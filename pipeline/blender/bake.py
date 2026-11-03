"""Cycles bakes: the body's normal, roughness, cavity and AO from its subdivided copy, and the gear atlas
(colour, roughness, normal, AO) from the procedural materials."""
import os

import bpy
import numpy as np

import util
from util import log


def cycles(samples=16):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.render.bake.margin = 12
    sc.render.bake.margin_type = 'EXTEND'


def _targets(obj, img):
    for slot in obj.material_slots:
        nt = slot.material.node_tree
        node = nt.nodes.get("BAKE_TARGET")
        if node is None:
            node = nt.nodes.new("ShaderNodeTexImage")
            node.name = "BAKE_TARGET"
            node.location = (-600, -600)
        node.image = img
        for n in nt.nodes:
            n.select = False
        node.select = True
        nt.nodes.active = node


def remove_targets(obj):
    for slot in obj.material_slots:
        nt = slot.material.node_tree
        node = nt.nodes.get("BAKE_TARGET")
        if node:
            nt.nodes.remove(node)


def bake(obj, img, kind, high=None, pass_filter=None, cage=0.012, max_ray=0.03):
    _targets(obj, img)
    if high is not None:
        util.select_only(obj, high)
        bpy.context.view_layer.objects.active = obj
    else:
        util.select_only(obj)
    kw = dict(type=kind, margin=12, use_clear=True, target='IMAGE_TEXTURES')
    if high is not None:
        kw.update(use_selected_to_active=True, cage_extrusion=cage, max_ray_distance=max_ray)
    if kind == 'NORMAL':
        kw.update(normal_space='TANGENT')
    if pass_filter:
        kw["pass_filter"] = pass_filter
    bpy.ops.object.bake(**kw)
    log(f"baked {kind} of {obj.name} into {img.name}")


def source_diffuse(obj):
    """The image the object's material feeds into Base Color (MPFB's GAMEENGINE skin)."""
    mat = obj.material_slots[0].material
    nt = mat.node_tree
    for link in nt.links:
        if link.to_socket.name == "Base Color" and link.from_node.type == 'TEX_IMAGE':
            return link.from_node.image
    for n in nt.nodes:
        if n.type == 'TEX_IMAGE' and n.image is not None:
            return n.image
    return None


def body_maps(low, high, size, outdir, tint=(1, 1, 1), hide=()):
    """Albedo from MakeHuman's skin texture (same UVs), and baked normal, roughness, cavity and AO."""
    src = source_diffuse(low)
    assert src is not None, "skin texture not found"
    a = util.image_array(src)[..., :3]
    a = util.resize(a, size)
    lin = util.srgb_to_linear(a) * np.array(util.srgb_to_linear(np.array(tint)), dtype=np.float32)
    albedo = util.image_from_array("body_albedo", util.linear_to_srgb(lin), os.path.join(outdir, "body_albedo.png"))

    # The high copy's material: the same texture, a little unevenness and pore-scale roughness.
    mat = util.new_material("skin_high")
    import shaders
    g = shaders.G(mat)
    co = g.coords("Object")
    tex = g.n("ShaderNodeTexImage", _image=src)
    uneven = g.noise(co, 180.0, 4.0, 0.55)
    fine = g.noise(co, 900.0, 2.0, 0.5)
    height = g.math('ADD', g.math('MULTIPLY', uneven, 0.6), g.math('MULTIPLY', fine, 0.4))
    rough = g.math('ADD', 0.47, g.math('MULTIPLY', g.noise(co, 60.0, 3.0), 0.16))
    g.principled(tex.outputs["Color"], rough, height, strength=0.12, distance=0.0008)
    high.data.materials.clear()
    high.data.materials.append(mat)
    low_mats = [s.material for s in low.material_slots]

    cycles(samples=4)
    nrm = util.new_image("body_normal", size, non_color=True)
    bake(low, nrm, 'NORMAL', high=high)
    rgh = util.new_image("body_rough", size, non_color=True)
    bake(low, rgh, 'ROUGHNESS', high=high)
    # Cavity: the high mesh's concavity through an emission shader.
    cav_mat = util.new_material("skin_cavity")
    g = shaders.G(cav_mat)
    geo = g.n("ShaderNodeNewGeometry")
    col = g.ramp(geo.outputs["Pointiness"], [(0.44, (0, 0, 0)), (0.52, (1, 1, 1))])
    em = g.n("ShaderNodeEmission")
    out = g.n("ShaderNodeOutputMaterial")
    g.link(col, em.inputs[0])
    g.link(em.outputs[0], out.inputs[0])
    high.data.materials[0] = cav_mat
    cav = util.new_image("body_cavity", size, non_color=True)
    bake(low, cav, 'EMIT', high=high)
    # AO on the low mesh itself, with the high copy and anything listed out of the way.
    high.hide_render = True
    for o in hide:
        o.hide_render = True
    cycles(samples=64)
    ao = util.new_image("body_ao", size, non_color=True)
    bake(low, ao, 'AO')
    for o in hide:
        o.hide_render = False
    remove_targets(low)

    R = util.image_array(ao)[..., 0]
    G = util.image_array(rgh)[..., 0]
    B = util.image_array(cav)[..., 0]
    R = 0.35 + 0.65 * R  # the renderer multiplies this into indirect light only; keep it gentle
    orm = util.image_from_array("body_orm", np.stack([R, G, B], axis=-1), os.path.join(outdir, "body_orm.png"), non_color=True)
    util.save_png(nrm, os.path.join(outdir, "body_normal.png"))
    return albedo, orm, nrm


def gear_maps(gear, size, outdir, hide=()):
    cycles(samples=8)
    col = util.new_image("gear_albedo", size)
    bake(gear, col, 'DIFFUSE', pass_filter={'COLOR'})
    rgh = util.new_image("gear_rough", size, non_color=True)
    bake(gear, rgh, 'ROUGHNESS')
    nrm = util.new_image("gear_normal", size, non_color=True)
    bake(gear, nrm, 'NORMAL')
    for o in hide:
        o.hide_render = True
    cycles(samples=64)
    ao = util.new_image("gear_ao", size, non_color=True)
    bake(gear, ao, 'AO')
    for o in hide:
        o.hide_render = False
    remove_targets(gear)
    R = 0.3 + 0.7 * util.image_array(ao)[..., 0]
    G = util.image_array(rgh)[..., 0]
    orm = util.image_from_array("gear_orm", np.stack([R, G, np.zeros_like(G)], axis=-1), os.path.join(outdir, "gear_orm.png"), non_color=True)
    util.save_png(col, os.path.join(outdir, "gear_albedo.png"))
    util.save_png(nrm, os.path.join(outdir, "gear_normal.png"))
    return col, orm, nrm
