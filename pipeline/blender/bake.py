"""Cycles bakes: the body's normal, roughness, cavity and AO from its subdivided copy, and the gear atlas
(colour, roughness, normal, AO) from the procedural materials."""
import os

import bpy
import numpy as np

import shaders
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


def _load_array(path, size):
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = 'Non-Color'
    a = util.image_array(img)[..., :3]
    bpy.data.images.remove(img)
    return util.resize(a, size)


def _blur(a, r):
    """Separable box blur (radius r texels, three passes ~ gaussian), edges clamped."""
    out = a.astype(np.float64)
    for _ in range(3):
        for axis in (0, 1):
            pad = [(0, 0)] * out.ndim
            pad[axis] = (r, r)
            p = np.pad(out, pad, mode='edge')
            c = np.cumsum(p, axis=axis)
            c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
            n = out.shape[axis]
            hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
            lo = np.take(c, np.arange(0, n), axis=axis)
            out = (hi - lo) / (2 * r + 1)
    return out


def rnm(base, detail, strength=1.0):
    """Reoriented normal mapping: detail laid onto base (both 0..1 encoded tangent-space normals)."""
    n1 = base * 2 - 1
    n2 = detail * 2 - 1
    n2[..., :2] *= strength
    n2 /= np.linalg.norm(n2, axis=-1, keepdims=True)
    t = n1 + np.array([0, 0, 1.0])
    u = n2 * np.array([-1, -1, 1.0])
    r = t * (np.sum(t * u, axis=-1, keepdims=True) / np.maximum(t[..., 2:3], 1e-4)) - u
    r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-6)
    return r * 0.5 + 0.5


def body_maps(low, high, size, outdir, tint=(1, 1, 1), hide=(), detail=None, hair=None):
    """Albedo from MakeHuman's skin texture (same UVs), and baked normal, roughness, cavity and AO. `detail`
    names a MakeHuman skin's normal and specular maps (same UV layout) laid over the bake for anatomy and
    skin texture: {normal, spec, strength, highpass (texels, 0 = keep everything), flip_y}."""
    src = source_diffuse(low)
    assert src is not None, "skin texture not found"
    a = util.image_array(src)[..., :3]
    a = util.resize(a, size)
    lin = util.srgb_to_linear(a) * np.array(util.srgb_to_linear(np.array(tint)), dtype=np.float32)
    albedo = util.image_from_array("body_albedo", util.linear_to_srgb(lin), os.path.join(outdir, "body_albedo.png"))

    # The high copy's material: the same texture, a little unevenness and pore-scale roughness.
    mat = util.new_material("skin_high")
    g = shaders.G(mat)
    co = g.coords("Object")
    tex = g.n("ShaderNodeTexImage", _image=src)
    uneven = g.noise(co, 180.0, 4.0, 0.55)
    fine = g.noise(co, 900.0, 2.0, 0.5)
    height = g.math('ADD', g.math('MULTIPLY', uneven, 0.6), g.math('MULTIPLY', fine, 0.4))
    rough = g.math('ADD', 0.5, g.math('MULTIPLY', g.noise(co, 60.0, 3.0), 0.16))
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
    # Hair painted on the scalp under the hair shell, fading out past the hairline.
    scalp_mask = None
    g_scalp = high.vertex_groups.get("scalp")
    if hair and g_scalp is not None:
        w = np.zeros(len(high.data.vertices), dtype=np.float32)
        for v in high.data.vertices:
            for gg in v.groups:
                if gg.group == g_scalp.index:
                    w[v.index] = gg.weight
        attr = high.data.color_attributes.new("scalp_w", 'FLOAT_COLOR', 'POINT')
        attr.data.foreach_set("color", np.repeat(w, 4))
        sm = util.new_material("scalp_mask")
        g = shaders.G(sm)
        at = g.n("ShaderNodeAttribute", _attribute_name="scalp_w", _attribute_type='GEOMETRY')
        em = g.n("ShaderNodeEmission")
        out = g.n("ShaderNodeOutputMaterial")
        g.link(at.outputs["Color"], em.inputs[0])
        g.link(em.outputs[0], out.inputs[0])
        high.data.materials[0] = sm
        scalp_img = util.new_image("body_scalp", size, non_color=True)
        bake(low, scalp_img, 'EMIT', high=high)
        scalp_mask = util.image_array(scalp_img)[..., 0]
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
    N = util.image_array(nrm)[..., :3]
    if detail:
        import mp
        dn = mp.AssetService.find_asset_absolute_path(detail["normal"], "skins")
        if dn:
            D = _load_array(dn, size)
            if detail.get("flip_y"):
                D[..., 1] = 1 - D[..., 1]
            hp = detail.get("highpass", 0)
            if hp:
                d = D * 2 - 1
                d = d - _blur(d, hp) + np.array([0, 0, 1.0])
                d /= np.linalg.norm(d, axis=-1, keepdims=True)
                D = d * 0.5 + 0.5
            N = rnm(N, D, detail.get("strength", 1.0))
            log("detail normal from", os.path.basename(dn))
        ds = mp.AssetService.find_asset_absolute_path(detail["spec"], "skins") if detail.get("spec") else None
        if ds:
            S = _load_array(ds, size)[..., 0]
            S = (S - S.mean()) / max(S.std(), 1e-3)
            G = np.clip(G - 0.05 * S, 0.48, 0.8)
            log("roughness from", os.path.basename(ds))
    if scalp_mask is not None:
        m = np.clip(scalp_mask * 1.25, 0, 1) ** 1.5
        yy, xx = np.mgrid[0:size, 0:size] / size
        streak = util.fbm3(np.stack([xx * 300, yy * 40, np.zeros_like(xx)], axis=-1), 2, seed=21)
        hc = util.srgb_to_linear(np.array(hair, dtype=np.float32)) * (0.7 + 0.6 * streak[..., None])
        a2 = util.srgb_to_linear(util.image_array(albedo)[..., :3])
        a2 = a2 * (1 - m[..., None]) + hc * m[..., None]
        albedo = util.image_from_array("body_albedo", util.linear_to_srgb(a2), os.path.join(outdir, "body_albedo.png"))
        G = G * (1 - m) + 0.6 * m
    orm = util.image_from_array("body_orm", np.stack([R, G, B], axis=-1), os.path.join(outdir, "body_orm.png"), non_color=True)
    nrm = util.image_from_array("body_normal", N, os.path.join(outdir, "body_normal.png"), non_color=True)
    return albedo, orm, nrm


def _emit_bake(obj, size, name, shader_for_slot):
    """Bake an emission per material slot into a float image (pattern coordinates, piece ids)."""
    saved = [s.material for s in obj.material_slots]
    temps = []
    for i, slot in enumerate(obj.material_slots):
        m = util.new_material(f"_tmp_{name}_{i}")
        g = shaders.G(m)
        em = g.n("ShaderNodeEmission")
        out = g.n("ShaderNodeOutputMaterial")
        shader_for_slot(g, em, i)
        g.link(em.outputs[0], out.inputs[0])
        slot.material = m
        temps.append(m)
    img = util.new_image(name, size, non_color=True, float_buffer=True)
    cycles(samples=1)
    bake(obj, img, 'EMIT')
    remove_targets(obj)
    for slot, m in zip(obj.material_slots, saved):
        slot.material = m
    for m in temps:
        bpy.data.materials.remove(m)
    return util.image_array(img)


def gear_maps(gear, size, outdir, hide=(), pattern=None):
    cycles(samples=8)
    col = util.new_image("gear_albedo", size)
    bake(gear, col, 'DIFFUSE', pass_filter={'COLOR'})
    rgh = util.new_image("gear_rough", size, non_color=True)
    bake(gear, rgh, 'ROUGHNESS')
    if pattern is not None:
        # Pattern coordinates and piece ids per texel, then the fighter's own paint over the bake.
        names = [s.material.name for s in gear.material_slots]

        def param_shader(g, em, i):
            p = g.attr("param")
            v = g.vmath('MULTIPLY_ADD', p, (0.25, 0.25, 0.25))
            node = v.node
            node.inputs[2].default_value = (0.5, 0.5, 0.5)
            g.link(v, em.inputs[0])

        def id_shader(g, em, i):
            em.inputs[0].default_value = ((i + 0.5) / 64, 0, 0, 1)
        prm = (_emit_bake(gear, size, "gear_param", param_shader)[..., :3] - 0.5) * 4
        sid = _emit_bake(gear, size, "gear_ids", id_shader)
        slot = np.floor(sid[..., 0] * 64).astype(int)
        covered = sid[..., 3] > 0.5 if sid.shape[-1] == 4 else np.ones(slot.shape, dtype=bool)
        A = util.srgb_to_linear(util.image_array(col)[..., :3])
        Rg = util.image_array(rgh)[..., 0].copy()
        A, Rg = pattern(names, slot, prm, A, Rg)
        util.set_image_array(col, util.linear_to_srgb(A))
        util.set_image_array(rgh, np.repeat(Rg[..., None], 3, axis=-1))
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
