"""Review renders (Cycles, CPU): turntable views in the fighting stance, a mask close-up, the rest pose, and
one frame from the game camera (side view, 3.5 m, 35 degree vertical FOV, eye height 1.3 m, warm key and
cool rim). Plus a contact sheet of them all."""
import math
import os

import bpy
import numpy as np
from mathutils import Euler, Vector

import pose
import util
from util import log


def _world(color=(0.010, 0.010, 0.012), strength=1.0):
    w = bpy.data.worlds.get("studio") or bpy.data.worlds.new("studio")
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (*color, 1)
    bg.inputs[1].default_value = strength
    bpy.context.scene.world = w


def _light(name, kind, loc, target, color, energy, size):
    data = bpy.data.lights.get(name) or bpy.data.lights.new(name, kind)
    data.energy = energy
    data.color = color
    if kind == 'AREA':
        data.size = size
    obj = bpy.data.objects.get(name) or util.link(bpy.data.objects.new(name, data))
    obj.location = loc
    util.look_at(obj, target)
    return obj


def _floor():
    if bpy.data.objects.get("floor"):
        return bpy.data.objects["floor"]
    obj = util.mesh_object("floor", [(-12, -12, 0), (12, -12, 0), (12, 12, 0), (-12, 12, 0)], [(0, 1, 2, 3)])
    mat = util.new_material("floor")
    import shaders
    g = shaders.G(mat)
    n = g.noise(g.coords("Object"), 3.0, 6.0, 0.6)
    col = g.ramp(n, [(0.3, (0.035, 0.034, 0.033)), (0.7, (0.06, 0.058, 0.055))])
    g.principled(col, 0.75)
    util.set_material(obj, mat)
    return obj


def _camera(name, loc, target, lens=None, vfov=None, sensor_fit='VERTICAL'):
    cam = bpy.data.cameras.get(name) or bpy.data.cameras.new(name)
    cam.sensor_fit = sensor_fit
    cam.sensor_height = 24.0
    if vfov is not None:
        cam.lens_unit = 'FOV'
        cam.angle_y = math.radians(vfov)
    else:
        cam.lens = lens
    obj = bpy.data.objects.get(name) or util.link(bpy.data.objects.new(name, cam))
    obj.location = loc
    util.look_at(obj, target)
    return obj


def _settings(w, h, samples=64):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    except TypeError:
        pass
    sc.cycles.max_bounces = 6
    sc.render.resolution_x = w
    sc.render.resolution_y = h
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'
    sc.view_settings.view_transform = 'AgX'
    for look in ("AgX - Medium High Contrast", "AgX - Base Contrast", "None"):
        try:
            sc.view_settings.look = look
            break
        except TypeError:
            continue


def _shot(path, cam, w, h, samples=64):
    sc = bpy.context.scene
    sc.camera = cam
    _settings(w, h, samples)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    log("rendered", os.path.basename(path))
    return path


def studio_lights(centre_z):
    t = Vector((0, 0, centre_z))
    _light("key", 'AREA', (-2.4, -2.6, 3.2), t, (1.0, 0.84, 0.66), 650, 1.2)
    _light("fill", 'AREA', (2.8, -2.4, 1.2), t, (0.8, 0.86, 1.0), 70, 3.0)
    _light("rim", 'AREA', (2.0, 2.8, 2.6), t + Vector((0, 0, 0.3)), (0.62, 0.76, 1.0), 700, 1.0)
    _light("rim2", 'AREA', (-2.2, 2.4, 2.0), t, (1.0, 0.78, 0.58), 300, 1.0)


def all_shots(arm, m, outdir, stance_name, fighter_id, hidden=()):
    os.makedirs(outdir, exist_ok=True)
    for o in hidden:
        o.hide_render = True
    _world()
    _floor()
    H = m["height"]
    cz = H * 0.5
    studio_lights(cz)
    out = {}
    vfov = 24.0
    dist = (H * 1.16) / (2 * math.tan(math.radians(vfov / 2)))
    # Rest pose, front: for checking the mesh itself.
    pose.reset(arm)
    cam = _camera("cam", (0, -dist, cz), (0, 0, cz), vfov=vfov)
    out["rest"] = _shot(os.path.join(outdir, f"{fighter_id}_rest.png"), cam, 768, 1024)
    pose.stance(arm, m, stance_name)
    cz2 = cz - 0.04
    for name, ang in (("front", 0), ("three_quarter", 38), ("back", 180)):
        a = math.radians(ang)
        cam = _camera("cam", (dist * math.sin(a), -dist * math.cos(a), cz2), (0, 0, cz2), vfov=vfov)
        out[name] = _shot(os.path.join(outdir, f"{fighter_id}_{name}.png"), cam, 768, 1024)
    # The mask close up, three-quarter, at head height.
    hp = pose.head(arm, "head")
    hc = hp + Vector((0, -0.02, 0.07))
    a = math.radians(28)
    cam = _camera("cam_mask", (hc.x + 0.95 * math.sin(a), hc.y - 0.95 * math.cos(a), hc.z + 0.05), hc, lens=85)
    _light("mask_key", 'AREA', (hc.x - 0.9, hc.y - 1.0, hc.z + 0.9), hc, (1.0, 0.9, 0.8), 70, 0.6)
    out["mask"] = _shot(os.path.join(outdir, f"{fighter_id}_mask.png"), cam, 768, 768, samples=96)
    bpy.data.objects.remove(bpy.data.objects["mask_key"], do_unlink=True)
    # The game camera: side on, P1 on the left facing right.
    for n in ("key", "fill", "rim", "rim2"):
        bpy.data.objects[n].hide_render = True
    arm.rotation_euler = Euler((0, 0, math.radians(90)))
    arm.location = (-0.55, 0, 0)
    util.object_mode()
    bpy.context.view_layer.update()
    _light("g_key", 'AREA', (-2.6, -2.4, 3.2), Vector((-0.55, 0, 1.0)), (1.0, 0.72, 0.42), 1100, 1.4)
    _light("g_rim", 'AREA', (0.6, 2.6, 2.4), Vector((-0.55, 0, 1.2)), (0.55, 0.72, 1.0), 1300, 1.0)
    _light("g_fill", 'AREA', (1.5, -3.0, 1.2), Vector((-0.55, 0, 1.0)), (0.75, 0.8, 0.95), 90, 3.0)
    _world((0.006, 0.0065, 0.008))
    cam = _camera("cam_game", (0, -3.5, 1.3), (0, 0, 1.0), vfov=35)
    out["game"] = _shot(os.path.join(outdir, f"{fighter_id}_game.png"), cam, 1664, 768, samples=64)
    for n in ("g_key", "g_rim", "g_fill"):
        bpy.data.objects.remove(bpy.data.objects[n], do_unlink=True)
    for n in ("key", "fill", "rim", "rim2"):
        bpy.data.objects[n].hide_render = False
    arm.rotation_euler = Euler((0, 0, 0))
    arm.location = (0, 0, 0)
    pose.reset(arm)
    for o in hidden:
        o.hide_render = False
    sheet(out, os.path.join(outdir, f"{fighter_id}_sheet.png"))
    return out


def sheet(shots, path):
    """One image: the four turntable views and the mask in a row, the game camera underneath."""
    def load(p):
        img = bpy.data.images.load(p, check_existing=False)
        a = util.image_array(img)[..., :3]
        bpy.data.images.remove(img)
        return a
    row = []
    for k in ("rest", "front", "three_quarter", "back"):
        row.append(util.resize(load(shots[k]), (512, 384)))
    m = util.resize(load(shots["mask"]), (512, 512))
    top = np.concatenate(row + [m], axis=1)
    g = load(shots["game"])
    gw = top.shape[1]
    gh = int(round(g.shape[0] * gw / g.shape[1]))
    g = util.resize(g, (gh, gw))
    full = np.concatenate([g, top], axis=0)  # image rows run bottom-up
    img = util.new_image("sheet", (full.shape[1], full.shape[0]))
    util.set_image_array(img, full)
    util.save_png(img, path)
    bpy.data.images.remove(img)
