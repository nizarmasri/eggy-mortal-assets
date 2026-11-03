"""Build one fighter: body, gear, mask, shards, textures, review renders, the raw GLB and its sidecars.

    blender -b -P pipeline/blender/fighter.py -- <id> <outdir> [--fast] [--no-previews]

Writes <outdir>/raw/<id>.glb (Blender's export, uncompressed), <outdir>/tex/*.png (every texture, the ORM
maps post.mjs attaches and the crack map it converts), <outdir>/<id>.bones.json, <outdir>/<id>.stats.json and
<outdir>/previews/*.png.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import util  # noqa: E402
from util import log  # noqa: E402
import body  # noqa: E402
import outfit  # noqa: E402
import mask  # noqa: E402
import designs  # noqa: E402
import shaders  # noqa: E402
import bake  # noqa: E402
import render  # noqa: E402
import export  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
FID = argv[0] if argv else "reiji"
OUT = os.path.abspath(argv[1] if len(argv) > 1 else os.path.join(ROOT, "out", FID))
FAST = "--fast" in argv
PREVIEWS = "--no-previews" not in argv
TEX = os.path.join(OUT, "tex")
os.makedirs(TEX, exist_ok=True)
os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)

T0 = time.time()


def stage(name):
    log(f"--- {name} ({time.time() - T0:.0f}s)")


def load(fid):
    with open(os.path.join(ROOT, "fighters", f"{fid}.json")) as f:
        return json.load(f)


preset = load(FID)
reference = load(preset.get("reference", "reiji"))
spec_mask = preset.get("mask", {})
size_body = preset.get("tex", {}).get("body", 1024)
size_gear = preset.get("tex", {}).get("gear", 1024)
size_mask = preset.get("tex", {}).get("mask", 1024)

util.clear_scene()
stage("reference rig")
ref_axes = body.reference_axes(reference)

stage("body")
arm, bod, eyes = body.build(preset)
body.normalise_rolls(arm, ref_axes)
m = body.measurements(arm, bod)
log("height", round(m["height"], 3))
teeth = body.teeth(bod, arm)

stage("gear")
pieces = []
info = {}
for item in preset.get("outfit", []):
    kind = item["kind"]
    if kind == "loose_pants":
        o, info = outfit.loose_pants(bod, arm, m, item)
        util.set_material(o, shaders.fabric("fabric_" + o.name, shaders.rgb(item["color"]), knee_z=info["knee_z"], hem_z=info["hem_z"], stains=item.get("stains", 0.0)))
        pieces.append(o)
        pants = o
    elif kind == "belt":
        o = outfit.belt(pants, arm, info, item)
        util.set_material(o, shaders.belt("belt_" + o.name, shaders.rgb(item["color"]), edge=shaders.rgb(item.get("edge", (0.5, 0.48, 0.45)))))
        pieces.append(o)
    elif kind == "wraps":
        o = outfit.wraps(bod, arm, m, item, name=item.get("name", "wraps"))
        util.set_material(o, shaders.tape("tape_" + o.name, shaders.rgb(item["color"]), band=item.get("band", 0.024), angle=item.get("angle", 0.55)))
        pieces.append(o)
    else:
        raise ValueError(f"unknown outfit kind {kind}")

stage("mask")
egg = mask.fit(bod, eyes, spec_mask.get("fit", {}))
mask_obj, centres = mask.build(egg, eyes, bod, spec_mask)

hair_pieces = []
if preset.get("hair"):
    h = preset["hair"]
    shell, knot, tie = outfit.short_hair(bod, arm, m, egg, h)
    hm = shaders.hair("hair_mat", shaders.rgb(h["color"]), tip=h.get("tip", (0.06, 0.05, 0.045)))
    for o in (shell, knot):
        util.set_material(o, hm)
    util.set_material(tie, shaders.cord("tie_mat", shaders.rgb(h.get("tie", (0.85, 0.82, 0.76)))))
    hair_pieces = [shell, knot, tie]
eye_y = 0.5 * (centres["eye_l"][1] + centres["eye_r"][1])
eye_th = float(np.interp(eye_y, egg.M, egg.th_t))
strap = outfit.strap([*(hair_pieces[:1]), bod], egg, eye_th + 0.05)
util.set_material(strap, shaders.leather("leather_mat", shaders.rgb(spec_mask.get("strap", (0.16, 0.10, 0.07)))))
pieces += hair_pieces + [strap]

stage("body low")
hi = body.high_copy(bod, levels=1 if FAST else 2)
body.decimate(bod, preset.get("body_tris", 18000))
# Faces the trousers hide for good (thighs and hips) go: fewer triangles, nothing to poke through.
if info:
    P = util.positions(bod)
    armw = util.weights_matrix(bod, outfit.sided(outfit.ARM_BONES)).sum(axis=1)
    hidden = (P[:, 2] > info["knee_z"] + 0.07) & (P[:, 2] < info["waist_z"] - 0.06) & (armw < 0.05)
    util.delete_faces(bod, util.faces_of_vertices(bod, hidden))
util.limit_and_normalize(bod)
for g in list(bod.vertex_groups):
    if g.name not in export.JOINTS:
        bod.vertex_groups.remove(g)

stage("gear atlas")
for o in pieces:
    me = o.data
    for uv in list(me.uv_layers):
        me.uv_layers.remove(uv)
    me.uv_layers.new(name="UVMap")
    for mm in list(o.modifiers):
        o.modifiers.remove(mm)
gear = util.join(pieces, "gear")
for g in list(gear.vertex_groups):
    if g.name not in export.JOINTS:
        gear.vertex_groups.remove(g)
util.limit_and_normalize(gear)
util.select_only(gear)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.003, area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
bpy.ops.uv.pack_islands(rotate=True, margin=0.003)
bpy.ops.object.mode_set(mode='OBJECT')

stage("bake body")
bake.cycles()
body_albedo, body_orm, body_nrm = bake.body_maps(bod, hi, size_body, TEX, tint=preset["human"].get("tint", (1, 1, 1)), hide=[mask_obj, gear], detail=preset["human"].get("detail"))
stage("bake gear")
gear_albedo, gear_orm, gear_nrm = bake.gear_maps(gear, size_gear, TEX, hide=[hi])

stage("mask textures")
cv = mask.canvas(egg, size_mask)
seed_pts, seed_xy = mask.seeds(egg, spec_mask.get("shards", 30), seed=spec_mask.get("seed", 1))
crack, cells, wall = mask.crack_map(cv, seed_pts)
ctx = {"slits": centres, "seeds": seed_xy, "cells": cells, "wall": wall, "egg": egg}
painted = designs.paint(FID, cv, ctx)
mask_albedo, mask_orm, mask_nrm = mask.textures(egg, cv, painted, TEX, bisque=spec_mask.get("bisque", (0.80, 0.78, 0.74)))
mask.crack_image(cv, crack, TEX)
mask.uv_unwrap(mask_obj, egg)

stage("final materials")
skin = shaders.image_material("skin", body_albedo, body_orm, body_nrm, subsurface=0.05)
util.set_material(bod, skin)
gear_mat = shaders.image_material("gear", gear_albedo, gear_orm, gear_nrm, double_sided=True)
gear.data.materials.clear()
gear.data.materials.append(gear_mat)
gear.data.polygons.foreach_set("material_index", [0] * len(gear.data.polygons))
mask_mat = shaders.image_material("mask", mask_albedo, mask_orm, mask_nrm, metallic_from_b=True)
util.set_material(mask_obj, mask_mat)
eye_src = bake.source_diffuse(eyes)
eye_arr = util.resize(util.image_array(eye_src)[..., :3], 256)
eye_img = util.image_from_array("eyes_albedo", eye_arr, os.path.join(TEX, "eyes_albedo.png"))
util.set_material(eyes, shaders.image_material("eyes", eye_img))
for o in (gear, mask_obj):
    util.add_armature(o, arm)
util.set_single_weight(mask_obj, "head")

stage("fracture")
shards = mask.fracture(mask_obj, seed_pts, egg)
shards = mask.finish_shards(shards, egg, arm, mask_mat)
log("shards:", len(shards))

for o in (bod, eyes, teeth, gear, mask_obj):
    export.tidy(o)
for o in shards:
    export.tidy(o)

stats = {
    "id": FID,
    "height_m": round(m["height"], 4),
    "triangles": {o.name: util.tri_count(o) for o in (bod, eyes, teeth, gear, mask_obj)},
    "shards": len(shards),
    "shard_triangles": sum(util.tri_count(o) for o in shards),
    "textures": {img.name: list(img.size) for img in (body_albedo, body_orm, body_nrm, gear_albedo, gear_orm, gear_nrm, mask_albedo, mask_orm, mask_nrm, eye_img)},
    "egg": {"C": egg.C.round(4).tolist(), "a": round(egg.a, 4), "b": round(egg.b, 4), "h": round(egg.h, 4), "phimax_deg": round(math.degrees(egg.phimax), 2)},
}
log("stats", json.dumps(stats))

if PREVIEWS:
    stage("previews")
    render.all_shots(arm, m, os.path.join(OUT, "previews"), preset.get("stance", "karate"), FID, hidden=[hi, *shards])

stage("export")
util.delete(hi)
raw = os.path.join(OUT, "raw", f"{FID}.glb")
export.bones_sidecar(arm, os.path.join(OUT, f"{FID}.bones.json"))
export.glb(arm, [bod, eyes, teeth, gear, mask_obj, *shards], raw)
stats["raw_bytes"] = os.path.getsize(raw)
with open(os.path.join(OUT, f"{FID}.stats.json"), "w") as f:
    json.dump(stats, f, indent=1)

stage("re-import check")
stats["reimport_meshes"] = export.reimport_check(raw)
with open(os.path.join(OUT, f"{FID}.stats.json"), "w") as f:
    json.dump(stats, f, indent=1)
stage("done")
