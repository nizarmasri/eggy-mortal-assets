"""The GLB as Blender writes it (post.mjs compresses it and attaches the ORM maps), the bone sidecar for
rig.json, and the re-import check: 53 joints with the UE4-style names."""
import json
import os

import bpy
import numpy as np
from mathutils import Matrix

import util
from util import log

JOINTS = [
    "Root", "pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head",
    *[f"{n}_{s}" for s in ("l", "r") for n in (
        "clavicle", "upperarm", "lowerarm", "hand",
        "thumb_01", "thumb_02", "thumb_03", "index_01", "index_02", "index_03", "middle_01", "middle_02", "middle_03",
        "ring_01", "ring_02", "ring_03", "pinky_01", "pinky_02", "pinky_03",
        "thigh", "calf", "foot", "ball")],
]
assert len(JOINTS) == 53


def tidy(obj, keep_uv="UVMap"):
    """One UV map named UVMap, no helper attributes."""
    me = obj.data
    if me.uv_layers:
        first = me.uv_layers.active or me.uv_layers[0]
        for uv in list(me.uv_layers):
            if uv.name != first.name:
                me.uv_layers.remove(uv)
        me.uv_layers[0].name = keep_uv
    for name in ("param",):
        if name in me.attributes:
            me.attributes.remove(me.attributes[name])


def to_gltf(v):
    """Blender (Z up, -Y forward) to glTF (Y up, +Z forward)."""
    return [round(float(v[0]), 6), round(float(v[2]), 6), round(float(-v[1]), 6)]


def bones_sidecar(arm, path):
    bones = []
    for b in arm.data.bones:
        bones.append({
            "name": b.name,
            "parent": b.parent.name if b.parent else None,
            "length": round(b.length, 6),
            "head": to_gltf(arm.matrix_world @ b.head_local),
            "tail": to_gltf(arm.matrix_world @ b.tail_local),
        })
    with open(path, "w") as f:
        json.dump({"bones": bones}, f, indent=1)


def glb(arm, objects, path):
    util.object_mode()
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    arm.select_set(True)
    for o in objects:
        o.hide_set(False)
        o.hide_render = False
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    kw = dict(
        filepath=path, export_format='GLB', use_selection=True, export_apply=False, export_yup=True,
        export_texcoords=True, export_normals=True, export_tangents=False, export_materials='EXPORT',
        export_image_format='AUTO', export_skins=True, export_all_influences=False, export_influence_nb=4,
        export_def_bones=False, export_animations=False, export_morph=False, export_attributes=True,
        export_extras=True, export_rest_position_armature=True, export_lights=False, export_cameras=False,
    )
    try:
        bpy.ops.export_scene.gltf(**kw)
    except TypeError as e:
        log("exporter refused an option, retrying without the optional ones:", e)
        for k in ("export_rest_position_armature", "export_influence_nb", "export_all_influences"):
            kw.pop(k, None)
        bpy.ops.export_scene.gltf(**kw)
    log("exported", path, os.path.getsize(path), "bytes")


def reimport_check(path):
    """Import the GLB into an empty scene: one armature, 53 joints, the expected names."""
    util.clear_scene()
    bpy.ops.import_scene.gltf(filepath=path)
    arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    assert len(arms) == 1, f"expected one armature, got {len(arms)}"
    names = [b.name for b in arms[0].data.bones]
    missing = sorted(set(JOINTS) - set(names))
    extra = sorted(set(names) - set(JOINTS))
    assert len(names) == 53 and not missing and not extra, f"joints: {len(names)}, missing {missing}, extra {extra}"
    meshes = sorted(o.name for o in bpy.context.scene.objects if o.type == 'MESH')
    log(f"re-import ok: 53 joints, meshes {meshes}")
    return meshes
