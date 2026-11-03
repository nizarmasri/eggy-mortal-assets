"""The fighter's body: an MPFB human from the preset, its export copy, the game-ready low mesh, eyes and teeth.

Coordinates while building: Blender's, metres, Z up, the fighter facing -Y, their left side at +X. The
glTF export turns that into Y up and +Z forward.
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector

import mp
import util
from util import log

BONES = 53


def human_info(preset, with_eyes=True):
    h = preset["human"]
    info = {
        "name": preset["id"],
        "phenotype": h["phenotype"],
        "rig": "game_engine",
        "eyes": "low-poly/low-poly.mhclo" if with_eyes else "",
        "eyebrows": "",
        "eyelashes": "",
        "tongue": "",
        "teeth": "",
        "hair": "",
        "proxy": "",
        "targets": h.get("targets", []),
        "clothes": [],
        "skin_mhmat": h["skin"] if with_eyes else "",
        "skin_material_type": "GAMEENGINE" if with_eyes else "NONE",
        "eyes_material_type": "GAMEENGINE",
        "skin_material_settings": {},
        "eyes_material_settings": {},
        "alternative_materials": {},
        "expressions": [],
    }
    return info


def deserialize(info):
    settings = mp.HumanService.get_default_deserialization_settings()
    settings["subdiv_levels"] = 0
    settings["load_clothes"] = False
    settings["material_instances"] = "NEVER"
    return mp.HumanService.deserialize_from_dict(info, settings)


def reference_axes(reference_preset):
    """Edit-bone Z axes (armature space) of the reference fighter's rig, for roll normalisation."""
    info = human_info(reference_preset, with_eyes=False)
    info["name"] = "reference"
    basemesh = deserialize(info)
    rig = basemesh.parent
    util.select_only(rig)
    bpy.ops.object.mode_set(mode='EDIT')
    axes = {eb.name: tuple(eb.z_axis) for eb in rig.data.edit_bones}
    bpy.ops.object.mode_set(mode='OBJECT')
    util.delete(basemesh, rig)
    return axes


CREDITS = []


def mhclo_credit(path, file):
    """The header lines of an MHCLO that credit it (name, author, licence, homepage)."""
    out = {"file": file}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip().lstrip("#").strip()
            k, _, v = text.partition(" ")
            k = k.rstrip(":").lower()
            if k in ("name", "author", "license", "licence", "homepage", "uuid", "copyright") and k not in out:
                out[k] = v.strip()
            if k == "verts":
                break
    log("credit:", out)
    return out


def build(preset):
    """The MPFB human, then its export copy with shape keys and helpers baked away.

    Returns (armature, body, eyes): the export copies, original deleted.
    """
    info = human_info(preset)
    basemesh = deserialize(info)
    rig = basemesh.parent
    assert rig is not None and rig.type == 'ARMATURE', "game_engine rig missing"
    log("human built:", basemesh.name, len(basemesh.data.vertices), "verts;", len(rig.data.bones), "bones")
    eyes = mp.ObjectService.find_object_of_type_amongst_nearest_relatives(basemesh, "Eyes")
    # MakeHuman garments that suit (fitted, rigged and given a delete group on the body by MPFB).
    for item in preset.get("mhclo", []):
        path = mp.AssetService.find_asset_absolute_path(item["file"], "clothes")
        assert path, f"MHCLO not found: {item['file']}"
        clo = mp.HumanService.add_mhclo_asset(path, basemesh, asset_type="Clothes", subdiv_levels=0, material_type="GAMEENGINE")
        clo["mortal_name"] = item["name"]
        log("mhclo:", item["file"], len(clo.data.vertices), "verts")
        CREDITS.append(mhclo_credit(path, item["file"]))

    root = mp.ExportService.create_character_copy(basemesh)
    copies = [c for c in root.children]
    body = next(c for c in copies if mp.ObjectService.object_is_basemesh(c))
    eyes_copy = next((c for c in copies if mp.ObjectService.object_is_eyes(c)), None)

    # Bake the phenotype into the mesh, then let MPFB remove the helper geometry.
    util.select_only(body)
    if body.data.shape_keys:
        bpy.ops.object.shape_key_remove(all=True, apply_mix=True)
    mp.ExportService.bake_modifiers_remove_helpers(body, bake_masks=True, bake_subdiv=False, remove_helpers=True)
    for g in list(body.vertex_groups):
        if g.name.startswith(("helper-", "joint-")) or g.name in ("Mid", "Left", "Right", "HelperGeometry", "JointCubes", "body"):
            body.vertex_groups.remove(g)
    if eyes_copy is not None:
        util.select_only(eyes_copy)
        if eyes_copy.data.shape_keys:
            bpy.ops.object.shape_key_remove(all=True, apply_mix=True)
        for m in list(eyes_copy.modifiers):
            if m.type != 'ARMATURE':
                util.apply_modifier(eyes_copy, m.name)

    clothes = []
    for c in copies:
        if c.get("mortal_name"):
            util.select_only(c)
            if c.data.shape_keys:
                bpy.ops.object.shape_key_remove(all=True, apply_mix=True)
            for mm in list(c.modifiers):
                if mm.type != 'ARMATURE':
                    util.apply_modifier(c, mm.name)
            c.name = c.data.name = c["mortal_name"]
            clothes.append(c)
    # Drop the originals so nothing else in the scene carries the same rig.
    util.delete(*[c for c in rig.children], rig)
    root.name = "Armature"
    root.data.name = "Armature"
    body.name = body.data.name = "body"
    if eyes_copy is not None:
        eyes_copy.name = eyes_copy.data.name = "eyes"
    for o in (body, eyes_copy):
        if o is not None:
            for m in o.modifiers:
                if m.type == 'ARMATURE':
                    m.object = root
    log("export copy:", len(body.data.vertices), "body verts,", util.tri_count(body), "tris")
    return root, body, eyes_copy, clothes


def normalise_rolls(arm, ref_axes):
    """MPFB issue #310: the same rig fitted to different bodies gets different bone rolls, which twists
    fingers and toes when one animation drives every fighter. Align each bone's Z axis to the reference
    fighter's, the way TalkingHead's fix_bone_axes does."""
    util.select_only(arm)
    bpy.ops.object.mode_set(mode='EDIT')
    changed = 0
    for eb in arm.data.edit_bones:
        z = ref_axes.get(eb.name)
        if z is None:
            continue
        before = eb.roll
        eb.align_roll(Vector(z))
        if abs(eb.roll - before) > 1e-4:
            changed += 1
    bpy.ops.object.mode_set(mode='OBJECT')
    log("rolls normalised:", changed, "bones changed")


def bone_world(arm, name, which="head"):
    b = arm.data.bones[name]
    return arm.matrix_world @ (b.head_local if which == "head" else b.tail_local)


def measurements(arm, body):
    P = util.positions(body)
    m = {
        "height": float(P[:, 2].max()),
        "head_top": float(P[:, 2].max()),
        "eye_z": None,
    }
    names = ["pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head"]
    for s in ("l", "r"):
        names += [f"{n}_{s}" for n in ("thigh", "calf", "foot", "ball", "clavicle", "upperarm", "lowerarm", "hand",
                                        "index_01", "middle_01", "ring_01", "pinky_01", "thumb_01")]
    for n in names:
        m[n] = tuple(bone_world(arm, n))
        m[n + "_tail"] = tuple(bone_world(arm, n, "tail"))
    return m


def target_vertices(names, share=0.5):
    """Vertex indices a set of MakeHuman targets moves by at least `share` of its largest move (a target fades
    out round what it moves), in the basemesh's own numbering (which the export copy keeps: MPFB's helper
    geometry comes after the body's vertices)."""
    import gzip
    out = set()
    for n in names:
        path = mp.TargetService.target_full_path(n)
        if not path:
            log("target not found:", n)
            continue
        opener = gzip.open if str(path).endswith(".gz") else open
        ids, mags = [], []
        with opener(path, "rt") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4 and not line.startswith("#"):
                    try:
                        ids.append(int(parts[0]))
                        mags.append(math.sqrt(sum(float(x) ** 2 for x in parts[1:4])))
                    except ValueError:
                        pass
        if ids:
            top = max(mags)
            out |= {i for i, mg in zip(ids, mags) if mg >= share * top}
    return out


def mark_ears(body, group="mortal_ears"):
    """A vertex group on the ears, read from the targets that move them (the basemesh has no ear group)."""
    ids = sorted(i for i in target_vertices(["l-ear-trans-backward", "r-ear-trans-backward"]) if i < len(body.data.vertices))
    if not ids:
        return None
    P = util.positions(body)[ids]
    log(f"ears: {len(ids)} vertices, |x| {np.abs(P[:, 0]).mean():.3f}, z {P[:, 2].mean():.3f}")
    vg = body.vertex_groups.get(group) or body.vertex_groups.new(name=group)
    vg.add(ids, 1.0, 'REPLACE')
    return vg


def sculpt(body, arm, m, spec):
    """Flesh the targets cannot give on their own: a big round belly and a thick waist, pushed out along the
    normals in a smooth field after the abs are calmed. Only the skin moves: bones, rest pose and weights stay
    exactly as they are."""
    if not spec:
        return
    k = m["height"] / 1.78
    P = util.positions(body)
    N = util.normals(body)
    arms = util.weights_matrix(body, [f"{n}_{s}" for s in ("l", "r") for n in (
        "upperarm", "lowerarm", "hand")]).sum(axis=1)
    z_lo = 0.5 * (m["thigh_l"][2] + m["thigh_r"][2])
    z_hi = float(m["spine_03"][2])
    zb = z_lo + spec.get("at", 0.4) * (z_hi - z_lo)
    gz = np.exp(-((P[:, 2] - zb) / (spec.get("height", 0.15) * k)) ** 2)
    gx = np.exp(-(P[:, 0] / (spec.get("width", 0.17) * k)) ** 2)
    keep = (arms < 0.05).astype(float)
    front = util.smoothstep(-0.2, 0.75, -N[:, 1])
    belly = gz * gx * front * keep
    E = util.edges(body)
    P = util.taubin(P, E, spec.get("calm", 14), weight=np.clip(belly * 1.6, 0, 0.9))
    d = k * (spec.get("belly", 0.07) * belly + spec.get("waist", 0.025) * gz * keep)
    util.set_positions(body, P + N * d[:, None])
    log(f"sculpt: belly {spec.get('belly', 0.07) * k:.3f} m at z {zb:.3f}, waist {spec.get('waist', 0.025) * k:.3f} m")


def head_vertices(body, threshold=0.5):
    W = util.weights_matrix(body, ["head", "neck_01"])
    return np.where(W[:, 0] >= threshold)[0]


def decimate(obj, target_tris):
    tris = util.tri_count(obj)
    if tris <= target_tris:
        return tris
    mod = obj.modifiers.new("Decimate", 'DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = target_tris / tris
    mod.use_symmetry = True
    mod.symmetry_axis = 'X'
    mod.use_collapse_triangulate = True
    # Apply before the armature so the weights are interpolated on the rest mesh.
    util.select_only(obj)
    while obj.modifiers[0].name != mod.name:
        bpy.ops.object.modifier_move_up(modifier=mod.name)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    util.limit_and_normalize(obj)
    out = util.tri_count(obj)
    log(f"decimated {obj.name}: {tris} -> {out} tris")
    return out


def high_copy(body, levels=2):
    """A subdivided rest-pose copy of the body for baking (no armature)."""
    hi = util.duplicate(body, "body_high")
    for m in list(hi.modifiers):
        hi.modifiers.remove(m)
    hi.parent = None
    hi.matrix_world = body.matrix_world.copy()
    sub = hi.modifiers.new("Subdiv", 'SUBSURF')
    sub.levels = levels
    sub.render_levels = levels
    sub.uv_smooth = 'PRESERVE_BOUNDARIES'
    util.apply_modifier(hi, sub.name)
    util.shade_smooth(hi)
    return hi


def teeth(body, arm, color=(0.86, 0.82, 0.72)):
    """A small low-poly set of teeth behind the lips, weighted to the head (no jaw bone in the rig)."""
    g = body.vertex_groups.get("lips")
    P = util.positions(body)
    if g is None:
        return None
    ids = [v.index for v in body.data.vertices if any(gg.group == g.index for gg in v.groups)]
    L = P[ids]
    cx, cz = 0.0, float(np.median(L[:, 2]))
    front = float(L[:, 1].min())  # lips' front (most -Y)
    width = float(L[:, 0].max() - L[:, 0].min())
    r = width * 0.42
    cy = front + 0.008 + r * 0.95
    bm = bmesh.new()
    for row, (z0, z1) in enumerate(((cz + 0.0005, cz + 0.0095), (cz - 0.0095, cz - 0.0012))):
        n = 7
        for i in range(n):
            a0 = math.radians(-70 + 140 * i / n)
            a1 = math.radians(-70 + 140 * (i + 0.86) / n)
            rr = r * (1.0 if row == 0 else 0.93)
            pts = []
            for a in (a0, a1):
                for depth in (0.0, 0.006):
                    x = math.sin(a) * (rr - depth)
                    y = cy - math.cos(a) * (rr - depth)
                    pts.append((x, y))
            verts = []
            for (x, y) in pts:
                for z in (z0, z1):
                    verts.append(bm.verts.new((x, y, z)))
            # pts order: a0 front, a0 back, a1 front, a1 back; each with z0, z1
            v = verts
            quads = [(0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5), (0, 2, 6, 4), (1, 5, 7, 3)]
            for q in quads:
                try:
                    bm.faces.new([v[k] for k in q])
                except ValueError:
                    pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    obj = util.bmesh_object("teeth", bm)
    util.set_single_weight(obj, "head")
    util.add_armature(obj, arm)
    mat = util.new_material("teeth")
    nt = mat.node_tree
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*util.srgb_to_linear(color).tolist(), 1)
    bsdf.inputs["Roughness"].default_value = 0.35
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    util.set_material(obj, mat)
    util.shade_smooth(obj)
    return obj
