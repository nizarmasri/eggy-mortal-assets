"""Small Blender helpers shared by every stage: selection, modifiers, meshes, vertex groups, images."""
import math
import os
import random

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector


def log(*args):
    print("[mortal]", *args, flush=True)


# ---------- selection and modes ----------

def object_mode():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def select_only(*objs):
    object_mode()
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0] if objs else None


def apply_modifier(obj, name):
    select_only(obj)
    bpy.ops.object.modifier_apply(modifier=name)


def apply_all_modifiers(obj, keep=("ARMATURE",)):
    for m in list(obj.modifiers):
        if m.type in keep:
            continue
        apply_modifier(obj, m.name)


def apply_transforms(obj):
    select_only(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def delete(*objs):
    for o in objs:
        if o is None:
            continue
        data = o.data if o.type == 'MESH' else None
        bpy.data.objects.remove(o, do_unlink=True)
        if data is not None and data.users == 0:
            bpy.data.meshes.remove(data)


def clear_scene():
    object_mode()
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for d in list(coll):
            if d.users == 0:
                coll.remove(d)


def link(obj, collection=None):
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def duplicate(obj, name):
    new = obj.copy()
    new.data = obj.data.copy()
    new.name = name
    new.data.name = name
    link(new)
    return new


# ---------- meshes ----------

def mesh_object(name, verts, faces, uvs=None, uv_name="UVMap"):
    """A mesh object from vertex positions, faces and optional per-loop UVs (list per face)."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    if uvs is not None:
        layer = me.uv_layers.new(name=uv_name)
        flat = []
        for f, fuv in zip(faces, uvs):
            flat.extend(fuv)
        for i, uv in enumerate(flat):
            layer.data[i].uv = uv
    obj = bpy.data.objects.new(name, me)
    return link(obj)


def bmesh_object(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    return link(obj)


def tri_count(obj):
    """Triangles of the evaluated mesh (modifiers applied)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    n = sum(len(p.vertices) - 2 for p in me.polygons)
    ev.to_mesh_clear()
    return n


def positions(obj):
    me = obj.data
    a = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def set_positions(obj, arr):
    obj.data.vertices.foreach_set("co", np.asarray(arr, dtype=np.float64).reshape(-1))
    obj.data.update()


def normals(obj):
    me = obj.data
    a = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("normal", a)
    return a.reshape(-1, 3)


def shade_smooth(obj, sharp_angle=None):
    me = obj.data
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    if sharp_angle is not None:
        me.set_sharp_from_angle(angle=math.radians(sharp_angle))
    me.update()


def weights_matrix(obj, names):
    """(n_verts, len(names)) weights of the given vertex groups."""
    idx = {g.name: g.index for g in obj.vertex_groups}
    cols = {idx[n]: j for j, n in enumerate(names) if n in idx}
    W = np.zeros((len(obj.data.vertices), len(names)))
    for v in obj.data.vertices:
        for g in v.groups:
            j = cols.get(g.group)
            if j is not None:
                W[v.index, j] = g.weight
    return W


def all_weights(obj):
    """Per vertex {group name: weight}."""
    names = {g.index: g.name for g in obj.vertex_groups}
    return [{names[g.group]: g.weight for g in v.groups if g.weight > 0} for v in obj.data.vertices]


def set_single_weight(obj, group, verts=None):
    """Every (or the given) vertex weighted 1.0 to one group, nothing else."""
    for g in list(obj.vertex_groups):
        obj.vertex_groups.remove(g)
    vg = obj.vertex_groups.new(name=group)
    ids = list(range(len(obj.data.vertices))) if verts is None else list(verts)
    vg.add(ids, 1.0, 'REPLACE')
    return vg


def limit_and_normalize(obj, limit=4):
    select_only(obj)
    if not obj.vertex_groups:
        return
    bpy.ops.object.vertex_group_clean(group_select_mode='ALL', limit=0.01)
    bpy.ops.object.vertex_group_limit_total(group_select_mode='ALL', limit=limit)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode='ALL', lock_active=False)


def transfer_weights(target, source, mapping='POLYINTERP_NEAREST'):
    """Copy skin weights from source to target through a Data Transfer modifier, then apply it."""
    for g in list(target.vertex_groups):
        target.vertex_groups.remove(g)
    mod = target.modifiers.new("WeightTransfer", 'DATA_TRANSFER')
    mod.object = source
    mod.use_vert_data = True
    mod.data_types_verts = {'VGROUP_WEIGHTS'}
    mod.vert_mapping = mapping
    mod.layers_vgroup_select_src = 'ALL'
    mod.layers_vgroup_select_dst = 'NAME'
    select_only(target)
    bpy.ops.object.datalayout_transfer(modifier=mod.name)
    # The modifier must run before any other (an armature would deform the copy first).
    while target.modifiers[0].name != mod.name:
        bpy.ops.object.modifier_move_up(modifier=mod.name)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    limit_and_normalize(target)


def add_armature(obj, arm):
    for m in list(obj.modifiers):
        if m.type == 'ARMATURE':
            obj.modifiers.remove(m)
    mod = obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm
    obj.parent = arm
    obj.matrix_parent_inverse = arm.matrix_world.inverted()
    return mod


def join(objs, name):
    objs = [o for o in objs if o is not None]
    select_only(*objs)
    bpy.ops.object.join()
    out = bpy.context.view_layer.objects.active
    out.name = name
    out.data.name = name
    return out


def delete_faces(obj, face_mask):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    doomed = [f for f in bm.faces if face_mask[f.index]]
    bmesh.ops.delete(bm, geom=doomed, context='FACES')
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def smooth_boundary(obj, iterations=12, lam=0.5, mu=-0.53, max_move=0.01):
    """Straighten the open edges (a garment's neckline, sleeve ends, hems): the zigzag a face selection leaves
    is smoothed along each boundary loop with Taubin's shrink-free filter; interior vertices stay put."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    nbrs = {}
    for e in bm.edges:
        if e.is_boundary:
            a, b = e.verts[0].index, e.verts[1].index
            nbrs.setdefault(a, []).append(b)
            nbrs.setdefault(b, []).append(a)
    bm.free()
    ids = np.array([i for i, n in nbrs.items() if len(n) == 2], dtype=np.int64)
    if not len(ids):
        return 0
    nb = np.array([nbrs[i] for i in ids], dtype=np.int64)
    P = positions(obj)
    P0 = P[ids].copy()
    # Only real loops: a vertex whose two boundary neighbours are far apart sits at a pinch; leave it.
    ok = np.linalg.norm(P[nb[:, 0]] - P[nb[:, 1]], axis=1) < 0.06
    for _ in range(iterations):
        for f in (lam, mu):
            avg = 0.5 * (P[nb[:, 0]] + P[nb[:, 1]])
            P[ids] = np.where(ok[:, None], P[ids] + f * (avg - P[ids]), P[ids])
    # Straightening moves an edge a centimetre at most.
    d = P[ids] - P0
    n = np.linalg.norm(d, axis=1, keepdims=True)
    P[ids] = P0 + d * np.minimum(1.0, max_move / np.maximum(n, 1e-9))
    set_positions(obj, P)
    return len(ids)


def despike(obj, rounds=3, factor=2.0, floor=0.012):
    """Pull back any vertex that sticks out of its neighbourhood: further from the mean of its neighbours
    than `factor` times its mean edge length (and more than `floor` metres)."""
    me = obj.data
    E = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", E)
    E = E.reshape(-1, 2)
    n = len(me.vertices)
    fixed = 0
    for _ in range(rounds):
        P = positions(obj)
        acc = np.zeros((n, 3))
        cnt = np.zeros(n)
        np.add.at(acc, E[:, 0], P[E[:, 1]])
        np.add.at(acc, E[:, 1], P[E[:, 0]])
        np.add.at(cnt, E[:, 0], 1)
        np.add.at(cnt, E[:, 1], 1)
        L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
        el = np.zeros(n)
        np.add.at(el, E[:, 0], L)
        np.add.at(el, E[:, 1], L)
        has = cnt > 0
        mean = np.where(has[:, None], acc / np.maximum(cnt, 1)[:, None], P)
        mel = np.where(has, el / np.maximum(cnt, 1), 0)
        d = np.linalg.norm(P - mean, axis=1)
        # A spike's own long edges inflate its mean edge length: compare with the neighbours' median instead.
        nb_el = np.zeros(n)
        np.add.at(nb_el, E[:, 0], mel[E[:, 1]])
        np.add.at(nb_el, E[:, 1], mel[E[:, 0]])
        nb_el = np.where(has, nb_el / np.maximum(cnt, 1), 0)
        bad = has & (d > np.maximum(factor * nb_el, floor))
        if not bad.any():
            break
        P[bad] = mean[bad]
        set_positions(obj, P)
        fixed += int(bad.sum())
    if fixed:
        log(f"despiked {obj.name}: {fixed} vertices")
    return fixed


def decimate_to(obj, target_tris, symmetric=True):
    """Collapse-decimate a mesh (no other modifiers on it) to about target_tris; vertex groups follow."""
    tris = tri_count(obj)
    if tris <= target_tris:
        return tris
    mod = obj.modifiers.new("Decimate", 'DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = target_tris / tris
    mod.use_symmetry = symmetric
    mod.symmetry_axis = 'X'
    mod.use_collapse_triangulate = True
    apply_modifier(obj, mod.name)
    out = tri_count(obj)
    log(f"decimated {obj.name}: {tris} -> {out} tris")
    return out


def faces_of_vertices(obj, vmask, mode="all"):
    """Face mask: faces whose vertices are all (or any) in vmask."""
    out = np.zeros(len(obj.data.polygons), dtype=bool)
    for p in obj.data.polygons:
        vals = [vmask[i] for i in p.vertices]
        out[p.index] = all(vals) if mode == "all" else any(vals)
    return out


def extract_faces(obj, face_mask, name):
    """A new object holding a copy of the masked faces (vertex groups and UVs kept)."""
    new = duplicate(obj, name)
    for m in list(new.modifiers):
        new.modifiers.remove(m)
    delete_faces(new, ~np.asarray(face_mask))
    return new


# ---------- materials and images ----------

def new_material(name):
    m = bpy.data.materials.get(name)
    if m:
        bpy.data.materials.remove(m)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    return m


def set_material(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def new_image(name, size, alpha=False, non_color=False, float_buffer=False):
    img = bpy.data.images.get(name)
    if img:
        bpy.data.images.remove(img)
    w, h = (size, size) if isinstance(size, int) else size
    img = bpy.data.images.new(name, w, h, alpha=alpha, float_buffer=float_buffer)
    if non_color:
        img.colorspace_settings.name = 'Non-Color'
    return img


def image_array(img):
    w, h = img.size
    a = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(h, w, 4)


def set_image_array(img, arr):
    h, w = arr.shape[:2]
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), dtype=arr.dtype)], axis=2)
    img.pixels.foreach_set(np.ascontiguousarray(arr, dtype=np.float32).reshape(-1))
    img.update()


def save_png(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()


def image_from_array(name, arr, path=None, non_color=False):
    h, w = arr.shape[:2]
    img = new_image(name, (w, h), non_color=non_color)
    set_image_array(img, arr)
    if path:
        save_png(img, path)
        # Reload from disk so the image is what was written (8 bits per channel).
        img.filepath = path
        img.source = 'FILE'
        img.reload()
    return img


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float32), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def resize(arr, size):
    """Box/bilinear resize of an (h, w, c) array to (size, size) or (h, w)."""
    h, w = arr.shape[:2]
    th, tw = (size, size) if isinstance(size, int) else size
    if h % th == 0 and w % tw == 0:
        fy, fx = h // th, w // tw
        return arr.reshape(th, fy, tw, fx, -1).mean(axis=(1, 3))
    ys = (np.arange(th) + 0.5) * h / th - 0.5
    xs = (np.arange(tw) + 0.5) * w / tw - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = (ys - y0)[:, None, None]
    fx = (xs - x0)[None, :, None]
    a = arr[y0][:, x0] * (1 - fx) + arr[y0][:, x1] * fx
    b = arr[y1][:, x0] * (1 - fx) + arr[y1][:, x1] * fx
    return a * (1 - fy) + b * fy


# ---------- noise (numpy, deterministic) ----------

def _hash3(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float64) / float(0xFFFFFF)


def value_noise3(p, seed=0):
    """Smooth value noise in [0, 1] at points p (..., 3)."""
    p = np.asarray(p, dtype=np.float64)
    i = np.floor(p).astype(np.int64)
    f = p - i
    u = f * f * (3 - 2 * f)
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[..., 0] if dx else 1 - u[..., 0]) * (u[..., 1] if dy else 1 - u[..., 1]) * (u[..., 2] if dz else 1 - u[..., 2])
                out = out + w * _hash3(i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz, seed)
    return out


def fbm3(p, octaves=4, seed=0, lacunarity=2.0, gain=0.5):
    p = np.asarray(p, dtype=np.float64)
    amp, tot, norm = 1.0, 0.0, 0.0
    for o in range(octaves):
        tot = tot + amp * value_noise3(p, seed + o * 17)
        norm += amp
        amp *= gain
        p = p * lacunarity
    return tot / norm


def value_noise1(x, seed=0):
    x = np.asarray(x, dtype=np.float64)
    i = np.floor(x).astype(np.int64)
    f = x - i
    u = f * f * (3 - 2 * f)
    z = np.zeros_like(i)
    return _hash3(i, z, z, seed) * (1 - u) + _hash3(i + 1, z, z, seed) * u


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def rng(seed):
    return random.Random(seed)


def look_at(cam, target):
    d = Vector(target) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def matrix_world_point(obj, co):
    return obj.matrix_world @ Vector(co)
