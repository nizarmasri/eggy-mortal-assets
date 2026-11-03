"""Fight gear built from the body itself, so it fits every body the presets make.

Tight gear (wraps, tape, shorts, rash guards) duplicates body faces chosen by bone weights, sits a few
millimetres off the skin and keeps the skin's weights. Loose gear (gi pants, belts, sashes) is shaped away
from the body and takes its weights by Data Transfer (nearest face, interpolated) from the body or from the
garment underneath. Each piece carries a corner attribute `param` for its procedural pattern (shaders.py).
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

import mask as egg_mask
import util
from util import log, smoothstep

ARM_BONES = ["clavicle", "upperarm", "lowerarm", "hand", "thumb_01", "thumb_02", "thumb_03", "index_01", "index_02", "index_03",
             "middle_01", "middle_02", "middle_03", "ring_01", "ring_02", "ring_03", "pinky_01", "pinky_02", "pinky_03"]
LEG_BONES = ["thigh", "calf", "foot", "ball"]


def sided(names, sides=("l", "r")):
    return [f"{n}_{s}" for s in sides for n in names]


def set_param(obj, fn, name="param"):
    """Corner attribute from fn(position (n,3), vertex index (n,), face index) -> (n, 3)."""
    me = obj.data
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
    attr = me.attributes.new(name, 'FLOAT_VECTOR', 'CORNER')
    co = util.positions(obj)
    lv = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", lv)
    out = np.zeros((len(me.loops), 3))
    for p in me.polygons:
        loops = np.arange(p.loop_start, p.loop_start + p.loop_total)
        out[loops] = fn(co[lv[loops]], lv[loops], p.index)
    attr.data.foreach_set("vector", out.reshape(-1))


def level_top(obj, reach=0.12, name="param", cell=0.02):
    """Bands measured down from a piece's top (waistbands, boot cuffs) should follow its top edge, not a
    horizontal plane. The top edge over each spot is the highest vertex in its vertical column (within two
    cells round it): near it, the pattern height (y) is shifted so every column's top reads as the piece's
    highest point, fading out over `reach` metres below. Works for open and closed tops and for pieces in
    several parts (two boots)."""
    me = obj.data
    attr = me.attributes.get(name)
    if attr is None or not len(me.vertices):
        return
    P = util.positions(obj)
    key = np.floor(P[:, :2] / cell).astype(np.int64)
    top = {}
    for (kx, ky), z in zip(map(tuple, key), P[:, 2]):
        if z > top.get((kx, ky), -1e9):
            top[(kx, ky)] = z
    local = np.empty(len(P))
    for i, (kx, ky) in enumerate(map(tuple, key)):
        best = -1e9
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                z = top.get((kx + dx, ky + dy))
                if z is not None and z > best:
                    best = z
        local[i] = best
    below = local - P[:, 2]
    gap = P[:, 2].max() - local
    # Only columns whose top is the piece's top edge (a few centimetres of dip at most): over a boot's foot
    # the column top is the instep, which is no edge.
    shift = gap * np.clip(1 - below / reach, 0, 1) * (gap < 0.08)
    lv = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", lv)
    vec = np.empty(len(me.loops) * 3)
    attr.data.foreach_get("vector", vec)
    vec = vec.reshape(-1, 3)
    vec[:, 1] += shift[lv]
    attr.data.foreach_set("vector", vec.reshape(-1))


def unwrap_angle(a):
    """Angles of one face's corners made continuous (no jump across +-pi inside a face)."""
    ref = a[0]
    return ref + (a - ref + math.pi) % (2 * math.pi) - math.pi


def bvh(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    return BVHTree.FromObject(obj, dg)


def closest_on_polyline(p, pts):
    """Closest point on a polyline for each p (n,3): returns (point, tangent, arc length)."""
    best = np.full(len(p), 1e9)
    bp = np.zeros_like(p)
    bt = np.zeros_like(p)
    bs = np.zeros(len(p))
    acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L = np.linalg.norm(d)
        t = np.clip(((p - a) @ d) / (L * L), 0, 1)
        q = a + t[:, None] * d
        dist = np.linalg.norm(p - q, axis=1)
        upd = dist < best
        best[upd] = dist[upd]
        bp[upd] = q[upd]
        bt[upd] = d / L
        bs[upd] = acc + t[upd] * L
        acc += L
    return bp, bt, bs


def lateral_param(co, legs, crotch):
    """Pattern coordinates for legwear: x = how far round to the outer side (1 outer, -1 inner, about the leg
    below the crotch and the body above it), y = height, z = how far round to the front (1 front, -1 back)."""
    out = np.zeros((len(co), 3))
    out[:, 1] = co[:, 2]
    for sign, pts in legs.items():
        sel = (co[:, 0] * sign) >= 0
        q = co[sel]
        axis_p, axis_t, _ = closest_on_polyline(q, pts)
        r = q - axis_p
        r[:, 2] = 0
        r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-6)
        rb = q.copy()
        rb[:, 2] = 0
        rb[:, 1] -= np.median(co[:, 1])
        rb /= np.maximum(np.linalg.norm(rb, axis=1, keepdims=True), 1e-6)
        above = (q[:, 2] > crotch + 0.04)[:, None]
        d = np.where(above, rb, r)
        out[sel, 0] = d[:, 0] * sign
        out[sel, 2] = -d[:, 1]
    return out


def crotch_height(body):
    P = util.positions(body)
    mid = (np.abs(P[:, 0]) < 0.012) & (P[:, 2] > 0.3 * P[:, 2].max()) & (P[:, 2] < 0.6 * P[:, 2].max())
    return float(P[mid, 2].min())


# ---------- loose pants ----------

def loose_pants(body, arm, m, spec, name="pants"):
    """Straight, roomy trousers (gi, abada, kung-fu, dobok): the body's legs and hips copied, rounded into
    tubes that hang wider towards the hem, with a rolled hem and a waistband."""
    Hm = m["height"]
    k = Hm / 1.78
    waist_z = spec.get("waist", 0.575) * Hm
    hem_z = spec.get("hem", 0.085) * Hm
    loose = spec.get("loose", 1.0)
    P = util.positions(body)
    N = util.normals(body)
    arms = util.weights_matrix(body, sided(ARM_BONES)).sum(axis=1)
    sel = (P[:, 2] <= waist_z) & (P[:, 2] >= hem_z) & (arms < 0.05)
    faces = util.faces_of_vertices(body, sel)
    obj = util.extract_faces(body, faces, name)
    P = util.positions(obj)
    N = util.normals(obj)
    crotch = crotch_height(body)
    out = P.copy()
    legs = {}
    for s, sign in (("l", 1), ("r", -1)):
        pts = np.array([m[f"thigh_{s}"], m[f"calf_{s}"], m[f"foot_{s}"]], dtype=np.float64)
        legs[sign] = pts
    knee_z = float(legs[1][1][2])
    for sign, pts in legs.items():
        side = (P[:, 0] * sign) > 0
        q = P[side]
        axis_p, axis_t, _ = closest_on_polyline(q, pts)
        r = q - axis_p
        r -= (r * axis_t).sum(axis=1)[:, None] * axis_t
        rl = np.linalg.norm(r, axis=1)
        rd = r / np.maximum(rl, 1e-6)[:, None]
        z = q[:, 2]
        # Ease grows from the crotch to the hem; a minimum tube radius keeps the shins from showing their shape.
        f_knee = np.clip((crotch - z) / max(crotch - knee_z, 1e-3), 0, 1)
        f_hem = np.clip((knee_z - z) / max(knee_z - hem_z, 1e-3), 0, 1)
        ease = (0.016 + 0.010 * f_knee + 0.008 * f_hem) * k * loose
        rmin = (0.072 * f_knee + 0.012 * f_hem + (0.082 * (1 - f_knee))) * k * loose
        newr = rl + ease
        newr = newr + np.maximum(0, rmin - newr) * 0.85
        flare = spec.get("flare", 0.0)
        if flare:
            newr += flare * k * np.clip((crotch - z) / max(crotch - hem_z, 1e-3), 0, 1) ** 1.3
        leg_pos = axis_p + rd * newr[:, None]
        # Above the crotch the hips are inflated along their normals instead.
        hip_pos = q + N[side] * (0.010 + 0.012 * np.clip((waist_z - z) / max(waist_z - crotch, 1e-3), 0, 1))[:, None] * k * loose
        blend = smoothstep(crotch - 0.03, crotch + 0.07, z)
        pos = leg_pos * (1 - blend[:, None]) + hip_pos * blend[:, None]
        # Never cross the middle: the two legs meet in a seam there.
        low = z < crotch + 0.08
        bad = low & (pos[:, 0] * sign < 0.004)
        pos[bad, 0] = sign * 0.004
        out[side] = pos
    util.set_positions(obj, out)
    # Calm the surface: loose cloth should not show abs or calves.
    sm = obj.modifiers.new("Smooth", 'SMOOTH')
    sm.factor = 0.7
    sm.iterations = 6
    util.select_only(obj)
    while obj.modifiers[0].name != sm.name:
        bpy.ops.object.modifier_move_up(modifier=sm.name)
    bpy.ops.object.modifier_apply(modifier=sm.name)
    # Then the cloth's own shape: long drape folds down the legs, bunching above the knee and at the hem.
    P = util.positions(obj)
    Nn = util.normals(obj)
    disp = np.zeros(len(P))
    for sign, pts in legs.items():
        side = (P[:, 0] * sign) > 0
        q = P[side]
        axis_p, axis_t, _ = closest_on_polyline(q, pts)
        r = q - axis_p
        ang = np.arctan2(r[:, 1], r[:, 0])
        z = q[:, 2]
        legness = 1 - smoothstep(crotch - 0.02, crotch + 0.08, z)
        drape = util.fbm3(np.stack([np.cos(ang) * 1.6, np.sin(ang) * 1.6, z * 2.2 + sign * 5], axis=1), 3, seed=4) - 0.5
        drape *= 0.030 * k * (0.35 + 0.9 * np.clip((knee_z + 0.15 - z) / 0.4, 0, 1)) * legness
        knee = np.exp(-((z - (knee_z + 0.03)) / 0.07) ** 2) * np.sin(z * 120 + 3 * util.fbm3(np.stack([np.cos(ang), np.sin(ang), z * 3], axis=1), 2, seed=8))
        hem = np.exp(-((z - (hem_z + 0.05)) / 0.05) ** 2) * np.sin(z * 150 + 4 * util.fbm3(np.stack([np.cos(ang) * 2, np.sin(ang) * 2, z * 2], axis=1), 2, seed=9))
        disp[side] = drape + (0.006 * knee + 0.005 * hem) * k * legness
    hips = P[:, 2] > crotch
    disp[hips] += (util.fbm3(P[hips] * np.array([12, 12, 4.0]), 2, seed=6) - 0.5) * 0.008 * k
    util.set_positions(obj, P + Nn * disp[:, None])
    # Rolled hem and waistband: extrude the open edges inward, so the cloth reads thick from any side.
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    boundary = [e for e in bm.edges if e.is_boundary]
    hem_edges = [e for e in boundary if all(v.co.z < waist_z - 0.1 for v in e.verts)]
    top_edges = [e for e in boundary if e not in hem_edges]
    for edges, up in ((hem_edges, 1), (top_edges, -1)):
        if not edges:
            continue
        verts = {v for e in edges for v in e.verts}
        # Even out the hem height first (the face selection leaves it ragged).
        if up > 0:
            zs = [v.co.z for v in verts]
            zt = float(np.median(zs))
            for v in verts:
                v.co.z = zt
        res = bmesh.ops.extrude_edge_only(bm, edges=edges)
        new_verts = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
        for v in new_verts:
            c = Vector((0, 0, v.co.z))
            if up > 0:
                side = 1 if v.co.x > 0 else -1
                pts = legs[side]
                ax = Vector(pts[2] if v.co.z < pts[1][2] else pts[1])
                c = Vector((ax.x, ax.y, v.co.z))
            d = (v.co - c)
            d.z = 0
            if d.length > 1e-6:
                v.co -= d.normalized() * 0.005
            v.co.z += 0.018 * up
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    util.shade_smooth(obj)
    set_param(obj, lambda co, vi, fi: lateral_param(co, legs, crotch))
    level_top(obj)
    log(f"{name}: {util.tri_count(obj)} tris (crotch {crotch:.3f}, knee {knee_z:.3f}, hem {hem_z:.3f})")
    return obj, {"waist_z": waist_z, "hem_z": hem_z, "knee_z": knee_z, "crotch": crotch}


# ---------- belt ----------

def ring_on(target_bvh, centre, z, n=72, offset=0.003, radius=0.6):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        d = Vector((math.cos(a), math.sin(a), 0))
        origin = Vector((centre[0], centre[1], z)) + d * radius
        hit = target_bvh.ray_cast(origin, -d, radius * 1.2)
        if hit[0] is None:
            pts.append(Vector((centre[0], centre[1], z)) + d * 0.15)
        else:
            pts.append(hit[0] + hit[1] * offset)
    return pts


def belt(pants, arm, info, spec, name="belt"):
    """A martial-arts belt: a band around the waist over the trousers, a square knot at the front and two
    tails hanging from it."""
    width = spec.get("width", 0.042)
    thick = spec.get("thick", 0.005)
    zc = info["waist_z"] - width * 0.5 - spec.get("drop", 0.012)
    tree = bvh(pants)
    P = util.positions(pants)
    near = np.abs(P[:, 2] - zc) < 0.03
    centre = P[near].mean(axis=0)
    top = ring_on(tree, centre, zc + width / 2, offset=0.0035)
    bot = ring_on(tree, centre, zc - width / 2, offset=0.0035)
    n = len(top)
    bm = bmesh.new()
    layers = []
    for lay_off in (0.0, thick):
        row_t, row_b = [], []
        for i in range(n):
            out_t = Vector((top[i].x - centre[0], top[i].y - centre[1], 0)).normalized()
            out_b = Vector((bot[i].x - centre[0], bot[i].y - centre[1], 0)).normalized()
            row_t.append(bm.verts.new(top[i] + out_t * lay_off))
            row_b.append(bm.verts.new(bot[i] + out_b * lay_off))
        layers.append((row_t, row_b))
    (it, ib), (ot, ob) = layers
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((ot[i], ot[j], ob[j], ob[i]))  # outer
        bm.faces.new((it[j], it[i], ib[i], ib[j]))  # inner
        bm.faces.new((it[i], it[j], ot[j], ot[i]))  # top
        bm.faces.new((ib[j], ib[i], ob[i], ob[j]))  # bottom
    # The knot: a rounded block at the front.
    front = min(range(n), key=lambda i: top[i].y + bot[i].y)
    fc = (top[front] + bot[front]) / 2
    knot = bmesh.ops.create_cube(bm, size=1.0)
    kv = knot["verts"]
    kw, kh, kd = width * 1.25, width * 1.05, 0.022
    for v in kv:
        v.co = Vector((v.co.x * kw, v.co.y * kd, v.co.z * kh)) + fc + Vector((0, -kd * 0.45, 0))
    bmesh.ops.bevel(bm, geom=[e for e in bm.edges if all(v in kv for v in e.verts)], offset=0.006, segments=2, affect='EDGES')
    # Two tails.
    tail_len = spec.get("tail", 0.24)
    tail_frames = []
    for sgn in (1, -1):
        start = fc + Vector((sgn * width * 0.35, -kd * 0.75, -kh * 0.35))
        direction = Vector((sgn * 0.28, -0.05, -1)).normalized()
        tail_frames.append((tuple(start), tuple(direction), tuple(Vector((direction.z, 0, -direction.x)).normalized())))
        segs = 8
        rows = []
        for s in range(segs + 1):
            t = s / segs
            c = start + direction * tail_len * t + Vector((0, -0.012 * math.sin(t * math.pi), 0))
            # Hang in front of the trousers.
            hit = tree.ray_cast(c + Vector((0, -0.3, 0)), Vector((0, 1, 0)), 0.6)
            if hit[0] is not None and c.y > hit[0].y - 0.012:
                c.y = hit[0].y - 0.012
            side = Vector((direction.z, 0, -direction.x)).normalized() * (width * 0.5)
            fwd = Vector((0, -thick, 0))
            rows.append([bm.verts.new(c - side), bm.verts.new(c + side), bm.verts.new(c + side + fwd), bm.verts.new(c - side + fwd)])
        for a, b in zip(rows[:-1], rows[1:]):
            for q in range(4):
                r_ = (q + 1) % 4
                bm.faces.new((a[q], a[r_], b[r_], b[q]))
        bm.faces.new(rows[0])
        bm.faces.new(list(reversed(rows[-1])))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    obj = util.bmesh_object(name, bm)
    util.shade_smooth(obj, sharp_angle=50)

    # Pattern coordinates: along the belt (metres) and across it (0..1), per piece.
    fcn = np.array(fc)

    def param(co, vi, fi):
        d = co - np.array(centre)
        ang = np.arctan2(d[:, 1], d[:, 0])
        along = unwrap_angle(ang) * 0.17
        across = np.clip((co[:, 2] - (zc - width / 2)) / width, 0, 1)
        for (start, direction, side_n) in tail_frames:
            st, dr, sn = np.array(start), np.array(direction), np.array(side_n)
            sel = (co[:, 2] < zc - width * 0.55) & (((co[:, 0] - fcn[0]) * dr[0]) > -0.004)
            if sel.any():
                q = co[sel] - st
                along[sel] = 2.0 + q @ dr
                across[sel] = np.clip(q @ sn / width + 0.5, 0, 1)
        knot_sel = (np.abs(co[:, 2] - fcn[2]) < kh * 0.62) & (co[:, 1] < fcn[1] - 0.003) & (np.abs(co[:, 0] - fcn[0]) < kw * 0.62)
        across[knot_sel] = 0.5 + 0.35 * (co[knot_sel, 2] - fcn[2]) / kh
        along[knot_sel] = 3.0 + co[knot_sel, 0]
        return np.stack([along, across, np.full(len(co), 0.3)], axis=1)
    set_param(obj, param)
    util.transfer_weights(obj, pants)
    # Tails and knot hang from the pelvis alone.
    W = util.positions(obj)
    tail = (W[:, 2] < zc - width * 0.6) | (W[:, 1] < centre[1] - 0.05)
    _reweight(obj, np.where(tail)[0], "pelvis")
    log(f"{name}: {util.tri_count(obj)} tris")
    return obj


def _reweight(obj, ids, group):
    ids = [int(i) for i in ids]
    for g in obj.vertex_groups:
        g.remove(ids)
    vg = obj.vertex_groups.get(group) or obj.vertex_groups.new(name=group)
    vg.add(ids, 1.0, 'REPLACE')


# ---------- tight wraps ----------

def wraps(body, arm, m, spec, name="wraps"):
    """Hand wraps or tape: the hands up to the first knuckles and the wrist, a few millimetres proud of the
    skin, with a rim so the edge reads as a layer."""
    length = spec.get("forearm", 0.07)
    offset = spec.get("offset", 0.0018)
    P = util.positions(body)
    vmask = np.zeros(len(P), dtype=bool)
    params = {}
    for s in ("l", "r"):
        W = util.weights_matrix(body, [f"hand_{s}", f"lowerarm_{s}", f"index_01_{s}", f"middle_01_{s}", f"ring_01_{s}", f"pinky_01_{s}", f"thumb_01_{s}",
                                       f"index_02_{s}", f"middle_02_{s}", f"ring_02_{s}", f"pinky_02_{s}", f"thumb_02_{s}",
                                       f"index_03_{s}", f"middle_03_{s}", f"ring_03_{s}", f"pinky_03_{s}", f"thumb_03_{s}"])
        wrist = np.array(m[f"hand_{s}"])
        elbow = np.array(m[f"lowerarm_{s}"])
        axis = wrist - elbow
        axis /= np.linalg.norm(axis)
        along = (P - wrist) @ axis  # metres past the wrist (towards the fingers)
        near_wrist = (along > -length) & (along < 0.02)
        # Knuckles: only close to the base of each finger, so the fingers themselves stay bare.
        kn = np.zeros(len(P), dtype=bool)
        for f in ("index", "middle", "ring", "pinky"):
            kh = np.array(m.get(f"{f}_01_{s}", m[f"hand_{s}_tail"]))
            kn |= np.linalg.norm(P - kh, axis=1) < spec.get("knuckle_r", 0.016)
        score = W[:, 0] + W[:, 2:6].sum(axis=1) * kn * spec.get("knuckles", 1.0) + W[:, 6] * 0.6 + W[:, 1] * near_wrist
        far = W[:, 7:].sum(axis=1)
        inc = (score >= 0.5) & (far < spec.get("fingers", 0.3)) & (along > -length)
        vmask |= inc
        params[s] = (wrist, axis)
    faces = util.faces_of_vertices(body, vmask)
    obj = util.extract_faces(body, faces, name)
    Pn = util.positions(obj) + util.normals(obj) * offset
    util.set_positions(obj, Pn)
    sol = obj.modifiers.new("Rim", 'SOLIDIFY')
    sol.thickness_clamp = 1.0  # no spikes at sharp open corners
    sol.thickness = offset
    sol.offset = -1
    sol.use_rim_only = True
    sol.use_even_offset = True
    util.select_only(obj)
    while obj.modifiers[0].name != sol.name:
        bpy.ops.object.modifier_move_up(modifier=sol.name)
    bpy.ops.object.modifier_apply(modifier=sol.name)
    util.shade_smooth(obj)

    def param(co, vi, fi):
        out = np.zeros((len(co), 3))
        for s, (wrist, axis) in params.items():
            sel = (co[:, 0] > 0) == (s == "l")
            if not sel.any():
                continue
            d = co[sel] - wrist
            u = d @ axis
            ref = np.cross(axis, [0, 0, 1.0])
            ref /= np.linalg.norm(ref)
            ref2 = np.cross(axis, ref)
            ang = unwrap_angle(np.arctan2(d @ ref2, d @ ref))
            out[sel] = np.stack([u, ang * 0.035, np.full(sel.sum(), 0.5 if s == "l" else 0.8)], axis=1)
        return out
    set_param(obj, param)
    log(f"{name}: {util.tri_count(obj)} tris")
    return obj


def _mirror(p):
    return (-p[0], p[1], p[2])


# ---------- hair ----------

def short_hair(body, arm, m, egg, spec, name="hair"):
    """A short hair shell on the scalp, combed towards a topknot at the crown's back, and the knot itself."""
    g = body.vertex_groups.get("scalp")
    P = util.positions(body)
    ids = np.zeros(len(P), dtype=bool)
    if g is not None:
        for v in body.data.vertices:
            if any(x.group == g.index for x in v.groups):
                ids[v.index] = True
    faces = util.faces_of_vertices(body, ids, mode=spec.get("grow", "all"))
    shell = util.extract_faces(body, faces, name)
    # A clean hairline: relax the boundary along itself, then put it back on the scalp.
    head_tree = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    bm = bmesh.new()
    bm.from_mesh(shell.data)
    edge_verts = [v for v in bm.verts if v.is_boundary]
    for _ in range(spec.get("hairline_smooth", 6)):
        new_co = {}
        for v in edge_verts:
            nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            if len(nb) == 2:
                new_co[v] = v.co * 0.4 + (nb[0].co + nb[1].co) * 0.3
        for v, c in new_co.items():
            loc, nrm, _, _ = head_tree.find_nearest(c)
            v.co = loc if loc is not None else c
    bm.to_mesh(shell.data)
    bm.free()
    shell.data.update()
    # Thickness tapers to almost nothing at the hairline, so the shell melts into the painted fade on the
    # scalp instead of ending in a step.
    bm = bmesh.new()
    bm.from_mesh(shell.data)
    bm.verts.ensure_lookup_table()
    ring = {v.index: 0 for v in bm.verts if v.is_boundary}
    frontier = list(ring)
    while frontier:
        nxt = []
        for i in frontier:
            for e in bm.verts[i].link_edges:
                j = e.other_vert(bm.verts[i]).index
                if j not in ring:
                    ring[j] = ring[i] + 1
                    nxt.append(j)
        frontier = nxt
    bm.free()
    S = util.positions(shell)
    shell_n = util.normals(shell)
    t = spec.get("thickness", 0.0035)
    taper = np.array([min(1.0, (ring.get(i, 9) + 0.15) / 2.5) for i in range(len(S))])
    # Clumps combed front to back: a lumpy, uneven surface instead of a smooth helmet.
    clump = util.fbm3(S * np.array([70.0, 22.0, 70.0]), 3, seed=31)
    t_v = t * (0.55 + 0.9 * clump) * taper
    util.set_positions(shell, S + shell_n * t_v[:, None])
    util.shade_smooth(shell)
    util.set_single_weight(shell, "head")
    if spec.get("no_knot"):
        cen = np.array(egg.C) + np.array([0, 0.05, 0.05])

        def plain(co, vi, fi):
            d = co - cen
            ang = unwrap_angle(np.arctan2(d[:, 0], -d[:, 2]))
            return np.stack([ang * 0.05, np.linalg.norm(d, axis=1), np.full(len(co), 0.2)], axis=1)
        set_param(shell, plain)
        log(f"{name}: shell {util.tri_count(shell)} tris")
        return shell, None, None
    # The knot: behind the top of the mask, at the crown's back (or low at the nape: a bun).
    S = util.positions(body)[ids]
    back = S[S[:, 1] > egg.C[1] + 0.035]
    lift = {"crown": 0.35, "nape": -2.5, "top": 0.05}[spec.get("knot_at", "crown")]
    top = back[np.argmax(back[:, 2] + lift * back[:, 1])] if spec.get("knot_at", "crown") != "nape" else back[np.argmax(back[:, 1] - 0.6 * np.abs(back[:, 2] - (egg.C[2] - 0.02)))]
    up = Vector(spec.get("knot_dir", (0, 0.55, 0.85))).normalized()
    knot_c = Vector(top) + up * spec.get("knot_lift", 0.016)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=1.0)
    rx, ry, rz = spec.get("knot", (0.019, 0.03, 0.017))
    rot = up.to_track_quat('Y', 'Z').to_matrix()
    for v in bm.verts:
        v.co = rot @ Vector((v.co.x * rx, v.co.y * ry, v.co.z * rz)) + knot_c
    knot = util.bmesh_object(name + "_knot", bm)
    util.shade_smooth(knot)
    util.set_single_weight(knot, "head")
    # A cord tied round the knot's base.
    bm = bmesh.new()
    tie_r = 1.12 * max(rx, rz) * math.sqrt(1 - 0.55 ** 2)
    bmesh.ops.create_cone(bm, cap_ends=False, segments=12, radius1=tie_r, radius2=tie_r, depth=0.006)
    for v in bm.verts:
        v.co = rot @ Vector((v.co.x, v.co.z - ry * 0.55, v.co.y)) + knot_c
    tie = util.bmesh_object(name + "_tie", bm)
    util.set_single_weight(tie, "head")
    centre = np.array(knot_c)

    def hair_param(co, vi, fi):
        d = co - centre
        d_local = d @ np.array(rot)
        ang = unwrap_angle(np.arctan2(d_local[:, 0], d_local[:, 2]))
        dist = np.linalg.norm(d, axis=1)
        return np.stack([ang * 0.05, dist, np.full(len(co), 0.2)], axis=1)
    set_param(shell, hair_param)
    set_param(knot, lambda co, vi, fi: np.stack([unwrap_angle(np.arctan2(((co - centre) @ np.array(rot))[:, 0], ((co - centre) @ np.array(rot))[:, 2])) * 0.02, ((co - centre) @ np.array(rot))[:, 1], np.full(len(co), 0.6)], axis=1))
    set_param(tie, lambda co, vi, fi: np.stack([unwrap_angle(np.arctan2(((co - centre) @ np.array(rot))[:, 0], ((co - centre) @ np.array(rot))[:, 2])) * 0.012, ((co - centre) @ np.array(rot))[:, 1] * 30, np.full(len(co), 0.9)], axis=1))
    log(f"{name}: shell {util.tri_count(shell)} + knot {util.tri_count(knot)} + tie {util.tri_count(tie)} tris")
    return shell, knot, tie


# ---------- the mask's strap ----------

def strap(surfaces, egg, th, name="strap", width=0.018, thick=0.0025):
    """A leather strap from under one edge of the mask, round the back of the head, to the other edge."""
    trees = [bvh(s) for s in surfaces]
    R = egg.R
    C = Vector(egg.C.tolist())
    # Run in the egg's tilted plane through the slits' latitude.
    zl = egg.h * math.cos(th)
    pts_top, pts_bot = [], []
    start = egg.phimax - math.radians(14)
    angles = np.linspace(start, 2 * math.pi - start, 40)
    for a in angles:
        rows = []
        for dz in (width / 2, -width / 2):
            local_dir = np.array([math.sin(a), -math.cos(a), 0.0])
            local_o = np.array([0, 0, zl + dz])
            o = Vector((local_o @ R.T + egg.C).tolist())
            d = Vector((local_dir @ R.T).tolist())
            origin = o + d * 0.3
            best = None
            for t in trees:
                hit = t.ray_cast(origin, -d, 0.35)
                if hit[0] is not None and (best is None or (hit[0] - origin).length < (best[0] - origin).length):
                    best = hit
            p = (best[0] + d * 0.0025) if best is not None else o + d * 0.09
            rows.append(p)
        pts_top.append(rows[0])
        pts_bot.append(rows[1])
    bm = bmesh.new()
    rows = []
    for t_, b_ in zip(pts_top, pts_bot):
        out = (((t_ + b_) / 2) - C)
        out.z = 0
        out.normalize()
        rows.append([bm.verts.new(t_), bm.verts.new(b_), bm.verts.new(b_ + out * thick), bm.verts.new(t_ + out * thick)])
    for a, b in zip(rows[:-1], rows[1:]):
        for q in range(4):
            r_ = (q + 1) % 4
            bm.faces.new((a[q], a[r_], b[r_], b[q]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    obj = util.bmesh_object(name, bm)
    util.shade_smooth(obj, sharp_angle=60)
    util.set_single_weight(obj, "head")
    n_rows = len(rows)

    def param(co, vi, fi):
        row = vi // 4
        k = vi % 4
        return np.stack([row / (n_rows - 1) * 0.5, np.where((k == 0) | (k == 3), 1.0, 0.0), np.full(len(co), 0.4)], axis=1)
    set_param(obj, param)
    return obj


# ---------- a luchador's hood ----------

def _mask_side(egg, P):
    """For points near the head: signed distance (metres, along the surface) from the egg mask's edge, positive
    under the mask, negative behind it; and whether the point is inside the egg."""
    phi, th, r = egg.params(P)
    aphi = np.abs(phi)
    under = aphi < egg.phimax
    X, Y = egg.canvas_xy(np.where(under, phi, np.sign(phi) * egg.phimax * 0.999), th)
    ti = np.clip(np.searchsorted(egg.th_t, th), 1, len(egg.th_t) - 1)
    inside_d = egg.S[ti, -1] - np.abs(X)
    behind_d = -(aphi - egg.phimax) * egg.a * np.maximum(egg.rho(th), 0.15)
    # In front of the edge but outside the egg (the throat under the chin's tip): how far outside it.
    outside_d = -(r - 1) * min(egg.a, egg.b)
    return np.where(under & (r <= 1), inside_d, np.where(under, outside_d, behind_d)), r, phi, th


def hood(body, arm, m, egg, spec, name="hood"):
    """A cloth hood under the egg mask, the way a luchador wears one: it covers the skull, the back and sides of
    the head and the neck down to a collar line at its base, laced up the back. Its face opening sits under the
    mask (a band of it tucks under the mask's edge, held inside the shell, so no skin shows and nothing pokes
    through the porcelain); the ears are smoothed over. Weights are the head and neck's, turning fully to the
    head under the mask so hood and mask move as one.

    Returns (hood, laces, info): info carries the lace panel for the pattern."""
    k = m["height"] / 1.78
    off = spec.get("offset", 0.004)
    overlap = spec.get("overlap", 0.016)
    P = util.positions(body)
    W = util.weights_matrix(body, ["head", "neck_01", "spine_03"] + sided(ARM_BONES))
    arm_w = W[:, 3:].sum(axis=1)
    # The collar line: a plane through the base of the neck, lower at the front (the notch between the
    # collarbones) than at the back.
    nb = np.array(m["neck_01"])
    neckv = W[:, 1] > 0.5
    y_front, y_back = float(P[neckv, 1].min()), float(P[neckv, 1].max())
    A = np.array([0.0, y_back, nb[2] + spec.get("back", 0.0)])
    B = np.array([0.0, y_front, min(m["clavicle_l"][2], m["clavicle_r"][2]) + spec.get("front", 0.03)])
    v = B - A
    n = np.array([0.0, v[2], -v[1]])
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    above = (P - A) @ n
    region = (above > -0.03) & (arm_w < 0.5) & (P[:, 2] > nb[2] - 0.12) & (W[:, :3].sum(axis=1) > 0.2)
    sd, r, phi, th = _mask_side(egg, P)
    # The face opening: only what the shell covers from the front. Below the mouth the shell narrows to its
    # tip, so the chin's underside and the throat stay in the hood (seen from below they would show skin).
    deep = (sd > overlap) & (r < 1.02) & (th < 0.8 * math.pi)
    obj = util.extract_faces(body, util.faces_of_vertices(body, region & ~deep), name)
    util.cut_plane(obj, A, n, keep_below=False)
    util.decimate_to(obj, spec.get("tris", 2600))
    # Off the skin, then smooth: a stretched hood does not follow the skull's bumps or the neck's folds.
    Q = util.positions(obj)
    E = util.edges(obj)
    rim = boundary_verts(obj)
    soft = np.full(len(Q), spec.get("calm", 0.55))
    soft[rim] = 0.0  # the edges are straightened along themselves, below
    Q = util.taubin(Q + util.normals(obj) * off, E, spec.get("smooth", 22), weight=soft)
    # Over the ears it lies flat: a membrane across each ear (fixed on the scalp round it), lifted to follow
    # the skull's curve. The body's ears are removed under it (fighter.py).
    ears = util.grow(util.weights_matrix(obj, ["mortal_ears"])[:, 0] > 0.5, E, 1)
    ears[rim] = False
    if ears.any():
        cnt = np.zeros(len(Q))
        np.add.at(cnt, E[:, 0], 1)
        np.add.at(cnt, E[:, 1], 1)
        cnt = np.maximum(cnt, 1)[:, None]
        centre = np.array([0.0, float(egg.C[1]) + 0.03 * k, float(egg.C[2]) + 0.02 * k])
        rad = np.linalg.norm(Q - centre, axis=1)
        for _ in range(300):
            acc = np.zeros_like(Q)
            np.add.at(acc, E[:, 0], Q[E[:, 1]])
            np.add.at(acc, E[:, 1], Q[E[:, 0]])
            Q[ears] = (acc / cnt)[ears]
            ra = np.zeros(len(Q))
            np.add.at(ra, E[:, 0], rad[E[:, 1]])
            np.add.at(ra, E[:, 1], rad[E[:, 0]])
            rad[ears] = (ra / cnt[:, 0])[ears]
        dirs = Q[ears] - centre
        Q[ears] = centre + dirs / np.linalg.norm(dirs, axis=1, keepdims=True) * rad[ears][:, None]
    ear_body = util.weights_matrix(body, ["mortal_ears"])[:, 0] >= 0.5
    skin = util.extract_faces(body, ~util.faces_of_vertices(body, ear_body), name + "_skin")
    tree = BVHTree.FromObject(skin, bpy.context.evaluated_depsgraph_get())
    Q, moved = util.push_out(Q, tree, off * 0.75)
    util.delete(skin)
    log(f"{name}: {int(ears.sum())} vertices laid flat over the ears")
    # Under the mask (and a few degrees past its edge) the hood stays inside the shell.
    sd, r, phi, th = _mask_side(egg, Q)
    limit = 1 - (egg_mask.THICK + 0.003) / min(egg.a, egg.b)
    past = np.clip((np.abs(phi) - egg.phimax) / math.radians(10), 0, 1)
    rmax = limit + past * past * 2.0
    # Only what is at or inside the shell is held in: past the shell's lower rim the hood carries on outside.
    clamp = (r > rmax) & (r < 1.06) & (th > 0.02) & (th < math.pi - 0.02)
    if clamp.any():
        q = egg.to_local(Q[clamp])
        f = (rmax[clamp] / r[clamp])[:, None]
        q[:, :2] *= f
        Q[clamp] = q @ egg.R.T + egg.C
    util.set_positions(obj, Q)
    log(f"{name}: {moved} pushed clear of the skin, {int(clamp.sum())} held inside the mask")
    util.smooth_boundary(obj, 8)
    sol = obj.modifiers.new("Rim", 'SOLIDIFY')
    sol.thickness_clamp = 1.0
    sol.thickness = 0.0025
    sol.offset = -1
    sol.use_rim_only = True
    util.apply_modifier(obj, sol.name)
    util.shade_smooth(obj)
    # Weights: the body's, blending to all head under the mask.
    Q = util.positions(obj)
    sd, r, phi, th = _mask_side(egg, Q)
    f = smoothstep(-0.012, 0.006, sd)
    hg = obj.vertex_groups.get("head") or obj.vertex_groups.new(name="head")
    for vtx in obj.data.vertices:
        fv = float(f[vtx.index])
        if fv <= 1e-3:
            continue
        old_head = 0.0
        for g in vtx.groups:
            if g.group == hg.index:
                old_head = g.weight
            else:
                g.weight *= (1 - fv)
        hg.add([vtx.index], old_head * (1 - fv) + fv, 'REPLACE')
    util.limit_and_normalize(obj)
    # Pattern coordinates: x across the back (metres), y height, z how far behind the mask's edge (the gold
    # frame round the mask; 1 where there is no edge to follow).
    yc = float(egg.C[1])
    # Behind the mask's edge: how far behind it (the gold frame); elsewhere no trim from the mask.
    trim_mask = np.where((np.abs(phi) > egg.phimax) & (th < math.pi - 0.15), -sd, 1.0)

    def param(co, vi, fi):
        # Across the back, x is the plain lateral offset (the laces are placed in it); elsewhere nothing reads it.
        x = np.where(co[:, 1] > yc, co[:, 0], np.sign(co[:, 0]) * 0.5)
        return np.stack([x, co[:, 2], trim_mask[vi]], axis=1)
    set_param(obj, param)
    # Laces up the back: a slit from the nape to the back of the skull, crossed in an X between eyelets.
    hood_tree = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    z_lo = float(A[2]) + spec.get("lace_from", 0.035) * k
    z_hi = float(egg.C[2]) + spec.get("lace_to", 0.03) * k
    crosses = spec.get("crosses", 7)
    w = spec.get("lace_w", 0.011) * k
    lr = spec.get("lace_r", 0.0016)

    def on_back(x, z):
        hit = hood_tree.ray_cast(Vector((x, yc + 0.5, z)), Vector((0, -1, 0)), 0.8)
        if hit[0] is None:
            return Vector((x, yc + 0.1, z))
        return hit[0] + hit[1] * (lr * 1.3)
    zs = np.linspace(z_lo, z_hi, crosses + 1)
    laces = []
    for start in (1, -1):
        pts = []
        for i in range(crosses):
            a, b = start * (1 if i % 2 == 0 else -1), start * (-1 if i % 2 == 0 else 1)
            for t in np.linspace(0, 1, 5, endpoint=False):
                pts.append(on_back(w * (a + (b - a) * t), zs[i] + (zs[i + 1] - zs[i]) * t))
        pts.append(on_back(w * (start * (1 if crosses % 2 == 0 else -1)), zs[-1]))
        laces.append(tube_along(pts, lr, sides=6, name=f"{name}_lace{start}"))
    # The tie at the nape: two short ends hanging from the bottom eyelets.
    for sgn in (1, -1):
        base = on_back(0.0, z_lo - 0.004)
        tail = [base + Vector((sgn * 0.004 * t, 0.006 * t, -0.028 * k * t)) for t in np.linspace(0, 1, 5)]
        laces.append(tube_along(tail, lr, sides=6, name=f"{name}_tie{sgn}"))
    lace = util.join(laces, name + "_laces")
    util.shade_smooth(lace)
    util.transfer_weights(lace, obj)
    info = {"lace_lo": z_lo, "lace_hi": z_hi, "lace_w": w, "crosses": crosses}
    log(f"{name}: {util.tri_count(obj)} tris, laces {util.tri_count(lace)} tris ({z_lo:.3f} to {z_hi:.3f})")
    return obj, lace, info


def boundary_verts(obj):
    """Indices of the vertices on open edges."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    out = sorted({v.index for e in bm.edges if e.is_boundary for v in e.verts})
    bm.free()
    return np.array(out, dtype=np.int64)


# ---------- tight gear chosen by region ----------

def select_region(body, m, spec):
    """Vertices in a region: bones (score of the listed bones' weights), optional limits along bones
    ({bone: [f0, f1]}, f = fraction from head to tail), heights (fractions of the body height), and
    excluded bones."""
    P = util.positions(body)
    H = m["height"]
    bones = spec["bones"]
    W = util.weights_matrix(body, bones)
    vm = W.sum(axis=1) >= spec.get("min_weight", 0.5)
    if spec.get("exclude"):
        X = util.weights_matrix(body, spec["exclude"])
        vm &= X.sum(axis=1) < spec.get("exclude_max", 0.3)
    if "z" in spec:
        z0, z1 = spec["z"]
        vm &= (P[:, 2] >= z0 * H) & (P[:, 2] <= z1 * H)
    for bone, (f0, f1) in spec.get("along", {}).items():
        a, b = np.array(m[bone]), np.array(m[bone + "_tail"])
        d = b - a
        t = ((P - a) @ d) / (d @ d)
        near = W[:, bones.index(bone)] > 0.2 if bone in bones else np.ones(len(P), dtype=bool)
        vm &= ~near | ((t >= f0) & (t <= f1))
    if spec.get("front_open"):
        # An open front: drop a band down the middle of the chest and belly.
        w = spec["front_open"]
        vm &= ~((np.abs(P[:, 0]) < w) & (P[:, 1] < np.median(P[vm, 1]) if vm.any() else False))
    return vm


def tight(body, arm, m, spec, name):
    """A layer just off the skin over a region (tights, crop tops, armbands, shin tape, boots, wrist wraps)."""
    vm = select_region(body, m, spec)
    faces = util.faces_of_vertices(body, vm)
    obj = util.extract_faces(body, faces, name)
    util.decimate_to(obj, spec.get("tris", 2400))
    if spec.get("cut_top") and "z" in spec:
        # A clean top edge (a waistband) instead of the zigzag the face selection leaves.
        util.cut_plane(obj, (0, 0, spec["z"][1] * m["height"] - spec.get("cut_below", 0.025)), (0, 0, 1), keep_below=True)
    off = spec.get("offset", 0.0018)
    P = util.positions(obj)
    N = util.normals(obj)
    extra = np.zeros(len(P))
    if spec.get("sole"):
        extra += spec["sole"] * (P[:, 2] < 0.012)
    util.set_positions(obj, P + N * (off + extra)[:, None])
    if spec.get("smooth"):
        sm = obj.modifiers.new("Smooth", 'SMOOTH')
        sm.factor = 0.5
        sm.iterations = spec["smooth"]
        util.apply_modifier(obj, sm.name)
    # Straighten the open edges last, after everything that moves vertices (the face selection leaves a
    # zigzag, decimation and smoothing make it wave).
    util.smooth_boundary(obj, spec.get("edge_smooth", 24))
    util.despike(obj)
    sol = obj.modifiers.new("Rim", 'SOLIDIFY')
    sol.thickness_clamp = 1.0  # no spikes at sharp open corners
    sol.thickness = off
    sol.offset = -1
    sol.use_rim_only = True
    util.apply_modifier(obj, sol.name)
    util.shade_smooth(obj)
    axis_bone = spec.get("axis")
    if axis_bone:
        # Pattern coordinates round a bone (wraps, tape, bands): along it, round it, and which side.
        def param(co, vi, fi):
            out = np.zeros((len(co), 3))
            for s in ("l", "r"):
                b = axis_bone.replace("*", s)
                if b not in m:
                    continue
                a, t = np.array(m[b]), np.array(m[b + "_tail"])
                ax = (t - a) / np.linalg.norm(t - a)
                sel = (co[:, 0] > 0) == (s == "l")
                if not sel.any():
                    continue
                d = co[sel] - a
                ref = np.cross(ax, [0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.cross(ax, [0, 1.0, 0])
                ref /= np.linalg.norm(ref)
                ref2 = np.cross(ax, ref)
                ang = unwrap_angle(np.arctan2(d @ ref2, d @ ref))
                out[sel] = np.stack([d @ ax, ang * 0.045, np.full(sel.sum(), 0.5 if s == "l" else 0.8)], axis=1)
            return out
        set_param(obj, param)
    else:
        legs = {1: np.array([m["thigh_l"], m["calf_l"], m["foot_l"]]), -1: np.array([m["thigh_r"], m["calf_r"], m["foot_r"]])}
        set_param(obj, lambda co, vi, fi: lateral_param(co, legs, crotch_height(body)))
        level_top(obj)
    log(f"{name}: {util.tri_count(obj)} tris")
    return obj


# ---------- boots and shoes ----------

def _convex_hull(points, name):
    """A closed convex hull of a point cloud, as an object."""
    bm = bmesh.new()
    for p in points:
        bm.verts.new(Vector(p))
    res = bmesh.ops.convex_hull(bm, input=list(bm.verts), use_existing_faces=False)
    doomed = list({g for g in res["geom_interior"] + res["geom_unused"] if isinstance(g, bmesh.types.BMVert)})
    if doomed:
        bmesh.ops.delete(bm, geom=doomed, context='VERTS')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return util.bmesh_object(name, bm)


def boots(body, arm, m, spec, name="boots"):
    """Boots or shoes as one smooth shape round each foot, not a layer on the skin (which shows every toe).

    Each foot below the ankle is replaced by its convex hull (a rounded toe box with no toes in it, the arch
    filled, a flat underside), joined to the leg from the ankle up to above the cuff and remeshed into one
    closed volume, smoothed like stiff leather and set a few millimetres off the skin. The top is cut clean
    across the shin at `cuff` (a fraction down the calf bone, 0 at the knee and 1 at the ankle) and given a
    rim; the underside is pressed flat with a welt round the sole. Weights come from the body, so the boot
    follows the foot and ball bones; the body's foot under it is removed for good (fighter.py), so no toe can
    poke through when the foot flexes."""
    cuff = spec.get("cuff", 0.4)
    off = spec.get("offset", 0.004)
    sole_h = spec.get("sole_h", 0.02)
    welt = spec.get("welt", 0.0025)
    P = util.positions(body)
    parts = []
    floor = 1e9
    cuts = []
    for s, sign in (("l", 1), ("r", -1)):
        W = util.weights_matrix(body, [f"foot_{s}", f"ball_{s}", f"calf_{s}"])
        knee, ankle = np.array(m[f"calf_{s}"]), np.array(m[f"foot_{s}"])
        d = ankle - knee
        t = ((P - knee) @ d) / (d @ d)
        side = (P[:, 0] * sign) > 0
        lower = side & (W.sum(axis=1) >= 0.5) & (t >= cuff - 0.12)
        z_ank = float(ankle[2])
        foot = lower & (P[:, 2] <= z_ank)
        floor = min(floor, float(P[foot, 2].min()))
        # The foot below the ankle as a hull: no toe separation, no arch, nothing for a toe to show through.
        parts.append(_convex_hull(P[foot], f"{name}_hull_{s}"))
        # The leg from just inside the hull up past the cuff, closed at both ends.
        leg = util.extract_faces(body, util.faces_of_vertices(body, lower & (P[:, 2] >= z_ank - 0.012)), f"{name}_leg_{s}")
        bm = bmesh.new()
        bm.from_mesh(leg.data)
        bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(leg.data)
        bm.free()
        leg.data.update()
        for g in list(leg.vertex_groups):
            leg.vertex_groups.remove(g)
        parts.append(leg)
        top = knee + d * cuff
        cuts.append((sign, top, -d / np.linalg.norm(d)))
    obj = util.join(parts, name)
    for mm in list(obj.modifiers):
        obj.modifiers.remove(mm)
    # One volume per boot (the hull and the leg merged), then calm it: leather does not show tendons or veins.
    rm = obj.modifiers.new("Remesh", 'REMESH')
    rm.mode = 'VOXEL'
    rm.voxel_size = spec.get("voxel", 0.004)
    rm.adaptivity = 0.0
    util.apply_modifier(obj, rm.name)
    E = util.edges(obj)
    util.set_positions(obj, util.taubin(util.positions(obj), E, spec.get("smooth", 24)))
    util.set_positions(obj, util.positions(obj) + util.normals(obj) * off)
    tree = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    Pn, moved = util.push_out(util.positions(obj), tree, off * 0.7)
    util.set_positions(obj, Pn)
    log(f"{name}: remeshed {len(Pn)} verts, {moved} pushed clear of the skin")
    # The cuff: a clean cut square to the shin.
    for sign, co, no in cuts:
        util.cut_plane(obj, co, no, keep_below=True, where=lambda c, sign=sign: c.x * sign > 0)
    util.decimate_to(obj, spec.get("tris", 3000))
    util.smooth_boundary(obj, 12)
    # The sole: pressed flat underneath, a welt standing a little proud round it.
    P = util.positions(obj)
    N = util.normals(obj)
    z0 = floor - spec.get("sole_drop", 0.002)
    flat = (P[:, 2] < z0 + 0.012) & (N[:, 2] < -0.45)
    P[flat, 2] = z0
    h = P[:, 2] - z0
    nh = N[:, :2] / np.maximum(np.linalg.norm(N[:, :2], axis=1, keepdims=True), 1e-6)
    side_ness = np.clip((np.linalg.norm(N[:, :2], axis=1) - 0.25) / 0.4, 0, 1)
    lip = welt * (1 - smoothstep(sole_h - 0.004, sole_h + 0.002, h)) * side_ness
    P[:, :2] += nh * lip[:, None]
    util.set_positions(obj, P)
    sol = obj.modifiers.new("Rim", 'SOLIDIFY')
    sol.thickness_clamp = 1.0
    sol.thickness = spec.get("rim", 0.003)
    sol.offset = -1
    sol.use_rim_only = True
    util.apply_modifier(obj, sol.name)
    util.shade_smooth(obj)
    util.transfer_weights(obj, body)
    legs = {1: np.array([m["thigh_l"], m["calf_l"], m["foot_l"]]), -1: np.array([m["thigh_r"], m["calf_r"], m["foot_r"]])}
    set_param(obj, lambda co, vi, fi: lateral_param(co, legs, crotch_height(body)))
    level_top(obj)
    # Heights in the pattern are measured from the underside of the sole.
    attr = obj.data.attributes["param"]
    vec = np.empty(len(obj.data.loops) * 3)
    attr.data.foreach_get("vector", vec)
    vec = vec.reshape(-1, 3)
    vec[:, 1] -= z0
    attr.data.foreach_set("vector", vec.reshape(-1))
    log(f"{name}: {util.tri_count(obj)} tris (sole at {z0:.3f}, cuff {cuff})")
    return obj


def foot_hidden(body):
    """Body vertices a pair of boots hides for good: the feet below the ankle."""
    return util.weights_matrix(body, sided(["foot", "ball"])).sum(axis=1) >= 0.5


# ---------- loose tops ----------

def loose_top(body, arm, m, spec, name="top"):
    """A jacket, tunic or dobok top: the torso (and sleeves to a given length) eased off the body, a V or an
    open front, a hem; weights kept from the body."""
    H = m["height"]
    # The hem hangs over the hips down to the asked height, but never below the crotch (it would wrap each
    # thigh like shorts): the hips and the tops of the thighs above it belong to the top.
    hem = max(spec.get("hem", 0.5), (crotch_height(body) + 0.035) / H)
    region = {
        "bones": ["pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r",
                  "lowerarm_l", "lowerarm_r", "thigh_l", "thigh_r"],
        "z": [hem, spec.get("neck", 0.83)],
        "exclude": ["head", "hand_l", "hand_r"],
        "along": {"upperarm_l": [-1, spec.get("sleeve", 1.0)], "upperarm_r": [-1, spec.get("sleeve", 1.0)],
                  "lowerarm_l": [-1, spec.get("forearm", 0.0)], "lowerarm_r": [-1, spec.get("forearm", 0.0)]},
    }
    if spec.get("sleeve", 1.0) <= 0.0:
        region["exclude"] += ["upperarm_l", "upperarm_r"]
        region["exclude_max"] = 0.45
    if spec.get("forearm", 0.0) <= 0.0:
        region["exclude"] += ["lowerarm_l", "lowerarm_r"]
    vm = select_region(body, m, region)
    P = util.positions(body)
    # A V neck or an open front: drop the faces in front of the chest inside the opening.
    neck_z = m["neck_01"][2]
    front = P[:, 1] < (m["spine_02"][1] - 0.04)
    if spec.get("v_depth"):
        vz = neck_z - spec["v_depth"]
        inside = (P[:, 2] > vz) & (np.abs(P[:, 0]) < (P[:, 2] - vz) * spec.get("v_width", 0.55)) & front
        vm &= ~inside
    if spec.get("open"):
        vm &= ~((np.abs(P[:, 0]) < spec["open"]) & front & (P[:, 2] > hem * H))
    faces = util.faces_of_vertices(body, vm)
    obj = util.extract_faces(body, faces, name)
    util.decimate_to(obj, spec.get("tris", 2000))
    P = util.positions(obj)
    N = util.normals(obj)
    ease = spec.get("ease", 0.012)
    zf = np.clip((neck_z - P[:, 2]) / 0.25, 0, 1)
    # Below the trousers' waistband (about 0.58 of the height) the top flares out over them.
    over = spec.get("over_pants", 0.02) * smoothstep(0.6 * H, 0.52 * H, P[:, 2])
    util.set_positions(obj, P + N * (ease * (0.35 + 0.65 * zf) + over)[:, None])
    sm = obj.modifiers.new("Smooth", 'SMOOTH')
    sm.factor = 0.6
    sm.iterations = spec.get("smooth", 5)
    util.apply_modifier(obj, sm.name)
    # Folds, then thickness at the open edges.
    P = util.positions(obj)
    N = util.normals(obj)
    f = (util.fbm3(P * np.array([10, 10, 3.0]), 3, seed=17) - 0.5) * spec.get("folds", 0.012)
    util.set_positions(obj, P + N * f[:, None])
    util.smooth_boundary(obj, spec.get("edge_smooth", 30))
    util.despike(obj)
    sol = obj.modifiers.new("Thick", 'SOLIDIFY')
    sol.thickness_clamp = 1.0  # no spikes at sharp open corners
    sol.thickness = spec.get("thickness", 0.003)
    sol.offset = 1
    sol.use_rim_only = False
    sol.use_even_offset = True
    util.apply_modifier(obj, sol.name)
    util.despike(obj)
    util.shade_smooth(obj)

    def param(co, vi, fi):
        out = np.zeros((len(co), 3))
        out[:, 0] = co[:, 0]
        out[:, 1] = co[:, 2]
        out[:, 2] = -co[:, 1]
        return out
    set_param(obj, param)
    log(f"{name}: {util.tri_count(obj)} tris")
    return obj


# ---------- cords, ropes, bands ----------

def tube_along(points, radius, sides=6, name="cord", closed=False):
    """A round tube along a polyline (cords, rope, a headband)."""
    pts = [Vector(p) for p in points]
    bm = bmesh.new()
    rings = []
    n = len(pts)
    prev_x = None
    for i, p in enumerate(pts):
        t = (pts[(i + 1) % n] - pts[i - 1]) if closed else (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)])
        t.normalize()
        x = t.orthogonal().normalized() if prev_x is None else (prev_x - t * prev_x.dot(t)).normalized()
        prev_x = x
        y = t.cross(x)
        rings.append([bm.verts.new(p + (x * math.cos(2 * math.pi * k / sides) + y * math.sin(2 * math.pi * k / sides)) * radius) for k in range(sides)])
    for a, b in zip(rings, rings[1:] + ([rings[0]] if closed else [])):
        for k in range(sides):
            j = (k + 1) % sides
            bm.faces.new((a[k], a[j], b[j], b[k]))
    if not closed:
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])
    obj = util.bmesh_object(name, bm)
    util.shade_smooth(obj)
    total = len(rings)

    def param(co, vi, fi):
        ring = vi // sides
        k = vi % sides
        return np.stack([ring / max(total - 1, 1), k / sides, np.zeros(len(co))], axis=1)
    set_param(obj, param)
    return obj


def waist_cord(pants, info, spec, name="cord"):
    """A capoeira corda: two turns of cord round the waist over the trousers, knotted at the side, two ends hanging."""
    tree = bvh(pants)
    P = util.positions(pants)
    zc = info["waist_z"] - 0.03
    centre = P[np.abs(P[:, 2] - zc) < 0.03].mean(axis=0)
    objs = []
    for turn, dz in enumerate((0.0, -0.016)):
        ring = ring_on(tree, centre, zc + dz, n=40, offset=0.007 + 0.006 * turn)
        objs.append(tube_along(ring, spec.get("radius", 0.0055), name=f"{name}_{turn}", closed=True))
    side = spec.get("knot_side", 1)
    ring = ring_on(tree, centre, zc, n=40, offset=0.012)
    k = max(range(len(ring)), key=lambda i: ring[i].x * side - abs(ring[i].y - centre[1]) * 0.5)
    start = ring[k] + Vector((0, -0.01, 0))
    for e, ang in enumerate((-0.12, 0.18)):
        pts = [start + Vector((side * (0.02 + 0.03 * t * t) + ang * t * 0.1, -0.012, -0.30 * t * spec.get("tail", 1.0))) for t in np.linspace(0, 1, 10)]
        objs.append(tube_along(pts, spec.get("radius", 0.0055), name=f"{name}_end{e}"))
    obj = util.join(objs, name)
    util.transfer_weights(obj, pants)
    W = util.positions(obj)
    _reweight(obj, np.where(W[:, 2] < zc - 0.03)[0], "pelvis")
    return obj


def headband(surfaces, egg, spec, name="mongkol"):
    """A thick rope ring worn round the head over the mask's brow, the back a little higher, a tail behind."""
    trees = [bvh(s) for s in surfaces]
    C = Vector(egg.C.tolist())
    from mathutils import Matrix
    R = Matrix.Rotation(math.radians(spec.get("tilt", 14)), 3, 'X')
    pts = []
    centre = C + Vector((0, 0, spec.get("height", 0.055)))
    for i in range(32):
        a = 2 * math.pi * i / 32
        d = (R @ Vector((math.sin(a), -math.cos(a), 0))).normalized()
        o = centre + d * 0.35
        best = None
        for t in trees:
            hit = t.ray_cast(o, -d, 0.4)
            if hit[0] is not None and (best is None or (hit[0] - o).length < (best[0] - o).length):
                best = hit
        p = best[0] + d * (spec.get("radius", 0.011) + 0.002) if best is not None else centre + d * 0.11
        pts.append(p)
    ring = tube_along(pts, spec.get("radius", 0.011), sides=8, name=name, closed=True)
    back = max(pts, key=lambda p: p.y)
    tail_pts = [back + Vector((0, 0.012 + 0.025 * t, -0.17 * t)) for t in np.linspace(0, 1, 6)]
    tail = tube_along(tail_pts, spec.get("radius", 0.011) * 0.75, sides=6, name=name + "_tail")
    obj = util.join([ring, tail], name)
    util.set_single_weight(obj, "head")
    return obj


def Matrix_rot_x(a):
    from mathutils import Matrix
    return Matrix.Rotation(a, 3, 'X')


# ---------- chain-driven pieces: cape, ribbons ----------

def set_chain(obj, along, across, strip=None):
    """`_CHAIN` (0 where the piece is fixed, 1 at its free end) and `_CHAINU` (-1..1 across it), and an
    optional `_STRIP` index: the renderer's spring chain bends each vertex by its chain position."""
    me = obj.data
    for name, vals in (("_CHAIN", along), ("_CHAINU", across), ("_STRIP", strip)):
        if vals is None:
            continue
        if name in me.attributes:
            me.attributes.remove(me.attributes[name])
        a = me.attributes.new(name, 'FLOAT', 'POINT')
        a.data.foreach_set("value", np.asarray(vals, dtype=np.float32))


def cape(body, arm, m, spec, name="cape"):
    """A long cape of heavy satin on a gathered collar round the base of the neck. It lies on the shoulders and
    the upper back, then falls straight from the shoulder blades and the seat to a hem at the back of the knees,
    in soft vertical folds that begin as small gathers at the collar; thick enough to show its edges.

    Each column is cast round a vertical axis through the neck onto the torso alone (never the arms, so it
    never sticks out like wings once the arms move) and hangs from the furthest point above it, as cloth does.
    Weighted to spine_03 and the clavicles at the collar, to spine_03 below it and, down the seat and legs, to
    the pelvis; the renderer swings it with a spring chain (_CHAIN 0 at the collar to 1 at the hem, _CHAINU
    -1 to 1 across)."""
    from mathutils.kdtree import KDTree
    armw = util.weights_matrix(body, sided(ARM_BONES[1:]) + ["head"]).sum(axis=1)
    torso = util.extract_faces(body, util.faces_of_vertices(body, armw < 0.25), name + "_torso")
    tree = bvh(torso)
    k = m["height"] / 1.78
    y0 = float(m["neck_01"][1])
    z_top = float(m["neck_01"][2]) + spec.get("collar_z", 0.0) * k
    z_hem = 0.5 * (m["calf_l"][2] + m["calf_r"][2]) + spec.get("hem_above_knee", 0.03) * k
    gap = spec.get("gap", 0.01) * k
    rows, cols = spec.get("rows", 22), spec.get("cols", 30)
    A_c, A_s, A_h = (math.radians(spec.get(key, dflt)) for key, dflt in (("collar_angle", 65), ("shoulder_angle", 45), ("hem_angle", 70)))
    hug = spec.get("hug", 0.3)  # how fast (metres in per metre down) it may come back in towards the body
    z_seat = crotch_height(body) + 0.05
    flare = spec.get("flare", 0.03) * k
    zs = np.append(np.arange(z_top, z_hem, -0.004), z_hem)
    tz = (z_top - zs) / (z_top - z_hem)
    span = A_c + (A_s - A_c) * smoothstep(0.0, 0.1, tz) + (A_h - A_s) * smoothstep(0.35, 1.0, tz)
    us = np.linspace(-1, 1, cols + 1)
    reach = 0.9
    G = np.zeros((rows + 1, cols + 1, 3))
    for c, u in enumerate(us):
        dx, dy = np.sin(u * span), np.cos(u * span)
        rho = np.zeros(len(zs))
        for j, z in enumerate(zs):
            hit = tree.ray_cast(Vector((dx[j] * reach, y0 + dy[j] * reach, z)), Vector((-dx[j], -dy[j], 0)), reach)
            if hit[0] is not None:
                rho[j] = reach - hit[3]
        # Hang from the furthest point above: below the shoulder blades it comes back in towards the small of
        # the back only slowly (heavy cloth lying on the back), and below the seat not at all.
        env = np.zeros(len(zs))
        prev = 0.06 * k
        for j in range(len(zs)):
            give = hug * 0.004 if (rho[j] > 0 and zs[j] > z_seat) else 0.0
            prev = max(rho[j] + gap if rho[j] > 0 else 0.0, prev - give)
            env[j] = prev
        env = env + flare * smoothstep(0.4, 1.0, tz)
        line = np.stack([dx * env, y0 + dy * env, zs], axis=1)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
        sr = np.linspace(0, s[-1], rows + 1)
        G[:, c] = np.stack([np.interp(sr, s, line[:, i]) for i in range(3)], axis=1)
    # Calm the column-to-column differences and the turn over the shoulders.
    for _ in range(5):
        G[:, 1:-1] += 0.5 * (0.5 * (G[:, :-2] + G[:, 2:]) - G[:, 1:-1])
        G[1:-1] += 0.3 * (0.5 * (G[:-2] + G[2:]) - G[1:-1])
    flat = G.reshape(-1, 3)
    flat, _ = util.push_out(flat, tree, gap * 0.6)
    G = flat.reshape(G.shape)
    # Folds along the surface normal: gathers at the collar fading out, broad folds deepening to the hem.
    du = np.gradient(G, axis=1)
    dv = np.gradient(G, axis=0)
    Nrm = np.cross(dv, du)
    Nrm /= np.maximum(np.linalg.norm(Nrm, axis=2, keepdims=True), 1e-9)
    radial = G - np.array([0.0, y0, 0.0])
    radial[..., 2] = 0
    Nrm *= np.where((Nrm * radial).sum(axis=2, keepdims=True) < 0, -1.0, 1.0)
    T = np.repeat(np.linspace(0, 1, rows + 1)[:, None], cols + 1, axis=1)
    U = np.repeat(us[None, :], rows + 1, axis=0)
    gathers = 0.0045 * k * (1 - smoothstep(0.0, 0.22, T)) * np.sin(math.pi * spec.get("gathers", 9) * (U + 1))
    phase = math.pi * spec.get("folds", 6) * (U + 1) + 0.7 * np.sin(2.3 * U + 0.5) + 1.2 * T * np.sin(1.7 * U + 2.0)
    folds = k * (0.004 + 0.02 * smoothstep(0.12, 1.0, T)) * np.sin(phase)
    G = G + Nrm * (gathers + folds)[..., None]
    flat, _ = util.push_out(G.reshape(-1, 3), tree, gap * 0.5)
    G = flat.reshape(G.shape)
    bm = bmesh.new()
    V = [[bm.verts.new(G[r, c]) for c in range(cols + 1)] for r in range(rows + 1)]
    for r in range(rows):
        for c in range(cols):
            bm.faces.new((V[r][c], V[r + 1][c], V[r + 1][c + 1], V[r][c + 1]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    f0 = bm.faces[(rows // 2) * cols + cols // 2]
    if f0.normal.dot(Vector((f0.calc_center_median().x, f0.calc_center_median().y - y0, 0))) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    sheet = util.bmesh_object(name, bm)
    sol = sheet.modifiers.new("Thick", 'SOLIDIFY')
    sol.thickness_clamp = 1.0  # no spikes at sharp open corners
    sol.thickness = spec.get("thickness", 0.007)
    sol.offset = 0
    sol.use_even_offset = True
    util.apply_modifier(sheet, sol.name)
    n_sheet = len(sheet.data.vertices)
    # The gathered collar: a rolled band along the top edge.
    band_r = spec.get("collar_r", 0.008) * k
    collar = tube_along([Vector(G[0, c] + Nrm[0, c] * band_r * 0.6) for c in range(cols + 1)], band_r, sides=8, name=name + "_collar")
    obj = util.join([sheet, collar], name)
    obj.name = obj.data.name = name
    util.shade_smooth(obj)
    # Chain and pattern values from the nearest grid point.
    kd = KDTree((rows + 1) * (cols + 1))
    for i, p in enumerate(G.reshape(-1, 3)):
        kd.insert(Vector(p), i)
    kd.balance()
    P = util.positions(obj)
    near = np.array([kd.find(Vector(p))[1] for p in P])
    rr, cc = near // (cols + 1), near % (cols + 1)
    seg = np.linalg.norm(np.diff(G, axis=1), axis=2)
    S = np.concatenate([np.zeros((rows + 1, 1)), np.cumsum(seg, axis=1)], axis=1)
    S -= S[:, cols // 2][:, None]
    half = np.minimum(-S[:, 0], S[:, -1])
    along = rr / rows
    across = us[cc]
    s_arc = S[rr, cc]
    edge = half[rr] - np.abs(s_arc)
    edge[n_sheet:] = -1.0
    set_chain(obj, along, across)
    util.transfer_weights(obj, torso)
    util.delete(torso)
    Wc = util.weights_matrix(obj, ["spine_03", "clavicle_l", "clavicle_r"])
    Wc[Wc.sum(axis=1) < 1e-6, 0] = 1.0
    Wc /= Wc.sum(axis=1, keepdims=True)
    ws = smoothstep(float(m["pelvis"][2]), float(m["spine_02"][2]), P[:, 2])
    b = smoothstep(0.06, 0.18, along)
    final = {"spine_03": (1 - b) * Wc[:, 0] + b * ws, "clavicle_l": (1 - b) * Wc[:, 1], "clavicle_r": (1 - b) * Wc[:, 2], "pelvis": b * (1 - ws)}
    for g in list(obj.vertex_groups):
        obj.vertex_groups.remove(g)
    for gname, wv in final.items():
        vg = obj.vertex_groups.new(name=gname)
        for i in np.nonzero(wv > 1e-3)[0]:
            vg.add([int(i)], float(wv[i]), 'REPLACE')
    util.limit_and_normalize(obj)
    set_param(obj, lambda co, vi, fi: np.stack([s_arc[vi], co[:, 2], edge[vi]], axis=1))
    log(f"{name}: {util.tri_count(obj)} tris, collar {z_top:.3f}, hem {z_hem:.3f}")
    return obj


def ribbons(egg, colours, spec, name="ribbons", surfaces=()):
    """Fitas: thin ribbons tied at the back of the mask's strap, hanging down the back of the head and the
    neck and fanning out over the shoulder blades, always a little clear of the hair and skin."""
    C = Vector(egg.C.tolist())
    trees = [bvh(sf) for sf in surfaces]
    n = len(colours)
    objs = []

    def back_of(x, z, fallback):
        """The furthest-back surface point at (x, z), cast from behind."""
        best = None
        for t in trees:
            hit = t.ray_cast(Vector((x, C.y + 0.6, z)), Vector((0, -1, 0)), 1.0)
            if hit[0] is not None and (best is None or hit[0].y > best):
                best = hit[0].y
        return fallback if best is None else best

    knot_z = C.z + 0.01
    for i in range(n):
        u = (i - (n - 1) / 2) / max(n - 1, 1)
        L = spec.get("length", 0.42) * (0.85 + 0.3 * ((i * 37) % 7) / 7)
        w = spec.get("width", 0.014)
        bm = bmesh.new()
        segs = 14
        rows = []
        y_prev = None
        for sgi in range(segs + 1):
            t = sgi / segs
            x = u * (0.02 + 0.13 * t) + 0.01 * math.sin(t * 5 + i * 1.7)
            z = knot_z - L * t
            y = back_of(x, z, C.y + egg.b * 0.9) + 0.008 + 0.004 * i / max(n - 1, 1)
            if y_prev is not None:
                y = max(y, y_prev - 0.015)  # hang off the back of the head, don't wrap under the skull
            y_prev = y
            c = Vector((x, y, z))
            twist = 0.6 * math.sin(t * 4 + i)
            side = Vector((math.cos(twist), math.sin(twist) * 0.3, 0)).normalized() * (w / 2)
            rows.append((bm.verts.new(c - side), bm.verts.new(c + side)))
        for a_, b_ in zip(rows[:-1], rows[1:]):
            bm.faces.new((a_[0], a_[1], b_[1], b_[0]))
        o = util.bmesh_object(f"{name}_{i}", bm)
        sol = o.modifiers.new("Thick", 'SOLIDIFY')
        sol.thickness_clamp = 1.0  # no spikes at sharp open corners
        sol.thickness = 0.0012
        util.apply_modifier(o, sol.name)
        util.shade_smooth(o)
        P = util.positions(o)
        set_chain(o, np.clip((knot_z - P[:, 2]) / L, 0, 1), np.clip((P[:, 0] - P[:, 0].mean()) / (w / 2), -1, 1), np.full(len(P), i))
        set_param(o, lambda co, vi, fi, i=i: np.stack([co[:, 2] * 10, np.zeros(len(co)), np.full(len(co), i / max(n, 1))], axis=1))
        o["colour_index"] = i
        objs.append(o)
    return objs


def gourd(body, arm, m, spec, name="gourd"):
    """A calabash wine gourd hanging from a cord at the hip: rigid, parented to the pelvis."""
    side = spec.get("side", 1)
    hip = Vector(m["thigh_l" if side > 0 else "thigh_r"])
    tree = bvh(body)
    o = hip + Vector((side * 0.3, 0.0, -0.02))
    hit = tree.ray_cast(o, Vector((-side, 0, 0)), 0.4)
    surf = hit[0] if hit[0] is not None else hip + Vector((side * 0.12, 0, 0))
    centre = surf + Vector((side * 0.06, 0.02, -0.14))
    bm = bmesh.new()
    prof = [(0.0, 0.004), (0.02, 0.03), (0.05, 0.048), (0.08, 0.05), (0.11, 0.04), (0.13, 0.024), (0.15, 0.022),
            (0.17, 0.03), (0.195, 0.033), (0.22, 0.026), (0.24, 0.014), (0.255, 0.011), (0.27, 0.011)]
    sides = 12
    rings = []
    for z, r in prof:
        rings.append([bm.verts.new(centre + Vector((math.cos(2 * math.pi * k / sides) * r, math.sin(2 * math.pi * k / sides) * r, z - 0.12))) for k in range(sides)])
    for a, b in zip(rings[:-1], rings[1:]):
        for k in range(sides):
            j = (k + 1) % sides
            bm.faces.new((a[k], a[j], b[j], b[k]))
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    g = util.bmesh_object(name, bm)
    util.shade_smooth(g)
    cord_pts = [centre + Vector((0, 0, 0.15)), centre.lerp(surf, 0.5) + Vector((0, 0, 0.12)), surf + Vector((0, 0, 0.05))]
    c = tube_along(cord_pts, 0.003, name=name + "_cord")
    obj = util.join([g, c], name)
    set_param(obj, lambda co, vi, fi: np.stack([co[:, 2] * 5, np.arctan2(co[:, 1] - centre.y, co[:, 0] - centre.x) * 0.2, np.zeros(len(co))], axis=1))
    util.set_single_weight(obj, "pelvis")
    return obj
