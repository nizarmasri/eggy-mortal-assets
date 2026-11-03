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
    """A short cape hanging from the shoulders down the back: a sheet a little clear of the back, cast onto
    the torso alone (never the arms, so it never sticks out like wings once the arms move), narrower at the
    collar than at the hem. Weighted to spine_03 and the clavicles at the collar (from the torso), to spine_03
    below; the renderer swings it with a spring chain (_CHAIN, _CHAINU)."""
    # The torso alone: the body without its arms, for casting and for the collar weights.
    armw = util.weights_matrix(body, sided(ARM_BONES[1:]) + ["head"]).sum(axis=1)
    torso = util.extract_faces(body, util.faces_of_vertices(body, armw < 0.25), name + "_torso")
    tree = bvh(torso)
    H = m["height"]
    top_z = m["neck_01"][2] - 0.015
    length = spec.get("length", 0.45) * H / 1.78
    w0 = spec.get("width", 0.2) * H / 1.78
    w1 = spec.get("bottom_width", w0 * 1.35) * H / 1.78
    back_y = m["spine_03"][1]
    rows, cols = 12, 12
    grid = []
    prev = None
    for r in range(rows + 1):
        t = r / rows
        z = top_z - length * t
        half = w0 + (w1 - w0) * t
        row = []
        for c in range(cols + 1):
            u = c / cols * 2 - 1
            x = u * half
            o = Vector((x * 0.85, back_y + 0.5, z))
            hit = tree.ray_cast(o, Vector((0, -1, 0)), 0.8)
            y = hit[0].y if hit[0] is not None else back_y + 0.12
            # Over the shoulders the collar wraps forward a little; lower down it hangs straight.
            y += 0.012 + 0.012 * t * t
            # Soft vertical folds, deeper towards the hem.
            y += 0.012 * t * math.sin(u * math.pi * 3.0 + 0.4)
            if prev is not None:
                y = max(y, prev[c].y - 0.02)  # hangs, never tucks back in under the shoulder blades
            row.append(Vector((x * (0.85 + 0.15 * t), y, z)))
        grid.append(row)
        prev = row
    bm = bmesh.new()
    V = [[bm.verts.new(p) for p in row] for row in grid]
    for r in range(rows):
        for c in range(cols):
            bm.faces.new((V[r][c], V[r + 1][c], V[r + 1][c + 1], V[r][c + 1]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    obj = util.bmesh_object(name, bm)
    sol = obj.modifiers.new("Thick", 'SOLIDIFY')
    sol.thickness_clamp = 1.0  # no spikes at sharp open corners
    sol.thickness = 0.004
    sol.offset = 0
    util.apply_modifier(obj, sol.name)
    util.shade_smooth(obj)
    P = util.positions(obj)
    along = np.clip((top_z - P[:, 2]) / length, 0, 1)
    across = np.clip(P[:, 0] / w1, -1, 1)
    set_chain(obj, along, across)
    util.transfer_weights(obj, torso)
    util.delete(torso)
    # Below the collar the cape hangs from the upper back and, lower down, from the hips, so a forward lean
    # does not swing its hem out behind like a board (the renderer's chain adds the swing).
    low = np.where(along > 0.12)[0]
    for g in obj.vertex_groups:
        g.remove([int(i) for i in low])
    ws = np.clip(1 - (along - 0.3) / 0.6, 0, 1)
    for gname, wv in (("spine_03", ws), ("pelvis", 1 - ws)):
        vg = obj.vertex_groups.get(gname) or obj.vertex_groups.new(name=gname)
        for i in low:
            if wv[i] > 1e-3:
                vg.add([int(i)], float(wv[i]), 'REPLACE')
    set_param(obj, lambda co, vi, fi: np.stack([co[:, 0], co[:, 2], np.zeros(len(co))], axis=1))
    log(f"{name}: {util.tri_count(obj)} tris")
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
