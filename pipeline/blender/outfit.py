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
        # Folds: a few soft vertical waves around the tube, more at the hem.
        ang = np.arctan2(rd[:, 1], rd[:, 0])
        wav = util.fbm3(np.stack([np.cos(ang) * 2.2, np.sin(ang) * 2.2, z * 7.0 + sign * 3], axis=1), 3, seed=4)
        newr += (wav - 0.5) * 0.012 * k * (0.4 + 0.8 * f_hem)
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
    set_param(obj, lambda co, vi, fi: co)
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
        score = W[:, 0] + W[:, 2:6].sum(axis=1) * spec.get("knuckles", 1.0) + W[:, 6] * 0.6 + W[:, 1] * near_wrist
        far = W[:, 7:].sum(axis=1)
        inc = (score >= 0.5) & (far < spec.get("fingers", 0.3)) & (along > -length)
        vmask |= inc
        params[s] = (wrist, axis)
    faces = util.faces_of_vertices(body, vmask)
    obj = util.extract_faces(body, faces, name)
    Pn = util.positions(obj) + util.normals(obj) * offset
    util.set_positions(obj, Pn)
    sol = obj.modifiers.new("Rim", 'SOLIDIFY')
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
    # Grow the scalp a ring or two so the hairline meets the mask's edge and the nape.
    faces = util.faces_of_vertices(body, ids, mode="any")
    shell = util.extract_faces(body, faces, name)
    S = util.positions(shell)
    shell_n = util.normals(shell)
    util.set_positions(shell, S + shell_n * spec.get("thickness", 0.0035))
    sol = shell.modifiers.new("Rim", 'SOLIDIFY')
    sol.thickness = spec.get("thickness", 0.0035)
    sol.offset = -1
    sol.use_rim_only = True
    util.apply_modifier(shell, sol.name)
    util.shade_smooth(shell)
    util.set_single_weight(shell, "head")
    # The knot: behind the top of the mask, at the crown's back.
    S = util.positions(body)[ids]
    back = S[S[:, 1] > egg.C[1] + 0.035]
    top = back[np.argmax(back[:, 2] + 0.35 * back[:, 1])]
    up = Vector((0, 0.55, 0.85)).normalized()
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
