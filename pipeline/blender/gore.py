"""Gore parts: cut loops, caps, a rigid stylised skeleton with pre-fractured pieces, and body segments.

Cuts. Each cut is a plane across one bone (CUTS: name, bone, fraction along it). Body and gear are bisected
there and split into two pieces along the new loop; both copies of the loop are weighted 100% to the cut's
bone, so the seam stays shut while the limb is attached and both caps fit it exactly once it is not. Each
vertex carries `_SEGMENT` (SEGMENTS) so the renderer knows which piece it belongs to.

Caps. Two per cut, filling the body's cross-section there: `cap_<cut>_p` on the stump (facing away from the
body), `cap_<cut>_d` on the severed piece. Skin rim, fat, muscle and a bone cross-section with marrow, all from
one small gore texture (`gore` material), weighted 100% to the cut's bone.

Skeleton. `skeleton` is one skinned mesh, every piece rigid on its bone, for the X-ray shots. Pre-fractured
pieces (`bone_frac_<part>_<n>`) are parented to their bones like the mask shards, for breaks.
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

import util
from util import log

SEGMENTS = {
    "torso": 0, "head": 1, "upperarm_l": 2, "forearm_l": 3, "upperarm_r": 4, "forearm_r": 5,
    "thigh_l": 6, "shin_l": 7, "thigh_r": 8, "shin_r": 9, "pelvis": 10,
}

# name, bone, fraction along the bone, segment on the far side, segment on the near side, chain bones (far side)
ARM = ["upperarm", "lowerarm", "hand", "thumb_01", "thumb_02", "thumb_03", "index_01", "index_02", "index_03", "middle_01",
       "middle_02", "middle_03", "ring_01", "ring_02", "ring_03", "pinky_01", "pinky_02", "pinky_03"]
LEG = ["thigh", "calf", "foot", "ball"]


def _side(names, s):
    return [f"{n}_{s}" for n in names]


CUTS = [
    ("neck", "neck_01", 0.55, "head", "torso", ["head"], 0.012, 0.11),
    ("waist", "spine_01", 0.5, "torso", "pelvis", ["spine_02", "spine_03", "neck_01", "head", "clavicle_l", "clavicle_r"] + _side(ARM, "l") + _side(ARM, "r"), 0.017, 0.26),
    ("shoulder_l", "upperarm_l", 0.24, "upperarm_l", "torso", _side(ARM, "l"), 0.011, 0.085),
    ("shoulder_r", "upperarm_r", 0.24, "upperarm_r", "torso", _side(ARM, "r"), 0.011, 0.085),
    ("elbow_l", "lowerarm_l", 0.16, "forearm_l", "upperarm_l", _side(ARM[1:], "l"), 0.010, 0.07),
    ("elbow_r", "lowerarm_r", 0.16, "forearm_r", "upperarm_r", _side(ARM[1:], "r"), 0.010, 0.07),
    ("hip_l", "thigh_l", 0.3, "thigh_l", "pelvis", _side(LEG, "l"), 0.014, 0.12),
    ("hip_r", "thigh_r", 0.3, "thigh_r", "pelvis", _side(LEG, "r"), 0.014, 0.12),
    ("knee_l", "calf_l", 0.14, "shin_l", "thigh_l", _side(LEG[1:], "l"), 0.013, 0.09),
    ("knee_r", "calf_r", 0.14, "shin_r", "thigh_r", _side(LEG[1:], "r"), 0.013, 0.09),
]


def plane(arm, bone, frac):
    b = arm.data.bones[bone]
    h, t = Vector(b.head_local), Vector(b.tail_local)
    n = (t - h).normalized()
    return h.lerp(t, frac), n


def _dominant(obj):
    """Each vertex's heaviest bone."""
    names = {g.index: g.name for g in obj.vertex_groups}
    out = []
    for v in obj.data.vertices:
        best, bw = None, -1
        for g in v.groups:
            if g.weight > bw and g.group in names:
                best, bw = names[g.group], g.weight
        out.append(best)
    return out


def _region_faces(bm, dom, co, n, chain, radius, band=0.04):
    """Faces near the cut plane, near the bone and on the cut's limb."""
    chain = set(chain)
    out = []
    for f in bm.faces:
        c = f.calc_center_median()
        d = (c - co).dot(n)
        if abs(d) > band:
            continue
        r = ((c - co) - n * d).length
        if r > radius:
            continue
        # The far side's chain, or the near side right next to the plane.
        doms = [dom[v.index] for v in f.verts if v.index < len(dom)]
        if any(x in chain for x in doms) or abs(d) < band * 0.5:
            out.append(f)
    return out


def cut(obj, arm, label=""):
    """Bisect obj at every cut, split it along the new loops, weight the loops to the cut bone and write
    `_SEGMENT`. Returns {cut name: number of loop vertices}."""
    dom = _dominant(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    deform = bm.verts.layers.deform.verify()
    gidx = {g.name: g.index for g in obj.vertex_groups}
    loops = {}
    # Normals before the split, so the seam does not show once the two sides have their own vertices.
    for c_name, bone, frac, far, near, chain, rb, radius in CUTS:
        co, n = plane(arm, bone, frac)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        dom_now = dom + [None] * (len(bm.verts) - len(dom))
        faces = _region_faces(bm, dom_now, co, n, chain, radius)
        if not faces:
            continue
        geom = list({v for f in faces for v in f.verts}) + list({e for f in faces for e in f.edges}) + faces
        res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=co, plane_no=n, dist=1e-5)
        on = [g for g in res["geom_cut"] if isinstance(g, bmesh.types.BMEdge)]
        if not on:
            continue
        bmesh.ops.split_edges(bm, edges=on)
        bm.verts.ensure_lookup_table()
        loop_verts = [v for v in bm.verts if abs((v.co - co).dot(n)) < 2e-5 and ((v.co - co) - n * (v.co - co).dot(n)).length < radius]
        g = gidx.get(bone)
        if g is None:
            g = obj.vertex_groups.new(name=bone).index
            gidx[bone] = g
        for v in loop_verts:
            dv = v[deform]
            dv.clear()
            dv[g] = 1.0
        loops[c_name] = len(loop_verts)
        dom = dom_now + [bone] * (len(bm.verts) - len(dom_now))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    seg = segments(obj, arm)
    if "_SEGMENT" in obj.data.attributes:
        obj.data.attributes.remove(obj.data.attributes["_SEGMENT"])
    attr = obj.data.attributes.new("_SEGMENT", 'FLOAT', 'POINT')
    attr.data.foreach_set("value", seg.astype(np.float32))
    log(f"cut {obj.name}{label}: loops {loops}; segments {np.bincount(seg.astype(int), minlength=11).tolist()}")
    return loops


def seal_seams(obj):
    """After a split, coincident vertices get one shared normal so the seam is invisible."""
    me = obj.data
    P = util.positions(obj)
    Nn = util.normals(obj)
    key = np.round(P / 1e-5).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    acc = np.zeros((inv.max() + 1, 3))
    np.add.at(acc, inv, Nn)
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-9)
    me.normals_split_custom_set_from_vertices([tuple(x) for x in acc[inv]])
    me.update()


def segments(obj, arm):
    """Segment of every vertex, from its faces: a face belongs to the far side of a cut when its centre is
    past the plane and its heaviest bone is on the cut's limb (cuts applied from the torso outwards)."""
    dom = _dominant(obj)
    me = obj.data
    P = util.positions(obj)
    face_seg = np.zeros(len(me.polygons), dtype=int)
    planes = {c[0]: (*plane(arm, c[1], c[2]), c) for c in CUTS}
    for p in me.polygons:
        c = Vector(p.center)
        vs = list(p.vertices)
        doms = [dom[i] for i in vs]
        best = max(set(doms), key=doms.count)
        s = "torso"
        co, n, spec = planes["waist"]
        below_waist = (c - co).dot(n) < 0
        if below_waist and (best in ("pelvis", "spine_01", None) or best.startswith(("thigh", "calf", "foot", "ball"))):
            s = "pelvis"
        for c_name in ("hip_l", "hip_r", "knee_l", "knee_r", "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "neck"):
            co, n, spec = planes[c_name]
            if best in spec[5] and (c - co).dot(n) > 0:
                s = spec[3]
        face_seg[p.index] = SEGMENTS[s]
    seg = np.zeros(len(me.vertices))
    for p in me.polygons:
        for i in p.vertices:
            seg[i] = face_seg[p.index]
    return seg


# ---------- caps ----------

def _section(obj, co, n, chain, radius):
    """The cross-section polyline of obj on the plane, near the bone, ordered around the bone axis."""
    dom = _dominant(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    faces = _region_faces(bm, dom, co, n, chain, radius)
    if not faces:
        bm.free()
        return None
    geom = list({v for f in faces for v in f.verts}) + list({e for f in faces for e in f.edges}) + faces
    res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=co, plane_no=n, dist=1e-5)
    pts = [Vector(g.co) for g in res["geom_cut"] if isinstance(g, bmesh.types.BMVert)]
    bm.free()
    if len(pts) < 6:
        return None
    u = n.orthogonal().normalized()
    w = n.cross(u)
    ang = [math.atan2((p - co).dot(w), (p - co).dot(u)) for p in pts]
    order = np.argsort(ang)
    pts = [pts[i] for i in order]
    # Resample evenly around the bone: the farthest point at each of 24 angles.
    out = []
    A = np.array(ang)[order]
    R = np.array([((p - co) - n * (p - co).dot(n)).length for p in pts])
    for k in range(24):
        a = -math.pi + 2 * math.pi * (k + 0.5) / 24
        d = np.abs((A - a + math.pi) % (2 * math.pi) - math.pi)
        near = d < (math.pi / 24) * 1.6
        if near.any():
            r = float(R[near].max())
        else:
            r = float(np.interp(a, A, R, period=2 * math.pi))
        out.append(co + (u * math.cos(a) + w * math.sin(a)) * r)
    return out, u, w


def caps(full_body, arm, mat):
    """Two caps per cut from the full body's cross-sections."""
    objs = []
    for c_name, bone, frac, far, near, chain, rb, radius in CUTS:
        co, n = plane(arm, bone, frac)
        sec = _section(full_body, co, n, chain, radius)
        if sec is None:
            log("cap: no section for", c_name)
            continue
        ring, u, w = sec
        minr = min((p - co).length for p in ring)
        rb_ = min(rb, minr * 0.45)
        for side, sgn in (("p", 1), ("d", -1)):
            bm = bmesh.new()
            uv = bm.loops.layers.uv.new("UVMap")
            M = len(ring)
            rows = []
            # (radius code u, point builder)
            ring_defs = [
                (1.0, lambda i: ring[i]),
                (0.94, lambda i: co + (ring[i] - co) * 0.95),
                (0.86, lambda i: co + (ring[i] - co) * 0.88),
                (0.55, lambda i: co + (ring[i] - co) * 0.55 + (ring[i] - co).normalized() * rb_ * 0.5),
                (0.2, lambda i: co + (ring[i] - co).normalized() * rb_),
                (0.12, lambda i: co + (ring[i] - co).normalized() * rb_ * 0.62),
            ]
            for code, fn in ring_defs:
                rows.append((code, [bm.verts.new(fn(i) + n * sgn * 0.0004) for i in range(M)]))
            centre = bm.verts.new(co + n * sgn * 0.0004)
            for (c0, r0), (c1, r1) in zip(rows[:-1], rows[1:]):
                for i in range(M):
                    j = (i + 1) % M
                    f = bm.faces.new((r0[i], r0[j], r1[j], r1[i]))
                    for lp, (cc, ii) in zip(f.loops, ((c0, i), (c0, i + 1), (c1, i + 1), (c1, i))):
                        lp[uv].uv = (cc, ii / M)
            last_c, last = rows[-1]
            for i in range(M):
                j = (i + 1) % M
                f = bm.faces.new((last[i], last[j], centre))
                for lp, (cc, ii) in zip(f.loops, ((last_c, i), (last_c, i + 1), (0.02, i + 0.5))):
                    lp[uv].uv = (cc, ii / M)
            # Face the open side: + bone direction on the stump, - on the severed piece.
            bm.normal_update()
            bm.faces.ensure_lookup_table()
            if bm.faces[0].normal.dot(n) * sgn < 0:
                bmesh.ops.reverse_faces(bm, faces=bm.faces)
            obj = util.bmesh_object(f"cap_{c_name}_{side}", bm)
            util.set_single_weight(obj, bone)
            util.add_armature(obj, arm)
            util.set_material(obj, mat)
            objs.append(obj)
    log("caps:", len(objs))
    return objs


def gore_texture(size=256):
    """Cross-section bands along u (marrow, bone, muscle, fat, dermis, skin), varied along v."""
    u = (np.arange(size) + 0.5) / size
    v = (np.arange(size) + 0.5) / size
    U, V = np.meshgrid(u, v)
    n1 = util.fbm3(np.stack([U * 6, V * 18, np.zeros_like(U)], axis=-1), 3, seed=31)
    fib = util.fbm3(np.stack([U * 60, V * 6, np.ones_like(U)], axis=-1), 2, seed=32)
    lin = util.srgb_to_linear
    marrow = lin(np.array([0.35, 0.05, 0.05]))
    bone = lin(np.array([0.86, 0.82, 0.72]))
    muscle = lin(np.array([0.48, 0.05, 0.05]))
    muscle2 = lin(np.array([0.30, 0.02, 0.03]))
    fat = lin(np.array([0.85, 0.72, 0.42]))
    dermis = lin(np.array([0.72, 0.38, 0.33]))
    col = np.zeros((size, size, 3))
    def band(lo, hi):
        return ((U >= lo) & (U < hi))[..., None]
    col = np.where(band(0, 0.1), marrow * (0.8 + 0.3 * n1[..., None]), col)
    col = np.where(band(0.1, 0.22), bone * (0.9 + 0.1 * n1[..., None]), col)
    m = muscle * (0.75 + 0.35 * fib[..., None]) * (1 - 0.3 * n1[..., None]) + muscle2 * 0.3 * n1[..., None]
    col = np.where(band(0.22, 0.84), m, col)
    col = np.where(band(0.84, 0.92), fat * (0.85 + 0.2 * n1[..., None]), col)
    col = np.where(band(0.92, 1.01), dermis * (0.9 + 0.15 * n1[..., None]), col)
    rough = np.where(U < 0.22, 0.45, np.where(U < 0.84, 0.32, 0.5)) + 0.1 * n1
    return util.linear_to_srgb(col), rough


# ---------- skeleton ----------

def _rod(bm, a, b, r, sides=6, knob=1.0, segs=2):
    """A bone shaft from a to b: a capped tube, swollen to `knob` x r at both ends."""
    a, b = Vector(a), Vector(b)
    d = b - a
    L = d.length
    if L < 1e-5:
        return []
    z = d.normalized()
    x = z.orthogonal().normalized()
    y = z.cross(x)
    rings = []
    ts = [0.0, 0.08] + [0.08 + 0.84 * (k + 1) / (segs + 1) for k in range(segs)] + [0.92, 1.0]
    for t in ts:
        end = min(t, 1 - t)
        rr = r * (knob if end < 0.1 else 1.0) * (0.75 if end == 0 else 1.0)
        c = a + d * t
        rings.append([bm.verts.new(c + (x * math.cos(2 * math.pi * i / sides) + y * math.sin(2 * math.pi * i / sides)) * rr) for i in range(sides)])
    faces = []
    for r0, r1 in zip(rings[:-1], rings[1:]):
        for i in range(sides):
            j = (i + 1) % sides
            faces.append(bm.faces.new((r0[i], r0[j], r1[j], r1[i])))
    faces.append(bm.faces.new(list(reversed(rings[0]))))
    faces.append(bm.faces.new(rings[-1]))
    return [v for ring in rings for v in ring]


def _blob(bm, c, radii, rot=None, segs=(10, 7)):
    res = bmesh.ops.create_uvsphere(bm, u_segments=segs[0], v_segments=segs[1], radius=1.0)
    R = rot if rot is not None else Matrix.Identity(3)
    for v in res["verts"]:
        v.co = Vector(c) + R @ Vector((v.co.x * radii[0], v.co.y * radii[1], v.co.z * radii[2]))
    return list(res["verts"])


class _Builder:
    def __init__(self):
        self.bm = bmesh.new()
        self.weights = {}

    def add(self, verts, bone):
        for v in verts:
            self.weights[v] = bone

    def object(self, name, arm, mat):
        bm = self.bm
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.verts.index_update()
        groups = {}
        for v, b in self.weights.items():
            groups.setdefault(b, []).append(v.index)
        uv = bm.loops.layers.uv.new("UVMap")
        for f in bm.faces:
            for lp in f.loops:
                lp[uv].uv = (0.16, (lp.vert.co.z * 3.0) % 1.0)
        obj = util.bmesh_object(name, bm)
        for b, ids in groups.items():
            vg = obj.vertex_groups.get(b) or obj.vertex_groups.new(name=b)
            vg.add(ids, 1.0, 'REPLACE')
        util.shade_smooth(obj, sharp_angle=70)
        util.add_armature(obj, arm)
        util.set_material(obj, mat)
        return obj


def skeleton(arm, body, egg, mat):
    """The stylised skeleton (one rigid-skinned mesh) and its pre-fractured pieces (bone-parented)."""
    B = {b.name: (Vector(b.head_local), Vector(b.tail_local)) for b in arm.data.bones}
    tree = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    sk = _Builder()
    frac = []  # (name, bone, builder function)

    def head(n):
        return B[n][0]

    def tail(n):
        return B[n][1]

    # Limbs.
    for s in ("l", "r"):
        sk.add(_rod(sk.bm, head(f"clavicle_{s}") + Vector((0, -0.02, 0)), tail(f"clavicle_{s}"), 0.006, knob=1.4), f"clavicle_{s}")
        sk.add(_rod(sk.bm, head(f"upperarm_{s}"), tail(f"upperarm_{s}"), 0.011, knob=1.9, segs=3), f"upperarm_{s}")
        for off in (-0.008, 0.008):
            a, b = head(f"lowerarm_{s}"), tail(f"lowerarm_{s}")
            side = (b - a).cross(Vector((0, 0, 1))).normalized() * off
            sk.add(_rod(sk.bm, a + side, b + side, 0.0065, knob=1.6), f"lowerarm_{s}")
        for f in ("index", "middle", "ring", "pinky", "thumb"):
            sk.add(_rod(sk.bm, head(f"hand_{s}"), head(f"{f}_01_{s}"), 0.004, sides=4, knob=1.4, segs=0), f"hand_{s}")
            for k in (1, 2, 3):
                sk.add(_rod(sk.bm, head(f"{f}_0{k}_{s}"), tail(f"{f}_0{k}_{s}"), 0.0035, sides=4, knob=1.3, segs=0), f"{f}_0{k}_{s}")
        sk.add(_rod(sk.bm, head(f"thigh_{s}"), tail(f"thigh_{s}"), 0.014, knob=1.9, segs=3), f"thigh_{s}")
        a, b = head(f"calf_{s}"), tail(f"calf_{s}")
        sk.add(_rod(sk.bm, a, b, 0.012, knob=1.8, segs=3), f"calf_{s}")
        sk.add(_rod(sk.bm, a + Vector((0.02 if s == "l" else -0.02, 0.01, -0.02)), b + Vector((0.015 if s == "l" else -0.015, 0.01, 0)), 0.005, knob=1.4), f"calf_{s}")
        sk.add(_blob(sk.bm, a + Vector((0, -0.035, 0.0)), (0.018, 0.01, 0.02), segs=(8, 6)), f"calf_{s}")  # kneecap
        sk.add(_rod(sk.bm, head(f"foot_{s}"), tail(f"foot_{s}"), 0.009, knob=1.6), f"foot_{s}")
        sk.add(_rod(sk.bm, head(f"foot_{s}"), head(f"foot_{s}") + Vector((0, 0.05, -0.04)), 0.011, knob=1.3, segs=0), f"foot_{s}")
        sk.add(_rod(sk.bm, head(f"ball_{s}"), tail(f"ball_{s}"), 0.006, knob=1.3, segs=0), f"ball_{s}")
    # Spine: a vertebra per few centimetres from the pelvis to the skull, set towards the back.
    chain = ["pelvis", "spine_01", "spine_02", "spine_03", "neck_01"]
    for bn in chain:
        a, b = head(bn), tail(bn)
        n = max(2, int((b - a).length / 0.028))
        for k in range(n):
            c = a.lerp(b, (k + 0.5) / n) + Vector((0, 0.035 if bn != "neck_01" else 0.015, 0))
            sk.add(_blob(sk.bm, c, (0.017, 0.014, 0.009), segs=(6, 4)), bn)
    # Ribs: arcs following the chest wall, inset from the skin.
    rib_bones = []
    z0, z1 = head("spine_02").z, tail("spine_03").z - 0.04
    for i in range(9):
        z = z1 - (z1 - z0) * i / 8
        bone = "spine_03" if i < 5 else "spine_02"
        for s in (1, -1):
            pts = []
            for k in range(7):
                ang = math.radians(10 + 160 * k / 6)  # from the back round to the front
                d = Vector((s * math.sin(ang), math.cos(ang), 0))
                c = Vector((0, head("spine_02").y, z - 0.03 * math.sin(ang)))
                hit = tree.ray_cast(c + d * 0.4, -d, 0.5)
                p = hit[0] - d * 0.022 if hit[0] is not None else c + d * 0.12
                pts.append(p)
            for a, b in zip(pts[:-1], pts[1:]):
                sk.add(_rod(sk.bm, a, b, 0.0045, sides=4, knob=1.0, segs=0), bone)
            rib_bones.append((i, s, pts, bone))
    # Sternum and pelvis.
    front = [p[-1] for (i, s, p, b) in rib_bones if s == 1]
    if front:
        sk.add(_rod(sk.bm, front[0] + Vector((-front[0].x, 0, 0)), front[-1] + Vector((-front[-1].x, 0, 0)), 0.01, knob=1.2), "spine_03")
    pc = head("pelvis")
    for s in (1, -1):
        sk.add(_blob(sk.bm, pc + Vector((s * 0.075, 0.0, 0.03)), (0.045, 0.02, 0.05), rot=Matrix.Rotation(math.radians(25 * s), 3, 'Y'), segs=(8, 6)), "pelvis")
    sk.add(_blob(sk.bm, pc + Vector((0, -0.05, -0.03)), (0.05, 0.015, 0.02), segs=(8, 5)), "pelvis")
    # Skull and jaw, sized from the mask's egg (which was fitted to the head).
    hc = Vector((0, egg.C[1] + 0.012, egg.C[2] + 0.02))
    skull_r = (egg.a * 0.78, (egg.b) * 0.62, egg.h * 0.62)
    sk.add(_blob(sk.bm, hc, skull_r, segs=(14, 10)), "head")
    jaw_c = hc + Vector((0, -skull_r[1] * 0.55, -skull_r[2] * 0.85))
    sk.add(_blob(sk.bm, jaw_c, (skull_r[0] * 0.75, skull_r[1] * 0.45, skull_r[2] * 0.22), segs=(10, 6)), "head")
    skel = sk.object("skeleton", arm, mat)

    # Pre-fractured pieces: rigid, parented to their bones.
    pieces = []

    def piece(name, bone, builder):
        b = _Builder()
        verts = builder(b.bm)
        obj = b.object(name, arm, mat)
        for m in list(obj.modifiers):
            obj.modifiers.remove(m)
        for g in list(obj.vertex_groups):
            obj.vertex_groups.remove(g)
        util.select_only(obj)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        mw = obj.matrix_world.copy()
        obj.parent = arm
        obj.parent_type = 'BONE'
        obj.parent_bone = bone
        obj.matrix_parent_inverse = Matrix.Identity(4)
        bpy.context.view_layer.update()
        obj.matrix_world = mw
        pieces.append(obj)

    # Skull in four (quadrants), jaw in two (left, right): the skull blob cut by planes.
    for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (1, -1), (-1, -1))):
        def skull_q(bm, sx=sx, sy=sy):
            _blob(bm, hc, skull_r, segs=(14, 10))
            for no in (Vector((-sx, 0, 0)), Vector((0, -sy, 0))):
                bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces), plane_co=hc, plane_no=no, clear_outer=True, dist=1e-5)
            bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
            return list(bm.verts)
        piece(f"bone_frac_skull_{k}", "head", skull_q)
    for k, sx in enumerate((1, -1)):
        def jaw_h(bm, sx=sx):
            _blob(bm, jaw_c, (skull_r[0] * 0.75, skull_r[1] * 0.45, skull_r[2] * 0.22), segs=(10, 6))
            bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces), plane_co=jaw_c, plane_no=Vector((-sx, 0, 0)), clear_outer=True, dist=1e-5)
            bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
            return list(bm.verts)
        piece(f"bone_frac_jaw_{k}", "head", jaw_h)
    # Ribs in six: left and right, upper, middle and lower thirds.
    for k, (s, lo, hi) in enumerate(((1, 0, 3), (1, 3, 6), (1, 6, 9), (-1, 0, 3), (-1, 3, 6), (-1, 6, 9))):
        sel = [r for r in rib_bones if r[1] == s and lo <= r[0] < hi]
        def ribs(bm, sel=sel):
            for (_, _, pts, _) in sel:
                for a, b in zip(pts[:-1], pts[1:]):
                    _rod(bm, a, b, 0.0045, sides=4, knob=1.0, segs=0)
            return list(bm.verts)
        piece(f"bone_frac_ribs_{k}", sel[0][3] if sel else "spine_03", ribs)
    # Spine in three: lumbar (spine_01), thoracic (spine_02/03), cervical (neck).
    for k, bones in enumerate((["pelvis", "spine_01"], ["spine_02", "spine_03"], ["neck_01"])):
        def spine(bm, bones=bones):
            for bn in bones:
                a, b = head(bn), tail(bn)
                n = max(2, int((b - a).length / 0.028))
                for j in range(n):
                    c = a.lerp(b, (j + 0.5) / n) + Vector((0, 0.035 if bn != "neck_01" else 0.015, 0))
                    _blob(bm, c, (0.017, 0.014, 0.009), segs=(6, 4))
            return list(bm.verts)
        piece(f"bone_frac_spine_{k}", bones[-1], spine)
    # Forearms and shins snap mid-shaft: two halves each.
    for s in ("l", "r"):
        for k in (0, 1):
            def fore(bm, s=s, k=k):
                a, b = head(f"lowerarm_{s}"), tail(f"lowerarm_{s}")
                m = a.lerp(b, 0.5 + 0.04 * (1 if k == 0 else -1))
                lo, hi = (a, m) if k == 0 else (m, b)
                for off in (-0.008, 0.008):
                    side = (b - a).cross(Vector((0, 0, 1))).normalized() * off
                    _rod(bm, lo + side, hi + side, 0.0065, knob=1.2, segs=1)
                return list(bm.verts)
            piece(f"bone_frac_forearm_{s}_{k}", f"lowerarm_{s}", fore)
            def shin(bm, s=s, k=k):
                a, b = head(f"calf_{s}"), tail(f"calf_{s}")
                m = a.lerp(b, 0.5 + 0.05 * (1 if k == 0 else -1))
                lo, hi = (a, m) if k == 0 else (m, b)
                _rod(bm, lo, hi, 0.012, knob=1.2, segs=1)
                return list(bm.verts)
            piece(f"bone_frac_shin_{s}_{k}", f"calf_{s}", shin)
    log(f"skeleton: {util.tri_count(skel)} tris; {len(pieces)} fractured pieces, {sum(util.tri_count(p) for p in pieces)} tris")
    return skel, pieces
