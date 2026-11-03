"""The ceramic egg-shell mask: a half egg fitted to the head, eye and mouth slits, a painted canvas in its UVs,
pre-fractured with Cell Fracture into 25-40 shards, and a crack map that matches those shards.

The shell is an analytic egg, so every texel of its painted canvas has an exact 3D point: designs are painted
in metres on the surface (designs.py), and the crack map is the exact distance to the Voronoi cell walls Cell
Fracture cut along.

Egg-local frame: centre C, tilted back by `tilt` about X. Parameters phi (around, 0 = front, + towards the
fighter's left, +X) and theta (0 at the top pole, pi at the chin tip). rho(theta) = sin(theta)(1 + k cos(theta))
makes the top end blunter than the chin end.

UV layout (Blender, v up):
  outer glaze    u 0..1,    v 0.25..1    the canvas, metric (same metres per UV both ways)
  inner bisque   u 0..0.62, v 0..0.24
  rim, slit walls and break faces   u 0.65..1, v 0..0.24
"""
import math
import random

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

import util
from util import log

OUTER_V0 = 0.25
THICK = 0.004


class Egg:
    def __init__(self, C, a, b, h, k=0.12, tilt=math.radians(8), phimax=math.radians(96), p=1.0):
        self.C = np.array(C, dtype=np.float64)
        self.a, self.b, self.h, self.k, self.tilt, self.phimax, self.p = a, b, h, k, tilt, phimax, p
        c, s = math.cos(tilt), math.sin(tilt)
        # Rotation about X by +tilt takes local to world (the top leans towards +Y, the back).
        self.R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        self._tables()

    def rho(self, th):
        # p < 1 fills the shoulders of the ends a little (a superellipse profile); k makes the top the blunt end.
        return np.sin(th) ** self.p * (1 + self.k * np.cos(th))

    def local(self, phi, th, inset=0.0):
        r = self.rho(th)
        a, b, h = self.a - inset, self.b - inset, self.h - inset
        return np.stack([a * r * np.sin(phi), -b * r * np.cos(phi), h * np.cos(th)], axis=-1)

    def world(self, phi, th, inset=0.0):
        return self.local(phi, th, inset) @ self.R.T + self.C

    def to_local(self, p):
        return (np.asarray(p, dtype=np.float64) - self.C) @ self.R

    def params(self, p_world):
        q = self.to_local(p_world)
        th = np.arccos(np.clip(q[..., 2] / self.h, -1, 1))
        phi = np.arctan2(q[..., 0] / self.a, -q[..., 1] / self.b)
        r = np.sqrt((q[..., 0] / self.a) ** 2 + (q[..., 1] / self.b) ** 2) / np.maximum(self.rho(th), 1e-6)
        return phi, th, r

    def normal(self, phi, th):
        e = 1e-4
        p = self.world(phi, th)
        du = self.world(phi + e, th) - p
        dv = self.world(phi, th + e) - p
        n = np.cross(dv, du)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        return n

    def _tables(self):
        """Arc lengths: S[i, j] along the latitude of theta_i from the front meridian to phi_j, and the
        meridian arc from the top pole to theta_i."""
        self.th_t = np.linspace(0, math.pi, 721)
        self.ph_t = np.linspace(0, self.phimax, 401)
        TH, PH = np.meshgrid(self.th_t, self.ph_t, indexing="ij")
        r = self.rho(TH)
        speed = r * np.sqrt((self.a * np.cos(PH)) ** 2 + (self.b * np.sin(PH)) ** 2)
        dph = self.ph_t[1] - self.ph_t[0]
        self.S = np.concatenate([np.zeros((len(self.th_t), 1)), np.cumsum((speed[:, 1:] + speed[:, :-1]) * 0.5 * dph, axis=1)], axis=1)
        p = self.local(np.zeros_like(self.th_t), self.th_t)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        self.M = np.concatenate([[0], np.cumsum(seg)])
        self.width = 2 * float(self.S[:, -1].max())
        self.height = float(self.M[-1])
        self.scale = min(1.0 / self.width, (1 - OUTER_V0 - 0.01) / self.height)  # UV units per metre

    def canvas_xy(self, phi, th):
        """Metric canvas coordinates (X along the latitude, signed; Y down the meridian from the top)."""
        Y = np.interp(th, self.th_t, self.M)
        ti = np.clip(np.searchsorted(self.th_t, th), 1, len(self.th_t) - 1)
        X = np.empty_like(np.asarray(phi, dtype=np.float64))
        aphi = np.abs(phi)
        # Interpolate S along phi for each point's latitude row (nearest row is plenty at 721 rows).
        rows = self.S[ti]
        j = np.clip(aphi / self.ph_t[-1] * (len(self.ph_t) - 1), 0, len(self.ph_t) - 1.000001)
        j0 = np.floor(j).astype(int)
        f = j - j0
        X = rows[np.arange(len(ti)), j0] * (1 - f) + rows[np.arange(len(ti)), j0 + 1] * f
        return np.sign(phi) * X, Y

    def uv_of(self, X, Y):
        return 0.5 + X * self.scale, 1 - 0.005 - Y * self.scale

    def xy_of_uv(self, u, v):
        return (u - 0.5) / self.scale, (1 - 0.005 - v) / self.scale

    def params_of_xy(self, X, Y):
        """Inverse of canvas_xy: (phi, theta, valid)."""
        th = np.interp(Y, self.M, self.th_t)
        ti = np.clip(np.searchsorted(self.th_t, th), 1, len(self.th_t) - 1)
        aX = np.abs(X)
        phi = np.empty_like(aX)
        valid = (Y >= 0) & (Y <= self.height) & (aX <= self.S[ti, -1])
        # Row-wise inverse interpolation: rows share a latitude when the canvas is sampled row by row.
        for t in np.unique(ti):
            sel = ti == t
            phi[sel] = np.interp(aX[sel], self.S[t], self.ph_t)
        return np.sign(X) * phi, th, valid


# ---------- fitting ----------

def _group_ids(body, name):
    g = body.vertex_groups.get(name)
    if g is None:
        return np.zeros(0, dtype=int)
    return np.array([v.index for v in body.data.vertices if any(x.group == g.index for x in v.groups)], dtype=int)


def fit(body, eyes, spec=None):
    """An egg sized from the face's landmarks: its sides end just in front of the ears, its chin tip a little
    under the chin, its top over the crown, and it clears the nose, brows and the hair shell by a margin."""
    spec = spec or {}
    P = util.positions(body)
    N = util.normals(body)
    W = util.weights_matrix(body, ["head"])
    head = W[:, 0] >= 0.5
    ears = _group_ids(body, "ears")
    lips = _group_ids(body, "lips")
    E = util.positions(eyes)
    eye_z = float(E[:, 2].mean())
    eye_y = float(E[:, 1].mean())
    H = P[head]
    crown = float(H[:, 2].max())
    mouth_z = float(P[lips, 2].mean()) if len(lips) else eye_z - 0.075
    chin_z = mouth_z - spec.get("chin_below_mouth", 0.048)
    ear_front = float(P[ears, 1].min()) if len(ears) else eye_y + 0.07
    face = head & (P[:, 2] > chin_z - 0.005) & (P[:, 1] < ear_front)
    face[ears] = False
    nose_band = face & (np.abs(P[:, 0]) < 0.02) & (P[:, 2] < eye_z) & (P[:, 2] > mouth_z)
    nose_y = float(P[nose_band, 1].min())
    xmax = float(np.abs(P[face, 0]).max())
    C = np.array([0.0, ear_front - spec.get("edge_ahead", 0.004), 0.5 * (chin_z - spec.get("tip_below_chin", 0.014) + crown + spec.get("top_over_crown", 0.004))])
    h = 0.5 * (crown + spec.get("top_over_crown", 0.004) - (chin_z - spec.get("tip_below_chin", 0.014)))
    a = xmax + spec.get("side_gap", 0.016)
    egg = Egg(C, a, 0.1, h, k=spec.get("k", 0.16), tilt=math.radians(spec.get("tilt", 6)), phimax=math.radians(90), p=spec.get("p", 0.82))
    # Depth: the nose tip and a clearance in front of it, at the nose's latitude.
    nz = egg.to_local(np.array([[0, nose_y, eye_z - 0.03]]))[0]
    th_n = math.acos(max(-1, min(1, nz[2] / h)))
    b = (-nz[1] + spec.get("nose_gap", 0.012) + THICK) / max(egg.rho(th_n), 0.3)
    egg = Egg(C, a, b, h, k=egg.k, tilt=egg.tilt, phimax=egg.phimax, p=egg.p)
    # Everything under the shell must clear its inner surface: the face, and the scalp pushed out by the hair.
    pts = np.concatenate([P[face] + N[face] * spec.get("clearance", 0.008), E + np.array([0, -0.004, 0])])
    for it in range(40):
        phi, th, r = egg.params(pts)
        covered = np.abs(phi) < egg.phimax
        limit = 1 - (THICK + 0.0015) / min(egg.a, egg.b)
        bad = covered & (r > limit)
        if bad.sum() < 3:
            break
        q = egg.to_local(pts[bad])
        top = q[:, 2] / egg.h > 0.45
        bottom = q[:, 2] / egg.h < -0.45
        side = (np.abs(q[:, 0]) / egg.a > 0.6) & ~top & ~bottom
        front = ~top & ~side & ~bottom
        a2, b2, h2, C2 = egg.a, egg.b, egg.h, egg.C.copy()
        if top.mean() > 0.25:
            # The skull's dome: lengthen the egg upwards, keeping the chin tip where it is.
            h2 = egg.h * 1.008
            C2[2] += (h2 - egg.h)
        if bottom.mean() > 0.25:
            # The jaw: lengthen it downwards (the narrow end slips under the chin), the top where it is.
            h2 = h2 * 1.008
            C2[2] -= (h2 - egg.h) if top.mean() <= 0.25 else 0.0
        if side.mean() > 0.25:
            a2 *= 1.008
        if front.mean() > 0.25:
            b2 *= 1.008
        egg = Egg(C2, a2, b2, h2, k=egg.k, tilt=egg.tilt, phimax=egg.phimax, p=egg.p)
    log(f"egg: C={egg.C.round(4).tolist()} a={egg.a:.4f} b={egg.b:.4f} h={egg.h:.4f} p={egg.p} k={egg.k} (nose {nose_y:.3f}, ears {ear_front:.3f}, chin {chin_z:.3f}, crown {crown:.3f}, {it} grow steps)")
    return egg


# ---------- geometry ----------

def _grid(egg, n_phi=36, n_th=30):
    verts, faces, idx = [], [], {}
    phis = np.linspace(-egg.phimax, egg.phimax, n_phi + 1)
    ths = np.linspace(0, math.pi, n_th + 1)
    # Denser near the eyes' latitude: remap theta a little.
    top = len(verts)
    verts.append(egg.world(0.0, 0.0))
    for i in range(1, n_th):
        for j, ph in enumerate(phis):
            idx[i, j] = len(verts)
            verts.append(egg.world(ph, ths[i]))
    bottom = len(verts)
    verts.append(egg.world(0.0, math.pi))
    for j in range(n_phi):
        faces.append((top, idx[1, j + 1], idx[1, j]))
        faces.append((bottom, idx[n_th - 1, j], idx[n_th - 1, j + 1]))
    for i in range(1, n_th - 1):
        for j in range(n_phi):
            faces.append((idx[i, j], idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j]))
    return verts, faces


def _slit_cutter(egg, center_world, width, height, tilt_out, side, name, droop=0.0):
    """A prism through the shell with an almond cross-section, centred where center_world projects."""
    phi, th, _ = egg.params(np.array([center_world]))
    phi, th = float(phi[0]), float(th[0])
    S = egg.world(phi, th)
    n = egg.normal(np.array(phi), np.array(th))
    e = 1e-3
    du = egg.world(phi + e, th) - S
    du /= np.linalg.norm(du)
    dv = np.cross(n, du)  # up-ish
    if dv[2] < 0:
        dv = -dv
    pts = []
    m = 28
    for i in range(m):
        t = 2 * math.pi * i / m
        x = 0.5 * width * math.cos(t)
        s = math.sin(t)
        shape = (1 - (2 * x / width) ** 2) ** 0.75
        y = 0.5 * height * s * shape
        if s < 0:
            y *= 0.8  # flatter lower lid
        # Outer corner raised (side = +1 on the fighter's left, where du points outward).
        y += tilt_out * (x / (0.5 * width)) * side * height * 0.5
        y -= droop * (1 - (2 * x / width) ** 2) * height
        pts.append(S + du * x + dv * y)
    bm = bmesh.new()
    inner = [bm.verts.new(tuple(p - n * 0.03)) for p in pts]
    outer = [bm.verts.new(tuple(p + n * 0.03)) for p in pts]
    for i in range(m):
        j = (i + 1) % m
        bm.faces.new((inner[i], inner[j], outer[j], outer[i]))
    bm.faces.new(list(reversed(inner)))
    bm.faces.new(outer)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return util.bmesh_object(name, bm), (phi, th)


def build(egg, eyes, body, spec):
    """The intact mask object (rest pose, not yet skinned) and the slit centres in canvas metres."""
    verts, faces = _grid(egg, spec.get("n_phi", 36), spec.get("n_th", 30))
    shell = util.mesh_object("mask", verts, faces)
    # Faces point outward?
    me = shell.data
    p = me.polygons[len(me.polygons) // 2]
    c = np.array(p.center)
    if np.dot(np.array(p.normal), c - egg.C) < 0:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
    sol = shell.modifiers.new("Solidify", 'SOLIDIFY')
    sol.thickness = THICK
    sol.offset = -1
    sol.use_even_offset = True
    sol.use_quality_normals = True
    sol.use_rim = True
    util.apply_modifier(shell, sol.name)

    # Eye slits over the eyes, and a mouth slit.
    E = util.positions(eyes)
    slits = []
    cutters = []
    for side in (1, -1):
        sel = E[:, 0] * side > 0
        ec = E[sel].mean(axis=0)
        cut, pt = _slit_cutter(egg, ec, spec.get("eye_w", 0.041), spec.get("eye_h", 0.0125), spec.get("eye_tilt", 0.35), side, f"cut_eye_{side}")
        cutters.append(cut)
        slits.append(("eye_l" if side > 0 else "eye_r", pt))
    lips = body.vertex_groups.get("lips")
    if spec.get("mouth", "slit") == "slit" and lips is not None:
        P = util.positions(body)
        ids = [v.index for v in body.data.vertices if any(g.group == lips.index for g in v.groups)]
        mc = P[ids].mean(axis=0) + np.array([0, -0.01, -0.002])
        cut, pt = _slit_cutter(egg, mc, spec.get("mouth_w", 0.034), spec.get("mouth_h", 0.0042), 0.0, 1, "cut_mouth")
        cutters.append(cut)
        slits.append(("mouth", pt))
    for cut in cutters:
        mod = shell.modifiers.new("Cut", 'BOOLEAN')
        mod.operation = 'DIFFERENCE'
        mod.solver = 'EXACT'
        mod.object = cut
        util.apply_modifier(shell, mod.name)
        util.delete(cut)
    util.shade_smooth(shell, sharp_angle=38)
    centres = {}
    for name, (phi, th) in slits:
        X, Y = egg.canvas_xy(np.array([phi]), np.array([th]))
        centres[name] = (float(X[0]), float(Y[0]))
    log("mask shell:", util.tri_count(shell), "tris; slits", {k: tuple(round(x, 4) for x in v) for k, v in centres.items()})
    return shell, centres


def uv_unwrap(obj, egg, break_faces=None):
    """UVs from the analytic egg: outer faces on the canvas, inner faces and walls in the bisque strip."""
    me = obj.data
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    uv = me.uv_layers[0]
    M = obj.matrix_world
    L = len(me.loops)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    co = co @ np.array(M.to_3x3()).T + np.array(M.translation)
    lv = np.empty(L, dtype=np.int64)
    me.loops.foreach_get("vertex_index", lv)
    out = np.zeros((L, 2))
    lmin = min(egg.a, egg.b)
    for p in me.polygons:
        loops = list(range(p.loop_start, p.loop_start + p.loop_total))
        pc = co[lv[loops]]
        cen = pc.mean(axis=0)
        n = np.array((M.to_3x3() @ p.normal).normalized())
        phi_c, th_c, r_c = egg.params(cen[None])
        grad = egg.normal(phi_c, th_c)[0]
        d = float(np.dot(n, grad))
        is_break = break_faces is not None and break_faces[p.index]
        phi, th, r = egg.params(pc)
        if not is_break and d > 0.5:
            X, Y = egg.canvas_xy(phi, th)
            u, v = egg.uv_of(X, Y)
            out[loops] = np.stack([u, v], axis=1)
        elif not is_break and d < -0.5:
            X, Y = egg.canvas_xy(phi, th)
            out[loops] = np.stack([0.31 + X * egg.scale * 0.6, 0.235 - Y * egg.scale * 0.3], axis=1)
        else:
            out[loops] = np.stack([0.66 + 0.33 * (phi + egg.phimax) / (2 * egg.phimax), 0.01 + 0.22 * th / math.pi], axis=1)
    uv.data.foreach_set("uv", out.reshape(-1))
    me.update()


def canvas(egg, size=1024):
    """Per-texel canvas metres, surface points and validity for the outer region of a size x size map."""
    h0 = int(round(size * OUTER_V0))
    rows = np.arange(h0, size)
    v = (rows + 0.5) / size  # Blender v of each image row (row 0 is v = 0, at the bottom)
    u = (np.arange(size) + 0.5) / size
    U, V = np.meshgrid(u, v)
    X, Y = egg.xy_of_uv(U, V)
    phi, th, valid = egg.params_of_xy(X.ravel(), Y.ravel())
    P = egg.world(phi, th)
    Nrm = egg.normal(phi, th)
    shape = U.shape
    ti = np.clip(np.searchsorted(egg.th_t, th), 1, len(egg.th_t) - 1)
    edge = (egg.S[ti, -1] - np.abs(X.ravel())).reshape(shape)
    return {
        "edge": edge,
        "size": size, "row0": h0, "X": X, "Y": Y,
        "phi": phi.reshape(shape), "th": th.reshape(shape),
        "P": P.reshape(shape + (3,)), "N": Nrm.reshape(shape + (3,)),
        "valid": valid.reshape(shape),
    }


# ---------- fracture ----------

def seeds(egg, count=30, seed=1):
    """Fracture seeds on the shell's mid-surface, spread by best-candidate sampling on the canvas."""
    r = random.Random(seed)
    pts = []
    tries = 0
    while len(pts) < count and tries < 20000:
        best, best_d = None, -1
        for _ in range(24):
            Y = r.uniform(0.02, egg.height - 0.02)
            th = float(np.interp(Y, egg.M, egg.th_t))
            ti = int(np.clip(np.searchsorted(egg.th_t, th), 1, len(egg.th_t) - 1))
            Xmax = float(egg.S[ti, -1]) - 0.012
            if Xmax <= 0:
                continue
            X = r.uniform(-Xmax, Xmax)
            d = min((math.hypot(X - x, Y - y) for x, y in pts), default=1e9)
            if d > best_d:
                best, best_d = (X, Y), d
        tries += 1
        if best:
            pts.append(best)
    XY = np.array(pts)
    phi, th, _ = egg.params_of_xy(XY[:, 0], XY[:, 1])
    P = egg.world(phi, th, inset=THICK / 2)
    return P, XY


def fracture(mask_obj, seed_points, egg):
    """Cell Fracture the mask along the seeds' Voronoi cells (at 10x scale: the add-on merges vertices
    closer than 5 mm, which would fold a 4 mm shell). Returns the shard objects, rest pose, world space."""
    import addon_utils
    for mod in addon_utils.modules():
        if mod.__name__.endswith("cell_fracture"):
            addon_utils.enable(mod.__name__, default_set=True)
    S = 10.0
    work = util.duplicate(mask_obj, "mask_frac")
    work.data.materials.clear()
    for name in ("mask", "mask_break"):
        m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        work.data.materials.append(m)
    work.matrix_world = Matrix.Scale(S, 4)
    util.apply_transforms(work)
    pts = util.mesh_object("frac_points", [tuple(p * S) for p in seed_points], [])
    pts.parent = work
    random.seed(7)
    util.select_only(work)
    bpy.ops.object.add_fracture_cell_objects(
        source={'VERT_CHILD'}, source_limit=0, source_noise=0.0, cell_scale=(1, 1, 1),
        recursion=0, use_smooth_faces=False, use_sharp_edges=True, use_sharp_edges_apply=False,
        use_data_match=True, use_island_split=True, margin=0.0, material_index=1,
        use_interior_vgroup=False, mass_mode='VOLUME', mass=1.0, use_recenter=True,
        use_remove_original=False, collection_name="", use_debug_points=False,
        use_debug_redraw=False, use_debug_bool=False)
    shards = [o for o in bpy.context.selected_objects if o is not work and o.type == 'MESH']
    util.delete(pts, work)
    out = []
    for o in shards:
        # Back to metres around the world origin.
        o.matrix_world = Matrix.Scale(1 / S, 4) @ o.matrix_world
        util.apply_transforms(o)
        if len(o.data.polygons) < 4:
            util.delete(o)
            continue
        out.append(o)
    log("fracture:", len(out), "shards")
    return out


def finish_shards(shards, egg, arm, mat):
    """Name the shards in a stable order (top to bottom, then left to right), UV them, origin at their centre,
    and parent each to the head bone so it rides the head until the renderer sets it free."""
    info = []
    for o in shards:
        P = util.positions(o)
        c = P.mean(axis=0)
        phi, th, _ = egg.params(c[None])
        info.append((round(float(th[0]), 2), float(phi[0]), o))
    info.sort(key=lambda t: (t[0], t[1]))
    head = arm.data.bones["head"]
    out = []
    for i, (_, _, o) in enumerate(info):
        name = f"mask_shard_{i:02d}"
        o.name = o.data.name = name
        me = o.data
        brk = np.array([p.material_index == 1 for p in me.polygons])
        uv_unwrap(o, egg, break_faces=brk)
        me.polygons.foreach_set("material_index", [0] * len(me.polygons))
        me.materials.clear()
        me.materials.append(mat)
        util.shade_smooth(o, sharp_angle=38)
        util.select_only(o)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        mw = o.matrix_world.copy()
        o.parent = arm
        o.parent_type = 'BONE'
        o.parent_bone = "head"
        o.matrix_parent_inverse = Matrix.Identity(4)
        bpy.context.view_layer.update()
        o.matrix_world = mw
        out.append(o)
    return out


def crack_map(cv, seed_points):
    """R: distance to the nearest shard wall (metres / 20 mm, clipped), G: a random value per cell, B: the
    cell's index / 64. Linear data; texels off the canvas get R = 1."""
    P = cv["P"].reshape(-1, 3)
    S = np.asarray(seed_points)
    d2 = ((P[:, None, :] - S[None, :, :]) ** 2).sum(axis=2)
    i1 = np.argmin(d2, axis=1)
    s1 = S[i1]
    dmin = np.full(len(P), 1e9)
    nbr = np.zeros(len(P), dtype=int)
    for j in range(len(S)):
        sj = S[j]
        den = 2 * np.linalg.norm(sj - s1, axis=1)
        num = d2[:, j] - d2[np.arange(len(P)), i1]
        dj = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 1e9)
        dj = np.where(i1 == j, 1e9, dj)
        upd = dj < dmin
        nbr[upd] = j
        dmin = np.minimum(dmin, dj)
    rnd = np.array([random.Random(1000 + k).random() for k in range(len(S))])
    shape = cv["X"].shape
    R = np.clip(dmin / 0.020, 0, 1).reshape(shape)
    G = rnd[i1].reshape(shape)
    B = ((i1 + 0.5) / 64.0).reshape(shape)
    R = np.where(cv["valid"], R, 1.0)
    return np.stack([R, G, B], axis=-1), i1.reshape(shape), dmin.reshape(shape), nbr.reshape(shape)


# ---------- textures ----------

def _fill_rows(a, valid):
    """Pad invalid texels with the nearest valid one along their row (then column), so filtering at the
    mask's edge never reaches an unpainted colour."""
    out = a.copy()
    h, w = valid.shape
    idx = np.where(valid, np.arange(w)[None, :], -1)
    left = np.maximum.accumulate(idx, axis=1)
    idx_r = np.where(valid, np.arange(w)[None, :], w)
    right = np.minimum.accumulate(idx_r[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(h)[:, None]
    take = np.where(left >= 0, left, np.where(right < w, right, 0))
    has = (left >= 0) | (right < w)
    filled = out[rows, take]
    out = np.where(has[..., None] if out.ndim == 3 else has, filled, out)
    return out


def textures(egg, cv, painted, outdir, bisque=(0.80, 0.78, 0.74), bisque_rough=0.85, name="mask"):
    """Full-size albedo, ORM (R occlusion 1, G roughness, B metalness) and normal maps for the mask."""
    import os
    alb, rough, metal, height = painted
    size = cv["size"]
    r0 = cv["row0"]
    valid = cv["valid"]
    alb = _fill_rows(alb, valid)
    rough = _fill_rows(rough, valid)
    metal = _fill_rows(metal, valid)
    height = _fill_rows(height, valid)
    bis = util.srgb_to_linear(np.array(bisque, dtype=np.float32))
    A = np.broadcast_to(bis, (size, size, 3)).copy()
    # The bisque gets a little speckle so the inside and the breaks are not flat.
    yy, xx = np.mgrid[0:size, 0:size] / size
    speck = util.fbm3(np.stack([xx * 40, yy * 40, np.zeros_like(xx)], axis=-1), 3, seed=12)
    A *= (0.93 + 0.1 * speck)[..., None]
    A[r0:] = alb
    Rg = np.full((size, size), bisque_rough)
    Rg[r0:] = rough
    Mt = np.zeros((size, size))
    Mt[r0:] = metal
    Ht = np.zeros((size, size))
    Ht[r0:] = height
    # Normal map from the height field: metres per texel from the canvas scale.
    dx = 1.0 / (egg.scale * size)
    gy, gx = np.gradient(Ht, dx)
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    N = n * 0.5 + 0.5
    os.makedirs(outdir, exist_ok=True)
    img_a = util.image_from_array(f"{name}_albedo", util.linear_to_srgb(A), os.path.join(outdir, f"{name}_albedo.png"))
    orm = np.stack([np.ones_like(Rg), np.clip(Rg, 0.02, 1), np.clip(Mt, 0, 1)], axis=-1)
    img_o = util.image_from_array(f"{name}_orm", orm, os.path.join(outdir, f"{name}_orm.png"), non_color=True)
    nn = util.resize(n, size // 2)
    nn /= np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-6)
    small = nn * 0.5 + 0.5
    img_n = util.image_from_array(f"{name}_normal", small, os.path.join(outdir, f"{name}_normal.png"), non_color=True)
    return img_a, img_o, img_n


def crack_image(cv, crack, outdir, name="mask_crack"):
    import os
    size = cv["size"]
    r0 = cv["row0"]
    full = np.zeros((size, size, 3), dtype=np.float32)
    full[..., 0] = 1.0
    full[r0:] = crack
    small = util.resize(full, size // 2)
    return util.image_from_array(name, small, os.path.join(outdir, f"{name}.png"), non_color=True)


def displace(obj, egg, height_fn):
    """Push the shell along its normals by height_fn(X, Y) (metres; negative = dent), inner and outer alike,
    so big dents show in the silhouette and in the shards."""
    P = util.positions(obj)
    phi, th, _ = egg.params(P)
    X, Y = egg.canvas_xy(phi, th)
    h = height_fn(X, Y)
    n = egg.normal(phi, th)
    util.set_positions(obj, P + n * h[:, None])
