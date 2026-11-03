"""Each fighter's painted mask, in metres on the egg's surface.

paint(fighter, cv, ctx) gets the canvas (mask.canvas: X across, Y down from the top pole, P the 3D point,
N the normal) and ctx (slit centres in canvas metres, the shard seeds and the crack-map cells) and returns
linear albedo (h, w, 3), roughness, metalness and height in metres (h, w). Everything is deterministic.
"""
import math

import numpy as np

import util
from util import fbm3, smoothstep, value_noise1


def lin(rgb_srgb):
    return util.srgb_to_linear(np.array(rgb_srgb, dtype=np.float32))


def porcelain(cv, tint=(0.94, 0.93, 0.90), seed=3):
    """Glazed white porcelain: a warm white, slow glaze pooling, glossy, a faint orange-peel ripple."""
    P = cv["P"]
    shape = cv["X"].shape
    pool = fbm3(P * 9.0, octaves=3, seed=seed)
    speck = fbm3(P * 140.0, octaves=2, seed=seed + 9)
    base = lin(tint)
    alb = np.broadcast_to(base, shape + (3,)).copy()
    alb *= (0.975 + 0.05 * pool)[..., None]
    alb *= (1 - 0.035 * smoothstep(0.72, 0.9, speck))[..., None]
    rough = 0.07 + 0.05 * fbm3(P * 30.0, octaves=2, seed=seed + 4)
    height = 0.00004 * (fbm3(P * 220.0, octaves=2, seed=seed + 5) - 0.5)
    metal = np.zeros(shape)
    return alb, rough, metal, height


def _bezier(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + (t ** 3) * p3


def stroke_field(X, Y, ctrl, width, seed=0, n=500):
    """Distance field of a brush stroke along a cubic Bezier: returns (t along, s across in [-1, 1], inside)."""
    pts = _bezier(*[np.array(c, dtype=np.float64) for c in ctrl], n)
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seglen)])
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    pad = width * 1.5
    lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    box = (X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1])
    t = np.zeros(X.shape)
    s = np.full(X.shape, 9.0)
    xs, ys = X[box], Y[box]
    best_d = np.full(xs.shape, 1e9)
    best_i = np.zeros(xs.shape, dtype=int)
    for k in range(0, n, 50):
        d = (xs[:, None] - pts[None, k:k + 50, 0]) ** 2 + (ys[:, None] - pts[None, k:k + 50, 1]) ** 2
        j = np.argmin(d, axis=1)
        dj = d[np.arange(len(j)), j]
        upd = dj < best_d
        best_d[upd] = dj[upd]
        best_i[upd] = j[upd] + k
    tt = L[best_i] / L[-1]
    off = (xs - pts[best_i, 0]) * nrm[best_i, 0] + (ys - pts[best_i, 1]) * nrm[best_i, 1]
    t[box] = tt
    s[box] = off
    return t, s, L[-1]


def brush_stroke(X, Y, ctrl, width, seed=0, dry_start=0.55):
    """Opacity (0..1) and ink thickness of a calligraphy stroke: a pressed start, a long taper, dry-brush
    streaks breaking up the tail, ragged edges and pooled ink at the rims."""
    t, off, length = stroke_field(X, Y, ctrl, width, seed)
    # Width profile: a pressed blob at the start, swelling, then a long taper.
    prof = (np.clip(t / 0.06, 0, 1) ** 0.5) * (1 - smoothstep(0.55, 1.0, t) ** 1.4) * (0.85 + 0.25 * np.exp(-((t - 0.12) / 0.1) ** 2))
    rag = 0.12 * (value_noise1(t * length * 90, seed) - 0.5) + 0.08 * (value_noise1(t * length * 23, seed + 1) - 0.5)
    half = 0.5 * width * np.maximum(prof, 0) * (1 + rag)
    s = off / np.maximum(half, 1e-6)
    inside = np.abs(s) < 1
    edge = smoothstep(1.0, 0.86, np.abs(s))
    # Bristle streaks run along the stroke; they open up as the brush dries.
    lanes = value_noise1((s + 1) * 26, seed + 3) * 0.6 + value_noise1((s + 1) * 71, seed + 4) * 0.4
    dryness = smoothstep(dry_start, 1.0, t) * 0.9 + 0.25 * smoothstep(0.75, 1.0, np.abs(s))
    streak = smoothstep(dryness - 0.08, dryness + 0.08, lanes + 0.15 * (value_noise1(t * length * 400 + (s + 1) * 13, seed + 5) - 0.5))
    alpha = np.where(inside, edge * streak, 0.0)
    pool = smoothstep(0.55, 0.95, np.abs(s)) * inside
    return alpha, pool, t, s


def splatter(X, Y, centre, direction, count, spread, rmin, rmax, seed):
    r = util.rng(seed)
    a = np.zeros(X.shape)
    d = np.array(direction) / np.linalg.norm(direction)
    for _ in range(count):
        along = r.uniform(0.0, spread)
        side = r.gauss(0, spread * 0.25)
        cx = centre[0] + d[0] * along - d[1] * side
        cy = centre[1] + d[1] * along + d[0] * side
        rad = r.uniform(rmin, rmax) * (1 - 0.6 * along / spread)
        dist = np.hypot(X - cx, Y - cy)
        a = np.maximum(a, smoothstep(rad, rad * 0.75, dist))
    return a


def seal(X, Y, centre, size, seed):
    """A small square red seal (hanko) with an abstract carved glyph: opacity field."""
    r = util.rng(seed)
    lx, ly = (X - centre[0]) / size, (Y - centre[1]) / size
    inside = (np.abs(lx) < 0.5) & (np.abs(ly) < 0.5)
    border = inside & ((np.abs(lx) > 0.42) | (np.abs(ly) > 0.42))
    glyph = np.zeros(X.shape, dtype=bool)
    for _ in range(7):
        if r.random() < 0.5:
            y = r.choice([-0.25, -0.08, 0.08, 0.25])
            x0, x1 = sorted([r.uniform(-0.32, 0.1), r.uniform(-0.1, 0.32)])
            glyph |= (np.abs(ly - y) < 0.045) & (lx > x0) & (lx < x1)
        else:
            x = r.choice([-0.25, -0.08, 0.08, 0.25])
            y0, y1 = sorted([r.uniform(-0.32, 0.1), r.uniform(-0.1, 0.32)])
            glyph |= (np.abs(lx - x) < 0.045) & (ly > y0) & (ly < y1)
    wear = value_noise1(lx * 37 + ly * 91, seed) > 0.18
    return ((border | (inside & ~glyph)) & wear).astype(np.float64)


# ---------- fighters ----------

def reiji(cv, ctx):
    """White porcelain, one bold vermilion brushstroke across the face, a small seal under the left eye."""
    X, Y = cv["X"], cv["Y"]
    alb, rough, metal, height = porcelain(cv)
    ey = 0.5 * (ctx["slits"]["eye_l"][1] + ctx["slits"]["eye_r"][1])
    ex = abs(ctx["slits"]["eye_l"][0])
    ctrl = [(-ex - 0.075, ey - 0.095), (-ex * 0.2, ey - 0.07), (ex * 0.3, ey + 0.035), (ex + 0.07, ey + 0.115)]
    a, pool, t, s = brush_stroke(X, Y, ctrl, width=0.052, seed=11)
    a = np.maximum(a, splatter(X, Y, (ctrl[0][0] - 0.004, ctrl[0][1] - 0.004), (-0.8, -0.5), 14, 0.05, 0.0012, 0.0045, 21))
    ink = lin((0.70, 0.05, 0.04))
    ink_dark = lin((0.42, 0.015, 0.02))
    col = ink * (1 - 0.55 * pool[..., None]) + ink_dark * 0.55 * pool[..., None]
    alb = alb * (1 - a[..., None]) + col * a[..., None]
    rough = rough * (1 - a) + 0.30 * a
    height = height + 0.00012 * a * (0.6 + 0.4 * pool)
    # The seal: below the fighter's left eye, near the jaw.
    sl = ctx["slits"]["eye_l"]
    sa = seal(X, Y, (sl[0] + 0.012, sl[1] + 0.095), 0.016, 5)
    alb = alb * (1 - sa[..., None]) + lin((0.62, 0.06, 0.05)) * sa[..., None]
    rough = rough * (1 - sa) + 0.4 * sa
    return alb, rough, metal, height


def default(cv, ctx):
    return porcelain(cv)


DESIGNS = {"reiji": reiji}


def paint(fighter, cv, ctx):
    return DESIGNS.get(fighter, default)(cv, ctx)
