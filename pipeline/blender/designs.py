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


def rim_line(cv, alb, rough, colour, inset=0.006, width=0.0018, gloss=0.12):
    """A thin painted line following the mask's edge, a hand's width in."""
    e = cv["edge"]
    wob = 0.0004 * (util.fbm3(cv["P"] * 60, 2, seed=41) - 0.5)
    a = smoothstep(width * 0.5 + 0.0003, width * 0.5 - 0.0003, np.abs(e - inset + wob)) * cv["valid"]
    alb = alb * (1 - a[..., None]) + lin(colour) * a[..., None]
    rough = rough * (1 - a) + gloss * a
    return alb, rough, a


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
    # Past either end the nearest sample is the end itself: those texels are outside the stroke.
    along = (xs - pts[best_i, 0]) * tang[best_i, 0] + (ys - pts[best_i, 1]) * tang[best_i, 1]
    beyond = ((best_i == 0) & (along < 0)) | ((best_i == n - 1) & (along > 0))
    off = np.where(beyond, 9.0, off)
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
    alb, rough, _ = rim_line(cv, alb, rough, (0.16, 0.03, 0.03))
    # The seal: below the fighter's left eye, near the jaw.
    sl = ctx["slits"]["eye_l"]
    sa = seal(X, Y, (sl[0] + 0.012, sl[1] + 0.095), 0.016, 5)
    alb = alb * (1 - sa[..., None]) + lin((0.62, 0.06, 0.05)) * sa[..., None]
    rough = rough * (1 - sa) + 0.4 * sa
    return alb, rough, metal, height


def _eyes(ctx):
    el, er = ctx["slits"]["eye_l"], ctx["slits"]["eye_r"]
    return abs(el[0]), 0.5 * (el[1] + er[1])


def _hash2(i, j, seed=0):
    a, b = np.minimum(i, j), np.maximum(i, j)
    h = (a * 73856093) ^ (b * 19349663) ^ (seed * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def seams(ctx, cv, width, pick=1.0, seed=0, wobble=0.0004):
    """Lines along the shard walls (the walls Cell Fracture cut along): opacity, and which wall."""
    wall = ctx["wall"]
    edge_id = _hash2(ctx["cells"], ctx["nbr"], seed)
    keep = edge_id < pick
    wv = width * (0.65 + 0.7 * util.fbm3(cv["P"] * 45, 2, seed=seed + 3))
    wob = wobble * (util.fbm3(cv["P"] * 90, 2, seed=seed + 4) - 0.5)
    a = smoothstep(wv * 0.5 + 0.00025, wv * 0.5 - 0.00025, np.abs(wall + wob)) * keep * cv["valid"]
    return a, edge_id


def mara(cv, ctx):
    """Black urushi lacquer, repaired in gold along some of the walls it will one day break along."""
    P = cv["P"]
    alb = np.broadcast_to(lin((0.028, 0.024, 0.026)), cv["X"].shape + (3,)).copy()
    alb *= (0.85 + 0.3 * util.fbm3(P * 6, 3, seed=51))[..., None]
    rough = 0.05 + 0.04 * util.fbm3(P * 25, 2, seed=52)
    metal = np.zeros(cv["X"].shape)
    height = 0.00003 * (util.fbm3(P * 200, 2, seed=53) - 0.5)
    a, _ = seams(ctx, cv, 0.0016, pick=0.5, seed=7)
    halo, _ = seams(ctx, cv, 0.006, pick=0.5, seed=7)
    gold = lin((1.0, 0.80, 0.42))
    alb = alb * (1 - 0.25 * halo[..., None]) + gold * 0.06 * halo[..., None]
    alb = alb * (1 - a[..., None]) + gold * a[..., None]
    rough = rough * (1 - a) + (0.18 + 0.1 * util.fbm3(P * 300, 2, seed=54)) * a
    metal = np.maximum(metal, a)
    height = height + 0.0002 * a
    return alb, rough, metal, height


def elyema(cv, ctx):
    """Luchador paint on a crimson enamel egg: gold flames rising from the eyes, a golden-yolk sun on the
    brow with a centre stripe over the crown, white trim round every gold edge, gold round the mouth."""
    X, Y = cv["X"], cv["Y"]
    ex, ey = _eyes(ctx)
    mx, my = ctx["slits"].get("mouth", (0, ey + 0.08))
    alb = np.broadcast_to(lin((0.62, 0.04, 0.07)), X.shape + (3,)).copy()
    alb *= (0.94 + 0.1 * util.fbm3(cv["P"] * 8, 2, seed=61))[..., None]
    rough = 0.16 + 0.04 * util.fbm3(cv["P"] * 30, 2, seed=62)
    gold_f = np.zeros(X.shape)
    for side in (1, -1):
        dx, dy = X - side * ex, Y - ey
        r = np.hypot(dx, dy)
        ang = np.arctan2(-dy, dx * side)  # up = +pi/2, outward = 0
        flame = 0.024 + 0.03 * np.clip(np.sin(ang), 0, 1) ** 1.5 * (0.6 + 0.4 * np.abs(np.sin(ang * 5 + 0.6)))
        flame += 0.012 * np.clip(np.cos(ang), 0, 1)
        gold_f = np.maximum(gold_f, smoothstep(flame + 0.0008, flame - 0.0008, r))
    # The golden yolk on the brow, with rays.
    sx, sy = 0.0, ey - 0.075
    r = np.hypot(X - sx, Y - sy)
    ang = np.arctan2(Y - sy, X - sx)
    rays = 0.03 + 0.017 * (np.cos(ang * 12) > 0.2) * smoothstep(0.02, 0.03, r)
    sun = smoothstep(rays + 0.0007, rays - 0.0007, r)
    stripe = (np.abs(X) < 0.005) & (Y < sy)
    mouth = (np.abs(X - mx) < 0.032) & (np.abs(Y - my) < 0.011)
    gold_f = np.clip(np.maximum.reduce([gold_f, sun, stripe.astype(float), mouth.astype(float) * 0.999]), 0, 1)
    # White trim: a band just outside every gold shape.
    def grow(m, k):
        out = m.copy()
        for _ in range(k):
            out = np.maximum.reduce([out, np.roll(out, 1, 0), np.roll(out, -1, 0), np.roll(out, 1, 1), np.roll(out, -1, 1)])
        return out
    trim = np.clip(grow(gold_f, 4) - gold_f, 0, 1)
    yolk = smoothstep(0.0215, 0.0195, r)
    gold = lin((0.98, 0.72, 0.18))
    alb = alb * (1 - trim[..., None]) + lin((0.95, 0.93, 0.88)) * trim[..., None]
    alb = alb * (1 - gold_f[..., None]) + gold * gold_f[..., None]
    alb = alb * (1 - yolk[..., None]) + lin((1.0, 0.62, 0.05)) * yolk[..., None]
    metal = 0.85 * gold_f * (1 - yolk)
    rough = rough * (1 - gold_f) + 0.24 * gold_f
    height = 0.00015 * gold_f + 0.0001 * trim
    return alb, rough, metal, height


def bia(cv, ctx):
    """Green enamel, a gold rhombus across the face, a blue disc on the brow crossed by a white band of
    stars, a dotted white border: painted by hand."""
    X, Y = cv["X"], cv["Y"]
    ex, ey = _eyes(ctx)
    alb = np.broadcast_to(lin((0.02, 0.50, 0.24)), X.shape + (3,)).copy()
    alb *= (0.93 + 0.12 * util.fbm3(cv["P"] * 7, 2, seed=71))[..., None]
    rough = 0.14 + 0.05 * util.fbm3(cv["P"] * 30, 2, seed=72)
    cy = ey + 0.012
    wob = 0.0015 * (util.fbm3(cv["P"] * 40, 2, seed=73) - 0.5)
    rh = np.abs(X) / 0.135 + np.abs(Y - cy) / 0.125
    gold = smoothstep(1.0 + 0.01, 1.0 - 0.01, rh + wob * 20)
    dots = np.zeros(X.shape)
    edge_d = np.abs(rh - 1.06)
    along = np.arctan2(Y - cy, X)
    dots = (edge_d < 0.03) * smoothstep(0.55, 0.75, np.cos(along * 70))
    by = ey - 0.07
    disc = smoothstep(0.047, 0.045, np.hypot(X, Y - by))
    band_c = by + 0.06 - np.sqrt(np.maximum(0, 0.075 ** 2 - X ** 2))
    band = (np.abs(Y - band_c - 0.004) < 0.0045) * disc
    stars = np.zeros(X.shape)
    r = util.rng(9)
    for _ in range(14):
        sx, sy = r.uniform(-0.035, 0.035), by + r.uniform(-0.03, 0.035)
        if sx * sx + (sy - by) ** 2 > 0.04 ** 2:
            continue
        stars = np.maximum(stars, smoothstep(0.0022, 0.0012, np.hypot(X - sx, Y - sy)))
    alb = alb * (1 - gold[..., None]) + lin((0.99, 0.80, 0.05)) * gold[..., None]
    alb = alb * (1 - disc[..., None]) + lin((0.03, 0.16, 0.55)) * disc[..., None]
    white = np.clip(band + stars * disc + dots * (1 - gold), 0, 1)
    alb = alb * (1 - white[..., None]) + lin((0.96, 0.95, 0.92)) * white[..., None]
    height = 0.0001 * gold + 0.00008 * white
    return alb, rough, np.zeros(X.shape), height


DENTS = [(-0.05, -0.06, 0.03, 0.005), (0.07, 0.02, 0.024, 0.004), (-0.02, 0.10, 0.02, 0.003), (0.03, -0.10, 0.035, 0.006), (-0.09, 0.06, 0.018, 0.003)]


def _dents(X, Y, ctx):
    ex, ey = _eyes(ctx)
    h = np.zeros(np.shape(X))
    for dx, dy, r, d in DENTS:
        h = h - d * np.exp(-(((X - dx) ** 2 + (Y - (ey + dy)) ** 2) / (r * r)))
    return h


def duke(cv, ctx):
    """Chrome, dented by years of punches, scuffed, wrapped in strips of white athletic tape."""
    X, Y = cv["X"], cv["Y"]
    ex, ey = _eyes(ctx)
    P = cv["P"]
    alb = np.broadcast_to(lin((0.86, 0.87, 0.9)), X.shape + (3,)).copy()
    scr = util.fbm3(np.stack([X * 900, Y * 25, P[..., 2] * 3], axis=-1), 2, seed=81)
    rough = 0.08 + 0.18 * smoothstep(0.62, 0.85, scr) + 0.08 * util.fbm3(P * 20, 3, seed=82)
    metal = np.ones(X.shape)
    height = 0.00002 * (scr - 0.5)
    # Tape strips: (start, end, width).
    tape = np.zeros(X.shape)
    for (x0, y0), (x1, y1), w, sd in (((-0.16, ey - 0.13), (0.06, ey - 0.03), 0.034, 1), ((0.075, ey + 0.035), (0.13, ey + 0.16), 0.03, 2),
                                      ((-0.14, ey + 0.09), (-0.02, ey + 0.13), 0.028, 3)):
        d = np.array([x1 - x0, y1 - y0])
        L = np.linalg.norm(d)
        d /= L
        t = (X - x0) * d[0] + (Y - y0) * d[1]
        o = (X - x0) * -d[1] + (Y - y0) * d[0]
        torn = 0.004 * (util.value_noise1(o * 900, sd) - 0.5)
        inside = (np.abs(o) < w / 2 + 0.0006 * (util.value_noise1(t * 600, sd + 7) - 0.5)) & (t > torn) & (t < L + torn)
        tape = np.maximum(tape, inside.astype(float))
    wrinkle = util.fbm3(np.stack([X * 300, Y * 300, np.zeros_like(X)], axis=-1), 3, seed=83)
    dirt = util.fbm3(P * 30, 3, seed=84)
    tcol = lin((0.88, 0.86, 0.80)) * (0.85 + 0.2 * wrinkle[..., None]) * (1 - 0.25 * smoothstep(0.55, 0.8, dirt)[..., None])
    alb = alb * (1 - tape[..., None]) + tcol * tape[..., None]
    rough = rough * (1 - tape) + 0.78 * tape
    metal = metal * (1 - tape)
    height = height + _dents(X, Y, ctx) * 0.0 + tape * (0.0004 + 0.0002 * wrinkle)
    return alb, rough, metal, height


def sorn(cv, ctx):
    """Porcelain painted in cobalt like a sak yant: nine spires on the brow over a row of dots, lines of
    looping script along the cheeks, a fine border."""
    X, Y = cv["X"], cv["Y"]
    ex, ey = _eyes(ctx)
    alb, rough, metal, height = porcelain(cv, tint=(0.95, 0.95, 0.93), seed=91)
    ink = np.zeros(X.shape)
    base_y = ey - 0.045
    for k in range(9):
        cx = (k - 4) * 0.0105
        top = base_y - (0.07 - 0.008 * abs(k - 4))
        w = 0.0032 * np.clip((Y - top) / (base_y - top), 0, 1)
        spire = (np.abs(X - cx) < w) & (Y > top) & (Y < base_y)
        ink = np.maximum(ink, spire.astype(float))
        ink = np.maximum(ink, smoothstep(0.0028, 0.0018, np.hypot(X - cx, Y - (top - 0.006))))
    arcs = np.abs(np.hypot(X, Y - (base_y + 0.05)) - 0.05) < 0.0011
    ink = np.maximum(ink, (arcs & (Y < base_y + 0.005) & (np.abs(X) < 0.05)).astype(float))
    # Script: rows of small loops following arcs under the eyes and along the jaw.
    for row, (yoff, curve) in enumerate(((0.03, 0.9), (0.05, 1.1), (0.115, 0.7), (0.135, 0.8))):
        yc = ey + yoff + curve * X ** 2
        band = np.abs(Y - yc) < 0.0065
        px = X / 0.0075
        loop = np.abs(np.hypot(np.mod(px, 1.0) - 0.5, (Y - yc) / 0.0075 * 1.6) - 0.32) < 0.075
        tick = (np.mod(px + 0.5 * row, 1.0) < 0.08) & (Y - yc > -0.006) & (Y - yc < 0.0)
        keep = util.value_noise1(np.floor(px), row) > 0.15
        mask = band & (loop | tick) & keep & (np.abs(X) < 0.095) & (np.abs(X) > (0.03 if row < 2 else 0.0))
        if row >= 2:
            mask &= ~((np.abs(X - ctx["slits"].get("mouth", (0, 0))[0]) < 0.03) & (np.abs(Y - ctx["slits"].get("mouth", (0, ey + 0.08))[1]) < 0.012))
        ink = np.maximum(ink, mask.astype(float))
    cobalt = lin((0.06, 0.16, 0.52))
    bleed = util.fbm3(cv["P"] * 120, 2, seed=92)
    a = np.clip(ink * (0.85 + 0.15 * bleed), 0, 1)
    alb = alb * (1 - a[..., None]) + cobalt * a[..., None]
    alb, rough, _ = rim_line(cv, alb, rough, (0.06, 0.16, 0.52), inset=0.007, width=0.0022)
    return alb, rough, metal, height


def han(cv, ctx):
    """White porcelain split red over blue by one sweeping S across a great circle on the face, short black
    bars at four corners."""
    X, Y = cv["X"], cv["Y"]
    ex, ey = _eyes(ctx)
    alb, rough, metal, height = porcelain(cv, tint=(0.96, 0.95, 0.93), seed=101)
    cy = ey + 0.03
    R = 0.115
    r = np.hypot(X, Y - cy)
    circle = smoothstep(R + 0.0008, R - 0.0008, r)
    # The S: red where above the line made of two half-circles.
    up = np.hypot(X + R / 2, Y - cy) < R / 2
    down = np.hypot(X - R / 2, Y - cy) < R / 2
    red = ((Y < cy) & ~down) | up
    red_c, blue_c = lin((0.78, 0.05, 0.12)), lin((0.02, 0.18, 0.52))
    col = np.where(red[..., None], red_c, blue_c)
    alb = alb * (1 - circle[..., None]) + col * circle[..., None]
    rough = rough * (1 - circle) + 0.12 * circle
    bars = np.zeros(X.shape)
    for sx, sy, ang, pattern in ((-1, -1, 0.6, (1, 1, 1)), (1, -1, -0.6, (1, 0, 1)), (-1, 1, -0.6, (0, 1, 0)), (1, 1, 0.6, (0, 0, 0))):
        cx, cyy = sx * 0.13, cy + sy * 0.125
        c, s_ = np.cos(ang), np.sin(ang)
        u = (X - cx) * c + (Y - cyy) * s_
        v = -(X - cx) * s_ + (Y - cyy) * c
        for k, solid in enumerate(pattern):
            vv = v - (k - 1) * 0.008
            bar = (np.abs(vv) < 0.0028) & (np.abs(u) < 0.017)
            if not solid:
                bar &= np.abs(u) > 0.003
            bars = np.maximum(bars, bar.astype(float))
    alb = alb * (1 - bars[..., None]) + lin((0.04, 0.04, 0.05)) * bars[..., None]
    height = height + 0.0001 * circle
    return alb, rough, metal, height


def gu(cv, ctx):
    """Unglazed clay, broken and glued back many times: every shard a slightly different batch, amber glue
    along every wall, iron staples across some, chips, soot and the ghost of old paint."""
    X, Y = cv["X"], cv["Y"]
    ex, ey = _eyes(ctx)
    P = cv["P"]
    cells = ctx["cells"]
    r = util.rng(5)
    shade = np.array([r.uniform(0.82, 1.15) for _ in range(64)])
    hue = np.array([r.uniform(-0.04, 0.04) for _ in range(64)])
    base = lin((0.60, 0.38, 0.25))
    alb = np.broadcast_to(base, X.shape + (3,)).copy()
    alb *= shade[cells][..., None]
    alb[..., 0] *= 1 + hue[cells]
    alb[..., 2] *= 1 - hue[cells]
    grit = util.fbm3(P * 160, 3, seed=111)
    alb *= (0.85 + 0.25 * grit)[..., None]
    soot = smoothstep(0.55, 0.85, util.fbm3(P * 9, 4, seed=112))
    alb *= (1 - 0.45 * soot)[..., None]
    rough = 0.82 + 0.1 * grit
    metal = np.zeros(X.shape)
    height = 0.00015 * (grit - 0.5)
    # Ghost of old paint: a faded black brow and a red cheek, rubbed away.
    worn = smoothstep(0.45, 0.7, util.fbm3(P * 35, 3, seed=113))
    brow = (np.abs(np.abs(X) - ex) < 0.03) & (np.abs(Y - (ey - 0.022 - 0.15 * (np.abs(X) - ex) ** 2)) < 0.004)
    cheek = np.hypot(np.abs(X) - ex, Y - (ey + 0.045)) < 0.016
    alb = alb * (1 - (brow * worn * 0.7)[..., None]) + lin((0.05, 0.04, 0.04)) * (brow * worn * 0.7)[..., None]
    alb = alb * (1 - (cheek * worn * 0.45)[..., None]) + lin((0.55, 0.1, 0.08)) * (cheek * worn * 0.45)[..., None]
    # Glue on every wall, with drips.
    glue, eid = seams(ctx, cv, 0.0026, pick=1.0, seed=17, wobble=0.0012)
    gcol = lin((0.45, 0.27, 0.08))
    alb = alb * (1 - glue[..., None] * 0.85) + gcol * 0.85 * glue[..., None]
    rough = rough * (1 - glue) + 0.22 * glue
    height = height + 0.00035 * glue
    # Staples across a few walls: short iron bars perpendicular to the wall.
    seeds = ctx["seeds"]
    staple = np.zeros(X.shape)
    count = 0
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            if count > 13:
                break
            mid = 0.5 * (seeds[i] + seeds[j])
            d = seeds[j] - seeds[i]
            if np.linalg.norm(d) > 0.075 or _hash2(np.array(i), np.array(j), 3) > 0.35:
                continue
            d /= np.linalg.norm(d)
            u = (X - mid[0]) * d[0] + (Y - mid[1]) * d[1]
            v = -(X - mid[0]) * d[1] + (Y - mid[1]) * d[0]
            bar = (np.abs(u) < 0.0065) & (np.abs(v) < 0.0011)
            pins = (np.hypot(np.abs(u) - 0.0065, v) < 0.0016)
            staple = np.maximum(staple, (bar | pins).astype(float) * ((ctx["cells"] == i) | (ctx["cells"] == j)))
            count += 1
    alb = alb * (1 - staple[..., None]) + lin((0.22, 0.2, 0.19)) * staple[..., None]
    metal = np.maximum(metal, staple)
    rough = rough * (1 - staple) + 0.45 * staple
    height = height + 0.0008 * staple
    return alb, rough, metal, height


def default(cv, ctx):
    return porcelain(cv)


GEOMETRY = {"duke": _dents}


DESIGNS = {"reiji": reiji, "mara": mara, "elyema": elyema, "bia": bia, "duke": duke, "sorn": sorn, "han": han, "gu": gu}


def paint(fighter, cv, ctx):
    return DESIGNS.get(fighter, default)(cv, ctx)
