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


def warp_index(cv, amp, freq, seed):
    """Index maps that displace a canvas field by up to about amp metres along a smooth noise: a straight
    wall sampled through them meanders like a real crack."""
    X, Y = cv["X"], cv["Y"]
    px = max(abs(float(X[0, 1] - X[0, 0])), 1e-6)
    py = max(abs(float(Y[1, 0] - Y[0, 0])), 1e-6)
    P = cv["P"]
    dx = amp * 2.5 * (util.fbm3(P * freq, 3, seed=seed) - 0.5)
    dy = amp * 2.5 * (util.fbm3(P * freq + 17.3, 3, seed=seed + 1) - 0.5)
    h, w = X.shape
    ii = np.clip(np.round(np.arange(h)[:, None] + dy / py), 0, h - 1).astype(np.int64)
    jj = np.clip(np.round(np.arange(w)[None, :] + dx / px), 0, w - 1).astype(np.int64)
    return ii, jj


def seams(ctx, cv, width, pick=1.0, seed=0, wobble=0.0004, warp=0.0, warp_freq=55.0):
    """Lines along the shard walls (the walls Cell Fracture cut along): opacity, and which wall. With warp,
    the line meanders up to about warp metres off the wall, as a hand-repaired crack does."""
    wall, cells, nbr = ctx["wall"], ctx["cells"], ctx["nbr"]
    if warp:
        ii, jj = warp_index(cv, warp, warp_freq, seed + 11)
        wall, cells, nbr = wall[ii, jj], cells[ii, jj], nbr[ii, jj]
    edge_id = _hash2(cells, nbr, seed)
    keep = edge_id < pick
    wv = width * (0.65 + 0.7 * util.fbm3(cv["P"] * 45, 2, seed=seed + 3))
    wob = wobble * (util.fbm3(cv["P"] * 90, 2, seed=seed + 4) - 0.5)
    a = smoothstep(wv * 0.5 + 0.00025, wv * 0.5 - 0.00025, np.abs(wall + wob)) * keep * cv["valid"]
    return a, edge_id


def veins(cv, near, width=0.0006, freq=38.0, seed=0):
    """Hairline cracks branching off a seam: thin noise ridges, only within `near` (a 0..1 field)."""
    r = 1 - np.abs(2 * util.fbm3(cv["P"] * freq, 4, seed=seed) - 1)
    line = smoothstep(1 - width * 22, 1 - width * 8, r)
    return line * near * cv["valid"]


def mara(cv, ctx):
    """Black urushi lacquer, broken and repaired in gold (kintsugi): meandering gold seams of uneven width
    along some of the walls it will one day break along, hairline gold veins branching off them."""
    P = cv["P"]
    alb = np.broadcast_to(lin((0.03, 0.026, 0.028)), cv["X"].shape + (3,)).copy()
    alb *= (0.85 + 0.3 * util.fbm3(P * 6, 3, seed=51))[..., None]
    rough = 0.11 + 0.06 * util.fbm3(P * 25, 2, seed=52)
    metal = np.zeros(cv["X"].shape)
    height = 0.00003 * (util.fbm3(P * 200, 2, seed=53) - 0.5)
    a, _ = seams(ctx, cv, 0.0021, pick=0.55, seed=7, warp=0.0045)
    near, _ = seams(ctx, cv, 0.03, pick=0.55, seed=7, warp=0.0045)
    v = veins(cv, near * smoothstep(0.2, 0.6, util.fbm3(P * 14, 2, seed=55)), seed=56)
    gold_a = np.clip(a + 0.85 * v, 0, 1)
    halo, _ = seams(ctx, cv, 0.0048, pick=0.55, seed=7, warp=0.0045)
    gold = lin((1.0, 0.78, 0.38))
    alb = alb * (1 - 0.1 * halo[..., None]) + gold * 0.03 * halo[..., None]
    alb = alb * (1 - gold_a[..., None]) + gold * (0.85 + 0.25 * util.fbm3(P * 400, 2, seed=57))[..., None] * gold_a[..., None]
    rough = rough * (1 - gold_a) + (0.2 + 0.12 * util.fbm3(P * 300, 2, seed=54)) * gold_a
    metal = np.maximum(metal, gold_a)
    height = height + 0.00022 * a + 0.00008 * v
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
        # Flames round each eye: an almond round the slit and three tongues of fire licking up and out
        # towards the temple, each tapering to a sharp point, and a short point down the cheek.
        def tongue(centre, reach, half):
            d = np.abs((ang - math.radians(centre) + np.pi) % (2 * np.pi) - np.pi)
            return reach * np.clip(1 - d / math.radians(half), 0, 1) ** 1.6
        rim = 0.02 + 0.006 * np.clip(np.cos(ang), 0, 1)
        flame = rim + np.maximum.reduce([tongue(18, 0.05, 24), tongue(48, 0.066, 22), tongue(82, 0.04, 20), tongue(-35, 0.016, 22)])
        gold_f = np.maximum(gold_f, smoothstep(flame + 0.0008, flame - 0.0008, r))
    # The golden yolk on the brow, with rays.
    sx, sy = 0.0, ey - 0.075
    r = np.hypot(X - sx, Y - sy)
    ang = np.arctan2(Y - sy, X - sx)
    rays = 0.03 + 0.017 * (np.cos(ang * 12) > 0.2) * smoothstep(0.02, 0.03, r)
    sun = smoothstep(rays + 0.0007, rays - 0.0007, r)
    stripe = (np.abs(X) < 0.005) & (Y < sy)
    mouth = (np.hypot((X - mx) / 0.034, (Y - my) / 0.014) < 1.0)
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
    """Green enamel crossed by three ribbons of colour sweeping up from the jaw to the temple, the way her
    fitas fly in the ginga: gold over the eyes, blue under it, a thin green-white pinstripe between them,
    and a line of white beads round the edge."""
    X, Y = cv["X"], cv["Y"]
    P = cv["P"]
    ex, ey = _eyes(ctx)
    alb = np.broadcast_to(lin((0.01, 0.42, 0.21)), X.shape + (3,)).copy()
    alb *= (0.92 + 0.14 * util.fbm3(P * 7, 2, seed=71))[..., None]
    rough = 0.13 + 0.05 * util.fbm3(P * 30, 2, seed=72)
    # A coordinate across the sweep: rotated, bent into an S, a little hand wobble.
    a = math.radians(-28)
    u = X * math.cos(a) + (Y - ey) * math.sin(a)
    w_ = -X * math.sin(a) + (Y - ey) * math.cos(a)
    f = w_ + 0.035 * np.sin(u / 0.05) + 0.0012 * (util.fbm3(P * 50, 2, seed=73) - 0.5) * 2
    gold = smoothstep(-0.0275, -0.0265, f) * smoothstep(0.0215, 0.0205, f)
    blue = smoothstep(0.0285, 0.0295, f) * smoothstep(0.0565, 0.0555, f)
    white = smoothstep(0.0215, 0.0225, f) * smoothstep(0.0285, 0.0275, f)
    pin = smoothstep(0.0236, 0.0246, f) * smoothstep(0.0264, 0.0254, f)
    alb = alb * (1 - gold[..., None]) + lin((0.98, 0.76, 0.06)) * gold[..., None]
    alb = alb * (1 - blue[..., None]) + lin((0.02, 0.15, 0.52)) * blue[..., None]
    alb = alb * (1 - white[..., None]) + lin((0.95, 0.94, 0.9)) * white[..., None]
    alb = alb * (1 - pin[..., None]) + lin((0.01, 0.42, 0.21)) * pin[..., None]
    # A smaller ribbon echo on the brow, gold edged in blue.
    f2 = f + 0.085
    g2 = smoothstep(-0.008, -0.007, f2) * smoothstep(0.008, 0.007, f2)
    alb = alb * (1 - g2[..., None]) + lin((0.98, 0.76, 0.06)) * g2[..., None]
    e2 = (smoothstep(0.0075, 0.0085, np.abs(f2)) * smoothstep(0.0115, 0.0105, np.abs(f2)))
    alb = alb * (1 - e2[..., None]) + lin((0.02, 0.15, 0.52)) * e2[..., None]
    # Beads round the edge.
    e = cv["edge"]
    beads = smoothstep(0.0026, 0.0016, np.hypot(e - 0.009, (np.mod(Y, 0.011) - 0.0055))) * cv["valid"]
    alb = alb * (1 - beads[..., None]) + lin((0.96, 0.95, 0.92)) * beads[..., None]
    paint = np.clip(gold + blue + white + g2 + e2, 0, 1)
    rough = rough * (1 - paint) + 0.1 * paint
    height = 0.00009 * paint + 0.00025 * beads
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
    """Porcelain painted in cobalt like a sak yant: nine bold spires on the brow over a bar and a row of dots,
    an unalome spiralling up the bridge of the nose, rows of looping script along both cheeks and the jaw,
    a fine border."""
    X, Y = cv["X"], cv["Y"]
    ex, ey = _eyes(ctx)
    alb, rough, metal, height = porcelain(cv, tint=(0.95, 0.95, 0.93), seed=91)
    ink = np.zeros(X.shape)
    base_y = ey - 0.038
    for k in range(9):
        cx = (k - 4) * 0.0128
        top = base_y - (0.082 - 0.009 * abs(k - 4))
        f = np.clip((Y - top) / (base_y - top), 0, 1)
        w = 0.0012 + 0.0042 * f ** 0.8
        spire = (np.abs(X - cx) < w) & (Y > top) & (Y < base_y)
        ink = np.maximum(ink, spire.astype(float))
        ink = np.maximum(ink, smoothstep(0.0034, 0.0024, np.hypot(X - cx, Y - (top - 0.007))))
    bar = (np.abs(Y - (base_y + 0.003)) < 0.0028) & (np.abs(X) < 0.062)
    ink = np.maximum(ink, bar.astype(float))
    for k in range(-5, 6):
        ink = np.maximum(ink, smoothstep(0.0024, 0.0016, np.hypot(X - k * 0.0112, Y - (base_y + 0.0115))))
    # The unalome: a spiral at the bottom, a zigzag rising from it, three dots above.
    sy = ey + 0.052
    rr = np.hypot(X, Y - sy)
    th = np.arctan2(Y - sy, X)
    spiral_r = 0.0035 + 0.0045 * ((th + np.pi) / (2 * np.pi) + np.floor(rr / 0.0045))
    spiral = (np.abs(rr - (0.0035 + 0.0045 * np.round((rr - 0.0035 - 0.0045 * (th + np.pi) / (2 * np.pi)) / 0.0045) + 0.0045 * (th + np.pi) / (2 * np.pi))) < 0.0009) & (rr < 0.0125)
    ink = np.maximum(ink, spiral.astype(float))
    zz_top, zz_bot = ey - 0.012, sy - 0.0125
    zt = np.clip((zz_bot - Y) / (zz_bot - zz_top), 0, 1)
    zig = 0.0045 * (2 * np.abs(np.mod(zt * 3, 1.0) - 0.5) * 2 - 1) * (1 - zt * 0.5)
    ink = np.maximum(ink, ((np.abs(X - zig) < 0.0012) & (Y > zz_top) & (Y < zz_bot)).astype(float))
    for k in range(3):
        ink = np.maximum(ink, smoothstep(0.0019, 0.0012, np.hypot(X, Y - (zz_top - 0.006 - 0.0055 * k))))
    # Script: running lines of looped, joined letters along arcs under the eyes and the jaw: a wave with a
    # small closed loop on every crest and a gap between words.
    mx, my = ctx["slits"].get("mouth", (0, ey + 0.08))
    for row, (yoff, curve, xmin) in enumerate(((0.034, 0.9, 0.03), (0.058, 1.0, 0.03), (0.118, 0.7, 0.0), (0.142, 0.75, 0.0))):
        yc = ey + yoff + curve * X ** 2
        per = 0.0105
        ph = X / per + 0.37 * row
        wave = yc + 0.0022 * np.sin(2 * np.pi * ph)
        line = np.abs(Y - wave) < 0.00095
        crest = np.round(ph - 0.25) + 0.25
        cxl = (crest - 0.37 * row) * per
        loop = np.abs(np.hypot(X - cxl, Y - (yc - 0.0022 - 0.0017)) - 0.0017) < 0.0007
        word = np.floor(ph / 4.0)
        gap = np.mod(ph, 4.0) > 3.55
        keep = util.value_noise1(word * 1.0, row + 3) > -1.0
        tick = (np.abs(X - cxl) < 0.0007) & (Y > yc + 0.0018) & (Y < yc + 0.0042) & (util.value_noise1(np.round(ph) * 1.0, row + 7) > 0.6)
        mask = (line | loop | tick) & keep & ~gap & (np.abs(X) < 0.1) & (np.abs(X) > xmin)
        mask &= ~((np.abs(X - mx) < 0.034) & (np.abs(Y - my) < 0.014))
        ink = np.maximum(ink, mask.astype(float))
    cobalt = lin((0.05, 0.14, 0.5))
    bleed = util.fbm3(cv["P"] * 120, 2, seed=92)
    a = np.clip(ink * (0.88 + 0.12 * bleed), 0, 1) * cv["valid"]
    alb = alb * (1 - a[..., None]) + cobalt * a[..., None]
    height = height + 0.00006 * a
    alb, rough, _ = rim_line(cv, alb, rough, (0.05, 0.14, 0.5), inset=0.007, width=0.0026)
    return alb, rough, metal, height


def han(cv, ctx):
    """White porcelain split red over blue by one sweeping S across a great circle on the face, outlined in a
    fine black brush line, a red line round the rim."""
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
    # A fine black brush line round the circle and down the S, and a red rim line.
    up_d = np.abs(np.hypot(X + R / 2, Y - cy) - R / 2)
    dn_d = np.abs(np.hypot(X - R / 2, Y - cy) - R / 2)
    s_line = np.where(X < 0, np.where(Y >= cy, up_d, 9.0), np.where(Y < cy, dn_d, 9.0))
    line = np.maximum(smoothstep(0.0016, 0.0008, np.abs(r - R)), smoothstep(0.0013, 0.0006, s_line) * circle)
    alb = alb * (1 - line[..., None]) + lin((0.03, 0.03, 0.035)) * line[..., None]
    alb, rough, _ = rim_line(cv, alb, rough, (0.62, 0.05, 0.1), inset=0.0065, width=0.002)
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
    base = lin((0.50, 0.34, 0.24))
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
