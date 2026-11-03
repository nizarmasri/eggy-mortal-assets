"""Paint over the baked gear atlas, per piece, from each piece's pattern coordinates (`param`).

apply(fighter, names, slot, prm, A, R, ctx): names[i] is the material of slot i ("g_<piece>"), slot the slot
of every texel, prm its pattern coordinates, A linear albedo, R roughness. ctx carries the trousers' heights
(waist_z, knee_z, hem_z, crotch). Legwear param: x = 1 on the outer side, -1 inner; y = height; z = 1 front.
Pieces made round a bone (tape, bands): x along the bone, y round it. Tops: x across, y height, z forward.
"""
import numpy as np

import util
from util import smoothstep


def lin(c):
    return util.srgb_to_linear(np.array(c, dtype=np.float32))


def _paint(A, R, m, colour, rough=None):
    m = np.clip(m, 0, 1)
    shade = A.mean(axis=-1, keepdims=True) / max(float(A[m > 0.5].mean()) if (m > 0.5).any() else 1.0, 1e-3)
    shade = np.clip(shade, 0.55, 1.35)
    A = A * (1 - m[..., None]) + lin(colour) * shade * m[..., None]
    if rough is not None:
        R = R * (1 - m) + rough * m
    return A, R


GOLD = (0.8, 0.47, 0.03)


def _stitch(v, period, width=0.12):
    """Dashes along a line coordinate."""
    return (np.mod(v / period, 1.0) < 0.55).astype(float)


def _laces(A, R, on, x, y, z, lo, hi, colour, tongue, width=0.32, pitch=0.018):
    """Boot laces up the front (legwear param): a tongue, and laces crossed in an X over it."""
    panel = on & (z > 0.78) & (np.abs(x) < width) & (y > lo) & (y < hi)
    A, R = _paint(A, R, panel, tongue, 0.6)
    a = np.abs(np.mod(y / pitch + x / (2 * width), 1.0) - 0.5) < 0.13
    b = np.abs(np.mod(y / pitch - x / (2 * width), 1.0) - 0.5) < 0.13
    A, R = _paint(A, R, panel & (a | b) & (np.abs(x) < width * 0.9), colour, 0.55)
    return A, R


def _sun(x, y, cx, cy, r0, r1, rays=12):
    """A disc and pointed rays round it (pattern coordinates in metres)."""
    rr = np.hypot(x - cx, y - cy)
    ang = np.arctan2(y - cy, x - cx)
    ray = np.abs(np.mod(ang * rays / (2 * np.pi) + 0.5, 1.0) - 0.5) < 0.45 * np.clip(1 - (rr - r0 * 1.15) / (r1 - r0 * 1.15), 0, 1)
    return (rr < r0) | ((rr > r0 * 1.15) & (rr < r1) & ray)


def elyema(names, slot, prm, A, R, ctx):
    x, y, z = prm[..., 0], prm[..., 1], prm[..., 2]
    knee = ctx.get("knee_z", 0.5)
    waist = ctx.get("waist_z", 1.0)
    for i, n in enumerate(names):
        on = slot == i
        if n == "g_tights":
            f = np.clip((x - 0.15) / 0.85, 0, 1)
            top = knee + 0.06 + (0.10 + 0.16 * f) * (0.55 + 0.45 * np.cos(z * np.pi * 2.6))
            gold = on & (x > 0.15) & (y < top)
            trim = on & (x > 0.12) & (y >= top) & (y < top + 0.009)
            band = on & (y > (y[on].max() if on.any() else waist) - 0.05)
            A, R = _paint(A, R, gold | band, GOLD, 0.3)
            A, R = _paint(A, R, trim, (0.95, 0.93, 0.88), 0.35)
        elif n == "g_boots":
            # Crimson wrestling boots: a gold cuff and side stripe, a black sole.
            top = y[on].max() if on.any() else 0.5
            A, R = _paint(A, R, on & (y > top - 0.035), GOLD, 0.3)
            A, R = _paint(A, R, on & (x > 0.82) & (y > 0.03), GOLD, 0.3)
            A, R = _paint(A, R, on & (y < 0.022), (0.05, 0.045, 0.045), 0.7)
        elif n == "g_cape":
            xm = np.abs(x[on]).max() if on.any() else 0.3
            ym = y[on].min() if on.any() else 0.9
            trim = on & ((np.abs(x) > xm - 0.028) | (y < ym + 0.028))
            A, R = _paint(A, R, trim, GOLD, 0.3)
            star = on & (np.hypot(x, y - (y[on].max() - 0.13 if on.any() else 1.2)) < 0.05)
            A, R = _paint(A, R, star, GOLD, 0.3)
    return A, R


def duke(names, slot, prm, A, R, ctx):
    x, y, z = prm[..., 0], prm[..., 1], prm[..., 2]
    for i, n in enumerate(names):
        on = slot == i
        if n == "g_trunks":
            # A white waistband with a green line through it, white piping at the leg hems.
            top = y[on].max() if on.any() else 1.0
            hem = np.percentile(y[on], 1.5) if on.any() else 0.6
            band = on & (y > top - 0.06)
            A, R = _paint(A, R, band, (0.95, 0.94, 0.9), 0.35)
            A, R = _paint(A, R, on & (y > top - 0.04) & (y < top - 0.027), (0.0, 0.42, 0.2), 0.35)
            A, R = _paint(A, R, on & (y < hem + 0.018), (0.95, 0.94, 0.9), 0.35)
        elif n == "g_boots":
            # Black boxing boots: a white cuff with a green line, a white sole.
            top = y[on].max() if on.any() else 0.5
            A, R = _laces(A, R, on, x, y, z, 0.075, top - 0.04, (0.92, 0.9, 0.86), (0.1, 0.1, 0.1))
            A, R = _paint(A, R, on & (y > top - 0.03), (0.92, 0.9, 0.86), 0.45)
            A, R = _paint(A, R, on & (np.abs(y - (top - 0.015)) < 0.004), (0.0, 0.42, 0.2), 0.45)
            A, R = _paint(A, R, on & (y < 0.022), (0.85, 0.83, 0.8), 0.7)
    return A, R


def sorn(names, slot, prm, A, R, ctx):
    x, y, z = prm[..., 0], prm[..., 1], prm[..., 2]
    for i, n in enumerate(names):
        on = slot == i
        if n == "g_shorts":
            top = y[on].max() if on.any() else 1.0
            hem = y[on].min() if on.any() else 0.7
            band = on & (y > top - 0.075)
            A, R = _paint(A, R, band, (0.95, 0.94, 0.9), 0.3)
            A, R = _paint(A, R, band & (np.abs(y - (top - 0.04)) < 0.008), (0.75, 0.05, 0.08), 0.3)
            side = on & (x > 0.55) & ~band
            A, R = _paint(A, R, side, (0.75, 0.05, 0.08), 0.28)
            A, R = _paint(A, R, on & ~band & (np.abs(x - 0.55) < 0.05), (0.92, 0.72, 0.25), 0.3)
            A, R = _paint(A, R, on & (y < hem + 0.012), (0.92, 0.72, 0.25), 0.3)
        elif n == "g_mongkol":
            st = np.mod(x * 18 + y * 1.0, 1.0)
            A, R = _paint(A, R, on & (st < 0.33), (0.75, 0.05, 0.08), 0.6)
            A, R = _paint(A, R, on & (st >= 0.66), (0.06, 0.16, 0.52), 0.6)
        elif n == "g_armbands":
            st = np.mod(x * 25 + y * 0.6, 1.0)
            A, R = _paint(A, R, on & (st < 0.5), (0.06, 0.16, 0.52), 0.6)
    return A, R


def bia(names, slot, prm, A, R, ctx):
    x, y, z = prm[..., 0], prm[..., 1], prm[..., 2]
    cols = [(0.02, 0.5, 0.24), (0.98, 0.8, 0.05), (0.03, 0.16, 0.55)]
    for i, n in enumerate(names):
        on = slot == i
        if n == "g_corda":
            k = np.floor(np.mod(x * 14 + y * 3, 3)).astype(int)
            for c in range(3):
                A, R = _paint(A, R, on & (k == c), cols[c], 0.65)
        elif n == "g_top":
            bottom = y[on].min() if on.any() else 1.2
            A, R = _paint(A, R, on & (y < bottom + 0.014), (0.98, 0.8, 0.05), 0.5)
    return A, R


def mara(names, slot, prm, A, R, ctx):
    x, y, z = prm[..., 0], prm[..., 1], prm[..., 2]
    knee = ctx.get("knee_z", 0.5)
    for i, n in enumerate(names):
        on = slot == i
        if n == "g_jacket":
            q1 = np.abs(np.mod((x + y) / 0.032, 1.0) - 0.5) < 0.04
            q2 = np.abs(np.mod((x - y) / 0.032, 1.0) - 0.5) < 0.04
            upper = y > (y[on].max() - 0.22 if on.any() else 1.3)
            A, R = _paint(A, R, on & upper & (q1 | q2), (0.16, 0.16, 0.17), 0.7)
        elif n == "g_pants":
            pad = on & (np.abs(y - (knee + 0.02)) < 0.075) & (z > 0.25)
            edge = on & (np.abs(np.abs(y - (knee + 0.02)) - 0.075) < 0.004) & (z > 0.25)
            A, R = _paint(A, R, pad, (0.2, 0.21, 0.21), 0.85)
            A, R = _paint(A, R, edge, (0.12, 0.12, 0.12), 0.85)
        elif n == "g_shoes":
            # Sambo shoes: a grey rubber sole under the black leather.
            A, R = _paint(A, R, on & (y < 0.017), (0.16, 0.155, 0.15), 0.75)
    return A, R


def gu(names, slot, prm, A, R, ctx):
    x, y, z = prm[..., 0], prm[..., 1], prm[..., 2]
    r = util.rng(12)
    for i, n in enumerate(names):
        on = slot == i
        if n in ("g_trousers", "g_tunic"):
            for _ in range(5):
                cx, cy = r.uniform(-0.8, 0.8) if n == "g_trousers" else r.uniform(-0.15, 0.15), r.uniform(0.3, 1.3)
                w, h = r.uniform(0.04, 0.08), r.uniform(0.04, 0.09)
                px = x if n == "g_tunic" else x * 0.12
                patch = on & (np.abs(px - cx * (0.12 if n == "g_trousers" else 1)) < w / 2) & (np.abs(y - cy) < h / 2)
                border = patch & ((np.abs(np.abs(px - cx * (0.12 if n == "g_trousers" else 1)) - w / 2) < 0.003) | (np.abs(np.abs(y - cy) - h / 2) < 0.003))
                shade = r.choice([(0.16, 0.18, 0.24), (0.24, 0.22, 0.2), (0.12, 0.13, 0.16)])
                A, R = _paint(A, R, patch, shade, 0.9)
                A, R = _paint(A, R, border & (_stitch(px + y, 0.008) > 0.5), (0.55, 0.5, 0.42), 0.9)
    return A, R


FIGHTERS = {"elyema": elyema, "duke": duke, "sorn": sorn, "bia": bia, "mara": mara, "gu": gu}


def for_fighter(fid, ctx):
    fn = FIGHTERS.get(fid)
    if fn is None:
        return None
    return lambda names, slot, prm, A, R: fn(names, slot, prm, A, R, ctx)
