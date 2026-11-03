"""Procedural Cycles materials for the gear, baked into the gear atlas (bake.py), and the final image-based
materials every exported surface wears (also what the previews render).

Gear pieces carry a corner attribute `param` (x along the piece, y across it, z a per-piece value) so a
pattern can follow a wrap, a belt or a strand of hair; fabrics use object (rest) coordinates.
"""
import bpy

import util


class G:
    """A tiny node-graph builder."""

    def __init__(self, mat):
        self.nt = mat.node_tree
        self.x = 0

    def n(self, kind, **kw):
        node = self.nt.nodes.new(kind)
        for k, v in kw.items():
            if k.startswith("_"):
                setattr(node, k[1:], v)
            else:
                node.inputs[k].default_value = v
        node.location = (self.x, 0)
        self.x += 200
        return node

    def link(self, a, b):
        self.nt.links.new(a, b)

    def math(self, op, a, b=None, clamp=False):
        node = self.n("ShaderNodeMath", _operation=op, _use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                node.inputs[i].default_value = v
            else:
                self.link(v, node.inputs[i])
        return node.outputs[0]

    def vmath(self, op, a, b=None):
        node = self.n("ShaderNodeVectorMath", _operation=op)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (tuple, list)):
                node.inputs[i].default_value = v
            else:
                self.link(v, node.inputs[i])
        return node.outputs[0] if op not in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else node.outputs["Value"]

    def mix(self, fac, a, b):
        node = self.n("ShaderNodeMix", _data_type='RGBA', _blend_type='MIX')
        for sock, v in ((node.inputs[0], fac), (node.inputs[6], a), (node.inputs[7], b)):
            if isinstance(v, (tuple, list)):
                sock.default_value = v
            elif isinstance(v, (int, float)):
                sock.default_value = v
            else:
                self.link(v, sock)
        return node.outputs[2]

    def ramp(self, value, stops):
        node = self.n("ShaderNodeValToRGB")
        cr = node.color_ramp
        while len(cr.elements) > len(stops):
            cr.elements.remove(cr.elements[-1])
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for el, (pos, col) in zip(cr.elements, stops):
            el.position = pos
            el.color = col if len(col) == 4 else (*col, 1)
        self.link(value, node.inputs[0])
        return node.outputs[0]

    def noise(self, vector, scale, detail=4.0, rough=0.5, dims='3D'):
        node = self.n("ShaderNodeTexNoise", _noise_dimensions=dims, Scale=scale, Detail=detail, Roughness=rough)
        if vector is not None:
            self.link(vector, node.inputs["Vector"])
        return node.outputs["Fac"]

    def coords(self, kind="Object"):
        return self.n("ShaderNodeTexCoord").outputs[kind]

    def attr(self, name):
        node = self.n("ShaderNodeAttribute", _attribute_name=name, _attribute_type='GEOMETRY')
        return node.outputs["Vector"]

    def sep(self, vec):
        node = self.n("ShaderNodeSeparateXYZ")
        self.link(vec, node.inputs[0])
        return node.outputs[0], node.outputs[1], node.outputs[2]

    def comb(self, x, y, z):
        node = self.n("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            if isinstance(v, (int, float)):
                node.inputs[i].default_value = v
            else:
                self.link(v, node.inputs[i])
        return node.outputs[0]

    def principled(self, color, rough, height=None, strength=1.0, distance=0.002):
        bsdf = self.n("ShaderNodeBsdfPrincipled")
        out = self.n("ShaderNodeOutputMaterial")
        for sock, v in ((bsdf.inputs["Base Color"], color), (bsdf.inputs["Roughness"], rough)):
            if isinstance(v, (tuple, list)):
                sock.default_value = v if len(v) == 4 else (*v, 1)
            elif isinstance(v, (int, float)):
                sock.default_value = v
            else:
                self.link(v, sock)
        bsdf.inputs["Metallic"].default_value = 0.0
        if height is not None:
            bump = self.n("ShaderNodeBump", Strength=strength, Distance=distance)
            self.link(height, bump.inputs["Height"])
            self.link(bump.outputs[0], bsdf.inputs["Normal"])
        self.link(bsdf.outputs[0], out.inputs[0])
        return bsdf


def rgb(srgb):
    c = util.srgb_to_linear(srgb).tolist()
    return (*c, 1.0)


def fabric(name, color, knee_z=0.5, hem_z=0.15, seed=0.0, weave=1.0, stains=0.0):
    """Heavy cotton: vertical drape folds, bunching at the knees and the hem, a canvas weave, worn edges."""
    mat = util.new_material(name)
    g = G(mat)
    co = g.coords("Object")
    x, y, z = g.sep(co)
    # Folds hang vertically: stretch the noise along Z.
    drape = g.noise(g.comb(g.math('MULTIPLY', x, 1.0), g.math('MULTIPLY', y, 1.0), g.math('MULTIPLY', z, 0.18)), 26.0, 3.0, 0.55)
    # Bunching: horizontal ripples near the knee and the hem.
    ripple_n = g.noise(co, 9.0, 2.0)
    ring = g.math('SINE', g.math('ADD', g.math('MULTIPLY', z, 150.0), g.math('MULTIPLY', ripple_n, 9.0)))
    near_knee = g.math('SUBTRACT', 1.0, g.math('MINIMUM', 1.0, g.math('ABSOLUTE', g.math('MULTIPLY', g.math('SUBTRACT', z, knee_z), 12.0))))
    near_hem = g.math('SUBTRACT', 1.0, g.math('MINIMUM', 1.0, g.math('ABSOLUTE', g.math('MULTIPLY', g.math('SUBTRACT', z, hem_z + 0.03), 16.0))))
    bunch = g.math('MULTIPLY', ring, g.math('ADD', g.math('MULTIPLY', near_knee, 0.8), g.math('MULTIPLY', near_hem, 0.6)))
    weave_n = g.noise(co, 900.0, 1.0, 0.3)
    height = g.math('ADD', g.math('MULTIPLY', drape, 1.0), g.math('MULTIPLY', bunch, 0.22))
    height = g.math('ADD', height, g.math('MULTIPLY', weave_n, 0.02 * weave))
    shade = g.ramp(drape, [(0.25, (0.82, 0.82, 0.82)), (0.75, (1, 1, 1))])
    base = g.mix(1.0, color, color)
    col = g.n("ShaderNodeMix", _data_type='RGBA', _blend_type='MULTIPLY')
    col.inputs[0].default_value = 1.0
    g.link(base, col.inputs[6])
    g.link(shade, col.inputs[7])
    out = col.outputs[2]
    grime = g.noise(co, 4.0, 4.0, 0.6)
    low = g.math('SUBTRACT', 1.0, g.math('MINIMUM', 1.0, g.math('MULTIPLY', g.math('SUBTRACT', z, hem_z), 6.0)))
    dirt_fac = g.math('MULTIPLY', g.math('MULTIPLY', grime, low), 0.35, clamp=True)
    out = g.mix(dirt_fac, out, (0.30, 0.27, 0.22, 1))
    if stains > 0:
        spots = g.n("ShaderNodeTexVoronoi", Scale=55.0, Randomness=1.0)
        g.link(co, spots.inputs["Vector"])
        pick = g.n("ShaderNodeTexWhiteNoise", _noise_dimensions='3D')
        g.link(spots.outputs["Color"], pick.inputs["Vector"])
        is_spot = g.math('LESS_THAN', spots.outputs["Distance"], g.math('MULTIPLY', pick.outputs["Value"], 0.25))
        keep = g.math('GREATER_THAN', g.noise(co, 2.5, 2.0), 1.0 - 0.32 * stains)
        stain = g.math('MULTIPLY', is_spot, keep)
        out = g.mix(g.math('MULTIPLY', stain, 0.85), out, rgb((0.33, 0.06, 0.05)))
    g.principled(out, 0.86, height, strength=0.8, distance=0.005)
    return mat


def belt(name, color, edge=(0.55, 0.52, 0.48)):
    """A karate belt: rows of stitching along it, frayed lighter edges, a dull sheen."""
    mat = util.new_material(name)
    g = G(mat)
    p = g.attr("param")
    u, v, w = g.sep(p)
    rows = g.math('SINE', g.math('MULTIPLY', v, 3.14159 * 9))
    stitch = g.math('POWER', g.math('ABSOLUTE', rows), 6.0)
    edge_d = g.math('MINIMUM', v, g.math('SUBTRACT', 1.0, v))
    fray = g.noise(g.comb(g.math('MULTIPLY', u, 60.0), g.math('MULTIPLY', v, 4.0), w), 3.0, 3.0)
    wear = g.math('MULTIPLY', g.math('SUBTRACT', 1.0, g.math('MINIMUM', 1.0, g.math('MULTIPLY', edge_d, 14.0))), g.math('ADD', 0.3, fray), clamp=True)
    col = g.mix(g.math('MULTIPLY', wear, 0.8), color, (*edge[:3], 1) if len(edge) == 3 else edge)
    col = g.mix(g.math('MULTIPLY', stitch, 0.15), col, (0.0, 0.0, 0.0, 1))
    height = g.math('ADD', g.math('MULTIPLY', stitch, -0.4), g.math('MULTIPLY', g.noise(p, 400.0, 1.0), 0.15))
    g.principled(col, 0.72, height, strength=0.6, distance=0.002)
    return mat


def tape(name, color, band=0.024, angle=0.55, dirt=(0.25, 0.08, 0.06)):
    """Cloth tape wound in overlapping diagonal turns: ridges at each edge, a fine weave, dirt and old blood."""
    mat = util.new_material(name)
    g = G(mat)
    p = g.attr("param")
    u, v, w = g.sep(p)
    s = g.math('DIVIDE', g.math('ADD', u, g.math('MULTIPLY', v, angle)), band)
    f = g.math('FRACT', s)
    ridge = g.math('POWER', g.math('SUBTRACT', 1.0, f), 10.0)
    weave = g.noise(g.comb(g.math('MULTIPLY', u, 2.0), g.math('MULTIPLY', v, 2.0), w), 700.0, 1.0)
    shade = g.ramp(f, [(0.0, (0.72, 0.72, 0.72)), (0.12, (1, 1, 1)), (1.0, (0.93, 0.93, 0.93))])
    col = g.n("ShaderNodeMix", _data_type='RGBA', _blend_type='MULTIPLY')
    col.inputs[0].default_value = 1.0
    col.inputs[6].default_value = color
    g.link(shade, col.inputs[7])
    out = col.outputs[2]
    grime = g.noise(g.coords("Object"), 30.0, 4.0, 0.6)
    out = g.mix(g.math('MULTIPLY', g.math('SUBTRACT', grime, 0.45), 0.9, clamp=True), out, (*util.srgb_to_linear(dirt).tolist(), 1))
    height = g.math('ADD', g.math('MULTIPLY', ridge, 0.5), g.math('MULTIPLY', weave, 0.12))
    g.principled(out, 0.74, height, strength=0.6, distance=0.0015)
    return mat


def hair(name, color, tip=(0.06, 0.05, 0.045)):
    """Short hair combed along param.y: fine strands, a slight brown in the tips, a soft sheen."""
    mat = util.new_material(name)
    g = G(mat)
    p = g.attr("param")
    u, v, w = g.sep(p)
    strands = g.noise(g.comb(g.math('MULTIPLY', u, 900.0), g.math('MULTIPLY', v, 25.0), w), 1.0, 3.0, 0.6)
    clumps = g.noise(g.comb(g.math('MULTIPLY', u, 90.0), g.math('MULTIPLY', v, 6.0), w), 1.0, 2.0)
    mixv = g.math('ADD', g.math('MULTIPLY', strands, 0.7), g.math('MULTIPLY', clumps, 0.3))
    col = g.mix(g.ramp(mixv, [(0.35, (0, 0, 0)), (0.75, (1, 1, 1))]), color, (*util.srgb_to_linear(tip).tolist(), 1))
    rough = g.math('ADD', 0.62, g.math('MULTIPLY', strands, 0.2))
    g.principled(col, rough, mixv, strength=0.35, distance=0.0008)
    return mat


def leather(name, color):
    mat = util.new_material(name)
    g = G(mat)
    co = g.coords("Object")
    grain = g.n("ShaderNodeTexVoronoi", Scale=320.0, _feature='DISTANCE_TO_EDGE')
    g.link(co, grain.inputs["Vector"])
    wear = g.noise(co, 60.0, 3.0)
    p = g.attr("param")
    u, v, w = g.sep(p)
    edge = g.math('MINIMUM', v, g.math('SUBTRACT', 1.0, v))
    stitch_line = g.math('LESS_THAN', g.math('ABSOLUTE', g.math('SUBTRACT', edge, 0.16)), 0.035)
    stitch_dash = g.math('GREATER_THAN', g.math('SINE', g.math('MULTIPLY', u, 900.0)), 0.0)
    stitch = g.math('MULTIPLY', stitch_line, stitch_dash)
    col = g.mix(g.math('MULTIPLY', wear, 0.35), color, (0.02, 0.012, 0.008, 1))
    col = g.mix(g.math('MULTIPLY', stitch, 0.8), col, (0.25, 0.2, 0.15, 1))
    height = g.math('ADD', g.math('MULTIPLY', grain.outputs["Distance"], 0.6), g.math('MULTIPLY', stitch, 0.5))
    g.principled(col, g.math('ADD', 0.42, g.math('MULTIPLY', wear, 0.18)), height, strength=0.25, distance=0.0006)
    return mat


def boot(name, color, sole=0.02):
    """Boot leather: fine grain and a little wear, a groove where the upper meets the sole and a stitched
    welt just above it (param y is the height above the sole's underside)."""
    mat = util.new_material(name)
    g = G(mat)
    co = g.coords("Object")
    grain = g.n("ShaderNodeTexVoronoi", Scale=320.0, _feature='DISTANCE_TO_EDGE')
    g.link(co, grain.inputs["Vector"])
    wear = g.noise(co, 60.0, 3.0)
    u, v, w = g.sep(g.attr("param"))
    cx, cy, cz = g.sep(co)
    groove = g.math('LESS_THAN', g.math('ABSOLUTE', g.math('SUBTRACT', v, sole)), 0.0016)
    line = g.math('LESS_THAN', g.math('ABSOLUTE', g.math('SUBTRACT', v, sole + 0.006)), 0.0009)
    dash = g.math('GREATER_THAN', g.math('SINE', g.math('MULTIPLY', g.math('ADD', cx, cy), 1100.0)), 0.2)
    stitch = g.math('MULTIPLY', line, dash)
    col = g.mix(g.math('MULTIPLY', wear, 0.3), color, (0.02, 0.012, 0.008, 1))
    col = g.mix(g.math('MULTIPLY', groove, 0.7), col, (0.01, 0.01, 0.01, 1))
    height = g.math('ADD', g.math('MULTIPLY', grain.outputs["Distance"], 0.5), g.math('SUBTRACT', g.math('MULTIPLY', stitch, 0.4), groove))
    g.principled(col, g.math('ADD', 0.38, g.math('MULTIPLY', wear, 0.16)), height, strength=0.3, distance=0.0008)
    return mat


def cord(name, color):
    """Twisted cord: a helix of plies along param.x."""
    mat = util.new_material(name)
    g = G(mat)
    p = g.attr("param")
    u, v, w = g.sep(p)
    tw = g.math('SINE', g.math('ADD', g.math('MULTIPLY', u, 900.0), g.math('MULTIPLY', v, 6.283 * 3)))
    col = g.mix(g.math('MULTIPLY', g.math('ADD', tw, 1.0), 0.18), color, (0.0, 0.0, 0.0, 1))
    g.principled(col, 0.7, tw, strength=0.5, distance=0.001)
    return mat


def satin(name, color, rough=0.32):
    """Shiny fight-shorts satin (or lycra): soft folds, a fine weave, low roughness."""
    mat = util.new_material(name)
    g = G(mat)
    co = g.coords("Object")
    x, y, z = g.sep(co)
    folds = g.noise(g.comb(x, y, g.math('MULTIPLY', z, 0.35)), 30.0, 3.0, 0.5)
    weave = g.noise(co, 1200.0, 1.0, 0.3)
    height = g.math('ADD', folds, g.math('MULTIPLY', weave, 0.03))
    g.principled(color, g.math('ADD', rough, g.math('MULTIPLY', folds, 0.1)), height, strength=0.5, distance=0.003)
    return mat


def gourd(name, color):
    """A dried calabash: lacquered, streaked lengthwise, darker at the neck."""
    mat = util.new_material(name)
    g = G(mat)
    p = g.attr("param")
    u, v, w = g.sep(p)
    streak = g.noise(g.comb(g.math('MULTIPLY', v, 40.0), g.math('MULTIPLY', u, 1.5), w), 4.0, 4.0, 0.6)
    col = g.mix(g.math('MULTIPLY', streak, 0.5), color, (0.12, 0.05, 0.02, 1))
    g.principled(col, g.math('ADD', 0.28, g.math('MULTIPLY', streak, 0.2)), streak, strength=0.2, distance=0.001)
    return mat


def image_material(name, albedo, orm=None, normal=None, metallic_from_b=False, subsurface=0.0, double_sided=False):
    """The surface as exported: base colour, ORM (G roughness, B metalness when metallic_from_b) and a
    tangent-space normal map. glTF's exporter reads this; the previews render it."""
    mat = util.new_material(name)
    g = G(mat)
    uvn = g.n("ShaderNodeUVMap", _uv_map="UVMap")
    tex = g.n("ShaderNodeTexImage", _image=albedo, _interpolation='Linear')
    g.link(uvn.outputs[0], tex.inputs[0])
    bsdf = g.n("ShaderNodeBsdfPrincipled")
    out = g.n("ShaderNodeOutputMaterial")
    g.link(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.6
    if orm is not None:
        t2 = g.n("ShaderNodeTexImage", _image=orm, _interpolation='Linear')
        g.link(uvn.outputs[0], t2.inputs[0])
        sepc = g.n("ShaderNodeSeparateColor")
        g.link(t2.outputs["Color"], sepc.inputs[0])
        g.link(sepc.outputs[1], bsdf.inputs["Roughness"])
        if metallic_from_b:
            g.link(sepc.outputs[2], bsdf.inputs["Metallic"])
    if normal is not None:
        t3 = g.n("ShaderNodeTexImage", _image=normal, _interpolation='Linear')
        g.link(uvn.outputs[0], t3.inputs[0])
        nm = g.n("ShaderNodeNormalMap", _uv_map="UVMap", Strength=1.0)
        g.link(t3.outputs["Color"], nm.inputs["Color"])
        g.link(nm.outputs[0], bsdf.inputs["Normal"])
    if subsurface > 0:
        bsdf.inputs["Subsurface Weight"].default_value = subsurface
        bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
        bsdf.inputs["Subsurface Scale"].default_value = 0.006
    g.link(bsdf.outputs[0], out.inputs[0])
    mat.use_backface_culling = not double_sided
    return mat
