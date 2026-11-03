"""Fighting stances for the previews: two-bone IK for the limbs, aimed spine and head, closed fists.

All targets are in the armature's rest space (metres, the fighter facing -Y, their left at +X), scaled by
height so a stance fits every body. The rig keeps the rest pose for export: previews pose, render, reset.
"""
import math

import bpy
from mathutils import Matrix, Quaternion, Vector

import util


def _pb(arm, name):
    return arm.pose.bones[name]


def _update():
    bpy.context.view_layer.update()


def rotate(arm, name, axis, degrees, pivot=None):
    pb = _pb(arm, name)
    M = pb.matrix.copy()
    c = pivot if pivot is not None else M.translation.copy()
    R = Matrix.Rotation(math.radians(degrees), 4, Vector(axis))
    pb.matrix = Matrix.Translation(c) @ R @ Matrix.Translation(-c) @ M
    _update()


def aim(arm, name, direction):
    pb = _pb(arm, name)
    M = pb.matrix.copy()
    cur = (M.to_3x3() @ Vector((0, 1, 0))).normalized()
    q = cur.rotation_difference(Vector(direction).normalized())
    c = M.translation.copy()
    pb.matrix = Matrix.Translation(c) @ q.to_matrix().to_4x4() @ Matrix.Translation(-c) @ M
    _update()


def head(arm, name):
    return _pb(arm, name).matrix.translation.copy()


def tail(arm, name):
    pb = _pb(arm, name)
    return (pb.matrix @ Matrix.Translation((0, pb.bone.length, 0))).translation.copy()


def two_bone(arm, upper, lower, end, target, pole):
    """Place `end`'s head (the wrist or ankle) at target, the middle joint towards pole."""
    S = head(arm, upper)
    L1 = (head(arm, lower) - S).length
    L2 = (head(arm, end) - head(arm, lower)).length
    T = Vector(target)
    d = T - S
    dist = min(d.length, (L1 + L2) * 0.999)
    u = d.normalized()
    pv = Vector(pole) - u * Vector(pole).dot(u)
    if pv.length < 1e-6:
        pv = Vector((0, -1, 0)) - u * u.y
    v = pv.normalized()
    a = (L1 * L1 - L2 * L2 + dist * dist) / (2 * dist)
    h = math.sqrt(max(L1 * L1 - a * a, 0))
    E = S + u * a + v * h
    aim(arm, upper, E - S)
    aim(arm, lower, (S + u * dist) - head(arm, lower))


def fist(arm, side, curl=1.0):
    """Close the fingers: rotate each joint about its bend axis (the bone's X after roll normalisation)."""
    for f, angles in (("index", (80, 95, 60)), ("middle", (85, 95, 60)), ("ring", (88, 95, 60)), ("pinky", (90, 95, 60))):
        for i, ang in enumerate(angles, start=1):
            pb = _pb(arm, f"{f}_0{i}_{side}")
            pb.rotation_mode = 'XYZ'
            pb.rotation_euler = (math.radians(ang * curl), 0, 0)
    for i, ang in enumerate((25, 35, 30), start=1):
        pb = _pb(arm, f"thumb_0{i}_{side}")
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (math.radians(ang * curl), 0, 0)
    _update()


def reset(arm):
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    _update()


STANCES = {
    # Karate kamae: left foot forward, bladed, lead fist out at chest height, rear fist chambered at the chin.
    "karate": dict(drop=0.06, blade=-30, lean=7, counter=12, feet=((0.15, -0.27), (-0.13, 0.21)),
                   lead=(0.03, -0.33, -0.2), rear=(-0.07, -0.17, -0.2), head_tilt=4, knee_out=0.35),
    # A boxer's guard: both fists high, chin down, weight forward.
    "boxing": dict(drop=0.05, blade=-22, lean=10, counter=10, feet=((0.13, -0.18), (-0.14, 0.18)),
                   lead=(0.05, -0.27, 0.02), rear=(-0.08, -0.16, 0.04), head_tilt=12, knee_out=0.3),
    # Muay thai: tall, square, hands high and open-ish.
    "muaythai": dict(drop=0.03, blade=-12, lean=2, counter=6, feet=((0.13, -0.15), (-0.14, 0.14)),
                     lead=(0.08, -0.28, 0.06), rear=(-0.09, -0.2, 0.06), head_tilt=6, knee_out=0.2),
}


def stance(arm, m, name="karate"):
    s = STANCES.get(name, STANCES["karate"])
    reset(arm)
    k = m["height"] / 1.78
    # Lower the hips and blade the body.
    pb = _pb(arm, "pelvis")
    M = pb.matrix.copy()
    M.translation.z -= s["drop"] * k
    pb.matrix = M
    _update()
    rotate(arm, "pelvis", (0, 0, 1), s["blade"])
    rotate(arm, "spine_01", (1, 0, 0), s["lean"] * 0.5)
    rotate(arm, "spine_02", (0, 0, 1), s["counter"])
    rotate(arm, "spine_02", (1, 0, 0), s["lean"] * 0.5)
    # Feet: ankles at their rest height, placed around the body's centre.
    for side, (fx, fy) in zip(("l", "r"), s["feet"]):
        ankle_z = m[f"foot_{side}"][2]
        target = Vector((fx * k, fy * k, ankle_z))
        pole = Vector((s["knee_out"] * (1 if side == "l" else -1), -1, 0.2))
        two_bone(arm, f"thigh_{side}", f"calf_{side}", f"foot_{side}", target, pole)
        # Keep the foot flat and pointing roughly forward.
        aim(arm, f"foot_{side}", Vector((0.2 * (1 if side == "l" else -1), -1.0, -0.55)))
    # Fists: lead in front of the lead shoulder, rear by the chin.
    neck = head(arm, "head")
    for side, off in (("l", s["lead"]), ("r", s["rear"])):
        sh = head(arm, f"upperarm_{side}")
        base = sh if side == "l" else neck
        target = base + Vector(off) * k
        pole = Vector((0.35 if side == "l" else -0.35, 0.3, -1.0))
        two_bone(arm, f"upperarm_{side}", f"lowerarm_{side}", f"hand_{side}", target, pole)
        fore = (head(arm, f"hand_{side}") - head(arm, f"lowerarm_{side}")).normalized()
        aim(arm, f"hand_{side}", (fore + Vector((0, -0.9, 0))).normalized())
        fist(arm, side)
    # Head facing the opponent, chin down a touch: turn the neck part way, then aim the face.
    rotate(arm, "neck_01", (0, 0, 1), -(s["blade"] + s["counter"]) * 0.6)
    look(arm, "head", Vector((0, -1, -math.tan(math.radians(s["head_tilt"])))))


def look(arm, name, direction):
    """Turn a bone about its head so its rest-pose forward (-Y) points along direction, keeping it upright."""
    pb = _pb(arm, name)
    rest = pb.bone.matrix_local.to_3x3()
    pose_m = pb.matrix.to_3x3()
    fwd = (pose_m @ rest.inverted() @ Vector((0, -1, 0))).normalized()
    q = fwd.rotation_difference(Vector(direction).normalized())
    c = pb.matrix.translation.copy()
    pb.matrix = Matrix.Translation(c) @ q.to_matrix().to_4x4() @ Matrix.Translation(-c) @ pb.matrix
    _update()
    # Remove any roll: the head's rest up (+Z) should stay as close to world up as the aim allows.
    pose_m = pb.matrix.to_3x3()
    up = (pose_m @ rest.inverted() @ Vector((0, 0, 1))).normalized()
    fwd = (pose_m @ rest.inverted() @ Vector((0, -1, 0))).normalized()
    want = (Vector((0, 0, 1)) - fwd * fwd.z).normalized()
    ang = up.angle(want)
    if ang > 1e-3:
        axis = up.cross(want).normalized()
        sign = 1 if axis.dot(fwd) > 0 else -1
        rotate(arm, name, tuple(fwd), math.degrees(ang) * sign)
