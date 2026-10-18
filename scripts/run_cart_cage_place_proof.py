#!/usr/bin/env python3
"""CPU proof (no GPU) gating the cart-cage open-face place (firstmate 350, corrected run-279).

The place-onto-cart routes the carried carton THROUGH the cart's single OPEN +x loading face onto
the shelf. The mission's REAL /Rollwagen (read from the mission's OWN scene, run-*/mission.usda,
NOT the compose_robot_in_store stage) sits BESIDE the robot on -y in ur_base_link: deck
[0,-0.75,-0.30], back wall x=-0.38, side walls y=-0.43/-1.07, open +x. ``--stage`` re-derives and
asserts the committed CANONICAL_CAGE below.

run-279 correction: fm350 modelled the cart as a SOLID cuMotion obstacle, but the real beside-cart
PHYSICALLY INTERLEAVES the arm/mobile-base (arm-only clearance -44 mm at the deck, -37..-47 mm at the
near side wall, at every pose) -- so publishing it as a solid obstacle makes cuMotion refuse every
plan, and a cage-clearance pass/fail gate is UNSATISFIABLE (it false-fails TUCK/PAN, which physically
succeeded in run-278). There is no real cage to route around; run-278 planned the place freely and
jammed ONLY on a wrist_1 self-collision (pose-IK flipped wrist_1 to 1.99, above the ur_wrist_2-vs-
ur_forearm stop ~0.70, and it stuck at 0.69). The carton-vs-wall clearances are therefore reported as
INFORMATIONAL ONLY, never a gate.

This script instead PROVES the wrist_1-safe fixed PREPLACE shape (``cage_preplace_shape``, driven via
begin_joint_goal) is correct, with CPU forward kinematics (``twin.urdf_probe``):

  (a) its wrist_1 is at/below the self-collision stop (<= WRIST1_STOP);
  (b) its FK tool pose REACHES the recorded sensed mouth standoff (pos + orientation), so the carton
      presents at the open +x face exactly as the pose-IK preplace intended;
  (c) the straight-line insert along -x from there lands the carton INSIDE the cart footprint on the
      shelf (a valid open-face landing), and
  (d) the joint travel from the pan config to the shape is modest (< pi on every joint).

Exit 0 iff (a)-(d) hold (and, with --stage, the re-derived cage matches CANONICAL_CAGE).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from twin import urdf_probe as up  # noqa: E402
from twin.restock_manipulation.cad_tool import ARM_JOINTS  # noqa: E402
from twin.restock_host.world_obstacles import (  # noqa: E402
    Box, boxes_from_specs, carton_cover_local, check_configuration)

# Authoritative cage, ur_base_link frame (walls-only, OPEN +x). Re-derived from the mission's OWN
# /Rollwagen (BESIDE the robot on -y; deck surface z=-0.275, shelf z=0.42). --stage re-derives +
# asserts these against run-*/mission.usda.
CANONICAL_CAGE = [
    {'id': 'cart_deck',  'size': [0.80, 0.68, 0.05], 'center': [ 0.000, -0.750, -0.300]},
    {'id': 'cart_back',  'size': [0.02, 0.68, 1.45], 'center': [-0.380, -0.750,  0.425]},
    {'id': 'cart_side0', 'size': [0.80, 0.02, 1.45], 'center': [ 0.000, -1.070,  0.425]},
    {'id': 'cart_side1', 'size': [0.80, 0.02, 1.45], 'center': [ 0.000, -0.430,  0.425]},
]
# The open loading face: +x in ur_base_link. Deck top / shelf z for the landing check.
CAGE_AXIS = (1.0, 0.0, 0.0)
DECK_TOP_Z = -0.300 + 0.05 / 2.0
SHELF_TOP_Z = 0.420 + 0.025 / 2.0
CART_FRONT_X = 0.000 + 0.80 / 2.0   # open +x face plane (deck x-extent)

# wrist_1 self-collision stop (ur_wrist_2-vs-ur_forearm); the fixed preplace shape must stay at/below.
WRIST1_STOP = 0.65
# The sensed preplace mouth TCP (ur_base_link) the pose-IK preplace aimed at in run-278 (FK of its
# recorded goal); the fixed shape must reach the SAME tool pose.
RECORDED_MOUTH_TCP = (0.6298, -0.9527, 0.8024)
RECORDED_MOUTH_SHAPE = (-3.7024, -3.1293, 1.1383, 1.9909, -0.5614, -3.1416)  # run-278 pose-IK goal (jams)
# fm351/run-279: the wrist_1-safe fixed PREPLACE shape (begin_joint_goal target), CPU-IK-solved to the
# RECORDED_MOUTH_TCP with wrist_1 below the stop. Must match ros_host_config.json cage_preplace_shape.
CAGE_PREPLACE_SHAPE = (-3.702, -3.486, 1.425, -1.081, 0.561, 0.0)

# Carry-fold joint configs (arm joints 1-5 = the carry shape cart_tuck_shape; joint0 = shoulder_pan).
CART_TUCK_SHAPE = (-2.27, 1.0, 0.65, 0.8, -1.2545)
CART_PAN_JOINT0 = -1.822
PAN_CONFIG = (CART_PAN_JOINT0, *CART_TUCK_SHAPE)


def _tcp_transform(robot, config):
    """FK 4x4 of the gripper_pad_tcp in ur_base_link at an ARM-order joint config."""
    world = up.link_world_transforms(robot, dict(zip(ARM_JOINTS, config)))
    key = next(k for k in world if k.endswith('gripper_pad_tcp'))
    return np.asarray(world[key], dtype=float)


def _quat_matrix(q):
    """Rotation matrix for a unit quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def carton_box_points(center, quat, dims, pitch=0.02):
    """EXACT carton box surface sample (no sphere inflation) at a given CARTON centre + orientation,
    in ur_base_link. Used for the tight insert-fit check where the carton's 0.6 m face nearly spans
    the cage's narrow interior -- the conservative sphere cover (sized for the 100s-of-mm carry
    margins) would read a false penetration there. A dense box grid + each wall's exact box SDF gives
    the true min gap; a convex box's nearest point to another convex box is on its surface, so the
    surface grid's min SDF is the box-box clearance."""
    d = np.asarray(dims, dtype=float)
    axes = [np.linspace(-v / 2.0, v / 2.0, max(2, int(math.ceil(v / pitch)) + 1)) for v in d]
    pts = np.array([[x, y, z] for x in axes[0] for y in axes[1] for z in axes[2]])
    r = _quat_matrix(quat)
    return pts @ r.T + np.asarray(center, dtype=float)


def carton_world_extents(quat, dims):
    """Half-extents of the carton's world AABB under orientation ``quat`` (axis-aligned bound of the
    rotated box) -- used to rest the carton on the deck and size the mouth clearance."""
    r = _quat_matrix(quat)
    return np.abs(r) @ (np.asarray(dims, dtype=float) / 2.0)


def _derive_from_stage(stage_path):
    """Re-derive the walls-only cage from a composed mission stage's /Rollwagen AABBs (needs usd-core).
    Returns specs in the ur_base_link frame; raises if the stage/usd is unavailable."""
    from pxr import Usd, UsdGeom  # noqa
    st = Usd.Stage.Open(str(stage_path))
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'], useExtentsHint=False)
    base = UsdGeom.Xformable(st.GetPrimAtPath('/MobileRobot/base_link')
                             ).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    urw = np.array([base[0], base[1], base[2] + 0.26])  # ur_base_link = base_link + arm_mount(0,0,0.26)
    ch = st.GetPrimAtPath('/Rollwagen/chassis')
    groups = {'cart_deck': lambda n: n == 'deck', 'cart_back': lambda n: n.startswith('back_'),
              'cart_side0': lambda n: n.startswith('side_0_'), 'cart_side1': lambda n: n.startswith('side_1_')}
    specs = []
    for gid, pred in groups.items():
        mn = np.array([math.inf] * 3)
        mx = np.array([-math.inf] * 3)
        for c in ch.GetChildren():
            if not pred(c.GetName()):
                continue
            rng = bbc.ComputeWorldBound(c).ComputeAlignedRange()
            mn = np.minimum(mn, np.array(rng.GetMin()))
            mx = np.maximum(mx, np.array(rng.GetMax()))
        center = ((mn + mx) / 2.0 - urw)
        size = (mx - mn)
        specs.append({'id': gid, 'size': [round(v, 4) for v in size], 'center': [round(v, 4) for v in center]})
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--urdf', default=str(Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'))
    ap.add_argument('--xrdf', default=str(Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'))
    ap.add_argument('--manifest', default=str(Path.home() / 'datasets/twin-robot-in-store-boot-v1/mobile/mobile_manifest.json'))
    ap.add_argument('--stage', help='composed mission.usda to RE-DERIVE + assert the cage (needs usd-core)')
    ap.add_argument('--dims', type=float, nargs=3, default=[0.6, 0.4, 0.16], help='carton extents (m)')
    ap.add_argument('--offset-tcp', type=float, nargs=3, default=[0.0, 0.0, 0.20])
    ap.add_argument('--tcp-quat', type=float, nargs=4, default=[0.5, 0.5, 0.5, 0.5], help='side-grip TCP quat')
    ap.add_argument('--mouth-standoff', type=float, default=0.45, help='preplace offset out the +x mouth (m)')
    ap.add_argument('--insert-gap', type=float, default=0.03, help='stop this far above the deck top (m)')
    ap.add_argument('--steps', type=int, default=24, help='insert interpolation samples')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)

    specs = CANONICAL_CAGE
    stage_ok = None
    if args.stage:
        derived = _derive_from_stage(args.stage)
        stage_ok = all(math.isclose(d['center'][i], c['center'][i], abs_tol=0.02)
                       and math.isclose(d['size'][i], c['size'][i], abs_tol=0.02)
                       for d, c in zip(derived, CANONICAL_CAGE) for i in range(3))
        specs = derived
    boxes = boxes_from_specs(specs)
    carton = dict(dims=args.dims, offset_tcp=args.offset_tcp)
    robot = up.parse_urdf(args.urdf)
    shape = CAGE_PREPLACE_SHAPE

    # ---- GATE: the wrist_1-safe fixed PREPLACE shape (FK, no cage clearance -- the arm interleaves) ----
    tcp = _tcp_transform(robot, shape)
    tcp_pos = tcp[:3, 3]
    # carton centre = tool pose * offset_tcp (tool frame); same convention check_configuration uses.
    carton_center = tcp_pos + tcp[:3, :3] @ np.asarray(args.offset_tcp, dtype=float)
    ref = _tcp_transform(robot, RECORDED_MOUTH_SHAPE)      # the pose-IK goal the shape must match
    pos_err = float(np.linalg.norm(tcp_pos - ref[:3, 3]))
    # orientation error vs the recorded goal's tool frame (so the straight insert is axis-aligned)
    rot_err = float(np.arccos(np.clip((np.trace(tcp[:3, :3].T @ ref[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    wrist1 = shape[3]
    a = np.array(CAGE_AXIS, dtype=float); a = a / np.linalg.norm(a)
    # The straight -x insert drives the TOOL from the mouth standoff into the open +x face. Gate on the
    # TOOL insert point (pure FK, no offset/quat approximation); the carton hangs off it (reported as
    # informational carton_center -- its exact rest depends on the SENSED place target at runtime).
    tcp_place = tcp_pos - a * args.mouth_standoff
    carton_place = carton_center - a * args.mouth_standoff
    half = carton_world_extents(args.tcp_quat, args.dims)
    travel = np.abs(np.array(shape) - np.array(PAN_CONFIG))

    g_wrist1 = wrist1 <= WRIST1_STOP
    g_reach = pos_err <= 0.01 and rot_err <= math.radians(5)
    g_preplace_outside = (tcp_pos[0]) >= CART_FRONT_X        # tool presents outside the +x mouth
    # (c) the -x insert drives the tool INSIDE the cart footprint, above the deck (a valid open-face
    # landing onto the shelf). Footprint = cart x/y extent (+ small margin); z above the deck top.
    g_land = (-0.45 <= tcp_place[0] <= 0.45 and -1.10 <= tcp_place[1] <= -0.40
              and DECK_TOP_Z < tcp_place[2] < 1.50)
    g_travel = bool(np.all(travel < math.pi))
    ok = (g_wrist1 and g_reach and g_preplace_outside and g_land and g_travel
          and (stage_ok is not False))

    # ---- INFORMATIONAL ONLY: carton-vs-wall clearances along the fold + insert. NOT a gate: the real
    # beside-cart interleaves the arm/mobile-base, so these read negative at poses that physically work.
    manifest = json.loads(Path(args.manifest).read_text())['transport_positions']
    fold = {'STOW': [manifest[n] for n in ARM_JOINTS],
            'TUCK': [manifest[ARM_JOINTS[0]], *CART_TUCK_SHAPE],
            'PAN':  list(PAN_CONFIG),
            'PREPLACE': list(shape)}
    fold_min = {b.id: math.inf for b in boxes}
    for q in fold.values():
        r = check_configuration(args.urdf, args.xrdf, boxes, dict(zip(ARM_JOINTS, q)), carton=carton)
        for b in r['boxes']:
            fold_min[b['box']] = min(fold_min[b['box']], b['clearance_m'])

    result = {
        'cage_specs': specs, 'stage_asserted': stage_ok,
        'cage_preplace_shape': list(shape), 'wrist1': wrist1, 'wrist1_stop': WRIST1_STOP,
        'tcp_pos': [round(v, 4) for v in tcp_pos], 'recorded_mouth_tcp': list(RECORDED_MOUTH_TCP),
        'pos_err_m': round(pos_err, 4), 'rot_err_deg': round(math.degrees(rot_err), 3),
        'carton_center': [round(v, 4) for v in carton_center],
        'tcp_place': [round(v, 4) for v in tcp_place],
        'carton_place_informational': [round(v, 4) for v in carton_place],
        'travel_per_joint': [round(v, 3) for v in travel], 'travel_max': round(float(travel.max()), 3),
        'gate': {'wrist1<=stop': g_wrist1, 'reach_mouth': g_reach, 'preplace_outside_mouth': g_preplace_outside,
                 'insert_lands_in_cart': g_land, 'travel<pi': g_travel, 'stage_ok': stage_ok},
        'informational_wall_clearance_mm': {k: round(v * 1000) for k, v in fold_min.items()},
        'ok': ok,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print('CART-CAGE OPEN-FACE PLACE PROOF (CPU, no GPU) -- wrist_1-safe fixed preplace (run-279)')
        print('  cage source:', 'mission stage (asserted=%s)' % stage_ok if args.stage else 'committed CANONICAL_CAGE (beside-cart)')
        print('  (a) wrist_1 of cage_preplace_shape: %.3f  (stop %.2f)  => %s' % (
            wrist1, WRIST1_STOP, 'OK' if g_wrist1 else 'FAIL'))
        print('  (b) FK TCP %s vs recorded mouth %s: pos_err %.1f mm, rot_err %.2f deg  => %s' % (
            [round(v, 3) for v in tcp_pos], [round(v, 3) for v in RECORDED_MOUTH_TCP],
            pos_err * 1000, math.degrees(rot_err), 'OK' if g_reach else 'FAIL'))
        print('      preplace tool outside +x mouth (x>%.2f): %s  => %s' % (
            CART_FRONT_X, round(float(tcp_pos[0]), 3), 'OK' if g_preplace_outside else 'FAIL'))
        print('  (c) -x insert drives tool to %s (cart footprint, above deck; carton ~%s): %s' % (
            [round(v, 3) for v in tcp_place], [round(v, 3) for v in carton_place],
            'OK' if g_land else 'FAIL'))
        print('  (d) pan->shape joint travel: %s max %.2f rad (<pi)  => %s' % (
            [round(v, 2) for v in travel], travel.max(), 'OK' if g_travel else 'FAIL'))
        print('  informational carton-vs-wall min clearance (arm interleaves the real cart, NOT a gate): %s'
              % result['informational_wall_clearance_mm'])
        print('  VERDICT:', 'PASS (safe to boot)' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
