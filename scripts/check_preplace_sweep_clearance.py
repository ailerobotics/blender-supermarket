#!/usr/bin/env python3
"""CPU proof: the carried PRE-PLACE leg-2 shoulder_pan sweep vs the Rollwagen cart cage
(sim-preplace-sweep-cage-clearance-v1).

No Isaac / ROS / GPU -- usd-core (the authored cart via ``twin.mobile_assets.author_cart``) + numpy + the
mobile URDF + the host XRDF sphere model. It reconstructs the coldwin70 collect place-on-cart leg-2 at the
real park pose and proves:

* OLD -- the executed wrist-first leg-2 EXTENDS the arm to the standoff shape, then sweeps shoulder_pan to
  the standoff. The extended gripper HOUSING (a robot link -- ``cube_without_surface_plates`` /
  ``surface_plate_1`` -- NOT the ``cage_collision``-filtered carton) rakes ``/Rollwagen/chassis/post_2``
  (the +x mouth post) and the upper side-wall bar ``side_0_horizontal_6`` at shoulder_pan ~ 0.20 rad, where
  the run froze (executed pan 0.2012) with a 1632.7 N*m effort spike. The cage gate REFUSES it by name
  (PREPLACE_SWEEP_CLEARANCE prim=post_2 min_clearance ~ -48 mm).
* NEW -- the PAN-FIRST reorder (``preplace_pan_first``): shoulder_pan sweeps to the standoff while the arm
  holds the compact carry shape, THEN extends. The tucked arc clears the mouth post/side bar by ~ +252 mm
  (>= the 40 mm design target) and the gate PASSES.

    PYTHONPATH=scripts python3 scripts/check_preplace_sweep_clearance.py [--margin 0.0]

Exit 0 when the OLD sweep is refused AND the NEW reorder clears; exit 1 otherwise. Skips (exit 0 with a
note) when usd-core or the dataset URDF/XRDF are absent.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ coldwin70 park-pose geometry (nav-trace)
# Robot base_link and cart (/Rollwagen chassis) world poses over the preplace window (t 185-235, static),
# from datasets/twin-robot-in-store-boot-v1/mission/coldwin70/nav-trace.jsonl. The cart row is
# [x, y, yaw_deg, z]; the base is the top-level x/y/yaw_deg. ur_base_link = base_link + (0, 0, 0.26) (URDF
# arm_mount). The cart COLLIDER boxes (rollwagen_collider_boxes) are authored in the cart ROOT frame (ground
# z=0), so the cart root world z is 0 (the chassis body origin logged at z~0.20 is the deck centre).
BASE_X, BASE_Y, BASE_YAW_DEG = -4.6140, 14.7690, 84.20
CART_X, CART_Y, CART_YAW_DEG = -4.6071, 15.9724, -93.61
# The sim's carton FROZE (physical jam) at this world z; used to calibrate base_link world z via FK so the
# reconstructed arm sits at the measured height (the x/y/yaw are the trusted nav base pose).
CARTON_JAM_WORLD_Z = 1.1441
JAM_Q = [0.2012, -1.4860, 1.4790, 0.0232, 0.6605, -0.0128]   # CARTON_TICK now=237.683 (settled jam config)

# coldwin70 leg-2 planned waypoints (host-stack.log "move hold joints to standoff" TRAJ_HOST_PLAN), names
# order [elbow, lift, pan, w1, w2, w3]: elbow/lift held, shoulder_pan sweeps 1.2678 -> -0.4053.
_LEG2_NAMES = ['ur_elbow_joint', 'ur_shoulder_lift_joint', 'ur_shoulder_pan_joint',
               'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']
_LEG2_PLAN = [
 [1.4788, -1.4856, 1.2678, -0.0131, 0.6609, 0.0118], [1.4788, -1.4856, 1.1084, -0.0087, 0.6612, 0.0099],
 [1.4788, -1.4856, 0.9491, -0.0037, 0.6614, 0.0072], [1.4788, -1.4856, 0.7898, 0.0014, 0.6615, 0.0037],
 [1.4788, -1.4856, 0.6304, 0.0067, 0.6616, -0.0003], [1.4788, -1.4856, 0.4711, 0.012, 0.6618, -0.0048],
 [1.4788, -1.4856, 0.3117, 0.0171, 0.6619, -0.0097], [1.4788, -1.4856, 0.2321, 0.0195, 0.662, -0.0122],
 [1.4788, -1.4856, 0.1524, 0.0219, 0.6621, -0.0148], [1.4788, -1.4856, -0.007, 0.0263, 0.6622, -0.0201],
 [1.4788, -1.4856, -0.1663, 0.0302, 0.6624, -0.0254], [1.4788, -1.4856, -0.246, 0.0319, 0.6625, -0.028],
 [1.4788, -1.4856, -0.4053, 0.0348, 0.6626, -0.0329]]
# The compact CARRY shape (leg-1 start, UR order pan,lift,elbow,w1,w2,w3) the pan-first reorder holds.
CARRY_SHAPE = [1.2678, -2.27, 1.0, 0.65, 0.0019, 0.6191]

_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _xlate(x, y, z):
    T = np.eye(4); T[:3, 3] = (x, y, z); return T


def _to_ur(p):
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    d = dict(zip(_LEG2_NAMES, p))
    return [d[j] for j in ARM_JOINTS]


def _reorder_pan_first(leg2_plan, carry):
    """The pan-first reorder path: sweep shoulder_pan from carry to the standoff pan while the arm holds the
    carry shape (leg A), then extend the wrists/lift/elbow to the standoff (leg B)."""
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    stand = _to_ur(leg2_plan[-1])
    n = 12
    legA = [[carry[0] + (stand[0] - carry[0]) * i / n, carry[1], carry[2], carry[3], carry[4], carry[5]]
            for i in range(n + 1)]
    start = [stand[0], carry[1], carry[2], carry[3], carry[4], carry[5]]
    legB = [[stand[0]] + [start[k] + (stand[k] - start[k]) * i / n for k in range(1, 6)]
            for i in range(n + 1)]
    return legA + legB, legA


def _t_base_cart(robot, geom):
    from twin.restock_manipulation import place_clearance as pc
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    Cu = pc._carton_in_base(robot, {j: JAM_Q[i] for i, j in enumerate(ARM_JOINTS)}, geom)
    bz = CARTON_JAM_WORLD_Z - 0.26 - float(Cu[2, 3])
    T_world_urbase = _xlate(BASE_X, BASE_Y, bz) @ _rotz(math.radians(BASE_YAW_DEG)) @ _xlate(0, 0, 0.26)
    T_world_cartroot = _xlate(CART_X, CART_Y, 0.0) @ _rotz(math.radians(CART_YAW_DEG))
    return np.linalg.inv(T_world_urbase) @ T_world_cartroot


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--margin', type=float, default=0.0, help='refuse threshold (m); default 0 = penetration')
    ap.add_argument('--json', type=Path, help='write the result JSON here')
    args = ap.parse_args(argv)

    try:
        from twin.restock_manipulation import place_clearance as pc
        from twin.urdf_probe import parse_urdf
    except Exception as exc:                                   # pragma: no cover
        print('SKIP: cannot import place_clearance/urdf_probe (%s)' % exc)
        return 0
    if not (_URDF.is_file() and _XRDF.is_file()):
        print('SKIP: dataset URDF/XRDF not on this box (%s)' % _URDF)
        return 0
    try:
        bodies = pc.rollwagen_collider_boxes()
    except Exception as exc:
        print('SKIP: usd-core / authored cart unavailable (%s)' % exc)
        return 0

    robot = parse_urdf(str(_URDF))
    geom = pc.GripperCartonGeometry.shipped()
    cage = pc.cage_body_names(bodies)
    T_base_cart = _t_base_cart(robot, geom)

    old = [_to_ur(p) for p in _LEG2_PLAN]
    new_full, new_sweep = _reorder_pan_first(_LEG2_PLAN, CARRY_SHAPE)

    def report(title, path):
        per_body = pc.preplace_cage_arm_clearance(path, str(_URDF), str(_XRDF), bodies, T_base_cart,
                                                  include=cage)
        reject = pc.preplace_sweep_reject(per_body, margin=args.margin)
        worst_v, worst_k = min((v, k) for k, v in per_body.items())
        print('\n== %s ==' % title)
        print(' min cage clearance: %s=%.1fmm' % (worst_k, worst_v * 1000.0))
        print(' verdict: %s' % ('REFUSED -- %s' % reject if reject else 'CLEAR (no cage body penetrated)'))
        return worst_v, worst_k, reject

    old_min, old_body, old_reject = report('OLD: wrist-first leg-2 pan sweep (coldwin70 executed)', old)
    # NEW: report the raking SWEEP clearance (the design-target >= 40 mm) and the FULL reorder gate verdict.
    sweep_per = pc.preplace_cage_arm_clearance(new_sweep, str(_URDF), str(_XRDF), bodies, T_base_cart,
                                               include=cage)
    sweep_min_v, sweep_min_k = min((v, k) for k, v in sweep_per.items())
    new_min, new_body, new_reject = report('NEW: pan-first reorder (preplace_pan_first)', new_full)
    print(' raking-sweep min (leg A, the tucked arc): %s=%.1fmm (target >= %.0f mm)'
          % (sweep_min_k, sweep_min_v * 1000.0, pc.PREPLACE_CAGE_CLEAR_TARGET_M * 1000.0))

    ok = (old_reject is not None) and (new_reject is None) \
        and (sweep_min_v >= pc.PREPLACE_CAGE_CLEAR_TARGET_M)
    print('\nSUMMARY: OLD refused=%s  NEW clear=%s  raking-sweep>=%.0fmm=%s  => %s'
          % (old_reject is not None, new_reject is None, pc.PREPLACE_CAGE_CLEAR_TARGET_M * 1000.0,
             sweep_min_v >= pc.PREPLACE_CAGE_CLEAR_TARGET_M, 'OK' if ok else 'FAIL'))
    if args.json:
        args.json.write_text(json.dumps({
            'margin_m': args.margin,
            'old_min_mm': round(old_min * 1000, 2), 'old_body': old_body,
            'old_refused': old_reject is not None,
            'new_min_mm': round(new_min * 1000, 2), 'new_body': new_body,
            'new_refused': new_reject is not None,
            'raking_sweep_min_mm': round(sweep_min_v * 1000, 2), 'raking_sweep_body': sweep_min_k,
            'ok': ok}, indent=1) + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
