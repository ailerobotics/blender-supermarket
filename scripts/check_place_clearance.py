#!/usr/bin/env python3
"""CPU proof: can the collect place-on-cart set-down lower the carton onto the support WITHOUT the
gripper pad striking a Rollwagen post / wall bar / drawbar? (sim-place-descent-residual-v1)

No Isaac, ROS or GPU -- usd-core (the authored cart via ``twin.mobile_assets.author_cart``) + numpy +
the mobile URDF. It settles the coldwin31..40 collect-set-down CARTON_TILT +2.0x deg as a GEOMETRIC
CONTACT PIVOT (not droop) and proves the chosen remedy clears it:

* OLD  -- the fixed ``collect_place_preplace_shape`` parks the held carton OVERHANGING the +x mouth
  (coldwin39 reconstruction: cart (0.243, -0.185), carton yaw -97 deg). The welded gripper pad -- a
  ROBOT link, NOT the carton, so the ``cage_collision: filtered`` carton-vs-cage filter does not cover
  it -- jams flush on ``/Rollwagen/chassis/post_2`` (SEAL_DIAG min_gap_mm 0.0), pivoting the carton off
  level. The clearance check REFUSES it.
* NEW  -- ``carton_level.preplace_above_place``: the standoff re-targets DIRECTLY ABOVE the place
  load-centre, then the carton descends VERTICALLY (``place_from_above``) onto the support. The pad
  never sweeps past the posts. The clearance check PASSES (minimum non-support gap ~ 89 mm >= 30 mm).

``--plan-gate`` (needs the dataset URDF) additionally re-runs the carton-level PLAN gate
(``plan_compensated_place_descent`` with the UNCHANGED droop model K=2850) on the new descent and reports
its predicted executed tilt, proving the gate still passes.

    PYTHONPATH=scripts python3 scripts/check_place_clearance.py [--margin 0.03] [--plan-gate]

Exit 0 when the NEW (shipped) geometry clears; exit 1 when it does not (a gate).
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

from twin.restock_manipulation import place_clearance as pc  # noqa: E402

# coldwin39 reconstruction (nav-trace poses + the ground-truth place slot). The place target is the
# run-config ground-truth slot (load_center (0.18, 0), support_z_world 0.9325 = the upper-shelf top,
# yaw_cart_rel -90 deg). The OLD preplace standoff is where the fixed collect_place_preplace_shape parks
# the carton (cart (0.243, -0.185), yaw -97 deg) -- the dwell where coldwin39 tripped.
PLACE_LOAD_CENTER = (0.18, 0.0)
SUPPORT_Z = 0.9325
PLACE_YAW_CART_REL_DEG = -90.0
OLD_STANDOFF_XY = (0.243, -0.185)
OLD_STANDOFF_YAW_DEG = -97.0
CARTON_H = pc.CARTON_DIMS_M[2]


def _report(title, carton_poses, geom, bodies, margin):
    per_wp, per_body = pc.sweep_clearance(carton_poses, geom, bodies)
    reject = pc.clearance_reject(per_body, margin=margin)
    print('\n== %s ==' % title)
    print(' ' + pc.clearance_summary(per_body))
    print(' verdict: %s' % ('REFUSED -- %s' % reject if reject else 'CLEAR (no non-support body within margin)'))
    return per_body, reject


def _plan_gate_extreme(load_center, support_z, yaw_cart_rel, standoff_m, margin):
    """Run the UNCHANGED carton-level droop-aware PLAN gate (``plan_compensated_place_descent``, K=2850)
    on the NEW directly-above-place vertical descent and report its predicted executed tilt.

    Anchored on the FK of the real ``collect_place_preplace_shape`` (gripper_pad_tcp in the arm frame),
    not a reconstructed cart-in-arm: the standoff TCP is that FK, the place target is the standoff lowered
    by ``standoff_m`` (arm +Z is world +Z, the base is gravity-level), so the descent is the far-extended
    loaded set-down the droop model cares about, with NO dependence on the fragile cart/base world
    reconstruction. Returns ``(pred_pitch, pred_roll, reject|None)`` or None when the dataset URDF is
    absent."""
    urdf = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
    if not urdf.is_file():
        return None
    from twin.urdf_probe import parse_urdf
    from twin.restock_manipulation import carton_level as cl
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    robot = parse_urdf(str(urdf))
    gc = cl.GravityCompensation().validate()           # K=2850 default, UNCHANGED
    tcp_link = 'gripper_pad_tcp'
    seed = [0.093, -1.91, 2.014, -0.103, 1.664, 0.0]    # collect_place_preplace_shape
    # the carton up-axis in the tool frame, captured from the reach shape (the weld attitude)
    up = cl.carton_up_in_tool(cl.tool_rotation(robot, seed, tcp_link))
    standoff_tcp = cl.tool_position(robot, seed, tcp_link)          # far-extended loaded standoff
    place_tcp = standoff_tcp - np.array([0.0, 0.0, standoff_m])     # directly below it (vertical descent)
    big = 2 * math.pi
    limits = {'ur_elbow_joint': (0.1, math.pi), 'ur_wrist_1_joint': (-big, 0.65),
              'ur_wrist_2_joint': (0.0, big)}
    try:
        standoff_cfg = cl.level_place_ik(robot, standoff_tcp, up, seed, tool_link=tcp_link,
                                         limits=limits, gravity_comp=gc)
        start = [float(standoff_cfg[j]) for j in ARM_JOINTS]
        plan = cl.plan_compensated_place_descent(
            robot, start, tuple(place_tcp), start, up, gc, bound_deg=2.0, tool_link=tcp_link,
            limits=limits, waypoints=8, approach_from_above=True, above_clearance_m=0.08)
        return plan.max_pitch, plan.max_roll, plan.reject(2.0)
    except ValueError as exc:
        return ('UNSOLVED', str(exc), None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--margin', type=float, default=pc.DEFAULT_MARGIN_M, help='clearance margin (m)')
    ap.add_argument('--standoff', type=float, default=0.12, help='NEW preplace standoff above support (m)')
    ap.add_argument('--plan-gate', action='store_true', help='also run the K=2850 plan gate (needs URDF)')
    ap.add_argument('--json', type=Path, help='write the result JSON here')
    args = ap.parse_args(argv)

    bodies = pc.rollwagen_collider_boxes()
    geom = pc.GripperCartonGeometry.shipped()
    yaw_place = math.radians(PLACE_YAW_CART_REL_DEG)

    old = [pc.carton_pose_cart(OLD_STANDOFF_XY, SUPPORT_Z, math.radians(OLD_STANDOFF_YAW_DEG),
                               carton_h=CARTON_H, z_above=0.0)]
    old_min, old_reject = _report('OLD: fixed over-mouth preplace standoff (coldwin39 cart (0.243,-0.185))',
                                  old, geom, bodies, args.margin)

    new = pc.place_descent_poses(PLACE_LOAD_CENTER, SUPPORT_Z, yaw_place, carton_h=CARTON_H,
                                 standoff_m=args.standoff, n=12)
    new_min, new_reject = _report('NEW: preplace_above_place standoff + vertical descent', new, geom,
                                  bodies, args.margin)

    result = {'margin_m': args.margin,
              'old_post_2_gap_mm': round(old_min.get('post_2', float('nan')) * 1000, 2),
              'old_refused': old_reject is not None,
              'new_min_nonsupport_mm': round(min(v for k, v in new_min.items()
                                                  if k not in pc.SUPPORT_BODIES) * 1000, 2),
              'new_refused': new_reject is not None,
              'new_per_body_min_mm': {k: round(v * 1000, 2) for k, v in sorted(new_min.items())}}

    if args.plan_gate:
        extreme = _plan_gate_extreme(PLACE_LOAD_CENTER, SUPPORT_Z, yaw_place, args.standoff, args.margin)
        print('\n== PLAN GATE (carton-level droop model K=2850, UNCHANGED) ==')
        if extreme is None:
            print(' skipped: dataset URDF not on this box')
        elif extreme[0] == 'UNSOLVED':
            print(' UNSOLVED: %s' % extreme[1]); result['plan_gate'] = 'unsolved'
        else:
            mp, mr, rej = extreme
            print(' predicted executed peak: pitch=%.3f roll=%.3f deg (bound 2.00) -> %s'
                  % (mp, mr, 'REJECT' if rej else 'PASS'))
            result['plan_gate'] = {'pred_pitch_deg': round(mp, 4), 'pred_roll_deg': round(mr, 4),
                                   'pass': rej is None}

    ok = old_reject is not None and new_reject is None
    print('\nSUMMARY: OLD refused=%s  NEW clear=%s  => %s'
          % (old_reject is not None, new_reject is None, 'OK' if ok else 'FAIL'))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
