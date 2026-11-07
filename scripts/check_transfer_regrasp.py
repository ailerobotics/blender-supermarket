#!/usr/bin/env python3
"""Offline CPU proof (no GPU) for the cage-aware re-grasp + reposition place-park
(sim-transfer-regrasp-reposition-v1). Reproduces the coldwin93 TRANSFER failure and proves the opt-in fix on
CPU, and emits the ON config block for the next GPU run.

coldwin93 ran couple + tow + uncouple + park on the GPU, then died at the TRANSFER PICK: cuMotion could not
inverse-kinematics-solve the near-face SIDE reach into the rollwagen +x cage mouth
(INVERSE_KINEMATICS_FAILURE x10 -> PLAN_TOTAL_ATTEMPTS_CAP -> HOST_FAULT_ESTOP phase=approach), and the
observed store-shelf place target lay > 2 m from the parked base. This script proves, on CPU, against the
AUTHORED rollwagen cage (usd-core) + the dataset arm URDF/XRDF:

  (1) RE-GRASP WORLD OBSTACLES -- the parked cart-body -> ur_base_link transform (from the parked prim pose,
      not the coupled hitch), the PR 350 world_obstacles built at it, and that they CLEAR the arm start
      config (a world obstacle overlapping the start rejects every plan -- INVALID_INITIAL_CSPACE);
  (2) PAD-DOWN RE-GRASP CLEARANCE -- the approach/descent/lift/withdraw pad-down path through the open mouth
      CLEARS every non-support cage body by >= the margin (the sweep tool), and the carried withdraw stays
      within the 2.0 deg level bound;
  (3) REPOSITION REACH -- the place-park pose computed from the endcap target puts it within ~0.9 m of the
      base facing the shelf, and the place + level descent onto the shelf top (0.73) sit inside the UR10
      envelope.

Run:  PYTHONPATH=scripts .venv-usd/bin/python scripts/check_transfer_regrasp.py \
        [--robot-park X Y YAWDEG] [--cart-park X Y YAWDEG] [--endcap X Y Z] [--emit-config OUT.json]
Exit 0 iff (1), (2) and (3) pass (prints TRANSFER_REGRASP PASS/FAIL)."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*

from twin.restock_manipulation import transfer_regrasp as tr

# coldwin93 geometry (nav-trace.jsonl terminal transfer poses + scenario).
DEF_ROBOT_PARK = (-4.294, 6.987, -89.8)        # ur_base_link world (x, y, yaw deg)
DEF_CART_PARK = (-4.2986, 8.2593, -89.81)      # parked cart prim world (x, y, yaw deg)
DEF_ENDCAP = (-3.05, 4.826, 0.73)              # scenario.initial_staging.target_surface_pose_world
CARTON_H = 0.18
SUPPORT_Z = 0.225                              # the deck (coldwin93 carton rest); shelf is 0.933
SUPPORT = 'deck'
_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'


def _record(report, name, passed, **kw):
    report['ok'] = report['ok'] and passed
    report['checks'].append(dict(name=name, passed=passed, **kw))
    print(('PASS ' if passed else 'FAIL ') + name + ' ' + json.dumps(kw), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--robot-park', type=float, nargs=3, default=DEF_ROBOT_PARK, metavar=('X', 'Y', 'YAWDEG'))
    ap.add_argument('--cart-park', type=float, nargs=3, default=DEF_CART_PARK, metavar=('X', 'Y', 'YAWDEG'))
    ap.add_argument('--endcap', type=float, nargs=3, default=DEF_ENDCAP, metavar=('X', 'Y', 'Z'))
    ap.add_argument('--support-z', type=float, default=SUPPORT_Z)
    ap.add_argument('--support', default=SUPPORT)
    ap.add_argument('--reach-m', type=float, default=tr.DEFAULT_PLACE_REACH_M)
    ap.add_argument('--margin', type=float, default=tr.DEFAULT_REGRASP_CAGE_MARGIN_M)
    ap.add_argument('--emit-config', type=Path, default=None, help='write the ON config block as JSON')
    ap.add_argument('--json', type=Path, default=None)
    args = ap.parse_args()

    try:
        from twin.restock_manipulation.place_clearance import rollwagen_collider_boxes
        bodies = rollwagen_collider_boxes()
    except Exception as exc:
        print('TRANSFER_REGRASP SKIP (authored rollwagen needs usd-core): %s' % exc, flush=True)
        return 0

    robot = (args.robot_park[0], args.robot_park[1], math.radians(args.robot_park[2]))
    cart = (args.cart_park[0], args.cart_park[1], math.radians(args.cart_park[2]))
    report = {'ok': True, 'checks': []}

    # (1) re-grasp world obstacles from the parked prim pose, cleared vs the arm start config
    cart_pose = tr.cart_pose_in_base(robot, cart)
    specs = tr.parked_cart_world_obstacles(cart_pose[:4], collider_boxes=bodies)
    cleared = None
    _JOINTS = ('ur_shoulder_pan_joint', 'ur_shoulder_lift_joint', 'ur_elbow_joint',
               'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint')
    stow = dict(zip(_JOINTS, (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)))
    if _URDF.is_file() and _XRDF.is_file():
        try:
            cleared = tr.check_regrasp_obstacles_clear_arm(specs, str(_URDF), str(_XRDF), configs=[{}, stow])
            ok = min(cleared.values()) > 0.0
        except ValueError as exc:
            ok = False
            cleared = {'error': str(exc)}
        _record(report, 'parked_cart_world_obstacles_clear_the_arm_start', ok,
                cart_pose=[round(v, 4) for v in cart_pose], obstacle_ids=[s['id'] for s in specs],
                min_clear_mm={k: round(v * 1000, 1) for k, v in cleared.items()} if ok else cleared)
    else:
        print('NOTE world-obstacle clear-arm proof skipped (dataset URDF/XRDF absent)', flush=True)

    # (2) pad-down re-grasp path clears the cage + carried withdraw level
    poses, carried = tr.mouth_pad_down_path((0.08, 0.0), args.support_z, 0.0, carton_h=CARTON_H)
    per_body = tr.regrasp_cage_clearance(poses, carried, bodies, support=args.support)
    reject = tr.regrasp_sweep_reject(per_body, margin=args.margin)
    worst_ns = min((v for k, v in per_body.items() if k not in ('deck', 'shelf')), default=float('inf'))
    _record(report, 'pad_down_regrasp_clears_the_cage', reject is None,
            min_nonsupport_mm=round(worst_ns * 1000, 1), margin_mm=round(args.margin * 1000, 1),
            summary=tr.regrasp_clearance_summary(per_body), reject=reject)
    tilt, level_ok = tr.carried_withdraw_level_ok(poses, carried, bound_deg=tr.DEFAULT_TILT_BOUND_DEG)
    _record(report, 'carried_withdraw_stays_level', level_ok,
            max_tilt_deg=round(tilt, 4), bound_deg=tr.DEFAULT_TILT_BOUND_DEG)

    # (3) reposition reach from the computed place-park
    pp = tr.place_park_pose(tuple(args.endcap), reach_m=args.reach_m, carton_h=CARTON_H)
    _record(report, 'reposition_place_target_within_reach',
            pp.within_goal and pp.faces_shelf and pp.within_envelope,
            shelf_approach=[round(v, 4) for v in pp.pose],
            reach_center_m=round(pp.reach_center_m, 4), reach_place_m=round(pp.reach_place_m, 4),
            reach_preplace_m=round(pp.reach_preplace_m, 4), max_reach_m=pp.max_reach_m,
            faces_shelf=pp.faces_shelf)

    if args.emit_config is not None:
        block = {
            'cart_pose_in_base': [round(v, 4) for v in cart_pose],
            'transfer_regrasp': True,
            'world_obstacles': specs,
            'preplace_cage_gate': {'cart_pose': [round(v, 4) for v in cart_pose]},
            'place_descent_cage_gate': {'cart_pose': [round(v, 4) for v in cart_pose]},
            'shelf_approach': [round(v, 6) for v in pp.pose],
        }
        args.emit_config.parent.mkdir(parents=True, exist_ok=True)
        args.emit_config.write_text(json.dumps(block, indent=2))
        print('wrote %s' % args.emit_config, flush=True)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print('wrote %s' % args.json, flush=True)

    print('TRANSFER_REGRASP ' + ('PASS' if report['ok'] else 'FAIL'), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
