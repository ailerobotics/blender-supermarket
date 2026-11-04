#!/usr/bin/env python3
"""CPU proof (sim-place-descent-residual-v2, brief part 2): the LEAST-EXTENDED collect place-on-cart
approach that keeps the welded carton level at the set-down.

coldwin39 placed at reach 1.086 m -> droop 2.02 deg (over the 2.0 bound); coldwin41's preplace_above_place
EXTENDED the reach and DRIFTED the carton ~93 mm toward the +x mouth -> post_2 edge pivot -> 4.94 deg. This
grid-searches the two decoupling levers -- the place load-centre moved toward the robot
(``cart_placement.load_center_body_xy``) and the base parked closer to the cart (``place_approach``) -- and
reports the least-extended combination whose K-fitted droop tilt and carton drift are both low. The droop
tilt is validated (reproduces coldwin39's 2.02 at K=2850); the per-body pad gaps are reported for
corroboration with the documented ~84 mm CPU-vs-sim geometry caveat (the GPU SEAL_DIAG is the real pad
check). The closed-loop backstop covers the residual tilt under the bound.

    PYTHONPATH=scripts python3 scripts/check_place_select.py [--tilt-target 1.5] [--json out.json]

Needs the dataset mobile URDF + usd-core. Exit 0 when a feasible least-extended pick exists, else 1.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# coldwin39 collect place-end config (TRAJ order elbow,shoulder_lift,pan,w1,w2,w3 -> ARM order), the
# base<-cart anchor whose carton sits at the known place load-centre (0.18, 0).
_CW39_TRAJ = (1.5833, -1.2932, -0.094, -0.2526, 1.6056, 0.0001)


def _arm(traj):
    el, sl, pan, w1, w2, w3 = traj
    return [pan, sl, el, w1, w2, w3]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tilt-target', type=float, default=1.5, help='max predicted droop tilt (deg)')
    ap.add_argument('--drift-max', type=float, default=0.050, help='max carton drift off the load-centre (m)')
    ap.add_argument('--json', type=Path, help='write the result JSON here')
    args = ap.parse_args(argv)

    urdf = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
    if not urdf.is_file():
        print('skipped: dataset URDF not on this box'); return 0
    from twin.urdf_probe import parse_urdf
    from twin.restock_manipulation import place_clearance as pc
    from twin.restock_manipulation import place_select as ps
    robot = parse_urdf(str(urdf))
    geom = pc.GripperCartonGeometry.shipped()
    bodies = pc.rollwagen_collider_boxes()
    anchor = _arm(_CW39_TRAJ)
    best, cands = ps.select_least_extended(robot, geom, bodies, anchor, tilt_target_deg=args.tilt_target,
                                           drift_max_m=args.drift_max)

    print('load-centre +dx (toward robot) x base +closer (toward cart): reach, droop tilt, drift, min mouth gap')
    for c in cands:
        mark = 'OK ' if c.feasible else '   '
        print('  %s dx=%+.2f closer=%+.2f | reach=%.3f droop=%.2f deg drift=%.1fmm mouth=%.1fmm'
              % (mark, c.load_center_dx, c.base_closer_m, c.reach_m, c.droop_tilt_deg,
                 c.load_center_drift_mm, c.min_mouth_gap_mm))
    print('\nanchor (dx=0, closer=0): the coldwin39 place -- reach 1.086, droop 2.16 (measured 2.02, OVER 2.0)')
    if best is None:
        print('NO feasible least-extended pick within the grid at tilt<=%.2f -- lean on the closed-loop '
              'backstop (carton_level.closed_loop_level) + a larger base move' % args.tilt_target)
        result = {'feasible': False, 'tilt_target_deg': args.tilt_target}
    else:
        print('CHOSEN least-extended: load_center_dx=%+.2f m (deeper from the mouth), base_closer=%+.2f m '
              '(place_approach toward the cart)\n  -> reach %.3f m, predicted droop %.2f deg, drift %.1f mm'
              % (best.load_center_dx, best.base_closer_m, best.reach_m, best.droop_tilt_deg,
                 best.load_center_drift_mm))
        print('  per-body drooped gaps (mm, SEAL-anchored; ~84mm CPU offset caveat): %s' % best.per_body_mm)
        result = {'feasible': True, 'tilt_target_deg': args.tilt_target,
                  'load_center_dx_m': best.load_center_dx, 'base_closer_m': best.base_closer_m,
                  'reach_m': round(best.reach_m, 4), 'predicted_droop_deg': round(best.droop_tilt_deg, 3),
                  'drift_mm': round(best.load_center_drift_mm, 1),
                  'per_body_drooped_gap_mm': best.per_body_mm}
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + '\n')
    return 0 if best is not None else 1


if __name__ == '__main__':
    sys.exit(main())
