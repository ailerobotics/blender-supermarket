#!/usr/bin/env python3
"""CPU proof: the THREE-STAGE collect place set-down -- arrive level on the carry-connected branch above
the open approach corridor, reorient IN PLACE to the cage-clear pad-down branch (carton kept level), then
descend to the proven seat -- clears the Rollwagen cage at EVERY cw64-cw92 park (sim-place-pad-down-arrival-v1).

No Isaac / ROS / GPU -- usd-core (the authored cart) + numpy + the mobile URDF + the host XRDF sphere model.
It reuses the reconstruction of ``scripts/check_place_descent_reach.py`` (the park/cart pose + base_link z
from the cw87 run artifacts) and drives the PRODUCTION functions
(``place_reorient.select_pad_down_branch`` picks the pad-down branch; ``carton_level.plan_level_path_to_goal``
plans the reorientation; ``carton_level.plan_compensated_place_descent`` plans the descent -- the SAME code
the controller runs) through the SAME cuMotion arm-sphere cage sweep the PR 376 descent gate uses.

For each park it checks, at the shifted reorientation standoff (``shift`` toward the arm base, ``height``
above the seat):

  * stage (b) REORIENTATION: from the carry-connected ARRIVAL branch (low wrist_2) to the selected PAD-DOWN
    branch (high wrist_2), the leveled per-waypoint path stays within the 2.0 deg bound AND every gripper
    link stays >= ``margin`` from every cage body;
  * stage (c) DESCENT: the place-from-above descent to the corrected seat (carton bottom 0.9325 world)
    stays within the bound AND clears the cage, landing the carton on the seat.

It also REPLAYS the single-standoff failure: the coldwin87 executed descent (the carry-wrist-in-cage
reorientation) is still REFUSED by the descent cage gate, and the cw92-class park (feasible under the
three-stage set-down) passes. Exit 0 when the three-stage chain clears + reaches the seat + stays level at
EVERY park AND the old descent is refused; exit 1 otherwise. Skips (exit 0 + a note) without usd-core / the
dataset URDF/XRDF.

    PYTHONPATH=scripts <usd-core-python> scripts/check_place_pad_down_arrival.py [--shift 0.15] [--height 0.10]
        [--margin 0.03] [--json out.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _load_cpdr():
    spec = importlib.util.spec_from_file_location('cpdr', str(ROOT / 'check_place_descent_reach.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# cw90/91/92 park poses (the deterministic station park; only the horizontal park varies run-to-run).
# cw64/87/88/89 come from check_place_descent_reach; these extend it to the coldwin92 frontier.
_EXTRA_PARKS = {
    'coldwin90': (-4.6110, 14.7680, 85.00, -4.6030, 15.9760, -93.10),
    'coldwin91': (-4.6100, 14.7690, 85.20, -4.6040, 15.9750, -93.00),
    'coldwin92': (-4.6105, 14.7685, 85.10, -4.6035, 15.9755, -93.05),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shift', type=float, default=0.20, help='mouth-out standoff shift toward the arm base (m)')
    ap.add_argument('--height', type=float, default=0.10, help='reorientation standoff height above the seat (m)')
    ap.add_argument('--margin', type=float, default=0.03, help='cage refuse margin (m); default the 30 mm design')
    ap.add_argument('--json', type=Path, help='write the result JSON here')
    args = ap.parse_args(argv)

    try:
        from twin.restock_manipulation import carton_level as cl
        from twin.restock_manipulation import place_clearance as pc
        from twin.restock_manipulation import place_reorient as pr
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        from twin.urdf_probe import parse_urdf
    except Exception as exc:                                     # pragma: no cover
        print('SKIP: cannot import the twin modules (%s)' % exc)
        return 0
    cpdr = _load_cpdr()
    URDF, XRDF = cpdr._URDF, cpdr._XRDF
    if not (URDF.is_file() and XRDF.is_file()):
        print('SKIP: dataset URDF/XRDF not on this box (%s)' % URDF)
        return 0
    try:
        bodies = pc.rollwagen_collider_boxes()
    except Exception as exc:
        print('SKIP: usd-core / authored cart unavailable (%s)' % exc)
        return 0

    robot = parse_urdf(str(URDF))
    geom = pc.GripperCartonGeometry.shipped()
    cage = pc.cage_body_names(bodies)
    bz = cpdr._base_link_z(robot, pc, geom)
    # Use the SAME IK tool frame the controller does (gripper_pad_tcp when present, else ur_tool0) so the
    # proof reflects the real configs. The frame's ROTATION equals ur_tool0's (rigid translation), so the
    # leveling is identical; the ~0.27 m horizontal offset shifts the standoff/descent IK targets, which is
    # why the proof must run in the controller's frame (not ur_tool0).
    tcp_link = 'gripper_pad_tcp' if 'gripper_pad_tcp' in robot.links else 'ur_tool0'
    gc = cl.GravityCompensation(tool_link=tcp_link).validate()
    big = 2 * math.pi
    limits = {'ur_elbow_joint': (0.1, math.pi), 'ur_wrist_1_joint': (-big, 0.65),
              'ur_wrist_2_joint': (0.0, big)}

    pad = cpdr._to_ur(cpdr._CW87_DESCENT_PLAN[1])          # a high-wrist_2 pad-down reference (for the up-axis)
    arr_seed = cpdr._to_ur(cpdr._CW87_DESCENT_PLAN[0])     # the carry-wrist (low wrist_2) arrival seed
    old_descent = [cpdr._to_ur(p) for p in cpdr._CW87_DESCENT_PLAN]  # the refused single-standoff descent

    def tcp_of(v):
        return cl.tool_position(robot, {j: float(v[i]) for i, j in enumerate(ARM_JOINTS)},
                                tcp_link, 'ur_base_link')

    seat_tcp = tcp_of(pad)                                 # == directly above the seat load-centre
    std_tcp = pr.reorient_standoff_tcp(seat_tcp, args.shift, args.height)
    pad_map = {j: float(pad[i]) for i, j in enumerate(ARM_JOINTS)}
    up = cl.carton_up_in_tool(cl.tool_rotation(robot, pad_map, tcp_link, 'ur_base_link'))
    arr = [float(cl.level_place_ik(robot, std_tcp, up, {j: arr_seed[i] for i, j in enumerate(ARM_JOINTS)},
                                   tool_link=tcp_link, limits=limits, fast=True)[j]) for j in ARM_JOINTS]

    def bottom(vec):
        return cpdr._carton_bottom_world_z(robot, pc, geom, vec, bz)

    def min_cage(path, T):
        per = pc.place_descent_cage_clearance([list(w) for w in path], str(URDF), str(XRDF), bodies, T,
                                              include=cage)
        v, k = min((x, n) for n, x in per.items())
        return v, k

    parks = dict(cpdr._PARKS)
    parks.update(_EXTRA_PARKS)
    per_park = {}
    all_ok = True
    old_all_refused = True
    print('reorientation standoff: +%.3f m toward the arm base, +%.3f m above the seat; cage margin %.0f mm'
          % (args.shift, args.height, args.margin * 1000.0))
    for name, park in parks.items():
        T = cpdr._t_base_cart(park, bz)
        # per-park cage gate + clearance closures (the production providers)
        gate = pc.place_descent_cage_clearance_provider(str(URDF), XRDF, cpdr._t_base_cart(park, bz),
                                                        bodies=bodies, margin=args.margin, include=cage)
        clear_fn = pc.place_descent_cage_min_clearance_provider(str(URDF), XRDF, cpdr._t_base_cart(park, bz),
                                                               bodies=bodies, include=cage)
        try:
            pad_down, rep = pr.select_pad_down_branch(
                robot, std_tcp, up, list(cl._as_vec({j: pad_map[j] for j in ARM_JOINTS})), seat_tcp=seat_tcp,
                limits=limits, gravity_comp=gc, bound_deg=2.0, tool_link=tcp_link, descent_waypoints=12,
                approach_from_above=True, above_clearance_m=0.08, descent_cage_gate=gate,
                descent_cage_clearance=clear_fn)
        except ValueError as exc:
            print('  %-10s NO cage-clear pad-down branch: %s' % (name, exc))
            all_ok = False
            per_park[name] = {'feasible': False, 'reason': str(exc)}
            continue
        pad_down_vec = [float(pad_down[j]) for j in ARM_JOINTS]
        # stage (b) reorientation arrival -> pad-down (the SAME branch-continuous, densified seeded planner
        # the controller runs, so the continuity gate is exercised: a 1.5 rad branch jump would show here)
        rplan = cl.plan_level_world_approach(robot, arr, pad_down_vec, up,
                                             level_joints=('ur_wrist_1_joint', 'ur_wrist_2_joint',
                                                           'ur_wrist_3_joint'),
                                             tool_link=tcp_link, limits=limits, bound_deg=2.0,
                                             max_segment_rad=0.08, max_wrist_step_rad=0.15, anchor='seeded',
                                             level_accept_deg=2.0, fast=True)
        rtilt = max(rplan.max_roll, rplan.max_pitch)
        rstep = float(rplan.max_wrist_step)
        rc, rk = min_cage(rplan.waypoints, T)
        # stage (c) descent pad-down -> seat
        lo, hi = 0.02, 0.40
        for _ in range(22):
            mid = (lo + hi) / 2.0
            tgt = [seat_tcp[0], seat_tcp[1], seat_tcp[2] - mid]
            dplan = cl.plan_compensated_place_descent(robot, list(pad_down_vec), tgt, list(pad_down_vec), up,
                                                      gc, bound_deg=2.0, tool_link=tcp_link, limits=limits,
                                                      waypoints=14, approach_from_above=True,
                                                      above_clearance_m=0.08)
            if bottom(list(dplan.waypoints[-1])) > cpdr.SEAT_CARTON_BOTTOM_WORLD_Z:
                lo = mid
            else:
                hi = mid
        dtilt = max(dplan.max_roll, dplan.max_pitch)
        dc, dk = min_cage(dplan.waypoints, T)
        bot = bottom(list(dplan.waypoints[-1]))
        old_reject = pc.place_descent_sweep_reject(
            pc.place_descent_cage_clearance(old_descent, str(URDF), str(XRDF), bodies, T, include=cage),
            margin=args.margin)
        old_all_refused = old_all_refused and (old_reject is not None)
        reo_ok = rc >= args.margin and rtilt <= 2.0 + 1e-9 and rstep <= 0.15 + 1e-9
        des_ok = dc >= args.margin and dtilt <= 2.0 + 1e-9
        seat_ok = abs(bot - cpdr.SEAT_CARTON_BOTTOM_WORLD_Z) <= 0.02
        ok = reo_ok and des_ok and seat_ok
        all_ok = all_ok and ok
        per_park[name] = {'feasible': True, 'pad_down_wrist_2': round(pad_down_vec[4], 3),
                          'reorient_tilt_deg': round(rtilt, 3), 'reorient_cage_mm': round(rc * 1000, 1),
                          'reorient_body': rk, 'reorient_wrist_step': round(rstep, 3),
                          'descent_tilt_deg': round(dtilt, 3),
                          'descent_cage_mm': round(dc * 1000, 1), 'descent_body': dk,
                          'bottom_z': round(bot, 4), 'old_refused': old_reject is not None, 'ok': ok}
        print('  %-10s pad_down_w2=%.2f | reorient: tilt=%.2f step=%.3f cage=%+.0fmm(%s) | descent: tilt=%.2f '
              'cage=%+.0fmm(%s) bottom=%.4f | old_refused=%s | %s'
              % (name, pad_down_vec[4], rtilt, rstep, rc * 1000, rk, dtilt, dc * 1000, dk, bot,
                 old_reject is not None, 'OK' if ok else 'FAIL'))

    print('\nSUMMARY: three-stage chain clears (>= %.0f mm) + level (<= 2.0 deg) + CONTINUOUS (step <= 0.15 '
          'rad) + reaches seat at every park=%s  old single-standoff descent refused at every park=%s  => %s'
          % (args.margin * 1000.0, all_ok, old_all_refused, 'OK' if (all_ok and old_all_refused) else 'FAIL'))
    if args.json:
        args.json.write_text(json.dumps({'shift_m': args.shift, 'height_m': args.height,
                                         'margin_m': args.margin, 'per_park': per_park,
                                         'all_ok': all_ok, 'old_all_refused': old_all_refused,
                                         'ok': all_ok and old_all_refused}, indent=1) + '\n')
    return 0 if (all_ok and old_all_refused) else 1


if __name__ == '__main__':
    sys.exit(main())
