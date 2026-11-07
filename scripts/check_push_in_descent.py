#!/usr/bin/env python3
"""CPU proof: the post-release collect push-in nudge (sim-place-post-release-push-in-v1).

Sibling of ``scripts/check_housing_seat_descent.py``. No Isaac / ROS / GPU -- usd-core (the COMPOSED
coldwin100 ``mission.usda`` cart, read from the run's own stage) + numpy + the mobile URDF + the host XRDF
sphere model. It drives the PRODUCTION housing-seat descent (PR 385) to the lip seat, then the PRODUCTION
push-in planner (``push_in.plan_push_in``) -- re-contact the free carton at the seat and push it inboard
(cart -x) toward the shelf centre, the stroke CAPPED so the side-grip housing keeps a clearance to the shelf
board -- and replays the resulting approach + stroke + retreat against the composed ``/Rollwagen/chassis``
colliders at the recorded cw100 cart pose, PRINTING the per-waypoint minimum signed gaps for HOUSING-vs-SHELF,
the wrist/arm links vs the CAGE MOUTH posts/bars, and vs the DECK (PR 383's shelf-inclusive arm-sphere sweep).

It proves, for the coldwin103 ON values (overhang 0.28 + 0.26 seats):

  1. STROKE CAP: the full-centre stroke (~0.37 m) is CAPPED to a bounded nudge (~0.03 m) so the housing
     keeps the configured ``shelf_clear_min`` (default 10 mm) to the shelf board -- the side-grip housing is
     COAXIAL with the pad normal and hangs ~0.14 m below the carton bottom, so it cannot follow the carton
     inboard over the board (a CENTRED rest needs a TOP-DOWN grip, the out-of-scope follow-up).

  2. CAGE / DECK CLEARANCE: through the whole approach + stroke + retreat the arm keeps >= 30 mm to every
     cage mouth post / bar AND the deck (the brief's cage+deck margin).

  3. CARTON LEVEL + IMPROVED REST: the carton stays within the 2.0 deg bound (the pad keeps it level), and
     the nudge moves the carton CoM inboard off the +x lip (the 0.28 seat's ~14 mm -> ~44 mm margin), raising
     the base-on-shelf support fraction -- WITHOUT the housing striking the board.

    PYTHONPATH=scripts <usd-core-python> scripts/check_push_in_descent.py
        [--overhang 0.28] [--shelf-clear-min 0.010] [--target-x 0.0] [--quick] [--json out.json]

Exit 0 when (1) the stroke is capped to keep the housing >= shelf_clear_min to the shelf, (2) the arm clears
every cage + deck body by >= 30 mm through the whole motion, and (3) the carton stays level and its CoM moves
inboard off the lip. Exit 1 otherwise. Skips (exit 0 + a note) without usd-core, the dataset URDF/XRDF, or
the coldwin100 mission.usda.
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

_MISSION = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mission/coldwin100/mission.usda'
_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'

BASE = (-4.6110, 14.7690, 85.10)
CART = (-4.6028, 15.9756, -93.22)
CARTON_WORLD_Z = 1.1581
CARTON_HALF_M = 0.09
_PAD_PLAN = [1.614, -1.4589, -0.0667, -0.1755, 1.6351, -0.0001]
_PLAN_NAMES = ['ur_elbow_joint', 'ur_shoulder_lift_joint', 'ur_shoulder_pan_joint',
               'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']
_JAM_PLAN = [1.8106, -1.8708, -0.0694, 0.059, 1.7243, -0.0005]
COMPOSED_FLAT_TOP_Z = 0.9639
CAGE_DECK_MARGIN_M = 0.030


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _xlate(x, y, z):
    T = np.eye(4); T[:3, 3] = (x, y, z); return T


def _to_ur(p):
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    d = dict(zip(_PLAN_NAMES, p))
    return {j: d[j] for j in ARM_JOINTS}


def composed_cart_boxes(mission_usda):
    """Reuse the housing-seat check's composed-cart reader (``/Rollwagen/chassis`` collider AABBs in the
    cart body frame, from the run's own stage)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('check_housing_seat_descent',
                                                  ROOT / 'check_housing_seat_descent.py')
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    return chk.composed_cart_boxes(mission_usda)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--overhang', type=float, default=0.28, help='the PR 385 lip-overhang seat to push from')
    ap.add_argument('--lift', type=float, default=0.0314, help='the PR 385 seat z-lift')
    ap.add_argument('--shelf-clear-min', type=float, default=0.010,
                    help='the housing must keep >= this (m) to the shelf board through the stroke')
    ap.add_argument('--target-x', type=float, default=0.0, help='the carton CoM cart-x the push aims for')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args(argv)

    try:
        from twin.restock_manipulation import carton_level as cl
        from twin.restock_manipulation import place_clearance as pc
        from twin.restock_manipulation import place_reorient as pr
        from twin.restock_manipulation import push_in as pin
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        from twin.urdf_probe import parse_urdf
    except Exception as exc:                                       # pragma: no cover
        print('SKIP: cannot import the twin modules (%s)' % exc)
        return 0
    if not (_URDF.is_file() and _XRDF.is_file() and _MISSION.is_file()):
        print('SKIP: dataset URDF/XRDF or coldwin100 mission.usda absent (%s)' % _MISSION)
        return 0
    try:
        boxes = composed_cart_boxes(_MISSION)
    except Exception as exc:
        print('SKIP: usd-core / composed cart unavailable (%s)' % exc)
        return 0

    robot = parse_urdf(str(_URDF))
    geom = pc.GripperCartonGeometry.shipped()
    tcp_link = 'gripper_pad_tcp' if 'gripper_pad_tcp' in robot.links else 'ur_tool0'
    gc = cl.GravityCompensation(tool_link=tcp_link).validate()
    big = 2 * math.pi
    limits = {'ur_elbow_joint': (0.1, math.pi), 'ur_wrist_1_joint': (-big, 0.65), 'ur_wrist_2_joint': (0.0, big)}

    jam = _to_ur(_JAM_PLAN)
    Cb0 = pc._carton_in_base(robot, jam, geom)
    bz = CARTON_WORLD_Z - 0.26 - float(Cb0[2, 3])
    T_world_urbase = _xlate(*BASE[:2], bz) @ _rotz(math.radians(BASE[2])) @ _xlate(0, 0, 0.26)
    T_world_cart = _xlate(*CART[:2], 0.0) @ _rotz(math.radians(CART[2]))
    T_base_cart = np.linalg.inv(T_world_urbase) @ T_world_cart
    T_cart_base = np.linalg.inv(T_base_cart)
    shelf_lo, shelf_hi = boxes['shelf']
    shelf_x_lip = float(shelf_hi[0])
    cage = pc.cage_body_names(boxes)
    sweep = pc.place_descent_sweep_bodies(boxes)      # cage + deck + shelf

    pad = _to_ur(_PAD_PLAN)
    padv = [pad[j] for j in ARM_JOINTS]
    up = cl.carton_up_in_tool(cl.tool_rotation(robot, pad, tcp_link, 'ur_base_link'))
    seat0 = np.asarray(cl.tool_position(robot, pad, tcp_link, 'ur_base_link'), float)

    def housing_seat_tcp(overhang, lift):
        seat = seat0.copy()
        if overhang:
            d = np.asarray(pr.mouth_out_unit(tuple(seat)), float)
            seat = seat + d * float(overhang)
        seat[2] += float(lift)
        return seat

    def carton_bottom_world(vec):
        C = pc._carton_in_base(robot, {j: vec[i] for i, j in enumerate(ARM_JOINTS)}, geom)
        return bz + 0.26 + float(C[2, 3]) - CARTON_HALF_M

    def com_cart_x(vec):
        C = pc._carton_in_base(robot, {j: vec[i] for i, j in enumerate(ARM_JOINTS)}, geom)
        return float((T_cart_base @ C)[0, 3])

    def build_seat_config(overhang, lift, seat_world_z):
        seat = housing_seat_tcp(overhang, lift)
        std = np.asarray(pr.reorient_standoff_tcp(tuple(seat), 0.20, 0.10), float)
        gate = pc.place_descent_cage_clearance_provider(str(_URDF), _XRDF, T_base_cart, bodies=boxes,
                                                        margin=0.03, include=cage)
        clr = pc.place_descent_cage_min_clearance_provider(str(_URDF), _XRDF, T_base_cart, bodies=boxes,
                                                           include=cage)
        pad_down, _ = pr.select_pad_down_branch(
            robot, tuple(std), up, list(padv), seat_tcp=tuple(seat), limits=limits, gravity_comp=gc,
            bound_deg=2.0, tool_link=tcp_link, descent_waypoints=12, approach_from_above=True,
            above_clearance_m=0.08, descent_cage_gate=gate, descent_cage_clearance=clr)
        pdv = [float(pad_down[j]) for j in ARM_JOINTS]
        lo, hi = 0.02, 0.45
        for _ in range(18):
            mid = (lo + hi) / 2.0
            tgt = [float(seat[0]), float(seat[1]), float(seat[2]) - mid]
            plan = cl.plan_compensated_place_descent(robot, list(pdv), tgt, list(pdv), up, gc, bound_deg=2.0,
                                                     tool_link=tcp_link, limits=limits, waypoints=14,
                                                     approach_from_above=True, above_clearance_m=0.08)
            if carton_bottom_world(list(plan.waypoints[-1])) > seat_world_z:
                lo = mid
            else:
                hi = mid
        return list(plan.waypoints[-1]), seat

    def shelf_clear_fn(cfgs):
        per = pc.place_descent_cage_clearance(list(cfgs), str(_URDF), str(_XRDF), boxes, T_base_cart,
                                              include=['shelf'])
        return float(per.get('shelf', float('inf')))

    def per_waypoint_gaps(cfg):
        per = pc.place_descent_cage_clearance([cfg], str(_URDF), str(_XRDF), boxes, T_base_cart, include=sweep)
        shelf = per.get('shelf', float('nan'))
        deck = per.get('deck', float('nan'))
        cmin = min((per[k], k) for k in per if k in cage)
        return shelf, deck, cmin

    print('composed cart: %d colliders; shelf flat top 0.9639, +x lip (cart x) %.4f; shelf_clear_min %.1f mm'
          % (len(boxes), shelf_x_lip, args.shelf_clear_min * 1000))

    overhangs = (args.overhang,) if args.quick else (0.28, 0.26)
    results = []
    overall_ok = True
    for oh in overhangs:
        seat_cfg, seat_tcp = build_seat_config(oh, args.lift, COMPOSED_FLAT_TOP_Z)
        com0 = com_cart_x(seat_cfg)
        tcp0 = np.asarray(cl.tool_position(robot, {j: seat_cfg[i] for i, j in enumerate(ARM_JOINTS)},
                                           tcp_link, 'ur_base_link'), float)
        pdir = pin.push_direction_base(T_base_cart)
        # Start from the release-lifted pose (seat raised +0.08 z), like the controller enters push_in.
        start_cfg = [float(cl.level_place_ik(robot, (tcp0 + np.array([0, 0, 0.08])).tolist(), up,
                     {j: seat_cfg[i] for i, j in enumerate(ARM_JOINTS)}, tool_link=tcp_link,
                     limits=limits, fast=True)[j]) for j in ARM_JOINTS]
        target_stroke = max(0.0, com0 - args.target_x)
        plan = pin.plan_push_in(robot, seat_tcp, up, start_cfg, pdir, target_stroke_m=target_stroke,
                                shelf_clear_min_m=args.shelf_clear_min, shelf_clear_fn=shelf_clear_fn,
                                standoff_m=0.08, backout_m=0.06, waypoints=6, tool_link=tcp_link, limits=limits)
        com_end = com_cart_x(plan.stroke[-1]) if plan.stroke else com0
        print('\n=== overhang %.2f ===' % oh)
        print(' target_stroke=%.1fmm capped=%.1fmm (capped=%s) max_tilt=%.3fdeg nwp=%d'
              % (plan.target_stroke_m * 1000, plan.achieved_stroke_m * 1000, plan.capped,
                 plan.max_tilt_deg, len(plan.waypoints)))
        print(' carton CoM cart-x %.4f -> %.4f (%.1f -> %.1f mm inboard of the +x lip)'
              % (com0, com_end, (shelf_x_lip - com0) * 1000, (shelf_x_lip - com_end) * 1000))
        print(' leg | wp | housing-vs-SHELF | arm-vs-CAGE(worst) | arm-vs-DECK')
        legs = [('approach', plan.approach), ('stroke', plan.stroke), ('retreat', plan.retreat)]
        min_shelf, min_cage, min_deck = float('inf'), float('inf'), float('inf')
        rows = []
        for name, leg in legs:
            for i, cfg in enumerate(leg):
                sh, dk, cm = per_waypoint_gaps(cfg)
                min_shelf = min(min_shelf, sh); min_cage = min(min_cage, cm[0]); min_deck = min(min_deck, dk)
                rows.append({'leg': name, 'wp': i, 'shelf_mm': round(sh * 1000, 1),
                             'cage_mm': round(cm[0] * 1000, 1), 'cage_body': cm[1], 'deck_mm': round(dk * 1000, 1)})
                print('  %-8s | %2d | %+9.1f mm    | %+9.1f mm (%-18s | %+9.1f mm'
                      % (name, i, sh * 1000, cm[0] * 1000, cm[1] + ')', dk * 1000))
        shelf_ok = min_shelf >= args.shelf_clear_min - 1e-4
        cage_ok = min_cage >= CAGE_DECK_MARGIN_M
        deck_ok = min_deck >= CAGE_DECK_MARGIN_M
        tilt_ok = plan.max_tilt_deg <= 2.0 + 1e-6
        improved = com_end < com0 - 1e-4
        ok = shelf_ok and cage_ok and deck_ok and tilt_ok and improved
        overall_ok = overall_ok and ok
        print('  -> min: shelf %+.1fmm (>= %.1f: %s), cage %+.1fmm (>= 30: %s), deck %+.1fmm (>= 30: %s) | '
              'tilt %.2fdeg (<= 2.0: %s) | CoM moved inboard: %s => %s'
              % (min_shelf * 1000, args.shelf_clear_min * 1000, shelf_ok, min_cage * 1000, cage_ok,
                 min_deck * 1000, deck_ok, plan.max_tilt_deg, tilt_ok, improved, 'OK' if ok else 'FAIL'))
        results.append({'overhang': oh, 'target_stroke_mm': round(plan.target_stroke_m * 1000, 1),
                        'capped_stroke_mm': round(plan.achieved_stroke_m * 1000, 1), 'capped': plan.capped,
                        'com_before': round(com0, 4), 'com_after': round(com_end, 4),
                        'com_inboard_before_mm': round((shelf_x_lip - com0) * 1000, 1),
                        'com_inboard_after_mm': round((shelf_x_lip - com_end) * 1000, 1),
                        'max_tilt_deg': round(plan.max_tilt_deg, 3), 'min_shelf_mm': round(min_shelf * 1000, 1),
                        'min_cage_mm': round(min_cage * 1000, 1), 'min_deck_mm': round(min_deck * 1000, 1),
                        'ok': ok, 'per_waypoint': rows})

    print('\nSUMMARY: the post-release push-in is a BOUNDED shelf-safe nudge (stroke capped so the housing '
          'stays off the shelf board); cage + deck >= 30 mm; carton level; CoM moved inboard off the +x lip. '
          'A CENTRED rest needs a top-down grip (out of scope) => %s' % ('OK' if overall_ok else 'FAIL'))
    if args.json:
        args.json.write_text(json.dumps({'shelf_clear_min_m': args.shelf_clear_min, 'target_x_m': args.target_x,
                                         'results': results, 'ok': overall_ok}, indent=1) + '\n')
    return 0 if overall_ok else 1


if __name__ == '__main__':
    sys.exit(main())
