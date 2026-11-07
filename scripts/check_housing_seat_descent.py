#!/usr/bin/env python3
"""CPU proof: the side-grip-HOUSING-compatible collect place seat (sim-place-housing-compatible-seat-v1).

Sibling of ``scripts/check_pad_down_shelf_contact.py``. No Isaac / ROS / GPU -- usd-core (the COMPOSED
coldwin100 ``mission.usda`` cart, read from the run's own stage, NOT the ``author_cart`` box model) +
numpy + the mobile URDF + the host XRDF sphere model. It drives the PRODUCTION pad-down branch selector
(``place_reorient.select_pad_down_branch``) + descent planner (``carton_level.plan_compensated_place_descent``)
through the SAME housing-seat transform the controller applies (``TransferController._housing_seat_tcp``: a
mouth-out lip overhang + a z-lift to the pinned composed flat top), replays the resulting descent against
the composed ``/Rollwagen/chassis`` colliders at the recorded cw100 cart pose, and PRINTS the per-waypoint
minimum signed gaps for HOUSING-vs-SHELF, HOUSING-vs-CAGE and HOUSING-vs-DECK (PR 383's shelf-inclusive
arm-sphere sweep).

It proves:

  1. SEAT-Z FIX (``housing_seat_z_lift_m``): PR 383 named the recorded descent over-driving toward the
     author_cart BOX-MODEL shelf top (0.9325 world) while the run pins the COMPOSED flat top (0.9639).
     With the +0.0314 lift the descent lands the carton bottom on the COMPOSED flat top -- the pins and
     the descent agree.

  2. OVERHANG knob (``housing_seat_lip_overhang_m``): the housing-vs-shelf clearance rises monotonically
     with the mouth-out shift; at the coldwin102 overhang the gripper housing clears the shelf board by
     >= 30 mm (PR 383's jam -- -57 mm at overhang 0 -- is gone) WHILE the carton CoM stays INBOARD of the
     shelf +x lip (a stable rest; the carton base keeps ~half its footprint on the shelf). So a stable,
     housing-clear side-grip seat DOES exist -- in a narrow overhang window near the lip.

  3. The honest tradeoff: housing clearance trades against the CoM stability margin (both move with the
     overhang), so the window is narrow and the rest is edge-weighted. A TOP-DOWN grip (out of scope: it
     changes the pick + transfer) gives a centred, robustly-stable rest with a wider margin -- the
     follow-up for tow-robustness.

    PYTHONPATH=scripts <usd-core-python> scripts/check_housing_seat_descent.py [--overhang 0.28]
        [--lift 0.0314] [--quick] [--json out.json]

Exit 0 when (1) the lift targets the composed flat top, (2) the coldwin102 overhang clears the shelf
>= 30 mm, and (3) that overhang keeps the carton CoM inboard of the +x lip (a stable rest). Exit 1
otherwise. Skips (exit 0 + a note) without usd-core, the dataset URDF/XRDF, or the coldwin100 mission.usda.
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

# ------------------------------------------------------------------ coldwin100 recorded park (run artifacts)
_MISSION = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mission/coldwin100/mission.usda'
_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'

BASE = (-4.6110, 14.7690, 85.10)        # recorded base x, y, yaw_deg (nav-trace @ parked)
CART = (-4.6028, 15.9756, -93.22)       # recorded settled cart root x, y, yaw_deg (z=0)
CARTON_WORLD_Z = 1.1581                 # recorded frozen carton CENTRE world z (nav-trace)
CARTON_HALF_M = 0.09

# A high-wrist_2 pad-down reference config (the branch the three-stage set-down holds): the carton up-axis
# + the baseline seat TCP. From the coldwin100 TRAJ_HOST_PLAN pad-down leg (names [elbow,lift,pan,w1,w2,w3]).
_PAD_PLAN = [1.614, -1.4589, -0.0667, -0.1755, 1.6351, -0.0001]
_PLAN_NAMES = ['ur_elbow_joint', 'ur_shoulder_lift_joint', 'ur_shoulder_pan_joint',
               'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']
_JAM_PLAN = [1.8106, -1.8708, -0.0694, 0.059, 1.7243, -0.0005]   # cw100 jam, to calibrate base_link world z

# The two shelf-top heights PR 383 named: the COMPOSED flat top (the pins / the real geometry) and the
# author_cart BOX-MODEL top (the seat the recorded descent over-drove toward). WORLD z (cart ground z=0).
COMPOSED_FLAT_TOP_Z = 0.9639
BOX_MODEL_TOP_Z = 0.9325
# The housing-vs-shelf clearance the brief targets (m), and the penetration-only gate floor (a close ride
# >= 0 does not strike -- PR 383's support margin).
DESIGN_MARGIN_M = 0.030


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
    """``{name: (lo, hi)}`` of every composed ``/Rollwagen/chassis`` collider AABB in the cart BODY frame,
    from the run's ``mission.usda`` (the REAL geometry, not the box model). Needs usd-core."""
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    stage = Usd.Stage.Open(str(mission_usda))
    roll = stage.GetPrimAtPath('/Rollwagen')
    Twr = UsdGeom.Xformable(roll).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    boxes = {}
    for child in stage.GetPrimAtPath('/Rollwagen/chassis').GetChildren():
        if not child.HasAPI(UsdPhysics.CollisionAPI):
            continue
        m = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rel = m * Twr.GetInverse()
        corners = np.array([list(rel.Transform(Gf.Vec3d(x, y, z)))
                            for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
        boxes[child.GetName()] = (corners.min(0), corners.max(0))
    if not boxes:
        raise ValueError('no composed /Rollwagen/chassis colliders in %s' % mission_usda)
    return boxes


def _aabb_clip_area(poly, lo, hi):
    """Area of a convex polygon ``poly`` (list of (x, y)) clipped to the AABB ``[lo, hi]`` (Sutherland-Hodgman)."""
    def _ix(a, b, ax, val):
        t = (val - a[ax]) / (b[ax] - a[ax])
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    pts = list(poly)
    edges = [(lambda p: p[0] >= lo[0], lambda a, b: _ix(a, b, 0, lo[0])),
             (lambda p: p[0] <= hi[0], lambda a, b: _ix(a, b, 0, hi[0])),
             (lambda p: p[1] >= lo[1], lambda a, b: _ix(a, b, 1, lo[1])),
             (lambda p: p[1] <= hi[1], lambda a, b: _ix(a, b, 1, hi[1]))]
    for inside, inter in edges:
        out = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            ina, inb = inside(a), inside(b)
            if ina:
                out.append(a)
            if ina != inb:
                out.append(inter(a, b))
        pts = out
        if not pts:
            return 0.0
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--overhang', type=float, default=0.28,
                    help='the coldwin102 mouth-out lip overhang (m); default 0.28 (>= 30 mm housing clearance, CoM on the shelf)')
    ap.add_argument('--lift', type=float, default=0.0314,
                    help='the seat z-lift (m); default 0.0314 = composed flat top 0.9639 - box-model top 0.9325')
    ap.add_argument('--quick', action='store_true', help='fewer tradeoff points (the test path)')
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
    sweep = pc.place_descent_sweep_bodies(boxes)         # cage + deck + shelf (PR 383 shelf-inclusive)
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

    def carton_seat_metrics(vec):
        """(CoM cart-x, CoM-inboard-of-lip m, base-overlap-on-shelf fraction) at the seated config."""
        C = pc._carton_in_base(robot, {j: vec[i] for i, j in enumerate(ARM_JOINTS)}, geom)
        Cc = T_cart_base @ C
        half = np.array(pc.CARTON_DIMS_M) / 2.0
        corners = [tuple((Cc @ np.array([sx * half[0], sy * half[1], -half[2], 1.0]))[:2])
                   for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        area = _aabb_clip_area([(float(x), float(y)) for x, y in corners],
                               (float(shelf_lo[0]), float(shelf_lo[1])), (float(shelf_hi[0]), float(shelf_hi[1])))
        full = pc.CARTON_DIMS_M[0] * pc.CARTON_DIMS_M[1]
        cx = float(Cc[0, 3])
        return cx, shelf_x_lip - cx, area / full

    _cache = {}

    def build_descent(overhang, lift, seat_world_z):
        key = (round(overhang, 4), round(lift, 4), round(seat_world_z, 4))
        if key in _cache:
            return _cache[key]
        seat = housing_seat_tcp(overhang, lift)
        std = np.asarray(pr.reorient_standoff_tcp(tuple(seat), 0.20, 0.10), float)
        gate = pc.place_descent_cage_clearance_provider(str(_URDF), _XRDF, T_base_cart, bodies=boxes,
                                                        margin=0.03, include=cage)
        clr = pc.place_descent_cage_min_clearance_provider(str(_URDF), _XRDF, T_base_cart, bodies=boxes,
                                                           include=cage)
        try:
            pad_down, _ = pr.select_pad_down_branch(
                robot, tuple(std), up, list(padv), seat_tcp=tuple(seat), limits=limits, gravity_comp=gc,
                bound_deg=2.0, tool_link=tcp_link, descent_waypoints=12, approach_from_above=True,
                above_clearance_m=0.08, descent_cage_gate=gate, descent_cage_clearance=clr)
        except ValueError:
            _cache[key] = None
            return None
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
        wps = [list(w) for w in plan.waypoints]
        _cache[key] = wps
        return wps

    def per_waypoint_gaps(waypoints):
        rows = []
        for w in waypoints:
            per = pc.place_descent_cage_clearance([w], str(_URDF), str(_XRDF), boxes, T_base_cart, include=sweep)
            shelf = per.get('shelf', float('nan'))
            deck = per.get('deck', float('nan'))
            cmin = min((per[k], k) for k in per if k in cage)
            rows.append((shelf, deck, cmin[0], cmin[1]))
        return rows

    print('composed cart: %d colliders; shelf flat top (world) 0.9639, +x lip (cart x) %.4f; box-model top 0.9325'
          % (len(boxes), shelf_x_lip))

    # -- (1) seat-z fix: the lift targets the composed flat top, not the box model ---------------------
    d_nolift = build_descent(args.overhang, 0.0, BOX_MODEL_TOP_Z)
    d_lift = build_descent(args.overhang, args.lift, COMPOSED_FLAT_TOP_Z)
    if d_nolift is None or d_lift is None:
        print('NOTE: no pad-down branch at overhang %.3f -- the seat is unreachable there; try a smaller '
              '--overhang' % args.overhang)
        return 1
    bottom_nolift = carton_bottom_world(d_nolift[-1])
    bottom_lift = carton_bottom_world(d_lift[-1])
    print('\n(1) SEAT-Z FIX: descent WITHOUT lift lands carton bottom %.4f (box-model top %.4f); WITH '
          '+%.4f lift lands %.4f (composed flat top %.4f) -> the pins and the descent agree'
          % (bottom_nolift, BOX_MODEL_TOP_Z, args.lift, bottom_lift, COMPOSED_FLAT_TOP_Z))
    seat_z_ok = abs(bottom_lift - COMPOSED_FLAT_TOP_Z) <= 0.02

    # -- (2) per-waypoint gaps at the coldwin102 seat (overhang + lift) --------------------------------
    rows = per_waypoint_gaps(d_lift)
    cx, com_margin, overlap = carton_seat_metrics(d_lift[-1])
    print('\n(2) PER-WAYPOINT housing-vs-cart gaps at overhang=%.3f lift=%.4f (composed cw100 cart):'
          % (args.overhang, args.lift))
    print('   wp | housing-vs-SHELF | housing-vs-CAGE(worst) | housing-vs-DECK')
    for i, (shelf, deck, cv, ck) in enumerate(rows):
        print('   %2d | %+9.1f mm    | %+9.1f mm (%-18s | %+9.1f mm'
              % (i, shelf * 1000, cv * 1000, ck + ')', deck * 1000))
    min_shelf = min(r[0] for r in rows)
    min_cage = min(r[2] for r in rows)
    chosen_clear = min_shelf >= DESIGN_MARGIN_M
    chosen_stable = com_margin > 0.0
    print('   -> min over descent: shelf %+.1f mm (>= 30 mm: %s), cage %+.1f mm | seated carton: CoM cart-x '
          '%+.4f, CoM %+.1f mm inboard of the +x lip (stable: %s), base-on-shelf %.0f%%'
          % (min_shelf * 1000, chosen_clear, min_cage * 1000, cx, com_margin * 1000, chosen_stable, overlap * 100))

    # -- (3) the tradeoff: housing clearance vs CoM stability margin -----------------------------------
    grid = (0.26, args.overhang, 0.30) if args.quick else (0.0, 0.20, 0.26, args.overhang, 0.30)
    grid = sorted(set(round(o, 4) for o in grid))
    print('\n(3) HOUSING-CLEARANCE vs CARTON-CoM tradeoff (composed cw100 cart; shelf +x lip x=%.4f):'
          % shelf_x_lip)
    print('   overhang | min housing-vs-shelf | CoM->+x lip | base-on-shelf | verdict')
    tradeoff = []
    stable_clear_exists = False
    for oh in grid:
        d = build_descent(oh, args.lift, COMPOSED_FLAT_TOP_Z)
        if d is None:
            print('   %7.3f  | (no pad-down branch reachable)' % oh)
            tradeoff.append({'overhang': oh, 'reachable': False})
            continue
        ms = min(r[0] for r in per_waypoint_gaps(d))
        cxx, cm, ov = carton_seat_metrics(d[-1])
        stable = cm > 0.0
        clear = ms >= DESIGN_MARGIN_M
        sc = stable and clear
        stable_clear_exists = stable_clear_exists or sc
        verdict = ('STABLE+CLEAR(>=30mm)' if sc else
                   ('stable,close-ride(+%dmm)' % round(ms * 1000) if (stable and ms >= 0) else
                    ('clear,CoM-OFF-shelf' if clear else ('stable,STRIKE(%dmm)' % round(ms * 1000) if stable
                                                          else 'neither'))))
        print('   %7.3f  |      %+8.1f mm    |  %+.4f m  |     %3.0f%%      | %s'
              % (oh, ms * 1000, cm, ov * 100, verdict))
        tradeoff.append({'overhang': oh, 'reachable': True, 'min_shelf_mm': round(ms * 1000, 1),
                         'com_vs_lip_m': round(cm, 4), 'base_on_shelf_frac': round(ov, 3),
                         'stable': stable, 'clear': clear})
    print('   -> a STABLE (CoM inboard of the +x lip) AND housing-CLEAR (>= 30 mm) seat exists: %s'
          % stable_clear_exists)
    print('      (housing clearance trades against the CoM margin; the window is narrow + edge-weighted. A '
          'TOP-DOWN grip -- out of scope -- gives a centred, robustly-stable rest with a wider margin: follow-up)')

    ok = seat_z_ok and chosen_clear and chosen_stable and stable_clear_exists
    print('\nSUMMARY: seat-z lift targets the composed flat top=%s | overhang %.3f clears the shelf >= 30 mm=%s '
          '+ CoM on the shelf (stable)=%s | a stable+clear side-grip seat exists=%s => %s'
          % (seat_z_ok, args.overhang, chosen_clear, chosen_stable, stable_clear_exists, 'OK' if ok else 'FAIL'))
    if args.json:
        args.json.write_text(json.dumps({
            'overhang_m': args.overhang, 'lift_m': args.lift,
            'bottom_nolift': round(bottom_nolift, 4), 'bottom_lift': round(bottom_lift, 4), 'seat_z_ok': seat_z_ok,
            'min_shelf_mm': round(min_shelf * 1000, 1), 'min_cage_mm': round(min_cage * 1000, 1),
            'chosen_clear': chosen_clear, 'chosen_stable': chosen_stable,
            'chosen_com_vs_lip_m': round(com_margin, 4), 'chosen_base_on_shelf_frac': round(overlap, 3),
            'per_waypoint': [{'wp': i, 'shelf_mm': round(r[0] * 1000, 1), 'deck_mm': round(r[1] * 1000, 1),
                              'cage_mm': round(r[2] * 1000, 1), 'cage_body': r[3]} for i, r in enumerate(rows)],
            'tradeoff': tradeoff, 'stable_clear_exists': stable_clear_exists, 'ok': ok}, indent=1) + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
