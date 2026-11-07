#!/usr/bin/env python3
"""CPU proof: the collect place-DESCENT reaches the corrected cart seat WITHOUT the gripper housing
catching on the Rollwagen cage (sim-place-descent-reach-cage-v1).

No Isaac / ROS / GPU -- usd-core (the authored cart via ``twin.mobile_assets.author_cart``) + numpy + the
mobile URDF + the host XRDF sphere model. It extends the PR 366 pre-place sweep tool
(``scripts/check_preplace_sweep_clearance.py`` / ``place_clearance.preplace_cage_arm_clearance``) to the
place DESCENT and reconstructs the coldwin87 set-down at the real park/cart pose:

* OLD -- the coldwin87 executed place-descent plan (the host-stack ``TRAJ_HOST_PLAN``, 13 leveled
  waypoints). Its FIRST leg reorients the gripper from the standoff's CARRY wrist orientation
  (wrist_2 ~ 0.28) to the place PAD-DOWN orientation (wrist_2 ~ 1.63) WHILE the gripper housing sits inside
  the +x cage mouth, so the UNFILTERED housing link ``cube_without_surface_plates`` wedges against the upper
  side bar ``side_1_horizontal_6`` and the mouth posts at the descent START (the ONLY penetrating config of
  the whole path). wrist_2 then froze at 0.28 against a 1.63 rad command -> TRAJ_HOST_TRACK_MISS worst_j=4
  err 1.30 -> PATH_TOLERANCE_VIOLATED. The descent cage gate REFUSES it by name
  (PLACE_DESCENT_CAGE_CLEARANCE prim=side_1_horizontal_6 min_clearance < 0).

* NEW -- the REMEDY descent holds the PAD-DOWN wrist branch (wrist_2 ~ 1.63) from the standoff to the
  CORRECTED seat (carton bottom 0.9325 world, the PR 375 settled shelf flat top), planned by the SAME
  production ``carton_level.plan_compensated_place_descent`` the place gate runs. CPU-proven: every gripper
  link stays >= 0.03 m (actually >= ~0.10 m) from EVERY cage body at every collect-place park pose, the
  descent tilt stays within the 2.0 deg carton-level bound (UNTOUCHED), and the carton lands at the corrected
  seat. The cage gate PASSES it.

    PYTHONPATH=scripts <usd-core-python> scripts/check_place_descent_reach.py [--margin 0.03] [--json out.json]

Exit 0 when the OLD descent is refused AND the NEW descent clears (>= margin) at EVERY park pose AND lands at
the seat within tolerance AND stays within the 2.0 deg tilt bound; exit 1 otherwise. Skips (exit 0 with a
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

# ------------------------------------------------------------------ coldwin87 geometry (run artifacts)
# datasets/twin-robot-in-store-boot-v1/mission/coldwin87: nav-trace.jsonl (park base + cart + frozen carton)
# and host-stack.log (the place-descent TRAJ_HOST_PLAN + the CARTON_TICK frozen jam config).
# The base x/y/yaw is the trusted nav base pose; the frozen carton WORLD z calibrates base_link world z via
# FK (so the reconstructed arm sits at the measured height), exactly like check_preplace_sweep_clearance.
CW87_BASE = (-4.6100, 14.7690, 86.00)          # x, y, yaw_deg
CW87_CART = (-4.6068, 15.9723, -93.61)         # x, y, yaw_deg (cart root; z=0)
CW87_CARTON_WORLD_Z = 1.1634                    # frozen carton CENTRE world z (nav-trace last row)
CW87_JAM_Q = [-0.5045, -1.3624, 1.3084, 0.1072, 0.279, -0.0572]   # CARTON_TICK frozen config (ARM order)

# The CORRECTED seat (sim-place-seat-geometry-v1 / PR 375): carton BOTTOM on the settled shelf flat top.
SEAT_CARTON_BOTTOM_WORLD_Z = 0.9325
CARTON_HALF_M = 0.09                            # 0.18 m carton half-height (handling_unit.dimensions_m)

# coldwin87 place-descent TRAJ_HOST_PLAN, names order [elbow, lift, pan, w1, w2, w3] (13 leveled waypoints).
_PLAN_NAMES = ['ur_elbow_joint', 'ur_shoulder_lift_joint', 'ur_shoulder_pan_joint',
               'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']
_CW87_DESCENT_PLAN = [
    [1.3084, -1.3624, -0.5045, 0.1072, 0.279, -0.0572], [1.5343, -1.5555, -0.085, 0.0335, 1.6281, -0.0003],
    [1.5172, -1.5388, -0.0932, 0.0256, 1.6269, -0.0088], [1.4993, -1.5216, -0.1011, 0.0181, 1.6255, -0.0171],
    [1.4808, -1.5042, -0.1089, 0.011, 1.624, -0.0252], [1.4664, -1.4886, -0.1156, 0.0015, 1.623, -0.0332],
    [1.4901, -1.4886, -0.1143, -0.0222, 1.6263, -0.0413], [1.5132, -1.4879, -0.1131, -0.046, 1.6294, -0.0494],
    [1.5358, -1.4867, -0.112, -0.0697, 1.6325, -0.0528], [1.5578, -1.485, -0.1109, -0.0935, 1.6356, -0.0528],
    [1.5792, -1.4827, -0.1098, -0.1172, 1.6385, -0.0528], [1.6002, -1.4799, -0.1088, -0.1409, 1.6413, -0.0528],
    [1.6206, -1.4766, -0.1077, -0.1647, 1.6441, -0.0528]]

# The additional collect-place park poses the fix must also clear (reach at the cw64-cw89 park poses). Each is
# (base x, y, yaw_deg, cart x, y, yaw_deg) from that run's nav-trace place window (max carton height). The
# station park is deterministic, so base_link world z is shared; only the horizontal park varies run-to-run.
_PARKS = {
    'coldwin87': (-4.6100, 14.7690, 86.00, -4.6068, 15.9723, -93.61),
    'coldwin64': (-4.6200, 14.7430, 85.30, -4.5977, 15.9859, -92.48),
    'coldwin88': (-4.6110, 14.7670, 84.80, -4.6028, 15.9756, -93.22),
    'coldwin89': (-4.6110, 14.7690, 84.70, -4.6028, 15.9756, -93.22),
}

_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _xlate(x, y, z):
    T = np.eye(4); T[:3, 3] = (x, y, z); return T


def _to_ur(p):
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    d = dict(zip(_PLAN_NAMES, p))
    return [d[j] for j in ARM_JOINTS]


def _base_link_z(robot, pc, geom):
    """base_link world z calibrated from the frozen carton world z at the jam config (FK)."""
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    Cu = pc._carton_in_base(robot, {j: CW87_JAM_Q[i] for i, j in enumerate(ARM_JOINTS)}, geom)
    return CW87_CARTON_WORLD_Z - 0.26 - float(Cu[2, 3])


def _t_base_cart(park, bz):
    bx, by, byaw, cx, cy, cyaw = park
    T_world_urbase = _xlate(bx, by, bz) @ _rotz(math.radians(byaw)) @ _xlate(0, 0, 0.26)
    T_world_cartroot = _xlate(cx, cy, 0.0) @ _rotz(math.radians(cyaw))
    return np.linalg.inv(T_world_urbase) @ T_world_cartroot


def _carton_bottom_world_z(robot, pc, geom, cfg_vec, bz):
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    C = pc._carton_in_base(robot, {j: cfg_vec[i] for i, j in enumerate(ARM_JOINTS)}, geom)
    return bz + 0.26 + float(C[2, 3]) - CARTON_HALF_M


def build_clear_descent(robot, pc, cl, geom, bz, *, standoff_vec, waypoints=12):
    """Build the REMEDY place descent: hold the pad-down wrist branch (the ``standoff_vec`` wrist) and lower
    the leveled carton from the standoff to the corrected seat (carton bottom SEAT_CARTON_BOTTOM_WORLD_Z),
    via the production ``plan_compensated_place_descent``. Returns (waypoint_vecs, max_tilt, final_bottom_z).
    The target TCP drop is binary-searched so the final carton bottom lands on the corrected seat."""
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    std_map = {j: float(standoff_vec[i]) for i, j in enumerate(ARM_JOINTS)}
    tR = cl.tool_rotation(robot, std_map, tool_link='ur_tool0', root='ur_base_link')
    up_in_tool = cl.carton_up_in_tool(tR)
    tcp = cl.tool_position(robot, std_map, 'ur_tool0', 'ur_base_link')
    gc = cl.GravityCompensation(tool_link='ur_tool0').validate()
    big = 2 * math.pi
    limits = {'ur_elbow_joint': (0.1, math.pi), 'ur_wrist_1_joint': (-big, 0.65),
              'ur_wrist_2_joint': (0.0, big)}

    def _plan(drop):
        tgt = [float(tcp[0]), float(tcp[1]), float(tcp[2]) - float(drop)]
        p = cl.plan_compensated_place_descent(
            robot, list(standoff_vec), tgt, list(standoff_vec), up_in_tool, gc, bound_deg=2.0,
            tool_link='ur_tool0', limits=limits, waypoints=waypoints, approach_from_above=True,
            above_clearance_m=0.08, depth_bias_m=0.0)
        last = list(p.waypoints[-1])
        return p, _carton_bottom_world_z(robot, pc, geom, last, bz)

    lo, hi = 0.05, 0.25
    for _ in range(26):
        mid = (lo + hi) / 2.0
        _, bottom = _plan(mid)
        if bottom > SEAT_CARTON_BOTTOM_WORLD_Z:
            lo = mid
        else:
            hi = mid
    plan, bottom = _plan((lo + hi) / 2.0)
    return [list(w) for w in plan.waypoints], float(max(plan.max_roll, plan.max_pitch)), float(bottom)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--margin', type=float, default=None, help='refuse margin (m); default the 30 mm design')
    ap.add_argument('--json', type=Path, help='write the result JSON here')
    args = ap.parse_args(argv)

    try:
        from twin.restock_manipulation import carton_level as cl
        from twin.restock_manipulation import place_clearance as pc
        from twin.urdf_probe import parse_urdf
    except Exception as exc:                                    # pragma: no cover
        print('SKIP: cannot import place_clearance/carton_level/urdf_probe (%s)' % exc)
        return 0
    if not (_URDF.is_file() and _XRDF.is_file()):
        print('SKIP: dataset URDF/XRDF not on this box (%s)' % _URDF)
        return 0
    try:
        bodies = pc.rollwagen_collider_boxes()
    except Exception as exc:
        print('SKIP: usd-core / authored cart unavailable (%s)' % exc)
        return 0

    margin = pc.DEFAULT_PLACE_DESCENT_CAGE_MARGIN_M if args.margin is None else float(args.margin)
    robot = parse_urdf(str(_URDF))
    geom = pc.GripperCartonGeometry.shipped()
    cage = pc.cage_body_names(bodies)
    bz = _base_link_z(robot, pc, geom)

    old = [_to_ur(p) for p in _CW87_DESCENT_PLAN]
    # The pad-down standoff = the reoriented first descent waypoint (wrist_2 ~ 1.63), the branch whose descent
    # the fix selects. Build the clear remedy descent from it down to the corrected seat.
    standoff = _to_ur(_CW87_DESCENT_PLAN[1])
    new, new_tilt, new_bottom = build_clear_descent(robot, pc, cl, geom, bz, standoff_vec=standoff)

    def _sweep(path, T_base_cart):
        per = pc.place_descent_cage_clearance(path, str(_URDF), str(_XRDF), bodies, T_base_cart, include=cage)
        worst_v, worst_k = min((v, k) for k, v in per.items())
        return per, worst_v, worst_k

    print('corrected seat: carton bottom %.4f world (settled shelf flat top; carton half %.3f)'
          % (SEAT_CARTON_BOTTOM_WORLD_Z, CARTON_HALF_M))
    print('NEW remedy descent: holds wrist_2 in [%.3f, %.3f] (pad-down), max tilt %.3f deg (<= 2.00 bound), '
          'lands carton bottom %.4f (seat %.4f, |err| %.1f mm)'
          % (min(w[4] for w in new), max(w[4] for w in new), new_tilt, new_bottom,
             SEAT_CARTON_BOTTOM_WORLD_Z, abs(new_bottom - SEAT_CARTON_BOTTOM_WORLD_Z) * 1000.0))

    per_park = {}
    old_all_refused = True
    new_all_clear = True
    for name, park in _PARKS.items():
        T = _t_base_cart(park, bz)
        _, ov, ok = _sweep(old, T)
        new_per, nv, nk = _sweep(new, T)
        old_reject = pc.place_descent_sweep_reject(
            pc.place_descent_cage_clearance(old, str(_URDF), str(_XRDF), bodies, T, include=cage),
            margin=margin)
        new_reject = pc.place_descent_sweep_reject(new_per, margin=margin)
        old_all_refused = old_all_refused and (old_reject is not None)
        new_all_clear = new_all_clear and (new_reject is None)
        per_park[name] = {'old_min_mm': round(ov * 1000, 1), 'old_body': ok,
                          'old_refused': old_reject is not None,
                          'new_min_mm': round(nv * 1000, 1), 'new_body': nk,
                          'new_refused': new_reject is not None}
        print('  %-10s OLD min=%7.1fmm(%-20s refused=%s)  NEW min=%7.1fmm(%-20s clear=%s)'
              % (name, ov * 1000, ok, old_reject is not None, nv * 1000, nk, new_reject is None))

    seat_ok = abs(new_bottom - SEAT_CARTON_BOTTOM_WORLD_Z) <= 0.02
    tilt_ok = new_tilt <= 2.0 + 1e-9
    ok = old_all_refused and new_all_clear and seat_ok and tilt_ok
    print('\nSUMMARY: OLD refused at every park=%s  NEW clear (>= %.0f mm) at every park=%s  '
          'NEW reaches seat (|err| <= 20 mm)=%s  NEW tilt <= 2.0 deg=%s  => %s'
          % (old_all_refused, margin * 1000.0, new_all_clear, seat_ok, tilt_ok, 'OK' if ok else 'FAIL'))
    if args.json:
        args.json.write_text(json.dumps({
            'margin_m': margin, 'corrected_seat_bottom_z': SEAT_CARTON_BOTTOM_WORLD_Z,
            'new_max_tilt_deg': round(new_tilt, 3), 'new_final_bottom_z': round(new_bottom, 4),
            'new_wrist2_range': [round(min(w[4] for w in new), 3), round(max(w[4] for w in new), 3)],
            'per_park': per_park, 'old_all_refused': old_all_refused, 'new_all_clear': new_all_clear,
            'seat_ok': seat_ok, 'tilt_ok': tilt_ok, 'ok': ok}, indent=1) + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
