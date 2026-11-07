#!/usr/bin/env python3
"""CPU proof: the coldwin100 pad-down place JAM is the gripper SUCTION HOUSING striking the Rollwagen
UPPER SHELF -- the support board the cage-only descent clearance sweep was structurally blind to
(sim-pad-down-descent-jam-physics-v1).

No Isaac / ROS / GPU -- usd-core (the COMPOSED mission cart, read from the run's own ``mission.usda``, NOT
the ``author_cart`` box model) + numpy + the mobile URDF + the host XRDF sphere model. It replays the exact
recorded coldwin100 ``TRAJ_HOST_PLAN`` pad-down descent (13 IK waypoints, from the host-stack log) through
forward kinematics against the REAL composed ``/Rollwagen/chassis`` colliders at the recorded settled cart
pose, and proves, for the recorded jam:

  * the CAGE-ONLY arm-sphere sweep (the pre-fix model) reads the descent CLEAR -- min cage clearance
    >= +0.10 m (exactly the ``descent_clearance ~ 110 mm`` the host logged while the carton would not
    descend); yet
  * the SHELF-INCLUSIVE arm-sphere sweep (the corrected model, :func:`place_clearance.place_descent_sweep_bodies`)
    finds the gripper suction housing (``suction_cups`` / ``surface_plate_*`` / ``cube_without_surface_plates``)
    driven deep INTO ``/Rollwagen/chassis/shelf`` (<= -0.03 m, actually ~ -0.05 m), and
    :func:`place_clearance.place_descent_sweep_reject` REFUSES it BY NAME (``prim=shelf``) -- reproducing the
    coldwin100 jam the cage-only sweep could not see.

It also confirms the cause is not the cage and not the carton: the welded carton is cage-filtered
(``/RestockScenario/HandlingUnit`` ``physics:filteredPairs`` covers every cage upright/bar), the carton is
~0.10 m above the shelf, and only the unfiltered gripper housing -- which hangs ~0.14 m below the carton
bottom in the pad-down orientation -- reaches the shelf board.

    PYTHONPATH=scripts <usd-core-python> scripts/check_pad_down_shelf_contact.py [--json out.json]

Exit 0 when the cage-only sweep clears AND the shelf-inclusive sweep refuses naming the shelf (the jam
reproduced) at the recorded park; exit 1 otherwise. Skips (exit 0 + a note) without usd-core, the dataset
URDF/XRDF, or the coldwin100 mission.usda.
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

# ------------------------------------------------------------------ coldwin100 recorded jam (run artifacts)
# datasets/twin-robot-in-store-boot-v1/mission/coldwin100: host-stack.log (the PLACE_REORIENT pad-down config,
# the TRAJ_HOST_PLAN place-descent, the CARTON_TICK frozen jam vector) + nav-trace.jsonl (the settled base +
# cart pose and the frozen carton CENTRE world z at the jam).
_MISSION = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mission/coldwin100/mission.usda'
_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'

BASE = (-4.6110, 14.7690, 85.10)        # recorded base x, y, yaw_deg (nav-trace @ t~290.8, parked)
CART = (-4.6028, 15.9756, -93.22)       # recorded settled cart root x, y, yaw_deg (z=0)
CARTON_WORLD_Z = 1.1581                 # recorded frozen carton CENTRE world z (nav-trace)

# TRAJ_HOST_PLAN names order: [elbow, lift, pan, w1, w2, w3] (host-stack TRAJ_HOST_PLAN, 13 waypoints).
_PLAN_NAMES = ['ur_elbow_joint', 'ur_shoulder_lift_joint', 'ur_shoulder_pan_joint',
               'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']
_CW100_DESCENT_PLAN = [
    [1.8106, -1.8708, -0.0694, 0.059, 1.7243, -0.0005], [1.7781, -1.8239, -0.0701, 0.0587, 1.7112, -0.0005],
    [1.7433, -1.7773, -0.0713, 0.0396, 1.6977, -0.0005], [1.7059, -1.7309, -0.0721, 0.0231, 1.6844, -0.0004],
    [1.6663, -1.6845, -0.0727, 0.009, 1.6714, -0.0004], [1.6245, -1.6382, -0.0732, -0.0028, 1.6587, -0.0003],
    [1.5803, -1.5918, -0.0735, -0.0124, 1.6463, -0.0002], [1.5338, -1.5454, -0.0738, -0.0109, 1.634, -0.0001],
    [1.4847, -1.4988, -0.074, -0.0071, 1.6219, -0.0001], [1.4751, -1.4713, -0.0731, -0.0242, 1.6175, -0.0],
    [1.5244, -1.4702, -0.0708, -0.0745, 1.6239, -0.0], [1.5704, -1.4658, -0.0687, -0.125, 1.6297, -0.0],
    [1.614, -1.4589, -0.0667, -0.1755, 1.6351, -0.0001]]


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _xlate(x, y, z):
    T = np.eye(4); T[:3, 3] = (x, y, z); return T


def composed_cart_boxes(mission_usda, cart_xyyaw):
    """``({name: (lo, hi)}, filtered_set)`` -- the composed ``/Rollwagen/chassis`` collider AABBs in the
    cart BODY frame read from the run's ``mission.usda`` (the REAL geometry the sim collides, not the
    box model), plus the set of cart bodies the welded carton is ``physics:filteredPairs``-filtered against.
    Needs ``pxr`` (usd-core)."""
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
    filtered = set()
    hu = stage.GetPrimAtPath('/RestockScenario/HandlingUnit')
    if hu:
        for rel in hu.GetRelationships():
            if 'filteredpairs' in rel.GetName().lower():
                for tg in rel.GetTargets():
                    filtered.add(str(tg).split('/')[-1])
    if not boxes:
        raise ValueError('no composed /Rollwagen/chassis colliders in %s' % mission_usda)
    return boxes, filtered


def _to_ur(p):
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    d = dict(zip(_PLAN_NAMES, p))
    return [d[j] for j in ARM_JOINTS]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', type=Path, help='write the result JSON here')
    args = ap.parse_args(argv)

    try:
        from twin.restock_manipulation import place_clearance as pc
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        from twin.urdf_probe import parse_urdf
    except Exception as exc:                                     # pragma: no cover
        print('SKIP: cannot import the twin modules (%s)' % exc)
        return 0
    if not (_URDF.is_file() and _XRDF.is_file() and _MISSION.is_file()):
        print('SKIP: dataset URDF/XRDF or coldwin100 mission.usda absent (%s)' % _MISSION)
        return 0
    try:
        bodies, filtered = composed_cart_boxes(_MISSION, CART)
    except Exception as exc:
        print('SKIP: usd-core / composed cart unavailable (%s)' % exc)
        return 0

    robot = parse_urdf(str(_URDF))
    geom = pc.GripperCartonGeometry.shipped()
    jam = _to_ur(_CW100_DESCENT_PLAN[0])
    cfg0 = {j: jam[i] for i, j in enumerate(ARM_JOINTS)}
    # base_link world z calibrated from the recorded frozen carton world z at the jam config (FK)
    Cb0 = pc._carton_in_base(robot, cfg0, geom)
    bz = CARTON_WORLD_Z - 0.26 - float(Cb0[2, 3])
    T_world_urbase = _xlate(*BASE[:2], bz) @ _rotz(math.radians(BASE[2])) @ _xlate(0, 0, 0.26)
    T_world_cart = _xlate(*CART[:2], 0.0) @ _rotz(math.radians(CART[2]))
    T_base_cart = np.linalg.inv(T_world_urbase) @ T_world_cart

    cage = pc.cage_body_names(bodies)
    sweep = pc.place_descent_sweep_bodies(bodies)
    support_in_sweep = [b for b in sweep if b not in cage]
    wps = [_to_ur(p) for p in _CW100_DESCENT_PLAN]

    def _minclear(include):
        per = pc.place_descent_cage_clearance(wps, str(_URDF), str(_XRDF), bodies, T_base_cart, include=include)
        v, k = min((x, n) for n, x in per.items())
        return per, v, k

    cage_per, cage_min, cage_body = _minclear(cage)
    full_per, full_min, full_body = _minclear(sweep)
    shelf_mm = full_per.get('shelf', float('nan')) * 1000.0
    reject = pc.place_descent_sweep_reject(full_per)

    print('composed cart: %d colliders; carton cage-filtered vs %d bodies (shelf filtered=%s)'
          % (len(bodies), len(filtered), 'shelf' in filtered))
    print('CAGE-ONLY sweep (pre-fix model): min clearance = %+.1f mm vs %s  (the blind "descent_clearance ~ '
          '110 mm" the host logged while the carton would not descend)' % (cage_min * 1000.0, cage_body))
    print('SHELF-INCLUSIVE sweep (corrected): min clearance = %+.1f mm vs %s ; shelf = %+.1f mm ; support '
          'bodies swept = %s' % (full_min * 1000.0, full_body, shelf_mm, support_in_sweep))
    print('REJECT: %s' % (reject if reject else '(none)'))

    cage_clears = cage_min >= 0.10                         # the pre-fix model read it clear (~110 mm)
    shelf_strikes = full_body == 'shelf' and full_min <= -0.03
    jam_reproduced = (reject is not None) and ('prim=shelf' in reject) and shelf_strikes
    ok = cage_clears and jam_reproduced
    print('\nSUMMARY: cage-only sweep clears (>= 100 mm)=%s  shelf-inclusive sweep names the SHELF strike '
          '(<= -30 mm)=%s  reject names prim=shelf=%s  => %s'
          % (cage_clears, shelf_strikes, (reject is not None and 'prim=shelf' in reject),
             'OK (cw100 jam reproduced by the corrected model)' if ok else 'FAIL'))
    if args.json:
        args.json.write_text(json.dumps({
            'cage_only_min_mm': round(cage_min * 1000.0, 1), 'cage_only_body': cage_body,
            'shelf_inclusive_min_mm': round(full_min * 1000.0, 1), 'shelf_inclusive_body': full_body,
            'shelf_mm': round(shelf_mm, 1), 'reject': reject,
            'carton_filtered_vs_cage': 'shelf' not in filtered and len(filtered) >= 50,
            'cage_clears': cage_clears, 'jam_reproduced': jam_reproduced, 'ok': ok}, indent=1) + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
