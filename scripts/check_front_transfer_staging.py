#!/usr/bin/env python3
"""CPU proof for the SALES-FLOOR (front) scripted transfer: replay the warm8 bay transfer at endcap G3_back_s2.

The fm459 backroom bay was authored as a replica of the sales-floor slot relative to the parked robot (bay
top z 0.73 == G3_back_s2 surface; bay 1.17 m to the robot's left). So the recorded bay handover, rotated by
pi and moved onto the endcap target, is a staging the scripted transfer already solves. This script:

* maps the recorded bay park (``shortcut.RECORDED_BAY_PARK_ROBOT_POSE``, warm8's measured carton on the
  rollwagen) onto the endcap and prints the ``start_staging.robot_base_pose`` the integration lane sets;
* runs ``ScriptedTransfer`` (still arm, the warm8 mode) for both place approaches and sweeps the carried
  carton's box against the endcap's shelf boards / s3 products (geometry from planogram.json, the same
  numbers build_supermarket.py builds from);
* exits 0 only when ``place_approach=front`` succeeds level, lands exactly on the target and never
  intersects a board -- the precondition for spending a GPU run on it. (Its plan is task-space only --
  ``ik_complete`` False -- so it is for the still arm, warm8's mode; see ``plan_transfer``.)

No Isaac, no GPU. Example:

  python3 scripts/check_front_transfer_staging.py \\
      --urdf ~/datasets/twin-robot-in-store-boot-v1/mission/warm8/mobile-import.urdf
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin.restock import scripted_transfer as st                      # noqa: E402
from twin.restock.runtime import Packet                               # noqa: E402
from twin.restock.shortcut import RECORDED_BAY_PARK_ROBOT_POSE        # noqa: E402

DEFAULT_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mission/warm8/mobile-import.urdf'
PLANOGRAM = Path(__file__).resolve().parent / 'planogram.json'
STOW = np.array([0., -2.5, 1.7, -0.7, -1.5708, 0.])
DIMS = (0.6, 0.4, 0.16)
UR_BASE_Z = 0.50
BAY_TARGET = (-2.0, 16.424, 0.73)              # warm8 transfer_ground_truth target (the bay top)
BAY_CARTON = (-3.162, 15.267, 1.013, -math.pi / 2)   # warm8 SCRIPTED_TRANSFER begin carton_world (measured)
ENDCAP, SLOT = 'G3_back', 2                    # scenario initial_staging.target_surface_pose_world


def endcap_geometry(planogram=PLANOGRAM, endcap_id=ENDCAP):
    """Board slabs + the s2 slot of an endcap, as build_supermarket.build_gondolas places them."""
    plan = json.loads(Path(planogram).read_text())
    fix = plan['fixture']
    ec = next(e for e in plan['endcaps'] if e['id'] == endcap_id)
    g = next(gg for gg in plan['gondolas'] if endcap_id.startswith(gg['id'] + '_'))
    if ec['facing'] != '+Y':
        raise ValueError('only the +Y-facing endcap layout is modelled here')
    run_len = g['n_bays'] * plan['bay_w']
    y_back = g['y0'] + run_len + 0.03 + fix['back_t'] / 2
    sy = y_back + fix['back_t'] / 2                     # shelf back edge
    half_w = (ec['w'] - fix['upright_w'] - 0.01) / 2
    boards = []
    for i, sh in enumerate(ec['shelves']):
        if i == 0:
            continue                                    # level 0 is the base deck
        front = sy + sh['depth']
        boards.append(dict(name=f's{i}', lo=(ec['x'] - half_w, sy, sh['z'] - fix['shelf_t']),
                           hi=(ec['x'] + half_w, front, sh['z'])))
        boards.append(dict(name=f's{i}_rail', lo=(ec['x'] - half_w, front, sh['z'] - 0.036),     # ticket rail
                           hi=(ec['x'] + half_w, front + 0.006, sh['z'] + 0.006)))
    nxt = ec['shelves'][SLOT + 1]
    products = dict(name=f's{SLOT + 1}_products', lo=(ec['x'] - half_w, sy, nxt['z']),
                    hi=(ec['x'] + half_w, sy + nxt['depth'], ec['shelves'][SLOT + 2]['z'] - fix['shelf_t']))
    slot = ec['shelves'][SLOT]
    target = (ec['x'], sy + 0.2 + 0.056, slot['z'])     # carton (0.4 deep) flush at the front, as the scenario
    return boards, products, target


def rigid_map(bay_target, shelf_target, yaw=math.pi):
    """World->world map taking the bay target onto the shelf target, rotated by ``yaw`` about z."""
    R = st.pose_T((0., 0., 0.), st.yaw_quat(yaw))
    p = np.asarray(shelf_target, float) - (R @ np.append(np.asarray(bay_target, float), 1.0))[:3]
    return st.pose_T((p[0], p[1], 0.0), st.yaw_quat(yaw))


class _Driver:
    def __init__(self, ik, base, carton):
        self.ik, self.base, self.c, self.q, self.trace = ik, base, carton.copy(), STOW.copy(), []

    def base_world_T(self):
        return self.base

    def carton_world_T(self):
        return self.c

    def tool_world_T(self):
        return self.base @ self.ik.fk(self.q)

    def arm_joint_positions(self):
        return self.q.copy()

    def attach_carton(self):
        pass

    def set_carton_world_T(self, T):
        self.c = np.asarray(T, float)
        self.trace.append(self.c.copy())

    def place_carton(self, T):
        self.c = np.asarray(T, float)


def run_transfer(urdf, base, carton, target, place_approach):
    ik = st.ArmIK(urdf)
    driver = _Driver(ik, base, carton)
    runner = st.ScriptedTransfer(driver, ik=ik, target_xyz_world=tuple(target), dims=DIMS, location='G3_back_s2',
                                 sku='juice_orange', quantity=1, place_approach=place_approach)
    plan = runner.begin()
    phases, result = [], None
    for i in range(20000):
        result = runner.step(1 / 30, Packet(f'physics:{i}', 'sensor', i / 30, None, None, None, None,
                                            True, True, 0., True))
        phases.append(runner.phase)
        if result.status != 'running':
            break
    return plan, driver, result, phases


def _box(T):
    hx, hy, hz = (v / 2 for v in DIMS)
    pts = np.array([[sx * hx, sy * hy, sz * hz, 1.] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    c = (T @ pts.T).T[:, :3]
    return c.min(0), c.max(0)


def _overlap(lo, hi, blo, bhi, tol=1e-3):
    return all(lo[i] < bhi[i] - tol and hi[i] > blo[i] + tol for i in range(3))


def intersections(trace, phases, boards, products):
    """[(frame, slab, phase)] where the carried carton's box enters a board / the upper-shelf products."""
    hits = []
    for k, T in enumerate(trace):
        lo, hi = _box(T)
        for slab in boards + [products]:
            if _overlap(lo, hi, slab['lo'], slab['hi']):
                hits.append((k, slab['name'], phases[min(k, len(phases) - 1)]))
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--urdf', type=Path, default=DEFAULT_URDF, help='mission mobile-import.urdf (read-only)')
    args = ap.parse_args(argv)
    boards, products, endcap_target = endcap_geometry()
    m = rigid_map(BAY_TARGET, endcap_target)
    bx, by, byaw = RECORDED_BAY_PARK_ROBOT_POSE
    base = m @ st.pose_T((bx, by, UR_BASE_Z), st.yaw_quat(byaw))
    cx, cy, cz, cyaw = BAY_CARTON
    carton = m @ st.pose_T((cx, cy, cz), st.yaw_quat(cyaw))
    staging = [round(float(base[0, 3]), 4), round(float(base[1, 3]), 4), round(st.T_yaw(base), 4)]
    print(f'FRONT_STAGING start_staging.robot_base_pose={staging} '
          f'transfer_ground_truth.target={[round(v, 4) for v in endcap_target]} '
          f'carton_on_cart={[round(float(v), 3) for v in carton[:3, 3]]}')
    ok = True
    for approach in (st.PLACE_ABOVE, st.PLACE_FRONT):
        plan, driver, result, phases = run_transfer(args.urdf, base, carton, endcap_target, approach)
        tilt = max(math.degrees(math.acos(min(1.0, abs(T[2, 2])))) for T in driver.trace)
        final = driver.c[:3, 3]
        err = float(np.linalg.norm(final[:2] - np.asarray(endcap_target[:2])))
        hits = intersections(driver.trace, phases, boards, products)
        by_slab = {}
        for k, name, phase in hits:
            by_slab.setdefault((name, phase), []).append(k)
        detail = '; '.join(f'{n}@{p} frames {v[0]}..{v[-1]}' for (n, p), v in by_slab.items()) or 'none'
        print(f'FRONT_TRANSFER place={approach} result={result.status} ik_complete={plan.ik_complete} '
              f'reach={plan.choice["worst_station_reach"]:.3f} frames={len(driver.trace)} '
              f'max_tilt_deg={tilt:.2f} xy_err_m={err:.4f} board_hits={len(hits)} ({detail})')
        if approach == st.PLACE_FRONT:
            ok = result.status == 'succeeded' and not hits and tilt < 1.0 and err < 1e-6
    print('FRONT_TRANSFER verdict:', 'PASS (place=front is clean at the endcap)' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
