"""CPU home-leg simulation for the uncouple-at-parking return (sim-video-verify-return-v1, vr1).

After the robot releases the cart at ``mission_poses.cart_parking`` it drives HOME EMPTY with the real
uncoupled navigation controller (``nav_factory`` -> ``ArticulatedController`` -> the core grid planner +
braking/coverage gates) on LIVE lidar returns -- the first uncoupled leg of the mission that is not a
trusted-static drive (couple + the collect reposition legs are static). This module replays that leg on
CPU: the real controller, the real ``IsaacRangeSensor`` + ``observation`` (so the production scan wiring,
incl. the ``sensor_to_base`` the bundle passes, is exercised verbatim) over a vectorised box ray-caster of
the static map + the mission support + the transfer bay + the camera map solids + the PARKED cart (now a
real obstacle) + the robot's own body (self), and unicycle kinematics on the commanded (v, w).

It is an emulation (box geometry, ideal kinematics, no coverage noise), not physics: it proves the
planner/gates can route the configured home from the parking spot, and it localises a geometric refusal
(e.g. a phantom return in the goal disk) before a GPU run. No Isaac / USD imports.
"""
from __future__ import annotations

import math
import os
from types import SimpleNamespace as NS

import numpy as np

CART_L, CART_W, CART_H = 0.80, 0.68, 1.65       # MobileSpec cart_length / cart_width / cart_height
HITCH_TO_CART_M = 0.62 + 0.65                    # |hitch_x| + cart_hitch_x
LIDAR_XYZ = (0.52, 0.0, 0.15)                    # URDF lidar_frame origin (x, y, z) in base_link
NAV_ENV = {'RESTOCK_NAV_STALE_AFTER_S': '3.0', 'RESTOCK_NAV_BLOCKED_TIMEOUT_S': '40',
           'RESTOCK_NAV_GOAL_TIMEOUT_S': '1200'}   # the orchestrator's sim-clock nav bounds (run-356)


def static_boxes(environment, scenario, camera_config=None):
    """3-D AABBs ``(xmin, ymin, zmin, xmax, ymax, zmax, path)`` the lidar can hit, from the same inputs the
    navigation grid is built from (manifest static obstacles + staging support) plus the transfer bay and
    the collision camera solids parented to the map (real colliders in the stage)."""
    out = [(*o['min'], *o['max'], '/World/static/' + str(o.get('name', 'obstacle')))
           for o in environment['static_obstacles']]
    st = scenario.get('initial_staging') or {}
    if st:
        sx, sy = st['support_center_xy']
        dx, dy, dz = st['support_dimensions_m']
        out.append((sx - dx / 2, sy - dy / 2, 0.0, sx + dx / 2, sy + dy / 2, dz, '/RestockScenario/Support'))
    bay = scenario.get('transfer_bay')
    if bay:
        bx, by = bay['center_xy']
        bw, bd = bay['dimensions_m'][:2]
        out.append((bx - bw / 2, by - bd / 2, 0.0, bx + bw / 2, by + bd / 2, bay['top_z'],
                    '/RestockScenario/TransferBay'))
    for solid in (camera_config or {}).get('solids', []):
        if solid.get('parent') != 'map' or not solid.get('collision'):
            continue
        m = np.asarray(solid['parent_from_solid'], float)
        ext = np.abs(m[:3, :3]) @ (np.asarray(solid['dimensions_m'], float) / 2)
        lo, hi = m[:3, 3] - ext, m[:3, 3] + ext
        out.append((*lo, *hi, '/World/camera_solids/' + str(solid.get('name', 'solid'))))
    return out


def cart_boxes(cx, cy, cyaw):
    """The parked rollwagen as the lidar sees it (posts, mesh side/back walls, deck, drawbar, casters),
    authored like mobile_assets.author_cart. Axis-aligned approximation (the parking pose is straight)."""
    if abs(math.sin(cyaw)) > 0.2:
        raise ValueError('cart_boxes approximates a near-axis-aligned parked cart only')
    flip = -1.0 if math.cos(cyaw) < 0 else 1.0
    L, W, H, z0 = CART_L, CART_W, CART_H, 0.20
    out = []
    for px in (-L / 2 + 0.02, L / 2 - 0.02):
        for py in (-W / 2 + 0.02, W / 2 - 0.02):
            x, y = cx + flip * px, cy + py
            out.append((x - 0.0125, y - 0.0125, z0, x + 0.0125, y + 0.0125, H, '/Rollwagen/chassis/post'))
    for sgn in (-1, 1):
        y = cy + sgn * (W / 2 - 0.02)
        out.append((cx - L / 2, y - 0.006, z0, cx + L / 2, y + 0.006, H, '/Rollwagen/chassis/side'))
    xb = cx - flip * (L / 2 - 0.02)
    out.append((xb - 0.006, cy - W / 2, z0, xb + 0.006, cy + W / 2, H, '/Rollwagen/chassis/back'))
    out.append((cx - L / 2, cy - W / 2, z0 - 0.025, cx + L / 2, cy + W / 2, z0 + 0.025, '/Rollwagen/chassis/deck'))
    xa, xd = sorted((cx + flip * L / 2, cx + flip * 0.65))
    out.append((xa, cy - 0.02, 0.26, xd, cy + 0.02, 0.30, '/Rollwagen/chassis/drawbar'))
    for px in (-L / 2 + 0.10, L / 2 - 0.10):
        for py in (-W / 2 + 0.08, W / 2 - 0.08):
            x, y = cx + flip * px, cy + py
            out.append((x - 0.10, y - 0.02, 0.0, x + 0.05, y + 0.02, 0.15, '/Rollwagen/wheel'))
    return out


def robot_boxes(x, y, yaw):
    """The robot's own chassis + stowed arm (self returns -> unknown beams), kept clear of the lidar
    origin at the chassis front (x = 0.52 in base_link)."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for lx in np.linspace(-0.45, 0.45, 10):
        for ly in np.linspace(-0.34, 0.34, 8):
            wx, wy = x + c * lx - s * ly, y + s * lx + c * ly
            out.append((wx - 0.05, wy - 0.05, 0.0, wx + 0.05, wy + 0.05, 0.34, '/World/Robot/base_link'))
    for lx in np.linspace(-0.3, 0.3, 4):
        for ly in np.linspace(-0.3, 0.3, 4):
            wx, wy = x + c * lx - s * ly, y + s * lx + c * ly
            out.append((wx - 0.08, wy - 0.08, 0.34, wx + 0.08, wy + 0.08, 1.2, '/World/Robot/arm'))
    return out


class BoxRaycaster:
    """``raycast_closest`` (the PhysX scene-query call ``IsaacRangeSensor`` makes) over AABBs."""

    def __init__(self, static):
        self.static = np.array([b[:6] for b in static], float)
        self.static_paths = [b[6] for b in static]
        self.set_dynamic([])

    def set_dynamic(self, boxes):
        self.boxes = (np.vstack([self.static, np.array([b[:6] for b in boxes], float)])
                      if boxes else self.static)
        self.paths = self.static_paths + [b[6] for b in boxes]

    def raycast_closest(self, origin, direction, rmax):
        b = self.boxes
        o, d = np.asarray(origin, float), np.asarray(direction, float)
        par = np.abs(d) <= 1e-12
        dd = np.where(par, 1.0, d)
        t1, t2 = (b[:, :3] - o) / dd, (b[:, 3:] - o) / dd
        inside = (b[:, :3] <= o) & (o <= b[:, 3:])
        tlo = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
        thi = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
        tmin, tmax = np.maximum(tlo.max(axis=1), 0.0), np.minimum(thi.min(axis=1), rmax)
        ok = tmin <= tmax
        if not ok.any():
            return {'hit': False}
        idx = np.where(ok)[0][np.argmin(tmin[ok])]
        return {'hit': True, 'distance': float(tmin[idx]), 'collision': self.paths[idx],
                'rigidBody': self.paths[idx]}


def return_legs(config, *, home=None, backaway='config'):
    """The uncoupled goals the adapter drives after the uncouple: [cart_parking_backaway?, home]."""
    poses = config['mission_poses']
    back = poses.get('cart_parking_backaway') if backaway == 'config' else backaway
    return ([tuple(back)] if back else []) + [tuple(home or poses['home'])]


def simulate_home_leg(config, scenario, environment, *, start=None, home=None, backaway='config',
                      camera_config=None, lidar_yaw_fix=None, max_s=300.0, dt=1 / 60, scan_every=6,
                      beams=180):
    """Drive the uncoupled legs ``start`` (default: cart_parking) -> [backaway] -> ``home`` with the real
    controller, one ``nav.start`` per leg exactly as the adapter sequences them (each leg must latch
    ``arrived_stopped``). ``max_s`` bounds the WHOLE drive (the return stage budget).

    Returns ``{status, reason, t, pose, leg, legs, reasons, near_goal}``; ``near_goal`` lists live returns
    inside the goal's footprint disk at the first refusal (the phantom-return diagnosis)."""
    from twin.navigation.core import Odometry, wrap
    from twin.navigation.factory import nav_factory
    from twin.navigation.sensors import IsaacRangeSensor, observation as nav_observation
    poses = config['mission_poses']
    start = tuple(start or poses['cart_parking'])
    legs = return_legs(config, home=home, backaway=backaway)
    fix = bool(config.get('lidar_sensor_yaw_fix')) if lidar_yaw_fix is None else bool(lidar_yaw_fix)
    to_base = (LIDAR_XYZ[0], LIDAR_XYZ[1], 0.0) if fix else LIDAR_XYZ   # == isaac_bundle wiring
    nav_cfg = dict(config['navigation'], environment=environment, scenario=scenario)
    nav_cfg.pop('environment_path', None)
    nav_cfg.pop('scenario_path', None)
    saved = {k: os.environ.get(k) for k in NAV_ENV}
    os.environ.update(NAV_ENV)
    try:
        nav = nav_factory(app=None, assembly=object(), plan=NS(nav_mode='reference'),
                          config=dict(config, navigation=nav_cfg), output='.')
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    cart = (start[0] - HITCH_TO_CART_M * math.cos(start[2]), start[1] - HITCH_TO_CART_M * math.sin(start[2]),
            start[2])
    query = BoxRaycaster(static_boxes(environment, scenario, camera_config))
    state = {'pose': list(start)}

    def lidar_pose():
        x, y, yaw = state['pose']
        return (x + LIDAR_XYZ[0] * math.cos(yaw), y + LIDAR_XYZ[0] * math.sin(yaw), yaw)

    sensor = IsaacRangeSensor(lidar_pose, sensor_to_base=to_base, self_paths=('/World/Robot',),
                              coupled_self_paths=('/Rollwagen',), query=query, beams=beams)
    t, v, w, obs, reasons, near_goal = 0.0, 0.0, 0.0, None, {}, None
    radius = nav.footprint.radius(False) + 0.05
    leg = 0
    nav.start(legs[leg], False)
    out = {'lidar_yaw_fix': fix, 'legs': [list(g) for g in legs], 'home': list(legs[-1])}
    for i in range(int(max_s / dt)):
        t += dt
        x, y, yaw = state['pose']
        odo = Odometry((x, y, yaw), t, v, w, position_uncertainty=0.05)
        if obs is None or i % scan_every == 0:
            query.set_dynamic(cart_boxes(*cart) + robot_boxes(x, y, yaw))
            obs = nav_observation(sensor.read(t), odo, coupled=False, arm_transport=True)
        r = nav.step(dt, odo, obs, now=t)
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
        if r.reason == 'blocked_no_route' and near_goal is None:
            goal = legs[leg]
            near_goal = sorted((round(math.dist(p, goal[:2]), 3), round(p[0], 3), round(p[1], 3))
                               for p in obs.obstacles if math.dist(p, goal[:2]) <= radius)[:5]
        if r.status == 'succeeded' and leg < len(legs) - 1:
            leg += 1                                  # next leg from the measured arrival pose (adapter)
            nav.start(legs[leg], False)
            v = w = 0.0
            continue
        if r.status in ('failed', 'succeeded'):
            return dict(out, status=r.status, reason=r.reason, t=round(t, 1), leg=leg,
                        pose=[round(c, 3) for c in state['pose']], reasons=reasons, near_goal=near_goal or [])
        v, w = r.linear, r.angular
        state['pose'] = [x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, wrap(yaw + w * dt)]
    return dict(out, status='timeout', reason='sim budget', t=round(t, 1), leg=leg,
                pose=[round(c, 3) for c in state['pose']], reasons=reasons, near_goal=near_goal or [])


def lidar_wall_probe(sensor_to_base, *, wall_x=3.0):
    """The lidar wiring proof: a wall at x=wall_x seen from a base at the origin. Returns the obstacle x
    span; the correct wiring reproduces the wall exactly (span 0)."""
    from twin.navigation.core import Odometry
    from twin.navigation.sensors import IsaacRangeSensor, observation as nav_observation
    query = BoxRaycaster([(wall_x, -5.0, 0.0, wall_x + 0.1, 5.0, 2.0, '/World/wall')])
    sensor = IsaacRangeSensor(lambda: (LIDAR_XYZ[0], LIDAR_XYZ[1], 0.0), sensor_to_base=sensor_to_base,
                              query=query, beams=360)
    obs = nav_observation(sensor.read(1.0), Odometry((0.0, 0.0, 0.0), 1.0), coupled=False, arm_transport=True)
    xs = [p[0] for p in obs.obstacles]
    return (round(min(xs), 3), round(max(xs), 3))


def check_home_leg(config, scenario, environment, rep, *, camera_config=None):
    """Report rows: the lidar wiring proof, then the configured home leg as the run would wire it (and,
    for the decision, the other lidar wiring)."""
    legacy = lidar_wall_probe(LIDAR_XYZ)
    fixed = lidar_wall_probe((LIDAR_XYZ[0], LIDAR_XYZ[1], 0.0))
    rep.add('PASS' if fixed == (3.0, 3.0) else 'FAIL', 'lidar_yaw_fix_reproduces_wall', fixed_span=fixed)
    rep.add('WARN' if not config.get('lidar_sensor_yaw_fix') else 'PASS', 'lidar_legacy_wiring_rotates_returns',
            legacy_span=legacy, note='legacy sensor_to_base=(x,y,z) rotates every return 0.15 rad; set '
                                     '"lidar_sensor_yaw_fix": true to use yaw 0')
    configured = simulate_home_leg(config, scenario, environment, camera_config=camera_config)
    rep.check(configured['status'] == 'succeeded', 'home_leg_configured_home_reached', **configured)
    other = simulate_home_leg(config, scenario, environment, camera_config=camera_config,
                              lidar_yaw_fix=not configured['lidar_yaw_fix'])
    rep.add('INFO', 'home_leg_other_lidar_wiring', **other)
    return configured
