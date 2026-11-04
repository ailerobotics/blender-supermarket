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
import sys
import time
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


# Within this band of |sin(cyaw)| the cart is treated as axis-aligned (the historical approximation); the
# parking pose used to be straight (yaw ~ 0 or ~ pi). PR 271 re-heads the cart SOUTH on the portal axis
# (cart_parking yaw = +pi/2), so a parked cart can now sit at an arbitrary yaw -- handled by rotating the
# canonical part boxes below. The band is kept so results stay byte-identical for the old near-axis poses.
_AXIS_ALIGNED_SIN_TOL = 0.2


def cart_boxes(cx, cy, cyaw):
    """The parked rollwagen as the lidar sees it (posts, mesh side/back walls, deck, drawbar, casters),
    authored like mobile_assets.author_cart, as 3-D AABBs at parking yaw ``cyaw``.

    For a near-axis-aligned pose (``|sin(cyaw)| <= 0.2``) this is the historical axis-aligned approximation
    (byte-identical, so every old near-axis call is unchanged); flip semantics (``cos(cyaw) < 0`` reflects
    the asymmetric back wall / drawbar to the far end, as a y-symmetric cart's 180 deg rotation does). For
    any other yaw (e.g. the PR 271 south-headed ``+pi/2`` parking pose) the canonical (flip = 1) part boxes
    are rigidly rotated about the cart centre and returned as the conservative axis-aligned bounds of the
    rotated corners -- so the boxes always CONTAIN the true rotated footprint and are never smaller than the
    axis-aligned approximation."""
    if abs(math.sin(cyaw)) <= _AXIS_ALIGNED_SIN_TOL:
        return _cart_boxes_axis_aligned(cx, cy, cyaw)
    return _cart_boxes_rotated(cx, cy, cyaw)


def _cart_boxes_axis_aligned(cx, cy, cyaw):
    """The historical near-axis-aligned approximation, verbatim: the cart as axis-aligned AABBs with the
    only yaw handling a 180 deg ``flip`` of the asymmetric back wall / drawbar."""
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


def _cart_local_boxes():
    """The canonical (parked yaw = 0, flip = 1) rollwagen part boxes in the cart-centre frame, each
    ``(lxmin, lymin, zmin, lxmax, lymax, zmax, path)`` -- the same geometry the axis-aligned path lays down
    at the origin, with the asymmetric back wall at the -x end and the drawbar protruding at the +x end.
    The rotated path rotates these about the centre; a true 180 deg rotation reproduces the flip layout
    (the cart is symmetric about its x-axis)."""
    L, W, H, z0 = CART_L, CART_W, CART_H, 0.20
    out = []
    for px in (-L / 2 + 0.02, L / 2 - 0.02):
        for py in (-W / 2 + 0.02, W / 2 - 0.02):
            out.append((px - 0.0125, py - 0.0125, z0, px + 0.0125, py + 0.0125, H, '/Rollwagen/chassis/post'))
    for sgn in (-1, 1):
        y = sgn * (W / 2 - 0.02)
        out.append((-L / 2, y - 0.006, z0, L / 2, y + 0.006, H, '/Rollwagen/chassis/side'))
    xb = -(L / 2 - 0.02)
    out.append((xb - 0.006, -W / 2, z0, xb + 0.006, W / 2, H, '/Rollwagen/chassis/back'))
    out.append((-L / 2, -W / 2, z0 - 0.025, L / 2, W / 2, z0 + 0.025, '/Rollwagen/chassis/deck'))
    dx0, dx1 = sorted((L / 2, 0.65))
    out.append((dx0, -0.02, 0.26, dx1, 0.02, 0.30, '/Rollwagen/chassis/drawbar'))
    for px in (-L / 2 + 0.10, L / 2 - 0.10):
        for py in (-W / 2 + 0.08, W / 2 - 0.08):
            out.append((px - 0.10, py - 0.02, 0.0, px + 0.05, py + 0.02, 0.15, '/Rollwagen/wheel'))
    return out


def _cart_boxes_rotated(cx, cy, cyaw):
    """A parked cart at arbitrary yaw: rotate the canonical part boxes about the centre by ``cyaw`` and
    return the conservative axis-aligned bounds of the rotated xy corners (heights unchanged). Each AABB
    contains its rotated box, so the union contains the true rotated footprint; always >= the canonical
    extent."""
    c, s = math.cos(cyaw), math.sin(cyaw)
    out = []
    for lx0, ly0, zlo, lx1, ly1, zhi, path in _cart_local_boxes():
        xs, ys = [], []
        for lx in (lx0, lx1):
            for ly in (ly0, ly1):
                xs.append(cx + c * lx - s * ly)
                ys.append(cy + s * lx + c * ly)
        out.append((min(xs), min(ys), zlo, max(xs), max(ys), zhi, path))
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
    """``raycast_closest`` (the PhysX scene-query call ``IsaacRangeSensor`` makes) over AABBs.

    The slab intersection below is unchanged; the only addition is a per-origin candidate cull so a
    horizontal beam of length ``rmax`` tests the handful of boxes it could actually reach rather than the
    whole store map (~900 boxes). A box survives the cull iff its z-span straddles the ray height AND its
    nearest xy point is within ``rmax`` -- both are the SAME conditions a real hit must satisfy (see
    ``_candidates``), so the cull is an exact superset and the returned hit is bit-identical to the full
    scan. The lidar reuses one origin (x, y, z) across all 180 bearings at a height, so the cull is
    memoised per origin and cleared whenever the dynamic boxes change (``set_dynamic``)."""

    def __init__(self, static):
        arr = np.array([b[:6] for b in static], float) if static else np.empty((0, 6), float)
        self.static_lo, self.static_hi = arr[:, :3].copy(), arr[:, 3:].copy()
        self.static_paths = [b[6] for b in static]
        self.set_dynamic([])

    def set_dynamic(self, boxes):
        if boxes:
            d = np.array([b[:6] for b in boxes], float)
            self.lo = np.vstack([self.static_lo, d[:, :3]])
            self.hi = np.vstack([self.static_hi, d[:, 3:]])
            self.paths = self.static_paths + [b[6] for b in boxes]
        else:
            self.lo, self.hi, self.paths = self.static_lo, self.static_hi, self.static_paths
        self._cull = {}
        self._read_cache = {}

    def prime_read(self, x, y, yaw, heights, beams, rmax):
        """Precompute every ``(height, bearing)`` closest-hit a ``IsaacRangeSensor.read`` from this pose
        will request, in a handful of batched numpy ops instead of ``beams * heights`` per-beam calls, and
        memoise them keyed exactly as ``read`` queries (same ``yaw - pi + i * 2pi/beams`` bearings). It is a
        pure speed cache: ``raycast_closest`` falls back to the identical per-beam path on any key it does
        not find, so a drift in the sensor's bearing formula only slows the cache, never changes a result."""
        inc = 2 * math.pi / beams
        dirs = [(math.cos(yaw - math.pi + i * inc), math.sin(yaw - math.pi + i * inc)) for i in range(beams)]
        # Batch only the non-axis-aligned bearings (both components clear of zero): for them the slab
        # bounds are a plain min/max of the division endpoints -- the bit-identical fast path of
        # raycast_closest. The handful of exactly-axis-aligned bearings are simply not cached, so
        # raycast_closest serves them through its parallel-ray branch (identical result, rare).
        dirs = [d for d in dirs if (d[0] > 1e-12 or d[0] < -1e-12) and (d[1] > 1e-12 or d[1] < -1e-12)]
        if not dirs:
            return
        D = np.array(dirs, float)                                  # (beams, 2)
        rows = np.arange(len(dirs))
        invx, invy = D[:, 0][:, None], D[:, 1][:, None]            # (beams, 1), divide (not reciprocal)
        for z in heights:
            lo_o, hi_o, _inside, idx = self._candidates(x, y, z, rmax)
            if len(idx) == 0:
                miss = {'hit': False}
                for dx, dy in dirs:
                    self._read_cache[(x, y, z, dx, dy, rmax)] = miss
                continue
            # Per-axis slab bounds as 2-D (beams, M) blocks -- same divisions as the 3-D form, but without
            # the (beams, M, 2) temporaries or the axis-2 reduction. tmin/tmax are the across-axis
            # max-of-mins / min-of-maxes, exactly as raycast_closest computes them.
            tx1 = lo_o[:, 0][None, :] / invx
            tx2 = hi_o[:, 0][None, :] / invx
            ty1 = lo_o[:, 1][None, :] / invy
            ty2 = hi_o[:, 1][None, :] / invy
            tmin = np.maximum(np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2)), 0.0)
            tmax = np.minimum(np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2)), rmax)
            cand = np.where(tmin <= tmax, tmin, np.inf)
            j = np.argmin(cand, axis=1)                             # closest box per bearing
            best = cand[rows, j]
            for b, (dx, dy) in enumerate(dirs):
                if np.isinf(best[b]):
                    self._read_cache[(x, y, z, dx, dy, rmax)] = {'hit': False}
                else:
                    p = self.paths[idx[j[b]]]
                    self._read_cache[(x, y, z, dx, dy, rmax)] = {
                        'hit': True, 'distance': float(best[b]), 'collision': p, 'rigidBody': p}

    def _candidates(self, ox, oy, oz, rmax):
        """The candidate boxes for an origin, plus the slab terms that depend only on the origin (not the
        beam direction) so the 180 bearings cast from here reuse them. Because every candidate straddles
        the ray height ``oz`` (dz=0 for a horizontal ray), the z slab never constrains the hit, so only the
        x/y slabs are kept -- the result is identical to the full 3-D test on all boxes."""
        key = (ox, oy, oz, rmax)
        c = self._cull.get(key)
        if c is None:
            lo, hi, n = self.lo, self.hi, len(self.lo)
            zero = np.zeros(n)
            dx = np.maximum.reduce([lo[:, 0] - ox, zero, ox - hi[:, 0]])   # xy distance to the box AABB
            dy = np.maximum.reduce([lo[:, 1] - oy, zero, oy - hi[:, 1]])
            mask = ((lo[:, 2] <= oz) & (oz <= hi[:, 2])               # ray height inside the box z-span
                    & (dx * dx + dy * dy <= rmax * rmax + 1e-9))      # nearest xy point within reach
            idx = np.nonzero(mask)[0]
            o_xy = np.array([ox, oy])
            lo_o = lo[idx, :2] - o_xy          # box-min relative to the origin (direction-independent)
            hi_o = hi[idx, :2] - o_xy
            inside = (lo[idx, :2] <= o_xy) & (o_xy <= hi[idx, :2])
            c = (lo_o, hi_o, inside, idx)
            self._cull[key] = c
        return c

    def raycast_closest(self, origin, direction, rmax):
        cached = self._read_cache.get((origin[0], origin[1], origin[2], direction[0], direction[1], rmax))
        if cached is not None:
            return cached
        lo_o, hi_o, inside, idx = self._candidates(origin[0], origin[1], origin[2], rmax)
        if len(idx) == 0:
            return {'hit': False}
        dx, dy = direction[0], direction[1]
        if dx > 1e-12 or dx < -1e-12:
            if dy > 1e-12 or dy < -1e-12:
                # Fast path: neither component is parallel, so the slab bounds are just min/max of the
                # two division endpoints (``par`` all-False collapses the np.where branches away). This is
                # the bit-identical result of the general path for a non-axis-aligned beam.
                t1, t2 = lo_o / (dx, dy), hi_o / (dx, dy)
                tlo, thi = np.minimum(t1, t2), np.maximum(t1, t2)
                tmin, tmax = np.maximum(tlo.max(axis=1), 0.0), np.minimum(thi.min(axis=1), rmax)
                ok = tmin <= tmax
                if not ok.any():
                    return {'hit': False}
                j = np.where(ok)[0][np.argmin(tmin[ok])]
                path = self.paths[idx[j]]
                return {'hit': True, 'distance': float(tmin[j]), 'collision': path, 'rigidBody': path}
        d = np.array((dx, dy), float)
        par = np.abs(d) <= 1e-12
        dd = np.where(par, 1.0, d)
        t1, t2 = lo_o / dd, hi_o / dd
        tlo = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
        thi = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
        tmin, tmax = np.maximum(tlo.max(axis=1), 0.0), np.minimum(thi.min(axis=1), rmax)
        ok = tmin <= tmax
        if not ok.any():
            return {'hit': False}
        j = np.where(ok)[0][np.argmin(tmin[ok])]
        path = self.paths[idx[j]]
        return {'hit': True, 'distance': float(tmin[j]), 'collision': path, 'rigidBody': path}


def return_legs(config, *, home=None, backaway='config'):
    """The uncoupled goals the adapter drives after the uncouple: [cart_parking_backaway?, home]."""
    poses = config['mission_poses']
    back = poses.get('cart_parking_backaway') if backaway == 'config' else backaway
    return ([tuple(back)] if back else []) + [tuple(home or poses['home'])]


def simulate_home_leg(config, scenario, environment, *, start=None, home=None, backaway='config',
                      camera_config=None, lidar_yaw_fix=None, max_s=300.0, dt=1 / 60, scan_every=6,
                      beams=180, heartbeat_s=3.0):
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
    fix = bool(config.get('lidar_sensor_yaw_fix', True)) if lidar_yaw_fix is None else bool(lidar_yaw_fix)
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
    cart_b = cart_boxes(*cart)                      # the parked cart is fixed: its boxes never change
    wall0 = last_beat = time.perf_counter()
    if heartbeat_s:
        print(f'[tail_nav] home leg (lidar_yaw_fix={fix}) start -> {len(legs)} goal(s), '
              f'budget {max_s:.0f}s sim', file=sys.stderr, flush=True)
    for i in range(int(max_s / dt)):
        t += dt
        x, y, yaw = state['pose']
        odo = Odometry((x, y, yaw), t, v, w, position_uncertainty=0.05)
        if obs is None or i % scan_every == 0:
            query.set_dynamic(cart_b + robot_boxes(x, y, yaw))
            lx, ly, lyaw = sensor.get_pose()      # the exact pose read() casts from
            query.prime_read(lx, ly, lyaw, sensor.heights, sensor.beams, sensor.range_max)
            obs = nav_observation(sensor.read(t), odo, coupled=False, arm_transport=True)
        if heartbeat_s and (now_wall := time.perf_counter()) - last_beat >= heartbeat_s:
            last_beat = now_wall
            print(f'[tail_nav] home leg lidar_yaw_fix={fix} leg {leg + 1}/{len(legs)} '
                  f'sim t={t:.1f}/{max_s:.0f}s pose=({x:.2f},{y:.2f}) '
                  f'wall={now_wall - wall0:.1f}s', file=sys.stderr, flush=True)
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


# warm3 (sim-video, branch fm/twin-robot-in-store-boot-v1) recorded END-OF-VERIFY train pose, read from
# the run's nav-trace.jsonl first RETURN tick (tick 4846): base (x, y, yaw) and the towed cart's
# articulation (base_yaw - cart_yaw). This is the measured pose from which the live coupled RETURN stalled
# -- the cart lags at ~0.2 rad after the transfer, so the authored return lane (start_cart_yaw 0) is
# tracked with cart_err ~0.23 and the follower drops its own validated route every tick unless
# navigation.tow_lane_cart_yaw_rebase is set. Any regression that drops that key reproduces the stall here.
RECORDED_END_OF_VERIFY_BASE = (-2.078, 15.239, math.radians(-1.7))
RECORDED_END_OF_VERIFY_ARTICULATION = 0.2001


def simulate_coupled_return(config, scenario, environment, *, start_pose=RECORDED_END_OF_VERIFY_BASE,
                            start_articulation=RECORDED_END_OF_VERIFY_ARTICULATION, goal=None,
                            camera_config=None, max_s=30.0, dt=1 / 60):
    """Drive the REAL COUPLED (robot+cart) ``ArticulatedController`` along the mission's return goal from a
    measured end-of-verify train pose, built through ``nav_factory`` so the controller reads the mission's
    actual ``navigation.*`` keys verbatim -- including ``tow_lane_cart_yaw_rebase``. If that key is dropped
    (or set false) the follower hits the cart-yaw ``stop_replan_tracking_error`` trap (cart_err ~0.23 >
    ``tow_lane_cart_yaw_tol`` because the cart lags at articulation ~0.2 while the authored lane starts at
    cart_yaw 0) and the drive never arrives -- so any caller asserting arrival fails, closing the gap that
    ``check_mission_tail`` left by stubbing the coupled return. Unlike ``simulate_home_leg`` this is coupled
    and follows the resolved return lane; the goal defaults to ``mission_poses.cart_parking``.

    Returns ``{status, reason, arrived, reasons, pose, moved, lane, goal, articulation}``. ``arrived`` is
    True iff the controller latched ``arrived_stopped`` at the goal.
    """
    from twin.navigation.core import Odometry, Observation, wrap
    from twin.navigation.factory import nav_factory
    from twin.navigation.articulated import TrainState, advance
    from twin.mobile import MobileSpec
    poses = config['mission_poses']
    goal = tuple(goal if goal is not None else poses['cart_parking'])
    ga = (config.get('goal_articulation') or {}).get('return')
    nav_cfg = dict(config['navigation'], environment=environment)
    nav_cfg.pop('environment_path', None)
    nav_cfg.pop('scenario_path', None)
    if scenario is not None:
        nav_cfg['scenario'] = scenario
    else:
        nav_cfg.pop('scenario', None)   # optional; a present-but-None key fails _load
    if camera_config is not None:
        nav_cfg['camera_config'] = camera_config
    nav_cfg.pop('camera_config_path', None)
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
    spec = MobileSpec()
    nav.start(goal, True, goal_articulation=ga)
    lane = getattr(nav, '_active_lane', None)
    pose = list(start_pose)
    art = float(start_articulation)
    cyaw = wrap(pose[2] - art)
    v = w = 0.0
    reasons = {}
    result = None
    for _ in range(int(max_s / dt)):
        od = Odometry(tuple(pose), (_ + 1) * dt, v, w)          # production source defaults -> accepted
        ob = Observation((_ + 1) * dt, (), True, True, art, coverage_radius=10.0)
        r = nav.step(dt, od, ob, now=(_ + 1) * dt)
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
        if r.status in ('succeeded', 'failed'):
            result = r
            break
        v, w = r.linear, r.angular
        x, y, yaw = pose
        ts = TrainState(x, y, yaw, cyaw)
        k = w / v if abs(v) > 1e-6 else 0.0
        ns = advance(ts, v * dt, k, spec)
        pose = [x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, wrap(yaw + w * dt)]
        cyaw = ns.cart_yaw
        art = abs(wrap(pose[2] - cyaw))
    arrived = bool(result and result.status == 'succeeded' and result.reason == 'arrived_stopped')
    return {'status': result.status if result else 'timeout',
            'reason': result.reason if result else 'sim budget',
            'arrived': arrived, 'reasons': reasons,
            'pose': [round(c, 3) for c in pose], 'moved': round(math.dist(pose[:2], start_pose[:2]), 3),
            'lane': lane.name if lane is not None else None,
            'goal': list(goal), 'articulation': round(start_articulation, 4)}


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
    # The fix is ON by default now (vr1 / sim-restock-lidar-yaw-fix-eval); WARN only when a config explicitly
    # selects the legacy rotated wiring with "lidar_sensor_yaw_fix": false.
    rep.add('WARN' if config.get('lidar_sensor_yaw_fix', True) is False else 'PASS',
            'lidar_legacy_wiring_rotates_returns',
            legacy_span=legacy, note='legacy sensor_to_base=(x,y,z) rotates every return 0.15 rad; the fix '
                                     '(yaw 0) is the default -- drop "lidar_sensor_yaw_fix": false to use it')
    # sim-cart-side-bay-v1: the side-bay return is a SINGLE uncoupled leg from the service area
    # (shelf_approach, where the base finished the transfer) straight HOME -- the cart stays parked in
    # the bay, so there is no cart_parking tow and no cart_parking_backaway. The legacy uncouple-at-parking
    # return keeps its default start=cart_parking -> [backaway] -> home.
    poses = config.get('mission_poses', {})
    side_bay = bool(config.get('cart_side_bay')) and 'cart_side_bay' in poses
    leg_kw = ({'start': poses.get('shelf_approach'), 'backaway': None} if side_bay else {})
    configured = simulate_home_leg(config, scenario, environment, camera_config=camera_config, **leg_kw)
    rep.check(configured['status'] == 'succeeded', 'home_leg_configured_home_reached', **configured)
    other = simulate_home_leg(config, scenario, environment, camera_config=camera_config,
                              lidar_yaw_fix=not configured['lidar_yaw_fix'], **leg_kw)
    rep.add('INFO', 'home_leg_other_lidar_wiring', **other)
    return configured


def check_return_tracking(config, scenario, environment, rep, *, camera_config=None):
    """Exercise the COUPLED return's cart-yaw tracking with the REAL controller instead of the kinematic
    stub ``check_tail`` uses. Drives ``simulate_coupled_return`` from the recorded end-of-verify train pose
    (cart lagging at articulation ~0.2) to ``mission_poses.cart_parking`` using the config's actual
    ``navigation.*`` keys. With ``tow_lane_cart_yaw_rebase`` set (and/or a wide enough ``tow_lane_cart_yaw_tol``)
    it reaches ``arrived_stopped``; if the runtime config regresses that key back to the warm3 state the
    follower drops its own validated route on cart_err and this check FAILS -- which the stub could never
    catch. No-op (INFO skip) when the coupled return resolves to no authored lane (e.g. a no-tow config)."""
    result = simulate_coupled_return(config, scenario, environment, camera_config=camera_config, max_s=30.0)
    if result['lane'] is None:
        rep.add('INFO', 'return_tracking_no_authored_lane', **result,
                note='coupled return resolves to no authored tow lane (free-plan); cart-yaw lane tracking N/A')
        return result
    rep.check(result['arrived'], 'return_tracking_reaches_cart_parking', **result)
    return result
