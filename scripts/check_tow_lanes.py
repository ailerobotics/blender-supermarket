#!/usr/bin/env python3
"""Offline proof (CPU-only) for the two sim-restock-nav-lanes-v1 deliverables, driven by the
REAL controller/planner against the mission's REAL inputs (environment manifest + mission
scenario + mission camera config, the full ~800-obstacle set the live mission gates against --
the same inputs scripts/check_couple_nav.py uses):

  (1) DIRECTION-AWARE braking zone (navigation/core.py Controller.gate_velocity): the ds-007 /
      run-319b backstock Support, sensed ~0.81 m in the robot's REAR hemisphere while it drives
      forward and away, no longer trips obstacle_stop; the SAME return the same distance AHEAD
      still stops the robot; and the legacy omnidirectional disk (directional_zone=False) DOES
      trip on the rear return (reproducing the bug the fix removes).

  (2) PRE-PLANNED TOW LANES (navigation/lanes.py): the authored outbound (couple dock -> shelf
      park) and return (shelf park -> cart parking spot) lanes expand against the static map,
      re-validate their swept robot+cart envelope against the full LIVE obstacle set, and the
      real ArticulatedController follows each lane closed-loop from a jittered start to the goal
      -- the return with NO reorientation loop (length within a small factor of the direct line),
      where free planning to the east-facing home timed out at 23.9 m for a 10.4 m direct.

Run:  python3 scripts/check_tow_lanes.py \
        --environment <environment.json> --scenario <mission-scenario.json> \
        --camera-config <camera-config.json> --lanes docs/examples/tow-lanes-mission-fragment.json
Exit 0 iff every check passes (prints TOW_LANES_PROOF PASS/FAIL)."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*
from twin.navigation.scenario import environment_grid
from twin.navigation.core import Controller, Footprint, GridMap
from twin.navigation.lanes import TowLaneGraph
from twin.navigation.articulated import ArticulatedController, ArticulatedMap, TrainState, advance
from twin.navigation import Odometry, Observation
from twin.mobile import MobileSpec


def _record(report, name, passed, **kw):
    report['ok'] = report['ok'] and passed
    report['checks'].append(dict(name=name, passed=passed, **kw))
    print(('PASS ' if passed else 'FAIL ') + name + ' ' + json.dumps(kw), flush=True)


# ---- (1) direction-aware zone -------------------------------------------------------------

def check_directional_zone(grid, report):
    """ds-007 geometry on the REAL uncoupled base gate: rear clears, ahead stops, legacy trips."""
    spec = MobileSpec()
    fp = Footprint(spec.transport_radius, abs(spec.hitch_x),
                   math.hypot(spec.cart_hitch_x + spec.cart_length / 2, spec.cart_width / 2),
                   margin=0.05, min_turn_radius=0.0, max_articulation=spec.max_articulation)
    # ds-007: robot stalled at (-5.64, 15.98, -26 deg); the Support sensed ~0.81 m in its rear
    # hemisphere. Model one return at 0.81 m BEHIND and one the same distance AHEAD.
    heading = math.radians(-26.0)
    robot = (-5.64, 15.98, heading)
    dist = 0.81
    rear = (robot[0] - dist * math.cos(heading), robot[1] - dist * math.sin(heading))
    ahead = (robot[0] + dist * math.cos(heading), robot[1] + dist * math.sin(heading))
    # Isolate the gate from the static map (the Support is modelled as the sensed return here;
    # the mission grid's own geometry is exercised by the lane proof below).
    empty = GridMap((-20, -20, 20, 20), (), 0.2)
    od = Odometry(robot, 0.1, 0.25, 0.0, 'test_fixture', 0.05)

    def gate(directional, obstacle):
        c = Controller(empty, fp, allow_test_fixtures=True, directional_zone=directional)
        c.start((robot[0] + math.cos(heading), robot[1] + math.sin(heading), heading), False)
        obs = Observation(0.1, (obstacle,), False, True, 0.0, 'test_fixture', 10.0)
        return c.gate_velocity(0.1, od, obs, 0.25, 0.0).reason

    _record(report, 'directional_rear_support_clears',
            gate(True, rear) == 'tracking', dist_m=dist, reason=gate(True, rear))
    _record(report, 'directional_ahead_same_distance_stops',
            gate(True, ahead) == 'obstacle_stop', dist_m=dist, reason=gate(True, ahead))
    _record(report, 'legacy_omnidirectional_disk_trips_on_rear',
            gate(False, rear) == 'obstacle_stop', dist_m=dist, reason=gate(False, rear))


# ---- (2) tow lanes ------------------------------------------------------------------------

def _obstacle_points(grid, step=0.25):
    """A dense LIVE obstacle cloud sampled on the static-obstacle AABB faces, so the lane's
    swept-envelope re-validation runs against the real geometry as sensed returns (not only the
    AABB test). Conservative: samples the full perimeter of every static obstacle."""
    pts = []
    for xmin, ymin, xmax, ymax in grid.obstacles:
        nx = max(1, int((xmax - xmin) / step))
        ny = max(1, int((ymax - ymin) / step))
        for i in range(nx + 1):
            x = xmin + (xmax - xmin) * i / nx
            pts.append((x, ymin)); pts.append((x, ymax))
        for j in range(ny + 1):
            y = ymin + (ymax - ymin) * j / ny
            pts.append((xmin, y)); pts.append((xmax, y))
    return tuple(pts)


def _proud_points(grid, delta, step=0.25):
    """A LIVE cloud of the static-obstacle face returns pushed `delta` m OUTWARD (away from each
    AABB centre) -- the documented lidar/ESDF effect that returns register a few cm PROUD of the
    true surface toward free space. This reproduces the run-321b outbound `lane_blocked`: the
    authored centreline clears the true static map but a proud return of a shelf/support it hugs
    (tightest outbound clearance ~0.089 m) intrudes the swept envelope."""
    pts = []
    for xmin, ymin, xmax, ymax in grid.obstacles:
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        nx = max(1, int((xmax - xmin) / step)); ny = max(1, int((ymax - ymin) / step))
        face = []
        for i in range(nx + 1):
            x = xmin + (xmax - xmin) * i / nx
            face.append((x, ymin)); face.append((x, ymax))
        for j in range(ny + 1):
            y = ymin + (ymax - ymin) * j / ny
            face.append((xmin, y)); face.append((xmax, y))
        for x, y in face:
            dx, dy = x - cx, y - cy
            n = math.hypot(dx, dy) or 1.0
            pts.append((x + delta * dx / n, y + delta * dy / n))
    return tuple(pts)


def check_proud_tolerance(grid, lanes_cfg, report):
    """fm423(a): reproduce the run-321b live `lane_blocked` with a lidar-proud cloud model and prove
    the opt-in proud tolerance (navigation.tow_lane_static_tol_m) reopens the lane WITHOUT blanket-
    trusting the static map -- a genuinely NEW obstacle on the lane still blocks the tow at plan time
    (and the live directional braking gate, untouched, still stops it)."""
    spec = MobileSpec()
    nav = lanes_cfg['navigation']
    graph = TowLaneGraph.from_config(nav['tow_lanes'])
    graph.expand_all(ArticulatedMap(grid, spec))
    lane = next((ln for ln in graph.lanes if 'outbound' in ln.name), graph.lanes[0])
    PROUD, TOL = 0.10, 0.12   # proud blocks (tightest outbound clearance ~0.089 m); tol covers it
    proud = _proud_points(grid, PROUD)

    def plan_reason(lane_static_tol, extra=()):
        c = ArticulatedController(grid, spec, allow_test_fixtures=True, lane_graph=graph,
                                  lane_static_tol=lane_static_tol)
        c.start(lane.goal, True)
        state = lane.start_state()
        obs = Observation(0.1, tuple(proud) + tuple(extra), True, True,
                          state.articulation, 'test_fixture', 10.0)
        return c.plan_observed_route(state, obs, 0.05).reason

    _record(report, 'proud_cloud_reproduces_live_lane_blocked',
            plan_reason(0.0) == 'lane_blocked', proud_m=PROUD, reason=plan_reason(0.0))
    _record(report, 'proud_tolerance_reopens_lane',
            plan_reason(TOL) == 'validated_lane', tol_m=TOL, reason=plan_reason(TOL))
    # SAFETY: a new obstacle on the lane centre, far from any static geometry, is NEVER discounted.
    mid = lane.route.states[len(lane.route.states) // 2]
    _record(report, 'new_obstacle_on_lane_still_blocks_tow',
            plan_reason(TOL, extra=[(mid.x, mid.y)]) == 'lane_blocked',
            new_obstacle=(round(mid.x, 2), round(mid.y, 2)), reason=plan_reason(TOL, extra=[(mid.x, mid.y)]))


def _drive_lane(controller, spec, lane, goal, *, jitter, report, label, direct_xy):
    """Drive the REAL ArticulatedController following the lane closed-loop from a jittered start."""
    import time as _time
    controller.start(goal, True)
    s0 = lane.start_state()
    pose = (s0.x + jitter[0], s0.y + jitter[1], s0.yaw + jitter[2])
    cart_yaw = s0.cart_yaw + jitter[3]
    v = w = 0.0
    dt = 0.1
    t0 = _time.monotonic()
    reached = False
    detail = 'timeout'
    for tick in range(8000):
        now = (tick + 1) * dt
        art = math.atan2(math.sin(pose[2] - cart_yaw), math.cos(pose[2] - cart_yaw))
        od = Odometry(pose, now, abs(v), w, 'test_fixture', 0.05)
        obs = Observation(now, (), True, True, art, 'test_fixture', 10.0)
        r = controller.step(dt, od, obs, now=now)
        if r.status == 'succeeded':
            reached = True; detail = r.reason; break
        if r.status == 'failed':
            detail = r.reason; break
        v, w = r.linear, r.angular
        th = pose[2] + w * dt
        ts = TrainState(pose[0], pose[1], pose[2], cart_yaw)
        k = w / v if abs(v) > 1e-6 else 0.0
        ns = advance(ts, v * dt, k, spec)
        pose = (pose[0] + v * dt * math.cos(th), pose[1] + v * dt * math.sin(th), th)
        cart_yaw = ns.cart_yaw
    cpu = _time.monotonic() - t0
    route = lane.route
    length = sum(math.dist((route.states[i].x, route.states[i].y),
                           (route.states[i + 1].x, route.states[i + 1].y))
                 for i in range(len(route.states) - 1))
    end_art = abs(route.states[-1].articulation)
    _record(report, f'lane_follows_to_goal[{label}]', reached,
            detail=detail, route_len_m=round(length, 2), direct_m=round(direct_xy, 2),
            loop_ratio=round(length / direct_xy, 2), end_articulation_rad=round(end_art, 3),
            final_pose=tuple(round(c, 3) for c in pose), follow_cpu_s=round(cpu, 2))
    return reached, length, direct_xy


def check_tow_lanes(grid, lanes_cfg, report):
    import time as _time
    spec = MobileSpec()
    world = ArticulatedMap(grid, spec)
    nav = lanes_cfg['navigation']
    graph = TowLaneGraph.from_config(nav['tow_lanes'])
    t0 = _time.monotonic()
    graph.expand_all(world)
    _record(report, 'both_lanes_expand_against_static_map',
            all(ln.route is not None for ln in graph.lanes),
            lanes=[ln.name for ln in graph.lanes], expand_cpu_s=round(_time.monotonic() - t0, 2))

    live = _obstacle_points(grid)
    report['live_obstacle_points'] = len(live)

    # Re-validate each lane's swept robot+cart envelope against the FULL live obstacle cloud.
    for ln in graph.lanes:
        t1 = _time.monotonic()
        planner = __import__('twin.navigation.articulated', fromlist=['ArticulatedPlanner']).ArticulatedPlanner(world)
        chk = planner.validate_seed(ln.route, ln.start_state(), ln.goal, live,
                                    0.05, ln.goal_articulation)
        _record(report, f'lane_swept_envelope_clear_live[{ln.name}]', bool(chk.states),
                reason=chk.reason, states=len(chk.states), validate_cpu_s=round(_time.monotonic() - t1, 3))

    # Build the REAL controller with the lane graph and follow each lane closed-loop.
    controller = ArticulatedController(grid, spec, allow_test_fixtures=True,
                                       goal_timeout=600.0, blocked_timeout=60.0, lane_graph=graph)
    for ln in graph.lanes:
        direct = math.dist(ln.start[:2], ln.goal[:2])
        _drive_lane(controller, spec, ln, ln.goal,
                    jitter=(0.06, -0.05, 0.03, 0.04), report=report, label=ln.name,
                    direct_xy=direct)

    # The return must NOT be a reorientation loop (free planning to the east-facing home was a
    # 23.9 m loop for a 10.4 m direct = ratio 2.3). Assert the authored return stays well under.
    ret = next((ln for ln in graph.lanes if 'return' in ln.name), None)
    if ret is not None and ret.route is not None:
        length = sum(math.dist((ret.route.states[i].x, ret.route.states[i].y),
                               (ret.route.states[i + 1].x, ret.route.states[i + 1].y))
                     for i in range(len(ret.route.states) - 1))
        direct = math.dist(ret.start[:2], ret.goal[:2])
        _record(report, 'return_lane_no_reorientation_loop', length / direct < 1.7,
                route_len_m=round(length, 2), direct_m=round(direct, 2),
                loop_ratio=round(length / direct, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--environment', type=Path, required=True)
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--camera-config', type=Path, required=True)
    ap.add_argument('--lanes', type=Path, required=True,
                    help='mission fragment with navigation.tow_lanes (e.g. docs/examples/tow-lanes-mission-fragment.json)')
    args = ap.parse_args()

    manifest = json.loads(args.environment.read_text())
    scenario = json.loads(args.scenario.read_text())
    camera = json.loads(args.camera_config.read_text())
    lanes_cfg = json.loads(args.lanes.read_text())
    grid = environment_grid(manifest, scenario, camera)   # the exact mission nav grid (+ camera mounts)

    report = {'ok': True, 'checks': [], 'static_obstacles': len(grid.obstacles)}
    print(f'grid: {len(grid.obstacles)} static obstacles, bounds {grid.bounds}', flush=True)
    check_directional_zone(grid, report)
    check_tow_lanes(grid, lanes_cfg, report)
    check_proud_tolerance(grid, lanes_cfg, report)
    print('TOW_LANES_PROOF ' + ('PASS' if report['ok'] else 'FAIL'), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
