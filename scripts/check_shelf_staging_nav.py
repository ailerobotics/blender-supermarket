#!/usr/bin/env python3
"""Offline proof (CPU-only) for the sim-shelf-staging-v1 sales-floor re-sequence, against the mission's
REAL nav grid (environment manifest + mission scenario, the full ~800-obstacle static map, the same
inputs scripts/check_tow_lanes.py uses) and the REAL twin.mobile train/base envelopes.

The captain's complaint was the ~46 deg in-aisle coupled turn to the externally-fixed west-facing endcap
service pose (and the cart exiting the doorway angled ~40 deg on the outbound). This re-sequences the
sales-floor leg so the COUPLED train never makes that turn:

  (1) STRAIGHT COUPLED TOW to the staging pose (navigation.tow_lanes 'outbound_dock_to_staging', expanded
      by check_tow_lanes too): the coupled articulation is ~0 deg EVERYWHERE and the cart crosses the
      doorway aligned -- vs the ~46 deg whole-lane / ~40 deg cart-in-doorway of the coupled-to-endcap lane.

  (2) UNCOUPLED ROBOT-ONLY SERVICE PATH: after the robot brakes + releases the cart at the staging, the
      FREE base drives ALONE to the endcap `shelf_approach` service pose. Proven with the base's PHYSICAL
      rectangle (not the conservative transport-radius disk -- which overlaps the just-released cart at the
      departure distance by construction) against the static map AND the parked-cart AABB, on a realistic
      forward-first departure path (the base pulls straight forward away from the cart, then to the endcap).
      The parked rollwagen is also checked clear of the y=6 service aisle.

  (3) RETURN CONSTRAINT (reported, not a failure): the rollwagen parked on the portal axis BLOCKS the free
      base from threading the single 2.6 m doorway to drive home empty (the real grid A* finds NO PATH).
      This is the documented integration/GPU hand-off (re-couple + aisle reorientation, OR treat the
      parked cart as a known-static for the final leg) -- see the fragment's _return_handoff_note.

Run:  python3 scripts/check_shelf_staging_nav.py \
        --environment <environment.json> --scenario <mission-scenario.json> \
        --lanes docs/examples/tow-lanes-mission-fragment.json
Exit 0 iff (1) and (2) pass (prints SHELF_STAGING_NAV PASS/FAIL)."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*
from twin.navigation.scenario import environment_grid
from twin.navigation.core import GridMap
from twin.navigation.lanes import TowLaneGraph
from twin.navigation.articulated import ArticulatedMap, polygon_hits_box
from twin.mobile import MobileSpec, Pose2

WALL_Y = 11.125
MAX_COUPLED_DEG = 10.0           # the captain target: coupled articulation under ~10 deg anywhere


def _record(report, name, passed, **kw):
    report['ok'] = report['ok'] and passed
    report['checks'].append(dict(name=name, passed=passed, **kw))
    print(('PASS ' if passed else 'FAIL ') + name + ' ' + json.dumps(kw), flush=True)


def _cart_centre(spec, s):
    r = Pose2(s.x, s.y, s.yaw)
    hx, hy = r.point(spec.hitch_x)
    return (hx - spec.cart_hitch_x * math.cos(s.cart_yaw),
            hy - spec.cart_hitch_x * math.sin(s.cart_yaw))


def _cart_aabb(spec, s, pad=0.02):
    cx, cy = _cart_centre(spec, s)
    cp = Pose2(cx, cy, s.cart_yaw)
    corners = [cp.point(x, y) for x in (-spec.cart_length / 2, spec.cart_hitch_x)
               for y in (-spec.cart_width / 2, spec.cart_width / 2)]
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _base_rect(spec, pose, margin=0.0):
    p = Pose2(pose[0], pose[1], pose[2])
    hx, hy = spec.length / 2 + margin, spec.width / 2 + margin
    return [p.point(hx, hy), p.point(hx, -hy), p.point(-hx, -hy), p.point(-hx, hy)]


def _seg_pt(a, b, p):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _poly_box_clear(poly, box):
    if polygon_hits_box(poly, box):
        return 0.0
    xmin, ymin, xmax, ymax = box
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    best = math.inf
    for a, b in zip(poly, poly[1:] + poly[:1]):
        for c in corners:
            best = min(best, _seg_pt(a, b, c))
    for a, b in zip(corners, corners[1:] + corners[:1]):
        for c in poly:
            best = min(best, _seg_pt(a, b, c))
    return best


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _staged_path(start, via, goal, n=30):
    """Forward-first departure: drive straight (yaw held) start->via, then face-travel via->goal, then
    rotate in place to the goal yaw -- a diff-drive-plausible uncoupled motion."""
    pts = []
    for i in range(n + 1):
        t = i / n
        pts.append((start[0] + t * (via[0] - start[0]), start[1] + t * (via[1] - start[1]), start[2]))
    heading = math.atan2(goal[1] - via[1], goal[0] - via[0])
    for i in range(1, n + 1):
        t = i / n
        pts.append((via[0] + t * (goal[0] - via[0]), via[1] + t * (goal[1] - via[1]), heading))
    dyaw = _wrap(goal[2] - heading)
    for i in range(1, n + 1):
        pts.append((goal[0], goal[1], _wrap(heading + dyaw * i / n)))
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--environment', type=Path, required=True)
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--lanes', type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.environment.read_text())
    scenario = json.loads(args.scenario.read_text())
    cfg = json.loads(args.lanes.read_text())
    grid = environment_grid(manifest, scenario, None)
    spec = MobileSpec()
    world = ArticulatedMap(grid, spec)
    report = {'ok': True, 'checks': [], 'static_obstacles': len(grid.obstacles)}
    print(f'grid: {len(grid.obstacles)} static obstacles, bounds {grid.bounds}', flush=True)

    poses = cfg['mission_poses']
    staging = tuple(poses['shelf_staging'])
    service = tuple(poses['shelf_approach'])

    # (1) straight coupled tow to the staging -- coupled articulation under the target everywhere.
    graph = TowLaneGraph.from_config(cfg['navigation']['tow_lanes'])
    graph.expand_all(world)
    lane = next(ln for ln in graph.lanes if 'staging' in ln.name)
    st = lane.route.states
    whole = max(math.degrees(abs(s.articulation)) for s in st)
    door = max((math.degrees(abs(s.articulation)) for s in st
                if abs(_cart_centre(spec, s)[1] - WALL_Y) <= 0.7), default=0.0)
    robot_cross = 0.0
    for a, b in zip(st, st[1:]):
        if (a.y - WALL_Y) * (b.y - WALL_Y) <= 0 and a.y != b.y:
            robot_cross = math.degrees(abs(a.articulation)); break
    _record(report, 'coupled_tow_to_staging_under_10deg', whole <= MAX_COUPLED_DEG,
            whole_lane_max_deg=round(whole, 2), target_deg=MAX_COUPLED_DEG)
    _record(report, 'cart_crosses_doorway_aligned', door <= 3.0,
            cart_in_doorway_max_deg=round(door, 2), robot_wall_cross_deg=round(robot_cross, 2))

    # the parked cart (terminal train state) and its clearance from the y=6 service aisle.
    cart_box = _cart_aabb(spec, st[-1])
    cart_cx, cart_cy = _cart_centre(spec, st[-1])
    _record(report, 'parked_cart_clear_of_service_aisle', cart_box[1] > 6.6 and cart_box[3] < WALL_Y,
            cart_centre=[round(cart_cx, 2), round(cart_cy, 2)],
            cart_y_range=[round(cart_box[1], 2), round(cart_box[3], 2)], aisle_y=6.0, wall_y=WALL_Y)

    # (2) uncoupled robot-only service path staging -> endcap, forward-first departure.
    via = (staging[0], 6.5, staging[2])
    path = _staged_path(staging, via, service)
    min_cart = math.inf; min_static = math.inf; hit = None
    for pose in path:
        rect = _base_rect(spec, pose, margin=0.05)
        c = _poly_box_clear(rect, cart_box); min_cart = min(min_cart, c)
        if c == 0 and hit is None:
            hit = {'cart': [round(pose[0], 2), round(pose[1], 2)]}
        for ob in grid.obstacles:
            cs = _poly_box_clear(rect, ob); min_static = min(min_static, cs)
            if cs == 0 and hit is None:
                hit = {'static': [round(v, 2) for v in ob], 'at': [round(pose[0], 2), round(pose[1], 2)]}
    _record(report, 'uncoupled_service_path_clears', min_static > 0.0 and min_cart >= 0.0 and hit is None,
            min_static_clear_m=round(min_static, 3), min_cart_clear_m=round(min_cart, 3),
            via=[round(v, 2) for v in via], hit=hit)

    # (3) RETURN CONSTRAINT (reported, informational -- the documented integration/GPU hand-off).
    # home is the backroom return pose from the environment manifest (fallback to the portal-axis home).
    home_pose = tuple(manifest.get('poses', {}).get('home', (-3.875, 14.0, 0.0)))
    g2 = GridMap(grid.bounds, tuple(list(grid.obstacles) + [cart_box]), grid.resolution)
    empty_return = g2.plan(service, home_pose, spec.transport_radius)
    print('NOTE return_empty_blocked_by_parked_cart=%s (integration/GPU hand-off: re-couple + aisle '
          'reorientation OR treat parked cart as known-static for the final leg; see fragment '
          '_return_handoff_note)' % (not bool(empty_return)), flush=True)

    print('SHELF_STAGING_NAV ' + ('PASS' if report['ok'] else 'FAIL'), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
