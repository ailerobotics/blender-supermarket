#!/usr/bin/env python3
"""fm390 offline proof #2: can the mobile BASE drive from the pick pose to place_approach (in front of
the rollwagen's open +x face) WITHOUT the base footprint touching the parked cart or the source
handling-unit support?

Why this must be proven offline: the reference/Nav2 planner does NOT carry the parked cart as a static
obstacle during collect (couple later drives the base right up to the same cart to back onto its hitch,
so the cart cannot be a nav obstacle at these working distances). The base-vs-cart clearance along the
reposition drive therefore holds by CONSTRUCTION, not by the in-sim obstacle gate -- so it is checked
here, outside the sim, with the physical base rectangle against the cart chassis + source AABBs.

Model (honest, labelled):
  * The base is its PHYSICAL rectangle (spec.length x spec.width), not the conservative transport-radius
    disk `occupied_envelope` uses: that disk (~0.71 m) is a free-navigation planning envelope and
    overlaps the parked cart at every close docking/placement distance the mission deliberately works at
    (pickup sits 0.75 m from the cart centre), so it cannot model actual contact here. Contact is a
    rectangle-vs-rectangle question.
  * The cart CHASSIS is a full-height 2D obstacle. The DRAWBAR/hitch (world x[-5.33,-5.08], z[0.26,0.30])
    is an elevated thin bar: the base body top (base_z) passes UNDER it, and the carried carton is tucked
    over the base centre (>= 0.5 m east of the drawbar tip), so the drawbar is reported with its vertical
    clearance, not treated as a base-footprint obstacle. The arm/carton-vs-drawbar clearance is the reach
    proof's job (scripts/check_place_approach_reach.py).

Prints, for several path models (direct face-travel, direct linear-yaw worst case, and a staged path via
an east waypoint), whether every sampled base rectangle clears both obstacles, with the min clearance.

Run:  .venv-usd/bin/python -I scripts/check_place_approach_nav.py \
        --mobile-manifest <mobile_manifest.json> --scenario <mission-scenario.json>
  (plain python3 -I also works; this proof needs no pxr.)
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*
from twin.mobile import MobileSpec, Pose2
from twin.navigation.articulated import polygon_hits_box


def base_rectangle(pose, spec, margin=0.0):
    """The physical base footprint polygon (world) at a (x, y, yaw) pose, optionally inflated."""
    p = Pose2(pose[0], pose[1], pose[2])
    hx, hy = spec.length / 2 + margin, spec.width / 2 + margin
    return [p.point(hx, hy), p.point(hx, -hy), p.point(-hx, -hy), p.point(-hx, hy)]


def _seg_point_dist(a, b, p):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def poly_box_clearance(poly, box):
    """Min distance between a convex polygon and an AABB (0.0 if they overlap/touch)."""
    if polygon_hits_box(poly, box):
        return 0.0
    xmin, ymin, xmax, ymax = box
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    edges_poly = list(zip(poly, poly[1:] + poly[:1]))
    edges_box = list(zip(corners, corners[1:] + corners[:1]))
    best = math.inf
    for a, b in edges_poly:
        for c in corners:
            best = min(best, _seg_point_dist(a, b, c))
    for a, b in edges_box:
        for c in poly:
            best = min(best, _seg_point_dist(a, b, c))
    return best


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def path_face_travel(start, goal, n=60):
    """Translate facing the travel heading, then rotate in place to the goal yaw (diff-drive-like)."""
    heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
    out = []
    for i in range(n + 1):           # translation leg, facing travel
        t = i / n
        out.append((start[0] + t * (goal[0] - start[0]),
                    start[1] + t * (goal[1] - start[1]), heading))
    dyaw = _wrap(goal[2] - heading)
    for i in range(1, n + 1):        # in-place rotation to the goal yaw
        out.append((goal[0], goal[1], heading + dyaw * i / n))
    return out


def path_linear_yaw(start, goal, n=120):
    """Worst case: translate AND linearly interpolate yaw at once (a continuous 180 deg turn en route)."""
    dyaw = _wrap(goal[2] - start[2])
    return [(start[0] + t * (goal[0] - start[0]), start[1] + t * (goal[1] - start[1]),
             start[2] + dyaw * t) for t in (i / n for i in range(n + 1))]


def path_staged(start, stage, goal, n=60):
    """pickup -> east staging waypoint (keep yaw~0, clear of the cart) -> place_approach."""
    return path_face_travel(start, stage, n) + path_face_travel(stage, goal, n)


def evaluate(path, obstacles, spec, margin=0.05):
    worst = math.inf
    first_hit = None
    for i, pose in enumerate(path):
        rect = base_rectangle(pose, spec, margin)
        for name, box in obstacles.items():
            clr = poly_box_clearance(rect, box)
            if clr < worst:
                worst = clr
            if clr <= 0.0 and first_hit is None:
                first_hit = (i, name, tuple(round(c, 3) for c in pose))
    return dict(clears=first_hit is None, min_clearance_m=round(worst, 4), first_hit=first_hit)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mobile-manifest', type=Path, required=True)
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--standoff', type=float, default=1.15,
                    help='place_approach standoff east of the deck centre (default 1.15, in the '
                         'reach-proof feasible base-x window)')
    ap.add_argument('--margin', type=float, default=0.05, help='footprint safety margin (m)')
    a = ap.parse_args()
    manifest = json.loads(a.mobile_manifest.read_text())
    scenario = json.loads(a.scenario.read_text())
    spec = MobileSpec(**{k: v for k, v in manifest['spec'].items()
                         if k in MobileSpec.__dataclass_fields__})
    st = scenario['initial_staging']
    cart_xy = tuple(st['cart_body_pose'][:2])
    deck_center = (cart_xy[0] + st['cart_load_center_body_xy'][0],
                   cart_xy[1] + st['cart_load_center_body_xy'][1])
    pickup = tuple(st['robot_base_pose'])
    place_approach = (deck_center[0] + a.standoff, deck_center[1], math.pi)   # +x mouth, facing it
    # Obstacle AABBs (world). Cart chassis from the spec footprint; source support from the scenario.
    cl, cw = spec.cart_length, spec.cart_width
    cart_chassis = (cart_xy[0] - cl / 2, cart_xy[1] - cw / 2, cart_xy[0] + cl / 2, cart_xy[1] + cw / 2)
    sx, sy = st['support_center_xy']
    sdx, sdy = st['support_dimensions_m'][0] / 2, st['support_dimensions_m'][1] / 2
    source_support = (sx - sdx, sy - sdy, sx + sdx, sy + sdy)
    obstacles = {'cart_chassis': cart_chassis, 'source_support': source_support}
    # East staging waypoint: place_approach's x, pickup's y, facing east (yaw 0) -- clear of the cart,
    # then rotate+translate south to the mouth. (The mission already stages couple this way: couple_approach.)
    stage = (place_approach[0], pickup[1], 0.0)

    report = dict(
        frame='world_metres',
        base_rect_m=[spec.length, spec.width], margin_m=a.margin,
        pickup=pickup, place_approach=[round(v, 4) for v in place_approach],
        east_staging_waypoint=[round(v, 4) for v in stage],
        obstacles={k: [round(c, 3) for c in v] for k, v in obstacles.items()},
        drawbar_note=('drawbar/hitch world x[-5.33,-5.08] z[0.26,0.30] is ELEVATED and thin: the base '
                      'body (top=base_z=%.3f) passes under it; the carried carton is tucked over the base '
                      'centre >= 0.5 m east of the drawbar tip. Arm/carton-vs-drawbar clearance = reach '
                      'proof. NOT modelled as a base-footprint obstacle here.' % spec.base_z),
        end_pose_static=evaluate([place_approach], obstacles, spec, a.margin),
        path_direct_face_travel=evaluate(path_face_travel(pickup, place_approach), obstacles, spec, a.margin),
        path_direct_linear_yaw=evaluate(path_linear_yaw(pickup, place_approach), obstacles, spec, a.margin),
        path_staged_via_east_waypoint=evaluate(path_staged(pickup, stage, place_approach), obstacles, spec, a.margin),
    )
    print(json.dumps(report, indent=2))
    end_ok = report['end_pose_static']['clears']
    staged_ok = report['path_staged_via_east_waypoint']['clears']
    if end_ok and staged_ok:
        print('\nFEASIBLE: the base footprint CLEARS the parked cart + source at place_approach, and a '
              'collision-free drive exists VIA the east staging waypoint %s (translate east clear of the '
              'cart, then turn to face the mouth). The direct face-travel path is %s.'
              % ([round(v, 2) for v in stage],
                 'also clear' if report['path_direct_face_travel']['clears'] else
                 'NOT clear -- route via the staging waypoint'))
        return 0
    print('\nNOT PROVEN: end_pose_clears=%s staged_path_clears=%s' % (end_ok, staged_ok))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
