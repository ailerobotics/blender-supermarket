#!/usr/bin/env python3
"""sim-collect-dock-reach-v1 offline proof (CPU-only): the SOUTH-headed cart dock satisfies BOTH jobs
the captain's step-by-step rule splits across the collect stage and the outbound tow.

coldwin26 failed at COLLECT with 'unreachable or undersized target': the cart, docked on the portal
axis at (-3.875, 15.77, -pi/2) for a gentle doorway crossing, sat ~1.9 m from the robot at its fixed
backstock collect pose (-5.73, 16, 0) -- past the arm place-reach gate (max_reach ~1.3 m) -- so the
carton could not be placed on the cart. Pushing the cart EAST next to the source made the place reach
but the SOUTH outbound lane could not be acquired. One cart pose was asked to serve two jobs.

Firstmate's decision: define the dock by BOTH constraints at once and prove, offline, that it:
  (1) REACH   -- the cart DECK place target is inside the arm place-reach gate from the FIXED collect
                 pose (the coldwin26 gate), with margin; PR 290's standoff+descent keeps the carton
                 LEVEL on the way down (scripts/twin/tools/carton_level_bench.run_place, reported here).
  (2) FACING  -- the cart is headed SOUTH (within a few degrees), so it crosses the doorway aligned.
  (3) CLEAR   -- the docked cart body clears the backstock Support corridor, the source shelf and the
                 robot's own footprint.
  (4) COUPLE  -- PR 293's cart-relative standoff + straight-reverse insert fits south of the dock
                 (the reverse base rectangle clears the static map).
  (5) OUTBOUND-- a gentle pre-planned tow from the dock reaches the portal-axis lead-in with the turn
                 in the OPEN BACKROOM away from the door (whole-lane |articulation| < 15 deg, cart
                 within 10 deg of the axis at the doorway, a straight lead-in >= 3 m before the wall).
                 (The lane geometry itself is also gated by scripts/check_tow_lanes.py /
                 scripts/check_cart_side_bay.py; this proof asserts the dock-derived seat + reach.)

Model is kinematic + swept-footprint only (collision_checked via the static map, NOT PhysX); labelled.

Run:  .venv-usd/bin/python -I scripts/check_collect_dock_reach.py \
        --mobile-manifest <mobile_manifest.json> \
        --environment <environment.json> --scenario scripts/restock_scenario.json \
        --lanes docs/examples/cart-side-bay-mission-fragment.json
Exit 0 iff every claim passes (prints COLLECT_DOCK_REACH PASS/FAIL)."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*
from twin.navigation.scenario import environment_grid
from twin.navigation.articulated import ArticulatedMap, polygon_hits_box
from twin.navigation.lanes import TowLane, TowLaneGraph, couple_seat_from_cart
from twin.mobile import MobileSpec, wrap

SOUTH = -math.pi / 2
DOOR_Y = 11.125
PORTAL_X = (-5.175, -2.575)      # the widened wall opening (PR 266), x extent
FACING_TOL_DEG = 5.0             # "within a few degrees" of south
ART_MAX_DEG = 15.0               # the captain target for the whole outbound lane
DOORWAY_ALIGNED_DEG = 10.0
STRAIGHT_LEADIN_M = 3.0
DOORWAY_CLEAR_M = 0.5            # >= 0.5 m clear each side at the crossing


def _rot(yaw, v):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def _aabb(cx, cy, yaw, length, width, pad=0.0):
    hl, hw = length / 2 + pad, width / 2 + pad
    xs, ys = [], []
    for dx in (-hl, hl):
        for dy in (-hw, hw):
            rx, ry = _rot(yaw, (dx, dy))
            xs.append(cx + rx); ys.append(cy + ry)
    return (min(xs), min(ys), max(xs), max(ys))


def _clear(a, b, gap=0.0):
    """True iff AABBs a, b are separated (no overlap) with at least `gap` between them."""
    return a[2] <= b[0] - gap or a[0] >= b[2] + gap or a[3] <= b[1] - gap or a[1] >= b[3] + gap


def _sep(a, b):
    """Signed separation (m) between two non-overlapping AABBs (max over axes; >0 => clear)."""
    return max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3])


def _base_rect(spec, pose, margin=0.05):
    x, y, yaw = pose
    hl, hw = spec.length / 2 + margin, spec.width / 2 + margin
    pts = []
    for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        rx, ry = _rot(yaw, (dx, dy))
        pts.append((x + rx, y + ry))
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mobile-manifest', type=Path, required=True)
    ap.add_argument('--environment', type=Path, required=True)
    ap.add_argument('--scenario', type=Path, default=Path('scripts/restock_scenario.json'))
    ap.add_argument('--lanes', type=Path, required=True,
                    help='mission fragment with navigation.tow_lanes carrying the dock as couple_dock')
    ap.add_argument('--lane-name', default=None,
                    help='which tow lane to read the dock from (default: the first outbound lane)')
    args = ap.parse_args(argv)

    mobile = json.loads(args.mobile_manifest.read_text())
    env = json.loads(args.environment.read_text())
    scenario = json.loads(args.scenario.read_text())
    cfg = json.loads(args.lanes.read_text())
    spec = MobileSpec()

    staging = scenario['initial_staging']
    unit = scenario['handling_unit']
    robot = tuple(float(v) for v in staging['robot_base_pose'])        # FIXED collect pose (-5.73,16,0)
    max_reach = float(unit['max_reach_m'])
    carton_h = unit['dimensions_m'][2]
    load_center = staging['cart_load_center_body_xy']
    deck_z = staging['cart_support_z_world']                           # the DECK (0.225), per spec
    arm_base_z = mobile['spec'].get('arm_mount_z', 0.5)

    # The dock is the ONE source of truth: the active outbound lane's couple_dock.
    lanes_cfg = cfg['navigation']['tow_lanes']
    graph = TowLaneGraph.from_config(lanes_cfg)
    outbound = None
    for ln in graph.lanes:
        if args.lane_name and ln.name == args.lane_name:
            outbound = ln; break
        if 'return' not in ln.name and outbound is None:
            outbound = ln
    if outbound is None:
        print('COLLECT_DOCK_REACH FAIL (no outbound lane in the fragment)'); return 1
    if outbound.couple_dock is None:
        print('COLLECT_DOCK_REACH FAIL (outbound lane declares no couple_dock = the dock source)'); return 1
    dock = outbound.couple_dock
    dock_x, dock_y, dock_yaw = dock

    report = {'ok': True, 'checks': []}
    def rec(name, passed, **kw):
        report['ok'] = report['ok'] and bool(passed)
        report['checks'].append((name, bool(passed), kw))
        print(('PASS ' if passed else 'FAIL ') + name + ' ' + json.dumps(kw), flush=True)

    # (2) FACING south (within a few degrees).
    facing_err = abs(math.degrees(wrap(dock_yaw - SOUTH)))
    rec('dock_faces_south', facing_err <= FACING_TOL_DEG,
        dock=[round(v, 3) for v in dock], facing_err_deg=round(facing_err, 2), tol_deg=FACING_TOL_DEG)

    # (1) REACH: the cart DECK place target within the arm reach gate from the FIXED collect pose.
    dcx, dcy = _rot(dock_yaw, (load_center[0], load_center[1]))
    deck = (dock_x + dcx, dock_y + dcy, deck_z)                        # deck load-centre, world
    # base-frame vector (robot-base origin): x,y rotated into the base, z world (the mission's base-frame
    # reach gate -- the coldwin26 'unreachable or undersized target' scale, ~1.9 m at the on-axis dock).
    rel = _rot(-robot[2], (deck[0] - robot[0], deck[1] - robot[1]))
    mag_baselink = math.sqrt(rel[0] ** 2 + rel[1] ** 2 + deck[2] ** 2)
    mag_armbase = math.sqrt(rel[0] ** 2 + rel[1] ** 2 + (deck[2] - arm_base_z) ** 2)
    margin = max_reach - mag_baselink
    rec('deck_place_target_within_reach', mag_baselink <= max_reach,
        deck_world=[round(v, 3) for v in deck], collect_pose=[round(v, 3) for v in robot],
        reach_mag_m=round(mag_baselink, 3), reach_mag_armframe_m=round(mag_armbase, 3),
        max_reach_m=max_reach, reach_margin_m=round(margin, 3))

    # (3) CLEAR: the docked cart body clears the Support corridor, the source shelf and the robot.
    sup_c = staging['support_center_xy']; sup_d = staging['support_dimensions_m']
    support = (sup_c[0] - sup_d[0] / 2, sup_c[1] - sup_d[1] / 2, sup_c[0] + sup_d[0] / 2, sup_c[1] + sup_d[1] / 2)
    cart_box = _aabb(dock_x, dock_y, dock_yaw, spec.cart_length, spec.cart_width)
    robot_box = _aabb(robot[0], robot[1], robot[2], spec.length, spec.width)
    sup_ok = _clear(cart_box, support, 0.05)
    rbt_ok = _clear(cart_box, robot_box, 0.05)
    rec('dock_clears_support_corridor', sup_ok,
        cart_aabb=[round(v, 2) for v in cart_box], support_aabb=[round(v, 2) for v in support],
        separation_m=round(_sep(cart_box, support), 3))
    rec('dock_clears_robot_footprint', rbt_ok,
        separation_m=round(_sep(cart_box, robot_box), 3))

    # (4) COUPLE: PR 293 cart-relative standoff + straight reverse clears the static map south of the dock.
    grid = environment_grid(env, scenario, None)
    world = ArticulatedMap(grid, spec)
    seat = couple_seat_from_cart(dock, hitch_x=spec.hitch_x, cart_hitch_x=spec.cart_hitch_x)
    standoff_off = 0.40
    # PR 293 standoff_from_cart: on the cart hitch axis, offset back along +(cos cart_yaw, sin cart_yaw);
    # the base holds the rear hitch standoff_off short of the drawbar and creeps STRAIGHT in (reverses).
    sx = seat[0] + standoff_off * math.cos(dock_yaw)
    sy = seat[1] + standoff_off * math.sin(dock_yaw)
    reverse_clear = True
    for i in range(13):
        d = standoff_off * (1 - i / 12.0)                    # standoff -> seat (straight reverse)
        px = seat[0] + d * math.cos(dock_yaw); py = seat[1] + d * math.sin(dock_yaw)
        rect = _base_rect(spec, (px, py, seat[2]), margin=0.05)
        if any(polygon_hits_box(rect, ob) for ob in grid.obstacles):
            reverse_clear = False; break
    rec('couple_straight_reverse_clears', reverse_clear,
        seat=[round(v, 3) for v in seat], standoff=[round(sx, 3), round(sy, 3), round(seat[2], 3)],
        standoff_offset_m=standoff_off)

    # (4b) COLLECT-DRIVE ROUTE: the captain's step -- "move the robot in front of the rollwagen where
    # there is space to put the box into the rollwagen, then place". With reposition_before_place the base
    # drives from the FIXED collect pose to a place_approach in front of the south cart's open (+x body)
    # mouth, faces it, and lowers the carton. Prove: (i) the deck is reachable from place_approach, (ii)
    # place_approach + a staged reposition route clear the static map + the parked cart + the Support.
    from twin.restock.isaac_bundle import derive_place_approach
    pa_standoff = 1.0
    pa = derive_place_approach((deck[0], deck[1]), dock_yaw, pa_standoff,
                               mouth_axis_body=(1.0, 0.0), deck_center_body_xy=(0.0, 0.0))
    pa_reach = math.hypot(deck[0] - pa[0], deck[1] - pa[1])
    pa_reach = math.sqrt(pa_reach ** 2 + (deck[2] - arm_base_z) ** 2)
    pa_box = _aabb(pa[0], pa[1], pa[2], spec.length, spec.width)
    pa_static = not any(polygon_hits_box(_base_rect(spec, pa, 0.05), ob) for ob in grid.obstacles)
    pa_cart = _clear(pa_box, cart_box, 0.0)
    rec('collect_place_approach_reaches_deck', pa_reach <= max_reach,
        place_approach=[round(v, 3) for v in pa], standoff_m=pa_standoff,
        deck_reach_m=round(pa_reach, 3), max_reach_m=max_reach)
    rec('collect_place_approach_clears', pa_static and pa_cart,
        clears_static=pa_static, cart_separation_m=round(_sep(pa_box, cart_box), 3))
    # staged reposition route: collect -> (collect_x, pa_y) south -> place_approach east+face mouth.
    route_pts = [robot, (robot[0], pa[1], robot[2]), pa]
    route_clear = True
    min_clr = 9.9
    for a, b in zip(route_pts[:-1], route_pts[1:]):
        for i in range(21):
            t = i / 20.0
            x = a[0] + t * (b[0] - a[0]); y = a[1] + t * (b[1] - a[1])
            yaw = a[2] + t * wrap(b[2] - a[2])
            rect = _base_rect(spec, (x, y, yaw), 0.02)
            if any(polygon_hits_box(rect, ob) for ob in grid.obstacles):
                route_clear = False
            rb = _aabb(x, y, yaw, spec.length, spec.width)
            if not _clear(rb, cart_box, 0.0):
                route_clear = False
            else:
                min_clr = min(min_clr, _sep(rb, cart_box))
    rec('collect_reposition_route_clears', route_clear,
        route=[[round(c, 2) for c in p] for p in route_pts], min_cart_separation_m=round(min_clr, 3))

    # the lane start must equal the dock-derived couple seat (the coldwin23-2 frame-match guard).
    seat_match = (math.dist(seat[:2], outbound.start[:2]) <= outbound.start_tol_xy
                  and abs(wrap(seat[2] - outbound.start[2])) <= outbound.start_tol_yaw)
    rec('outbound_lane_start_follows_dock', seat_match,
        lane_start=[round(v, 3) for v in outbound.start], derived_seat=[round(v, 3) for v in seat])

    # (5) OUTBOUND: expand the dock lane and measure the gentle-curve metrics.
    try:
        route = outbound.expand(world, build_timeout=45.0)
    except Exception as exc:
        rec('outbound_lane_expands', False, error=str(exc)[:160])
        print('COLLECT_DOCK_REACH FAIL', flush=True); return 1
    rec('outbound_lane_expands', bool(route.states), states=len(route.states))
    max_art = max(abs(s.articulation) for s in route.states)
    peak = max(route.states, key=lambda s: abs(s.articulation))
    door = [s for s in route.states if abs(s.y - DOOR_Y) < 0.4]
    door_art = max((abs(s.articulation) for s in door), default=float('nan'))
    cross_x = door[0].x if door else float('nan')
    wclr = cross_x - spec.cart_width / 2 - PORTAL_X[0]
    eclr = PORTAL_X[1] - (cross_x + spec.cart_width / 2)
    # straight lead-in: span of near-aligned, near-straight states above the doorway.
    straight = [s.y for s in route.states if DOOR_Y - 0.2 <= s.y <= DOOR_Y + 5
                and abs(wrap(s.yaw - SOUTH)) < math.radians(4) and abs(s.articulation) < math.radians(5)]
    span = (max(straight) - min(straight)) if len(straight) > 1 else 0.0
    turn_in_open_backroom = peak.y > DOOR_Y + STRAIGHT_LEADIN_M
    rec('outbound_articulation_under_15deg', math.degrees(max_art) <= ART_MAX_DEG,
        whole_lane_max_deg=round(math.degrees(max_art), 2), peak_at_y=round(peak.y, 2),
        turn_in_open_backroom_away_from_door=bool(turn_in_open_backroom), target_deg=ART_MAX_DEG)
    rec('outbound_cart_aligned_at_doorway',
        math.degrees(door_art) <= DOORWAY_ALIGNED_DEG and PORTAL_X[0] <= cross_x <= PORTAL_X[1],
        cart_in_doorway_deg=round(math.degrees(door_art), 2), crossing_x=round(cross_x, 3),
        west_clearance_m=round(wclr, 3), east_clearance_m=round(eclr, 3), aligned_tol_deg=DOORWAY_ALIGNED_DEG)
    rec('outbound_straight_leadin_before_wall', span >= STRAIGHT_LEADIN_M,
        straight_leadin_m=round(span, 2), target_m=STRAIGHT_LEADIN_M)
    rec('doorway_crossing_clearance_each_side', wclr >= DOORWAY_CLEAR_M and eclr >= DOORWAY_CLEAR_M,
        west_clearance_m=round(wclr, 3), east_clearance_m=round(eclr, 3), target_m=DOORWAY_CLEAR_M)

    print('COLLECT_DOCK_REACH ' + ('PASS' if report['ok'] else 'FAIL'), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
