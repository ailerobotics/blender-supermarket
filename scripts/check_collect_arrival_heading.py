#!/usr/bin/env python3
"""sim-collect-arrival-heading-v1 offline gate (CPU-only): the collect place_approach standoff must be
reachable by a DRIVE that arrives already facing it -- the final heading must come from the lead-in, not
an in-place spin.

Why this exists. coldwin29's cold COLLECT drove to place_approach and then sat there ~160 s until the
300 s stage timeout, never entering PREPLACE. The mission's place_approach pose was authored for the
PRE-304 EAST-facing cart -- its standoff faces the mouth at heading pi. PR 304 rotated the dock to the
SOUTH-facing cart (mouth opens -y), whose standoff must face +pi/2. The stale pi heading is ~1.43 rad
from the heading the base reaches by driving south into the standoff (~-pi/2); the arrival test needs xy
AND yaw within tolerance at once, so the ~1.43 rad in-place rotation drifts the differential base off the
xy circle, the drive branch pulls it back, and the heading limit-cycles forever. check_collect_dock_reach
validates a swept FOOTPRINT of a route but never that the controller can CONVERGE on the standoff heading,
so the stale pose passed that gate. This script closes that gap.

It derives the reach-correct standoff (derive_place_approach from the dock's deck + heading) and the
spin-free lead-in (derive_place_approach_stage), then asserts:
  (A) SPIN-FREE  -- the final leg (stage -> place_approach) is a straight drive whose direction equals
                    place_approach's heading, so the base arrives aligned (no in-place spin).
  (B) REACH      -- the cart DECK place target is within the arm reach gate from the derived standoff.
  (C) AUTHORED   -- if --authored-place-approach x,y,yaw is given (the mission's current pose), its
                    heading matches the spin-free arrival heading; a stale heading FAILS here with the
                    measured error (this is the coldwin29 catch).
It then prints the corrected mission_poses (place_approach + place_approach_stage) for the integration lane.

Model is kinematic only (headings + reach), never a physics run.

Run:  .venv-usd/bin/python -I scripts/check_collect_arrival_heading.py \
        --scenario scripts/restock_scenario.json \
        --lanes docs/examples/cart-side-bay-mission-fragment.json \
        [--authored-place-approach=-4.5,15.4,3.14159265]
Exit 0 iff every claim passes (prints COLLECT_ARRIVAL_HEADING PASS/FAIL)."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*
from twin.navigation.lanes import TowLaneGraph
from twin.mobile import MobileSpec, wrap
from twin.restock.isaac_bundle import derive_place_approach, derive_place_approach_stage

YAW_MATCH_TOL = math.radians(10.0)   # a lead-in should deliver the heading well inside the 0.20 rad arrival tol


def _rot(yaw, v):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def _base_rect(spec, pose, margin=0.05):
    x, y, yaw = pose
    hl, hw = spec.length / 2 + margin, spec.width / 2 + margin
    return [(x + rx, y + ry) for rx, ry in
            (_rot(yaw, (hl, hw)), _rot(yaw, (hl, -hw)), _rot(yaw, (-hl, -hw)), _rot(yaw, (-hl, hw)))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', type=Path, default=Path('scripts/restock_scenario.json'))
    ap.add_argument('--lanes', type=Path, required=True,
                    help='mission fragment with navigation.tow_lanes carrying the dock as couple_dock')
    ap.add_argument('--lane-name', default=None)
    ap.add_argument('--standoff', type=float, default=1.0, help='arm-reach standoff out the cart mouth (m)')
    ap.add_argument('--lead-in', type=float, default=0.6, help='straight lead-in before the standoff (m)')
    ap.add_argument('--environment', type=Path, default=None,
                    help='optional environment manifest: when given, sweep the lead-in route footprint '
                         'against the real static map (route clearance for the new lead-in).')
    ap.add_argument('--authored-place-approach', default=None,
                    help='x,y,yaw of the mission\'s CURRENT place_approach pose to audit (optional)')
    ap.add_argument('--authored-stage', default=None,
                    help='x,y,yaw of the mission\'s CURRENT place_approach_stage (the waypoint BEFORE '
                         'place_approach). Defaults to the legacy corner formula (pa.x, pickup.y, pickup.yaw) '
                         'when --authored-place-approach is given without it.')
    args = ap.parse_args(argv)

    scenario = json.loads(args.scenario.read_text())
    cfg = json.loads(args.lanes.read_text())
    st = scenario['initial_staging']
    pickup = tuple(float(v) for v in st['robot_base_pose'])
    load_center = st['cart_load_center_body_xy']
    deck_z = float(st['cart_support_z_world'])
    max_reach = float(scenario['handling_unit']['max_reach_m'])
    arm_base_z = 0.5  # arm mount height (mobile manifest spec.arm_mount_z default; reach is slack at ~0.5 m)

    graph = TowLaneGraph.from_config(cfg['navigation']['tow_lanes'])
    outbound = None
    for ln in graph.lanes:
        if args.lane_name and ln.name == args.lane_name:
            outbound = ln; break
        if 'return' not in ln.name and outbound is None:
            outbound = ln
    if outbound is None or outbound.couple_dock is None:
        print('COLLECT_ARRIVAL_HEADING FAIL (no outbound lane / couple_dock in the fragment)'); return 1
    dock_x, dock_y, dock_yaw = outbound.couple_dock

    report = {'ok': True, 'checks': []}
    def rec(name, passed, **kw):
        report['ok'] = report['ok'] and bool(passed)
        report['checks'].append((name, bool(passed), kw))
        print(('PASS ' if passed else 'FAIL ') + name + ' ' + json.dumps(kw), flush=True)

    # Deck load-centre (world) from the dock + the cart-body load centre.
    dcx, dcy = _rot(dock_yaw, (load_center[0], load_center[1]))
    deck = (dock_x + dcx, dock_y + dcy)

    pa = derive_place_approach(deck, dock_yaw, args.standoff,
                               mouth_axis_body=(1.0, 0.0), deck_center_body_xy=(0.0, 0.0))
    stage = derive_place_approach_stage(pa, pickup, lead_in=args.lead_in)

    # (A) SPIN-FREE: the final leg direction == place_approach heading (arrives aligned).
    drive_in = math.atan2(pa[1] - stage[1], pa[0] - stage[0])
    spin = abs(wrap(pa[2] - drive_in))
    rec('final_leg_arrives_aligned', spin <= 1e-6,
        place_approach=[round(v, 4) for v in pa], lead_in_stage=[round(v, 4) for v in stage],
        final_leg_drive_in_rad=round(drive_in, 4), in_place_spin_rad=round(spin, 6))

    # (B) REACH: the deck within the arm reach gate from the derived standoff.
    reach = math.sqrt((deck[0] - pa[0]) ** 2 + (deck[1] - pa[1]) ** 2 + (deck_z - arm_base_z) ** 2)
    rec('deck_within_reach_from_standoff', reach <= max_reach,
        deck_world=[round(deck[0], 3), round(deck[1], 3), round(deck_z, 3)],
        standoff_m=args.standoff, deck_reach_m=round(reach, 3), max_reach_m=max_reach)

    # (C) AUDIT the authored pose (optional): the heading the base reaches driving the ACTUAL configured
    # final leg (authored stage -> authored place_approach) must match the authored place_approach heading.
    # This is the coldwin29 catch: the base drives the configured leg and arrives facing the DRIVE
    # direction, not the authored heading, so a mismatch is an unreachable in-place spin.
    if args.authored_place_approach:
        ax, ay, ayaw = (float(v) for v in args.authored_place_approach.split(','))
        if args.authored_stage:
            asx, asy, _asyaw = (float(v) for v in args.authored_stage.split(','))
        else:
            # the legacy corner-route staging the mission carried pre-fix: (pa.x, pickup.y, pickup.yaw)
            asx, asy = ax, pickup[1]
        a_drive_in = math.atan2(ay - asy, ax - asx)
        a_err = abs(wrap(ayaw - a_drive_in))
        rec('authored_place_approach_heading_is_spin_free', a_err <= YAW_MATCH_TOL,
            authored_place_approach=[round(ax, 4), round(ay, 4), round(ayaw, 4)],
            authored_stage=[round(asx, 4), round(asy, 4)],
            arrives_facing_rad=round(a_drive_in, 4), in_place_spin_rad=round(a_err, 4),
            spin_deg=round(math.degrees(a_err), 2), tol_deg=round(math.degrees(YAW_MATCH_TOL), 1))

    # (D) ROUTE CLEARANCE (optional): sweep the lead-in route footprint (pickup -> stage -> place_approach)
    # against the real ~800-obstacle static map, so the new, further-south lead-in is proven clear.
    if args.environment is not None:
        from twin.navigation.scenario import environment_grid
        from twin.navigation.articulated import polygon_hits_box
        spec = MobileSpec()
        grid = environment_grid(json.loads(args.environment.read_text()), scenario, None)
        route = [pickup, stage, pa]
        clear = True
        for a, b in zip(route[:-1], route[1:]):
            for i in range(21):
                t = i / 20.0
                x = a[0] + t * (b[0] - a[0]); y = a[1] + t * (b[1] - a[1])
                yaw = a[2] + t * wrap(b[2] - a[2])
                if any(polygon_hits_box(_base_rect(spec, (x, y, yaw), 0.05), ob) for ob in grid.obstacles):
                    clear = False
        rec('lead_in_route_clears_static_map', clear,
            route=[[round(c, 3) for c in p] for p in route], n_obstacles=len(grid.obstacles))

    corrected = {'mission_poses': {'place_approach': [round(v, 4) for v in pa],
                                   'place_approach_stage': [round(v, 4) for v in stage]}}
    print('CORRECTED_MISSION_POSES ' + json.dumps(corrected), flush=True)
    print('COLLECT_ARRIVAL_HEADING ' + ('PASS' if report['ok'] else 'FAIL'), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
