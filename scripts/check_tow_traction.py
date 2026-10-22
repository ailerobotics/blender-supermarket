#!/usr/bin/env python3
"""Offline proof (CPU-only) for sim-robust-coupled-traction-v1: the coupled outbound tow's stall at
x~=-3.93 is a PhysX DRIVE-AUTHORITY deficit, NOT geometry, and names the acceptance gate the bounded GPU
proof must clear. Driven by the REAL controller/planner against the REAL mission inputs (environment
manifest + scenario + camera config -> the full ~800-obstacle set) and the REAL recorded nav-traces of
the stalled (outbound1/warm9/warm10) and clean (warm8/warm11) runs. No GPU, no Isaac.

Checks (exit 0 iff ALL pass; prints TOW_TRACTION_PROOF PASS/FAIL):

  (1) GEOMETRY CLEAR. The outbound_dock_to_bay lane expands against the static map, its swept robot+cart
      envelope re-validates clear against the full LIVE obstacle cloud, and the REAL ArticulatedController
      follows it closed-loop to the goal in the kinematic (no-slip) model. Reaching the goal here while the
      GPU run stalls isolates the stall to sim physics, not geometry or the controller logic.

  (2) RECORDED FOOTPRINTS CLEAR. Every recorded robot+cart footprint of BOTH the stalled and the clean
      runs -- including the exact x~=-3.93 stall pose -- is collision-free against the static map with the
      mandatory 0.05 m margin. There is nothing to clip at the stall.

  (3) TRACTION MARGIN (the acceptance gate, scripts/twin/traction.coupled_tow_margin). The stalled runs
      FAIL the gate (never reached goal; pinned many sim seconds) and the clean runs PASS it (reached goal;
      no multi-second pin). This reproduces the stall and defines PASS/FAIL for a new run.

  (4) LEVER DIRECTION (scripts/twin/traction.static_traction_budget). The recommended physical config
      (caster lift 0.03 + drive force 200 N.m) strictly dominates the stuck config (0.005 / 80 N.m) in the
      quasi-static force budget -- confirming the levers' sign. Absolute margin is NOT an efficacy claim;
      the real (dynamic re-breakaway) efficacy is the bounded GPU bench.

Also: --gpu-run <run-dir> grades a SINGLE new nav-trace against gate (3) and prints GPU_RUN PASS/FAIL, so
the integration lane runs the SAME gate on its bounded couple+outbound proof.

Run:  python3 scripts/check_tow_traction.py \
        --environment ~/datasets/sim-store-realism-v2/environment.json \
        --scenario .../mission-scenario.json --camera-config .../camera-config-bay.json \
        --fragment docs/examples/coupled-traction-fragment.json \
        --runs-dir ~/datasets/twin-robot-in-store-boot-v1/mission \
        --stalled outbound1 warm9 warm10 --clean warm8 warm11 [--gpu-run <dir>] [--json out.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.* and sibling checks
from twin.navigation.scenario import environment_grid
from twin.navigation.lanes import TowLaneGraph
from twin.navigation.articulated import ArticulatedController, ArticulatedMap, ArticulatedPlanner, TrainState
from twin.mobile import MobileSpec
from twin.traction import coupled_tow_margin, static_traction_budget
from check_tow_lanes import _obstacle_points, _drive_lane, _record


def _outbound_rows(run_dir):
    rows = []
    path = Path(run_dir) / 'nav-trace.jsonl'
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue                                   # tolerate a truncated final line
        if d.get('action') == 'outbound':
            rows.append(d)
    return rows


def _bay_lane(fragment):
    nav = fragment['navigation']
    graph = TowLaneGraph.from_config(nav['tow_lanes'])
    lane = next((ln for ln in graph.lanes if 'outbound' in ln.name), graph.lanes[0])
    return graph, lane


# ---- (1) geometry clear -------------------------------------------------------------------------

def check_geometry_clear(grid, fragment, report):
    spec = MobileSpec()
    world = ArticulatedMap(grid, spec)
    graph, lane = _bay_lane(fragment)
    graph.expand_all(world)
    _record(report, 'bay_lane_expands_against_static_map', lane.route is not None,
            lane=lane.name)
    if lane.route is None:
        return
    live = _obstacle_points(grid)
    report['live_obstacle_points'] = len(live)
    planner = ArticulatedPlanner(world)
    chk = planner.validate_seed(lane.route, lane.start_state(), lane.goal, live,
                                0.05, lane.goal_articulation)
    _record(report, 'bay_lane_swept_envelope_clear_live', bool(chk.states),
            reason=chk.reason, states=len(chk.states))
    # The real controller follows the lane closed-loop (kinematic, no slip) -> isolates the stall to physics.
    controller = ArticulatedController(grid, spec, allow_test_fixtures=True,
                                       goal_timeout=600.0, blocked_timeout=60.0, lane_graph=graph)
    direct = math.dist(lane.start[:2], lane.goal[:2])
    _drive_lane(controller, spec, lane, lane.goal, jitter=(0.06, -0.05, 0.03, 0.04),
                report=report, label=lane.name, direct_xy=direct)


# ---- (2) recorded footprints clear --------------------------------------------------------------

def check_recorded_footprints_clear(grid, runs_dir, runs, report):
    spec = MobileSpec()
    world = ArticulatedMap(grid, spec)
    for run in runs:
        rows = _outbound_rows(Path(runs_dir) / run)
        if not rows:
            _record(report, f'recorded_footprints_clear[{run}]', False, reason='no_outbound_rows')
            continue
        collisions = 0
        first_bad = None
        for d in rows:
            yaw = math.radians(d['yaw_deg'])
            cart = d.get('cart')
            cart_yaw = math.radians(cart[2]) if cart else yaw
            st = TrainState(d['x'], d['y'], yaw, cart_yaw)
            if not world.valid(st, (), 0.0):
                collisions += 1
                if first_bad is None:
                    first_bad = (round(d['x'], 3), round(d['y'], 3))
        _record(report, f'recorded_footprints_clear[{run}]', collisions == 0,
                ticks=len(rows), collisions=collisions, first_bad=first_bad)


# ---- (3) traction margin (the acceptance gate) --------------------------------------------------

def _margin(run_dir, goal):
    return coupled_tow_margin(_outbound_rows(run_dir), goal)


def check_traction_margin(runs_dir, stalled, clean, goal, report):
    for run in stalled:
        m = _margin(Path(runs_dir) / run, goal)
        _record(report, f'stalled_run_fails_gate[{run}]', not m.passed,
                reached_goal=m.reached_goal, longest_pin_s=m.longest_pin_s,
                max_advance_m=m.max_advance_m, reason=m.reason)
    for run in clean:
        m = _margin(Path(runs_dir) / run, goal)
        _record(report, f'clean_run_passes_gate[{run}]', m.passed,
                reached_goal=m.reached_goal, longest_pin_s=m.longest_pin_s,
                max_advance_m=m.max_advance_m, reason=m.reason)


# ---- (4) lever direction ------------------------------------------------------------------------

def check_lever_direction(fragment, report):
    cfg = fragment['_traction_budget']
    stuck = static_traction_budget(**cfg['stuck'])
    rec = static_traction_budget(**cfg['recommended'])
    _record(report, 'recommended_config_dominates_stuck_margin', rec['margin'] > stuck['margin'],
            stuck_margin=stuck['margin'], recommended_margin=rec['margin'],
            stuck_share=stuck['drive_wheel_share'], recommended_share=rec['drive_wheel_share'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--environment', type=Path, required=True)
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--camera-config', type=Path, required=True)
    ap.add_argument('--fragment', type=Path, required=True,
                    help='docs/examples/coupled-traction-fragment.json (lanes + budget inputs)')
    ap.add_argument('--runs-dir', type=Path, required=True)
    ap.add_argument('--stalled', nargs='+', default=['outbound1', 'warm9', 'warm10'])
    ap.add_argument('--clean', nargs='+', default=['warm8', 'warm11'])
    ap.add_argument('--gpu-run', type=Path, help='grade a single new nav-trace against the acceptance gate')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()

    fragment = json.loads(args.fragment.read_text())
    goal = fragment['navigation']['tow_lanes'][0]['goal']

    # --gpu-run: grade one run against the acceptance gate and exit (the integration lane's PASS/FAIL).
    if args.gpu_run is not None:
        m = _margin(args.gpu_run, goal)
        verdict = 'PASS' if m.passed else 'FAIL'
        out = dict(run=str(args.gpu_run), passed=m.passed, reached_goal=m.reached_goal,
                   longest_pin_s=m.longest_pin_s, max_advance_m=m.max_advance_m,
                   drive_cruise_mps=m.drive_cruise_mps, reason=m.reason)
        print('GPU_RUN ' + verdict + ' ' + json.dumps(out), flush=True)
        if args.json:
            args.json.write_text(json.dumps(out, indent=2) + '\n')
        return 0 if m.passed else 1

    manifest = json.loads(args.environment.read_text())
    scenario = json.loads(args.scenario.read_text())
    camera = json.loads(args.camera_config.read_text())
    grid = environment_grid(manifest, scenario, camera)

    report = {'ok': True, 'checks': [], 'static_obstacles': len(grid.obstacles), 'goal': goal}
    print(f'grid: {len(grid.obstacles)} static obstacles, bounds {grid.bounds}', flush=True)
    check_geometry_clear(grid, fragment, report)
    check_recorded_footprints_clear(grid, args.runs_dir, list(args.stalled) + list(args.clean), report)
    check_traction_margin(args.runs_dir, args.stalled, args.clean, goal, report)
    check_lever_direction(fragment, report)

    print(('TOW_TRACTION_PROOF PASS' if report['ok'] else 'TOW_TRACTION_PROOF FAIL')
          + f" ({sum(c['passed'] for c in report['checks'])}/{len(report['checks'])} checks)", flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
