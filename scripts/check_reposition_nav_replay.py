#!/usr/bin/env python3
"""fm392 proof (c): replay a reposition-leg obstacle set through the PRODUCTION uncoupled planner and
report whether a route to the staging waypoint + place_approach exists -- with and without the candidate
fixes. Reproduces run-303's 'blocked_no_route' offline so a fix is proven on CPU before run-305.

The uncoupled reposition uses twin.navigation.core.GridMap.plan/free (ArticulatedController delegates to
its base Controller when not coupled). This harness builds the REAL nav grid + footprint radius from the
mission runtime-config (navigation block), then evaluates free(start)/free(goal)/segment_free/plan against
an injected dynamic-obstacle set -- the live lidar returns that the static grid lacks. Populate the
obstacle set from the run's NAV_DIAG (the nearest returns to the base) or an explicit JSON.

Candidate fixes (both sim-only, byte-identical off in the mission):
  * mask_radius R : drop injected obstacle points within R m of the base pose -- models correctly masking
    a self-return (folded arm / carried carton / gripper) that leaked into obs.obstacles at the base.
  * drop_unknown  : (placeholder) exclude points flagged 'unknown' in the injected set -- models treating
    self-occluded/unknown returns as free for the reposition leg's route.

NOT a physics or sensor run; a geometry replay of the production planner.

Run:  .venv-usd/bin/python -I scripts/check_reposition_nav_replay.py \
        --runtime-config <mission-runtime-cc-collect.json> [--obstacles pts.json] [--mask-radius 0.75]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*
from twin.navigation.factory import _build_grid, _build_spec  # noqa: E402
from twin.navigation.articulated import ArticulatedController  # noqa: E402


def _filter(points, pose, *, mask_radius=0.0, drop_unknown=False):
    """Apply the candidate fixes to an injected obstacle set. Each point is (x, y) or (x, y, unknown)."""
    out = []
    for p in points:
        x, y = p[0], p[1]
        unknown = bool(p[2]) if len(p) > 2 else False
        if drop_unknown and unknown:
            continue
        if mask_radius > 0 and math.hypot(x - pose[0], y - pose[1]) <= mask_radius:
            continue
        out.append((x, y))
    return out


def evaluate(grid, radius, pose, goal, obstacles):
    free_start = grid.free(pose, radius, obstacles)
    free_goal = grid.free(goal, radius, obstacles)
    seg = grid.segment_free(pose, goal, radius, obstacles)
    path = grid.plan(pose, goal, radius, obstacles)
    return dict(free_start=free_start, free_goal=free_goal, segment_free=seg,
                route_found=bool(path), waypoints=len(path) if path else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime-config', type=Path, required=True)
    ap.add_argument('--obstacles', type=Path,
                    help='JSON list of [x,y] or [x,y,unknown] dynamic obstacle points (default: none)')
    ap.add_argument('--pose', type=float, nargs=3, default=[-5.73, 16.0, 0.0], help='base x y yaw')
    ap.add_argument('--stage', type=float, nargs=3, default=[-4.5, 16.0, 0.0])
    ap.add_argument('--place', type=float, nargs=3, default=[-4.5, 15.25, math.pi])
    ap.add_argument('--mask-radius', type=float, default=0.0, help='fix: drop obstacle points within R m of the base')
    ap.add_argument('--drop-unknown', action='store_true', help='fix: drop points flagged unknown')
    ap.add_argument('--position-uncertainty', type=float, default=0.05)
    a = ap.parse_args()
    cfg = json.loads(a.runtime_config.read_text())
    nav = cfg['navigation']
    grid = _build_grid(nav)
    spec = _build_spec(nav)
    ctl = ArticulatedController(grid, spec)
    radius = ctl.footprint.radius(False) + a.position_uncertainty
    raw = json.loads(a.obstacles.read_text()) if a.obstacles else []
    pose, stage, place = tuple(a.pose), tuple(a.stage), tuple(a.place)

    report = {'footprint_radius': round(radius, 4), 'n_raw_obstacles': len(raw),
              'legs': {}, 'fix': {'mask_radius': a.mask_radius, 'drop_unknown': a.drop_unknown}}
    for label, (p, g) in (('pickup->stage', (pose, stage)),
                          ('stage->place_approach', (stage, place)),
                          ('pickup->place_approach(direct)', (pose, place))):
        before = evaluate(grid, radius, p, g, [(q[0], q[1]) for q in raw])
        fixed_obs = _filter(raw, p, mask_radius=a.mask_radius, drop_unknown=a.drop_unknown)
        after = evaluate(grid, radius, p, g, fixed_obs)
        report['legs'][label] = {'before_fix': before, 'after_fix': after,
                                 'n_obs_after': len(fixed_obs)}
    print(json.dumps(report, indent=2))
    staged = report['legs']['pickup->stage']['after_fix']['route_found'] and \
        report['legs']['stage->place_approach']['after_fix']['route_found']
    print('\nSTAGED REPOSITION ROUTE (after fix) =', 'FOUND' if staged else 'NOT FOUND')
    return 0 if staged else 1


if __name__ == '__main__':
    raise SystemExit(main())
