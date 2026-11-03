#!/usr/bin/env python3
"""Offline proof (CPU-only) for sim-collect-reposition-block-v1, driven by a RECORDED run (no GPU, Isaac).

THE FAILURE (boxc1a, 2026-10-25, collect-only). The collect "drive->rollwagen" reposition to
place_approach_stage (-4.5, 16.0) failed with blocked_no_route_timeout. The brief framed it as a ~15 deg
base YAW shove during the grasp; the recorded ticks say otherwise:

  * During the GRASP (collect step 2, grasp.lift.tuck) the base is rock-steady -- the pre-weld full base
    brake holds it (boxc1a: max |yaw| 0.2 deg, ~0 xy). The "15" was the planner's in_start=15 obstacle-
    POINT count inside the footprint disk, misread as degrees.
  * During the REPOSITION (step 3, drive->rollwagen) the loaded base DRIFTS ~0.28 m NORTH + ~0.13 m WEST
    off its clean departure pose into the LIVE backstock Support, because every blocked/stopped tick issued
    mobile.command(0,0,...) which RELEASES the base brake "to resume driving" even with no route; the
    cantilevered carried carton then rolls the un-pinned base. The drifted footprint overlaps the Support
    (free_start=False) -> permanent blocked_no_route.

THIS CHECK reads a run's mission.json (stage_timeline) + nav-trace.jsonl and measures, over the first
collect attempt, (1) the max base yaw drift during the grasp window and (2) the max base xy drift during
the reposition window. It PASSES iff the grasp yaw stayed within --grasp-yaw-bound-deg (confirms the base
was held during contact) AND the reposition base drift stayed within --reposition-drift-bound-m (the fix's
pass criterion: with hold_base_on_reposition_block on, the base no longer drifts off the departure pose).

On the unfixed boxc1a it PASSES the grasp bound and FAILS the reposition bound -- flagging exactly the bug.
It is the grader the integration lane points at a fixed collect-only GPU run.

Run:  python3 scripts/check_reposition_base_drift.py --run <run-dir> \\
        [--grasp-yaw-bound-deg 2.0 --reposition-drift-bound-m 0.10] [--json out.json]
Exit 0 iff both bounds hold (prints REPOSITION_BASE_DRIFT PASS/FAIL)."""
import argparse
import json
import math
from pathlib import Path


def _load_timeline(mission):
    """Return (grasp_t0, reposition_t0, reposition_t1) for the FIRST collect attempt, in mission time.

    grasp_t0  = begin of step 2 (grasp.lift.tuck); reposition_t0 = begin of step 3 (drive->rollwagen);
    reposition_t1 = the next stage_failed/stage_begin after reposition_t0 (end of that reposition)."""
    tl = mission.get('stage_timeline') or []
    grasp_t0 = repos_t0 = repos_t1 = None
    for ev in tl:
        d = ev.get('data', {})
        if ev.get('type') == 'stage_step' and d.get('stage') == 'collect' and d.get('attempt') == 1:
            if d.get('step') == 2 and grasp_t0 is None:
                grasp_t0 = float(ev.get('t'))
            if d.get('step') == 3 and repos_t0 is None:
                repos_t0 = float(ev.get('t'))
    if repos_t0 is not None:
        for ev in tl:
            t = float(ev.get('t'))
            if t > repos_t0 and ev.get('type') in ('stage_failed', 'stage_begin'):
                repos_t1 = t
                break
    return grasp_t0, repos_t0, repos_t1


def _rows(nav_trace_path, t0, t1):
    out = []
    with open(nav_trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get('t')
            if t is None or not (t0 <= t <= t1):
                continue
            out.append((float(t), float(d['x']), float(d['y']), float(d['yaw_deg'])))
    return out


def _yaw_drift_deg(rows):
    if not rows:
        return 0.0
    y0 = rows[0][3]
    return max(abs(r[3] - y0) for r in rows)


def _xy_drift_m(rows):
    if not rows:
        return 0.0, 0.0, 0.0
    x0, y0 = rows[0][1], rows[0][2]
    dxy = max(math.hypot(r[1] - x0, r[2] - y0) for r in rows)
    dnorth = max(r[2] - y0 for r in rows)            # +y toward the backstock Support
    dwest = max(x0 - r[1] for r in rows)             # -x away from the +x goal
    return dxy, dnorth, dwest


def analyze(run_dir, grasp_yaw_bound_deg, reposition_drift_bound_m):
    run = Path(run_dir)
    mission = json.loads((run / 'mission.json').read_text())
    g0, r0, r1 = _load_timeline(mission)
    if g0 is None or r0 is None or r1 is None:
        raise SystemExit(f'could not locate the collect grasp/reposition steps in {run}/mission.json '
                         f'(grasp_t0={g0} reposition_t0={r0} reposition_t1={r1})')
    nav = run / 'nav-trace.jsonl'
    grasp_rows = _rows(nav, g0, r0)
    repos_rows = _rows(nav, r0, r1)
    grasp_yaw = _yaw_drift_deg(grasp_rows)
    repos_dxy, repos_north, repos_west = _xy_drift_m(repos_rows)
    repos_yaw = _yaw_drift_deg(repos_rows)
    grasp_ok = grasp_yaw <= grasp_yaw_bound_deg
    repos_ok = repos_dxy <= reposition_drift_bound_m
    return {
        'run': str(run),
        'reason': mission.get('reason'),
        'windows_mission_time': {'grasp': [g0, r0], 'reposition': [r0, r1]},
        'grasp': {'yaw_drift_deg': round(grasp_yaw, 3), 'bound_deg': grasp_yaw_bound_deg,
                  'within_bound': grasp_ok, 'ticks': len(grasp_rows)},
        'reposition': {'base_xy_drift_m': round(repos_dxy, 4), 'north_m': round(repos_north, 4),
                       'west_m': round(repos_west, 4), 'yaw_drift_deg': round(repos_yaw, 3),
                       'bound_m': reposition_drift_bound_m, 'within_bound': repos_ok,
                       'ticks': len(repos_rows)},
        'pass': bool(grasp_ok and repos_ok),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='run dir with mission.json + nav-trace.jsonl')
    ap.add_argument('--grasp-yaw-bound-deg', type=float, default=2.0)
    ap.add_argument('--reposition-drift-bound-m', type=float, default=0.10)
    ap.add_argument('--json', type=Path, help='also write the report JSON here')
    a = ap.parse_args()
    report = analyze(a.run, a.grasp_yaw_bound_deg, a.reposition_drift_bound_m)
    print(json.dumps(report, indent=2))
    if a.json:
        a.json.write_text(json.dumps(report, indent=2))
    ok = report['pass']
    print('\nREPOSITION_BASE_DRIFT %s: grasp yaw %.2f deg (<=%.2f: %s); reposition base drift %.3f m '
          '(<=%.2f: %s, north %.3f m)' % (
              'PASS' if ok else 'FAIL', report['grasp']['yaw_drift_deg'], a.grasp_yaw_bound_deg,
              report['grasp']['within_bound'], report['reposition']['base_xy_drift_m'],
              a.reposition_drift_bound_m, report['reposition']['within_bound'],
              report['reposition']['north_m']))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
