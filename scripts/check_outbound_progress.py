#!/usr/bin/env python3
"""Offline proof (CPU-only) for sim-robust-outbound-drive-v1: the coupled outbound tow no longer dies on a
no-base-progress stall. Driven by the RECORDED nav-trace.jsonl of the warm runs (no GPU, no Isaac).

THE STALL (warm9/warm10). The coupled outbound tow pinned at x~=-3.9 (y=15.25) while the follower kept
COMMANDING forward -- reason='tracking' on most NAV_STEP ticks, only a few 'measured_turn_radius' -- and the
base jittered in place: measured |rv| noise to ~0.23 m/s but ~0 NET displacement. Under ~20x render pacing
that frozen window spanned ~14 s SIM but ~285 s WALL, so the WALL-clock no-base-progress budget
(restock.runtime `_nav_stuck_check`, 120 s) failed the stage -- on the wrong clock, after burning GPU wall.
coupled_creep_escape (fm482) missed it (its trigger needs |odom.linear| inside a narrow noise band the
jitter spikes leave) and measured_turn_fault_s (fm446) missed it (it accumulates only on measured_turn_radius
COASTS, not the dominant 'tracking' ticks).

THE FIX (navigation.coupled_progress_escape, default off => byte-identical). A SIM-CLOCK net-progress monitor
in the ArticulatedController: while the coupled base commands forward but has not moved >= min_m from its
anchor for escape_s SIM seconds it FORCES the committed creep, and after fault_s SIM seconds of no net
progress it FAULTS 'coupled_no_progress' so the stage StageRetry-replans -- in a few SIM seconds, before (and
on the correct clock for) the 120 s WALL budget. This module measures, straight off the RECORDED traces, the
longest no-progress window per run in BOTH clocks, and reports when the monitor would have fired.

The window math here is a faithful mirror of ArticulatedController.step's monitor (anchor on net
displacement >= min_m; otherwise accumulate SIM time since the anchor). It is exercised against the synthetic
warm10 signature by tests/test_articulated_navigation.py (the REAL controller); this script grounds the same
logic in the REAL recorded runs.

Run:  python3 scripts/check_outbound_progress.py --runs-dir <.../mission> \\
        --stalled warm9 warm10 --clean warm8 warm11 [--min-m 0.05 --escape-s 3.0 --fault-s 8.0] [--json out]
Exit 0 iff every stalled run would fire the monitor AND every clean run would not (prints
OUTBOUND_PROGRESS_PROOF PASS/FAIL)."""
import argparse
import json
import math
import sys
from pathlib import Path


def _read_outbound(run_dir):
    """Return the outbound ticks of a run's nav-trace.jsonl (tolerates a truncated final line)."""
    rows = []
    path = Path(run_dir) / 'nav-trace.jsonl'
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue   # a run killed mid-write leaves one partial final line
            if r.get('action') == 'outbound':
                rows.append(r)
    return rows


def analyze(rows, *, min_m, escape_s, fault_s):
    """Longest no-progress window (the controller's anchor logic) over the recorded outbound ticks.

    Returns the max stuck SIM seconds, the WALL seconds that same window spanned, and the first SIM time the
    monitor would escalate (escape_s) / fault (fault_s), plus the WALL time at the fault (to show how much
    earlier-and-on-the-right-clock it fires vs the 120 s WALL budget)."""
    anchor = None                         # (sim, wall, x, y)
    max_stuck_sim = max_stuck_wall = 0.0
    escalate_sim = fault_sim = fault_wall = None
    for r in rows:
        sim, wall, x, y = r['world_t'], r['wall'], r['x'], r['y']
        if anchor is None:
            anchor = (sim, wall, x, y)
            continue
        a_sim, a_wall, ax, ay = anchor
        if math.hypot(x - ax, y - ay) >= min_m:
            anchor = (sim, wall, x, y)     # real net progress -> re-anchor
            continue
        stuck_sim = max(0.0, sim - a_sim)
        stuck_wall = max(0.0, wall - a_wall)
        if stuck_sim > max_stuck_sim:
            max_stuck_sim, max_stuck_wall = stuck_sim, stuck_wall
        if escalate_sim is None and stuck_sim >= escape_s:
            escalate_sim = stuck_sim
        if fault_sim is None and stuck_sim >= fault_s:
            fault_sim, fault_wall = stuck_sim, stuck_wall
    return dict(ticks=len(rows), max_stuck_sim=round(max_stuck_sim, 2),
                max_stuck_wall=round(max_stuck_wall, 1),
                would_escalate=escalate_sim is not None,
                would_fault=fault_sim is not None,
                fault_at_sim=None if fault_sim is None else round(fault_sim, 2),
                fault_at_wall=None if fault_wall is None else round(fault_wall, 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs-dir', required=True, help='mission dataset dir holding the warm*/ run dirs')
    ap.add_argument('--stalled', nargs='+', default=['warm9', 'warm10'],
                    help='runs that DID stall (the monitor must fire)')
    ap.add_argument('--clean', nargs='+', default=['warm8', 'warm11'],
                    help='runs that towed cleanly (the monitor must NOT fire)')
    ap.add_argument('--min-m', type=float, default=0.05)
    ap.add_argument('--escape-s', type=float, default=3.0)
    ap.add_argument('--fault-s', type=float, default=8.0)
    ap.add_argument('--wall-budget-s', type=float, default=120.0,
                    help='the restock.runtime WALL no-base-progress budget, for the comparison')
    ap.add_argument('--json', help='write the full report here')
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    report = {'params': dict(min_m=args.min_m, escape_s=args.escape_s, fault_s=args.fault_s),
              'stalled': {}, 'clean': {}, 'checks': []}
    ok = True

    def grade(label, name, want_fault):
        nonlocal ok
        rows = _read_outbound(runs_dir / name)
        if not rows:
            report['checks'].append(f'{name}: NO outbound ticks in nav-trace (SKIP/FAIL)')
            ok = False
            return
        res = analyze(rows, min_m=args.min_m, escape_s=args.escape_s, fault_s=args.fault_s)
        report[label][name] = res
        passed = (res['would_fault'] == want_fault)
        ok = ok and passed
        verb = 'fires' if res['would_fault'] else 'silent'
        detail = (f"stuck {res['max_stuck_sim']}s SIM / {res['max_stuck_wall']}s WALL; monitor {verb}")
        if res['would_fault']:
            detail += (f" (fault at {res['fault_at_sim']}s SIM = {res['fault_at_wall']}s WALL, "
                       f"vs {args.wall_budget_s:g}s WALL budget)")
        report['checks'].append(f"{'PASS' if passed else 'FAIL'} {name} (want fault={want_fault}): {detail}")

    for name in args.stalled:
        grade('stalled', name, True)
    for name in args.clean:
        grade('clean', name, False)

    report['verdict'] = 'PASS' if ok else 'FAIL'
    for line in report['checks']:
        print(line)
    print(f'OUTBOUND_PROGRESS_PROOF {report["verdict"]}')
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
