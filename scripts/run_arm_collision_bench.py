#!/usr/bin/env python3
"""Arm collision-avoidance evaluation bench -- CLI (sim-arm-collision-bench-v1).

CPU mock-ESDF mode (the default; no Isaac/ROS/GPU): score a declarative scenario set against an
arm-collision GATE and write the one-page markdown + CSV + a pass/fail verdict.

    PYTHONPATH=scripts python3 scripts/run_arm_collision_bench.py \
        --scenarios docs/examples/arm-collision-scenarios \
        --gate baseline \
        --out /home/manishsaini/firstmate/data/sim-arm-collision-bench-v1/before-baseline

Gates:
  * ``baseline``   -- the CURRENT stack (no perceived-ESDF gate on these arm legs): the BEFORE numbers.
  * ``reference``  -- the bench's transparent collision-aware reference instrument (what the sibling
                      lanes target).
  * ``module:function`` -- a sibling lane's real gate (the AFTER column) once it is importable.

``--observe-all`` models the perception coverage the sibling features add (mast at collect/carry, the
cart as a world obstacle) -- the AFTER-coverage column, answering "once the stack SEES these obstacles,
does the gate make the right call?" without fabricating a planner.

Exit code is 0 iff every scenario passed its thresholds (so CI/a pre-send gate can use it).  The GPU
mode -- booting one bounded mission segment with the same obstacles spawned and reading the live
contact reports / ESDF -- lives in ``scripts/twin/tools/arm_collision_scene.py`` +
``scripts/run_arm_collision_bench_gpu.py`` and emits the SAME schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ on the path for `twin`

from twin.tools import arm_collision_bench as bench  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenarios', type=Path,
                    default=Path(__file__).resolve().parent.parent / 'docs/examples/arm-collision-scenarios',
                    help='directory of scenario JSON files (default: docs/examples/arm-collision-scenarios)')
    ap.add_argument('--gate', default='baseline', help='baseline | reference | module:function')
    ap.add_argument('--observe-all', action='store_true',
                    help='model full perception coverage (AFTER-coverage column)')
    ap.add_argument('--out', type=Path, help='write report.md, results.csv, summary.json here')
    ap.add_argument('--urdf', type=Path, help='override the mission URDF')
    ap.add_argument('--curobo', type=Path, help='override the curobo collision-sphere yml')
    ap.add_argument('--thresholds', type=Path, help='JSON thresholds (else scenario-margin defaults)')
    ap.add_argument('--title', default='Arm collision bench (CPU mock-ESDF)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    scenarios = bench.load_scenarios(args.scenarios)
    if not scenarios:
        ap.error('no scenarios found in %s' % args.scenarios)
    if args.observe_all:
        scenarios = [bench.with_full_coverage(s) for s in scenarios]
    thresholds = bench.Thresholds.from_dict(
        json.loads(args.thresholds.read_text()) if args.thresholds else None)

    arm = bench.ArmModel.load(args.urdf, args.curobo)
    gate, gate_name = bench.resolve_gate(args.gate)
    if args.observe_all:
        gate_name += '+observe_all'
    results = [bench.run_scenario(s, arm, gate, thresholds) for s in scenarios]
    summary = bench.summarise(results, gate_name, thresholds)
    md = bench.report_markdown(results, summary, title=args.title)
    csv = bench.results_csv(results)

    if not args.quiet:
        print(md)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'report.md').write_text(md)
        (args.out / 'results.csv').write_text(csv)
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if not args.quiet:
            print('wrote %s/{report.md,results.csv,summary.json}' % args.out)
    return 0 if summary['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
