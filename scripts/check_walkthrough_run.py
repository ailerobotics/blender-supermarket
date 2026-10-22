#!/usr/bin/env python3
"""Pass/fail check of a restock mission run dir before its demo video is sent (CPU only, read-only).

    python3 scripts/check_walkthrough_run.py <run-dir> [--json out.json]

Prints a short table (stages in order, mission succeeded, frames per stage, carton on the rollwagen
through couple/outbound/park, carton on the bay top after transfer, demo video present) and exits 0
only when the OVERALL verdict is PASS or WARN (WARN = passes with a labelled workaround, e.g. a
start-phase shortcut). Checks and evidence sources: twin/demo/run_check.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.demo import run_check as C           # noqa: E402
from twin.demo import run_evidence as E        # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--scenario', type=Path,
                        help='mission scenario JSON (default: <run>/mission-scenario.json or ../mission-scenario.json)')
    parser.add_argument('--console-log', type=Path, help='the run console log (default: ../<run>.console.log)')
    parser.add_argument('--json', type=Path, help='also write the results as JSON here')
    args = parser.parse_args(argv)
    run = E.load_run(args.run_dir, console_log=args.console_log, scenario=args.scenario)
    checks = C.run_checks(run)
    print(C.format_table(run, checks))
    verdict = C.overall(checks)
    if args.json:
        args.json.write_text(json.dumps({
            'run': str(run.run_dir), 'overall': verdict, 'result': E.result_line(run),
            'checks': [{'name': c.name, 'result': c.result, 'evidence': c.evidence, 'details': c.details}
                       for c in checks],
            'labels': [{'stage': l.stage, 'key': l.key, 'value': l.value} for l in E.workaround_labels(run)],
        }, indent=2, default=str) + '\n')
    return 0 if verdict in (C.PASS, C.WARN) else 1


if __name__ == '__main__':
    sys.exit(main())
