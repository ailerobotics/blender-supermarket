#!/usr/bin/env python3
"""Smart post-restock planogram/realogram verification (CPU, read-only, opt-in).

Consumes a COMPLETED mission's artifacts and emits an ADDITIONAL compliance report
plus any residual reset jobs in the ``sim-smart-job-feed-v1`` ``--job-from`` shape.
This never changes the mission's own stock-commit gate (the mission ``verify`` stage
stays the source of truth); it is a smarter, planogram-level close of the loop:
"one SKU landed" -> "the destination slot now matches the planogram, or here are
the residual reset jobs that would make it".

Evidence basis is labelled honestly on every slot verdict (see
:mod:`twin.restock.smart_verify`):

* default: ``mission_verify_sensor`` -- only the just-placed handling unit was
  scanned at the destination slot by the mission ``verify`` stage;
* ``--inventory DB``: ``bookkeeping`` -- the transactional DB's committed counts;
* ``--shelf-record JSON``: ``full_shelf_sensor`` -- a complete labelled shelf
  observation supplied out of band (e.g. a store-vision ``realogram`` export).

Optionally cross-checks the verdict through store-vision's read-only
``planoreset.verify`` interfaces (``--storevision``); that is advisory and degrades
cleanly when store-vision is not importable.

Examples (RUNBOOK §11a):

    # default: consume a finished run's mission.json, verdict for the job's slot
    scripts/run_smart_verify.py --mission <run>/mission.json

    # bookkeeping basis + residual jobs emitted to the job-feed contract
    scripts/run_smart_verify.py --mission <run>/mission.json \\
        --inventory <run>/inventory.sqlite --job-feed-out <run>/reset-jobs.json

    # store-vision cross-check (advisory)
    scripts/run_smart_verify.py --mission <run>/mission.json --storevision
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Derive ROOT from the script location (env SUPERMARKET_ROOT overrides), mirroring
# the other CLIs so this runs in any worktree/clone.
ROOT = Path(os.environ.get('SUPERMARKET_ROOT') or Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(ROOT / 'scripts'))

from twin.restock import smart_verify as sv  # noqa: E402

DEFAULT_PLANOGRAM = ROOT / 'scripts' / 'planogram.json'


def _resolve_slot(mission, evidence, explicit):
    """Choose the destination slot: explicit --slot wins, else the verify location."""
    if explicit:
        return explicit
    loc = (evidence or {}).get('location')
    if isinstance(loc, str) and loc:
        return loc
    raise SystemExit('cannot determine destination slot: pass --slot (mission verify '
                     'evidence carried no location)')


def build(args):
    mission = json.loads(Path(args.mission).read_text())
    plan = sv.load_planogram(args.planogram)
    evidence = sv.mission_verify_evidence(mission)
    if evidence is None:
        raise SystemExit('mission verify did not succeed; smart-verify has nothing to '
                         'check (this is the gate: run only after a succeeded verify)')
    slot_id = _resolve_slot(mission, evidence, args.slot)

    stock = sv.load_stock(args.inventory) if args.inventory else None
    shelf_record = json.loads(Path(args.shelf_record).read_text()) if args.shelf_record else None

    report = sv.report_from_mission(
        mission=mission, plan=plan, slot_id=slot_id, stock=stock,
        shelf_record=shelf_record, source_bin=args.source_bin,
        quantity_policy=args.quantity_policy, priority=args.priority,
        mission_json_path=args.mission)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mission', required=True, type=Path,
                   help="a completed run's mission.json (events carry the verify evidence)")
    p.add_argument('--planogram', type=Path, default=DEFAULT_PLANOGRAM,
                   help='planogram JSON (default: scripts/planogram.json)')
    p.add_argument('--slot', help='destination slot id (default: the mission verify location)')
    p.add_argument('--inventory', type=Path,
                   help='inventory.sqlite for a bookkeeping-basis realogram (read-only)')
    p.add_argument('--shelf-record', type=Path,
                   help='full-shelf labelled observation JSON (full_shelf_sensor basis)')
    p.add_argument('--quantity-policy', choices=sv.QUANTITY_POLICIES, default='at_least',
                   help='how to judge quantity vs the planogram target (default: at_least)')
    p.add_argument('--source-bin', help='source cell to stamp on residual restock jobs')
    p.add_argument('--priority', type=int, default=50, help='residual job priority (default: 50)')
    p.add_argument('--out', type=Path, help='write the full smart-verify report JSON here')
    p.add_argument('--job-feed-out', type=Path,
                   help='write ONLY the residual reset jobs (job-feed --job-from shape) here')
    p.add_argument('--storevision', action='store_true',
                   help='also cross-check through store-vision planoreset.verify (advisory)')
    p.add_argument('--storevision-src',
                   help='store-vision src/ path for the cross-check (default: $STOREVISION_SRC '
                        'or the sibling checkout)')
    p.add_argument('--quiet', action='store_true', help='suppress the stdout summary')
    args = p.parse_args(argv)

    report = build(args)
    out = report.to_dict()

    if args.storevision:
        from twin.restock import smart_verify_storevision as svx
        out['storevision_crosscheck'] = svx.cross_check(
            report, storevision_src=args.storevision_src)

    if args.out:
        args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    if args.job_feed_out:
        jobs = [j.to_job_feed() for j in report.residual_jobs]
        args.job_feed_out.write_text(json.dumps(jobs, indent=2, allow_nan=False) + '\n')

    if not args.quiet:
        residual = out['residual_jobs']
        print(json.dumps({
            'job_id': out['job_id'],
            'compliant': out['compliant'],
            'evidence_basis': out['evidence_basis'],
            'slots_checked': len(out['slots']),
            'residual_jobs': len(residual),
            'storevision': out.get('storevision_crosscheck', {}).get('available'),
            'authority_note': out['authority_note'],
        }, indent=2))

    # Exit 0 always: smart-verify is an ADDITIONAL report, never a gate. A
    # non-compliant verdict is signalled in the JSON, not via a failure exit code.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
