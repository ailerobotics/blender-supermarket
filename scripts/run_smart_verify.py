#!/usr/bin/env python3
"""Smart post-restock planogram/realogram verification (CPU, read-only, opt-in).

Consumes a COMPLETED mission's artifacts and emits an ADDITIONAL compliance report
plus any residual reset jobs. This NEVER changes the mission's own stock-commit gate
(the mission ``verify`` stage stays the source of truth); it is a smarter,
planogram-level close of the loop: "one SKU landed" -> "the destination slot now
matches the planogram, or here are the residual reset jobs that would make it".

sim-smart-verify-v2 adds the **post-transfer shelf observation** export: the resolved
verify region (the bay the shelf-staging resolver pinned) + the saved verify frames,
assembled into a realogram (``shelf-observation.json``) that can be run through
store-vision ``realogram``/``planoreset`` read-only (``--storevision-realogram``).

Modes (``--mode``, default ``report``):

* ``report`` -- WORKING DEFAULT: an additional report only. The mission gate is
  unchanged; exit is always 0.
* ``strict`` -- SMART variant: additionally writes the residual reset jobs in lane-1's
  ``storevision.restock.job_plan/v1`` shape (``--job-plan-out`` / ``reset-jobs.job-plan.json``)
  so the NEXT mission can consume them via ``run_mobile_restock.py --job-from`` (closed
  loop; store-vision ``restockloop`` is the analogue). STILL not a blocker.
* ``block`` -- SMART variant: like ``strict`` and additionally exits non-zero when the
  slot is non-compliant. This does NOT retroactively un-commit the mission's stock
  (the mission ``verify`` stage already committed it); it is smart-verify's OWN gate,
  for a caller that wants a hard signal. Opt-in and explicit.

Evidence basis is labelled honestly on every slot verdict (see
:mod:`twin.restock.smart_verify`):

* default: ``mission_verify_sensor`` -- only the just-placed handling unit was
  scanned at the destination slot by the mission ``verify`` stage;
* ``--inventory DB``: ``bookkeeping`` -- the transactional DB's committed counts;
* ``--shelf-record JSON``: ``full_shelf_sensor`` -- a complete labelled shelf
  observation supplied out of band (e.g. a store-vision ``realogram`` export).

Examples (RUNBOOK §11b):

    # default: a finished run dir -> report + shelf observation next to mission.json
    scripts/run_smart_verify.py --run-dir <run>

    # strict: also emit residual jobs the next mission can consume (--job-from)
    scripts/run_smart_verify.py --run-dir <run> --mode strict

    # store-vision realogram/planoreset cross-check (advisory, needs the sv .venv)
    scripts/run_smart_verify.py --run-dir <run> --storevision --storevision-realogram
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

# Derive ROOT from the script location (env SUPERMARKET_ROOT overrides), mirroring
# the other CLIs so this runs in any worktree/clone.
ROOT = Path(os.environ.get('SUPERMARKET_ROOT') or Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(ROOT / 'scripts'))

from twin.restock import smart_verify as sv  # noqa: E402
from twin.restock import shelf_realogram as sr  # noqa: E402

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


def _mission_and_rundir(args):
    """Resolve (mission.json path, run_dir) from --mission / --run-dir."""
    if args.mission:
        mission_path = Path(args.mission)
        run_dir = args.run_dir or mission_path.parent
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        mission_path = run_dir / 'mission.json'
    else:
        raise SystemExit('pass --mission <mission.json> or --run-dir <run directory>')
    return mission_path, (Path(run_dir) if run_dir else None)


def build(args):
    """Build the v1 report (verdict + residual jobs). Unchanged semantics."""
    mission_path, _ = _mission_and_rundir(args)
    mission = json.loads(Path(mission_path).read_text())
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
        mission_json_path=str(mission_path))
    return mission, plan, evidence, slot_id, report, stock, shelf_record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group('inputs')
    src.add_argument('--mission', type=Path,
                     help="a completed run's mission.json (events carry the verify evidence)")
    src.add_argument('--run-dir', type=Path,
                     help='a completed run directory (mission.json + mission-frames/)')
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
    p.add_argument('--mode', choices=('report', 'strict', 'block'), default='report',
                   help='report (default, additional report only) | strict (also emit '
                        'job_plan/v1 residual jobs for the next mission) | block (strict + '
                        'exit non-zero when non-compliant; smart-verify own gate, never '
                        'un-commits the mission)')
    p.add_argument('--store-config', help='store_config label to stamp on job_plan/v1 residuals')
    out = p.add_argument_group('outputs')
    out.add_argument('--out', type=Path, help='write the full smart-verify report JSON here')
    out.add_argument('--out-dir', type=Path,
                     help='write smart-verify.json + shelf-observation.json (+ reset-jobs.job-plan.json '
                          'in strict/block) into this dir (default: the run dir)')
    out.add_argument('--shelf-observation-out', type=Path,
                     help='write ONLY the post-transfer shelf observation (realogram) JSON here')
    out.add_argument('--job-feed-out', type=Path,
                     help='write the residual jobs in the legacy job-feed shape here')
    out.add_argument('--job-plan-out', type=Path,
                     help='write the consumable residual jobs in job_plan/v1 shape here '
                          '(default in strict/block: <out-dir>/reset-jobs.job-plan.json)')
    sx = p.add_argument_group('store-vision cross-check (read-only, advisory)')
    sx.add_argument('--storevision', action='store_true',
                    help='cross-check through store-vision planoreset.verify_from_compliance '
                         '(in-process, no pydantic)')
    sx.add_argument('--storevision-realogram', action='store_true',
                    help='also run the realogram->planoreset pipeline in the store-vision '
                         '.venv subprocess (needs pydantic; degrades cleanly)')
    sx.add_argument('--storevision-src',
                    help='store-vision src/ path (default: $STOREVISION_SRC or the sibling checkout)')
    sx.add_argument('--storevision-venv-python',
                    help='store-vision .venv python for --storevision-realogram')
    p.add_argument('--quiet', action='store_true', help='suppress the stdout summary')
    args = p.parse_args(argv)

    mission, plan, evidence, slot_id, report, stock, shelf_record = build(args)

    # v2: the post-transfer shelf observation (resolved verify region + saved frames).
    _, run_dir = _mission_and_rundir(args)
    extra = None
    # When a bookkeeping snapshot is available, enrich the exported realogram with the
    # believed shelf stock (recorded as a weaker belief; the verdict is unchanged).
    if stock is not None and shelf_record is None:
        bk = sv.observation_from_bookkeeping(stock, slot_id).observed
        extra = {k: v for k, v in bk.items() if k not in report.slots[0].observed}
    realogram = sr.build_shelf_realogram(
        mission, evidence, slot_id, run_dir=run_dir, shelf_record=shelf_record,
        extra_observed=extra)

    out = report.to_dict()
    out['shelf_observation'] = realogram.to_dict()
    out['mode'] = args.mode

    if args.storevision:
        from twin.restock import smart_verify_storevision as svx
        out['storevision_crosscheck'] = svx.cross_check(
            report, storevision_src=args.storevision_src)
    if args.storevision_realogram:
        from twin.restock import smart_verify_storevision as svx
        expectation = sv.planogram_slot_index(plan).get(slot_id)
        out['storevision_realogram'] = svx.cross_check_realogram(
            realogram, expectation, venv_python=args.storevision_venv_python,
            storevision_src=args.storevision_src)

    # Residual jobs in lane-1 job_plan/v1 shape (strict/block emit them for the next mission).
    generated_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    consumable, advisory = sv.residual_job_plans(
        report, store_config=args.store_config, generated_at=generated_at)
    out['residual_job_plans'] = {'consumable': consumable, 'advisory': advisory}

    # Auto-write the trio next to mission.json only when a run dir / out dir is
    # explicit; the bare --mission form keeps v1's "write only what is asked" contract
    # (so a read-only saved run is not written into unless the caller opts in).
    out_dir = args.out_dir or (run_dir if args.run_dir else None)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / 'smart-verify.json').write_text(
            json.dumps(out, indent=2, allow_nan=False) + '\n')
        (Path(out_dir) / 'shelf-observation.json').write_text(
            json.dumps(realogram.to_dict(), indent=2, allow_nan=False) + '\n')
    if args.shelf_observation_out:
        args.shelf_observation_out.write_text(
            json.dumps(realogram.to_dict(), indent=2, allow_nan=False) + '\n')
    if args.job_feed_out:
        jobs = [j.to_job_feed() for j in report.residual_jobs]
        args.job_feed_out.write_text(json.dumps(jobs, indent=2, allow_nan=False) + '\n')
    if args.mode in ('strict', 'block'):
        job_plan_out = args.job_plan_out or (Path(out_dir) / 'reset-jobs.job-plan.json'
                                             if out_dir else None)
        if job_plan_out:
            Path(job_plan_out).write_text(json.dumps(consumable, indent=2, allow_nan=False) + '\n')

    if not args.quiet:
        print(json.dumps({
            'job_id': out['job_id'],
            'mode': args.mode,
            'compliant': out['compliant'],
            'evidence_basis': out['evidence_basis'],
            'shelf_observation': {'region_source': realogram.region.source,
                                  'frames_basis': realogram.frames_basis,
                                  'frames': len(realogram.frames)},
            'slots_checked': len(out['slots']),
            'residual_jobs': len(out['residual_jobs']),
            'residual_job_plans_consumable': len(consumable),
            'storevision': out.get('storevision_crosscheck', {}).get('available'),
            'storevision_realogram': out.get('storevision_realogram', {}).get('available'),
            'authority_note': out['authority_note'],
        }, indent=2))

    # Exit 0 except in block mode on a non-compliant slot (smart-verify's OWN gate;
    # the mission commit is unchanged either way).
    if args.mode == 'block' and not out['compliant']:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
