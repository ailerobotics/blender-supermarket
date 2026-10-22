#!/usr/bin/env python3
"""Scan demo-compile source dirs frame by frame and write frame-exact CLEAN cut ranges (CPU only).

    python3 scripts/cut_qa_ranges.py --out data/sim-video-cut-qa-v1/cut_ranges.json \
        [--mission-root DIR] [--visual data/sim-video-cut-qa-v1/visual_faults.json] [SOURCE ...]

For every source (a run dir or a choreographed demo dir under ``--mission-root``) and every stage in
it, writes the usable ``keep`` frame pieces, the ``cuts`` with a reason each, and every fault found
(pose / image / by-eye ``visual`` channel). Detectors + rules: twin/demo/cut_qa.py. Source dirs are
read-only; a source that is not there yet (e.g. door-src-001 still rendering) is recorded as missing.
Check a compile spec against the result with scripts/check_cut_ranges.py.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.demo import cut_qa as Q        # noqa: E402

MISSION_ROOT = Path('/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission')
DEFAULT_SOURCES = ('collect-src-001', 'warm11', 'warm8', 'demo-005', 'demo-004', 'door-src-001',
                   'run-326', 'run-325', 'run-322', 'warm7')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sources', nargs='*', default=list(DEFAULT_SOURCES))
    parser.add_argument('--mission-root', type=Path, default=MISSION_ROOT)
    parser.add_argument('--visual', type=Path, help='by-eye faults JSON {source: [{kind, frame, end, detail}]}')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    visual, notes = {}, {}
    if args.visual:
        data = json.loads(args.visual.read_text())
        visual = data.get('faults', {})
        notes = data.get('notes', {})
    out = Q.build(args.mission_root, args.sources, visual, notes)
    out['generated'] = date.today().isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    for name, rec in out['sources'].items():
        if rec.get('missing'):
            print(f'{name:<16} MISSING (not rendered yet)')
            continue
        for stage, s in rec['stages'].items():
            ff = s['first_fault']
            print(f"{name:<16} {stage:<9} span{s['stage_span']} keep={s['keep']}"
                  + (f"  first fault {ff['kind']}#{ff['frame']}" if ff else ''))
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
