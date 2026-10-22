#!/usr/bin/env python3
"""Check a demo compile spec against the clean cut ranges (CPU only, no video decode).

    python3 scripts/check_cut_ranges.py [--cut-ranges data/sim-video-cut-qa-v1/cut_ranges.json] \
        [--spec data/sim-video-compile-v1/compile-demo-006.json]

Always validates cut_ranges.json itself (every keep piece inside its stage span and source, ordered,
no keep frame inside a fault window dilated by the 0.5 s margin, every cut carries a reason). With
``--spec``, also checks every compile segment: its [frame_start, frame_end] must lie inside ONE keep
piece of the same source + stage. Exit 0 = clean, 1 = a violation (each printed), 2 = bad input.
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / 'data' / 'sim-video-cut-qa-v1' / 'cut_ranges.json'
ENDING = ('carton_tilt', 'unlatch', 'position_jump', 'carton_floating', 'carton_off_cart', 'whiteout',
          'black_frame', 'image_jump', 'clip', 'story')


def check_ranges(cr):
    bad = []
    margin = int(cr.get('margin_frames', 8))
    for name, rec in cr.get('sources', {}).items():
        if rec.get('missing'):
            continue
        n = int(rec['n_frames'])
        for stage, s in rec.get('stages', {}).items():
            lo, hi = s['stage_span']
            prev = None
            for a, b in s.get('keep', []):
                if not (lo <= a <= b <= hi < n):
                    bad.append(f'{name}/{stage}: keep [{a},{b}] outside stage span [{lo},{hi}] / {n} frames')
                if prev is not None and a <= prev:
                    bad.append(f'{name}/{stage}: keep pieces overlap or are out of order at [{a},{b}]')
                prev = b
                for f in s.get('faults', []):
                    if f['kind'] in ENDING and a <= f['end'] + margin and f['frame'] - margin <= b:
                        bad.append(f"{name}/{stage}: keep [{a},{b}] within {margin} frames of fault "
                                   f"{f['kind']} #{f['frame']}-{f['end']}")
            for c in s.get('cuts', []):
                if not str(c.get('reason', '')).strip():
                    bad.append(f"{name}/{stage}: cut [{c.get('start')},{c.get('end')}] has no reason")
    return bad


def clean_runs(rec):
    """A source's keep pieces over ALL its stages, merged where contiguous: cleanliness is per frame, so a
    segment may straddle a stage boundary (the compile and the trace can disagree on it by a frame or two)."""
    pieces = sorted(tuple(k) for s in rec.get('stages', {}).values() for k in s.get('keep', []))
    runs = []
    for a, b in pieces:
        if runs and a <= runs[-1][1] + 1:
            runs[-1][1] = max(runs[-1][1], b)
        else:
            runs.append([a, b])
    return runs


def check_spec(cr, spec):
    bad = []
    for seg in spec.get('segments', []):
        name, stage = Path(seg['source']).name, seg['stage']
        a, b = int(seg['frame_start']), int(seg['frame_end'])
        rec = cr.get('sources', {}).get(name)
        if rec is None or rec.get('missing'):
            bad.append(f'{stage} <- {name} [{a},{b}]: source not scanned (run scripts/cut_qa_ranges.py)')
            continue
        runs = clean_runs(rec)
        if not any(ka <= a and b <= kb for ka, kb in runs):
            cuts = sorted((c for s in rec.get('stages', {}).values() for c in s.get('cuts', [])
                           if c['start'] <= b and a <= c['end']), key=lambda c: c['start'])
            why = (f"cut #{cuts[0]['start']}-{cuts[0]['end']}: {cuts[0]['reason'][:140]}" if cuts
                   else 'frames outside every scanned stage')
            bad.append(f'{stage} <- {name} [{a},{b}]: not inside one clean keep run {runs} ({why})')
    return bad


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cut-ranges', type=Path, default=DEFAULT)
    parser.add_argument('--spec', type=Path, help='compile spec JSON (scripts/compile_demo_video.py input)')
    args = parser.parse_args(argv)
    try:
        cr = json.loads(args.cut_ranges.read_text())
        spec = json.loads(args.spec.read_text()) if args.spec else None
    except (OSError, ValueError) as exc:
        print(f'bad input: {exc}', file=sys.stderr)
        return 2
    bad = check_ranges(cr) + (check_spec(cr, spec) if spec else [])
    for line in bad:
        print('FAIL', line)
    print('OK' if not bad else f'{len(bad)} violation(s)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
