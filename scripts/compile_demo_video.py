#!/usr/bin/env python3
"""Compile one continuous seven-stage demo video from per-stage segments of different runs (CPU only).

    python3 scripts/compile_demo_video.py <spec.json> --out <dir>/walkthrough.mp4 [--ffmpeg PATH] [--tmp DIR]

The spec JSON lists, in mission order, which run supplies each stage and the frame range to take; the
compiler re-renders the decision/reason-tree panel so stage numbers, the mission clock and the ladder
run continuously across the source seams, drops short cross-fades at every source boundary, and writes
the mp4 plus ``<out>.table.json`` (the stage-to-source provenance table). Spec format: see
``data/sim-video-compile-v1/compile-demo-006.json`` and twin/demo/stitch.py. Source runs are read-only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.demo import stitch as S        # noqa: E402
from twin.demo import walkthrough_cut as W  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('spec', type=Path, help='compile spec JSON')
    parser.add_argument('--out', type=Path, required=True, help='output mp4')
    parser.add_argument('--ffmpeg', help='ffmpeg binary (default: $FFMPEG, PATH, imageio_ffmpeg)')
    parser.add_argument('--tmp', type=Path, help='scratch dir for decoded choreo frames (default: <out>.tmp)')
    parser.add_argument('--overwrite', action='store_true', help='replace an existing output mp4 + .table.json')
    args = parser.parse_args(argv)
    ffmpeg = W.find_ffmpeg(args.ffmpeg)
    if ffmpeg is None:
        parser.error('no ffmpeg found: pass --ffmpeg or set $FFMPEG')
    out = args.out
    if out.exists():
        if not args.overwrite:
            parser.error(f'{out} exists (pass --overwrite to replace it)')
        out.unlink()
        side = out.with_suffix('.table.json')
        if side.exists():
            side.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.tmp or out.with_suffix('.tmp')
    spec = S.CompileSpec.from_json(args.spec)
    t0 = time.monotonic()
    summary = S.render(spec, out, ffmpeg=ffmpeg, tmp_dir=tmp)
    print(json.dumps({k: summary[k] for k in ('video', 'size', 'fps', 'output_frames', 'duration_s',
                                              'xfade_frames')}, indent=1))
    print('stage -> source:')
    for row in summary['segments']:
        print(f"  {row['stage']:<9} <- {row['source']:<16} {row['kind']:<7} "
              f"frames[{row['frame_start']}:{row['frame_end']}] emitted={row['emitted']}"
              + (f"  labels={row['labels']}" if row['labels'] else ''))
    print(f'built in {time.monotonic() - t0:.1f}s; table: {out.with_suffix(".table.json")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
