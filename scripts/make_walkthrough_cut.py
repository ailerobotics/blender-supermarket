#!/usr/bin/env python3
"""Assemble the demo cut of one restock mission run (CPU only; the run dir is only read).

    python3 scripts/make_walkthrough_cut.py <run-dir> [--out <run-dir>/walkthrough-demo.mp4] [--ffmpeg PATH]

Writes a 1280x720 30 fps H.264 mp4 (title card, a caption per stage for all seven stages, the
workaround labels from mission.json on screen, slow stretches time-lapsed, a closing result card
with the run checker's verdicts) plus ``<out>.cut.json`` (the edit decision list). ffmpeg comes from
--ffmpeg, $FFMPEG, PATH, or imageio_ffmpeg. Layout + honesty rules: twin/demo/walkthrough_cut.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.demo import run_check as C           # noqa: E402
from twin.demo import run_evidence as E        # noqa: E402
from twin.demo import walkthrough_cut as W     # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--out', type=Path, help='output mp4 (default: <run-dir>/walkthrough-demo.mp4)')
    parser.add_argument('--overwrite', action='store_true', help='replace an existing output mp4 + .cut.json')
    parser.add_argument('--ffmpeg', help='ffmpeg binary (default: $FFMPEG, PATH, imageio_ffmpeg)')
    parser.add_argument('--scenario', type=Path, help='mission scenario JSON (default: auto-discovered)')
    parser.add_argument('--console-log', type=Path, help='run console log (default: ../<run>.console.log)')
    parser.add_argument('--no-checks', action='store_true', help='skip the checker verdicts on the closing card')
    parser.add_argument('--bt-panel', action='store_true',
                        help='replace the route-map panel with the demo-004 decision/reason tree, driven by '
                             'the recorded bt_events + traced poses (1920x1080; default out walkthrough-bt.mp4)')
    args = parser.parse_args(argv)
    ffmpeg = W.find_ffmpeg(args.ffmpeg)
    if ffmpeg is None:
        parser.error('no ffmpeg found: pass --ffmpeg or set $FFMPEG')
    default_name = 'walkthrough-bt.mp4' if args.bt_panel else 'walkthrough-demo.mp4'
    out = args.out or (args.run_dir / default_name)
    if out.exists():
        if not args.overwrite:
            parser.error(f'{out} exists (pass --overwrite to replace it)')
        out.unlink()
        side = out.with_suffix('.cut.json')
        if side.exists():
            side.unlink()
    t0 = time.monotonic()
    run = E.load_run(args.run_dir, console_log=args.console_log, scenario=args.scenario)
    checks = [] if args.no_checks else C.run_checks(run)
    plan = W.build_plan(run, checks=checks, views=W.discover_views(run), bt_panel=args.bt_panel)
    summary = W.render_video(plan, out, ffmpeg=ffmpeg)
    print(json.dumps({k: summary[k] for k in ('video', 'size', 'bt_panel', 'duration_s', 'output_frames',
                                              'recorded_frames_used', 'recorded_frames_total', 'result',
                                              'clock_source')}, indent=1))
    print(f'built in {time.monotonic() - t0:.1f}s; edit list: {out.with_suffix(".cut.json")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
