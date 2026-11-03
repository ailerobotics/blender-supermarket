#!/usr/bin/env python3
"""Generate the restock behaviour-tree diagram (SVG + self-contained HTML) FROM the code, aligned
to a recorded run.

The structure is reconstructed by :mod:`twin.restock.bt.diagram` from the authored plans
(``stage_phases.PLANS``) and the event contract (``events.py``) -- never drawn by hand. When a run
directory is given, each stage/step node is coloured by the sim-time range it was active (replayed
from the run's ``bt_events.jsonl`` via ``twin.demo.bt_timeline``) and linked to the moment in the
run's whole-run walkthrough video.

Usage:
    python3 scripts/build_bt_diagram.py \
        --run   /home/.../datasets/twin-robot-in-store-boot-v1/mission/coldwin16 \
        --video coldwin16-whole-run.mp4 \
        --out   /home/.../firstmate/data/sim-bt-stage-map-v1 \
        [--fps 20] [--title "..."]

With no ``--run`` it emits the structural (uncoloured) diagram of the current tree.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ on the path

from twin.restock.bt import diagram as D   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', help='recorded run dir (mission-frames + bt_events.jsonl); omit for a '
                                  'structural-only diagram')
    ap.add_argument('--out', required=True, help='output directory for bt-diagram.svg/.html')
    ap.add_argument('--video', default='', help='URL/path the HTML embeds + seeks (relative or '
                                                'absolute)')
    ap.add_argument('--video-label', default='', help='caption under the video')
    ap.add_argument('--fps', type=float, default=20.0, help='fps the whole-run mp4 was built at '
                                                            '(make_whole_run_video.py default 20)')
    ap.add_argument('--title', default='Restock mission behaviour tree')
    ap.add_argument('--basename', default='bt-diagram')
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    align = None
    timeline = None
    if args.run:
        run_dir = Path(args.run)
        align = D.align_to_run(run_dir, fps=args.fps)
        from twin.demo import bt_timeline as B
        bt_events = D._read_jsonl(run_dir / 'bt_events.jsonl')
        mission = {}
        mj = run_dir / 'mission.json'
        if mj.exists():
            mission = json.loads(mj.read_text())
        timeline = B.BtTimeline.from_run(D._RunShim(bt_events, mission))

    model = D.build_model(timeline=timeline)
    if align is not None:
        D.annotate_model(model, align)

    svg = D.render_svg(model, align=align, title=args.title)
    svg_path = out_dir / f'{args.basename}.svg'
    svg_path.write_text(svg)

    html = D.render_html(model, align=align, video_url=args.video,
                         video_label=args.video_label, title=args.title,
                         svg_filename=svg_path.name)
    html_path = out_dir / f'{args.basename}.html'
    html_path.write_text(html)

    summary = {'svg': str(svg_path), 'html': str(html_path),
               'nodes': sum(1 for _ in D.iter_nodes(model)),
               'run': args.run, 'fps': args.fps,
               'mission_state': align['mission_state'] if align else None,
               'video_duration_s': round(align['video_duration'], 1)
               if (align and align.get('video_duration')) else None}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
