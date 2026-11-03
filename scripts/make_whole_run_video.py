#!/usr/bin/env python3
"""Build one continuous H.264 MP4 of an entire mission from its mission-frames, with a
stage-name + sim-time overlay (OpenCV only; no system ffmpeg needed).

The stage per frame comes from the run's ``mission-progress.jsonl`` stage->sim_time ladder
(falling back to ``nav-trace.jsonl`` actions). The overlay drawing + ladder live in the shared
``twin.demo.stage_video`` module, which the per-stage audit clips (``scripts/stage_audit.py``)
render with too, so a clip's band matches the whole-run video frame for frame.

    python3 scripts/make_whole_run_video.py <run_dir> <out.mp4> [fps]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.demo import stage_video            # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        raise SystemExit("usage: make_whole_run_video.py <run_dir> <out.mp4> [fps]")
    run_dir, out = argv[0], argv[1]
    fps = float(argv[2]) if len(argv) > 2 else 20.0
    summary = stage_video.whole_run_video(run_dir, out, fps=fps)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
