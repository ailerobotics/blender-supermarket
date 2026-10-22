#!/usr/bin/env python3
"""Extract one mission stage's recorded footage into a self-contained segment directory.

The compiled demo video (sim-video-compile-v1) is stitched from per-stage segments, each
a clean clip of one stage drawn from the run that performed it best. This tool carves that
segment out of a full mission run: it selects the frames the trace assigns to the stage
(reusing ``run_evidence`` for the stage<->frame mapping and the frame/trace clock offset),
copies them renumbered from 0, and writes the matching ``frames.json`` (original per-frame
metadata preserved), ``bt_events.jsonl`` (stage events only), ``nav-trace.jsonl`` (stage
window) and a ``README.md`` that names the source run and the ORIGINAL frame range.

CPU only; no Blender, no Isaac. See ``tests/test_extract_stage_segment.py``.

Usage:
    python -m twin.demo.extract_stage_segment --run <run_dir> --stage collect --out <seg_dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import run_evidence as E


def _collect_frames(run, stage):
    """The Frame objects the trace assigns to ``stage`` (in record order)."""
    frames = E.frames_by_stage(run).get(stage) or []
    return sorted(frames, key=lambda f: f.index)


def _stage_bt_events(run_dir, stage):
    """bt_events.jsonl lines whose data.stage == stage, as parsed dicts."""
    out = []
    path = Path(run_dir) / "bt_events.jsonl"
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if (ev.get("data") or {}).get("stage") == stage:
            out.append(ev)
    return out


def _stage_nav_rows(run_dir, stage):
    """nav-trace.jsonl raw lines whose ``action`` == stage (original JSON preserved)."""
    out = []
    path = Path(run_dir) / "nav-trace.jsonl"
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("action") == stage:
            out.append(line)
    return out


def extract(run_dir, stage, out_dir, *, require_succeeded=True):
    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve()
    run = E.load_run(run_dir)

    status = E.stage_status(run).get(stage, "missing")
    if require_succeeded and status != "succeeded":
        raise SystemExit(
            f"refusing: stage {stage!r} status is {status!r}, not 'succeeded' in {run.name}. "
            f"pass --allow-incomplete to extract a partial stage anyway."
        )

    frames = _collect_frames(run, stage)
    if not frames:
        raise SystemExit(f"no recorded frames assigned to stage {stage!r} in {run.name}")

    # Source frames.json metadata, keyed by file name, so we copy render_id/sha256/shape verbatim.
    src_manifest = json.loads((run_dir / "mission-frames" / "frames.json").read_text())
    by_file = {f["file"]: f for f in src_manifest.get("frames", [])}

    frames_out = out_dir / "mission-frames"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    frames_out.mkdir(parents=True)

    new_frames = []
    for new_i, fr in enumerate(frames):
        new_name = f"{new_i:06d}.png"
        shutil.copy2(fr.file, frames_out / new_name)
        meta = dict(by_file.get(fr.file.name, {}))
        meta["file"] = new_name
        meta.setdefault("simulation_time", fr.sim_time)
        new_frames.append(meta)

    manifest = {
        "nominal_fps": src_manifest.get("nominal_fps", 30),
        "timing": src_manifest.get("timing", "variable acquisition intervals"),
        "source_run": run.name,
        "source_stage": stage,
        "source_frame_range": [frames[0].index, frames[-1].index],
        "frames": new_frames,
    }
    (frames_out / "frames.json").write_text(json.dumps(manifest, indent=1))

    bt = _stage_bt_events(run_dir, stage)
    (out_dir / "bt_events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in bt)
    )

    nav_lines = _stage_nav_rows(run_dir, stage)
    (out_dir / "nav-trace.jsonl").write_text(
        "".join(l + "\n" for l in nav_lines)
    )

    sim_lo, sim_hi = frames[0].sim_time, frames[-1].sim_time
    readme = f"""# {stage} segment — source {run.name}

Carved by `twin.demo.extract_stage_segment` from the full mission run `{run.name}`
(`{run_dir}`).

- **source run**: `{run.name}`
- **stage**: `{stage}` (mission status: `{status}`)
- **source frame range**: `{frames[0].index}`..`{frames[-1].index}` of
  `{run.name}/mission-frames/` ({len(frames)} frames, renumbered 0..{len(frames)-1} here)
- **sim-time**: {sim_lo:.3f}s .. {sim_hi:.3f}s (Isaac stage clock, as recorded)
- **frame shape**: {new_frames[0].get('shape')} (same walkthrough camera as warm8)

## Contents
- `mission-frames/*.png` + `mission-frames/frames.json` — the {len(frames)} stage frames,
  original per-frame metadata (render_id, sha256, shape, simulation_time) preserved.
- `bt_events.jsonl` — behaviour-tree events for this stage only ({len(bt)} events).
- `nav-trace.jsonl` — trace rows while this stage was active ({len(nav_lines)} rows).

Regenerate with:

    python -m twin.demo.extract_stage_segment --run {run_dir} --stage {stage} --out {out_dir}
"""
    (out_dir / "README.md").write_text(readme)

    return {
        "run": run.name,
        "stage": stage,
        "status": status,
        "frame_range": [frames[0].index, frames[-1].index],
        "n_frames": len(frames),
        "sim_time": [sim_lo, sim_hi],
        "out_dir": str(out_dir),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="source mission run directory")
    ap.add_argument("--stage", required=True, choices=E.STAGES, help="stage to extract")
    ap.add_argument("--out", required=True, help="output segment directory (overwritten)")
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="extract even if the stage did not reach 'succeeded' (partial clip)",
    )
    args = ap.parse_args(argv)
    summary = extract(
        args.run, args.stage, args.out, require_succeeded=not args.allow_incomplete
    )
    json.dump(summary, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
