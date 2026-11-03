"""Shared stage + sim-time video overlay (OpenCV only; no system ffmpeg).

This is the single home for the "stage N/M: NAME | sim t = … s" band that the whole-run
mission video and the per-stage audit clips both burn into their frames. It was lifted out of
the integration lane's ``make_whole_run_video.py`` so ``stage_audit.py`` renders byte-identical
overlays instead of a drifting copy: ``scripts/make_whole_run_video.py`` is now a thin wrapper
over :func:`whole_run_video`, and ``scripts/stage_audit.py`` cuts per-stage clips via
:func:`write_overlay_video` over the same drawing code.

Clocks: the stage ladder and every frame are keyed on Isaac stage time (frame
``simulation_time`` == ``mission-progress.jsonl`` / ``stage_timeline`` ``sim_time``), so a clip's
overlay matches exactly what the whole-run video shows at that moment.

CPU only; no Blender, no Isaac. See ``tests/test_stage_video.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2


def load_stage_ladder(run_dir):
    """``[(sim_time, stage), …]`` sorted by sim time: the stage the mission entered and when.

    Primary source is ``mission-progress.jsonl`` (one record per stage entry, crash-safe); when
    that is absent or empty it falls back to the first appearance of each ``action`` in
    ``nav-trace.jsonl``. Identical to the integration lane's whole-run derivation."""
    run_dir = Path(run_dir)
    ladder = []
    mp = run_dir / "mission-progress.jsonl"
    if mp.is_file():
        for line in mp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            st = r.get("stage")
            t = r.get("sim_time", r.get("t"))
            if st is not None and t is not None:
                ladder.append((float(t), str(st)))
    if not ladder:
        nav = run_dir / "nav-trace.jsonl"
        seen = {}
        if nav.is_file():
            for line in nav.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                a = r.get("action")
                t = r.get("t")
                if a is not None and t is not None and a not in seen:
                    seen[a] = float(t)
        ladder = sorted((t, a) for a, t in seen.items())
    ladder.sort()
    return ladder


def stage_at(ladder, sim_t):
    """The stage active at Isaac stage time ``sim_t`` per ``ladder`` (the last entry whose time
    is at or before ``sim_t``)."""
    cur = ladder[0][1] if ladder else "?"
    for t, st in ladder:
        if sim_t + 1e-6 >= t:
            cur = st
        else:
            break
    return cur


def draw_overlay(img, *, run_name, stage, stage_idx, total_stages, sim_t, state):
    """Burn the whole-run overlay onto ``img`` in place: a translucent top band with
    ``<run> | STAGE i/n: NAME | sim t = … s`` and a bottom-right ``mission <STATE>`` tag."""
    h, w = img.shape[0], img.shape[1]
    band_h = 54
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, band_h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    label = (f"{run_name}  |  STAGE {stage_idx}/{total_stages}: {str(stage).upper()}"
             f"  |  sim t = {sim_t:6.1f} s")
    cv2.putText(img, label, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 180), 2, cv2.LINE_AA)
    tag = f"mission {state}"
    cv2.putText(img, tag, (w - 250, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2,
                cv2.LINE_AA)


def open_writer(out, fps, w, h):
    """Open a cv2.VideoWriter trying H.264 codecs first, then mp4v. Returns ``(writer, codec)``.

    Raises RuntimeError if no codec opens (so a silent 0-byte clip never ships)."""
    for cc in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*cc)
        writer = cv2.VideoWriter(str(out), fourcc, float(fps), (int(w), int(h)))
        if writer.isOpened():
            return writer, cc
        writer.release()
    raise RuntimeError("could not open a VideoWriter with avc1/H264/mp4v")


def read_state(run_dir):
    """Mission state (upper-case) from ``mission.json``, or ``COMPLETE`` when absent."""
    mj = Path(run_dir) / "mission.json"
    if mj.is_file():
        try:
            return str(json.loads(mj.read_text()).get("state", "?")).upper()
        except ValueError:
            pass
    return "COMPLETE"


def _load_manifest_frames(run_dir):
    """``[(abs_png_path, sim_time)]`` for every recorded frame, sorted by file name."""
    fdir = Path(run_dir) / "mission-frames"
    man = json.loads((fdir / "frames.json").read_text())
    frames = man["frames"] if isinstance(man, dict) and "frames" in man else man
    frames = sorted(frames, key=lambda f: f["file"])
    return [(fdir / f["file"], float(f["simulation_time"])) for f in frames]


def write_overlay_video(items, out, *, run_name, state, ladder, fps=20.0):
    """Write ``items`` (``[(png_path, sim_time)]``, in order) to ``out`` with the stage overlay.

    ``ladder`` drives the stage name/index shown; ``state`` the corner tag. Returns a summary
    dict (out path, frame count, fps, size, duration, stage_order, sim span). Missing/unreadable
    frames are skipped. Raises if no frame could be read."""
    items = list(items)
    if not items:
        raise RuntimeError(f"no frames to write for {out}")
    total_stages = len(ladder)
    first = cv2.imread(str(items[0][0]))
    if first is None:
        # Find the first readable frame to size the writer.
        for path, _ in items:
            first = cv2.imread(str(path))
            if first is not None:
                break
    if first is None:
        raise RuntimeError(f"no readable frames among {len(items)} for {out}")
    h, w = first.shape[0], first.shape[1]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer, codec = open_writer(out, fps, w, h)
    written = 0
    stage_order = []
    sim_lo = sim_hi = None
    try:
        for path, sim_t in items:
            img = cv2.imread(str(path))
            if img is None:
                continue
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            st = stage_at(ladder, sim_t)
            if not stage_order or stage_order[-1] != st:
                stage_order.append(st)
            stage_idx = len([s for s in ladder if s[0] <= sim_t + 1e-6]) or 1
            draw_overlay(img, run_name=run_name, stage=st, stage_idx=stage_idx,
                         total_stages=total_stages, sim_t=sim_t, state=state)
            writer.write(img)
            written += 1
            sim_lo = sim_t if sim_lo is None else min(sim_lo, sim_t)
            sim_hi = sim_t if sim_hi is None else max(sim_hi, sim_t)
    finally:
        writer.release()
    size = os.path.getsize(out) if os.path.exists(out) else 0
    return {
        "out": str(out), "frames": written, "fps": float(fps), "codec": codec,
        "size_bytes": size, "duration_s": round(written / fps, 2) if fps else 0.0,
        "state": state, "stage_order": stage_order,
        "sim_span_s": [round(sim_lo, 2), round(sim_hi, 2)] if sim_lo is not None else None,
    }


def whole_run_video(run_dir, out, fps=20.0):
    """Render the whole mission (every recorded frame) with the stage overlay -- the behaviour of
    the integration lane's ``make_whole_run_video.py``, now sharing the drawing code above."""
    run_dir = Path(run_dir).resolve()
    run_name = run_dir.name
    items = _load_manifest_frames(run_dir)
    ladder = load_stage_ladder(run_dir)
    state = read_state(run_dir)
    return write_overlay_video(items, out, run_name=run_name, state=state, ladder=ladder, fps=fps)
