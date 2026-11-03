#!/usr/bin/env python3
"""Per-stage audit of one or more finished mission runs — the captain's one question answered
for any run: which stage failed, why, when, and what did it look like. CPU only; the run dirs
are only read.

    python3 scripts/stage_audit.py <run-dir> [<run-dir>...] --out <dir> [--fps 20] [--no-clips]

Writes into ``<dir>``:
  - ``stage_audit.md``   one table per run (stage · PASS/FAIL/SKIPPED · failure reason by name ·
                         sim-time span · wall-time span · frames · key metrics the stage logs)
  - ``stage_audit.json`` the same, structured, plus the clip list
  - ``clips/<run>__<stage>.mp4`` one short overlaid clip per stage cut from mission-frames, with
                         the SAME stage + sim-time band as the whole-run video (shared
                         ``twin.demo.stage_video`` drawing code, not a copy)
  - ``index.html``       a self-contained page (table + inline <video> per clip) for the watch page
"""
import argparse
import json
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.demo import stage_audit_core as A      # noqa: E402
from twin.demo import stage_video                 # noqa: E402


def _render_clips(audit, out_dir, *, fps):
    """Cut one overlaid clip per stage that has recorded frames. Returns ``(clips, clip_info)``:
    ``clips`` maps stage -> relative clip path (for the Markdown/HTML), ``clip_info`` the per-clip
    render summaries."""
    clips_dir = out_dir / "clips"
    ladder = stage_video.load_stage_ladder(audit["run_dir"])
    state = str(audit["state"]).upper()
    clips, clip_info = {}, []
    for stage, frames in audit["_frames"].items():
        if not frames:
            continue
        items = [(f.file, f.sim_time) for f in sorted(frames, key=lambda f: f.index)]
        name = f"{audit['run']}__{stage}.mp4"
        rel = f"clips/{name}"
        summary = stage_video.write_overlay_video(
            items, clips_dir / name, run_name=audit["run"], state=state, ladder=ladder, fps=fps)
        summary["stage"] = stage
        clips[stage] = rel
        clip_info.append(summary)
    return clips, clip_info


def _html(audits):
    """A self-contained HTML page: per run, the stage table + an inline player per clip."""
    css = ("body{font:14px/1.5 system-ui,sans-serif;margin:16px;background:#0e1116;color:#d7dde5}"
           "h2{color:#8ff0c8}table{border-collapse:collapse;width:100%;margin:8px 0 20px}"
           "td,th{border:1px solid #2a3240;padding:5px 8px;text-align:left;font-size:13px}"
           "th{background:#161b22}.PASS{color:#5fe0a0}.FAIL{color:#ff7b72}.SKIPPED{color:#9aa4b2}"
           "video{width:320px;margin:4px 8px 4px 0;background:#000;vertical-align:top}"
           ".clips{display:flex;flex-wrap:wrap}.cap{font-size:12px;color:#9aa4b2}")
    parts = ["<!doctype html><meta charset=utf-8><title>Stage audit</title>",
             f"<style>{css}</style>", "<h1>Mission stage audit</h1>"]
    for audit in audits:
        parts.append(f"<h2>{escape(audit['run'])} — {escape(audit['result_line'])}</h2>")
        meta = f"state: {audit['state']}"
        if audit["reason"]:
            meta += f" · reason: {audit['reason']}"
        parts.append(f"<p class=cap>{escape(meta)}</p>")
        parts.append("<table><tr><th>Stage<th>Result<th>Reason<th>Att<th>Sim (s)<th>Wall (s)"
                     "<th>Frames<th>Key metrics<th>Note</tr>")
        for r in audit["rows"]:
            sim = A._fmt_span(r["sim_start_s"], r["sim_end_s"])
            wall = A._fmt_span(r["wall_start_s"], r["wall_end_s"])
            metrics = A._fmt_metrics(r["metrics"])
            parts.append(
                f"<tr><td>{r['stage']}<td class={r['result']}>{r['result']}"
                f"<td>{escape(r['reason'] or '—')}<td>{r['attempts'] or '—'}"
                f"<td>{escape(sim)}<td>{escape(wall)}<td>{r['n_frames'] or '—'}"
                f"<td>{escape(metrics)}<td>{escape(r['note'] or '')}</tr>")
        parts.append("</table>")
        clips = audit.get("_clips") or {}
        if clips:
            parts.append("<div class=clips>")
            for stage, rel in clips.items():
                parts.append(f"<div><video controls preload=none src='{escape(rel)}'></video>"
                             f"<div class=cap>{escape(stage)}</div></div>")
            parts.append("</div>")
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="mission run directories to audit")
    ap.add_argument("--out", required=True, help="output directory (created; files overwritten)")
    ap.add_argument("--fps", type=float, default=20.0, help="clip frame rate (default 20)")
    ap.add_argument("--no-clips", action="store_true", help="table only; skip the MP4 clips")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    audits, md_sections, json_runs = [], [], []
    for run_dir in args.run_dirs:
        audit = A.audit_run(run_dir)
        clips, clip_info = ({}, [])
        if not args.no_clips:
            clips, clip_info = _render_clips(audit, out_dir, fps=args.fps)
        audit["_clips"] = clips
        md_sections.append(A.markdown_table(audit, clips=clips))
        json_runs.append({k: v for k, v in audit.items() if not k.startswith("_")}
                         | {"clips": clips, "clip_renders": clip_info})
        audits.append(audit)

    title = ("# Mission stage audit\n\n"
             "One row per stage per run: result (PASS/FAIL/SKIPPED), failure reason by name, the\n"
             "sim-time and wall-time spans, recorded frame count, the key metrics the stage logs,\n"
             "and a link to a short overlaid clip cut from that stage's frames.\n\n")
    (out_dir / "stage_audit.md").write_text(title + "\n".join(md_sections) + "\n")
    (out_dir / "stage_audit.json").write_text(json.dumps({"runs": json_runs}, indent=1) + "\n")
    (out_dir / "index.html").write_text(_html(audits))

    summary = {"out": str(out_dir), "runs": [a["run"] for a in audits],
               "clips": sum(len(a["_clips"]) for a in audits)}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
