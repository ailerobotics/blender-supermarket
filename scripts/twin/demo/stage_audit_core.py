"""Per-stage audit of one finished mission run: which stage failed, why, when, with what.

Pure (no cv2/Isaac/Blender); reads a run dir only. The stage ladder, result and failure reason
come from the durable record ``run_evidence`` already reads -- ``mission.json`` (the inventory
event log + ``stage_timeline``) -- so the verdict matches the mission's own stock-commit gate.
Sim-time boundaries come from ``stage_timeline`` (Isaac stage clock); wall-time boundaries are
interpolated from ``nav-trace.jsonl``'s ``wall`` field against the event mission clock ``t``.

``scripts/stage_audit.py`` turns :func:`audit_run` into Markdown + JSON and (via
``twin.demo.stage_video``) cuts one overlaid MP4 clip per stage. See ``tests/test_stage_audit.py``.
"""
from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import run_evidence as E

STAGES = E.STAGES

# Status (run_evidence.stage_status) -> the three audit results the captain asked for.
_RESULT = {
    "succeeded": "PASS",
    "failed": "FAIL",
    "unavailable": "FAIL",   # a provisioned-capability gap is recorded FAILED by the mission
    "started": "FAIL",       # began but never reached success in a terminal run
    "shortcut": "SKIPPED",   # synthesized by a start_at bench, not performed
    "missing": "SKIPPED",    # never driven (bounded / cut / not reached)
}

# Evidence keys that are bookkeeping, not a stage metric.
_SKIP_METRIC_KEYS = {"observation_id", "simulation_time", "job_id", "sequence", "shortcut", "reason"}


@dataclass
class StageRow:
    stage: str
    result: str
    reason: str = ""
    attempts: int = 0
    sim_start: float | None = None
    sim_end: float | None = None
    wall_start: float | None = None
    wall_end: float | None = None
    n_frames: int = 0
    metrics: dict = field(default_factory=dict)
    note: str = ""

    def as_dict(self):
        return {
            "stage": self.stage, "result": self.result, "reason": self.reason,
            "attempts": self.attempts,
            "sim_start_s": _round(self.sim_start), "sim_end_s": _round(self.sim_end),
            "wall_start_s": _round(self.wall_start), "wall_end_s": _round(self.wall_end),
            "n_frames": self.n_frames, "metrics": self.metrics, "note": self.note,
        }


def _round(v, nd=2):
    return round(float(v), nd) if isinstance(v, (int, float)) else None


def _read_wall_map(run_dir):
    """``(ts, walls)`` parallel sorted arrays mapping mission clock ``t`` -> wall seconds from
    ``nav-trace.jsonl``; ``([], [])`` when the trace has no ``wall`` field."""
    path = Path(run_dir) / "nav-trace.jsonl"
    pairs = []
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t, wall = r.get("t"), r.get("wall")
                if isinstance(t, (int, float)) and isinstance(wall, (int, float)):
                    pairs.append((float(t), float(wall)))
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _wall_at(ts, walls, t):
    """Wall second at mission time ``t`` by linear interpolation (clamped to the ends); None
    when the mapping is empty or ``t`` is None."""
    if not ts or t is None:
        return None
    if t <= ts[0]:
        return walls[0]
    if t >= ts[-1]:
        return walls[-1]
    i = bisect.bisect_right(ts, t)
    t0, t1, w0, w1 = ts[i - 1], ts[i], walls[i - 1], walls[i]
    if t1 == t0:
        return w0
    return w0 + (w1 - w0) * (t - t0) / (t1 - t0)


def _timeline_facts(mission):
    """Per stage, from ``mission.json`` ``stage_timeline``: sim-time span, mission-time span,
    attempt count, the last failure reason, and whether it ever succeeded."""
    facts = {}
    for ev in mission.get("stage_timeline") or []:
        data = ev.get("data") or {}
        stage = data.get("stage")
        if stage is None:
            continue
        f = facts.setdefault(stage, {"sim": [], "t": [], "attempts": set(),
                                     "fail_reason": None, "succeeded": False})
        if isinstance(data.get("sim_time"), (int, float)):
            f["sim"].append(float(data["sim_time"]))
        if isinstance(ev.get("t"), (int, float)):
            f["t"].append(float(ev["t"]))
        if isinstance(data.get("attempt"), int):
            f["attempts"].add(data["attempt"])
        typ = ev.get("type")
        if typ == "stage_failed" and data.get("reason"):
            f["fail_reason"] = str(data["reason"])
        elif typ == "stage_succeeded":
            f["succeeded"] = True
    return facts


def _stage_metrics(run, stage):
    """Compact scalar/short-string metrics the stage logged in its success (or latest) evidence."""
    evidence = {}
    for ev in run.events:
        if E.event_stage(ev.get("state")) != stage:
            continue
        ev_ev = ev.get("evidence") or {}
        if ev.get("state") == stage + "_succeeded" or ev_ev:
            evidence = ev_ev
    out = {}
    for key, value in evidence.items():
        if key in _SKIP_METRIC_KEYS:
            continue
        if isinstance(value, bool) or isinstance(value, (int, float)):
            out[key] = value
        elif isinstance(value, str) and len(value) <= 60:
            out[key] = value
        if len(out) >= 6:
            break
    return out


def audit_run(run_dir):
    """Audit one mission run dir into ``{run, state, reason, labels, clock_*, rows: [StageRow…]}``.

    ``rows`` covers the seven canonical stages in order. ``frames_by_stage`` is returned under
    ``_frames`` for the clip step (the CLI), keyed by stage -> list of ``run_evidence.Frame``."""
    run = E.load_run(run_dir)
    status = E.stage_status(run)
    facts = _timeline_facts(run.mission)
    frames = E.frames_by_stage(run)
    ts, walls = _read_wall_map(run_dir)
    windows = E.stage_windows(run)                  # mission-time span per stage from the trace
    stages_driven = run.mission.get("stages_driven")
    not_run = set(run.mission.get("not_run") or [])
    stop_after = run.mission.get("stop_after")
    start_at = (run.mission.get("shortcut") or {}).get("start_at")

    rows = []
    for stage in STAGES:
        st = status.get(stage, "missing")
        result = _RESULT.get(st, "SKIPPED")
        fact = facts.get(stage, {})
        sim = fact.get("sim") or []
        tt = fact.get("t") or []
        sim_start = min(sim) if sim else None
        sim_end = max(sim) if sim else None
        t_start = min(tt) if tt else (windows.get(stage, (None, None))[0])
        t_end = max(tt) if tt else (windows.get(stage, (None, None))[1])
        if sim_start is None and stage in frames and frames[stage]:
            sim_start = frames[stage][0].sim_time
            sim_end = frames[stage][-1].sim_time
        attempts = max(fact.get("attempts") or {0}) if fact.get("attempts") else (1 if sim else 0)

        reason = ""
        if result == "FAIL":
            if st == "unavailable":
                reason = "capability unavailable"
            else:
                reason = fact.get("fail_reason") or E.short_reason(run.mission.get("reason"), 200)

        note = ""
        if st == "shortcut":
            note = f"synthesized (start_at={start_at})" if start_at else "synthesized prefix"
        elif st == "unavailable":
            note = "capability unavailable"
        elif st == "missing":
            if stage in not_run and stop_after:
                note = f"not run (stop_after={stop_after})"
            elif isinstance(stages_driven, list) and stage not in stages_driven:
                note = "not run (bounded)"
            else:
                note = "not reached"

        rows.append(StageRow(
            stage=stage, result=result, reason=reason, attempts=attempts,
            sim_start=sim_start, sim_end=sim_end,
            wall_start=_wall_at(ts, walls, t_start), wall_end=_wall_at(ts, walls, t_end),
            n_frames=len(frames.get(stage, [])),
            metrics=_stage_metrics(run, stage), note=note,
        ))

    # The burned-in bench caption (key 'caption') is long and already implied by stop_after/start_at
    # plus the mode/source labels, so it is dropped here to keep the audit readable.
    labels = [lbl.text + (f" [{lbl.stage}]" if lbl.stage else "")
              for lbl in E.workaround_labels(run) if lbl.key != "caption"]
    return {
        "run": run.name,
        "run_dir": str(run.run_dir),
        "state": run.mission.get("state", "unknown"),
        "reason": E.short_reason(run.mission.get("reason"), 200),
        "result_line": E.result_line(run),
        "stages_driven": stages_driven,
        "not_run": sorted(not_run),
        "stop_after": stop_after,
        "start_at": start_at,
        "clock_offset_s": round(run.clock_offset, 4),
        "clock_source": run.clock_source,
        "labels": labels,
        "rows": [r.as_dict() for r in rows],
        "_rows": rows,
        "_frames": frames,
    }


# ---------------- rendering helpers (text; the CLI adds the clips) ----------------
def _fmt_span(lo, hi):
    if lo is None and hi is None:
        return "—"
    return f"{_round(lo)}→{_round(hi)}"


def _fmt_metrics(metrics):
    if not metrics:
        return "—"
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            v = round(v, 4)
        parts.append(f"{k}={v}")
    return "; ".join(parts)


def markdown_table(audit, clips=None):
    """The per-run Markdown section. ``clips`` maps stage -> clip relative path (or None)."""
    clips = clips or {}
    out = []
    out.append(f"### {audit['run']} — {audit['result_line']}")
    out.append("")
    meta = [f"state: `{audit['state']}`"]
    if audit["reason"]:
        meta.append(f"reason: `{audit['reason']}`")
    if audit["stop_after"]:
        meta.append(f"stop_after: `{audit['stop_after']}`")
    if audit["start_at"]:
        meta.append(f"start_at: `{audit['start_at']}`")
    meta.append(f"clock offset: {audit['clock_offset_s']} s ({audit['clock_source']})")
    out.append(" · ".join(meta))
    if audit["labels"]:
        out.append("")
        out.append("workaround/provenance labels: " + ", ".join(f"`{x}`" for x in audit["labels"]))
    out.append("")
    header = ("| Stage | Result | Reason | Att | Sim start→end (s) | Wall start→end (s) "
              "| Frames | Clip | Key metrics | Note |")
    out.append(header)
    out.append("|---|---|---|--:|---|---|--:|---|---|---|")
    for r in audit["_rows"]:
        clip = clips.get(r.stage)
        clip_cell = f"[clip]({clip})" if clip else "—"
        out.append("| {stage} | {result} | {reason} | {att} | {sim} | {wall} | {nf} | {clip} "
                   "| {metrics} | {note} |".format(
                       stage=r.stage, result=r.result, reason=r.reason or "—", att=r.attempts or "—",
                       sim=_fmt_span(r.sim_start, r.sim_end), wall=_fmt_span(r.wall_start, r.wall_end),
                       nf=r.n_frames or "—", clip=clip_cell, metrics=_fmt_metrics(r.metrics),
                       note=r.note or ""))
    out.append("")
    return "\n".join(out)
