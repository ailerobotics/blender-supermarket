"""Pure aggregation for the fleet dashboard — no network, no GPU, no simulator.

Every function here takes already-parsed data (dicts, paths) and returns a plain
JSON-serialisable dict. The server (:mod:`twin.dashboard.server`) owns all I/O
(fetching the upstream ``status.json``, reading files); keeping the merge pure is
what lets the whole view be unit tested from fixtures.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any

# The mission pipeline, in order. Mirrors ``twin.restock.mission.ACTIONS`` with the
# leading ``reserved`` reservation state the mission records before ``collect``.
# Kept as a literal (not imported) so the dashboard never pulls the mission module.
PIPELINE = ("reserved", "collect", "couple", "outbound", "park", "transfer",
            "verify", "return")
TERMINAL_OK = ("complete",)
TERMINAL_BAD = ("failed",)


def utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def latest_run_dir(run_root: str) -> str | None:
    """Newest mission run directory under ``run_root`` (by mtime), or ``None``.

    A "run directory" is any immediate child holding a ``mission.json``; this is the
    fallback when the upstream pulse is unavailable and so cannot name the live run.
    """
    try:
        entries = [os.path.join(run_root, n) for n in os.listdir(run_root)]
    except OSError:
        return None
    runs = [p for p in entries
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "mission.json"))]
    if not runs:
        return None
    return max(runs, key=lambda p: os.path.getmtime(p))


def read_run_mission(run_root: str, run_name: str | None) -> tuple[str | None, dict | None]:
    """Resolve ``mission.json`` for ``run_name`` under ``run_root``.

    Returns ``(resolved_run_name, mission_dict_or_None)``. When ``run_name`` is
    falsy or has no ``mission.json``, falls back to the newest run directory.
    """
    candidate = None
    if run_name:
        candidate = os.path.join(run_root, run_name)
        if not os.path.exists(os.path.join(candidate, "mission.json")):
            candidate = None
    if candidate is None:
        candidate = latest_run_dir(run_root)
    if candidate is None:
        return None, None
    return os.path.basename(candidate), _read_json(os.path.join(candidate, "mission.json"))


def stage_timeline(mission: dict | None, current_action: str | None = None) -> dict:
    """Turn a run's ``mission.json`` into a phone-friendly pipeline view.

    ``current_action`` (from the live trace, when available) overrides the stage
    inferred from the recorded events so a still-running stage shows as active even
    before its event is written.
    """
    events = []
    if isinstance(mission, dict) and isinstance(mission.get("events"), list):
        events = [e for e in mission["events"] if isinstance(e, dict)]

    reached = {}  # stage -> last state seen for it
    for ev in events:
        st = ev.get("state")
        if isinstance(st, str):
            reached[st] = ev

    result_state = (mission or {}).get("state")
    reason = (mission or {}).get("reason")
    physics_verified = (mission or {}).get("physics_verified")

    # The "active" stage: prefer the live trace action, else the last recorded event's
    # state if it is a pipeline stage, else the furthest pipeline stage reached.
    active = None
    if current_action in PIPELINE:
        active = current_action
    elif events:
        last = events[-1].get("state")
        active = last if last in PIPELINE else None
    if active is None:
        for st in reversed(PIPELINE):
            if st in reached:
                active = st
                break

    failed_stage = None
    if result_state in TERMINAL_BAD and events:
        # the recorded failure event sometimes names the stage in its evidence
        for ev in reversed(events):
            if ev.get("state") in TERMINAL_BAD:
                failed_stage = active
                break

    steps = []
    active_idx = PIPELINE.index(active) if active in PIPELINE else -1
    for i, stage in enumerate(PIPELINE):
        if i == active_idx and result_state not in TERMINAL_OK:
            # The stage the run is sitting on: failed if the run failed here (a
            # recorded stage-entry event means "entered", not "completed"), else active.
            status = "failed" if result_state in TERMINAL_BAD else "active"
        elif stage in reached and reached[stage].get("state") not in TERMINAL_BAD:
            status = "done"
        elif i < active_idx:
            status = "done"
        else:
            status = "pending"
        steps.append({"stage": stage, "status": status})

    if result_state in TERMINAL_OK:
        for s in steps:
            s["status"] = "done"

    verdict = "running"
    if result_state in TERMINAL_OK:
        verdict = "complete"
    elif result_state in TERMINAL_BAD:
        verdict = "failed"

    return {
        "active": active,
        "verdict": verdict,
        "reason": reason,
        "physics_verified": physics_verified,
        "failed_stage": failed_stage,
        "steps": steps,
        "event_count": len(events),
    }


def normalise_upstream(upstream: dict | None) -> dict:
    """Project the firstmate watch page ``status.json`` into the fields we surface.

    Tolerant: a missing/garbled upstream yields a sane empty pulse with
    ``alive=False`` rather than raising.
    """
    up = upstream if isinstance(upstream, dict) else {}
    trace = up.get("trace") if isinstance(up.get("trace"), dict) else {}
    return {
        "run": up.get("run"),
        "alive": bool(up.get("alive", False)),
        "frames": up.get("frames", 0),
        "run_age_s": up.get("run_age_s"),
        "ffmpeg": up.get("ffmpeg"),
        "trace": {
            "tick": trace.get("tick"),
            "t": trace.get("t"),
            "action": trace.get("action"),
            "coupled": trace.get("coupled"),
            "parked": trace.get("parked"),
            "x": trace.get("x"),
            "y": trace.get("y"),
            "lin": trace.get("lin"),
        },
        "result": up.get("mission") if isinstance(up.get("mission"), dict) else {},
        "videos": up.get("videos") if isinstance(up.get("videos"), list) else [],
        "runs": up.get("runs") if isinstance(up.get("runs"), list) else [],
    }


def build_state(*, upstream: dict | None, upstream_ok: bool,
                run_name: str | None, mission: dict | None,
                store: dict | None, behaviour_tree: dict | None,
                triggers: list | None, upstream_base: str) -> dict:
    """Merge the live pulse, the resolved run, store state, BT view and triggers.

    The single read-only JSON the page (and any phone app) consumes.
    """
    up = normalise_upstream(upstream)
    timeline = stage_timeline(mission, up["trace"].get("action"))
    return {
        "schema": "fm-fleet-dashboard.v1",
        "generated_at": utcnow_iso(),
        "upstream": {"ok": bool(upstream_ok), "base": upstream_base},
        "mission": {
            "run": run_name or up["run"],
            "alive": up["alive"],
            "frames": up["frames"],
            "run_age_s": up["run_age_s"],
            "trace": up["trace"],
            "result": up["result"] or {
                k: (mission or {}).get(k) for k in ("state", "reason", "physics_verified")
            },
            "timeline": timeline,
            "videos": up["videos"],
            "runs": up["runs"],
        },
        "behaviour_tree": behaviour_tree,
        "store": store,
        "triggers": triggers or [],
        "media": {
            "latest_frame": f"{upstream_base}/latest.jpg",
            "report": f"{upstream_base}/report",
        },
    }
