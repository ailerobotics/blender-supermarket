"""Static run-directory panel — the zero-process variant of the fleet dashboard.

The live server (:mod:`twin.dashboard.server`) needs a Python process running and
fetches the firstmate watch page's ``status.json``. This module builds the *same*
aggregated view **purely from a run directory's own files** (``mission.json``,
``nav-trace.jsonl``, ``bt_snapshot.json``, the rendered frames) and writes it as a
self-contained ``fleet.html`` + ``fleet.json`` pair *into* a directory, so the panel
can be served with nothing more than ``python3 -m http.server`` over that directory
(the ``python3 -m http.server`` pattern the watch page already uses) and reached from
the watch page. All media (newest frame, ``walkthrough.mp4``, ``bt_view.html``) are
linked **relative** when the output directory is the run directory, so every link
resolves over the one static mount with no extra routes.

Read-only and CPU-only: it never imports the mission, never boots Isaac, never writes
into the simulator's state — it only reads a finished/running run's artifacts and
writes ``fleet.html``/``fleet.json`` (into a separate ``--static-out`` directory by
default). A mission run is byte-identical whether or not this ever runs.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Any

from twin.dashboard import aggregate, store_state as store_state_mod
from twin.dashboard.page_static import STATIC_PAGE


def _tail_json_line(path: str) -> dict | None:
    """Last well-formed JSON object line of a (possibly large) .jsonl file."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            tail = fh.read().decode("utf-8", "ignore")
    except OSError:
        return None
    for line in reversed([ln for ln in tail.splitlines() if ln.strip()]):
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _newest_frame_rel(run_dir: str) -> str | None:
    """Run-relative path of the newest rendered frame, or ``None``."""
    frames = glob.glob(os.path.join(run_dir, "mission-frames", "*.png"))
    if not frames:
        return None
    newest = max(frames, key=os.path.getmtime)
    return os.path.relpath(newest, run_dir)


def pulse_from_run_dir(run_dir: str) -> dict:
    """Build an upstream-shaped pulse from a run directory's files alone.

    Matches the subset of the watch page ``status.json`` that
    :func:`aggregate.normalise_upstream` consumes. ``alive`` is *unknown* from files,
    so it is reported honestly as ``False`` (a static snapshot is a point in time);
    ``run_age_s`` lets the page show staleness instead.
    """
    name = os.path.basename(run_dir.rstrip("/"))
    frames = glob.glob(os.path.join(run_dir, "mission-frames", "*.png"))
    trace = _tail_json_line(os.path.join(run_dir, "nav-trace.jsonl")) or {}
    mission = aggregate._read_json(os.path.join(run_dir, "mission.json")) or {}
    try:
        age = int(time.time() - os.path.getmtime(run_dir))
    except OSError:
        age = None
    return {
        "run": name,
        "alive": False,
        "frames": len(frames),
        "run_age_s": age,
        "trace": {k: trace.get(k) for k in
                  ("tick", "t", "action", "coupled", "parked", "x", "y", "lin")},
        "mission": {k: mission.get(k) for k in
                    ("state", "reason", "physics_verified")},
        "videos": [],
        "runs": [],
    }


def build_run_snapshot(run_root: str, run_name: str | None = None, *,
                       store_state_path: str | None = None,
                       bt_path: str | None = None,
                       media_base: str = "",
                       triggers: list | None = None,
                       approval_required: bool | None = None) -> dict | None:
    """Aggregate one run directory into the read-only ``fleet.json`` view.

    Resolves ``run_name`` (newest run when absent/unknown) under ``run_root`` and
    merges its files through :func:`aggregate.build_state`. ``media_base`` points the
    frame/report links at an absolute base (e.g. the watch page) when set; empty ⇒
    run-relative links for a static mount rooted at the run directory. Returns
    ``None`` when no run directory resolves.
    """
    run_dir = aggregate.resolve_run_dir(run_root, run_name)
    if run_dir is None:
        return None
    resolved = os.path.basename(run_dir)
    mission = aggregate._read_json(os.path.join(run_dir, "mission.json"))
    pulse = pulse_from_run_dir(run_dir)
    store = store_state_mod.load_store_state(store_state_path)

    snap_path = bt_path or os.path.join(run_dir, "bt_snapshot.json")
    bt_snapshot = aggregate._read_json(snap_path)
    # bt_view.html sits beside fleet.html in the run dir ⇒ relative link; an absolute
    # media_base points it back through the base instead.
    if bt_snapshot is not None:
        view_url = (f"{media_base.rstrip('/')}/bt_view.html"
                    if media_base else "bt_view.html")
    else:
        view_url = None
    bt = aggregate.summarise_bt(bt_snapshot, view_url=view_url)

    base = media_base.rstrip("/") if media_base else ""
    if base:
        media = {"latest_frame": f"{base}/latest.jpg", "report": f"{base}/report"}
    else:
        frame = _newest_frame_rel(run_dir)
        media = {
            "latest_frame": frame,
            "walkthrough": ("walkthrough.mp4"
                            if os.path.exists(os.path.join(run_dir, "walkthrough.mp4"))
                            else None),
            "report": None,
        }

    return aggregate.build_state(
        upstream=pulse, upstream_ok=True, run_name=resolved, mission=mission,
        store=store, behaviour_tree=bt, triggers=triggers or [],
        upstream_base=base, media=media, approval_required=approval_required,
    )


def render_static_site(out_dir: str, snapshot: dict) -> tuple[str, str]:
    """Write ``fleet.html`` + ``fleet.json`` into ``out_dir``. Returns their paths.

    ``fleet.html`` is self-contained and polls ``./fleet.json`` (relative), so the
    pair serves correctly under any static mount (``python3 -m http.server``) rooted
    at ``out_dir``.
    """
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "fleet.json")
    html_path = os.path.join(out_dir, "fleet.html")
    tmp = json_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(snapshot, fh, indent=2)
    os.replace(tmp, json_path)
    with open(html_path, "w") as fh:
        fh.write(STATIC_PAGE)
    return html_path, json_path


def generate(run_root: str, out_dir: str, run_name: str | None = None,
             **kwargs: Any) -> dict | None:
    """One-shot: aggregate the resolved run and write the static site into ``out_dir``.

    Returns the snapshot dict (``None`` when no run resolves, in which case nothing is
    written). ``out_dir`` defaulting to the run directory itself gives relative media.
    """
    snapshot = build_run_snapshot(run_root, run_name, **kwargs)
    if snapshot is None:
        return None
    render_static_site(out_dir, snapshot)
    return snapshot
