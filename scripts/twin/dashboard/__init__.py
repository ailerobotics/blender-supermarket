"""Phone-first, read-only fleet dashboard for the Isaac restocking mission.

Opt-in and strictly additive: this package is never imported or launched by the
mission (``scripts/twin/restock/mission.py``), so a mission run is byte-identical
whether or not the dashboard is up. It runs in two shapes, same aggregation and same
page: a standalone CPU server (:mod:`twin.dashboard.server`) and a zero-process
*static panel* (:mod:`twin.dashboard.snapshot` + :mod:`twin.dashboard.page_static`)
rendered into a run directory and served with ``python3 -m http.server``.

It does NOT replace the existing firstmate watch page (``:8787``): it *reads* that
page's ``status.json`` (configurable URL) for the live simulator pulse and links
its media (latest frame, run/stage videos) back to it, and it *reads* the mission
run directories under ``datasets/.../mission/`` for per-run detail (stage timeline
from ``mission.json`` events, an optional behaviour-tree view when a lane publishes
one). It additionally overlays store-vision store state (shelf fill / depletion /
alerts) from a documented JSON export, and exposes an opt-in app trigger/monitor
queue. Pure aggregation lives in :mod:`twin.dashboard.aggregate` and is unit tested
with no network, no GPU and no simulator.
"""
