#!/usr/bin/env python3
"""Launch (or dry-run) the phone-first fleet dashboard — CPU only, no GPU (RUNBOOK §17).

Read-only: it reads the mission run directories and the upstream watch page's
``status.json``; it never writes to a run, never boots Isaac and is never imported by
the mission, so a mission run is byte-identical whether or not this is up.

  # serve on the tailnet (read-only; no trigger path)
  PYTHONPATH=scripts python3 scripts/run_fleet_dashboard.py

  # print the aggregated status once and exit (the dry-run transcript)
  PYTHONPATH=scripts python3 scripts/run_fleet_dashboard.py --dry-run

  # with the store overlay + opt-in phone trigger queue
  PYTHONPATH=scripts python3 scripts/run_fleet_dashboard.py \
      --store-state /path/to/store-state.json \
      --enable-trigger --requests-dir /tmp/fleet-dashboard-requests
"""
import argparse
import json
import os
import sys

# Derive ROOT from the script location so it runs in any worktree/clone, and make the
# sibling ``scripts/`` importable as the ``twin`` package without a preset PYTHONPATH.
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from twin.dashboard.server import (  # noqa: E402
    Config, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_RUN_ROOT, DEFAULT_UPSTREAM,
    compute_status, serve,
)


def build_config(argv=None) -> tuple[Config, bool]:
    p = argparse.ArgumentParser(description="Read-only fleet mission + store dashboard (CPU).")
    p.add_argument("--host", default=DEFAULT_HOST, help="bind address (default tailnet IP)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port (8794+; 8787/90/91/93 taken)")
    p.add_argument("--upstream", default=DEFAULT_UPSTREAM, help="existing watch-page base URL (read-only)")
    p.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="mission run directories root")
    p.add_argument("--store-state", default=None, help="store-vision store-state JSON export (overlay)")
    p.add_argument("--bt", default=None,
                   help="override path to a bt_snapshot.json (default: auto-discover "
                        "the per-run bt_snapshot.json sim-restock-bt-v1 writes into the run dir)")
    p.add_argument("--requests-dir", default=None, help="trigger-request queue directory")
    p.add_argument("--enable-trigger", action="store_true",
                   help="allow POST /trigger to enqueue requests (default off: read-only)")
    p.add_argument("--upstream-timeout", type=float, default=2.0, help="upstream fetch timeout (s)")
    p.add_argument("--dry-run", action="store_true", help="print the aggregated status once and exit")
    a = p.parse_args(argv)

    requests_dir = a.requests_dir
    if a.enable_trigger and not requests_dir:
        p.error("--enable-trigger requires --requests-dir")

    cfg = Config(
        host=a.host, port=a.port, upstream_base=a.upstream, run_root=a.run_root,
        store_state_path=a.store_state, bt_path=a.bt, requests_dir=requests_dir,
        trigger_enabled=bool(a.enable_trigger), upstream_timeout=a.upstream_timeout,
    )
    return cfg, a.dry_run


def main(argv=None) -> int:
    cfg, dry = build_config(argv)
    if dry:
        print(json.dumps(compute_status(cfg), indent=2))
        return 0
    print(f"fleet dashboard: http://{cfg.host}:{cfg.port}/  "
          f"(upstream {cfg.upstream_base}, run-root {cfg.run_root}, "
          f"trigger {'ON' if cfg.trigger_enabled else 'off'})", file=sys.stderr)
    serve(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
