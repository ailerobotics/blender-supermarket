#!/usr/bin/env python3
"""Launch, dry-run, or statically render the phone-first fleet dashboard — CPU only,
no GPU (RUNBOOK §17).

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

  # smart variant: a phone approval step honoured by the lane-1 job feed
  PYTHONPATH=scripts python3 scripts/run_fleet_dashboard.py \
      --enable-trigger --approval-trigger --requests-dir /tmp/fleet-dashboard-requests

  # static panel over a run directory (no process; serve with python3 -m http.server)
  PYTHONPATH=scripts python3 scripts/run_fleet_dashboard.py --static-out RUN_DIR
  python3 -m http.server --directory RUN_DIR 8794 --bind 100.100.47.65   # then open /fleet.html
"""
import argparse
import json
import os
import sys
import time

# Derive ROOT from the script location so it runs in any worktree/clone, and make the
# sibling ``scripts/`` importable as the ``twin`` package without a preset PYTHONPATH.
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from twin.dashboard.server import (  # noqa: E402
    Config, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_RUN_ROOT, DEFAULT_UPSTREAM,
    compute_status, serve,
)
from twin.dashboard import snapshot  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only fleet mission + store dashboard (CPU).")
    p.add_argument("--host", default=DEFAULT_HOST, help="bind address (default tailnet IP)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port (8794+; 8787/90/91/93 taken)")
    p.add_argument("--upstream", default=DEFAULT_UPSTREAM, help="existing watch-page base URL (read-only)")
    p.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="mission run directories root")
    p.add_argument("--run", default=None, help="run name to resolve (default: newest run)")
    p.add_argument("--store-state", default=None, help="store-vision store-state JSON export (overlay)")
    p.add_argument("--bt", default=None,
                   help="override path to a bt_snapshot.json (default: auto-discover "
                        "the per-run bt_snapshot.json sim-restock-bt-v1 writes into the run dir)")
    p.add_argument("--requests-dir", default=None, help="trigger-request queue directory")
    p.add_argument("--enable-trigger", action="store_true",
                   help="allow POST /trigger to enqueue requests (default off: read-only)")
    p.add_argument("--approval-trigger", action="store_true",
                   help="smart variant: requests land pending_approval and need a phone "
                        "approve/reject; an approved request is written as a job-plan the "
                        "lane-1 feed honours (requires --enable-trigger; default off)")
    p.add_argument("--upstream-timeout", type=float, default=2.0, help="upstream fetch timeout (s)")
    p.add_argument("--dry-run", action="store_true", help="print the aggregated status once and exit")
    p.add_argument("--static-out", default=None, metavar="DIR",
                   help="render the static fleet.html + fleet.json into DIR and exit "
                        "(serve over DIR with python3 -m http.server; no live process). "
                        "Pass the run dir itself for relative media links.")
    p.add_argument("--media-base", default="",
                   help="absolute base URL for the static panel's frame/report links "
                        "(default: empty ⇒ run-relative links over the static mount)")
    p.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                   help="with --static-out, re-render every SECONDS (keeps the static "
                        "mount fresh); 0 ⇒ one shot")
    return p


def build_config(argv=None) -> tuple[Config, argparse.Namespace]:
    p = build_parser()
    a = p.parse_args(argv)

    requests_dir = a.requests_dir
    if a.enable_trigger and not requests_dir:
        p.error("--enable-trigger requires --requests-dir")
    if a.approval_trigger and not a.enable_trigger:
        p.error("--approval-trigger requires --enable-trigger")

    cfg = Config(
        host=a.host, port=a.port, upstream_base=a.upstream, run_root=a.run_root,
        store_state_path=a.store_state, bt_path=a.bt, requests_dir=requests_dir,
        trigger_enabled=bool(a.enable_trigger),
        approval_required=bool(a.approval_trigger),
        upstream_timeout=a.upstream_timeout,
    )
    return cfg, a


def _render_static(cfg: Config, a: argparse.Namespace) -> int:
    kwargs = dict(store_state_path=cfg.store_state_path, bt_path=cfg.bt_path,
                  media_base=a.media_base,
                  approval_required=cfg.approval_required or None)

    def once() -> bool:
        snap = snapshot.generate(cfg.run_root, a.static_out, a.run, **kwargs)
        if snap is None:
            print(f"no run directory under {cfg.run_root}", file=sys.stderr)
            return False
        print(f"static panel: {os.path.join(a.static_out, 'fleet.html')} "
              f"(run {snap['mission']['run']})", file=sys.stderr)
        return True

    if a.watch and a.watch > 0:
        print(f"re-rendering every {a.watch:g}s (Ctrl-C to stop)", file=sys.stderr)
        try:
            while True:
                once()
                time.sleep(a.watch)
        except KeyboardInterrupt:
            return 0
    return 0 if once() else 1


def main(argv=None) -> int:
    cfg, a = build_config(argv)
    if a.static_out:
        return _render_static(cfg, a)
    if a.dry_run:
        print(json.dumps(compute_status(cfg), indent=2))
        return 0
    print(f"fleet dashboard: http://{cfg.host}:{cfg.port}/  "
          f"(upstream {cfg.upstream_base}, run-root {cfg.run_root}, "
          f"trigger {'ON' if cfg.trigger_enabled else 'off'}"
          f"{', approval' if cfg.approval_required else ''})", file=sys.stderr)
    serve(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
