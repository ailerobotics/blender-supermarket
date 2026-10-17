"""The read-only dashboard HTTP server (stdlib only; CPU; no GPU).

Routes:
  GET  /               -> phone-first HTML page
  GET  /status.json    -> aggregated mission + store + triggers view
  GET  /config.json    -> {trigger_enabled} (the page uses this to show the form)
  GET  /triggers.json  -> stored trigger requests (newest first)
  GET  /healthz        -> "ok"
  POST /trigger        -> enqueue a restock request (ONLY when trigger_enabled)

The server fetches the upstream watch-page ``status.json`` with a short timeout and
degrades gracefully (``upstream.ok=false``) when it is down; it never proxies the
upstream's media (those links point back at the upstream base).
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from twin.dashboard import aggregate, store_state, triggers
from twin.dashboard.page import PAGE

DEFAULT_RUN_ROOT = "/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission"
DEFAULT_UPSTREAM = "http://100.100.47.65:8787"
DEFAULT_HOST = "100.100.47.65"
DEFAULT_PORT = 8794


@dataclass
class Config:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    upstream_base: str = DEFAULT_UPSTREAM
    run_root: str = DEFAULT_RUN_ROOT
    store_state_path: str | None = None
    bt_path: str | None = None
    requests_dir: str | None = None
    trigger_enabled: bool = False
    upstream_timeout: float = 2.0


def fetch_upstream(base: str, timeout: float) -> tuple[dict | None, bool]:
    """Fetch ``<base>/status.json``. Returns ``(dict|None, ok)``; never raises."""
    url = base.rstrip("/") + "/status.json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed tailnet URL)
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None, False
    return (data if isinstance(data, dict) else None), isinstance(data, dict)


def resolve_bt(cfg: Config, run_dir: str | None, run_name: str | None,
               upstream_base: str) -> dict | None:
    """Resolve the behaviour-tree snapshot for the live run (sim-restock-bt-v1).

    An explicit ``--bt`` path wins; otherwise the per-run ``bt_snapshot.json`` the
    bt lane writes into the run directory is auto-discovered, so the panel lights up
    on the first bt-enabled run with zero config. The richer self-contained
    ``bt_view.html`` phone page (same directory) is linked via the watch page, which
    serves the run tree under ``<media_prefix>/<run>/``.
    """
    view_url = None
    snapshot = None
    if cfg.bt_path:
        snapshot = aggregate._read_json(cfg.bt_path)
    elif run_dir:
        snapshot = aggregate._read_json(os.path.join(run_dir, "bt_snapshot.json"))
        if snapshot is not None and run_name:
            media_prefix = os.path.basename(cfg.run_root.rstrip("/"))
            view_url = f"{upstream_base}/{media_prefix}/{run_name}/bt_view.html"
    return aggregate.summarise_bt(snapshot, view_url=view_url)


def compute_status(cfg: Config) -> dict:
    """Assemble the full aggregated status dict (pure except for the file/URL reads)."""
    upstream, ok = fetch_upstream(cfg.upstream_base, cfg.upstream_timeout)
    run_name = (upstream or {}).get("run") if isinstance(upstream, dict) else None
    run_dir = aggregate.resolve_run_dir(cfg.run_root, run_name)
    resolved_run = os.path.basename(run_dir) if run_dir else None
    mission = aggregate._read_json(os.path.join(run_dir, "mission.json")) if run_dir else None
    store = store_state.load_store_state(cfg.store_state_path)
    upstream_base = cfg.upstream_base.rstrip("/")
    bt = resolve_bt(cfg, run_dir, resolved_run, upstream_base)
    trig = triggers.list_triggers(cfg.requests_dir) if cfg.requests_dir else []
    return aggregate.build_state(
        upstream=upstream, upstream_ok=ok,
        run_name=resolved_run, mission=mission,
        store=store, behaviour_tree=bt, triggers=trig,
        upstream_base=upstream_base,
    )


class _Handler(BaseHTTPRequestHandler):
    cfg: Config = None  # set on the server instance's handler class

    def log_message(self, *a):  # quiet by default
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/status.json":
            self._send(200, compute_status(self.cfg))
        elif path == "/config.json":
            self._send(200, {"trigger_enabled": self.cfg.trigger_enabled})
        elif path == "/triggers.json":
            self._send(200, triggers.list_triggers(self.cfg.requests_dir) if self.cfg.requests_dir else [])
        elif path == "/healthz":
            self._send(200, "ok", "text/plain")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/trigger":
            self._send(404, {"error": "not found"})
            return
        if not self.cfg.trigger_enabled or not self.cfg.requests_dir:
            self._send(403, {"error": "trigger path disabled (start with --enable-trigger)"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except (ValueError, OSError):
            self._send(400, {"error": "invalid JSON body"})
            return
        try:
            rec = triggers.enqueue_trigger(self.cfg.requests_dir, payload)
        except triggers.TriggerError as exc:
            self._send(400, {"error": str(exc)})
            return
        self._send(201, rec)


def make_server(cfg: Config) -> ThreadingHTTPServer:
    handler = type("_BoundHandler", (_Handler,), {"cfg": cfg})
    return ThreadingHTTPServer((cfg.host, cfg.port), handler)


def serve(cfg: Config) -> None:
    httpd = make_server(cfg)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
