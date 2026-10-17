"""App trigger / monitor queue — the opt-in phone path.

A phone app (store-vision ``missions`` / ``shelf-lens`` / ``lovable``, or any
operator) can POST a restock *request* here. The request is written as a JSON file
to a requests directory and listed back read-only; it is a **monitored queue, not
an actuator** — writing one does not drive the robot. A demand-driven job feed
(the rank-1 ``sim-smart-job-feed-v1`` adapter) or an operator can later consume the
queue to build a mission job. Enqueuing is gated behind ``--enable-trigger``
(default off), so the dashboard is read-only unless explicitly opened up.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from typing import Any

VALID_STATUSES = ("queued", "accepted", "rejected", "done")


class TriggerError(ValueError):
    """A trigger request that fails validation (honest refusal, no file written)."""


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def validate_request(payload: Any) -> dict:
    """Validate an inbound trigger payload. Raises :class:`TriggerError` on bad input.

    Required: ``sku`` (non-empty str), ``quantity`` (positive int). Optional:
    ``source`` (collect cell), ``shelf`` (destination slot), ``priority`` (int),
    ``note``, ``provenance`` (free-form origin, e.g. the app/user).
    """
    if not isinstance(payload, dict):
        raise TriggerError("request must be a JSON object")
    sku = payload.get("sku")
    if not isinstance(sku, str) or not sku.strip():
        raise TriggerError("'sku' is required (non-empty string)")
    qty = payload.get("quantity", 1)
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        raise TriggerError("'quantity' must be a positive integer")
    priority = payload.get("priority", 0)
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise TriggerError("'priority' must be an integer")
    clean = {
        "sku": sku.strip(),
        "quantity": qty,
        "priority": priority,
        "source": payload.get("source"),
        "shelf": payload.get("shelf"),
        "note": payload.get("note"),
        "provenance": payload.get("provenance", "unknown"),
    }
    return clean


def enqueue_trigger(requests_dir: str, payload: Any) -> dict:
    """Validate + persist a trigger request. Returns the stored record."""
    clean = validate_request(payload)
    os.makedirs(requests_dir, exist_ok=True)
    now = _utcnow_iso()
    rec = {
        "id": uuid.uuid4().hex[:12],
        "created_at": now,
        "status": "queued",
        "request": clean,
    }
    ts = now.replace(":", "").replace("-", "")
    fname = f"{ts}-{rec['id']}.json"
    tmp = os.path.join(requests_dir, "." + fname + ".tmp")
    final = os.path.join(requests_dir, fname)
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=2)
    os.replace(tmp, final)  # atomic publish
    return rec


def list_triggers(requests_dir: str, limit: int = 50) -> list[dict]:
    """List stored trigger requests, newest first (by ``created_at``)."""
    try:
        names = [n for n in os.listdir(requests_dir)
                 if n.endswith(".json") and not n.startswith(".")]
    except OSError:
        return []
    out = []
    for n in names:
        try:
            with open(os.path.join(requests_dir, n), "r") as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict):
            out.append(rec)
    out.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
    return out[:limit]
