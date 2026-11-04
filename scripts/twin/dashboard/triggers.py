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

VALID_STATUSES = ("queued", "pending_approval", "approved", "accepted", "rejected", "done")

# The job-plan contract the rank-1 demand-feed consumer (`twin.restock.job_from_plan`)
# reads via ``run_mobile_restock.py --job-from``. The approval variant emits an
# approved request in this shape so the mission job feed *honours* a phone approval.
JOB_PLAN_SCHEMA = "storevision.restock.job_plan/v1"


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


def _atomic_write(requests_dir: str, fname: str, rec: dict) -> str:
    os.makedirs(requests_dir, exist_ok=True)
    tmp = os.path.join(requests_dir, "." + fname + ".tmp")
    final = os.path.join(requests_dir, fname)
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=2)
    os.replace(tmp, final)  # atomic publish
    return final


def enqueue_trigger(requests_dir: str, payload: Any,
                    *, approval_required: bool = False) -> dict:
    """Validate + persist a trigger request. Returns the stored record.

    ``approval_required`` is the smart ``--approval-trigger`` variant: the request
    lands as ``pending_approval`` (carrying an ``approval`` block) instead of
    ``queued`` and does nothing until a phone ``approve``/``reject`` decision. The
    working default (``approval_required=False``) is the plain monitored queue.
    """
    clean = validate_request(payload)
    now = _utcnow_iso()
    rec = {
        "id": uuid.uuid4().hex[:12],
        "created_at": now,
        "status": "pending_approval" if approval_required else "queued",
        "request": clean,
    }
    if approval_required:
        rec["approval"] = {"state": "pending", "required": True, "by": None, "at": None}
    ts = now.replace(":", "").replace("-", "")
    _atomic_write(requests_dir, f"{ts}-{rec['id']}.json", rec)
    return rec


def _record_path(requests_dir: str, trigger_id: str) -> str | None:
    try:
        names = [n for n in os.listdir(requests_dir)
                 if n.endswith(".json") and not n.startswith(".")]
    except OSError:
        return None
    for n in names:
        if n.endswith(f"-{trigger_id}.json") or n == f"{trigger_id}.json":
            return os.path.join(requests_dir, n)
    return None


def _as_job_plan(rec: dict) -> dict:
    """Project an approved request into the ``job_from_plan`` contract (honest shape).

    The source cell / destination slot come from the request; the mission job feed
    (`--job-from`) re-validates both against ``scripts/planogram.json`` and refuses an
    unknown slot or SKU, so this is a *proposal* carrying the approval, never a silent
    actuation. Pose/xyz are store-vision's to fill at plan time and are left absent.
    """
    req = rec.get("request", {})
    return {
        "schema": JOB_PLAN_SCHEMA,
        "generated_at": rec.get("created_at"),
        "sku": req.get("sku"),
        "quantity": req.get("quantity"),
        "source": {"location": req.get("source")},
        "destination": {"slot_id": req.get("shelf")},
        "priority": req.get("priority", 0),
        "provenance": {
            "planner": "fleet-dashboard.approval-trigger",
            "task_id": rec.get("id"),
            "approval": dict(rec.get("approval", {})),
        },
    }


def decide_trigger(requests_dir: str, trigger_id: str, *, approve: bool,
                   by: str = "phone") -> dict:
    """Apply a phone approve/reject decision to a ``pending_approval`` request.

    On approve, also writes the approved request as a job-plan proposal under
    ``<requests_dir>/approved/`` in the ``job_from_plan`` contract, so an operator or
    the rank-1 feed can honour it with ``run_mobile_restock.py --job-from``. Raises
    :class:`TriggerError` for an unknown id or a request not awaiting approval.
    """
    path = _record_path(requests_dir, trigger_id)
    if path is None:
        raise TriggerError(f"no such request {trigger_id!r}")
    try:
        with open(path, "r") as fh:
            rec = json.load(fh)
    except (OSError, ValueError) as exc:
        raise TriggerError(f"request {trigger_id!r} unreadable") from exc
    if rec.get("status") != "pending_approval":
        raise TriggerError(
            f"request {trigger_id!r} is {rec.get('status')!r}, not awaiting approval")
    now = _utcnow_iso()
    rec["status"] = "approved" if approve else "rejected"
    rec["approval"] = {
        "state": "approved" if approve else "rejected",
        "required": True, "by": by, "at": now,
    }
    _atomic_write(requests_dir, os.path.basename(path), rec)
    if approve:
        _atomic_write(os.path.join(requests_dir, "approved"),
                      f"{rec['id']}.json", _as_job_plan(rec))
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
