"""Map a store-vision approved restock plan into the mission job (CPU, opt-in).

The Isaac restocking mission normally runs the hardcoded job carried by its
scenario (`scripts/restock_scenario.json`: `job` = sku/quantity/source_bin/shelf_id).
store-vision already *plans* exactly this -- its `restock` module turns shelf/3D
signals into a prioritised, approved `RobotMission` (pick -> travel -> place). The
producer `scripts/integrations/export_restock_job.py` serialises one approved item
into the ``storevision.restock.job_plan/v1`` contract; this module maps that export
onto the mission's :class:`~twin.restock.inventory.Job` fields and validates the
SKU + destination slot against the planogram (`scripts/planogram.json`), refusing a
job whose slot or SKU is unknown there (honest failure, never a silent guess).

Opt-in and default-off: nothing here runs unless ``--job-from`` is passed to
`run_mobile_restock.py`. Absent, the mission job is byte-identical to today's.

The producer side lives next to store-vision and imports it read-only; this
consumer has no store-vision dependency -- it only reads the exported JSON.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .inventory import Job

SCHEMA = "storevision.restock.job_plan/v1"

# A gondola-side slot id in the store-vision namespace carries an underscore between
# the gondola id and its side letter (``G3_R_b0_s2``); the planogram/catalog_geom
# namespace omits it (``G3R_b0_s2``). Endcap/back/front/wall/chiller slots already
# match (``G3_back_s2``), so only this one case needs bridging.
_GONDOLA_SIDE = re.compile(r"^(G\d+)_([LR])(_b\d+_s\d+)$")


class PlanError(ValueError):
    """The plan export is malformed or does not map onto the planogram."""


def load_plan(path) -> dict:
    """Read and shape-check a job-plan export JSON."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise PlanError("plan export must be a JSON object")
    schema = data.get("schema")
    if schema != SCHEMA:
        raise PlanError(f"unsupported plan schema {schema!r} (expected {SCHEMA!r})")
    for key in ("sku", "quantity", "source", "destination"):
        if key not in data:
            raise PlanError(f"plan export missing required field {key!r}")
    if not isinstance(data["source"], dict) or "location" not in data["source"]:
        raise PlanError("plan export 'source' must be an object with a 'location'")
    if not isinstance(data["destination"], dict) or "slot_id" not in data["destination"]:
        raise PlanError("plan export 'destination' must be an object with a 'slot_id'")
    return data


def slot_candidates(slot_id: str) -> list[str]:
    """Planogram slot ids to try for a store-vision slot id (most specific first)."""
    out = [slot_id]
    m = _GONDOLA_SIDE.match(slot_id)
    if m:
        collapsed = f"{m.group(1)}{m.group(2)}{m.group(3)}"
        if collapsed not in out:
            out.append(collapsed)
    return out


def _planogram_slots(planogram_path) -> dict[str, set[str]]:
    """Map planogram slot_id -> set of SKUs it stocks, via catalog_geom (authoritative)."""
    try:
        from scenarios import catalog_geom as geom
    except ImportError as exc:  # pragma: no cover - requires scripts/ on sys.path
        raise PlanError(
            "planogram validation needs the 'scenarios.catalog_geom' module on the path"
        ) from exc
    plan = geom.load_planogram(str(planogram_path))
    return {s.slot_id: set(s.facing_count()) for s in geom.iter_slots(plan)}


def resolve_slot(destination_slot_id: str, slots: dict[str, set[str]]) -> str:
    """Resolve a plan slot id to a real planogram slot id, or refuse."""
    for cand in slot_candidates(destination_slot_id):
        if cand in slots:
            return cand
    raise PlanError(
        f"destination slot {destination_slot_id!r} is unknown in the planogram "
        f"(tried {slot_candidates(destination_slot_id)})"
    )


def resolve_job(plan: dict, planogram_path, *, job_id: str | None = None) -> tuple[Job, dict]:
    """Map a plan export to a mission :class:`Job`, validated against the planogram.

    Returns ``(job, meta)`` where ``meta`` carries the resolution provenance (the
    resolved slot, the source id used, and the plan's own provenance block). Raises
    :class:`PlanError` on an unknown SKU, an unknown slot, or a slot that does not
    stock the requested SKU.
    """
    slots = _planogram_slots(planogram_path)
    sku = plan["sku"]
    known_skus = set().union(*slots.values()) if slots else set()
    if sku not in known_skus:
        raise PlanError(f"SKU {sku!r} is unknown in the planogram")

    resolved_slot = resolve_slot(plan["destination"]["slot_id"], slots)
    if sku not in slots[resolved_slot]:
        raise PlanError(
            f"planogram slot {resolved_slot!r} stocks {sorted(slots[resolved_slot])}, "
            f"not the requested SKU {sku!r}"
        )

    provenance = plan.get("provenance", {}) if isinstance(plan.get("provenance"), dict) else {}
    resolved_job_id = (
        job_id
        or provenance.get("mission_id")
        or provenance.get("task_id")
        or f"plan-{sku}-{resolved_slot}"
    )
    source_bin = plan["source"]["location"]
    try:
        job = Job(
            job_id=str(resolved_job_id),
            sku=str(sku),
            quantity=plan["quantity"],
            source_bin=str(source_bin),
            shelf_id=resolved_slot,
        )
    except (TypeError, ValueError) as exc:
        raise PlanError(f"plan does not form a valid mission job: {exc}") from exc

    meta = {
        "resolved_slot_id": resolved_slot,
        "plan_slot_id": plan["destination"]["slot_id"],
        "source_bin": source_bin,
        "priority": plan.get("priority"),
        "provenance": provenance,
        "store_config": plan.get("store_config"),
    }
    return job, meta


def apply_to_scenario(scenario: dict, job: Job) -> dict:
    """Return a copy of ``scenario`` with the demand-driven job substituted.

    The dependent fields the mission validator couples to the job are kept coherent:
    ``initial_inventory`` gains/keeps a row for (source_bin, sku) covering the
    quantity, and ``handling_unit.sku`` (when present) follows the job SKU. Every
    other field is preserved unchanged, so a plan that matches the existing job
    reproduces the original scenario exactly.
    """
    from dataclasses import asdict

    out = json.loads(json.dumps(scenario))  # deep copy via JSON (scenario is JSON data)
    out["job"] = asdict(job)

    inv = out.get("initial_inventory")
    if not isinstance(inv, list):
        inv = []
    merged = []
    seen = False
    for row in inv:
        if isinstance(row, dict) and (row.get("location"), row.get("sku")) == (job.source_bin, job.sku):
            row = dict(row, quantity=max(int(row.get("quantity", 0)), job.quantity))
            seen = True
        merged.append(row)
    if not seen:
        merged.append({"location": job.source_bin, "sku": job.sku, "quantity": job.quantity})
    out["initial_inventory"] = merged

    if isinstance(out.get("handling_unit"), dict):
        out["handling_unit"] = dict(out["handling_unit"], sku=job.sku)
    return out
