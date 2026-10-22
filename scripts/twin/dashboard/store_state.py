"""Read-only overlay of store-vision store state (shelf fill / depletion / alerts).

store-vision is a separate, local-only project and is READ-ONLY for this lane, so
the dashboard does not import it. Instead it consumes a small, documented JSON
*export* at a configurable path (``--store-state``); see
``docs/fleet_dashboard.md`` for the contract and the proposed one-line store-vision
exporter. Absent/garbled export -> the store panel reports "not configured" rather
than failing the dashboard.

Contract (all keys optional, tolerated if extra keys appear)::

    {
      "generated_at": "2026-10-17T20:40:00Z",
      "source": "storevision.shelf+hq",
      "store_id": "supermarket-twin",
      "on_shelf_availability_pct": 94.2,
      "slots": [
        {"slot": "G3_back_s2", "sku": "SKU-1234", "fill_pct": 0.0, "state": "oos"},
        {"slot": "G3_back_s1", "sku": "SKU-1200", "fill_pct": 62.0, "state": "low"}
      ],
      "alerts": [
        {"type": "spill", "severity": "warn", "count": 1, "location": "aisle-3"},
        {"type": "obstruction", "severity": "info", "count": 0}
      ]
    }
"""
from __future__ import annotations

import json
from typing import Any

_SEVERITY_RANK = {"critical": 3, "crit": 3, "warn": 2, "warning": 2, "info": 1}
# A slot counts as needing attention when empty/low. Accepts both the dashboard's
# own labels and store-vision's native availability vocabulary
# (availability.types.AVAILABILITY_STATES: in_stock|limited|out_of_stock|unknown), so
# a near-native export works without a translation step.
_NEEDS_RESTOCK = {"oos", "out_of_stock", "empty", "low", "limited"}
_OUT_OF_STOCK = {"oos", "out_of_stock", "empty"}


def load_store_state(path: str | None) -> dict | None:
    """Load + summarise a store-state export. ``None`` path or bad file -> ``None``."""
    if not path:
        return None
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return summarise(data)


def summarise(data: dict) -> dict:
    """Normalise a raw export into the panel view with a computed summary block."""
    slots = data.get("slots") if isinstance(data.get("slots"), list) else []
    slots = [s for s in slots if isinstance(s, dict)]
    alerts = data.get("alerts") if isinstance(data.get("alerts"), list) else []
    alerts = [a for a in alerts if isinstance(a, dict)]

    needs = [s for s in slots if str(s.get("state", "")).lower() in _NEEDS_RESTOCK]
    oos = [s for s in slots if str(s.get("state", "")).lower() in _OUT_OF_STOCK]

    active_alerts = [a for a in alerts if int(a.get("count", 0) or 0) > 0]
    top_severity = None
    if active_alerts:
        top_severity = max(
            active_alerts,
            key=lambda a: _SEVERITY_RANK.get(str(a.get("severity", "info")).lower(), 0),
        ).get("severity")

    return {
        "generated_at": data.get("generated_at"),
        "source": data.get("source"),
        "store_id": data.get("store_id"),
        "on_shelf_availability_pct": data.get("on_shelf_availability_pct"),
        "slots": slots,
        "alerts": alerts,
        "summary": {
            "slot_count": len(slots),
            "needs_restock": len(needs),
            "out_of_stock": len(oos),
            "active_alerts": len(active_alerts),
            "top_severity": top_severity,
        },
    }
