#!/usr/bin/env python3
"""Export a store-vision tray/handling-unit catalogue as a mission fit-preflight input.

Producer side of the OPT-IN smart variant of the tray/slot fit preflight (lane
sim-tray-fit-preflight-v2). The mission's working default keeps the scenario's own
handling unit in scope (scripts/restock_scenario.json: the 0.45x0.30x0.18 m carton).
This producer widens that set to every tray SKU store-vision knows: it reuses
store-vision's AGPL-free ``storevision.restock.providers.ProductInfo`` READ-ONLY --
the per-SKU physical record (``dims_mm`` = (L, W, H) millimetres, ``weight_kg``,
``tray_packed``) that store-vision's own restock feasibility check uses -- and
serialises it into the ``storevision.restock.product_catalogue/v1`` contract
consumed by ``scripts/twin/restock/fit_preflight.py`` (:func:`load_catalogue`).

It never actuates and touches no store-vision state on disk. Run it with the
store-vision virtualenv's interpreter, pointing --store-vision-src at that checkout.
Only store-vision's AGPL-free modules are imported (``restock.providers``); the
AGPL YOLO detector paths (``products.detect.tray_plugin``, ``trays.backends``) are
never touched.

The consumer imports no store-vision code -- it only reads the exported JSON, so a
mission box without store-vision still runs the preflight against the catalogue.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "storevision.restock.product_catalogue/v1"


def build_catalogue_doc(products, *, axis_order=(0, 1, 2), note: str | None = None) -> dict:
    """Serialise an iterable of store-vision ProductInfo-shaped records into the export.

    ``products`` is any iterable of objects exposing ``sku``, ``dims_mm`` (an (L, W, H)
    millimetre triple), ``weight_kg`` and ``tray_packed`` -- store-vision's
    ``ProductInfo`` satisfies this structurally, and so does any duck-typed stand-in
    (which is all this pure builder needs, so it is testable without store-vision).

    ``axis_order`` records the catalogue (L, W, H) -> mission grasp-frame (x, y, z)
    mapping the consumer must apply; it is emitted verbatim (default identity, since
    store-vision's dims and the mission's own unit share the planogram orientation).
    """
    order = tuple(int(i) for i in axis_order)
    if sorted(order) != [0, 1, 2]:
        raise ValueError(f"axis_order must be a permutation of (0,1,2): {axis_order}")
    units = []
    seen = set()
    for p in products:
        sku = str(p.sku)
        if sku in seen:
            raise ValueError(f"duplicate sku in catalogue: {sku!r}")
        seen.add(sku)
        dims_mm = [float(x) for x in p.dims_mm]
        if len(dims_mm) != 3 or not all(d > 0 for d in dims_mm):
            raise ValueError(f"product {sku!r} dims_mm must be 3 positive mm: {dims_mm}")
        units.append({
            "sku": sku,
            "dims_mm": dims_mm,
            "weight_kg": float(getattr(p, "weight_kg", 0.0)),
            "tray_packed": bool(getattr(p, "tray_packed", True)),
            "source": str(getattr(p, "source", "storevision.restock.ProductInfo")),
        })
    doc = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "axis_order": list(order),
        "producer": "blender-supermarket/scripts/integrations/export_tray_catalogue.py",
        "catalogue_type": "storevision.restock.ProductInfo",
        "units": units,
    }
    if note:
        doc["note"] = str(note)
    return doc


def _parse_product(spec: str):
    """Parse a ``sku:LxWxH[:kg]`` CLI product spec into a ProductInfo kwargs dict.

    Dimensions are millimetres in store-vision's (L, W, H) order.
    """
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"product must be 'sku:LxWxH[:kg]' (mm), got {spec!r}")
    sku = parts[0]
    try:
        dims_mm = tuple(float(v) for v in parts[1].lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad dims in {spec!r}: {exc}") from exc
    if len(dims_mm) != 3:
        raise argparse.ArgumentTypeError(f"dims must be LxWxH (mm), got {parts[1]!r}")
    weight_kg = float(parts[2]) if len(parts) == 3 else 0.5
    return {"sku": sku, "dims_mm": dims_mm, "weight_kg": weight_kg}


def build_products(args):
    """Build store-vision ``ProductInfo`` records from the CLI specs (read-only import).

    store-vision ships no concrete populated catalogue (only the ``ProductInfo`` record
    type + a ``MockProductCatalog``), so the per-SKU geometry is supplied by the
    operator here and validated through store-vision's own record type. The result is
    exactly the shape store-vision's ``restock`` feasibility consumes.
    """
    sys.path.insert(0, str(Path(args.store_vision_src).resolve()))
    from storevision.restock import ProductInfo  # AGPL-free (restock.providers)

    products = []
    for spec in args.product:
        kw = _parse_product(spec)
        products.append(ProductInfo(sku=kw["sku"], dims_mm=kw["dims_mm"],
                                    weight_kg=kw["weight_kg"]))
    return products


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-vision-src", required=True, type=Path,
                    help="path to the store-vision checkout (its src/ on sys.path)")
    ap.add_argument("--product", action="append", required=True, metavar="sku:LxWxH[:kg]",
                    help="a tray SKU + geometry in mm (store-vision ProductInfo order L,W,H); repeatable")
    ap.add_argument("--axis-order", type=int, nargs=3, default=(0, 1, 2),
                    help="catalogue (L,W,H) -> mission grasp (x,y,z) permutation (default identity)")
    ap.add_argument("--note", help="free-text provenance note recorded in the export")
    ap.add_argument("--output", type=Path, required=True, help="write the catalogue JSON here")
    args = ap.parse_args(argv)

    products = build_products(args)
    doc = build_catalogue_doc(products, axis_order=tuple(args.axis_order), note=args.note)
    args.output.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {len(doc['units'])} tray SKU(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
