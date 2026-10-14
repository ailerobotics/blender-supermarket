"""Deterministic store geometry derived from the planogram and product catalog.

This is the single source of truth for *where every product facing sits in world
space*. It is pure Python (no ``bpy``) so it can be unit-tested offline and shared
verbatim between the Blender scene builder (``build_scene.py`` instantiates one box
per facing at these transforms) and the annotator (``annotate.py`` projects these
boxes into each camera). Keeping both sides on the same facing list is what makes
the rendered pixels and the ground-truth boxes agree.

The placement math mirrors ``scripts/build_supermarket.py`` (``place_products``,
``shelf_frame``, the gondola/wall/chiller/endcap loops) but is **deterministic**:
no RNG jitter, no random empties. The scenario layer (``state.py``) is what removes
or moves facings over time, so ground truth stays exact.

Coordinates: metres, Blender world axes, Z up. The store floor is at z=0, centred
on the origin; +X is to the right looking from the entrance, matching the builder.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

# catalog.py lives one level up in scripts/; callers put scripts/ on sys.path.
try:
    from catalog import BY_ID, CATEGORIES  # type: ignore
except ImportError:  # pragma: no cover - exercised via scripts/ on path
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from catalog import BY_ID, CATEGORIES  # type: ignore

# Unit axis vectors keyed by a fixture's facing direction (shopper faces this way).
FACING: Dict[str, Tuple[float, float]] = {
    "-X": (-1.0, 0.0), "+X": (1.0, 0.0), "-Y": (0.0, -1.0), "+Y": (0.0, 1.0)}
ROT_Z: Dict[str, float] = {
    "-X": -math.pi / 2, "+X": math.pi / 2, "-Y": 0.0, "+Y": math.pi}

# Placement spacing constants, copied from build_supermarket.place_products.
GAP = 0.012          # lateral gap between facings
FRONT_GAP = 0.02     # setback of the front row from the shelf lip
ROW_STEP_PAD = 0.006  # extra depth per row


def shelf_frame(facing: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """(right, front) unit vectors: shopper's right, and back-edge -> shopper."""
    fx, fy = FACING[facing]
    return (-fy, fx), (fx, fy)


@dataclass
class Facing:
    """One product unit at the front of a shelf column (the visible facing)."""
    slot_id: str
    facing_index: int          # column index within the slot, left to right
    sku: str
    center: Tuple[float, float, float]   # world centre of the box
    size: Tuple[float, float, float]     # world-axis-aligned full extents (x, y, z)
    rot_z: float               # yaw applied to the product mesh (radians)

    @property
    def aabb(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        cx, cy, cz = self.center
        sx, sy, sz = self.size
        return ((cx - sx / 2, cy - sy / 2, cz - sz / 2),
                (cx + sx / 2, cy + sy / 2, cz + sz / 2))

    def corners(self) -> List[Tuple[float, float, float]]:
        (x0, y0, z0), (x1, y1, z1) = self.aabb
        return [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]


@dataclass
class Slot:
    """One shelf level of one bay: an ordered row of facings sharing a plan SKU set."""
    slot_id: str
    fixture_id: str
    fixture_type: str          # gondola | wall | chiller | endcap
    cat: str
    facing_dir: str            # key into FACING
    shelf_z: float
    depth: float
    width: float
    origin_xy: Tuple[float, float]
    bay_index: int
    level: int
    facings: List[Facing] = field(default_factory=list)

    def facing_count(self) -> Dict[str, int]:
        """Per-SKU facing count — the depletion/misplaced ground-truth primitive."""
        out: Dict[str, int] = {}
        for f in self.facings:
            out[f.sku] = out.get(f.sku, 0) + 1
        return out


def _place_shelf(slot_id: str, origin_xy, z, facing, width, depth, items) -> List[Facing]:
    (rx, ry), (fxv, fyv) = shelf_frame(facing)
    ox, oy = origin_xy
    total = sum(n * (BY_ID[sid]["w"] + GAP) for sid, n in items)
    x = -min(total, width) / 2.0
    out: List[Facing] = []
    fi_global = 0
    for sid, n in items:
        s = BY_ID[sid]
        w, d, h = s["w"], s["d"], s["h"]
        front_off = depth - FRONT_GAP - 0.5 * (d + ROW_STEP_PAD)
        for _ in range(n):
            if x + w > width / 2 + 1e-6:
                break
            cx = ox + rx * (x + w / 2) + fxv * front_off
            cy = oy + ry * (x + w / 2) + fyv * front_off
            cz = z + h / 2
            # right/front are axis-aligned unit vectors for every fixture facing,
            # so the world AABB extents follow directly from (w, d, h).
            sx = abs(rx) * w + abs(fxv) * d
            sy = abs(ry) * w + abs(fyv) * d
            out.append(Facing(slot_id=slot_id, facing_index=fi_global, sku=sid,
                              center=(cx, cy, cz), size=(sx, sy, h),
                              rot_z=ROT_Z[facing]))
            fi_global += 1
            x += w + GAP
    return out


def iter_slots(plan: dict, fixtures: Optional[List[str]] = None) -> Iterator[Slot]:
    """Yield every product Slot in the planogram (optionally limited to fixture ids)."""
    bay_w = plan["bay_w"]
    fix = plan["fixture"]

    def want(fid: str) -> bool:
        return fixtures is None or fid in fixtures

    # ---- gondolas: double-sided runs, sides L=-X / R=+X ----
    for g in plan["gondolas"]:
        if not want(g["id"]):
            continue
        gx, y0 = g["x"], g["y0"]
        for side, facing in (("L", "-X"), ("R", "+X")):
            sx = gx + FACING[facing][0] * fix["back_t"] / 2
            for bay in g["sides"][side]["bays"]:
                by = y0 + (bay["index"] + 0.5) * bay_w
                width = bay_w - fix["upright_w"] - 0.01
                for li, sh in enumerate(bay["shelves"]):
                    slot_id = f"{g['id']}{side}_b{bay['index']}_s{li}"
                    s = Slot(slot_id, g["id"], "gondola", bay.get("cat", ""), facing,
                             sh["z"], sh["depth"], width, (sx, by), bay["index"], li)
                    s.facings = _place_shelf(slot_id, (sx, by), sh["z"], facing,
                                             width, sh["depth"], sh["items"])
                    yield s

    # ---- wall units: single-sided ----
    for wu in plan["wall_units"]:
        if not want(wu["id"]):
            continue
        facing = wu["facing"]
        fxv, fyv = FACING[facing]
        along = (-fyv, fxv)
        start = (wu["x"], wu["y0"])
        for bay in wu["bays"]:
            bcx = start[0] + along[0] * ((bay["index"] + 0.5) * bay_w) + fxv * fix["back_t"] / 2
            bcy = start[1] + along[1] * ((bay["index"] + 0.5) * bay_w) + fyv * fix["back_t"] / 2
            width = bay_w - fix["upright_w"] - 0.01
            for li, sh in enumerate(bay["shelves"]):
                slot_id = f"{wu['id']}_b{bay['index']}_s{li}"
                s = Slot(slot_id, wu["id"], "wall", bay.get("cat", ""), facing,
                         sh["z"], sh["depth"], width, (bcx, bcy), bay["index"], li)
                s.facings = _place_shelf(slot_id, (bcx, bcy), sh["z"], facing,
                                         width, sh["depth"], sh["items"])
                yield s

    # ---- chillers: one cabinet per bay; products set back 0.06 from bay centre ----
    for ch in plan["chillers"]:
        if not want(ch["id"]):
            continue
        facing = ch["facing"]
        fxv, fyv = FACING[facing]
        along = (-fyv, fxv)
        start = (ch["x"], ch["y0"])
        for bay in ch["bays"]:
            bcx = start[0] + along[0] * ((bay["index"] + 0.5) * bay_w)
            bcy = start[1] + along[1] * ((bay["index"] + 0.5) * bay_w)
            ibx, iby = bcx + fxv * 0.06, bcy + fyv * 0.06
            width = bay_w - 0.14
            for li, sh in enumerate(bay["shelves"]):
                slot_id = f"{ch['id']}_b{bay['index']}_s{li}"
                s = Slot(slot_id, ch["id"], "chiller", bay.get("cat", ""), facing,
                         sh["z"], sh["depth"], width, (ibx, iby), bay["index"], li)
                s.facings = _place_shelf(slot_id, (ibx, iby), sh["z"], facing,
                                         width, sh["depth"], sh["items"])
                yield s

    # ---- endcaps: single bay capping a gondola run ----
    for ec in plan["endcaps"]:
        if not want(ec["id"]):
            continue
        facing = ec["facing"]
        fxv, fyv = FACING[facing]
        g = next(gg for gg in plan["gondolas"] if ec["id"].startswith(gg["id"] + "_"))
        run_len = g["n_bays"] * bay_w
        y_back = (g["y0"] - 0.03 - fix["back_t"] / 2) if facing == "-Y" \
            else (g["y0"] + run_len + 0.03 + fix["back_t"] / 2)
        sy = y_back + FACING[facing][1] * fix["back_t"] / 2
        width = ec["w"] - fix["upright_w"] - 0.01
        for li, sh in enumerate(ec["shelves"]):
            slot_id = f"{ec['id']}_s{li}"
            s = Slot(slot_id, ec["id"], "endcap", ec.get("sku", ""), facing,
                     sh["z"], sh["depth"], width, (ec["x"], sy), 0, li)
            s.facings = _place_shelf(slot_id, (ec["x"], sy), sh["z"], facing,
                                     width, sh["depth"], sh["items"])
            yield s


def fixture_footprint(plan: dict, fixture_id: str) -> Optional[dict]:
    """A coarse box (centre + size, z from floor) approximating a fixture's volume.

    Used by the scene builder to stand non-target fixtures as cheap occluders and
    to draw the floor map, without instancing their products.
    """
    bay_w = plan["bay_w"]
    fix = plan["fixture"]
    for g in plan["gondolas"]:
        if g["id"] == fixture_id:
            run = g["n_bays"] * bay_w
            depth = 2 * fix["base_depth"] + fix["back_t"]
            return dict(center=(g["x"], g["y0"] + run / 2, fix["height"] / 2),
                        size=(depth, run, fix["height"]), kind="gondola")
    for wu in plan["wall_units"]:
        if wu["id"] == fixture_id:
            run = wu["n_bays"] * bay_w
            fxv, fyv = FACING[wu["facing"]]
            along = (-fyv, fxv)
            cx = wu["x"] + along[0] * run / 2 - fxv * fix["base_depth"] / 2
            cy = wu["y0"] + along[1] * run / 2 - fyv * fix["base_depth"] / 2
            size = (abs(along[0]) * run + abs(fxv) * fix["base_depth"],
                    abs(along[1]) * run + abs(fyv) * fix["base_depth"], 2.2)
            return dict(center=(cx, cy, 2.2 / 2), size=size, kind="wall")
    for ch in plan["chillers"]:
        if ch["id"] == fixture_id:
            run = ch["n_bays"] * bay_w
            depth = 0.8 if ch["id"] == "C_right" else 0.9
            height = 2.1 if ch["id"] == "C_right" else 2.0
            fxv, fyv = FACING[ch["facing"]]
            along = (-fyv, fxv)
            cx = ch["x"] + along[0] * run / 2 + fxv * depth / 2
            cy = ch["y0"] + along[1] * run / 2 + fyv * depth / 2
            size = (abs(along[0]) * run + abs(fxv) * depth,
                    abs(along[1]) * run + abs(fyv) * depth, height)
            return dict(center=(cx, cy, height / 2), size=size, kind="chiller")
    return None


def load_planogram(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def all_fixture_ids(plan: dict) -> List[str]:
    ids = [g["id"] for g in plan["gondolas"]]
    ids += [w["id"] for w in plan["wall_units"]]
    ids += [c["id"] for c in plan["chillers"]]
    return ids
