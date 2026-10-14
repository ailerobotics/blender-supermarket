"""Helpers shared by the scenario builders: entity construction, fixture geometry,
and a stable pass-index allocator. Pure Python (no ``bpy``)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import catalog_geom as geom
from state import Entity

Vec3 = Tuple[float, float, float]


class PassIndex:
    """Hands out unique positive object indices (0 is reserved for background)."""
    def __init__(self, start: int = 1):
        self._n = start

    def next(self) -> int:
        v = self._n
        self._n += 1
        return v


def product_entities(plan: dict, fixture_ids: List[str], pidx: PassIndex,
                     annotate: bool = True) -> List[Entity]:
    """One :class:`Entity` per product facing on the given fixtures."""
    out: List[Entity] = []
    for slot in geom.iter_slots(plan, fixtures=fixture_ids):
        for f in slot.facings:
            out.append(Entity(
                id=f"{f.slot_id}_f{f.facing_index}",
                kind="product", cls=f.sku, pass_index=pidx.next(),
                center=f.center, size=f.size, rot_z=f.rot_z, annotate=annotate,
                meta=dict(slot_id=f.slot_id, facing_index=f.facing_index,
                          fixture_id=slot.fixture_id, bay=slot.bay_index,
                          level=slot.level, cat=slot.cat)))
    return out


def footprint_entities(plan: dict, fixture_ids: List[str], pidx: PassIndex) -> List[Entity]:
    """Coarse occluder boxes for non-target fixtures (not annotated)."""
    out: List[Entity] = []
    for fid in fixture_ids:
        fp = geom.fixture_footprint(plan, fid)
        if fp is None:
            continue
        out.append(Entity(id=f"footprint_{fid}", kind="fixture", cls="fixture",
                          pass_index=pidx.next(), center=fp["center"], size=fp["size"],
                          annotate=False, meta=dict(fixture_id=fid, kind=fp["kind"])))
    return out


def gondola_side_front(plan: dict, fixture_id: str, side: str) -> Tuple[float, str]:
    """(front-face X, facing-dir) for a gondola side; shelves face into the aisle."""
    g = next(gg for gg in plan["gondolas"] if gg["id"] == fixture_id)
    bd = plan["fixture"]["base_depth"]
    if side == "R":           # +X side, aisle is to the right
        return g["x"] + bd, "+X"
    return g["x"] - bd, "-X"


def bay_centers_y(plan: dict, fixture_id: str) -> List[float]:
    g = next(gg for gg in plan["gondolas"] if gg["id"] == fixture_id)
    bw = plan["bay_w"]
    return [g["y0"] + (i + 0.5) * bw for i in range(g["n_bays"])]


def slot_depths_for_bay(plan: dict, fixture_id: str, side: str, bay_index: int) -> Dict[int, dict]:
    """Map shelf level -> {z, depth, front_x} for a gondola bay (RGB-D fill GT)."""
    front_x, facing = gondola_side_front(plan, fixture_id, side)
    out: Dict[int, dict] = {}
    for slot in geom.iter_slots(plan, fixtures=[fixture_id]):
        if slot.fixture_id != fixture_id or slot.facing_dir != facing:
            continue
        if slot.bay_index != bay_index:
            continue
        out[slot.level] = dict(z=slot.shelf_z, depth=slot.depth, front_x=front_x)
    return out


def world_size(sku: str, rot_z: float) -> Vec3:
    """World-axis-aligned extents of a product facing at yaw ``rot_z``.

    Matches :mod:`catalog_geom` placement: ±X facings (|sin rot|≈1) span depth in X
    and width in Y; ±Y facings span width in X and depth in Y.
    """
    s = geom.BY_ID[sku]
    w, d, h = s["w"], s["d"], s["h"]
    import math
    if abs(math.sin(rot_z)) > 0.5:      # facing ±X
        return (d, w, h)
    return (w, d, h)                     # facing ±Y


def floor_marking(id: str, cls: str, center: Vec3, size: Vec3, pidx: PassIndex,
                  **meta) -> Entity:
    return Entity(id=id, kind="distractor", cls=cls, pass_index=pidx.next(),
                  center=center, size=size, annotate=True, meta=meta)
