"""Pure pose / clearance maths for placing the robot in the store.

No ``pxr``, no ``isaacsim`` — just numpy/scipy — so it is unit tested without a
simulator. ``usd_probe`` gathers the real footprints from the USD and feeds them
here; the same functions back the tests with synthetic footprints.

Conventions (verified against the store USD, see docs/twin_usd_facts.json):
  * World is Z-up, 1 unit = 1 metre.
  * The store floor is a horizontal plane at ``floor_z`` (measured 0.0).
  * Footprints are axis-aligned XY rectangles ``(min_x, min_y, max_x, max_y)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

# The shelfbot_newgripper URDF mounts the pedestal at the world link + this offset
# (joint world->pedestal_link origin xyz [0, -1.2, 0.35]); the pedestal collision
# box (0.2 x 0.4 x 0.7) is centred on the link, so its bottom rests on Z=0. Only
# the XY offset matters for where the base stands on the floor.
PEDESTAL_OFFSET_FROM_ROOT: Tuple[float, float, float] = (0.0, -1.2, 0.0)

Rect = Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)


def root_translate_for_pedestal(
    pedestal_xyz: Sequence[float],
    floor_z: float,
    pedestal_offset: Sequence[float] = PEDESTAL_OFFSET_FROM_ROOT,
) -> Tuple[float, float, float]:
    """World translate to give the imported robot's ROOT prim so that its pedestal
    base lands at ``pedestal_xyz`` (x, y) with the box bottom on ``floor_z``.

    pedestal_world = root_translate + pedestal_offset  =>
    root_translate = pedestal_xyz - pedestal_offset.

    NB: this assumes the import root prim coincides with the URDF ``world`` link.
    The importer's root convention varies (fix_base / base_link), so the compose
    script re-measures the pedestal prim after import and corrects — this is the
    pre-import estimate and the value the dry-run reports.
    """
    px, py = float(pedestal_xyz[0]), float(pedestal_xyz[1])
    ox, oy = float(pedestal_offset[0]), float(pedestal_offset[1])
    return (px - ox, py - oy, float(floor_z))


def aisle_centerlines(gondola_x_ranges: Sequence[Rect] | Sequence[Tuple[float, float]]
                      ) -> list[Tuple[float, float]]:
    """Given the X extents of the parallel gondola runs, return the (centre_x,
    gap_width) of each aisle between adjacent runs, ordered left→right.

    Accepts either full rects or bare (x_min, x_max) pairs.
    """
    spans: list[Tuple[float, float]] = []
    for r in gondola_x_ranges:
        if len(r) == 2:
            spans.append((float(r[0]), float(r[1])))
        else:
            spans.append((float(r[0]), float(r[2])))
    spans.sort(key=lambda s: s[0])
    out = []
    for a, b in zip(spans, spans[1:]):
        gap = b[0] - a[1]  # from right edge of left run to left edge of right run
        centre = (a[1] + b[0]) / 2.0
        out.append((round(centre, 4), round(gap, 4)))
    return out


@dataclass
class ClearanceGrid:
    """A 2D occupancy grid over the floor and its Euclidean clearance field (m)."""
    extent: Tuple[float, float, float, float]  # x0, x1, y0, y1
    res: float
    occ: np.ndarray  # bool (nx, ny), True = obstacle / out of bounds
    clearance: np.ndarray  # float (nx, ny), metres to nearest obstacle

    def _idx(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        x0, x1, y0, y1 = self.extent
        ix = int(round((x - x0) / self.res))
        iy = int(round((y - y0) / self.res))
        if 0 <= ix < self.occ.shape[0] and 0 <= iy < self.occ.shape[1]:
            return ix, iy
        return None

    def clearance_at(self, x: float, y: float) -> float:
        ij = self._idx(x, y)
        return float(self.clearance[ij]) if ij else 0.0

    def index_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        x0, _, y0, _ = self.extent
        return (round(x0 + ix * self.res, 4), round(y0 + iy * self.res, 4))


def build_clearance_grid(
    footprints: Iterable[Rect],
    extent: Tuple[float, float, float, float],
    res: float = 0.05,
    wall_margin: float = 0.30,
) -> ClearanceGrid:
    """Rasterise obstacle ``footprints`` into an occupancy grid over ``extent``
    (x0, x1, y0, y1), erode the domain edge by ``wall_margin`` (the perimeter
    walls bound the domain rather than being stamped — a wall ring's bbox would
    otherwise fill the whole floor), and compute the clearance field.

    Only import scipy here so the module imports cheaply for callers that only
    need the pure pose helpers above.
    """
    from scipy import ndimage

    x0, x1, y0, y1 = extent
    nx = max(1, int(np.ceil((x1 - x0) / res)))
    ny = max(1, int(np.ceil((y1 - y0) / res)))
    occ = np.zeros((nx, ny), dtype=bool)

    for (mnx, mny, mxx, mxy) in footprints:
        ix0 = max(0, int(np.floor((mnx - x0) / res)))
        ix1 = min(nx, int(np.ceil((mxx - x0) / res)))
        iy0 = max(0, int(np.floor((mny - y0) / res)))
        iy1 = min(ny, int(np.ceil((mxy - y0) / res)))
        if ix1 > ix0 and iy1 > iy0:
            occ[ix0:ix1, iy0:iy1] = True

    m = int(round(wall_margin / res))
    if m > 0:
        occ[:m, :] = True
        occ[-m:, :] = True
        occ[:, :m] = True
        occ[:, -m:] = True

    clearance = ndimage.distance_transform_edt(~occ) * res
    return ClearanceGrid(extent=extent, res=res, occ=occ, clearance=clearance)


@dataclass
class SpawnCandidate:
    x: float
    y: float
    z: float
    clearance_m: float
    label: str = ""
    root_translate: Optional[Tuple[float, float, float]] = None


def best_along_line(grid: ClearanceGrid, x: float, y_lo: float, y_hi: float,
                    step: float = 0.1) -> SpawnCandidate:
    """Max-clearance point on the vertical line X=x within [y_lo, y_hi]."""
    best = None
    y = y_lo
    while y <= y_hi + 1e-9:
        c = grid.clearance_at(x, y)
        if best is None or c > best.clearance_m:
            best = SpawnCandidate(x=round(x, 4), y=round(y, 4), z=0.0, clearance_m=round(c, 4))
        y += step
    return best


def global_max_clearance(grid: ClearanceGrid) -> SpawnCandidate:
    ij = np.unravel_index(int(np.argmax(grid.clearance)), grid.clearance.shape)
    x, y = grid.index_to_world(int(ij[0]), int(ij[1]))
    return SpawnCandidate(x=x, y=y, z=0.0,
                          clearance_m=round(float(grid.clearance[ij]), 4),
                          label="global_max_clearance")


def resolve_spawn(
    grid: ClearanceGrid,
    floor_z: float,
    pedestal_xy: Tuple[float, float],
    y_range: Tuple[float, float],
    min_clearance: float,
    label: str = "aisle",
) -> SpawnCandidate:
    """Pick the pedestal spot: the best-clearance point on the aisle centreline
    X=pedestal_xy[0] within ``y_range``, and attach the root translate estimate.
    Raises ValueError if the best clearance is below ``min_clearance``.
    """
    cand = best_along_line(grid, pedestal_xy[0], y_range[0], y_range[1])
    # Prefer the requested Y if it already clears the bar (keeps a human-chosen,
    # mid-aisle pose instead of drifting to the aisle end).
    at_req = grid.clearance_at(pedestal_xy[0], pedestal_xy[1])
    if at_req >= min_clearance:
        cand = SpawnCandidate(x=round(pedestal_xy[0], 4), y=round(pedestal_xy[1], 4),
                              z=0.0, clearance_m=round(at_req, 4))
    if cand.clearance_m < min_clearance:
        raise ValueError(
            f"no spot on X={pedestal_xy[0]} in Y{y_range} clears {min_clearance} m "
            f"(best {cand.clearance_m} m) — pick another aisle")
    cand.z = float(floor_z)
    cand.label = label
    cand.root_translate = root_translate_for_pedestal((cand.x, cand.y), floor_z)
    return cand
