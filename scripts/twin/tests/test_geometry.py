"""Pose / clearance maths — no simulator needed."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from twin import geometry as G  # noqa: E402


def test_root_translate_compensates_pedestal_offset():
    # Pedestal offset (0,-1.2,0): to stand the pedestal at (-1.52, 0.75) on floor 0,
    # the robot root must be at (-1.52, 0.75 + 1.2, 0) = (-1.52, 1.95, 0).
    rt = G.root_translate_for_pedestal((-1.52, 0.75), floor_z=0.0)
    assert rt == pytest.approx((-1.52, 1.95, 0.0))


def test_root_translate_nonzero_floor():
    rt = G.root_translate_for_pedestal((1.0, 2.0), floor_z=0.4)
    assert rt == pytest.approx((1.0, 3.2, 0.4))


def test_aisle_centerlines_from_gondola_ranges():
    # Two 1.55 m runs at x-centres -9.14 and -6.09 -> gap 1.5, centre -7.62.
    ranges = [(-9.92, -8.37), (-6.87, -5.32)]
    aisles = G.aisle_centerlines(ranges)
    assert aisles == [(-7.62, 1.5)]


def test_aisle_centerlines_sorts_and_handles_rects():
    ranges = [(-6.87, -3.65, -5.32, 5.15), (-9.92, -3.65, -8.37, 5.15)]  # rects, unsorted
    aisles = G.aisle_centerlines(ranges)
    assert len(aisles) == 1
    assert aisles[0][0] == pytest.approx(-7.62)


def _grid_with_two_walls():
    # Domain x in [0,10], y in [0,10]; obstacles at x<2 and x>8 (a 6 m aisle, centre x=5).
    fps = [(0.0, 0.0, 2.0, 10.0), (8.0, 0.0, 10.0, 10.0)]
    return G.build_clearance_grid(fps, (0.0, 10.0, 0.0, 10.0), res=0.1, wall_margin=0.0)


def test_clearance_peaks_at_aisle_centre():
    grid = _grid_with_two_walls()
    c_centre = grid.clearance_at(5.0, 5.0)
    c_edge = grid.clearance_at(2.5, 5.0)
    assert c_centre > c_edge
    assert c_centre == pytest.approx(3.0, abs=0.15)  # 3 m to either wall


def test_resolve_spawn_keeps_requested_y_when_clear():
    grid = _grid_with_two_walls()
    cand = G.resolve_spawn(grid, floor_z=0.0, pedestal_xy=(5.0, 4.0),
                           y_range=(1.0, 9.0), min_clearance=1.0)
    assert (cand.x, cand.y) == pytest.approx((5.0, 4.0))
    assert cand.z == 0.0
    assert cand.root_translate == pytest.approx((5.0, 5.2, 0.0))  # +1.2 y offset


def test_resolve_spawn_raises_when_too_tight():
    grid = _grid_with_two_walls()
    # X=2.2 is 0.2 m from the wall -> below a 1.0 m bar everywhere on that line.
    with pytest.raises(ValueError):
        G.resolve_spawn(grid, 0.0, (2.05, 5.0), (1.0, 9.0), min_clearance=1.0)


def test_global_max_clearance_is_in_the_aisle():
    grid = _grid_with_two_walls()
    gm = G.global_max_clearance(grid)
    assert 4.0 <= gm.x <= 6.0
    assert gm.clearance_m == pytest.approx(3.0, abs=0.15)
