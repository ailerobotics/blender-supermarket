"""Settle analysis of the robot pose time-series — no simulator needed."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from twin import settle as S  # noqa: E402


def _series(min_zs, cxs=None, cys=None, z_exts=None):
    """Build a sample series; XY defaults to a fixed spot, z_ext to a stable 1.9 m."""
    n = len(min_zs)
    cxs = cxs if cxs is not None else [-1.525] * n
    cys = cys if cys is not None else [0.75] * n
    z_exts = z_exts if z_exts is not None else [1.9] * n
    return [{"step": i, "t": i / 60.0, "min_z": mz, "max_z": mz + ze,
             "cx": cx, "cy": cy, "z_ext": ze}
            for i, (mz, cx, cy, ze) in enumerate(zip(min_zs, cxs, cys, z_exts))]


def test_rests_on_floor_is_ok():
    v = S.analyze_settling(_series([0.02, 0.005, 0.0, 0.0, 0.0]), floor_z=0.0)
    assert v["ok"] is True
    assert v["rests_on_floor"] is True
    assert v["fell_through"] is False
    assert v["drifted"] is False
    assert v["exploded"] is False


def test_fall_through_flagged():
    # Robot sinks well below the floor plane.
    v = S.analyze_settling(_series([0.0, -0.3, -1.2, -2.0]), floor_z=0.0)
    assert v["fell_through"] is True
    assert v["ok"] is False
    assert any("below floor" in r for r in v["reasons"])


def test_hovering_above_floor_not_ok():
    # Pedestal stuck 0.5 m above the floor (bad import root / offset).
    v = S.analyze_settling(_series([0.5, 0.5, 0.5, 0.5]), floor_z=0.0)
    assert v["fell_through"] is False
    assert v["rests_on_floor"] is False
    assert v["ok"] is False


def test_drift_flagged():
    # Footprint centre wanders 0.4 m in X — a pinned pedestal should not.
    v = S.analyze_settling(_series([0.0, 0.0, 0.0, 0.0],
                                   cxs=[-1.525, -1.6, -1.8, -1.925]), floor_z=0.0)
    assert v["drifted"] is True
    assert v["drift_xy"] == pytest.approx(0.4, abs=1e-6)
    assert v["ok"] is False


def test_explosion_by_zext_growth():
    v = S.analyze_settling(_series([0.0, 0.0, 0.0, 0.0],
                                   z_exts=[1.9, 3.0, 6.0, 12.0]), floor_z=0.0)
    assert v["exploded"] is True
    assert v["ok"] is False


def test_nonfinite_is_explosion():
    v = S.analyze_settling(_series([0.0, 0.0, float("nan"), 0.0]), floor_z=0.0)
    assert v["any_nonfinite"] is True
    assert v["exploded"] is True
    assert v["ok"] is False


def test_empty_series():
    v = S.analyze_settling([], floor_z=0.0)
    assert v["ok"] is False
    assert v["n_samples"] == 0
    assert any("no samples" in r for r in v["reasons"])


def test_nonzero_floor_offset():
    # Floor at Z=0.4: resting there is ok, resting at 0 is a fall-through.
    ok = S.analyze_settling(_series([0.42, 0.41, 0.40, 0.40]), floor_z=0.4)
    assert ok["ok"] is True
    fell = S.analyze_settling(_series([0.4, 0.1, 0.0, -0.1]), floor_z=0.4)
    assert fell["fell_through"] is True


def test_settle_tail_ignores_initial_drop():
    # A small initial settle from 0.05 -> 0.0 is fine; the tail is on the floor.
    v = S.analyze_settling(_series([0.05, 0.03, 0.01, 0.0, 0.0, 0.0]), floor_z=0.0)
    assert v["rests_on_floor"] is True
    assert v["ok"] is True


def test_format_verdict_runs():
    v = S.analyze_settling(_series([0.0, 0.0, 0.0]), floor_z=0.0)
    text = S.format_verdict(v)
    assert "settle analysis" in text
    assert "VERDICT" in text
