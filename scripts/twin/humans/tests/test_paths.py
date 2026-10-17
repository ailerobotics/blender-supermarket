"""Offline tests for the pure scripted-path core. No simulator, ROS or GPU."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans.paths import (CYCLE, ONCE, PINGPONG, ScriptedPath, crosses_segment,
                               min_distance_between, min_distance_to_point)


def test_endpoints_and_midpoint():
    p = ScriptedPath(waypoints=((0, 0), (10, 0)), speed_mps=1.0, mode=ONCE)
    assert p.pose_at(0.0)[:2] == (0.0, 0.0)
    x, y, yaw = p.pose_at(5.0)
    assert math.isclose(x, 5.0) and math.isclose(y, 0.0)
    assert math.isclose(yaw, 0.0)
    # once mode holds at the end forever
    assert p.pose_at(100.0)[:2] == (10.0, 0.0)


def test_constant_speed_covers_distance():
    p = ScriptedPath(waypoints=((0, 0), (3, 4)), speed_mps=2.5, mode=ONCE)  # 5 m leg
    # half the travel time -> half the distance
    x, y, _ = p.pose_at(1.0)
    assert math.isclose(x, 1.5) and math.isclose(y, 2.0)


def test_start_offset_waits_at_first_waypoint():
    p = ScriptedPath(waypoints=((1, 2), (5, 2)), speed_mps=1.0, start_offset_s=3.0)
    assert p.pose_at(0.0)[:2] == (1.0, 2.0)
    assert p.pose_at(2.9)[:2] == (1.0, 2.0)
    x, _y, _ = p.pose_at(3.0 + 1.0)
    assert math.isclose(x, 2.0)


def test_pingpong_returns_to_start_each_period():
    p = ScriptedPath(waypoints=((0, 0), (4, 0)), speed_mps=1.0, mode=PINGPONG)
    # period = 2 * 4 m / 1 = 8 s; at t=0 and t=8 we are at the start.
    assert p.pose_at(0.0)[:2] == (0.0, 0.0)
    x8, _, _ = p.pose_at(8.0)
    assert math.isclose(x8, 0.0, abs_tol=1e-9)
    # at t=4 we are at the far end, heading reversed on the way back
    x4, _, _ = p.pose_at(4.0)
    assert math.isclose(x4, 4.0)


def test_dwell_holds_position_and_tiles_time():
    p = ScriptedPath(waypoints=((0, 0), (2, 0)), speed_mps=1.0, mode=PINGPONG, dwell_s=1.0)
    # leg travel 2 s, then dwell 1 s at (2,0). At t=2.5 still dwelling at the end.
    assert math.isclose(p.pose_at(2.5)[0], 2.0)
    # period = travel(2+2) + dwell(1*2 legs) = 6 s -> back to start at t=6
    assert math.isclose(p.pose_at(6.0)[0], 0.0, abs_tol=1e-9)


def test_cycle_closes_the_loop():
    p = ScriptedPath(waypoints=((0, 0), (2, 0), (2, 2)), speed_mps=1.0, mode=CYCLE)
    # legs: 2 + 2 + close(2*sqrt2); period returns to origin
    per = p._period()
    assert math.isclose(p.pose_at(per)[0], 0.0, abs_tol=1e-6)
    assert math.isclose(p.pose_at(per)[1], 0.0, abs_tol=1e-6)


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        ScriptedPath(waypoints=((0, 0),), speed_mps=1.0)          # need >= 2
    with pytest.raises(ValueError):
        ScriptedPath(waypoints=((0, 0), (1, 1)), speed_mps=0.0)   # speed > 0
    with pytest.raises(ValueError):
        ScriptedPath(waypoints=((1, 1), (1, 1)), speed_mps=1.0)   # zero length
    with pytest.raises(ValueError):
        ScriptedPath(waypoints=((0, 0), (1, 1)), speed_mps=1.0, mode='sprint')


def test_distance_helpers():
    p = ScriptedPath(waypoints=((0, -3), (0, 3)), speed_mps=1.0, mode=ONCE)
    # passes through the origin -> min distance to it is ~0
    assert min_distance_to_point(p, (0.0, 0.0), 6.0, dt=0.05) < 0.1
    # crosses the x-axis segment from (-1,0) to (1,0)
    assert crosses_segment(p, ((-1, 0), (1, 0)), 6.0, clearance_m=0.1, dt=0.05)
    q = ScriptedPath(waypoints=((5, -3), (5, 3)), speed_mps=1.0, mode=ONCE)
    assert min_distance_between(p, q, 6.0, dt=0.05) >= 4.9
