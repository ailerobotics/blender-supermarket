"""CPU tests for the carried-carton level constraint (no Isaac/ROS/GPU).

Run with the repo convention that dodges the box ROS pytest plugins:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest scripts/twin/tests/test_carton_level.py
"""
import math
import os

import numpy as np
import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin16/mobile-import.urdf'
_has_urdf = os.path.exists(URDF)
need_urdf = pytest.mark.skipif(not _has_urdf, reason='mission URDF not present on this box')


# ----------------------------------------------------------------- decompose

def _Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_level_carton_reads_zero():
    assert cl.up_axis_rpy([0, 0, 1]) == pytest.approx((0, 0, 0), abs=1e-9)


def test_pure_pitch_and_roll_decompose():
    roll, pitch, tilt = cl.up_axis_rpy(_Ry(math.radians(5)) @ np.array([0, 0, 1.0]))
    assert (roll, pitch, tilt) == pytest.approx((0, 5, 5), abs=1e-6)
    roll, pitch, tilt = cl.up_axis_rpy(_Rx(math.radians(7)) @ np.array([0, 0, 1.0]))
    assert (roll, pitch, tilt) == pytest.approx((7, 0, 7), abs=1e-6)


def test_quat_rpy_matches_body_roll_pitch_formula():
    # the isaac_port.body_roll_pitch_deg convention (the live carton body read)
    def bodyrp(w, x, y, z):
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
        return math.degrees(roll), math.degrees(pitch)
    for ax in ('x', 'y'):
        for deg in (-9, -2, 3, 11):
            a = math.radians(deg)
            q = (math.cos(a / 2),
                 math.sin(a / 2) if ax == 'x' else 0.0,
                 math.sin(a / 2) if ax == 'y' else 0.0, 0.0)
            roll, pitch, _ = cl.quat_rpy(q)
            br, bp = bodyrp(*q)
            assert roll == pytest.approx(br, abs=1e-6)
            assert pitch == pytest.approx(bp, abs=1e-6)


def test_forward_down_tilt_side_vs_topdown():
    assert cl.forward_down_tilt([0, 0, -1]) == pytest.approx(0, abs=1e-6)   # top-down grasp
    assert cl.forward_down_tilt([0, 1, 0]) == pytest.approx(90, abs=1e-6)    # level side grasp


# -------------------------------------------------------------------- trip

def test_tilt_trip_within_bound_is_none():
    assert cl.tilt_trip(1.2, -1.9, bound_deg=2.0) is None


def test_tilt_trip_over_bound_names_carton_tilt():
    out = cl.tilt_trip(0.5, 6.3, bound_deg=2.0)
    assert out is not None
    msg, worst = out
    assert msg.startswith(cl.CARTON_TILT)
    assert worst == pytest.approx(6.3)
    assert 'pitch=+6.30' in msg and 'bound 2.00' in msg


# ------------------------------------------------------------------- config

def test_resolve_level_config_off_variants():
    for off in (None, False, '', '0', 'off', 'false', 'none'):
        assert cl.resolve_level_config(off) is None


def test_resolve_level_config_on_defaults():
    c = cl.resolve_level_config(True)
    assert c.bound_deg == cl.DEFAULT_TILT_BOUND_DEG and c.trip and c.level_goals


def test_resolve_level_config_mapping_and_validation():
    c = cl.resolve_level_config({'bound_deg': 1.5, 'trip': False, 'joints': ['ur_wrist_1_joint']})
    assert c.bound_deg == 1.5 and c.trip is False and c.joints == ('ur_wrist_1_joint',)
    with pytest.raises(ValueError):
        cl.resolve_level_config({'bound_deg': -1})
    with pytest.raises(ValueError):
        cl.resolve_level_config({'joints': ['not_a_joint']})
    with pytest.raises(ValueError):
        cl.resolve_level_config('weird')


# --------------------------------------------------------------- weld level

def test_weld_attach_level_check_passes_level_carton():
    roll, pitch, tilt, trip = cl.weld_attach_level_check((1.0, 0.0, 0.0, 0.0))  # identity = level
    assert tilt == pytest.approx(0, abs=1e-9) and trip is None


def test_weld_attach_level_check_trips_tilted_carton():
    a = math.radians(9)
    q = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)   # 9 deg about Y
    roll, pitch, tilt, trip = cl.weld_attach_level_check(q, bound_deg=2.0)
    assert tilt == pytest.approx(9, abs=1e-6)
    assert trip is not None and trip.startswith(cl.CARTON_TILT)


# ------------------------------------------------------------- FK + leveling

@need_urdf
def test_weld_pose_is_side_grip():
    from twin.urdf_probe import parse_urdf
    r = parse_urdf(URDF)
    from twin.tools.carton_level_bench import WELD_CONFIG
    R = cl.tool_rotation(r, WELD_CONFIG)
    # tool +Z points world +Y at the measured coldwin16 grasp attitude
    assert R[:, 2] == pytest.approx([0, 1, 0], abs=1e-3)


@need_urdf
def test_level_config_drives_tilt_to_zero():
    from twin.urdf_probe import parse_urdf
    r = parse_urdf(URDF)
    W = dict(zip(ARM_JOINTS, [0, -math.pi / 2, math.pi / 2, 0, 0, 0]))
    u = cl.carton_up_in_tool(cl.tool_rotation(r, W))
    carry = dict(W)
    carry['ur_shoulder_lift_joint'] = -1.0
    carry['ur_elbow_joint'] = 1.1
    carry['ur_wrist_1_joint'] = 0.6
    carry['ur_wrist_2_joint'] = 0.3
    before = cl.carried_rpy(r, carry, u)
    assert before[2] > 10            # the raw fold tilts the carton well off level
    lev = cl.level_config(r, carry, u)
    after = cl.carried_rpy(r, lev, u)
    assert abs(after[0]) < 0.1 and abs(after[1]) < 0.1   # leveled


@need_urdf
def test_level_config_respects_clear_wrist_limits():
    from twin.urdf_probe import parse_urdf
    from twin.tools.carton_level_bench import clear_wrist_limits
    r = parse_urdf(URDF)
    W = dict(zip(ARM_JOINTS, [0, -math.pi / 2, math.pi / 2, 0, 0, 0]))
    u = cl.carton_up_in_tool(cl.tool_rotation(r, W))
    carry = dict(W)
    carry['ur_shoulder_lift_joint'] = -1.0
    carry['ur_wrist_1_joint'] = 0.6
    carry['ur_wrist_2_joint'] = 0.3
    lev = cl.level_config(r, carry, u, limits=clear_wrist_limits())
    assert cl.carried_rpy(r, lev, u)[2] < 0.2
    assert lev['ur_wrist_1_joint'] <= 0.65 + 1e-9
    assert lev['ur_wrist_2_joint'] >= 0.0 - 1e-9


@need_urdf
def test_interpolate_level_keeps_every_waypoint_level():
    from twin.urdf_probe import parse_urdf
    r = parse_urdf(URDF)
    W = dict(zip(ARM_JOINTS, [0, -math.pi / 2, math.pi / 2, 0, 0, 0]))
    u = cl.carton_up_in_tool(cl.tool_rotation(r, W))
    g = dict(W)
    g['ur_shoulder_lift_joint'] = -1.0
    g['ur_wrist_1_joint'] = 0.5
    wps = cl.interpolate_level(r, W, g, u, n=8)
    assert len(wps) == 9
    for wp in wps:
        roll, pitch, _ = cl.carried_rpy(r, wp, u)
        assert abs(roll) < 0.2 and abs(pitch) < 0.2
