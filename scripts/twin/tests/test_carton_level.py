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


# -------------------------------------------- level-place IK (reach + level), sim-carton-level-place-on-cart-v1

# The TOP-DOWN collect grasp up-axis (PR 287): tool0 +Y points world-down, so the level carton's
# up-axis is [0,-1,0] in the tool frame. The real reach-authored place-on-cart standoff.
TOP_GRASP_WELD = (-math.pi, -2.2, 2.2, -math.pi, 0.0, 0.0)
COLLECT_PLACE_PREPLACE = [0.093, -1.91, 2.014, -0.103, 1.664, 0.0]


@need_urdf
def test_tool_position_matches_fk_transform():
    from twin.urdf_probe import parse_urdf
    r = parse_urdf(URDF)
    T = cl.tool_transform(r, COLLECT_PLACE_PREPLACE)
    assert cl.tool_position(r, COLLECT_PLACE_PREPLACE) == pytest.approx(T[:3, 3], abs=1e-9)
    assert np.allclose(cl.tool_rotation(r, COLLECT_PLACE_PREPLACE), T[:3, :3])


@need_urdf
def test_reach_authored_place_standoff_holds_the_top_grasped_carton_inverted():
    from twin.urdf_probe import parse_urdf
    r = parse_urdf(URDF)
    u = cl.carton_up_in_tool(cl.tool_rotation(r, list(TOP_GRASP_WELD)))
    assert u == pytest.approx([0, -1, 0], abs=1e-3)      # top-down grasp up-axis
    tilt = cl.carried_rpy(r, COLLECT_PLACE_PREPLACE, u)[2]
    assert tilt > 175                                     # ~180 deg: a real inverted hold, not an artifact


@need_urdf
def test_level_preserve_tcp_levels_the_standoff_at_the_same_reach():
    from twin.urdf_probe import parse_urdf
    from twin.tools.carton_level_bench import clear_wrist_limits
    r = parse_urdf(URDF)
    u = cl.carton_up_in_tool(cl.tool_rotation(r, list(TOP_GRASP_WELD)))
    tcp0 = cl.tool_position(r, COLLECT_PLACE_PREPLACE)
    lev = cl.level_preserve_tcp(r, COLLECT_PLACE_PREPLACE, u, limits=clear_wrist_limits())
    levv = [lev[j] for j in ARM_JOINTS]
    roll, pitch, _ = cl.carried_rpy(r, levv, u)
    assert abs(roll) < 0.1 and abs(pitch) < 0.1          # level
    assert cl.tool_position(r, levv) == pytest.approx(tcp0, abs=1e-3)   # SAME reach (TCP preserved)
    # the non-wrist_3 joints barely move (same arm branch -> inherits the reach/self-collision validation);
    # the leveling is a wrist_3 flange roll
    for k in range(5):
        assert abs(cl._wrap(levv[k] - COLLECT_PLACE_PREPLACE[k])) < math.radians(2)
    assert abs(cl._wrap(levv[5] - COLLECT_PLACE_PREPLACE[5])) > math.radians(90)


@need_urdf
def test_level_place_ik_reaches_a_lowered_deck_target_while_level():
    from twin.urdf_probe import parse_urdf
    from twin.tools.carton_level_bench import clear_wrist_limits
    r = parse_urdf(URDF)
    u = cl.carton_up_in_tool(cl.tool_rotation(r, list(TOP_GRASP_WELD)))
    limits = clear_wrist_limits()
    limits['ur_elbow_joint'] = (0.1, math.pi)
    lev = cl.level_preserve_tcp(r, COLLECT_PLACE_PREPLACE, u, limits=limits)
    levv = [lev[j] for j in ARM_JOINTS]
    target = cl.tool_position(r, levv).copy()
    target[2] -= 0.12                                    # a vertical set-down descent
    sd = cl.level_place_ik(r, target, u, levv, limits=limits)
    sdv = [sd[j] for j in ARM_JOINTS]
    assert cl.tool_position(r, sdv) == pytest.approx(target, abs=2e-3)   # reaches the lowered deck
    roll, pitch, _ = cl.carried_rpy(r, sdv, u)
    assert abs(roll) < 0.1 and abs(pitch) < 0.1          # and stays level
    # the straight-line (2-point) descent from standoff to set-down stays level throughout
    for i in range(21):
        a = i / 20.0
        wp = [(1 - a) * levv[k] + a * sdv[k] for k in range(6)]
        rr, pp, _ = cl.carried_rpy(r, wp, u)
        assert abs(rr) < 0.1 and abs(pp) < 0.1


@need_urdf
def test_level_place_ik_raises_when_target_is_unreachable():
    from twin.urdf_probe import parse_urdf
    r = parse_urdf(URDF)
    u = cl.carton_up_in_tool(cl.tool_rotation(r, list(TOP_GRASP_WELD)))
    with pytest.raises(ValueError):
        cl.level_place_ik(r, [5.0, 5.0, 5.0], u, COLLECT_PLACE_PREPLACE)   # far outside the envelope


def test_level_place_flag_resolves():
    assert cl.resolve_level_config(True).level_place is True
    assert cl.resolve_level_config({'level_place': False}).level_place is False


# ----------------------------------------------- sustained-tilt debounce (grace window)
# sim-carton-tilt-transient-debounce-v1. These are PURE (no URDF/Isaac): they drive the debounce
# with a crafted (roll, pitch, sim-time) sequence -- the exact shape the controller feeds it.

def test_tilt_grace_resolves_and_defaults():
    assert cl.resolve_level_config(True).tilt_grace_s == cl.DEFAULT_TILT_GRACE_S
    assert cl.resolve_level_config({'tilt_grace_s': 0.25}).tilt_grace_s == 0.25
    assert cl.resolve_level_config({'tilt_grace_s': 0.0}).tilt_grace_s == 0.0
    with pytest.raises(ValueError):
        cl.resolve_level_config({'tilt_grace_s': -0.1})


def test_one_tick_spike_passes_and_is_logged():
    """A single tick over the bound, then back within it => a CARTON_TILT_TRANSIENT (no trip)."""
    d = cl.TiltDebounce(bound_deg=2.0, grace_s=0.15)
    dt = 1.0 / 60.0
    assert d.observe(0 * dt, 0.1, 0.5).trip is None                 # level
    v = d.observe(1 * dt, -0.56, 2.08)                              # one-tick spike over 2.0
    assert v.trip is None and v.transient is None                  # within grace -> hold, no log yet
    assert d.exceeding
    v = d.observe(2 * dt, 0.1, 1.9)                                 # back within the bound
    assert v.trip is None
    assert v.transient is not None and v.transient.startswith(cl.CARTON_TILT_TRANSIENT)
    assert 'peak=2.08' in v.transient
    assert not d.exceeding                                          # exceedance window cleared


def test_sustained_tilt_trips_at_window_end():
    """Tilt held over the bound for the whole grace window trips -- at the window end, not before."""
    grace = 0.15
    d = cl.TiltDebounce(bound_deg=2.0, grace_s=grace)
    dt = 1.0 / 60.0
    tripped_at = None
    for i in range(40):
        now = i * dt
        v = d.observe(now, 0.2, 2.5)                                # continuously off level
        if v.trip is not None:
            tripped_at = now
            break
    assert tripped_at is not None
    # trips the first tick at/after grace, never before it
    assert tripped_at >= grace - 1e-9
    assert tripped_at < grace + dt + 1e-9
    assert cl.CARTON_TILT in v.trip and 'sustained' in v.trip


def test_broken_exceedance_does_not_trip():
    """A tilt that dips back within the bound before the window elapses never trips (window unbroken)."""
    d = cl.TiltDebounce(bound_deg=2.0, grace_s=0.15)
    dt = 1.0 / 60.0
    for i in range(100):
        roll, pitch = (0.0, 2.3) if (i % 4 == 0) else (0.0, 0.5)   # spike every 4th tick only
        v = d.observe(i * dt, roll, pitch)
        assert v.trip is None                                      # never sustained -> never trips


def test_grace_zero_is_immediate_trip_legacy():
    """grace 0 reproduces the legacy immediate trip on the first crossing tick (byte-identical)."""
    d = cl.TiltDebounce(bound_deg=2.0, grace_s=0.0)
    assert d.observe(0.0, 0.1, 1.5).trip is None
    v = d.observe(1.0 / 60.0, -0.56, 2.01)
    assert v.trip is not None and v.trip.startswith(cl.CARTON_TILT)
    assert 'sustained' not in v.trip                               # the plain legacy message


def test_window_scales_with_physics_dt():
    """The grace window is a FRAME-COUNT budget: scaling it by freshness_scale(dt) trips after the
    SAME number of physics ticks at any step (1/60, 1/36, 1/30)."""
    from twin.restock.physics_step import freshness_scale, DEFAULT_DT
    base_grace = 0.15                                              # at 1/60
    def ticks_to_trip(dt):
        d = cl.TiltDebounce(bound_deg=2.0, grace_s=base_grace * freshness_scale(dt))
        for i in range(200):
            if d.observe(i * dt, 0.0, 3.0).trip is not None:
                return i
        return None
    n60 = ticks_to_trip(DEFAULT_DT)
    n36 = ticks_to_trip(1.0 / 36.0)
    n30 = ticks_to_trip(1.0 / 30.0)
    assert n60 == n36 == n30                                       # identical frame budget across steps
    assert n60 == math.ceil(base_grace / DEFAULT_DT)               # 9 ticks at 1/60
