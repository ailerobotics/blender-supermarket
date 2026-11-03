"""CPU-only: the carton-level opt-in wiring in TransferController (off => byte-identical;
on => levels the carried fold goal + CARTON_TILT trip). No Isaac/ROS/GPU.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest scripts/twin/tests/test_carton_level_controller.py
"""
import math
import os

import pytest

from twin.restock_manipulation import TransferController
from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.tests.test_restock_manipulation import PROFILE, FixtureMotion

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin16/mobile-import.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='mission URDF not present')

# a level grasp pose (tool +Z = world +Y side grip) and a fold goal that would tilt the carton
WELD = (0.0, -math.pi / 2, math.pi / 2, 0.0, 0.0, 0.0)
TILTED_FOLD = (0.0, -1.0, 1.1, 0.6, 0.3, 0.0)


def _ctrl(**kw):
    return TransferController(FixtureMotion(), {'sku-box': PROFILE}, fixture_mode=True, **kw)


def test_off_is_passthrough():
    c = _ctrl()
    assert c._carton_level is None
    assert c._level_carried_goal(TILTED_FOLD, WELD) == TILTED_FOLD
    assert c._carton_tilt_trip(TILTED_FOLD) is None


@need_urdf
def test_on_levels_the_fold_goal():
    c = _ctrl(carton_level=cl.resolve_level_config(True), fk_urdf=URDF,
              carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    robot = c._fk_model()
    c._capture_level_up(WELD)              # grasp pose is level on the shelf
    up = c._carton_up_in_tool
    # the raw fold tilts the carton; the leveled goal does not
    assert cl.carried_rpy(robot, list(TILTED_FOLD), up)[2] > 10
    leveled = c._level_carried_goal(TILTED_FOLD, WELD)
    assert leveled != TILTED_FOLD
    roll, pitch, _ = cl.carried_rpy(robot, list(leveled), up)
    assert abs(roll) < 0.2 and abs(pitch) < 0.2
    # leveling stayed in the clear wrist region
    assert dict(zip(ARM_JOINTS, leveled))['ur_wrist_1_joint'] <= 0.65 + 1e-9
    assert dict(zip(ARM_JOINTS, leveled))['ur_wrist_2_joint'] >= -1e-9


@need_urdf
def test_on_trips_a_tilted_carried_pose():
    c = _ctrl(carton_level=cl.resolve_level_config({'bound_deg': 2.0}), fk_urdf=URDF)
    c._capture_level_up(WELD)
    assert c._carton_tilt_trip(WELD) is None            # the weld pose itself is level
    trip = c._carton_tilt_trip(TILTED_FOLD)             # a tilted carried pose
    assert trip is not None and trip.startswith(cl.CARTON_TILT)


@need_urdf
def test_trip_disabled_flag_suppresses_trip_but_keeps_leveling():
    c = _ctrl(carton_level=cl.resolve_level_config({'trip': False}), fk_urdf=URDF,
              carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    c._capture_level_up(WELD)
    assert c._carton_tilt_trip(TILTED_FOLD) is None     # trip off
    assert c._level_carried_goal(TILTED_FOLD, WELD) != TILTED_FOLD  # leveling still on


# ---------------------------------------------- level place-on-cart wiring (sim-carton-level-place-on-cart-v1)

# The TOP-DOWN collect grasp weld + the real reach-authored place standoff (holds the carton ~180 deg off).
TOP_GRASP_WELD = (-math.pi, -2.2, 2.2, -math.pi, 0.0, 0.0)
PLACE_PREPLACE = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)


def test_level_place_goal_off_is_passthrough():
    c = _ctrl(carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    assert c._carton_level is None
    assert c._level_place_goal(PLACE_PREPLACE, TOP_GRASP_WELD) == PLACE_PREPLACE


@need_urdf
def test_level_place_flag_off_is_passthrough_even_with_carton_level_on():
    c = _ctrl(carton_level=cl.resolve_level_config({'level_place': False}), fk_urdf=URDF,
              carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    c._capture_level_up(TOP_GRASP_WELD)
    assert c._level_place_goal(PLACE_PREPLACE, TOP_GRASP_WELD) == PLACE_PREPLACE


@need_urdf
def test_level_place_goal_levels_the_standoff_preserving_reach():
    c = _ctrl(carton_level=cl.resolve_level_config(True), fk_urdf=URDF,
              carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    robot = c._fk_model()
    c._capture_level_up(TOP_GRASP_WELD)
    up = c._carton_up_in_tool
    pad = 'gripper_pad_tcp'                                           # the reach frame (port IK tcp)
    assert cl.carried_rpy(robot, list(PLACE_PREPLACE), up)[2] > 175   # reach shape holds it inverted
    leveled = c._level_place_goal(PLACE_PREPLACE, TOP_GRASP_WELD)
    assert leveled != PLACE_PREPLACE
    roll, pitch, _ = cl.carried_rpy(robot, list(leveled), up)
    assert abs(roll) < 0.1 and abs(pitch) < 0.1                       # level
    assert cl.tool_position(robot, list(leveled), pad) == pytest.approx(
        cl.tool_position(robot, list(PLACE_PREPLACE), pad), abs=3e-3)  # SAME deck reach (pad TCP)


@need_urdf
def test_level_place_goal_reaches_a_lowered_sensed_target():
    c = _ctrl(carton_level=cl.resolve_level_config(True), fk_urdf=URDF,
              carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    robot = c._fk_model()
    c._capture_level_up(TOP_GRASP_WELD)
    up = c._carton_up_in_tool
    pad = 'gripper_pad_tcp'
    standoff = c._level_place_goal(PLACE_PREPLACE, TOP_GRASP_WELD)
    target = list(cl.tool_position(robot, list(standoff), pad))       # the sensed deck is a pad-TCP pose
    target[2] -= 0.12
    setdown = c._level_place_goal(standoff, TOP_GRASP_WELD, target_pos=target)
    assert cl.tool_position(robot, list(setdown), pad) == pytest.approx(target, abs=3e-3)  # reaches the deck
    roll, pitch, _ = cl.carried_rpy(robot, list(setdown), up)
    assert abs(roll) < 0.1 and abs(pitch) < 0.1                       # level


@need_urdf
def test_level_place_goal_unsolved_returns_goal_for_the_trip(capsys):
    c = _ctrl(carton_level=cl.resolve_level_config(True), fk_urdf=URDF,
              carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    c._capture_level_up(TOP_GRASP_WELD)
    # an unreachable target: the helper must not crash and must return the seed (CARTON_TILT trip owns it)
    out = c._level_place_goal(PLACE_PREPLACE, TOP_GRASP_WELD, target_pos=(5.0, 5.0, 5.0))
    assert out == PLACE_PREPLACE
    assert 'CARTON_LEVEL_UNSOLVED' in capsys.readouterr().out


@need_urdf
def test_end_to_end_collect_place_levels_the_preplace_standoff_and_reaches_verify():
    """End-to-end wiring: driving the REAL collect reposition place with carton_level ON issues a LEVELED
    preplace standoff (not the raw reach shape) and still completes to verify. trip=False so the fixture's
    constant measured joints do not false-trip the release level gate (the trip is unit-tested above)."""
    from twin.tests import test_restock_manipulation as T
    c = T.make_controller(
        'collect', transport_joints=T._SL_STOW, staged_cart_swing=True,
        cart_pan_joint0=-0.8605, cart_tuck_shape=T._SL_TUCK, reposition_before_place=True,
        collect_place_preplace_shape=T._SL_PREPLACE_SHAPE, collect_place_wrist_guard=True,
        collect_place_straight_line=True,
        carton_level=cl.resolve_level_config({'trip': False}), fk_urdf=URDF,
        carry_wrist1_cap=0.65, carry_wrist2_min=0.0)[0]
    m = c.motion
    robot = c._fk_model()
    c._capture_level_up(TOP_GRASP_WELD)              # the real top-down grasp up-axis (idempotent)
    up = c._carton_up_in_tool
    assert cl.carried_rpy(robot, list(T._SL_PREPLACE_SHAPE), up)[2] > 175   # raw shape holds it inverted
    _by_phase, seq, jg_by_phase = T._drive_collect_place(c, m)
    assert c.phase == 'verify', seq
    preplace_goal = jg_by_phase['preplace']['goal']
    assert list(preplace_goal) != list(T._SL_PREPLACE_SHAPE)               # the standoff was re-planned level
    roll, pitch, _ = cl.carried_rpy(robot, list(preplace_goal), up)
    assert abs(roll) < 0.1 and abs(pitch) < 0.1                            # ... and it is level
    assert jg_by_phase['preplace']['straight_line'] is False               # standoff stays collision-aware
    # same deck reach preserved (pad TCP within a few mm of the raw reach shape)
    assert cl.tool_position(robot, list(preplace_goal), 'gripper_pad_tcp') == pytest.approx(
        cl.tool_position(robot, list(T._SL_PREPLACE_SHAPE), 'gripper_pad_tcp'), abs=5e-3)


@need_urdf
def test_end_to_end_collect_place_off_is_byte_identical_preplace():
    """carton_level OFF: the collect place preplace is the raw reach shape, byte-identical to the baseline."""
    from twin.tests import test_restock_manipulation as T
    c, m = T._collect_place_controller(straight_line=True)
    assert c._carton_level is None
    _by_phase, _seq, jg_by_phase = T._drive_collect_place(c, m)
    assert list(jg_by_phase['preplace']['goal']) == list(T._SL_PREPLACE_SHAPE)


# ----------------------------------------- sustained-tilt debounce wiring (sim-carton-tilt-transient-debounce-v1)

def test_debounce_off_is_passthrough():
    c = _ctrl()
    assert c._tilt_debounce is None
    assert c._carton_tilt_observe(0.0, TILTED_FOLD) is None     # off => no decision, byte-identical


@need_urdf
def test_controller_debounces_a_one_tick_tilt_spike():
    c = _ctrl(carton_level=cl.resolve_level_config({'bound_deg': 2.0}), fk_urdf=URDF)
    c._capture_level_up(WELD)                                   # level at the weld
    dt = 1.0 / 60.0
    v = c._carton_tilt_observe(1 * dt, TILTED_FOLD)            # one tilted tick (>10 deg off)
    assert v is not None and v.trip is None                    # within grace -> held, no abort
    v = c._carton_tilt_observe(2 * dt, WELD)                   # back to level before the window ends
    assert v.trip is None and v.transient is not None          # logged transient, still no abort
    assert v.transient.startswith(cl.CARTON_TILT_TRANSIENT)


@need_urdf
def test_controller_trips_on_sustained_tilt_at_window_end():
    c = _ctrl(carton_level=cl.resolve_level_config({'bound_deg': 2.0}), fk_urdf=URDF)
    c._capture_level_up(WELD)
    dt = 1.0 / 60.0
    trip_at = None
    for i in range(60):
        v = c._carton_tilt_observe(i * dt, TILTED_FOLD)        # held off level continuously
        if v.trip is not None:
            trip_at = i * dt
            break
    assert trip_at is not None
    assert trip_at >= cl.DEFAULT_TILT_GRACE_S - 1e-9           # not before the grace window
    assert cl.CARTON_TILT in v.trip and 'sustained' in v.trip


@need_urdf
def test_transition_resets_the_debounce_window():
    c = _ctrl(carton_level=cl.resolve_level_config({'bound_deg': 2.0}), fk_urdf=URDF)
    c._capture_level_up(WELD)
    dt = 1.0 / 60.0
    c._carton_tilt_observe(0.0, TILTED_FOLD)                   # start an exceedance
    assert c._tilt_debounce.exceeding
    c._transition('tuck')                                      # a new move boundary
    assert not c._tilt_debounce.exceeding                      # window cleared -> no cross-phase charge
