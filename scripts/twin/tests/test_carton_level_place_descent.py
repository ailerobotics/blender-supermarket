"""CPU-only: the GENTLED, OBSERVABLE collect place-on-cart SET-DOWN descent
(sim-carton-level-preplace-v2).

collectlevel3 tripped CARTON_TILT at phase=PLACE (roll+0.10 pitch+2.22 deg, sustained 0.167 s > the
0.15 s grace) -- NOT at the pre-place segment PR 298 covered (its leveled approach ran at predicted
0.11 deg pitch and never tripped). The place-on-cart SET-DOWN is the ONE carried segment still driven
as a single 2-point straight-line begin_joint_goal. Its COMMANDED straight joint line is already LEVEL
(standoff and set-down both full-DOF leveled; the straight interior stays < 0.05 deg -- proved here on
the REAL mission kinematics), so the 2.22 deg is an EXECUTION transient of the loaded set-down: at the
standard 0.08 rad/s the welded carton's DIFFERENTIAL joint tracking during the heavy shoulder/elbow
descent leaves level. The fix TIME-PARAMETERISES the SAME level path slower
(carton_level.place_descent_speed/accel); the bound and grace are UNCHANGED. This replays the executed
joints through a representative tracking-lag model to reproduce 2.22 and shows the gentled descent stays
under 1 deg, and checks the controller wiring + the new CARTON_SEGMENT_TILT observability.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_carton_level_place_descent.py
"""
import math
import os

import pytest

from twin.restock_manipulation import carton_level as cl
from twin.tools import carton_level_bench as bench

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/collectlevel3/mobile-import.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='mission URDF not present')

# sim-carton-level-place-descent-v1: the coldwin34 run (the complete side-bay cold cycle with PR 318 active
# that tripped CARTON_TILT roll -0.13 pitch +2.06 at sim 202.95, phase=place -- carton_level:true =>
# gravity_compensation None, so the set-down commanded a geometric-level config with NO droop pre-tilt).
CW34_URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin34/mobile-import.urdf'
need_cw34 = pytest.mark.skipif(not os.path.exists(CW34_URDF), reason='coldwin34 URDF not present')


# ------------------------------------------------------------------ config (no URDF)

def test_resolve_place_descent_defaults():
    lc = cl.resolve_level_config(True)
    assert lc.place_descent_speed == pytest.approx(0.02)
    assert lc.place_descent_accel == pytest.approx(0.04)


def test_resolve_place_descent_override():
    lc = cl.resolve_level_config({'place_descent_speed': 0.03, 'place_descent_accel': 0.05})
    assert lc.place_descent_speed == pytest.approx(0.03)
    assert lc.place_descent_accel == pytest.approx(0.05)


@pytest.mark.parametrize('bad', [{'place_descent_speed': 0.0}, {'place_descent_speed': -1},
                                 {'place_descent_accel': 0.0}, {'place_descent_accel': float('inf')}])
def test_resolve_place_descent_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        cl.resolve_level_config(bad)


def test_off_leaves_place_descent_inert():
    assert cl.resolve_level_config(False) is None
    assert cl.resolve_level_config('off') is None


# ------------------------------------------------------------------ bench proof (REAL kinematics)

@need_urdf
def test_commanded_place_descent_straight_line_is_level():
    """The COMMANDED single-2-point straight joint line standoff->set-down is LEVEL: the trip is NOT a
    planned-pose tilt. (This is the correction to the brief's pre-place framing.)"""
    r = bench.run_place_descent(URDF)
    assert r.commanded_max_pitch < 0.1, r.commanded_max_pitch
    assert r.commanded_max_tilt < 0.1, r.commanded_max_tilt


@need_urdf
def test_ungentled_execution_reproduces_the_2p22_trip():
    """The representative tracking-lag model reproduces the collectlevel3 executed peak (~2.22 deg pitch)
    on the standard 0.08 rad/s descent -- an EXECUTION transient, while the command is level."""
    r = bench.run_place_descent(URDF)
    assert r.ungentled_speed == pytest.approx(0.08)
    assert r.ungentled_max_pitch == pytest.approx(2.22, abs=0.1), r.ungentled_max_pitch
    # it is the execution, not the plan: the ungentled executed pitch is >> the commanded pitch.
    assert r.ungentled_max_pitch > 10 * r.commanded_max_pitch


@need_urdf
def test_gentled_execution_stays_under_one_degree():
    """Time-parameterising the SAME level path at the default gentled speed keeps the executed carton
    under 1 deg (the brief's target) with comfortable margin to the 2.0 deg bound -- bound unchanged."""
    r = bench.run_place_descent(URDF)
    assert r.gentled_speed == pytest.approx(0.02)
    assert r.gentled_max_pitch < 1.0, r.gentled_max_pitch
    assert r.gentled_max_pitch < 0.5 * cl.DEFAULT_TILT_BOUND_DEG      # < 1.0 deg, well under the 2.0 bound
    assert r.gentled_duration > r.ungentled_duration                 # slower (time-parameterised)


@need_urdf
def test_required_speed_ceiling_monotone():
    """State the required ceiling: the executed peak falls monotonically as the descent is gentled, and
    crosses under 1 deg at or below the default 0.02 rad/s."""
    peaks = {}
    for s, a in [(0.08, 0.15), (0.04, 0.07), (0.03, 0.05), (0.02, 0.04)]:
        peaks[s] = bench.run_place_descent(URDF, gentled_speed=s, gentled_accel=a).gentled_max_pitch
    assert peaks[0.08] > peaks[0.04] > peaks[0.03] > peaks[0.02]      # monotone in speed
    assert peaks[0.02] < 1.0


# ------------------------------------------------------------------ controller wiring + observability

@need_urdf
def test_place_descent_passes_gentled_ceilings_to_motion():
    """sim-carton-level-place-descent-v1: the collect-deck level place descent dispatches as a PER-POINT
    wrist-compensated collision-off begin_joint_path (>2 waypoints) carrying the gentled
    place_descent_speed/accel; carton_level OFF leaves it byte-identical (a begin() pose straight-line,
    no joint path -- test_place_descent_off_is_byte_identical)."""
    from twin.tests import test_restock_manipulation as T
    # level_path OFF so the preplace uses the single leveled goal (the leveled-waypoint preplace hits a
    # pre-existing fixture-URDF inversion unrelated to this change); the place descent is independent.
    c = T.make_controller(
        'collect', transport_joints=T._SL_STOW, staged_cart_swing=True,
        cart_pan_joint0=-0.8605, cart_tuck_shape=T._SL_TUCK, reposition_before_place=True,
        collect_place_preplace_shape=T._SL_PREPLACE_SHAPE, collect_place_wrist_guard=True,
        collect_place_straight_line=True,
        carton_level=cl.resolve_level_config({'trip': False, 'level_path': False}), fk_urdf=URDF,
        carry_wrist1_cap=0.65, carry_wrist2_min=0.0)[0]
    m = c.motion
    # a realistic sensed deck TCP + realistic carried joints (the synthetic (.5,.2,.1) + the fixture's zero
    # joints are near-singular for the loaded level-place IK; the real mission supplies both).
    _by_phase, _seq, _jg = T._drive_collect_place(
        c, m, target_pose=(.9, .05, .35, 0., 0., 0., 1.), carry_joints=T._SL_PREPLACE_SHAPE)
    paths = getattr(m, 'joint_paths', [])
    assert paths, 'the level place descent must dispatch a begin_joint_path'
    place = paths[-1]
    assert len(place['waypoints']) > 2                       # per-point, not a 2-point endpoint-only line
    assert place['speed'] == pytest.approx(0.02)             # gentled place_descent ceilings
    assert place['accel'] == pytest.approx(0.04)
    assert place['waypoints'][0] == place['measured']        # the path starts exactly where the arm is


# ------------------------------------------ per-point wrist-compensated descent (place-descent-v1)

@need_cw34
def test_per_point_descent_reproduces_coldwin34_and_lands_level_both_docks():
    """sim-carton-level-place-descent-v1: on coldwin34's REAL kinematics, the UNCOMPENSATED geometric-level
    set-down droops ~2 deg under the two-regime model (reproducing the +2.06 CARTON_TILT), while the
    PER-POINT wrist-compensated descent lands the DEEPEST set-down -- and every compensated waypoint -- level
    (<= 1 deg) at BOTH the south dock and the east cart. Bound + grace untouched."""
    from twin.tools import carton_level_bench as bench
    for geom in ('south', 'east'):
        r = bench.run_place_descent_per_point(CW34_URDF, geometry=geom)
        # BEFORE: the uncompensated set-down reproduces the ~2 deg trip regime (coldwin34 +2.06) ...
        assert r.uncomp_setdown_pitch > 1.5, (geom, r.uncomp_setdown_pitch)
        # ... AFTER: the per-point fix lands the DEEPEST (most-extended) set-down level within the margin ...
        assert r.deepest_exec_pitch <= 1.0, (geom, r.deepest_exec_pitch)
        assert r.deepest_exec_pitch < 0.1, (geom, r.deepest_exec_pitch)       # in fact ~0
        # ... and EVERY compensated waypoint along the insert stays level (not only the endpoint) ...
        assert r.max_compensated_exec_pitch <= 1.0, (geom, r.max_compensated_exec_pitch)
        # ... strictly better than the uncompensated droop, every waypoint in the clear carry wrist region.
        assert r.deepest_exec_pitch < r.uncomp_setdown_pitch
        assert r.all_waypoints_clear
        assert r.passes(1.0)
    # the deeper south dock needs (and the compensation absorbs) more droop than the nearer east cart.
    south = bench.run_place_descent_per_point(CW34_URDF, geometry='south')
    east = bench.run_place_descent_per_point(CW34_URDF, geometry='east')
    assert south.uncomp_setdown_pitch > east.uncomp_setdown_pitch


@need_cw34
def test_per_point_descent_is_extension_following():
    """Each waypoint's wrist pre-rotation is solved against ITS OWN loaded droop (the sag is tau/K and tau
    grows with reach), so the commanded wrist GROWS monotonically into the deepest set-down -- a per-pose
    solve, not one endpoint pre-tilt interpolated. The COMMANDED pitch rises with extension while the
    two-regime EXECUTED stays ~0 throughout."""
    import numpy as np
    from twin.urdf_probe import parse_urdf
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    robot = parse_urdf(CW34_URDF)
    tl = 'gripper_pad_tcp'
    u = cl.carton_up_in_tool(cl.tool_rotation(robot, [1.3208, -1.2763, 2.0895, -0.8131, 1.2988, 0.0], tl))
    big = 2 * math.pi
    limits = {'ur_wrist_1_joint': (-big, 0.65), 'ur_wrist_2_joint': (0.0, big),
              'ur_elbow_joint': (0.1, math.pi)}
    seed = [cl.level_preserve_tcp(robot, [0.093, -1.91, 2.014, -0.103, 1.664, 0.0], u,
                                  tool_link=tl, limits=limits)[j] for j in ARM_JOINTS]
    gc = cl.GravityCompensation(tool_link=tl)
    plan = cl.plan_compensated_place_descent(robot, seed, np.array((1.042, 0.063, 0.436)), seed, u, gc,
                                             tool_link=tl, limits=limits, waypoints=10)
    # the commanded (pre-tilted) carton pitch on the COMPENSATED waypoints rises toward the deepest set-down
    cmd_pitch = [abs(cl.carried_rpy(robot, list(w), u, tl)[1]) for w in plan.waypoints[1:]]
    assert cmd_pitch[-1] > cmd_pitch[0] + 0.3            # extension-following: more pre-tilt deeper in
    assert cmd_pitch[-1] > 1.5                            # the deepest command is pre-tilted ~2 deg ...
    # ... yet the droop-aware predicted executed stays under the bound across the whole insert.
    assert max(plan.max_roll, plan.max_pitch) < cl.DEFAULT_TILT_BOUND_DEG


def test_place_descent_off_is_byte_identical():
    """carton_level OFF: the collect place descent is unchanged -- a begin() POSE straight-line move
    (the leveled begin_joint_goal descent branch is gated off), so no gentled joint goal is issued."""
    from twin.tests import test_restock_manipulation as T
    c, m = T._collect_place_controller(straight_line=True)
    assert c._carton_level is None
    by_phase, _seq, jg_by_phase = T._drive_collect_place(c, m)
    assert by_phase.get('place') is True            # place stays a collision-off begin() pose straight-line
    assert 'place' not in jg_by_phase               # ... never a (gentled) joint goal when the level fix is off
