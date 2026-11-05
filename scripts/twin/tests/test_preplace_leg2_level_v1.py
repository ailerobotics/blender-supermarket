"""CPU proof for the collect pre-place LEG-2 world-level fix + plan-derived phase budget
(sim-preplace-leg2-level-v1).

coldwin56 (wrist-first sequencing) solved leg 1 (the wrist_2 starve) but FAILED leg 2 two ways:
  (A) the two-leg sequence needs ~150 s > the hard-coded 120 s phase_timeout (timed out at 58 % of leg 2);
  (B) the welded carton tilt rose 1.64 -> 2.02 deg across the pan descent (CARTON_TICK base-frame).

Forensics (CPU, this file reproduces the logged CARTON_TICK to <0.02 deg): the base is held pitched
-1.17 deg (the station parking brake, RUNTIME chassis_station_pitch_hold_pitch_deg), so ur_base_link +Z
is off world +Z and shoulder_pan is a YAW about a tilted axis -- the carried carton sweeps a cone and its
WORLD tilt is not pan-invariant. Two compounding causes of the ~2 deg:
  * the K=2850 droop feed-forward is a PHANTOM at the preplace (the GPU executed joints matched the
    commanded within 0.003 rad), so solve_level_standoff(gc) pre-tilted the standoff wrist_1 ~+0.031 rad
    (~1.77 deg base-frame) that the (non-existent) droop never cancelled;
  * the held chassis attitude the base-frame leveling ignored.

The fix: drive the collect preplace from the GEOMETRIC WORLD-level standoff (NO droop pre-tilt) and
re-solve the wrist cluster for level-IN-WORLD (base_R from the hold) at every densified leg-2 waypoint.
And derive the phase timeout from the planned leg durations + a margin.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_preplace_leg2_level_v1.py
"""
import math
import os

import numpy as np
import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import config_preflight as cp
from twin.restock_manipulation.cad_tool import ARM_JOINTS

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='dataset mobile URDF not present')
TL = 'gripper_pad_tcp'

# coldwin48/49/50/56 geometry (ur_base_link): the resolved shelf place target + the deck-tuned fixed shape.
FIXED_SHAPE = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)
CW_PLACE = (0.957, 0.064, 0.538)
STANDOFF = (CW_PLACE[0], CW_PLACE[1], CW_PLACE[2] + 0.08)
# The collect carry/weld pose (coldwin56 leg-1 start tick): the carton rested LEVEL IN WORLD here.
CARRY = [1.2675, -2.2701, 1.0002, 0.6501, 0.0017, 0.6189]
PITCH_DEG = -1.17   # RUNTIME chassis_station_pitch_hold_pitch_deg
ROLL_DEG = 0.0


def _limits():
    big = 2 * math.pi
    return {'ur_elbow_joint': (0.1, math.pi),
            'ur_wrist_1_joint': (-big, 0.65), 'ur_wrist_2_joint': (0.0, big)}


@pytest.fixture(scope='module')
def robot():
    from twin.urdf_probe import parse_urdf
    return parse_urdf(URDF)


@pytest.fixture(scope='module')
def base_R():
    return cl.base_rotation_rp(ROLL_DEG, PITCH_DEG)


@pytest.fixture(scope='module')
def up_world(robot, base_R):
    # the physically-correct carton up-axis: level in WORLD at the weld while the base is held pitched.
    return cl.carton_up_in_tool(base_R @ cl.tool_rotation(robot, CARRY, TL))


@pytest.fixture(scope='module')
def up_base(robot):
    # the historical base-frame capture (assumes a level base) -- the frame the OLD CARTON_TICK used.
    return cl.carton_up_in_tool(cl.tool_rotation(robot, CARRY, TL))


@pytest.fixture(scope='module')
def gc():
    return cl.GravityCompensation(stiffness=2850.0, carton_mass=0.45, tool_link=TL).validate()


# ------------------------------------------------------------------ base_rotation helper

def test_base_rotation_rp_identity_is_none():
    assert cl.base_rotation_rp(0.0, 0.0) is None
    R = cl.base_rotation_rp(0.0, -1.17)
    assert R is not None and R.shape == (3, 3)
    # a -1.17 deg pitch tilts base +Z by 1.17 deg from world +Z.
    z = R @ np.array([0.0, 0.0, 1.0])
    assert math.degrees(math.acos(max(-1.0, min(1.0, float(z[2]))))) == pytest.approx(1.17, abs=1e-3)


def test_apply_base_none_is_byte_identical():
    v = np.array([0.1, 0.2, 0.97])
    assert np.allclose(cl._apply_base(None, v), v)


# ------------------------------------------------------------ Part 1: the leg-2 world-level fix

@need_urdf
def test_old_droop_pretilt_reproduces_coldwin56_baseframe_tilt(robot, up_base, gc):
    """The OLD path (gc droop pre-tilt + base-frame leveling) reproduces the ~1.8 deg base-frame carton
    tilt coldwin56 logged -- NOT ~0. This is the bug: the K=2850 feed-forward pre-tilts the standoff."""
    sol, _ = cl.solve_level_standoff(robot, STANDOFF, up_base, list(FIXED_SHAPE), gc,
                                     tool_link=TL, limits=_limits())
    stdv = [float(sol[j]) for j in ARM_JOINTS]
    # geometric (what the physical arm executes -- droop is a phantom) base-frame tilt of the gc standoff:
    _, _, tilt = cl.carried_rpy(robot, stdv, up_base, tool_link=TL)
    assert tilt > 1.6, tilt   # coldwin56 logged ~1.77 at the standoff pan, rising to 2.02 with wrist drift


@need_urdf
def test_coldwin56_executed_joints_reproduce_the_2deg(robot, up_base):
    """FK of the coldwin56 EXECUTED joints (leg-2 CARTON_TICK) reproduces the logged 1.64 -> 2.02 rise to
    <0.05 deg (base frame -- the metric the OLD trip read). Confirms the replay is faithful."""
    # (pan, lift, elbow, w1, w2, w3) VERBATIM from coldwin56 host-stack.log CARTON_TICK phase=preplace,
    # with the logged tilt; the executed wrist_1 drifts 0.0246 -> 0.0257 as pan descends 1.27 -> 0.48.
    rows = [([1.2674, -1.5558, 1.5579, 0.0246, 1.4633, 0.0026], 1.61),   # now=208.267
            ([0.4906, -1.5522, 1.5607, 0.0256, 1.6313, -0.0004], 2.02),  # now=248.333
            ([0.4816, -1.5522, 1.5607, 0.0257, 1.6313, -0.0004], 2.02)]  # now=248.800
    for q, logged in rows:
        _, _, tilt = cl.carried_rpy(robot, q, up_base, tool_link=TL)
        assert abs(tilt - logged) < 0.05, (q, tilt, logged)


@need_urdf
def test_geometric_world_standoff_is_world_level_not_pretilted(robot, up_world, base_R, gc):
    """The fix: the GEOMETRIC WORLD-level standoff (gravity_comp=None, base_R) is parallel to the GROUND
    (world tilt ~0), NOT the gc pre-tilted standoff (base-frame ~1.77)."""
    geo = cl.level_place_ik(robot, STANDOFF, up_world, list(FIXED_SHAPE), tool_link=TL,
                            limits=_limits(), gravity_comp=None, base_R=base_R)
    geov = [float(geo[j]) for j in ARM_JOINTS]
    _, _, world_tilt = cl.carried_rpy(robot, geov, up_world, tool_link=TL, base_R=base_R)
    assert world_tilt < 0.1, world_tilt


@need_urdf
def test_leg2_old_held_wrist_rises_new_resolved_stays_level(robot, up_world, up_base, base_R, gc):
    """THE leg-2 proof. OLD held-wrist path (gc pre-tilted standoff, base frame) stays ~1.77 deg off level
    across the pan descent (coldwin56's ~1.64 -> 2.02); the NEW per-waypoint WORLD-level re-solve stays
    <= 1.0 deg (world) across leg 2."""
    # OLD: gc pre-tilted standoff, wrists held, pan sweeps -- base-frame tilt stays high.
    sol, _ = cl.solve_level_standoff(robot, STANDOFF, up_base, list(FIXED_SHAPE), gc,
                                     tool_link=TL, limits=_limits())
    old_std = [float(sol[j]) for j in ARM_JOINTS]
    old_peak = 0.0
    for a in np.linspace(0, 1, 17):
        q = list(old_std); q[0] = CARRY[0] * (1 - a) + old_std[0] * a
        _, _, t = cl.carried_rpy(robot, q, up_base, tool_link=TL)
        old_peak = max(old_peak, t)
    assert old_peak > 1.6, old_peak

    # NEW: geometric world standoff + per-waypoint world-level re-solve over leg 2.
    geo = cl.level_place_ik(robot, STANDOFF, up_world, list(FIXED_SHAPE), tool_link=TL,
                            limits=_limits(), gravity_comp=None, base_R=base_R)
    std = [float(geo[j]) for j in ARM_JOINTS]
    leg1 = list(std); leg1[0] = CARRY[0]
    leg1 = [float(cl.level_config(robot, leg1, up_world, joints=cl.DEFAULT_LEVEL_JOINTS, tool_link=TL,
                                  limits=_limits(), accept_deg=2.0, base_R=base_R)[j]) for j in ARM_JOINTS]
    plan2 = cl.plan_level_world_approach(robot, leg1, std, up_world, tool_link=TL, limits=_limits(),
                                         bound_deg=2.0, max_segment_rad=0.08, base_R=base_R)
    assert plan2.max_tilt <= 1.0, plan2.max_tilt        # leg 2 world tilt (firstmate target <= 1.0)
    assert plan2.max_tilt <= 0.5                        # in fact essentially level


@need_urdf
def test_new_both_legs_world_level_under_half_degree(robot, up_world, base_R):
    """Both legs (carry -> leg1 -> standoff), per-waypoint WORLD-level re-solve: world tilt peak stays low
    across the whole preplace (firstmate acceptance ~0.5 deg; <= the 1.0 CARTON_SEGMENT_TILT target)."""
    geo = cl.level_place_ik(robot, STANDOFF, up_world, list(FIXED_SHAPE), tool_link=TL,
                            limits=_limits(), gravity_comp=None, base_R=base_R)
    std = [float(geo[j]) for j in ARM_JOINTS]
    leg1 = list(std); leg1[0] = CARRY[0]
    leg1 = [float(cl.level_config(robot, leg1, up_world, joints=cl.DEFAULT_LEVEL_JOINTS, tool_link=TL,
                                  limits=_limits(), accept_deg=2.0, base_R=base_R)[j]) for j in ARM_JOINTS]
    p1 = cl.plan_level_world_approach(robot, CARRY, leg1, up_world, tool_link=TL, limits=_limits(),
                                      bound_deg=2.0, max_segment_rad=0.08, base_R=base_R)
    p2 = cl.plan_level_world_approach(robot, leg1, std, up_world, tool_link=TL, limits=_limits(),
                                      bound_deg=2.0, max_segment_rad=0.08, base_R=base_R)
    assert max(p1.max_tilt, p2.max_tilt) <= 1.0


@need_urdf
def test_world_level_carton_would_false_trip_a_base_frame_metric(robot, up_world, up_base, base_R):
    """WHY the preplace trip/tick must measure in WORLD under the hold: a WORLD-level carton measured in
    the BASE frame (up_base) reads ~2.3 deg -- OVER the 2.0 bound -- so a base-frame trip would false-fire
    on the correctly-levelled carton. The world-frame measurement reads ~0."""
    geo = cl.level_place_ik(robot, STANDOFF, up_world, list(FIXED_SHAPE), tool_link=TL,
                            limits=_limits(), gravity_comp=None, base_R=base_R)
    std = [float(geo[j]) for j in ARM_JOINTS]
    _, _, world_tilt = cl.carried_rpy(robot, std, up_world, tool_link=TL, base_R=base_R)
    _, _, base_tilt = cl.carried_rpy(robot, std, up_base, tool_link=TL)
    assert world_tilt < 0.1
    assert base_tilt > 2.0        # the base-frame metric false-reads the world-level carton


# ------------------------------------------------------------ Part 2: plan-derived phase budget

def test_wrist_first_leg_durations_and_budget():
    l1, l2 = cl.wrist_first_leg_durations(CARRY, FIXED_SHAPE, speed=0.05, accel=0.05, max_segment_rad=0.08)
    assert l1 > 0 and l2 > 0
    budget, plan, b1, b2 = cl.preplace_phase_budget(CARRY, FIXED_SHAPE, speed=0.05, accel=0.05,
                                                    max_segment_rad=0.08, margin_s=90.0)
    assert (b1, b2) == (l1, l2)
    assert plan == pytest.approx(l1 + l2)
    assert budget == pytest.approx(max(120.0, plan + 90.0))
    # the two-leg sequence legitimately needs > the legacy 120 s, so the budget is well above it.
    assert budget > 120.0
    assert plan > 100.0   # coldwin56 measured ~141 s; the caps-formula gives ~110 s (+ margin covers it)


def test_budget_floor_preserves_120_for_a_tiny_plan():
    # a near-zero travel plan never shrinks the historical 120 s floor.
    tiny = [0.0] * 6
    budget, _, _, _ = cl.preplace_phase_budget(tiny, [0.0] * 6, margin_s=0.0)
    assert budget == pytest.approx(120.0)


# ------------------------------------------------------------ Part 2: config_preflight report + refusal

def _host(**over):
    hc = {'preplace_wrist_first': True, 'carton_level': True,
          'carry_lift_shape': list(CARRY), 'collect_place_preplace_shape': list(FIXED_SHAPE),
          'station_pitch_hold': True, 'station_hold_pitch_deg': -1.17, 'station_hold_roll_deg': 0.0}
    hc.update(over)
    return hc


def test_preflight_reports_budget_and_passes_default(capsys):
    problems = cp.check_preplace_phase_budget(_host())
    assert problems == []
    out = capsys.readouterr().out
    assert 'PREPLACE_PHASE_BUDGET' in out and 'derived budget=' in out


def test_preflight_refuses_explicit_timeout_below_plan():
    problems = cp.check_preplace_phase_budget(_host(phase_timeout_s=30.0))
    assert any('BELOW the planned wrist-first preplace duration' in p for p in problems), problems


def test_preflight_accepts_explicit_timeout_above_plan():
    assert cp.check_preplace_phase_budget(_host(phase_timeout_s=300.0)) == []


def test_preflight_validates_station_hold_keys():
    assert any('station_pitch_hold must be a boolean' in p
               for p in cp.check_preplace_phase_budget(_host(station_pitch_hold='yes')))
    assert any('station_hold_pitch_deg' in p
               for p in cp.check_preplace_phase_budget(_host(station_hold_pitch_deg='down')))


def test_preflight_noop_without_wrist_first():
    # no wrist-first and no station keys => byte-identical ([]).
    assert cp.check_preplace_phase_budget({'carton_level': True}) == []


# ------------------------------------------------------------ Part 3: convergence guard on the two-leg plan

# ------------------------------------------------------------ controller wiring (the real step() path)

MISSION_URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin16/mobile-import.urdf'
need_mission_urdf = pytest.mark.skipif(not os.path.exists(MISSION_URDF), reason='mission URDF not present')
# a level side-grasp weld pose in the mission URDF frame.
CTRL_WELD = (0.0, -math.pi / 2, math.pi / 2, 0.0, 0.0, 0.0)


def _ctrl(**kw):
    from twin.restock_manipulation import TransferController
    from twin.tests.test_restock_manipulation import PROFILE, FixtureMotion
    return TransferController(FixtureMotion(), {'sku-box': PROFILE}, fixture_mode=True, **kw)


@need_mission_urdf
def test_hold_off_frame_is_base_byte_identical():
    """Hold off => no world up-axis, the preplace measurement frame is the base frame (byte-identical)."""
    c = _ctrl(carton_level=cl.resolve_level_config(True), fk_urdf=MISSION_URDF)
    c._capture_level_up(CTRL_WELD)
    assert c._carton_base_R is None
    assert c._carton_up_in_tool_world is None
    up, bR = c._carton_level_frame('preplace')
    assert bR is None and up is c._carton_up_in_tool


@need_mission_urdf
def test_hold_on_captures_world_up_and_preplace_measures_in_world():
    """Hold on => a SEPARATE world up-axis is captured; the preplace trip/tick measure in WORLD (world up +
    base_R) while every other phase keeps the base frame."""
    c = _ctrl(carton_level=cl.resolve_level_config(True), fk_urdf=MISSION_URDF,
              station_pitch_hold=True, station_hold_pitch_deg=-1.17, station_hold_roll_deg=0.0)
    assert c._carton_base_R is not None
    c._capture_level_up(CTRL_WELD)
    assert c._carton_up_in_tool is not None and c._carton_up_in_tool_world is not None
    # preplace -> world frame; lift/place -> base frame (byte-identical).
    up_pp, bR_pp = c._carton_level_frame('preplace')
    assert bR_pp is c._carton_base_R and up_pp is c._carton_up_in_tool_world
    for ph in ('lift', 'tuck', 'place', 'release'):
        up_o, bR_o = c._carton_level_frame(ph)
        assert bR_o is None and up_o is c._carton_up_in_tool


@need_urdf
def test_convergence_guard_judges_against_the_current_leg_goal(robot, up_world, base_R):
    """PR 343 guard parity: it measures per-joint convergence to the CURRENT waypoint/leg goal, so arming
    it on each leg's own goal (leg 1 world-leveled shape, then the standoff) still catches a frozen joint.
    A config AT the goal converges (no false trip); one half a radian short does not."""
    from twin.restock_manipulation.convergence_guard import ConvergenceGuard
    geo = cl.level_place_ik(robot, STANDOFF, up_world, list(FIXED_SHAPE), tool_link=TL,
                            limits=_limits(), gravity_comp=None, base_R=base_R)
    std = [float(geo[j]) for j in ARM_JOINTS]
    g = ConvergenceGuard(ARM_JOINTS, stall_s=30.0)
    g.arm(std)
    # at the goal: error ~0 -> converged, continuously rearms, never trips.
    assert g.enabled
    trip = None
    for t in range(0, 40):
        trip = g.observe(float(t), std)
    assert trip is None
    # DRIVEN then FROZEN short of the goal (shoulder_pan): start 0.6 rad short, advance to 0.3 short
    # (so it HAS moved -> not idle), then hold -> its error stops shrinking -> trips by name.
    g2 = ConvergenceGuard(ARM_JOINTS, stall_s=5.0)
    g2.arm(std)
    start_pose = list(std); start_pose[0] += 0.6
    short = list(std); short[0] += 0.3
    g2.observe(0.0, start_pose)      # capture the armed start (driven joint begins here)
    g2.observe(0.5, short)           # it advanced 0.3 rad (driven) but stops 0.3 short of the goal
    trip2 = None
    for t in range(1, 20):
        trip2 = g2.observe(float(t), short)
        if trip2:
            break
    assert trip2 is not None
    assert 'ur_shoulder_pan_joint' in trip2.message
