"""CPU-only: the leveled WAYPOINT-PATH carried lift (sim-carry-level-lift-v1).

level_goals alone levels only the ENDPOINT, so the executor's joint-space path from the grasp pose to
the leveled carry_lift_shape passes through a tilted configuration (coldwin25: CPU-reconstructed ~11 deg
mid-path to the leveled endpoint, far over the 2 deg bound). This proves the fix -- the lift is driven
as a leveled waypoint path that keeps the carton parallel to the ground THROUGHOUT -- on the REAL mission
kinematics + REAL carry_lift_shape, and that a plan still over the bound is CARTON_LEVEL_PLAN_REJECTED
before any motion streams.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest scripts/twin/tests/test_carton_level_lift_path.py
"""
import json
import math
import os

import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.tests.test_restock_manipulation import FixtureMotion, make_controller, packet, run_to

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin16/mobile-import.urdf'
HOST = os.path.join(os.path.dirname(__file__), '..', 'restock_manipulation', 'ros_host_config.json')
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='mission URDF not present')

# The coldwin25 grasp/weld pose (ARM order) -- the measured arm config where the carton welds level and
# the lift begins. Reconstructed from the run's last grasp-track creep endpoint.
GRASP = (1.2655, -1.2755, 2.0905, -0.8149, 1.2621, 0.0)


def _host():
    return json.load(open(HOST))


def _robot_up(tool_link='ur_tool0'):
    from twin.urdf_probe import parse_urdf
    robot = parse_urdf(URDF)
    u = cl.carton_up_in_tool(cl.tool_rotation(robot, list(GRASP), tool_link))
    return robot, u


def _clear_limits(c):
    big = 2 * math.pi
    return {'ur_wrist_1_joint': (-big, float(c['carry_wrist1_cap'])),
            'ur_wrist_2_joint': (float(c['carry_wrist2_min']), big)}


# ----------------------------------------------------------------------------- config wiring

def test_level_path_defaults_on():
    cfg = cl.resolve_level_config(True)
    assert cfg.level_path is True and cfg.path_waypoints == 12


def test_level_path_can_be_disabled_and_count_set():
    cfg = cl.resolve_level_config({'level_path': False, 'path_waypoints': 20})
    assert cfg.level_path is False and cfg.path_waypoints == 20


def test_bad_path_waypoints_rejected():
    with pytest.raises(ValueError):
        cl.resolve_level_config({'path_waypoints': 1})


# ----------------------------------------------------------------------------- the planner

@need_urdf
def test_single_goal_path_tilts_but_leveled_waypoint_path_stays_under_bound():
    robot, u = _robot_up()
    c = _host()
    limits = _clear_limits(c)
    lift = [float(v) for v in c['carry_lift_shape']]
    leveled_goal = [cl.level_config(robot, lift, u, limits=limits, accept_deg=2.0)[j] for j in ARM_JOINTS]

    # BEFORE: a single joint goal -> the executor drives the straight joint segment grasp -> leveled goal.
    # Both ENDPOINTS are level, but the path between them tilts the carton well over the bound.
    _, _, single_goal_tilt = cl.predict_path_tilt(robot, [list(GRASP), leveled_goal], u, interior=40)
    assert single_goal_tilt > 5.0, single_goal_tilt

    # AFTER: the leveled waypoint path keeps the predicted executed tilt under the 2 deg bound.
    plan = cl.plan_level_path(robot, list(GRASP), lift, u, limits=limits)
    assert plan.reject(2.0) is None
    assert max(plan.max_roll, plan.max_pitch) <= 2.0
    assert len(plan.waypoints) >= 3
    # first waypoint is the measured start, last is level
    assert max(abs(a - b) for a, b in zip(plan.waypoints[0], GRASP)) < 1e-6
    assert cl.carried_rpy(robot, list(plan.waypoints[-1]), u)[2] <= 0.2
    # every waypoint stays in the clear carry wrist region (w1<=cap, w2>=min)
    for w in plan.waypoints:
        assert w[3] <= float(c['carry_wrist1_cap']) + 1e-6
        assert w[4] >= float(c['carry_wrist2_min']) - 1e-6


@need_urdf
def test_pan_carry_path_is_level():
    robot, u = _robot_up()
    c = _host()
    limits = _clear_limits(c)
    lift = [float(v) for v in c['carry_lift_shape']]
    start = [cl.level_config(robot, lift, u, limits=limits, accept_deg=2.0)[j] for j in ARM_JOINTS]
    pan = [float(c['cart_pan_joint0'])] + [float(v) for v in c['cart_tuck_shape']]
    plan = cl.plan_level_path(robot, start, pan, u, limits=limits)
    assert plan.reject(2.0) is None and max(plan.max_roll, plan.max_pitch) <= 2.0


@need_urdf
def test_plan_reject_names_the_token_when_over_bound():
    robot, u = _robot_up()
    c = _host()
    limits = _clear_limits(c)
    lift = [float(v) for v in c['carry_lift_shape']]
    plan = cl.plan_level_path(robot, list(GRASP), lift, u, limits=limits)
    # The real lift is UNDER the 2 deg bound; a far tighter bound forces a rejection by name.
    msg = plan.reject(0.5)
    assert msg is not None and msg.startswith(cl.CARTON_LEVEL_PLAN_REJECTED)
    assert 'refusing to execute' in msg


@need_urdf
def test_ungoalable_goal_raises():
    # A goal with the carton inverted and the clear-region limits that admit no level solution raises
    # (the caller then falls back / fails honestly -- never a silent tilt).
    robot, u = _robot_up()
    limits = {'ur_wrist_1_joint': (-0.65, 0.65), 'ur_wrist_2_joint': (0.0, 0.01),
              'ur_wrist_3_joint': (0.0, 0.01)}
    with pytest.raises(ValueError):
        cl.plan_level_path(robot, list(GRASP), list(GRASP), u, limits=limits,
                           joints=('ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint'),
                           bound_deg=0.05)


# ----------------------------------------------------------------------------- controller wiring

def _collect_ctrl(**kw):
    c, m = make_controller('collect', carry_wrist1_cap=0.65, carry_wrist2_min=0.0, **kw)
    return c, m


@need_urdf
def test_lift_dispatches_a_leveled_waypoint_path():
    shape = [float(v) for v in _host()['carry_lift_shape']]
    c, m = _collect_ctrl(carry_lift_shape=shape, carton_level=cl.resolve_level_config(True), fk_urdf=URDF)
    t = run_to(c, 'lift')
    assert c.phase == 'lift'
    c.step(.1, packet(c, t, contact=True, joint_positions=GRASP))
    assert getattr(m, 'joint_paths', []), 'lift must dispatch a multi-waypoint begin_joint_path'
    path = m.joint_paths[-1]
    assert len(path['waypoints']) >= 3                       # a decomposed path, not a single goal
    assert max(abs(a - b) for a, b in zip(path['waypoints'][0], GRASP)) < 1e-6

    # the dispatched path keeps the carton level throughout (replay)
    robot, u = _robot_up()
    _, _, tilt = cl.predict_path_tilt(robot, [list(w) for w in path['waypoints']], u, interior=20)
    assert tilt <= 2.0, tilt


@need_urdf
def test_level_path_off_uses_single_goal():
    shape = [float(v) for v in _host()['carry_lift_shape']]
    c, m = _collect_ctrl(carry_lift_shape=shape,
                         carton_level=cl.resolve_level_config({'level_path': False}), fk_urdf=URDF)
    t = run_to(c, 'lift')
    c.step(.1, packet(c, t, contact=True, joint_positions=GRASP))
    assert not getattr(m, 'joint_paths', []), 'level_path off must NOT use the waypoint path'
    assert m.stows, 'lift must still dispatch a single leveled-endpoint begin_joint_goal'


def test_carton_level_off_is_byte_identical_single_goal():
    # No carton_level -> the lift is the plain fixed-shape begin_joint_goal (byte-identical to pre-fix).
    shape = [1.2675, -2.27, 1.0, 0.65, 0.8, -1.2545]
    c, m = _collect_ctrl(carry_lift_shape=shape)
    t = run_to(c, 'lift')
    c.step(.1, packet(c, t, contact=True))
    assert not getattr(m, 'joint_paths', [])
    assert any(list(g[0]) == shape for g in m.stows), m.stows


@need_urdf
def test_lift_rejects_when_path_would_tilt_over_bound():
    # A tiny bound the real lift's irreducible wrist-branch transition cannot meet -> the move is
    # CARTON_LEVEL_PLAN_REJECTED before any motion (the controller fails by name, never executes a tilt).
    shape = [float(v) for v in _host()['carry_lift_shape']]
    c, m = _collect_ctrl(carry_lift_shape=shape,
                         carton_level=cl.resolve_level_config({'bound_deg': 0.5}), fk_urdf=URDF)
    t = run_to(c, 'lift')
    r = c.step(.1, packet(c, t, contact=True, joint_positions=GRASP))
    assert r.status == 'failed'
    assert r.reason.startswith(cl.CARTON_LEVEL_PLAN_REJECTED)
    assert not getattr(m, 'joint_paths', []), 'a rejected plan must not dispatch any motion'
