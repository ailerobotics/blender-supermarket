"""CPU-only: the leveled WAYPOINT-PATH carry/tuck -> place-on-cart standoff APPROACH
(sim-carton-level-preplace-v1).

PR 290 leveled the place-on-cart standoff ENDPOINT and PR 296 the lift/carry, but the carried move INTO
that standoff was still a single begin_joint_goal: both endpoints level, yet the joint-space path between
them passes through a pitched configuration (collectlevel1: FK-reconstructed ~13 deg mid-path; pitch peak
2.25 deg tripped CARTON_TILT at the pre-place transition). This proves the fix -- the approach is driven
as a leveled waypoint path to the SAME PR 290 standoff goal (pinned verbatim, interior wrists re-solved
level) so the carton stays parallel to the ground THROUGHOUT -- on the REAL mission kinematics, and that a
plan still over the bound is CARTON_LEVEL_PLAN_REJECTED before any motion streams.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest scripts/twin/tests/test_carton_level_preplace_path.py
"""
import json
import math
import os

import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.tests.test_restock_manipulation import make_controller, packet

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin16/mobile-import.urdf'
HOST = os.path.join(os.path.dirname(__file__), '..', 'restock_manipulation', 'ros_host_config.json')
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='mission URDF not present')

# The REAL collectlevel1 collect carry (ARM order), reconstructed from the saved run's host-stack
# TRAJ_HOST_PLAN dense trajectories: the grasp fixes the carton up-axis at the weld, the preplace START is
# the pan/tuck carry pose, the preplace GOAL is the PR 290 full-DOF leveled place-on-cart standoff (the
# executed endpoint). Both ENDPOINTS are level; the single begin_joint_goal path between them tilted.
GRASP = (1.1923, -1.2833, 2.1071, -0.8249, 1.1932, -0.0001)
PREPLACE_START = (1.2672, -2.27, 1.0, 0.65, 0.0017, 0.6187)
PREPLACE_GOAL = (0.093, -1.9103, 2.0154, -0.1061, 1.664, -0.0006)
_SL_STOW = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
_SL_TUCK = (-2.3, 1.0, 2.1066, 0.8, -1.2545)


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


# ----------------------------------------------------------------------------- the planner

@need_urdf
def test_single_goal_approach_tilts_but_leveled_waypoint_path_stays_under_bound():
    robot, u = _robot_up()
    limits = _clear_limits(_host())
    # BEFORE: the single begin_joint_goal drives the straight joint segment start -> leveled standoff.
    # Both ENDPOINTS are level, but the path between them tilts the carton well over the bound.
    assert cl.carried_rpy(robot, list(PREPLACE_START), u)[2] <= 0.2     # start level
    assert cl.carried_rpy(robot, list(PREPLACE_GOAL), u)[2] <= 0.2      # PR 290 standoff level
    _, _, single_tilt = cl.predict_path_tilt(robot, [list(PREPLACE_START), list(PREPLACE_GOAL)], u,
                                              interior=40)
    assert single_tilt > 5.0, single_tilt

    # AFTER: the leveled waypoint path to the SAME standoff keeps the predicted executed tilt under bound.
    plan = cl.plan_level_path_to_goal(robot, list(PREPLACE_START), list(PREPLACE_GOAL), u, limits=limits)
    assert plan.reject(2.0) is None
    assert max(plan.max_roll, plan.max_pitch) <= 2.0
    assert plan.max_pitch < 1.0                                        # brief target: < 1 deg pitch
    assert len(plan.waypoints) >= 3                                    # a decomposed path, not a single goal


@need_urdf
def test_plan_to_goal_pins_endpoints_and_does_not_relevel_the_standoff():
    robot, u = _robot_up()
    limits = _clear_limits(_host())
    plan = cl.plan_level_path_to_goal(robot, list(PREPLACE_START), list(PREPLACE_GOAL), u, limits=limits)
    # the FIRST waypoint is the measured start (begin_joint_path requires it) ...
    assert max(abs(a - b) for a, b in zip(plan.waypoints[0], PREPLACE_START)) < 1e-9
    # ... and the LAST waypoint is the PR 290 standoff VERBATIM (not a wrist-re-leveled branch).
    assert max(abs(a - b) for a, b in zip(plan.waypoints[-1], PREPLACE_GOAL)) < 1e-9
    # every waypoint stays in the clear carry wrist region (w1<=cap, w2>=min)
    c = _host()
    for w in plan.waypoints:
        assert w[3] <= float(c['carry_wrist1_cap']) + 1e-6
        assert w[4] >= float(c['carry_wrist2_min']) - 1e-6


@need_urdf
def test_plan_to_goal_does_not_relevel_whereas_plan_level_path_would():
    # plan_level_path re-levels its goal wrist-only, which for the PR 290 full-DOF standoff picks a DIFFERENT
    # wrist branch (moving the standoff off its reach-validated TCP); plan_level_path_to_goal must NOT.
    robot, u = _robot_up()
    limits = _clear_limits(_host())
    relevel = cl.plan_level_path(robot, list(PREPLACE_START), list(PREPLACE_GOAL), u, limits=limits)
    to_goal = cl.plan_level_path_to_goal(robot, list(PREPLACE_START), list(PREPLACE_GOAL), u, limits=limits)
    # the re-leveling variant drifts the standoff wrist cluster; the to_goal variant preserves it.
    assert max(abs(a - b) for a, b in zip(relevel.waypoints[-1], PREPLACE_GOAL)) > 0.5
    assert max(abs(a - b) for a, b in zip(to_goal.waypoints[-1], PREPLACE_GOAL)) < 1e-9


@need_urdf
def test_plan_reject_names_the_token_when_over_bound():
    robot, u = _robot_up()
    limits = _clear_limits(_host())
    plan = cl.plan_level_path_to_goal(robot, list(PREPLACE_START), list(PREPLACE_GOAL), u, limits=limits)
    # The real approach is UNDER the 2 deg bound; a far tighter bound forces a rejection by name.
    msg = plan.reject(0.1)
    assert msg is not None and msg.startswith(cl.CARTON_LEVEL_PLAN_REJECTED)
    assert 'refusing to execute' in msg


@need_urdf
def test_bench_preplace_approach_single_tilts_waypoint_path_level():
    from twin.tools import carton_level_bench as b
    _, r = b.run_preplace_approach(URDF)
    assert r.name == 'preplace_approach'
    assert r.single_goal_max_tilt > 5.0          # the single-goal path tilts the carton off level mid-path
    assert r.path_max_pitch < 1.0                # the leveled waypoint path holds pitch under 1 deg
    assert r.passes(cl.DEFAULT_TILT_BOUND_DEG)   # ... and under the bound on both axes


# ----------------------------------------------------------------------------- controller wiring

def _collect_place_controller(level, straight_line=True):
    kw = dict(transport_joints=_SL_STOW, staged_cart_swing=True, cart_pan_joint0=-0.8605,
              cart_tuck_shape=_SL_TUCK, reposition_before_place=True,
              carry_lift_shape=[float(v) for v in _host()['carry_lift_shape']],
              collect_place_preplace_shape=PREPLACE_GOAL, collect_place_wrist_guard=True,
              collect_place_straight_line=straight_line)
    if level is not None:
        kw['carton_level'] = level
        kw['fk_urdf'] = URDF
    return make_controller('collect', **kw)


def _drive_to_verify(c, m):
    """Drive the collect reposition place-on-cart to verify feeding the real grasp as the measured joints
    (so the carton-level up-axis is captured from the top-down grasp). Returns the preplace dispatch kind:
    the begin_joint_path recorded for the preplace phase (level path) or None (single begin_joint_goal)."""
    t0 = 1.0
    preplace_path = None
    njp = 0
    for i in range(160):
        if c.phase == 'verify':
            break
        if c.phase == 'transit':
            c.base_repositioned = True
        contact = c.held or (c.phase == 'grasp' and c.issued)
        if c.phase == 'release' and c.issued:
            contact = False
        jp = _SL_STOW if c.phase == 'stow' else GRASP
        pb = c.phase
        r = c.step(.1, packet(c, t0 + i * .1, contact=contact, joint_positions=jp))
        assert r.status != 'failed', (pb, r.reason)
        paths = getattr(m, 'joint_paths', [])
        if pb == 'preplace' and len(paths) > njp and preplace_path is None:
            preplace_path = paths[-1]
        njp = len(paths)
    assert c.phase == 'verify', c.phase
    return preplace_path


@need_urdf
def test_preplace_dispatches_a_leveled_waypoint_path_to_the_standoff():
    c, m = _collect_place_controller(level=cl.resolve_level_config(True))
    path = _drive_to_verify(c, m)
    assert path is not None, 'preplace must dispatch a multi-waypoint begin_joint_path'
    assert len(path['waypoints']) >= 3                                 # decomposed, not a single goal
    assert max(abs(a - b) for a, b in zip(path['waypoints'][0], GRASP)) < 1e-6   # starts at the arm
    # the last waypoint is the PR 290 leveled standoff (preserved); the dispatched path keeps level
    robot, u = _robot_up()
    _, _, tilt = cl.predict_path_tilt(robot, [list(w) for w in path['waypoints']], u, interior=20)
    assert tilt <= 2.0, tilt
    # the preplace begin_joint_goal (single-goal) path was NOT used for preplace
    assert not any(jg.get('goal') == PREPLACE_GOAL and not jg.get('straight_line')
                   for jg in getattr(m, 'joint_goals', []))


@need_urdf
def test_level_path_off_uses_single_begin_joint_goal_for_preplace():
    c, m = _collect_place_controller(level=cl.resolve_level_config({'level_path': False}))
    path = _drive_to_verify(c, m)
    assert path is None, 'level_path off must NOT dispatch a preplace waypoint path'
    # preplace falls back to a single leveled-endpoint begin_joint_goal (level_place still levels the goal)
    robot, u = _robot_up()
    assert any(not jg['straight_line'] and cl.carried_rpy(robot, list(jg['goal']), u)[2] <= 0.2
               for jg in getattr(m, 'joint_goals', []))


def test_carton_level_off_preplace_is_byte_identical_single_goal():
    # No carton_level -> preplace is the plain fixed-shape begin_joint_goal to the raw standoff shape
    # (byte-identical to pre-fix): no waypoint path, and the raw PREPLACE_GOAL is the dispatched goal.
    c, m = _collect_place_controller(level=None)
    path = _drive_to_verify(c, m)
    assert path is None
    assert any(g == PREPLACE_GOAL for g, _measured, _slow in m.stows), m.stows
