"""CPU proof for the collect pre-place LEG-2 wrist-divergence fix (sim-preplace-wrist-continuity-v1).

coldwin57 reached 97 % of the leg-2 pan descent with the carton at 0.00 deg world tilt, then e-stopped on
an ARM TRACKING FAULT (TRAJ_HOST_TRACK_MISS worst_j=4 ref 1.63 vs meas 0.33). The forensics (coldwin57-
jump.txt / -onset.txt) corrected two premises:

  * the leg-2 PLAN never jumped -- TRAJ_HOST_PLAN HELD wrist_2 = 1.6301 across all 18 waypoints; and
  * there is NO contact and NO second code writer -- the installed plan is the single source of truth
    (drive target == monitor reference). The executed wrist_2 RELAXED from the commanded HIGH level branch
    (1.63) to the LOW branch (~0.33) because wrist_2 is a NEAR-NULL DOF for the carton attitude AND gravity
    (sweeping it 1.63->0.33 changes world tilt <0.1 deg and the holding torque stays <0.02 N*m), so the
    soft wrist drive cannot distinguish/hold the high branch. Both branches are world-level, so the carton
    stayed level -- the fault is purely the plan-vs-executed wrist_2 divergence.

The fix: COMMAND the branch the drive settles on -- the reach+level standoff branch with the MINIMUM
|wrist_2| (``select_reaction_natural_standoff``) -- so leg 1 drives to it and leg 2 holds it with no
divergence (plan == drive == monitor all target ~0.32). The gravity/geometry holding-torque model is
computed and REPORTED (margin vs the 56 N*m URDF wrist effort) but is NOT the selector: the torque is
~0.5-1.0 N*m on every branch (far under 56), and minimising it would pick the FARTHEST / uphill branch. A
branch-continuous per-waypoint re-solve + a per-waypoint wrist-step guard are kept as cheap safety.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_preplace_wrist_continuity_v1.py
"""
import math
import os

import numpy as np
import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import config_preflight as cp
from twin.restock_manipulation.cad_tool import ARM_JOINTS

# Prefer the coldwin57 run URDF (where the low-|w2| and min-torque branches genuinely differ -- the honest
# finding), else the committed-test mobile URDF. Skip when neither is present.
_URDF_CANDIDATES = (
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin57/mobile-import.urdf',
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf',
)
URDF = next((p for p in _URDF_CANDIDATES if os.path.exists(p)), _URDF_CANDIDATES[0])
need_urdf = pytest.mark.skipif(not any(os.path.exists(p) for p in _URDF_CANDIDATES),
                               reason='no mobile URDF present')
TL = 'ur_tool0'
PITCH_DEG = -1.17   # RUNTIME chassis_station_pitch_hold_pitch_deg (the held chassis attitude)

# coldwin57 leg-2 geometry (ur_base_link / ARM order), read from the host-stack TRAJ_HOST_PLAN leg-2 plan
# (18 pts): the leg-2 START (high-branch standoff at the carry pan, world-level) and the leg-2 END (the
# standoff re-levelled at the standoff pan). Both are world-level (CARTON_TICK tilt 0.00 at both ends).
STANDOFF_HI = [1.2675, -1.5477, 1.5431, 0.0111, 1.6302, -0.0013]   # leg-2 start (== the standoff shape)
LEG2_START = STANDOFF_HI
LEG2_END = [-0.079, -1.5477, 1.543, 0.0242, 1.6302, -0.0224]       # leg-2 end (re-levelled standoff)
CARRY = [1.2675, -2.2701, 1.0002, 0.6501, 0.0017, 0.6189]


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
    return cl.base_rotation_rp(0.0, PITCH_DEG)


@pytest.fixture(scope='module')
def up_world(robot, base_R):
    # The carton's up-axis in the tool frame, fixed so base_R @ tool_R(standoff) @ up == world +Z.
    R = cl.tool_rotation(robot, STANDOFF_HI, TL)
    return cl._unit(R.T @ (np.asarray(base_R).T @ np.array([0.0, 0.0, 1.0])))


@pytest.fixture(scope='module')
def standoff_target(robot):
    return cl.tool_position(robot, STANDOFF_HI, TL)


# ----------------------------------------------------------------- branch selection (the primary fix)

@need_urdf
def test_reaction_branch_selects_min_abs_w2(robot, up_world, standoff_target, base_R):
    std, rep = cl.select_reaction_natural_standoff(
        robot, standoff_target, up_world, STANDOFF_HI, CARRY, tool_link=TL,
        limits=_limits(), base_R=base_R, effort_limit=56.0)
    # The chosen standoff is the branch with the smallest |wrist_2| across the enumerated level branches.
    chosen_w2 = abs(cl._wrap(std['ur_wrist_2_joint']))
    assert chosen_w2 == min(abs(cl._wrap(b['wrist_2'])) for b in rep['branches'])
    # It is the LOW branch (settling point ~0.32), decisively below the commanded HIGH branch (1.63).
    assert chosen_w2 < 0.6
    assert rep['chosen_wrist_2'] == std['ur_wrist_2_joint']
    # The chosen standoff still reaches the target and is world-level.
    assert np.max(np.abs(cl.tool_position(robot, [std[j] for j in ARM_JOINTS], TL) - standoff_target)) < 2e-3
    _, _, tilt = cl.carried_rpy(robot, [std[j] for j in ARM_JOINTS], up_world, TL, base_R=base_R)
    assert tilt <= 0.5


@need_urdf
def test_holding_torque_reported_and_far_below_56(robot, up_world, standoff_target, base_R):
    _, rep = cl.select_reaction_natural_standoff(
        robot, standoff_target, up_world, STANDOFF_HI, CARRY, tool_link=TL,
        limits=_limits(), base_R=base_R, effort_limit=56.0)
    # Honest finding: the peak soft-wrist holding torque is < 1.5 N*m on EVERY branch -- far below the
    # 56 N*m URDF wrist effort ceiling, so no branch saturates and the torque is not the selector.
    assert all(b['peak_torque_nm'] < 1.5 for b in rep['branches'])
    assert rep['effort_limit_nm'] == 56.0
    assert rep['chosen_margin_nm'] > 50.0   # margin to the 56 N*m ceiling is huge
    # The report names which branch minimum-torque WOULD pick (the honest comparison).
    assert 'min_torque_wrist_2' in rep


# ----------------------------------------------------------------- branch continuity (cheap safety guard)

@need_urdf
def test_leg2_plan_is_branch_continuous(robot, up_world, base_R):
    # Plan the real leg-2 pan sweep (start -> re-levelled standoff end) on the high branch, at the real
    # densification (preplace_max_segment_rad ~0.08 -> n~17, matching the coldwin57 gate).
    plan = cl.plan_level_world_approach(
        robot, dict(zip(ARM_JOINTS, LEG2_START)), dict(zip(ARM_JOINTS, LEG2_END)), up_world,
        limits=_limits(), base_R=base_R, max_segment_rad=0.08)
    # The plan HOLDS the branch: every per-waypoint wrist step is far under the 0.15 rad threshold, and the
    # carton stays level (<= 0.5 deg) across the whole sweep -- coldwin57's PLAN was already continuous.
    assert plan.max_wrist_step < 0.15
    assert plan.reject_wrist_step(0.15) is None
    assert plan.max_tilt <= 0.5
    assert all(abs(cl._wrap(w[4] - 1.6302)) < 0.05 for w in plan.waypoints)   # wrist_2 held on the branch


@need_urdf
def test_continuous_solve_prefers_near_branch_over_lowest_tilt(robot, up_world, base_R):
    # At a leg-2 waypoint (pan mid-sweep) seeded on the HIGH branch, level_config ranks by lowest-tilt then
    # nearest the GOAL (it can prefer a far branch); continuous_level_solve picks the MINIMUM-travel branch
    # from the previous waypoint, so it stays on the near (high) branch with a tiny step.
    base = dict(zip(ARM_JOINTS, STANDOFF_HI))
    base['ur_shoulder_pan_joint'] = 0.4
    prev = dict(zip(ARM_JOINTS, STANDOFF_HI))   # previous waypoint on the high branch (w2~1.63)
    sol, tilt, step = cl.continuous_level_solve(
        robot, base, up_world, prev, limits=_limits(), accept_deg=2.0, base_R=base_R)
    assert tilt <= 0.5
    assert step < 0.15                                   # minimal travel from the previous waypoint
    assert abs(cl._wrap(sol['ur_wrist_2_joint'] - prev['ur_wrist_2_joint'])) < 0.15   # stayed on the branch


def test_reject_wrist_step_message():
    # A plan whose max per-waypoint wrist step exceeds the threshold is refused by name; under it, None.
    p = cl.LevelPathPlan(waypoints=((0,)*6, (0,)*6), max_roll=0.0, max_pitch=0.0, max_tilt=0.0,
                         waypoints_n=1, max_wrist_step=0.9, worst_step_wp=1)
    msg = p.reject_wrist_step(0.15)
    assert msg is not None and cl.CARTON_LEVEL_PLAN_REJECTED in msg and '0.900 rad' in msg
    assert p.reject_wrist_step(1.0) is None


# ----------------------------------------------------------------- config + preflight

def test_resolve_config_new_keys_and_effort_default():
    c = cl.resolve_level_config({'preplace_level_gate': True, 'preplace_reaction_branch': True,
                                 'preplace_max_wrist_step_rad': 0.2, 'preplace_soft_effort_nm': 56})
    assert c.preplace_reaction_branch is True
    assert c.preplace_max_wrist_step_rad == 0.2
    assert c.preplace_soft_effort_nm == 56.0
    # Default soft-wrist effort ceiling is the 56 N*m URDF value (not the old 400).
    assert cl.resolve_level_config(True).preplace_soft_effort_nm == 56.0
    with pytest.raises(ValueError):
        cl.resolve_level_config({'preplace_max_wrist_step_rad': 0.0})
    with pytest.raises(ValueError):
        cl.resolve_level_config({'preplace_soft_effort_nm': -1})


def test_preflight_reports_continuity(capsys):
    hc = {'preplace_wrist_first': True, 'carton_level': {'preplace_level_gate': True,
          'preplace_reaction_branch': True}, 'carry_lift_shape': list(CARRY),
          'collect_place_preplace_shape': list(STANDOFF_HI)}
    problems = cp.check_preplace_phase_budget(hc)
    assert problems == []
    out = capsys.readouterr().out
    assert 'PREPLACE_WRIST_CONTINUITY' in out and 'reaction_branch=ON' in out
    # A malformed continuity threshold is refused at resolve time (loud at boot).
    bad = dict(hc); bad['carton_level'] = {'preplace_max_wrist_step_rad': -1}
    assert any('carton_level' in p for p in cp.check_preplace_phase_budget(bad))


# -------------------------------------------- one source of truth: installed plan == monitored plan

@need_urdf
def test_installed_plan_equals_monitored_plan_sample_by_sample(robot, up_world, base_R):
    """The leg-2 waypoints the controller installs via begin_joint_path are the SAME object the trajectory
    host monitors: _waypoints_goal copies the positions VERBATIM, and trajectory_host.Plan.sample(t)
    reconstructs them at the knots. So the drive reference (the executor's `desired`) and the monitor
    reference are identical sample by sample -- there is no second writer. Skipped where ROS msgs are
    unavailable (the plain CPU env); runs in a ROS-sourced CI."""
    try:
        from twin.restock.trajectory_host import Plan
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
    except Exception as exc:
        pytest.skip('ROS trajectory_msgs unavailable: %s' % exc)
    wps = [list(w) for w in cl.plan_level_world_approach(
        robot, dict(zip(ARM_JOINTS, LEG2_START)), dict(zip(ARM_JOINTS, LEG2_END)), up_world,
        limits=_limits(), base_R=base_R, max_segment_rad=0.08).waypoints]
    # Build the installed trajectory exactly as ros_adapter._waypoints_goal does (positions verbatim).
    jt = JointTrajectory()
    jt.joint_names = list(ARM_JOINTS)
    t = 0.0
    knots = []
    for i, q in enumerate(wps):
        p = JointTrajectoryPoint()
        p.positions = list(q)
        p.velocities = [0.0] * 6
        if i > 0:
            dmax = max((abs(b - a) for a, b in zip(wps[i - 1], q)), default=0.0)
            t += max(1.0, 1.5 * dmax / 0.08, math.sqrt(6 * dmax / 0.15))
        knots.append(t)
        p.time_from_start = Duration(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))
        jt.points.append(p)
    plan = Plan(jt)
    # The monitor reference (Plan.sample) reconstructs each installed waypoint EXACTLY at its knot time.
    for tk, w in zip(knots, wps):
        assert np.max(np.abs(np.asarray(plan.sample(tk), float) - np.asarray(w, float))) < 1e-6
