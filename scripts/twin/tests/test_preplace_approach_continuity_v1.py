"""CPU proof for the leg-1 APPROACH branch-continuity work (sim-preplace-select-live-cost-v1, coldwin61).

coldwin61 completed the reaction-branch SELECT (chosen wrist_2 = 0.666) but then
``plan_level_world_approach`` generated a 0.439 rad wrist step at the FINAL waypoint (10/10) of the leg-1
approach (carry -> standoff) and ``CARTON_LEVEL_PLAN_REJECTED`` refused it before motion -- the reaction-
branch fix was never exercised. The legacy FORWARD per-waypoint re-solve seeds from the START, so the
min-travel chain holds the carry-side branch and the terminal pin to the selected standoff jumps.

This reconstructs the leg-1 approach on the real coldwin60/61 URDF + inputs and shows:

  1. ``anchor='start'`` (the legacy forward generator, byte-identical to the prior default) reproduces a
     per-waypoint wrist jump at the FINAL waypoint above the 0.15 rad continuity threshold.
  2. The reject message now NAMES the jump: the before/after waypoint joint vectors, which wrist stepped,
     and the standoff (the coldwin61 instrumentation ask -- the joints were otherwise unrecoverable).
  3. ``anchor='goal'`` (solve the interior backward from the selected standoff) removes the jump from the
     FINAL waypoint -- the terminal pin is reached continuously.
  4. ``anchor='auto'`` (what the controller uses) is NEVER worse than the forward plan.

Honest CPU finding (documented for the lane): when the carry pose and the selected standoff are on
genuinely DIFFERENT wrist branches, the jump is a real discrete gap that neither anchor removes -- it only
moves WHICH waypoint crosses it (here, from the final waypoint to the first). The complete remedy is a
standoff branch reachable from the carry pose; the reaction-natural standoff is meant to be exactly that,
and the named reject message lets the next GPU run (coldwin62) confirm the real before/after joints.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_preplace_approach_continuity_v1.py
"""
import math
import os

import numpy as np
import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.restock_manipulation.preplace_sequence import wrist_first_subgoals

_URDF_CANDIDATES = (
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin60/mobile-import.urdf',
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin57/mobile-import.urdf',
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf',
)
URDF = next((p for p in _URDF_CANDIDATES if os.path.exists(p)), _URDF_CANDIDATES[0])
need_urdf = pytest.mark.skipif(not any(os.path.exists(p) for p in _URDF_CANDIDATES),
                               reason='no mobile URDF present')
TL = 'ur_tool0'
PITCH_DEG = -1.17
# coldwin60/61 reconstruction (coldwin60-select-cost.txt).
GRASP = [1.1923, -1.2833, 2.1071, -0.8249, 1.1932, -0.0001]
SEED = [0.093, -1.91, 2.014, -0.103, 1.664, 0.0]      # collect_place_preplace_shape
CARRY = [1.2675, -2.27, 1.0, 0.65, 0.8, -1.2545]       # carry_lift_shape
TARGET = (0.53, 0.45, 0.54)
LJ = cl.DEFAULT_LEVEL_JOINTS


def _limits():
    big = 2 * math.pi
    return {'ur_elbow_joint': (0.1, math.pi),
            'ur_wrist_1_joint': (-big, 0.65), 'ur_wrist_2_joint': (0.0, big)}


@pytest.fixture(scope='module')
def ctx():
    from twin.urdf_probe import parse_urdf
    robot = parse_urdf(URDF)
    base_R = cl.base_rotation_rp(0.0, PITCH_DEG)
    up = cl.carton_up_in_tool(np.asarray(base_R, float) @ cl.tool_rotation(robot, GRASP, TL))
    lim = _limits()

    def leveled(cfg):
        lev = cl.level_config(robot, list(cfg), up, joints=LJ, tool_link=TL, limits=lim,
                              accept_deg=2.0, base_R=base_R, fast=True)
        return tuple(float(lev[j]) for j in ARM_JOINTS)

    sol, _rep = cl.select_reaction_natural_standoff(
        robot, TARGET, up, SEED, CARRY, tool_link=TL, limits=lim, base_R=base_R,
        effort_limit=56.0, bound_deg=2.0, max_segment_rad=0.08, fast=True)
    standoff = tuple(float(sol[j]) for j in ARM_JOINTS)
    leg1, _ = wrist_first_subgoals(CARRY, standoff)
    start = leveled(CARRY)
    goal = leveled(leg1)
    return robot, base_R, up, lim, start, goal


def _plan(ctx, anchor):
    robot, base_R, up, lim, start, goal = ctx
    return cl.plan_level_world_approach(
        robot, list(start), list(goal), up, tool_link=TL, limits=lim, bound_deg=2.0,
        max_segment_rad=0.08, level_joints=LJ, base_R=base_R, anchor=anchor, fast=True)


@need_urdf
def test_forward_reproduces_final_waypoint_jump(ctx):
    """The legacy forward generator (coldwin61) jumps a wrist branch at the FINAL waypoint above 0.15 rad."""
    p = _plan(ctx, 'start')
    assert p.max_wrist_step > 0.15
    assert p.worst_step_wp == p.waypoints_n          # the jump is at the final (standoff) waypoint
    assert p.max_tilt <= 2.0                          # the path itself is level; only the branch steps


@need_urdf
def test_reject_message_names_the_jump(ctx):
    """The CARTON_LEVEL_PLAN_REJECTED message captures the before/after joints + the stepping wrist +
    the standoff (the coldwin61 instrumentation ask -- otherwise unrecoverable after the fact)."""
    p = _plan(ctx, 'start')
    msg = p.reject_wrist_step(0.15, level_joints=LJ)
    assert msg is not None and cl.CARTON_LEVEL_PLAN_REJECTED in msg
    assert 'joint=ur_wrist_' in msg
    assert ('wp%d=' % (p.waypoints_n - 1)) in msg and ('wp%d=' % p.waypoints_n) in msg
    assert 'standoff=' in msg


@need_urdf
def test_goal_anchor_removes_the_final_waypoint_jump(ctx):
    """Solving backward from the selected standoff reaches the terminal pin continuously -- the FINAL
    waypoint is no longer the branch jump (the specific coldwin61 symptom)."""
    pg = _plan(ctx, 'goal')
    # the terminal (standoff) pin is reached with a small step; the worst step is not at the final waypoint
    terminal_step = max(abs(pg.waypoints[-1][k] - pg.waypoints[-2][k])
                        for k, j in enumerate(ARM_JOINTS) if j in LJ)
    assert terminal_step < 0.15
    assert pg.worst_step_wp != pg.waypoints_n
    assert pg.max_tilt <= 2.0


@need_urdf
def test_auto_never_worse_than_forward(ctx):
    """``anchor='auto'`` (the controller's choice) returns the smaller-jump plan, so it never regresses
    the legacy forward behaviour and takes the goal-anchored plan when that removes the jump."""
    pf = _plan(ctx, 'start')
    pa = _plan(ctx, 'auto')
    assert pa.max_wrist_step <= pf.max_wrist_step + 1e-9


def test_reject_message_detail_pure():
    """The reject-message instrumentation runs without a URDF (pure LevelPathPlan)."""
    wps = ((0.0,) * 6, (0.0, 0.0, 0.0, 0.5, 0.0, 0.0), (0.0, 0.0, 0.0, 1.4, 0.0, 0.0))
    p = cl.LevelPathPlan(waypoints=wps, max_roll=0.0, max_pitch=0.0, max_tilt=0.0,
                         waypoints_n=2, max_wrist_step=0.9, worst_step_wp=2)
    msg = p.reject_wrist_step(0.15)
    assert cl.CARTON_LEVEL_PLAN_REJECTED in msg and '0.900 rad' in msg
    assert 'joint=ur_wrist_1_joint' in msg and 'wp1=' in msg and 'wp2=' in msg and 'standoff=' in msg
