"""CPU proof for the CARRY-CONNECTED pre-place standoff (sim-preplace-pan-first-seeded-leg2-v1, coldwin72).

coldwin72 -- the first run with every physical fix stacked (pick sealed, the pan swept the cage clear with
the base locked + low arm effort, the wrist held) -- was stopped before any motion by the carton-level
continuity gate in the re-ordered PAN-FIRST leg 2 (PR 366): the reaction-natural standoff (min |wrist_2| =
0.668) sits on a DIFFERENT carton-level basin than the carry/tucked shape (wrist_2 ~0), so the leg-2 wrist_2
holds ~0 for nine waypoints then JUMPS 0.300 rad in one step (> the 0.15 rad continuity limit) ->
CARTON_LEVEL_PLAN_REJECTED. No continuous monotonic <=0.15 level path connects the two basins at the
standoff pan (the wrist_1 self-collision cap chops the standoff branch off below t~0.5; CPU-proven the
crossing tilt ridge minimum is ~2.44 deg > the 2.0 bound), so the PR 349 seeding cannot help.

The fix (firstmate decision B): choose the standoff on the branch CONNECTED to the carry shape (wrist_2 ~
the carry wrist_2) so leg 2 is a monotonic extension the gate accepts. The PR 359 firm wrist hold now
covers the soft-wrist drift that pushed PR 345 to the 0.668 settling branch, so the ~0 branch is admissible.

This reconstructs the cw72 pre-place on the real cw72 URDF + logged joints (host-stack.log ln 18573-18618)
and shows:

  1. the reaction-natural (min-|wrist_2|) standoff is 0.668 and its pan-first leg 2 is REJECTED by the
     continuity gate -- wrist_2 holds ~0 then steps 0.300 rad (the logged failure);
  2. ``select_carry_connected_standoff`` instead returns a reach+level+self-clear standoff on the
     carry-connected branch (wrist_2 ~0.09) whose leg-2 the gate ACCEPTS (every wrist step <= 0.15 rad,
     tilt <= 2.0 deg);
  3. the PLACE DESCENT from that standoff stays level (<= 2.0 deg);
  4. the whole search stays well under the host-status watchdog budget;
  5. the HOST key resolves, default OFF is byte-identical, and the preflight refuses it without the
     reaction branch.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_preplace_pan_first_seeded_leg2_v1.py
"""
import math
import os
import time

import numpy as np
import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS

_URDF_CANDIDATES = (
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin72/mobile-import.urdf',
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin70/mobile-import.urdf',
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf',
)
URDF = next((p for p in _URDF_CANDIDATES if os.path.exists(p)), _URDF_CANDIDATES[0])
need_urdf = pytest.mark.skipif(not any(os.path.exists(p) for p in _URDF_CANDIDATES),
                               reason='no mobile URDF present')
TL = 'ur_tool0'
PITCH_DEG = -1.17
LJ = cl.DEFAULT_LEVEL_JOINTS
# cw72 logged joints (host-stack.log). Order [pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3].
WELD = [1.3158, -1.2899, 2.1087, -0.8199, 1.2941, 0.0]          # first carried/lift pose (tilt 0) -> up-axis
LEG2_START = [-0.4222, -2.2701, 1.0002, 0.6501, 0.0017, 0.6190]  # leg-1 end = pan-first leg-2 start
STD_0668 = [-0.4222, -1.4853, 1.4784, 0.0352, 0.6681, -0.0334]   # reaction-natural standoff (w2 0.668)
SEED = [0.093, -1.91, 2.014, -0.103, 1.664, 0.0]                 # collect_place_preplace_shape
MAX_SEG = 0.08
CLAMP = 0.15
GATE = 1.0
W2I = ARM_JOINTS.index('ur_wrist_2_joint')


def _limits():
    big = 2 * math.pi
    return {'ur_elbow_joint': (0.1, math.pi),
            'ur_wrist_1_joint': (-big, 0.65), 'ur_wrist_2_joint': (0.0, big)}


@pytest.fixture(scope='module')
def ctx():
    from twin.urdf_probe import parse_urdf
    robot = parse_urdf(URDF)
    base_R = cl.base_rotation_rp(0.0, PITCH_DEG)
    up = cl.carton_up_in_tool(np.asarray(base_R, float) @ cl.tool_rotation(robot, WELD, TL))
    lim = _limits()
    target = cl.tool_position(robot, STD_0668, TL)   # the place-above standoff target (TCP of the chosen 0.668)
    return robot, base_R, up, lim, target


def _leg2_plan(ctx, standoff, anchor='seeded'):
    robot, base_R, up, lim, _ = ctx
    kw = dict(tool_link=TL, limits=lim, bound_deg=2.0, max_segment_rad=MAX_SEG, level_joints=LJ,
              base_R=base_R, anchor=anchor, fast=True, max_wrist_step_rad=CLAMP)
    if anchor == 'seeded':
        kw['level_accept_deg'] = GATE
    # leg-2 start: carry tucked wrists at the standoff's pan
    start = list(LEG2_START)
    start[0] = float(standoff[0])
    return cl.plan_level_world_approach(robot, start, list(standoff), up, **kw)


@need_urdf
def test_min_wrist2_standoff_leg2_is_rejected(ctx):
    """The reaction-natural (min-|wrist_2| = 0.668) standoff's pan-first leg 2 is REJECTED by the continuity
    gate: wrist_2 holds ~0 for the interior then steps 0.300 rad in one waypoint (the logged cw72 failure)."""
    p = _leg2_plan(ctx, STD_0668, anchor='seeded')
    w2 = [round(float(w[W2I]), 3) for w in p.waypoints]
    assert w2[0] == pytest.approx(0.0017, abs=0.01)
    # interior holds ~0 then jumps
    assert any(w2[i] == 0.0 for i in range(1, len(w2) // 2))
    assert p.max_wrist_step == pytest.approx(0.300, abs=0.03)
    msg = p.reject_wrist_step(CLAMP, level_joints=LJ)
    assert msg is not None and cl.CARTON_LEVEL_PLAN_REJECTED in msg
    assert 'joint=ur_wrist_2_joint' in msg


@need_urdf
def test_carry_connected_standoff_leg2_is_accepted(ctx):
    """``select_carry_connected_standoff`` returns a reach+level+self-clear standoff on the carry-connected
    branch (wrist_2 near the carry wrist_2, ~0.09), and ITS pan-first leg 2 the continuity gate ACCEPTS:
    every per-waypoint wrist step <= 0.15 rad and predicted tilt <= 2.0 deg."""
    robot, base_R, up, lim, target = ctx
    sol, rep = cl.select_carry_connected_standoff(
        robot, target, up, SEED, LEG2_START, tool_link=TL, limits=lim, base_R=base_R,
        level_joints=LJ, max_segment_rad=MAX_SEG, bound_deg=2.0, max_wrist_step_rad=CLAMP,
        seeded_interior=True, level_accept_deg=GATE, budget_s=8.0)
    assert rep['reason'] == 'carry-connected'
    # the chosen standoff is carry-connected: wrist_2 far below the 0.668 settling branch
    assert float(sol['ur_wrist_2_joint']) < 0.3
    assert abs(float(sol['ur_wrist_2_joint']) - LEG2_START[W2I]) < 0.3
    # the reported leg-2 gate metrics PASS
    assert rep['leg2_max_wrist_step'] <= CLAMP + 1e-9
    assert rep['leg2_max_tilt_deg'] <= 2.0
    # the standoff reaches the SAME place-above target as the 0.668 branch (reach preserved)
    tcp = cl.tool_position(robot, [float(sol[j]) for j in ARM_JOINTS], TL)
    assert float(np.linalg.norm(np.asarray(tcp) - np.asarray(target))) < 2e-3
    # and re-planning its leg 2 directly confirms the gate accepts (no reject)
    p = _leg2_plan(ctx, [float(sol[j]) for j in ARM_JOINTS], anchor='seeded')
    assert p.reject_wrist_step(CLAMP, level_joints=LJ) is None
    assert p.reject(2.0) is None
    # wrist_2 stays bounded (no 0.300 cross-branch jump)
    w2 = [float(w[W2I]) for w in p.waypoints]
    assert max(abs(w2[i] - w2[i - 1]) for i in range(1, len(w2))) <= CLAMP + 1e-9


@need_urdf
def test_carry_connected_place_descent_stays_level(ctx):
    """The place DESCENT from the carry-connected standoff (straight vertical to the support, re-leveled per
    waypoint) stays parallel to the ground (<= 2.0 deg) -- the standoff is reach+level so the descent starts
    and stays level."""
    robot, base_R, up, lim, target = ctx
    sol, _ = cl.select_carry_connected_standoff(
        robot, target, up, SEED, LEG2_START, tool_link=TL, limits=lim, base_R=base_R,
        level_joints=LJ, max_segment_rad=MAX_SEG, bound_deg=2.0, max_wrist_step_rad=CLAMP,
        seeded_interior=True, level_accept_deg=GATE, budget_s=8.0)
    std = [float(sol[j]) for j in ARM_JOINTS]
    z0 = float(target[2])
    z1 = z0 - 0.08   # the 0.08 m from-above descent onto the support top
    n = 4
    vecs = [std]
    seed = dict(sol)
    for i in range(1, n + 1):
        z = z0 + (z1 - z0) * i / n
        s = cl.level_place_ik(robot, (float(target[0]), float(target[1]), z), up, seed,
                              tool_link=TL, limits=lim, gravity_comp=None, base_R=base_R, fast=True)
        vecs.append([float(s[j]) for j in ARM_JOINTS])
        seed = s
    mr, mp, mt = cl.predict_path_tilt_executed(robot, vecs, up, tool_link=TL, base_R=base_R)
    assert max(mr, mp) <= 2.0


@need_urdf
def test_carry_connected_search_under_budget(ctx):
    """The carry-connected search stays well under a generous host-status watchdog budget."""
    robot, base_R, up, lim, target = ctx
    t0 = time.perf_counter()
    _, rep = cl.select_carry_connected_standoff(
        robot, target, up, SEED, LEG2_START, tool_link=TL, limits=lim, base_R=base_R,
        level_joints=LJ, max_segment_rad=MAX_SEG, bound_deg=2.0, max_wrist_step_rad=CLAMP,
        seeded_interior=True, level_accept_deg=GATE, budget_s=8.0)
    assert time.perf_counter() - t0 < 8.0
    assert rep['select_seconds'] < 8.0


@need_urdf
def test_carry_connected_budget_overrun_raises(ctx):
    """A starved search (tiny budget) raises BranchSelectBudgetExceeded so the caller refuses before motion
    rather than starving the heartbeat (same contract as the min-|wrist_2| selector)."""
    robot, base_R, up, lim, target = ctx
    with pytest.raises(cl.BranchSelectBudgetExceeded):
        cl.select_carry_connected_standoff(
            robot, target, up, SEED, LEG2_START, tool_link=TL, limits=lim, base_R=base_R,
            level_joints=LJ, max_segment_rad=MAX_SEG, bound_deg=2.0, max_wrist_step_rad=CLAMP,
            seeded_interior=True, level_accept_deg=GATE, budget_s=1e-6)


def test_resolve_enables_carry_connected():
    """The HOST carton_level key resolves the flag; default off => byte-identical."""
    assert cl.resolve_level_config(True).preplace_branch_prefer_carry_connected is False
    cfg = cl.resolve_level_config({'preplace_branch_prefer_carry_connected': True})
    assert cfg.preplace_branch_prefer_carry_connected is True


def test_preflight_carry_connected_without_reaction_branch_refused():
    """``preplace_branch_prefer_carry_connected`` ON with ``preplace_reaction_branch`` OFF is inert -> refused."""
    from twin.restock_manipulation import config_preflight as cp
    probs = cp.check_carton_level({'carton_level': {'preplace_branch_prefer_carry_connected': True}})
    assert any('preplace_branch_prefer_carry_connected' in p and 'inert' in p for p in probs)


def test_preflight_carry_connected_with_reaction_branch_not_flagged():
    """With the reaction branch + level gate + above-support on, the carry-connected key is not flagged."""
    from twin.restock_manipulation import config_preflight as cp
    probs = cp.check_carton_level({'carton_level': {
        'preplace_branch_prefer_carry_connected': True, 'preplace_reaction_branch': True,
        'preplace_level_gate': True, 'preplace_above_support': True, 'preplace_seeded_interior': True}})
    assert not any('preplace_branch_prefer_carry_connected' in p for p in probs)
