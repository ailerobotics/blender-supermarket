"""CPU proof for the branch-SEEDED interior approach solve (sim-preplace-approach-seeded-v1, coldwin62).

coldwin62 was the 14th GPU run to die at the collect pre-place: the reaction-branch SELECT succeeded
(chosen wrist_2 = 0.663, 2.0 s) but ``plan_level_world_approach`` then produced a 0.433 rad wrist_2 step at
waypoint 5/10 of the carried approach and ``CARTON_LEVEL_PLAN_REJECTED`` refused it before any motion. The
legacy per-waypoint leveler solves each interior waypoint INDEPENDENTLY (``continuous_level_solve`` ranks the
min-travel branch, but the free Gauss-Newton descent from the position geometry still falls into a FAR level
basin), so wrist_2 DIPS onto a lower branch at the interior (~0.19 at wp4) and SNAPS back up toward the
standoff (0.62 at wp5, +0.433). PR 348's ``anchor='auto'`` only chooses between two ENDPOINT-anchored plans;
the interior branch dip survives in whichever it keeps.

This reconstructs the coldwin62 approach on the real coldwin62 URDF + logged joints (steering fixture
msg001; host-stack.log ln10873) and shows:

  1. the LEGACY ``anchor='auto'`` reproduces the logged dip -- wrist_2 ~0.03 -> 0.19 (wp4) -> 0.62 (wp5,
     +0.433 rad) -> 0.66 (standoff) -- and the continuity guard REFUSES it (max wrist step > 0.15);
  2. ``anchor='seeded'`` (the fix) sweeps wrist_2 MONOTONICALLY start -> standoff with every per-waypoint
     wrist step <= the 0.15 rad continuity threshold and the executed interior tilt <= 0.5 deg, and the guard
     ACCEPTS it;
  3. the local-densification-then-refuse path: a goal on a genuinely different wrist branch no densification
     bridges still leaves a step > the clamp, which the guard names and refuses (never a silent truncation);
  4. the whole seeded approach plan stays well under 2 s on the coldwin62 inputs (the Isaac host-status
     watchdog budget).

The logged ``carry_lift_shape`` wrist_2 = 0.80 is the UN-leveled carry config; the arm's actual measured
start of the approach is the LEVELED transit pose (CARTON_TICK phase=transit q=[1.2674,-2.2701,1.0002,0.6501,
0.0018,0.6189], tilt 0.03 -- wrist_2 ~0.00), so the physically-real monotonic sweep is ~0.03 -> 0.66. The
up-axis is captured at the weld (first carried/lift pose), exactly as ``controller._capture_level_up`` does.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_preplace_approach_seeded_v1.py
"""
import math
import os
import time

import numpy as np
import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS

_URDF_CANDIDATES = (
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin62/mobile-import.urdf',
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin60/mobile-import.urdf',
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf',
)
URDF = next((p for p in _URDF_CANDIDATES if os.path.exists(p)), _URDF_CANDIDATES[0])
need_urdf = pytest.mark.skipif(not any(os.path.exists(p) for p in _URDF_CANDIDATES),
                               reason='no mobile URDF present')
TL = 'ur_tool0'
PITCH_DEG = -1.17
LJ = cl.DEFAULT_LEVEL_JOINTS
# coldwin62 logged joints (steering fixture msg001 + host-stack.log ln10873). Joint order
# [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3].
WELD = [1.3158, -1.2900, 2.1088, -0.8199, 1.2941, 0.0]        # first carried/lift pose (tilt 0) -> up-axis
TRANSIT = [1.2674, -2.2701, 1.0002, 0.6501, 0.0018, 0.6189]   # measured approach start (phase=transit, level)
STANDOFF = [1.2677, -1.4848, 1.4779, -0.013, 0.6615, 0.0117]  # selected reaction-natural standoff (w2 0.661)
MAX_SEG = 0.08     # coldwin62 preplace_max_segment_rad (gives n=10, the logged path)
CLAMP = 0.15       # preplace_max_wrist_step_rad (the continuity threshold)
GATE = 1.0         # preplace_gate_bound_deg


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
    # the real measured start is the LEVELED transit pose (the arm carries the carton level)
    lev = cl.level_config(robot, list(TRANSIT), up, joints=LJ, tool_link=TL, limits=lim,
                          accept_deg=2.0, base_R=base_R, fast=True)
    start = [float(lev[j]) for j in ARM_JOINTS]
    return robot, base_R, up, lim, start


def _plan(ctx, anchor, goal=STANDOFF, seg=MAX_SEG):
    robot, base_R, up, lim, start = ctx
    kw = dict(tool_link=TL, limits=lim, bound_deg=2.0, max_segment_rad=seg, level_joints=LJ,
              base_R=base_R, anchor=anchor, fast=True)
    if anchor == 'seeded':
        kw['level_accept_deg'] = GATE
    return cl.plan_level_world_approach(robot, list(start), list(goal), up, **kw)


def _w2(plan):
    return [round(float(w[4]), 3) for w in plan.waypoints]


def _steps(plan, ctx):
    idx = [ARM_JOINTS.index(j) for j in LJ]
    return [max(abs(plan.waypoints[i][k] - plan.waypoints[i - 1][k]) for k in idx)
            for i in range(1, len(plan.waypoints))]


@need_urdf
def test_legacy_auto_reproduces_the_logged_dip(ctx):
    """The legacy ``anchor='auto'`` reproduces the coldwin62 interior branch dip: wrist_2 dips to ~0.19
    at wp4 then JUMPS +0.433 to ~0.62 at wp5, exceeding the 0.15 rad continuity threshold."""
    p = _plan(ctx, 'auto')
    assert p.waypoints_n == 10                               # n=10, the logged path
    w2 = _w2(p)
    assert w2[4] == pytest.approx(0.191, abs=0.02)           # wp4 DIP (logged 0.1899)
    assert w2[5] == pytest.approx(0.624, abs=0.02)           # wp5 JUMP (logged 0.6229)
    assert w2[-1] == pytest.approx(0.661, abs=0.01)          # standoff
    assert p.max_wrist_step == pytest.approx(0.433, abs=0.02)
    assert p.worst_step_wp == 5
    # the continuity guard REFUSES it before motion, naming the jump
    msg = p.reject_wrist_step(CLAMP, level_joints=LJ)
    assert msg is not None and cl.CARTON_LEVEL_PLAN_REJECTED in msg
    assert 'joint=ur_wrist_2_joint' in msg


@need_urdf
def test_seeded_sweeps_wrist2_monotonically_and_is_accepted(ctx):
    """``anchor='seeded'`` sweeps wrist_2 MONOTONICALLY start -> standoff with no dip, every per-waypoint
    wrist step <= 0.15 rad and the executed interior tilt <= 0.5 deg, and the continuity guard ACCEPTS it."""
    robot, base_R, up, lim, start = ctx
    p = _plan(ctx, 'seeded')
    w2 = _w2(p)
    # no dip: wrist_2 never goes below the start value then back up -- it is non-decreasing to the standoff
    assert all(w2[i] >= w2[i - 1] - 1e-3 for i in range(1, len(w2))), w2
    assert w2[0] == pytest.approx(float(start[4]), abs=0.01)
    assert w2[-1] == pytest.approx(0.661, abs=0.01)          # ends exactly at the selected standoff
    assert tuple(p.waypoints[-1]) == pytest.approx(tuple(float(v) for v in STANDOFF), abs=1e-6)
    assert tuple(p.waypoints[0]) == pytest.approx(tuple(float(v) for v in start), abs=1e-6)
    # every per-waypoint wrist step is within the continuity threshold
    steps = _steps(p, ctx)
    assert max(steps) <= CLAMP + 1e-9, max(steps)
    # the executed interior path stays level well within the gate (both legs)
    mr, mp, mt = cl.predict_path_tilt_executed(robot, [list(w) for w in p.waypoints], up,
                                               tool_link=TL, base_R=base_R)
    assert max(mr, mp) <= 0.5
    # accepted by BOTH gate arms (wrist-step continuity AND the tilt bound)
    assert p.reject_wrist_step(CLAMP, level_joints=LJ) is None
    assert p.reject(2.0) is None


@need_urdf
def test_seeded_timing_under_2s(ctx):
    """The whole seeded approach plan is well under the 2 s Isaac host-status watchdog budget."""
    t0 = time.perf_counter()
    _plan(ctx, 'seeded')
    assert time.perf_counter() - t0 < 2.0


@need_urdf
def test_seeded_densifies_then_refuses_a_true_branch_gap(ctx):
    """A goal on a genuinely DIFFERENT wrist branch that no local densification bridges still leaves a
    per-waypoint step > the clamp on the returned plan, which the guard names and refuses -- the seeded
    solve never silently truncates or accepts a discontinuous path."""
    robot, base_R, up, lim, start = ctx
    # force an un-bridgeable gap: a goal whose wrist_3 is a full +2*pi branch away from the start, inside
    # the generous limits, so no continuous level path reaches it within 0.15 rad/step.
    far_goal = list(STANDOFF)
    far_goal[5] = float(STANDOFF[5]) + 2 * math.pi
    p = cl.plan_level_world_approach(robot, list(start), list(far_goal), up, tool_link=TL,
                                     limits=None, bound_deg=2.0, max_segment_rad=MAX_SEG, level_joints=LJ,
                                     base_R=base_R, anchor='seeded', fast=True, level_accept_deg=GATE)
    # the last waypoint is still pinned to the goal exactly (no silent truncation)
    assert tuple(p.waypoints[-1]) == pytest.approx(tuple(far_goal), abs=1e-6)
    # and the unbridged branch gap leaves a step over the clamp -> the guard refuses it by name
    assert p.max_wrist_step > CLAMP
    assert p.reject_wrist_step(CLAMP, level_joints=LJ) is not None


def test_seeded_pure_densification_bound():
    """The densification bounds are sane module constants (pure, no URDF)."""
    assert cl._SEEDED_MAX_DEPTH >= 4
    assert cl._SEEDED_MAX_WAYPOINTS >= 100


def test_resolve_enables_seeded_interior():
    """The HOST carton_level key resolves the flag; default off => byte-identical."""
    assert cl.resolve_level_config(True).preplace_seeded_interior is False
    cfg = cl.resolve_level_config({'preplace_seeded_interior': True})
    assert cfg.preplace_seeded_interior is True


def test_preflight_seeded_without_level_gate_refused():
    """``preplace_seeded_interior`` ON with ``preplace_level_gate`` OFF is silently inert -> refused by name."""
    from twin.restock_manipulation import config_preflight as cp
    probs = cp.check_carton_level({'carton_level': {'preplace_seeded_interior': True}})
    assert any('preplace_seeded_interior' in p and 'inert' in p for p in probs)


def test_preflight_seeded_with_level_gate_not_flagged():
    """With the gate on (and above-support, so the gate itself is coherent) the seeded key is not flagged."""
    from twin.restock_manipulation import config_preflight as cp
    probs = cp.check_carton_level({'carton_level': {'preplace_seeded_interior': True,
                                                    'preplace_level_gate': True,
                                                    'preplace_above_support': True}})
    assert not any('preplace_seeded_interior' in p for p in probs)
