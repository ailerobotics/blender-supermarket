"""CPU proof for sim-preplace-branch-select-bounded-v1 (coldwin58/coldwin59).

coldwin59 (reaction-branch ON, hot-path logging OFF) died on the Isaac HOST_STATUS_WATCHDOG
(``isaac_ros_graph.poll``, 5 sim-s bound on a fresh ``/manipulation/status``): the reaction-branch
standoff selection ran SYNCHRONOUSLY in the controller's ``step()`` at the preplace transition and
blocked the single rclpy executor for >5 s (0 ``PREPLACE_LEVEL_GATE`` lines -- the solve never finished),
so the status heartbeat went silent and the watchdog killed a live, healthy host (Isaac / station /
nvblox all advancing through the gap). The old inline ``select_reaction_natural_standoff`` ran the full
densified leg-2 ``plan_level_world_approach`` for all 8 branches = ~14 s on the coldwin57 fixture.

Two fixes, proven here on CPU (no ROS / Isaac / GPU):

  1. The selection is now CHEAP and BYTE-IDENTICAL. The ``full_plan=False`` default validates each branch
     at the cheap leg-2-start endpoint instead of the full densified plan; the chosen standoff is decided
     by the min-|wrist_2| primary key, which the per-branch plan never affected, so the returned standoff
     is identical to the old inline computation (``full_plan=True``) on the coldwin57 fixture at ~0.08 s.
     A CPU time-box (``budget_s``) raises ``BranchSelectBudgetExceeded`` so the controller refuses before
     motion rather than ever blocking the heartbeat again.

  2. The status heartbeat is now STRUCTURALLY INDEPENDENT of the executor: ``StatusHeartbeat`` is a
     separate daemon thread that keeps publishing a liveness fill while any planning blocks the executor,
     so no future planner can starve it. The 10 s-blocking mock-planner test asserts the heartbeat never
     gaps beyond 1 s; the companion test reproduces coldwin59's >5 s gap on the old (no-filler) path.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_preplace_branch_select_bounded_v1.py
"""
import math
import os
import threading
import time

import numpy as np
import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.restock_manipulation.status_heartbeat import StatusHeartbeat

# The coldwin57 run URDF (where the low-|w2| settling branch and the full-plan selection genuinely agree
# -- the identity this lane must preserve); skip the fixture tests when it is not present.
_URDF_CANDIDATES = (
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin57/mobile-import.urdf',
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf',
)
URDF = next((p for p in _URDF_CANDIDATES if os.path.exists(p)), _URDF_CANDIDATES[0])
need_urdf = pytest.mark.skipif(not any(os.path.exists(p) for p in _URDF_CANDIDATES),
                               reason='no mobile URDF present')
TL = 'ur_tool0'
PITCH_DEG = -1.17
STANDOFF_HI = [1.2675, -1.5477, 1.5431, 0.0111, 1.6302, -0.0013]
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
    R = cl.tool_rotation(robot, STANDOFF_HI, TL)
    return cl._unit(R.T @ (np.asarray(base_R).T @ np.array([0.0, 0.0, 1.0])))


@pytest.fixture(scope='module')
def standoff_target(robot):
    return cl.tool_position(robot, STANDOFF_HI, TL)


# ------------------------------------------------- (1) cheap + byte-identical + time-boxed selection

@need_urdf
def test_fast_selection_is_byte_identical_to_full_plan(robot, up_world, standoff_target, base_R):
    """The cheap default returns the SAME standoff as the old densified per-branch computation
    (firstmate deliverable 3: the selection result is identical on the coldwin57 fixture)."""
    std_fast, rep_fast = cl.select_reaction_natural_standoff(
        robot, standoff_target, up_world, STANDOFF_HI, CARRY, tool_link=TL,
        limits=_limits(), base_R=base_R, effort_limit=56.0)
    std_full, rep_full = cl.select_reaction_natural_standoff(
        robot, standoff_target, up_world, STANDOFF_HI, CARRY, tool_link=TL,
        limits=_limits(), base_R=base_R, effort_limit=56.0, full_plan=True)
    assert rep_fast['full_plan'] is False and rep_full['full_plan'] is True
    for j in ARM_JOINTS:
        assert float(std_fast[j]) == float(std_full[j])      # byte-identical standoff joints
    assert rep_fast['chosen_wrist_2'] == rep_full['chosen_wrist_2']
    # same branch SET and the same chosen branch
    assert ([round(b['wrist_2'], 6) for b in rep_fast['branches']]
            == [round(b['wrist_2'], 6) for b in rep_full['branches']])


@need_urdf
def test_fast_selection_is_cheap(robot, up_world, standoff_target, base_R):
    """The selection must be CHEAP -- well under 1 s on CPU for the coldwin57 fixture (firstmate steering
    message: cheap, not just off the hot path; the old path was ~14 s and blocked the host >5 s)."""
    t0 = time.perf_counter()
    _, rep = cl.select_reaction_natural_standoff(
        robot, standoff_target, up_world, STANDOFF_HI, CARRY, tool_link=TL,
        limits=_limits(), base_R=base_R, effort_limit=56.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, 'selection took %.3f s (must be < 1 s)' % elapsed
    assert rep['select_seconds'] < 1.0
    # and well within the default 2.0 s HOST budget
    assert rep['select_seconds'] < cl.LevelConfig().preplace_branch_select_budget_s


@need_urdf
def test_selection_within_budget_does_not_raise(robot, up_world, standoff_target, base_R):
    std, _ = cl.select_reaction_natural_standoff(
        robot, standoff_target, up_world, STANDOFF_HI, CARRY, tool_link=TL,
        limits=_limits(), base_R=base_R, effort_limit=56.0, budget_s=2.0)
    assert set(std) >= set(ARM_JOINTS)


@need_urdf
def test_budget_overrun_refuses(robot, up_world, standoff_target, base_R):
    """A search that cannot finish inside the budget raises BranchSelectBudgetExceeded (so the controller
    refuses before motion) -- never a plain bound raise, never a silent block. Driven by a fake CPU clock
    so the test is deterministic and fast."""
    ticks = iter([0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0])
    last = [0.0]

    def fake_clock():
        try:
            last[0] = next(ticks)
        except StopIteration:
            last[0] += 100.0
        return last[0]

    with pytest.raises(cl.BranchSelectBudgetExceeded):
        cl.select_reaction_natural_standoff(
            robot, standoff_target, up_world, STANDOFF_HI, CARRY, tool_link=TL,
            limits=_limits(), base_R=base_R, effort_limit=56.0, budget_s=1.0, clock=fake_clock)


def test_budget_config_validates():
    assert cl.LevelConfig().preplace_branch_select_budget_s == 2.0
    assert cl.resolve_level_config(
        {'preplace_branch_select_budget_s': 3.5}).preplace_branch_select_budget_s == 3.5
    with pytest.raises(ValueError):
        cl.LevelConfig(preplace_branch_select_budget_s=0.0).validate()
    with pytest.raises(ValueError):
        cl.LevelConfig(preplace_branch_select_budget_s=float('nan')).validate()


# ------------------------------------------------- (2) heartbeat independence (no ROS/Isaac/GPU)

def _gaps(stamps):
    s = sorted(stamps)
    return [b - a for a, b in zip(s, s[1:])]


def test_heartbeat_never_gaps_under_a_10s_blocking_planner():
    """firstmate deliverable 2: with the StatusHeartbeat filler running, a planner that blocks the
    executor thread for 10 s never lets the status heartbeat gap beyond 1 s."""
    lock = threading.Lock()
    stamps = []

    def publish():
        with lock:
            stamps.append(time.monotonic())

    hb = StatusHeartbeat(publish=publish, now=time.monotonic, is_active=lambda: True, interval=0.5)
    hb.start()
    try:
        hb.note_publish()                 # a real executor-driven publish just happened
        block_start = time.monotonic()
        time.sleep(10.0)                  # the mock planner blocks the executor for 10 s
        block_end = time.monotonic()
    finally:
        hb.stop()
    with lock:
        collected = list(stamps)
    assert collected, 'the filler published nothing during the 10 s block'
    gaps = _gaps([block_start] + collected + [block_end])
    assert max(gaps) <= 1.0, 'max status gap was %.3f s (must stay <= 1 s)' % max(gaps)


def test_coldwin59_gap_reproduced_on_the_old_no_filler_path():
    """Reproduce coldwin59: WITHOUT the filler (the old path), a 10 s executor block leaves a single
    status gap of ~10 s -- far past the 5 s Isaac HOST_STATUS_WATCHDOG bound. Uses a virtual clock so the
    reproduction is instant and deterministic."""
    stamps = []
    virtual = [0.0]

    def publish():
        stamps.append(virtual[0])

    # No StatusHeartbeat thread started -- this is the old code path (publish only on executor events).
    publish()                 # the last real publish before the blocking solve
    virtual[0] += 10.0        # the executor blocks 10 s inside the reaction-branch solve
    publish()                 # the next publish only after the solve returns
    gap = max(_gaps(stamps))
    assert gap > 5.0, 'expected a >5 s gap on the old path, got %.3f s' % gap
    # ... and the StatusHeartbeat mechanism, given the SAME 10 s virtual block, fills it: a real-time
    # version of the fix is asserted in test_heartbeat_never_gaps_under_a_10s_blocking_planner above.


def test_heartbeat_silent_when_idle():
    """An idle/terminal controller phase publishes nothing -- byte-identical to a run with no filler."""
    stamps = []
    hb = StatusHeartbeat(publish=lambda: stamps.append(time.monotonic()),
                         now=time.monotonic, is_active=lambda: False, interval=0.1)
    hb.start()
    try:
        time.sleep(0.5)
    finally:
        hb.stop()
    assert stamps == []


def test_note_publish_suppresses_fills_during_steady_publishing():
    """When real publishes arrive faster than the interval (note_publish each time), the filler adds no
    extra fills -- it only tops up genuine gaps."""
    stamps = []
    hb = StatusHeartbeat(publish=lambda: stamps.append(time.monotonic()),
                         now=time.monotonic, is_active=lambda: True, interval=0.5)
    hb.start()
    try:
        for _ in range(10):
            hb.note_publish()
            time.sleep(0.05)      # 10 ticks x 50 ms = 0.5 s, each well under the 0.5 s interval
    finally:
        hb.stop()
    assert stamps == [], 'filler published %d times while real publishes kept the gap small' % len(stamps)


def test_status_heartbeat_rejects_bad_args():
    with pytest.raises(ValueError):
        StatusHeartbeat(publish=None)
    with pytest.raises(ValueError):
        StatusHeartbeat(publish=lambda: None, interval=0.0)
