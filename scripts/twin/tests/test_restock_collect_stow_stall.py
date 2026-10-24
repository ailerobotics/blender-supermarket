"""sim-robust-collect-stow-stall-v1 -- CPU proof (no Isaac/ROS/GPU) that a move_group PLANNER CRASH/HANG
during the post-release collect STOW fails FAST with a NAMED cause instead of looping to the per-phase
timeout, and that a genuine crash+respawn still recovers.

Root cause (coldliveapproach, ~/datasets/twin-robot-in-store-boot-v1/mission/coldliveapproach): move_group
SIGABRT'd (``malloc_consolidate(): invalid chunk size``, host-stack.log:13869) at the retreat_tuck->stow
boundary and produced NO further output -- it died and did not respawn. The stow plan goal was sent to the
dead ``/move_action`` server; goal ACCEPTANCE (a pacing-independent ms handshake) never returned, so the
port sat in ``plan_accept`` until PR 228's SIM-clock planning deadline (PLAN_SIM_DEADLINE_S=30 sim-s) fired,
4x, and the controller looped STAGED_MOVE_RETRY to the 120-sim-s phase timeout, reported only as the opaque
``phase timeout: stow``.

Regression PR 228 left: a SIGABRT-HUNG move_group keeps ``server_is_ready()==True`` (no clean DDS
deregistration), so ``_handle_server_crash`` (gated on not-ready) stayed inert; and ``_should_deadline_crash_retry``
is gated on ``clock=='wall'`` -- unreachable now every planning deadline is on the SIM clock -- so the fm404
belt-and-braces crash recovery could never fire. coldliveapproach's passing sibling (hosttimers) had ZERO
move_group crashes, so the stall is this crash, not the economy publish interval (which never feeds the
plan_accept budget).

Fix (sim-robust-collect-stow-stall-v1): a SIM-clock PLANNING-state deadline is positive evidence of a
dead/hung planner (healthy planning is ~1-2 sim-s per PR 228's own rationale). The port re-sends the plan a
bounded PLAN_ACCEPT_CRASH_RETRY_MAX times (so a crash+respawn whose server stale-reads ready recovers), then
flags ``_planner_unrecoverable``; the controller fails the op FAST with a named cause at each move 'failed'
branch instead of its own retry loop.
"""
import sys
from pathlib import Path
from types import SimpleNamespace as S

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from twin.restock_manipulation.ros_adapter import (  # noqa: E402
    RosMotionPort, PLAN_ACCEPT_CRASH_RETRY_MAX, PLAN_SIM_DEADLINE_S, MOVE_GROUP_CRASH_RETRY_OP_MAX,
    _should_deadline_crash_retry)
# the controller-tier harness (FixtureMotion / make_controller / run_to / packet) lives with the
# manipulation tests; reuse it so the collect drive to the stow phase is the known-good one.
from scripts.twin.tests.test_restock_manipulation import (  # noqa: E402
    FixtureMotion, PROFILE, JOB, packet, run_to)
from scripts.twin.restock_manipulation import TransferController  # noqa: E402


# --------------------------------------------------------------------------------------------------
# Port tier: the SIM-clock plan_accept deadline (move_group crash/hang) -> bounded re-send then named fail
# --------------------------------------------------------------------------------------------------
class _FakeFuture:
    def add_done_callback(self, cb):
        self._cb = cb


class _FakePlan:
    """MoveGroup action-client stub. ``ready`` models a crash (False) vs a respawned/stale-ready (True)
    server; ``sent`` records every re-sent goal."""
    def __init__(self, ready=True):
        self.ready = ready
        self.sent = []

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal):
        self.sent.append(goal)
        return _FakeFuture()


def _port_at_fired_sim_plan_deadline(state='plan_accept', plan_accept_crash_retries=0, ready=True):
    """A port whose SIM-clock planning deadline has already fired (goal never serviced), as in the
    coldliveapproach stow: move_group is unresponsive but server_is_ready() stale-reads True."""
    # RosMotionPort._poll() imports moveit_msgs at entry; the port tier below drives _poll() directly, so
    # skip precisely (not fail) where ROS is off PYTHONPATH (e.g. PYTHONPATH= CPU invocation) and moveit_msgs
    # is absent. The controller tier (FixtureMotion) never touches _poll and runs regardless.
    pytest.importorskip('moveit_msgs')
    p = RosMotionPort.__new__(RosMotionPort)
    p.state = state
    p.timeout = 25.0
    p.generation = 3
    p.handle = None
    p._plan_goal = S(request=S(start_state=S(joint_state=S(name=[], position=[]))))
    p._server_crash_retries = 0
    p._op_crash_retries = 0
    p._plan_accept_crash_retries = plan_accept_crash_retries
    p._planner_unrecoverable = False
    p._awaiting_server_respawn = False
    p._replan_attempts = 5
    p._last_fault_sim_deadline = False
    p._attached_profile = None          # empty-handed stow -> _resync_scene_after_respawn is a no-op
    p._live_arm_state = None            # real-rig-less -> _reanchor_plan_goal_to_live is a no-op
    p._plan_dispatches = 0
    p._sim_now = lambda: 100.0
    p.deadline_clock = p._sim_now       # SIM clock (-> _clock == 'sim')
    p.deadline = 0.0                    # already exceeded (100 > 0) -> the deadline branch fires
    p.plan = _FakePlan(ready)
    p.execute = _FakePlan(True)
    p._estops = []
    p.emergency_stop = lambda: p._estops.append(True)
    p.payload_sync = lambda held, profile: True
    return p


def test_sim_plan_accept_deadline_re_sends_once_as_a_crash_retry():
    # The FIRST sim-clock plan_accept deadline (goal never accepted) is treated as a probable move_group
    # crash/hang: re-send the plan (so a genuine crash+respawn recovers), NOT a terminal failure yet.
    if PLAN_ACCEPT_CRASH_RETRY_MAX < 1:
        pytest.skip('crash-retry budget disabled by env')
    p = _port_at_fired_sim_plan_deadline(plan_accept_crash_retries=0)
    assert p._poll() == 'running'
    assert p._plan_accept_crash_retries == 1
    assert p.state == 'plan_accept'                 # back into the plan handshake on the re-sent goal
    assert p.plan.sent, 'the plan goal was re-sent to the (possibly respawned) server'
    assert p._planner_unrecoverable is False        # not terminal yet
    assert p.last_fault_was_planner_unrecoverable() is False
    assert p._estops == []                           # a re-send never fires the mission-fatal global stop
    # the re-send re-armed the SIM planning deadline (fresh budget), so it is no longer in the past
    assert p.deadline_clock() <= p.deadline


def test_sim_plan_accept_deadline_fails_fast_named_when_planner_does_not_recover():
    # Once the bounded re-sends are exhausted, the planner is confirmed unrecoverable: flag it terminal so
    # the controller fails fast with a NAMED cause (NOT a sim-deadline retry).
    p = _port_at_fired_sim_plan_deadline(plan_accept_crash_retries=PLAN_ACCEPT_CRASH_RETRY_MAX)
    assert p._poll() == 'failed'
    assert p._planner_unrecoverable is True
    assert p.last_fault_was_planner_unrecoverable() is True
    assert p.last_fault_was_sim_deadline() is False  # terminal takes precedence over the deadline retry
    assert p._estops == []                            # _deadline_fail('sim') holds; the controller's _fail stops


def test_plan_result_sim_deadline_also_detected_as_planner_crash():
    # A crash that strikes AFTER acceptance (plan_result sim deadline, plan never returned) is the same hang
    # class and routes the same way.
    p = _port_at_fired_sim_plan_deadline(state='plan_result',
                                         plan_accept_crash_retries=PLAN_ACCEPT_CRASH_RETRY_MAX)
    assert p._poll() == 'failed'
    assert p._planner_unrecoverable is True


def test_execute_state_sim_deadline_is_not_a_planner_crash_byte_identical():
    # A SIM-clock deadline in an EXECUTION state (a render-paced slow execute) is the existing fm367
    # transient: HOLD + flag sim-deadline for the controller's MOVE_DEADLINE_RETRY. It must NOT be treated
    # as a planner crash (byte-identical to the pre-fix execute path).
    p = _port_at_fired_sim_plan_deadline(state='execute_result')
    assert p._poll() == 'failed'
    assert p._planner_unrecoverable is False
    assert p.last_fault_was_planner_unrecoverable() is False
    assert p.last_fault_was_sim_deadline() is True
    assert p._estops == []


def test_existing_deadline_crash_retry_predicate_unchanged():
    # The pure fm404 predicate is deliberately LEFT wall-only: the new sim-clock crash handling is a
    # separate path, so a sim-clock deadline never flips the old predicate (no behaviour change for its
    # existing callers / tests).
    assert _should_deadline_crash_retry('sim', 'plan_accept', True, 0, server_ready=True) is False
    assert _should_deadline_crash_retry('sim', 'plan_accept', True, 0, server_ready=False) is False
    assert _should_deadline_crash_retry('wall', 'plan_accept', True, 0, server_ready=False) is True


def test_crash_retry_budget_is_bounded_below_a_phase_timeout():
    # The named failure must land BEFORE a per-phase timeout: (re-sends + 1) deadlines x PLAN_SIM_DEADLINE_S
    # of sim time, which at the default budget (1 re-send, 30 sim-s) is ~60 sim-s, well under the mission's
    # 120-sim-s collect phase timeout.
    assert 0 <= PLAN_ACCEPT_CRASH_RETRY_MAX <= 5
    assert (PLAN_ACCEPT_CRASH_RETRY_MAX + 1) * PLAN_SIM_DEADLINE_S < 120.0


# --------------------------------------------------------------------------------------------------
# Controller tier: a terminal planner crash fails the STOW fast + named; a transient still retries
# --------------------------------------------------------------------------------------------------
class _CrashMotion(FixtureMotion):
    """FixtureMotion + the port's terminal planner-crash signal, so the controller tier can be driven
    through a move_group crash/hang without ROS."""
    def __init__(self):
        super().__init__()
        self.planner_unrecoverable = False

    def last_fault_was_planner_unrecoverable(self):
        return self.planner_unrecoverable


_STOW = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
_TUCK = (-2.3, 1.0, 2.1066, 0.8, -1.2545)


def _collect_to_stow():
    """A collect controller driven (through the known-good staged-swing path) to the post-release STOW,
    with its begin_joint_goal issued. Returns (controller, motion, sim_time)."""
    m = _CrashMotion()
    c = TransferController(m, {'sku-box': PROFILE}, fixture_mode=True,
                           transport_joints=_STOW, staged_cart_swing=True, cart_pan_joint0=-0.8605,
                           cart_tuck_shape=_TUCK)
    c.start('collect', JOB)
    t = run_to(c, 'retreat_tuck')
    c.step(.1, packet(c, t))             # issue retreat_tuck
    c.step(.1, packet(c, t + .1))        # succeeds -> stow
    assert c.phase == 'stow', c.phase
    before = len(m.stows)
    c.step(.1, packet(c, t + .2, joint_positions=_STOW))   # stow issues begin_joint_goal
    assert len(m.stows) > before, 'the stow begin_joint_goal was issued'
    return c, m, t + .3


def test_stow_fails_fast_named_on_a_terminal_planner_crash():
    # coldliveapproach repro: the stow plan never returns because move_group crashed+hung. The port reports
    # the terminal planner-unrecoverable fault and the controller FAILS FAST with a named cause instead of
    # looping STAGED_MOVE_RETRY to the 120-sim-s phase timeout.
    c, m, t = _collect_to_stow()
    m.outcome = 'failed'
    m.planner_unrecoverable = True
    res = c.step(.1, packet(c, t, joint_positions=_STOW))
    assert res.status == 'failed'
    assert 'move_group planner crashed or hung' in res.reason, res.reason
    assert 'stow' in res.reason


def test_stow_transient_sim_deadline_still_retries_not_fast_fail():
    # A plain SIM-clock deadline WITHOUT the terminal flag (a render-pacing transient) keeps the existing
    # bounded STAGED_MOVE_RETRY -- the fast-fail fires ONLY on the terminal planner-crash signal, so the
    # healthy/transient path is unchanged.
    c, m, t = _collect_to_stow()
    m.outcome = 'failed'
    m.sim_deadline_fault = True
    m.planner_unrecoverable = False
    res = c.step(.1, packet(c, t, joint_positions=_STOW))
    assert res.status == 'running', res.reason   # re-issued, not failed
    assert c.phase == 'stow'


def test_stow_without_the_probe_is_byte_identical():
    # A motion port WITHOUT last_fault_was_planner_unrecoverable (older port / fixture) keeps the exact
    # legacy behaviour: a 'failed' stow does STAGED_MOVE_RETRY (getattr(...,None) -> not callable -> skip).
    m = FixtureMotion()
    c = TransferController(m, {'sku-box': PROFILE}, fixture_mode=True,
                           transport_joints=_STOW, staged_cart_swing=True, cart_pan_joint0=-0.8605,
                           cart_tuck_shape=_TUCK)
    c.start('collect', JOB)
    t = run_to(c, 'retreat_tuck')
    c.step(.1, packet(c, t)); c.step(.1, packet(c, t + .1))
    assert c.phase == 'stow'
    c.step(.1, packet(c, t + .2, joint_positions=_STOW))   # issue
    m.outcome = 'failed'
    res = c.step(.1, packet(c, t + .3, joint_positions=_STOW))
    assert res.status == 'running'           # legacy STAGED_MOVE_RETRY, no fast-fail
    assert c.phase == 'stow'


def test_stow_never_arriving_sensing_ack_fails_fast_named():
    # The spec's "never-arriving ack" during the stow: the per-tick sensing/freshness gate is already a
    # bounded (immediate) sim-clock check with a NAMED failure -- a stale collision world (the ESDF
    # incorporation ack never refreshes) trips CONTROLLER_INVARIANT_TRIP and fails named, never a silent
    # hang. (The empty-handed stow is not held, so the held-only collision-stale PAUSE does not apply.)
    c, m, t = _collect_to_stow()
    res = c.step(.1, packet(c, t, joint_positions=_STOW, collision_time=t - 100.0))
    assert res.status == 'failed'
    assert 'collision' in res.reason, res.reason
