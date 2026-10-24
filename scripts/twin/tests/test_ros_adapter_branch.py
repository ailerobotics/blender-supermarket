"""CPU tests for the joint-space IK branch selection (host verdict 2026-10-14).

cuMotion's task-space IK is systemically broken in this XRDF, so the port uses KDL compute_ik ->
cuMotion joint-space planning, hardened with: stow-first seeds, nearest-in-limit-branch
normalisation (so KDL's far wrapped branches -- run-80's joint2 = -4.45 -- do not reach cuMotion as
self-colliding targets), and a retry to the next seed when cuMotion rejects a c-space target. No ROS
node: the port is built via __new__ and only the pure branch helpers are exercised.
"""
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace as S
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock_manipulation.ros_adapter import RosMotionPort

LIMITS = tuple((-2 * math.pi, 2 * math.pi) for _ in range(6))
STOW = (0.0, -1.57, 1.57, -1.57, -1.57, 0.0)


def _port(stow=STOW, limits=LIMITS):
    p = RosMotionPort.__new__(RosMotionPort)
    p.limits = limits
    p.stow = stow
    return p


class NearestBranch(unittest.TestCase):
    def test_all_joints_stay_in_limits(self):
        out = _port()._nearest_branch([5.1284, -4.44626, 2.48316, -1.17849, 1.1506, -3.14159])
        self.assertIsNotNone(out)
        for v, (lo, hi) in zip(out, LIMITS):
            self.assertTrue(lo <= v <= hi, v)

    def test_unwraps_toward_nominal(self):
        # joint0 = 5.1284 with nominal 0 -> nearest branch is 5.1284 - 2*pi (~-1.155), not the +value
        out = _port()._nearest_branch([5.1284, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.assertAlmostEqual(out[0], 5.1284 - 2 * math.pi, places=3)

    def test_identity_when_already_at_nominal(self):
        out = _port()._nearest_branch(list(STOW))
        self.assertEqual([round(v, 6) for v in out], [round(v, 6) for v in STOW])

    def test_none_when_no_in_limit_branch(self):
        tight = _port(limits=tuple((-0.1, 0.1) for _ in range(6)))
        self.assertIsNone(tight._nearest_branch([3.0] * 6))

    def test_no_stow_normalises_toward_zero(self):
        out = _port(stow=None)._nearest_branch([6.4, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(out[0], 6.4 - 2 * math.pi, places=3)   # nearest 0


class Seeds(unittest.TestCase):
    def test_stow_first_then_measured_then_perturbations(self):
        seeds = _port()._seed_configs((0.1,) * 6)
        self.assertEqual(seeds[0], STOW)                 # stow seeded first (central, in-limits)
        self.assertIn((0.1,) * 6, seeds)                 # measured start present
        self.assertGreaterEqual(len(seeds), 3)           # + perturbations
        self.assertEqual(len(seeds), len(set(seeds)))    # de-duplicated

    def test_no_stow_falls_back_to_measured_only(self):
        self.assertEqual(_port(stow=None)._seed_configs((0.2,) * 6), [(0.2,) * 6])

    def test_perturbations_clamped_into_limits(self):
        tight = _port(limits=tuple((-1.6, 1.6) for _ in range(6)))
        for seed in tight._seed_configs((0.0,) * 6):
            for v, (lo, hi) in zip(seed, tight.limits):
                self.assertTrue(lo <= v <= hi)


if __name__ == '__main__':
    unittest.main()


class ReplanRetry(unittest.TestCase):
    """Bounded fresh-seed re-plan on a MoveIt plan rejection (INVALID_MOTION_PLAN). The validator is
    ground truth and cuMotion is stochastic, so re-planning the same goal up to REPLAN_MAX times turns
    the seed-variant base/plate collision into a near-certain pass before falling back / failing."""
    def test_replans_while_budget_and_goal_remain(self):
        from twin.restock_manipulation.ros_adapter import _should_replan, REPLAN_MAX
        for n in range(REPLAN_MAX):
            self.assertTrue(_should_replan(n, has_plan_goal=True), n)   # retry each attempt within budget

    def test_stops_replanning_at_budget(self):
        from twin.restock_manipulation.ros_adapter import _should_replan, REPLAN_MAX
        self.assertFalse(_should_replan(REPLAN_MAX, has_plan_goal=True))   # budget spent -> fall back

    def test_never_replans_without_a_stored_goal(self):
        from twin.restock_manipulation.ros_adapter import _should_replan
        self.assertFalse(_should_replan(0, has_plan_goal=False))

    def test_pose_goal_keeps_full_budget(self):
        # Default joint_space=False (pose goal) is byte-identical to the old REPLAN_MAX behaviour.
        from twin.restock_manipulation.ros_adapter import _should_replan, REPLAN_MAX
        self.assertTrue(_should_replan(REPLAN_MAX - 1, True, joint_space=False))
        self.assertFalse(_should_replan(REPLAN_MAX, True, joint_space=False))


class JointSpaceReplanBudget(unittest.TestCase):
    """A joint-space (cart PLACE / fixed-shape) goal gets a TIGHTER same-goal re-plan budget: a
    self-colliding joint target is rejected deterministically every re-plan, so the port must advance
    to the next IK seed/branch fast instead of hammering the GPU on an unplannable target (collectsrc
    place-on-cart OOM). Pose goals keep REPLAN_MAX (their rejections are stochastic seed-variant clips)."""
    def test_joint_space_budget_is_tighter(self):
        from twin.restock_manipulation.ros_adapter import _replan_budget, REPLAN_MAX, REPLAN_MAX_JOINT_SPACE
        self.assertLessEqual(REPLAN_MAX_JOINT_SPACE, REPLAN_MAX)
        self.assertEqual(_replan_budget(joint_space=True), REPLAN_MAX_JOINT_SPACE)
        self.assertEqual(_replan_budget(joint_space=False), REPLAN_MAX)

    def test_joint_space_stops_at_its_budget(self):
        from twin.restock_manipulation.ros_adapter import _should_replan, REPLAN_MAX_JOINT_SPACE
        # Within the joint-space budget -> re-plan; at/over it -> fall through to the next seed.
        if REPLAN_MAX_JOINT_SPACE > 0:
            self.assertTrue(_should_replan(REPLAN_MAX_JOINT_SPACE - 1, True, joint_space=True))
        self.assertFalse(_should_replan(REPLAN_MAX_JOINT_SPACE, True, joint_space=True))

    def test_joint_space_falls_through_before_pose_goal_would(self):
        # At the joint-space budget the joint goal stops re-planning while a pose goal (same attempt
        # count) still would -- the whole point: reach the next branch faster for a self-colliding target.
        from twin.restock_manipulation.ros_adapter import _should_replan, REPLAN_MAX, REPLAN_MAX_JOINT_SPACE
        if REPLAN_MAX_JOINT_SPACE < REPLAN_MAX:
            n = REPLAN_MAX_JOINT_SPACE
            self.assertFalse(_should_replan(n, True, joint_space=True))
            self.assertTrue(_should_replan(n, True, joint_space=False))


class PlanAttemptsCap(unittest.TestCase):
    """Hard COUNT backstop on cuMotion plan dispatches per move (PLAN_TOTAL_ATTEMPTS_CAP). An unplannable
    goal that churns the whole re-plan x seed budget OOMs the 12 GB card (collectsrc, ~13 plans) and
    starves the shared ESDF service into a station-heartbeat gap; the cap fails the move cleanly first."""
    def test_cap_is_below_observed_oom_count_and_above_healthy_moves(self):
        from twin.restock_manipulation.ros_adapter import PLAN_TOTAL_ATTEMPTS_CAP
        self.assertLess(PLAN_TOTAL_ATTEMPTS_CAP, 13)   # below the collectsrc OOM count (~13 plans)
        self.assertGreaterEqual(PLAN_TOTAL_ATTEMPTS_CAP, 5)  # above any healthy move (plans in 1, retries 2-4)

    def test_exhausts_at_cap(self):
        from twin.restock_manipulation.ros_adapter import _plan_attempts_exhausted, PLAN_TOTAL_ATTEMPTS_CAP
        self.assertFalse(_plan_attempts_exhausted(PLAN_TOTAL_ATTEMPTS_CAP - 1))
        self.assertTrue(_plan_attempts_exhausted(PLAN_TOTAL_ATTEMPTS_CAP))
        self.assertTrue(_plan_attempts_exhausted(PLAN_TOTAL_ATTEMPTS_CAP + 5))

    def test_send_counts_only_plan_dispatches(self):
        # _send increments _plan_dispatches for a plan-accept on self.plan, never for ik/execute sends.
        port = RosMotionPort.__new__(RosMotionPort)
        port.generation = 0
        port.plan = object()
        port.execute = object()
        port.ik = object()
        sent = []

        class _Client:
            def __init__(self, owner, name):
                self.owner, self.name = owner, name
            def send_goal_async(self, goal):
                sent.append(self.name)
                return S(add_done_callback=lambda cb: None)
        port.plan = _Client(port, 'plan')
        port.execute = _Client(port, 'execute')
        port._plan_dispatches = 0
        port._send(port.plan, object(), 'plan_accept')
        port._send(port.plan, object(), 'plan_accept')
        port._send(port.execute, object(), 'execute_accept')
        self.assertEqual(port._plan_dispatches, 2)   # only the two plan-accept dispatches counted


class PressBoundUnderHeartbeat(unittest.TestCase):
    """The place press must be BOUNDED below the station heartbeat-unavailable window, so an unplannable
    place fails with a clean planning reason rather than silently running until the 30 s heartbeat gap
    e-stops the mission (collectsrc). The plan-wall cap is the time bound; the attempts cap the GPU bound.
    Both must sit under the heartbeat window the sim profile runs with."""
    def test_plan_wall_cap_under_heartbeat_window(self):
        from twin.restock_manipulation.ros_adapter import PLAN_TOTAL_WALL_CAP_S
        # The sim profile runs RESTOCK_HEARTBEAT_UNAVAILABLE_S=30 (collectsrc). The per-move planning wall
        # cap must be under it so planning fails cleanly before the heartbeat guard can trip on the press.
        self.assertLess(PLAN_TOTAL_WALL_CAP_S, 30.0)

    def test_attempts_cap_bounds_the_press(self):
        # A finite attempts cap guarantees the press cannot issue unbounded plans (which is what drove the
        # ESDF-service starvation behind the 30 s heartbeat gap), independent of the wall cap.
        from twin.restock_manipulation.ros_adapter import PLAN_TOTAL_ATTEMPTS_CAP
        self.assertTrue(0 < PLAN_TOTAL_ATTEMPTS_CAP < 64)


class FailedResultPending(unittest.TestCase):
    """failed_result_pending() lets the /sensors-independent bridge watchdog surface a GOAL_MISS/abort
    during the lift (which suspends sensing, stalling poll) within ~0.1 s instead of hanging (run-224)."""
    @staticmethod
    def _port(state, future):
        p = RosMotionPort.__new__(RosMotionPort)
        p.state, p.future = state, future
        return p

    class _F:
        def __init__(self, done, status=None, raises=False):
            self._done, self._status, self._raises = done, status, raises
        def done(self):
            return self._done
        def result(self):
            if self._raises:
                raise RuntimeError('result raised')
            return type('R', (), {'status': self._status})()

    def test_aborted_execute_result_is_pending_failure(self):
        self.assertTrue(self._port('execute_result', self._F(True, status=6)).failed_result_pending())  # ABORTED

    def test_succeeded_result_is_not_failure(self):
        self.assertFalse(self._port('execute_result', self._F(True, status=4)).failed_result_pending())  # SUCCEEDED

    def test_result_not_yet_arrived_is_not_failure(self):
        self.assertFalse(self._port('execute_result', self._F(False)).failed_result_pending())

    def test_only_result_states_count(self):
        self.assertFalse(self._port('execute_accept', self._F(True, status=6)).failed_result_pending())

    def test_raising_execute_result_future_is_a_failure(self):
        self.assertTrue(self._port('execute_result', self._F(True, raises=True)).failed_result_pending())

    def test_failed_plan_result_is_not_pending_retries_own_it(self):
        # A PLANNING failure must NOT trip the watchdog -- poll()'s REPLAN/seed retries recover the
        # stochastic cuMotion failure (run-229); only terminal EXECUTION aborts count.
        self.assertFalse(self._port('plan_result', self._F(True, status=6)).failed_result_pending())

    def test_none_future_is_safe(self):
        self.assertFalse(self._port('execute_result', None).failed_result_pending())


class PlanningWallExceeded(unittest.TestCase):
    """planning_wall_exceeded() must NOT fire when move_group has already delivered a SUCCESSFUL plan
    result that the /sensors-paced poll() has not consumed yet (firstmate 290 / run-234 false trip),
    while still catching a genuine hang (future not done) or an exhausted failure within the budget.

    sim-robust-host-timers-audit-fix-v1: the total-move planning budget is now SIM-primary (the
    ``plan_total_sim_cap_s`` leg) with a derived-generous FROZEN-SIM WALL backstop (``plan_wall_backstop_s``),
    replacing the pure-wall cap that false-tripped coldlive sample-4 (120 s wall, ~4-8 sim-s). The fixture
    drives the sim and wall legs independently."""
    _SIM_CAP = 22.0
    _WALL_BACKSTOP = 1100.0

    @classmethod
    def _port(cls, state, future, *, sim_elapsed=None, wall_elapsed=0.0,
              sim_cap=None, wall_backstop=None):
        import time
        p = RosMotionPort.__new__(RosMotionPort)
        p.state, p.future = state, future
        p.plan_total_sim_cap_s = cls._SIM_CAP if sim_cap is None else sim_cap
        p.plan_wall_backstop_s = cls._WALL_BACKSTOP if wall_backstop is None else wall_backstop
        p._plan_started_wall = time.monotonic() - wall_elapsed
        if sim_elapsed is None:          # no sim clock wired (frozen/unseeded) -> only the wall backstop
            p._plan_started_sim = None
            p._sim_now = lambda: 0.0
        else:
            p._plan_started_sim = 0.0
            p._sim_now = lambda: sim_elapsed
        return p

    class _PlanF:
        """Fake plan-result future: status (GoalStatus) + result.error_code.val (MoveIt)."""
        def __init__(self, done, status=4, err=1, raises=False):
            self._done, self._status, self._err, self._raises = done, status, err, raises
        def done(self):
            return self._done
        def result(self):
            if self._raises:
                raise RuntimeError('result raised')
            res = type('R', (), {'error_code': type('E', (), {'val': self._err})()})()
            return type('G', (), {'status': self._status, 'result': res})()

    def test_delivered_success_pending_consume_does_not_exhaust(self):
        self.assertFalse(self._port('plan_result', self._PlanF(True, status=4, err=1),
                                    sim_elapsed=self._SIM_CAP + 5).planning_wall_exceeded())

    def test_hang_without_result_exhausts_past_sim_cap(self):
        self.assertTrue(self._port('plan_result', self._PlanF(False),
                                   sim_elapsed=self._SIM_CAP + 5).planning_wall_exceeded())

    def test_delivered_failed_result_exhausts_past_sim_cap(self):
        self.assertTrue(self._port('plan_result', self._PlanF(True, status=6, err=1),
                                   sim_elapsed=self._SIM_CAP + 5).planning_wall_exceeded())  # run-220

    def test_within_sim_cap_never_exhausts(self):
        self.assertFalse(self._port('plan_result', self._PlanF(False),
                                    sim_elapsed=self._SIM_CAP - 5).planning_wall_exceeded())

    def test_execute_phase_is_excluded(self):
        self.assertFalse(self._port('execute_result', self._PlanF(False),
                                    sim_elapsed=self._SIM_CAP + 5).planning_wall_exceeded())

    def test_render_paced_slow_but_alive_plan_does_not_trip(self):
        # coldlive sample-4 REGRESSION GUARD: a render-starved-but-alive plan burns huge WALL time
        # (120 s+) while only a few SIM seconds pass. Under the OLD pure-wall cap this e-stopped bare;
        # the SIM-primary budget must tolerate it -- wall well under the generous backstop, sim under cap.
        self.assertFalse(self._port('plan_result', self._PlanF(False),
                                    sim_elapsed=6.0, wall_elapsed=120.0).planning_wall_exceeded())

    def test_wall_backstop_catches_frozen_sim(self):
        # A FROZEN sim never advances the sim leg (sim_elapsed stuck/None), so the generous WALL
        # backstop is the only thing that can hard-stop a wholly-stalled simulator.
        self.assertTrue(self._port('plan_result', self._PlanF(False),
                                   sim_elapsed=None, wall_elapsed=self._WALL_BACKSTOP + 5)
                        .planning_wall_exceeded())
        self.assertTrue(self._port('plan_result', self._PlanF(False),
                                   sim_elapsed=1.0, wall_elapsed=self._WALL_BACKSTOP + 5)
                        .planning_wall_exceeded())

    def test_sim_configurable_cap_defers_the_fire(self):
        # The SIM budget is the real bound: a plan 30 sim-s in exhausts the 22 s default but is within a
        # widened 60 s cap; 65 sim-s exhausts even that.
        self.assertTrue(self._port('plan_result', self._PlanF(False), sim_elapsed=30).planning_wall_exceeded())
        self.assertFalse(self._port('plan_result', self._PlanF(False), sim_elapsed=30, sim_cap=60.0)
                         .planning_wall_exceeded())
        self.assertTrue(self._port('plan_result', self._PlanF(False), sim_elapsed=65, sim_cap=60.0)
                        .planning_wall_exceeded())


class _FakeFuture:
    """send_goal_async future stub: records that a goal re-send happened; accepts a done callback."""
    def add_done_callback(self, cb):
        self._cb = cb


class _FakePlan:
    """MoveGroup/ExecuteTrajectory action-client stub with a toggleable server_is_ready() to model a
    move_group PROCESS crash (server vanishes) and its respawn (server returns)."""
    def __init__(self, ready=True):
        self.ready = ready
        self.sent = []
    def server_is_ready(self):
        return self.ready
    def send_goal_async(self, goal):
        self.sent.append(goal)
        return _FakeFuture()


class MoveGroupCrashRetry(unittest.TestCase):
    """Resilience to a move_group PROCESS crash (SIGABRT double-free, run-266/268): the launch now
    RESPAWNS move_group instead of tearing the host stack down, and the planning client holds the move
    open for the respawn, then re-sends the SAME plan goal -- a logged, bounded retry instead of a lost
    20-30 min GPU run. These exercise the pure budget helper and the _handle_server_crash state path on
    a port built via __new__ (no ROS node, no GPU)."""

    def test_pure_helper_retries_within_budget(self):
        from twin.restock_manipulation.ros_adapter import _should_crash_retry, MOVE_GROUP_CRASH_RETRY_MAX
        for n in range(MOVE_GROUP_CRASH_RETRY_MAX):
            self.assertTrue(_should_crash_retry(n, has_plan_goal=True), n)

    def test_pure_helper_stops_at_budget(self):
        from twin.restock_manipulation.ros_adapter import _should_crash_retry, MOVE_GROUP_CRASH_RETRY_MAX
        self.assertFalse(_should_crash_retry(MOVE_GROUP_CRASH_RETRY_MAX, has_plan_goal=True))

    def test_pure_helper_needs_a_stored_goal(self):
        from twin.restock_manipulation.ros_adapter import _should_crash_retry
        self.assertFalse(_should_crash_retry(0, has_plan_goal=False))

    def test_deadline_crash_retry_covers_respawn_between_polls_fm404(self):
        # fm404 B-lite (run-312): a move_group SIGABRT that RESPAWNS between polls is invisible to
        # _handle_server_crash (server ready again), so the lost plan request hangs to its deadline.
        # A plan deadline with a stored goal + budget AND POSITIVE DEATH EVIDENCE (the move_group action
        # server GONE) must re-send (crash-retry), not fail. sim-robust-plan-deadline-pacing-v1: the
        # server-ready gate is new -- a wall plan deadline on a LIVE-but-slow server is NOT a crash.
        from twin.restock_manipulation.ros_adapter import (
            _should_deadline_crash_retry as R, MOVE_GROUP_CRASH_RETRY_MAX)
        # server GONE (not ready) -> a wall plan deadline re-sends as a crash-retry.
        self.assertTrue(R('wall', 'plan_result', True, 0, server_ready=False))
        self.assertTrue(R('wall', 'plan_accept', True, MOVE_GROUP_CRASH_RETRY_MAX - 1, server_ready=False))
        # sim-robust-plan-deadline-pacing-v1 (coldlive4): a LIVE server (ready) that is merely render-paced
        # slow is NEVER a crash, even on a wall deadline -- the heuristic must not burn the crash budget.
        self.assertFalse(R('wall', 'plan_result', True, 0, server_ready=True))
        self.assertFalse(R('wall', 'plan_result', True, 0))                   # default server_ready=True
        # NOT this case: sim-clock (render pacing), execute state, no goal, or budget exhausted.
        self.assertFalse(R('sim', 'plan_result', True, 0, server_ready=False))
        self.assertFalse(R('wall', 'execute_result', True, 0, server_ready=False))
        self.assertFalse(R('wall', 'await_settle', True, 0, server_ready=False))
        self.assertFalse(R('wall', 'plan_result', False, 0, server_ready=False))
        self.assertFalse(R('wall', 'plan_result', True, MOVE_GROUP_CRASH_RETRY_MAX, server_ready=False))

    def test_crash_resend_reanchors_plan_start_to_live_state_fm405(self):
        # fm405: a crash MID-MOVE must resume from the LIVE pose -- re-anchor the plan goal's start_state
        # to the live measured joints before re-sending, so cuMotion plans from where the arm actually is
        # (the grasp descent resumes and the weld latches fully, not half-latched).
        from twin.restock_manipulation.ros_adapter import JOINTS
        p = self._port()
        live = {j: float(i) for i, j in enumerate(JOINTS)}          # distinctive live joints
        p._live_arm_state = lambda: (live, {j: 0.0 for j in JOINTS}, True)
        js = S(name=list(JOINTS), position=[9.0] * 6)               # stale pre-crash start
        p._plan_goal = S(request=S(start_state=S(joint_state=js)))
        p._reanchor_plan_goal_to_live()
        self.assertEqual(list(js.position), [float(i) for i in range(len(JOINTS))])  # re-anchored, in name order

    def test_crash_resend_reanchor_is_noop_without_live_state_fm405(self):
        # real rig / no fresh live state -> request unchanged (validate still re-anchors the RESULT).
        p = self._port()
        p._live_arm_state = None
        js = S(name=['a'], position=[1.0])
        p._plan_goal = S(request=S(start_state=S(joint_state=js)))
        p._reanchor_plan_goal_to_live()
        self.assertEqual(list(js.position), [1.0])

    @staticmethod
    def _port(state='plan_result', ready=True, plan_goal='GOAL', crash_retries=0, awaiting=False):
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = state
        p.timeout = 25.0
        p.generation = 7
        p.handle = object()
        p._plan_goal = plan_goal
        p._server_crash_retries = crash_retries
        p._op_crash_retries = 0
        p._awaiting_server_respawn = awaiting
        p._replan_attempts = 5
        p.plan = _FakePlan(ready)
        p.execute = _FakePlan(ready)
        # sim-robust-plan-deadline-pacing-v1: the crash-retry re-send now arms the planning deadline on the
        # SIM clock (_arm_plan_deadline -> self._sim_now), so the __new__-built port needs a sim clock stub.
        p._sim_now = lambda: 0.0
        return p

    def test_live_server_is_inert(self):
        p = self._port(ready=True)
        self.assertIsNone(p._handle_server_crash())          # server up -> poll() proceeds normally
        self.assertFalse(p._awaiting_server_respawn)
        self.assertEqual(p.plan.sent, [])

    def test_server_loss_holds_the_move_open(self):
        p = self._port(ready=False)
        self.assertEqual(p._handle_server_crash(), 'running')  # wait for the respawn, do not fail
        self.assertTrue(p._awaiting_server_respawn)
        self.assertEqual(p.plan.sent, [])                      # nothing re-sent until the server is back

    def test_respawn_resends_the_same_plan_goal_as_a_retry(self):
        p = self._port(ready=False)
        p._handle_server_crash()                               # notice the crash, start awaiting
        p.plan.ready = p.execute.ready = True                  # move_group respawned
        self.assertEqual(p._handle_server_crash(), 'running')
        self.assertEqual(p.plan.sent, ['GOAL'])                # SAME goal re-sent
        self.assertEqual(p.state, 'plan_accept')               # back into the plan handshake
        self.assertEqual(p._server_crash_retries, 1)           # retry counted
        self.assertFalse(p._awaiting_server_respawn)
        self.assertEqual(p._replan_attempts, 0)                # fresh re-plan budget for the retry

    def test_execute_phase_crash_recovers_by_replanning(self):
        p = self._port(state='execute_result', ready=False)
        self.assertEqual(p._handle_server_crash(), 'running')  # execute server lost -> hold open
        p.plan.ready = p.execute.ready = True
        self.assertEqual(p._handle_server_crash(), 'running')
        self.assertEqual(p.plan.sent, ['GOAL'])                # re-plan (not re-execute) the same goal
        self.assertEqual(p.state, 'plan_accept')

    def test_budget_exhausted_falls_through_to_failure(self):
        from twin.restock_manipulation.ros_adapter import MOVE_GROUP_CRASH_RETRY_MAX
        p = self._port(ready=False, crash_retries=MOVE_GROUP_CRASH_RETRY_MAX)
        self.assertIsNone(p._handle_server_crash())            # no retry left -> normal failure path
        self.assertEqual(p.plan.sent, [])

    def test_no_stored_goal_falls_through(self):
        p = self._port(ready=False, plan_goal=None)
        self.assertIsNone(p._handle_server_crash())            # e.g. straight-line descent -> normal path

    def test_non_action_state_is_inert(self):
        for state in ('ik', 'idle', 'succeeded', 'failed'):
            p = self._port(state=state, ready=False)
            self.assertIsNone(p._handle_server_crash(), state)

    def test_respawn_wait_cap_surfaces_an_unrecoverable_crash(self):
        import time
        from twin.restock_manipulation.ros_adapter import RESPAWN_WAIT_CAP_S
        p = self._port(ready=False, awaiting=True)
        p._respawn_wait_started = time.monotonic() - (RESPAWN_WAIT_CAP_S + 1)  # server never came back
        with self.assertRaises(RuntimeError):
            p._handle_server_crash()                           # poll() wraps this into a clean 'failed'

    def test_begin_joint_goal_stores_a_resendable_plan_goal_fm410(self):
        # fm410 (run-316): a begin_joint_goal move (the fm403 carry_lift_shape LIFT, cart tuck/pan/place)
        # MUST store _plan_goal so a WALL plan_result deadline re-sends it (bounded crash-retry) instead of
        # hard-failing the move -> host e-stop. run-316 died on exactly this: a joint-space (straight=False)
        # plan_result wall overrun of 30 ms with _plan_goal=None -> has_plan_goal False -> no retry.
        import math, time as _t, types
        from twin.restock_manipulation import ros_adapter as A
        from twin.restock_manipulation.ros_adapter import (
            RosMotionPort, JOINTS, _should_deadline_crash_retry as R, MOVE_GROUP_CRASH_RETRY_MAX)

        class _Goal:                                   # nested-attr MoveGroup.Goal stand-in
            def __init__(self):
                js = S(name=[], position=[])
                self.request = S(group_name=None, pipeline_id=None, allowed_planning_time=0.0,
                                 num_planning_attempts=0, max_velocity_scaling_factor=0.0,
                                 max_acceleration_scaling_factor=0.0,
                                 start_state=S(joint_state=js), goal_constraints=[])
                self.planning_options = S(plan_only=False)
        fake_action = types.ModuleType('moveit_msgs.action')
        fake_action.MoveGroup = S(Goal=_Goal)
        fake_action.ExecuteTrajectory = S(Goal=_Goal)
        fake_msg = types.ModuleType('moveit_msgs.msg')
        fake_msg.Constraints = lambda: S(joint_constraints=[])
        fake_msg.JointConstraint = lambda **kw: S(**kw)
        saved = {k: sys.modules.get(k) for k in ('moveit_msgs', 'moveit_msgs.action', 'moveit_msgs.msg')}
        sys.modules['moveit_msgs'] = types.ModuleType('moveit_msgs')
        sys.modules['moveit_msgs.action'] = fake_action
        sys.modules['moveit_msgs.msg'] = fake_msg
        try:
            p = RosMotionPort.__new__(RosMotionPort)
            p.state = 'idle'
            p.plan, p.execute = _FakePlan(True), _FakePlan(True)
            p.limits = LIMITS
            p.group, p.pipeline = 'arm', 'cuMotion'
            p.velocity_scale = 0.05
            p.generation, p.timeout = 3, 25.0
            p._publish_goal_tol = lambda tol: None
            p._sim_now = lambda: 0.0   # sim-robust-plan-deadline-pacing-v1: begin_joint_goal arms the sim deadline
            shape = (0.0, -1.5, 1.5, 0.65, 0.8, -1.25)   # a carry_lift_shape-like 6-joint goal
            measured = (0.0, -1.57, 1.57, -1.57, -1.57, 0.0)
            p.begin_joint_goal(shape, measured)
            self.assertIsNotNone(p._plan_goal)                       # stored...
            self.assertIs(p._plan_goal, p.plan.sent[-1])             # ...and it is the goal actually sent
            self.assertEqual(p.state, 'plan_accept')
            # The decisive consequence: a stored goal makes the deadline crash-retry reachable -- but only
            # with POSITIVE death evidence (the move_group server GONE). sim-robust-plan-deadline-pacing-v1:
            # a live-but-slow server (ready) is NOT a crash, so the re-send fires only when server_ready=False.
            self.assertTrue(R('wall', 'plan_result', p._plan_goal is not None,
                              getattr(p, '_server_crash_retries', 0), server_ready=False))
            self.assertFalse(R('wall', 'plan_result', p._plan_goal is not None,
                               getattr(p, '_server_crash_retries', 0), server_ready=True))
            self.assertTrue(_t and MOVE_GROUP_CRASH_RETRY_MAX and math)  # keep imports referenced
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


class MoveGroupRespawnSceneResync(unittest.TestCase):
    """sim-robust-movegroup-respawn-v1: a respawned move_group starts with a FRESH planning scene that no
    longer holds restock_carried_unit (the coldlive crash landed between the grasp attach and the detach),
    so a crash-retry must RE-ATTACH the carried carton from the authoritative object_attachment state
    BEFORE it resends the plan goal -- otherwise the resent plan ignores the attached carton and plans
    through it. Also pins the serialization invariant: a scene mutation (update_payload) must never be
    issued while a plan/execute is in flight."""

    @staticmethod
    def _port(state='plan_result', ready=True, plan_goal='GOAL', attached=None, order=None):
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = state
        p.timeout = 25.0
        p.generation = 7
        p.handle = object()
        p._plan_goal = plan_goal
        p._server_crash_retries = 0
        p._awaiting_server_respawn = False
        p._replan_attempts = 5
        p._attached_profile = attached
        p._live_arm_state = None            # real-rig-style: _reanchor is a no-op
        p._sim_now = lambda: 0.0            # sim-robust-plan-deadline-pacing-v1: the re-send arms the sim deadline
        p.plan = _FakePlan(ready)
        p.execute = _FakePlan(ready)

        def _payload(held, profile):
            if order is not None:
                order.append(('attach', held, profile))
            return True
        p.payload_sync = _payload
        # record plan re-sends in the same order list so the ordering is provable
        if order is not None:
            real_send = p.plan.send_goal_async
            def _send(goal):
                order.append(('plan_send', goal))
                return real_send(goal)
            p.plan.send_goal_async = _send
        return p

    def test_respawn_mid_carry_reattaches_before_resending_plan(self):
        order = []
        p = self._port(ready=False, attached='PROFILE', order=order)
        p._handle_server_crash()                       # notice the crash, start awaiting respawn
        p.plan.ready = p.execute.ready = True          # move_group respawned
        self.assertEqual(p._handle_server_crash(), 'running')
        self.assertEqual(p.plan.sent, ['GOAL'])        # the SAME plan goal was resent
        # Decisive ordering: the carton was re-attached BEFORE the plan was resent.
        self.assertEqual([o[0] for o in order], ['attach', 'plan_send'])
        self.assertEqual(order[0], ('attach', True, 'PROFILE'))

    def test_respawn_without_carry_does_not_reattach_byte_identical(self):
        order = []
        p = self._port(ready=False, attached=None, order=order)
        p._handle_server_crash()
        p.plan.ready = p.execute.ready = True
        self.assertEqual(p._handle_server_crash(), 'running')
        self.assertEqual(p.plan.sent, ['GOAL'])
        self.assertEqual([o[0] for o in order], ['plan_send'])   # no attach when nothing is held

    def test_resync_helper_reattaches_held_carton(self):
        order = []
        p = self._port(attached='PROFILE', order=order)
        p._resync_scene_after_respawn()
        self.assertEqual(order, [('attach', True, 'PROFILE')])

    def test_resync_helper_noop_when_nothing_held(self):
        order = []
        p = self._port(attached=None, order=order)
        p._resync_scene_after_respawn()
        self.assertEqual(order, [])

    def test_resync_helper_does_not_mask_a_reattach_raise(self):
        p = self._port(attached='PROFILE')
        def _boom(held, profile):
            raise RuntimeError('attach server gone')
        p.payload_sync = _boom
        p._resync_scene_after_respawn()                # logged, swallowed: the resent plan still gates
        # (no exception propagates out of the crash-retry resync)

    def test_update_payload_tracks_authoritative_attachment_state(self):
        prof = S(sku='x', dimensions=(0.1, 0.1, 0.1))   # profile is carried opaquely by the port
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = 'succeeded'
        p._attached_profile = None
        p.payload_sync = lambda held, profile: True
        self.assertTrue(p.update_payload(True, prof))
        self.assertIs(p._attached_profile, prof)        # mirror set on a confirmed attach
        self.assertTrue(p.update_payload(False, prof))
        self.assertIsNone(p._attached_profile)          # cleared on detach

    def test_update_payload_failed_attach_leaves_mirror_untouched(self):
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = 'succeeded'
        p._attached_profile = None
        p.payload_sync = lambda held, profile: False    # attach not acknowledged
        self.assertFalse(p.update_payload(True, 'PROFILE'))
        self.assertIsNone(p._attached_profile)          # not mirrored unless the attach succeeded

    def test_update_payload_refuses_mutation_during_in_flight_plan(self):
        from twin.restock_manipulation.ros_adapter import _PLAN_IN_FLIGHT_STATES
        for st in _PLAN_IN_FLIGHT_STATES:
            p = RosMotionPort.__new__(RosMotionPort)
            p.state = st
            p._attached_profile = None
            p.payload_sync = lambda held, profile: True
            with self.assertRaises(RuntimeError, msg=st):
                p.update_payload(True, 'PROFILE')

    def test_update_payload_allowed_between_moves(self):
        for st in ('idle', 'succeeded', 'failed', 'ik'):
            p = RosMotionPort.__new__(RosMotionPort)
            p.state = st
            p._attached_profile = None
            p.payload_sync = lambda held, profile: True
            self.assertTrue(p.update_payload(True, 'PROFILE'), st)   # never raises between moves


class MoveGroupDoubleDeathRecovery(unittest.TestCase):
    """sim-robust-movegroup-double-death-v1 (coldwin4): two recovery gaps a move_group death during collect
    exposed, beyond the single-death recovery.

    GAP 1 -- the post-respawn carton RE-ATTACH must not trust a SINGLE payload_sync at the respawn instant.
    coldwin4 PAYLOAD_SYNC_DIAG showed held=True s_contact=True fresh and both servers ready, yet the freshly
    respawned move_group REJECTED the attach goal (it had not finished re-initialising) and the recovery then
    proceeded CARTON-LESS (the resend planned through the held carton). Retry the attach bounded on the SIM
    clock (spanning poll()s so the node clock advances and the respawned server finishes init) and NEVER
    resend without the carton -- fail the op BY NAME if it never succeeds.

    GAP 2 -- a MID-OP respawn+reseed leaves the pre-crash FollowJointTrajectory goal LIVE on the executor,
    so the reseeded resend's new goal is rejected 'a previous goal is still executing' -> execution failed ->
    the stale goal tracking-faults at 1.314 rad -> e-stop. The crash-retry resend force-clears it via the
    /arm_controller/abort_active signal BEFORE resending (PR 229's op-start reset never clears a LIVE goal)."""

    @staticmethod
    def _port(*, attached='PROFILE', payload=None, sim=None, state='execute_result'):
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = state
        p.timeout = 25.0
        p.generation = 7
        p.handle = None
        p._plan_goal = 'GOAL'
        p._server_crash_retries = 0
        p._op_crash_retries = 0
        p._awaiting_server_respawn = False
        p._replan_attempts = 5
        p._attached_profile = attached
        p._reattach_active = False
        p._reattach_started_sim = 0.0
        p._reattach_failed = False
        p._last_fault_sim_deadline = False
        p._live_arm_state = None            # real-rig-style: _reanchor is a no-op
        p._plan_dispatches = 0
        p._sim_now = sim if sim is not None else (lambda: 0.0)
        p._estops = []
        p.emergency_stop = lambda: p._estops.append(True)
        p.plan = _FakePlan(True)
        p.execute = _FakePlan(True)
        p.payload_sync = payload if payload is not None else (lambda held, profile: True)
        # Record the gap-2 abort_active signal without the std_msgs ROS message (unavailable on CPU); the
        # separate plumbing test proves _publish_abort_active publishes Bool(data=True) to a real pub.
        p.aborts = []
        p._publish_abort_active = lambda: p.aborts.append('abort_active')
        return p

    # ---- GAP 1: bounded sim-clock re-attach; never resend carton-less -----------------------------
    def test_reattach_retries_on_the_sim_clock_then_resends_when_accepted(self):
        t = [0.0]
        attempts = {'n': 0}
        def payload(held, profile):
            attempts['n'] += 1
            return attempts['n'] >= 3            # the respawned planner rejects twice, accepts on the 3rd
        p = self._port(payload=payload, sim=lambda: t[0])
        p._begin_post_respawn_recovery()         # starts the re-attach; first attempt rejected
        self.assertTrue(p._reattach_active)
        self.assertEqual(p.plan.sent, [])        # NOT resent carton-less
        t[0] = 1.0
        p._step_reattach()                       # 2nd attempt rejected -> still deferred
        self.assertTrue(p._reattach_active)
        self.assertEqual(p.plan.sent, [])
        t[0] = 2.0
        p._step_reattach()                       # 3rd attempt ACCEPTED -> re-attached, now resend
        self.assertFalse(p._reattach_active)
        self.assertFalse(p._reattach_failed)
        self.assertEqual(p.plan.sent, ['GOAL'])  # resent only ONCE the carton was back in the scene
        self.assertEqual(attempts['n'], 3)

    def test_reattach_fails_op_by_name_when_the_respawned_planner_never_accepts(self):
        from twin.restock_manipulation.ros_adapter import REATTACH_SIM_BUDGET_S
        t = [0.0]
        p = self._port(payload=lambda held, profile: False, sim=lambda: t[0])
        p._begin_post_respawn_recovery()
        self.assertTrue(p._reattach_active)
        t[0] = REATTACH_SIM_BUDGET_S + 1.0       # sim budget exhausted
        p._step_reattach()
        self.assertFalse(p._reattach_active)
        self.assertTrue(p._reattach_failed)
        self.assertTrue(p.last_fault_was_reattach_failure())
        self.assertFalse(p.last_fault_was_sim_deadline())   # terminal: the controller must NOT retry it
        self.assertEqual(p.state, 'failed')
        self.assertEqual(p.plan.sent, [])                    # NEVER resent carton-less
        self.assertEqual(p._estops, [])                      # _fail_hold (no global stop); _fail() stops honestly

    def test_reattach_noop_and_immediate_resend_when_nothing_held(self):
        p = self._port(attached=None)
        p._begin_post_respawn_recovery()
        self.assertFalse(p._reattach_active)
        self.assertEqual(p.plan.sent, ['GOAL'])              # no-carry resend is immediate (byte-identical)

    def test_handle_server_crash_defers_resend_until_carton_reattached(self):
        # The full respawn-back path: a held carton + a transiently-rejecting respawned planner does NOT
        # resend until the re-attach succeeds across polls, and the crash retry is counted.
        attempts = {'n': 0}
        def payload(held, profile):
            attempts['n'] += 1
            return attempts['n'] >= 2
        p = self._port(payload=payload, state='plan_result')
        p._awaiting_server_respawn = True
        self.assertEqual(p._handle_server_crash(), 'running')
        self.assertTrue(p._reattach_active)
        self.assertEqual(p.plan.sent, [])                    # deferred, never carton-less
        self.assertEqual(p._server_crash_retries, 1)         # the retry was counted at respawn detect
        p._step_reattach()                                   # 2nd attempt accepts -> resend
        self.assertFalse(p._reattach_active)
        self.assertEqual(p.plan.sent, ['GOAL'])

    # ---- GAP 2: force-clear the crash-abandoned live trajectory_host goal before resending ---------
    def test_crash_retry_publishes_abort_active_before_resending(self):
        # Every crash-retry resend (carry or not, mid-plan or mid-execute = pick/descent/stow) force-clears a
        # crash-abandoned LIVE executor goal first, so the reseeded resend is accepted (coldwin4 death 2).
        for attached in ('PROFILE', None):
            for state in ('plan_result', 'execute_result'):
                p = self._port(attached=attached, state=state)
                p._awaiting_server_respawn = True
                p._handle_server_crash()
                self.assertEqual(p.aborts, ['abort_active'], (attached, state))
                self.assertEqual(p.plan.sent, ['GOAL'], (attached, state))   # re-attach ok -> resend

    def test_two_deaths_in_one_op_each_recover_and_accrue_the_op_budget(self):
        # PR 227 per-op budget: the per-MOVE retry resets each move (begin/begin_joint_goal), but the per-OP
        # total accrues across moves. Each death's respawn re-attaches + force-clears the stale goal + resends.
        p = self._port(attached='PROFILE', state='execute_result')
        p._awaiting_server_respawn = True
        p._handle_server_crash()                                   # death 1 (e.g. the pick)
        self.assertEqual((p._server_crash_retries, p._op_crash_retries), (1, 1))
        self.assertEqual(p.plan.sent, ['GOAL'])
        p._server_crash_retries = 0                                # per-move reset at the next move's begin()
        p._awaiting_server_respawn = True
        p.state = 'execute_result'
        p._handle_server_crash()                                   # death 2 (e.g. the stow) in the SAME op
        self.assertEqual(p._op_crash_retries, 2)                   # op total accrues across moves
        self.assertEqual(p.aborts, ['abort_active', 'abort_active'])

    def test_publish_abort_active_sends_bool_true_and_tolerates_a_missing_publisher(self):
        import types as _types
        fake = _types.ModuleType('std_msgs')
        fakemsg = _types.ModuleType('std_msgs.msg')
        fakemsg.Bool = lambda data: S(data=data)
        saved = {k: sys.modules.get(k) for k in ('std_msgs', 'std_msgs.msg')}
        sys.modules['std_msgs'], sys.modules['std_msgs.msg'] = fake, fakemsg
        try:
            p = RosMotionPort.__new__(RosMotionPort)
            rec = []
            p._abort_active_pub = S(publish=lambda m: rec.append(m.data))
            p._publish_abort_active()
            self.assertEqual(rec, [True])                          # forces the host to clear the stale goal
            RosMotionPort.__new__(RosMotionPort)._publish_abort_active()   # no publisher -> no-op, no raise
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


class _AttachServer:
    """A faithful model of the object_attachment container that SURVIVES a move_group crash. A bare ADD
    while it still holds the carton is refused as a duplicate ('Object already attached'); a DETACH clears
    it (acked iff something was attached)."""
    def __init__(self, holds=True):
        self.holds = holds
        self.calls = []

    def attach_add(self):
        self.calls.append('add')
        if self.holds:
            return False          # duplicate ADD refused -> send_goal_async accepted=False
        self.holds = True
        return True

    def detach(self):
        self.calls.append('detach')
        was = self.holds
        self.holds = False
        return was                # acked only if something was attached (tolerated otherwise)


def _reconcile_payload(server, record=None):
    """A payload_sync double with the real reconcile= contract: a bare ADD against a holding server is
    rejected; reconcile=True detaches-first (tolerant) then re-ADDs, which now succeeds."""
    def payload(held, profile, reconcile=False):
        if record is not None:
            record.append((held, reconcile))
        if not held:
            return server.detach()
        if reconcile:
            server.detach()       # detach-first: clear the surviving attach (best-effort/tolerant)
        return server.attach_add()
    return payload


class _ReconcilePayloadObj:
    """Callable payload double that ALSO exposes resync_world_after_respawn, so the adapter's best-effort
    world-obstacle resync hook is exercised (a plain lambda has no such attribute -> skipped)."""
    def __init__(self, server, world_ops):
        self._server, self._world_ops, self.world_resyncs = server, world_ops, 0

    def __call__(self, held, profile, reconcile=False):
        return _reconcile_payload(self._server)(held, profile, reconcile)

    def resync_world_after_respawn(self):
        self.world_resyncs += 1
        return list(self._world_ops)


class MoveGroupReattachReconcileAfterRespawn(unittest.TestCase):
    """sim-robust-movegroup-reattach-after-respawn-v1 (respawn2): object_attachment survives the move_group
    SIGABRT in a separate container, so after the respawn it STILL holds restock_carried_unit and refuses the
    re-attach's bare ADD as a duplicate ('Object already attached. Detach first ...') -- the recovery rejected
    forever and PR 242 fast-failed. The re-attach now RECONCILES that surviving attach (detach-first, re-ADD)
    so it succeeds; the world-obstacle resync re-establishes the rest of the scene; no-respawn is unchanged."""

    @staticmethod
    def _port(payload, *, attached='PROFILE', sim=None):
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = 'execute_result'
        p.timeout = 25.0
        p.generation = 7
        p.handle = None
        p._plan_goal = 'GOAL'
        p._server_crash_retries = p._op_crash_retries = 0
        p._awaiting_server_respawn = False
        p._replan_attempts = 5
        p._attached_profile = attached
        p._reattach_active = p._reattach_failed = False
        p._reattach_started_sim = 0.0
        p._last_fault_sim_deadline = False
        p._live_arm_state = None
        p._plan_dispatches = 0
        p._sim_now = sim if sim is not None else (lambda: 0.0)
        p._estops = []
        p.emergency_stop = lambda: p._estops.append(True)
        p.plan = _FakePlan(True)
        p.execute = _FakePlan(True)
        p.payload_sync = payload
        p.aborts = []
        p._publish_abort_active = lambda: p.aborts.append('abort_active')
        return p

    def test_surviving_attach_blocks_the_legacy_bare_ADD(self):
        # Documents the respawn2 bug: a holding server refuses the bare ADD a non-reconcile call sends.
        server = _AttachServer(holds=True)
        self.assertFalse(_reconcile_payload(server)(True, 'PROFILE'))        # rejected 'already attached'
        self.assertEqual(server.calls, ['add'])

    def test_reconcile_reattach_clears_the_surviving_attach_then_resends(self):
        server = _AttachServer(holds=True)
        record = []
        p = self._port(_reconcile_payload(server, record))
        p._begin_post_respawn_recovery()
        # The re-attach went through reconcile=True, so it detached-first then re-ADDed and SUCCEEDED.
        self.assertEqual(record, [(True, True)])
        self.assertEqual(server.calls, ['detach', 'add'])
        self.assertFalse(p._reattach_active)
        self.assertFalse(p._reattach_failed)
        self.assertEqual(p.plan.sent, ['GOAL'])            # resent only once the carton was back

    def test_reattach_passes_reconcile_true(self):
        record = []
        p = self._port(_reconcile_payload(_AttachServer(holds=True), record))
        self.assertTrue(p._resync_scene_after_respawn())
        self.assertEqual(record, [(True, True)])

    def test_world_obstacles_reestablished_before_reattach(self):
        payload = _ReconcilePayloadObj(_AttachServer(holds=True), world_ops=[('world_obstacle', 'cart_cage')])
        p = self._port(payload)
        self.assertTrue(p._resync_scene_after_respawn())
        self.assertEqual(payload.world_resyncs, 1)         # the static scene was re-established on respawn

    def test_world_resync_raise_does_not_abort_the_reattach(self):
        class _Boom(_ReconcilePayloadObj):
            def resync_world_after_respawn(self):
                raise RuntimeError('apply_planning_scene not ready')
        payload = _Boom(_AttachServer(holds=True), world_ops=[])
        p = self._port(payload)
        self.assertTrue(p._resync_scene_after_respawn())   # carton re-attach still proceeds + succeeds

    def test_no_respawn_update_payload_never_passes_reconcile(self):
        # Byte-identical no-respawn path: update_payload issues a plain attach with reconcile defaulting off.
        record = []
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = 'succeeded'
        p._attached_profile = None
        p.payload_sync = _reconcile_payload(_AttachServer(holds=False), record)
        self.assertTrue(p.update_payload(True, 'PROFILE'))
        self.assertEqual(record, [(True, False)])          # reconcile NOT set outside the respawn recovery

    def test_lambda_double_without_reconcile_still_works(self):
        # A minimal fixture lambda(held, profile) is detected and called two-arg (no masked TypeError).
        p = self._port(lambda held, profile: True)
        self.assertTrue(p._resync_scene_after_respawn())
        self.assertFalse(p._payload_accepts_reconcile())


class Drain(unittest.TestCase):
    """drain() advances the PLANNING state machine off the independent watchdog tick (firstmate 290
    option b / run-237), so a move_group result is consumed within 0.1 s of arrival regardless of the
    /sensors cadence -- but never in execution/idle/terminal states."""
    @staticmethod
    def _port(state):
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = state
        p._polls = 0
        p.poll = lambda: setattr(p, '_polls', p._polls + 1)
        return p

    def test_drains_planning_states(self):
        for st in ('ik', 'plan_accept', 'plan_result'):
            p = self._port(st); p.drain()
            self.assertEqual(p._polls, 1, st)

    def test_does_not_drain_execution_or_terminal_states(self):
        for st in ('execute_accept', 'execute_result', 'idle', 'succeeded', 'failed'):
            p = self._port(st); p.drain()
            self.assertEqual(p._polls, 0, st)


class MoveGroupDeathDuringOpRoutesToRetry(unittest.TestCase):
    """sim-robust-movegroup-estop-retry-v1: an upstream move_group PROCESS death (SIGABRT heap corruption,
    coldlive: exit -6 'malloc(): unaligned tcache chunk detected') that strikes mid-op can surface as a
    RAISE inside _poll (a lost-server result future, an ABORTED result with no payload, a service failure)
    rather than the clean server-down poll or the wall-deadline hang the existing crash-retry already
    covers. That raise used to go straight to stop() -> emergency_stop() (the global 'host emergency
    stop'), ending the run even though the launch respawns move_group. These pin that such a raise now
    HOLDS for the respawn + scene re-sync + a bounded re-send instead of e-stopping, while a genuine
    rejection (server still alive) and the no-crash path stay byte-identical."""

    @staticmethod
    def _port(state='plan_result', ready=True, plan_goal='GOAL', crash_retries=0):
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = state
        p.timeout = 25.0
        p.generation = 7
        p.handle = None                 # stop() only cancels a live handle; keep None for the e-stop tests
        p._plan_goal = plan_goal
        p._server_crash_retries = crash_retries
        p._awaiting_server_respawn = False
        p._replan_attempts = 5
        p._last_fault_sim_deadline = False
        p._sim_now = lambda: 0.0            # sim-robust-plan-deadline-pacing-v1: the respawn re-send arms the sim deadline
        p.plan = _FakePlan(ready)
        p.execute = _FakePlan(ready)
        p._estops = []
        p.emergency_stop = lambda: p._estops.append(True)
        return p

    def _poll_raising(self, p, exc):
        """Drive the real poll() wrapper with a _poll that raises ``exc`` (the move_group-death raise)."""
        p._poll = lambda: (_ for _ in ()).throw(exc)
        return p.poll()

    # --- pure helper truth table -------------------------------------------------------------------
    def test_pure_helper_recovers_only_a_gone_server_with_goal_and_budget(self):
        from twin.restock_manipulation.ros_adapter import (
            _should_recover_from_poll_fault as R, MOVE_GROUP_CRASH_RETRY_MAX)
        self.assertTrue(R(server_ready=False, has_plan_goal=True, crash_retries=0, is_respawn_timeout=False))
        self.assertTrue(R(False, True, MOVE_GROUP_CRASH_RETRY_MAX - 1, False))
        # A LIVE server delivered a genuine rejection -> surface it, do not mask as a crash.
        self.assertFalse(R(server_ready=True, has_plan_goal=True, crash_retries=0, is_respawn_timeout=False))
        # No stored goal (straight-line contact descent) / budget spent / respawn-cap timeout -> escalate.
        self.assertFalse(R(False, False, 0, False))
        self.assertFalse(R(False, True, MOVE_GROUP_CRASH_RETRY_MAX, False))
        self.assertFalse(R(False, True, 0, True))

    # --- poll() routing ----------------------------------------------------------------------------
    def test_death_raise_with_server_gone_holds_without_estop(self):
        p = self._port(state='plan_result', ready=False)
        out = self._poll_raising(p, AttributeError("'NoneType' object has no attribute 'error_code'"))
        self.assertEqual(out, 'running')               # held for the respawn, NOT failed
        self.assertEqual(p._estops, [])                # the global emergency stop never fired
        self.assertTrue(p._awaiting_server_respawn)    # _handle_server_crash takes over next poll
        self.assertEqual(p.state, 'plan_result')       # state preserved so the same goal re-sends

    def test_death_raise_at_each_op_move_recovers(self):
        # A death during any stored-goal op move -- the pick reach, the descent plan, the release-retreat
        # and the collect stow (coldlive) are all MoveGroup plan / ExecuteTrajectory states -- holds for
        # the respawn instead of e-stopping. (pick=plan_accept reach, descent=plan_result, retreat=
        # execute_accept, stow=execute_result are representative of the four server-interaction states.)
        for state in ('plan_accept', 'plan_result', 'execute_accept', 'execute_result'):
            p = self._port(state=state, ready=False)
            out = self._poll_raising(p, RuntimeError('result raised: server connection lost'))
            self.assertEqual(out, 'running', state)
            self.assertEqual(p._estops, [], state)
            self.assertTrue(p._awaiting_server_respawn, state)

    def test_genuine_rejection_with_live_server_still_estops(self):
        # Server ALIVE -> the raise is a real planning/validation fault (IK failed, planning failed,
        # invalid trajectory): surface it honestly via stop() -> emergency_stop, byte-identical.
        p = self._port(state='plan_result', ready=True)
        out = self._poll_raising(p, RuntimeError('planning failed'))
        self.assertEqual(out, 'failed')
        self.assertEqual(p._estops, [True])            # genuine fault still e-stops
        self.assertFalse(p._awaiting_server_respawn)

    def test_straight_line_descent_without_goal_estops(self):
        # The contact straight-line descent stores NO _plan_goal, so a death there has nothing to re-send
        # -> the normal e-stop path owns it (byte-identical with the pre-fix behaviour).
        p = self._port(state='plan_result', ready=False, plan_goal=None)
        out = self._poll_raising(p, RuntimeError('result raised'))
        self.assertEqual(out, 'failed')
        self.assertEqual(p._estops, [True])

    def test_retry_bound_exhausts_to_estop(self):
        from twin.restock_manipulation.ros_adapter import MOVE_GROUP_CRASH_RETRY_MAX
        p = self._port(state='plan_result', ready=False, crash_retries=MOVE_GROUP_CRASH_RETRY_MAX)
        out = self._poll_raising(p, RuntimeError('result raised'))
        self.assertEqual(out, 'failed')                # budget spent -> escalate with the honest fault
        self.assertEqual(p._estops, [True])

    def test_respawn_timeout_raise_is_never_masked(self):
        # The RESPAWN_WAIT_CAP_S 'did not respawn' raise IS the 'respawn never came' escalation -- it must
        # e-stop, not loop back into another hold.
        p = self._port(state='plan_result', ready=False)
        out = self._poll_raising(p, RuntimeError('move_group did not respawn within 45s'))
        self.assertEqual(out, 'failed')
        self.assertEqual(p._estops, [True])

    def test_non_server_state_raise_estops(self):
        # Outside a plan/execute state (e.g. the IK service state) a raise is a genuine fault.
        for state in ('ik', 'idle', 'succeeded', 'failed'):
            p = self._port(state=state, ready=False)
            out = self._poll_raising(p, RuntimeError('IK failed'))
            self.assertEqual(out, 'failed', state)
            self.assertEqual(p._estops, [True], state)

    def test_recovery_hands_off_to_handle_server_crash(self):
        # After the hold, the NEXT poll()'s _handle_server_crash owns the respawn wait + re-send: once the
        # server is back it re-sends the SAME goal and counts the retry (no emergency stop in between).
        p = self._port(state='plan_result', ready=False)
        self._poll_raising(p, RuntimeError('result raised'))     # hold
        self.assertEqual(p.plan.sent, [])
        p._attached_profile = None                               # no carry -> resync is a no-op
        p._live_arm_state = None
        p.plan.ready = p.execute.ready = True                    # move_group respawned
        self.assertEqual(p._handle_server_crash(), 'running')
        self.assertEqual(p.plan.sent, ['GOAL'])                  # same goal re-sent
        self.assertEqual(p._server_crash_retries, 1)             # retry counted once
        self.assertEqual(p._estops, [])

    def test_planning_wall_cap_suspended_during_respawn_wait(self):
        # While HELD for a respawn the dedicated RESPAWN_WAIT_CAP_S bounds the wait; the stale per-move
        # anchors must NOT let the bridge-watchdog planning budget fire and e-stop the hold.
        # sim-robust-host-timers-audit-fix-v1: drive the SIM leg (the primary) past its cap.
        import time
        p = self._port(state='plan_result', ready=False)
        p.future = None
        p.plan_total_sim_cap_s = 22.0
        p.plan_wall_backstop_s = 1100.0
        p._plan_started_sim = 0.0
        p._sim_now = lambda: 27.0                                # 27 sim-s > 22 s cap -> a genuine hang
        p._plan_started_wall = time.monotonic()
        self.assertTrue(p.planning_wall_exceeded())              # a genuine hang still exhausts...
        p._awaiting_server_respawn = True
        self.assertFalse(p.planning_wall_exceeded())             # ...but a respawn hold is suspended

    def test_no_crash_path_does_not_touch_recovery(self):
        # The no-crash path never raises in poll(), so _recover_movegroup_death_or_none is never reached
        # and emergency_stop never fires: a clean _poll returning 'succeeded' is byte-identical.
        p = self._port(state='plan_result', ready=True)
        p._poll = lambda: 'succeeded'
        self.assertEqual(p.poll(), 'succeeded')
        self.assertEqual(p._estops, [])
        self.assertFalse(p._awaiting_server_respawn)


class MoveGroupSecondDeathAndExecution(unittest.TestCase):
    """sim-robust-movegroup-second-death-v1. The estopretry proof run recovered a NATURAL move_group death
    during the collect pick (PR 225) but a SECOND, hand-injected death during the STOW still ended the run
    MANIP_HOST_FAILED 'host emergency stop'. Two gaps, both pinned here:

    (1) DEATH DURING EXECUTION. move_group's SIGABRT can strike while a trajectory is EXECUTING -- but the
        EXECUTE action server is the trajectory host, which does NOT die with move_group, so watching only
        it was blind to a move_group death mid-execution. Now the port watches move_group (self.plan) during
        an execute state too: a death there holds the arm for the respawn, re-syncs the scene, and re-plans
        the move from the live measured pose (resume). A straight-line execute (no _plan_goal, bypasses
        move_group) is unaffected -- byte-identical.
    (2) THE REAL estopretry e-stop. The stow RETREAT hit a SIM-clock move deadline (over 0.01 s, a render-
        pacing transient the controller re-issues via MOVE_DEADLINE_RETRY), but poll() called stop() ->
        emergency_stop() -- the mission-FATAL global stop -- BEFORE the retry re-issued. A SIM-clock miss now
        HOLDS without the global stop (_fail_hold); only a WALL miss / exhausted retry e-stops.
    Plus: a per-OP crash-retry cap (per-move budget already resets, so a death in a LATER step recovers; the
    cap bounds a storm across moves), and a TRUTHFUL re-attach report (ok=False now says FAILED, not
    're-attached ...')."""

    @staticmethod
    def _port(state='execute_result', plan_ready=False, execute_ready=True, plan_goal='GOAL',
              crash_retries=0, op_crash_retries=0, attached=None):
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = state
        p.timeout = 25.0
        p.generation = 7
        p.handle = None
        p._plan_goal = plan_goal
        p._server_crash_retries = crash_retries
        p._op_crash_retries = op_crash_retries
        p._awaiting_server_respawn = False
        p._replan_attempts = 5
        p._last_fault_sim_deadline = False
        p._attached_profile = attached
        p._live_arm_state = None
        p._sim_now = lambda: 0.0            # sim-robust-plan-deadline-pacing-v1: the re-plan re-send arms the sim deadline
        p.plan = _FakePlan(plan_ready)
        p.execute = _FakePlan(execute_ready)
        p._estops = []
        p.emergency_stop = lambda: p._estops.append(True)
        p.payload_sync = lambda held, profile: True
        return p

    # --- (1) move_group death DURING trajectory execution -----------------------------------------
    def test_server_action_watches_movegroup_during_execution(self):
        # Execute state, trajectory host ALIVE, move_group GONE -> the crash-relevant server is move_group.
        for state in ('execute_accept', 'execute_result'):
            p = self._port(state=state, plan_ready=False, execute_ready=True)
            self.assertIs(p._server_action_for_state(), p.plan, state)

    def test_normal_execution_still_watches_trajectory_host_byte_identical(self):
        # move_group ALIVE during an execute -> unchanged: the watched server is the trajectory host.
        for state in ('execute_accept', 'execute_result'):
            p = self._port(state=state, plan_ready=True, execute_ready=True)
            self.assertIs(p._server_action_for_state(), p.execute, state)
            self.assertIsNone(p._handle_server_crash(), state)   # both alive -> inert
            self.assertEqual(p.plan.sent, [], state)

    def test_execution_death_at_pick_descent_stow_holds_then_replans(self):
        # The pick reach, the descent and the collect STOW all execute a cuMotion-planned trajectory, so a
        # move_group death during their EXECUTION must recover like one during planning: hold, respawn,
        # re-plan the move from the live pose (plan -> execute resumes the segment). execute_accept and
        # execute_result stand in for the execution phase of each of those moves.
        for label, state in (('pick', 'execute_accept'), ('descent', 'execute_result'),
                             ('stow', 'execute_result')):
            p = self._port(state=state, plan_ready=False, execute_ready=True)
            self.assertEqual(p._handle_server_crash(), 'running', label)   # move_group gone -> hold
            self.assertTrue(p._awaiting_server_respawn, label)
            self.assertEqual(p.plan.sent, [], label)                       # nothing re-sent while down
            p.plan.ready = True                                            # move_group respawned
            self.assertEqual(p._handle_server_crash(), 'running', label)
            self.assertEqual(p.plan.sent, ['GOAL'], label)                 # re-PLAN the move (resume)
            self.assertEqual(p.state, 'plan_accept', label)                # back into the plan handshake
            self.assertEqual(p._estops, [], label)                         # never e-stopped

    def test_execution_death_during_straight_line_is_not_a_crash(self):
        # A straight-line execute stores no _plan_goal (it bypasses move_group), so a move_group death there
        # is irrelevant: fall through to the normal path, do NOT hijack it as a crash recovery.
        p = self._port(state='execute_result', plan_ready=False, execute_ready=True, plan_goal=None)
        self.assertIs(p._server_action_for_state(), p.execute)   # traj host (alive) -> inert
        self.assertIsNone(p._handle_server_crash())

    def test_execution_death_poll_raise_holds_without_estop(self):
        # A move_group death during execution that surfaces as a RAISE in _poll (lost-server result future)
        # routes through _recover_movegroup_death_or_none too: hold, no global stop.
        p = self._port(state='execute_result', plan_ready=False, execute_ready=True)
        p._poll = lambda: (_ for _ in ()).throw(RuntimeError('result raised: server connection lost'))
        self.assertEqual(p.poll(), 'running')
        self.assertEqual(p._estops, [])
        self.assertTrue(p._awaiting_server_respawn)

    # --- (2) a SIM-clock deadline miss must not fire the mission-fatal emergency stop ---------------
    def test_sim_deadline_holds_without_global_stop(self):
        # The estopretry root cause: a SIM-clock move-deadline miss is a render-pacing transient the
        # controller re-issues (MOVE_DEADLINE_RETRY). It must HOLD (cancel + keep the pose) WITHOUT the
        # global emergency stop, else the mission dies before the retry.
        p = self._port(state='execute_result')
        p.handle = None
        p._deadline_fail('sim')
        self.assertEqual(p.state, 'failed')          # move failed so the controller re-issues
        self.assertEqual(p._estops, [])              # ...but the mission-fatal global stop never fired

    def test_wall_deadline_still_emergency_stops(self):
        # A WALL miss is a genuine hang / planner fail-fast and keeps the immediate emergency stop.
        p = self._port(state='plan_result')
        p._deadline_fail('wall')
        self.assertEqual(p.state, 'failed')
        self.assertEqual(p._estops, [True])

    def test_fail_hold_cancels_goal_and_bumps_generation_without_estop(self):
        p = self._port(state='execute_result')
        cancelled = []
        p.handle = S(cancel_goal_async=lambda: cancelled.append(True))
        gen = p.generation
        p._fail_hold()
        self.assertEqual(p.state, 'failed')
        self.assertEqual(cancelled, [True])          # the in-flight goal is cancelled, like stop()...
        self.assertEqual(p.generation, gen + 1)      # ...and late callbacks invalidated...
        self.assertEqual(p._estops, [])              # ...but NO emergency stop

    # --- two deaths in one op, per-move reset + per-op bound ---------------------------------------
    def test_two_deaths_in_one_op_both_recover(self):
        # Death #1 during the pick execution, then (after the controller re-issues the NEXT move, which
        # resets the per-move budget) death #2 during the stow execution -- both recover; the per-op total
        # counts both.
        p = self._port(state='execute_result', plan_ready=False, execute_ready=True)
        # death #1
        self.assertEqual(p._handle_server_crash(), 'running')
        p.plan.ready = True
        self.assertEqual(p._handle_server_crash(), 'running')
        self.assertEqual(p._server_crash_retries, 1)
        self.assertEqual(p._op_crash_retries, 1)
        # the controller re-issues the next move -> per-move budget resets (begin/begin_joint_goal), the
        # per-op total persists; simulate that reset:
        p._server_crash_retries = 0
        p._awaiting_server_respawn = False
        p.state = 'execute_result'
        p.plan.ready = False
        # death #2 in the later step
        self.assertEqual(p._handle_server_crash(), 'running')
        p.plan.ready = True
        self.assertEqual(p._handle_server_crash(), 'running')
        self.assertEqual(p._server_crash_retries, 1)     # fresh per-move budget was available
        self.assertEqual(p._op_crash_retries, 2)         # ...and the per-op total counted both deaths
        self.assertEqual(p._estops, [])

    def test_per_op_cap_bounds_a_crash_storm_across_moves(self):
        from twin.restock_manipulation.ros_adapter import (
            MOVE_GROUP_CRASH_RETRY_OP_MAX, _op_crash_budget_ok)
        self.assertTrue(_op_crash_budget_ok(MOVE_GROUP_CRASH_RETRY_OP_MAX - 1))
        self.assertFalse(_op_crash_budget_ok(MOVE_GROUP_CRASH_RETRY_OP_MAX))
        # At the per-op cap, a fresh-per-move budget is NOT enough: the recovery falls through to e-stop.
        p = self._port(state='plan_result', plan_ready=False, crash_retries=0,
                       op_crash_retries=MOVE_GROUP_CRASH_RETRY_OP_MAX)
        self.assertIsNone(p._handle_server_crash())      # per-op budget spent -> normal failure path
        self.assertEqual(p.plan.sent, [])

    def test_reset_crash_budget_clears_both_counters(self):
        p = self._port(crash_retries=2, op_crash_retries=5)
        p.reset_crash_budget()
        self.assertEqual(p._op_crash_retries, 0)
        self.assertEqual(p._server_crash_retries, 0)

    # --- truthful re-attach report -----------------------------------------------------------------
    def test_resync_returns_true_on_success(self):
        p = self._port(attached='PROFILE')
        p.payload_sync = lambda held, profile: True
        self.assertTrue(p._resync_scene_after_respawn())

    def test_resync_returns_true_when_nothing_held(self):
        p = self._port(attached=None)
        self.assertTrue(p._resync_scene_after_respawn())   # no-op scene is consistent

    def test_resync_reports_failure_truthfully_when_payload_not_ok(self):
        # The estopretry 'ok=False' re-attach: payload_sync returns not-ok at its sensor precondition gate,
        # so the carton is NOT in the respawned scene. The helper must report that as a FAILURE, not claim
        # it 're-attached'.
        p = self._port(attached='PROFILE')
        p.payload_sync = lambda held, profile: False
        self.assertFalse(p._resync_scene_after_respawn())

    def test_resync_returns_false_on_reattach_raise(self):
        p = self._port(attached='PROFILE')
        def _boom(held, profile):
            raise RuntimeError('attach server gone')
        p.payload_sync = _boom
        self.assertFalse(p._resync_scene_after_respawn())   # raised -> not restored, reported, swallowed


class PlanDeadlineOnSimClockPacing(unittest.TestCase):
    """sim-robust-plan-deadline-pacing-v1: the PLANNING-phase per-move deadline is on the SIM clock, so a
    render-paced slow-but-alive plan is never mis-read as a move_group crash (coldlive4: a wall plan-accept
    deadline fired ~0.03-0.10 s over under the live_pose arm-feed render pacing, with ZERO actual move_group
    deaths, and the crash-retry heuristic burned its whole budget -> host e-stop). Exercised on a __new__
    port with a FAKE wall/sim clock pair (no ROS node, no GPU) and the public pure helpers."""

    class _Clock:
        """A wall/sim clock pair: wall advances ``pace`` x faster than sim (render pacing)."""
        def __init__(self, pace):
            self.sim = 0.0
            self.wall = 0.0
            self.pace = pace
        def advance_sim(self, ds):
            self.sim += ds
            self.wall += ds * self.pace

    def _port(self, clock, server_ready=True):
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = 'plan_accept'
        p.timeout = 25.0                       # the legacy WALL deadline budget
        p._plan_goal = 'GOAL'
        p._server_crash_retries = 0
        p._op_crash_retries = 0
        p._sim_now = lambda: clock.sim
        p.plan = _FakePlan(server_ready)
        p.execute = _FakePlan(server_ready)
        return p

    @staticmethod
    def _deadline_fired(p):
        """The exact predicate _poll() checks: deadline_clock() > deadline."""
        return p.deadline_clock() > p.deadline

    def test_arm_plan_deadline_uses_the_sim_clock(self):
        from twin.restock_manipulation.ros_adapter import PLAN_SIM_DEADLINE_S
        clock = self._Clock(pace=3.0)
        p = self._port(clock)
        p._arm_plan_deadline()
        self.assertIs(p.deadline_clock, p._sim_now)                 # SIM clock, not time.monotonic
        self.assertAlmostEqual(p.deadline, clock.sim + PLAN_SIM_DEADLINE_S)

    def test_slow_plan_under_3x_pacing_does_not_trip(self):
        # A slow-but-alive plan: delivery takes 12 sim-s. At 3x pacing that is 36 WALL-s -- PAST the legacy
        # 25 s WALL deadline (which would have false-tripped the crash-retry), but WELL under the sim budget.
        from twin.restock_manipulation.ros_adapter import (
            PLAN_SIM_DEADLINE_S, _should_deadline_crash_retry as R)
        self.assertGreater(PLAN_SIM_DEADLINE_S, 12.0)               # the default tolerates this slow plan
        clock = self._Clock(pace=3.0)
        p = self._port(clock, server_ready=True)                   # move_group ALIVE, just render-paced slow
        p._arm_plan_deadline()
        clock.advance_sim(12.0)                                    # 12 sim-s == 36 wall-s of slow delivery
        self.assertGreater(clock.wall, p.timeout)                  # the OLD wall deadline would have fired...
        self.assertFalse(self._deadline_fired(p))                  # ...but the SIM deadline has NOT
        # And even if a deadline did fire with the server alive, it is NOT a crash (positive-death gate):
        self.assertFalse(R('sim', 'plan_accept', True, 0, server_ready=True))

    def test_hung_planner_fails_by_name_after_the_bounded_sim_budget(self):
        # A genuinely hung planner delivers NO result; the simulator keeps stepping, so the SIM deadline
        # fires after the bounded budget. It is a SIM-clock miss (deadline_clock is _sim_now) -> flagged as
        # a render-pacing transient the controller re-issues (bounded MOVE_DEADLINE_RETRY) then e-stops
        # honestly -- NEVER a crash-retry (the server is alive).
        from twin.restock_manipulation.ros_adapter import (
            PLAN_SIM_DEADLINE_S, _should_deadline_crash_retry as R)
        clock = self._Clock(pace=3.0)
        p = self._port(clock, server_ready=True)
        p._arm_plan_deadline()
        clock.advance_sim(PLAN_SIM_DEADLINE_S + 1.0)               # sim advances past the budget, no result
        self.assertTrue(self._deadline_fired(p))                   # the bounded sim budget fires
        # The fired deadline is on the SIM clock (deadline_clock is _sim_now, not time.monotonic), so _poll
        # reads _clock='sim' -> a render-pacing transient the controller re-issues (bounded), NEVER a crash.
        self.assertIsNot(p.deadline_clock, time.monotonic)
        self.assertFalse(R('sim', 'plan_accept', True, 0, server_ready=True))

    def test_real_death_recovers_with_positive_evidence(self):
        # A REAL move_group death: the action server VANISHES (server_is_ready False). _handle_server_crash
        # holds the move open for the respawn (positive death evidence), and the wall-deadline crash-retry
        # heuristic is permitted ONLY with that same evidence (server gone).
        from twin.restock_manipulation.ros_adapter import _should_deadline_crash_retry as R
        clock = self._Clock(pace=3.0)
        p = self._port(clock, server_ready=False)                  # move_group gone
        self.assertEqual(p._handle_server_crash(), 'running')      # held for respawn, not failed
        self.assertTrue(p._awaiting_server_respawn)
        self.assertTrue(R('wall', 'plan_accept', True, 0, server_ready=False))   # death evidence -> crash-retry allowed

    def test_normal_path_is_byte_identical(self):
        # The no-crash healthy path: a plan that delivers within the sim budget with the server alive never
        # fires the deadline and is never a crash-retry -- unchanged from before this lane.
        from twin.restock_manipulation.ros_adapter import _should_deadline_crash_retry as R
        clock = self._Clock(pace=3.0)
        p = self._port(clock, server_ready=True)
        p._arm_plan_deadline()
        clock.advance_sim(1.5)                                     # healthy plan delivers in ~1-2 sim-s
        self.assertFalse(self._deadline_fired(p))
        self.assertFalse(R('wall', 'plan_accept', True, 0, server_ready=True))


if __name__ == '__main__':
    unittest.main()
