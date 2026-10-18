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
    while still catching a genuine hang (future not done) or an exhausted failure within the cap."""
    import math as _math
    from twin.restock_manipulation.ros_adapter import PLAN_TOTAL_WALL_CAP_S as _CAP

    @staticmethod
    def _port(state, future, elapsed, cap=None):
        import time
        p = RosMotionPort.__new__(RosMotionPort)
        p.state, p.future = state, future
        p._plan_started_wall = time.monotonic() - elapsed
        if cap is not None:
            p.plan_wall_cap_s = cap
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
        over = self._CAP + 5
        self.assertFalse(self._port('plan_result', self._PlanF(True, status=4, err=1), over)
                         .planning_wall_exceeded())

    def test_hang_without_result_exhausts_past_cap(self):
        over = self._CAP + 5
        self.assertTrue(self._port('plan_result', self._PlanF(False), over).planning_wall_exceeded())

    def test_delivered_failed_result_exhausts_past_cap(self):
        over = self._CAP + 5
        self.assertTrue(self._port('plan_result', self._PlanF(True, status=6, err=1), over)
                        .planning_wall_exceeded())      # aborted plan delivered -> surface it (run-220)

    def test_within_cap_never_exhausts(self):
        self.assertFalse(self._port('plan_result', self._PlanF(False), self._CAP - 5)
                         .planning_wall_exceeded())

    def test_execute_phase_is_excluded(self):
        over = self._CAP + 5
        self.assertFalse(self._port('execute_result', self._PlanF(False), over).planning_wall_exceeded())

    def test_sim_configurable_cap_defers_the_fire(self):
        # firstmate 293: the sim profile raises the cap to 60 s (result-delivery latency), so an
        # in-flight plan at 30 s -- which would exhaust at the 22 s default -- is still within budget.
        self.assertTrue(self._port('plan_result', self._PlanF(False), 30).planning_wall_exceeded())
        self.assertFalse(self._port('plan_result', self._PlanF(False), 30, cap=60.0).planning_wall_exceeded())
        self.assertTrue(self._port('plan_result', self._PlanF(False), 65, cap=60.0).planning_wall_exceeded())


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

    @staticmethod
    def _port(state='plan_result', ready=True, plan_goal='GOAL', crash_retries=0, awaiting=False):
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = state
        p.timeout = 25.0
        p.generation = 7
        p.handle = object()
        p._plan_goal = plan_goal
        p._server_crash_retries = crash_retries
        p._awaiting_server_respawn = awaiting
        p._replan_attempts = 5
        p.plan = _FakePlan(ready)
        p.execute = _FakePlan(ready)
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
