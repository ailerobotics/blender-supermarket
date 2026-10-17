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
    def _port(state, future, elapsed):
        import time
        p = RosMotionPort.__new__(RosMotionPort)
        p.state, p.future = state, future
        p._plan_started_wall = time.monotonic() - elapsed
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
