"""fm367: close the SIM-CLOCK move/settle deadline family in collect.

(1) RESTOCK_MOVE_SIM_DEADLINE_SCALE multiplies EVERY sim-clock move/settle deadline's added budget
    (the trajectory's own sim duration + the settle margin) on the arm path: the host's execute
    deadline (ros_adapter.sim_execute_budget, scaling duration + EXECUTE_SETTLE_MARGIN_S) and the
    executor's terminal-settle window (trajectory_host.goal_settle_budget, scaling GOAL_SETTLE_MARGIN_S).
    Default 1.0 => byte-identical real-rig; the sim sets 1.5. Clamped [1.0, 5.0]. WALL deadlines are OUT.
(2) A general-path collect move that returns 'failed' on a SIM-clock deadline miss (not a collision/
    plan rejection) is re-issued with a fresh deadline up to MOVE_DEADLINE_RETRY_MAX, then fails.

Run under: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest .../test_fm367_sim_deadline.py
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPTS = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, SCRIPTS)

from twin.restock_manipulation import ros_adapter as ra
from twin.restock.trajectory_host import goal_settle_budget, GOAL_SETTLE_MARGIN_S
from twin.restock_manipulation.ros_adapter import (
    sim_execute_budget, EXECUTE_SETTLE_MARGIN_S)

# Reuse the controller fixtures (FixtureMotion now reports last_fault_was_sim_deadline()).
from scripts.twin.tests.test_restock_manipulation import make_controller, packet, run_to
from twin.restock_manipulation.controller import MOVE_DEADLINE_RETRY_MAX


def _probe_scale(env_value):
    """Read MOVE_SIM_DEADLINE_SCALE + the scaled budgets from BOTH modules in a clean subprocess so the
    import-time env read is fresh (no reload pollution of the shared modules)."""
    env = dict(os.environ)
    env['PYTHONPATH'] = SCRIPTS
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    env.pop('RESTOCK_MOVE_SIM_DEADLINE_SCALE', None)
    if env_value is not None:
        env['RESTOCK_MOVE_SIM_DEADLINE_SCALE'] = env_value
    code = (
        'from twin.restock_manipulation import ros_adapter as ra;'
        'from twin.restock import trajectory_host as th;'
        'print(ra.MOVE_SIM_DEADLINE_SCALE, th.MOVE_SIM_DEADLINE_SCALE,'
        ' ra.sim_execute_budget(10.0), th.goal_settle_budget())')
    out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    ra_s, th_s, exec_b, goal_b = out.stdout.split()
    return float(ra_s), float(th_s), float(exec_b), float(goal_b)


class ScaleConstant(unittest.TestCase):
    def test_default_byte_identical_when_unset(self):
        ra_s, th_s, exec_b, goal_b = _probe_scale(None)
        self.assertEqual((ra_s, th_s), (1.0, 1.0))
        # Exact real-rig budgets: 10 + 25 execute, 20 goal-settle -- unchanged.
        self.assertEqual(exec_b, 10.0 + EXECUTE_SETTLE_MARGIN_S)
        self.assertEqual(goal_b, GOAL_SETTLE_MARGIN_S)

    def test_sim_sets_1_5x_on_every_sim_deadline(self):
        ra_s, th_s, exec_b, goal_b = _probe_scale('1.5')
        self.assertEqual((ra_s, th_s), (1.5, 1.5))
        self.assertAlmostEqual(exec_b, (10.0 + EXECUTE_SETTLE_MARGIN_S) * 1.5)   # 52.5
        self.assertAlmostEqual(goal_b, GOAL_SETTLE_MARGIN_S * 1.5)              # 30.0

    def test_clamped_sane(self):
        self.assertEqual(_probe_scale('99')[0], 5.0)    # upper clamp
        self.assertEqual(_probe_scale('0.5')[0], 1.0)   # below real-rig floor -> clamp to 1.0


class ScaledBudgetMath(unittest.TestCase):
    """The pure, importable deadline expressions scale the added budget by the passed scale."""
    def test_execute_budget_scales_duration_plus_margin(self):
        self.assertEqual(sim_execute_budget(12.0, scale=1.0), 12.0 + EXECUTE_SETTLE_MARGIN_S)
        self.assertAlmostEqual(sim_execute_budget(12.0, scale=1.5), (12.0 + EXECUTE_SETTLE_MARGIN_S) * 1.5)
        self.assertAlmostEqual(sim_execute_budget(0.0, scale=1.5), EXECUTE_SETTLE_MARGIN_S * 1.5)

    def test_goal_settle_budget_scales_margin(self):
        self.assertEqual(goal_settle_budget(scale=1.0), GOAL_SETTLE_MARGIN_S)
        self.assertAlmostEqual(goal_settle_budget(scale=1.5), GOAL_SETTLE_MARGIN_S * 1.5)

    def test_goal_settle_stays_below_execute_under_any_scale(self):
        # Ordering (GOAL_MISS prints terminal per-joint before the host execute deadline) must hold for
        # every scale and trajectory duration: the scaled settle window < the scaled execute budget.
        for scale in (1.0, 1.5, 3.0, 5.0):
            for duration in (0.0, 5.0, 60.0, 110.0):
                self.assertLess(goal_settle_budget(scale=scale),
                                sim_execute_budget(duration, scale=scale),
                                f'scale={scale} duration={duration}')


class GeneralMoveDeadlineRetry(unittest.TestCase):
    """A general-path collect move (approach/contact/lift/preplace/place/retreat) that 'failed' on a
    SIM-clock deadline miss re-issues with a fresh deadline up to MOVE_DEADLINE_RETRY_MAX, then fails.
    A collision/plan-rejection (flag False) still fails immediately (fail-fast preserved)."""

    def test_sim_deadline_fault_retries_then_fails_when_exhausted(self):
        c, m = make_controller()
        t = run_to(c, 'approach')                # approach is already issued here (moves == 1)
        issued_before = len(m.moves)
        m.outcome = 'failed'
        m.sim_deadline_fault = True              # a SIM-clock deadline miss -> retriable
        result = None
        for i in range(10):
            result = c.step(.1, packet(c, t + (i + 1) * .1))
            if c.phase == 'failed':
                break
        self.assertEqual(c.phase, 'failed')      # bounded: still fails once budget is spent
        self.assertEqual(result.reason, 'collision planning or execution failed')
        # Exactly MOVE_DEADLINE_RETRY_MAX fresh re-issues happened before the final fail.
        self.assertEqual(len(m.moves) - issued_before, MOVE_DEADLINE_RETRY_MAX)

    def test_sim_deadline_fault_stays_in_phase_while_retrying(self):
        c, m = make_controller()
        t = run_to(c, 'approach')
        m.outcome, m.sim_deadline_fault = 'failed', True
        r = c.step(.1, packet(c, t + .1))        # first failed poll -> retry
        self.assertEqual(r.status, 'running')
        self.assertEqual(c.phase, 'approach')
        self.assertFalse(c.issued)               # re-issue scheduled for next tick

    def test_collision_fault_fails_immediately_no_retry(self):
        c, m = make_controller()
        t = run_to(c, 'approach')
        issued_before = len(m.moves)
        m.outcome = 'failed'
        m.sim_deadline_fault = False             # collision/plan-rejection -> NOT a deadline fault
        r = c.step(.1, packet(c, t + .1))
        self.assertEqual(r.status, 'failed')
        self.assertEqual(len(m.moves), issued_before)   # no re-issue -> current behavior


class PortDeadlineFlag(unittest.TestCase):
    """The port's last-fault signal: a SIM-clock deadline miss flags True; a port raise (collision/plan
    rejection) clears it. Exercised on a __new__-built port (no ROS node) via the public accessor."""

    def _port(self):
        p = ra.RosMotionPort.__new__(ra.RosMotionPort)
        return p

    def test_accessor_defaults_false(self):
        self.assertFalse(self._port().last_fault_was_sim_deadline())   # no fault yet / off

    def test_sim_deadline_flag_true(self):
        p = self._port()
        p._last_fault_sim_deadline = True
        self.assertTrue(p.last_fault_was_sim_deadline())

    def test_poll_raise_clears_flag(self):
        # poll() wraps _poll(); a raising _poll (collision) must clear the flag and report 'failed'.
        p = self._port()
        p._last_fault_sim_deadline = True
        p.generation = 0
        p.handle = None
        p.state = 'plan_result'
        p.emergency_stop = lambda: None
        def _boom():
            raise RuntimeError('execution failed')
        p._poll = _boom
        self.assertEqual(p.poll(), 'failed')
        self.assertFalse(p.last_fault_was_sim_deadline())              # collision -> no deadline retry


if __name__ == '__main__':
    unittest.main()
