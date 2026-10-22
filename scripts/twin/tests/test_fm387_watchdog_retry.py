"""fm387: the host-stack watchdog's motion faults (goal-tolerance / track-miss / watchdog-drain) are
routed through the bounded fm372 move retry (retry_or_fail_move) instead of a direct _fail, so a
stochastic-crash-induced or convergence-in-time goal-miss RE-ROLLS. Pure controller-side tests."""
import pytest
pytest.importorskip('rclpy')  # make_controller builds the real controller (mirrors the fm372 suite)

from scripts.twin.tests.test_restock_manipulation import make_controller


def _armed(phase):
    c, _motion = make_controller()
    c.phase = phase
    c.issued = True
    return c


def _status(r):
    return getattr(r, 'status', None)


def test_watchdog_goal_miss_general_phase_retries_not_fails():
    c = _armed('lift')                      # general move phase
    r = c.retry_or_fail_move('arm motion failed (goal tolerance not reached)')
    assert _status(r) != 'failed'           # RUNNING -> mission continues, not killed
    assert c.phase == 'lift' and c.issued is False   # re-issue the move next tick
    assert c._move_deadline_retries == 1 and c._staged_retries == 0
    assert c.motion.stops == 0              # NOT emergency-stopped


def test_watchdog_goal_miss_staged_phase_uses_staged_budget():
    c = _armed('tuck')                      # staged cart-swing move
    r = c.retry_or_fail_move('arm motion failed (goal tolerance not reached)')
    assert _status(r) != 'failed' and c.issued is False
    assert c._staged_retries == 1 and c._move_deadline_retries == 0


def test_watchdog_fault_on_nonmove_phase_hard_fails():
    c = _armed('grasp')                     # gripper phase, no begin to re-issue
    r = c.retry_or_fail_move('arm motion failed (goal tolerance not reached)')
    assert _status(r) == 'failed' and c.motion.stops == 1


def test_watchdog_retry_is_bounded_then_fails():
    c = _armed('lift')                      # default _move_deadline_retry_max == 2
    assert _status(c.retry_or_fail_move('x')) != 'failed'; c.issued = True
    assert _status(c.retry_or_fail_move('x')) != 'failed'; c.issued = True
    r = c.retry_or_fail_move('x')                               # budget spent -> fail (no infinite loop)
    assert _status(r) == 'failed' and c._move_deadline_retries == 2 and c.motion.stops == 1


def test_watchdog_retry_acks_failed_result_to_prevent_refire():
    c = _armed('lift')
    acked = []
    c.motion.ack_failed_result = lambda: acked.append(True)   # port clear hook
    c.retry_or_fail_move('arm motion failed (goal tolerance not reached)')
    assert acked == [True]                  # cleared the stale result so the 0.1 s watchdog won't re-fire


def test_watchdog_retry_tolerates_motion_without_ack_hook():
    c = _armed('lift')                      # FixtureMotion has no ack_failed_result -> must not raise
    r = c.retry_or_fail_move('x')
    assert _status(r) != 'failed' and c.issued is False
