"""sim-robust-phase-timeout-during-recovery-v1 -- CPU proof (no Isaac/ROS/GPU) that a NAMED move_group
crash recovery does not get charged against the controller's per-phase timeout, so a recovered crash
cannot convert into 'phase timeout: <phase>'.

Root cause (respawn2b, ~/datasets/twin-robot-in-store-boot-v1/mission/respawn2b): two move_group SIGABRT
deaths inside the collect LIFT consumed ~119 of the 120 sim-s lift budget (orphaned-goal waits + re-sends)
while nothing paused the phase clock, so TransferController.step failed the phase 'phase timeout: lift'
even though each crash had recovered. The recovery flowed through the launch respawn supervisor + the
controller's SIM-deadline MOVE_DEADLINE_RETRY path, NOT the adapter's _awaiting_server_respawn crash-retry
(the launch respawned inside the render-paced poll gap), so the crash signal must also cover that path.

Fix: while motion.crash_recovery_active() is True the step credits that sim time out of the timeout
comparison (elapsed - credit vs phase_timeout), bounded by _recovery_pause_cap_s so a hung recovery still
fails fast BY NAME; no recovery => credit 0 => byte-identical.

Run under: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
    scripts/twin/tests/test_restock_phase_timeout_recovery.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from twin.restock_manipulation import ros_adapter as ra  # noqa: E402
from twin.restock_manipulation.controller import RECOVERY_PAUSE_CAP_S  # noqa: E402
# Reuse the controller-tier harness that every other manipulation test uses.
from scripts.twin.tests.test_restock_manipulation import make_controller, packet, run_to  # noqa: E402


# --------------------------------------------------------------------------------------------------
# Controller tier: the phase-clock credit while the adapter reports a named crash recovery
# --------------------------------------------------------------------------------------------------
def _hold_at_approach(phase_timeout=10.0):
    """A collect controller parked in the (already-issued) 'approach' move phase with the motion stub
    returning 'running' forever, so we can feed ticks and drive crash_recovery_active() by hand."""
    c, m = make_controller('collect', phase_timeout=phase_timeout)
    t = run_to(c, 'approach')
    m.outcome = 'running'
    assert c.phase == 'approach'
    return c, m, t


def _feed(c, m, t0, n, recovering):
    """Step the controller n ticks of dt=0.1, with crash_recovery_active() reporting `recovering`.
    Returns the next t0. Asserts the phase stays 'approach' and never fails (the caller tests failure)."""
    m._recovering = recovering
    m.crash_recovery_active = lambda: m._recovering
    for i in range(n):
        r = c.step(0.1, packet(c, t0 + (i + 1) * 0.1))
        assert r.status != 'failed', r.reason
        assert c.phase == 'approach'
    return t0 + n * 0.1


def test_two_recovery_windows_keep_a_phase_within_budget():
    # phase_timeout 10 sim-s. Real (non-recovery) time = 2 + 2 + 3 = 7 sim-s < 10; two 8 sim-s recovery
    # windows (16 sim-s) are credited out. Total elapsed 23 sim-s -> would have tripped the legacy
    # `elapsed > phase_timeout` twice over, but the credited pause keeps the phase alive.
    c, m, t = _hold_at_approach(phase_timeout=10.0)
    t = _feed(c, m, t, 20, recovering=False)   # 2.0 real
    t = _feed(c, m, t, 80, recovering=True)    # 8.0 credited
    t = _feed(c, m, t, 20, recovering=False)   # 2.0 real
    t = _feed(c, m, t, 80, recovering=True)    # 8.0 credited
    t = _feed(c, m, t, 30, recovering=False)   # 3.0 real
    assert c.phase == 'approach'
    assert c.elapsed > c.phase_timeout                     # the raw clock DID pass the budget ...
    assert c._recovery_credit_s == pytest.approx(16.0, abs=1e-6)   # ... but 16 s was credited to recovery
    assert c.elapsed - c._recovery_credit_s < c.phase_timeout      # so the effective budget is intact


def test_hung_recovery_fails_fast_by_name_at_the_recovery_bound():
    # A recovery that never ends must still terminate BY NAME at the pause bound -- not pause forever, and
    # not surface as the opaque 'phase timeout'. Tighten the bound so the test is quick.
    c, m, t = _hold_at_approach(phase_timeout=10.0)
    c._recovery_pause_cap_s = 5.0
    m._recovering = True
    m.crash_recovery_active = lambda: True
    r = None
    for i in range(500):
        r = c.step(0.1, packet(c, t + (i + 1) * 0.1))
        if c.phase == 'failed':
            break
    assert c.phase == 'failed'
    assert 'crash recovery exceeded' in r.reason            # named by the recovery bound
    assert 'phase timeout' not in r.reason                  # NOT the opaque phase timeout
    assert c._recovery_credit_s == pytest.approx(5.0)       # credit capped at the bound
    # The fail fired at the bound, long before the raw clock would have hit phase_timeout (10 s): the
    # effective (credited) elapsed is still ~0 when the bound trips.
    assert c.elapsed - c._recovery_credit_s < 1.0


def test_no_recovery_run_has_identical_elapsed_accounting():
    # With no crash recovery (the probe absent on the stub), the phase times out EXACTLY when the raw
    # elapsed first exceeds phase_timeout -- byte-identical to the legacy `elapsed > phase_timeout`.
    c, m, t = _hold_at_approach(phase_timeout=5.0)
    assert not hasattr(m, 'crash_recovery_active')          # probe absent -> credit never accrues
    r = None
    for i in range(200):
        r = c.step(0.1, packet(c, t + (i + 1) * 0.1))
        if c.phase == 'failed':
            break
    assert c.phase == 'failed'
    assert r.reason == 'phase timeout: approach'
    assert c._recovery_credit_s == 0.0
    # Fired the first tick the raw clock passed 5.0 s (~5.1 with fp accumulation) -- unchanged timing.
    assert 5.0 < c.elapsed < 5.25


def test_crash_recovery_active_false_leaves_the_clock_untouched():
    # A probe that is present but always False must also be byte-identical: credit stays 0.
    c, m, t = _hold_at_approach(phase_timeout=5.0)
    m.crash_recovery_active = lambda: False
    r = None
    for i in range(200):
        r = c.step(0.1, packet(c, t + (i + 1) * 0.1))
        if c.phase == 'failed':
            break
    assert c.phase == 'failed'
    assert r.reason == 'phase timeout: approach'
    assert c._recovery_credit_s == 0.0


def test_recovery_pause_cap_zero_disables_the_feature():
    # The escape hatch: cap 0 => the credit path is skipped entirely => byte-identical to no fix even if
    # the adapter reports recovery.
    c, m, t = _hold_at_approach(phase_timeout=5.0)
    c._recovery_pause_cap_s = 0.0
    m.crash_recovery_active = lambda: True
    r = None
    for i in range(200):
        r = c.step(0.1, packet(c, t + (i + 1) * 0.1))
        if c.phase == 'failed':
            break
    assert c.phase == 'failed'
    assert r.reason == 'phase timeout: approach'   # not the recovery-bound reason
    assert c._recovery_credit_s == 0.0


def test_default_cap_is_generous_and_env_tunable(monkeypatch):
    # Default covers a real multi-death recovery (respawn2b credited ~119 s); well above it.
    c, _ = make_controller('collect')
    assert c._recovery_pause_cap_s == RECOVERY_PAUSE_CAP_S == 300.0


# --------------------------------------------------------------------------------------------------
# Adapter tier: crash_recovery_active() and the move_group liveness latch (respawn2b path)
# --------------------------------------------------------------------------------------------------
class _FakePlan:
    """A move_group (plan) action-client stub whose readiness we toggle to model a SIGABRT + respawn."""
    def __init__(self, ready=True):
        self.ready = ready

    def server_is_ready(self):
        return self.ready


def _bare_port():
    """A RosMotionPort with no ROS node (its recovery flags default via getattr)."""
    return ra.RosMotionPort.__new__(ra.RosMotionPort)


def test_crash_recovery_active_defaults_false():
    assert _bare_port().crash_recovery_active() is False


def test_crash_recovery_active_from_respawn_hold_and_reattach():
    p = _bare_port()
    p._awaiting_server_respawn = True
    assert p.crash_recovery_active() is True
    p._awaiting_server_respawn = False
    p._reattach_active = True
    assert p.crash_recovery_active() is True


def test_liveness_latch_catches_a_death_the_crash_retry_path_misses():
    # respawn2b: the launch respawns move_group inside the poll gap, so _awaiting_server_respawn never
    # sets. The plan-server ready->not-ready transition during an active move latches the death instead.
    p = _bare_port()
    p.plan = _FakePlan(ready=True)
    p._track_movegroup_liveness()                      # baseline: ready
    assert p.crash_recovery_active() is False
    p.plan.ready = False                               # move_group SIGABRT
    p._track_movegroup_liveness()
    assert p._movegroup_death_seen is True
    assert p.crash_recovery_active() is True
    p.plan.ready = True                                # launch respawned it -> the latch HOLDS (the
    p._track_movegroup_liveness()                      # orphaned goal is still being recovered)
    assert p.crash_recovery_active() is True


def test_liveness_latch_noop_without_a_plan_client():
    # A port mid-teardown / a stub without server_is_ready must be a safe no-op (never raise, never set).
    p = _bare_port()
    p.plan = SimpleNamespace()                          # no server_is_ready
    p._track_movegroup_liveness()
    assert p.crash_recovery_active() is False


def test_reset_crash_budget_clears_the_latch():
    p = _bare_port()
    p._movegroup_death_seen = True
    p.reset_crash_budget()
    assert p._movegroup_death_seen is False
    assert p.crash_recovery_active() is False


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
