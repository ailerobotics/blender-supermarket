"""CPU tests for the coupled-tow traction core (no Isaac/ROS/GPU) -- sim-robust-coupled-traction-v1.

Covers the recorded-trace acceptance grader (coupled_tow_margin) and the quasi-static lever-direction
budget (drive_wheel_normal_share / static_traction_budget). The grader's separation of the REAL stalled
vs clean runs is exercised end-to-end by scripts/check_tow_traction.py; here it is pinned on synthetic
rows so the invariant is fast and offline.
"""
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.traction import (coupled_tow_margin, drive_wheel_normal_share,
                           static_traction_budget, TowMargin)

GOAL = (-2.0, 15.25, 0.0)


def _row(x, t, *, coupled=True, parked=False, lin=0.22, y=15.25):
    return dict(action='outbound', x=x, y=y, coupled=coupled, parked=parked, lin=lin, world_t=t)


def _clean_tow():
    """A healthy tow: advances +x from -4.46 to the goal at ~0.22 m/s, one row per 0.1 s sim."""
    rows, x, t = [], -4.46, 0.0
    while x <= -2.0 + 1e-9:
        rows.append(_row(round(x, 3), round(t, 2)))
        x += 0.022; t += 0.1
    return rows


def _stalled_tow(pin_s=20.0):
    """Advances 0.52 m to x=-3.93, then pins there for pin_s sim seconds (the outbound1 signature)."""
    rows, x, t = [], -4.46, 0.0
    while x < -3.93:
        rows.append(_row(round(x, 3), round(t, 2)))
        x += 0.022; t += 0.1
    end = t + pin_s
    while t <= end:
        rows.append(_row(-3.932, round(t, 2), parked=True, lin=0.02))
        t += 0.1
    return rows


def test_clean_tow_passes_gate():
    m = coupled_tow_margin(_clean_tow(), GOAL)
    assert isinstance(m, TowMargin)
    assert m.reached_goal and m.passed
    assert m.longest_pin_s < 5.0
    assert m.max_advance_m > 2.0
    assert m.reason == 'passed'


def test_stalled_tow_fails_gate():
    m = coupled_tow_margin(_stalled_tow(pin_s=20.0), GOAL)
    assert not m.reached_goal and not m.passed
    assert m.longest_pin_s >= 15.0           # the ~24 s outbound1 pin, reproduced
    assert m.max_advance_m == pytest.approx(0.53, abs=0.03)
    assert m.reason == 'never_reached_goal'


def test_reached_goal_but_long_pin_still_fails():
    """A tow that eventually crawls to the goal but pinned for many sim seconds is NOT robust."""
    rows = _stalled_tow(pin_s=12.0)
    x, t = -3.93, rows[-1]['world_t'] + 0.1
    while x <= -2.0 + 1e-9:                   # then it finally crawls in
        rows.append(_row(round(x, 3), round(t, 2)))
        x += 0.022; t += 0.1
    m = coupled_tow_margin(rows, GOAL)
    assert m.reached_goal
    assert not m.passed and m.longest_pin_s >= 10.0


def test_empty_or_uncoupled_is_fail_not_silent_pass():
    assert coupled_tow_margin([], GOAL).reason == 'no_coupled_rows'
    uncoupled = [_row(-4.46, 0.0, coupled=False), _row(-4.4, 0.1, coupled=False)]
    assert not coupled_tow_margin(uncoupled, GOAL).passed


def test_coupling_gap_reanchors_so_a_legitimate_wait_is_not_charged():
    """A long uncoupled wait between two clean coupled drives must not count as a pin."""
    rows = _clean_tow()[:10]                  # drive a bit
    t = rows[-1]['world_t']
    rows += [_row(rows[-1]['x'], round(t + 1 + i * 0.1, 2), coupled=False) for i in range(100)]  # 10 s gap
    x, t = rows[10]['x'], t + 12
    while x <= -2.0 + 1e-9:
        rows.append(_row(round(x, 3), round(t, 2)))
        x += 0.022; t += 0.1
    m = coupled_tow_margin(rows, GOAL)
    assert m.reached_goal and m.passed and m.longest_pin_s < 5.0


def test_bad_params_raise():
    with pytest.raises(ValueError):
        coupled_tow_margin(_clean_tow(), GOAL, pin_fault_s=0.0)


# ---- quasi-static budget (lever direction) ------------------------------------------------------

def test_drive_wheel_share_monotone_and_clamped():
    assert drive_wheel_normal_share(0.0) == pytest.approx(0.5)
    assert drive_wheel_normal_share(0.03) == pytest.approx(0.95)       # past lift_unload -> clamped
    assert drive_wheel_normal_share(1.0) == pytest.approx(0.95)        # saturates, never exceeds
    lifts = [0.0, 0.005, 0.01, 0.02, 0.03]
    shares = [drive_wheel_normal_share(l) for l in lifts]
    assert all(b >= a for a, b in zip(shares, shares[1:]))             # non-decreasing
    with pytest.raises(ValueError):
        drive_wheel_normal_share(-0.01)


def test_budget_non_decreasing_in_each_lever():
    base = dict(wheel_friction=1.0, caster_lift_m=0.005, drive_max_force_nm=80.0)
    m0 = static_traction_budget(**base)['margin']
    # Each lever is non-decreasing (never hurts).
    assert static_traction_budget(**{**base, 'caster_lift_m': 0.03})['margin'] >= m0
    assert static_traction_budget(**{**base, 'wheel_friction': 1.5})['margin'] >= m0
    assert static_traction_budget(**{**base, 'drive_max_force_nm': 200.0})['margin'] >= m0


def test_torque_cap_binds_at_80nm_so_caster_lift_needs_the_drive_bump():
    """Key finding: at the stuck 80 N.m the drive is TORQUE-capped (2*80/0.16 = 1000 N) below the friction
    limit even at full wheel_friction, so caster lift ALONE cannot raise the steady budget -- the drive
    force must come up too. Raising the force unbinds the cap and lets caster lift help."""
    stuck = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.005, drive_max_force_nm=80.0)
    lift_only = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.03, drive_max_force_nm=80.0)
    assert stuck['binding_cap'] == 'drive_torque' and lift_only['binding_cap'] == 'drive_torque'
    assert lift_only['margin'] == pytest.approx(stuck['margin'])     # caster lift alone: no change (capped)
    # Raise drive force: the torque cap lifts and the margin strictly increases.
    force_up = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.03, drive_max_force_nm=200.0)
    assert force_up['margin'] > lift_only['margin']
    # With the force high enough, friction binds and caster lift THEN strictly helps.
    lo_lift = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.005, drive_max_force_nm=400.0)
    hi_lift = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.03, drive_max_force_nm=400.0)
    assert lo_lift['binding_cap'] == 'friction' and hi_lift['binding_cap'] == 'friction'
    assert hi_lift['margin'] > lo_lift['margin']


def test_recommended_config_strictly_dominates_stuck():
    stuck = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.005, drive_max_force_nm=80.0)
    rec = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.03, drive_max_force_nm=200.0)
    assert rec['margin'] > stuck['margin']
    assert rec['drive_wheel_share'] > stuck['drive_wheel_share']


def test_budget_rejects_bad_inputs():
    for bad in (dict(wheel_friction=0.0), dict(drive_max_force_nm=0.0), dict(wheel_radius_m=0.0)):
        with pytest.raises(ValueError):
            static_traction_budget(caster_lift_m=0.03, **{'wheel_friction': 1.0, **bad})
