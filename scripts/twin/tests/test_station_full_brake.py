"""sim-station-full-brake-place-v1: the FULL base brake for the LOADED collect place descent.

coldwin49 reached the shelf place (PREPLACE_ABOVE_SUPPORT, real positive descent, 0 CARTON_TILT) but the
loaded from-above descent onto the 0.9707 m shelf shoved the yaw-lock-ONLY braked base ~0.08-0.18 m SOUTH
(away from the cart, a pure translational recoil) past the 0.080 m collect_place_base_settle_m stationary
guard, so the place never completed -> arm_not_in_transport_timeout / 'collect stuck 120s'. The forensics
(datasets .../coldwin49-shove.txt) traced it to the station hold leaving TRANSLATION FREE ("weld-loop
relief"). This lane FULL-locks the base (translation + rotation) over the loaded place window, then
returns to the rotation-only hold.

These are pure/CPU tests (no Isaac/USD/py_trees): the window predicate + effort guard
(twin.restock.station_full_brake), the port brake state machine (IsaacPort._apply_base_brake on a mock
mobile), a coldwin49 shove-timeline replay, the effort-guard HOLD abort path, and the config_preflight
key validation.
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock import station_full_brake as sfb
from twin.restock.station_full_brake import LOADED_PLACE_PHASES, EffortGuard, in_loaded_place_window
from twin.restock.isaac_port import IsaacPort, SENSING_ACTIONS
from twin.restock_manipulation.config_preflight import check_station_full_brake


# ============================================================ (0) the pure window predicate
def test_loaded_place_window_is_collect_place_release_retreat():
    assert LOADED_PLACE_PHASES == frozenset({'place', 'release', 'retreat'})
    for phase in ('place', 'release', 'retreat'):
        assert in_loaded_place_window('collect', phase)
    # not the standoff (arm still at preplace), nor stow/verify (empty arm back in transport), nor a
    # different station (transfer/live_pose reuse place/retreat but are NOT the collect deck place).
    for phase in ('preplace', 'stow', 'verify', 'approach', 'grasp', 'lift'):
        assert not in_loaded_place_window('collect', phase)
    assert not in_loaded_place_window('transfer', 'place')
    assert not in_loaded_place_window('couple', 'place')


def test_max_abs_effort_is_tolerant():
    assert sfb.max_abs_effort([1.0, -5.0, 3.0]) == 5.0
    assert sfb.max_abs_effort([1.0, None, 'x', float('nan'), -2.0]) == 2.0
    assert sfb.max_abs_effort(None) is None
    assert sfb.max_abs_effort([]) is None
    assert sfb.max_abs_effort([float('nan')]) is None


# ============================================================ (1) the brake state machine
class _StatefulMobile:
    """A mock MobileController that tracks the sim-only brake state (no USD), recording each real
    transition so the state machine can be asserted: full lock engaged exactly over the place window,
    rotation-only hold elsewhere, and NO engage/release cycling (idempotent re-engage is a no-op)."""

    def __init__(self, *, station_pitch_hold=True, station_place_full_brake=True):
        self.station_pitch_hold = station_pitch_hold
        self.station_place_full_brake = station_place_full_brake
        self.full_locked = False
        self.rotation_hold = False
        self.base_full = False
        self.full_lock_from = None
        self.events = []

    def engage_station_full_lock(self, from_phase=None):   # idempotent while locked (like the real method)
        if not self.full_locked:
            self.full_locked = True
            self.rotation_hold = False           # release_brake() inside the real engage clears the hold
            self.full_lock_from = from_phase     # sim-station-lock-from-preplace-v1: the engage phase
            self.events.append('full_lock_on')

    def release_station_full_lock(self):
        if self.full_locked:
            self.full_locked = False
            self.events.append('full_lock_off')

    def engage_station_hold(self):               # idempotent while latched
        if not self.rotation_hold:
            self.rotation_hold = True
            self.events.append('rot_hold_on')

    def release_station_hold(self):
        if self.rotation_hold:
            self.rotation_hold = False
            self.events.append('rot_hold_off')

    def engage_base_full_brake(self):
        if not self.base_full:
            self.base_full = True
            self.events.append('base_full_on')

    def release_base_full_brake(self):
        if self.base_full:
            self.base_full = False
            self.events.append('base_full_off')


def _drive(phases, **mobile_kw):
    """Step _apply_base_brake through a (action, phase) sequence, recording the locked state at each."""
    port = IsaacPort.__new__(IsaacPort)
    port.mobile = _StatefulMobile(**mobile_kw)
    trace = []
    for action, phase in phases:
        port.current_action = action
        port._phase = phase
        in_arm = action in SENSING_ACTIONS
        port._apply_base_brake(in_arm, True, phase)   # welded through the place window
        trace.append((phase, port.mobile.full_locked, port.mobile.rotation_hold))
    return port.mobile, trace


# A full collect place: standoff -> descend -> set-down/release -> empty withdraw -> stow/verify -> drive.
_COLLECT_PLACE = [('collect', 'preplace'), ('collect', 'place'), ('collect', 'place'),
                  ('collect', 'release'), ('collect', 'retreat'), ('collect', 'stow'),
                  ('collect', 'verify'), ('couple', 'navigate')]


def test_full_lock_engaged_exactly_over_the_place_window():
    mobile, trace = _drive(_COLLECT_PLACE)
    locked = {phase for phase, full, _ in trace if full}
    assert locked == {'place', 'release', 'retreat'}            # lock covers leave-standoff -> withdraw
    # the standoff + the post-withdraw phases hold the rotation-only station brake (not the full lock).
    for phase, full, rot in trace:
        if phase in ('preplace', 'stow', 'verify'):
            assert rot and not full
    # the drive leg (not an arm phase) releases everything.
    assert trace[-1] == ('navigate', False, False)


def test_full_lock_does_not_cycle_one_engage_one_release():
    mobile, _ = _drive(_COLLECT_PLACE)
    assert mobile.events.count('full_lock_on') == 1            # engaged once at place
    assert mobile.events.count('full_lock_off') == 1          # released once at withdraw (stow)
    # rotation-only hold restored AFTER the full lock is released (the withdraw hand-back).
    assert mobile.events.index('full_lock_off') < mobile.events.index('rot_hold_on', 1)


def test_full_brake_off_is_byte_identical_rotation_only_hold():
    # station_place_full_brake OFF: every arm phase takes the rotation-only station hold; the full lock is
    # never engaged (byte-identical to the sim-chassis-pitch-stability-v1 behaviour).
    mobile, trace = _drive(_COLLECT_PLACE, station_place_full_brake=False)
    assert all(not full for _, full, _ in trace)
    assert 'full_lock_on' not in mobile.events
    assert mobile.events.count('rot_hold_on') == 1            # one latched hold across the arm phase


def test_full_lock_only_at_collect_station():
    # a transfer place reuses the 'place'/'retreat' phase names but must NOT take the collect full lock.
    mobile, trace = _drive([('transfer', 'place'), ('transfer', 'retreat')])
    assert all(not full for _, full, _ in trace)
    assert 'full_lock_on' not in mobile.events


# =========================================== (1b) sim-station-lock-from-preplace-v1 engage-phase predicate
def test_resolve_engage_phase_defaults_and_validates():
    # None / '' / the literal 'place' => today's behaviour.
    assert sfb.resolve_engage_phase(None) == 'place'
    assert sfb.resolve_engage_phase('') == 'place'
    assert sfb.resolve_engage_phase('place') == 'place'
    # the opt-in triggers (case / whitespace tolerant).
    assert sfb.resolve_engage_phase('preplace') == 'preplace'
    assert sfb.resolve_engage_phase('  HOLD_FIRM ') == 'hold_firm'
    for bad in ('grasp', 'lift', 'x', 0, True):
        with pytest.raises(ValueError):
            sfb.resolve_engage_phase(bad)


def test_full_lock_window_place_is_todays_window():
    # default 'place': the descent window only; pre-place is NOT locked (rotation-only hold).
    for phase in ('place', 'release', 'retreat'):
        assert sfb.full_lock_window('collect', phase, 'place')
    assert not sfb.full_lock_window('collect', 'preplace', 'place')
    assert not sfb.full_lock_window('collect', 'preplace', 'place', hold_firm_fired=True)
    for phase in ('approach', 'grasp', 'lift', 'stow', 'verify'):
        assert not sfb.full_lock_window('collect', phase, 'place')


def test_full_lock_window_preplace_extends_backward_before_leg_1():
    # 'preplace': locked for the WHOLE pre-place phase (both legs), regardless of hold-firm, plus the
    # descent window -- the base is pinned BEFORE leg 1 starts.
    assert sfb.full_lock_window('collect', 'preplace', 'preplace')
    assert sfb.full_lock_window('collect', 'preplace', 'preplace', hold_firm_fired=False)
    for phase in ('place', 'release', 'retreat'):
        assert sfb.full_lock_window('collect', phase, 'preplace')
    # still never before the arm reaches pre-place.
    assert not sfb.full_lock_window('collect', 'lift', 'preplace')


def test_full_lock_window_hold_firm_arms_at_the_firm_event():
    # 'hold_firm': locked during pre-place only once a carried joint is firmed (end of leg 1), then the
    # descent window as usual.
    assert not sfb.full_lock_window('collect', 'preplace', 'hold_firm', hold_firm_fired=False)
    assert sfb.full_lock_window('collect', 'preplace', 'hold_firm', hold_firm_fired=True)
    for phase in ('place', 'release', 'retreat'):
        assert sfb.full_lock_window('collect', phase, 'hold_firm', hold_firm_fired=False)


def test_full_lock_window_only_collect_station():
    for ep in ('place', 'preplace', 'hold_firm'):
        assert not sfb.full_lock_window('transfer', 'place', ep, hold_firm_fired=True)
        assert not sfb.full_lock_window('couple', 'preplace', ep, hold_firm_fired=True)


# ===================================== (1c) engage-phase state machine (preplace / hold_firm before leg 2)
def _drive_engage(steps, engage_phase):
    """Step _apply_base_brake through (action, phase, hold_firm_fired) steps under an engage_phase key,
    recording the locked state. Mirrors the port's per-tick state: gate() sets _hold_firm_fired before
    advance() -> _apply_base_brake reads it."""
    port = IsaacPort.__new__(IsaacPort)
    port.mobile = _StatefulMobile()
    port._full_brake_engage_phase = engage_phase
    trace = []
    for action, phase, firmed in steps:
        port.current_action = action
        port._phase = phase
        port._hold_firm_fired = firmed
        port._apply_base_brake(action in SENSING_ACTIONS, True, phase)
        trace.append((phase, firmed, port.mobile.full_locked))
    return port.mobile, trace


# coldwin68/69: parked at (-4.61,14.77) -- the same spot as the no-shove reference coldwin64 -- but the
# firm-held carried PRE-PLACE's leg-2 shoulder_pan sweep shoved the base ~0.33 m sideways over sim t~191-227
# (carton came down ~0.5 m off its cart-relative target onto a cage post). Leg 1 settles, the wrists firm
# (PREPLACE_HOLD_FIRM) at the end of leg 1 (~t191), THEN leg 2 sweeps and the shove builds. (t, phase,
# hold_firm_fired, base_dxy_m from the standoff.)
COLDWIN68_PREPLACE = [
    (185.0, 'preplace', False, 0.000),   # leg 1 begins, held at the standoff
    (190.0, 'preplace', False, 0.004),   # leg 1 running (still tiny)
    (191.0, 'preplace', True,  0.006),   # <<< PREPLACE_HOLD_FIRM: wrists firmed, end of leg 1
    (200.0, 'preplace', True,  0.090),   # leg 2 shoulder_pan sweep -> shove building, past 0.080
    (210.0, 'preplace', True,  0.210),
    (227.0, 'preplace', True,  0.330),   # <<< peak shove ~0.33 m (carton ~0.5 m off target)
    (232.0, 'place',    True,  0.332),   # descent BEGINS -- today's 'place' engage point (too late)
]
LEG2_SWEEP_START_DXY = 0.080   # the base is still under this when leg 2 begins at t~200


def test_preplace_engage_locks_before_leg_1_and_before_the_shove():
    mobile, trace = _drive_engage([('collect', ph, f) for _, ph, f, _ in COLDWIN68_PREPLACE], 'preplace')
    # locked from the FIRST pre-place tick (t185, dxy 0.000) -- before leg 1, before any shove.
    first = trace[0]
    assert first == ('preplace', False, True)
    assert all(full for _, _, full in trace)                 # stays locked through place
    assert mobile.events.count('full_lock_on') == 1          # one engage, no cycling
    assert mobile.full_lock_from == 'preplace'               # logged as engaged from pre-place


def test_hold_firm_engage_locks_at_the_firm_event_before_leg_2_sweep():
    mobile, trace = _drive_engage([('collect', ph, f) for _, ph, f, _ in COLDWIN68_PREPLACE], 'hold_firm')
    # NOT locked during leg 1 (before the firm event)...
    assert trace[0] == ('preplace', False, False)
    assert trace[1] == ('preplace', False, False)
    # ...locked from the firm event (t191), and the base is still under 0.080 there -- BEFORE the leg-2
    # sweep shove builds.
    firm_idx = next(i for i, s in enumerate(COLDWIN68_PREPLACE) if s[2])
    assert COLDWIN68_PREPLACE[firm_idx][3] < LEG2_SWEEP_START_DXY
    assert trace[firm_idx][2] is True                        # locked exactly at the firm event
    assert all(full for _, _, full in trace[firm_idx:])      # stays locked
    assert mobile.events.count('full_lock_on') == 1          # one engage, no cycling
    assert mobile.full_lock_from == 'preplace'               # firm fires while phase is pre-place


def test_place_engage_only_locks_at_the_descent_after_the_shove():
    # the default 'place' key reproduces coldwin68: the base is NOT locked through the whole pre-place
    # shove, only once the descent begins at 'place' -- far too late (carton already landed off target).
    mobile, trace = _drive_engage([('collect', ph, f) for _, ph, f, _ in COLDWIN68_PREPLACE], 'place')
    for (_, ph, _firm, _dxy), (_, _, full) in zip(COLDWIN68_PREPLACE, trace):
        assert full == (ph == 'place')
    # the lock engages only at the final 'place' tick, by which point the shove already peaked at 0.33 m.
    assert trace[-1][2] is True and trace[-1][0] == 'place'
    shove_before_place = max(dxy for _, ph, _f, dxy in COLDWIN68_PREPLACE if ph == 'preplace')
    assert shove_before_place == pytest.approx(0.330)
    assert mobile.events.count('full_lock_on') == 1


def test_engage_phase_default_is_byte_identical_to_todays_place():
    # omitting the key entirely (getattr default) == engage_phase 'place' == today's behaviour.
    port = IsaacPort.__new__(IsaacPort)
    port.mobile = _StatefulMobile()
    # no _full_brake_engage_phase attr set, no _hold_firm_fired attr set (the __new__ bare port)
    for _, ph, _firm, _dxy in COLDWIN68_PREPLACE:
        port.current_action = 'collect'
        port._phase = ph
        port._apply_base_brake(True, True, ph)
    # locked only at the 'place' descent (pre-place never locked) -- the pre-key behaviour.
    assert port.mobile.events.count('full_lock_on') == 1
    assert port.mobile.full_lock_from == 'place'             # only 'place' triggered a lock


class _HoldFirmStub:
    """Stand-in for HoldFirmOnArrival: fires the given joint on the Nth observe (the end-of-leg-1 arrival)."""

    def __init__(self, fire_on_tick):
        self.fire_on_tick = fire_on_tick
        self.n = 0
        self.resets = 0

    def observe(self, cmd, measured):
        self.n += 1
        return ['ur_wrist_2_joint'] if self.n == self.fire_on_tick else []

    def reset(self):
        self.resets += 1


def test_apply_hold_firm_arms_and_clears_the_full_lock_latch():
    # the integration seam: _apply_hold_firm SETS _hold_firm_fired when a carried joint is firmed (so
    # _apply_base_brake's 'hold_firm' trigger fires), and CLEARS it on leaving pre-place (a later op re-arms).
    port = IsaacPort.__new__(IsaacPort)
    port._hold_firm = _HoldFirmStub(fire_on_tick=2)
    port._hold_firm_cfg = {'firm': {'kp': 1e6, 'kd': 1e4, 'eff': 2000.0}}
    port._hold_firm_phase = None
    port._hold_firm_fired = False
    port.robot = NS()                 # non-None; set_arm_drive_gains_per_joint raises + is caught (latch set first)
    port.native_transport = None
    port.command_gate = NS(_last_cmd_q=(0.0,) * 6, measured=lambda: (0.0,) * 6)
    # leg 1, tick 1: no joint firmed yet -> latch stays disarmed.
    port._apply_hold_firm('preplace')
    assert port._hold_firm_fired is False
    # tick 2: the wrist firms (PREPLACE_HOLD_FIRM) -> latch arms, before the leg-2 sweep.
    port._apply_hold_firm('preplace')
    assert port._hold_firm_fired is True
    # leaving pre-place (descent) clears the latch (place/release/retreat lock via the descent window).
    port._apply_hold_firm('place')
    assert port._hold_firm_fired is False
    assert port._hold_firm.resets >= 1


def test_apply_hold_firm_noop_when_off():
    # config off (_hold_firm is None) => never touches the latch => byte-identical.
    port = IsaacPort.__new__(IsaacPort)
    port._hold_firm = None
    port.robot = NS()
    port._hold_firm_fired = False
    port._hold_firm_phase = None
    port._apply_hold_firm('preplace')
    assert port._hold_firm_fired is False


# ============================================================ (2) coldwin49 shove-timeline replay
# Measured from coldwin49 nav-trace (datasets .../coldwin49-shove.txt). Base displacement (m) from the
# pre-place standoff vs the controller phase at that time. Reference base=(-4.610,14.768) at the standoff.
# Descent begins ~t193 (place); the carton meets the 0.97 shelf and the weld loop fights (release) ~t204+;
# the empty arm would retreat afterward. Every sample where the strict 0.080 guard would trip lies within
# the place/release window -> the full lock (place..retreat) covers the entire shove interval.
COLDWIN49_SHOVE = [
    # (t,     phase,      disp_m)
    (185.0, 'preplace', 0.000),   # held at the standoff, pre-descent
    (192.5, 'preplace', 0.000),   # descent not yet begun
    (193.0, 'place',    0.006),   # descent / press BEGINS (arm leaves the standoff)
    (194.5, 'place',    0.059),   # still under the 0.080 guard
    (195.08, 'place',   0.080),   # <<< 0.080 CROSSED, carton still ~0.22 m ABOVE the shelf
    (197.0, 'place',    0.148),
    (204.0, 'place',    0.149),
    (204.5, 'place',    0.178),   # parked flips False; peak pre-contact shove
    (211.0, 'release',  0.218),
    (215.0, 'release',  0.234),   # carton z starts to blow up (weld-loop fight)
    (221.0, 'release',  0.386),
    (224.9, 'release',  0.746),   # MAX (carton launched; contact numerically unstable)
    (230.0, 'retreat',  0.452),   # settles back; arm never stows -> 120 s no-progress fast-fail
]
COLLECT_PLACE_BASE_SETTLE_M = 0.080   # unchanged by this lane (full lock makes the real drift ~0)


def test_coldwin49_strict_guard_trips_on_the_recorded_shove():
    # the recorded shove crosses AND far exceeds the 0.080 stationary-guard tolerance -- this is the
    # coldwin49 blocker (the yaw-lock-only brake left translation free).
    peak = max(d for _, _, d in COLDWIN49_SHOVE)
    assert peak == pytest.approx(0.746)
    # reaches the 0.080 guard at t=195.08 (the documented crossing) and strictly exceeds it just after.
    crossed = [(t, ph, d) for t, ph, d in COLDWIN49_SHOVE if d >= COLLECT_PLACE_BASE_SETTLE_M]
    assert crossed and crossed[0][0] == pytest.approx(195.08)
    over = [(t, ph, d) for t, ph, d in COLDWIN49_SHOVE if d > COLLECT_PLACE_BASE_SETTLE_M]
    assert over, 'the shove strictly exceeds the settle tolerance'


def test_full_lock_window_covers_every_shove_sample():
    # every recorded tick where the base drifted past the 0.080 guard is inside the collect place window
    # (place/release/retreat) -> the full lock (engaged at 'place', held through 'retreat') is active for
    # the ENTIRE shove interval, so the base is pinned before and throughout the recoil.
    for t, phase, disp in COLDWIN49_SHOVE:
        if disp > COLLECT_PLACE_BASE_SETTLE_M:
            assert in_loaded_place_window('collect', phase), (t, phase, disp)
    # and the lock engages BEFORE the shove: the first in-window ('place') sample is still ~0 displacement
    # (base at the standoff), i.e. the lock is on well under the 0.080 guard.
    first_place = next(d for _, ph, d in COLDWIN49_SHOVE if ph == 'place')
    assert first_place < COLLECT_PLACE_BASE_SETTLE_M


def test_settle_tolerance_unchanged_by_this_lane():
    # item 3: collect_place_base_settle_m stays 0.080 (a wider tolerance would drift the place target
    # 17 cm off the shelf); the fix is the brake, not a looser guard.
    assert COLLECT_PLACE_BASE_SETTLE_M == 0.080


# ============================================================ (3) the weld-loop effort guard
def test_effort_guard_latches_abort_after_consecutive_overlimit():
    g = EffortGuard(100.0, consecutive=3)
    assert g.observe(120.0, True) == 'ok'       # 1st over
    assert g.observe(130.0, True) == 'ok'       # 2nd over
    assert g.observe(110.0, True) == 'abort'    # 3rd over -> latch
    assert g.aborted and g.peak == pytest.approx(130.0)
    assert g.observe(10.0, True) == 'abort'     # stays latched
    assert 'weld-loop effort limit exceeded' in g.reason


def test_effort_guard_resets_streak_below_limit_and_out_of_window():
    g = EffortGuard(100.0, consecutive=3)
    g.observe(150.0, True); g.observe(150.0, True)
    assert g.observe(50.0, True) == 'ok'        # a dip resets the streak
    assert g.observe(150.0, True) == 'ok'       # count restarts
    g.observe(150.0, True)
    assert g.observe(10.0, False) == 'ok'       # leaving the window also resets (no abort)
    assert not g.aborted


def test_effort_guard_inert_without_limit_or_reading():
    assert EffortGuard(None).observe(1e9, True) == 'ok'     # unconfigured => never trips
    g = EffortGuard(100.0, consecutive=1)
    assert g.observe(None, True) == 'ok'                    # no reading => cannot judge
    assert g.observe(10.0, False) == 'ok'                   # out of window
    with pytest.raises(ValueError):
        EffortGuard(0.0)
    with pytest.raises(ValueError):
        EffortGuard(-5.0)


class _RecMobile:
    station_place_full_brake = True

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append(name)


def _monitor_port(limit, efforts, *, phase='place', action='collect', tick=30, consecutive=2):
    port = IsaacPort.__new__(IsaacPort)
    port.mobile = _RecMobile()
    port.current_action = action
    port._phase = phase
    port.tick = tick
    port._welded_stop = False
    port._welded_stop_reason = None
    port._arm_joints = ('j1', 'j2')
    port.robot = NS(get_measured_joint_efforts=lambda: efforts, dof_names=['j1', 'j2'])
    port._full_brake_guard = EffortGuard(limit, consecutive=consecutive) if limit else None
    port.command_gate = NS(suppress_heartbeat=False)
    port.holds = []
    port.hold_arm = lambda: port.holds.append('hold')
    for name in ('_station_full_brake_monitor', '_read_arm_effort', 'enter_welded_controlled_stop'):
        setattr(port, name, getattr(IsaacPort, name).__get__(port, IsaacPort))
    return port


def test_monitor_holds_on_sustained_effort_spike_not_a_shove():
    # a sustained over-limit spike in the locked place window ABORTS to a HOLD: the base stays FULL-braked
    # and the arm is held (the controlled-stop path) -- never a brake release (which would shove the base).
    port = _monitor_port(100.0, [150.0, 160.0], consecutive=2)
    port._station_full_brake_monitor()
    assert not port._welded_stop                      # one spike is not enough (consecutive=2)
    port._station_full_brake_monitor()
    assert port._welded_stop is True                  # latched HOLD
    assert 'latch_welded_hold' in port.mobile.calls   # base pinned
    assert 'engage_base_full_brake' in port.mobile.calls
    assert port.holds == ['hold']                     # arm held energised
    assert port.command_gate.suppress_heartbeat is True
    assert 'weld-loop effort limit exceeded' in port._welded_stop_reason


def test_monitor_tolerates_below_limit_and_out_of_window():
    # below the limit: never aborts, however long the place runs.
    port = _monitor_port(100.0, [40.0, 50.0], consecutive=2)
    for _ in range(10):
        port._station_full_brake_monitor()
    assert not port._welded_stop
    # out of the window (stow): a huge effort does not trip the guard.
    port2 = _monitor_port(100.0, [999.0, 999.0], phase='stow', consecutive=1)
    port2._station_full_brake_monitor()
    port2._station_full_brake_monitor()
    assert not port2._welded_stop


def test_monitor_noop_when_guard_unconfigured():
    port = _monitor_port(None, [999.0])            # no limit -> _full_brake_guard is None
    port._station_full_brake_monitor()
    assert not port._welded_stop
    assert port.mobile.calls == []


def test_read_arm_effort_best_effort_degrades_to_none():
    # no articulation getter -> None (the guard cannot judge, keeps placing).
    port = IsaacPort.__new__(IsaacPort)
    port.robot = NS()
    port._arm_joints = ('j1',)
    assert port._read_arm_effort() is None
    # a raising getter -> None (never crashes the tick).
    def boom():
        raise RuntimeError('sensor glitch')
    port.robot = NS(get_measured_joint_efforts=boom)
    assert port._read_arm_effort() is None


# ============================================================ (4) config_preflight validation
def test_preflight_off_is_noop():
    assert check_station_full_brake({}) == []
    assert check_station_full_brake({'station_place_full_brake': False}) == []
    assert check_station_full_brake(None) == []


def test_preflight_refuses_full_brake_without_pitch_hold():
    problems = check_station_full_brake({'station_place_full_brake': True})
    assert problems and 'chassis_station_pitch_hold' in problems[0]


def test_preflight_accepts_full_brake_with_pitch_hold():
    assert check_station_full_brake(
        {'station_place_full_brake': True, 'chassis_station_pitch_hold': True}) == []
    assert check_station_full_brake(
        {'station_place_full_brake': True, 'chassis_station_pitch_hold': True,
         'station_place_full_brake_effort_limit': 250.0}) == []


def test_preflight_refuses_bad_effort_limit():
    for bad in (-3, 0, 'x', float('inf')):
        problems = check_station_full_brake(
            {'station_place_full_brake': True, 'chassis_station_pitch_hold': True,
             'station_place_full_brake_effort_limit': bad})
        assert any('effort_limit' in p for p in problems), bad


# ==================================== (4b) sim-station-lock-from-preplace-v1 engage-phase key validation
_WITH_BRAKE = {'station_place_full_brake': True, 'chassis_station_pitch_hold': True}


def test_preflight_accepts_place_and_preplace_engage_with_brake():
    assert check_station_full_brake(dict(_WITH_BRAKE, station_full_brake_engage_phase='place')) == []
    assert check_station_full_brake(dict(_WITH_BRAKE, station_full_brake_engage_phase='preplace')) == []


def test_preflight_refuses_unknown_engage_phase():
    problems = check_station_full_brake(dict(_WITH_BRAKE, station_full_brake_engage_phase='grasp'))
    assert any('station_full_brake_engage_phase' in p for p in problems)


def test_preflight_refuses_hold_firm_without_hold_firm_config():
    # 'hold_firm' needs the pre-place hold-firm-on-arrival config or its trigger never fires.
    problems = check_station_full_brake(dict(_WITH_BRAKE, station_full_brake_engage_phase='hold_firm'))
    assert any('hold_firm_on_arrival' in p for p in problems)
    # a disabled block is still refused.
    problems = check_station_full_brake(dict(
        _WITH_BRAKE, station_full_brake_engage_phase='hold_firm',
        phase_drive={'hold_firm_on_arrival': {'enabled': False}}))
    assert any('hold_firm_on_arrival' in p for p in problems)


def test_preflight_accepts_hold_firm_with_hold_firm_config():
    assert check_station_full_brake(dict(
        _WITH_BRAKE, station_full_brake_engage_phase='hold_firm',
        phase_drive={'hold_firm_on_arrival': {'enabled': True}})) == []


def test_preflight_refuses_engage_phase_without_the_brake():
    # a dangling engage key without the full brake would silently never activate -- refuse it.
    problems = check_station_full_brake({'station_full_brake_engage_phase': 'preplace'})
    assert problems and 'station_place_full_brake' in problems[0]
    # but the absent / default 'place' key is a no-op even without the brake (byte-identical).
    assert check_station_full_brake({'station_full_brake_engage_phase': 'place'}) == []
    assert check_station_full_brake({}) == []
    # a dangling UNKNOWN value is still refused (and names the bad value).
    problems = check_station_full_brake({'station_full_brake_engage_phase': 'bogus'})
    assert problems and 'station_place_full_brake' in problems[0]
