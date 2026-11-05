"""sim-preplace-per-joint-gains-v1: per-joint carried-preplace drive override (stiffen shoulder_pan, keep
the wrists soft), the frozen-joint convergence guard, the wrist-first sequencing fallback, and the
six-joint drive bench.

coldwin53 reached the carried preplace on the SOFT drive (the swing fix: CARTON_TICK tilt 1.83 deg, no
CARTON_TILT trip), but shoulder_pan could not HOLD its goal against the 1.64 rad wrist_2 sweep: it drifted
back +0.479 and FROZE 0.558 rad short -> 'phase timeout: preplace'. cw50-52 (STIFF all joints) held the pan
but swung the carton 3-8 deg. The pan HOLD and the carton SWING want opposite gains; the fix is a PER-JOINT
override (shoulder_pan stiff, wrists soft). These tests are CPU-only (no Isaac/ROS/GPU).
"""
import json
import os
import time

import numpy as np
import pytest

from twin.restock.physics_setup import (phase_drive_gains, phase_drive_plan, resolve_joint_overrides,
                                        set_arm_drive_gains_per_joint, _ARM_JOINT_NAMES, _WRIST_JOINT_NAMES)
from twin.restock_manipulation import config_preflight as cp
from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.restock_manipulation.convergence_guard import ConvergenceGuard
from twin.restock_manipulation.preplace_sequence import wrist_first_subgoals
from twin.restock_manipulation import (Observation, ProductProfile, Sensors, Target, TransferController)
import twin.tools.preplace_drive_bench as bench

SOFT = {'kp': 3e5, 'kd': 8e3, 'eff': 400.0}
STIFF = {'kp': 1e6, 'kd': 1e4, 'eff': 2000.0}
PAN = 'ur_shoulder_pan_joint'
# The lane's fix: preplace SOFT (swing killed) + shoulder_pan force-stiffened (holds against the sweep).
FIX_PD = {'soft': SOFT, 'stiff': STIFF, 'carried_soft_phases': ['preplace'],
          'carried_joint_profile': {'preplace': {PAN: dict(STIFF)}}}

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'preplace_sweep_commanded.json')
URDF = os.path.join(os.path.dirname(__file__), 'fixtures', 'collect_stow_arm.urdf')


# ------------------------------------------------- 1) per-joint drive plan resolution (physics_setup)

def test_phase_drive_plan_stiffens_pan_keeps_wrists_soft():
    plan = phase_drive_plan('preplace', FIX_PD, _ARM_JOINT_NAMES)
    assert plan[PAN] == STIFF                       # shoulder_pan force-stiffened
    for jn in _WRIST_JOINT_NAMES:
        assert plan[jn] == SOFT                     # wrists stay soft (swing fix kept)
    assert plan['ur_shoulder_lift_joint'] == SOFT and plan['ur_elbow_joint'] == SOFT


def test_phase_drive_plan_place_stays_uniform_stiff():
    # place is not force-soft and has no override -> uniform stiff on every joint (unchanged descent).
    plan = phase_drive_plan('place', FIX_PD, _ARM_JOINT_NAMES)
    assert all(v == STIFF for v in plan.values())


def test_phase_drive_plan_byte_identical_without_override():
    # Absent carried_joint_profile => the plan is the uniform phase gains on every joint (physically the
    # same as the old set_arm_drive_gains path).
    pd = {'soft': SOFT, 'stiff': STIFF, 'carried_soft_phases': ['preplace']}
    plan = phase_drive_plan('preplace', pd, _ARM_JOINT_NAMES)
    assert all(v == SOFT for v in plan.values())
    assert resolve_joint_overrides('preplace', pd) == {}
    assert phase_drive_plan('preplace', None, _ARM_JOINT_NAMES) is None


def test_resolve_joint_overrides_validates_names_and_gains():
    with pytest.raises(ValueError):
        resolve_joint_overrides('preplace', {'soft': SOFT, 'carried_joint_profile':
                                             {'preplace': {'not_a_joint': dict(STIFF)}}})
    with pytest.raises(ValueError):
        resolve_joint_overrides('preplace', {'soft': SOFT, 'carried_joint_profile':
                                             {'preplace': {PAN: {'kp': -1, 'kd': 0, 'eff': 1}}}})


class _FakeCtrl:
    def __init__(self, n):
        self.kps, self.kds = np.zeros(n), np.zeros(n)
        self.effs = self.eff_idx = None

    def get_gains(self):
        return self.kps, self.kds

    def set_gains(self, kps, kds):
        self.kps, self.kds = np.asarray(kps, float), np.asarray(kds, float)

    def set_max_efforts(self, effs, joint_indices=None):
        self.effs, self.eff_idx = np.asarray(effs, float), joint_indices


class _FakeRobot:
    _DOF = {jn: i for i, jn in enumerate(ARM_JOINTS)}
    _DOF['base'] = 6

    def __init__(self):
        self.ctrl = _FakeCtrl(len(self._DOF))

    def get_dof_index(self, n):
        return self._DOF[n]

    def get_articulation_controller(self):
        return self.ctrl


def test_set_arm_drive_gains_per_joint_applies_pan_stiff_wrists_soft():
    robot = _FakeRobot()
    plan = phase_drive_plan('preplace', FIX_PD, ARM_JOINTS)
    set_arm_drive_gains_per_joint(robot, plan)
    pan_i = ARM_JOINTS.index(PAN)
    w2_i = ARM_JOINTS.index('ur_wrist_2_joint')
    assert robot.ctrl.kps[pan_i] == 1e6 and robot.ctrl.kps[w2_i] == 3e5
    assert robot.ctrl.kps[6] == 0.0                # non-arm dof untouched
    effs = {int(i): e for i, e in zip(robot.ctrl.eff_idx, robot.ctrl.effs)}
    assert effs[pan_i] == 2000.0 and effs[w2_i] == 400.0


# ------------------------------------------------------------------ 2) config_preflight validation

def test_preflight_accepts_pan_stiffen_rejects_wrist_stiffen():
    assert cp.check_phase_drive_carried_joint_profile({'phase_drive': FIX_PD}) == []
    bad = dict(FIX_PD, carried_joint_profile={'preplace': {'ur_wrist_2_joint': dict(STIFF)}})
    probs = cp.check_phase_drive_carried_joint_profile({'phase_drive': bad})
    assert probs and 'wrist' in probs[0].lower()


def test_preflight_rejects_unknown_phase_and_joint():
    bad_phase = dict(FIX_PD, carried_joint_profile={'lift': {PAN: dict(STIFF)}})
    assert cp.check_phase_drive_carried_joint_profile({'phase_drive': bad_phase})
    bad_joint = dict(FIX_PD, carried_joint_profile={'preplace': {'ur_nope_joint': dict(STIFF)}})
    assert cp.check_phase_drive_carried_joint_profile({'phase_drive': bad_joint})


def test_preflight_off_is_byte_identical():
    assert cp.check_phase_drive_carried_joint_profile({}) == []
    assert cp.check_phase_drive_carried_joint_profile({'phase_drive': {'soft': SOFT, 'stiff': STIFF}}) == []


def test_preflight_convergence_keys():
    assert cp.check_preplace_convergence_keys({'preplace_convergence_stall_s': 30.0,
                                               'preplace_wrist_first': True}) == []
    assert cp.check_preplace_convergence_keys({}) == []            # absent => byte-identical
    assert cp.check_preplace_convergence_keys({'preplace_convergence_stall_s': -1})
    assert cp.check_preplace_convergence_keys({'preplace_convergence_stall_s': 200})  # >= phase timeout
    assert cp.check_preplace_convergence_keys({'preplace_wrist_first': 'yes'})         # not a bool


def test_keys_registered_in_consumer_map():
    assert cp.KEY_CONSUMER['preplace_convergence_stall_s'] == cp.HOST
    assert cp.KEY_CONSUMER['preplace_wrist_first'] == cp.HOST


# ---------------------------------------------------------------------- 3) convergence guard (pure)
# sim-preplace-convergence-guard-fix-v1: the guard measures per-joint convergence of the DRIVEN joint, not
# each joint's error to the FINAL goal. These replay the two real runs the fix must separate: coldwin54
# (the PR342 false trip -- a sequential path's not-yet-moved joint) and coldwin53 (the real pan freeze).

_FIXDIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def _load_replay(name):
    with open(os.path.join(_FIXDIR, 'preplace_convergence_%s.json' % name)) as f:
        return json.load(f)


def test_convergence_guard_coldwin54_replay_no_false_trip_on_idle_joint():
    # coldwin54 CARTON_TICK replay (128->158 s, the real pan leg): shoulder_pan descends cleanly while
    # wrist_2 is IDLE at its carry value 1.63 rad from its eventual goal (its sweep leg has not begun). The
    # PR342 guard false-tripped here naming wrist_2; the fixed guard must stay silent for the whole leg.
    d = _load_replay('coldwin54')
    g = ConvergenceGuard(ARM_JOINTS, stall_s=30.0)
    g.arm(d['preplace_standoff_goal'])
    for t in d['ticks']:
        assert g.observe(t['now'], t['q']) is None, 'false trip at now=%s' % t['now']


def test_convergence_guard_coldwin53_replay_names_frozen_pan_within_window():
    # coldwin53 CARTON_TICK replay (128->247 s): shoulder_pan descends to the standoff (err ~0.06), then the
    # wrist_2 sweep drives it BACK to +0.48 and it freezes 0.56 rad short. The guard must catch the retreat
    # WITHIN the stall window and name shoulder_pan (not wrist_2, which converges).
    d = _load_replay('coldwin53')
    g = ConvergenceGuard(ARM_JOINTS, stall_s=30.0)
    g.arm(d['preplace_standoff_goal'])
    fired = fired_t = None
    for t in d['ticks']:
        fired = g.observe(t['now'], t['q'])
        if fired:
            fired_t = t['now']
            break
    assert fired is not None, 'never caught the pan freeze'
    assert fired.joint == PAN
    assert fired_t < d['ticks'][-1]['now']                  # caught before the series/120 s timeout ends
    assert fired.error > 0.3 and fired.best_error < 0.1     # driven near goal, then retreated far
    assert 'PHASE_CONVERGENCE' in fired.message and 'shoulder_pan' in fired.message
    assert ('commanded=%.4f' % fired.commanded) in fired.message
    assert ('executed=%.4f' % fired.executed) in fired.message


def test_convergence_guard_idle_joint_never_trips():
    # A joint held at its carry value (never moving -- its leg has not started), far from the final goal,
    # must NOT trip while another joint converges, no matter how long it sits.
    goal = [0.0, 0.0, 0.0, 0.0, 1.64, 0.0]
    g = ConvergenceGuard(ARM_JOINTS, stall_s=20.0)
    g.arm(goal)
    for now in np.arange(0.0, 80.0, 1.0):
        f = min(1.0, now / 20.0)
        pan = 1.2 * (1 - f)                                 # pan descends to its goal 0 and holds
        q = [pan, 0.0, 0.0, 0.0, 0.002, 0.0]               # wrist_2 idle at carry 0.002 (goal 1.64)
        assert g.observe(float(now), q) is None


def test_convergence_guard_rearms_on_improvement_and_disarm():
    g = ConvergenceGuard(ARM_JOINTS, stall_s=10.0)
    g.arm([0.0] * 6)
    assert g.observe(0.0, [0.8] + [0.0] * 5) is None           # first observe captures the start pose
    assert g.observe(1.0, [0.4] + [0.0] * 5) is None           # pan DRIVEN (moved > engage) + improving
    assert g.observe(9.0, [0.4] + [0.0] * 5) is None           # 8 s since best < 10 stall
    assert g.observe(10.0, [0.25] + [0.0] * 5) is None         # improved -> rearm
    assert g.observe(19.0, [0.25] + [0.0] * 5) is None         # only 9 s since rearm
    fired = g.observe(21.0, [0.25] + [0.0] * 5)                # 11 s since rearm -> fire
    assert fired is not None and fired.joint == PAN and abs(fired.error - 0.25) < 1e-6
    g.disarm()
    assert g.observe(40.0, [0.25] + [0.0] * 5) is None


def test_convergence_guard_off_key_is_inert():
    off = ConvergenceGuard(ARM_JOINTS, stall_s=0.0)            # preplace_convergence_stall_s=0 => disabled
    off.arm([-0.079, -1.558, 1.586, 0.0, 1.640, 0.0])
    assert not off.enabled
    for now in np.arange(0.0, 200.0, 5.0):
        assert off.observe(float(now), [0.6, -1.558, 1.586, 0.0, 1.631, 0.0]) is None


# ------------------------------------------------------------------- 4) wrist-first sequencing (pure)

def test_wrist_first_holds_pan_moves_rest():
    carry = [1.2675, -2.27, 1.0, 0.65, 0.002, 0.619]
    standoff = [-0.079, -1.558, 1.586, 0.0, 1.640, 0.0]
    leg1, leg2 = wrist_first_subgoals(carry, standoff)
    assert leg1[0] == carry[0]                      # shoulder_pan held at carry in leg 1
    assert list(leg1[1:]) == list(standoff[1:])     # every other joint at the standoff
    assert list(leg2) == [float(v) for v in standoff]
    with pytest.raises(ValueError):
        wrist_first_subgoals([0, 1, 2], standoff)   # non-6


# ------------------------------------------------------------------------ 5) six-joint drive bench

@pytest.fixture(scope='module')
def report():
    return bench.run_per_joint(URDF, FIX)


def test_bench_tilt_jacobian_pan_swing_safe_wrists_inject():
    r = bench.run_per_joint(URDF, FIX)
    jac = {jn: (dr, dp, dt) for jn, dr, dp, dt in r['tilt_jacobian']}
    assert jac[PAN][2] < 2.0                        # shoulder_pan: small tilt sensitivity -> swing-safe
    for jn in ('ur_shoulder_lift_joint', 'ur_elbow_joint', 'ur_wrist_1_joint', 'ur_wrist_3_joint'):
        assert jac[jn][2] > 10.0                    # tilt-injecting joints -> must stay soft


def test_bench_pan_is_vertical_and_coupling_tiny(report):
    assert abs(report['pan_gravity_nm']) < 1e-3     # vertical axis -> zero gravity torque
    assert abs(report['coupling_pan_w2']) < 1.0     # negligible rigid-body pan<->wrist_2 coupling


def test_bench_reproduces_cw53_freeze_and_chosen_holds(report):
    h = report['pan_hold']
    b = report['authority_bracket']
    # soft eff 400 reproduces the cw53 drift; the chosen pan effort holds within 0.02 rad
    assert abs(h['soft_peak_rad'] - 0.558) < 0.05
    assert h['chosen_peak_rad'] < 0.02
    # required pan holding torque is bracketed strictly between the soft and stiff ceilings
    assert b['soft_eff'] < b['inferred_required_nm'] <= b['stiff_eff']
    assert b['chosen_eff'] == 2000.0


def test_bench_predicted_overshoot_under_1_5_deg(report):
    assert all(v < 1.5 for v in report['predicted_overshoot_deg'].values())


def test_bench_sequencing_decoupled_and_level(report):
    s = report['sequencing']
    assert s['pan_travel_during_sweep_rad'] == 0.0        # pan held while the wrist sweeps
    assert s['wrist2_travel_during_pan_rad'] == 0.0       # wrist_2 set before the pan moves
    assert s['path_tilt_deg'] < 2.0                       # carton stays level across both legs


def test_bench_existing_sweep_report_still_runs():
    # the original sweep-tilt characterisation is unchanged (regression on the existing public API)
    rep = bench.run(URDF, FIX)
    assert rep['fix']['stiff_phases_fixed'] == ['pan', 'place']


# ------------------------------------------------------- 6) controller integration (convergence guard)

PROFILE = ProductProfile('sku-box', (.10, .10, .10), (1., 0., 0., 0.))
JOB = dict(job_id='job-1', sku='sku-box', quantity=1, source_bin='backroom', shelf_id='shelf-a')


class _RunningMotion:
    """Minimal MotionPort whose poll() stays 'running' (no transition), so a manually-injected preplace
    accumulates convergence-guard observations across steps."""
    outcome = 'running'

    def poll(self):
        return 'running'

    def begin_joint_goal(self, *a, **k):
        pass

    def begin(self, *a, **k):
        pass

    def update_payload(self, *a, **k):
        return True

    def stop(self):
        pass

    def take_exec_abort_reason(self):
        return None


def _preplace_packet(now, joint_positions):
    obs = Observation(f'src-{now}', 'test', now, 'sku-box', 'backroom',
                      (.4, 0., .2, 0., 0., 0., 1.), .99)
    tgt = Target(f'tgt-{now}', 'test', now, 'shelf-a', (.5, .2, .1, 0., 0., 0., 1.), True, (.3, .3, .4))
    return Sensors(simulation_time=now, observations=(obs,), target=tgt, parked=True, contact=True,
                   feedback_time=now, collision_time=now, joint_positions=tuple(joint_positions),
                   scanned_locations=('backroom', 'shelf-a'), scan_time=now,
                   valid_until_steady=time.monotonic() + 100.)


def test_controller_convergence_guard_fails_frozen_preplace_by_name():
    c = TransferController(_RunningMotion(), {'sku-box': PROFILE}, fixture_mode=True,
                           preplace_convergence_stall_s=15.0)
    c.start('collect', JOB)
    # Inject the preplace directly (the full dispatch path needs the collect-deck-place setup; the guard
    # wiring in step() is what we exercise here). Pre-drive the guard to the cw53 pattern -- pan DRIVEN to
    # the standoff (engages + converges) then RETREATS -- so the ONE step() below fires at the convergence
    # block (which returns before the dispatch), exactly as a real frozen preplace would.
    c.phase = 'preplace'
    c.issued = True                                   # skip the move-issue branch
    c.held = True
    c.elapsed = 0.0
    c.last_time = None
    standoff = [-0.079, -1.558, 1.586, 0.0, 1.640, 0.0]
    carry = [1.267, -1.558, 1.586, 0.0, 1.640, 0.0]   # pan at carry (rest already at goal)
    frozen = [0.5, -1.558, 1.586, 0.0, 1.631, 0.0]    # pan retreats ~0.58 and holds
    g = c._preplace_convergence
    g.arm(standoff)
    assert g.observe(100.0, carry) is None            # capture start
    assert g.observe(101.0, standoff) is None         # pan driven to goal (engaged + converged)
    assert g.observe(110.0, frozen) is None           # retreat, but only 9 s < 15 s window
    result = c.step(1.0, _preplace_packet(117.0, frozen))   # 16 s since the best -> fires in step()
    assert result.status == 'failed'
    assert 'PHASE_CONVERGENCE' in result.reason and 'shoulder_pan' in result.reason
    assert any('phase_convergence' in e for e in c.evidence)


def test_controller_off_is_inert_even_on_frozen_preplace():
    # stall_s=0 (default) => guard disabled => the step() convergence block is a no-op (the coarse phase
    # timeout owns a frozen preplace, as before the lane). Replaces the old scalar-internal pre-seed test.
    c = TransferController(_RunningMotion(), {'sku-box': PROFILE}, fixture_mode=True)
    c.start('collect', JOB)
    assert not c._preplace_convergence.enabled
    assert c.preplace_convergence_stall_s == 0.0
    c.phase = 'preplace'
    g = c._preplace_convergence
    g.arm([-0.079, -1.558, 1.586, 0.0, 1.640, 0.0])
    carry = [1.267, -1.558, 1.586, 0.0, 1.640, 0.0]
    frozen = [0.5, -1.558, 1.586, 0.0, 1.631, 0.0]
    assert g.observe(100.0, carry) is None
    for now in np.arange(101.0, 260.0, 5.0):           # well past any window -> a disabled guard never fires
        assert g.observe(float(now), frozen) is None


def test_controller_stores_wrist_first_flag():
    c = TransferController(_RunningMotion(), {'sku-box': PROFILE}, fixture_mode=True,
                           preplace_wrist_first=True)
    assert c.preplace_wrist_first is True
    # the wrist-first sub-stage flag resets when leaving preplace
    c.phase = 'preplace'
    c._preplace_wristed = True
    c._transition('place')
    assert c._preplace_wristed is False
