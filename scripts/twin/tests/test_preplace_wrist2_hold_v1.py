"""sim-preplace-wrist2-hold-v1: the carried-preplace wrist give-way fix -- drift re-capture on the
convergence guard and hold-firm-on-arrival.

cw66 sealed its pick and carried the carton to the set-down pose, then stalled in the carried preplace:
wrist_2 reached its commanded 0.6678 rad (closest approach 0.0006), was held there on the SOFT carried drive
(kp 3e5 / eff 400) through the end of leg 1, and was then BACK-DRIVEN off its target by leg 2's 1.69 rad
shoulder_pan sweep -- collapsing to a -0.3264 rad limit cycle, so the convergence guard faulted at 30 s.
cw64 ran the identical config/trajectory, dipped at the same pan angle, and recovered: the soft held wrist
holding the dynamic pan-sweep reaction is a coin toss. The static hold is trivial (gravity on wrist_2 is
0.045 N*m), so (a) once a joint arrives it is swing-safe to firm (hold-firm-on-arrival), and (b) a drifted
joint is re-driven by re-issuing the goal (drift re-capture) rather than waited out to a fault.

CPU-only: no Isaac / ROS / GPU.
"""
import json
import os
import types

import pytest

from twin.restock_manipulation.convergence_guard import (ConvergenceGuard, ConvergenceStall,
                                                         DriftRecapture)
from twin.restock_manipulation.preplace_hold import HoldFirmOnArrival
from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.restock_manipulation import config_preflight as cp
from twin.restock_manipulation import ProductProfile, TransferController
from twin.restock.physics_setup import resolve_hold_firm, _ARM_JOINT_NAMES


class _MotionStub:
    """Minimal MotionPort: records begin_joint_goal dispatches + stop() for the controller-wiring tests."""

    def __init__(self):
        self.joint_goals = []
        self.stops = 0

    def begin_joint_goal(self, goal, measured, slow=False, goal_tol=None, straight_line=False,
                         speed=None, accel=None):
        self.joint_goals.append(dict(goal=tuple(goal), measured=tuple(measured), slow=slow,
                                     goal_tol=goal_tol, straight_line=bool(straight_line)))

    def stop(self):
        self.stops += 1


_PROFILE = ProductProfile('sku-box', (.10, .10, .10), (1., 0., 0., 0.))


def _make_controller(**kwargs):
    motion = _MotionStub()
    return TransferController(motion, {'sku-box': _PROFILE}, fixture_mode=True, **kwargs), motion

W2 = ARM_JOINTS.index('ur_wrist_2_joint')            # 4
CMD = 0.6678
SOFT = {'kp': 3e5, 'kd': 8e3, 'eff': 400.0}
STIFF = {'kp': 1e6, 'kd': 1e4, 'eff': 2000.0}
PAN = 'ur_shoulder_pan_joint'
CW66 = os.path.join(os.path.dirname(__file__), 'fixtures', 'preplace_wrist2_cw66.json')


def _goal(w2=CMD):
    g = [0.0] * 6
    g[W2] = w2
    return tuple(g)


def _meas(w2):
    m = [0.0] * 6
    m[W2] = w2
    return tuple(m)


# ---------------------------------------------------------------- 1) drift re-capture (ConvergenceGuard)

def test_recapture_off_is_byte_identical_terminal_fault():
    """Re-capture off (either bound <= 0) => only the terminal ConvergenceStall path runs, identical to the
    convergence-guard-fix-v1 behaviour (message closest-approach from the all-time best)."""
    for kw in ({}, {'recapture_rad': 0.05, 'recapture_max': 0}, {'recapture_rad': 0.0, 'recapture_max': 3}):
        g = ConvergenceGuard(ARM_JOINTS, stall_s=30.0, **kw)
        assert not g.recapture_enabled
        g.arm(_goal())
        g.observe(0.0, _meas(0.0))                   # capture start
        for t in (1.0, 2.0, 3.0):
            g.observe(t, _meas(0.667))               # converge (closest approach)
        verdict = None
        for t in range(4, 60):
            verdict = g.observe(float(t), _meas(-0.326))
            if verdict is not None:
                break
        assert isinstance(verdict, ConvergenceStall)
        assert verdict.joint == 'ur_wrist_2_joint'
        assert verdict.best_error == pytest.approx(0.0007, abs=1e-3)   # true closest approach preserved
        assert 'PHASE_CONVERGENCE' in verdict.message


def test_recapture_on_emits_before_fault_and_preserves_closest_approach():
    g = ConvergenceGuard(ARM_JOINTS, stall_s=30.0, recapture_rad=0.05, recapture_max=3)
    assert g.recapture_enabled
    g.arm(_goal())
    g.observe(0.0, _meas(0.0))
    for t in (1.0, 2.0, 3.0):
        g.observe(t, _meas(0.667))
    v = g.observe(4.0, _meas(-0.326))                 # first drift tick
    assert isinstance(v, DriftRecapture)
    assert v.joint == 'ur_wrist_2_joint'
    assert v.to_pos == pytest.approx(CMD)
    assert v.from_pos == pytest.approx(-0.326)
    assert v.best_error == pytest.approx(0.0007, abs=1e-3)   # the closest approach, not the drifted err
    assert v.count == 1 and v.limit == 3
    assert 'PREPLACE_DRIFT_RECAPTURE' in v.message


def test_recapture_budget_is_bounded_then_faults():
    """After the per-joint budget is spent on a joint that keeps re-drifting WITHOUT recovering, the guard
    falls through to the terminal ConvergenceStall (never an unbounded re-issue storm)."""
    g = ConvergenceGuard(ARM_JOINTS, stall_s=5.0, recapture_rad=0.05, recapture_max=2)
    g.arm(_goal())
    g.observe(0.0, _meas(0.0))
    g.observe(0.5, _meas(0.66))
    recaps, stall = 0, None
    # each tick: drift a bit FURTHER so err keeps exceeding best+recapture_rad until budget is spent
    drift = 0.60
    for i in range(1, 200):
        t = 1.0 + i * 0.5
        drift = min(drift + 0.1, 1.5)
        v = g.observe(t, _meas(CMD - drift))
        if isinstance(v, DriftRecapture):
            recaps += 1
        elif isinstance(v, ConvergenceStall):
            stall = v
            break
    assert recaps == 2                                 # exactly the budget
    assert isinstance(stall, ConvergenceStall)         # then a terminal fault, by name


def test_recapture_resets_window_so_a_reconverging_joint_never_faults():
    """Model the re-issue EFFECT: the joint arrives, is back-driven once (a drift episode), the DriftRecapture
    re-drives it back to target, and it then HOLDS converged. The guard must re-arm on the re-convergence and
    never fault (the intended runtime loop: re-capture -> re-converge -> done)."""
    g = ConvergenceGuard(ARM_JOINTS, stall_s=10.0, recapture_rad=0.05, recapture_max=3)
    g.arm(_goal())
    g.observe(0.0, _meas(0.0))
    pos, t, recovered = 0.667, 1.0, False
    recaps, faulted = 0, False
    for _ in range(400):
        t += 0.5
        v = g.observe(t, _meas(pos))
        if isinstance(v, ConvergenceStall):
            faulted = True
            break
        if isinstance(v, DriftRecapture):
            recaps += 1
            pos, recovered = CMD, True               # the re-issue re-drives it back, and it then holds
        elif not recovered and t > 3.0:
            pos = -0.30                               # the one back-driven drift episode
    assert recaps >= 1 and recovered and not faulted


# ---------------------------------------------------------------- 2) cw66 replay regression (real trace)

def _cw66():
    with open(CW66) as fh:
        return json.load(fh)


def test_cw66_replay_off_faults_on_with_key_does_not():
    fx = _cw66()
    assert fx['joint'] == 'ur_wrist_2_joint' and fx['commanded'] == pytest.approx(CMD)
    stall_s = fx['stall_s']

    def replay(recapture_rad, recapture_max):
        g = ConvergenceGuard(ARM_JOINTS, stall_s=stall_s,
                             recapture_rad=recapture_rad, recapture_max=recapture_max)
        g.arm(_goal())
        stalls, recaps = [], []
        for now, w2 in fx['trace']:
            v = g.observe(float(now), _meas(float(w2)))
            if isinstance(v, ConvergenceStall):
                stalls.append((now, v.joint))
                break
            if isinstance(v, DriftRecapture):
                recaps.append(now)
        return stalls, recaps

    # KEY OFF: the known PHASE_CONVERGENCE fault, at the recorded fault time, on wrist_2.
    off_stalls, off_recaps = replay(0.0, 0)
    assert off_recaps == []
    assert len(off_stalls) == 1
    assert off_stalls[0][1] == 'ur_wrist_2_joint'
    assert off_stalls[0][0] == pytest.approx(fx['recorded_fault']['fault_now'], abs=0.1)

    # KEY ON: a drift re-capture fires (near the recorded drift onset) and NO terminal fault is reached
    # within the recorded trace.
    on_stalls, on_recaps = replay(0.05, 3)
    assert on_stalls == []
    assert on_recaps, 'the re-capture must fire on the drift'
    assert on_recaps[0] < fx['recorded_fault']['fault_now']   # well before the off-mode fault time


# ---------------------------------------------------------------- 3) hold-firm-on-arrival decider (pure)

def test_decider_latches_firm_on_settled_arrival():
    d = HoldFirmOnArrival(ARM_JOINTS, tol_rad=0.05, settle_eps_rad=0.01, dwell_ticks=3, enabled=True,
                          hold_joints=['ur_wrist_2_joint', 'ur_wrist_1_joint'])
    # wrist_2 sweeps (command moving) -> must NOT firm; then settles arrived -> firms after the dwell.
    newly = []
    for c in (0.1, 0.3, 0.5):                          # moving command
        newly += d.observe(_cmd(c), _meas(c))
    assert newly == [] and d.firm == set()             # a moving (sweeping) joint is never firmed
    for c in (0.666, 0.666, 0.666, 0.666):             # settled + arrived
        newly += d.observe(_cmd(0.666), _meas(0.666))
    assert 'ur_wrist_2_joint' in d.firm
    # monotonic: it stays firm even if later nudged
    d.observe(_cmd(0.666), _meas(0.60))
    assert 'ur_wrist_2_joint' in d.firm


def test_decider_only_firms_eligible_joints_and_off_is_noop():
    d = HoldFirmOnArrival(ARM_JOINTS, dwell_ticks=1, enabled=True,
                          hold_joints=['ur_wrist_2_joint'])
    for _ in range(3):
        d.observe(_goal(0.666), _meas(0.666))
    assert d.firm == {'ur_wrist_2_joint'}              # the eligible one only (others held at 0 too)
    # disabled => never firms, returns [] (byte-identical)
    off = HoldFirmOnArrival(ARM_JOINTS, enabled=False)
    assert off.observe(_goal(0.666), _meas(0.666)) == []
    assert off.firm == set()


def test_decider_default_eligible_excludes_shoulder_pan():
    cfg = resolve_hold_firm({'soft': SOFT, 'stiff': STIFF,
                             'carried_joint_profile': {'preplace': {PAN: dict(STIFF)}},
                             'hold_firm_on_arrival': {'enabled': True}})
    assert cfg is not None
    assert PAN not in cfg['joints']                    # already firm via carried_joint_profile
    assert 'ur_wrist_2_joint' in cfg['joints']
    assert cfg['firm'] == STIFF                         # defaults to the pan firm block


def _cmd(w2):
    c = [0.0] * 6
    c[W2] = w2
    return tuple(c)


# ---------------------------------------------------------------- 4) resolve_hold_firm + preflight

def test_resolve_hold_firm_off_and_defaults():
    assert resolve_hold_firm(None) is None
    assert resolve_hold_firm({'soft': SOFT}) is None                       # no block
    assert resolve_hold_firm({'hold_firm_on_arrival': {'enabled': False}}) is None
    cfg = resolve_hold_firm({'hold_firm_on_arrival': {'enabled': True, 'firm': STIFF}})
    assert cfg['enabled'] and cfg['firm'] == STIFF
    assert cfg['tol_rad'] == 0.05 and cfg['dwell_ticks'] == 3


def test_resolve_hold_firm_rejects_malformed():
    with pytest.raises(ValueError):                    # no firm block resolvable
        resolve_hold_firm({'hold_firm_on_arrival': {'enabled': True}})
    with pytest.raises(ValueError):                    # bad dwell
        resolve_hold_firm({'hold_firm_on_arrival': {'enabled': True, 'firm': STIFF, 'dwell_ticks': 0}})
    with pytest.raises(ValueError):                    # unknown joint
        resolve_hold_firm({'hold_firm_on_arrival': {'enabled': True, 'firm': STIFF, 'joints': ['nope']}})
    with pytest.raises(ValueError):                    # bad gain
        resolve_hold_firm({'hold_firm_on_arrival': {'enabled': True, 'firm': {'kp': -1, 'kd': 0, 'eff': 1}}})


def test_preflight_hold_firm_and_recapture_keys():
    # hold_firm runtime validation
    assert cp.check_phase_drive_hold_firm({}) == []
    assert cp.check_phase_drive_hold_firm({'phase_drive': {'hold_firm_on_arrival': {'enabled': True,
                                                                                    'firm': STIFF}}}) == []
    assert cp.check_phase_drive_hold_firm({'phase_drive': {'hold_firm_on_arrival': {'enabled': True}}})
    # drift re-capture host keys
    good = {'preplace_convergence_stall_s': 30.0, 'preplace_drift_recapture_rad': 0.05,
            'preplace_drift_recapture_max': 3}
    assert cp.check_preplace_convergence_keys(good) == []
    assert cp.check_preplace_convergence_keys({}) == []                    # absent => byte-identical
    # re-capture without the guard armed
    assert cp.check_preplace_convergence_keys({'preplace_drift_recapture_rad': 0.05,
                                               'preplace_drift_recapture_max': 3})
    # bound below the eps => jitter storm
    assert cp.check_preplace_convergence_keys({'preplace_convergence_stall_s': 30.0,
                                               'preplace_drift_recapture_rad': 0.005,
                                               'preplace_drift_recapture_max': 3})
    # non-integer / negative budget
    assert cp.check_preplace_convergence_keys({'preplace_convergence_stall_s': 30.0,
                                               'preplace_drift_recapture_max': -1})
    assert cp.KEY_CONSUMER['preplace_drift_recapture_rad'] == cp.HOST
    assert cp.KEY_CONSUMER['preplace_drift_recapture_max'] == cp.HOST


# ---------------------------------------------------------------- 5) controller wiring (re-issue + log once)

def _arm_preplace(c, goal=None):
    c.phase = 'preplace'
    c.issued = True
    c._preplace_recapture_logged = False
    c._preplace_convergence.arm(goal or _goal())


def _sensors(w2):
    return types.SimpleNamespace(joint_positions=_meas(w2))


def test_controller_recapture_reissues_goal_and_logs_once(capsys):
    c, motion = _make_controller(preplace_convergence_stall_s=30.0,
                                 preplace_drift_recapture_rad=0.05, preplace_drift_recapture_max=3)
    assert c.preplace_drift_recapture_rad == 0.05 and c.preplace_drift_recapture_max == 3
    assert c._preplace_convergence.recapture_enabled
    _arm_preplace(c)
    c._handle_preplace_convergence(0.0, _sensors(0.0))               # capture start
    for t in (1.0, 2.0, 3.0):
        c._handle_preplace_convergence(t, _sensors(0.667))          # converge
    before = len(motion.joint_goals)
    handled, _res = c._handle_preplace_convergence(4.0, _sensors(-0.326))
    assert handled
    assert len(motion.joint_goals) == before + 1                    # re-issued the preplace goal
    assert motion.joint_goals[-1]['goal'][W2] == pytest.approx(CMD)  # toward the armed wrist_2 target
    assert c.issued is True
    assert any('phase_drift_recapture' in e for e in c.evidence)
    # a SECOND re-capture does not re-print PREPLACE_DRIFT_RECAPTURE (logged once)
    capsys.readouterr()
    c._handle_preplace_convergence(5.0, _sensors(-0.40))
    assert 'PREPLACE_DRIFT_RECAPTURE' not in capsys.readouterr().out


def test_controller_off_faults_without_reissue():
    c, motion = _make_controller(preplace_convergence_stall_s=30.0)  # recapture off
    _arm_preplace(c)
    c._handle_preplace_convergence(0.0, _sensors(0.0))
    for t in (1.0, 2.0, 3.0):
        c._handle_preplace_convergence(t, _sensors(0.667))
    before = len(motion.joint_goals)
    handled = res = None
    for t in range(4, 60):
        handled, res = c._handle_preplace_convergence(float(t), _sensors(-0.326))
        if handled:
            break
    assert handled
    assert res.status == 'failed'                                    # terminal PHASE_CONVERGENCE fail
    assert 'PHASE_CONVERGENCE' in c.reason
    assert len(motion.joint_goals) == before                        # no re-issue in off mode
