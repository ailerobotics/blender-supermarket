"""fm347 (1): phase-dependent arm drive -- pure selection logic + the runtime gain-set primitive.
CPU-only (no Isaac): physics_setup imports cleanly; set_arm_drive_gains is exercised against a fake
articulation sink so the mid-mission re-gain path is covered without a simulator."""
import numpy as np
import pytest

from twin.restock.physics_setup import (phase_drive_gains, resolve_stiff_phases, set_arm_drive_gains,
                                        _PHASE_DRIVE_STIFF_PHASES)

PHASE_DRIVE = {'soft': {'kp': 3e5, 'kd': 8e3, 'eff': 400.0},
               'stiff': {'kp': 1e6, 'kd': 1e4, 'eff': 2000.0}}
SOFT = {'kp': 3e5, 'kd': 8e3, 'eff': 400.0}
STIFF = {'kp': 1e6, 'kd': 1e4, 'eff': 2000.0}


def test_phase_drive_gains_stiff_for_loaded_extend_soft_otherwise():
    # Stiff for the loaded base rotation + extend + place (pan/preplace/place); soft for the tuck fold
    # and the empty-handed post-release retreat/stow/retreat_tuck.
    for p in ('pan', 'preplace', 'place'):
        assert phase_drive_gains(p, PHASE_DRIVE) == {'kp': 1e6, 'kd': 1e4, 'eff': 2000.0}
    for p in ('tuck', 'lift', 'approach', 'retreat', 'retreat_tuck', 'stow', 'release'):
        assert phase_drive_gains(p, PHASE_DRIVE) == {'kp': 3e5, 'kd': 8e3, 'eff': 400.0}


# ---------------------------------------------------- sim-preplace-sweep-tilt-v1: carried_soft_phases

def test_resolve_stiff_phases_default_is_byte_identical():
    assert resolve_stiff_phases(None) == frozenset(_PHASE_DRIVE_STIFF_PHASES)
    assert resolve_stiff_phases(PHASE_DRIVE) == frozenset(_PHASE_DRIVE_STIFF_PHASES)
    assert resolve_stiff_phases(dict(PHASE_DRIVE, carried_soft_phases=[])) == frozenset(_PHASE_DRIVE_STIFF_PHASES)


def test_carried_soft_phases_forces_preplace_soft_keeps_place_stiff():
    # The verdict fix: run the carried PREPLACE approach SOFT (it swings the welded carton under the
    # stiff drive), but keep the gravity-resisting place DESCENT and the free-arm pan STIFF.
    cfg = dict(PHASE_DRIVE, carried_soft_phases=['preplace'])
    assert phase_drive_gains('preplace', cfg) == SOFT
    assert phase_drive_gains('place', cfg) == STIFF
    assert phase_drive_gains('pan', cfg) == STIFF
    assert resolve_stiff_phases(cfg) == frozenset({'pan', 'place'})


def test_carried_soft_phases_rejects_non_stiff_or_bad_value():
    # a phase that is not one of the default STIFF phases is a typo, not a silent no-op
    with pytest.raises(ValueError):
        resolve_stiff_phases(dict(PHASE_DRIVE, carried_soft_phases=['lift']))
    with pytest.raises(ValueError):
        resolve_stiff_phases(dict(PHASE_DRIVE, carried_soft_phases='preplace'))  # must be a list


def test_carried_soft_phases_off_is_byte_identical_everywhere():
    # Absent/empty => every phase selects exactly what it did before the lane.
    base = {p: phase_drive_gains(p, PHASE_DRIVE)
            for p in ('pan', 'preplace', 'place', 'tuck', 'lift', 'transit', 'retreat')}
    cfg = dict(PHASE_DRIVE)  # no carried_soft_phases
    for p, g in base.items():
        assert phase_drive_gains(p, cfg) == g


def test_phase_drive_gains_absent_config_is_noop():
    assert phase_drive_gains('preplace', None) is None
    assert phase_drive_gains('preplace', {}) is None
    # a config missing the needed band is a no-op, never a crash
    assert phase_drive_gains('preplace', {'soft': PHASE_DRIVE['soft']}) is None


class _FakeArmController:
    def __init__(self, ndof):
        self.kps = np.zeros(ndof)
        self.kds = np.zeros(ndof)
        self.effs = None
        self.eff_idx = None

    def get_gains(self):
        return self.kps, self.kds

    def set_gains(self, kps, kds):
        self.kps, self.kds = np.asarray(kps, float), np.asarray(kds, float)

    def set_max_efforts(self, effs, joint_indices=None):
        self.effs, self.eff_idx = np.asarray(effs, float), joint_indices


class _FakeRobot:
    # dof 0..5 are the arm joints a..f; 6,7 are non-arm dofs that must stay untouched.
    _DOF = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'base': 6, 'cart': 7}

    def __init__(self):
        self.ctrl = _FakeArmController(len(self._DOF))

    def get_dof_index(self, name):
        return self._DOF[name]

    def get_articulation_controller(self):
        return self.ctrl


def test_set_arm_drive_gains_only_touches_arm_dofs():
    robot = _FakeRobot()
    arm = ('a', 'b', 'c', 'd', 'e', 'f')
    set_arm_drive_gains(robot, arm, kp=1e6, kd=1e4, eff=2000.0)
    assert list(robot.ctrl.kps[:6]) == [1e6] * 6 and list(robot.ctrl.kds[:6]) == [1e4] * 6
    assert robot.ctrl.kps[6] == 0.0 and robot.ctrl.kps[7] == 0.0   # non-arm dofs untouched
    assert list(robot.ctrl.effs) == [2000.0] * 6
    assert list(robot.ctrl.eff_idx) == [0, 1, 2, 3, 4, 5]          # effort set on the arm dof indices
