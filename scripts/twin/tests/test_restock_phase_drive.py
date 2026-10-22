"""fm347 (1): phase-dependent arm drive -- pure selection logic + the runtime gain-set primitive.
CPU-only (no Isaac): physics_setup imports cleanly; set_arm_drive_gains is exercised against a fake
articulation sink so the mid-mission re-gain path is covered without a simulator."""
import numpy as np

from twin.restock.physics_setup import phase_drive_gains, set_arm_drive_gains

PHASE_DRIVE = {'soft': {'kp': 3e5, 'kd': 8e3, 'eff': 400.0},
               'stiff': {'kp': 1e6, 'kd': 1e4, 'eff': 2000.0}}


def test_phase_drive_gains_stiff_for_loaded_extend_soft_otherwise():
    # Stiff for the loaded base rotation + extend + place (pan/preplace/place); soft for the tuck fold
    # and the empty-handed post-release retreat/stow/retreat_tuck.
    for p in ('pan', 'preplace', 'place'):
        assert phase_drive_gains(p, PHASE_DRIVE) == {'kp': 1e6, 'kd': 1e4, 'eff': 2000.0}
    for p in ('tuck', 'lift', 'approach', 'retreat', 'retreat_tuck', 'stow', 'release'):
        assert phase_drive_gains(p, PHASE_DRIVE) == {'kp': 3e5, 'kd': 8e3, 'eff': 400.0}


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
