"""CPU tests for the load-aware gravity feedforward (sim-robust-executor-shoulder-sag-v1).

The executor's LOADED pick settled ur_shoulder_lift 0.098 rad short of its goal (execretest, PR 214):
TRAJ_HOST_GOAL_MISS per_joint_err=[8e-5, 0.0981, 5.7e-5, ...] -> TRAJ_HOST_FAULT. The cause, established
by a static torque balance at the pick pose (these tests), is a COMBINATION:
  1. the gravity FF was evaluated at the COMMANDED (goal) arm config while the arm physically lagged at
     the measured pose, so a heavily gravity-loaded joint was over-compensated; and
  2. the 0.45 kg welded carton was omitted from the FF mass model.
On the soft shoulder_lift joint (effective J*Kp ~= 10 Nm/rad, back-calculated from the run and matching
the exact 0.098 reproduction) the ~1 Nm residual the two leave produces the miss. Only evaluating the FF
at the MEASURED pose AND adding the held carton drives the net torque to ~0 (each correction ALONE is
worse). The pure helper gravity_ff_effort holds that logic; build_payload_gravity the carton torque.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from twin.restock.physics_setup import (  # noqa: E402
    build_arm_gravity, build_payload_gravity, gravity_ff_effort, _ISAAC_BODY_MASSES)
from twin.restock_manipulation.cad_tool import ARM_JOINTS  # noqa: E402

URDF = os.path.expanduser('~/datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf')
ARM = list(ARM_JOINTS)
LIFT = 1

# The execretest terminal state (host-stack.log TRAJ_HOST_GOAL_MISS), arm order [pan,lift,elbow,w1,w2,w3].
GOAL = np.array([0.3481, -3.6472, 2.2171, -1.7122, 1.2201, 3.1418])
MEAS = np.array([0.3482, -3.5491, 2.2172, -1.7122, 1.2201, 3.1418])
CARTON_KG = 0.45
# Effective torque-per-rad of the shoulder_lift drive, back-calculated from the run: the drive supplied
# ~-0.965 Nm at a +0.098 rad error (elbow, same Kp, supplied -9.5 Nm at +1e-4 rad => J*Kp ~= 95000, so
# shoulder is ~1e4x softer -- the asymmetry that makes only shoulder_lift miss). Used to turn a residual
# net torque into the predicted steady-state offset so the sizing test is a real pass/fail, not a replot.
JKP_SHOULDER = 9.85
GOAL_POS_TOL = 0.010


class GravityFfEffortPureTests(unittest.TestCase):
    """gravity_ff_effort without Isaac/URDF: a stub grav depends on q so pose-selection is observable."""

    @staticmethod
    def _grav(q):
        # A single gravity-loaded joint (index 1) whose torque tracks its own angle, so commanded-vs-
        # measured pose selection changes the result; everything else sub-threshold.
        return np.array([0.0, 100.0 * q[1], 0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def _payload(q, kg):
        # Same sign as the gravity load (a carried mass adds to the joint's downward torque).
        return np.array([0.0, -10.0 * kg, 0.0, 0.0, 0.0, 0.0])

    def test_off_is_legacy_commanded_pose_no_payload(self):
        # The loaded-carry OFF path: apply() passes at_measured=False, held=False and payload_grav_fn=None,
        # so this is byte-identical to the legacy FF (robot gravity at the COMMANDED pose, no payload).
        qc = np.array([0.0, -2.0, 0, 0, 0, 0])
        qm = np.array([0.0, -1.0, 0, 0, 0, 0])
        ff = gravity_ff_effort(self._grav, qc, qm, at_measured=False, ff_sign=-1.0, ff_min_nm=3.0,
                               payload_grav_fn=None, payload_kg=0.0, held=False)
        self.assertAlmostEqual(ff[LIFT], 200.0, places=6)   # COMMANDED q (-2.0 -> g=-200) -> ff=+200

    def test_at_measured_uses_measured_pose(self):
        qc = np.array([0.0, -2.0, 0, 0, 0, 0])
        qm = np.array([0.0, -1.0, 0, 0, 0, 0])
        ff = gravity_ff_effort(self._grav, qc, qm, at_measured=True, ff_sign=-1.0, ff_min_nm=3.0)
        # Uses MEASURED q (-1.0 -> g=-100): ff=+100 (not +200 from the commanded pose).
        self.assertAlmostEqual(ff[LIFT], 100.0, places=6)

    def test_payload_added_only_when_held(self):
        q = np.array([0.0, -1.0, 0, 0, 0, 0])  # g_lift=-100
        kw = dict(at_measured=True, ff_sign=-1.0, ff_min_nm=3.0,
                  payload_grav_fn=self._payload, payload_kg=CARTON_KG)
        held = gravity_ff_effort(self._grav, q, q, held=True, **kw)
        free = gravity_ff_effort(self._grav, q, q, held=False, **kw)
        # payload torque = 10*0.45 = 4.5; held adds it (g=-104.5 -> ff=+104.5), not-held does not.
        self.assertAlmostEqual(held[LIFT], 104.5, places=6)
        self.assertAlmostEqual(free[LIFT], 100.0, places=6)

    def test_payload_zero_is_a_noop_even_when_held(self):
        q = np.array([0.0, -1.0, 0, 0, 0, 0])
        ff = gravity_ff_effort(self._grav, q, q, at_measured=True, ff_sign=-1.0, ff_min_nm=3.0,
                               payload_grav_fn=self._payload, payload_kg=0.0, held=True)
        self.assertAlmostEqual(ff[LIFT], 100.0, places=6)

    def test_threshold_zeros_subthreshold_joints(self):
        grav = lambda q: np.array([0.0, 50.0, 2.0, 0.0, 0.0, 0.0])  # elbow 2 Nm < 3 Nm threshold
        ff = gravity_ff_effort(grav, np.zeros(6), np.zeros(6), at_measured=False, ff_sign=-1.0, ff_min_nm=3.0)
        self.assertAlmostEqual(ff[LIFT], -50.0, places=6)
        self.assertEqual(ff[2], 0.0)


@unittest.skipUnless(os.path.isfile(URDF), f'mobile URDF absent: {URDF}')
class LoadedPickEquilibriumTests(unittest.TestCase):
    """Static torque balance at the execretest pick pose: reproduce the 0.098 rad miss and prove the fix."""

    @classmethod
    def setUpClass(cls):
        from twin import urdf_probe as up
        model = up.parse_urdf(URDF)
        # staticmethod: a bare function stored as a class attribute would bind self and get a spurious
        # second positional arg when called as self.grav(q).
        cls.grav = staticmethod(build_arm_gravity(model, ARM, _ISAAC_BODY_MASSES))
        cls.payload = staticmethod(build_payload_gravity(model, ARM, 'gripper_pad_tcp'))

    def _predicted_shoulder_err(self, ff):
        # True physical gravity + carton at the resting (measured) pose; the sim FF model matches PhysX
        # (it IS the sim's own FF function), confirmed by the exact 0.098 reproduction below. The drive
        # must hold the residual net = ff + G_actual; offset = -net/(J*Kp).
        net = (ff + self.grav(MEAS) + self.payload(MEAS, CARTON_KG))[LIFT]
        return -net / JKP_SHOULDER

    def test_model_matches_sim_ff_log_at_start_pose(self):
        # The sim logged ARM_GRAVITY_FF g=[0,-33.1,78.8,13.7,...] at the start pose; the CPU model must
        # match (same code) so the goal-pose balance below is trustworthy.
        start = np.array([-0.0, -2.5002, 1.7004, -0.7001, -1.5708, -0.0001])
        g = self.grav(start)
        for got, ref in zip(g, [0.0, -33.1, 78.8, 13.7, 0.0, 0.0]):
            self.assertLess(abs(got - ref), 1.0)

    def test_legacy_ff_reproduces_the_098_miss(self):
        ff = gravity_ff_effort(self.grav, GOAL, MEAS, at_measured=False, ff_sign=-1.0, ff_min_nm=3.0)
        err = self._predicted_shoulder_err(ff)
        self.assertAlmostEqual(abs(err), 0.098, delta=0.01)   # reproduces the observed 0.0981 rad
        self.assertGreater(abs(err), GOAL_POS_TOL)             # and it fails the tight goal tolerance

    def test_each_correction_alone_is_not_better(self):
        # Adding the carton to the commanded-pose FF, or evaluating at measured WITHOUT the carton, both
        # LEAVE a larger offset -- so the two must ship together (not as a half-fix footgun).
        carton_only = gravity_ff_effort(self.grav, GOAL, MEAS, at_measured=False, ff_sign=-1.0,
                                        ff_min_nm=3.0, payload_grav_fn=self.payload, payload_kg=CARTON_KG,
                                        held=True)
        measured_only = gravity_ff_effort(self.grav, GOAL, MEAS, at_measured=True, ff_sign=-1.0,
                                          ff_min_nm=3.0)
        self.assertGreater(abs(self._predicted_shoulder_err(carton_only)), GOAL_POS_TOL)
        self.assertGreater(abs(self._predicted_shoulder_err(measured_only)), GOAL_POS_TOL)

    def test_measured_pose_plus_held_payload_reaches_goal(self):
        ff = gravity_ff_effort(self.grav, GOAL, MEAS, at_measured=True, ff_sign=-1.0, ff_min_nm=3.0,
                               payload_grav_fn=self.payload, payload_kg=CARTON_KG, held=True)
        err = self._predicted_shoulder_err(ff)
        self.assertLessEqual(abs(err), GOAL_POS_TOL)           # the fix: within the tight 0.010 rad goal
        # and it does not merely nudge -- the residual net torque is driven essentially to zero.
        net = (ff + self.grav(MEAS) + self.payload(MEAS, CARTON_KG))[LIFT]
        self.assertLess(abs(net), 0.1)

    def test_shoulder_pan_ff_stays_zero(self):
        # The vertical yaw axis carries no gravity torque; the fix must not invent one on it.
        ff = gravity_ff_effort(self.grav, GOAL, MEAS, at_measured=True, ff_sign=-1.0, ff_min_nm=3.0,
                               payload_grav_fn=self.payload, payload_kg=CARTON_KG, held=True)
        self.assertEqual(ff[0], 0.0)


if __name__ == '__main__':
    unittest.main()
