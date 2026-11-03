"""CPU tests for the opt-in load-aware gravity feedforward helper (``arm_gravity_ff_loaded_carry``, default
off) AND for the correct diagnosis of the executor LOADED-pick shoulder_lift miss.

HISTORY/CORRECTION (sim-robust-executor-shoulder-drive-v1): PR 218 added the load-aware FF (robot gravity at
the measured pose + held carton) believing the executor pick's ur_shoulder_lift ~0.098 rad miss was a
gravity sag, and shipped an equilibrium test built on a *fitted* effective "J*Kp ~= 10 Nm/rad" for the
shoulder drive. That number has no physical basis -- the drive stiffness read back from the sim is 3e5 with
a 400 Nm force limit -- and the FF changed nothing in the live run (execff shoulder_lift 0.1002 rad, vs
0.0981 without the key: feedforward-INDEPENDENT). The real cause is a PHYSICAL BLOCK: the scripted side-grip
pick jams ~40 mm short of the carton, so no FF, stiffness, or goal tolerance on shoulder_lift closes it.
This module now (1) keeps the pure helper tests (the opt-in key stays, default off, harmless), and (2)
replaces the fitted-sag equilibrium test with an honest static balance proving the miss is NOT a gravity
sag. The named goal-miss diagnostic that reports it lives in trajectory_host.classify_goal_miss
(test_restock_trajectory_host). See data/sim-robust-executor-shoulder-drive-v1/report.md.
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

# The execretest/execff terminal state (host-stack.log TRAJ_HOST_GOAL_MISS), arm order [pan,lift,elbow,w*].
GOAL = np.array([0.3481, -3.6472, 2.2171, -1.7122, 1.2201, 3.1418])
MEAS = np.array([0.3482, -3.5491, 2.2172, -1.7122, 1.2201, 3.1418])
CARTON_KG = 0.45
OBSERVED_LIFT_MISS = 0.098           # the live shoulder_lift shortfall (rad), FF on or off
# The shoulder_lift position drive as read back from the sim (ARM_GAINS_READBACK): a stiff position PD with
# a hard per-joint force limit. These are the physical numbers the balance below uses -- NOT a fitted
# softness.
DRIVE_KP = 3.0e5                     # rad -> Nm position-drive stiffness
DRIVE_MAX_FORCE = 400.0              # Nm per-joint effort ceiling (set_max_efforts readback)
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
class ShoulderLiftMissIsNotGravitySagTests(unittest.TestCase):
    """Honest static torque balance at the execff pick pose: the shoulder_lift miss is NOT a gravity sag.

    Replaces the removed fitted "J*Kp ~= 10 Nm/rad" reproduction. These assert the real drive physics, so the
    suite no longer encodes a non-existent sag model (sim-robust-executor-shoulder-drive-v1)."""

    @classmethod
    def setUpClass(cls):
        from twin import urdf_probe as up
        model = up.parse_urdf(URDF)
        # staticmethod: a bare function stored as a class attribute would bind self and get a spurious
        # second positional arg when called as self.grav(q).
        cls.grav = staticmethod(build_arm_gravity(model, ARM, _ISAAC_BODY_MASSES))
        cls.payload = staticmethod(build_payload_gravity(model, ARM, 'gripper_pad_tcp'))

    def _total_lift_gravity(self):
        # True generalized gravity the drive must hold on shoulder_lift at the resting (measured) pose,
        # robot + the 0.45 kg welded carton.
        return (self.grav(MEAS) + self.payload(MEAS, CARTON_KG))[LIFT]

    def test_gravity_is_far_below_the_drive_force_limit(self):
        # If the miss were a gravity sag, the drive would be saturating against gravity. It is not: the
        # shoulder_lift gravity (robot + carton) is well under the 400 Nm per-joint force ceiling, so the
        # drive has ample authority to hold the joint at its goal.
        g = abs(self._total_lift_gravity())
        self.assertGreater(g, 50.0)                 # it IS a meaningfully gravity-loaded joint (~127 Nm)...
        self.assertLess(g, 0.5 * DRIVE_MAX_FORCE)   # ...but far under the 400 Nm force limit (no saturation)

    def test_position_drive_demand_dwarfs_gravity(self):
        # The position-drive torque demanded to hold the OBSERVED 0.098 rad error is Kp*err ~= 29 kNm, which
        # the force limit clamps to 400 Nm -- still >> the ~127 Nm gravity. A drive with this much authority
        # over gravity cannot sit 0.098 rad short unless something PHYSICAL blocks the joint.
        demand = DRIVE_KP * OBSERVED_LIFT_MISS
        self.assertGreater(demand, 10000.0)
        self.assertGreater(DRIVE_MAX_FORCE, 2.0 * abs(self._total_lift_gravity()))

    def test_gravity_droop_offset_is_microradian_not_0098(self):
        # A position PD drive's gravity-droop steady-state error is gravity/Kp. Here that is ~4e-4 rad --
        # two orders of magnitude below the observed 0.098 rad. So 0.098 rad is NOT gravity droop, and no
        # feedforward (which only cancels gravity/Kp) could account for it. Matches the FF-independent run
        # (execff 0.1002 with the FF on vs 0.0981 off).
        droop = abs(self._total_lift_gravity()) / DRIVE_KP
        self.assertLess(droop, 1e-3)
        self.assertGreater(OBSERVED_LIFT_MISS, 50.0 * droop)

    def test_ff_cannot_move_the_equilibrium_enough_to_explain_the_miss(self):
        # Even a maximally wrong/right feedforward (the full +/- gravity) shifts the drive equilibrium by at
        # most 2*gravity/Kp -- still microradian. The load-aware FF (measured pose + carton) is correct and
        # tiny; it cannot be the lever for a 0.098 rad miss. This is why PR 218's FF changed nothing.
        ff = gravity_ff_effort(self.grav, GOAL, MEAS, at_measured=True, ff_sign=-1.0, ff_min_nm=3.0,
                               payload_grav_fn=self.payload, payload_kg=CARTON_KG, held=True)
        # the FF is sized near the gravity it cancels (same order), not near the 29 kNm drive demand
        self.assertLess(abs(ff[LIFT]), DRIVE_MAX_FORCE)
        max_equilibrium_shift = 2.0 * abs(self._total_lift_gravity()) / DRIVE_KP
        self.assertLess(max_equilibrium_shift, 0.01 * OBSERVED_LIFT_MISS)


if __name__ == '__main__':
    unittest.main()
