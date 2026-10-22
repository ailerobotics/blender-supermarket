"""CPU unit test for the fm356 option-A gravity_ff (build_arm_gravity).

Computes the arm's generalized gravity on CPU (URDF FK + fixed per-body masses), bypassing Isaac's
get_generalized_gravity_forces (which mis-ordered the arm DOFs: run-279 fed +78.76 Nm to the vertical
shoulder_pan; run-281's full-DOF+dof_names mapping read all zeros). Asserts the fix reproduces
gravity_ff_proof.py: shoulder_pan G=0 at every config (the invariant), lift/elbow the dominant loads,
and the magnitudes match the proof. Skips if the machine-specific mobile URDF is absent.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from twin.restock.physics_setup import build_arm_gravity, _ISAAC_BODY_MASSES  # noqa: E402
from twin.restock_manipulation.cad_tool import ARM_JOINTS  # noqa: E402

URDF = os.path.expanduser('~/datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf')

# Reference G (Nm) from gravity_ff_proof.py (Isaac masses incl. payload), arm order [pan,lift,elbow,w1,w2,w3].
PROOF_G = {
    'home':     [0.0, -33.0, 78.8, 13.7, 0.0, 0.0],
    'pan':      [0.0, -41.2, 48.7, 21.0, -7.9, 1.0],
    'preplace': [0.0, -186.3, -55.0, -11.0, 0.0, 0.0],
    'place':    [0.0, -183.8, -55.0, -11.0, 0.0, 0.0],
}
CONFIGS = {
    'home':     [0.0002, -2.5, 1.7003, -0.6999, -1.5708, 0.0],
    'pan':      [-1.822, -2.27, 1.0, 0.65, 0.8, -1.2545],
    'preplace': [-3.702, -3.486, 1.425, -1.081, 0.561, 0.0],
    'place':    [-3.702, -3.536, 1.475, -1.081, 0.561, 0.0],
}
PAN, LIFT, ELBOW = 0, 1, 2


@unittest.skipUnless(os.path.isfile(URDF), f'mobile URDF absent: {URDF}')
class BuildArmGravityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from twin import urdf_probe as up
        cls.grav = staticmethod(build_arm_gravity(up.parse_urdf(URDF), list(ARM_JOINTS), _ISAAC_BODY_MASSES))

    def test_pan_is_zero_every_config(self):
        # The decisive invariant: shoulder_pan is a vertical yaw axis -> gravity torque identically 0.
        for name, cfg in CONFIGS.items():
            self.assertLess(abs(self.grav(cfg)[PAN]), 1e-2, f'pan G != 0 at {name}')

    def test_lift_elbow_dominant_and_nontrivial(self):
        # Not all-zeros (the run-281 failure); the horizontal-axis joints carry the real load.
        for name, cfg in CONFIGS.items():
            g = np.abs(self.grav(cfg))
            self.assertGreater(max(g[LIFT], g[ELBOW]), 10.0, f'no lift/elbow load at {name}')
            self.assertIn(int(np.argmax(g)), (LIFT, ELBOW), f'dominant joint not lift/elbow at {name}')

    def test_matches_proof_within_tolerance(self):
        for name, cfg in CONFIGS.items():
            got = self.grav(cfg)
            for j, (g, ref) in enumerate(zip(got, PROOF_G[name])):
                self.assertLess(abs(g - ref), 1.5,
                                f'{name} joint {j}: {g:.2f} vs proof {ref:.2f}')


class MassConstantTests(unittest.TestCase):
    def test_payload_and_arm_links_present(self):
        for ln in ('ur_shoulder_link', 'ur_upper_arm_link', 'ur_forearm_link', 'big_cube',
                   'gripper_pad_tcp'):
            self.assertIn(ln, _ISAAC_BODY_MASSES)


if __name__ == '__main__':
    unittest.main()
