"""CPU tests for the joint-space IK branch selection (host verdict 2026-10-14).

cuMotion's task-space IK is systemically broken in this XRDF, so the port uses KDL compute_ik ->
cuMotion joint-space planning, hardened with: stow-first seeds, nearest-in-limit-branch
normalisation (so KDL's far wrapped branches -- run-80's joint2 = -4.45 -- do not reach cuMotion as
self-colliding targets), and a retry to the next seed when cuMotion rejects a c-space target. No ROS
node: the port is built via __new__ and only the pure branch helpers are exercised.
"""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock_manipulation.ros_adapter import RosMotionPort

LIMITS = tuple((-2 * math.pi, 2 * math.pi) for _ in range(6))
STOW = (0.0, -1.57, 1.57, -1.57, -1.57, 0.0)


def _port(stow=STOW, limits=LIMITS):
    p = RosMotionPort.__new__(RosMotionPort)
    p.limits = limits
    p.stow = stow
    return p


class NearestBranch(unittest.TestCase):
    def test_all_joints_stay_in_limits(self):
        out = _port()._nearest_branch([5.1284, -4.44626, 2.48316, -1.17849, 1.1506, -3.14159])
        self.assertIsNotNone(out)
        for v, (lo, hi) in zip(out, LIMITS):
            self.assertTrue(lo <= v <= hi, v)

    def test_unwraps_toward_nominal(self):
        # joint0 = 5.1284 with nominal 0 -> nearest branch is 5.1284 - 2*pi (~-1.155), not the +value
        out = _port()._nearest_branch([5.1284, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.assertAlmostEqual(out[0], 5.1284 - 2 * math.pi, places=3)

    def test_identity_when_already_at_nominal(self):
        out = _port()._nearest_branch(list(STOW))
        self.assertEqual([round(v, 6) for v in out], [round(v, 6) for v in STOW])

    def test_none_when_no_in_limit_branch(self):
        tight = _port(limits=tuple((-0.1, 0.1) for _ in range(6)))
        self.assertIsNone(tight._nearest_branch([3.0] * 6))

    def test_no_stow_normalises_toward_zero(self):
        out = _port(stow=None)._nearest_branch([6.4, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(out[0], 6.4 - 2 * math.pi, places=3)   # nearest 0


class Seeds(unittest.TestCase):
    def test_stow_first_then_measured_then_perturbations(self):
        seeds = _port()._seed_configs((0.1,) * 6)
        self.assertEqual(seeds[0], STOW)                 # stow seeded first (central, in-limits)
        self.assertIn((0.1,) * 6, seeds)                 # measured start present
        self.assertGreaterEqual(len(seeds), 3)           # + perturbations
        self.assertEqual(len(seeds), len(set(seeds)))    # de-duplicated

    def test_no_stow_falls_back_to_measured_only(self):
        self.assertEqual(_port(stow=None)._seed_configs((0.2,) * 6), [(0.2,) * 6])

    def test_perturbations_clamped_into_limits(self):
        tight = _port(limits=tuple((-1.6, 1.6) for _ in range(6)))
        for seed in tight._seed_configs((0.0,) * 6):
            for v, (lo, hi) in zip(seed, tight.limits):
                self.assertTrue(lo <= v <= hi)


if __name__ == '__main__':
    unittest.main()
