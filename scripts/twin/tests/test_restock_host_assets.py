"""Coverage proof and exclusion policy tests, CPU only."""
import itertools
import unittest
from types import SimpleNamespace as S

import numpy as np

from twin.restock_host.generate_assets import cover_bounds, kinematic_exclusions


class AssetTests(unittest.TestCase):
    def test_cell_circumspheres_cover_box_interior_and_boundary(self):
        low, high = np.array([-.17, -.02, .1]), np.array([.27, .03, .103])
        centers, radius = cover_bounds(low, high, .05)
        points = np.array(list(itertools.product(*[np.linspace(a, b, 17) for a, b in zip(low, high)])))
        nearest = np.linalg.norm(points[:, None] - centers[None, :], axis=2).min(axis=1)
        self.assertLessEqual(float(nearest.max()), radius)
        self.assertLessEqual(float(np.sqrt(3)*.05/2), .044)

    def test_degenerate_and_nonfinite_bounds(self):
        centers, radius = cover_bounds([0, 0, 0], [0, 0, 0], .05)
        self.assertEqual(len(centers), 1)
        self.assertGreater(radius, 0)
        for low, high, pitch in [([0]*3, [-1]*3, .05), ([0]*3, [1]*3, 0),
                                 ([0]*3, [float('nan')]*3, .05)]:
            with self.assertRaises(ValueError):
                cover_bounds(low, high, pitch)

    def test_excludes_only_rigid_or_directly_adjacent_assemblies(self):
        robot = S(links={n: S(collisions=[True]) for n in 'abcde'}, joints=[
            S(parent='a', child='b', jtype='fixed'),
            S(parent='b', child='c', jtype='revolute'),
            S(parent='c', child='d', jtype='revolute'),
            S(parent='c', child='e', jtype='prismatic')])
        excluded = kinematic_exclusions(robot)
        self.assertEqual(excluded[('a', 'b')], 'SameRigidAssembly')
        self.assertIn(('a', 'c'), excluded)
        self.assertNotIn(('a', 'd'), excluded)
        self.assertNotIn(('d', 'e'), excluded)


if __name__ == '__main__':
    unittest.main()
