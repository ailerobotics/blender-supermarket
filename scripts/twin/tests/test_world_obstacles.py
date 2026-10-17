"""CPU-only tests for the static world-obstacle box geometry + clearance (no ROS, no GPU, no dataset)."""
import math
import os
import unittest

import numpy as np

from twin.restock_host.world_obstacles import (Box, clearance, boxes_from_specs,
                                               carton_arm_clearance, carton_cover_local)

_URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'
_JOINTS = ['ur_shoulder_pan_joint', 'ur_shoulder_lift_joint', 'ur_elbow_joint',
           'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']


class TestBoxSignedDistance(unittest.TestCase):
    def setUp(self):
        self.box = Box('b', size=(2.0, 2.0, 2.0), center=(0.0, 0.0, 0.0))  # [-1,1]^3

    def test_center_is_inside_negative(self):
        self.assertAlmostEqual(float(self.box.signed_distance([[0, 0, 0]])[0]), -1.0, places=6)

    def test_face_is_zero(self):
        self.assertAlmostEqual(float(self.box.signed_distance([[1, 0, 0]])[0]), 0.0, places=6)

    def test_outside_axis_distance(self):
        self.assertAlmostEqual(float(self.box.signed_distance([[2.5, 0, 0]])[0]), 1.5, places=6)

    def test_outside_corner_distance(self):
        # point (2,2,1): distance to corner (1,1,1) is sqrt(1+1+0)
        self.assertAlmostEqual(float(self.box.signed_distance([[2, 2, 1]])[0]), math.sqrt(2.0), places=6)

    def test_offcenter_box(self):
        box = Box('b', size=(1.0, 1.0, 1.0), center=(5.0, 0.0, 0.0))  # [4.5,5.5]x[-.5,.5]^2
        self.assertAlmostEqual(float(box.signed_distance([[4.0, 0, 0]])[0]), 0.5, places=6)

    def test_yaw_rotation_distance_invariant_at_center(self):
        box = Box('b', size=(2.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, math.pi / 2))
        # yaw +90deg about z maps the box long axis (half 1.0) onto world y: a point 2 m along +y
        # is 1.0 m outside, while 2 m along +x meets the short axis (half 0.5) and is 1.5 m outside.
        self.assertAlmostEqual(float(box.signed_distance([[0, 2, 0]])[0]), 1.0, places=6)
        self.assertAlmostEqual(float(box.signed_distance([[2, 0, 0]])[0]), 1.5, places=6)


class TestClearance(unittest.TestCase):
    def test_clearance_subtracts_radius(self):
        box = Box('b', size=(2.0, 2.0, 2.0), center=(0.0, 0.0, 0.0))
        centres = np.array([[3.0, 0.0, 0.0]])   # surface distance 2.0
        radii = np.array([0.5])
        self.assertAlmostEqual(clearance(box, centres, radii), 1.5, places=6)

    def test_penetration_is_negative(self):
        box = Box('b', size=(2.0, 2.0, 2.0), center=(0.0, 0.0, 0.0))
        centres = np.array([[1.1, 0.0, 0.0]])   # 0.1 outside the face
        radii = np.array([0.3])                 # sphere reaches 0.2 inside
        self.assertLess(clearance(box, centres, radii), 0.0)
        self.assertAlmostEqual(clearance(box, centres, radii), 0.1 - 0.3, places=6)

    def test_min_over_sphere_set(self):
        box = Box('b', size=(2.0, 2.0, 2.0), center=(0.0, 0.0, 0.0))
        centres = np.array([[5.0, 0, 0], [1.2, 0, 0]])
        radii = np.array([0.1, 0.1])
        self.assertAlmostEqual(clearance(box, centres, radii), 0.2 - 0.1, places=6)


class TestBoxesFromSpecs(unittest.TestCase):
    def test_build_with_defaults(self):
        boxes = boxes_from_specs([{'id': 'cart', 'size': [0.8, 0.68, 0.6], 'center': [0, -0.75, 1.0]}])
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].id, 'cart')
        self.assertEqual(boxes[0].rpy, (0.0, 0.0, 0.0))

    def test_rejects_missing_fields(self):
        with self.assertRaises(ValueError):
            boxes_from_specs([{'id': 'x', 'size': [1, 1, 1]}])

    def test_rejects_nonpositive_size(self):
        with self.assertRaises(ValueError):
            boxes_from_specs([{'id': 'x', 'size': [1, 0, 1], 'center': [0, 0, 0]}])


class TestCartonCoverLocal(unittest.TestCase):
    """The shared carton/placeholder cover: a conservative sphere grid of a box in the local frame."""

    def test_cover_encloses_box_corners(self):
        grid, radius = carton_cover_local((0.6, 0.4, 0.16), offset_tcp=(0.0, 0.0, 0.0), pitch=0.08)
        self.assertGreater(len(grid), 0)
        self.assertGreater(radius, 0.0)
        # every box corner is within `radius` of some grid centre (conservative enclosure)
        half = np.array([0.3, 0.2, 0.08])
        corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        for c in corners:
            self.assertLessEqual(float(np.min(np.linalg.norm(grid - c, axis=1))), radius + 1e-9)

    def test_rejects_nonpositive_dims(self):
        for dims in [(0.1, 0.0, 0.1), (-0.1, 0.1, 0.1), (0.1, 0.1, float('nan'))]:
            with self.assertRaises(ValueError):
                carton_cover_local(dims)


@unittest.skipUnless(os.path.isfile(_URDF) and os.path.isfile(_XRDF),
                     'mission CAD assets not present (dataset-only regression)')
class TestCartonArmClearance(unittest.TestCase):
    """Regression for the planner-side finding: the tuck GOAL drives the carton into ur_upper_arm
    even though cuMotion reports the plan collision-free -- so the arm planner is not catching the
    carton-vs-arm contact (run-249/250). The attached_object self-collision declaration (declaration
    + placeholder spheres) is what makes cuMotion catch it. Reproduced on CPU from the asset URDF FK
    + XRDF spheres."""

    def _cfg(self, w1):
        return dict(zip(_JOINTS, [1.2251, -2.2422, 1.0802, w1, 1.0608, -1.1675]))

    def test_carton_clear_at_pinned_but_hits_upper_arm_at_full_fold(self):
        pinned = carton_arm_clearance(_URDF, _XRDF, self._cfg(0.6965))
        full = carton_arm_clearance(_URDF, _XRDF, self._cfg(2.1066))
        self.assertTrue(pinned['clear'], f"expected clear at pinned fold: {pinned}")
        self.assertFalse(full['clear'], f"expected carton-vs-arm contact at full fold: {full}")
        self.assertEqual(full['worst_link'], 'ur_upper_arm_link')
        # monotone approach as the fold completes
        self.assertLess(full['clearance_m'], pinned['clearance_m'])


if __name__ == '__main__':
    unittest.main()
