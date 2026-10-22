"""Tests for the obstruction->dynobs Detection seam (pure; no store-vision)."""
import math
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.navigation.dynamic import Detection
from twin.route_alerts.costmap import footprint_to_detection, obstructions_to_detections


@dataclass
class _Ob:
    """Duck-typed stand-in for store-vision obstruction.passability.Obstruction."""
    footprint: tuple

    @property
    def center(self):
        x, y, w, h = self.footprint
        return (x + w / 2, y + h / 2)

    @property
    def radius_m(self):
        _, _, w, h = self.footprint
        return 0.5 * math.hypot(w, h)


class CostmapSeamTests(unittest.TestCase):
    def test_footprint_center_and_radius(self):
        d = footprint_to_detection((2.0, 3.0, 2.0, 2.0), source="route_obstruction")
        self.assertIsInstance(d, Detection)
        self.assertEqual(d.position, (3.0, 4.0))  # centre
        self.assertAlmostEqual(d.radius, 0.5 * math.hypot(2.0, 2.0))
        self.assertEqual(d.source, "route_obstruction")

    def test_pad_adds_to_radius_only(self):
        d = footprint_to_detection((0, 0, 2, 0), source="s", pad_m=0.5)
        self.assertEqual(d.position, (1.0, 0.0))
        self.assertAlmostEqual(d.radius, 1.0 + 0.5)  # half of width-2 box + pad

    def test_obstructions_roundtrip_center_radius(self):
        obs = [_Ob((1.0, 1.0, 2.0, 2.0)), _Ob((5.0, 0.0, 1.0, 1.0))]
        dets = obstructions_to_detections(obs, source="route_obstruction", pad_m=0.1)
        self.assertEqual(len(dets), 2)
        self.assertEqual(dets[0].position, (2.0, 2.0))
        self.assertTrue(all(d.source == "route_obstruction" for d in dets))
        # radius is the object's own extent + pad, never the robot clearance
        self.assertAlmostEqual(dets[0].radius, 0.5 * math.hypot(2, 2) + 0.1)

    def test_non_finite_footprint_skipped(self):
        dets = obstructions_to_detections([_Ob((float("nan"), 0, 1, 1))],
                                          source="s")
        self.assertEqual(dets, [])

    def test_detections_are_dynobs_contract(self):
        # The produced objects are exactly what SensorNavigation.step(detections=)
        # and DynamicObstacleLayer.update() accept.
        from twin.navigation.dynamic import DynamicObstacleLayer
        layer = DynamicObstacleLayer()
        dets = obstructions_to_detections([_Ob((5.0, 0.0, 0.6, 0.6))], source="route_obstruction")
        layer.update(dets, 0.0)
        layer.update(dets, 0.1)  # confirm_frames default 2
        self.assertTrue(len(layer.emitted_points()) > 0)


if __name__ == "__main__":
    unittest.main()
