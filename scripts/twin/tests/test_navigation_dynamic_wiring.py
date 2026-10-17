"""CPU contract tests for the opt-in moving-obstacle wiring.

Prove the factory builds the layer only when configured, that an unconfigured
run is byte-identical, that detections=None leaves the sensed observation
untouched, and that supplied detections are folded in add-only. No ROS, GPU,
simulator or fixture-provenance input drives a base here.
"""
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.navigation.factory import nav_factory
from twin.navigation.runtime import SensorNavigation
from twin.navigation.dynamic import Detection, DynamicObstacleLayer, DynamicObstacleConfig
from twin.navigation.core import Odometry, Result
from twin.navigation.sensors import Scan

MANIFEST = {'map_bounds': [-6., 6., -4., 4.],
            'static_obstacles': [{'min': [-6., -4.], 'max': [-5.6, 4.]}]}
SENSOR_CFG = {'wheel_radius': 0.16, 'wheel_separation': 0.70,
              'initial_pose': [0., 0., 0.], 'initial_map_pose': [0., 0., 0.]}


def _build(nav_mode, **nav_extra):
    nav = {'environment': MANIFEST}
    nav.update(nav_extra)
    return nav_factory(app=None, assembly=object(), plan=SimpleNamespace(nav_mode=nav_mode),
                       config={'nav_factory': 'x:y', 'navigation': nav},
                       output=tempfile.gettempdir())


class FactoryWiringTests(unittest.TestCase):
    def test_default_off_no_layer(self):
        obj = _build('sensor', sensor=SENSOR_CFG)
        self.assertIsNone(obj.dynamic_layer)

    def test_layer_built_from_config(self):
        obj = _build('sensor', sensor=SENSOR_CFG,
                     dynamic_obstacles={'moving_speed': 0.2, 'prediction_horizon': 2.0})
        self.assertIsInstance(obj.dynamic_layer, DynamicObstacleLayer)
        self.assertEqual(obj.dynamic_layer.config.moving_speed, 0.2)
        self.assertEqual(obj.dynamic_layer.config.prediction_horizon, 2.0)

    def test_dynamic_obstacles_rejected_for_non_sensor_mode(self):
        with self.assertRaises(ValueError):
            _build('reference', dynamic_obstacles={})

    def test_bad_dynamic_config_rejected(self):
        with self.assertRaises(ValueError):
            _build('sensor', sensor=SENSOR_CFG, dynamic_obstacles={'moving_speed': -1})

    def test_blocked_timeout_patience_flows_through(self):
        obj = _build('reference', blocked_timeout=45.)
        self.assertEqual(obj.blocked_timeout, 45.)


class _RecordingController:
    """Captures the observation SensorNavigation hands to the inner controller."""
    def __init__(self):
        self.terminal = None
        self.seen = None

    def step(self, dt, localized, sensed, *, now):
        self.seen = sensed
        return Result(reason='tracking')


class _CannedWheels:
    def update(self, left, right, stamp):
        return Odometry((0., 0., 0.), stamp, 0., 0., source='wheel_odometry')


class _CannedLocalizer:
    def update(self, raw, scan):
        return Odometry((0., 0., 0.), raw.stamp, 0., 0.,
                        source='wheel_odometry_localized', position_uncertainty=0.10)


class StepWiringTests(unittest.TestCase):
    def _nav(self, layer):
        return SensorNavigation(_RecordingController(), _CannedWheels(),
                                _CannedLocalizer(), dynamic_layer=layer)

    def _scan(self, t):
        # All beams return a far (but < range_max) hit, so the base observation
        # already carries static points; the layer only ever adds more.
        return Scan(t, (8.0,) * 72, -math.pi, 2 * math.pi / 72, 10.0)

    def test_detections_none_is_byte_identical(self):
        layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=1))
        nav = self._nav(layer)
        base = self._nav(None)
        nav.step(0.05, (0., 0.), self._scan(1.), now=1., coupled=True,
                 arm_transport=True, articulation=0.0, detections=None)
        base.step(0.05, (0., 0.), self._scan(1.), now=1., coupled=True,
                  arm_transport=True, articulation=0.0)
        self.assertEqual(nav.controller.seen.obstacles, base.controller.seen.obstacles)

    def test_detections_are_folded_in_add_only(self):
        layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=1))
        nav = self._nav(layer)
        plain = self._nav(None)
        det = [Detection((2.0, 0.0), 0.3)]
        nav.step(0.05, (0., 0.), self._scan(1.), now=1., coupled=True,
                 arm_transport=True, articulation=0.0, detections=det)
        plain.step(0.05, (0., 0.), self._scan(1.), now=1., coupled=True,
                   arm_transport=True, articulation=0.0)
        augmented = nav.controller.seen.obstacles
        raw = plain.controller.seen.obstacles
        # Every raw return survives, and extra occupancy around the detection added.
        self.assertTrue(set(raw).issubset(set(augmented)))
        self.assertGreater(len(augmented), len(raw))
        self.assertTrue(any(math.hypot(x - 2.0, y) <= 0.3 + 1e-9 for x, y in augmented))


if __name__ == '__main__':
    unittest.main()
