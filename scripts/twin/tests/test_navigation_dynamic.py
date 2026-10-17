"""Deterministic CPU fixtures for the moving-obstacle layer.

These do not certify PhysX dynamics, perception or a sensing run; they prove the
tracker/predictor contract and the add-only safety invariant.
"""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.navigation import GridMap, Observation
from twin.navigation.dynamic import (Detection, DynamicObstacleConfig,
    DynamicObstacleLayer, TrackedObstacle, predicted_points)


class ConfigValidationTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        DynamicObstacleConfig()

    def test_rejects_bad_values(self):
        for bad in (dict(association_distance=0), dict(moving_speed=-1),
                    dict(prediction_horizon=-0.1), dict(prediction_step=0),
                    dict(velocity_gain=0), dict(velocity_gain=1.5),
                    dict(confirm_frames=0), dict(confirm_frames=1.5),
                    dict(drop_after=0), dict(point_spacing=0), dict(max_tracks=0)):
            with self.assertRaises(ValueError):
                DynamicObstacleConfig(**bad)

    def test_detection_validation(self):
        Detection((1.0, 2.0), 0.3)
        for bad in (((1.0,), 0.3), ((1.0, float('nan')), 0.3), ((0, 0), -1)):
            with self.assertRaises(ValueError):
                Detection(*bad)
        with self.assertRaises(ValueError):
            Detection((0, 0), 0.3, source='')


class TrackerTests(unittest.TestCase):
    def test_confirm_frames_debounce(self):
        layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=3))
        self.assertEqual(layer.update([Detection((1, 1), 0.3)], 0.0), ())
        self.assertEqual(layer.update([Detection((1, 1), 0.3)], 0.1), ())
        tracks = layer.update([Detection((1, 1), 0.3)], 0.2)
        self.assertEqual(len(tracks), 1)

    def test_velocity_estimate_of_moving_detection(self):
        layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=1, velocity_gain=1.0))
        layer.update([Detection((0.0, 0.0), 0.3)], 0.0)
        tracks = layer.update([Detection((0.5, 0.0), 0.3)], 0.5)
        self.assertEqual(len(tracks), 1)
        self.assertAlmostEqual(tracks[0].velocity[0], 1.0, places=6)
        self.assertAlmostEqual(tracks[0].velocity[1], 0.0, places=6)
        self.assertGreaterEqual(tracks[0].speed, 0.10)

    def test_association_keeps_single_track(self):
        layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=1))
        ids = set()
        x = 0.0
        for i in range(5):
            tracks = layer.update([Detection((x, 0.0), 0.3)], i * 0.2)
            x += 0.1
            ids.update(t.track_id for t in tracks)
        self.assertEqual(len(ids), 1)

    def test_far_jump_starts_new_track(self):
        cfg = DynamicObstacleConfig(confirm_frames=1, association_distance=0.5)
        layer = DynamicObstacleLayer(cfg)
        layer.update([Detection((0.0, 0.0), 0.3)], 0.0)
        tracks = layer.update([Detection((5.0, 0.0), 0.3)], 0.1)
        # The old track is retained (not yet aged out) and a new one is seeded.
        self.assertEqual(len(tracks), 2)

    def test_drop_after_ages_out(self):
        cfg = DynamicObstacleConfig(confirm_frames=1, drop_after=0.3)
        layer = DynamicObstacleLayer(cfg)
        layer.update([Detection((0.0, 0.0), 0.3)], 0.0)
        self.assertEqual(len(layer.update([], 0.1)), 1)   # kept briefly
        self.assertEqual(len(layer.update([], 0.5)), 0)   # aged out

    def test_out_of_order_stamp_raises(self):
        layer = DynamicObstacleLayer()
        layer.update([], 1.0)
        with self.assertRaises(ValueError):
            layer.update([], 0.5)

    def test_max_tracks_cap(self):
        cfg = DynamicObstacleConfig(confirm_frames=1, max_tracks=2, association_distance=0.1)
        layer = DynamicObstacleLayer(cfg)
        layer.update([Detection((0, 0), 0.2), Detection((10, 0), 0.2),
                      Detection((20, 0), 0.2)], 0.0)
        self.assertLessEqual(len(layer._tracks), 2)

    def test_determinism(self):
        def run():
            layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=1))
            out = []
            for i in range(6):
                layer.update([Detection((i * 0.2, 0.0), 0.3)], i * 0.2)
                out.append(layer.emitted_points())
            return out
        self.assertEqual(run(), run())


class PredictorTests(unittest.TestCase):
    def test_static_obstacle_emits_only_current_extent(self):
        cfg = DynamicObstacleConfig(prediction_horizon=2.0)
        obs = TrackedObstacle(0, (0.0, 0.0), (0.0, 0.0), 0.3, 0.0, 5, 'x')
        pts = predicted_points(obs, cfg)
        self.assertTrue(all(math.hypot(x, y) <= 0.3 + 1e-9 for x, y in pts))

    def test_moving_obstacle_blocks_forward_path(self):
        cfg = DynamicObstacleConfig(prediction_horizon=1.0, prediction_step=0.25)
        obs = TrackedObstacle(0, (0.0, 0.0), (1.0, 0.0), 0.3, 0.0, 5, 'x')
        pts = predicted_points(obs, cfg)
        # Predicted path reaches ~1 m ahead (+ radius), none behind beyond radius.
        self.assertGreater(max(x for x, _ in pts), 1.0)
        self.assertGreaterEqual(min(x for x, _ in pts), -0.3 - 1e-9)

    def test_points_cover_disk_without_gaps(self):
        cfg = DynamicObstacleConfig(point_spacing=0.2)
        obs = TrackedObstacle(0, (0.0, 0.0), (0.0, 0.0), 0.5, 0.0, 5, 'x')
        pts = predicted_points(obs, cfg)
        # A probe anywhere in the disk is within point_spacing of some emitted point.
        for px, py in ((0.25, 0.0), (0.0, 0.4), (-0.3, 0.3)):
            self.assertLessEqual(min(math.hypot(px - x, py - y) for x, y in pts), 0.2 + 1e-9)


class AddOnlyInvariantTests(unittest.TestCase):
    def _obs(self, obstacles):
        return Observation(0.2, tuple(obstacles), True, True, 0.0, 'test_fixture', 10)

    def test_apply_is_superset_of_raw_returns(self):
        layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=1))
        layer.update([Detection((2.0, 0.0), 0.3)], 0.1)
        layer.update([Detection((2.0, 0.0), 0.3)], 0.2)
        raw = ((5.0, 5.0), (-4.0, 1.0))
        out = layer.apply(self._obs(raw))
        for p in raw:
            self.assertIn(p, out.obstacles)
        self.assertGreater(len(out.obstacles), len(raw))

    def test_no_tracks_leaves_observation_untouched(self):
        layer = DynamicObstacleLayer()
        obs = self._obs(((1.0, 1.0),))
        self.assertIs(layer.apply(obs), obs)

    def test_predicted_crossing_blocks_a_cell_the_raw_return_leaves_free(self):
        """The gate's free() primitive: a moving obstacle's predicted path blocks a
        lane cell that its current (off-lane) detection would leave clear."""
        grid = GridMap((-10, -10, 10, 10))
        layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=1, velocity_gain=1.0,
                                     prediction_horizon=1.5, prediction_step=0.25))
        layer.update([Detection((2.0, 1.1), 0.3)], 0.1)
        layer.update([Detection((2.0, 0.6), 0.3)], 0.2)  # crossing toward the lane (-y)
        lane_cell, clearance = (2.0, 0.0), 0.5
        raw = ((2.0, 0.6),)  # the latest raw detection, still off the lane
        # Raw detection alone leaves the lane cell free (0.6 m away > clearance).
        self.assertTrue(grid.free(lane_cell, clearance, raw))
        # The predicted crossing path puts occupancy on the lane cell: now blocked.
        self.assertFalse(grid.free(lane_cell, clearance, layer.emitted_points()))

    def test_route_detours_around_predicted_path(self):
        """A planned straight route is forced to detour by the predicted sweep."""
        grid = GridMap((-10, -10, 10, 10), resolution=0.25)
        start, goal = (0.0, 0.0), (6.0, 0.0)
        direct = grid.plan(start, goal, 0.3)
        self.assertEqual(direct, [goal])  # unobstructed straight shot
        layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=1, velocity_gain=1.0,
                                     prediction_horizon=1.2, prediction_step=0.2))
        layer.update([Detection((3.0, 1.0), 0.4)], 0.1)
        layer.update([Detection((3.0, 0.5), 0.4)], 0.2)  # sweeps across the lane
        detour = grid.plan(start, goal, 0.3, layer.emitted_points())
        self.assertTrue(detour)
        self.assertGreater(len(detour), 1)  # no longer a single straight segment


if __name__ == '__main__':
    unittest.main()
