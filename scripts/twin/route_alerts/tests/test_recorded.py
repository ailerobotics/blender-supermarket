"""Recorded-frames bridge + evidence seeders (pure; numpy only, no PNG backend).

The recorded bridge turns a saved run's ``mission-frames/frames.json`` + PNGs into
RouteFrame-ready data stamped with the run's REAL timestamps. PNG decode is injected
here (``image_loader``) so the test needs no OpenCV/Pillow; the index parser and the
pure numpy seeders carry the rest.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from twin.route_alerts.recorded import (RecordedFrameRef, floor_calibration,  # noqa: E402
                                        iter_recorded_frames, load_frame_index,
                                        seed_obstruction, seed_spill)


def _write_run(d, n=5, h=12, w=16):
    frames_dir = Path(d) / "mission-frames"
    frames_dir.mkdir(parents=True)
    index = {"nominal_fps": 30, "frames": [
        {"seq": i, "file": f"{i:06d}.png", "simulation_time": 1.0 + 0.25 * i,
         "shape": [h, w, 3]} for i in range(n)]}
    (frames_dir / "frames.json").write_text(json.dumps(index))
    return frames_dir, h, w, n


class IndexTests(unittest.TestCase):
    def test_parse_index(self):
        with tempfile.TemporaryDirectory() as d:
            _write_run(d)
            refs = load_frame_index(d)
            self.assertEqual(len(refs), 5)
            self.assertIsInstance(refs[0], RecordedFrameRef)
            self.assertEqual(refs[2].simulation_time, 1.5)
            self.assertEqual(refs[2].file, "000002.png")

    def test_accepts_mission_frames_dir_directly(self):
        with tempfile.TemporaryDirectory() as d:
            frames_dir, *_ = _write_run(d)
            self.assertEqual(len(load_frame_index(frames_dir)), 5)

    def test_missing_index_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_frame_index(d)


class IterTests(unittest.TestCase):
    def _loader_factory(self, h, w):
        seen = []

        def loader(path):
            seen.append(Path(path).name)
            return np.zeros((h, w, 3), dtype=np.uint8)

        return loader, seen

    def test_iter_yields_real_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            _, h, w, n = _write_run(d)
            loader, seen = self._loader_factory(h, w)
            out = list(iter_recorded_frames(d, image_loader=loader))
            self.assertEqual(len(out), n)
            rgb, t, seq = out[3]
            self.assertEqual(rgb.shape, (h, w, 3))
            self.assertEqual(t, 1.0 + 0.25 * 3)
            self.assertEqual(seq, 3)
            self.assertEqual(seen[0], "000000.png")

    def test_stride_and_limit(self):
        with tempfile.TemporaryDirectory() as d:
            _, h, w, n = _write_run(d, n=9)
            loader, _ = self._loader_factory(h, w)
            out = list(iter_recorded_frames(d, stride=3, limit=2, image_loader=loader))
            self.assertEqual([seq for _, _, seq in out], [0, 3])

    def test_bad_stride(self):
        with tempfile.TemporaryDirectory() as d:
            _write_run(d)
            with self.assertRaises(ValueError):
                list(iter_recorded_frames(d, stride=0))


class SeederTests(unittest.TestCase):
    def test_seed_obstruction_box_is_closer_than_floor(self):
        depth = seed_obstruction((100, 120), box_px=(40, 50, 70, 80),
                                 floor_depth_m=3.0, box_depth_m=2.0)
        self.assertEqual(depth.shape, (100, 120))
        self.assertAlmostEqual(float(depth[10, 10]), 3.0)  # floor
        self.assertAlmostEqual(float(depth[60, 55]), 2.0)  # inside the box
        # a floor pixel is strictly farther than a box pixel (protrudes toward cam)
        self.assertGreater(float(depth[10, 10]), float(depth[60, 55]))

    def test_seed_spill_changes_center_only(self):
        rgb = np.full((80, 100, 3), 200, dtype=np.uint8)
        out = seed_spill(rgb, center_px=(50, 40), radius_px=10, color=(10, 10, 10))
        self.assertFalse(np.array_equal(rgb, out))      # a copy was modified
        self.assertTrue(np.array_equal(rgb, np.full_like(rgb, 200)))  # input untouched
        self.assertLess(int(out[40, 50].max()), 200)    # center darkened
        self.assertEqual(int(out[0, 0].max()), 200)     # a far corner untouched

    def test_floor_calibration_maps_corners(self):
        calib = floor_calibration((720, 960), aisle_x=(0.0, 4.0), aisle_y=(0.0, 6.0))
        H = calib.image_to_floor
        self.assertEqual(H.shape, (3, 3))

        def to_store(u, v):
            p = H @ np.array([u, v, 1.0])
            return p[0] / p[2], p[1] / p[2]

        # bottom-left pixel -> (x~0, y~0); bottom-right -> (x~4, y~0)
        x, y = to_store(0, 719)
        self.assertAlmostEqual(x, 0.0, places=3)
        self.assertAlmostEqual(y, 0.0, places=2)
        x, y = to_store(959, 719)
        self.assertAlmostEqual(x, 4.0, places=2)
        # higher up the image is farther (larger y)
        _, y_far = to_store(0, 360)
        self.assertGreater(y_far, 0.0)
        self.assertIn("route-aisle", calib.aisle)


if __name__ == "__main__":
    unittest.main()
