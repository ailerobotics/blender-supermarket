"""RouteFrame construction / validation (needs numpy; to_store_frame gated)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:
    _HAVE_NUMPY = False

try:
    import storevision  # noqa: F401
    _HAVE_SV = True
except Exception:
    _HAVE_SV = False

if _HAVE_NUMPY:
    from twin.route_alerts.frames import RouteFrame


@unittest.skipUnless(_HAVE_NUMPY, "numpy required")
class RouteFrameTests(unittest.TestCase):
    def _rgb(self, h=8, w=8, c=3):
        return np.zeros((h, w, c), dtype=np.uint8)

    def test_basic_and_size(self):
        f = RouteFrame(camera_id="nav", seq=1, rgb=self._rgb(6, 10), capture_time=1.0)
        self.assertEqual(f.size, (10, 6))  # (w, h)

    def test_rgba_truncated_to_rgb(self):
        f = RouteFrame(camera_id="nav", seq=0, rgb=self._rgb(4, 4, 4), capture_time=0.0)
        self.assertEqual(f.rgb.shape[2], 3)

    def test_depth_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            RouteFrame(camera_id="n", seq=0, rgb=self._rgb(4, 4),
                       depth=np.zeros((5, 5), dtype=np.float32), capture_time=0.0)

    def test_bad_rgb_rejected(self):
        with self.assertRaises(ValueError):
            RouteFrame(camera_id="n", seq=0, rgb=np.zeros((4, 4), dtype=np.uint8),
                       capture_time=0.0)

    @unittest.skipUnless(_HAVE_SV, "store-vision required")
    def test_to_store_frame_folds_agents_for_masking(self):
        f = RouteFrame(camera_id="nav", seq=2, rgb=self._rgb(), capture_time=3.0,
                       agents=[{"bbox": (1, 1, 2, 2), "label": "person"},
                               (3, 3, 2, 2)])
        sf = f.to_store_frame()
        self.assertEqual(sf.camera_id, "nav")
        self.assertEqual(len(sf.meta["agents"]), 2)
        self.assertEqual(sf.meta["agents"][0].label, "person")


if __name__ == "__main__":
    unittest.main()
