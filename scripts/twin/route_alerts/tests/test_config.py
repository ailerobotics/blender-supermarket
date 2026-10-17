"""Pure-logic tests for route-alerts config parsing (no store-vision / numpy)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.route_alerts.config import DetectorPref, RouteAlertsConfig


class DetectorPrefTests(unittest.TestCase):
    def test_bool_shorthand(self):
        self.assertEqual(DetectorPref.from_obj(True), DetectorPref(enabled=True))
        self.assertFalse(DetectorPref.from_obj(False).enabled)
        self.assertFalse(DetectorPref.from_obj(None).enabled)

    def test_dict_fields(self):
        p = DetectorPref.from_obj({"enabled": True, "gpu": True, "device": "cuda",
                                   "tuning": {"persist_seconds": 0.4}})
        self.assertTrue(p.enabled and p.gpu)
        self.assertEqual(p.device, "cuda")
        self.assertEqual(p.tuning, {"persist_seconds": 0.4})

    def test_invalid(self):
        with self.assertRaises(ValueError):
            DetectorPref.from_obj(3.14)


class RouteAlertsConfigTests(unittest.TestCase):
    def test_defaults_cpu_classical(self):
        c = RouteAlertsConfig()
        self.assertTrue(c.spill.enabled and not c.spill.gpu)
        self.assertTrue(c.obstruction.enabled and not c.obstruction.gpu)
        self.assertFalse(c.strays.enabled)  # semantic / GPU path, off by default
        self.assertTrue(c.strays.gpu)
        self.assertTrue(c.any_enabled)

    def test_true_means_defaults(self):
        self.assertEqual(RouteAlertsConfig.from_obj(True), RouteAlertsConfig())

    def test_absent_is_error(self):
        # Absence is handled by the caller NOT constructing the runner; from_obj on a
        # falsey value is a programming error.
        for bad in (None, False):
            with self.assertRaises(ValueError):
                RouteAlertsConfig.from_obj(bad)

    def test_dict_override(self):
        c = RouteAlertsConfig.from_obj({
            "spill": {"enabled": False},
            "obstruction": {"enabled": True, "gpu": True, "device": "cuda"},
            "cadence": 3, "staff_tasks": False,
            "aisles": {"a1": {"x0": 0, "y0": 0, "x1": 2, "y1": 10}},
            "footprint_pad_m": 0.15, "obstruction_source": "route_obs",
        })
        self.assertFalse(c.spill.enabled)
        self.assertTrue(c.obstruction.gpu)
        self.assertEqual(c.cadence, 3)
        self.assertFalse(c.staff_tasks)
        self.assertEqual(c.footprint_pad_m, 0.15)
        self.assertIn("a1", c.aisles)

    def test_bad_cadence_and_pad(self):
        with self.assertRaises(ValueError):
            RouteAlertsConfig(cadence=0)
        with self.assertRaises(ValueError):
            RouteAlertsConfig(footprint_pad_m=-1.0)


if __name__ == "__main__":
    unittest.main()
