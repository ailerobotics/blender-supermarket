"""The findings/act mode split: the nav seam, the record, the live wiring (pure CPU).

No store-vision: the runner imports the bridge lazily, so a runner constructed over a
config with no frames never touches store-vision. ``twin.navigation.dynamic.Detection``
is pure. These tests pin that the WORKING DEFAULT (findings) feeds navigation nothing
and the SMART variant (act) feeds the obstruction -- the one byte-identical-vs-not
guarantee the captain asked to keep one working version of.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from twin.navigation.dynamic import Detection  # noqa: E402
from twin.route_alerts.nav_source import source_from_hook  # noqa: E402
from twin.route_alerts.runner import RouteAlertsRunner  # noqa: E402


def _runner(mode, sink=None):
    r = RouteAlertsRunner({"mode": mode}, robot_id="shelfbot", sink_dir=sink)
    # Stand in a confirmed obstruction detection (as the bridge would have produced),
    # bypassing store-vision which is not importable in the pure test interpreter.
    r._latest_detections = [Detection((2.0, 2.3), 0.5, source="route_obstruction")]
    return r


class LatestDetectionsGateTests(unittest.TestCase):
    def test_findings_feeds_nav_nothing(self):
        self.assertEqual(_runner("findings").latest_detections(), [])

    def test_act_feeds_the_obstruction(self):
        dets = _runner("act").latest_detections()
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].source, "route_obstruction")

    def test_default_mode_is_findings(self):
        r = RouteAlertsRunner(True)
        r._latest_detections = [Detection((1.0, 1.0), 0.3)]
        self.assertEqual(r.latest_detections(), [])


class EvidenceRecordTests(unittest.TestCase):
    def _record(self, mode):
        with tempfile.TemporaryDirectory() as d:
            r = _runner(mode, sink=d)
            from twin.route_alerts.runner import RouteAlertsResult
            res = RouteAlertsResult(ran=True,
                                    detections=list(r._latest_detections))
            r._write(3.5, res)
            r.close()
            line = (Path(d) / "route-alerts.jsonl").read_text().strip().splitlines()[-1]
            return json.loads(line)

    def test_record_carries_mode_and_fed_to_nav(self):
        findings = self._record("findings")
        self.assertEqual(findings["mode"], "findings")
        self.assertEqual(len(findings["detections"]), 1)  # detection recorded for evidence
        self.assertEqual(findings["fed_to_nav"], [])       # ... but nav receives nothing

    def test_act_record_feeds_nav(self):
        act = self._record("act")
        self.assertEqual(act["mode"], "act")
        self.assertEqual(len(act["fed_to_nav"]), 1)
        self.assertEqual(act["fed_to_nav"][0]["source"], "route_obstruction")


class NavSourceTests(unittest.TestCase):
    class _Hook:
        def __init__(self, dets):
            self._dets = dets

        def latest_detections(self):
            return list(self._dets)

    def test_source_returns_hook_detections(self):
        d = Detection((1.0, 2.0), 0.4)
        src = source_from_hook(self._Hook([d]))
        self.assertEqual(src(0.0), [d])

    def test_source_defensive_on_raise(self):
        class _Boom:
            def latest_detections(self):
                raise RuntimeError("boom")

        self.assertEqual(source_from_hook(_Boom())(1.0), [])


class BundleActWiringTests(unittest.TestCase):
    """The isaac_bundle helper that wires act mode into the existing dynobs seam."""

    class _Nav:
        def __init__(self, *, sensor=True, layer=True, source=None):
            self.sensor_navigation = sensor
            self.detection_source = source
            self.controller = type("C", (), {"dynamic_layer": object() if layer else None})()

    class _Hook:
        def __init__(self, acts):
            self.acts = acts

        def latest_detections(self):
            return [Detection((2.0, 2.0), 0.5, source="route_obstruction")]

    def _wire(self, nav, hook):
        from twin.restock.isaac_bundle import _wire_route_alerts_act
        _wire_route_alerts_act(nav, hook)

    def test_findings_hook_leaves_nav_untouched(self):
        nav = self._Nav()
        self._wire(nav, self._Hook(acts=False))
        self.assertIsNone(nav.detection_source)

    def test_act_hook_wires_the_source(self):
        nav = self._Nav()
        self._wire(nav, self._Hook(acts=True))
        self.assertIsNotNone(nav.detection_source)
        fed = nav.detection_source(0.0)
        self.assertEqual(fed[0].source, "route_obstruction")

    def test_act_requires_sensor_nav(self):
        with self.assertRaises(ValueError):
            self._wire(self._Nav(sensor=False), self._Hook(acts=True))

    def test_act_refuses_to_share_an_existing_source(self):
        with self.assertRaises(ValueError):
            self._wire(self._Nav(source=lambda now: []), self._Hook(acts=True))

    def test_act_requires_the_dynobs_layer(self):
        with self.assertRaises(ValueError):
            self._wire(self._Nav(layer=False), self._Hook(acts=True))


if __name__ == "__main__":
    unittest.main()
