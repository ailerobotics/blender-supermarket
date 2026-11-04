"""Full-runner integration over store-vision synthetic aisle frames.

Gated on store-vision being importable (run under the store-vision venv):

  PYTHONPATH=scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
    /home/manishsaini/firstmate/projects/store-vision/.venv/bin/python \\
    -m pytest scripts/twin/route_alerts/tests/test_integration.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import numpy  # noqa: F401
    from storevision.obstruction.scenes import default_scenes
    _HAVE_SV = True
except Exception:
    _HAVE_SV = False

_TUNE = {"persist_seconds": 0.4, "warmup_frames": 2, "clear_seconds": 0.8,
         "min_object_height_m": 0.06}


@unittest.skipUnless(_HAVE_SV, "store-vision + numpy required")
class RouteAlertsIntegrationTests(unittest.TestCase):
    def _run(self, *, spill=True, obstruction=True, staff_tasks=True, mask_people=True,
             cadence=1):
        from twin.route_alerts import RouteAlertsRunner, RouteFrame

        scene = default_scenes()[0]  # carton_blocks_aisle1 + a moving shopper
        frames, agents_by_seq = scene.render()
        itf = scene.image_to_floor()
        aisles = {aid: dict(x0=s.x0, y0=s.y0, x1=s.x1, y1=s.y1, axis=s.axis,
                            min_width_m=s.min_width_m) for aid, s in scene.aisles.items()}
        cfg = {
            "spill": {"enabled": spill, "tuning": _TUNE},
            "obstruction": {"enabled": obstruction, "tuning": _TUNE},
            "floors": {scene.camera_id: {"image_to_floor": itf.tolist(), "zone_id": "aisle"}},
            "aisles": aisles, "staff_tasks": staff_tasks, "cadence": cadence,
            "obstruction_source": "route_obstruction",
        }
        runner = RouteAlertsRunner(cfg, robot_id="shelfbot")
        evs, robot_evs, tasks, dets, passability = [], [], [], [], {}
        ran_count = 0
        for i, fr in enumerate(frames):
            agents = agents_by_seq.get(i, []) if mask_people else []
            rf = RouteFrame(camera_id=fr.camera_id, seq=i, rgb=fr.rgb, depth=fr.depth,
                            capture_time=fr.capture_time, zone_id="aisle", agents=agents)
            r = runner.observe([rf], now=fr.capture_time)
            if r.ran:
                ran_count += 1
            evs += [e.type for e in r.events]
            robot_evs += [e.type for e in r.robot_events]
            tasks += r.tasks
            if r.detections:
                dets = r.detections
            if r.passability:
                passability = r.passability
        return runner, set(evs), robot_evs, tasks, dets, passability, ran_count

    def test_detects_obstruction_blocks_aisle_and_emits_dynobs_detection(self):
        runner, evs, robot_evs, _, dets, passability, _ = self._run(spill=False)
        self.assertIsNone(runner.disabled_reason)
        self.assertTrue(any(t.startswith("obstruction.") for t in evs))
        self.assertTrue(passability.get("blocked"))
        self.assertTrue(dets)  # dynobs Detection(s) for navigation
        self.assertTrue(all(d.source == "route_obstruction" for d in dets))

    def test_detects_spill(self):
        runner, evs, *_ = self._run(obstruction=False)
        self.assertIsNone(runner.disabled_reason)
        self.assertTrue(any(t.startswith("spill.") for t in evs))

    def test_robot_events_mirror_every_finding(self):
        _, evs, robot_evs, *_ = self._run()
        self.assertTrue(robot_evs)
        self.assertTrue(all(t.startswith("robot.route_alert.") for t in robot_evs))

    def test_staff_tasks_raised(self):
        _, _, _, tasks, *_ = self._run()
        self.assertTrue(tasks)

    def test_people_are_masked_out(self):
        # With agents fed, no detection lands on the shopper's path; without them, a
        # person is (wrongly) detected -> strictly more detections. Masking must not
        # increase the detection count.
        _, _, _, _, masked_dets, _, _ = self._run(spill=False, mask_people=True)
        _, _, _, _, unmasked_dets, _, _ = self._run(spill=False, mask_people=False)
        self.assertLessEqual(len(masked_dets), len(unmasked_dets))

    def test_cadence_skips_frames(self):
        _, _, _, _, _, _, ran = self._run(cadence=3)
        scene_frames = len(default_scenes()[0].render()[0])
        self.assertEqual(ran, scene_frames // 3)

    def test_disabled_detectors_are_byte_quiet(self):
        runner, evs, robot_evs, tasks, dets, passability, _ = self._run(
            spill=False, obstruction=False)
        self.assertEqual(evs, set())
        self.assertEqual(robot_evs, [])
        self.assertEqual(tasks, [])
        self.assertEqual(dets, [])


@unittest.skipUnless(_HAVE_SV, "store-vision + numpy required")
class RouteAlertsModeSplitTests(unittest.TestCase):
    """The two delivered modes over the same store-vision frames (findings vs act)."""

    def _run(self, mode):
        from twin.route_alerts import RouteAlertsRunner, RouteFrame

        scene = default_scenes()[0]
        frames, agents_by_seq = scene.render()
        itf = scene.image_to_floor()
        cfg = {
            "spill": {"enabled": False},
            "obstruction": {"enabled": True, "tuning": _TUNE},
            "floors": {scene.camera_id: {"image_to_floor": itf.tolist(), "zone_id": "aisle"}},
            "obstruction_source": "route_obstruction", "mode": mode,
        }
        runner = RouteAlertsRunner(cfg, robot_id="shelfbot")
        events, result_dets, fed = [], [], []
        for i, fr in enumerate(frames):
            rf = RouteFrame(camera_id=fr.camera_id, seq=i, rgb=fr.rgb, depth=fr.depth,
                            capture_time=fr.capture_time, zone_id="aisle",
                            agents=agents_by_seq.get(i, []))
            r = runner.observe([rf], now=fr.capture_time)
            events += [e.type for e in r.events]
            if r.detections:
                result_dets = r.detections
            fed = runner.latest_detections()
        return runner, events, result_dets, fed

    def test_findings_detects_but_feeds_nav_nothing(self):
        runner, events, result_dets, fed = self._run("findings")
        self.assertIsNone(runner.disabled_reason)
        self.assertTrue(any(t.startswith("obstruction.") for t in events))  # still a finding
        self.assertTrue(result_dets)        # the detection is computed (recorded for evidence)
        self.assertEqual(fed, [])           # ... but navigation receives nothing

    def test_act_feeds_the_obstruction_to_nav(self):
        runner, events, result_dets, fed = self._run("act")
        self.assertIsNone(runner.disabled_reason)
        self.assertTrue(result_dets)
        self.assertTrue(fed)                # act exposes the obstruction to the dynobs layer
        self.assertTrue(all(d.source == "route_obstruction" for d in fed))


if __name__ == "__main__":
    unittest.main()
