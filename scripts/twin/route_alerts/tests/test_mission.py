"""Opt-in mission-hook wiring tests (pure; fakes stand in for runner/port/packet)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.route_alerts.mission import (RouteAlertsMissionHook, _DisabledHook,
                                       build_from_config, recording_camera_frame_builder)


class _FakeRunner:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.disabled_reason = None
        self._dets = ["det"]

    def observe(self, frames, now, *, robot_pose=None):
        self.calls.append((len(frames), now, robot_pose))
        return {"ran": True}

    def latest_detections(self):
        return list(self._dets)

    def close(self):
        self.closed = True


class _FakePacket:
    simulation_time = 4.0
    pose = (1.0, 2.0, 0.5)


class BuildFromConfigTests(unittest.TestCase):
    def test_absent_section_returns_none(self):
        self.assertIsNone(build_from_config({}))
        self.assertIsNone(build_from_config({"route_alerts": None}))
        self.assertIsNone(build_from_config(None))


class MissionHookTests(unittest.TestCase):
    def test_disabled_without_frame_builder(self):
        hook = RouteAlertsMissionHook(_FakeRunner(), frame_builder=None)
        self.assertIsNotNone(hook.disabled_reason)
        self.assertIsNone(hook.on_navigate(object(), _FakePacket(), 1.0))

    def test_on_navigate_runs_and_passes_pose(self):
        runner = _FakeRunner()
        hook = RouteAlertsMissionHook(runner, frame_builder=lambda port, pkt: "frame")
        res = hook.on_navigate(object(), _FakePacket(), 4.0)
        self.assertEqual(res, {"ran": True})
        self.assertEqual(runner.calls, [(1, 4.0, (1.0, 2.0, 0.5))])

    def test_frame_builder_none_skips(self):
        runner = _FakeRunner()
        hook = RouteAlertsMissionHook(runner, frame_builder=lambda port, pkt: None)
        self.assertIsNone(hook.on_navigate(object(), _FakePacket(), 1.0))
        self.assertEqual(runner.calls, [])

    def test_self_disables_on_exception_never_raises(self):
        runner = _FakeRunner()

        def boom(port, pkt):
            raise RuntimeError("camera exploded")

        hook = RouteAlertsMissionHook(runner, frame_builder=boom)
        self.assertIsNone(hook.on_navigate(object(), _FakePacket(), 1.0))
        self.assertIn("camera exploded", hook.disabled_reason)
        # once disabled, later ticks are quiet
        self.assertIsNone(hook.on_navigate(object(), _FakePacket(), 2.0))

    def test_latest_detections_passthrough_and_close(self):
        runner = _FakeRunner()
        hook = RouteAlertsMissionHook(runner, frame_builder=lambda p, k: "f")
        self.assertEqual(hook.latest_detections(), ["det"])
        hook.close()
        self.assertTrue(runner.closed)

    def test_disabled_hook_is_inert(self):
        h = _DisabledHook("why")
        self.assertEqual(h.disabled_reason, "why")
        self.assertIsNone(h.on_navigate(None, None, 0))
        self.assertEqual(h.latest_detections(), [])
        h.close()


class FrameBuilderGdprTests(unittest.TestCase):
    def test_no_agents_provider_refuses_to_analyse(self):
        build = recording_camera_frame_builder({"route_alerts": {"camera": "route"}},
                                               agents_provider=None)
        # No people-mask source => every tick returns None (never analyse people).
        self.assertIsNone(build(object(), _FakePacket()))


class AdapterWiringTests(unittest.TestCase):
    def test_adapter_default_route_alerts_is_none(self):
        from twin.restock.runtime import RuntimeAdapter
        import inspect
        sig = inspect.signature(RuntimeAdapter.__init__)
        self.assertIn("route_alerts", sig.parameters)
        self.assertIsNone(sig.parameters["route_alerts"].default)


if __name__ == "__main__":
    unittest.main()
