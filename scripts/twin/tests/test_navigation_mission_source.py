"""CPU contract tests for the mission-side moving-obstacle detection feed.

Prove, with no ROS / GPU / simulator / USD stage:

* NavigationBridge passes `detections=None` when no source is attached (byte-identical
  to a run with no moving-obstacle layer), feeds the source's detections through on the
  sensor route, and refuses a source on a non-sensor route.
* The lane-owned scenario source reuses the humans scripted person: it is None when the
  humans block is absent, and otherwise turns the actor's KNOWN pose into a deterministic
  map-frame Detection matching PeopleStage.world_positions.
* The bundle's _build_detection_source resolves the module:function builder, is None when
  the key is absent, and is rejected for a non-sensor nav mode.
* The humans-perception adapter converts a MissionSafety-shaped provider through
  to_nav_detections onto the same seam.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.navigation.dynamic import Detection
from twin.navigation.mission_source import (
    build_dynamic_obstacle_source, humans_detection_source, SCRIPTED_SOURCE)
from twin.restock.runtime import NavigationBridge
from twin.restock import isaac_bundle


CROSSER = {'person_id': 'crosser', 'role': 'shopper',
           'waypoints': [[3.0, -1.5], [3.0, 1.5]], 'speed_mps': 0.9, 'radius_m': 0.3}


def humans_config():
    return {'humans': {'enabled': True, 'people': [CROSSER]}}


class _RecordingController:
    """Stand-in nav controller that records the kwargs each step() receives."""
    def __init__(self):
        self.calls = []

    def start(self, goal, coupled, goal_articulation=None):
        pass

    def cancel(self):
        pass

    def step(self, dt, odometry, navigation, *, now, coupled, arm_transport,
             articulation, detections=None):
        self.calls.append(dict(now=now, detections=detections))
        return SimpleNamespace(status='running', linear=0.0, angular=0.0, reason='ok')


def _packet(now):
    return SimpleNamespace(odometry=object(), navigation=object(), simulation_time=now,
                           coupled=False, transport_ready=False, articulation=0.0)


class BridgeFeedTests(unittest.TestCase):
    def test_no_source_passes_detections_none(self):
        ctrl = _RecordingController()
        bridge = NavigationBridge(ctrl, sensor_navigation=True)
        bridge.step(0.1, _packet(1.0))
        self.assertIsNone(ctrl.calls[-1]['detections'])

    def test_source_detections_flow_to_controller(self):
        ctrl = _RecordingController()
        det = Detection(position=(3.0, 0.0), radius=0.3, source=SCRIPTED_SOURCE)
        bridge = NavigationBridge(ctrl, sensor_navigation=True,
                                  detection_source=lambda now: [det])
        bridge.step(0.1, _packet(2.0))
        self.assertEqual(ctrl.calls[-1]['detections'], [det])
        self.assertEqual(ctrl.calls[-1]['now'], 2.0)

    def test_source_called_with_sim_clock(self):
        seen = []
        ctrl = _RecordingController()
        bridge = NavigationBridge(ctrl, sensor_navigation=True,
                                  detection_source=lambda now: seen.append(now) or [])
        bridge.step(0.1, _packet(4.25))
        self.assertEqual(seen, [4.25])

    def test_source_rejected_on_non_sensor_route(self):
        with self.assertRaises(ValueError):
            NavigationBridge(_RecordingController(), nav2=True,
                             detection_source=lambda now: [])
        with self.assertRaises(ValueError):
            NavigationBridge(_RecordingController(),
                             detection_source=lambda now: [])


class ScenarioSourceTests(unittest.TestCase):
    def test_none_without_humans_block(self):
        src = build_dynamic_obstacle_source(assembly=None,
                                            plan=SimpleNamespace(poses={}), config={})
        self.assertIsNone(src)

    def test_none_when_humans_disabled(self):
        src = build_dynamic_obstacle_source(
            assembly=None, plan=SimpleNamespace(poses={}),
            config={'humans': {'enabled': False, 'people': [CROSSER]}})
        self.assertIsNone(src)

    def test_scripted_detection_matches_world_positions(self):
        # No stage => detections flow without a visible prim (world_positions is pure).
        src = build_dynamic_obstacle_source(
            assembly=SimpleNamespace(stage=None),
            plan=SimpleNamespace(poses={}), config=humans_config())
        self.assertIsNotNone(src)
        from twin.humans.config import HumansConfig
        from twin.humans.isaac_people import PeopleStage
        people = HumansConfig.from_config(humans_config()).people
        ref = PeopleStage(HumansConfig.from_config(humans_config()).with_people(people))
        for now in (0.0, 1.0, 2.0, 5.0):
            dets = src(now)
            self.assertEqual(len(dets), 1)
            d = dets[0]
            self.assertEqual(d.source, SCRIPTED_SOURCE)
            self.assertEqual(d.radius, 0.3)
            x, y, _z = ref.world_positions(now)['crosser']
            self.assertAlmostEqual(d.position[0], x)
            self.assertAlmostEqual(d.position[1], y)

    def test_deterministic_replay(self):
        src = build_dynamic_obstacle_source(
            assembly=SimpleNamespace(stage=None),
            plan=SimpleNamespace(poses={}), config=humans_config())
        a = [tuple(src(t)[0].position) for t in (0.0, 1.3, 2.7, 4.0)]
        b = [tuple(src(t)[0].position) for t in (0.0, 1.3, 2.7, 4.0)]
        self.assertEqual(a, b)


class BundleResolveTests(unittest.TestCase):
    def _cfg(self, **nav):
        base = {'navigation': {'environment': {}}}
        base['navigation'].update(nav)
        base.update(humans_config())
        return base

    def test_absent_key_is_none(self):
        src = isaac_bundle._build_detection_source(
            app=None, assembly=SimpleNamespace(stage=None),
            plan=SimpleNamespace(nav_mode='sensor', poses={}),
            config=self._cfg(), output='/tmp/x')
        self.assertIsNone(src)

    def test_resolves_module_function(self):
        src = isaac_bundle._build_detection_source(
            app=None, assembly=SimpleNamespace(stage=None),
            plan=SimpleNamespace(nav_mode='sensor', poses={}),
            config=self._cfg(dynamic_obstacle_source=
                             'twin.navigation.mission_source:build_dynamic_obstacle_source'),
            output='/tmp/x')
        self.assertTrue(callable(src))
        self.assertEqual(src(0.0)[0].source, SCRIPTED_SOURCE)

    def test_rejected_for_non_sensor_mode(self):
        with self.assertRaises(ValueError):
            isaac_bundle._build_detection_source(
                app=None, assembly=SimpleNamespace(stage=None),
                plan=SimpleNamespace(nav_mode='reference', poses={}),
                config=self._cfg(dynamic_obstacle_source='m:f'), output='/tmp/x')

    def test_bad_reference_rejected(self):
        with self.assertRaises(ValueError):
            isaac_bundle._build_detection_source(
                app=None, assembly=SimpleNamespace(stage=None),
                plan=SimpleNamespace(nav_mode='sensor', poses={}),
                config=self._cfg(dynamic_obstacle_source='not-a-ref'), output='/tmp/x')


class HumansAdapterTests(unittest.TestCase):
    def test_wraps_mission_safety_detections(self):
        from twin.humans.detection import Detection as HumanDetection

        class _Safety:
            def detections(self, now):
                return [HumanDetection(position=(1.0, 2.0), radius=0.4,
                                       source='human_ground_truth', frame='map')]

        src = humans_detection_source(_Safety())
        out = src(0.0)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], Detection)
        self.assertEqual(out[0].position, (1.0, 2.0))
        self.assertEqual(out[0].source, 'human_ground_truth')


if __name__ == '__main__':
    unittest.main()
