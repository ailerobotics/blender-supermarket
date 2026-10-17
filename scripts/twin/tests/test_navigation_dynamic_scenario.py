"""Deterministic CPU fixtures for the scripted dynamic-obstacle scenario.

Ground-truth scenario actors, not a detector or a physics run.
"""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.navigation.dynamic import Detection
from twin.navigation.dynamic_scenario import (Keyframe, ScriptedActor, DynamicScenario,
    scenario_detections, crossing_person, rolled_cart, dropped_box)


class ActorValidationTests(unittest.TestCase):
    def test_keyframe_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            Keyframe(float('inf'), 0, 0)

    def test_actor_rejects_bad_shapes(self):
        k = (Keyframe(0, 0, 0), Keyframe(1, 1, 0))
        for bad in (dict(name=''), dict(radius=0), dict(radius=-1),
                    dict(keyframes=()), dict(keyframes=(Keyframe(1, 0, 0), Keyframe(0, 0, 0))),
                    dict(keyframes=(Keyframe(0, 0, 0), Keyframe(0, 1, 0))),  # duplicate t
                    dict(appear=5.0, disappear=1.0), dict(source='')):
            kw = dict(name='a', radius=0.3, keyframes=k)
            kw.update(bad)
            with self.assertRaises(ValueError):
                ScriptedActor(**kw)


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.actor = ScriptedActor('a', 0.3, (Keyframe(1.0, 0.0, 0.0), Keyframe(3.0, 2.0, -2.0)),
                                   appear=1.0, disappear=3.0)

    def test_clamps_before_and_after(self):
        self.assertEqual(self.actor.position_at(0.0), (0.0, 0.0))
        self.assertEqual(self.actor.position_at(9.0), (2.0, -2.0))

    def test_linear_interpolation_midpoint(self):
        x, y = self.actor.position_at(2.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, -1.0)

    def test_detection_window(self):
        self.assertIsNone(self.actor.detection_at(0.5))
        self.assertIsNone(self.actor.detection_at(3.5))
        d = self.actor.detection_at(2.0)
        self.assertIsInstance(d, Detection)
        self.assertEqual(d.radius, 0.3)
        self.assertEqual(d.source, 'scenario_ground_truth')


class CannedActorTests(unittest.TestCase):
    def test_crossing_person_reaches_midlane_then_leaves(self):
        p = crossing_person(lane_x=4.0, enter=2.0, cross_duration=6.0, dwell=1.5)
        self.assertIsNone(p.detection_at(1.0))               # not yet entered
        mid = p.detection_at(5.0)                            # mid cross -> in lane
        self.assertIsNotNone(mid)
        self.assertAlmostEqual(mid.position[0], 4.0)
        self.assertAlmostEqual(mid.position[1], 0.0, places=6)
        self.assertIsNone(p.detection_at(20.0))              # has left

    def test_rolled_cart_comes_to_rest_and_stays(self):
        c = rolled_cart(lane_x=4.0, rest_y=0.6, enter=1.0, roll_duration=3.0)
        self.assertIsNone(c.detection_at(0.5))
        rest = c.detection_at(10.0)
        self.assertIsNotNone(rest)                           # still present (no disappear)
        self.assertAlmostEqual(rest.position[1], 0.6)

    def test_dropped_box_appears_at_drop(self):
        b = dropped_box(at_x=4.0, at_y=0.0, drop=3.0)
        self.assertIsNone(b.detection_at(2.99))
        d = b.detection_at(5.0)
        self.assertIsNotNone(d)
        self.assertEqual(d.position, (4.0, 0.0))


class ScenarioTests(unittest.TestCase):
    def test_aggregates_active_actors(self):
        s = DynamicScenario('mix', 'person + box',
                            (crossing_person(enter=2.0), dropped_box(drop=3.0)))
        self.assertEqual(len(s.detections_at(0.0)), 0)       # none active
        self.assertEqual(len(s.detections_at(5.0)), 2)       # both active

    def test_duplicate_actor_names_rejected(self):
        with self.assertRaises(ValueError):
            DynamicScenario('x', 'dup', (dropped_box(name='b'), dropped_box(name='b')))

    def test_scenario_detections_determinism(self):
        actors = (crossing_person(), rolled_cart(), dropped_box())
        ts = [i * 0.3 for i in range(40)]
        self.assertEqual([scenario_detections(actors, t) for t in ts],
                         [scenario_detections(actors, t) for t in ts])


if __name__ == '__main__':
    unittest.main()
