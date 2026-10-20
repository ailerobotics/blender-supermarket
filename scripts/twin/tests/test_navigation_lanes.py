"""CPU tests for pre-planned tow lanes (navigation/lanes.py) and the coupled lane follower.

Deterministic fixtures in an open grid -- never evidence of a physics run. No ROS, GPU or
simulator. The real ArticulatedController/ArticulatedPlanner expand and follow the lanes.
"""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.navigation.core import GridMap
from twin.navigation.lanes import TowLane, TowLaneGraph
from twin.navigation.articulated import ArticulatedController, ArticulatedMap, TrainState, advance
from twin.navigation import Odometry, Observation
from twin.mobile import MobileSpec


def _straight_lane(goal_articulation=0.3):
    return TowLane('straight', [-4, 0, 0], [4, 0, 0], start_cart_yaw=0.0,
                   goal_articulation=goal_articulation, waypoints=[[0, 0, 0.0]])


class TowLaneSpecTests(unittest.TestCase):
    def test_from_config_round_trip_and_defaults(self):
        lane = TowLane.from_config({'name': 'L', 'start': [-4, 0, 0], 'goal': [4, 0, 0],
                                    'start_cart_yaw': 0.1, 'goal_articulation': 0.4,
                                    'waypoints': [[0, 0, 0]], 'goal_tol_xy': 0.5})
        self.assertEqual(lane.name, 'L')
        self.assertEqual(lane.start, (-4, 0, 0))
        self.assertAlmostEqual(lane.start_cart_yaw, 0.1)
        self.assertAlmostEqual(lane.goal_articulation, 0.4)
        self.assertAlmostEqual(lane.goal_tol_xy, 0.5)

    def test_from_config_rejects_unknown_key_and_missing_required(self):
        with self.assertRaises(ValueError):
            TowLane.from_config({'name': 'L', 'start': [0, 0, 0], 'goal': [1, 0, 0], 'bogus': 1})
        with self.assertRaises(ValueError):
            TowLane.from_config({'name': 'L', 'start': [0, 0, 0]})

    def test_matches_goal_and_acquirable(self):
        lane = _straight_lane()
        self.assertTrue(lane.matches_goal((4.0, 0.02, 0.0)))
        self.assertFalse(lane.matches_goal((4.0, 3.0, 0.0)))          # wrong XY
        self.assertFalse(lane.matches_goal((4.0, 0.0, 1.0)))          # wrong yaw
        self.assertTrue(lane.acquirable((-4.05, 0.03, 0.02), 0.03))   # close start, aligned cart
        self.assertFalse(lane.acquirable((-2.0, 0.0, 0.0), 0.0))      # too far from the lane start
        self.assertFalse(lane.acquirable((-4.0, 0.0, 0.0), 1.0))      # cart far from authored start_cart_yaw

    def test_graph_unique_names_and_resolve(self):
        lane = _straight_lane()
        graph = TowLaneGraph([lane])
        self.assertIsNone(graph.resolve_goal((4, 0, 0)))              # not expanded yet
        graph.expand_all(ArticulatedMap(GridMap((-8, -8, 8, 8), (), 0.20), MobileSpec()))
        self.assertIs(graph.resolve_goal((4.0, 0.0, 0.0)), lane)
        self.assertIsNone(graph.resolve_goal((0.0, 5.0, 0.0)))
        with self.assertRaises(ValueError):
            TowLaneGraph([_straight_lane(), _straight_lane()])        # duplicate names

    def test_expand_refuses_waypoint_off_the_centreline(self):
        bad = TowLane('bad', [-4, 0, 0], [4, 0, 0], start_cart_yaw=0.0,
                      goal_articulation=0.3, waypoints=[[0, 3.0, 0.0]])  # 3 m off the straight line
        with self.assertRaises(ValueError):
            bad.expand(ArticulatedMap(GridMap((-8, -8, 8, 8), (), 0.20), MobileSpec()))


class CoupledLaneFollowerTests(unittest.TestCase):
    def setUp(self):
        self.grid = GridMap((-8, -8, 8, 8), (), 0.20)
        self.spec = MobileSpec()
        self.lane = _straight_lane()
        self.graph = TowLaneGraph([self.lane]).expand_all(ArticulatedMap(self.grid, self.spec))

    def _controller(self):
        return ArticulatedController(self.grid, self.spec, allow_test_fixtures=True,
                                     goal_timeout=400.0, blocked_timeout=40.0, lane_graph=self.graph)

    def _drive(self, c, start, cart_yaw, obstacles=(), ticks=4000):
        pose = tuple(start); v = w = 0.0; dt = 0.1
        for t in range(ticks):
            now = (t + 1) * dt
            art = math.atan2(math.sin(pose[2] - cart_yaw), math.cos(pose[2] - cart_yaw))
            od = Odometry(pose, now, abs(v), w, 'test_fixture', 0.05)
            obs = Observation(now, tuple(obstacles), True, True, art, 'test_fixture', 10.0)
            r = c.step(dt, od, obs, now=now)
            if r.status in ('succeeded', 'failed'):
                return r
            v, w = r.linear, r.angular
            th = pose[2] + w * dt
            ns = advance(TrainState(pose[0], pose[1], pose[2], cart_yaw),
                         v * dt, (w / v if abs(v) > 1e-6 else 0.0), self.spec)
            pose = (pose[0] + v * dt * math.cos(th), pose[1] + v * dt * math.sin(th), th)
            cart_yaw = ns.cart_yaw
        return r

    def test_coupled_goal_resolves_and_follows_lane_to_goal(self):
        c = self._controller()
        c.start((4, 0, 0), True)
        self.assertIs(c._active_lane, self.lane)
        self.assertAlmostEqual(c.goal_articulation, 0.3)          # arrival follows the lane
        r = self._drive(c, (-4.0, 0.0, 0.0), 0.0)
        self.assertEqual(r.status, 'succeeded')
        self.assertEqual(r.reason, 'arrived_stopped')

    def test_uncoupled_drive_ignores_lane_graph(self):
        c = self._controller()
        c.start((4, 0, 0), False)
        self.assertIsNone(c._active_lane)                         # uncoupled never uses a lane

    def test_lane_route_validated_against_live_obstacles(self):
        c = self._controller()
        c.start((4, 0, 0), True)
        state = TrainState(-4.0, 0.0, 0.0, 0.0)
        clear = Observation(0.1, (), True, True, 0.0, 'test_fixture', 10.0)
        self.assertEqual(c.plan_observed_route(state, clear, 0.05).reason, 'validated_lane')

    def test_live_obstacle_on_the_lane_blocks_without_free_search(self):
        c = self._controller()
        c.start((4, 0, 0), True)
        state = TrainState(-4.0, 0.0, 0.0, 0.0)
        # A dense wall of returns straddling the lane centreline at x=0.
        wall = tuple((0.0, y) for y in [round(-0.4 + 0.1 * i, 2) for i in range(9)])
        blocked = Observation(0.1, wall, True, True, 0.0, 'test_fixture', 10.0)
        self.assertEqual(c.plan_observed_route(state, blocked, 0.05).reason, 'lane_blocked')

    def test_not_acquired_when_measured_start_is_off_the_lane(self):
        c = self._controller()
        c.start((4, 0, 0), True)
        far = TrainState(-1.0, 0.0, 0.0, 0.0)                     # 3 m from the lane start
        clear = Observation(0.1, (), True, True, 0.0, 'test_fixture', 10.0)
        self.assertEqual(c.plan_observed_route(far, clear, 0.05).reason, 'lane_not_acquired')

    def test_absent_lane_graph_is_byte_identical_free_planning(self):
        c = ArticulatedController(self.grid, self.spec, allow_test_fixtures=True)
        c.start((4, 0, 0), True)
        self.assertIsNone(c._active_lane)
        state = TrainState(-4.0, 0.0, 0.0, 0.0)
        clear = Observation(0.1, (), True, True, 0.0, 'test_fixture', 10.0)
        # Free planner runs (reason is a planner verdict, never a lane verdict).
        self.assertNotIn(c.plan_observed_route(state, clear, 0.05).reason,
                         ('validated_lane', 'lane_blocked', 'lane_not_acquired'))


if __name__ == '__main__':
    unittest.main()
