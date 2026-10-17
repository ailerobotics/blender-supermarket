"""Closed-loop CPU evidence fixtures for dynamic-obstacle avoidance.

Quasi-static kinematic demonstrations; NOT PhysX, sensing or perception runs.
They assert the behaviours, so the evidence CLI's verdict is regression-covered.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.mobile import MobileSpec
from twin.navigation.core import GridMap
from twin.navigation.articulated import ArticulatedMap, ArticulatedPlanner, TrainState
from twin.navigation.dynamic import DynamicObstacleConfig, DynamicObstacleLayer, Detection
from twin.navigation.dynamic_scenario import DynamicScenario, crossing_person, dropped_box
from twin.navigation.dynamic_evidence import simulate

CFG = DynamicObstacleConfig(confirm_frames=2, moving_speed=0.08,
                            prediction_horizon=1.2, prediction_step=0.3)


def _aisle():
    # A 2 m-wide aisle (walls at y = +-1) running along +x; no coupled detour fits.
    return GridMap((-3, -2, 12, 2), ((-3, -2, 12, -1.0), (-3, 1.0, 12, 2)))


class ClosedLoopTests(unittest.TestCase):
    def test_person_crossing_stops_then_resumes_and_arrives(self):
        scen = DynamicScenario('person', 'shopper crosses the aisle',
                               (crossing_person(lane_x=5.0, enter=16.0, cross_duration=6.0, dwell=6.0),))
        run = simulate(scen, grid=_aisle(), goal=(9.0, 0.0, 0.0), config=CFG,
                       dt=0.1, horizon=160.0, blocked_timeout=50.0)
        self.assertTrue(run.stopped_for_obstacle, 'robot should hold while the aisle is blocked')
        self.assertTrue(run.resumed_after_clear, 'robot should resume once the path clears')
        self.assertTrue(run.arrived, 'robot should reach the goal after the shopper passes')
        # It never drove into the shopper.
        self.assertGreater(run.min_clearance, 0.0)

    def test_dropped_box_permanent_block_gives_up_honestly(self):
        scen = DynamicScenario('box', 'box dropped on the route and left',
                               (dropped_box(at_x=5.0, at_y=0.0, drop=12.0),))
        run = simulate(scen, grid=_aisle(), goal=(9.0, 0.0, 0.0), config=CFG,
                       dt=0.1, horizon=120.0, blocked_timeout=15.0)
        self.assertTrue(run.stopped_for_obstacle)
        self.assertFalse(run.arrived, 'a permanently blocked aisle must not report arrival')
        self.assertEqual(run.ticks[-1].status, 'failed')
        self.assertIn('timeout', run.ticks[-1].reason)

    def test_determinism(self):
        scen = DynamicScenario('person', 'd',
                               (crossing_person(lane_x=5.0, enter=16.0, cross_duration=6.0, dwell=6.0),))
        a = simulate(scen, grid=_aisle(), goal=(9.0, 0.0, 0.0), config=CFG, horizon=80.0)
        b = simulate(scen, grid=_aisle(), goal=(9.0, 0.0, 0.0), config=CFG, horizon=80.0)
        self.assertEqual([vars(k) for k in a.ticks], [vars(k) for k in b.ticks])


class PlannerDetourTests(unittest.TestCase):
    """The coupled planner re-plans a route AROUND an off-centre obstacle cluster."""
    def test_detour_route_differs_from_straight(self):
        spec = MobileSpec()
        world = ArticulatedMap(GridMap((-4, -6, 14, 6)), spec)
        planner = ArticulatedPlanner(world, timeout=10)
        start, goal = TrainState(2.0, 0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
        straight = planner.plan(start, goal)
        self.assertTrue(straight.states)
        self.assertLess(max(abs(s.y) for s in straight.states), 0.05)  # straight shot
        layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=1))
        layer.update([Detection((5.0, 0.9), 0.5)], 0.0)  # cart rolled in, off-centre
        detour = planner.plan(start, goal, layer.emitted_points())
        self.assertTrue(detour.states, detour.reason)
        self.assertGreater(max(abs(s.y) for s in detour.states), 0.3)  # curved around it


if __name__ == '__main__':
    unittest.main()
