"""sim-collect-reposition-block-v1: planner reproduction of the boxc1a mid-collect reposition no-route.

Recorded (``~/datasets/twin-robot-in-store-boot-v1/mission/boxc1a``): the "drive->rollwagen" reposition
to place_approach_stage (-4.5, 16.0) failed with blocked_no_route_timeout. During the carry the loaded
base drifted ~0.28 m NORTH off its clean departure pose (-5.734, 16.003) into the LIVE backstock Support
(AABB x[-6.07,-5.39] y[16.86,17.34], scenario initial_staging; NOT in the nav static map). At the
drifted pose the footprint (r=0.75) overlaps the Support -> free_start=False, in_start~=15, route=None
(NAV_DIAG). This locks the planner-visible failure and proves the fixes:
  * HOLD (the shipped fix): the base stays on the clean departure pose, where a route exists even with
    the Support as a live obstacle (seg_free True) -- so keeping the base there restores the route.
  * trusted-static drive (the reposition already dispatches static_drive=True): empty live obstacles ->
    the Support is ignored -> a route exists even from the drifted pose.
No Isaac/ROS/GPU imports; the planner is pure.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.navigation import GridMap

# Backstock Support as the sensed (LIVE) return points the lidar sees on its south face + body, from the
# recorded scenario geometry (support_center_xy=[-5.73,17.1], dims 0.68x0.48 -> AABB below).
SUP = (-6.07, 16.86, -5.39, 17.34)
SUPPORT_LIVE = tuple(
    (SUP[0] + i * (SUP[2] - SUP[0]) / 24.0, y)
    for i in range(25) for y in (SUP[1], 17.00, 17.10)
)
GOAL = (-4.5, 16.0)                 # place_approach_stage (the reposition goal)
RADIUS = 0.75                        # base footprint radius (circumscribes the train)
CLEAN = (-5.734, 16.003)             # the grasp/departure park pose (nav-trace, pre-drift)
DRIFTED = (-5.72, 16.17)             # NAV_DIAG free_start=False pose (loaded carry drifted north)


class RepositionBlockPlannerTests(unittest.TestCase):
    def setUp(self):
        # Empty static map over a backroom-sized rectangle: the Support is a LIVE obstacle, not static
        # (environment.json static_obstacles has nothing within 0.8 m of these poses).
        self.grid = GridMap((-9, 11, 3, 20), resolution=0.20)

    def test_drifted_pose_reproduces_free_start_false_and_no_route(self):
        # The recorded failure: with the Support live, the drifted start is boxed in -> no route.
        self.assertFalse(self.grid.free(DRIFTED, RADIUS, SUPPORT_LIVE))    # free_start=False (recorded)
        self.assertEqual(self.grid.plan(DRIFTED, GOAL, RADIUS, SUPPORT_LIVE), [])  # blocked_no_route

    def test_clean_departure_pose_has_a_route(self):
        # The HOLD fix keeps the base on this pose; a route exists here even with the Support LIVE, so
        # not drifting off it is what resolves the stage.
        self.assertTrue(self.grid.free(CLEAN, RADIUS, SUPPORT_LIVE))       # free_start=True
        self.assertTrue(self.grid.plan(CLEAN, GOAL, RADIUS, SUPPORT_LIVE)) # route found

    def test_trusted_static_drive_routes_from_either_pose(self):
        # The reposition dispatches static_drive=True -> the planner uses EMPTY live obstacles, so the
        # live Support is ignored and a route exists from the clean AND the drifted pose.
        self.assertTrue(self.grid.plan(CLEAN, GOAL, RADIUS, ()))
        self.assertTrue(self.grid.plan(DRIFTED, GOAL, RADIUS, ()))

    def test_goal_is_clear(self):
        # The failure is the START, not the goal: the rollwagen-mouth goal clears the Support either way.
        self.assertTrue(self.grid.free(GOAL, RADIUS, SUPPORT_LIVE))


if __name__ == '__main__':
    unittest.main()
