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


class LaneStaticProudToleranceTests(unittest.TestCase):
    """fm423(a) / run-321b outbound lane_blocked: discount lidar-proud returns of KNOWN static
    geometry (a few cm proud of a shelf/support the lane hugs) when re-validating a tow lane, so
    they do not false-block the authored centreline -- while a genuinely NEW obstacle on the lane
    still blocks it and the live braking gate is untouched. Geometry: the straight lane's swept
    envelope half-width is ~0.77 m (blocks a return at |y|<=0.76, clears at |y|>=0.78)."""

    def setUp(self):
        self.spec = MobileSpec()
        # A static box whose SOUTH edge (y=0.80) just clears the lane's ~0.77 m envelope, so the
        # lane expands + validates against the true static map exactly as before.
        self.box = (-0.5, 0.80, 0.5, 2.0)
        self.grid = GridMap((-8, -8, 8, 8), (self.box,), 0.20)
        self.lane = _straight_lane()
        self.graph = TowLaneGraph([self.lane]).expand_all(ArticulatedMap(self.grid, self.spec))
        self.state = TrainState(-4.0, 0.0, 0.0, 0.0)
        # A proud return of that box: at y=0.74 it INTRUDES the envelope (blocks) but is only 6 cm
        # (<= 8 cm tol) from the box's south edge, so it is a known-static proud return.
        self.proud = (0.0, 0.74)
        # A genuinely new obstacle on the lane centre, 0.80 m from the box (NOT near static).
        self.newobs = (0.0, 0.0)

    def _controller(self, lane_static_tol):
        c = ArticulatedController(self.grid, self.spec, allow_test_fixtures=True,
                                  lane_graph=self.graph, lane_static_tol=lane_static_tol)
        c.start((4, 0, 0), True)
        return c

    def _obs(self, points):
        return Observation(0.1, tuple(points), True, True, 0.0, 'test_fixture', 10.0)

    def test_near_static_classifies_proud_vs_new(self):
        world = ArticulatedMap(self.grid, self.spec)
        self.assertTrue(world.near_static(*self.proud, 0.08))    # 6 cm proud of the box edge
        self.assertFalse(world.near_static(*self.proud, 0.04))   # tighter tol no longer matches
        self.assertFalse(world.near_static(*self.newobs, 0.08))  # 0.80 m away: a new obstacle

    def test_default_off_proud_return_blocks_byte_identical(self):
        c = self._controller(lane_static_tol=0.0)               # default: full live cloud
        self.assertEqual(c.plan_observed_route(self.state, self._obs([self.proud]), 0.05).reason,
                         'lane_blocked')

    def test_proud_static_return_discounted_reopens_lane(self):
        c = self._controller(lane_static_tol=0.08)
        self.assertEqual(c.plan_observed_route(self.state, self._obs([self.proud]), 0.05).reason,
                         'validated_lane')

    def test_new_obstacle_on_lane_still_blocks_with_tolerance(self):
        # SAFETY: a new obstacle not near any static geometry is NEVER discounted, so the tow still
        # stops at plan time even with the proud tolerance enabled (plus the live braking gate).
        c = self._controller(lane_static_tol=0.08)
        self.assertEqual(
            c.plan_observed_route(self.state, self._obs([self.proud, self.newobs]), 0.05).reason,
            'lane_blocked')

    def test_lane_blocked_diag_names_the_vseed_reason(self):
        # The opaque bare 'lane_blocked' (run-321b burned an hour on it) now logs the specific
        # validate_seed rejection + cloud sizes.
        import contextlib, io
        c = self._controller(lane_static_tol=0.0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            c.plan_observed_route(self.state, self._obs([self.proud]), 0.05)
        out = buf.getvalue()
        self.assertIn('LANE_BLOCKED_DIAG', out)
        self.assertIn('vseed_reason=', out)

    def test_lane_static_tol_validation(self):
        with self.assertRaises(ValueError):
            ArticulatedController(self.grid, self.spec, allow_test_fixtures=True, lane_static_tol=1.0)
        with self.assertRaises(ValueError):
            ArticulatedController(self.grid, self.spec, allow_test_fixtures=True, lane_static_tol=-0.1)


class LaneTrustStaticTests(unittest.TestCase):
    """fm425(a) / run-322b: the coupled OUTBOUND tow stayed lane_blocked with tow_lane_static_tol_m=0.12
    ACTIVE because 128 live returns sat > 0.12 m from any mapped AABB -- with no dynamic-obstacle source
    every return is static, so they are proud-beyond-tol OR off finer geometry than the AABB map carries
    (unmapped static), which near_static can never discount. tow_lane_trust_static re-validates the authored
    lane against the static map ALONE (what it was build-validated against) and reopens it, WITHOUT setting
    self.static_drive -- so the live braking gate still stops the tow for a genuinely new obstacle."""

    def setUp(self):
        self.spec = MobileSpec()
        self.box = (-0.5, 0.80, 0.5, 2.0)
        self.grid = GridMap((-8, -8, 8, 8), (self.box,), 0.20)
        self.lane = _straight_lane()
        self.graph = TowLaneGraph([self.lane]).expand_all(ArticulatedMap(self.grid, self.spec))
        self.state = TrainState(-4.0, 0.0, 0.0, 0.0)
        # An UNMAPPED static return on the lane centre, 0.80 m from the only box (NOT near static at ANY
        # tol < 1 m) -- the class the proud tolerance cannot reach.
        self.unmapped = (0.0, 0.0)

    def _controller(self, **kw):
        c = ArticulatedController(self.grid, self.spec, allow_test_fixtures=True,
                                  lane_graph=self.graph, **kw)
        c.start((4, 0, 0), True)
        return c

    def _obs(self, points):
        return Observation(0.1, tuple(points), True, True, 0.0, 'test_fixture', 10.0)

    def test_nearest_static_measures_distance(self):
        world = ArticulatedMap(self.grid, self.spec)
        self.assertAlmostEqual(world.nearest_static(0.0, 0.74), 0.06, places=2)   # 6 cm proud of the box
        self.assertGreater(world.nearest_static(*self.unmapped), 0.12)            # genuinely far from static

    def test_tol_cannot_discount_unmapped_static_return(self):
        # Even a generous tol keeps an unmapped-static return -> lane_blocked (reproduces run-322b).
        c = self._controller(lane_static_tol=0.12)
        self.assertEqual(c.plan_observed_route(self.state, self._obs([self.unmapped]), 0.05).reason,
                         'lane_blocked')

    def test_trust_static_reopens_lane_against_static_map(self):
        c = self._controller(lane_trust_static=True)
        self.assertEqual(c.plan_observed_route(self.state, self._obs([self.unmapped]), 0.05).reason,
                         'validated_lane')

    def test_trust_static_new_obstacle_still_stops_via_brake_gate(self):
        # SAFETY: trust_static trusts the static map for the PLAN only; the live braking gate keeps the
        # full live cloud, so a new obstacle directly ahead still stops the tow (obstacle_stop).
        c = self._controller(lane_trust_static=True)
        pose = (self.state.x, self.state.y, self.state.yaw)
        ahead = (pose[0] + 0.5, pose[1])     # 0.5 m ahead along +x heading
        od = Odometry(pose, 0.1, 0.2, 0.0, 'test_fixture', 0.05)
        clear = c.gate_velocity(0.1, od, self._obs([]), 0.2, 0.0)
        blocked = c.gate_velocity(0.1, od, self._obs([ahead]), 0.2, 0.0)
        self.assertEqual(clear.reason, 'tracking')
        self.assertEqual(blocked.reason, 'obstacle_stop')

    def test_trust_static_does_not_enable_static_drive(self):
        # trust_static is scoped to lane plan-validation; it must NEVER flip the base static_drive (which
        # would zero the braking gate's live cloud). Byte-identical braking behaviour.
        c = self._controller(lane_trust_static=True)
        self.assertFalse(c.static_drive)

    def test_default_off_is_byte_identical(self):
        c = self._controller()   # no trust_static, no tol
        self.assertFalse(c.lane_trust_static)
        self.assertEqual(c.plan_observed_route(self.state, self._obs([self.unmapped]), 0.05).reason,
                         'lane_blocked')


class LaneReacquireAlongRouteTests(unittest.TestCase):
    """fm427 (B1) / run-323b: the follower drove ~0.36 m along the validated lane, dropped the route on a
    tracking transient, then could NOT re-acquire -- `acquirable` checked only the fixed START (0.36 m >
    start_tol_xy) though the train was 0.06 m from the route. tow_lane_reacquire_along_route re-acquires at
    the train's progress (nearest route point, SAME tolerances)."""

    def setUp(self):
        self.spec = MobileSpec()
        self.grid = GridMap((-8, -8, 8, 8), (), 0.20)
        self.lane = _straight_lane()
        self.graph = TowLaneGraph([self.lane]).expand_all(ArticulatedMap(self.grid, self.spec))
        # A pose well ALONG the straight lane (start is x=-4; drive to x=+1), on the centreline (y=0),
        # 5 m from the start (>> start_tol_xy 0.25) but ~0 m from the route.
        self.advanced = (1.0, 0.0, 0.0)

    def _obs(self):
        return Observation(0.1, (), True, True, 0.0, 'test_fixture', 10.0)

    def test_acquirable_start_only_rejects_advanced_pose(self):
        self.assertFalse(self.lane.acquirable(self.advanced, 0.0, along_route=False))

    def test_acquirable_along_route_accepts_advanced_pose(self):
        self.assertTrue(self.lane.acquirable(self.advanced, 0.0, along_route=True))

    def test_along_route_still_rejects_off_route_pose(self):
        # a pose far from EVERY route point (off the lane laterally) is never acquirable, even along_route.
        self.assertFalse(self.lane.acquirable((1.0, 3.0, 0.0), 0.0, along_route=True))

    def test_controller_default_traps_drifted_follower(self):
        c = ArticulatedController(self.grid, self.spec, allow_test_fixtures=True, lane_graph=self.graph)
        self.assertFalse(c.lane_reacquire_along_route)
        c.start((4, 0, 0), True)
        state = TrainState(*self.advanced, 0.0)
        self.assertEqual(c.plan_observed_route(state, self._obs(), 0.05).reason, 'lane_not_acquired')

    def test_controller_reacquire_recovers_drifted_follower(self):
        c = ArticulatedController(self.grid, self.spec, allow_test_fixtures=True, lane_graph=self.graph,
                                  lane_reacquire_along_route=True)
        c.start((4, 0, 0), True)
        state = TrainState(*self.advanced, 0.0)
        self.assertEqual(c.plan_observed_route(state, self._obs(), 0.05).reason, 'validated_lane')


if __name__ == '__main__':
    unittest.main()
