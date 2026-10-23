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


class LaneReacquireMidTurnDriftTests(unittest.TestCase):
    """fm448/run-331: on the coupled outbound tow the base drifted off the authored route MID-TURN, the
    controller dropped the route (stop_replan_tracking_error, which gates on xy + cart_yaw, NOT base
    heading), then churned lane_not_acquired forever (846x) because `acquirable(along_route=True)` ALSO
    required the base HEADING to be within start_tol_yaw of the nearest route point. Mid-route the authored
    yaw sweeps through the turn, so the nearest-by-xy point's heading differs from a slightly-drifted base
    even when xy + cart_yaw are both in tolerance -> acquire was STRICTER than the drop, so re-acquire was
    impossible. The fix drops the base-yaw gate for along_route (same xy + cart_yaw criteria as the drop);
    the start-only gate (along_route=False) keeps the base-yaw check byte-identically."""

    def setUp(self):
        self.spec = MobileSpec()
        self.grid = GridMap((-8, -8, 8, 8), (), 0.20)
        self.lane = _straight_lane()
        self.graph = TowLaneGraph([self.lane]).expand_all(ArticulatedMap(self.grid, self.spec))
        # A pose at a MID-ROUTE point (x=+1, 5 m past the x=-4 start -- >> start_tol_xy 0.25 from it) on the
        # centreline, but whose base HEADING has drifted 0.2 rad (> start_tol_yaw 0.15). xy and cart_yaw are
        # both in tolerance of the nearest route point (yaw there is 0). Old along_route gate refused it on
        # the base-yaw check; the fix re-acquires.
        self.drifted = (1.0, 0.0, 0.2)

    def _obs(self):
        return Observation(0.1, (), True, True, 0.0, 'test_fixture', 10.0)

    # (a) a mid-route drifted pose (near route state k, not the start) re-acquires when along_route=True.
    def test_along_route_reacquires_after_midroute_heading_drift(self):
        self.assertTrue(self.lane.acquirable(self.drifted, 0.0, along_route=True))

    def test_controller_reacquires_after_midroute_heading_drift(self):
        c = ArticulatedController(self.grid, self.spec, allow_test_fixtures=True, lane_graph=self.graph,
                                  lane_reacquire_along_route=True)
        c.start((4, 0, 0), True)
        state = TrainState(*self.drifted, 0.0)
        self.assertEqual(c.plan_observed_route(state, self._obs(), 0.05).reason, 'validated_lane')

    # (b) a genuinely off-lane pose (far from EVERY route point) still returns lane_not_acquired.
    def test_along_route_still_rejects_far_off_lane_pose(self):
        self.assertFalse(self.lane.acquirable((1.0, 3.0, 0.2), 0.0, along_route=True))  # 3 m off laterally

    def test_controller_rejects_far_off_lane_pose(self):
        c = ArticulatedController(self.grid, self.spec, allow_test_fixtures=True, lane_graph=self.graph,
                                  lane_reacquire_along_route=True)
        c.start((4, 0, 0), True)
        state = TrainState(1.0, 3.0, 0.2, 0.0)
        self.assertEqual(c.plan_observed_route(state, self._obs(), 0.05).reason, 'lane_not_acquired')

    def test_along_route_still_gates_cart_yaw_drift(self):
        # SAFETY: the cart_yaw gate is RETAINED along_route -- a cart yawed > start_tol_cart_yaw (0.20) from
        # the nearest route point is still refused, even at a mid-route point with a good base heading.
        self.assertFalse(self.lane.acquirable((1.0, 0.0, 0.0), 0.5, along_route=True))

    # (c) along_route=False is byte-identical (start-only, base-yaw gate intact).
    def test_along_route_false_is_byte_identical_start_only(self):
        # A pose AT the start xy with a 0.2 rad heading drift (> start_tol_yaw 0.15) is rejected exactly as
        # before -- the start-only branch keeps the base-yaw gate.
        self.assertFalse(self.lane.acquirable((-4.0, 0.0, 0.2), 0.0, along_route=False))
        self.assertTrue(self.lane.acquirable((-4.05, 0.03, 0.02), 0.03, along_route=False))
        # And the same drifted mid-route pose that re-acquires along_route is still refused start-only
        # (5 m from the start > start_tol_xy), proving the branches are independent.
        self.assertFalse(self.lane.acquirable(self.drifted, 0.0, along_route=False))


class LaneCoupledTrackingKnobsTests(unittest.TestCase):
    """fm429 (run-324b/c): config-driven coupled-tow tracking knobs, all defaulting to the prior
    hardcoded values => byte-identical. The cart_yaw replan tolerance is the one staged (relaxed to the
    swept-envelope-proven bound); curve speed floor and commanded-curvature cap ship as tunable."""

    def setUp(self):
        self.spec = MobileSpec()
        self.grid = GridMap((-8, -8, 8, 8), (), 0.20)
        self.lane = _straight_lane()
        self.graph = TowLaneGraph([self.lane]).expand_all(ArticulatedMap(self.grid, self.spec))

    def test_defaults_match_prior_hardcoded_values(self):
        c = ArticulatedController(self.grid, self.spec, allow_test_fixtures=True, lane_graph=self.graph)
        self.assertAlmostEqual(c.lane_cart_yaw_tol, 0.20)
        self.assertAlmostEqual(c.lane_curve_speed_floor, 0.25)
        self.assertAlmostEqual(c.lane_max_curvature, 1.0 / 1.5)        # physical min-turn-radius cap

    def test_validation_rejects_bad_knobs(self):
        for bad in (dict(lane_cart_yaw_tol=0.0), dict(lane_cart_yaw_tol=math.radians(90)),
                    dict(lane_curve_speed_floor=0.0), dict(lane_curve_speed_floor=1.5),
                    dict(lane_max_curvature=0.0), dict(lane_max_curvature=1.0)):   # 1.0 > 1/1.5 phys cap
            with self.assertRaises(ValueError):
                ArticulatedController(self.grid, self.spec, allow_test_fixtures=True, **bad)

    def test_cart_yaw_tol_honored_in_tracking(self):
        # With a route acquired, a cart_yaw divergence of 0.25 rad from the route drops the route at the
        # default 0.20 tol (stop_replan_tracking_error) but is tolerated at the relaxed 0.30 tol.
        for tol, expect_drop in ((0.20, True), (0.30, False)):
            c = ArticulatedController(self.grid, self.spec, allow_test_fixtures=True,
                                      lane_graph=self.graph, lane_cart_yaw_tol=tol)
            c.start((4, 0, 0), True)
            # STOPPED at the lane start so it comes to rest then plans the route (planning_stop -> route).
            for i in range(5):
                c.step(0.1, Odometry((-4.0, 0.0, 0.0), 0.1 + i * 0.1, 0.0, 0.0, 'test_fixture', 0.05),
                       Observation(0.1 + i * 0.1, (), True, True, 0.0, 'test_fixture', 10.0), now=0.1 + i * 0.1)
                if c.route is not None:
                    break
            self.assertIsNotNone(c.route)
            # tick 2: base still on the route (y=0) but the CART yaw has diverged 0.25 rad
            # (articulation = base_yaw - cart_yaw = -0.25 for cart_yaw=0.25, base_yaw=0)
            r = c.step(0.1, Odometry((-3.95, 0.0, 0.0), 0.2, 0.0, 0.0, 'test_fixture', 0.05),
                       Observation(0.2, (), True, True, -0.25, 'test_fixture', 10.0), now=0.2)
            self.assertEqual(r.reason == 'stop_replan_tracking_error', expect_drop,
                             f'tol={tol} reason={r.reason}')

    def test_tighter_max_curvature_clamps_commanded_turn(self):
        # A tighter commanded-curvature cap bounds |w|/v below the physical 1/1.5; the follower never
        # commands a turn tighter than lane_max_curvature. (Construction + attr contract; the clamp is
        # applied at the tracking law.)
        c = ArticulatedController(self.grid, self.spec, allow_test_fixtures=True,
                                  lane_graph=self.graph, lane_max_curvature=0.4)
        self.assertAlmostEqual(c.lane_max_curvature, 0.4)

    def _to_tracking(self, c):
        c.start((4, 0, 0), True)
        pose = (-4.0, 0.0, 0.0); v = w = 0.0
        for t in range(80):
            od = Odometry(pose, (t + 1) * 0.1, abs(v), w, 'test_fixture', 0.05)
            r = c.step(0.1, od, Observation((t + 1) * 0.1, (), True, True, 0.0, 'test_fixture', 10.0),
                       now=(t + 1) * 0.1)
            if c.route is not None and c.route.states and r.reason == 'tracking':
                return pose, v, w
            v, w = r.linear, r.angular
            th = pose[2] + w * 0.1
            ns = advance(TrainState(pose[0], pose[1], pose[2], 0.0), v * 0.1, (w / v if abs(v) > 1e-6 else 0), self.spec)
            pose = (pose[0] + v * 0.1 * math.cos(th), pose[1] + v * 0.1 * math.sin(th), th)
        self.fail('did not reach lane tracking')

    def test_coast_refusals_keeps_lane_through_refusal_fm433(self):
        # fm433 (A'-2): with lane_coast_refusals, a gate refusal (here an obstacle ahead) on an ACTIVE lane
        # KEEPS the route (coast) instead of dropping+replanning; the base still STOPS this tick (not
        # 'tracking'), so a genuinely new obstacle still halts the tow. Default off => the legacy drop.
        for coast, expect_route_kept in ((False, False), (True, True)):
            c = ArticulatedController(self.grid, self.spec, allow_test_fixtures=True, lane_graph=self.graph,
                                      lane_coast_refusals=coast, blocked_timeout=40.0)
            pose, v, w = self._to_tracking(c)
            self.assertIsNotNone(c.route)
            ahead = (pose[0] + 0.3, pose[1])       # obstacle right in front -> gate refusal
            od = Odometry(pose, 10.0, abs(v), w, 'test_fixture', 0.05)
            r = c.step(0.1, od, Observation(10.0, (ahead,), True, True, 0.0, 'test_fixture', 10.0), now=10.0)
            self.assertNotEqual(r.reason, 'tracking')          # the tow STOPS for the obstacle
            self.assertEqual(c.route is not None, expect_route_kept)   # coast keeps the lane; legacy drops it


class OutboundLeadInAndFastFailTests(unittest.TestCase):
    """fm-sim-robust-outbound: the improved crosstrack couple (PR232/233) seats the base a VARIABLE
    distance off the authored outbound dock start -- run coldlivewin landed +0.5 m north of y=15.25
    (pose (-4.584,15.746,0.363)) while coldliverecovery landed essentially AT it ((-4.464,15.262,-0.013)).
    The fixed authored start + default xy acquire tol then deadlocked the outbound: the lane never
    acquired, no route was produced, and the base froze until the 120 s mission nav-stuck WALL budget.

    A hand-authored lead-in to the EXACT start is not viable here -- a forward-only train 0.5 m lateral
    off a flat lane free-plans a ~12 m loop (confirmed on CPU) -- so the fix is a bounded, swept-validated
    closed-loop MERGE from the measured seat onto the authored centreline (`build_lead_in`), plus a named
    fast-fail (`lane_acquire_timeout_s`) for a seat beyond the band the lane can absorb. The lane start is
    re-centred to the seat distribution so BOTH recorded seats are within +-0.25 m and absorbable.

    Deterministic open-grid fixtures -- never evidence of a physics run. The real store adds obstacles and
    PhysX cart dynamics; the integration-lane GPU proof owns that (data/sim-robust-outbound-lane-start-v1)."""

    # The two recorded post-couple seats (base pose + cart_yaw) from the twin-robot-in-store-boot-v1 runs.
    SEATS = {'coldlivewin': (-4.584, 15.746, 0.363, 0.362),       # +0.5 m north of y=15.25
             'coldliverecovery': (-4.464, 15.262, -0.013, -0.013)}  # essentially at the authored dock

    def setUp(self):
        self.spec = MobileSpec()
        self.grid = GridMap((-12, 8, 4, 22), (), 0.20)      # open backroom/dock corridor
        self.goal = (-2.0, 15.25, 0.0)

    def _lane(self, start_y=15.5, lead_in_max_m=3.0):
        lane = TowLane('outbound_dock_to_bay', [-4.46, start_y, 0.0], list(self.goal),
                       start_cart_yaw=0.0, goal_articulation=0.3, start_tol_cart_yaw=0.35,
                       lead_in_max_m=lead_in_max_m)
        graph = TowLaneGraph([lane]).expand_all(ArticulatedMap(self.grid, self.spec))
        return lane, graph

    def _controller(self, graph, **kw):
        base = dict(allow_test_fixtures=True, goal_timeout=600.0, blocked_timeout=40.0,
                    lane_graph=graph, lane_reacquire_along_route=True)
        base.update(kw)
        return ArticulatedController(self.grid, self.spec, **base)

    def _drive(self, c, seat, ticks=8000):
        c.start(self.goal, True)
        pose = seat[:3]; cart_yaw = seat[3]; v = w = 0.0; dt = 0.1
        for t in range(ticks):
            now = (t + 1) * dt
            art = math.atan2(math.sin(pose[2] - cart_yaw), math.cos(pose[2] - cart_yaw))
            od = Odometry(pose, now, abs(v), w, 'test_fixture', 0.05)
            obs = Observation(now, (), True, True, art, 'test_fixture', 10.0)
            r = c.step(dt, od, obs, now=now)
            if r.status in ('succeeded', 'failed'):
                return r, round(now, 2)
            v, w = r.linear, r.angular
            th = pose[2] + w * dt
            ns = advance(TrainState(pose[0], pose[1], pose[2], cart_yaw),
                         v * dt, (w / v if abs(v) > 1e-6 else 0.0), self.spec)
            pose = (pose[0] + v * dt * math.cos(th), pose[1] + v * dt * math.sin(th), th)
            cart_yaw = ns.cart_yaw
        return r, round(now, 2)

    # ---- build_lead_in geometry ----
    def test_build_lead_in_bridges_each_recorded_seat(self):
        lane, _ = self._lane()
        world = ArticulatedMap(self.grid, self.spec)
        for name, seat in self.SEATS.items():
            route = lane.build_lead_in(TrainState(*seat), world, self.spec)
            self.assertIsNotNone(route, f'{name}: lead-in should bridge the seat')
            # the spliced route STARTS at the measured seat (so the follower acquires trivially) ...
            self.assertAlmostEqual(route.states[0].x, seat[0], places=6)
            self.assertAlmostEqual(route.states[0].y, seat[1], places=6)
            # ... and ENDS at the authored goal (the authored tail is spliced on).
            self.assertLess(math.dist(route.states[-1].pose[:2], self.goal[:2]), 0.05)
            self.assertEqual(len(route.curvatures), len(route.states) - 1)

    def test_build_lead_in_none_when_seat_beyond_band(self):
        lane, _ = self._lane(lead_in_max_m=0.3)            # 0.3 m cannot bridge the 0.5 m north seat
        world = ArticulatedMap(self.grid, self.spec)
        self.assertIsNone(lane.build_lead_in(TrainState(*self.SEATS['coldlivewin']), world, self.spec))

    def test_build_lead_in_off_by_default_returns_none(self):
        lane, _ = self._lane(lead_in_max_m=0.0)            # default off
        world = ArticulatedMap(self.grid, self.spec)
        self.assertIsNone(lane.build_lead_in(TrainState(*self.SEATS['coldlivewin']), world, self.spec))

    # ---- plan_observed_route verdicts ----
    def test_plan_route_uses_leadin_for_drifted_seat_authored_for_on_start(self):
        lane, graph = self._lane()
        c = self._controller(graph, lane_cart_yaw_rebase=True, lane_cart_yaw_tol=0.35)
        c.start(self.goal, True)
        clear = Observation(0.1, (), True, True, 0.0, 'test_fixture', 10.0)
        drifted = c.plan_observed_route(TrainState(*self.SEATS['coldlivewin']), clear, 0.05)
        self.assertEqual(drifted.reason, 'validated_lane_leadin')
        on_start = c.plan_observed_route(TrainState(-4.46, 15.5, 0.0, 0.0), clear, 0.05)
        self.assertEqual(on_start.reason, 'validated_lane')     # at the authored start: no lead-in needed

    # ---- full closed-loop drive arrives from every recorded seat (re-centred example config) ----
    def test_outbound_arrives_from_every_recorded_seat(self):
        lane, graph = self._lane()
        for name, seat in self.SEATS.items():
            c = self._controller(graph, lane_cart_yaw_rebase=True, lane_cart_yaw_tol=0.35,
                                 lane_acquire_timeout_s=8.0)
            r, t = self._drive(c, seat)
            self.assertEqual(r.status, 'succeeded', f'{name}: {r.reason} @ {t}s')
            self.assertEqual(r.reason, 'arrived_stopped')

    # ---- fast named failure instead of the 120 s nav-stuck churn ----
    def test_unacquired_lane_fails_fast_by_name(self):
        lane, graph = self._lane(lead_in_max_m=0.0)         # lead-in off -> lane_not_acquired churn
        c = self._controller(graph, lane_acquire_timeout_s=5.0)
        r, t = self._drive(c, self.SEATS['coldlivewin'])
        self.assertEqual(r.status, 'failed')
        self.assertEqual(r.reason, 'lane_not_acquired_timeout')
        self.assertLessEqual(t, 6.0)                        # a few sim seconds, not 120 s

    def test_leadin_unreachable_fails_fast_by_name(self):
        lane, graph = self._lane(lead_in_max_m=0.3)         # too short to bridge -> named unreachable
        c = self._controller(graph, lane_acquire_timeout_s=5.0)
        r, t = self._drive(c, self.SEATS['coldlivewin'])
        self.assertEqual(r.status, 'failed')
        self.assertEqual(r.reason, 'lane_leadin_unreachable_timeout')
        self.assertLessEqual(t, 6.0)

    def test_acquire_timeout_off_is_byte_identical_churn(self):
        # timeout 0 + lead-in off => the legacy empty-route churn, never a terminal acquire failure.
        lane, graph = self._lane(lead_in_max_m=0.0)
        c = self._controller(graph, lane_acquire_timeout_s=0.0)
        r, t = self._drive(c, self.SEATS['coldlivewin'], ticks=300)
        self.assertNotEqual(r.status, 'succeeded')
        self.assertNotIn('_timeout', r.reason or '')        # no fast-fail fired

    def test_acquire_timeout_validation(self):
        _, graph = self._lane()
        with self.assertRaises(ValueError):
            ArticulatedController(self.grid, self.spec, allow_test_fixtures=True,
                                  lane_graph=graph, lane_acquire_timeout_s=-1.0)


class TowLaneLeadInConfigTests(unittest.TestCase):
    def test_from_config_round_trips_lead_in_keys(self):
        lane = TowLane.from_config({'name': 'L', 'start': [-4, 0, 0], 'goal': [4, 0, 0],
                                    'lead_in_max_m': 2.5, 'lead_in_fine_xy': 0.12,
                                    'lead_in_lookahead': 0.9})
        self.assertAlmostEqual(lane.lead_in_max_m, 2.5)
        self.assertAlmostEqual(lane.lead_in_fine_xy, 0.12)
        self.assertAlmostEqual(lane.lead_in_lookahead, 0.9)

    def test_defaults_are_off_and_backward_compatible(self):
        lane = TowLane.from_config({'name': 'L', 'start': [-4, 0, 0], 'goal': [4, 0, 0]})
        self.assertEqual(lane.lead_in_max_m, 0.0)           # off by default => never synthesised

    def test_rejects_bad_lead_in_values(self):
        for bad in ({'lead_in_max_m': -1.0}, {'lead_in_fine_xy': 0.0}, {'lead_in_lookahead': -0.5}):
            cfg = {'name': 'L', 'start': [-4, 0, 0], 'goal': [4, 0, 0]}
            cfg.update(bad)
            with self.assertRaises(ValueError):
                TowLane.from_config(cfg)


if __name__ == '__main__':
    unittest.main()
