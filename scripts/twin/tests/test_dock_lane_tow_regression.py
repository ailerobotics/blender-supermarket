"""CPU proof for fm/sim-dock-lane-tow-regression-v1.

Drives the REAL ``twin.navigation.articulated.ArticulatedController`` to reproduce and fix the
coupled-tow regression seen on the GPU side-bay runs: after the PR 293 dock-frame straight-REVERSE
couple, the first forward tow from rest is intermittent (coldwinint1 passed; coldwinint2 on the 304
dock AND coldwinint3 on the SAME proven portal-axis lane both failed ``coupled_no_progress``). The
recorded nav-traces show the common, lane-INDEPENDENT mechanism: the base launches, creps ~0.5 m,
the follower's speed schedule collapses to a creep/coast and leaves the front caster COCKED at
~10-14 deg, and the caster's scrub arc then trips ``gate_velocity``'s ``measured_turn_radius`` coast
(int3) or sits under the heading-hold creep (int2) so the tow pins from rest -- while the reactive
0.18-0.30 m/s break-away rock is too weak/late to straighten the cocked caster.

The fixes proved here (both opt-in; default off => byte-identical, see ``test_defaults_off``):
  * ``coupled_from_rest_launch`` -- a proactive from-rest launch that commands the committed
    ``_breakaway_cmd`` forward override for the first ``coupled_launch_distance_m`` of a coupled
    authored-lane tow, so the base carries momentum through the caster-swivel-in transient BEFORE a
    throttle can collapse the command. The override bypasses the ``measured_turn_radius`` coast
    (braking along the COMMANDED curvature), reproducing the coldwinint3 caster-scrub pin and
    showing the launch drives through it.
  * the ``coupled_heading_hold`` straight-branch now references the LOCAL authored lane tangent, not
    the fixed goal heading, so it no longer vetoes a legitimate gentle authored curve at an off-axis
    dock (the coldwinint2 compounding factor).
  * ``TOW_NO_PROGRESS`` names the real cause before ``coupled_no_progress``.

CPU-only; no Isaac/GPU/ROS. Run with:
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -I -m pytest \
    scripts/twin/tests/test_dock_lane_tow_regression.py -q
"""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.mobile import MobileSpec
from twin.navigation import GridMap, Odometry, Observation
from twin.navigation.articulated import ArticulatedController, ArticulatedMap, TrainState, advance
from twin.navigation.lanes import TowLaneGraph

# A cocked front caster's measured scrub arc exceeds the coupled min-turn-radius curvature
# (1/1.5 + 0.03 ~= 0.697 1/m) -- this is the signature that trips gate_velocity.measured_turn_radius.
SCRUB_K = 1.2

STRAIGHT_LANE = {'name': 'outbound_straight', 'start': [0.0, 0.0, 0.0], 'goal': [6.0, 0.0, 0.0],
                 'start_cart_yaw': 0.0, 'start_tol_cart_yaw': 0.35,
                 'waypoints': [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]], 'route_through_waypoints': True}

# The coldwinint2 off-axis 304 dock lane: the first segment legitimately points a few degrees off the
# due-south goal heading to shift x from the dock (-4.6) toward the portal (-4.3).
OFF_AXIS_LANE = {'name': 'outbound_dock_to_side_bay',
                 'start': [-4.6, 14.73, -math.pi / 2], 'start_cart_yaw': -math.pi / 2,
                 'goal': [-4.3, 6.95, -math.pi / 2], 'goal_articulation': 0.35, 'start_tol_cart_yaw': 0.35,
                 'waypoints': [[-4.3, 12.0, -math.pi / 2], [-4.3, 9.9, -math.pi / 2]],
                 'route_through_waypoints': True}


def _graph(lane_cfg, spec, grid):
    g = TowLaneGraph.from_config([lane_cfg])
    g.expand_all(ArticulatedMap(grid, spec))
    return g


def _controller(grid, spec, graph, **kw):
    return ArticulatedController(grid, spec, allow_test_fixtures=True, lane_graph=graph,
                                 goal_timeout=600.0, blocked_timeout=60.0, **kw)


def _reach_tracking(c, spec, lane, scrub=False):
    """Drive the lane follower closed-loop from the lane start until the first tracking tick.
    Returns (state, cart_yaw, v, w, now). When scrub is True, every odometry sample carries a cocked
    front caster's scrub arc (measured angular = SCRUB_K * linear) so measured_turn_radius is live."""
    c.start(lane.goal, True)
    s0 = lane.start_state()
    state = TrainState(s0.x, s0.y, s0.yaw, s0.cart_yaw)
    v = w = 0.0
    now = 0.0
    for i in range(400):
        now = (i + 1) * 0.05
        ang = SCRUB_K * abs(v) if (scrub and abs(v) > 0.02) else w
        od = Odometry(state.pose, now, abs(v), ang, 'test_fixture', 0.02)
        obs = Observation(now, (), True, True, state.articulation, 'test_fixture', 10.0)
        r = c.step(0.05, od, obs, now=now)
        v, w = r.velocity
        if c.route is not None and c.route.states and r.reason == 'tracking':
            return state, state.cart_yaw, v, w, now
        k = w / v if abs(v) > 1e-6 else 0.0
        state = advance(state, v * 0.05, k, spec)
    raise AssertionError('controller never reached tracking')


class FromRestLaunchTests(unittest.TestCase):
    def setUp(self):
        self.spec = MobileSpec()
        self.grid = GridMap((-12, -12, 18, 18))

    # ---- (1) the caster-scrub pin reproduction: coasts to rest WITHOUT the launch ------------------

    def test_caster_scrub_trips_measured_turn_coast_without_launch(self):
        """coldwinint3 mechanism: a cocked-caster scrub arc (|measured omega / v| > 1/1.5) repeatedly
        trips gate_velocity's measured_turn_radius coast, so the from-rest tow stutters toward a pin
        instead of a clean committed forward -- the controller refuses the tow on its own scrub."""
        graph = _graph(STRAIGHT_LANE, self.spec, self.grid)
        c = _controller(self.grid, self.spec, graph)                 # launch OFF (default)
        state, cart_yaw, v, w, now = _reach_tracking(c, self.spec, graph.lanes[0])
        self.assertEqual(c.from_rest_launches, 0)
        reasons = []
        for i in range(60):
            now += 0.05
            ang = SCRUB_K * abs(v) if abs(v) > 0.02 else w             # cocked-caster scrub measured arc
            od = Odometry(state.pose, now, abs(v), ang, 'test_fixture', 0.02)
            obs = Observation(now, (), True, True, state.articulation, 'test_fixture', 10.0)
            r = c.step(0.05, od, obs, now=now)
            reasons.append(r.reason)
            v, w = r.velocity
            k = w / v if abs(v) > 1e-6 else 0.0
            state = advance(state, v * 0.05, k, self.spec)
        self.assertGreaterEqual(reasons.count('measured_turn_radius'), 10)   # the scrub keeps coasting it
        self.assertEqual(c.from_rest_launches, 0)                     # launch off: never fires

    # ---- (2) the proactive from-rest launch drives THROUGH the same scrub ------------------------

    def test_from_rest_launch_drives_through_caster_scrub(self):
        graph = _graph(STRAIGHT_LANE, self.spec, self.grid)
        c = _controller(self.grid, self.spec, graph, coupled_progress_escape=True,
                        coupled_progress_recovery=True, coupled_from_rest_launch=True,
                        coupled_launch_distance_m=1.0, coupled_launch_mps=0.35)
        state, cart_yaw, v, w, now = _reach_tracking(c, self.spec, graph.lanes[0])
        self.assertGreaterEqual(c.from_rest_launches, 1)              # the launch armed at tow start
        self.assertIsNotNone(c._launch_anchor)
        launch_speeds = []
        launch_reasons = []
        for i in range(160):
            now += 0.05
            ang = SCRUB_K * abs(v) if abs(v) > 0.02 else w             # SAME cocked-caster scrub
            od = Odometry(state.pose, now, abs(v), ang, 'test_fixture', 0.02)
            obs = Observation(now, (), True, True, state.articulation, 'test_fixture', 10.0)
            r = c.step(0.05, od, obs, now=now)
            if not c._launch_done:
                launch_reasons.append(r.reason)
                launch_speeds.append(r.velocity[0])
            v, w = r.velocity
            k = w / v if abs(v) > 1e-6 else 0.0
            state = advance(state, v * 0.05, k, self.spec)
            if c._launch_done:
                break
        # the launch commanded a committed forward speed (full cap, never a creep), bypassed the
        # measured_turn coast throughout, and cleared the full metre from the anchor.
        self.assertTrue(launch_speeds)
        self.assertGreaterEqual(min(launch_speeds), 0.18)            # committed forward, never a creep
        self.assertNotIn('measured_turn_radius', launch_reasons)     # launch bypasses the scrub coast
        self.assertTrue(c._launch_done)                              # one-shot: disarmed past the distance
        self.assertGreaterEqual(math.hypot(state.x - c._launch_anchor[0],
                                           state.y - c._launch_anchor[1]), 0.95)  # drove the first metre

    def test_launch_is_one_shot_and_hands_back(self):
        """Past coupled_launch_distance_m the launch disarms and never re-arms for the goal."""
        graph = _graph(STRAIGHT_LANE, self.spec, self.grid)
        c = _controller(self.grid, self.spec, graph, coupled_progress_escape=True,
                        coupled_progress_recovery=True, coupled_from_rest_launch=True,
                        coupled_launch_distance_m=0.5, coupled_launch_mps=0.35)
        state, cart_yaw, v, w, now = _reach_tracking(c, self.spec, graph.lanes[0])
        for i in range(120):
            now += 0.05
            od = Odometry(state.pose, now, abs(v), w, 'test_fixture', 0.02)
            obs = Observation(now, (), True, True, state.articulation, 'test_fixture', 10.0)
            r = c.step(0.05, od, obs, now=now)
            v, w = r.velocity
            k = w / v if abs(v) > 1e-6 else 0.0
            state = advance(state, v * 0.05, k, self.spec)
            if r.status != 'running':
                break
        self.assertTrue(c._launch_done)
        launched = c.from_rest_launches
        # keep driving: the counter must NOT climb again (one-shot per goal)
        for i in range(40):
            now += 0.05
            od = Odometry(state.pose, now, abs(v), w, 'test_fixture', 0.02)
            obs = Observation(now, (), True, True, state.articulation, 'test_fixture', 10.0)
            r = c.step(0.05, od, obs, now=now)
            v, w = r.velocity
            k = w / v if abs(v) > 1e-6 else 0.0
            state = advance(state, v * 0.05, k, self.spec)
            if r.status != 'running':
                break
        self.assertEqual(c.from_rest_launches, launched)

    # ---- (3) default-off is byte-identical (no new behaviour) ------------------------------------

    def test_defaults_off(self):
        c = ArticulatedController(self.grid, self.spec, allow_test_fixtures=True)
        self.assertFalse(c.coupled_from_rest_launch)
        self.assertEqual(c.from_rest_launches, 0)

    def test_launch_requires_progress_recovery(self):
        with self.assertRaises(ValueError):
            ArticulatedController(self.grid, self.spec, allow_test_fixtures=True,
                                  coupled_from_rest_launch=True, coupled_progress_recovery=False)

    def test_launch_speed_at_least_breakaway(self):
        with self.assertRaises(ValueError):
            ArticulatedController(self.grid, self.spec, allow_test_fixtures=True,
                                  coupled_progress_escape=True, coupled_progress_recovery=True,
                                  coupled_from_rest_launch=True, coupled_launch_mps=0.05,
                                  coupled_breakaway_speed_mps=0.18)


class HeadingHoldCurveTests(unittest.TestCase):
    """The straight-branch heading hold references the LOCAL authored tangent, so a legitimate gentle
    authored curve at an off-axis dock is no longer vetoed (coldwinint2)."""

    def setUp(self):
        self.spec = MobileSpec()
        self.grid = GridMap((-12, -12, 18, 18))

    def _drive(self, lane_cfg, **kw):
        graph = _graph(lane_cfg, self.spec, self.grid)
        lane = graph.lanes[0]
        c = _controller(self.grid, self.spec, graph, **kw)
        c.start(lane.goal, True)
        s0 = lane.start_state()
        state = TrainState(s0.x, s0.y, s0.yaw, s0.cart_yaw)
        v = w = 0.0
        now = 0.0
        min_x = state.x
        for i in range(6000):
            now = (i + 1) * 0.05
            od = Odometry(state.pose, now, abs(v), w, 'test_fixture', 0.02)
            obs = Observation(now, (), True, True, state.articulation, 'test_fixture', 10.0)
            r = c.step(0.05, od, obs, now=now)
            if r.status == 'succeeded':
                return True, state, c
            if r.status == 'failed':
                return False, state, c
            v, w = r.velocity
            k = w / v if abs(v) > 1e-6 else 0.0
            state = advance(state, v * 0.05, k, self.spec)
            min_x = min(min_x, state.x)
        return False, state, c

    def test_off_axis_curve_follows_to_goal_with_heading_hold(self):
        # With heading hold ON, the kinematic follower reaches the side-bay goal on the off-axis 304
        # lane: the guard no longer force-holds the base dead-south off the authored curve. (The
        # kinematic model has no physics caster pin, so reaching the goal isolates the guard-veto fix.)
        reached, state, c = self._drive(OFF_AXIS_LANE, coupled_heading_hold=True,
                                        heading_hold_bound_deg=1.0, heading_hold_creep_mps=0.14,
                                        heading_hold_straight_kappa=0.08)
        self.assertTrue(reached, 'off-axis authored curve was vetoed/stalled by the heading hold')
        # the base actually tracked EAST onto the portal axis (x -4.6 -> ~-4.3), not held at the dock x
        self.assertGreater(state.x, -4.45)

    def test_straight_lane_heading_hold_unchanged(self):
        # On a straight-to-goal lane ref_yaw == goal heading, so the guard behaviour is unchanged and
        # the follower still reaches the goal.
        reached, state, c = self._drive(STRAIGHT_LANE, coupled_heading_hold=True,
                                        heading_hold_bound_deg=1.0, heading_hold_creep_mps=0.14,
                                        heading_hold_straight_kappa=0.08)
        self.assertTrue(reached)


class TowNoProgressDiagTests(unittest.TestCase):
    def setUp(self):
        self.spec = MobileSpec()
        self.grid = GridMap((-12, -12, 18, 18))

    def _controller_with_window(self):
        graph = _graph(STRAIGHT_LANE, self.spec, self.grid)
        c = _controller(self.grid, self.spec, graph)
        c.start(graph.lanes[0].goal, True)
        c._active_lane = graph.lanes[0]
        c._prog_anchor = (5.0, 0.0, 0.0)          # stuck window opened at sim t=5.0
        c._prog_stuck_s = 8.0
        return c

    def test_diag_names_caster_scrub_cause(self):
        c = self._controller_with_window()
        c._mtr_window = 1.5                        # measured_turn coasts dominated the stuck window
        diag = c._tow_no_progress_diag(10.0, TrainState(0.5, 0.0, 0.0, 0.0))
        self.assertIn('TOW_NO_PROGRESS', diag)
        self.assertIn('from_rest_caster_scrub_measured_turn_coast', diag)

    def test_diag_names_heading_hold_veto_cause(self):
        c = self._controller_with_window()
        c._hh_last_fire_now = 6.0                   # heading hold overrode inside the window (>= 5.0)
        c._hh_last_ae = math.radians(1.3)
        c._hh_last_ref_yaw = -0.1
        diag = c._tow_no_progress_diag(10.0, TrainState(-4.58, 14.2, -1.55, -1.55))
        self.assertIn('heading_hold_veto_of_authored_heading', diag)

    def test_diag_names_bare_from_rest_pin_cause(self):
        c = self._controller_with_window()
        diag = c._tow_no_progress_diag(10.0, TrainState(0.5, 0.0, 0.0, 0.0))
        self.assertIn('from_rest_contact_pin', diag)


if __name__ == '__main__':
    unittest.main()
