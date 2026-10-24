"""CPU contract tests for the navigation lane factory `nav_factory`.

The factory is consumed by `twin.restock.runtime.NavigationBridge` on the
integration branch `fm/twin-robot-in-store-boot-v1` (head 6b24902). That module
is not on main, so these tests mirror the EXACT calls the bridge makes per
nav_mode rather than importing it, and prove the returned object answers them.
No ROS node, GPU boot, simulator or fixture/ground-truth input is used.
"""
import inspect
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.navigation.factory import nav_factory, _require_bridge_contract, NAV_MODES
from twin.navigation.articulated import ArticulatedController
from twin.navigation.runtime import SensorNavigation
from twin.navigation.core import Odometry, Observation, Result
from twin.navigation.sensors import Scan

MANIFEST = {
    'map_bounds': [-6., 6., -4., 4.],              # xmin, xmax, ymin, ymax
    'static_obstacles': [{'min': [-6., -4.], 'max': [-5.6, 4.]},
                         {'min': [5.6, -4.], 'max': [6., 4.]},
                         {'min': [-1., 1.5], 'max': [1., 2.]}],
}
SENSOR_CFG = {'wheel_radius': 0.16, 'wheel_separation': 0.70,
              'initial_pose': [0., 0., 0.], 'initial_map_pose': [0., 0., 0.]}


def _plan(nav_mode):
    return SimpleNamespace(nav_mode=nav_mode)


def _config(nav_mode, **nav_extra):
    nav = {'environment': MANIFEST}
    nav.update(nav_extra)
    return {'nav_factory': 'twin.navigation.factory:nav_factory', 'navigation': nav}


def _build(nav_mode, *, assembly=None, output=None, **nav_extra):
    return nav_factory(app=None, assembly=assembly if assembly is not None else object(),
                       plan=_plan(nav_mode), config=_config(nav_mode, **nav_extra),
                       output=output or tempfile.gettempdir())


class _FakeNav2Adapter:
    """Records the exact bridge calls; stands in for the ROS ArticulatedNav2Adapter."""
    def __init__(self, controller):
        self.controller, self.calls = controller, []

    def start(self, goal, coupled, *, goal_articulation=None):
        self.calls.append(('start', tuple(goal), coupled, goal_articulation))
        if goal_articulation is None:
            self.controller.start(goal, coupled)
        else:
            self.controller.start(goal, coupled, goal_articulation=goal_articulation)

    def cancel(self):
        self.calls.append(('cancel',)); return self.controller.cancel()

    def publish_sensors(self, odometry, scan):
        self.calls.append(('publish_sensors',))

    def step(self, dt, odometry, scan, *, coupled, arm_transport, articulation=0.0, temporal_map=None):
        self.calls.append(('step', dt, coupled, arm_transport, articulation))
        return Result(reason='fake_nav2')


class FactoryDispatchTests(unittest.TestCase):
    def test_reference_mode_returns_production_articulated_controller(self):
        obj = _build('reference')
        self.assertIsInstance(obj, ArticulatedController)
        self.assertFalse(obj.allow_test_fixtures)
        # Grid built from the static manifest (3 obstacles) + map walls.
        self.assertEqual(len(obj.grid.obstacles), 3)

    def test_sensor_mode_returns_sensor_navigation(self):
        obj = _build('sensor', sensor=SENSOR_CFG)
        self.assertIsInstance(obj, SensorNavigation)
        self.assertIsInstance(obj.controller, ArticulatedController)
        self.assertFalse(obj.controller.allow_test_fixtures)

    def test_controller_options_and_spec_flow_through(self):
        obj = _build('reference', goal_timeout=240., require_temporal_map=True,
                     mobile_spec={'cart_width': 0.72})
        self.assertEqual(obj.goal_timeout, 240.)
        self.assertTrue(obj.require_temporal_map)
        self.assertAlmostEqual(obj.spec.cart_width, 0.72)

    def test_docking_absent_leaves_controller_unchanged(self):
        self.assertIsNone(_build('reference').docking)

    def test_planner_timeout_absent_is_default_30s(self):
        # Absent navigation.planner_timeout_s => the coupled planner keeps its 30 s default budget.
        self.assertEqual(_build('reference').planner.timeout, 30.0)

    def test_planner_timeout_flows_through(self):
        # Config-driven coupled-plan wall budget: a loaded host can be given more time so the full
        # search is not starved into 'planning_timeout' (a compute budget, not a collision tolerance).
        self.assertEqual(_build('reference', planner_timeout_s=120.0).planner.timeout, 120.0)

    def test_planner_timeout_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            _build('reference', planner_timeout_s=0)

    def test_docking_block_flows_through(self):
        obj = _build('reference', docking={'goal': [-4.46, 15.25, 0.0], 'speed': 0.08,
                                           'approach_distance': 1.5, 'goal_tolerance': 0.3})
        self.assertIsNotNone(obj.docking)
        self.assertEqual(obj.docking.goal, (-4.46, 15.25, 0.0))
        self.assertAlmostEqual(obj.docking.speed, 0.08)
        self.assertAlmostEqual(obj.docking.approach_distance, 1.5)
        self.assertAlmostEqual(obj.docking.goal_tolerance, 0.3)

    def test_docking_requires_goal_and_fields(self):
        for bad in ({'speed': 0.08, 'approach_distance': 1.5},
                    {'goal': [-4.46, 15.25, 0.0], 'approach_distance': 1.5},
                    {'goal': [-4.46, 15.25, 0.0], 'speed': 0.08},
                    [1, 2, 3]):
            with self.assertRaises(ValueError):
                _build('reference', docking=bad)

    def test_directional_zone_default_on_and_config_toggle(self):
        self.assertTrue(_build('reference').directional_zone)                 # default on
        self.assertFalse(_build('reference', directional_zone=False).directional_zone)
        self.assertTrue(_build('reference', directional_zone=True).directional_zone)

    def test_directional_zone_rejects_non_bool(self):
        with self.assertRaises(ValueError):
            _build('reference', directional_zone='yes')

    def test_tow_lanes_absent_leaves_no_lane_graph(self):
        self.assertIsNone(_build('reference').lane_graph)

    def test_tow_lanes_built_and_expanded(self):
        # A straight coupled lane through the open centre of the fixture grid expands at build time.
        lane = {'name': 'demo', 'start': [-3, 0, 0], 'goal': [3, 0, 0],
                'start_cart_yaw': 0.0, 'goal_articulation': 0.3, 'waypoints': [[0, 0, 0]]}
        obj = _build('reference', tow_lanes=[lane])
        self.assertIsNotNone(obj.lane_graph)
        self.assertEqual([ln.name for ln in obj.lane_graph.lanes], ['demo'])
        self.assertIsNotNone(obj.lane_graph.lanes[0].route)                   # expanded at build
        self.assertIs(obj.lane_graph.resolve_goal((3.0, 0.0, 0.0)), obj.lane_graph.lanes[0])

    def test_tow_lane_that_cannot_expand_fails_the_build(self):
        # A lane whose start sits inside a wall cannot be realised -> the build raises (fail-closed).
        lane = {'name': 'in_wall', 'start': [5.9, 0, 0], 'goal': [3, 0, 0],
                'start_cart_yaw': 0.0, 'goal_articulation': 0.3}
        with self.assertRaises(ValueError):
            _build('reference', tow_lanes=[lane])

    def test_tow_lane_static_tol_flows_through(self):
        self.assertEqual(_build('reference').lane_static_tol, 0.0)             # default off
        self.assertAlmostEqual(_build('reference', tow_lane_static_tol_m=0.12).lane_static_tol, 0.12)

    def test_tow_lane_trust_static_flows_through(self):
        self.assertFalse(_build('reference').lane_trust_static)               # default off
        self.assertTrue(_build('reference', tow_lane_trust_static=True).lane_trust_static)

    def test_tow_lane_reacquire_along_route_flows_through(self):
        self.assertFalse(_build('reference').lane_reacquire_along_route)      # default off
        self.assertTrue(_build('reference', tow_lane_reacquire_along_route=True).lane_reacquire_along_route)

    def test_coupled_tracking_knobs_flow_through(self):
        c = _build('reference')
        self.assertAlmostEqual(c.lane_cart_yaw_tol, 0.20)                     # defaults
        self.assertAlmostEqual(c.lane_curve_speed_floor, 0.25)
        self.assertAlmostEqual(c.lane_max_curvature, 1.0 / 1.5)
        c2 = _build('reference', tow_lane_cart_yaw_tol=0.30,
                    tow_lane_curve_speed_floor=0.15, tow_lane_max_curvature=0.4)
        self.assertAlmostEqual(c2.lane_cart_yaw_tol, 0.30)
        self.assertAlmostEqual(c2.lane_curve_speed_floor, 0.15)
        self.assertAlmostEqual(c2.lane_max_curvature, 0.4)

    def test_tow_lane_coast_refusals_flows_through(self):
        self.assertFalse(_build('reference').lane_coast_refusals)             # default off
        self.assertTrue(_build('reference', tow_lane_coast_refusals=True).lane_coast_refusals)

    def test_tow_lane_acquire_timeout_flows_through(self):
        self.assertEqual(_build('reference').lane_acquire_timeout_s, 0.0)     # default off
        self.assertAlmostEqual(_build('reference', tow_lane_acquire_timeout_s=8.0).lane_acquire_timeout_s, 8.0)

    def test_tow_lane_trust_unknown_flows_through(self):
        self.assertFalse(_build('reference').lane_trust_unknown)              # default off
        self.assertTrue(_build('reference', tow_lane_trust_unknown=True).lane_trust_unknown)

    def test_coupled_progress_escape_flows_through(self):
        c = _build('reference')
        self.assertFalse(c.coupled_progress_escape)                          # default off
        self.assertAlmostEqual(c.coupled_progress_min_m, 0.05)               # defaults
        self.assertAlmostEqual(c.coupled_progress_escape_s, 3.0)
        self.assertAlmostEqual(c.coupled_progress_fault_s, 8.0)
        c2 = _build('reference', coupled_progress_escape=True, coupled_progress_min_m=0.08,
                    coupled_progress_escape_s=2.0, coupled_progress_fault_s=6.0)
        self.assertTrue(c2.coupled_progress_escape)
        self.assertAlmostEqual(c2.coupled_progress_min_m, 0.08)
        self.assertAlmostEqual(c2.coupled_progress_escape_s, 2.0)
        self.assertAlmostEqual(c2.coupled_progress_fault_s, 6.0)

    def test_coupled_progress_escape_rejects_non_bool(self):
        with self.assertRaises(ValueError):
            _build('reference', coupled_progress_escape='true')              # _strict_bool

    def test_coupled_breakaway_recovery_flows_through(self):
        c = _build('reference')
        self.assertFalse(c.coupled_progress_recovery)                        # default off
        self.assertAlmostEqual(c.coupled_breakaway_speed, 0.18)             # defaults
        self.assertAlmostEqual(c.coupled_breakaway_window_s, 2.0)
        self.assertAlmostEqual(c.coupled_breakaway_pulse_s, 0.5)
        self.assertEqual(c.coupled_breakaway_attempts, 3)
        c2 = _build('reference', coupled_progress_recovery=True, coupled_breakaway_speed_mps=0.22,
                    coupled_breakaway_window_s=2.5, coupled_breakaway_pulse_s=0.4,
                    coupled_breakaway_attempts=4)
        self.assertTrue(c2.coupled_progress_recovery)
        self.assertAlmostEqual(c2.coupled_breakaway_speed, 0.22)
        self.assertAlmostEqual(c2.coupled_breakaway_window_s, 2.5)
        self.assertAlmostEqual(c2.coupled_breakaway_pulse_s, 0.4)
        self.assertEqual(c2.coupled_breakaway_attempts, 4)

    def test_coupled_progress_recovery_rejects_non_bool(self):
        with self.assertRaises(ValueError):
            _build('reference', coupled_progress_recovery='true')            # _strict_bool

    def test_coupled_heading_hold_flows_through(self):
        c = _build('reference')
        self.assertFalse(c.coupled_heading_hold)                             # default off
        self.assertAlmostEqual(c.heading_hold_bound, math.radians(1.0))      # defaults
        self.assertAlmostEqual(c.heading_hold_creep, 0.14)
        self.assertAlmostEqual(c.heading_hold_straight_kappa, 0.08)
        c2 = _build('reference', coupled_heading_hold=True, heading_hold_bound_deg=1.5,
                    heading_hold_creep_mps=0.16, heading_hold_straight_kappa=0.06)
        self.assertTrue(c2.coupled_heading_hold)
        self.assertAlmostEqual(c2.heading_hold_bound, math.radians(1.5))
        self.assertAlmostEqual(c2.heading_hold_creep, 0.16)
        self.assertAlmostEqual(c2.heading_hold_straight_kappa, 0.06)

    def test_coupled_heading_hold_rejects_non_bool(self):
        with self.assertRaises(ValueError):
            _build('reference', coupled_heading_hold='true')                 # _strict_bool

    def test_heading_hold_prethroat_flows_through(self):
        # sim-robust-heading-hold-bound-v1: the pre-throat alignment gate keys thread through.
        c = _build('reference')
        self.assertEqual(c.heading_hold_prethroat_m, 0.0)                    # default off
        self.assertIsNone(c.heading_hold_throat)
        self.assertIsNone(c.heading_hold_axis)
        self.assertAlmostEqual(c.heading_hold_align_timeout_s, 20.0)
        c2 = _build('reference', coupled_heading_hold=True, heading_hold_prethroat_m=0.8,
                    heading_hold_throat=[-3.875, 15.26], heading_hold_align_timeout_s=15.0,
                    heading_hold_axis_deg=0.0)
        self.assertAlmostEqual(c2.heading_hold_prethroat_m, 0.8)
        self.assertEqual(c2.heading_hold_throat, (-3.875, 15.26))
        self.assertAlmostEqual(c2.heading_hold_align_timeout_s, 15.0)
        self.assertAlmostEqual(c2.heading_hold_axis, 0.0)

    def test_heading_hold_prethroat_requires_throat(self):
        with self.assertRaises(ValueError):
            _build('reference', coupled_heading_hold=True, heading_hold_prethroat_m=0.8)


class FailClosedTests(unittest.TestCase):
    def test_invalid_nav_mode_refused(self):
        with self.assertRaises(ValueError):
            nav_factory(app=None, assembly=object(), plan=_plan('teleport'),
                        config=_config('teleport'), output=tempfile.gettempdir())

    def test_missing_assembly_refused_in_production(self):
        with self.assertRaises(ValueError):
            nav_factory(app=None, assembly=None, plan=_plan('reference'),
                        config=_config('reference'), output=tempfile.gettempdir())

    def test_missing_navigation_config_refused(self):
        with self.assertRaises(ValueError):
            nav_factory(app=None, assembly=object(), plan=_plan('reference'),
                        config={'nav_factory': 'x:y'}, output=tempfile.gettempdir())

    def test_fixture_knob_forbidden(self):
        with self.assertRaises(ValueError):
            _build('reference', allow_test_fixtures=True)

    def test_sensor_mode_requires_calibration(self):
        with self.assertRaises(ValueError):
            _build('sensor')  # no navigation.sensor block

    def test_missing_environment_refused(self):
        with self.assertRaises(ValueError):
            nav_factory(app=None, assembly=object(), plan=_plan('reference'),
                        config={'navigation': {}}, output=tempfile.gettempdir())


class BridgeContractDriveTests(unittest.TestCase):
    """Drive the returned object through the bridge's exact per-mode call sequence."""

    def _odom(self, t):
        return Odometry((0., 0., 0.), t, 0., 0., source='wheel_odometry_localized',
                        position_uncertainty=0.10)

    def _obs(self, t):
        return Observation(t, (), True, True, 0.0, 'simulated_lidar', 0.0)

    def _scan(self, t):
        return Scan(t, (5.0,) * 360, -math.pi, 2 * math.pi / 360, 10.0)

    def test_reference_bridge_calls(self):
        obj = _build('reference')
        # bridge.start(goal, coupled, goal_articulation=...) for an articulated action
        obj.start((2., 0., 0.), True, goal_articulation=0.5)
        self.assertEqual(obj.goal_articulation, 0.5)
        # bridge.step(dt, odometry, navigation, now=simulation_time)
        result = obj.step(0.05, self._odom(1.), self._obs(1.), now=1.)
        self.assertIsInstance(result, Result)
        obj.cancel()
        self.assertEqual(obj.terminal.reason, 'cancelled')

    def test_reference_rejects_fixture_provenance_in_production(self):
        obj = _build('reference')
        obj.start((2., 0., 0.), True)
        fixture_odom = Odometry((0., 0., 0.), 1., source='test_fixture', position_uncertainty=0.10)
        fixture_obs = Observation(1., (), True, True, 0., 'test_fixture', 0.)
        result = obj.step(0.05, fixture_odom, fixture_obs, now=1.)
        self.assertEqual(result.reason, 'unsupported_observation_source')

    def test_sensor_bridge_calls(self):
        obj = _build('sensor', sensor=SENSOR_CFG)
        obj.start((2., 0., 0.), True, goal_articulation=0.4)
        self.assertEqual(obj.controller.goal_articulation, 0.4)
        # bridge.step(dt, packet.odometry(=wheel angles), packet.navigation(=scan), now=, ...)
        result = obj.step(0.05, (0., 0.), self._scan(1.), now=1.,
                          coupled=True, arm_transport=True, articulation=0.0)
        self.assertIsInstance(result, Result)
        obj.cancel()

    def test_nav2_bridge_calls_with_injected_adapter(self):
        env = {'ROS_DOMAIN_ID': '88', 'ROS_LOCALHOST_ONLY': '1',
               'ROS_AUTOMATIC_DISCOVERY_RANGE': 'LOCALHOST'}
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            obj = nav_factory(app=None, assembly=object(), plan=_plan('nav2'),
                              config=_config('nav2'), output=tempfile.gettempdir(),
                              _nav2_adapter=_FakeNav2Adapter)
        finally:
            for k, v in saved.items():
                os.environ[k] = v if v is not None else ''
                if v is None:
                    del os.environ[k]
        self.assertIsInstance(obj, _FakeNav2Adapter)
        obj.start((2., 0., 0.), True, goal_articulation=0.5)
        self.assertEqual(obj.controller.goal_articulation, 0.5)
        obj.publish_sensors(self._odom(1.), self._scan(1.))
        result = obj.step(0.05, self._odom(1.), self._scan(1.),
                          coupled=True, arm_transport=True, articulation=0.0)
        self.assertEqual(result.reason, 'fake_nav2')
        self.assertEqual([c[0] for c in obj.calls], ['start', 'publish_sensors', 'step'])

    def test_nav2_requires_domain88_isolation(self):
        saved = {k: os.environ.get(k) for k in
                 ('ROS_DOMAIN_ID', 'ROS_LOCALHOST_ONLY', 'ROS_AUTOMATIC_DISCOVERY_RANGE')}
        for k in saved:
            os.environ.pop(k, None)
        try:
            with self.assertRaises(RuntimeError):
                nav_factory(app=None, assembly=object(), plan=_plan('nav2'),
                            config=_config('nav2'), output=tempfile.gettempdir(),
                            _nav2_adapter=_FakeNav2Adapter)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


class RealNav2ClassContractTests(unittest.TestCase):
    def test_real_articulated_nav2_adapter_signatures_match_bridge(self):
        # Static signature check of the real ROS overlay class (no instantiation).
        from twin.navigation.nav2 import ArticulatedNav2Adapter, Nav2Adapter
        s = object()
        def binds(func, args, kwargs):
            try:
                inspect.signature(func).bind(*args, **kwargs); return True
            except TypeError:
                return False
        # unbound methods include self
        self.assertTrue(binds(ArticulatedNav2Adapter.start, (s, s, s), {'goal_articulation': s}))
        self.assertTrue(binds(ArticulatedNav2Adapter.start, (s, s, s), {}))
        self.assertTrue(binds(Nav2Adapter.publish_sensors, (s, s, s), {}))
        self.assertTrue(binds(ArticulatedNav2Adapter.step, (s, s, s, s),
                              {'coupled': s, 'arm_transport': s, 'articulation': s}))
        self.assertTrue(binds(Nav2Adapter.cancel, (s,), {}))

    def test_contract_checker_rejects_incomplete_object(self):
        class Missing:  # no cancel/step
            def start(self, goal, coupled, *, goal_articulation=None): ...
        with self.assertRaises(ValueError):
            _require_bridge_contract(Missing(), 'reference')


if __name__ == '__main__':
    unittest.main()
