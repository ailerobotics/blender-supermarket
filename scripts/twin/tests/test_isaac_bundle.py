"""CPU contract tests for the isaac_bundle:create factory and its pure plan.

No Isaac, no GPU: exercises the config->mission-contract mapping (plan_bundle) and the
factory's assembly WIRING via injection seams. The injected assembly/runtime are
test fixtures; they never stand in for a physical mission run, and the production
factory path (real Isaac builders) is not exercised here.
"""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import isaac_bundle
from twin.restock.isaac_bundle import plan_bundle, create, BundlePlan, NAV_GOALS
from twin.restock.runtime import NavigationBridge


def base_config(**over):
    cfg = {
        'mission_poses': {'pickup': [-5.73, 16.0, 0.0], 'cart_dock': [-7.0, 16.0, 0.0],
                          'shelf_approach': [-3.05, 6.0, math.pi], 'home': [0.0, 0.0, 0.0]},
        'goal_articulation': {'outbound': math.radians(10), 'return': math.radians(10)},
        'package_roots': ['/opt/ros/jazzy/share', '/home/x/shelfbot'],
        'recording_fps': 30,
        'nav_factory': 'mod:nav', 'manip_factory': 'mod:manip', 'sensing_factory': 'mod:sense',
    }
    cfg.update(over)
    return cfg


SCENARIO = {'navigation_model': 'articulated', 'job': {'sku': 'x', 'quantity': 1}}
MANIFEST = {'frame': 'map'}


class PlanBundleTests(unittest.TestCase):
    def test_valid(self):
        p = plan_bundle(MANIFEST, SCENARIO, base_config())
        self.assertIsInstance(p, BundlePlan)
        self.assertEqual(set(p.poses), set(NAV_GOALS))
        self.assertEqual(p.poses['shelf_approach'], (-3.05, 6.0, math.pi))
        self.assertEqual(p.nav_mode, 'sensor')  # from articulated scenario
        self.assertEqual(p.recording_fps, 30)
        self.assertEqual(p.verification_source, 'rollwagen')
        self.assertAlmostEqual(p.goal_articulation['outbound'], math.radians(10))

    def test_nav_mode_override_and_default(self):
        self.assertEqual(plan_bundle(MANIFEST, {'navigation_model': 'reference', 'job': {'sku': 'x', 'quantity': 1}},
                                     base_config()).nav_mode, 'reference')
        self.assertEqual(plan_bundle(MANIFEST, SCENARIO, base_config(nav_mode='nav2')).nav_mode, 'nav2')

    def test_missing_pose(self):
        cfg = base_config()
        del cfg['mission_poses']['home']
        with self.assertRaises(ValueError):
            plan_bundle(MANIFEST, SCENARIO, cfg)

    def test_extra_pose(self):
        cfg = base_config()
        cfg['mission_poses']['extra'] = [0, 0, 0]
        with self.assertRaises(ValueError):
            plan_bundle(MANIFEST, SCENARIO, cfg)

    def test_nonfinite_pose(self):
        cfg = base_config()
        cfg['mission_poses']['pickup'] = [float('inf'), 0, 0]
        with self.assertRaises(ValueError):
            plan_bundle(MANIFEST, SCENARIO, cfg)

    def test_place_approach_optional_pose_accepted(self):
        # fm390: place_approach is an optional nav goal (collect drives there before placing).
        cfg = base_config()
        cfg['mission_poses']['place_approach'] = [-4.5, 15.25, math.pi]
        p = plan_bundle(MANIFEST, SCENARIO, cfg)
        self.assertEqual(p.poses['place_approach'], (-4.5, 15.25, math.pi))
        # absent => not present, bundle still valid (byte-identical with today)
        self.assertNotIn('place_approach', plan_bundle(MANIFEST, SCENARIO, base_config()).poses)

    def test_place_approach_stage_optional_pose_accepted(self):
        # fm390 nav proof #2: place_approach_stage is the east staging waypoint the base drives to before
        # place_approach (a direct diagonal clips the cart). Optional + plumbed when supplied.
        cfg = base_config()
        cfg['mission_poses']['place_approach'] = [-4.5, 15.25, math.pi]
        cfg['mission_poses']['place_approach_stage'] = [-4.5, 16.0, 0.0]
        p = plan_bundle(MANIFEST, SCENARIO, cfg)
        self.assertEqual(p.poses['place_approach_stage'], (-4.5, 16.0, 0.0))
        self.assertNotIn('place_approach_stage', plan_bundle(MANIFEST, SCENARIO, base_config()).poses)

    def test_derive_place_approach_stage_translates_east_at_pick_heading(self):
        # The staging pose sits at place_approach's x but the pick pose's lateral line + heading, so leg 1
        # is a straight translate out of the cart's north gap before the turn onto the mouth.
        pickup = (-5.73, 16.0, 0.0)
        place_approach = isaac_bundle.derive_place_approach((-5.73, 15.25), 0.0, 1.15)
        stage = isaac_bundle.derive_place_approach_stage(place_approach, pickup)
        self.assertAlmostEqual(stage[0], place_approach[0], places=6)  # place_approach's standoff x
        self.assertAlmostEqual(stage[1], 16.0, places=6)               # pick pose's y (clear of the cart)
        self.assertAlmostEqual(stage[2], 0.0, places=6)                # pick heading (straight translate)

    def test_derive_place_approach_from_mission_cart(self):
        # Mission cart /Rollwagen at (-5.73,15.25) yaw 0, open +x face; proven standoff ~1.23 m east.
        x, y, yaw = isaac_bundle.derive_place_approach((-5.73, 15.25), 0.0, 1.23)
        self.assertAlmostEqual(x, -4.50, places=2)
        self.assertAlmostEqual(y, 15.25, places=2)
        self.assertAlmostEqual(math.cos(yaw), -1.0, places=6)  # faces the mouth (-x); +-pi both valid
        self.assertAlmostEqual(math.sin(yaw), 0.0, places=6)
        # a cart yawed -90deg rotates the mouth axis with it (derived, not pinned)
        x2, y2, yaw2 = isaac_bundle.derive_place_approach((-5.73, 15.25), -math.pi / 2, 1.23)
        self.assertAlmostEqual(x2, -5.73, places=2)
        self.assertAlmostEqual(y2, 15.25 - 1.23, places=2)   # mouth now points -y
        self.assertAlmostEqual(math.cos(yaw2), 0.0, places=6)  # faces +y back toward the mouth
        self.assertAlmostEqual(math.sin(yaw2), 1.0, places=6)

    def test_articulation_bounds(self):
        with self.assertRaises(ValueError):
            plan_bundle(MANIFEST, SCENARIO, base_config(goal_articulation={'outbound': math.radians(80)}))
        with self.assertRaises(ValueError):
            plan_bundle(MANIFEST, SCENARIO, base_config(goal_articulation={'outbound': 0.0}))
        with self.assertRaises(ValueError):
            plan_bundle(MANIFEST, SCENARIO, base_config(goal_articulation={'park': 0.1}))

    def test_bad_nav_mode(self):
        with self.assertRaises(ValueError):
            plan_bundle(MANIFEST, SCENARIO, base_config(nav_mode='teleport'))

    def test_bad_fps_and_roots(self):
        with self.assertRaises(ValueError):
            plan_bundle(MANIFEST, SCENARIO, base_config(recording_fps=0))
        with self.assertRaises(ValueError):
            plan_bundle(MANIFEST, SCENARIO, base_config(package_roots=[]))


class TransportPositionsTests(unittest.TestCase):
    def test_valid_mapping_returned(self):
        positions = {'shoulder_pan_joint': 0.0, 'elbow_joint': -1.57}
        self.assertEqual(isaac_bundle.require_transport_positions(
            {'transport_positions': positions}), positions)

    def test_missing_or_empty_refused_clearly(self):
        for manifest in ({}, {'transport_positions': {}}, {'transport_positions': None},
                         {'transport_positions': []}, 'not-a-dict'):
            with self.assertRaisesRegex(ValueError, 'transport_positions'):
                isaac_bundle.require_transport_positions(manifest)

    def test_nonfinite_or_boolean_joint_value_refused(self):
        for bad in (float('nan'), float('inf'), True, 'x'):
            with self.assertRaisesRegex(ValueError, 'transport_positions'):
                isaac_bundle.require_transport_positions({'transport_positions': {'elbow_joint': bad}})


# ---- fixtures for the create() wiring seam (test-only, not physical evidence) ----
class _Ctrl:
    cart_location = 'rollwagen'
    def start(self, *a, **k): pass
    def step(self, *a, **k): pass
    def cancel(self): pass
    def stop(self): pass


class _Mobile:
    def park(self): pass


class _Port:
    def advance(self, dt): raise AssertionError('fixture port not stepped in contract test')
    def controls(self, c): pass
    def stop(self): pass


class CreateWiringTests(unittest.TestCase):
    def _fixture_runtime(self, flags):
        def build(*, app, assembly, plan, config, output, scenario=None, manifest=None):
            flags['built'] = True
            flags['plan'] = plan
            return {'navigation': NavigationBridge(_Ctrl(), sensor_navigation=plan.nav_mode == 'sensor'),
                    'mobile': _Mobile(), 'manipulation': _Ctrl(), 'port': _Port(),
                    'closers': (lambda: flags.__setitem__('closed', True),),
                    'recording': {'fps': plan.recording_fps}}
        return build

    def test_create_builds_production_adapter(self):
        flags = {}
        bundle = create(app=object(), manifest=MANIFEST, scenario=SCENARIO, config=base_config(),
                        output='/tmp/does-not-matter',
                        _assemble=lambda **k: object(),
                        _runtime=self._fixture_runtime(flags))
        self.assertTrue(flags['built'])
        self.assertIs(bundle.adapter.fixture_mode, False)  # launcher requires non-fixture
        self.assertEqual(bundle.adapter.poses['home'], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(bundle.adapter.goal_articulation['return'], math.radians(10))
        self.assertEqual(bundle.recording['fps'], 30)
        bundle.close()
        self.assertTrue(flags.get('closed'))

    def test_create_rejects_bad_config_before_isaac(self):
        cfg = base_config()
        del cfg['mission_poses']['home']
        called = {'assembled': False}
        with self.assertRaises(ValueError):
            create(app=object(), manifest=MANIFEST, scenario=SCENARIO, config=cfg, output='/tmp/x',
                   _assemble=lambda **k: called.__setitem__('assembled', True),
                   _runtime=self._fixture_runtime({}))
        self.assertFalse(called['assembled'])  # plan validated before any assembly

    def test_resolve_factory_requires_module_function(self):
        with self.assertRaises(ValueError):
            isaac_bundle._resolve_factory({'nav_factory': 'not_a_ref'}, 'nav_factory')
        with self.assertRaises(ValueError):
            isaac_bundle._resolve_factory({}, 'nav_factory')


if __name__ == '__main__':
    unittest.main()
