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
        self.assertEqual(plan_bundle(MANIFEST, {'navigation_model': 'reference'},
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
        def build(*, app, assembly, plan, config, output):
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
