"""fm386: close the SIM-clock nav bound family (sibling of the fm366 wall sweep + fm385 sensor watchdog).

The couple/outbound/park nav (SensorNavigation/NavigationController) has SIM-clock bounds -- the per-frame
stale_after (now-stamp staleness), blocked_timeout (no-progress) and goal_timeout -- that were hardcoded
ctor defaults. On a render-paced tow/dock the odom/obs stamp can lag past the tight 0.3 s stale_after and
false-trip 'stale_sensing' (exactly the shape of the 1.0 s /sensors watchdog that tripped at retreat).
These assert: byte-identical when unset, env WINS over a config value (the run-290 shadow lesson),
malformed env is ignored, and a GUARD that fails BY NAME if any listed nav sim bound stops being
env-overridable. CPU-only; no Isaac."""
import os
import unittest

from twin.navigation import factory


class NavSimBoundEnvTests(unittest.TestCase):
    def setUp(self):
        for var, _lo, _hi in factory._NAV_SIM_BOUNDS.values():
            os.environ.pop(var, None)

    tearDown = setUp

    def test_byte_identical_off_adds_nothing(self):
        self.assertEqual(factory._controller_kwargs({}), {})

    def test_config_value_preserved_when_env_unset(self):
        self.assertEqual(factory._controller_kwargs({'stale_after': 0.3})['stale_after'], 0.3)
        self.assertEqual(factory._controller_kwargs({'goal_timeout': 180.0})['goal_timeout'], 180.0)

    def test_env_wins_over_config(self):
        os.environ['RESTOCK_NAV_STALE_AFTER_S'] = '5'
        os.environ['RESTOCK_NAV_BLOCKED_TIMEOUT_S'] = '60'
        os.environ['RESTOCK_NAV_GOAL_TIMEOUT_S'] = '600'
        k = factory._controller_kwargs({'stale_after': 0.3, 'blocked_timeout': 8.0, 'goal_timeout': 180.0})
        self.assertEqual((k['stale_after'], k['blocked_timeout'], k['goal_timeout']), (5.0, 60.0, 600.0))

    def test_env_clamped(self):
        os.environ['RESTOCK_NAV_STALE_AFTER_S'] = '999'   # above hi=30
        self.assertEqual(factory._controller_kwargs({})['stale_after'], 30.0)

    def test_malformed_env_ignored(self):
        os.environ['RESTOCK_NAV_STALE_AFTER_S'] = 'abc'
        self.assertNotIn('stale_after', factory._controller_kwargs({}))

    def test_guard_every_nav_sim_bound_is_env_overridable(self):
        # Fails BY NAME if a nav sim bound is dropped from the env-overridable set.
        expected = {'stale_after', 'blocked_timeout', 'goal_timeout'}
        self.assertEqual(set(factory._NAV_SIM_BOUNDS), expected,
                         'a nav SIM-clock bound is no longer env-overridable: %s'
                         % (expected ^ set(factory._NAV_SIM_BOUNDS)))
        for limit, (var, _lo, _hi) in factory._NAV_SIM_BOUNDS.items():
            probe = 12.0   # within all three [lo,hi] ranges (stale<=30, blocked>=1, goal>=10)
            os.environ[var] = str(probe)
            self.assertEqual(factory._controller_kwargs({}).get(limit), probe,
                             'nav sim bound %r (%s) not honoured from env' % (limit, var))
            os.environ.pop(var, None)


if __name__ == '__main__':
    unittest.main()
