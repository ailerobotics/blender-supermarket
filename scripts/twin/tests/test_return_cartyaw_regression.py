"""CPU regression for the coupled RETURN cart-yaw stop_replan trap (sim-video-return-workaround-v1).

warm3's live RETURN stalled: the authored tow lane ``return_bay_to_parking`` (``start_cart_yaw`` 0) was
followed from an end-of-verify train whose towed cart lags at articulation ~0.2, so the follower's
tracking law saw ``cart_err`` ~0.23 > ``tow_lane_cart_yaw_tol`` (default 0.20) and dropped its own
validated route every tick (``stop_replan_tracking_error``), never moving. The fix is the mission nav
key ``tow_lane_cart_yaw_rebase`` (re-base the expected cart_yaw to the measured aligned entry) -- a key
that was once silently dropped in a fork, which this test exists to catch.

Unlike ``scripts/check_mission_tail.py`` (which STUBS the coupled return kinematically and so never
exercised the tracking law), this drives the REAL coupled ``ArticulatedController`` built through
``nav_factory`` from the recorded end-of-verify pose, reading the mission's actual ``navigation.*`` keys
verbatim. The fixture ``fixtures/return_cartyaw_nav.json`` mirrors the mission runtime-config nav block
(keep in sync) with a minimal embedded open grid so the test needs no ~/datasets inputs.
"""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.tail_nav import simulate_coupled_return   # noqa: E402

FIXTURE = json.loads((Path(__file__).resolve().parent / 'fixtures' / 'return_cartyaw_nav.json').read_text())
ENV = FIXTURE['navigation']['environment']
DEFAULT_TOL = 0.20   # the ArticulatedController / factory default; the value at which the trap bit warm3
_DROP = object()     # sentinel: remove the key entirely (simulate a fork that dropped it)


def _drive(nav_overrides):
    cfg = copy.deepcopy(FIXTURE)
    nav = cfg['navigation']
    for k, v in nav_overrides.items():
        if v is _DROP:
            nav.pop(k, None)
        else:
            nav[k] = v
    return simulate_coupled_return(cfg, None, ENV, max_s=20.0)


class ReturnCartYawRegression(unittest.TestCase):
    def test_fixed_config_coupled_return_arrives(self):
        """The mission's actual (fixed) nav keys drive the coupled return to arrival from the recorded pose."""
        r = _drive({})
        self.assertEqual(r['lane'], 'return_bay_to_parking', r)
        self.assertTrue(r['arrived'], msg=f'fixed config must reach the cart-parking goal: {r}')
        self.assertEqual(r['reason'], 'arrived_stopped', r)
        # the fixture must actually carry the load-bearing fix key (a dropped key here fails the test)
        self.assertIs(FIXTURE['navigation'].get('tow_lane_cart_yaw_rebase'), True,
                      'fixture lost tow_lane_cart_yaw_rebase=true (keep in sync with the mission)')

    def test_dropped_rebase_key_reproduces_warm3_stall(self):
        """Dropping tow_lane_cart_yaw_rebase at the default cart-yaw tol reproduces the warm3 stall: the
        follower drops its validated route on cart_err and never arrives."""
        r = _drive({'tow_lane_cart_yaw_rebase': _DROP, 'tow_lane_cart_yaw_tol': DEFAULT_TOL})
        self.assertFalse(r['arrived'], msg=f'without rebase at default tol the return must stall: {r}')
        self.assertEqual(r['moved'], 0.0, r)
        self.assertIn('stop_replan_tracking_error', r['reasons'],
                      msg=f'the stall must be the cart-yaw stop_replan trap, not another refusal: {r}')

    def test_rebase_alone_recovers_at_default_tol(self):
        """tow_lane_cart_yaw_rebase=true alone (even at the historical default tol) recovers the return,
        isolating it as the load-bearing fix."""
        r = _drive({'tow_lane_cart_yaw_rebase': True, 'tow_lane_cart_yaw_tol': DEFAULT_TOL})
        self.assertTrue(r['arrived'], msg=f'rebase=true must recover the return at the default tol: {r}')
        self.assertEqual(r['reason'], 'arrived_stopped', r)


if __name__ == '__main__':
    unittest.main()
