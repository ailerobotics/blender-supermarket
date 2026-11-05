"""CPU tests for sim-preflight-nested-cart-placement-v1.

The Isaac factory reads the collect placement from the NESTED block
``config['manipulation']['cart_placement']``; a TOP-LEVEL ``config['cart_placement']`` is
IGNORED. coldwin42-47 edited the top-level block, so five GPU runs ran on settings the sim
never read, and the pre-flight passed because it too validated the ignored top-level block.

These tests prove: the factory and the pre-flight resolve the placement through ONE shared
accessor (``config_access``); the pre-flight refuses a stale top-level duplicate, naming both
blocks; it passes on the coldwin48 shape (nested set correctly) and fails on the coldwin47
shape (top-level block, nested stale).
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from twin.restock_manipulation import config_access as ca  # noqa: E402
from twin.restock_manipulation import config_preflight as cp  # noqa: E402
from twin.restock_manipulation.isaac_factory import parse_manipulation_config  # noqa: E402

BASELINE = {k: 0.0 for k in cp.BASELINE_REQUIRED}

# The CORRECT (coldwin48) nested placement and the STALE (coldwin47) nested placement.
SHELF = {'location': 'rollwagen', 'support_prim': 'shelf',
         'load_center_body_xy': [0.135, 0.0], 'support_z_world': 0.9712,
         'free_dimensions': [0.65, 0.45, 0.30]}
STALE_DECK = {'location': 'rollwagen', 'load_center_body_xy': [0.08, 0.0],
              'support_z_world': 0.9325, 'free_dimensions': [0.65, 0.45, 0.30]}


def _manip(cart_placement):
    """A minimal-but-valid config.manipulation block carrying a cart_placement."""
    return {
        'profiles': [{'sku': 'juice_orange', 'dimensions': [.45, .30, .18],
                      'tcp_quaternion': [.5, .5, .5, .5], 'grasp_mode': 'side'}],
        'stations': {
            'source': {'slots': [{'id': 's0'}],
                       'base_from_camera': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
            'shelf': {'slots': [{'id': 'd0'}],
                      'base_from_camera': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}},
        'shelfbot_root': '/home/x/shelfbot_newgripper', 'walls': {},
        'voxel_dims': [2.4, 2.4, 2.4], 'voxel_size': .02,
        'cart_placement': cart_placement}


def _coldwin47():
    """Top-level block set (shelf), nested block STALE (deck) -- the slip the factory ran on."""
    return {'manipulation': _manip(STALE_DECK), 'cart_placement': SHELF}


def _coldwin48():
    """Nested block set correctly (shelf); the top-level duplicate (if any) is identical."""
    return {'manipulation': _manip(SHELF), 'cart_placement': dict(SHELF)}


class ConfigAccessTests(unittest.TestCase):
    def test_accessor_reads_nested_not_top_level(self):
        cfg = {'manipulation': {'cart_placement': STALE_DECK}, 'cart_placement': SHELF}
        self.assertEqual(ca.cart_placement_block(cfg), STALE_DECK)
        self.assertEqual(ca.top_level_cart_placement(cfg), SHELF)
        self.assertEqual(ca.CART_PLACEMENT_SOURCE, 'manipulation.cart_placement')

    def test_accessor_empty_when_absent(self):
        self.assertIsNone(ca.cart_placement_block({}))
        self.assertIsNone(ca.top_level_cart_placement({}))
        self.assertEqual(ca.manipulation_block({}), {})


class FactoryAndPreflightAgreeTests(unittest.TestCase):
    def test_factory_reads_the_nested_block_the_accessor_names(self):
        # The factory's parsed cart_placement must come from the NESTED block (coldwin47: the stale deck),
        # NOT the valid top-level shelf -- exactly the bug. The pre-flight reads the same block.
        cfg = _coldwin47()
        parsed = parse_manipulation_config(cfg, 'juice_orange')
        # factory took the nested (stale) values, not the top-level shelf
        self.assertNotIn('support_prim', parsed.cart_placement)   # STALE_DECK has none
        self.assertEqual(parsed.cart_placement['support_z_world'], 0.9325)
        self.assertEqual(parsed.cart_placement['load_center_body_xy'], (0.08, 0.0))
        # the pre-flight accessor resolves the SAME block the factory parsed
        self.assertEqual(ca.cart_placement_block(cfg), STALE_DECK)

    def test_factory_and_accessor_match_on_the_correct_config(self):
        cfg = _coldwin48()
        parsed = parse_manipulation_config(cfg, 'juice_orange')
        self.assertEqual(parsed.cart_placement['support_prim'], 'shelf')
        self.assertEqual(parsed.cart_placement['support_z_world'], 0.9712)
        self.assertEqual(ca.cart_placement_block(cfg), SHELF)


class CheckCartPlacementSourceTests(unittest.TestCase):
    def test_stale_divergent_top_level_is_refused_naming_both_blocks(self):
        probs = cp.check_cart_placement_source(_coldwin47())
        self.assertEqual(len(probs), 1)
        msg = probs[0]
        self.assertIn('manipulation.cart_placement', msg)         # the block that is READ
        self.assertIn('top-level', msg)                           # the block that is IGNORED
        self.assertIn('IGNORED', msg)
        self.assertIn('coldwin', msg)

    def test_identical_top_level_duplicate_passes(self):
        # coldwin48 carries an identical top-level duplicate -- harmless, the factory reads the nested
        # block which equals it.
        self.assertEqual(cp.check_cart_placement_source(_coldwin48()), [])

    def test_nested_only_passes(self):
        # The correct shape: no top-level block at all.
        self.assertEqual(cp.check_cart_placement_source({'manipulation': _manip(SHELF)}), [])

    def test_orphan_top_level_with_no_nested_is_refused(self):
        # Only the top-level block authored (the pure coldwin42-45 case): the factory reads an ABSENT
        # nested block, so the whole placement is ignored.
        probs = cp.check_cart_placement_source({'cart_placement': SHELF})
        self.assertEqual(len(probs), 1)
        self.assertIn('ABSENT', probs[0])

    def test_no_cart_placement_anywhere_is_byte_identical(self):
        self.assertEqual(cp.check_cart_placement_source({}), [])


class MainCliTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.host_path = os.path.join(self.tmp, 'ros_host_config.json')
        self.runtime_path = os.path.join(self.tmp, 'runtime.json')
        json.dump(BASELINE, open(self.host_path, 'w'))

    def _run(self, runtime):
        json.dump(runtime, open(self.runtime_path, 'w'))
        return cp.main(['--host-config', self.host_path, '--runtime-config', self.runtime_path])

    def test_main_passes_on_coldwin48_nested_config(self):
        self.assertEqual(self._run(_coldwin48()), 0)

    def test_main_fails_on_coldwin47_config(self):
        self.assertEqual(self._run(_coldwin47()), 2)


if __name__ == '__main__':
    unittest.main()
