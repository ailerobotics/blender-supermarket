"""CPU tests for the fm360 VRAM pre-launch guard (vram_guard)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from twin.tools import vram_guard as vg  # noqa: E402


class ParseTests(unittest.TestCase):
    def test_parse_typical_row(self):
        self.assertEqual(vg.parse_smi('9800, 2400, 12288\n'), (9800, 2400, 12288))

    def test_parse_first_gpu_only(self):
        self.assertEqual(vg.parse_smi('5000, 7000, 12288\n3000, 9000, 12288\n')[0], 5000)

    def test_parse_rejects_malformed(self):
        with self.assertRaises(ValueError):
            vg.parse_smi('')
        with self.assertRaises(ValueError):
            vg.parse_smi('n/a, x, y')


class DecideTests(unittest.TestCase):
    def test_enough_free_passes(self):
        ok, _ = vg.decide(9800, 9500)
        self.assertTrue(ok)

    def test_too_little_free_refuses(self):
        ok, msg = vg.decide(8000, 9500)
        self.assertFalse(ok)
        self.assertIn('REFUSE', msg)

    def test_boundary_is_inclusive(self):
        self.assertTrue(vg.decide(9500, 9500)[0])

    def test_run285_condition_would_refuse(self):
        # run-285: ~2.2 GB whispr + others left < the derived minimum -> the guard refuses.
        self.assertFalse(vg.decide(9000)[0])

    def test_refusal_names_holders_and_derivation(self):
        _, bd = vg.required_min_free_mib(None)
        ok, msg = vg.decide(8000, bd['min_free_mib'], breakdown=bd,
                            holders='whispr (pid 1339, 2498 MiB)')
        self.assertFalse(ok)
        self.assertIn('REFUSE', msg)
        self.assertIn('whispr', msg)                 # names what holds the memory
        self.assertIn('mission peak', msg)           # states how the minimum was derived
        self.assertIn('NEVER whispr', msg)           # never touch the protected service


class DeriveTests(unittest.TestCase):
    def test_default_minimum_is_derived_not_a_magic_constant(self):
        # ~= mission 9630 - teardown 948 + margin 400 = 9082, NOT the old fixed 9500.
        self.assertEqual(vg.DEFAULT_MIN_FREE_MIB, 9082)
        mn, bd = vg.required_min_free_mib(None)
        self.assertEqual(mn, vg.DEFAULT_MIN_FREE_MIB)
        self.assertTrue(bd['stage_scoped_teardown'])
        self.assertEqual(bd['teardown_saving_mib'], vg.STAGE_SCOPED_TEARDOWN_SAVING_MIB)

    def test_honest_minimum_lets_a_run_boot_beside_244gb_whispr(self):
        # The brief's failure: free 9473 beside a 2.44 GB whispr was refused by the fixed 9500.
        # The derived minimum (default-on trims) is below that, so the same card now boots.
        mn, _ = vg.required_min_free_mib(None)
        self.assertLess(mn, 9473)
        self.assertTrue(vg.decide(9473, mn)[0])

    def test_opting_out_the_teardown_raises_the_minimum(self):
        # Without the trim the mission needs its full footprint free -> a higher (honest) minimum.
        on, _ = vg.required_min_free_mib({'sensing_teardown': True})
        off, bd = vg.required_min_free_mib({'sensing_teardown': False})
        self.assertGreater(off, on)
        self.assertFalse(bd['stage_scoped_teardown'])

    def test_env_forces_the_teardown_state(self):
        off, _ = vg.required_min_free_mib(None, teardown_env='0')
        on, _ = vg.required_min_free_mib(None, teardown_env='1')
        self.assertGreater(off, on)

    def test_extra_feed_raises_the_minimum(self):
        one, _ = vg.required_min_free_mib({'masked_depth': {'enabled': True}})
        two, _ = vg.required_min_free_mib({'masked_depth': {'enabled': True},
                                           'mast_depth': {'enabled': True}})
        self.assertGreater(two, one)

    def test_teardown_reader_parity_with_sensing_sources(self):
        # The guard's inlined reader must track sensing_sources (the authoritative copy).
        from twin.restock.sensing_sources import stage_scoped_teardown_enabled as auth
        for cfg in ({}, {'sensing_teardown': False}, {'sensing_teardown': True},
                    {'sensing_teardown': {'stage_scoped': False}}, {'sensing_teardown': {}}):
            self.assertEqual(vg._stage_scoped_teardown_enabled(cfg, None), auth(cfg), cfg)

    def test_holders_text_names_top_consumer(self):
        smi = (
            '+---------------------------------------------------------------------------------------+\n'
            '| Processes:                                                                            |\n'
            '|  GPU   GI   CI        PID   Type   Process name                           GPU Memory  |\n'
            '|        ID   ID                                                            Usage       |\n'
            '|=======================================================================================|\n'
            '|    0   N/A  N/A      1339      C   whispr                                  2498MiB    |\n'
            '+---------------------------------------------------------------------------------------+\n')
        self.assertIn('whispr', vg.holders_text(smi))
        self.assertEqual(vg.holders_text('no table here'), '')


if __name__ == '__main__':
    unittest.main()
