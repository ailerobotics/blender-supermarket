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
        # run-285: ~2.2 GB whispr + others left < 9.5 GB free -> the guard would have refused.
        self.assertFalse(vg.decide(9000)[0])


if __name__ == '__main__':
    unittest.main()
