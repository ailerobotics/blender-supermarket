"""CPU tests for the batch-key activation preflight (config_preflight)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from twin.restock_manipulation import config_preflight as cp  # noqa: E402

CAGE_KEYS = ['cart_cage_place', 'world_obstacles', 'cage_axis',
             'cage_mouth_standoff', 'cage_insert_gap']


class CheckActivationTests(unittest.TestCase):
    def test_present_in_correct_file_passes(self):
        host = {k: True for k in CAGE_KEYS}
        runtime = {'phase_drive': True}
        self.assertEqual(cp.check_activation(host, runtime, CAGE_KEYS + ['phase_drive']), [])

    def test_run277_slip_detected(self):
        # The exact run-277 bug: cage keys authored into the RUNTIME config, absent from
        # the HOST config their consumer reads.
        host = {}
        runtime = {k: True for k in CAGE_KEYS}
        problems = cp.check_activation(host, runtime, CAGE_KEYS)
        self.assertEqual(len(problems), len(CAGE_KEYS))
        self.assertTrue(all('MIS-PLACED' in p for p in problems))
        self.assertTrue(all('run-277' in p for p in problems))

    def test_missing_everywhere_detected(self):
        problems = cp.check_activation({}, {}, ['cart_cage_place'])
        self.assertEqual(len(problems), 1)
        self.assertIn('MISSING', problems[0])

    def test_unregistered_key_detected(self):
        problems = cp.check_activation({'made_up': 1}, {}, ['made_up'])
        self.assertEqual(len(problems), 1)
        self.assertIn('unregistered', problems[0])

    def test_runtime_key_in_runtime_passes(self):
        self.assertEqual(cp.check_activation({}, {'phase_drive': True}, ['phase_drive']), [])

    def test_runtime_key_misplaced_in_host_detected(self):
        problems = cp.check_activation({'phase_drive': True}, {}, ['phase_drive'])
        self.assertEqual(len(problems), 1)
        self.assertIn('MIS-PLACED', problems[0])


class MainCliTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.host_path = os.path.join(self.tmp, 'ros_host_config.json')
        self.runtime_path = os.path.join(self.tmp, 'runtime.json')

    def _write(self, host, runtime):
        json.dump(host, open(self.host_path, 'w'))
        json.dump(runtime, open(self.runtime_path, 'w'))

    def test_main_passes_when_correct(self):
        self._write({k: True for k in CAGE_KEYS}, {})
        rc = cp.main(['--host-config', self.host_path, '--runtime-config',
                      self.runtime_path, '--require', ','.join(CAGE_KEYS)])
        self.assertEqual(rc, 0)

    def test_main_fails_on_run277_slip(self):
        self._write({}, {k: True for k in CAGE_KEYS})
        rc = cp.main(['--host-config', self.host_path, '--runtime-config',
                      self.runtime_path, '--require', ','.join(CAGE_KEYS)])
        self.assertEqual(rc, 2)

    def test_main_noop_without_require(self):
        self._write({}, {})
        rc = cp.main(['--host-config', self.host_path, '--runtime-config',
                      self.runtime_path])
        self.assertEqual(rc, 0)

    def test_committed_host_config_has_cage_keys(self):
        # Guards against a regression of the run-277 placement on the real artifact.
        here = os.path.dirname(__file__)
        host_cfg = json.load(open(os.path.join(
            here, '..', 'restock_manipulation', 'ros_host_config.json')))
        self.assertEqual(cp.check_activation(host_cfg, {}, CAGE_KEYS), [])


if __name__ == '__main__':
    unittest.main()
