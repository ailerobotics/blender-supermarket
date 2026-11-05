"""CPU tests for the batch-key activation preflight (config_preflight)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from twin.restock_manipulation import config_preflight as cp  # noqa: E402

CAGE_KEYS = ['cart_cage_place', 'cage_preplace_shape', 'cage_axis',
             'cage_mouth_standoff', 'cage_insert_gap']
# fm402: the cap/limit baseline the preflight ALWAYS enforces (unioned into every run's --require).
BASELINE = {k: True for k in cp.BASELINE_REQUIRED}


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
        self._write({**BASELINE, **{k: True for k in CAGE_KEYS}}, {})
        rc = cp.main(['--host-config', self.host_path, '--runtime-config',
                      self.runtime_path, '--require', ','.join(CAGE_KEYS)])
        self.assertEqual(rc, 0)

    def test_main_fails_on_run277_slip(self):
        self._write(dict(BASELINE), {k: True for k in CAGE_KEYS})
        rc = cp.main(['--host-config', self.host_path, '--runtime-config',
                      self.runtime_path, '--require', ','.join(CAGE_KEYS)])
        self.assertEqual(rc, 2)

    def test_main_enforces_cap_baseline_even_without_require(self):
        # fm402: with no --require the cap/limit BASELINE is still asserted, so an absent cap fails the
        # launch loudly (never a silent un-cap). Empty host config => the baseline caps are missing.
        self._write({}, {})
        rc = cp.main(['--host-config', self.host_path, '--runtime-config', self.runtime_path])
        self.assertEqual(rc, 2)

    def test_main_passes_baseline_when_caps_present(self):
        # fm402: the baseline passes once every cap/limit key is present in the host config.
        self._write(dict(BASELINE), {})
        rc = cp.main(['--host-config', self.host_path, '--runtime-config', self.runtime_path])
        self.assertEqual(rc, 0)

    def test_absent_carry_wrist1_cap_fails_loudly(self):
        # fm402: the specific carry_wrist1_cap slip (runs 300d/305/311) must be caught by the baseline.
        host = {k: True for k in cp.BASELINE_REQUIRED if k != 'carry_wrist1_cap'}
        self._write(host, {})
        rc = cp.main(['--host-config', self.host_path, '--runtime-config', self.runtime_path])
        self.assertEqual(rc, 2)

    def test_committed_host_config_has_cage_and_cap_keys(self):
        # Guards against a regression of the run-277 placement AND the carry_wrist1_cap omission on the
        # real artifact (the runs 300d/305/311 silent un-cap).
        here = os.path.dirname(__file__)
        host_cfg = json.load(open(os.path.join(
            here, '..', 'restock_manipulation', 'ros_host_config.json')))
        self.assertEqual(cp.check_activation(host_cfg, {}, CAGE_KEYS + list(cp.BASELINE_REQUIRED)), [])
        self.assertIn('carry_wrist1_cap', host_cfg)


class FlowConsistencyTests(unittest.TestCase):
    # sim-service-carry-drive-to-shelf-v1: the cross-LAYER shelf_staging/transfer_service_carry split.

    def test_split_shelf_staging_on_carry_off_is_refused(self):
        # transferobs1: runtime shelf_staging ON, host transfer_service_carry OFF/absent => the service
        # carry silently does nothing (arm never folds to TRANSIT, base never carries) and the controller
        # emergency-stops on the far endcap reach. Named + refused at plan time.
        problems = cp.check_flow_consistency({}, {'shelf_staging': True})
        self.assertEqual(len(problems), 1)
        self.assertIn('SPLIT', problems[0])
        self.assertIn('transfer_service_carry', problems[0])
        self.assertIn('shelf_approach', problems[0])

    def test_both_on_passes(self):
        self.assertEqual(
            cp.check_flow_consistency({'transfer_service_carry': True}, {'shelf_staging': True}), [])

    def test_both_off_passes(self):
        self.assertEqual(cp.check_flow_consistency({}, {}), [])

    def test_carry_on_without_staging_is_not_refused(self):
        # One-directional on purpose: carry ON + staging OFF is inert / bounded (TRANSIT phase-times-out
        # by name), and refusing it would false-block every non-staging run that shares the repo HOST
        # config now that transfer_service_carry defaults true there (collect-only, legacy transfer).
        self.assertEqual(
            cp.check_flow_consistency({'transfer_service_carry': True}, {'shelf_staging': False}), [])

    def test_committed_host_config_pairs_with_shelf_staging(self):
        # The repo host default has transfer_service_carry ON, so a shelf_staging runtime pairs cleanly.
        here = os.path.dirname(__file__)
        host_cfg = json.load(open(os.path.join(
            here, '..', 'restock_manipulation', 'ros_host_config.json')))
        self.assertTrue(host_cfg.get('transfer_service_carry'))
        self.assertEqual(cp.check_flow_consistency(host_cfg, {'shelf_staging': True}), [])


class PhaseDriveCarriedSoftTests(unittest.TestCase):
    # sim-preplace-sweep-tilt-v1: validate phase_drive.carried_soft_phases (run carried preplace SOFT).
    SOFT = {'kp': 3e5, 'kd': 8e3, 'eff': 400.0}
    STIFF = {'kp': 1e6, 'kd': 1e4, 'eff': 2000.0}

    def _pd(self, **extra):
        return {'soft': self.SOFT, 'stiff': self.STIFF, **extra}

    def test_absent_override_is_noop(self):
        self.assertEqual(cp.check_phase_drive_carried_soft({}), [])
        self.assertEqual(cp.check_phase_drive_carried_soft({'phase_drive': self._pd()}), [])

    def test_valid_override_passes(self):
        self.assertEqual(
            cp.check_phase_drive_carried_soft({'phase_drive': self._pd(carried_soft_phases=['preplace'])}),
            [])

    def test_non_stiff_phase_rejected(self):
        problems = cp.check_phase_drive_carried_soft(
            {'phase_drive': self._pd(carried_soft_phases=['lift'])})
        self.assertTrue(problems and 'carried_soft_phases' in problems[0])

    def test_non_list_rejected(self):
        problems = cp.check_phase_drive_carried_soft(
            {'phase_drive': self._pd(carried_soft_phases='preplace')})
        self.assertTrue(problems)

    def test_missing_soft_block_rejected(self):
        problems = cp.check_phase_drive_carried_soft(
            {'phase_drive': {'stiff': self.STIFF, 'carried_soft_phases': ['preplace']}})
        self.assertTrue(any('no "soft"' in p for p in problems))


if __name__ == '__main__':
    unittest.main()
