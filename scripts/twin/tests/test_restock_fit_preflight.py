"""CPU tests for the opt-in tray/slot fit preflight.

No Isaac/ROS/Blender imports. Covers the shared axiswise predicate (parity with the
place-phase gate), the fit verdict's three honest cases, the record loaders
(mission depth capture + store-vision twin3d export with an explicit axis order),
and the default-off mission gate (a 'wont_fit' refuses before any reservation; an
absent verdict is byte-identical).
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.fit_preflight import (
    HandlingUnit, SlotFreeSpace, evaluate_fit, free_space_from_depth_capture,
    free_space_from_twin3d, handling_unit_from_scenario, load_free_space,
    preflight_from_config)
from twin.restock.inventory import Inventory, Job
from twin.restock.mission import ACTIONS, Mission
import twin.restock_manipulation.controller as controller_mod
from twin.restock_manipulation.fit import axis_fits

UNIT = HandlingUnit('juice_orange', (0.6, 0.4, 0.16))
SCENARIO = {'job': {'sku': 'juice_orange'},
            'handling_unit': {'sku': 'juice_orange', 'dimensions_m': [0.6, 0.4, 0.16]}}


class AxisFitsTests(unittest.TestCase):
    def test_fits_with_margin(self):
        self.assertTrue(axis_fits((0.63, 0.43, 0.19), (0.6, 0.4, 0.16)))

    def test_margin_is_required_on_every_axis(self):
        # The gate requires free >= dim + margin (same float arithmetic the place gate uses);
        # a free dimension short of the margin on any single axis fails.
        self.assertTrue(axis_fits((0.625, 0.425, 0.185), (0.6, 0.4, 0.16), margin=0.02))
        self.assertFalse(axis_fits((0.61, 0.425, 0.185), (0.6, 0.4, 0.16), margin=0.02))  # x < +margin

    def test_undersize_any_axis_fails(self):
        self.assertFalse(axis_fits((0.6, 0.43, 0.19), (0.6, 0.4, 0.16)))  # x short of margin

    def test_nonfinite_and_bad_length_fail(self):
        self.assertFalse(axis_fits((math.inf, 0.43, 0.19), (0.6, 0.4, 0.16)))
        self.assertFalse(axis_fits((0.63, 0.43), (0.6, 0.4, 0.16)))

    def test_matches_place_gate_semantics(self):
        # The place-phase controller gate uses the SAME shared predicate, so the preflight's
        # 'ok'/'wont_fit' agree with what the gate accepts/rejects for the same free/unit dims.
        self.assertIs(controller_mod.axis_fits, axis_fits)
        for free, expect_fit in [((0.63, 0.43, 0.19), True), ((0.6, 0.43, 0.19), False)]:
            self.assertEqual(axis_fits(free, UNIT.dimensions_m), expect_fit)
            self.assertEqual(evaluate_fit(UNIT, SlotFreeSpace(free)).status,
                             'ok' if expect_fit else 'wont_fit')


class EvaluateFitTests(unittest.TestCase):
    def test_ok(self):
        v = evaluate_fit(UNIT, SlotFreeSpace((0.63, 0.43, 0.19), slot_id='G3_back_s2'))
        self.assertEqual(v.status, 'ok')
        self.assertTrue(v)
        self.assertFalse(v.low_confidence)
        self.assertEqual(v.slot_id, 'G3_back_s2')

    def test_wont_fit_undersize_names_axis(self):
        v = evaluate_fit(UNIT, SlotFreeSpace((0.6, 0.42, 0.18)))
        self.assertEqual(v.status, 'wont_fit')
        self.assertFalse(v)
        self.assertTrue(any('on x' in r for r in v.reasons))
        self.assertIn('tray exceeds slot free dimensions', v.message)

    def test_wont_fit_blocked_or_occupied(self):
        self.assertEqual(evaluate_fit(UNIT, SlotFreeSpace((0.7, 0.5, 0.2), blocked=True)).status,
                         'wont_fit')
        self.assertEqual(evaluate_fit(UNIT, SlotFreeSpace((0.7, 0.5, 0.2), clear=False)).status,
                         'wont_fit')

    def test_unknown_absent_record(self):
        v = evaluate_fit(UNIT, None)
        self.assertEqual(v.status, 'unknown')
        self.assertTrue(v.low_confidence)

    def test_unknown_unmeasured_falls_back_not_refuses(self):
        # Even a clearly-too-small configured stand-in is 'unknown' (fall back), never 'wont_fit':
        # we refuse only on trustworthy measurement.
        v = evaluate_fit(UNIT, SlotFreeSpace((0.1, 0.1, 0.1), measured=False))
        self.assertEqual(v.status, 'unknown')
        self.assertTrue(v.low_confidence)

    def test_unknown_low_confidence(self):
        v = evaluate_fit(UNIT, SlotFreeSpace((0.62, 0.42, 0.18), confidence=0.3))
        self.assertEqual(v.status, 'unknown')
        self.assertIn('confidence', v.message)

    def test_event_is_json_safe(self):
        json.dumps(evaluate_fit(UNIT, SlotFreeSpace((0.6, 0.42, 0.18))).event())


class HandlingUnitTests(unittest.TestCase):
    def test_from_scenario_default_sku(self):
        self.assertEqual(handling_unit_from_scenario(SCENARIO).dimensions_m, (0.6, 0.4, 0.16))

    def test_units_override_is_configurable(self):
        u = handling_unit_from_scenario(SCENARIO, sku='tray_small',
                                        units={'tray_small': (0.3, 0.2, 0.1)})
        self.assertEqual((u.sku, u.dimensions_m), ('tray_small', (0.3, 0.2, 0.1)))

    def test_unknown_sku_raises(self):
        with self.assertRaises(ValueError):
            handling_unit_from_scenario(SCENARIO, sku='not_in_scope')


class LoaderTests(unittest.TestCase):
    def test_depth_capture_metres_passthrough(self):
        fs = free_space_from_depth_capture((0.62, 0.42, 0.18), slot_id='s')
        self.assertEqual(fs.free_dimensions_m, (0.62, 0.42, 0.18))
        self.assertTrue(fs.measured)

    def test_twin3d_mm_to_m_with_axis_order(self):
        # Bay frame (width,height,depth) mm; map comparison (x,y,z) <- (depth,width,height).
        rec = {'free_width_mm': 430.0, 'free_height_mm': 190.0, 'free_depth_mm': 630.0,
               'measured': True, 'confidence': 0.9, 'blocked': False}
        fs = free_space_from_twin3d(rec, slot_id='G3_back_s2', axis_order=(2, 0, 1))
        self.assertEqual(fs.free_dimensions_m, (0.63, 0.43, 0.19))
        self.assertEqual(evaluate_fit(UNIT, fs).status, 'ok')

    def test_twin3d_bad_axis_order_raises(self):
        rec = {'free_width_mm': 1, 'free_height_mm': 1, 'free_depth_mm': 1}
        with self.assertRaises(ValueError):
            free_space_from_twin3d(rec, axis_order=(0, 0, 1))

    def test_twin3d_blocked_propagates(self):
        rec = {'free_width_mm': 700, 'free_height_mm': 300, 'free_depth_mm': 700, 'blocked': True}
        self.assertTrue(free_space_from_twin3d(rec).blocked)

    def test_load_free_space_shapes(self):
        self.assertIsNone(load_free_space(None))
        self.assertIsNone(load_free_space({}))
        a = load_free_space({'free_dimensions_m': [0.62, 0.42, 0.18]})
        self.assertEqual(a.free_dimensions_m, (0.62, 0.42, 0.18))
        b = load_free_space({'free_space': {'free_dimensions_m': [0.6, 0.42, 0.18],
                                            'measured': True, 'confidence': 1.0}})
        self.assertEqual(b.free_dimensions_m, (0.6, 0.42, 0.18))

    def test_load_free_space_twin3d_export_file(self):
        doc = {'schema': 'storevision.twin3d.slot_volumes.v1',
               'slots': [{'slot_id': 'G3_back_s2', 'fill': {},
                          'free_width_mm': 400.0, 'free_height_mm': 160.0,
                          'free_depth_mm': 620.0, 'measured': True, 'confidence': 0.95}]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'slots.json'
            p.write_text(json.dumps(doc))
            fs = load_free_space({'twin3d_export': str(p), 'slot_id': 'G3_back_s2',
                                  'axis_order': [2, 0, 1]})
        self.assertEqual(fs.free_dimensions_m, (0.62, 0.4, 0.16))
        self.assertAlmostEqual(fs.confidence, 0.95)


class PreflightFromConfigTests(unittest.TestCase):
    def test_absent_option_returns_none(self):
        self.assertIsNone(preflight_from_config(SCENARIO, None))
        self.assertIsNone(preflight_from_config(SCENARIO, {}))
        self.assertIsNone(preflight_from_config(SCENARIO, {'fit_preflight': None}))

    def test_ok_and_wont_fit(self):
        ok = preflight_from_config(SCENARIO, {'fit_preflight': {
            'free_dimensions_m': [0.63, 0.43, 0.19]}})
        self.assertEqual(ok.status, 'ok')
        bad = preflight_from_config(SCENARIO, {'fit_preflight': {
            'free_dimensions_m': [0.5, 0.42, 0.18]}})
        self.assertEqual(bad.status, 'wont_fit')


class _Fixture:
    def __init__(self):
        self.started = self.stopped = False
        self.time = 0.

    def start(self, action, job):
        self.started = True
        self.action, self.job = action, job

    def step(self, dt):
        from twin.restock.mission import Result
        self.time += dt
        return Result('succeeded', dict(
            source='test_fixture', observation_id=f'{self.action}-{self.time}',
            simulation_time=self.time, loaded=True, coupled=True, arrived=True,
            stationary=True, released=True, home=True, sku=self.job.sku,
            quantity=self.job.quantity, location=self.job.shelf_id, transfer_confirmed=True))

    def stop(self):
        self.stopped = True


class MissionGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inv = Inventory(str(Path(self.tmp.name) / 'stock.sqlite'))
        self.inv.seed('backroom', 'juice_orange', 2)
        self.job = Job('one', 'juice_orange', 1, 'backroom', 'shelf')
        self.adapter = _Fixture()

    def tearDown(self):
        self.inv.close()
        self.tmp.cleanup()

    def test_none_is_byte_identical_full_cycle(self):
        m = Mission(self.inv, self.job, self.adapter, fixture_mode=True)
        m.start()
        for _ in ACTIONS:
            m.tick(.1)
        self.assertEqual(m.state, 'complete')
        self.assertEqual(self.inv.count('shelf', 'juice_orange'), 1)

    def test_wont_fit_refuses_before_reservation(self):
        verdict = evaluate_fit(UNIT, SlotFreeSpace((0.5, 0.42, 0.18)))
        m = Mission(self.inv, self.job, self.adapter, fixture_mode=True, fit_preflight=verdict)
        m.start()
        self.assertEqual(m.state, 'failed')
        self.assertTrue(m.reason.startswith('unsupported:'))
        self.assertFalse(self.adapter.started)       # robot never started
        self.assertEqual(self.inv.events('one'), [])  # job never reserved: no events
        self.assertEqual(self.inv.count('backroom', 'juice_orange'), 2)  # stock untouched

    def test_ok_and_unknown_proceed(self):
        for verdict in [evaluate_fit(UNIT, SlotFreeSpace((0.63, 0.43, 0.19))),
                        evaluate_fit(UNIT, None)]:
            inv = Inventory(':memory:')
            inv.seed('backroom', 'juice_orange', 2)
            m = Mission(inv, Job('j', 'juice_orange', 1, 'backroom', 'shelf'),
                        _Fixture(), fixture_mode=True, fit_preflight=verdict)
            m.start()
            self.assertEqual(m.state, 'collect')  # proceeded into the cycle
            inv.close()


if __name__ == '__main__':
    unittest.main()
