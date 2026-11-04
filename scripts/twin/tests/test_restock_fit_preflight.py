"""CPU tests for the opt-in tray/slot fit preflight.

No Isaac/ROS/Blender imports. Covers the shared axiswise predicate (parity with the
place-phase gate), the fit verdict's three honest cases, the record loaders
(mission depth capture + store-vision twin3d export with an explicit axis order),
and the default-off mission gate (a 'wont_fit' refuses before any reservation; an
absent verdict is byte-identical).
"""
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.fit_preflight import (
    CATALOGUE_SCHEMA, CatalogueError, HandlingUnit, SlotFreeSpace, evaluate_fit,
    free_space_from_depth_capture, free_space_from_twin3d, handling_unit_from_scenario,
    load_catalogue, load_free_space, preflight_from_config)
from twin.restock.inventory import Inventory, Job
from twin.restock.mission import ACTIONS, Mission
import twin.restock_manipulation.controller as controller_mod
from twin.restock_manipulation.fit import axis_fits

UNIT = HandlingUnit('juice_orange', (0.6, 0.4, 0.16))
SCENARIO = {'job': {'sku': 'juice_orange'},
            'handling_unit': {'sku': 'juice_orange', 'dimensions_m': [0.6, 0.4, 0.16]}}

# The mission's working-default handling unit is the scenario carton (sim-carton-size-v1).
CARTON_SCENARIO = {'job': {'sku': 'juice_orange'},
                   'handling_unit': {'sku': 'juice_orange', 'dimensions_m': [0.45, 0.3, 0.18]}}


def _catalogue_doc(units, axis_order=(0, 1, 2)):
    """A minimal storevision.restock.product_catalogue/v1 document for tests."""
    return {'schema': CATALOGUE_SCHEMA, 'axis_order': list(axis_order),
            'units': [dict(u) for u in units]}


def _load_producer():
    """Load the store-vision-free producer module from its file (no package import)."""
    path = Path(__file__).resolve().parents[2] / 'integrations' / 'export_tray_catalogue.py'
    spec = importlib.util.spec_from_file_location('export_tray_catalogue', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _DuckProduct:
    """A duck-typed store-vision ProductInfo stand-in (no store-vision import needed)."""
    def __init__(self, sku, dims_mm, weight_kg=0.5, tray_packed=True):
        self.sku, self.dims_mm, self.weight_kg, self.tray_packed = sku, dims_mm, weight_kg, tray_packed


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


class CatalogueLoaderTests(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertEqual(load_catalogue(None), {})

    def test_inline_doc_builds_units_in_metres(self):
        cat = load_catalogue(_catalogue_doc([
            {'sku': 'juice_orange', 'dims_mm': [450, 300, 180]},
            {'sku': 'cereal_big', 'dims_mm': [600, 400, 300], 'source': 'sv'}]))
        self.assertEqual(set(cat), {'juice_orange', 'cereal_big'})
        self.assertEqual(cat['juice_orange'].dimensions_m, (0.45, 0.3, 0.18))
        self.assertTrue(cat['cereal_big'].source.startswith('catalogue:'))

    def test_bad_schema_raises(self):
        with self.assertRaises(CatalogueError):
            load_catalogue({'schema': 'nope', 'units': []})

    def test_axis_order_maps_LWH_to_xyz(self):
        # (L,W,H)=(600,400,300) with order (2,0,1) -> (x,y,z)=(H,L,W)=(0.3,0.6,0.4).
        cat = load_catalogue(_catalogue_doc([{'sku': 't', 'dims_mm': [600, 400, 300]}],
                                            axis_order=(2, 0, 1)))
        self.assertEqual(cat['t'].dimensions_m, (0.3, 0.6, 0.4))

    def test_bad_axis_order_raises(self):
        with self.assertRaises(CatalogueError):
            load_catalogue(_catalogue_doc([{'sku': 't', 'dims_mm': [1, 1, 1]}], axis_order=(0, 0, 1)))

    def test_missing_or_bad_dims_raise(self):
        with self.assertRaises(CatalogueError):
            load_catalogue(_catalogue_doc([{'sku': 't'}]))
        with self.assertRaises(CatalogueError):
            load_catalogue(_catalogue_doc([{'sku': 't', 'dims_mm': [1, 2]}]))
        with self.assertRaises(CatalogueError):
            load_catalogue(_catalogue_doc([{'sku': 't', 'dims_mm': [0, 2, 3]}]))

    def test_units_must_be_list(self):
        with self.assertRaises(CatalogueError):
            load_catalogue({'schema': CATALOGUE_SCHEMA, 'units': {}})

    def test_load_from_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'cat.json'
            p.write_text(json.dumps(_catalogue_doc([{'sku': 'x', 'dims_mm': [100, 200, 300]}])))
            cat = load_catalogue(str(p))
        self.assertEqual(cat['x'].dimensions_m, (0.1, 0.2, 0.3))

    def test_unreadable_path_raises(self):
        with self.assertRaises(CatalogueError):
            load_catalogue('/no/such/catalogue.json')


class ResolverPrecedenceTests(unittest.TestCase):
    def test_default_is_scenario(self):
        u = handling_unit_from_scenario(CARTON_SCENARIO)
        self.assertEqual((u.dimensions_m, u.source), ((0.45, 0.3, 0.18), 'scenario'))

    def test_catalogue_adds_new_sku_in_scope(self):
        cat = load_catalogue(_catalogue_doc([{'sku': 'cereal_big', 'dims_mm': [600, 400, 300]}]))
        u = handling_unit_from_scenario(CARTON_SCENARIO, sku='cereal_big', catalogue=cat)
        self.assertEqual(u.dimensions_m, (0.6, 0.4, 0.3))
        self.assertTrue(u.source.startswith('catalogue:'))

    def test_catalogue_overrides_scenario_for_shared_sku(self):
        # The store-vision record is the richer source for a SKU the scenario also carries.
        cat = load_catalogue(_catalogue_doc([{'sku': 'juice_orange', 'dims_mm': [460, 310, 190]}]))
        u = handling_unit_from_scenario(CARTON_SCENARIO, catalogue=cat)
        self.assertEqual(u.dimensions_m, (0.46, 0.31, 0.19))
        self.assertTrue(u.source.startswith('catalogue:'))

    def test_units_override_catalogue(self):
        cat = load_catalogue(_catalogue_doc([{'sku': 'juice_orange', 'dims_mm': [460, 310, 190]}]))
        u = handling_unit_from_scenario(CARTON_SCENARIO, catalogue=cat,
                                        units={'juice_orange': (0.4, 0.25, 0.15)})
        self.assertEqual((u.dimensions_m, u.source), ((0.4, 0.25, 0.15), 'config_units'))

    def test_unknown_sku_error_lists_scope(self):
        cat = load_catalogue(_catalogue_doc([{'sku': 'cereal_big', 'dims_mm': [600, 400, 300]}]))
        with self.assertRaises(ValueError) as ctx:
            handling_unit_from_scenario(CARTON_SCENARIO, sku='mystery', catalogue=cat)
        self.assertIn('cereal_big', str(ctx.exception))
        self.assertIn('juice_orange', str(ctx.exception))


class CataloguePreflightTests(unittest.TestCase):
    def test_default_off_is_scenario_unit(self):
        # No catalogue key => working default (scenario carton), byte-identical to v1.
        v = preflight_from_config(CARTON_SCENARIO, {'fit_preflight': {
            'free_dimensions_m': [0.48, 0.33, 0.21]}})
        self.assertEqual(v.status, 'ok')
        self.assertEqual((v.unit_sku, v.unit_source), ('juice_orange', 'scenario'))

    def test_variant_inline_catalogue_ok(self):
        v = preflight_from_config(CARTON_SCENARIO, {'fit_preflight': {
            'sku': 'cereal_big',
            'catalogue': _catalogue_doc([{'sku': 'cereal_big', 'dims_mm': [600, 400, 300]}]),
            'free_dimensions_m': [0.63, 0.43, 0.33]}})
        self.assertEqual(v.status, 'ok')
        self.assertEqual(v.unit_sku, 'cereal_big')
        self.assertTrue(v.unit_source.startswith('catalogue:'))

    def test_variant_catalogue_wont_fit(self):
        v = preflight_from_config(CARTON_SCENARIO, {'fit_preflight': {
            'sku': 'cereal_big',
            'catalogue': _catalogue_doc([{'sku': 'cereal_big', 'dims_mm': [600, 400, 300]}]),
            'free_dimensions_m': [0.5, 0.43, 0.33]}})  # x too small for 0.6 + margin
        self.assertEqual(v.status, 'wont_fit')
        self.assertIn('tray exceeds slot free dimensions', v.message)

    def test_variant_from_export_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'cat.json'
            p.write_text(json.dumps(_catalogue_doc([{'sku': 'wide_tray', 'dims_mm': [700, 300, 160]}])))
            v = preflight_from_config(CARTON_SCENARIO, {'fit_preflight': {
                'sku': 'wide_tray', 'catalogue_export': str(p),
                'free_dimensions_m': [0.73, 0.33, 0.19]}})
        self.assertEqual(v.status, 'ok')
        self.assertEqual(v.unit_sku, 'wide_tray')

    def test_event_carries_unit_fields_and_is_json_safe(self):
        v = preflight_from_config(CARTON_SCENARIO, {'fit_preflight': {
            'catalogue': _catalogue_doc([{'sku': 'juice_orange', 'dims_mm': [450, 300, 180]}]),
            'free_dimensions_m': [0.48, 0.33, 0.21]}})
        ev = v.event()
        json.dumps(ev)
        self.assertEqual(ev['unit_sku'], 'juice_orange')
        self.assertTrue(ev['unit_source'].startswith('catalogue:'))


class ProducerTests(unittest.TestCase):
    """The scripts/integrations/export_tray_catalogue.py producer (store-vision-free core)."""
    def setUp(self):
        self.prod = _load_producer()

    def test_build_doc_round_trips_through_consumer(self):
        doc = self.prod.build_catalogue_doc(
            [_DuckProduct('juice_orange', (450, 300, 180), 0.45),
             _DuckProduct('cereal_big', (600, 400, 300), 1.2)], note='demo')
        self.assertEqual(doc['schema'], CATALOGUE_SCHEMA)
        cat = load_catalogue(doc)
        self.assertEqual(set(cat), {'juice_orange', 'cereal_big'})
        self.assertEqual(cat['juice_orange'].dimensions_m, (0.45, 0.3, 0.18))

    def test_duplicate_sku_raises(self):
        with self.assertRaises(ValueError):
            self.prod.build_catalogue_doc(
                [_DuckProduct('a', (1, 2, 3)), _DuckProduct('a', (4, 5, 6))])

    def test_bad_axis_order_raises(self):
        with self.assertRaises(ValueError):
            self.prod.build_catalogue_doc([_DuckProduct('a', (1, 2, 3))], axis_order=(0, 0, 1))

    def test_bad_dims_raise(self):
        with self.assertRaises(ValueError):
            self.prod.build_catalogue_doc([_DuckProduct('a', (0, 2, 3))])

    def test_parse_product(self):
        self.assertEqual(self.prod._parse_product('sku:450x300x180:0.45'),
                         {'sku': 'sku', 'dims_mm': (450., 300., 180.), 'weight_kg': 0.45})
        self.assertEqual(self.prod._parse_product('s:1x2x3')['weight_kg'], 0.5)

    def test_repo_example_loads(self):
        # The committed example export is a valid, consumable catalogue.
        example = Path(__file__).resolve().parents[2] / 'tray_catalogue.example.json'
        cat = load_catalogue(str(example))
        self.assertIn('juice_orange', cat)
        self.assertEqual(cat['juice_orange'].dimensions_m, (0.45, 0.3, 0.18))


class CatalogueMissionGateTests(unittest.TestCase):
    """A catalogue-sourced won't-fit refuses before any reservation or robot motion."""
    def test_catalogue_wont_fit_refuses_before_reservation(self):
        verdict = preflight_from_config(CARTON_SCENARIO, {'fit_preflight': {
            'sku': 'cereal_big',
            'catalogue': _catalogue_doc([{'sku': 'cereal_big', 'dims_mm': [600, 400, 300]}]),
            'free_dimensions_m': [0.5, 0.43, 0.33]}})
        self.assertEqual(verdict.status, 'wont_fit')
        inv = Inventory(':memory:')
        inv.seed('backroom', 'cereal_big', 2)
        adapter = _Fixture()
        m = Mission(inv, Job('one', 'cereal_big', 1, 'backroom', 'shelf'),
                    adapter, fixture_mode=True, fit_preflight=verdict)
        m.start()
        self.assertEqual(m.state, 'failed')
        self.assertTrue(m.reason.startswith('unsupported:'))
        self.assertFalse(adapter.started)
        self.assertEqual(inv.count('backroom', 'cereal_big'), 2)
        inv.close()


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
