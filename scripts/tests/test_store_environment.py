"""CPU-only semantic contract checks; geometry checks live in check_environment.py."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from store_environment import layout, PORTAL_X, PORTAL_WIDTH
from twin.colliders import classify, PrimInfo, SKIP, TRIANGLE_MESH

PLAN = json.loads((Path(__file__).resolve().parents[1] / 'planogram.json').read_text())


class EnvironmentTests(unittest.TestCase):
    def test_determinism_and_no_planogram_mutation(self):
        before = json.dumps(PLAN)
        self.assertEqual(layout(PLAN), layout(PLAN))
        self.assertEqual(before, json.dumps(PLAN))

    def test_stations_reference_real_slots_and_skus(self):
        data = layout(PLAN)
        stations = data['stocking_stations']
        self.assertEqual(len(stations), 72)
        self.assertEqual(len({s['id'] for s in stations}), len(stations))
        for station in stations:
            self.assertEqual(len(station['pose']), 3)
            self.assertTrue(station['slots'])
        all_skus = {sku for s in stations for slot in s['slots'] for sku in slot['sku_ids']}
        self.assertTrue(all(s['sku'] in all_skus for s in data['backroom_stocks']))

    def test_portal_fits_existing_rear_fixture_gap(self):
        wall = next(w for w in PLAN['wall_units'] if w['id'] == 'W_back')
        wall_end = wall['x'] + wall['n_bays'] * PLAN['bay_w']
        chiller_start = next(w['x'] for w in PLAN['chillers'] if w['id'] == 'C_back')
        self.assertGreater(PORTAL_X - PORTAL_WIDTH / 2, wall_end)
        self.assertLess(PORTAL_X + PORTAL_WIDTH / 2, chiller_start)

    def test_collision_groups(self):
        for group, kind in [('GRP_EnvironmentDecor', SKIP), ('GRP_Structure', TRIANGLE_MESH),
                            ('GRP_Fixtures', TRIANGLE_MESH)]:
            self.assertEqual(classify(PrimInfo('/Supermarket/' + group + '/mesh', 'Mesh', group)).kind, kind)


if __name__ == '__main__':
    unittest.main()
