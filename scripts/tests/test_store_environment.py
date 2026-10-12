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
        self.assertEqual(len(stations), 73)
        self.assertEqual(len({s['id'] for s in stations}), len(stations))
        for station in stations:
            self.assertEqual(len(station['pose']), 3)
            self.assertTrue(station['slots'])
        all_skus = {sku for s in stations for slot in s['slots'] for sku in slot['sku_ids']}
        self.assertTrue(all(s['sku'] in all_skus for s in data['backroom_stocks']))

    def test_body_specific_dock_and_chassis_headings(self):
        import math
        data = layout(PLAN)
        robot, cart = data['poses']['robot_dock'], data['poses']['cart_dock']
        self.assertAlmostEqual(robot[0] - cart[0], 1.27)
        self.assertEqual(robot[1:], cart[1:])
        self.assertEqual(data['pose_bodies']['cart_dock'], 'cart_body')
        for station in data['stocking_stations'][:-1]:
            self.assertEqual(station['pose'][2], -math.pi / 2)
            self.assertIn(station['arm_shelf_facing_yaw'], [0, math.pi])
        first = data['stocking_stations'][-1]
        self.assertEqual(first['id'], data['first_cycle_station'])
        self.assertEqual(first['pose'][2], math.pi)
        self.assertEqual(first['arm_shelf_facing_yaw'], -math.pi / 2)
        self.assertEqual(data['backroom_stocks'][0]['sku'], data['first_cycle_sku'])
        self.assertTrue(any(data['first_cycle_sku'] in s['sku_ids'] for s in first['slots']))

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
