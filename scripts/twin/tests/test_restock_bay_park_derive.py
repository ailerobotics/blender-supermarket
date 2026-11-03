"""CPU tests for the v3 re-derivation of the ``recorded_bay_park`` transfer-bench staging pose
(sim-rerecord-bay-park-v3-v1).

The frozen v2 pose (0.6 m carton, v2 store) read as misaligned at the staged dock on the v3-portal26
store + 0.45 m carton (ledger stage-transfer2). The preset now DERIVES its pose from the live scenario
geometry so a store or carton change re-derives rather than replaying a stale pose, and carries a
provenance stamp so a frozen fragment goes stale LOUDLY. These tests prove the derivation, the reach
feasibility gate, the provenance staleness check and the staged-scenario wiring -- all on CPU.
"""
import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.mobile import Coupler, DockObservation, MobileSpec, Pose2            # noqa: E402
from twin.restock.shortcut import (bay_park_provenance, check_bay_park_provenance,  # noqa: E402
                                   derive_bay_park, staged_scenario,
                                   RECORDED_BAY_PARK_ROBOT_POSE)


def v3_scenario():
    """The fields the staging reads, mirroring the live v3-portal26 mission scenario."""
    return {'job': {'job_id': 'restock-one', 'sku': 'juice_orange', 'quantity': 1,
                    'source_bin': 'backstock_00', 'shelf_id': 'G3_back_s2'},
            'handling_unit': {'dimensions_m': [0.45, 0.3, 0.18], 'max_reach_m': 1.3},
            'transfer_bay': {'center_xy': [-2.0, 16.42], 'dimensions_m': [0.7, 0.5, 0.73], 'top_z': 0.73},
            'initial_staging': {'robot_base_pose': [-5.73, 16.0, 0.0],
                                'cart_body_pose': [-5.73, 15.25, 0.0],
                                'cart_load_center_body_xy': [0.18, 0.0],
                                'cart_support_z_world': 0.9325,
                                'unit_yaws_world': {'source': math.pi, 'cart': -math.pi / 2,
                                                    'shelf': 0.0}}}


def bench_config(**over):
    cfg = {'start_at': 'transfer', 'start_staging': 'recorded_bay_park'}
    cfg.update(over)
    return cfg


def yaw_err(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


class DeriveTests(unittest.TestCase):
    def test_derived_pose_is_grounded_in_bay_and_corridor(self):
        d = derive_bay_park(v3_scenario())
        x, y, yaw = d['robot_base_pose']
        # x centred under the transfer bay; y + yaw on the tow corridor (the scenario's cart_body_pose).
        self.assertAlmostEqual(x, -2.0, places=4)       # transfer_bay.center_xy[0]
        self.assertAlmostEqual(y, 15.25, places=4)      # initial_staging.cart_body_pose[1]
        self.assertAlmostEqual(yaw, 0.0, places=6)      # corridor heading

    def test_derived_pose_keeps_pick_and_place_in_reach(self):
        d = derive_bay_park(v3_scenario())
        self.assertLessEqual(d['reach']['carton_pick_m'], d['reach']['max_m'])
        self.assertLessEqual(d['reach']['bay_place_m'], d['reach']['max_m'])
        self.assertAlmostEqual(d['reach']['carton_pick_m'], 1.09, places=2)
        self.assertAlmostEqual(d['reach']['bay_place_m'], 1.17, places=2)

    def test_pose_moved_east_vs_the_v2_record(self):
        # The re-derivation is the 'after'; the frozen v2 mean is the 'before' (robot x -2.079).
        d = derive_bay_park(v3_scenario())
        self.assertAlmostEqual(d['robot_base_pose'][0] - RECORDED_BAY_PARK_ROBOT_POSE[0], 0.079, places=3)

    def test_infeasible_reach_raises_named(self):
        sc = v3_scenario()
        sc['transfer_bay']['center_xy'] = [-2.0, 18.5]      # bay 3.25 m north of the corridor
        with self.assertRaises(ValueError) as ei:
            derive_bay_park(sc)
        self.assertIn('bay place reach', str(ei.exception))
        self.assertIn('max', str(ei.exception))

    def test_missing_geometry_fails_closed(self):
        for drop in ('transfer_bay', 'initial_staging'):
            sc = v3_scenario()
            del sc[drop]
            with self.assertRaises(ValueError):
                derive_bay_park(sc)
        sc = v3_scenario()
        sc['handling_unit']['dimensions_m'] = [0.45, 0.3]   # malformed
        with self.assertRaises(ValueError):
            derive_bay_park(sc)


class ProvenanceTests(unittest.TestCase):
    def test_stamp_carries_the_geometry_and_pose(self):
        prov = bay_park_provenance(v3_scenario(), dataset='sim-store-realism-v3-portal26', git_head='abc1234')
        self.assertEqual(prov['carton_dimensions_m'], [0.45, 0.3, 0.18])
        self.assertEqual(prov['transfer_bay_center_xy'], [-2.0, 16.42])
        self.assertEqual(prov['derived_robot_base_pose'], [-2.0, 15.25, 0.0])
        self.assertEqual(prov['dataset'], 'sim-store-realism-v3-portal26')
        self.assertEqual(prov['git_head'], 'abc1234')

    def test_fresh_provenance_passes(self):
        prov = bay_park_provenance(v3_scenario())
        self.assertEqual(check_bay_park_provenance(prov, v3_scenario())['derived_robot_base_pose'],
                         [-2.0, 15.25, 0.0])

    def test_stale_carton_raises_by_name(self):
        prov = bay_park_provenance(v3_scenario())
        stale = dict(prov, carton_dimensions_m=[0.6, 0.4, 0.16])
        with self.assertRaises(ValueError) as ei:
            check_bay_park_provenance(stale, v3_scenario())
        self.assertIn('carton_dimensions_m', str(ei.exception))
        self.assertIn('STALE', str(ei.exception))

    def test_stale_bay_raises_by_name(self):
        prov = bay_park_provenance(v3_scenario())
        stale = dict(prov, transfer_bay_center_xy=[-2.0, 15.9])
        with self.assertRaises(ValueError) as ei:
            check_bay_park_provenance(stale, v3_scenario())
        self.assertIn('transfer_bay_center_xy', str(ei.exception))

    def test_no_stamp_returns_live(self):
        self.assertEqual(check_bay_park_provenance(None, v3_scenario())['derived_robot_base_pose'],
                         [-2.0, 15.25, 0.0])


class StagedScenarioTests(unittest.TestCase):
    def test_derives_on_v3_and_records_provenance(self):
        variant, record = staged_scenario(v3_scenario(), bench_config())
        self.assertEqual(record['robot_base_pose'], [-2.0, 15.25, 0.0])
        self.assertEqual(record['robot_base_pose_origin'], 'derived_from_scenario')
        self.assertIn('provenance', record)
        self.assertEqual(record['provenance']['carton_dimensions_m'], [0.45, 0.3, 0.18])
        self.assertIn('reach', record)

    def test_staged_train_passes_the_coupler_attach_gate(self):
        variant, _ = staged_scenario(v3_scenario(), bench_config())
        st = variant['initial_staging']
        spec = MobileSpec()
        robot, cart = Pose2(*st['robot_base_pose']), Pose2(*st['cart_body_pose'])
        # Hitch coincident by construction, articulation 0.
        self.assertLess(math.dist(robot.point(spec.hitch_x), cart.point(spec.cart_hitch_x)), 1e-9)
        obs = DockObservation(robot, cart, spec.hitch_z, spec.hitch_z, 0.0, 0.0, 0.0, 0.0)
        made = []
        Coupler(spec).attach(obs, lambda: made.append(True))
        self.assertEqual(made, [True])

    def test_falls_back_to_v2_without_a_transfer_bay(self):
        sc = v3_scenario()
        del sc['transfer_bay']
        _, record = staged_scenario(sc, bench_config())
        self.assertEqual(record['robot_base_pose_origin'], 'v2_fallback')
        self.assertEqual(tuple(record['robot_base_pose']), RECORDED_BAY_PARK_ROBOT_POSE)
        self.assertNotIn('provenance', record)

    def test_explicit_pose_overrides_derivation(self):
        _, record = staged_scenario(v3_scenario(), bench_config(
            start_staging={'robot_base_pose': [-2.1, 15.25, 0.0]}))
        self.assertEqual(record['robot_base_pose'], [-2.1, 15.25, 0.0])
        self.assertEqual(record['robot_base_pose_origin'], 'explicit_config')

    def test_frozen_stale_provenance_is_refused(self):
        stale = bay_park_provenance(v3_scenario())
        stale['carton_dimensions_m'] = [0.6, 0.4, 0.16]
        with self.assertRaises(ValueError):
            staged_scenario(v3_scenario(), bench_config(start_staging={
                'robot_base_pose': [-2.0, 15.25, 0.0], 'provenance': stale}))

    def test_scenario_not_mutated(self):
        sc = v3_scenario()
        before = copy.deepcopy(sc)
        staged_scenario(sc, bench_config())
        self.assertEqual(sc, before)


if __name__ == '__main__':
    unittest.main()
