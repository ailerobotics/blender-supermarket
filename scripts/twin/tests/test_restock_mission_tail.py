"""Guard the mission TAIL (transfer succeeded -> verify -> uncouple-at-parking return -> mission end) for the
backroom-bay demo with the CPU dry-run (``scripts/check_mission_tail.py``, sim-video-verify-return-v1 / vr1).

The fixture ``fixtures/mission_tail_bay.json`` is the tail-relevant excerpt of the run-356 runtime config +
the fm459 bay scenario + the backroom static map, unchanged; the tests add the vr1 switches on top. The BT
dry-runs need py_trees (skipped without it, like the other BT tests); the geometry/gate tests do not.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts:<bt-pythonpath> python3 -m pytest \
        scripts/twin/tests/test_restock_mission_tail.py -q
"""
import copy
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import check_mission_tail as tail  # noqa: E402
from twin.restock.ground_truth_verify import (GroundTruthInventoryVerifier,  # noqa: E402
                                              GroundTruthVerifyError, ground_truth_place_actions,
                                              mission_ground_truth_sources, mission_modes)
from twin.restock.inventory import Inventory, Job  # noqa: E402
from twin.restock.mission import Mission, Result  # noqa: E402

FIXTURE = json.loads((Path(__file__).resolve().parent / 'fixtures' / 'mission_tail_bay.json').read_text())
HAVE_PT = tail.have_py_trees()


def _config(**extra):
    config = copy.deepcopy(FIXTURE['config'])
    config.update(extra)
    return config


def _rows(rep, prefix):
    return {name: (verdict, detail) for verdict, name, detail in rep.rows if name.startswith(prefix)}


class GroundTruthVerifierGeometry(unittest.TestCase):
    """The mission-level ground-truth verify against the bay top (0.7 x 0.5 at z 0.73) and the placed
    tray (0.6 x 0.4 x 0.16): rests-on-bay accepted, off-bay refused fast and named, two frames."""

    def setUp(self):
        self.scenario = FIXTURE['scenario']
        self.unit = [-2.0, 16.42, 0.73 + 0.08, 0.0]
        self.contact = [False]
        self.v = tail.gt_verifier_from_config(_config(verify_ground_truth={'enabled': True}), self.scenario,
                                              lambda: tuple(self.unit), read_contact=lambda: self.contact[0])
        self.job = Job(**self.scenario['job'])

    def test_disabled_without_the_switch(self):
        self.assertIsNone(tail.gt_verifier_from_config(_config(), self.scenario, lambda: None))

    def test_two_frames_then_labelled_evidence(self):
        self.v.begin(self.job, 10.0)
        self.assertIsNone(self.v(10.0))                  # not after the boundary
        self.assertIsNone(self.v(10.1))                  # first confirmation
        ev = self.v(10.2)
        self.assertEqual((ev['source'], ev['verify_mode']), ('ground_truth', 'ground_truth'))
        self.assertEqual((ev['sku'], ev['quantity'], ev['location']), ('juice_orange', 1, 'G3_back_s2'))
        self.assertIs(ev['transfer_confirmed'], True)
        self.assertEqual(len(ev['confirming_observation_ids']), 2)
        self.assertIn('NOT a camera measurement', ev['note'])
        self.assertFalse(self.v.active)

    def test_tray_off_the_bay_fails_fast_named(self):
        self.unit[2] = 0.08                              # on the floor
        self.v.begin(self.job, 0.0)
        self.assertIsNone(self.v(0.1))
        self.assertIsNone(self.v(0.2))
        with self.assertRaisesRegex(GroundTruthVerifyError, 'not resting on the bay'):
            self.v(0.3)

    def test_still_held_is_refused(self):
        self.contact[0] = True
        self.v.begin(self.job, 0.0)
        with self.assertRaisesRegex(GroundTruthVerifyError, 'still held'):
            self.v(0.1)

    def test_settling_band_waits(self):
        self.unit[2] += 0.05                             # 5 cm above the top: not on, not clearly off
        self.v.begin(self.job, 0.0)
        for k in range(1, 6):
            self.assertIsNone(self.v(0.1 * k))
        self.assertTrue(self.v.active)

    def test_tray_on_bay_cases(self):
        rep = tail.Report(io.StringIO())
        tail.check_tray_on_bay(_config(verify_ground_truth={'enabled': True}), self.scenario, rep)
        failed = [name for verdict, name, _ in rep.rows if verdict == 'FAIL']
        self.assertEqual(failed, [])


class GroundTruthSwitches(unittest.TestCase):
    def test_default_config_is_byte_identical(self):
        self.assertEqual(mission_modes({}), {})
        self.assertEqual(ground_truth_place_actions({}), ())
        self.assertEqual(mission_ground_truth_sources({}), ())

    def test_run_config_modes(self):
        config = _config(verify_ground_truth={'enabled': True})
        self.assertEqual(mission_modes(config), {'verify_mode': 'ground_truth',
                                                 'transfer_verify_mode': 'ground_truth',
                                                 'return_mode': 'uncouple_at_cart_parking'})
        self.assertEqual(ground_truth_place_actions(config), ('transfer',))
        self.assertEqual(mission_ground_truth_sources(config), ('ground_truth',))

    def test_mission_refuses_ground_truth_evidence_unless_opted_in(self):
        class Adapter:
            def start(self, action, job): pass
            def stop(self): pass
            def step(self, dt):
                return Result('succeeded', {'source': 'ground_truth', 'observation_id': 'gt:1',
                                            'simulation_time': 0.1, 'released': True})
        with tempfile.TemporaryDirectory() as tmp:
            for sources, ok in (((), False), (('ground_truth',), True)):
                db = Inventory(str(Path(tmp) / f'{ok}.sqlite'))
                db.seed('backstock_00', 'juice_orange', 1)
                m = Mission(db, Job(**FIXTURE['scenario']['job']), Adapter(), start_at='transfer',
                            ground_truth_sources=sources)
                m.start()
                out = m.step_stage(0.1)
                self.assertEqual(out == 'succeeded', ok, out)
                db.close()
        with self.assertRaises(ValueError):
            Mission(None, Job(**FIXTURE['scenario']['job']), Adapter(), ground_truth_sources=('sensor_x',))


class ConfigChecks(unittest.TestCase):
    def test_run356_config_needs_the_verify_switch(self):
        # The run-356 config as-is: everything in place except a way for VERIFY to confirm on the
        # uncommissioned bay -- the dry-run names the missing key.
        rep = tail.Report(io.StringIO())
        tail.check_config(_config(), FIXTURE['scenario'], rep)
        rows = _rows(rep, 'config_')
        self.assertEqual(rows['config_verify_stage_can_confirm'][0], 'FAIL')
        self.assertIn('verify_ground_truth', rows['config_verify_stage_can_confirm'][1]['fix'])
        others = [n for n, (v, _) in rows.items() if v == 'FAIL' and n != 'config_verify_stage_can_confirm']
        self.assertEqual(others, [])
        self.assertEqual(rows['config_return_lane_resolves'][1]['lane'], 'return_bay_to_parking')

    def test_with_the_verify_switch_config_passes(self):
        rep = tail.Report(io.StringIO())
        tail.check_config(_config(verify_ground_truth={'enabled': True}), FIXTURE['scenario'], rep)
        self.assertEqual([n for v, n, _ in rep.rows if v == 'FAIL'], [])


@unittest.skipUnless(HAVE_PT, 'py_trees not installed (BT pythonpath)')
class TailDryRun(unittest.TestCase):
    """The real tree drives transfer -> verify -> return -> finalize to mission success, and the latent
    failure modes found by this dry-run stay fixed."""

    def _run(self, config, **inject):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        mission, summary, rig = tail.run_tail(config, FIXTURE['scenario'], tmp.name, **inject)
        mj = json.loads((Path(tmp.name) / 'mission.json').read_text())
        lines = (Path(tmp.name) / 'bt_events.jsonl').read_text().splitlines()
        return mission, summary, rig, mj, lines

    def test_all_dryrun_checks_pass_with_the_vr1_switches(self):
        rep = tail.Report(io.StringIO())
        with tempfile.TemporaryDirectory() as tmp:
            tail.check_tail(_config(verify_ground_truth={'enabled': True}), FIXTURE['scenario'], rep, tmp)
            tail.check_default_off(_config(verify_ground_truth={'enabled': True}), FIXTURE['scenario'], rep, tmp)
        failed = [(n, d) for v, n, d in rep.rows if v == 'FAIL']
        self.assertEqual(failed, [])
        self.assertGreaterEqual(len([1 for v, n, _ in rep.rows if v == 'PASS' and n.startswith('tail_')]), 12)

    def test_nominal_tail_events_and_mission_json(self):
        mission, summary, rig, mj, lines = self._run(_config(verify_ground_truth={'enabled': True}))
        self.assertEqual(mission.state, 'complete')
        self.assertEqual(mj['state'], 'complete')
        self.assertEqual((mj['verify_mode'], mj['transfer_verify_mode'], mj['return_mode']),
                         ('ground_truth', 'ground_truth', 'uncouple_at_cart_parking'))
        states = [e['state'] for e in summary['events']]
        for s in ('transfer_succeeded', 'inventory_verified', 'verify_succeeded', 'return_succeeded'):
            self.assertIn(s, states)
        self.assertEqual(states[-1], 'complete')
        # the uncouple happens in RETURN at cart_parking, AFTER the transfer + verify at the park pose
        self.assertLess(states.index('verify_succeeded'), states.index('return'))
        self.assertEqual(rig.mobile.calls, ['hold_cart_pin', 'uncouple'])
        # walkthrough timeline: every tail stage begins + succeeds with each step on the sim clock
        timeline = [json.loads(line) for line in lines]
        begins = [e['data']['stage'] for e in timeline if e['type'] == 'stage_begin']
        self.assertEqual(begins, ['transfer', 'verify', 'return'])
        ret_steps = [e['data']['label'] for e in timeline
                     if e['type'] == 'stage_step' and e['data']['stage'] == 'return']
        self.assertEqual(ret_steps, ['tow→cart park', 'brake cart', 'uncouple', 'back away', 'drive→home'])
        self.assertTrue(all('sim_time' in e['data'] for e in timeline if e['type'].startswith('stage_')))
        self.assertEqual(timeline[-1]['type'], 'mission_complete')
        self.assertEqual([e['type'] for e in mj['stage_timeline']], [e['type'] for e in timeline
                                                                     if e['type'] in tail_types()])

    def test_run356_config_as_is_times_out_in_verify(self):
        # Reproduces the predicted first live attempt without the vr1 verify switch.
        mission, _summary, _rig, _mj, _lines = self._run(_config())
        self.assertEqual(mission.state, 'failed')
        self.assertIn('verify timed out', mission.reason)

    def test_default_off_reproduces_transfer_evidence_mismatch(self):
        config = _config()
        config['transfer_ground_truth'] = dict(config['transfer_ground_truth'], enabled=False)
        mission, *_ = self._run(config)
        self.assertEqual(mission.state, 'failed')
        self.assertIn('transfer evidence mismatch', mission.reason)

    def test_backaway_waypoint_is_driven_uncoupled(self):
        poses = dict(FIXTURE['config']['mission_poses'], cart_parking_backaway=[-0.6, 13.8, math.pi])
        mission, _s, rig, _mj, _l = self._run(_config(verify_ground_truth={'enabled': True}, mission_poses=poses))
        self.assertEqual(mission.state, 'complete')
        self.assertEqual([(g['goal'], g['coupled']) for g in rig.nav.goals],
                         [([-1.2, 15.25, 0.0], True), ([-0.6, 13.8, math.pi], False),
                          ([-5.73, 16.0, 0.0], False)])


def tail_types():
    from twin.restock.bt.driver import STAGE_EVENT_TYPES
    return STAGE_EVENT_TYPES


class HomeLegNavigation(unittest.TestCase):
    """The lidar wiring proof and the uncoupled home leg with the REAL controller on the bay map."""

    def test_legacy_lidar_wiring_rotates_returns_and_the_fix_does_not(self):
        from twin.restock.tail_nav import LIDAR_XYZ, lidar_wall_probe
        self.assertEqual(lidar_wall_probe((LIDAR_XYZ[0], LIDAR_XYZ[1], 0.0)), (3.0, 3.0))
        lo, hi = lidar_wall_probe(LIDAR_XYZ)          # the legacy (x, y, z) tuple read as (x, y, yaw)
        self.assertGreater(hi - lo, 1.0)
