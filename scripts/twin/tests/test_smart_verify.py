"""CPU tests for smart post-restock verification (no Isaac/ROS/GPU/Blender).

Covers the planogram slot index + its consistency with store_environment.layout,
the per-policy slot diff, honest evidence bases, residual reset jobs in the
sim-smart-job-feed-v1 contract shape, the end-to-end report from a fixture mission,
and the lazy store-vision cross-check (degradation + real read-only integration).
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock import smart_verify as sv
from twin.restock import smart_verify_storevision as svx
from twin.restock.inventory import Inventory, Job
from twin.restock.mission import Mission, Result

ROOT = Path(__file__).resolve().parents[2]
PLANOGRAM = ROOT / 'planogram.json'


def _plan():
    return sv.load_planogram(PLANOGRAM)


# ---- a fixture adapter identical in spirit to test_restock_mission.Fixture ----
class _Fixture:
    def __init__(self, job):
        self.t = 0.
        self.job = job

    def start(self, action, job):
        self.action = action

    def step(self, dt):
        self.t += dt
        ev = dict(source='test_fixture', observation_id=f'{self.action}-{self.t}',
                  simulation_time=self.t, loaded=True, coupled=True, arrived=True,
                  stationary=True, released=True, home=True, sku=self.job.sku,
                  quantity=self.job.quantity, location=self.job.shelf_id,
                  transfer_confirmed=True)
        return Result('succeeded', ev, '')

    def stop(self):
        pass


def _run_mission(tmp, *, dest='G3_back_s2', sku='juice_orange', qty=1,
                 seed_dest=0, source='backstock_00'):
    """Run a complete fixture mission; return (mission_summary_dict, inventory_path)."""
    inv_path = str(Path(tmp) / 'inventory.sqlite')
    inv = Inventory(inv_path)
    inv.seed(source, sku, qty)
    if seed_dest:
        inv.seed(dest, sku, seed_dest)
    job = Job('restock-one', sku, qty, source, dest)
    m = Mission(inv, job, _Fixture(job), fixture_mode=True)
    m.start()
    while not m.terminal:
        m.tick(.1)
    summary = {'state': m.state, 'reason': m.reason, 'events': inv.events('restock-one')}
    inv.close()
    return summary, inv_path


class PlanogramIndexTests(unittest.TestCase):
    def test_index_covers_known_demo_slot(self):
        idx = sv.planogram_slot_index(_plan())
        self.assertIn('G3_back_s2', idx)
        self.assertEqual(idx['G3_back_s2'].expected, {'juice_orange': 11})

    def test_slot_ids_consistent_with_store_environment_layout(self):
        """The slot-id scheme and sku sets must match store_environment.layout exactly
        so the planogram index and the mission's stocking targets can never drift."""
        from store_environment import layout
        plan = _plan()
        idx = sv.planogram_slot_index(plan)
        lay = layout(plan)
        for st in lay['stocking_stations']:
            for slot in st['slots']:
                self.assertIn(slot['id'], idx, f"layout slot {slot['id']} missing from index")
                self.assertEqual(set(slot['sku_ids']), set(idx[slot['id']].expected),
                                 f"sku set drift at {slot['id']}")

    def test_aggregate_rejects_bad_items(self):
        with self.assertRaises(ValueError):
            sv._aggregate_items([['sku', -1]])
        with self.assertRaises(ValueError):
            sv._aggregate_items([['', 1]])

    def test_duplicate_sku_facings_aggregate(self):
        self.assertEqual(sv._aggregate_items([['a', 2], ['a', 3], ['b', 1]]),
                         {'a': 5, 'b': 1})


class DiffTests(unittest.TestCase):
    def _exp(self, expected):
        return sv.SlotExpectation(slot_id='S', expected=expected)

    def _obs(self, observed, basis='full_shelf_sensor'):
        return sv.SlotObservation(slot_id='S', observed=observed, evidence_basis=basis)

    def test_at_least_shortfall_and_compliant(self):
        v = sv.diff_slot(self._exp({'a': 5}), self._obs({'a': 3}))
        self.assertFalse(v.compliant)
        self.assertEqual(v.diffs[0].kind, 'shortfall')
        self.assertEqual(v.diffs[0].delta, 2)
        v2 = sv.diff_slot(self._exp({'a': 5}), self._obs({'a': 6}))  # overstock ok under at_least
        self.assertTrue(v2.compliant)
        self.assertEqual(v2.diffs[0].kind, 'compliant')

    def test_exact_flags_overstock(self):
        v = sv.diff_slot(self._exp({'a': 5}), self._obs({'a': 6}), quantity_policy='exact')
        self.assertFalse(v.compliant)
        self.assertEqual(v.diffs[0].kind, 'overstock')

    def test_identity_only_presence(self):
        v = sv.diff_slot(self._exp({'a': 5}), self._obs({'a': 1}), quantity_policy='identity')
        self.assertTrue(v.compliant)
        v2 = sv.diff_slot(self._exp({'a': 5}), self._obs({}), quantity_policy='identity')
        self.assertFalse(v2.compliant)
        self.assertEqual(v2.diffs[0].kind, 'shortfall')

    def test_misplaced_sku(self):
        v = sv.diff_slot(self._exp({'a': 2}), self._obs({'a': 2, 'b': 3}))
        self.assertFalse(v.compliant)
        kinds = {d.sku: d.kind for d in v.diffs}
        self.assertEqual(kinds['a'], 'compliant')
        self.assertEqual(kinds['b'], 'misplaced')

    def test_bad_quantity_policy(self):
        with self.assertRaises(ValueError):
            sv.diff_slot(self._exp({'a': 1}), self._obs({'a': 1}), quantity_policy='nope')


class ObservationTests(unittest.TestCase):
    def test_bad_evidence_basis_rejected(self):
        with self.assertRaises(ValueError):
            sv.SlotObservation(slot_id='S', observed={}, evidence_basis='made_up')

    def test_from_verify_location_mismatch(self):
        ev = dict(sku='a', quantity=1, location='X', observation_id='o1')
        with self.assertRaises(ValueError):
            sv.observation_from_verify(ev, 'Y')

    def test_from_verify_basis_and_ids(self):
        ev = dict(sku='a', quantity=2, location='S', observation_id='o1', confidence=0.9)
        obs = sv.observation_from_verify(ev, 'S')
        self.assertEqual(obs.evidence_basis, 'mission_verify_sensor')
        self.assertEqual(obs.observed, {'a': 2})
        self.assertEqual(obs.observation_ids, ('o1',))
        self.assertEqual(obs.confidence, 0.9)

    def test_from_bookkeeping(self):
        stock = {('S', 'a'): 3, ('S', 'b'): 0, ('OTHER', 'a'): 9}
        obs = sv.observation_from_bookkeeping(stock, 'S')
        self.assertEqual(obs.observed, {'a': 3})  # zero and other-slot excluded
        self.assertEqual(obs.evidence_basis, 'bookkeeping')


class ResetJobTests(unittest.TestCase):
    def test_job_feed_contract_fields(self):
        exp = sv.SlotExpectation(slot_id='G3_back_s2', expected={'juice_orange': 11})
        obs = sv.SlotObservation(slot_id='G3_back_s2', observed={'juice_orange': 1},
                                 evidence_basis='bookkeeping')
        v = sv.diff_slot(exp, obs)
        jobs = sv.reset_jobs_for(v, source_bin='backstock_00')
        self.assertEqual(len(jobs), 1)
        feed = jobs[0].to_job_feed()
        self.assertEqual(set(feed), {'sku', 'quantity', 'source_bin', 'shelf_id',
                                     'slot', 'priority', 'kind', 'provenance'})
        self.assertEqual(feed['sku'], 'juice_orange')
        self.assertEqual(feed['quantity'], 10)
        self.assertEqual(feed['kind'], 'restock')
        self.assertEqual(feed['shelf_id'], feed['slot'])
        self.assertEqual(feed['provenance']['emitted_by'], 'smart-verify')

    def test_remove_job_for_misplaced(self):
        exp = sv.SlotExpectation(slot_id='S', expected={'a': 2})
        obs = sv.SlotObservation(slot_id='S', observed={'a': 2, 'b': 3},
                                 evidence_basis='full_shelf_sensor')
        v = sv.diff_slot(exp, obs)
        jobs = sv.reset_jobs_for(v)
        kinds = {j.sku: (j.kind, j.quantity) for j in jobs}
        self.assertEqual(kinds['b'], ('remove', 3))

    def test_compliant_slot_no_jobs(self):
        exp = sv.SlotExpectation(slot_id='S', expected={'a': 2})
        obs = sv.SlotObservation(slot_id='S', observed={'a': 2}, evidence_basis='bookkeeping')
        self.assertEqual(sv.reset_jobs_for(sv.diff_slot(exp, obs)), [])


class ReportTests(unittest.TestCase):
    def test_weakest_basis_and_overall_and(self):
        plan = _plan()
        strong = sv.SlotObservation(slot_id='G3_back_s2', observed={'juice_orange': 11},
                                    evidence_basis='full_shelf_sensor')
        weak = sv.SlotObservation(slot_id='G1L_b0_s0', observed={'cereal_choco': 2, 'cereal_honey': 3},
                                  evidence_basis='bookkeeping')
        rep = sv.build_report(job_id='j', plan=plan, observations=[strong, weak])
        self.assertEqual(rep.evidence_basis, 'bookkeeping')  # weakest wins
        self.assertTrue(rep.compliant)  # both slots fully stocked

    def test_mission_verify_evidence_selection(self):
        self.assertIsNone(sv.mission_verify_evidence({'events': [{'state': 'collect_succeeded'}]}))
        ev = {'sku': 'a', 'quantity': 1, 'location': 'S', 'observation_id': 'o'}
        mission = {'events': [{'state': 'verify_succeeded', 'evidence': ev},
                              {'state': 'inventory_verified', 'evidence': dict(ev, observation_id='o2')}]}
        chosen = sv.mission_verify_evidence(mission)
        self.assertEqual(chosen['observation_id'], 'o2')  # last verified wins

    def test_report_from_mission_refuses_unverified(self):
        with self.assertRaises(ValueError):
            sv.report_from_mission(mission={'events': []}, plan=_plan(), slot_id='G3_back_s2')

    def test_end_to_end_from_fixture_mission_shortfall(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, inv_path = _run_mission(tmp)
            # bookkeeping basis: only the 1 landed unit is at the slot; planogram wants 11.
            stock = sv.load_stock(inv_path)
            rep = sv.report_from_mission(mission=summary, plan=_plan(), slot_id='G3_back_s2',
                                         stock=stock, source_bin='backstock_00')
            self.assertFalse(rep.compliant)
            self.assertEqual(rep.evidence_basis, 'bookkeeping')
            self.assertEqual(len(rep.residual_jobs), 1)
            self.assertEqual(rep.residual_jobs[0].quantity, 10)
            d = rep.to_dict()
            self.assertEqual(d['schema'], 'smart-verify.v1')
            json.dumps(d)  # must be JSON-serializable

    def test_end_to_end_compliant_when_preseeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, inv_path = _run_mission(tmp, seed_dest=10)  # 10 + 1 landed = 11 == target
            stock = sv.load_stock(inv_path)
            rep = sv.report_from_mission(mission=summary, plan=_plan(), slot_id='G3_back_s2',
                                         stock=stock)
            self.assertTrue(rep.compliant)
            self.assertEqual(rep.residual_jobs, [])

    def test_load_stock_is_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, inv_path = _run_mission(tmp)
            stock = sv.load_stock(inv_path)
            self.assertIn(('G3_back_s2', 'juice_orange'), stock)


class StoreVisionCrossCheckTests(unittest.TestCase):
    def _report(self):
        exp = sv.SlotExpectation(slot_id='G3_back_s2', expected={'juice_orange': 11})
        obs = sv.SlotObservation(slot_id='G3_back_s2', observed={'juice_orange': 1},
                                 evidence_basis='bookkeeping')
        return sv.build_report(job_id='j', plan={'gondolas': [], 'endcaps': [
            {'id': 'G3_back', 'shelves': [{}, {}, {'items': [['juice_orange', 11]]}]}]},
            observations=[obs])

    def test_degrades_when_src_missing(self):
        res = svx.cross_check(self._report(), storevision_src='/nonexistent/store-vision/src')
        self.assertFalse(res['available'])
        self.assertIn('reason', res)

    def test_compliance_shim_mapping(self):
        v = sv.diff_slot(sv.SlotExpectation(slot_id='S', expected={'a': 5}),
                         sv.SlotObservation(slot_id='S', observed={'a': 2}, evidence_basis='bookkeeping'))
        shim = svx._compliance_from_verdict(v)
        self.assertEqual(shim.status, 'understocked')
        self.assertFalse(shim.compliant)
        self.assertEqual(shim.facings_expected, 5)
        self.assertEqual(shim.facings_present, 2)

    def test_real_integration_if_available(self):
        """If the sibling store-vision checkout is importable, the cross-check must run
        its REAL read-only verify_from_compliance and return per-slot verifications."""
        src = svx.DEFAULT_STOREVISION_SRC
        verify, _, reason = svx._import_planoreset(src)
        if verify is None:
            self.skipTest(f'store-vision not importable: {reason}')
        res = svx.cross_check(self._report(), storevision_src=src)
        self.assertTrue(res['available'], res.get('reason'))
        self.assertEqual(len(res['verifications']), 1)
        self.assertEqual(res['verifications'][0]['slot_id'], 'G3_back_s2')
        self.assertFalse(res['verifications'][0]['ok'])


class CliTests(unittest.TestCase):
    def _write_mission(self, tmp):
        summary, inv_path = _run_mission(tmp)
        mission_path = Path(tmp) / 'mission.json'
        mission_path.write_text(json.dumps(summary) + '\n')
        return mission_path, inv_path

    def test_cli_default_basis_and_exit_zero(self):
        import run_smart_verify
        with tempfile.TemporaryDirectory() as tmp:
            mission_path, _ = self._write_mission(tmp)
            out = Path(tmp) / 'sv.json'
            rc = run_smart_verify.main(['--mission', str(mission_path), '--planogram',
                                        str(PLANOGRAM), '--out', str(out), '--quiet'])
            self.assertEqual(rc, 0)  # additional report never fails via exit code
            d = json.loads(out.read_text())
            self.assertEqual(d['evidence_basis'], 'mission_verify_sensor')
            self.assertFalse(d['compliant'])

    def test_cli_bookkeeping_and_job_feed_out(self):
        import run_smart_verify
        with tempfile.TemporaryDirectory() as tmp:
            mission_path, inv_path = self._write_mission(tmp)
            feed = Path(tmp) / 'jobs.json'
            rc = run_smart_verify.main(['--mission', str(mission_path), '--planogram',
                                        str(PLANOGRAM), '--inventory', str(inv_path),
                                        '--source-bin', 'backstock_00',
                                        '--job-feed-out', str(feed), '--quiet'])
            self.assertEqual(rc, 0)
            jobs = json.loads(feed.read_text())
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['kind'], 'restock')
            self.assertEqual(jobs[0]['quantity'], 10)
            self.assertEqual(jobs[0]['source_bin'], 'backstock_00')

    def test_cli_explicit_slot_overrides(self):
        import run_smart_verify
        with tempfile.TemporaryDirectory() as tmp:
            mission_path, _ = self._write_mission(tmp)
            out = Path(tmp) / 'sv.json'
            # Explicit slot that disagrees with the verify location is rejected by the
            # verify-observation guard (location != slot).
            with self.assertRaises(ValueError):
                run_smart_verify.main(['--mission', str(mission_path), '--planogram',
                                       str(PLANOGRAM), '--slot', 'G1L_b0_s0',
                                       '--out', str(out), '--quiet'])


if __name__ == '__main__':
    unittest.main()
