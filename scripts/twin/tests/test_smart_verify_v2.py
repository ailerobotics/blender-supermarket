"""CPU tests for sim-smart-verify-v2 (no Isaac/ROS/GPU/Blender/pydantic).

Covers the post-transfer shelf observation export (resolved verify region + saved
verify frames), the lane-1 ``job_plan/v1`` residual jobs (round-tripped through the
real ``job_from_plan`` loader for the closed loop), the ``report``/``strict``/``block``
CLI modes, and the store-vision realogram cross-check (degradation always; the real
read-only venv subprocess when the sibling store-vision ``.venv`` is present). A
recorded saved run (``coldwin22`` under the mission dataset) is used when available so
the export is proven on REAL verify frames, not only a synthetic fixture.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock import smart_verify as sv
from twin.restock import shelf_realogram as sr
from twin.restock import smart_verify_storevision as svx
from twin.restock import job_from_plan as jfp

ROOT = Path(__file__).resolve().parents[2]
PLANOGRAM = ROOT / 'planogram.json'
SAVED_RUN = Path('/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin22')


def _gt_verify_evidence(*, sku='juice_orange', qty=1, slot='G3_back_s2',
                        start=373.75, end=373.78):
    """A ground-truth verify evidence block shaped like a real saved run's."""
    return {
        'observation_id': f'ground_truth_verify:{end}',
        'source': 'ground_truth', 'verify_mode': 'ground_truth',
        'simulation_time': end, 'verification_started': start,
        'sku': sku, 'quantity': qty, 'location': slot,
        'confirming_observation_ids': [f'gt:{start}', f'gt:{end}'],
        'ground_truth_check': {'unit_center_world': [-2.0027, 16.4264, 0.82],
                               'on_bay': True, 'tilt_deg': 0.0},
        'tolerances': {'bay_center_xy': [-2.0, 16.42], 'bay_half_xy': [0.35, 0.25],
                       'bay_top_z': 0.73},
    }


def _mission(evidence, *, state='complete'):
    return {
        'state': state,
        'stage_timeline': [
            {'type': 'stage_begin', 'data': {'stage': 'verify',
             'sim_time': evidence['verification_started']}},
            {'type': 'stage_succeeded', 'data': {'stage': 'verify',
             'sim_time': evidence['simulation_time']}},
        ],
        'events': [{'state': 'inventory_verified', 'evidence': evidence}],
    }


def _write_run_dir(tmp, evidence, *, frames=True):
    """A synthetic run dir: mission.json + (optional) mission-frames/frames.json."""
    run = Path(tmp)
    (run).mkdir(parents=True, exist_ok=True)
    (run / 'mission.json').write_text(json.dumps(_mission(evidence)))
    if frames:
        fdir = run / 'mission-frames'
        fdir.mkdir(exist_ok=True)
        # frames straddling the verify instant at ~0.33 s cadence
        manifest = []
        t = evidence['verification_started'] - 3.0
        i = 0
        while t <= evidence['simulation_time'] + 3.0:
            name = f'{i:06d}.png'
            (fdir / name).write_bytes(b'')
            manifest.append({'file': name, 'simulation_time': round(t, 4), 'sha256': f'h{i}'})
            t += 0.3333
            i += 1
        (fdir / 'frames.json').write_text(json.dumps(
            {'nominal_fps': 30, 'timing': 'variable', 'frames': manifest}))
    return run


class BayDeckTests(unittest.TestCase):
    def test_bay_and_deck(self):
        self.assertEqual(sr.bay_and_deck('G3_back_s2'), ('G3_back', 2))
        self.assertEqual(sr.bay_and_deck('G1L_b0_s0'), ('G1L_b0', 0))
        self.assertEqual(sr.bay_and_deck('weird'), ('weird', None))


class RegionTests(unittest.TestCase):
    def test_from_tolerances(self):
        r = sr.extract_verify_region(_gt_verify_evidence())
        self.assertTrue(r.available)
        self.assertEqual(r.source, 'verify_tolerances')
        self.assertEqual(r.center_xy, (-2.0, 16.42))
        self.assertEqual(r.size_xy, (0.7, 0.5))
        self.assertEqual(r.top_z, 0.73)
        self.assertTrue(r.on_bay)
        self.assertEqual(r.store_region(), [-2.35, 16.17, 0.7, 0.5])

    def test_placed_unit_only_when_no_bay(self):
        ev = _gt_verify_evidence()
        ev.pop('tolerances')
        r = sr.extract_verify_region(ev)
        self.assertEqual(r.source, 'placed_unit_only')
        self.assertEqual(r.center_xy, (-2.0027, 16.4264))
        self.assertIsNone(r.size_xy)
        self.assertFalse(r.available)
        self.assertIsNone(r.store_region())

    def test_unavailable_when_no_geometry(self):
        r = sr.extract_verify_region({'sku': 'a', 'quantity': 1, 'location': 'S'})
        self.assertEqual(r.source, 'unavailable')
        self.assertFalse(r.available)


class FrameTests(unittest.TestCase):
    def test_locate_pads_the_window(self):
        ev = _gt_verify_evidence()
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run_dir(tmp, ev)
            frames, basis = sr.locate_verify_frames(run, _mission(ev), ev)
            self.assertEqual(basis, sr.FRAME_WALKTHROUGH)
            self.assertTrue(frames)
            self.assertLessEqual(len(frames), 8)
            # every located frame sits inside the padded window and carries its path
            for f in frames:
                self.assertTrue(ev['verification_started'] - 2.0 - 1e-6
                                <= f.simulation_time <= ev['simulation_time'] + 2.0 + 1e-6)
                self.assertTrue(f.path.endswith('.png'))
                self.assertEqual(f.camera, sr.FRAME_WALKTHROUGH)

    def test_no_frames_dir_gives_none(self):
        ev = _gt_verify_evidence()
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run_dir(tmp, ev, frames=False)
            frames, basis = sr.locate_verify_frames(run, _mission(ev), ev)
            self.assertEqual((frames, basis), ([], sr.FRAME_NONE))


class BuildShelfRealogramTests(unittest.TestCase):
    def test_verify_basis_build(self):
        ev = _gt_verify_evidence()
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run_dir(tmp, ev)
            rg = sr.build_shelf_realogram(_mission(ev), ev, 'G3_back_s2', run_dir=run)
        self.assertEqual(rg.bay_id, 'G3_back')
        self.assertEqual(rg.deck_index, 2)
        self.assertEqual(rg.observed, {'juice_orange': 1})
        self.assertEqual(rg.evidence_basis, 'mission_verify_sensor')
        self.assertEqual(rg.frames_basis, sr.FRAME_WALKTHROUGH)
        self.assertTrue(rg.region.available)
        d = rg.to_dict()
        self.assertEqual(d['schema'], 'smart-verify.shelf-observation.v2')
        json.dumps(d)  # JSON-serialisable
        # the slot observation feeds the planogram diff unchanged
        obs = rg.to_slot_observation()
        self.assertEqual(obs.evidence_basis, 'mission_verify_sensor')
        self.assertEqual(obs.observed, {'juice_orange': 1})

    def test_shelf_record_basis(self):
        ev = _gt_verify_evidence()
        rec = {'slot_id': 'G3_back_s2', 'observed': {'juice_orange': 11}, 'confidence': 0.95}
        rg = sr.build_shelf_realogram(_mission(ev), ev, 'G3_back_s2', shelf_record=rec)
        self.assertEqual(rg.evidence_basis, 'full_shelf_sensor')
        self.assertEqual(rg.frames_basis, sr.FRAME_SHELF_SENSOR)
        self.assertEqual(rg.observed, {'juice_orange': 11})

    def test_extra_observed_merge_is_weaker_belief(self):
        ev = _gt_verify_evidence()
        rg = sr.build_shelf_realogram(_mission(ev), ev, 'G3_back_s2',
                                      extra_observed={'juice_orange': 3, 'cola': 0})
        # merged into observed, recorded in provenance; basis unchanged
        self.assertEqual(rg.observed, {'juice_orange': 4})
        self.assertEqual(rg.evidence_basis, 'mission_verify_sensor')
        self.assertEqual(rg.provenance['extra_observed_bookkeeping'], {'juice_orange': 3})


class JobPlanTests(unittest.TestCase):
    def _shortfall_report(self):
        exp = sv.SlotExpectation(slot_id='G3_back_s2', expected={'juice_orange': 11})
        obs = sv.SlotObservation(slot_id='G3_back_s2', observed={'juice_orange': 1},
                                 evidence_basis='mission_verify_sensor')
        return sv.build_report(job_id='restock-one',
                               plan={'gondolas': [], 'endcaps': [
                                   {'id': 'G3_back', 'shelves': [{}, {},
                                    {'items': [['juice_orange', 11]]}]}]},
                               observations=[obs], source_bin='backstock_00')

    def test_to_job_plan_shape(self):
        rep = self._shortfall_report()
        plan = rep.residual_jobs[0].to_job_plan(store_config='demo', generated_at='2026-01-01T00:00:00Z')
        self.assertEqual(plan['schema'], 'storevision.restock.job_plan/v1')
        self.assertEqual(plan['sku'], 'juice_orange')
        self.assertEqual(plan['quantity'], 10)
        self.assertEqual(plan['source']['location'], 'backstock_00')
        self.assertEqual(plan['destination']['slot_id'], 'G3_back_s2')
        self.assertEqual(plan['provenance']['planner'], 'sim-smart-verify-v2')

    def test_job_plan_round_trips_through_lane1_loader(self):
        """The closed loop: a residual job_plan MUST load + resolve via job_from_plan."""
        rep = self._shortfall_report()
        consumable, advisory = sv.residual_job_plans(rep, store_config='demo')
        self.assertEqual(len(consumable), 1)
        self.assertEqual(advisory, [])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'job.json'
            p.write_text(json.dumps(consumable[0]))
            loaded = jfp.load_plan(p)   # schema + shape check
            job, meta = jfp.resolve_job(loaded, PLANOGRAM)
            self.assertEqual(job.sku, 'juice_orange')
            self.assertEqual(job.quantity, 10)
            self.assertEqual(job.shelf_id, 'G3_back_s2')
            self.assertEqual(job.source_bin, 'backstock_00')

    def test_remove_is_advisory_not_consumable(self):
        exp = sv.SlotExpectation(slot_id='G3_back_s2', expected={'juice_orange': 11})
        obs = sv.SlotObservation(slot_id='G3_back_s2',
                                 observed={'juice_orange': 11, 'cola': 3},
                                 evidence_basis='full_shelf_sensor')
        rep = sv.build_report(job_id='j', plan={'gondolas': [], 'endcaps': [
            {'id': 'G3_back', 'shelves': [{}, {}, {'items': [['juice_orange', 11]]}]}]},
            observations=[obs], source_bin='backstock_00')
        consumable, advisory = sv.residual_job_plans(rep)
        self.assertEqual(consumable, [])  # a remove has no mission action
        self.assertEqual(len(advisory), 1)
        self.assertEqual(advisory[0]['provenance']['kind'], 'remove')

    def test_missing_source_bin_is_advisory(self):
        rep = sv.build_report(job_id='j', plan={'gondolas': [], 'endcaps': [
            {'id': 'G3_back', 'shelves': [{}, {}, {'items': [['juice_orange', 11]]}]}]},
            observations=[sv.SlotObservation(slot_id='G3_back_s2',
                          observed={'juice_orange': 1}, evidence_basis='bookkeeping')])
        consumable, advisory = sv.residual_job_plans(rep)  # no source_bin stamped
        self.assertEqual(consumable, [])
        self.assertEqual(len(advisory), 1)


class RealogramCrossCheckTests(unittest.TestCase):
    def _realogram(self):
        ev = _gt_verify_evidence()
        return sr.build_shelf_realogram(_mission(ev), ev, 'G3_back_s2')

    def test_degrades_when_venv_missing(self):
        exp = sv.planogram_slot_index(sv.load_planogram(PLANOGRAM))['G3_back_s2']
        res = svx.cross_check_realogram(self._realogram(), exp,
                                        venv_python='/nonexistent/python')
        self.assertFalse(res['available'])
        self.assertIn('reason', res)

    def test_real_realogram_if_venv_available(self):
        py = svx._venv_python()
        if py is None:
            self.skipTest('store-vision .venv python not present')
        exp = sv.planogram_slot_index(sv.load_planogram(PLANOGRAM))['G3_back_s2']
        res = svx.cross_check_realogram(self._realogram(), exp)
        self.assertTrue(res.get('available'), res.get('reason'))
        v = res['verifications']
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]['slot_id'], 'G3_back_s2')
        self.assertFalse(v[0]['ok'])  # 1 facing observed vs 11 expected
        self.assertEqual(v[0]['status'], 'facing_changed')


class CliV2Tests(unittest.TestCase):
    def test_run_dir_report_mode(self):
        import run_smart_verify
        ev = _gt_verify_evidence()
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run_dir(tmp, ev)
            rc = run_smart_verify.main(['--run-dir', str(run), '--planogram',
                                        str(PLANOGRAM), '--source-bin', 'backstock_00',
                                        '--quiet'])
            self.assertEqual(rc, 0)
            sv_json = json.loads((run / 'smart-verify.json').read_text())
            self.assertEqual(sv_json['mode'], 'report')
            self.assertFalse(sv_json['compliant'])
            self.assertEqual(sv_json['shelf_observation']['region']['source'], 'verify_tolerances')
            # report mode writes no consumable job-plan file
            self.assertFalse((run / 'reset-jobs.job-plan.json').exists())
            obs = json.loads((run / 'shelf-observation.json').read_text())
            self.assertEqual(obs['schema'], 'smart-verify.shelf-observation.v2')

    def test_strict_mode_emits_consumable_job_plan(self):
        import run_smart_verify
        ev = _gt_verify_evidence()
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run_dir(tmp, ev)
            rc = run_smart_verify.main(['--run-dir', str(run), '--planogram',
                                        str(PLANOGRAM), '--source-bin', 'backstock_00',
                                        '--mode', 'strict', '--quiet'])
            self.assertEqual(rc, 0)  # strict is NOT a blocker
            jobs = json.loads((run / 'reset-jobs.job-plan.json').read_text())
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['schema'], 'storevision.restock.job_plan/v1')

    def test_block_mode_exit_nonzero_when_noncompliant(self):
        import run_smart_verify
        ev = _gt_verify_evidence()
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run_dir(tmp, ev)
            rc = run_smart_verify.main(['--run-dir', str(run), '--planogram',
                                        str(PLANOGRAM), '--source-bin', 'backstock_00',
                                        '--mode', 'block', '--quiet'])
            self.assertEqual(rc, 2)  # smart-verify's OWN gate

    def test_block_mode_exit_zero_when_compliant(self):
        import run_smart_verify
        ev = _gt_verify_evidence()
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run_dir(tmp, ev)
            rec = Path(tmp) / 'shelf.json'
            rec.write_text(json.dumps({'slot_id': 'G3_back_s2',
                                       'observed': {'juice_orange': 11}}))
            rc = run_smart_verify.main(['--run-dir', str(run), '--planogram',
                                        str(PLANOGRAM), '--shelf-record', str(rec),
                                        '--mode', 'block', '--quiet'])
            self.assertEqual(rc, 0)


class RecordedRunTests(unittest.TestCase):
    """Prove the export on a REAL saved run's verify frames when it is present."""

    def test_coldwin22_shelf_observation(self):
        if not (SAVED_RUN / 'mission.json').is_file():
            self.skipTest(f'saved run not present: {SAVED_RUN}')
        mission = json.loads((SAVED_RUN / 'mission.json').read_text())
        ev = sv.mission_verify_evidence(mission)
        self.assertIsNotNone(ev, 'coldwin22 must carry a succeeded verify')
        rg = sr.build_shelf_realogram(mission, ev, ev['location'], run_dir=SAVED_RUN)
        self.assertEqual(rg.region.source, 'verify_tolerances')
        self.assertTrue(rg.region.available)
        # real chase-camera frames bracket the verify instant
        self.assertEqual(rg.frames_basis, sr.FRAME_WALKTHROUGH)
        self.assertTrue(rg.frames)
        for f in rg.frames:
            self.assertTrue(Path(f.path).is_file(), f.path)


if __name__ == '__main__':
    unittest.main()
