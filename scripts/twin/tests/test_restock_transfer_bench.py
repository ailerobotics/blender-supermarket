"""CPU tests for the warm ``start_at='transfer'`` bench (recorded bay-park start staging).

Proves offline: the staged robot pose equals the recorded park->transfer handover of runs 351, 353,
354 and 356 within tolerance; the staged cart sits on the hitch exactly where those runs measured the
deck (TRANSFER_CAPTURE) and passes the Coupler attach gate; the carton rests on the DECK at the fm479
slot; the mission/tree enter transfer directly with the skipped prefix recorded as ``*_shortcut``; and
the bench is labelled in mission.json and the walkthrough caption. The absent-config path is the
identity. ``scripts/check_transfer_bench_staging.py`` repeats the pose proof against the raw run files.
"""
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.mobile import Coupler, DockObservation, MobileSpec, Pose2
from twin.restock.inventory import Inventory, Job
from twin.restock.mission import Mission, Result
from twin.restock import shortcut
from twin.restock.shortcut import (DECK_TOP_Z_WORLD, RECORDED_BAY_PARK_CART_IN_BASE_XY,
                                   RECORDED_BAY_PARK_ROBOT_POSE, RECORDED_BAY_PARK_RUNS,
                                   plan_start_state, shortcut_summary, staged_scenario,
                                   walkthrough_caption)

try:
    import py_trees  # noqa: F401
    HAVE_PT = True
except ImportError:
    HAVE_PT = False

XY_TOL = 0.01                 # m   (nav-trace resolution 1 mm; run spread 6 mm)
YAW_TOL = 0.01                # rad (nav-trace resolution 0.1 deg; run spread 0.7 deg)
SKIPPED = ['collect', 'couple', 'outbound', 'park']


def mission_scenario():
    """The fields of the bay-route mission-scenario.json that the staging reads (dock spawn)."""
    return {'job': {'job_id': 'restock-one', 'sku': 'juice_orange', 'quantity': 1,
                    'source_bin': 'backstock_00', 'shelf_id': 'G3_back_s2'},
            'handling_unit': {'dimensions_m': [0.6, 0.4, 0.16]},
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


class StagedPoseTests(unittest.TestCase):
    def setUp(self):
        self.variant, self.record = staged_scenario(mission_scenario(), bench_config())
        self.st = self.variant['initial_staging']

    def test_absent_staging_is_identity(self):
        sc = mission_scenario()
        out, record = staged_scenario(sc, {'start_at': 'transfer'})
        self.assertIs(out, sc)
        self.assertIsNone(record)

    def test_robot_pose_matches_every_recorded_handover(self):
        rx, ry, ryaw = self.st['robot_base_pose']
        self.assertEqual(sorted(RECORDED_BAY_PARK_RUNS), ['run-351', 'run-353', 'run-354', 'run-356'])
        for run, (x, y, yaw) in RECORDED_BAY_PARK_RUNS.items():
            self.assertLessEqual(math.hypot(rx - x, ry - y), XY_TOL, run)
            self.assertLessEqual(yaw_err(ryaw, yaw), YAW_TOL, run)
        self.assertEqual(tuple(self.st['robot_base_pose']), RECORDED_BAY_PARK_ROBOT_POSE)

    def test_cart_is_on_the_hitch_where_the_runs_measured_the_deck(self):
        robot = Pose2(*self.st['robot_base_pose'])
        cart = Pose2(*self.st['cart_body_pose'])
        spec = MobileSpec()
        self.assertLess(math.dist(robot.point(spec.hitch_x), cart.point(spec.cart_hitch_x)), 1e-9)
        # Deck centre (cart body origin) in the robot frame == TRANSFER_CAPTURE cart=(-1.27, 0.00).
        dx, dy = cart.x - robot.x, cart.y - robot.y
        c, s = math.cos(robot.yaw), math.sin(robot.yaw)
        local = (c * dx + s * dy, -s * dx + c * dy)
        self.assertLessEqual(math.dist(local, RECORDED_BAY_PARK_CART_IN_BASE_XY), 0.005)
        self.assertLessEqual(yaw_err(cart.yaw, robot.yaw), 1e-9)       # articulation 0

    def test_staged_train_passes_the_coupler_attach_gate(self):
        spec = MobileSpec()
        robot, cart = Pose2(*self.st['robot_base_pose']), Pose2(*self.st['cart_body_pose'])
        obs = DockObservation(robot, cart, spec.hitch_z, spec.hitch_z, 0.0, 0.0, 0.0, 0.0)
        made = []
        Coupler(spec).attach(obs, lambda: made.append(True))
        self.assertEqual(made, [True])

    def test_carton_rests_on_the_deck_at_the_fm479_slot(self):
        plan = plan_start_state(self.variant, 'transfer')
        self.assertTrue(plan.couple_cart)
        self.assertEqual(list(plan.skipped), SKIPPED)
        cx, cy, cyaw = self.st['cart_body_pose']
        self.assertAlmostEqual(plan.carton_position[0], cx + 0.18 * math.cos(cyaw), places=9)
        self.assertAlmostEqual(plan.carton_position[1], cy + 0.18 * math.sin(cyaw), places=9)
        self.assertAlmostEqual(plan.carton_position[2], DECK_TOP_Z_WORLD + 0.16 / 2, places=9)
        self.assertAlmostEqual(DECK_TOP_Z_WORLD, 0.20 + 0.05 / 2)       # deck centre z + half deck
        # carton keeps its on-cart yaw (-pi/2 on a yaw-0 cart), turned with the staged cart yaw.
        w, _, _, z = plan.carton_orientation_wxyz
        self.assertLessEqual(yaw_err(2 * math.atan2(z, w), -math.pi / 2 + cyaw), 1e-9)

    def test_record_names_the_workaround(self):
        self.assertEqual(self.record['mode'], 'start_phase_shortcut:recorded_bay_park')
        self.assertTrue(self.record['coupled'])
        self.assertEqual(self.record['source_runs'], sorted(RECORDED_BAY_PARK_RUNS))
        self.assertEqual(self.record['carton_load_center_body_xy'], [0.18, 0.0])

    def test_scenario_is_not_mutated(self):
        sc = mission_scenario()
        staged_scenario(sc, bench_config())
        self.assertEqual(sc, mission_scenario())

    def test_overrides_and_refusals(self):
        variant, _ = staged_scenario(mission_scenario(), bench_config(start_staging={
            'preset': 'recorded_bay_park', 'robot_base_pose': [-2.0, 15.25, 0.0],
            'carton_load_center_body_xy': [0.1, 0.0]}))
        self.assertEqual(variant['initial_staging']['robot_base_pose'], [-2.0, 15.25, 0.0])
        self.assertEqual(variant['initial_staging']['cart_load_center_body_xy'], [0.1, 0.0])
        for bad in (bench_config(start_at='couple'), bench_config(start_staging='nope'),
                    bench_config(start_staging={'coupled': False}),
                    bench_config(start_staging={'articulation': 0.5}),
                    bench_config(start_staging={'robot_base_pose': [0, 0]}), bench_config(start_staging=3)):
            with self.assertRaises(ValueError):
                staged_scenario(mission_scenario(), bad)


class LabelTests(unittest.TestCase):
    def test_caption_only_for_a_shortcut_run(self):
        self.assertIsNone(walkthrough_caption({}))
        text = walkthrough_caption(bench_config())
        self.assertIn('start_at=transfer', text)
        self.assertIn('collect, couple, outbound, park SYNTHESIZED', text)
        self.assertIn('recorded_bay_park', text)

    def test_summary_fields(self):
        _, record = staged_scenario(mission_scenario(), bench_config())
        out = shortcut_summary({'start_at': 'transfer', 'skipped': SKIPPED}, bench_config(), record)
        self.assertEqual(out['start_at'], 'transfer')
        self.assertEqual(out['start_staging_mode'], 'start_phase_shortcut:recorded_bay_park')
        self.assertEqual(out['shortcut']['staging'], record)
        self.assertEqual(out['walkthrough_caption'], walkthrough_caption(bench_config()))

    def test_recorder_persists_caption_only_when_set(self):
        import json
        import numpy as np
        from twin.restock.recording import Recorder
        with tempfile.TemporaryDirectory() as tmp:
            for caption in (None, 'BENCH RUN'):
                rec = Recorder(Path(tmp) / f'f-{bool(caption)}', fps=30, caption=caption)
                rec.add(np.zeros((4, 4, 3), np.uint8), render_id='a', simulation_time=0.1)
                data = json.loads((rec.directory / 'frames.json').read_text())
                self.assertEqual(data.get('caption'), caption)
                self.assertEqual(Recorder.from_manifest(rec.directory / 'frames.json').caption, caption)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg not on PATH')
    def test_captioned_encode_keeps_frames_and_timing(self):
        import numpy as np
        from twin.restock.recording import Recorder
        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder(Path(tmp) / 'frames', fps=30, caption=walkthrough_caption(bench_config()))
            for i in range(6):
                rec.add(np.full((64, 96, 3), 40 * i, np.uint8), render_id=str(i), simulation_time=0.3 * (i + 1))
            info = rec.encode(Path(tmp) / 'walkthrough.mp4', ffmpeg=shutil.which('ffmpeg'))
            self.assertEqual(info['frames'], 6)
            self.assertEqual(info['caption'], walkthrough_caption(bench_config()))
            self.assertTrue((Path(tmp) / 'walkthrough.caption.srt').is_file())


class _Fixture:
    """Records which stage the mission begins; every stage succeeds with fixture evidence."""
    def __init__(self):
        self.begun, self.time, self.action = [], 0., None

    def start(self, action, job):
        self.begun.append(action)
        self.action, self.job = action, job

    def step(self, dt):
        self.time += dt
        return Result('succeeded', dict(
            source='test_fixture', observation_id=f'{self.action}-{self.time}', simulation_time=self.time,
            loaded=True, coupled=True, arrived=True, stationary=True, released=True, home=True,
            sku=self.job.sku, quantity=self.job.quantity, location=self.job.shelf_id,
            transfer_confirmed=True), '')

    def stop(self):
        pass


class EntersTransferDirectlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inv = Inventory(str(Path(self.tmp.name) / 'stock.sqlite'))
        self.inv.seed('backstock_00', 'juice_orange', 1)
        self.job = Job('restock-one', 'juice_orange', 1, 'backstock_00', 'G3_back_s2')

    def tearDown(self):
        self.inv.close()
        self.tmp.cleanup()

    def _check(self, mission, adapter):
        self.assertEqual(mission.state, 'complete')
        self.assertEqual(adapter.begun[0], 'transfer')
        self.assertEqual(adapter.begun, ['transfer', 'verify', 'return'])
        events = [e['state'] for e in self.inv.events('restock-one')]
        self.assertEqual(events[1:5], [s + '_shortcut' for s in SKIPPED])
        for s in SKIPPED:
            self.assertNotIn(s + '_succeeded', events)
        self.assertEqual(mission.shortcut, {'start_at': 'transfer', 'skipped': SKIPPED})
        self.assertEqual(self.inv.count('G3_back_s2', 'juice_orange'), 1)   # verify still commits

    def test_state_machine_enters_transfer(self):
        adapter = _Fixture()
        m = Mission(self.inv, self.job, adapter, fixture_mode=True, start_at='transfer')
        m.start()
        for _ in range(100):
            if m.terminal:
                break
            m.tick(1 / 60)
        self._check(m, adapter)

    @unittest.skipUnless(HAVE_PT, 'py_trees not importable (task .venv-bt / RESTOCK_BT_PYTHONPATH)')
    def test_tree_enters_transfer(self):
        from twin.restock.bt.driver import build_bus, run_mission_tree
        from twin.restock.bt.tree import active_stages
        adapter = _Fixture()
        m = Mission(self.inv, self.job, adapter, fixture_mode=True, start_at='transfer')
        ctx, root, summary = run_mission_tree(m, build_bus(None)[0], max_ticks=200)
        self.assertEqual(active_stages(m), ['transfer', 'verify', 'return'])
        self.assertEqual(summary['shortcut']['start_at'], 'transfer')
        self._check(m, adapter)


class CreateWiringTests(unittest.TestCase):
    def test_build_gets_the_staged_scenario_runtime_keeps_the_original(self):
        from test_isaac_bundle import MANIFEST, CreateWiringTests as Base, base_config
        from twin.restock.isaac_bundle import create
        seen, flags = {}, {}
        sc = dict(mission_scenario(), navigation_model='articulated')
        cfg = base_config(**bench_config())
        runtime = Base._fixture_runtime(None, flags)

        def build_runtime(**k):
            seen['runtime_scenario'] = k['scenario']
            return runtime(**k)
        with mock.patch.object(shortcut, 'apply_start_state',
                               lambda assembly, config, scenario: seen.__setitem__('applied', scenario)):
            bundle = create(app=object(), manifest=MANIFEST, scenario=sc, config=cfg, output='/tmp/x',
                            _assemble=lambda **k: seen.__setitem__('built', k['scenario']) or object(),
                            _runtime=build_runtime)
        built = seen['built']['initial_staging']
        self.assertEqual(tuple(built['robot_base_pose']), RECORDED_BAY_PARK_ROBOT_POSE)
        self.assertIs(seen['applied'], seen['built'])
        self.assertIs(seen['runtime_scenario'], sc)
        self.assertEqual(bundle.start_staging['mode'], 'start_phase_shortcut:recorded_bay_park')

    def test_absent_staging_passes_the_scenario_through(self):
        from test_isaac_bundle import MANIFEST, SCENARIO, CreateWiringTests as Base, base_config
        from twin.restock.isaac_bundle import create
        seen = {}
        bundle = create(app=object(), manifest=MANIFEST, scenario=SCENARIO, config=base_config(),
                        output='/tmp/x', _assemble=lambda **k: seen.__setitem__('built', k['scenario']),
                        _runtime=Base._fixture_runtime(None, {}))
        self.assertIs(seen['built'], SCENARIO)
        self.assertIsNone(bundle.start_staging)


if __name__ == '__main__':
    unittest.main()
