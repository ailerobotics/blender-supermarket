"""CPU tests for the ``stop_after`` targeted test: end the mission right after the stage under test.

Covers the pure stage cut (``mission.bounded_stages``), the launcher key/flag application
(``run_mobile_restock.apply_bench_keys``), the mission.json + caption labels, and that the state
machine and behaviour tree drive exactly the bounded stages (collect-only, couple->park from a
start_at=couple shortcut, transfer->verify from the transfer bench).
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.inventory import Inventory, Job
from twin.restock.mission import ACTIONS, Mission, Result, bounded_stages
from twin.restock.shortcut import bounded_summary, walkthrough_caption
from run_mobile_restock import apply_bench_keys

try:
    import py_trees  # noqa: F401
    HAVE_PT = True
except ImportError:
    HAVE_PT = False


class BoundedStagesTests(unittest.TestCase):
    def test_cut_after_the_stage_under_test(self):
        self.assertEqual(bounded_stages(None, 'collect'), ('collect',))
        self.assertEqual(bounded_stages(None, 'park', 'couple'), ('collect', 'couple', 'outbound', 'park'))
        self.assertEqual(bounded_stages(None, 'return'), ACTIONS)
        self.assertEqual(bounded_stages(ACTIONS[:4], None), ACTIONS[:4])

    def test_refusals(self):
        with self.assertRaises(ValueError):
            bounded_stages(None, 'nonsense')
        with self.assertRaises(ValueError):
            bounded_stages(ACTIONS[:2], 'park')                 # not in the configured prefix
        with self.assertRaises(ValueError):
            bounded_stages(None, 'couple', 'transfer')          # stops before it starts

    def test_launcher_keys(self):
        self.assertEqual(apply_bench_keys({}), {})
        cfg = apply_bench_keys({'start_at': 'couple', 'stop_after': 'park'})
        self.assertEqual(cfg['stages'], ['collect', 'couple', 'outbound', 'park'])
        cfg = apply_bench_keys({'stages': list(ACTIONS)}, start_at='transfer',
                               start_staging='recorded_bay_park', stop_after='verify')
        self.assertEqual((cfg['start_at'], cfg['start_staging'], cfg['stop_after']),
                         ('transfer', 'recorded_bay_park', 'verify'))
        self.assertEqual(cfg['stages'], list(ACTIONS[:6]))
        with self.assertRaises(ValueError):
            apply_bench_keys({'start_at': 'transfer'}, stop_after='park')


class LabelTests(unittest.TestCase):
    def test_caption_and_summary(self):
        cfg = apply_bench_keys({}, start_at='couple', stop_after='park')
        text = walkthrough_caption(cfg)
        self.assertEqual(text, 'BENCH RUN start_at=couple stop_after=park: collect SYNTHESIZED, not '
                               'performed; transfer, verify, return NOT RUN | settle hold 20s; host clock sync')
        out = bounded_summary(cfg)
        self.assertEqual(out['stop_after'], 'park')
        self.assertEqual(out['stages_driven'], ['couple', 'outbound', 'park'])
        self.assertEqual(out['not_run'], ['transfer', 'verify', 'return'])
        self.assertEqual(out['walkthrough_caption'], text)
        self.assertEqual(walkthrough_caption(apply_bench_keys({}, stop_after='collect')),
                         'BENCH RUN stop_after=collect: couple, outbound, park, transfer, verify, return NOT RUN')
        self.assertEqual(bounded_summary({}), {})

    def test_start_only_caption_unchanged(self):
        # A staged start names its staged-start switches (twin.restock.staged_start) on screen too.
        self.assertEqual(walkthrough_caption({'start_at': 'transfer', 'start_staging': 'recorded_bay_park'}),
                         'BENCH RUN start_at=transfer: collect, couple, outbound, park SYNTHESIZED, not '
                         'performed | staged: recorded_bay_park | settle hold 20s; host clock sync')
        self.assertEqual(walkthrough_caption({'start_at': 'transfer', 'start_staging': 'recorded_bay_park',
                                              'staged_start_settle_s': 0, 'staged_start_clock_sync': False}),
                         'BENCH RUN start_at=transfer: collect, couple, outbound, park SYNTHESIZED, not '
                         'performed | staged: recorded_bay_park')


class _Fixture:
    def __init__(self):
        self.begun, self.time, self.stopped = [], 0., False

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
        self.stopped = True


CASES = (  # (bench keys, stages the adapter must begin, stock committed at the shelf)
    ({'stop_after': 'collect'}, ['collect'], 0),
    ({'start_at': 'couple', 'stop_after': 'park'}, ['couple', 'outbound', 'park'], 0),
    ({'start_at': 'transfer', 'stop_after': 'verify'}, ['transfer', 'verify'], 1),
)


class DrivesExactlyTheBoundedStagesTests(unittest.TestCase):
    def _mission(self, keys):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        inv = Inventory(str(Path(tmp.name) / 'stock.sqlite'))
        self.addCleanup(inv.close)
        inv.seed('backstock_00', 'juice_orange', 1)
        cfg = apply_bench_keys(dict(keys))
        adapter = _Fixture()
        m = Mission(inv, Job('restock-one', 'juice_orange', 1, 'backstock_00', 'G3_back_s2'), adapter,
                    fixture_mode=True, stages=cfg.get('stages'), start_at=cfg.get('start_at'))
        return m, adapter, inv

    def _check(self, m, adapter, inv, begun, committed):
        self.assertEqual(m.state, 'complete')
        self.assertEqual(adapter.begun, begun)
        self.assertTrue(adapter.stopped)
        events = [e['state'] for e in inv.events('restock-one')]
        for stage in ACTIONS[ACTIONS.index(begun[-1]) + 1:]:
            self.assertNotIn(stage, events)
        self.assertEqual(inv.count('G3_back_s2', 'juice_orange'), committed)

    def test_state_machine(self):
        for keys, begun, committed in CASES:
            with self.subTest(keys=keys):
                m, adapter, inv = self._mission(keys)
                m.start()
                for _ in range(200):
                    if m.terminal:
                        break
                    m.tick(1 / 60)
                self._check(m, adapter, inv, begun, committed)

    @unittest.skipUnless(HAVE_PT, 'py_trees not importable (task .venv-bt / RESTOCK_BT_PYTHONPATH)')
    def test_behaviour_tree(self):
        from twin.restock.bt.driver import build_bus, run_mission_tree
        for keys, begun, committed in CASES:
            with self.subTest(keys=keys):
                m, adapter, inv = self._mission(keys)
                run_mission_tree(m, build_bus(None)[0], max_ticks=300)
                self._check(m, adapter, inv, begun, committed)


if __name__ == '__main__':
    unittest.main()
