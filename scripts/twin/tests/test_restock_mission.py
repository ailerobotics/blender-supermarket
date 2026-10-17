"""CPU fixtures test bookkeeping and failure handling, never simulated autonomy."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.inventory import Inventory, Job
from twin.restock.mission import ACTIONS, Mission, Result


class Fixture:
    def __init__(self):
        self.time = 0.
        self.stopped = False
        self.override = {}
        self.status = 'succeeded'

    def start(self, action, job):
        self.action, self.job = action, job

    def step(self, dt):
        self.time += dt
        evidence = dict(source='test_fixture', observation_id=f'{self.action}-{self.time}',
                        simulation_time=self.time, loaded=True, coupled=True,
                        arrived=True, stationary=True, released=True, home=True,
                        sku=self.job.sku, quantity=self.job.quantity,
                        location=self.job.shelf_id, transfer_confirmed=True)
        evidence.update(self.override)
        return Result(self.status, evidence, 'obstacle blocks path')

    def stop(self):
        self.stopped = True


class MissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / 'stock.sqlite')
        self.inv = Inventory(self.path)
        self.inv.seed('backroom', 'milk', 2)
        self.job = Job('one', 'milk', 1, 'backroom', 'shelf')
        self.adapter = Fixture()

    def tearDown(self):
        self.inv.close()
        self.tmp.cleanup()

    def mission(self, **kw):
        m = Mission(self.inv, self.job, self.adapter, **kw)
        m.start()
        return m

    def test_success_credits_only_after_verification(self):
        m = self.mission(fixture_mode=True)
        for action in ACTIONS:
            self.assertEqual(m.state, action)
            if action != 'return':
                self.assertEqual(self.inv.count('shelf', 'milk'), 0)
            m.tick(.1)
        self.assertEqual(m.state, 'complete')
        self.assertEqual(self.inv.count('backroom', 'milk'), 1)
        self.assertEqual(self.inv.count('shelf', 'milk'), 1)
        self.assertTrue(self.adapter.stopped)
        m.tick(.1)
        self.assertEqual(self.inv.count('shelf', 'milk'), 1)

    def test_unavailable_stage_records_failed_continues_and_ends_failed(self):
        # Honest degradation: a stage that reports 'unavailable' is recorded FAILED, the
        # mission keeps exercising every later stage, and the overall outcome is 'failed'.
        class Degraded(Fixture):
            def step(self, dt):
                result = super().step(dt)
                if self.action == 'transfer':
                    return Result('unavailable', {}, 'manipulation unavailable: cuRobo not provisioned')
                return result
        self.adapter = Degraded()
        m = self.mission(fixture_mode=True)
        seen = []
        for _ in range(len(ACTIONS)):
            seen.append(m.state)
            m.tick(.1)
        # Every stage was attempted, including the ones after the unavailable transfer.
        self.assertEqual(seen, list(ACTIONS))
        self.assertEqual(m.state, 'failed')
        self.assertIn('transfer', m.reason)
        self.assertEqual(m.degraded, ['transfer'])
        # Transfer was recorded as unavailable, never as a success.
        actions = [e['state'] for e in self.inv.events('one')]
        self.assertIn('transfer_unavailable', actions)
        self.assertNotIn('transfer_succeeded', actions)
        self.assertTrue(self.adapter.stopped)

    def test_production_rejects_fixture_and_ground_truth(self):
        m = self.mission()
        m.tick(.1)
        self.assertEqual(m.state, 'failed')
        self.assertTrue(self.adapter.stopped)
        self.assertEqual(self.inv.count('shelf', 'milk'), 0)

    def test_timeout_retains_reservation(self):
        m = self.mission(timeout_s=.2, fixture_mode=True)
        self.adapter.status = 'running'
        m.tick(.1)
        m.tick(.2)
        self.assertEqual(m.state, 'failed')
        with self.assertRaises(ValueError):
            self.inv.reserve(Job('two', 'milk', 2, 'backroom', 'shelf'))

    def test_restart_refuses_replay(self):
        self.mission(fixture_mode=True)
        other = Inventory(self.path)
        try:
            with self.assertRaisesRegex(ValueError, 'reconciliation'):
                other.reserve(self.job)
        finally:
            other.close()

    def test_wrong_sku_and_stale_evidence_fail_without_credit(self):
        m = self.mission(fixture_mode=True)
        for _ in range(5):
            m.tick(.1)
        self.adapter.override = {'sku': 'wrong'}
        m.tick(.1)
        self.assertEqual(m.state, 'failed')
        self.assertEqual(self.inv.count('shelf', 'milk'), 0)

    def test_stale(self):
        m = self.mission(fixture_mode=True)
        self.adapter.override = {'simulation_time': -10}
        m.tick(.1)
        self.assertEqual(m.state, 'failed')

    def test_missing_coupling_measurement(self):
        m = self.mission(fixture_mode=True)
        m.tick(.1)
        self.adapter.override = {'coupled': False}
        m.tick(.1)
        self.assertEqual(m.state, 'failed')

    def test_obstacle_failure_stops(self):
        m = self.mission(fixture_mode=True)
        for _ in range(2):
            m.tick(.1)
        self.adapter.status = 'failed'
        m.tick(.1)
        self.assertEqual(m.state, 'failed')
        self.assertIn('obstacle', m.reason)
        self.assertTrue(self.adapter.stopped)

    def test_return_failure_preserves_verified_stock(self):
        m = self.mission(fixture_mode=True)
        for _ in range(6):
            m.tick(.1)
        self.adapter.status = 'failed'
        m.tick(.1)
        self.assertEqual(m.state, 'failed')
        self.assertEqual(self.inv.count('shelf', 'milk'), 1)

    def test_atomic_rollback_on_reused_observation(self):
        m = self.mission(fixture_mode=True)
        for _ in range(6):
            m.tick(.1)
        evidence = self.inv.events('one')[-2]['evidence']
        job2 = Job('two', 'milk', 1, 'backroom', 'shelf')
        self.inv.reserve(job2)
        self.inv.record('two', 'verify', {})
        with self.assertRaises(Exception):
            self.inv.verify(job2, evidence)
        self.assertEqual(self.inv.count('backroom', 'milk'), 1)
        self.assertEqual(self.inv.count('shelf', 'milk'), 1)

    def test_invalid_quantities(self):
        for qty in (True, 0, -1, 1.2):
            with self.assertRaises(ValueError):
                Job('bad', 'milk', qty, 'backroom', 'shelf')

    def test_start_at_records_shortcut_prefix_and_runs_remaining_stages(self):
        # A start-phase shortcut skips the in-order prefix before `start_at`, records each
        # skipped stage as a `<stage>_shortcut` event (shortcut=True, never *_succeeded), then
        # runs the real state machine from `start_at` onward -- here transfer -> verify -> return.
        m = self.mission(fixture_mode=True, start_at='transfer')
        skipped = ('collect', 'couple', 'outbound', 'park')
        self.assertEqual(m.shortcut, {'start_at': 'transfer', 'skipped': list(skipped)})
        self.assertEqual(m.state, 'transfer')  # began at the shortcut target, not collect
        events = [e['state'] for e in self.inv.events('one')]
        for stage in skipped:
            self.assertIn(stage + '_shortcut', events)
            self.assertNotIn(stage + '_succeeded', events)
        # Skipped stages carry honest shortcut provenance.
        shortcut_ev = next(e for e in self.inv.events('one') if e['state'] == 'collect_shortcut')
        self.assertTrue(shortcut_ev['evidence']['shortcut'])
        self.assertEqual(shortcut_ev['evidence']['source'], 'start_phase_shortcut')
        for _ in range(3):  # transfer, verify, return
            m.tick(.1)
        self.assertEqual(m.state, 'complete')
        # verify still committed the real stock move exactly once.
        self.assertEqual(self.inv.count('shelf', 'milk'), 1)
        self.assertEqual(self.inv.count('backroom', 'milk'), 1)
        actions = [e['state'] for e in self.inv.events('one')]
        self.assertIn('transfer_succeeded', actions)
        self.assertIn('inventory_verified', actions)

    def test_start_at_absent_is_identical_to_production(self):
        m = self.mission(fixture_mode=True)
        self.assertIsNone(m.shortcut)
        self.assertEqual(m.start_index, 0)
        self.assertEqual(m.state, 'collect')
        actions = [e['state'] for e in self.inv.events('one')]
        self.assertFalse(any(a.endswith('_shortcut') for a in actions))

    def test_start_at_invalid_stage_raises(self):
        with self.assertRaises(ValueError):
            Mission(self.inv, self.job, self.adapter, start_at='not_a_stage')

    def test_start_at_cannot_skip_verify_commit_gate(self):
        with self.assertRaisesRegex(ValueError, 'verify'):
            Mission(self.inv, self.job, self.adapter, start_at='return')


if __name__ == '__main__':
    unittest.main()
