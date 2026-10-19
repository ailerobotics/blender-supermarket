"""CPU tests for the opt-in behaviour-tree mission layer (scripts/twin/restock/bt).

Two tiers:

* **Pure** (no py_trees): the event contract -- Event (de)serialisation, scenario-events validation,
  stage-failure classification, detection conversion -- and that ``scenario.validate`` accepts/
  rejects an ``events`` list. These run under any interpreter.
* **Tree** (needs py_trees): the tree ticks the REAL Mission through mocked stage leaves -- full
  success parity with the state machine, event injection (human hold/resume, obstacle, stuck abort,
  baseline re-commission), bounded retry + recovery, degraded outcome, and snapshot output. Skipped
  if py_trees is not importable (the task-owned ``.venv-bt`` has it).

Run (task venv):  .venv-bt/bin/python -m pytest scripts/twin/tests/test_restock_bt.py -q
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock.inventory import Inventory, Job
from twin.restock.mission import ACTIONS, Mission, Result
from twin.restock import scenario as scenario_mod
from twin.restock.bt import events as ev

try:
    import py_trees  # noqa: F401
    HAVE_PT = True
except ImportError:
    HAVE_PT = False


# --------------------------------------------------------------------------------------------
# Fixture adapter: mirrors test_restock_mission.Fixture, with per-stage scripted behaviour so a
# tree test can drive running/succeeded/failed/unavailable deterministically.
# --------------------------------------------------------------------------------------------
class Fixture:
    def __init__(self, *, running_ticks=0, scripts=None):
        self.time = 0.
        self.stopped = False
        self.running_ticks = running_ticks
        self.scripts = scripts or {}      # stage -> list of status tokens consumed per begin()
        self._attempt = {}                # stage -> how many begins so far
        self._run_left = 0
        self.action = None

    def start(self, action, job):
        self.action, self.job = action, job
        self._attempt[action] = self._attempt.get(action, 0) + 1
        self._run_left = self.running_ticks

    def _evidence(self):
        return dict(source='test_fixture', observation_id=f'{self.action}-{self.time}',
                    simulation_time=self.time, loaded=True, coupled=True, arrived=True,
                    stationary=True, released=True, home=True, sku=self.job.sku,
                    quantity=self.job.quantity, location=self.job.shelf_id,
                    transfer_confirmed=True)

    def step(self, dt):
        self.time += dt
        script = self.scripts.get(self.action)
        if script is not None:
            token = script[min(self._attempt[self.action] - 1, len(script) - 1)]
            if token == 'fail':
                return Result('failed', {}, f'{self.action} reach stalled: no accepted frame')
            if token == 'unavailable':
                return Result('unavailable', {}, f'{self.action} unavailable: cuRobo not provisioned')
            # token == 'ok' falls through to normal success below
        if self._run_left > 0:
            self._run_left -= 1
            return Result('running', {}, '')
        return Result('succeeded', self._evidence(), '')

    def stop(self):
        self.stopped = True


def _inv():
    tmp = tempfile.TemporaryDirectory()
    inv = Inventory(str(Path(tmp.name) / 'stock.sqlite'))
    inv.seed('backroom', 'milk', 2)
    return tmp, inv


class PureContractTests(unittest.TestCase):
    def test_event_roundtrip_and_validation(self):
        e = ev.Event('human_detected', {'zone': 'stop', 'distance_m': 1.2}, stamp_ns=5)
        self.assertEqual(ev.Event.from_json(e.to_json()), e)
        with self.assertRaises(ValueError):
            ev.Event('not_a_type')
        with self.assertRaises(ValueError):
            ev.Event.from_json('{bad json')
        with self.assertRaises(ValueError):
            ev.Event.from_json('{"no":"type"}')

    def test_detection_to_event(self):
        class D:
            position = (1.5, -2.0)
            radius = 0.4
            source = 'lidar'
        e = ev.detection_to_event(D(), stamp_ns=9)
        self.assertEqual(e.type, 'dynamic_obstacle')
        self.assertEqual(e.data, {'position': [1.5, -2.0], 'radius': 0.4, 'source': 'lidar'})
        self.assertEqual(e.stamp_ns, 9)

    def test_scenario_events_validation(self):
        self.assertEqual(ev.validate_scenario_events({}), [])          # absent -> valid
        good = {'events': [{'t': 0.5, 'type': 'human_detected', 'data': {'zone': 'stop'}},
                           {'t': 2.0, 'type': 'human_cleared'}]}
        self.assertEqual(ev.validate_scenario_events(good), [])
        self.assertEqual(len(ev.parse_scenario_events(good)), 2)
        bad = {'events': [{'t': 1.0, 'type': 'human_detected'},
                          {'t': 0.5, 'type': 'nope'}]}       # out of order + unknown type
        errs = ev.validate_scenario_events(bad)
        self.assertTrue(any('nondecreasing' in e for e in errs))
        self.assertTrue(any('unknown type' in e for e in errs))
        with self.assertRaises(ValueError):
            ev.parse_scenario_events(bad)

    def test_collect_substep_view_label(self):
        # fm390: the view shows the position-first collect arc. The adapter drive/warm phase overrides the
        # finer manipulation phase; otherwise the controller phase maps to the step. No phase => no label.
        from twin.restock.bt import view_labels as vl

        class Adapter:
            def __init__(self, phase=None, manip=None):
                self.phase, self._last_manip_phase = phase, manip

        class M:
            def __init__(self, adapter): self.adapter = adapter

        self.assertEqual(vl.collect_substep(M(Adapter('navigate', 'transit'))), 'nav (navigate)')
        self.assertEqual(vl.collect_substep(M(Adapter('rewarm', 'transit'))), 'perceive (rewarm)')
        self.assertEqual(vl.collect_substep(M(Adapter('execute', 'preplace'))), 'place (preplace)')
        self.assertEqual(vl.collect_substep(M(Adapter('execute', 'grasp'))), 'pick (grasp)')
        self.assertEqual(vl.collect_substep(M(Adapter('execute', 'verify'))), 'verify (verify)')
        self.assertIsNone(vl.collect_substep(M(Adapter(None, None))))      # no phase exposed
        self.assertIsNone(vl.collect_substep(M(None)))                     # no adapter
        self.assertIsNone(vl.collect_substep(object()))                    # plain mission

    def test_classify_stage_failure(self):
        self.assertEqual(ev.classify_stage_failure('collect', 'reach stalled: no frame'), 'grasp_failed')
        self.assertEqual(ev.classify_stage_failure('transfer', 'transfer evidence mismatch'), 'seal_failed')
        self.assertEqual(ev.classify_stage_failure('outbound', 'blocked_no_route_timeout'), 'blocked_aisle')
        self.assertEqual(ev.classify_stage_failure('return', 'goal_timeout'), 'goal_miss')
        self.assertEqual(ev.classify_stage_failure('couple', 'cart lost before transfer'), 'coupling_failed')
        self.assertEqual(ev.classify_stage_failure('park', 'mystery'), 'goal_miss')   # stage fallback

    def test_scenario_validate_rejects_bad_events(self):
        # A minimal manifest/scenario that is otherwise valid enough to reach the events check.
        manifest = {'frame': 'map', 'units': 'metres', 'up_axis': 'Z', 'ground_truth_only': True}
        scenario = {'job': {'job_id': 'j', 'sku': 'milk', 'quantity': 1,
                            'source_bin': 'backroom', 'shelf_id': 'shelf'},
                    'events': [{'t': 1.0, 'type': 'bogus'}]}
        result = scenario_mod.validate(manifest, scenario)
        self.assertFalse(result['ok'])
        self.assertTrue(any('bogus' in e for e in result['errors']))


@unittest.skipUnless(HAVE_PT, 'py_trees not installed (task-owned .venv-bt has it)')
class TreeTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.inv = _inv()
        self.job = Job('one', 'milk', 1, 'backroom', 'shelf')

    def tearDown(self):
        self.inv.close()
        self.tmp.cleanup()

    def _run(self, adapter, *, reactions=None, scenario=None, max_retries=2, snapshot=None,
             max_ticks=2000):
        from twin.restock.bt.driver import build_bus, run_mission_tree
        from twin.restock.bt.context import RecordingReactions
        reactions = reactions or RecordingReactions()
        m = Mission(self.inv, self.job, adapter, fixture_mode=True)
        bus, queue = build_bus(scenario)
        ctx, root, summary = run_mission_tree(
            m, bus, reactions=reactions, snapshot=snapshot, max_retries=max_retries,
            max_ticks=max_ticks)
        return m, ctx, root, summary, reactions

    def test_full_success_matches_state_machine(self):
        m, ctx, root, summary, _ = self._run(Fixture())
        self.assertEqual(m.state, 'complete')
        self.assertEqual(summary['orchestrator'], 'behaviour_tree')
        self.assertEqual(self.inv.count('backroom', 'milk'), 1)   # 2 seeded - 1 moved
        self.assertEqual(self.inv.count('shelf', 'milk'), 1)      # credited only at verify
        # The durable record must carry the same per-stage *_succeeded + complete events the
        # state machine writes (no fabricated success, stock committed once).
        events = [e['state'] for e in self.inv.events('one')]
        for action in ACTIONS:
            self.assertIn(action + '_succeeded', events)
        self.assertIn('complete', events)

    def test_running_ticks_then_success(self):
        m, *_ = self._run(Fixture(running_ticks=3))
        self.assertEqual(m.state, 'complete')

    def test_parity_with_state_machine(self):
        # The default (no-event) path must be byte-identical to Mission.tick looping: same final
        # state and the same durable per-stage event sequence + stock. Run both on twin inventories.
        def sm_events(adapter):
            tmp, inv = _inv()
            try:
                m = Mission(inv, Job('one', 'milk', 1, 'backroom', 'shelf'), adapter, fixture_mode=True)
                m.start()
                for _ in range(2000):
                    if m.terminal:
                        break
                    m.tick(1 / 60)
                return m.state, [e['state'] for e in inv.events('one')], inv.count('shelf', 'milk')
            finally:
                inv.close(); tmp.cleanup()

        for scripts in (None, {'collect': ['fail', 'ok']}, {'transfer': ['unavailable']}):
            with self.subTest(scripts=scripts):
                sm_state, sm_log, sm_shelf = sm_events(Fixture(scripts=scripts) if scripts else Fixture())
                # fresh inventory for the tree run
                self.tearDown(); self.setUp()
                adapter = Fixture(scripts=scripts) if scripts else Fixture()
                m, ctx, root, summary, _ = self._run(adapter)
                tree_log = [e['state'] for e in self.inv.events('one')]
                tree_shelf = self.inv.count('shelf', 'milk')
                # The retry case is NOT expected to match a no-retry state machine run (the tree
                # recovers where the state machine fails) -- only the pure-success and degraded
                # paths must match byte-for-byte.
                if scripts is None or 'transfer' in (scripts or {}):
                    self.assertEqual(m.state, sm_state)
                    self.assertEqual(tree_log, sm_log)
                    self.assertEqual(tree_shelf, sm_shelf)

    def test_human_event_holds_then_resumes(self):
        # Human enters the stop zone early and leaves later; the mission must HOLD (not advance)
        # while held and still complete afterwards. A fixture with running ticks gives a window.
        scenario = {'events': [
            {'t': 0.02, 'type': 'human_detected', 'data': {'zone': 'stop', 'distance_m': 0.5}},
            {'t': 0.20, 'type': 'human_cleared'}]}
        from twin.restock.bt.context import RecordingReactions
        reactions = RecordingReactions()
        m, ctx, root, summary, _ = self._run(Fixture(running_ticks=2), reactions=reactions,
                                             scenario=scenario)
        self.assertEqual(m.state, 'complete')
        calls = [c[0] for c in reactions.calls]
        self.assertIn('stop', calls)       # safety hold on entering stop zone
        self.assertIn('resume', calls)     # released when cleared
        # The timeline recorded the hold and the resume.
        log = ctx.bb.get('/event/log')
        actions = [e['action'] for e in log]
        self.assertIn('stop+pause_arm', actions)
        self.assertIn('resume', actions)

    def test_queue_obstacle_injection_sets_replan(self):
        # Drive a few ticks by hand so we can assert the pause/replan state while the obstacle is
        # present, independent of mission timing.
        from twin.restock.bt.driver import build_bus
        from twin.restock.bt.context import MissionContext, RecordingReactions, CLOCK
        from twin.restock.bt.tree import build_tree
        import py_trees
        m = Mission(self.inv, self.job, Fixture(running_ticks=50), fixture_mode=True)
        m.start(begin=False)
        bus, queue = build_bus(None)
        reactions = RecordingReactions()
        ctx = MissionContext(m, bus, reactions=reactions)
        root = build_tree(ctx)
        tree = py_trees.trees.BehaviourTree(root)
        tree.setup()

        def tick(now):
            ctx.dt = 1 / 60
            ctx.now = now
            ctx.bb.set(CLOCK, now)
            tree.tick()

        tick(0.01)
        self.assertFalse(ctx.paused)
        queue.push(ev.Event('dynamic_obstacle', {'position': [1.0, 0.0], 'radius': 0.5, 'source': 's'}))
        tick(0.02)
        self.assertTrue(ctx.paused)
        self.assertTrue(ctx.bb.get('/control/replan'))
        self.assertIn('wait', [c[0] for c in reactions.calls])
        queue.push(ev.Event('obstacle_cleared'))
        tick(0.03)
        self.assertFalse(ctx.paused)

    def test_bounded_retry_then_recovery_succeeds(self):
        # collect fails twice then succeeds; the tree re-observes + retries within budget.
        from twin.restock.bt.context import RecordingReactions
        reactions = RecordingReactions()
        adapter = Fixture(scripts={'collect': ['fail', 'fail', 'ok']})
        m, ctx, root, summary, _ = self._run(adapter, reactions=reactions, max_retries=2)
        self.assertEqual(m.state, 'complete')
        self.assertEqual(adapter._attempt['collect'], 3)       # 1 initial + 2 retries
        self.assertEqual([c[0] for c in reactions.calls].count('reobserve'), 2)
        log = ctx.bb.get('/event/log')
        self.assertTrue(any(e['action'].startswith('recover') for e in log))

    def test_retry_exhaustion_is_honest_abort(self):
        adapter = Fixture(scripts={'collect': ['fail']})      # always fails
        m, ctx, root, summary, _ = self._run(adapter, max_retries=2)
        self.assertEqual(m.state, 'failed')
        self.assertIn('stalled', m.reason)
        self.assertEqual(self.inv.count('shelf', 'milk'), 0)   # nothing fabricated
        self.assertTrue(adapter.stopped)

    def test_degraded_stage_ends_failed(self):
        # Parity with the state machine: a degraded (unavailable) stage is recorded, the mission
        # keeps exercising later stages, and finalize forces a 'failed' outcome. (A degraded
        # transfer still lets the later verify stage run -- exactly as Mission.tick does -- so this
        # asserts the SAME end state, not a divergent one.)
        adapter = Fixture(scripts={'collect': ['unavailable']})
        m, ctx, root, summary, _ = self._run(adapter)
        self.assertEqual(m.state, 'failed')
        self.assertIn('collect', m.degraded)
        self.assertIn('stages unavailable', m.reason)

    def test_health_guard_aborts_to_safe_park(self):
        scenario = {'events': [{'t': 0.02, 'type': 'stuck', 'data': {'reason': 'wheel slip'}}]}
        from twin.restock.bt.context import RecordingReactions
        reactions = RecordingReactions()
        m, ctx, root, summary, _ = self._run(Fixture(running_ticks=50), reactions=reactions,
                                             scenario=scenario)
        self.assertEqual(m.state, 'failed')
        self.assertIn('stuck', m.reason)
        self.assertIn('safe_park', [c[0] for c in reactions.calls])
        self.assertIsNotNone(ctx.abort_reason)

    def test_baseline_guard_recommission_refusal_vs_success(self):
        from twin.restock.bt.context import RecordingReactions
        # Refusing provider: event stays active, mission held (never completes within few ticks).
        scenario = {'events': [{'t': 0.02, 'type': 'baseline_expired', 'data': {'station': 'cart'}}]}
        reactions = RecordingReactions(recommission_ok=False)
        m, ctx, root, summary, _ = self._run(Fixture(running_ticks=50), reactions=reactions,
                                             scenario=scenario, max_ticks=20)
        self.assertIn('baseline_expired', ctx.active())
        self.assertEqual([c[0] for c in reactions.calls].count('recommission') >= 1, True)
        # Succeeding provider on a fresh run: event cleared, mission proceeds to complete.
        self.tearDown(); self.setUp()
        reactions = RecordingReactions(recommission_ok=True)
        m2, ctx2, root2, summary2, _ = self._run(Fixture(), reactions=reactions, scenario=scenario)
        self.assertEqual(m2.state, 'complete')
        self.assertNotIn('baseline_expired', ctx2.active())

    def test_snapshot_writer_outputs(self):
        from twin.restock.bt.view import SnapshotWriter
        out = tempfile.TemporaryDirectory()
        writer = SnapshotWriter(out.name, title='bt-test')
        scenario = {'events': [{'t': 0.02, 'type': 'human_detected', 'data': {'zone': 'stop'}},
                               {'t': 0.08, 'type': 'human_cleared'}]}
        m, ctx, root, summary, _ = self._run(Fixture(running_ticks=2), scenario=scenario,
                                             snapshot=writer)
        snap = json.loads((Path(out.name) / 'bt_snapshot.json').read_text())
        self.assertIn('tree', snap)
        self.assertEqual(snap['tree']['name'], 'restock_root')
        names = {c['name'] for c in snap['tree']['children']}
        self.assertEqual(names, {'event_pump', 'reactive', 'mission'})
        self.assertTrue((Path(out.name) / 'bt_view.html').exists())
        lines = (Path(out.name) / 'bt_events.jsonl').read_text().strip().splitlines()
        self.assertTrue(len(lines) >= 2)       # at least the human hold + resume
        self.assertTrue(all(json.loads(l) for l in lines))
        out.cleanup()


if __name__ == '__main__':
    unittest.main()
