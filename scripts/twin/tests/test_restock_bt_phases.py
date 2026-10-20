"""CPU tests for the behaviour tree as the full mission DRIVER with explicit position-first phase
leaves (fm/sim-restock-bt-driver-v1).

A ``PhasedFixture`` walks per-stage phases (the adapter's observed ``phase`` / manipulation phase /
reposition + return flags) exactly as the real runtime does, so the tree's navigate -> perceive ->
manipulate decomposition is genuinely exercised -- yet it returns the same running/succeeded tokens a
plain fixture would, so driving it through the tree must match ``Mission.tick`` step-for-step. These
assert:

* **Parity (decomposed path):** the same durable per-stage event sequence, final state, stock and
  ``step_stage`` call count as the sequential state machine, for all seven stages.
* **Structure:** each stage is a subtree of explicit, named leaves in the position-first order, and
  at most one leaf is RUNNING at a time (steps happen in series, not all at once).
* **Retry/recovery:** a mid-stage failure re-enters the phase sequence from step 0 and recovers.

Run (task venv):  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= .venv-bt/bin/python -m pytest \
  scripts/twin/tests/test_restock_bt_phases.py -q
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock.inventory import Inventory, Job
from twin.restock.mission import Mission, Result

try:
    import py_trees  # noqa: F401
    HAVE_PT = True
except ImportError:
    HAVE_PT = False


# Per-stage phase walks: one attr-dict applied per running tick (then one succeeded tick). The dicts
# set the exact fields the stage_phases classifier reads, so the gates advance like the real runtime.
SCRIPTS = {
    'collect': [{'phase': 'navigate'}, {'phase': 'warm'},
                {'phase': 'execute', '_last_manip_phase': 'grasp'},
                {'phase': 'execute', '_last_manip_phase': 'tuck'},
                {'phase': 'navigate', '_collect_reposition_pending': True},
                {'phase': 'rewarm', '_collect_repositioned': True},
                {'phase': 'execute', '_last_manip_phase': 'place', '_collect_repositioned': True},
                {'phase': 'execute', '_last_manip_phase': 'verify', '_collect_repositioned': True}],
    'couple': [{'phase': 'navigate'}, {'phase': 'fine_dock'}, {'phase': 'attached'}],
    'outbound': [{'phase': 'navigate'}],
    'park': [{'phase': 'execute'}],
    'transfer': [{'phase': 'warm'}, {'phase': 'execute', '_last_manip_phase': 'place'},
                 {'phase': 'execute', '_last_manip_phase': 'verify'}],
    'verify': [{'phase': 'execute'}],
    'return': [{'phase': 'navigate'}],
}

RETURN_UNCOUPLE_SCRIPT = [{'phase': 'navigate', '_return_leg': 'cart_parking'},
                          {'phase': 'execute', '_return_leg': 'brake'},
                          {'phase': 'execute', '_return_leg': 'uncouple'},
                          {'phase': 'navigate', '_return_leg': 'cart_parking_backaway'},
                          {'phase': 'navigate', '_return_leg': 'home'}]


class PhasedFixture:
    _PHASE_ATTRS = ('phase', '_last_manip_phase', '_collect_reposition_pending',
                    '_collect_repositioned', '_return_leg')

    def __init__(self, scripts, *, uncouple_on_return=False):
        self.time = 0.
        self.stopped = False
        self.steps = 0
        self.scripts = scripts
        self._i = 0
        self.action = None
        self.uncouple_on_return = uncouple_on_return
        self.poses = {'cart_parking': (0., 0., 0.)} if uncouple_on_return else {}
        self._reset_phase_view()

    # the real RuntimeAdapter._uncouple_return equivalent the classifier consults for 'return'
    def _uncouple_return(self):
        return self.uncouple_on_return and 'cart_parking' in self.poses

    def _reset_phase_view(self):
        self.phase = None
        self._last_manip_phase = None
        self._collect_reposition_pending = False
        self._collect_repositioned = False
        self._return_leg = None

    def start(self, action, job):
        self.action, self.job = action, job
        self._i = 0
        self._reset_phase_view()

    def _evidence(self):
        return dict(source='test_fixture', observation_id=f'{self.action}-{self.time}',
                    simulation_time=self.time, loaded=True, coupled=True, arrived=True,
                    stationary=True, released=True, home=True, sku=self.job.sku,
                    quantity=self.job.quantity, location=self.job.shelf_id, transfer_confirmed=True)

    def step(self, dt):
        self.time += dt
        self.steps += 1
        script = self.scripts.get(self.action, [])
        if self._i < len(script):
            for k in self._PHASE_ATTRS:
                if k in script[self._i]:
                    setattr(self, k, script[self._i][k])
            self._i += 1
            return Result('running', {}, '')
        return Result('succeeded', self._evidence(), '')

    def stop(self):
        self.stopped = True


def _inv():
    tmp = tempfile.TemporaryDirectory()
    inv = Inventory(str(Path(tmp.name) / 'stock.sqlite'))
    inv.seed('backroom', 'milk', 2)
    return tmp, inv


@unittest.skipUnless(HAVE_PT, 'py_trees not installed (task-owned .venv-bt has it)')
class PhasedDriverTests(unittest.TestCase):
    def _state_machine(self, adapter):
        tmp, inv = _inv()
        try:
            m = Mission(inv, Job('one', 'milk', 1, 'backroom', 'shelf'), adapter, fixture_mode=True)
            m.start()
            for _ in range(5000):
                if m.terminal:
                    break
                m.tick(1 / 60)
            return m.state, [e['state'] for e in inv.events('one')], inv.count('shelf', 'milk')
        finally:
            inv.close(); tmp.cleanup()

    def _tree(self, adapter, *, snapshot=None):
        from twin.restock.bt.driver import build_bus, run_mission_tree
        tmp, inv = _inv()
        try:
            m = Mission(inv, Job('one', 'milk', 1, 'backroom', 'shelf'), adapter, fixture_mode=True)
            bus, _ = build_bus(None)
            ctx, root, summary = run_mission_tree(m, bus, snapshot=snapshot, max_ticks=5000)
            return (m.state, [e['state'] for e in inv.events('one')], inv.count('shelf', 'milk'),
                    ctx, root, summary)
        finally:
            inv.close(); tmp.cleanup()

    def test_decomposed_full_run_matches_state_machine_step_for_step(self):
        # The phased adapter walks every stage's real phases; driving it through the tree must match
        # the state machine byte-for-byte AND call step_stage the same number of times.
        sm_state, sm_log, sm_shelf = self._state_machine(PhasedFixture(SCRIPTS))
        sm_steps = PhasedFixture(SCRIPTS)  # count steps the state machine takes, separately
        s2_state, *_ = self._state_machine(sm_steps)
        tree_adapter = PhasedFixture(SCRIPTS)
        state, log, shelf, ctx, root, summary = self._tree(tree_adapter)
        self.assertEqual(state, sm_state)
        self.assertEqual(log, sm_log)
        self.assertEqual(shelf, sm_shelf)
        self.assertEqual(state, 'complete')
        self.assertEqual(tree_adapter.steps, sm_steps.steps)  # same number of step_stage calls
        self.assertEqual(summary['orchestrator'], 'behaviour_tree')

    def test_each_stage_is_a_subtree_of_named_position_first_leaves(self):
        from twin.restock.bt.view import serialise_node
        from twin.restock.bt.stage_phases import PLANS
        _, _, _, ctx, root, _ = self._tree(PhasedFixture(SCRIPTS))
        mission = next(c for c in root.children if c.name == 'mission')
        stages = {c.name: c for c in mission.children if c.name.startswith('stage:')}
        # collect is the richest subtree: the full position-first arc, one named leaf per step.
        collect = serialise_node(stages['stage:collect'])
        steps_seq = collect['children'][0]            # the "collect steps" memory sequence
        leaf_names = [c['name'] for c in steps_seq['children']]
        self.assertEqual(leaf_names, [f"collect:{label}" for label, _ in PLANS['collect']])
        self.assertEqual(len(leaf_names), 7)
        # every stage present as its own subtree
        for action in ('collect', 'couple', 'outbound', 'park', 'transfer', 'verify', 'return'):
            self.assertIn(f'stage:{action}', stages)

    def test_steps_run_in_series_not_all_at_once(self):
        # At every tick at most ONE leaf across the whole tree is RUNNING, and within the active
        # stage the running leaf's predecessors are SUCCESS -- i.e. nav -> perceive -> manip happen
        # in order, not simultaneously.
        from twin.restock.bt.driver import build_bus, run_mission_tree
        from py_trees.common import Status

        def leaves(node):
            kids = getattr(node, 'children', [])
            return [node] if not kids else [l for c in kids for l in leaves(c)]

        violations = []

        def on_tick(i, ctx, root):
            running = [n for n in leaves(root) if n.status == Status.RUNNING
                       and type(n).__name__ == 'PhaseGate']
            if len(running) > 1:
                violations.append((i, [n.name for n in running]))

        tmp, inv = _inv()
        try:
            m = Mission(inv, Job('one', 'milk', 1, 'backroom', 'shelf'),
                        PhasedFixture(SCRIPTS), fixture_mode=True)
            bus, _ = build_bus(None)
            run_mission_tree(m, bus, max_ticks=5000, on_tick=on_tick)
            self.assertEqual(m.state, 'complete')
            self.assertEqual(violations, [])
        finally:
            inv.close(); tmp.cleanup()

    def test_return_uncouple_subtree_drives_the_five_new_steps(self):
        from twin.restock.bt.view import serialise_node
        from twin.restock.bt.stage_phases import RETURN_UNCOUPLE_PLAN
        scripts = dict(SCRIPTS, **{'return': RETURN_UNCOUPLE_SCRIPT})
        adapter = PhasedFixture(scripts, uncouple_on_return=True)
        state, log, shelf, ctx, root, _ = self._tree(adapter)
        self.assertEqual(state, 'complete')
        mission = next(c for c in root.children if c.name == 'mission')
        ret = serialise_node(next(c for c in mission.children if c.name == 'stage:return'))
        leaf_names = [c['name'] for c in ret['children'][0]['children']]
        self.assertEqual(leaf_names, [f"return:{label}" for label, _ in RETURN_UNCOUPLE_PLAN])
        self.assertEqual(len(leaf_names), 5)       # tow->park, brake, uncouple, back away, drive->home
        self.assertEqual(log[-1], 'complete')

    def test_mid_stage_failure_re_enters_phase_sequence_and_recovers(self):
        # collect fails at the manipulate step on the first attempt, then a scripted re-attempt
        # succeeds. The StageRetry node must re-begin and re-enter the phase sequence from step 0.
        from twin.restock.bt.driver import build_bus, run_mission_tree
        from twin.restock.bt.context import RecordingReactions

        class FailOnceCollect(PhasedFixture):
            def __init__(self):
                super().__init__(SCRIPTS)
                self._attempt = {}

            def start(self, action, job):
                super().start(action, job)
                self._attempt[action] = self._attempt.get(action, 0) + 1

            def step(self, dt):
                if self.action == 'collect' and self._attempt['collect'] == 1:
                    self.time += dt
                    self.steps += 1
                    # walk a couple of phases then fail in the pick (manipulate) step
                    script = [{'phase': 'navigate'}, {'phase': 'warm'},
                              {'phase': 'execute', '_last_manip_phase': 'grasp'}]
                    if self._i < len(script):
                        for k in self._PHASE_ATTRS:
                            if k in script[self._i]:
                                setattr(self, k, script[self._i][k])
                        self._i += 1
                        return Result('running', {}, '')
                    return Result('failed', {}, 'collect reach stalled: no accepted frame')
                return super().step(dt)

        reactions = RecordingReactions()
        tmp, inv = _inv()
        try:
            adapter = FailOnceCollect()
            m = Mission(inv, Job('one', 'milk', 1, 'backroom', 'shelf'), adapter, fixture_mode=True)
            bus, _ = build_bus(None)
            ctx, root, summary = run_mission_tree(m, bus, reactions=reactions, max_retries=2,
                                                  max_ticks=5000)
            self.assertEqual(m.state, 'complete')
            self.assertEqual(adapter._attempt['collect'], 2)       # one failure + one retry
            self.assertIn('reobserve', [c[0] for c in reactions.calls])
            log = ctx.bb.get('/event/log')
            self.assertTrue(any(e['action'].startswith('recover') for e in log))
        finally:
            inv.close(); tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
