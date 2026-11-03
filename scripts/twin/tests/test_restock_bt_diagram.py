"""CPU tests for the behaviour-tree diagram generator (scripts/twin/restock/bt/diagram.py) and the
add-a-stage recipe (scripts/twin/restock/bt/stage_template.py).

Two tiers:

* **Pure** (no py_trees): the model is reconstructed from the code (PLANS + the event contract), the
  metadata registry resolves to real symbols, a run aligns to sim windows + video timecodes, and the
  add-a-stage template makes a dummy stage appear in BOTH the diagram model and a replayed timeline.
  These run under any interpreter.
* **Parity** (needs py_trees): the reconstructed structure matches the REAL serialised tree
  node-for-node, so the diagram cannot drift from the tree it documents. Skipped without py_trees
  (the task-owned ``.venv-bt`` has it).

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -m pytest \
      scripts/twin/tests/test_restock_bt_diagram.py -q
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock.bt import diagram as D
from twin.restock.bt import events as ev
from twin.restock.bt import stage_phases as sp
from twin.restock.bt import stage_template as tmpl

try:
    import py_trees  # noqa: F401
    HAVE_PT = True
except ImportError:
    HAVE_PT = False


def _names_cls(node):
    """(name, cls) pairs in depth-first tick order -- the structural signature."""
    return [(n['name'], n['cls']) for n, _ in D.iter_nodes(node)]


class ModelFromCode(unittest.TestCase):
    def test_root_shape(self):
        m = D.build_model()
        self.assertEqual(m['name'], 'restock_root')
        self.assertEqual(m['cls'], 'Sequence')
        self.assertEqual(m['memory'], False)
        top = [c['name'] for c in m['children']]
        self.assertEqual(top, ['event_pump', 'reactive', 'mission'])

    def test_reactive_guards_in_order(self):
        m = D.build_model()
        reactive = next(c for c in m['children'] if c['name'] == 'reactive')
        self.assertEqual([c['name'] for c in reactive['children']],
                         ['health_guard', 'human_safety', 'obstacle_avoid', 'baseline_guard'])

    def test_every_plans_stage_and_step_present(self):
        m = D.build_model()
        mission = next(c for c in m['children'] if c['name'] == 'mission')
        stage_nodes = [c for c in mission['children'] if c['role'] == 'stage_retry']
        self.assertEqual([n['stage'] for n in stage_nodes], list(sp.PLANS))
        for stage, plan in sp.PLANS.items():
            sn = next(n for n in stage_nodes if n['stage'] == stage)
            steps_seq = sn['children'][0]
            gate_labels = [c['name'] for c in steps_seq['children']]
            self.assertEqual(gate_labels, [f'{stage}:{label}' for label, _ in plan])
        self.assertEqual(mission['children'][-1]['name'], 'finalize')

    def _stage_gates(self, model, stage):
        mission = next(c for c in model['children'] if c['name'] == 'mission')
        node = next(n for n in mission['children']
                    if n['role'] == 'stage_retry' and n['stage'] == stage)
        return [c['name'] for c in node['children'][0]['children']]

    def _outbound_gates(self, model):
        return self._stage_gates(model, 'outbound')

    def test_shelf_staging_outbound_plan_shows_the_new_nodes(self):
        # sim-shelf-staging-v1 (+ sim-service-leg-carry-v1): with shelf_staging the outbound stage
        # decomposes into the re-sequenced position-first leaves (tow->staging, brake cart, uncouple); the
        # base is left parked at the staging pose (the endcap carton-carry is the transfer service leg).
        gates = self._outbound_gates(D.build_model(shelf_staging=True))
        self.assertEqual(gates, [f'outbound:{label}' for label, _ in sp.OUTBOUND_STAGING_PLAN])
        self.assertEqual(gates, ['outbound:tow→staging', 'outbound:brake cart', 'outbound:uncouple'])
        # Default (flag off) keeps the single tow->shelf leaf (byte-identical).
        self.assertEqual(self._outbound_gates(D.build_model()), ['outbound:tow→shelf'])

    def test_shelf_staging_transfer_service_leg_shows_the_new_nodes(self):
        # sim-service-leg-carry-v1: with shelf_staging the transfer stage decomposes into the seven
        # position-first service-leg leaves (perceive carton -> pick@staging -> carry->endcap -> perceive slot
        # -> place@endcap -> verify -> drive->staging); the diagram generator must show them in series.
        gates = self._stage_gates(D.build_model(shelf_staging=True), 'transfer')
        self.assertEqual(gates, [f'transfer:{label}' for label, _ in sp.TRANSFER_STAGING_PLAN])
        self.assertEqual(gates, ['transfer:perceive carton', 'transfer:transfer-pick@staging',
                                 'transfer:carry→endcap', 'transfer:perceive slot',
                                 'transfer:place@endcap', 'transfer:verify placed',
                                 'transfer:drive→staging'])
        # Default (flag off) keeps the legacy single-pose transfer plan (byte-identical).
        self.assertEqual(self._stage_gates(D.build_model(), 'transfer'),
                         [f'transfer:{label}' for label, _ in sp.PLANS['transfer']])

    def test_cart_side_bay_shows_tow_to_bay_and_home_empty(self):
        # sim-cart-side-bay-v1: with cart_side_bay the OUTBOUND decomposes into tow->side bay / brake /
        # uncouple, the TRANSFER keeps the (re-targeted) service leg, and the RETURN baseline is the single
        # drive-home-empty leaf (the robot threads the freed doorway; the cart stays parked in the bay).
        m = D.build_model(cart_side_bay=True)
        self.assertEqual(self._outbound_gates(m),
                         ['outbound:tow→side bay', 'outbound:brake cart', 'outbound:uncouple'])
        self.assertEqual(self._stage_gates(m, 'transfer'),
                         [f'transfer:{label}' for label, _ in sp.TRANSFER_STAGING_PLAN])
        self.assertEqual(self._stage_gates(m, 'return'), ['return:drive→home (empty)'])

    def test_cart_side_bay_tow_home_followup_shows_the_recouple_steps(self):
        # The coupled tow-home (cart retrieved to the backroom) is a named config follow-up; when selected
        # the RETURN diagram shows the five retrieval steps (re-couple@bay -> ... -> drive home).
        m = D.build_model(cart_side_bay=True, cart_tow_home=True)
        self.assertEqual(self._stage_gates(m, 'return'),
                         [f'return:{label}' for label, _ in sp.RETURN_SIDE_BAY_RECOUPLE_PLAN])
        self.assertEqual(self._stage_gates(m, 'return')[0], 'return:re-couple@bay')

    def test_stage_steps_sequences_have_memory(self):
        m = D.build_model()
        for n, _ in D.iter_nodes(m):
            if n['role'] == 'stage_steps':
                self.assertTrue(n['memory'])

    def test_categories_are_the_brief_vocabulary(self):
        allowed = {D.SEQUENCE, D.DECORATOR, D.ACTION, D.CONDITION}
        for n, _ in D.iter_nodes(D.build_model()):
            self.assertIn(n['category'], allowed, n['name'])


class MetadataRegistry(unittest.TestCase):
    def test_impl_points_at_real_symbols(self):
        """Every file:function in the registry must be defined in the named source file, so the
        diagram can never label a node with a leaf that does not exist. Checked by source inspection
        (not import) so it runs without py_trees -- the leaf modules import py_trees at module load."""
        bt_dir = Path(D.__file__).resolve().parent
        for role, (fname, symbol) in D.IMPL.items():
            src = (bt_dir / fname).read_text()
            self.assertTrue(f'def {symbol}(' in src or f'class {symbol}(' in src
                            or f'class {symbol}:' in src,
                            f'{role}: {fname} defines no {symbol}')

    def test_guard_events_are_real_event_types(self):
        for role, evs in D.GUARD_EVENTS.items():
            for e in evs:
                self.assertIn(e, ev.EVENT_TYPES, f'{role} reacts to unknown event {e}')

    def test_stage_retry_reacts_to_the_failure_contract(self):
        m = D.build_model()
        sr = next(n for n, _ in D.iter_nodes(m) if n['role'] == 'stage_retry')
        self.assertEqual(set(sr['reacts']), set(ev.STAGE_FAILURE_TYPES))


class RunAlignment(unittest.TestCase):
    """Align a synthetic two-stage run (frames + bt_events) and check windows + video timecodes."""
    def _write_run(self, tmp):
        run = Path(tmp)
        # two stages, a nav step then a verify step each; sim 0->10->20
        bt = []
        for k, stage in enumerate(('collect', 'couple')):
            t0 = k * 10.0
            bt += tmpl.synthetic_bt_events(stage=stage, plan=[('a', 'nav'), ('b', 'verify')],
                                           t0=t0, dt=5.0)
        bt.append({'t': 20.0, 'type': 'mission_complete',
                   'data': {'state': 'complete', 'sim_time': 20.0}, 'action': 'stage'})
        (run / 'bt_events.jsonl').write_text('\n'.join(json.dumps(e) for e in bt) + '\n')
        (run / 'mission.json').write_text(json.dumps({'state': 'complete', 'reason': ''}))
        # 200 frames over sim 0..20s -> 10 fps source; video at 20 fps => 10 video seconds
        fdir = run / 'mission-frames'
        fdir.mkdir()
        frames = [{'file': f'{i:06d}.png', 'simulation_time': i * 0.1} for i in range(201)]
        (fdir / 'frames.json').write_text(json.dumps({'frames': frames}))
        return run

    def test_align_windows_and_video_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._write_run(tmp)
            align = D.align_to_run(run, fps=20.0)
            self.assertEqual(align['mission_state'], 'complete')
            self.assertEqual(align['stage']['collect']['status'], 'success')
            self.assertAlmostEqual(align['stage']['couple']['sim_start'], 10.0)
            # couple begins at sim 10 => frame 100 => video 100/20 = 5.0s
            self.assertAlmostEqual(align['stage']['couple']['video_start'], 5.0, places=1)
            self.assertAlmostEqual(align['video_duration'], 201 / 20.0, places=2)

    def test_annotate_colours_stage_and_step_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._write_run(tmp)
            align = D.align_to_run(run, fps=20.0)
            model = D.build_model(stage_order=['collect', 'couple'],
                                  plans={'collect': [('a', 'nav'), ('b', 'verify')],
                                         'couple': [('a', 'nav'), ('b', 'verify')]})
            D.annotate_model(model, align)
            stage = next(n for n, _ in D.iter_nodes(model)
                         if n['role'] == 'stage_retry' and n['stage'] == 'couple')
            self.assertEqual(stage['active']['status'], 'success')
            self.assertIsNotNone(stage['active']['video_start'])
            guard = next(n for n, _ in D.iter_nodes(model) if n['role'] == 'human_safety')
            self.assertEqual(guard['active']['status'], 'armed')  # never fired in this run


class RenderSmoke(unittest.TestCase):
    def test_svg_and_html_render(self):
        model = D.build_model()
        svg = D.render_svg(model, title='t')
        self.assertTrue(svg.startswith('<svg'))
        self.assertIn('restock_root', svg)
        html = D.render_html(model, video_url='v.mp4', title='t', svg_filename='x.svg')
        self.assertIn('<!doctype html>', html)
        self.assertIn('stage:collect', html)
        # self-contained: no external script/style/src besides the one video we pass in
        self.assertNotIn('http://', html)
        self.assertNotIn('https://', html)


class AddStageRecipe(unittest.TestCase):
    """The tested one-file recipe: the template registers a dummy stage; it must show up in the
    diagram model AND in a replayed timeline (diagram alignment), with no edit to the real PLANS."""
    def test_dummy_stage_appears_in_diagram(self):
        plans = tmpl.register_plan()          # real PLANS + the example 'inspect' stage
        self.assertNotIn('inspect', sp.PLANS)  # real PLANS untouched (non-mutating)
        model = D.build_model(plans=plans)
        stages = [n['stage'] for n, _ in D.iter_nodes(model) if n['role'] == 'stage_retry']
        self.assertIn('inspect', stages)
        gates = [n['name'] for n, _ in D.iter_nodes(model)
                 if n['role'] == 'phase_gate' and n['stage'] == 'inspect']
        self.assertEqual(gates, [f'inspect:{label}' for label, _ in tmpl.TEMPLATE_PLAN])

    def test_dummy_stage_appears_in_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            bt = tmpl.synthetic_bt_events(t0=0.0, dt=2.0)
            (run / 'bt_events.jsonl').write_text('\n'.join(json.dumps(e) for e in bt) + '\n')
            (run / 'mission.json').write_text(json.dumps({'state': 'complete', 'reason': ''}))
            align = D.align_to_run(run, fps=20.0)
            self.assertIn('inspect', align['stage'])
            self.assertEqual(align['stage']['inspect']['status'], 'success')
            # each template leaf got its own window
            n_steps = len(tmpl.TEMPLATE_PLAN)
            got = [k for k in align['step'] if k[0] == 'inspect']
            self.assertEqual(len(got), n_steps)

    def test_milestone_classifier_shape(self):
        # mirrors the real stage_phases classifiers: phase -> monotonic step index
        self.assertEqual(tmpl.inspect_milestone(SimpleNamespace(phase='navigate')), 0)
        self.assertEqual(tmpl.inspect_milestone(SimpleNamespace(phase='warm')), 1)
        self.assertEqual(tmpl.inspect_milestone(
            SimpleNamespace(phase='execute', _last_manip_phase='verify')), 2)


@unittest.skipUnless(HAVE_PT, 'py_trees not installed (task-owned .venv-bt)')
class ParityWithRealTree(unittest.TestCase):
    """The reconstructed model must match the REAL serialised py_trees tree node-for-node."""
    def _real_tree_signature(self):
        from twin.restock.bt.tree import build_tree
        from twin.restock.bt.view import serialise_node
        from twin.restock.bt.context import MissionContext
        from twin.restock.bt.driver import build_bus
        from twin.restock.mission import ACTIONS

        # A minimal phased adapter (exposes `phase`, no mode tags, coupled return) + a stub mission
        # carrying the full stage tuple -- enough for build_tree to author the structure.
        adapter = SimpleNamespace(phase='navigate')
        mission = SimpleNamespace(adapter=adapter, stages=list(ACTIONS), start_index=0)
        bus, _ = build_bus()
        ctx = MissionContext(mission, bus)
        root = build_tree(ctx)
        return [(n['name'], n['kind']) for n in _iter_serialised(serialise_node(root))]

    def test_structure_matches(self):
        real = self._real_tree_signature()
        model = _names_cls(D.build_model())
        self.assertEqual(model, real)


def _iter_serialised(node):
    yield node
    for c in node.get('children', []):
        yield from _iter_serialised(c)


if __name__ == '__main__':
    unittest.main()
