"""Template + one-file recipe for adding a new mission stage to the behaviour tree.

Read this top-to-bottom with ``docs/behaviour_tree.md`` -- it is the worked example the add-a-stage
recipe points at, and it is exercised by ``scripts/twin/tests/test_restock_bt_diagram.py`` so the
recipe cannot rot. Nothing here is wired into a live mission; it is a copy-me scaffold.

A stage in this tree is *data-driven*: the tree (``tree.build_stage``) and the diagram
(``diagram.build_model``) both build a stage's subtree from its ordered ``(label, kind)`` plan, and
the timeline/diagram read that same plan back. So adding a stage is three small, local edits -- no
new leaf class, no tree surgery:

1. **Declare the plan** -- add a ``<stage>: [(label, kind), ...]`` entry to
   ``stage_phases.PLANS`` (one tuple == one position-first leaf: navigate -> perceive -> manipulate).
   This alone makes the stage + its steps appear in the diagram and the live view.
2. **Classify its progress** (only if the stage has >1 step) -- add a ``_<stage>_milestone(adapter)``
   that maps the adapter's observed ``phase`` to the current step index, and register it in
   ``stage_phases._CLASSIFIERS``. A single-step stage needs none (it sits on step 0 until it
   succeeds).
3. **Drive it** -- make ``mission.Mission`` run the stage (add it to the stage tuple and implement
   its ``begin_stage`` / ``step_stage`` behaviour). The tree then picks it up automatically because
   ``tree.active_stages`` reads ``mission.stages``.

The constants and functions below are a filled-in example for a fictional ``inspect`` stage (drive to
a QR gantry, perceive the code, confirm it). ``register_plan`` returns a *new* plans dict with the
example added -- pure and non-mutating -- which is exactly how the unit test proves the stage shows
up in both the diagram and a replayed timeline without touching the real ``PLANS``.
"""
from __future__ import annotations

from . import stage_phases as _sp

# 1. The example stage's plan. kinds are drawn from stage_phases' vocabulary (NAV/PERCEIVE/MANIP/...).
TEMPLATE_STAGE = 'inspect'
TEMPLATE_PLAN = [
    ('drive→gantry', _sp.NAV),
    ('perceive code', _sp.PERCEIVE),
    ('confirm code', _sp.VERIFY),
]


# 2. The example progress classifier: observed adapter phase -> monotonic step index. Mirrors the
#    shape of stage_phases._collect_milestone / _couple_milestone. Pure; reads only attributes the
#    real RuntimeAdapter exposes.
def inspect_milestone(adapter) -> int:
    phase = getattr(adapter, 'phase', None)
    manip = getattr(adapter, '_last_manip_phase', None)
    if phase == 'navigate':
        return 0                       # driving to the gantry
    if phase in ('warm', 'rewarm'):
        return 1                       # perceiving the code
    return 2 if manip == 'verify' else 1   # confirm


def register_plan(plans=None, *, stage=TEMPLATE_STAGE, plan=None):
    """Return a NEW plans dict with the example stage appended (non-mutating).

    This is the hook the unit test and the recipe use to show a dummy stage flowing through the
    diagram + timeline without editing the real ``stage_phases.PLANS``. In a real change you instead
    add the entry to ``PLANS`` directly."""
    base = dict(_sp.PLANS if plans is None else plans)
    base[stage] = list(plan if plan is not None else TEMPLATE_PLAN)
    return base


def synthetic_bt_events(stage=TEMPLATE_STAGE, plan=None, *, t0=0.0, dt=1.0):
    """A minimal ``bt_events.jsonl``-shaped list for the example stage: one begin, one step per leaf,
    one succeeded -- exactly what the BT driver (``leaves.StageRun``) writes. The timeline replay and
    :func:`diagram.align_to_run` consume this, so the test can prove the new stage colours + links in
    the diagram from a recorded run too."""
    plan = list(plan if plan is not None else TEMPLATE_PLAN)
    events = [{'t': t0, 'type': 'stage_begin',
               'data': {'stage': stage, 'attempt': 1, 'sim_time': t0,
                        'steps': [label for label, _ in plan]}, 'action': 'stage'}]
    t = t0
    for i, (label, kind) in enumerate(plan):
        events.append({'t': t, 'type': 'stage_step',
                       'data': {'stage': stage, 'attempt': 1, 'sim_time': t, 'step': i,
                                'label': label, 'kind': kind}, 'action': 'stage'})
        t += dt
    events.append({'t': t, 'type': 'stage_succeeded',
                   'data': {'stage': stage, 'attempt': 1, 'sim_time': t}, 'action': 'stage'})
    return events
