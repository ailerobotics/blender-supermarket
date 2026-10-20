"""Compose the restock mission behaviour tree -- the mission's run DRIVER.

Shape (left-to-right = tick order under memory=False roots)::

    restock_root  (Sequence, memory=False)        -- re-evaluated every tick
      |- event_pump                               -- drain bus -> blackboard, recompute control
      |- reactive  (Sequence, memory=False)       -- all handlers evaluate each tick
      |    |- health_guard      (stuck/low_battery -> safe park, FAILS to tear down)
      |    |- human_safety      (slow -> stop + pause arm -> resume)
      |    |- obstacle_avoid    (dynamic obstacle / blocked aisle -> wait + re-plan)
      |    `- baseline_guard    (baseline_expired -> re-commission via stations provider)
      `- mission   (Sequence, memory=True)         -- resumes at the running stage
           |- stage:collect  (StageRetry) -> "collect steps" (Sequence, memory=True)
           |      |- collect:drive->pick         (navigate)
           |      |- collect:perceive carton     (perceive)
           |      |- collect:grasp.lift.tuck     (manipulate)
           |      |- collect:drive->rollwagen    (navigate)   <- position-first: move, THEN place
           |      |- collect:perceive deck       (perceive)
           |      |- collect:place               (manipulate)
           |      `- collect:verify placed       (verify)
           |- stage:couple / outbound / park / transfer / verify / return  (one subtree each)
           `- finalize        (mission.finalize once)

Each stage is a **subtree of explicit position-first leaves** (navigate -> perceive -> manipulate,
``stage_phases.PLANS``): a new step is a new leaf. One shared :class:`leaves.StageRun` steps the real
``step_stage`` seam once per tick; the leaves are faithful views of that one stepping gated on the
adapter's observed phase, so the decomposed drive is byte-identical to the state machine (parity).
The :class:`leaves.StageRetry` wrapper holds bounded retry + the recovery reaction as one named node.
A non-phased adapter (a plain fixture) collapses each stage to a single leaf (byte-identical to the
former opaque ``Stage``). On ``return`` an opt-in uncouple-at-parking-spot plan adds the new steps
tow->cart park / brake cart / uncouple / back away / drive->home.

Why this shape: ``event_pump`` and ``reactive`` run before ``mission`` every tick, so a safety
event is observed and the control flags/physical reactions are applied *before* the active stage
decides whether to step. The active leaf reads the pause flag and HOLDS (safety stop, payload
preserved) rather than stepping -- so "the reactive subtree handles the event while the mission
continues (then resumes)" is literally what the tree does, and the live view shows it node by node.
``health_guard`` is first in the reactive sequence and the only reactive leaf that FAILS, so an
abort short-circuits before the mission ticks.
"""
from __future__ import annotations

import py_trees

from .leaves import (BaselineGuard, EventPump, Finalize, HealthGuard, HumanSafety,
                     ObstacleAvoid, PhaseGate, StageRetry, StageRun)
from .stage_phases import stage_plan


def active_stages(mission):
    """The stages the tree actually drives: the mission's stage tuple from its ``start_index`` on
    (start_at-aware). The skipped prefix was recorded as ``*_shortcut`` by ``start(begin=False)``."""
    return list(mission.stages[mission.start_index:])


def build_stage(ctx, action, *, max_retries=2):
    """Build the subtree that drives ONE mission stage: a memory sequence of explicit position-first
    leaves (navigate -> perceive -> manipulate -> ...), wrapped in a bounded-retry/recovery node.

    When the adapter exposes per-stage phases (the real runtime), the sequence has one
    :class:`PhaseGate` per step in the stage's plan (``stage_phases.PLANS``); otherwise (a plain
    mock/fixture) it collapses to a single leaf that drives the whole stage -- byte-identical to the
    pre-decomposition ``Stage`` so the offline/mock parity harnesses are unchanged. One shared
    :class:`StageRun` steps the mission; the leaves are faithful views of that one stepping."""
    adapter = getattr(ctx.mission, 'adapter', None)
    plan = stage_plan(action, adapter)
    steps = plan if plan else [(action, 'stage')]
    run = StageRun(ctx, action, len(steps))
    seq = py_trees.composites.Sequence(name=f'{action} steps', memory=True)
    seq.add_children([PhaseGate(run, i, label, kind) for i, (label, kind) in enumerate(steps)])
    return StageRetry(seq, run, max_retries=max_retries)


def build_reactive(ctx):
    reactive = py_trees.composites.Sequence(name='reactive', memory=False)
    reactive.add_children([
        HealthGuard(ctx),
        HumanSafety(ctx),
        ObstacleAvoid(ctx),
        BaselineGuard(ctx),
    ])
    return reactive


def build_mission(ctx, *, max_retries=2):
    mission_seq = py_trees.composites.Sequence(name='mission', memory=True)
    mission_seq.add_children([build_stage(ctx, action, max_retries=max_retries)
                              for action in active_stages(ctx.mission)])
    mission_seq.add_child(Finalize(ctx))
    return mission_seq


def build_tree(ctx, *, max_retries=2):
    """Build the full mission tree for ``ctx``. Returns the root behaviour (not yet set up)."""
    root = py_trees.composites.Sequence(name='restock_root', memory=False)
    root.add_children([
        EventPump(ctx),
        build_reactive(ctx),
        build_mission(ctx, max_retries=max_retries),
    ])
    return root
