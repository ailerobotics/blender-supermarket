"""Compose the restock mission behaviour tree.

Shape (left-to-right = tick order under memory=False roots)::

    restock_root  (Sequence, memory=False)        -- re-evaluated every tick
      |- event_pump                               -- drain bus -> blackboard, recompute control
      |- reactive  (Sequence, memory=False)       -- all handlers evaluate each tick
      |    |- health_guard      (stuck/low_battery -> safe park, FAILS to tear down)
      |    |- human_safety      (slow -> stop + pause arm -> resume)
      |    |- obstacle_avoid    (dynamic obstacle / blocked aisle -> wait + re-plan)
      |    `- baseline_guard    (baseline_expired -> re-commission via stations provider)
      `- mission   (Sequence, memory=True)         -- resumes at the running stage
           |- stage:collect   (bounded retry + re-observe recovery)
           |- stage:couple    (bounded retry)
           |- ... (one per active stage, start_at-aware) ...
           `- finalize        (mission.finalize once)

Why this shape: ``event_pump`` and ``reactive`` run before ``mission`` every tick, so a safety
event is observed and the control flags/physical reactions are applied *before* the active stage
decides whether to step. The stage itself reads the pause flag and HOLDS (safety stop, payload
preserved) rather than stepping -- so "the reactive subtree handles the event while the mission
continues (then resumes)" is literally what the tree does, and the live view shows it node by node.
``health_guard`` is first in the reactive sequence and the only reactive leaf that FAILS, so an
abort short-circuits before the mission ticks.
"""
from __future__ import annotations

import py_trees

from .leaves import (BaselineGuard, EventPump, Finalize, HealthGuard, HumanSafety,
                     ObstacleAvoid, Stage)


def active_stages(mission):
    """The stages the tree actually drives: the mission's stage tuple from its ``start_index`` on
    (start_at-aware). The skipped prefix was recorded as ``*_shortcut`` by ``start(begin=False)``."""
    return list(mission.stages[mission.start_index:])


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
    mission_seq.add_children([Stage(ctx, action, max_retries=max_retries)
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
