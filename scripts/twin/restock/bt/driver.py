"""Tick the restock mission tree against a real :class:`mission.Mission`.

The driver owns the tick clock (so scenario events fire deterministically off tick time, even while
a stage is held), ticks the tree until the mission is terminal or the step budget is exhausted, and
optionally writes a live snapshot each tick. It is a behaviour-preserving *alternative* to
``Mission.tick`` looping: it drives the SAME ``begin_stage`` / ``step_stage`` / ``finalize`` seams,
so inventory, evidence validation and the terminal record are byte-identical to the state machine;
only the orchestration (bounded retry + reactive handling) is new.
"""
from __future__ import annotations

import py_trees

from .bus import EventBus, QueueEventSource, ScenarioEventSource
from .context import CLOCK, MissionContext
from .tree import build_tree


def build_bus(scenario=None, *, extra_sources=()):
    """An :class:`EventBus` with a scenario source (if the scenario declares ``events``) plus a
    thread-safe :class:`QueueEventSource` a ROS subscriber (or a test) can ``push`` onto. Returns
    ``(bus, queue)`` so the caller keeps a handle on the queue."""
    sources = []
    if scenario is not None and scenario.get('events'):
        sources.append(ScenarioEventSource(scenario))
    queue = QueueEventSource()
    sources.append(queue)
    sources.extend(extra_sources)
    return EventBus(sources), queue


def run_mission_tree(mission, bus, *, reactions=None, dt=1 / 60, max_ticks=60 * 20 * 60,
                     snapshot=None, max_retries=2, on_tick=None, is_running=None):
    """Drive ``mission`` to a terminal state through the behaviour tree. Returns a summary dict
    (same shape the state-machine run persists, plus ``orchestrator='behaviour_tree'`` and a tick
    count). ``mission`` must be freshly constructed (state ``'new'``); this calls
    ``mission.start(begin=False)`` so the tree begins the first stage itself.

    * ``reactions`` -- a :class:`Reactions` implementation (defaults to a recording no-op).
    * ``snapshot``  -- an optional :class:`view.SnapshotWriter`; written once per tick.
    * ``on_tick(i, ctx, root)`` -- optional callback after each tick (e.g. picture-in-picture).
    * ``is_running()`` -- optional liveness check (GPU run passes ``app.is_running``).
    """
    mission.start(begin=False)
    ctx = MissionContext(mission, bus, reactions=reactions, snapshot=snapshot)
    root = build_tree(ctx, max_retries=max_retries)
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup()

    ticks = 0
    now = 0.0
    for i in range(int(max_ticks)):
        if mission.terminal:
            break
        if is_running is not None and not is_running():
            mission.fail('simulation app stopped')
            break
        now += dt
        ctx.dt = dt
        ctx.now = round(now, 6)
        ctx.bb.set(CLOCK, round(now, 3))
        tree.tick()
        ticks = i + 1
        if snapshot is not None:
            snapshot.write(ctx, root, tick=i)
        if on_tick is not None:
            on_tick(i, ctx, root)
        if root.status == py_trees.common.Status.FAILURE and mission.terminal:
            break
    if not mission.terminal:
        mission.fail('tree step budget exceeded')
        if snapshot is not None:
            snapshot.write(ctx, root, tick=ticks)

    summary = {'state': mission.state, 'reason': mission.reason,
               'orchestrator': 'behaviour_tree', 'ticks': ticks,
               'degraded': list(mission.degraded),
               'physics_verified': False,
               'note': 'Mission success alone does not verify video, contacts or autonomy.'}
    if mission.shortcut is not None:
        summary['shortcut'] = mission.shortcut
    return ctx, root, summary
