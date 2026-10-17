"""Opt-in behaviour-tree mission layer for the Isaac supermarket restock mission.

The production path is the ``mission.Mission`` state machine; this package adds an
*alternative orchestrator* that drives the SAME ``Mission`` (and therefore the same
adapter, inventory and evidence validation) through a py_trees behaviour tree, so it
adds reactive event handling (human / dynamic obstacle / grasp-seal / commissioning /
coupling / stuck-battery) and a live, phone-friendly visualization on top without
duplicating stage logic. Default-off: nothing here runs unless a caller opts in.

Only ``events`` is imported eagerly here-adjacent (it is pure Python with no py_trees /
ROS dependency, so ``scenario.py`` can validate the scenario ``events`` list without
pulling py_trees). The tree/leaf modules import py_trees lazily via their own imports.
"""
from .events import (Event, EVENT_TOPIC, EVENT_TYPES, ScenarioEvent,
                     classify_stage_failure, detection_to_event,
                     parse_scenario_events, validate_scenario_events)

__all__ = [
    'Event', 'EVENT_TOPIC', 'EVENT_TYPES', 'ScenarioEvent',
    'classify_stage_failure', 'detection_to_event',
    'parse_scenario_events', 'validate_scenario_events',
]


def build_mission_tree(ctx, **kw):
    """Lazy accessor for :func:`tree.build_tree` (imports py_trees only when called)."""
    from .tree import build_tree
    return build_tree(ctx, **kw)


def run_mission_tree(*args, **kw):
    """Lazy accessor for :func:`driver.run_mission_tree` (imports py_trees only when called)."""
    from .driver import run_mission_tree as _run
    return _run(*args, **kw)
