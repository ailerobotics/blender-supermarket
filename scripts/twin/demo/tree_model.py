"""The PR153 behaviour-tree structure, rendered as a static tree the demo panel draws.

The node structure is taken VERBATIM from the real behaviour tree's authored plans
(:mod:`twin.restock.bt.stage_phases`) so the panel always matches what the autonomous
mission would tick: the seven stage subtrees and their ``(label, kind)`` child steps.
Nothing here imports ``py_trees`` (it is not installed in the sim env); only the pure
plan constants are read, so the panel stays faithful without the orchestrator.

A per-frame ``annotate(states)`` stamps each node with a status + reason and returns a
nested dict in exactly the shape of ``twin.restock.bt.view.serialise_node`` (name /
status / feedback / kind / children), so the panel renderer is the same mental model as
the live ``bt_view.html``.
"""
from __future__ import annotations

from twin.restock.bt import stage_phases

# Status vocabulary (mirrors py_trees.common.Status via bt.view._STATUS_NAME).
IDLE, RUNNING, SUCCESS, FAILURE = 'idle', 'running', 'success', 'failure'
STATUSES = (IDLE, RUNNING, SUCCESS, FAILURE)

# The demo's return stage uncouples the cart at its parking spot (captain 2026-10-20),
# so it uses the RETURN_UNCOUPLE_PLAN rather than the coupled tow-home default.
STAGE_ORDER = tuple(stage_phases.PLANS)  # ('collect','couple','outbound','park','transfer','verify','return')

# Human-readable stage titles for the panel header row.
STAGE_TITLES = {
    'collect': '1 · Collect',
    'couple': '3 · Couple',
    'outbound': '4 · Tow outbound',
    'park': '5 · Park',
    'transfer': '6 · Transfer',
    'verify': '7 · Verify',
    'return': '8 · Return',
}


def stage_steps(stage):
    """The ordered ``(label, kind)`` plan for a stage, with ``return`` resolved to the
    uncouple-at-parking plan used by the demo."""
    if stage == 'return':
        return list(stage_phases.RETURN_UNCOUPLE_PLAN)
    return list(stage_phases.PLANS[stage])


def build_structure():
    """The static tree: a list of ``(stage, [(label, kind), ...])`` in tick order."""
    return [(stage, stage_steps(stage)) for stage in STAGE_ORDER]


def _node(name, status, feedback, kind, children=()):
    return {'name': name, 'status': status, 'feedback': feedback,
            'kind': kind, 'children': list(children)}


def annotate(states):
    """Build the serialised tree dict for one frame.

    ``states`` maps ``stage -> {'status': <status>, 'steps': [{'status','reason'}, ...]}``.
    A stage or step absent from ``states`` renders IDLE with no reason. Returns the root
    node dict (name/status/feedback/kind/children), ready for the panel renderer.
    """
    stage_nodes = []
    for stage, steps in build_structure():
        st = states.get(stage, {})
        step_states = st.get('steps', [])
        step_nodes = []
        for i, (label, kind) in enumerate(steps):
            ss = step_states[i] if i < len(step_states) else {}
            step_nodes.append(_node(f'{stage}:{label}', ss.get('status', IDLE),
                                    ss.get('reason', ''), kind))
        stage_nodes.append(_node(f'stage:{stage}', st.get('status', IDLE),
                                 st.get('feedback', ''), 'StageRetry', step_nodes))
    return _node('restock mission', _roll_up([n['status'] for n in stage_nodes]),
                 '', 'Sequence', stage_nodes)


def _roll_up(child_statuses):
    """Sequence roll-up: FAILURE if any failed, else RUNNING if any running or a mix of
    done/idle (in progress), else SUCCESS if all success, else IDLE."""
    if FAILURE in child_statuses:
        return FAILURE
    if RUNNING in child_statuses:
        return RUNNING
    if child_statuses and all(s == SUCCESS for s in child_statuses):
        return SUCCESS
    if any(s == SUCCESS for s in child_statuses):
        return RUNNING  # some stages done, others idle -> mission in progress
    return IDLE
