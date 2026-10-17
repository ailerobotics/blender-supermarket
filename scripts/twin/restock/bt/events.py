"""Event contract for the behaviour-tree mission layer -- pure Python.

No py_trees, no ROS, no simulator imports: this module is the single source of truth for
*what an event is*, so it is importable from ``scenario.py`` (to validate the deterministic
scenario ``events`` list) and from a ROS subscriber node (to parse the wire payload) without
dragging in py_trees or rclpy.

Two producers feed the same :class:`Event` type into the tree's event bus:

* **Deterministic scenario events** -- an optional top-level ``"events"`` list in the mission
  scenario JSON, each ``{"t": <sim seconds>, "type": <event type>, "data": {...}}``. Injected
  once when the mission clock reaches ``t`` (see :class:`ScenarioEventSource`). This is how a
  scenario *declares* what happens so the tree can *show* how it is handled -- the path used by
  CPU tests and the scripted GPU proof.
* **Runtime detections** -- the humans and dynamic-obstacle lanes publish ``std_msgs/msg/String``
  JSON events on the ROS 2 topic :data:`EVENT_TOPIC`. :func:`Event.from_json` parses them and
  :func:`detection_to_event` converts a ``navigation.dynamic.Detection`` into a canonical
  ``dynamic_obstacle`` event (so the dynobs lane need not restructure PR #114).

See ``data/sim-restock-bt-v1/report.md`` §2 for the full table and
``docs/restock_bt.md`` for the reactive handling of each type.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

# ROS 2 topic the humans / dynamic-obstacle lanes publish detections onto (std_msgs/String JSON).
EVENT_TOPIC = '/restock_bt/events'

# Closed set of canonical event types. Unknown types on the wire are logged and ignored, never
# raised, so a lane publishing a future type cannot crash the mission tree.
EVENT_TYPES = frozenset({
    'human_detected', 'human_cleared',
    'dynamic_obstacle', 'obstacle_cleared',
    'blocked_aisle',
    'grasp_failed', 'seal_failed',
    'tracking_miss', 'goal_miss',
    'baseline_expired',
    'coupling_failed',
    'stuck', 'low_battery',
})

# Persistent-condition events and the event that clears each. Everything else is momentary
# (consumed by its handler / logged by the failing stage); these stay 'active' until cleared.
CLEARED_BY = {'human_cleared': 'human_detected', 'obstacle_cleared': 'dynamic_obstacle'}

# Momentary stage-failure events a Stage leaf derives from a failed step_stage() reason, so the
# live timeline shows WHAT failed; the bounded Retry decorator owns the actual retry/abort.
STAGE_FAILURE_TYPES = frozenset({
    'grasp_failed', 'seal_failed', 'tracking_miss', 'goal_miss', 'coupling_failed', 'blocked_aisle',
})

# The physical recovery a Stage leaf requests (via Reactions) before retrying each failure type.
# Manipulation re-observes before re-grasping; nav re-plans; coupling simply re-attempts.
RECOVERY_FOR = {
    'grasp_failed': 'reobserve', 'seal_failed': 'reobserve',
    'blocked_aisle': 'replan', 'goal_miss': 'replan', 'tracking_miss': 'replan',
    'coupling_failed': 'wait',
}


def classify_stage_failure(stage: str, reason: str) -> str:
    """Map a failed ``step_stage`` (stage, reason) onto a canonical stage-failure event type, so the
    live timeline names WHAT failed. Pure and best-effort: the reason string still carries the full
    detail; this only picks the label. Falls back by stage when the reason is unrecognised.

    Known runtime reasons (see report §1): nav emits ``blocked_no_route_timeout`` /
    ``goal_timeout`` / ``obstacle_stop_timeout``; coupling emits ``base moved during stationary
    operation`` / ``cart lost before transfer``; manipulation emits ``reach stalled: ...`` /
    ``transfer evidence mismatch``."""
    text = (reason or '').lower()
    if 'seal' in text or 'transfer evidence' in text:
        return 'seal_failed'
    if 'block' in text or 'no_route' in text or 'no route' in text:
        return 'blocked_aisle'
    if 'goal' in text:
        return 'goal_miss'
    if 'obstacle' in text or 'track' in text:
        return 'tracking_miss'
    if 'cart' in text or 'coupl' in text or 'stationary' in text:
        return 'coupling_failed'
    if 'reach' in text or 'grasp' in text or 'stall' in text:
        return 'grasp_failed'
    # Stage fallback: manipulation stages grasp, coupling couples, nav stages miss goals.
    if stage in ('collect',):
        return 'grasp_failed'
    if stage in ('transfer',):
        return 'seal_failed'
    if stage in ('couple',):
        return 'coupling_failed'
    return 'goal_miss'


@dataclass(frozen=True)
class Event:
    """One mission event. ``data`` is a JSON-serialisable dict of type-specific fields."""
    type: str
    data: dict = field(default_factory=dict)
    stamp_ns: int = 0   # producer clock (sim or wall) in nanoseconds; 0 = unstamped

    def __post_init__(self):
        if self.type not in EVENT_TYPES:
            raise ValueError(f'unknown event type {self.type!r}; expected one of {sorted(EVENT_TYPES)}')
        if not isinstance(self.data, dict):
            raise ValueError('event data must be a dict')
        if isinstance(self.stamp_ns, bool) or not isinstance(self.stamp_ns, int):
            raise ValueError('stamp_ns must be an int (nanoseconds)')

    def to_json(self) -> str:
        return json.dumps({'type': self.type, 'data': self.data, 'stamp_ns': self.stamp_ns},
                          allow_nan=False, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> 'Event':
        """Parse a wire payload. Raises ValueError on malformed JSON / unknown type so a bad
        publisher is rejected at the edge rather than corrupting the blackboard."""
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f'invalid event JSON: {exc}')
        if not isinstance(obj, dict) or 'type' not in obj:
            raise ValueError('event payload must be a JSON object with a "type"')
        return cls(type=obj['type'], data=obj.get('data') or {}, stamp_ns=int(obj.get('stamp_ns', 0)))


def detection_to_event(detection, *, stamp_ns: int = 0) -> Event:
    """Convert a ``navigation.dynamic.Detection(position, radius, source)`` into the canonical
    ``dynamic_obstacle`` event. Accepts any object exposing those attributes so the dynobs lane
    can hand us its own detection without importing this module's types."""
    return Event('dynamic_obstacle',
                 {'position': [float(detection.position[0]), float(detection.position[1])],
                  'radius': float(detection.radius),
                  'source': str(getattr(detection, 'source', 'dynamic_detection'))},
                 stamp_ns)


@dataclass(frozen=True)
class ScenarioEvent:
    """A validated deterministic scenario event: fire :attr:`event` once at sim time :attr:`t`."""
    t: float
    event: Event


def validate_scenario_events(scenario) -> list:
    """Return a list of human-readable error strings for the optional scenario ``events`` list
    (empty => valid or absent). Backward compatible: a scenario with no ``events`` key is valid.

    Called by ``scenario.py:validate``; pure, so it adds no py_trees/ROS dependency there."""
    raw = scenario.get('events')
    if raw is None:
        return []
    errors = []
    if not isinstance(raw, list):
        return ['scenario events must be a list']
    last_t = -math.inf
    for i, row in enumerate(raw):
        where = f'events[{i}]'
        if not isinstance(row, dict):
            errors.append(f'{where}: must be an object'); continue
        t = row.get('t')
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            errors.append(f'{where}: t must be a finite, nonnegative sim time')
        elif t < last_t:
            errors.append(f'{where}: events must be in nondecreasing sim-time order')
        else:
            last_t = t
        etype = row.get('type')
        if etype not in EVENT_TYPES:
            errors.append(f'{where}: unknown type {etype!r}')
        data = row.get('data', {})
        if data is not None and not isinstance(data, dict):
            errors.append(f'{where}: data must be an object')
    return errors


def parse_scenario_events(scenario) -> list:
    """Parse the scenario ``events`` list into sorted :class:`ScenarioEvent`s. Raises ValueError
    if invalid (call :func:`validate_scenario_events` first for friendly errors)."""
    errors = validate_scenario_events(scenario)
    if errors:
        raise ValueError('invalid scenario events: ' + '; '.join(errors))
    out = []
    for row in scenario.get('events') or []:
        out.append(ScenarioEvent(float(row['t']),
                                 Event(row['type'], dict(row.get('data') or {}),
                                       int(row.get('stamp_ns', 0)))))
    out.sort(key=lambda se: se.t)
    return out
