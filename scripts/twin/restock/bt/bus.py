"""Event sources and the bus that drains them into the tree each tick -- pure Python.

A *source* answers ``poll(now) -> list[Event]`` with the events to apply this tick. The tree's
``EventPump`` leaf polls the :class:`EventBus` once per tick (passing the mission/sim clock) and
applies the results to the blackboard. Sources are deliberately simple and side-effect free so
the whole reactive path is deterministic and CPU-testable without ROS or a simulator.
"""
from __future__ import annotations

import collections
import threading

from .events import Event, parse_scenario_events


class ScenarioEventSource:
    """Deterministic, sim-time-triggered events parsed from the scenario ``events`` list.

    Fires each event exactly once, the first time ``poll(now)`` is called with ``now >= t``.
    Monotonic: the internal cursor only moves forward, so a non-monotonic ``now`` never
    re-fires a past event."""
    def __init__(self, scenario_or_events):
        if isinstance(scenario_or_events, dict):
            self._pending = parse_scenario_events(scenario_or_events)
        else:
            self._pending = sorted(scenario_or_events, key=lambda se: se.t)
        self._i = 0

    def poll(self, now):
        out = []
        while self._i < len(self._pending) and self._pending[self._i].t <= now:
            out.append(self._pending[self._i].event)
            self._i += 1
        return out

    @property
    def remaining(self):
        return len(self._pending) - self._i


class QueueEventSource:
    """Thread-safe FIFO for externally produced events (e.g. a ROS subscriber callback pushes
    here from its executor thread; the tree drains on its own tick thread)."""
    def __init__(self):
        self._q = collections.deque()
        self._lock = threading.Lock()

    def push(self, event: Event):
        if not isinstance(event, Event):
            raise TypeError('push expects an Event')
        with self._lock:
            self._q.append(event)

    def poll(self, now):
        out = []
        with self._lock:
            while self._q:
                out.append(self._q.popleft())
        return out


class EventBus:
    """Fan-in over several sources. ``poll(now)`` concatenates each source's events in order."""
    def __init__(self, sources=()):
        self.sources = list(sources)

    def add_source(self, source):
        self.sources.append(source)
        return source

    def poll(self, now):
        out = []
        for source in self.sources:
            out.extend(source.poll(now))
        return out
