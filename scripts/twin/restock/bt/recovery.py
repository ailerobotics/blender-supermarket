"""Feed recovery observations into the tree's event bus -- pure Python, no py_trees / ROS / sim.

A *recovery* (a nav stall recovery during the coupled tow, a VRAM headroom shed, a host
station-liveness e-stop) is handled ENTIRELY by its owning lane's code, OUTSIDE the behaviour tree
(see :data:`events.RECOVERY_SPECS`). This module is the thin seam that lets the tree *observe* one
as it happens, so the live ``/bt`` view and the generated diagram/timeline show the recovery node
light up and return to nominal. It never performs or changes a recovery.

Two producers, both reducing to the same edge-detected :class:`events.Event` on the bus:

* **Push** -- a recovery code path that already holds a handle calls
  :meth:`RecoveryObserver.note(recovery_type, active, data)` when it engages / clears. The observer
  de-dups (no event while the active state + data are unchanged), so a per-tick call is cheap and
  the timeline is not spammed.
* **Poll** -- :class:`RecoveryPollSource` is a bus source (``poll(now) -> [Event]``) that reads a
  best-effort *provider* once per tick and emits the same edge-detected events. A provider returns
  the current recovery signals; absent / ``None`` / raising ⇒ no events ⇒ the run is byte-identical
  (this is why the seam is default-on: it is inert until a provider is wired).

A *signal* is ``(recovery_type, active: bool, data: dict|None)``. ``active`` True while the recovery
is engaged; the edge to False emits the ``cleared_by`` event (from :data:`events.RECOVERY_SPECS`).
"""
from __future__ import annotations

from .events import RECOVERY_SPECS, Event


def _normalise(signal):
    """Coerce a provider signal into ``(recovery_type, bool active, dict data)``; raise ValueError on
    an unknown recovery type so a typo is caught at the edge, never silently dropped."""
    rtype, active = signal[0], bool(signal[1])
    data = signal[2] if len(signal) > 2 and signal[2] is not None else {}
    if rtype not in RECOVERY_SPECS:
        raise ValueError(f'unknown recovery type {rtype!r}; expected one of {sorted(RECOVERY_SPECS)}')
    if not isinstance(data, dict):
        raise ValueError('recovery signal data must be a dict')
    return rtype, active, data


class _EdgeState:
    """Per-recovery-type edge detector shared by the push and poll producers: remembers the last
    (active, data) and yields the Event(s) for a transition. Emitting on first-active, on a
    data change while active, and on the active->inactive edge (the cleared event)."""

    def __init__(self):
        self._last = {}   # recovery_type -> (active, data) last emitted

    def transition(self, rtype, active, data):
        """Return the Event to emit for this observation, or ``None`` when nothing changed."""
        rtype, active, data = _normalise((rtype, active, data))
        prev = self._last.get(rtype)
        if active:
            if prev is not None and prev[0] and prev[1] == data:
                return None                       # unchanged active observation: no new event
            self._last[rtype] = (True, dict(data))
            return Event(rtype, dict(data))
        # inactive
        if prev is None or not prev[0]:
            self._last[rtype] = (False, {})
            return None                           # was already clear (or never seen): nothing to clear
        self._last[rtype] = (False, {})
        return Event(RECOVERY_SPECS[rtype]['cleared_by'], dict(data))


class RecoveryObserver:
    """Push seam: a recovery code path calls :meth:`note` when it engages or clears; the observer
    pushes the edge-detected :class:`events.Event` onto ``queue`` (a :class:`bus.QueueEventSource`).
    Best-effort and de-duped, so a producer can call it every tick without spamming the timeline."""

    def __init__(self, queue):
        self._queue = queue
        self._edges = _EdgeState()

    def note(self, recovery_type, active, data=None):
        event = self._edges.transition(recovery_type, active, data)
        if event is not None:
            self._queue.push(event)
        return event

    def engaged(self, recovery_type, data=None):
        return self.note(recovery_type, True, data)

    def cleared(self, recovery_type):
        return self.note(recovery_type, False)


class RecoveryPollSource:
    """Poll seam: a bus source that reads a best-effort recovery *provider* each tick and emits the
    edge-detected recovery events. ``provider`` is a zero-arg callable returning an iterable of
    ``(recovery_type, active, data)`` signals (or ``None``); absent / ``None`` / raising ⇒ no events,
    so an unwired runtime is byte-identical."""

    def __init__(self, provider):
        self._provider = provider
        self._edges = _EdgeState()

    def poll(self, now):
        if self._provider is None:
            return []
        try:
            signals = self._provider()
        except Exception:
            return []                             # a broken provider must never crash the tick
        if not signals:
            return []
        out = []
        for signal in signals:
            try:
                event = self._edges.transition(signal[0], signal[1],
                                               signal[2] if len(signal) > 2 else None)
            except Exception:
                continue                          # skip a malformed signal, keep the rest
            if event is not None:
                out.append(event)
        return out
