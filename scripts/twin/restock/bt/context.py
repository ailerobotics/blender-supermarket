"""Shared mission context + reaction hooks for the behaviour-tree layer.

The :class:`MissionContext` is the single object every leaf shares. It holds the real
``mission.Mission`` (so leaves drive production stage code, never a copy), the per-tick sim
timestep ``dt`` and clock ``now`` (set by the driver before each tick), the :class:`EventBus`,
the blackboard client, and the :class:`Reactions` hooks the capability lanes implement.

Blackboard layout (all under the two namespaces so a viewer can read them generically):

* ``/event/active``  -- dict ``{event_type: data}`` of currently-active persistent events.
* ``/event/log``     -- append-only list of ``{t, type, data, action}`` timeline entries.
* ``/event/last``    -- most recent event ``{t, type, data}`` (for a compact view header).
* ``/control/pause`` -- bool; a reactive handler sets it so the active stage HOLDS (safety stop
  without releasing a held payload) instead of stepping. The sim clock holds with it.
* ``/control/slow``  -- bool; a human in the slow zone requests reduced speed before a full stop.
* ``/control/replan``-- bool; an obstacle handler requests a nav re-plan.
* ``/control/abort`` -- None or a reason string; set by stuck/low-battery -> the mission aborts to
  a safe park (honest failure, never a spurious 'complete').
* ``/clock/now``     -- the tick clock (advances every tick, even while paused, so cleared-after
  timeouts and scenario clear events still fire during a hold).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import py_trees

from .events import Event

EVENT_ACTIVE = '/event/active'
EVENT_LOG = '/event/log'
EVENT_LAST = '/event/last'
CTRL_PAUSE = '/control/pause'
CTRL_SLOW = '/control/slow'
CTRL_REPLAN = '/control/replan'
CTRL_ABORT = '/control/abort'
CLOCK = '/clock/now'

_ALL_KEYS = (EVENT_ACTIVE, EVENT_LOG, EVENT_LAST, CTRL_PAUSE, CTRL_SLOW, CTRL_REPLAN, CTRL_ABORT, CLOCK)


@runtime_checkable
class Reactions(Protocol):
    """The physical reactions a capability lane wires in. The tree decides WHEN to react; the
    lane owns HOW. Every method is best-effort and must not raise on a no-op runtime.

    ``recommission(station)`` returns truthy on a fresh baseline. ``safe_park()`` brings the
    train to a braked, safe stop. The rest are commands with no return."""
    def slow(self) -> None: ...
    def stop(self) -> None: ...
    def pause_arm(self) -> None: ...
    def resume(self) -> None: ...
    def wait(self) -> None: ...
    def replan(self) -> None: ...
    def reobserve(self) -> None: ...
    def recommission(self, station: str) -> bool: ...
    def safe_park(self) -> None: ...


class RecordingReactions:
    """Default reactions: record every call (for tests and for an honest no-op runtime).

    A real lane supplies its own object implementing :class:`Reactions`; this one neither
    touches the adapter nor fabricates anything. ``recommission`` returns True by default so the
    recommission subtree can be exercised offline; pass ``recommission_ok=False`` to test refusal."""
    def __init__(self, *, recommission_ok=True):
        self.calls = []
        self._recommission_ok = recommission_ok

    def _rec(self, name, *args):
        self.calls.append((name, *args))

    def slow(self): self._rec('slow')
    def stop(self): self._rec('stop')
    def pause_arm(self): self._rec('pause_arm')
    def resume(self): self._rec('resume')
    def wait(self): self._rec('wait')
    def replan(self): self._rec('replan')
    def reobserve(self): self._rec('reobserve')
    def recommission(self, station): self._rec('recommission', station); return self._recommission_ok
    def safe_park(self): self._rec('safe_park')


class AdapterReactions:
    """Reactions backed by the running adapter/mobile base: a safe default for a GPU run when a
    lane has not wired a richer reaction. Everything maps to the adapter's EXISTING,
    payload-preserving primitives (``RuntimeAdapter.stop`` never commands a release); unknown
    hooks are no-ops. ``recommission`` is refused here (returns False) because re-commissioning is
    the stations lane's provider, not the adapter's -- a caller that needs it wires that lane in."""
    def __init__(self, adapter, *, recommission_provider=None):
        self.adapter = adapter
        self._recommission = recommission_provider

    def _hold(self):
        mobile = getattr(self.adapter, 'mobile', None)
        if mobile is not None and hasattr(mobile, 'hold'):
            mobile.hold()

    def slow(self): self._hold()
    def stop(self):
        self._hold()
        if hasattr(self.adapter, 'stop'):
            try:
                self.adapter.stop()   # stops arm + base; never releases a held payload
            except Exception:
                pass
    def pause_arm(self): self._hold()
    def resume(self): pass            # the stage re-begins its goal on resume (honest re-plan-from-here)
    def wait(self): self._hold()
    def replan(self): pass
    def reobserve(self): pass
    def recommission(self, station):
        return bool(self._recommission(station)) if self._recommission else False
    def safe_park(self): self._hold()


class MissionContext:
    """Everything a leaf needs, wired once. The driver sets :attr:`dt` before each tick."""
    def __init__(self, mission, bus, *, reactions=None, snapshot=None, name='restock_bt',
                 wait_ticks=8):
        self.mission = mission
        self.bus = bus
        self.reactions = reactions if reactions is not None else RecordingReactions()
        self.snapshot = snapshot                # optional view writer (set in the viz layer)
        self.wait_ticks = max(1, int(wait_ticks))
        self.dt = 0.0
        self.now = 0.0
        self.degraded_seen = False
        self.last_failure = None
        self._held = False
        self._begin_error = None
        self.bb = py_trees.blackboard.Client(name=name)
        for key in _ALL_KEYS:
            self.bb.register_key(key=key, access=py_trees.common.Access.WRITE)
        self.reset_blackboard()

    def reset_blackboard(self):
        self.bb.set(EVENT_ACTIVE, {})
        self.bb.set(EVENT_LOG, [])
        self.bb.set(EVENT_LAST, None)
        self.bb.set(CTRL_PAUSE, False)
        self.bb.set(CTRL_SLOW, False)
        self.bb.set(CTRL_REPLAN, False)
        self.bb.set(CTRL_ABORT, None)
        self.bb.set(CLOCK, 0.0)

    # --- event application (called by EventPump) ---------------------------------------
    def apply_event(self, event: Event):
        from .events import CLEARED_BY
        active = self.bb.get(EVENT_ACTIVE)
        if event.type in CLEARED_BY:
            active.pop(CLEARED_BY[event.type], None)
        elif event.type not in ('grasp_failed', 'seal_failed', 'tracking_miss', 'goal_miss',
                                 'coupling_failed'):
            # Persistent conditions stay active until cleared / handled. Momentary stage-failure
            # events are logged only (the Stage leaf + Retry own them).
            active[event.type] = event.data
        self.bb.set(EVENT_ACTIVE, active)
        self.bb.set(EVENT_LAST, {'t': round(self.now, 3), 'type': event.type, 'data': event.data})
        self.log(event.type, event.data, action='received')

    def clear_active(self, event_type):
        active = self.bb.get(EVENT_ACTIVE)
        if active.pop(event_type, None) is not None:
            self.bb.set(EVENT_ACTIVE, active)

    def active(self):
        return self.bb.get(EVENT_ACTIVE)

    def log(self, etype, data=None, action=''):
        entry = {'t': round(self.now, 3), 'type': etype, 'data': data or {}, 'action': action}
        log = self.bb.get(EVENT_LOG)
        log.append(entry)
        self.bb.set(EVENT_LOG, log)
        return entry

    def emit_stage_failure(self, event: Event):
        """A Stage leaf records what failed onto the timeline (the Retry decorator handles the
        retry/abort). Momentary, so it does not persist in /event/active."""
        self.bb.set(EVENT_LAST, {'t': round(self.now, 3), 'type': event.type, 'data': event.data})
        self.log(event.type, event.data, action='stage-failure')

    # --- control flags -----------------------------------------------------------------
    def recompute_control(self):
        """Derive the control flags from the currently-active persistent events, deterministically
        and in ONE place, so the reactive handler behaviours stay side-effecting (call reactions,
        log, show status) without racing each other over a shared pause bool.

        A human in the stop zone or any present dynamic obstacle HOLDS the active stage (pause);
        a human in the slow zone requests reduced speed. ``EventPump`` calls this each tick after
        draining the bus; the ``Stage`` leaf reads :attr:`paused` to hold (safety stop, payload
        preserved) instead of stepping the mission."""
        active = self.bb.get(EVENT_ACTIVE)
        human = active.get('human_detected')
        human_slow = bool(human) and human.get('zone') == 'slow'
        human_stop = bool(human) and not human_slow
        obstacle = 'dynamic_obstacle' in active
        self.request_pause(human_stop or obstacle)
        self.request_slow(human_slow)
        if obstacle:
            self.request_replan(True)

    @property
    def paused(self):
        return bool(self.bb.get(CTRL_PAUSE))

    @property
    def abort_reason(self):
        return self.bb.get(CTRL_ABORT)

    def request_pause(self, on=True):
        self.bb.set(CTRL_PAUSE, bool(on))

    def request_slow(self, on=True):
        self.bb.set(CTRL_SLOW, bool(on))

    def request_replan(self, on=True):
        self.bb.set(CTRL_REPLAN, bool(on))

    def request_abort(self, reason):
        self.bb.set(CTRL_ABORT, reason)

    # --- one-time safety hold (idempotent across held ticks) ---------------------------
    def safety_hold(self):
        if not self._held:
            self.reactions.stop()
            self._held = True

    def release_hold(self):
        if self._held:
            self.reactions.resume()
            self._held = False
