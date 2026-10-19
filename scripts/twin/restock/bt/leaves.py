"""py_trees leaf behaviours for the restock mission tree.

Every leaf drives the REAL :class:`mission.Mission` (through the ``begin_stage`` / ``step_stage`` /
``finalize`` seams) or reads/writes the shared blackboard; none reimplements stage logic. Leaves
are intentionally small and expose a ``feedback_message`` each tick, because the live visualisation
(``view.py``) snapshots exactly that string per node so a phone viewer can read "how is this event
being handled" straight off the tree.

Status contract used throughout:

* ``RUNNING`` -- stage still in progress, or held by a safety event (payload preserved).
* ``SUCCESS`` -- stage validated + recorded (or a degraded stage recorded unavailable, mission kept
  moving); a reactive handler always SUCCEEDs so the root sequence proceeds to the mission.
* ``FAILURE`` -- a stage exhausted its bounded retries (honest abort, ``mission.fail`` already
  called), or the health guard tripped an abort (stuck / low battery) -- either way the root
  sequence stops and the driver sees ``mission.terminal``.
"""
from __future__ import annotations

import py_trees
from py_trees.common import Status

from .events import (Event, RECOVERY_FOR, STAGE_FAILURE_TYPES, classify_stage_failure)

from .view_labels import collect_substep as _collect_substep

RUNNING, SUCCESS, FAILURE = Status.RUNNING, Status.SUCCESS, Status.FAILURE


class EventPump(py_trees.behaviour.Behaviour):
    """Drain the event bus into the blackboard once per tick, then recompute control flags.

    Always SUCCEEDs (it is pure bookkeeping): the reactive handlers and the mission run after it in
    the root sequence. The driver has already set ``ctx.now`` / ``ctx.dt`` / the clock key before
    this ticks, so the scenario source fires deterministically off the tick clock."""
    def __init__(self, ctx, name='event_pump'):
        super().__init__(name)
        self.ctx = ctx

    def update(self):
        events = self.ctx.bus.poll(self.ctx.now)
        for event in events:
            self.ctx.apply_event(event)
        self.ctx.recompute_control()
        active = self.ctx.active()
        self.feedback_message = (f'active={sorted(active)}' if active else 'clear')
        return SUCCESS


class HumanSafety(py_trees.behaviour.Behaviour):
    """React to a human in the slow/stop zone: slow, then stop + pause the arm; resume when clear.

    Edge-triggered so the physical reaction fires once on entering/leaving a zone (not every tick);
    the pause/slow flags themselves are owned by :meth:`MissionContext.recompute_control`. Always
    SUCCEEDs so the mission (which reads the pause flag and holds) still ticks."""
    def __init__(self, ctx, name='human_safety'):
        super().__init__(name)
        self.ctx = ctx
        self._engaged = None   # None | 'slow' | 'stop'

    def update(self):
        human = self.ctx.active().get('human_detected')
        zone = None if human is None else (human.get('zone') or 'stop')
        zone = 'slow' if zone == 'slow' else ('stop' if zone is not None else None)
        if zone != self._engaged:
            if zone == 'slow':
                self.ctx.reactions.slow()
                self.ctx.log('human_detected', human, action='slow')
            elif zone == 'stop':
                self.ctx.safety_hold()
                self.ctx.reactions.pause_arm()
                self.ctx.log('human_detected', human, action='stop+pause_arm')
            else:  # cleared
                self.ctx.release_hold()
                self.ctx.log('human_cleared', {}, action='resume')
            self._engaged = zone
        self.feedback_message = {None: 'clear', 'slow': 'human slow-zone: slowing',
                                 'stop': 'human stop-zone: HOLD, arm paused'}[zone]
        return SUCCESS


class ObstacleAvoid(py_trees.behaviour.Behaviour):
    """React to a dynamic obstacle / blocked aisle: hold and request a re-plan; resume when clear.

    Edge-triggered. The pause flag is set by ``recompute_control``; here we fire the wait/replan
    reaction and log the transition. Always SUCCEEDs."""
    def __init__(self, ctx, name='obstacle_avoid'):
        super().__init__(name)
        self.ctx = ctx
        self._engaged = False

    def update(self):
        obstacle = self.ctx.active().get('dynamic_obstacle')
        present = obstacle is not None
        if present and not self._engaged:
            self.ctx.safety_hold()
            self.ctx.reactions.wait()
            self.ctx.reactions.replan()
            self.ctx.log('dynamic_obstacle', obstacle, action='wait+replan')
            self._engaged = True
        elif not present and self._engaged:
            self.ctx.release_hold()
            self.ctx.request_replan(False)
            self.ctx.log('obstacle_cleared', {}, action='resume')
            self._engaged = False
        self.feedback_message = ('obstacle: HOLD, re-planning' if present else 'clear')
        return SUCCESS


class HealthGuard(py_trees.behaviour.Behaviour):
    """Abort to a safe park on ``stuck`` / ``low_battery``. FAILS so the root sequence stops.

    This is the only reactive leaf that can FAIL: a stuck base or a low battery is not something the
    mission can drive through, so we brake to a safe park (payload preserved), record an honest
    ``mission.fail`` and tear the tree down. Otherwise SUCCEEDs."""
    def __init__(self, ctx, name='health_guard'):
        super().__init__(name)
        self.ctx = ctx

    def update(self):
        active = self.ctx.active()
        trigger = next((t for t in ('stuck', 'low_battery') if t in active), None)
        if trigger is None:
            self.feedback_message = 'nominal'
            return SUCCESS
        reason = f'{trigger}: {active[trigger].get("reason") or active[trigger]}'
        if self.ctx.abort_reason is None:
            self.ctx.request_abort(reason)
            self.ctx.reactions.safe_park()
            self.ctx.log(trigger, active[trigger], action='safe_park+abort')
            if not self.ctx.mission.terminal:
                self.ctx.mission.fail(reason)
        self.feedback_message = f'ABORT to safe park: {reason}'
        return FAILURE


class BaselineGuard(py_trees.behaviour.Behaviour):
    """Re-commission an expired station baseline through the stations lane's provider.

    On a ``baseline_expired`` event, request a fresh baseline via ``reactions.recommission(station)``
    and HOLD (pause) until it returns truthy; clear the event on success. If the provider refuses
    (no provider wired, or it returns falsey) this stays held and the timeline shows the refusal --
    honest, never fabricated. SUCCEEDs (the hold is via the pause flag)."""
    def __init__(self, ctx, name='baseline_guard'):
        super().__init__(name)
        self.ctx = ctx
        self._pending = None

    def update(self):
        active = self.ctx.active()
        expired = active.get('baseline_expired')
        if expired is None:
            self.feedback_message = 'baselines valid'
            return SUCCESS
        station = expired.get('station') or 'unknown'
        self.ctx.request_pause(True)          # hold while re-commissioning
        ok = False
        try:
            ok = bool(self.ctx.reactions.recommission(station))
        except Exception as exc:              # a refusing provider must not crash the tree
            self.ctx.log('baseline_expired', {'station': station, 'error': str(exc)},
                         action='recommission_error')
            self.feedback_message = f'recommission {station}: error {exc}'
            return SUCCESS
        if ok:
            self.ctx.clear_active('baseline_expired')
            self.ctx.log('baseline_expired', {'station': station}, action='recommissioned')
            self.feedback_message = f'recommissioned {station}'
        else:
            self.ctx.log('baseline_expired', {'station': station}, action='recommission_refused')
            self.feedback_message = f'recommission {station}: REFUSED, holding'
        return SUCCESS


class Stage(py_trees.behaviour.Behaviour):
    """Drive ONE mission stage through the real ``begin_stage`` / ``step_stage`` seams, with bounded
    retry + a recovery reaction, and honest abort on exhaustion.

    * First tick begins the stage (records the action, starts the adapter).
    * While a safety event holds the mission (``ctx.paused``) the stage performs an idempotent
      safety hold and returns RUNNING -- it does NOT step the mission, so the sim clock and the
      stage make no progress during the hold; it resumes stepping when the hold clears.
    * ``step_stage`` outcomes map to ``RUNNING`` / ``SUCCESS`` (succeeded or degraded-unavailable,
      which keeps the mission moving but forces a failed finalize) / a failure path.
    * On failure: emit a classified stage-failure event onto the timeline, and if retries remain
      request the recovery reaction (re-observe / re-plan / wait) and re-begin the stage; otherwise
      ``mission.fail`` and return FAILURE.
    """
    def __init__(self, ctx, action, name=None, max_retries=2):
        super().__init__(name or f'stage:{action}')
        self.ctx = ctx
        self.action = action
        self.max_retries = max(0, int(max_retries))
        self._begun = False
        self._retries = 0

    def _begin(self):
        self.ctx.mission.begin_stage(self.action)
        self._begun = True

    def update(self):
        ctx = self.ctx
        if ctx.mission.terminal:
            self.feedback_message = f'{self.action}: mission terminal'
            return FAILURE if ctx.mission.state == 'failed' else SUCCESS
        if not self._begun:
            self._begin()
            self.feedback_message = f'{self.action}: begun'
            return RUNNING
        if ctx.paused:
            ctx.safety_hold()
            self.feedback_message = f'{self.action}: HOLD (event)'
            return RUNNING
        ctx.release_hold()
        outcome = ctx.mission.step_stage(ctx.dt)
        if outcome == 'running':
            # fm390: for collect, show the live position-first sub-step (nav -> perceive -> place ->
            # verify) so the view renders the step-by-step placement, not an opaque 'collect: running'.
            substep = _collect_substep(ctx.mission) if self.action == 'collect' else None
            self.feedback_message = (f'{self.action}: {substep}' if substep
                                     else f'{self.action}: running')
            return RUNNING
        if outcome in ('succeeded', 'degraded'):
            self.feedback_message = f'{self.action}: {outcome}'
            return SUCCESS
        # outcome == ('failed', reason)
        reason = outcome[1]
        etype = classify_stage_failure(self.action, reason)
        ctx.emit_stage_failure(Event(etype, {'reason': reason, 'stage': self.action}))
        if etype in STAGE_FAILURE_TYPES and self._retries < self.max_retries:
            self._retries += 1
            recovery = RECOVERY_FOR.get(etype)
            if recovery:
                getattr(ctx.reactions, recovery)()
            ctx.log(etype, {'reason': reason, 'retry': self._retries, 'recovery': recovery},
                    action=f'recover:{recovery}')
            self._begin()
            self.feedback_message = (f'{self.action}: recover {etype} '
                                     f'({self._retries}/{self.max_retries})')
            return RUNNING
        # honest abort: retries exhausted (or non-recoverable)
        ctx.log(etype, {'reason': reason}, action='abort')
        ctx.mission.fail(reason)
        self.feedback_message = f'{self.action}: FAILED {reason}'
        return FAILURE


class Finalize(py_trees.behaviour.Behaviour):
    """Run ``mission.finalize`` once after the last stage: stop the adapter and record the overall
    outcome (complete, or failed if any stage degraded). SUCCESS if complete, FAILURE if the
    finalize recorded a failed outcome."""
    def __init__(self, ctx, name='finalize'):
        super().__init__(name)
        self.ctx = ctx
        self._done = False

    def update(self):
        if not self._done:
            if not self.ctx.mission.terminal:
                self.ctx.mission.finalize()
            self._done = True
        state = self.ctx.mission.state
        self.feedback_message = f'mission {state}'
        return SUCCESS if state == 'complete' else FAILURE
