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
from .stage_phases import milestone as _milestone

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


class StageRun:
    """Shared per-stage driver: begins ONE mission stage and steps it through the real
    ``begin_stage`` / ``step_stage`` seams **at most once per tree tick**, exposing which position-
    first step it is on (``milestone``) and its outcome. The phase-gate leaves share one of these
    (so they are pure views, never a second stepping path) and the :class:`StageRetry` decorator
    reads ``fail_reason`` to retry or abort. Keeping stepping here -- in one place, one call per tick,
    identical in count and order to ``Mission.tick`` -- is what makes the decomposition byte-identical
    to the state machine (parity): the tree never issues a nav/manip command, it only steps + reads.

    * First tick of a stage begins it (records the action, starts the adapter) and does NOT step --
      the state machine's begin is likewise free (folded into ``start`` / the advancing tick).
    * While a safety event holds the mission (``ctx.paused``) the run performs an idempotent safety
      hold and does not step, so the sim clock and the stage make no progress during the hold.
    * ``step_stage`` -> ``running`` (advance ``milestone`` off the adapter's observed phase),
      ``succeeded`` / ``degraded`` (``done``), or ``('failed', reason)`` (``fail_reason``).
    """
    def __init__(self, ctx, action, n_steps, steps=None):
        self.ctx = ctx
        self.action = action
        self.n_steps = max(1, int(n_steps))
        # vr1: the (label, kind) plan, so the timeline names each position-first step the walkthrough
        # cuts on; defaults to one leaf named after the stage (a non-phased fixture).
        self.steps = list(steps) if steps else [(action, 'stage')]
        self.begun = False
        self.done = None             # None | 'succeeded' | 'degraded'
        self.fail_reason = None      # set the tick step_stage reports ('failed', reason)
        self.milestone = 0
        self.attempt = 0
        self._held = False
        self._tick = None            # ctx.now of the last pump (one step per tree tick)

    def _sim_time(self):
        # The mission's dt-accumulated clock == the packet/frame simulation_time the recorder stamps on
        # walkthrough frames (the tree tick clock runs ahead of it by the free begin ticks).
        return round(float(getattr(self.ctx.mission, 'time', 0.0)), 4)

    def _log_stage(self, etype, **data):
        """vr1: durable stage timeline entry (bt_events.jsonl + mission.json stage_timeline) so the
        walkthrough can cut + caption every stage and step from the run itself."""
        self.ctx.log(etype, dict({'stage': self.action, 'attempt': self.attempt,
                                  'sim_time': self._sim_time()}, **data), action='stage')

    def _log_step(self, index, *, skipped=False):
        label, kind = self.steps[min(index, len(self.steps) - 1)]
        extra = {'skipped': True} if skipped else {}
        self._log_stage('stage_step', step=index, label=label, kind=kind, **extra)

    def begin(self):
        self.ctx.mission.begin_stage(self.action)
        self.begun = True
        self.milestone = 0
        self.fail_reason = None
        self.done = None
        self.attempt += 1
        self._log_stage('stage_begin', steps=[label for label, _ in self.steps])
        self._log_step(0)

    def _classify(self):
        idx = _milestone(self.action, getattr(self.ctx.mission, 'adapter', None))
        # monotonic within an attempt, clamped to the plan length (a non-phased adapter -> step 0).
        return max(self.milestone, min(int(idx), self.n_steps - 1))

    def pump(self):
        """Advance the stage by at most one ``step_stage`` per tree tick (idempotent per ``ctx.now``),
        so a success-cascade across several gates in one tick never double-steps the mission."""
        ctx = self.ctx
        if self._tick == ctx.now:
            return
        self._tick = ctx.now
        if ctx.mission.terminal or self.done is not None or self.fail_reason is not None:
            return
        if not self.begun:
            self.begin()
            return                            # begin tick: no step (matches the free begin)
        if ctx.paused:
            ctx.safety_hold()
            self._held = True
            return
        if self._held:
            ctx.release_hold()
            self._held = False
        outcome = ctx.mission.step_stage(ctx.dt)
        if outcome == 'running':
            before = self.milestone
            self.milestone = self._classify()
            for index in range(before + 1, self.milestone + 1):   # vr1: one entry per step reached
                self._log_step(index, skipped=index < self.milestone)
        elif outcome in ('succeeded', 'degraded'):
            self.done = outcome
            self._log_stage('stage_' + outcome)
        else:
            self.fail_reason = outcome[1]      # ('failed', reason)
            self._log_stage('stage_failed', reason=str(self.fail_reason)[:300])

    def reset_for_retry(self):
        self.begin()


class PhaseGate(py_trees.behaviour.Behaviour):
    """One explicit position-first step (navigate / perceive / manipulate / couple / uncouple /
    verify) of a stage. It drives the shared :class:`StageRun` (``pump`` steps the mission once per
    tick) and reports: RUNNING while the stage is on this step (or held by a safety event), SUCCESS
    once the stage has advanced past this step (or the whole stage finished), FAILURE when the step
    failed -- naming the exact failed field in its feedback (the fm412 diagnostic). Preconditions are
    structural: a gate only ticks once every earlier gate has SUCCEEDED (the memory sequence order)."""
    def __init__(self, run, index, label, kind):
        super().__init__(name=f'{run.action}:{label}')
        self.run = run
        self.index = index
        self.kind = kind

    def update(self):
        run = self.run
        run.pump()
        if run.fail_reason is not None:
            self.feedback_message = f'{self.kind}: FAILED {run.fail_reason}'
            return FAILURE
        if run.done is not None:
            self.feedback_message = f'{self.kind}: {run.done}'
            return SUCCESS
        if run.ctx.mission.terminal:
            # mission went terminal out-of-band (an upstream abort); do not claim this step passed.
            self.feedback_message = f'{self.kind}: mission {run.ctx.mission.state}'
            return SUCCESS if run.ctx.mission.state == 'complete' else FAILURE
        if run.ctx.paused:
            self.feedback_message = f'{self.kind}: HOLD (event)'
            return RUNNING
        if run.milestone > self.index:
            self.feedback_message = f'{self.kind}: done'
            return SUCCESS
        self.feedback_message = f'{self.kind}: running'
        return RUNNING


class StageRetry(py_trees.decorators.Decorator):
    """Wrap a stage's phase sequence with bounded retry + an explicit recovery reaction, aborting
    honestly on exhaustion -- so retry/recovery is one named node in the tree (not hidden in a leaf)
    and a new step is just a new leaf in the wrapped sequence.

    On a step FAILURE: emit a classified stage-failure event onto the timeline (naming WHAT failed);
    if the failure type is recoverable and retries remain, request the recovery reaction
    (re-observe / re-plan / wait), re-begin the stage and re-enter the phase sequence from step 0
    (RUNNING); otherwise record an honest ``mission.fail`` and FAIL."""
    def __init__(self, child, run, *, max_retries=2, name=None):
        super().__init__(name=name or f'stage:{run.action}', child=child)
        self.run = run
        self.max_retries = max(0, int(max_retries))
        self.retries = 0

    def update(self):
        status = self.decorated.status
        run = self.run
        if status != FAILURE or run.fail_reason is None:
            return status
        reason = run.fail_reason
        etype = classify_stage_failure(run.action, reason)
        run.ctx.emit_stage_failure(Event(etype, {'reason': reason, 'stage': run.action}))
        if etype in STAGE_FAILURE_TYPES and self.retries < self.max_retries:
            self.retries += 1
            recovery = RECOVERY_FOR.get(etype)
            if recovery:
                getattr(run.ctx.reactions, recovery)()
            run.ctx.log(etype, {'reason': reason, 'retry': self.retries, 'recovery': recovery},
                        action=f'recover:{recovery}')
            run.reset_for_retry()
            self.decorated.stop(Status.INVALID)      # re-enter the phase sequence from step 0
            self.feedback_message = f'recover {etype} ({self.retries}/{self.max_retries})'
            return RUNNING
        run.ctx.log(etype, {'reason': reason}, action='abort')
        if not run.ctx.mission.terminal:
            run.ctx.mission.fail(reason)
        self.feedback_message = f'FAILED {reason}'
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
            mission = self.ctx.mission
            self.ctx.log('mission_' + mission.state,     # vr1: the walkthrough's closing card
                         {'state': mission.state, 'reason': mission.reason,
                          'sim_time': round(float(getattr(mission, 'time', 0.0)), 4)}, action='stage')
        state = self.ctx.mission.state
        self.feedback_message = f'mission {state}'
        return SUCCESS if state == 'complete' else FAILURE
