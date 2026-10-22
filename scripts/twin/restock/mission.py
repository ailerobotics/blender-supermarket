"""Bounded mission state machine independent of Isaac/ROS imports.

Runtime adapters own sensor freshness, collision checks and actuator control.
Each successful action must carry measured evidence. Tests opt into fixture
provenance; production rejects it. No default/mock adapter is installed.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Protocol

from ..restock_manipulation import ground_truth_collect as _gtc
from .inventory import Inventory, Job

ACTIONS = ('collect', 'couple', 'outbound', 'park', 'transfer', 'verify', 'return')

# fm363 universal step-level bounded retry (OUTER net; inner per-move retries stay the first line).
# SIM-ONLY: RESTOCK_STEP_RETRY_MAX default 0 == byte-identical real-rig behaviour (no outer retry); the
# sim orchestrator sets 2. Scoped to the re-attemptable MANEUVER stages only: re-beginning one re-drives
# a nav/dock/home maneuver from the LIVE pose, which is idempotent-ish. collect/transfer make irreversible
# payload acts (grasp/place/release) and verify commits stock, so re-beginning them would DOUBLE-ACT --
# those are excluded here and rely on their inner per-move retries instead (fm363's explicit scoping).
_RETRYABLE_STAGES = ('couple', 'outbound', 'park', 'return')
# collect-gt: stages whose evidence may be explicitly-labelled sim GROUND TRUTH (source 'ground_truth' AND
# ground_truth=True: the fm389 place check converted by the runtime). Never 'verify' -- the stock-commit
# gate stays sensor-only. Unlabelled or other stages, or RESTOCK_COLLECT_GROUND_TRUTH unset, keep the
# sensor-only rule.
_GT_EVIDENCE_STAGES = ('collect',)


def bounded_stages(stages=None, stop_after=None, start_at=None):
    """The in-order stage prefix a run drives: ``stages`` (default ACTIONS) cut right after
    ``stop_after`` (dev targeted test: e.g. collect-only, or start_at=couple stop_after=park).
    ``stop_after=None`` returns ``stages`` unchanged. Refuses a stop stage that is not in the run or
    that precedes ``start_at`` (the run would drive nothing)."""
    stages = tuple(stages) if stages else ACTIONS
    if stop_after is None:
        return stages
    if stop_after not in stages:
        raise ValueError(f'stop_after {stop_after!r} not in mission stages {stages}')
    if start_at is not None and start_at in stages and stages.index(stop_after) < stages.index(start_at):
        raise ValueError(f'stop_after {stop_after!r} precedes start_at {start_at!r}')
    return stages[:stages.index(stop_after) + 1]


@dataclass(frozen=True)
class Result:
    status: str
    evidence: dict = field(default_factory=dict)
    reason: str = ''


class Adapter(Protocol):
    def start(self, action: str, job: Job) -> None: ...
    def step(self, dt: float) -> Result: ...
    def stop(self) -> None: ...


class Mission:
    def __init__(self, inventory: Inventory, job: Job, adapter: Adapter,
                 timeout_s=120., fixture_mode=False, max_evidence_age_s=1., stages=None,
                 start_at=None, fit_preflight=None):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('timeout must be finite and positive')
        if not math.isfinite(max_evidence_age_s) or max_evidence_age_s < 0:
            raise ValueError('evidence age must be finite and nonnegative')
        # A bounded stage subset must be an in-order prefix of ACTIONS so each stage's
        # physical prerequisites (coupled, parked, ...) still hold. Default: the full cycle.
        self.stages = tuple(stages) if stages else ACTIONS
        if self.stages != ACTIONS[:len(self.stages)]:
            raise ValueError(f'stages must be an in-order prefix of {ACTIONS}')
        # Start-phase shortcut (dev iteration ONLY): begin the real state machine at `start_at`
        # instead of stages[0]. The stages BEFORE it are NOT performed -- the adapter/runtime
        # synthesizes their physical end state (robot pose, braked base, carton attach, coupling)
        # out of band, and the mission records each skipped stage HONESTLY as a `<stage>_shortcut`
        # event (never as a performed success). Absent (None) => no shortcut, byte-identical
        # production behaviour (start_index 0). Stock is still only committed at the real `verify`,
        # so a shortcut that begins at or before verify credits inventory correctly; a shortcut that
        # begins after verify is refused (it would skip the only stock-committing gate).
        self.start_index = 0
        self.shortcut = None
        if start_at is not None:
            if start_at not in self.stages:
                raise ValueError(f'start_at {start_at!r} not in mission stages {self.stages}')
            self.start_index = self.stages.index(start_at)
            if 'verify' in self.stages and self.start_index > self.stages.index('verify'):
                raise ValueError('start_at cannot skip the verify stock-commit gate')
            self.shortcut = {'start_at': start_at,
                             'skipped': list(self.stages[:self.start_index])}
        # Opt-in measured tray/slot fit preflight (default None => disabled, byte-identical).
        # A FitVerdict (twin.restock.fit_preflight) whose status is 'wont_fit' refuses the job in
        # start() BEFORE any reservation or robot motion with an honest 'unsupported' message
        # routed to the caller, instead of failing opaquely at the place phase. 'ok'/'unknown'
        # never block here: the place-phase gate stays the measured backstop and verify stays the
        # stock-commit source of truth.
        self.fit_preflight = fit_preflight
        self.inventory, self.job, self.adapter = inventory, job, adapter
        self.timeout_s, self.fixture_mode = timeout_s, fixture_mode
        self.max_evidence_age_s = max_evidence_age_s
        self.state, self.reason = 'new', ''
        self.time = self.elapsed = 0.
        # fm363: per-stage outer-retry budget, read at construction (after the orchestrator sets env).
        # 0 => byte-identical (no outer retry). Only _RETRYABLE_STAGES consult it.
        self.step_retry_max = max(0, int(os.environ.get('RESTOCK_STEP_RETRY_MAX', 0)))
        self._step_retries = {}
        # Stages attempted but reported 'unavailable' (a provisioned-capability gap, e.g.
        # in-process cuRobo). They are recorded FAILED and the mission keeps running so the
        # full choreography is exercised, but the overall outcome is forced to 'failed'.
        self.degraded = []

    @property
    def terminal(self):
        return self.state in ('complete', 'failed')

    def _begin(self, action):
        self.state, self.elapsed = action, 0.
        self.inventory.record(self.job.job_id, action, {})
        self.adapter.start(action, self.job)

    def begin_stage(self, action):
        """Public seam: begin a named stage (records the action, starts the adapter).

        The opt-in behaviour-tree layer drives stages itself through this; Mission.start()
        and _advance() use _begin internally. Same single implementation either way."""
        self._begin(action)

    def start(self, begin=True):
        if self.state != 'new':
            raise ValueError('mission already started')
        # Refuse a proven won't-fit BEFORE reserving stock or sending the robot. The job was
        # never reserved, so the inventory tables stay untouched; the honest reason is surfaced
        # on self.reason for the launcher to log and persist. 'ok'/'unknown' fall through.
        if self.fit_preflight is not None and self.fit_preflight.status == 'wont_fit':
            self.state = 'failed'
            self.reason = 'unsupported: ' + self.fit_preflight.message
            return
        self.inventory.reserve(self.job)
        # Record any shortcut-skipped prefix HONESTLY -- as `<stage>_shortcut` events marked
        # shortcut=True, never as performed successes -- so the durable record and mission.json
        # cannot be mistaken for real actuation. The physical end state is synthesized by the
        # runtime before the first real tick.
        for action in self.stages[:self.start_index]:
            self.inventory.record(self.job.job_id, action + '_shortcut',
                                  {'shortcut': True, 'source': 'start_phase_shortcut',
                                   'reason': 'physical and job state synthesized, not performed'})
        # `begin=False` reserves the job + records the shortcut prefix but does NOT begin the
        # first stage, so the behaviour-tree layer can begin stages itself (via begin_stage);
        # the default (begin=True) is byte-identical to the production state machine.
        if not begin:
            return
        try:
            self._begin(self.stages[self.start_index])
        except Exception as exc:
            self.fail(f'adapter start: {exc}')

    def fail(self, reason):
        self.state, self.reason = 'failed', reason
        try:
            self.adapter.stop()
        except Exception as exc:
            self.reason += f'; STOP FAILED: {exc}'
        self.inventory.record(self.job.job_id, 'failed', {'reason': self.reason})

    def _validate(self, evidence):
        sources = {'sensor'} | ({'test_fixture'} if self.fixture_mode else set())
        if (self.state in _GT_EVIDENCE_STAGES and _gtc.enabled() and evidence.get('source') == 'ground_truth'
                and evidence.get('ground_truth') is True):
            sources = sources | {'ground_truth'}    # collect-gt: labelled sim ground-truth (never verify)
        if evidence.get('source') not in sources:
            raise ValueError('action requires sensor evidence')
        if not isinstance(evidence.get('observation_id'), str) or not evidence['observation_id']:
            raise ValueError('missing observation ID')
        stamp = evidence.get('simulation_time')
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or
                not math.isfinite(stamp) or not -1e-6 <= self.time - stamp <= self.max_evidence_age_s):
            raise ValueError('missing, future or stale evidence timestamp')
        required = {'collect': ('loaded', True), 'couple': ('coupled', True),
                    'outbound': ('arrived', True), 'park': ('stationary', True),
                    'transfer': ('released', True), 'return': ('home', True)}
        if self.state in required:
            key, value = required[self.state]
            if evidence.get(key) is not value:
                raise ValueError(f'missing measured {key}')
        if self.state == 'verify':
            if (evidence.get('sku') != self.job.sku or
                    type(evidence.get('quantity')) is not int or
                    evidence['quantity'] != self.job.quantity or
                    evidence.get('location') != self.job.shelf_id or
                    evidence.get('transfer_confirmed') is not True):
                raise ValueError('observed transfer does not match job')

    def step_stage(self, dt):
        """Advance the CURRENT stage by one tick WITHOUT transitioning to the next stage and
        WITHOUT failing the mission. Returns a token the caller reconciles into mission state:

          'running'          -- stage still in progress
          'succeeded'        -- stage's measured evidence validated + recorded (verify also
                                committed stock); caller should advance
          'degraded'         -- stage reported capability 'unavailable' (recorded
                                <stage>_unavailable, degraded tracked); caller should advance
                                but the overall outcome must end 'failed'
          ('failed', reason) -- stage/adapter raised or returned a failed status (nothing was
                                recorded as a success); caller should fail the mission

        Mission.tick() wraps this with _advance()/fail() for the production state machine; the
        opt-in behaviour-tree layer wraps it with bounded retry + recovery subtrees instead.
        Keeping evidence validation + recording here means it lives in exactly one place."""
        if not math.isfinite(dt) or dt <= 0:
            return ('failed', 'invalid simulation timestep')
        self.time += dt
        self.elapsed += dt
        try:
            if self.elapsed > self.timeout_s:
                raise TimeoutError(f'{self.state} timed out')
            result = self.adapter.step(dt)
            if result.status == 'running':
                return 'running'
            if result.status == 'unavailable':
                # Honest degradation: the stage was attempted but its capability is not
                # provisioned. Record it FAILED (never success), keep the mission moving so
                # the full loop is recorded, and force an overall 'failed' outcome at the end.
                self.degraded.append(self.state)
                self.inventory.record(self.job.job_id, self.state + '_unavailable',
                                      {'reason': result.reason})
                return 'degraded'
            if result.status != 'succeeded':
                raise RuntimeError(result.reason or f'invalid/failed status: {result.status}')
            self._validate(result.evidence)
            if self.state == 'verify':
                self.inventory.verify(self.job, result.evidence)
            self.inventory.record(self.job.job_id, self.state + '_succeeded', result.evidence)
            return 'succeeded'
        except Exception as exc:
            return ('failed', str(exc))

    def tick(self, dt):
        if self.terminal:
            return self.state
        if self.state == 'new':
            raise ValueError('start mission first')
        outcome = self.step_stage(dt)
        if outcome == 'running':
            return self.state
        if isinstance(outcome, tuple):     # ('failed', reason)
            reason = outcome[1]
            # fm363 universal OUTER retry: a maneuver stage that fails OR raised (step_stage turns both
            # into ('failed', reason)) is re-attempted with a FRESH budget (re-_begin restarts the
            # adapter + resets elapsed) up to step_retry_max, so one unlucky seed streak / timing trip /
            # transient cannot end the mission. One log line per retry. Off (0) or a non-maneuver stage
            # => the original immediate fail (byte-identical). Inner per-move retries still run first.
            if (self.step_retry_max > 0 and self.state in _RETRYABLE_STAGES
                    and self._step_retries.get(self.state, 0) < self.step_retry_max):
                self._step_retries[self.state] = self._step_retries.get(self.state, 0) + 1
                print('STEP_RETRY stage=%s attempt=%d/%d cause=%s'
                      % (self.state, self._step_retries[self.state], self.step_retry_max, reason),
                      flush=True)
                self._begin(self.state)     # re-drive the maneuver from the live pose, fresh budget
                return self.state
            self.fail(reason)
            return self.state
        self._advance()                     # 'succeeded' or 'degraded'
        return self.state

    def _advance(self):
        index = self.stages.index(self.state) + 1
        if index == len(self.stages):
            self.finalize()
        else:
            self._begin(ACTIONS[index])

    def finalize(self):
        """Terminal bookkeeping after the last stage: stop the adapter and record the overall
        outcome -- 'failed' if any stage degraded (unavailable), else 'complete'. Reused by the
        behaviour-tree layer's finalize leaf so the end-of-mission record is written once."""
        self.adapter.stop()
        if self.degraded:
            self.state, self.reason = 'failed', 'stages unavailable: ' + ', '.join(self.degraded)
            self.inventory.record(self.job.job_id, 'failed', {'reason': self.reason})
        else:
            self.inventory.record(self.job.job_id, 'complete', {})
            self.state = 'complete'
