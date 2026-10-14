"""Bounded mission state machine independent of Isaac/ROS imports.

Runtime adapters own sensor freshness, collision checks and actuator control.
Each successful action must carry measured evidence. Tests opt into fixture
provenance; production rejects it. No default/mock adapter is installed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .inventory import Inventory, Job

ACTIONS = ('collect', 'couple', 'outbound', 'park', 'transfer', 'verify', 'return')


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
                 timeout_s=120., fixture_mode=False, max_evidence_age_s=1., stages=None):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('timeout must be finite and positive')
        if not math.isfinite(max_evidence_age_s) or max_evidence_age_s < 0:
            raise ValueError('evidence age must be finite and nonnegative')
        # A bounded stage subset must be an in-order prefix of ACTIONS so each stage's
        # physical prerequisites (coupled, parked, ...) still hold. Default: the full cycle.
        self.stages = tuple(stages) if stages else ACTIONS
        if self.stages != ACTIONS[:len(self.stages)]:
            raise ValueError(f'stages must be an in-order prefix of {ACTIONS}')
        self.inventory, self.job, self.adapter = inventory, job, adapter
        self.timeout_s, self.fixture_mode = timeout_s, fixture_mode
        self.max_evidence_age_s = max_evidence_age_s
        self.state, self.reason = 'new', ''
        self.time = self.elapsed = 0.
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

    def start(self):
        if self.state != 'new':
            raise ValueError('mission already started')
        self.inventory.reserve(self.job)
        try:
            self._begin(self.stages[0])
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
        if evidence.get('source') not in sources:
            raise ValueError('action requires sensor evidence')
        if not isinstance(evidence.get('observation_id'), str) or not evidence['observation_id']:
            raise ValueError('missing observation ID')
        stamp = evidence.get('simulation_time')
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or
                not math.isfinite(stamp) or not 0 <= self.time - stamp <= self.max_evidence_age_s):
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

    def tick(self, dt):
        if self.terminal:
            return self.state
        if self.state == 'new':
            raise ValueError('start mission first')
        if not math.isfinite(dt) or dt <= 0:
            self.fail('invalid simulation timestep')
            return self.state
        self.time += dt
        self.elapsed += dt
        try:
            if self.elapsed > self.timeout_s:
                raise TimeoutError(f'{self.state} timed out')
            result = self.adapter.step(dt)
            if result.status == 'running':
                return self.state
            if result.status == 'unavailable':
                # Honest degradation: the stage was attempted but its capability is not
                # provisioned. Record it FAILED (never success), keep the mission moving so
                # the full loop is recorded, and force an overall 'failed' outcome at the end.
                self.degraded.append(self.state)
                self.inventory.record(self.job.job_id, self.state + '_unavailable',
                                      {'reason': result.reason})
                self._advance()
                return self.state
            if result.status != 'succeeded':
                raise RuntimeError(result.reason or f'invalid/failed status: {result.status}')
            self._validate(result.evidence)
            if self.state == 'verify':
                self.inventory.verify(self.job, result.evidence)
            self.inventory.record(self.job.job_id, self.state + '_succeeded', result.evidence)
            self._advance()
        except Exception as exc:
            self.fail(str(exc))
        return self.state

    def _advance(self):
        index = self.stages.index(self.state) + 1
        if index == len(self.stages):
            self.adapter.stop()
            if self.degraded:
                self.state, self.reason = 'failed', 'stages unavailable: ' + ', '.join(self.degraded)
                self.inventory.record(self.job.job_id, 'failed', {'reason': self.reason})
            else:
                self.inventory.record(self.job.job_id, 'complete', {})
                self.state = 'complete'
        else:
            self._begin(ACTIONS[index])
