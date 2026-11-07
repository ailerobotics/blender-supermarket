"""Planner / render VRAM spike guard (sim-transfer-vram-preshed-v1).

coldwin93 died because the cuMotion planner's ten PLANNING_FAILED retries allocated a ~1.35 GB working
set into a card with only 117 MiB free -- a spike from 11791 MiB used to 13142 MiB (a native SIGSEGV).
The ``vram_headroom`` monitor samples AFTER the fact and cannot catch a sub-second allocation spike.

This guard is the BEFORE-the-fact backstop: a caller about to do a large GPU allocation (dispatch another
cuMotion plan retry, or boot a new render product) calls :meth:`SpikeGuard.wait_for_headroom` first. While
free VRAM is below ``floor_mib`` (default 400) the guard PAUSES -- polling with a bounded wait -- instead
of letting the allocator spike into a crash; it proceeds the moment free recovers above the floor, and
FAILS BY NAME if free does not recover within ``max_wait_s`` so the caller aborts the op cleanly instead
of OOMing. An unavailable sampler (no NVML) proceeds (the guard never wedges a run it cannot measure).

Opt-in (default OFF => a bare ``SpikeGuard()`` / absent config => disabled => callers skip it => byte-
identical). The decision logic is PURE: the VRAM sampler, the sleep and the clock are all injected, so the
cw93 spike trace is replayed in a unit test with no GPU.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, fields

FLOOR_MIB_DEFAULT = 400.0
POLL_PERIOD_S_DEFAULT = 0.5
MAX_WAIT_S_DEFAULT = 20.0


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or value <= 0:
        raise ValueError(f'spike_guard.{name} must be a positive number')
    return float(value)


@dataclass(frozen=True)
class SpikeGuard:
    """Resolved spike-guard levers. ``SpikeGuard()`` (default) is the DISABLED sentinel."""
    enabled: bool = False
    floor_mib: float = FLOOR_MIB_DEFAULT
    poll_period_s: float = POLL_PERIOD_S_DEFAULT
    max_wait_s: float = MAX_WAIT_S_DEFAULT

    @property
    def active(self):
        return bool(self.enabled)

    def describe(self):
        return ('VRAM_SPIKE_GUARD enabled=%s floor_mib=%g poll_period_s=%g max_wait_s=%g'
                % (self.enabled, self.floor_mib, self.poll_period_s, self.max_wait_s))

    def summary(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def wait_for_headroom(self, sampler, *, what='allocation', sleep=time.sleep,
                          clock=time.monotonic, log=print):
        """Pause until free VRAM clears ``floor_mib``, then return a :class:`SpikeGuardDecision`.

        ``sampler`` -> ``(used_mib, free_mib)`` or None. If disabled, or the sampler is unavailable / the
        first reading already clears the floor, proceeds immediately (``decision.proceed`` True, no sleep).
        Otherwise it logs ``VRAM_SPIKE_GUARD_PAUSE`` and polls every ``poll_period_s`` until free recovers
        above the floor (proceed) or ``max_wait_s`` elapses with free still below it, in which case
        ``decision.proceed`` is False and ``decision.reason`` names the failure (the caller aborts by name).
        """
        if not self.active or sampler is None:
            return SpikeGuardDecision(proceed=True, free_mib=None, waited_s=0.0, reason='inactive')
        free = _read_free(sampler, log)
        if free is None:
            # cannot measure -> never wedge the run; proceed (the pre-launch budget is the other gate).
            return SpikeGuardDecision(proceed=True, free_mib=None, waited_s=0.0, reason='unmeasured')
        if free >= self.floor_mib:
            return SpikeGuardDecision(proceed=True, free_mib=free, waited_s=0.0, reason='clear')
        start = clock()
        waited = 0.0
        log('VRAM_SPIKE_GUARD_PAUSE what=%s free=%.0f floor=%.0f max_wait=%.0f -- pausing before the '
            'allocation to avoid an OOM spike' % (what, free, self.floor_mib, self.max_wait_s))
        while waited < self.max_wait_s:
            sleep(self.poll_period_s)
            waited = float(clock() - start)
            free = _read_free(sampler, log)
            if free is None:
                return SpikeGuardDecision(proceed=True, free_mib=None, waited_s=waited,
                                          reason='unmeasured')
            if free >= self.floor_mib:
                log('VRAM_SPIKE_GUARD_RESUME what=%s free=%.0f floor=%.0f waited=%.1fs'
                    % (what, free, self.floor_mib, waited))
                return SpikeGuardDecision(proceed=True, free_mib=free, waited_s=waited,
                                          reason='recovered')
        log('VRAM_SPIKE_GUARD_TIMEOUT what=%s free=%.0f floor=%.0f waited=%.1fs -- free did not recover; '
            'aborting the allocation by name instead of spiking the allocator'
            % (what, free, self.floor_mib, waited))
        return SpikeGuardDecision(proceed=False, free_mib=free, waited_s=waited,
                                  reason='vram spike guard: free VRAM %.0f MiB stayed below the %.0f MiB '
                                         'floor for %.0fs (%s)' % (free, self.floor_mib, waited, what))


@dataclass
class SpikeGuardDecision:
    proceed: bool
    free_mib: float | None
    waited_s: float
    reason: str

    def __bool__(self):
        return self.proceed


def _read_free(sampler, log):
    try:
        reading = sampler()
    except Exception as exc:  # noqa: BLE001 -- a sampler fault must not crash the guard
        log('VRAM_SPIKE_GUARD_SAMPLE_ERROR %r' % (exc,))
        return None
    if reading is None:
        return None
    try:
        return float(reading[1])
    except (TypeError, IndexError, ValueError):
        return None


def from_config(config):
    """Resolve ``config['transfer_spike_guard']`` (DEFAULT OFF; absent/``false`` => disabled). ``true`` =>
    enabled defaults. Fails CLOSED on a typo / bad value."""
    block = config.get('transfer_spike_guard') if isinstance(config, dict) else None
    return _resolve(block)


def from_env(environ=None):
    """Resolve the spike guard from environment (the ros_adapter host process has no mission config in
    hand): ``RESTOCK_TRANSFER_SPIKE_GUARD=1`` enables it; ``RESTOCK_SPIKE_GUARD_FLOOR_MIB`` /
    ``_POLL_S`` / ``_MAX_WAIT_S`` tune the levers. Unset => disabled (byte-identical)."""
    env = os.environ if environ is None else environ
    if env.get('RESTOCK_TRANSFER_SPIKE_GUARD', '') not in ('1', 'true', 'True'):
        return SpikeGuard()
    return SpikeGuard(
        enabled=True,
        floor_mib=float(env.get('RESTOCK_SPIKE_GUARD_FLOOR_MIB', FLOOR_MIB_DEFAULT)),
        poll_period_s=float(env.get('RESTOCK_SPIKE_GUARD_POLL_S', POLL_PERIOD_S_DEFAULT)),
        max_wait_s=float(env.get('RESTOCK_SPIKE_GUARD_MAX_WAIT_S', MAX_WAIT_S_DEFAULT)))


def _resolve(block):
    if block is None or block is False:
        return SpikeGuard()
    if block is True:
        block = {'enabled': True}
    if not isinstance(block, dict):
        raise ValueError('transfer_spike_guard must be an object such as '
                         '{"enabled": true, "floor_mib": 400}')
    known = {f.name for f in fields(SpikeGuard)}
    unknown = set(block) - known
    if unknown:
        raise ValueError(f'transfer_spike_guard has unknown keys {sorted(unknown)}; '
                         f'known: {sorted(known)}')
    enabled = bool(block.get('enabled', True))
    floor = _positive('floor_mib', block.get('floor_mib', FLOOR_MIB_DEFAULT))
    poll = _positive('poll_period_s', block.get('poll_period_s', POLL_PERIOD_S_DEFAULT))
    max_wait = _positive('max_wait_s', block.get('max_wait_s', MAX_WAIT_S_DEFAULT))
    return SpikeGuard(enabled=enabled, floor_mib=floor, poll_period_s=poll, max_wait_s=max_wait)


def summary_fields(config):
    cfg = from_config(config)
    if not cfg.active:
        return {}
    return {'transfer_spike_guard_mode': 'on', 'transfer_spike_guard': cfg.summary()}
