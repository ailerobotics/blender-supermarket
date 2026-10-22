"""Staged-start (``start_at`` other than collect) settle hold + host clock sync.

A start-phase shortcut (``shortcut.apply_start_state``) settles the synthesized carton/hitch with
``world.step`` calls BEFORE the first mission tick. Those steps advance Isaac's world clock -- the
``/clock`` the host stack, the station acks and the nvblox ``collision_time`` all ride -- while the
mission packet clock (``Packet.simulation_time`` = tick*dt) starts at 0 on the first tick. The two
clocks therefore differ by a CONSTANT offset for the whole run: 0.05 s on a cold run, 0.55 s on
``start_at=couple`` (30 settle steps) and 1.55 s on ``start_at=transfer`` (90 settle steps). The host
compares the two domains in exactly two places, and both bench1 failures are that offset, not a
boot transient:

* ``ros_adapter`` packet freshness ``/clock now - packet stamp`` = offset (bench1 attempt 1:
  ``now-stamp=1.550`` > max_age 1.5);
* the controller's ``collision_dt = packet stamp - collision_time`` = true age - offset (bench1
  attempt 2: 0.217 - 1.550 = ``-1.3333`` < -render_lead).

Two switches, both engaged ONLY on a staged start (``config['start_at']`` set and not ``collect``),
so a cold run is byte-identical whatever the runtime config carries:

* ``staged_start_clock_sync`` (default True): the Isaac runtime stamps every host sensor packet with
  the exact offset (``Sensors.clock_offset`` = /clock - packet clock, read on the same tick) and moves
  the packet's ``collision_time`` into the packet clock, so both host checks measure TRUE age, as they
  do on a cold run. ``false`` restores the raw packet.
* ``staged_start_settle_s`` (default 20 s; 0 disables): the warm-up of the mission's FIRST
  manipulation stage (transfer on the bench) holds the dispatch until the mission has run that much
  sim time AND the station ack + collision-world stamps (``StationClient.ready``: both fresh within
  0.5 s of /clock) have stayed fresh with the base parked for the last ``_STEADY_S``. Bounded: past
  ``settle_s + _MAX_EXTRA_S`` of warm-up it releases anyway (logged) and the host guards decide, the
  same contract as the existing warm-up tick cap.

The ARM path is handled explicitly OUT OF BAND of these two packet switches. The host trajectory
executor (``trajectory_host``) does not ride the mission packet clock at all -- it subscribes to /clock
(use_sim_time) and /joint_states directly on domain 88 -- so ``staged_start_clock_sync`` does not touch
it. Instead it re-stamps its trajectory time origin to the first post-install /joint_states stamp
(``trajectory_host.anchored_start``), which makes the tracking reference ride the MEASURED-FEEDBACK clock
regardless of any /clock-vs-packet or /clock-vs-feedback offset (transfer1, 2026-10-22: anchoring to
now() at execute charged an ~11 s accept->first-feedback sim burst as arm progress and faulted the
still-parked arm on the first sample). CPU-proven in tests/test_restock_trajectory_host.py
(two-clock-offset + transfer1 replay).

Pure Python, CPU-tested (tests/test_restock_staged_start.py).
"""
from __future__ import annotations

import math
from dataclasses import replace

STAGED_SETTLE_DEFAULT_S = 20.0
# Continuous window the station ack + collision-world stamps must stay fresh (parked base) before the
# hold releases. The 'simulation ran settle_s' half of the condition is on the mission clock, so on a
# start_at=couple run (transfer ~200 s in) the hold costs only this window, not another settle_s.
_STEADY_S = 2.0
# Bound past settle_s: a stream that never steadies releases the hold (logged) instead of holding forever.
_MAX_EXTRA_S = 60.0
_SETTLE_MAX_S = 600.0
MANIPULATION_ACTIONS = ('collect', 'transfer')


def is_staged_start(config) -> bool:
    """True for a start-phase shortcut run that begins after collect (start_at couple/transfer/...)."""
    start_at = config.get('start_at') if isinstance(config, dict) else None
    return bool(start_at) and start_at != 'collect'


def staged_settle_seconds(config) -> float:
    """``config['staged_start_settle_s']`` on a staged start (default 20 s), else 0.0 (hold off)."""
    if not is_staged_start(config):
        return 0.0
    value = config.get('staged_start_settle_s', STAGED_SETTLE_DEFAULT_S)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not 0.0 <= value <= _SETTLE_MAX_S):
        raise ValueError(f'staged_start_settle_s must be a number of seconds in [0, {_SETTLE_MAX_S:g}]')
    return float(value)


def staged_clock_sync(config) -> bool:
    """``config['staged_start_clock_sync']`` on a staged start (default True), else False."""
    if not is_staged_start(config):
        return False
    value = config.get('staged_start_clock_sync', True)
    if not isinstance(value, bool):
        raise ValueError('staged_start_clock_sync must be true or false')
    return value


def sync_packet_clock(packet, offset):
    """The host sensor packet with the exact /clock - packet-clock ``offset`` recorded and its
    ``collision_time`` (a /clock stamp from the station ack) moved into the packet clock, so the host's
    packet-freshness and collision-freshness checks both measure true age. The -1 'no collision world
    yet' sentinel (any negative read) passes through untouched. Pure."""
    offset = float(offset)
    if not math.isfinite(offset):
        raise ValueError('clock offset must be finite')
    collision = packet.collision_time
    if math.isfinite(collision) and collision >= 0:
        collision = collision - offset
    return replace(packet, clock_offset=offset, collision_time=collision)


class SettleHold:
    """Hold the first manipulation dispatch of a staged start until the simulation has settled.

    ``update(now, ready=, parked=)`` is called once per warm-up tick with the mission clock and the
    station's ``ready`` (ack + collision stamps fresh in the /clock domain); it returns True once the
    hold has released (latched). Releases with reason ``settled`` when ``now >= settle_s`` and the
    streams have been continuously fresh with a parked base for ``steady_s``; with reason ``cap`` once
    the warm-up has lasted ``settle_s + max_extra_s`` (bounded, never an indefinite hold).
    ``release_line`` is the one console line naming when and why it released."""

    def __init__(self, settle_s, *, steady_s=_STEADY_S, max_extra_s=_MAX_EXTRA_S):
        self.settle_s = float(settle_s)
        if not math.isfinite(self.settle_s) or self.settle_s <= 0:
            raise ValueError('settle hold needs a positive finite settle_s')
        self.steady_s = min(float(steady_s), self.settle_s)
        self.max_extra_s = float(max_extra_s)
        self.released = False
        self.reason = None
        self.release_time = None
        self.release_line = None
        self.status = 'staged-start settle hold: waiting for the first warm-up tick'
        self._start = None
        self._steady_since = None
        self._ready_ticks = 0
        self._ticks = 0

    def update(self, now, *, ready, parked):
        if self.released:
            return True
        now = float(now)
        if self._start is None:
            self._start = now
        self._ticks += 1
        steady = bool(ready) and bool(parked)
        if steady:
            self._ready_ticks += 1
            if self._steady_since is None:
                self._steady_since = now
        else:
            self._steady_since = None
        steady_for = now - self._steady_since if steady else 0.0
        if steady and now >= self.settle_s and steady_for >= self.steady_s:
            return self._release('settled', now, steady_for)
        if now - self._start >= self.settle_s + self.max_extra_s:
            return self._release('cap', now, steady_for)
        self.status = (f'staged-start settle hold: sim {now:.1f}/{self.settle_s:g}s, station ack + '
                       f'collision stamps steady {steady_for:.1f}/{self.steady_s:g}s')
        return False

    def _release(self, reason, now, steady_for):
        self.released, self.reason, self.release_time = True, reason, now
        why = ('simulation settled: station ack + collision-world stamps fresh and base parked '
               f'for {steady_for:.2f}s >= {self.steady_s:g}s at sim {now:.2f}s >= {self.settle_s:g}s'
               if reason == 'settled' else
               f'cap: streams not steady after {now - self._start:.1f}s of warm-up '
               f'(> {self.settle_s:g}+{self.max_extra_s:g}s); dispatching, host guards decide')
        self.release_line = (f'STAGED_SETTLE_HOLD released reason={reason} sim={now:.3f} '
                             f'steady={steady_for:.3f} ready_ticks={self._ready_ticks}/{self._ticks} '
                             f'-- {why}')
        self.status = f'staged-start settle hold released ({reason})'
        return True


def summary_fields(config):
    """mission.json fields + caption fragment naming the staged-start switches ({} on a cold run)."""
    settle, sync = staged_settle_seconds(config), staged_clock_sync(config)
    if not is_staged_start(config):
        return {}
    parts = []
    if settle > 0:
        parts.append(f'settle hold {settle:g}s')
    if sync:
        parts.append('host clock sync')
    return {'staged_start_settle_s': settle, 'staged_start_clock_sync': sync,
            'staged_start_mode': '; '.join(parts) or 'off'}
