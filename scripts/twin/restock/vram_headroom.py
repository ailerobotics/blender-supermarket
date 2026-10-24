"""Opt-in VRAM headroom monitor for the Isaac restock mission (sim-robust-vram-headroom-v1).

The full capstone runs within ~500 MiB of the 12 GiB card top (completes peak ~11.7 GB used, with
whispr ~2.2 GB resident). coldwin13 -- byte-identical to the passing coldwin11 -- died in a native
SIGSEGV during the transfer approach with sampled VRAM at 10470 MiB used / 1438 free; the 20 s
orchestrator sampler can miss a transient spike, so the ceiling, a sub-sample spike, and a non-memory
native fault are all live hypotheses. This monitor gives the run real headroom: it samples used VRAM
on the SIM clock (sub-second), and when free headroom drops below a configurable floor it SHEDS
optional load in a fixed, named order, logging each shed by name with the reading, and names the
ceiling by name if it is reached anyway.

One runtime-config key turns it on (absent/`false` => byte-identical, no monitor constructed)::

    "vram_headroom": {"enabled": true, "floor_mib": 800}

Design:
* ``floor_mib`` (default 800) -- shed when FREE VRAM drops below this.
* ``sample_period_s`` (default 0.2 SIM seconds) -- sub-second so a spike between the orchestrator's
  20 s samples is still seen. The reading comes from a cheap in-process sampler (NVML if importable,
  else a background ``nvidia-smi`` poller thread), injectable so the policy is CPU-tested.
* ``shed_order`` (default ``capture_rate`` -> ``arm_feed_economy`` -> ``secondary_cameras``) -- a
  RATCHET: each sample below the floor fires the NEXT still-available shed and never un-sheds (no
  thrash). A shed that is not safe in the current stage (``arm_feed_economy`` depends on the nvblox
  feed in collect/transfer/verify) is skipped and the monitor escalates.
* **Never shed** a load-bearing honest-stage input (the live_pose transfer arm feed, nvblox, the
  planner). ``secondary_cameras`` frees only the station decode render products this mission never
  reads (the fm380/fm381 teardown, VRAM-measured ~0.95 GB+), so it is safe by construction.
* **Ceiling:** once every available shed has fired and free VRAM is still below the floor, log
  ``VRAM_CEILING_REACHED`` once and record it -- the cause is named, not silent.
* **Durable last reading:** ``vram-headroom.json`` (latest + peak + fired sheds + ceiling flag) is
  written atomically ~1 sim-s so the crash record can carry the last sampled headroom (``crash_record``).

Pure: the config resolution + the shed POLICY import nothing from Isaac/ROS, so they are unit-tested
on a recorded VRAM timeline (coldwin13's included). The actuators (widen the record stride, cap the
arm feed, destroy the unread cameras) are INJECTED by the bundle; the monitor only decides + logs.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field, fields

CARD_MIB_DEFAULT = 12288.0
FLOOR_MIB_DEFAULT = 800.0
SAMPLE_PERIOD_S_DEFAULT = 0.2
DEFAULT_SHED_ORDER = ('capture_rate', 'arm_feed_economy', 'secondary_cameras')
KNOWN_SHEDS = frozenset(DEFAULT_SHED_ORDER)

# Stages whose honest work consumes the arm eye-in-hand masked-depth feed (nvblox collision map): the
# arm-feed economy shed must NOT fire here. Mirrors isaac_port.SENSING_ACTIONS.
NVBLOX_FEED_STAGES = frozenset({'collect', 'transfer', 'verify'})


# --------------------------------------------------------------------------- config (pure)

@dataclass(frozen=True)
class VramHeadroom:
    """Resolved headroom-monitor levers. Defaults = disabled (byte-identical)."""
    enabled: bool = False
    floor_mib: float = FLOOR_MIB_DEFAULT
    card_mib: float = CARD_MIB_DEFAULT
    sample_period_s: float = SAMPLE_PERIOD_S_DEFAULT
    shed_order: tuple = DEFAULT_SHED_ORDER
    durable_write_period_s: float = 1.0   # throttle the vram-headroom.json rewrite (sim seconds)

    @property
    def active(self):
        return bool(self.enabled)

    def describe(self):
        return ('VRAM_HEADROOM enabled=%s floor_mib=%g card_mib=%g sample_period_s=%g shed_order=%s'
                % (self.enabled, self.floor_mib, self.card_mib, self.sample_period_s,
                   list(self.shed_order)))

    def summary(self):
        return {f.name: (list(getattr(self, f.name)) if isinstance(getattr(self, f.name), tuple)
                         else getattr(self, f.name)) for f in fields(self)}


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or value <= 0:
        raise ValueError(f'vram_headroom.{name} must be a positive number')
    return float(value)


def from_config(config):
    """Resolve ``config['vram_headroom']`` (absent/false => disabled). Fails closed on a typo so a bad
    key refuses the boot rather than silently running unprotected."""
    block = config.get('vram_headroom') if isinstance(config, dict) else None
    if block is None or block is False:
        return VramHeadroom()
    if block is True:
        block = {'enabled': True}
    if not isinstance(block, dict):
        raise ValueError('vram_headroom must be an object such as {"enabled": true, "floor_mib": 800}')
    known = {f.name for f in fields(VramHeadroom)}
    unknown = set(block) - known
    if unknown:
        raise ValueError(f'vram_headroom has unknown keys {sorted(unknown)}; known: {sorted(known)}')
    enabled = bool(block.get('enabled', True))
    floor = _positive('floor_mib', block.get('floor_mib', FLOOR_MIB_DEFAULT))
    card = _positive('card_mib', block.get('card_mib', CARD_MIB_DEFAULT))
    period = _positive('sample_period_s', block.get('sample_period_s', SAMPLE_PERIOD_S_DEFAULT))
    write_period = _positive('durable_write_period_s',
                             block.get('durable_write_period_s', 1.0))
    order = block.get('shed_order', DEFAULT_SHED_ORDER)
    if isinstance(order, (list, tuple)):
        order = tuple(order)
    else:
        raise ValueError('vram_headroom.shed_order must be a list of shed names')
    unknown_shed = set(order) - KNOWN_SHEDS
    if unknown_shed:
        raise ValueError(f'vram_headroom.shed_order has unknown sheds {sorted(unknown_shed)}; '
                         f'known: {sorted(KNOWN_SHEDS)}')
    if len(set(order)) != len(order):
        raise ValueError('vram_headroom.shed_order must not repeat a shed')
    return VramHeadroom(enabled=enabled, floor_mib=floor, card_mib=card, sample_period_s=period,
                        shed_order=order, durable_write_period_s=write_period)


def summary_fields(config):
    """mission.json provenance: ``{'vram_headroom_mode': 'on', 'vram_headroom': {...}}`` when enabled,
    else {} (byte-identical summaries)."""
    cfg = from_config(config)
    if not cfg.active:
        return {}
    return {'vram_headroom_mode': 'on', 'vram_headroom': cfg.summary()}


# --------------------------------------------------------------------------- shed policy (pure)

@dataclass
class Decision:
    fire: str | None = None       # a shed name to actuate this sample
    ceiling: bool = False         # the ceiling was named this sample (once)
    low: bool = False             # below floor but no shed available right now


class ShedPolicy:
    """The fixed-order shed ratchet. Stateful but PURE (no I/O): unit-tested on a VRAM timeline.

    ``observe(free_mib, available)`` fires AT MOST ONE shed per sample -- the first of ``shed_order``
    not yet fired AND currently ``available`` (safe in the stage + has an actuator). It never
    un-sheds. When free is below the floor and every shed in the order has fired, it names the ceiling
    once (``Decision.ceiling``). When below the floor but the only remaining sheds are unavailable
    (stage-blocked), it reports ``Decision.low`` (a throttled warning, no action)."""

    def __init__(self, floor_mib, shed_order):
        self.floor_mib = float(floor_mib)
        self.shed_order = tuple(shed_order)
        self.fired = []            # in fire order
        self.ceiling_named = False

    def observe(self, free_mib, available, fireable_ever=None):
        """``available`` = sheds fireable NOW (stage-safe + has an actuator); ``fireable_ever`` = sheds
        that could EVER fire (have an actuator, any stage; default: the whole order). The ceiling is
        named only when every ever-fireable shed has fired and free is still below the floor -- a shed
        that is merely stage-blocked right now (in ``fireable_ever`` but not ``available``) keeps the
        verdict at ``low`` (it may yet fire in a later stage), while a shed with no actuator at all is
        not in ``fireable_ever`` and so never holds the ceiling open."""
        if free_mib is None:
            return Decision()
        if float(free_mib) >= self.floor_mib:
            return Decision()
        available = set(available)
        fireable_ever = set(self.shed_order if fireable_ever is None else fireable_ever)
        for shed in self.shed_order:
            if shed not in self.fired and shed in available:
                self.fired.append(shed)
                return Decision(fire=shed)
        # Below floor, nothing fireable right now. Ceiling iff every EVER-fireable shed has fired.
        remaining = [s for s in self.shed_order if s in fireable_ever and s not in self.fired]
        if not remaining:
            if not self.ceiling_named:
                self.ceiling_named = True
                return Decision(ceiling=True)
            return Decision(low=True)
        return Decision(low=True)


def shed_is_stage_safe(shed, stage):
    """Whether ``shed`` may fire in ``stage`` without starving an honest-stage input. Pure.

    ``arm_feed_economy`` throttles the eye-in-hand masked-depth feed, which nvblox consumes in
    collect/transfer/verify -- unsafe there. ``capture_rate`` (walkthrough recording) and
    ``secondary_cameras`` (frees only never-read decode products) are safe everywhere."""
    if shed == 'arm_feed_economy':
        return stage not in NVBLOX_FEED_STAGES
    return True


# --------------------------------------------------------------------------- samplers (impure)

def nvml_sampler():
    """Return a ``() -> (used_mib, free_mib)`` reading NVML (cheap, ~1 ms), or None if unavailable."""
    try:
        import pynvml  # type: ignore
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None

    def sample():
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / (1024 * 1024), info.free / (1024 * 1024)
        except Exception:
            return None
    return sample


class SmiPoller:
    """Background ``nvidia-smi`` poller: samples every ``period_s`` WALL seconds in a daemon thread and
    exposes the latest (used_mib, free_mib). Used only when NVML is unavailable, so the per-tick cost
    never forks a subprocess. Best-effort: a missing/odd nvidia-smi leaves the latest reading None."""

    def __init__(self, period_s=0.2):
        self.period_s = max(0.05, float(period_s))
        self._latest = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='vram-smi-poller', daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        import subprocess
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.free',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5, check=True).stdout
                row = out.strip().splitlines()[0]
                used, free = [int(float(p.strip())) for p in row.split(',')[:2]]
                self._latest = (float(used), float(free))
            except Exception:
                pass  # leave the previous latest; never raise out of the poller
            self._stop.wait(self.period_s)

    def sample(self):
        return self._latest

    def close(self):
        self._stop.set()


def make_sampler(period_s=0.2):
    """Best in-process VRAM sampler: NVML if importable, else a background nvidia-smi poller. Returns
    ``(sample_callable, closer)``; ``sample_callable()`` -> (used_mib, free_mib) or None."""
    nvml = nvml_sampler()
    if nvml is not None:
        return nvml, (lambda: None)
    poller = SmiPoller(period_s).start()
    return poller.sample, poller.close


# --------------------------------------------------------------------------- monitor (impure)

def _log(line):
    print(line, flush=True)


class VramHeadroomMonitor:
    """Samples VRAM on the sim clock and actuates the shed ratchet. Constructed only when enabled.

    ``actuators`` maps a shed name to a zero-arg callable returning a short description of what it did
    (or raising -> logged, treated as unavailable). ``begin_stage`` tracks the current stage for the
    stage-safety gate; ``tick(sim_time)`` is called once per mission tick (cheap: it only reads the
    sampler's latest in-memory value and, at most ~1/sim-s, rewrites the durable file)."""

    def __init__(self, cfg, *, sampler=None, closer=None, actuators=None, run_dir=None,
                 clock=time.monotonic, log=_log, stage_safe=shed_is_stage_safe):
        self.cfg = cfg
        self.policy = ShedPolicy(cfg.floor_mib, cfg.shed_order)
        self._sampler = sampler
        self._closer = closer or (lambda: None)
        self.actuators = dict(actuators or {})
        self.run_dir = run_dir
        self._clock = clock
        self._log = log
        self._stage_safe = stage_safe
        self.stage = None
        self._last_sample_sim = None
        self._last_write_sim = None
        self.peak_used = 0.0
        self.last_reading = None        # (used, free, sim_time)
        self.ceiling_reached = False
        if self._sampler is None:
            self._sampler, self._closer = make_sampler(cfg.sample_period_s)
        self._log(cfg.describe())

    def begin_stage(self, stage):
        self.stage = stage

    def _available(self):
        return [s for s in self.cfg.shed_order
                if s in self.actuators and self._stage_safe(s, self.stage)]

    def tick(self, sim_time):
        if sim_time is None:
            return
        sim_time = float(sim_time)
        if self._last_sample_sim is not None \
                and sim_time - self._last_sample_sim < self.cfg.sample_period_s:
            return
        self._last_sample_sim = sim_time
        reading = None
        try:
            reading = self._sampler()
        except Exception as exc:  # noqa: BLE001
            self._log('VRAM_SAMPLE_ERROR %r' % (exc,))
        if reading is None:
            return
        used, free = float(reading[0]), float(reading[1])
        self.last_reading = (used, free, sim_time)
        if used > self.peak_used:
            self.peak_used = used
        decision = self.policy.observe(free, self._available(), fireable_ever=set(self.actuators))
        if decision.fire:
            self._actuate(decision.fire, used, free, sim_time)
        elif decision.ceiling:
            self.ceiling_reached = True
            self._log('VRAM_CEILING_REACHED stage=%s used=%.0f free=%.0f floor=%.0f sim=%.3f -- every '
                      'available shed fired and free VRAM is still below the floor; the 12 GiB ceiling '
                      'is the limit (sheds fired: %s)'
                      % (self.stage, used, free, self.cfg.floor_mib, sim_time,
                         self.policy.fired or 'none'))
        elif decision.low:
            self._log('VRAM_HEADROOM_LOW stage=%s used=%.0f free=%.0f floor=%.0f sim=%.3f -- below '
                      'floor; no shed safe to fire in this stage (fired: %s)'
                      % (self.stage, used, free, self.cfg.floor_mib, sim_time,
                         self.policy.fired or 'none'))
        self._maybe_write(sim_time)

    def _actuate(self, shed, used, free, sim_time):
        try:
            detail = self.actuators[shed]()
        except Exception as exc:  # noqa: BLE001 -- a failed shed must not crash the tick
            detail = 'FAILED %r' % (exc,)
        self._log('VRAM_SHED name=%s stage=%s used=%.0f free=%.0f floor=%.0f sim=%.3f detail=%s'
                  % (shed, self.stage, used, free, self.cfg.floor_mib, sim_time, detail))
        self._write(sim_time)  # persist the shed immediately

    def _maybe_write(self, sim_time):
        if self._last_write_sim is None \
                or sim_time - self._last_write_sim >= self.cfg.durable_write_period_s:
            self._write(sim_time)

    def _write(self, sim_time):
        if not self.run_dir:
            return
        self._last_write_sim = sim_time
        used, free = (self.last_reading[0], self.last_reading[1]) if self.last_reading else (None, None)
        record = {
            'used_mib': used,
            'free_mib': free,
            'headroom_mib': free,
            'sim_time': sim_time,
            'peak_used_mib': self.peak_used,
            'floor_mib': self.cfg.floor_mib,
            'card_mib': self.cfg.card_mib,
            'fired_sheds': list(self.policy.fired),
            'ceiling_reached': self.ceiling_reached,
            'wall_time': self._clock(),
        }
        try:
            from pathlib import Path
            out = Path(self.run_dir) / 'vram-headroom.json'
            tmp = out.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(record, allow_nan=False) + '\n')
            os.replace(tmp, out)
        except Exception as exc:  # noqa: BLE001
            self._log('VRAM_HEADROOM_WRITE skipped: %r' % (exc,))

    def close(self):
        try:
            if self.last_reading is not None:
                self._write(self.last_reading[2])
        finally:
            try:
                self._closer()
            except Exception:
                pass
