"""Transfer-entry VRAM pre-shed (sim-transfer-vram-preshed-v1).

coldwin93 crashed the whole simulation natively (SIGSEGV at 13.14 GB) during the live_pose TRANSFER
stage: the run entered transfer with the station decode cameras AND the collect-workspace ESDF bounds
STILL resident (nothing freed them before transfer), so steady VRAM sat at 11791 MiB used / 117 free,
and the ten PLANNING_FAILED cuMotion retries then allocated a ~1.35 GB working set into that sliver.
The :mod:`vram_headroom` monitor sheds REACTIVELY and its only runtime actuator in transfer
(``capture_rate``, a recording stride) frees no VRAM, so it could not open headroom; and a spike that
fast cannot be caught after the fact.

This module is the PROACTIVE, DETERMINISTIC counterpart: on ENTERING the transfer stage, BEFORE its
second full-res (arm masked-depth) render boots, it fires a fixed, named sequence of releases --
    1. ``mast_feed``          -- drop the secondary (mast, camera_1) masked-depth feed if present;
    2. ``decode_economy``     -- switch the station decode/source/cart render to the economy resolution;
    3. ``collect_capture``    -- release the collect-stage capture buffers (station decode products);
    4. ``collect_esdf_bounds``-- release the nvblox ESDF voxels covering the collect-only workspace;
-- logs ``VRAM_PRESHED`` with the before/after free headroom, and REQUIRES at least ``min_free_mib``
(default 1.5 GB) free before the transfer render is allowed to boot, FAILING BY NAME otherwise (so an
over-budget transfer refuses up front instead of crashing the allocator ~17 min in).

Opt-in (HOST/RUNTIME key ``transfer_preshed``), DEFAULT OFF => absent block => byte-identical run. The
config resolution and the sequencing POLICY are pure (no Isaac/ROS/GPU import); the actuators (drop the
feed, retarget the decode render, free the ESDF bounds) and the VRAM sampler are INJECTED by the bundle,
so the sequence + gate are unit-tested on recorded headroom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

# The fixed pre-shed order. Each name maps to an injected actuator; a name with no actuator in the
# current run is skipped (logged), never an error (e.g. a start_at=couple run never created the collect
# capture buffers). NOTHING load-bearing is ever here: the arm feed, nvblox transfer map and planner stay.
PRESHED_ORDER = ('mast_feed', 'decode_economy', 'collect_capture', 'collect_esdf_bounds')
KNOWN_SHEDS = frozenset(PRESHED_ORDER)
# Default floor: the transfer render must boot into >= this much free VRAM (brief scope 1: >= 1.5 GB).
MIN_FREE_MIB_DEFAULT = 1536.0


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or value <= 0:
        raise ValueError(f'transfer_preshed.{name} must be a positive number')
    return float(value)


@dataclass(frozen=True)
class TransferPreshed:
    """Resolved transfer-entry pre-shed levers. The bare ``TransferPreshed()`` is the DISABLED sentinel
    (absent key / explicit ``false``); the pre-shed only ever runs when ``enabled``."""
    enabled: bool = False
    min_free_mib: float = MIN_FREE_MIB_DEFAULT
    order: tuple = PRESHED_ORDER
    require_min_free: bool = True   # fail by name if the post-shed free is below min_free_mib

    @property
    def active(self):
        return bool(self.enabled)

    def describe(self):
        return ('TRANSFER_PRESHED enabled=%s min_free_mib=%g require_min_free=%s order=%s'
                % (self.enabled, self.min_free_mib, self.require_min_free, list(self.order)))

    def summary(self):
        return {f.name: (list(getattr(self, f.name)) if isinstance(getattr(self, f.name), tuple)
                         else getattr(self, f.name)) for f in fields(self)}


def from_config(config):
    """Resolve ``config['transfer_preshed']``. DEFAULT OFF: absent/``false``/``{"enabled": false}`` =>
    disabled (byte-identical). ``true`` => enabled defaults. Fails CLOSED on a typo / bad value."""
    block = config.get('transfer_preshed') if isinstance(config, dict) else None
    if block is None or block is False:
        return TransferPreshed()
    if block is True:
        block = {'enabled': True}
    if not isinstance(block, dict):
        raise ValueError('transfer_preshed must be an object such as {"enabled": true, "min_free_mib": 1536}')
    known = {f.name for f in fields(TransferPreshed)}
    unknown = set(block) - known
    if unknown:
        raise ValueError(f'transfer_preshed has unknown keys {sorted(unknown)}; known: {sorted(known)}')
    enabled = bool(block.get('enabled', True))
    min_free = _positive('min_free_mib', block.get('min_free_mib', MIN_FREE_MIB_DEFAULT))
    require = bool(block.get('require_min_free', True))
    order = block.get('order', PRESHED_ORDER)
    if not isinstance(order, (list, tuple)):
        raise ValueError('transfer_preshed.order must be a list of shed names')
    order = tuple(order)
    unknown_shed = set(order) - KNOWN_SHEDS
    if unknown_shed:
        raise ValueError(f'transfer_preshed.order has unknown sheds {sorted(unknown_shed)}; '
                         f'known: {sorted(KNOWN_SHEDS)}')
    if len(set(order)) != len(order):
        raise ValueError('transfer_preshed.order must not repeat a shed')
    return TransferPreshed(enabled=enabled, min_free_mib=min_free, order=order, require_min_free=require)


def summary_fields(config):
    """mission.json provenance: ``{'transfer_preshed_mode': 'on', 'transfer_preshed': {...}}`` when
    enabled, else {} (byte-identical summaries)."""
    cfg = from_config(config)
    if not cfg.active:
        return {}
    return {'transfer_preshed_mode': 'on', 'transfer_preshed': cfg.summary()}


class TransferPreshedInsufficientFree(RuntimeError):
    """Raised (named) when the post-shed free VRAM is still below the required floor."""


@dataclass
class PreshedResult:
    before_free_mib: float | None
    after_free_mib: float | None
    fired: list            # (name, detail) in order actually fired
    skipped: list          # names with no actuator this run
    min_free_mib: float
    ok: bool               # post-shed free cleared the floor (or the floor was not required / unknown)


def run_preshed(cfg, actuators, *, sampler=None, log=print):
    """Execute the transfer-entry pre-shed ONCE and gate the transfer render on the freed headroom.

    ``cfg`` is a resolved :class:`TransferPreshed`; ``actuators`` maps a shed name to a zero-arg callable
    returning a short description (or raising -> logged, counted as a failed shed, never crashing the
    tick). ``sampler`` -> ``(used_mib, free_mib)`` or None reads the card before and after the sequence.

    Logs ``VRAM_PRESHED`` with before/after free, fires every configured shed that has an actuator (in
    ``cfg.order``), and -- when ``cfg.require_min_free`` -- RAISES :class:`TransferPreshedInsufficientFree`
    if the measured post-shed free is still below ``cfg.min_free_mib``. Returns a :class:`PreshedResult`.
    Pure except for the injected ``sampler``/``actuators``/``log``; unit-tested on recorded headroom."""
    def _free():
        if sampler is None:
            return None
        try:
            reading = sampler()
        except Exception as exc:  # noqa: BLE001 -- a sampler fault must not crash the pre-shed
            log('VRAM_PRESHED_SAMPLE_ERROR %r' % (exc,))
            return None
        if reading is None:
            return None
        return float(reading[1])

    before = _free()
    fired, skipped = [], []
    for name in cfg.order:
        act = actuators.get(name)
        if act is None:
            skipped.append(name)
            continue
        try:
            detail = act()
        except Exception as exc:  # noqa: BLE001
            detail = 'FAILED %r' % (exc,)
        fired.append((name, detail))
        log('VRAM_PRESHED_STEP name=%s detail=%s' % (name, detail))
    after = _free()
    freed = (after - before) if (before is not None and after is not None) else None
    log('VRAM_PRESHED before_free=%s after_free=%s freed=%s min_free=%.0f fired=%s skipped=%s'
        % ('%.0f' % before if before is not None else 'n/a',
           '%.0f' % after if after is not None else 'n/a',
           '%.0f' % freed if freed is not None else 'n/a',
           cfg.min_free_mib, [n for n, _ in fired], skipped))

    ok = True
    if cfg.require_min_free and after is not None and after < cfg.min_free_mib:
        ok = False
    result = PreshedResult(before_free_mib=before, after_free_mib=after, fired=fired,
                           skipped=skipped, min_free_mib=cfg.min_free_mib, ok=ok)
    if not ok:
        raise TransferPreshedInsufficientFree(
            'transfer pre-shed left only %.0f MiB free (< required %.0f MiB) after firing %s; refusing '
            'to boot the transfer render into a near-full card' % (after, cfg.min_free_mib,
                                                                   [n for n, _ in fired] or 'nothing'))
    return result
