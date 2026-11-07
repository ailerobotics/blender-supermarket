"""Advisory VRAM budget for the restock mission's camera feeds on the 12 GB card.

The live_pose transfer runs the full RTX render + nvblox ESDF + cuMotion planner AND one (or
two) eye-in-hand masked-depth feeds on a 12 GB card that ALSO holds the always-on ``whispr``
service (~2.2 GB, which must never be touched). The transfer2 proof peaked at 11882 MiB used /
**26 MiB free** -- it fits with a single arm feed only by a sliver, and a second full-res feed
(the opt-in mast camera_1) would not fit. This module is the pre-boot GATE for that: given the
mission config's feeds + ESDF grid it estimates peak VRAM and reports whether the card has
headroom after the resident services, so the integration lane can refuse an over-budget config
BEFORE booting Isaac beside the live stack instead of OOMing ~17 min in.

The model is ADVISORY and transparent: every constant is anchored to the transfer2 measurement
(2026-10-22) and overridable. It is NOT a substitute for the on-card peak sampler -- it is a
cheap, CPU-only sanity gate that the configured feeds plausibly fit with whispr resident.

Evidence anchor (transfer2, live_pose transfer, ONE full-res arm masked-depth feed):
  card total 12288 MiB; used 11882 MiB (incl. whispr ~2252); free 26 MiB.
  => mission (Isaac + host + nvblox + cuMotion + 1 arm feed) ~= 11882 - 2252 = 9630 MiB.
"""
from __future__ import annotations

from dataclasses import dataclass

# 12 GB card (MiB). Override per host if the card differs.
CARD_MIB = 12288
# Always-on whispr service VRAM (~2.2 GB). Resident; must not be touched (AGENTS.md / brief).
WHISPR_MIB = 2252
# One eye-in-hand masked-depth feed: a fixed RTX render-product/context cost plus a per-pixel
# term (RGB + distance_to_image_plane f32 + instance-id seg i32 render targets, double-buffered).
# The fixed term dominates; fm373 found render RESOLUTION saves only ~160 MiB, i.e. dropping a
# whole feed is the real lever, not shrinking one.
FEED_FIXED_MIB = 430.0
FEED_BYTES_PER_PIXEL = 64.0
# Mission base WITHOUT any eye-in-hand feed (Isaac + host + nvblox ESDF + cuMotion). Anchored so
# base + one reference arm feed == the measured 9630 MiB mission footprint (see module docstring).
_REFERENCE_FEED_WIDTH = 1280
_REFERENCE_FEED_HEIGHT = 720
BASE_MISSION_MIB = 9630.0 - (FEED_FIXED_MIB
                             + _REFERENCE_FEED_WIDTH * _REFERENCE_FEED_HEIGHT
                             * FEED_BYTES_PER_PIXEL / (1024 * 1024))
# Safety headroom to keep free on the card (driver churn, fragmentation, transient spikes).
DEFAULT_MARGIN_MIB = 256.0

# Phase-dependent base (sim-rgbd-arm-collision-aware-v1 / sim-perception-architecture-s1 s7): the
# mission VRAM peak is set by the STAGE, not a single number. The anchor above (BASE_MISSION_MIB)
# is the live_pose TRANSFER peak (transfer2: 26 MiB free -- nothing more fits). The COLLECT / carry
# / set-down stages peak lower: coldwin60-62 measured ~10.5-10.7 GB used (incl. whispr ~2252 + ONE
# wrist feed), i.e. collect mission base = 10736 - 2252 - one wrist feed(424x240). So the mast
# (~455) AND/OR a dynamic layer fit in collect but NOT in transfer -- the schedule (arm_feed_schedule)
# keeps the mast OFF during live_pose transfer, and this base lets the pre-boot gate confirm a
# mast-on COLLECT config is fits=True.
# Wrist feed term (424x240, FEED_FIXED + per-pixel) computed inline so this constant is module-level.
_WRIST_FEED_MIB = FEED_FIXED_MIB + 424 * 240 * FEED_BYTES_PER_PIXEL / (1024 * 1024)
_COLLECT_PEAK_WITH_ONE_WRIST_MIB = 10736.0   # coldwin60 peak used (incl. whispr + 1 wrist feed)
COLLECT_BASE_MISSION_MIB = _COLLECT_PEAK_WITH_ONE_WRIST_MIB - WHISPR_MIB - _WRIST_FEED_MIB


# --------------------------------------------------------------------------- transfer model (sim-transfer-vram-preshed-v1)
# The live_pose TRANSFER stage runs the card hardest (transfer2: 26 MiB free). cw93 proved the pre-boot
# feed-only budget above is NOT enough: that run entered transfer with the station decode cameras AND the
# collect-workspace ESDF bounds STILL resident (no transfer-entry pre-shed existed), so steady VRAM sat at
# 11791 MiB used / 117 free, and the ten PLANNING_FAILED cuMotion retries then allocated a ~1.35 GB working
# set into that sliver -> 13142 MiB -> native SIGSEGV. This section models the transfer AS IT ACTUALLY RUNS
# so the pre-launch gate can REFUSE such a config (predicted peak over card - 1 GB) and confirm the
# transfer-entry pre-shed + planner spike guard bring it back under the line.
#
# The card-minus-1-GB transfer ceiling (brief scope 2). A modelled transfer peak above this is refused.
TRANSFER_PEAK_MARGIN_MIB = 1024.0
# The pre-shed must leave at least this much FREE before the transfer render boots (brief scope 1: >=1.5 GB).
PRESHED_MIN_FREE_MIB = 1536.0
#
# BASE_MISSION_MIB is decomposed into the IRREDUCIBLE lean base + the two SHEDDABLE steady components that
# were still resident entering cw93's transfer (the transfer-entry pre-shed releases them):
#   * DECODE_CAMERAS_MIB       -- the 2560x1920 station decode/source/cart render products. The fm380
#                                 stage-scoped teardown measured ~0.95 GB returned; the pre-shed switches
#                                 them to the economy render resolution (DECODE_CAMERAS_ECONOMY_MIB).
#   * ESDF_COLLECT_BOUNDS_MIB  -- the nvblox ESDF voxels covering the collect workspace beyond the transfer
#                                 reach, released when the collect bounds are dropped at transfer entry.
# TRANSFER_LEAN_BASE_MIB is the remainder (Isaac runtime + the walkthrough RTX render + cuMotion resident +
# the transfer-only ESDF). lean + decode + collect == BASE_MISSION_MIB, so the legacy anchor is preserved.
DECODE_CAMERAS_MIB = 950.0
DECODE_CAMERAS_ECONOMY_MIB = 150.0
ESDF_COLLECT_BOUNDS_MIB = 600.0
TRANSFER_LEAN_BASE_MIB = BASE_MISSION_MIB - DECODE_CAMERAS_MIB - ESDF_COLLECT_BOUNDS_MIB
# The cuMotion planner-retry working set. cw93's spike delta was 13142 - 11791 = 1351 MiB across the ten
# retries. WITHOUT the spike guard the model charges the full working set (the crash); WITH the guard the
# retries PAUSE below the 400 MiB free floor, so at most ONE dispatch's working set is ever resident.
PLAN_RETRY_WORKING_SET_MIB = 1350.0
PLAN_RETRY_GUARDED_MIB = 350.0


def feed_mib(width, height, *, fixed=FEED_FIXED_MIB, bytes_per_pixel=FEED_BYTES_PER_PIXEL):
    """Estimated VRAM (MiB) for ONE masked-depth feed at ``width`` x ``height``.

    Lowering an arm feed's render resolution is the config lever the integration lane can apply to
    cut pressure; this ties the estimate to it (the per-pixel term shrinks, the fixed term stays).
    """
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        raise ValueError('feed resolution must be positive')
    return float(fixed) + w * h * float(bytes_per_pixel) / (1024 * 1024)


@dataclass
class BudgetVerdict:
    fits: bool
    estimated_mib: float          # mission + resident estimate
    free_after_mib: float         # card - estimate (negative => over budget)
    margin_mib: float
    detail: dict

    def __bool__(self):
        return self.fits


def estimate_mission_mib(feeds, *, base=BASE_MISSION_MIB):
    """Estimated mission VRAM (MiB), EXCLUDING resident services.

    ``feeds`` is an iterable of ``(width, height)`` for each ENABLED masked-depth feed (e.g. the
    arm; plus the mast when ``mast_depth.enabled``). An empty iterable is the no-eye-in-hand base.
    """
    total = float(base)
    per_feed = [feed_mib(w, h) for (w, h) in feeds]
    return total + sum(per_feed), per_feed


def check_budget(feeds, *, card_mib=CARD_MIB, resident_mib=WHISPR_MIB, base=BASE_MISSION_MIB,
                 margin_mib=DEFAULT_MARGIN_MIB):
    """Does the configured mission (its feeds) fit the card with resident services + margin?

    Returns a :class:`BudgetVerdict`; ``fits`` is True iff
    ``resident + mission_estimate + margin <= card``. Used as a pre-boot gate.
    """
    mission, per_feed = estimate_mission_mib(feeds, base=base)
    estimated = float(resident_mib) + mission
    free_after = float(card_mib) - estimated
    fits = free_after >= float(margin_mib)
    return BudgetVerdict(
        fits=fits, estimated_mib=estimated, free_after_mib=free_after, margin_mib=float(margin_mib),
        detail={'card_mib': float(card_mib), 'resident_mib': float(resident_mib),
                'base_mib': float(base), 'mission_mib': mission, 'per_feed_mib': per_feed,
                'num_feeds': len(per_feed)})


def _economy_helpers():
    """Return ``(from_block, scaled_resolution)`` from masked_depth_economy, import-path robust."""
    try:
        from .masked_depth_economy import from_block as _economy_from_block, scaled_resolution
    except ImportError:
        # Run by file path (``python3 .../gpu_budget.py``) there is no parent package, so the
        # relative import fails; put the ``scripts/`` root on sys.path and import absolutely. The
        # ``-m twin.restock.gpu_budget`` path keeps the relative import and never reaches here.
        import sys as _sys
        from pathlib import Path as _Path
        _scripts_root = str(_Path(__file__).resolve().parents[2])
        if _scripts_root not in _sys.path:
            _sys.path.insert(0, _scripts_root)
        from twin.restock.masked_depth_economy import (  # type: ignore[no-redef]
            from_block as _economy_from_block, scaled_resolution)
    return _economy_from_block, scaled_resolution


def feed_resolution_from_block(block):
    """``(width, height)`` for one masked-depth feed block, applying its economy resolution_scale.

    Returns None when the block is absent/not a dict/not ``enabled``. Shared by :func:`feeds_from_config`
    and the transfer model so the arm/mast feeds size identically everywhere. Pure (no Isaac)."""
    if not isinstance(block, dict) or not block.get('enabled'):
        return None
    _economy_from_block, scaled_resolution = _economy_helpers()
    res = block.get('resolution')
    if res and isinstance(res, (list, tuple)) and len(res) == 2:
        w, h = int(res[0]), int(res[1])
    else:
        w, h = _REFERENCE_FEED_WIDTH, _REFERENCE_FEED_HEIGHT
    # sim-robust-arm-feed-economy-v1: a feed's economy resolution_scale shrinks the per-pixel VRAM
    # term (fm373: resolution is a small VRAM lever, but reflect it so the gate is self-consistent).
    scale = _economy_from_block(block.get('economy')).resolution_scale
    return scaled_resolution(w, h, scale) if scale != 1.0 else (w, h)


def feeds_from_config(config):
    """Extract the enabled masked-depth feeds' ``(width, height)`` from a mission-runtime config.

    Reads ``masked_depth`` (the arm feed, camera_0) and ``mast_depth`` (camera_1, opt-in). A feed's
    resolution comes from its ``resolution: [w, h]`` when set, else the reference arm resolution.
    A feed is counted only when its block opts in (``enabled``). Pure dict reader (no Isaac)."""
    feeds = []
    for key in ('masked_depth', 'mast_depth'):
        res = feed_resolution_from_block(config.get(key) or {})
        if res is not None:
            feeds.append(res)
    return feeds


def _feed_mib_for_key(config, key):
    """Estimated VRAM (MiB) for the feed under ``config[key]`` (0 when absent/disabled)."""
    res = feed_resolution_from_block(config.get(key) or {})
    return 0.0 if res is None else feed_mib(*res)


@dataclass
class TransferBudgetVerdict:
    """Pre-launch verdict for the live_pose TRANSFER stage modelled as it actually runs."""
    fits: bool
    predicted_peak_mib: float          # steady + planner retry working set (the crash-relevant peak)
    steady_mib: float                  # resident + lean base + feeds + resident sheddables (no spike)
    planner_mib: float                 # the cuMotion retry working set charged (bounded by the guard)
    free_at_peak_mib: float            # card - predicted peak (negative => over budget)
    free_at_steady_mib: float          # card - steady (the headroom the transfer render boots into)
    preshed: bool
    spike_guard: bool
    margin_mib: float
    preshed_min_free_mib: float
    detail: dict

    def __bool__(self):
        return self.fits


def check_transfer_budget(config, *, preshed=False, spike_guard=False, card_mib=CARD_MIB,
                          resident_mib=WHISPR_MIB, margin_mib=TRANSFER_PEAK_MARGIN_MIB,
                          preshed_min_free_mib=PRESHED_MIN_FREE_MIB):
    """Does the live_pose TRANSFER stage fit the card as it actually runs? (sim-transfer-vram-preshed-v1)

    Models the transfer peak from the mission config: the lean mission base (Isaac + walkthrough render +
    cuMotion + transfer ESDF), the arm masked-depth feed, and -- unless ``preshed`` releases them -- the
    opt-in mast feed, the station decode/source/cart cameras, and the collect-workspace ESDF bounds; plus
    the cuMotion planner-retry working set, charged in FULL unless ``spike_guard`` pauses the retries below
    the free floor. ``fits`` is True iff the predicted peak leaves at least ``margin_mib`` (card - 1 GB by
    default) free. When ``preshed`` is set, ``detail['preshed_free_ok']`` also reports whether the post-shed
    STEADY headroom clears ``preshed_min_free_mib`` (the >=1.5 GB the transfer render must boot into)."""
    arm_mib = _feed_mib_for_key(config, 'masked_depth')
    mast_mib = _feed_mib_for_key(config, 'mast_depth')
    decode_mib = DECODE_CAMERAS_ECONOMY_MIB if preshed else DECODE_CAMERAS_MIB
    collect_esdf_mib = 0.0 if preshed else ESDF_COLLECT_BOUNDS_MIB
    mast_charged = 0.0 if preshed else mast_mib   # the pre-shed drops the secondary (mast) feed
    planner_mib = PLAN_RETRY_GUARDED_MIB if spike_guard else PLAN_RETRY_WORKING_SET_MIB

    steady = (float(resident_mib) + TRANSFER_LEAN_BASE_MIB + arm_mib
              + mast_charged + decode_mib + collect_esdf_mib)
    predicted_peak = steady + planner_mib
    free_at_peak = float(card_mib) - predicted_peak
    free_at_steady = float(card_mib) - steady
    fits = free_at_peak >= float(margin_mib)
    preshed_free_ok = (free_at_steady >= float(preshed_min_free_mib)) if preshed else None
    return TransferBudgetVerdict(
        fits=fits, predicted_peak_mib=predicted_peak, steady_mib=steady, planner_mib=planner_mib,
        free_at_peak_mib=free_at_peak, free_at_steady_mib=free_at_steady,
        preshed=preshed, spike_guard=spike_guard, margin_mib=float(margin_mib),
        preshed_min_free_mib=float(preshed_min_free_mib),
        detail={'card_mib': float(card_mib), 'resident_mib': float(resident_mib),
                'lean_base_mib': TRANSFER_LEAN_BASE_MIB, 'arm_feed_mib': arm_mib,
                'mast_feed_mib': mast_charged, 'decode_cameras_mib': decode_mib,
                'collect_esdf_bounds_mib': collect_esdf_mib, 'preshed_free_ok': preshed_free_ok})


def check_collect_budget(config, *, card_mib=CARD_MIB, resident_mib=WHISPR_MIB,
                         margin_mib=DEFAULT_MARGIN_MIB):
    """Pre-boot budget for a COLLECT/carry/set-down proof run (the ~2.9 GB-free stage).

    Uses the lower collect-stage base (:data:`COLLECT_BASE_MISSION_MIB`) so a mast-on collect
    config -- wrist (camera_0) + mast (camera_1) -- can be confirmed ``fits=True`` before boot, the
    headroom window the phase schedule (:mod:`arm_feed_schedule`) relies on. The feeds are the
    configured masked_depth/mast_depth feeds (same reader as the transfer check). Keep the mast OFF
    for the ``live_pose`` transfer stage -- use :func:`check_budget` (the transfer peak) there.
    """
    return check_budget(feeds_from_config(config), card_mib=card_mib, resident_mib=resident_mib,
                        base=COLLECT_BASE_MISSION_MIB, margin_mib=margin_mib)


def _main(argv):
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description='Pre-boot VRAM budget gate for the restock camera feeds.')
    ap.add_argument('config', nargs='?', help='mission-runtime config JSON (reads masked_depth/mast_depth)')
    ap.add_argument('--card-mib', type=float, default=CARD_MIB)
    ap.add_argument('--resident-mib', type=float, default=WHISPR_MIB,
                    help='resident service VRAM to reserve (default: whispr ~2252)')
    ap.add_argument('--margin-mib', type=float, default=DEFAULT_MARGIN_MIB)
    ap.add_argument('--feed', action='append', default=[], metavar='WxH',
                    help='add an explicit feed resolution (repeatable), overrides the config feeds')
    ap.add_argument('--transfer', action='store_true',
                    help='model the live_pose TRANSFER stage as it runs (requires a config)')
    ap.add_argument('--preshed', action='store_true', help='model the transfer-entry pre-shed as applied')
    ap.add_argument('--spike-guard', action='store_true',
                    help='model the planner-retry spike guard as active')
    args = ap.parse_args(argv)

    if args.transfer:
        if not args.config:
            ap.error('--transfer requires a mission-runtime config')
        config = json.loads(Path(args.config).read_text())
        v = check_transfer_budget(config, preshed=args.preshed, spike_guard=args.spike_guard,
                                  card_mib=args.card_mib, resident_mib=args.resident_mib)
        print('GPU_TRANSFER_BUDGET fits=%s preshed=%s spike_guard=%s predicted_peak=%.0f MiB '
              'steady=%.0f MiB free_at_peak=%.0f free_at_steady=%.0f (margin=%.0f) detail=%s'
              % (v.fits, v.preshed, v.spike_guard, v.predicted_peak_mib, v.steady_mib,
                 v.free_at_peak_mib, v.free_at_steady_mib, v.margin_mib, v.detail))
        return 0 if v.fits else 1

    if args.feed:
        feeds = []
        for spec in args.feed:
            w, h = spec.lower().split('x')
            feeds.append((int(w), int(h)))
    elif args.config:
        feeds = feeds_from_config(json.loads(Path(args.config).read_text()))
    else:
        feeds = [(_REFERENCE_FEED_WIDTH, _REFERENCE_FEED_HEIGHT)]  # the current single arm feed

    v = check_budget(feeds, card_mib=args.card_mib, resident_mib=args.resident_mib,
                     margin_mib=args.margin_mib)
    print('GPU_BUDGET fits=%s feeds=%d estimated=%.0f MiB free_after=%.0f MiB (margin=%.0f) detail=%s'
          % (v.fits, v.detail['num_feeds'], v.estimated_mib, v.free_after_mib, v.margin_mib, v.detail))
    return 0 if v.fits else 1


if __name__ == '__main__':
    import sys
    sys.exit(_main(sys.argv[1:]))
