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


def feeds_from_config(config):
    """Extract the enabled masked-depth feeds' ``(width, height)`` from a mission-runtime config.

    Reads ``masked_depth`` (the arm feed, camera_0) and ``mast_depth`` (camera_1, opt-in). A feed's
    resolution comes from its ``resolution: [w, h]`` when set, else the reference arm resolution.
    A feed is counted only when its block opts in (``enabled``). Pure dict reader (no Isaac)."""
    from .masked_depth_economy import from_block as _economy_from_block, scaled_resolution
    feeds = []
    for key in ('masked_depth', 'mast_depth'):
        block = config.get(key) or {}
        if not isinstance(block, dict) or not block.get('enabled'):
            continue
        res = block.get('resolution')
        if res and isinstance(res, (list, tuple)) and len(res) == 2:
            w, h = int(res[0]), int(res[1])
        else:
            w, h = _REFERENCE_FEED_WIDTH, _REFERENCE_FEED_HEIGHT
        # sim-robust-arm-feed-economy-v1: a feed's economy resolution_scale shrinks the per-pixel VRAM
        # term (fm373: resolution is a small VRAM lever, but reflect it so the gate is self-consistent).
        scale = _economy_from_block(block.get('economy')).resolution_scale
        feeds.append(scaled_resolution(w, h, scale) if scale != 1.0 else (w, h))
    return feeds


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
    args = ap.parse_args(argv)

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
