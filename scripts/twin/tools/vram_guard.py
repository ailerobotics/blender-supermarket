#!/usr/bin/env python3
"""fm360 / sim-robust-vram-headroom-default-on-v1: cheap GPU VRAM guard so a run is never lost SILENTLY
to over-subscription again -- with an HONEST, DERIVED minimum.

run-285 segfaulted inside the Isaac render loop at ~13.14 GB VRAM on a 12 GiB card once the run got
long enough (full collect), with whispr's ~2.2 GB always-on competing. This guard, called pre-launch
by the orchestrator, refuses to boot when the card does not have enough FREE VRAM for the mission's
footprint -- a one-second CPU check, not a 20-minute wasted boot.

The minimum is **not a magic constant**. It is DERIVED from the measured mission footprint
(``twin.restock.gpu_budget``, anchored to the transfer2 measurement 2026-10-22: mission ~= 9630 MiB
excluding resident whispr) MINUS the default-on footprint trims (the fm380 stage-scoped sensing
teardown frees ~948 MiB of shelf-camera render product proactively at collect->couple), PLUS a stated
safety margin. The old fixed 9500 was both DISHONEST (unrelated to any measured peak) and self-
defeating: it refused at free=9473 beside a 2.44 GB whispr, yet 9500 was itself below the real ~9630
mission footprint, so it was simultaneously too high to boot and too low to be safe. Deriving the
minimum from the (lowered) measured footprint lets a run boot beside 2.44 GB whispr -- the trims make
the mission fit the headroom that is actually free.

The refusal line names WHAT HOLDS the memory (the resident GPU consumers parsed from ``nvidia-smi``,
e.g. whispr) and how the minimum was derived, so an operator sees the cause, not just a number.
whispr is a protected live service and is NEVER touched (run-285 crash history).

Reads `nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv,noheader,nounits`
(and the full process table to name holders). Pure parse/decision/derivation helpers are unit-tested;
main() runs nvidia-smi and exits nonzero on refusal.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import gpu_budget  # noqa: E402

# Measured mission footprint (MiB, EXCLUDING resident whispr), transfer2 anchor via gpu_budget:
# estimate_mission_mib([reference arm feed]) ~= 9630. This is the free VRAM the un-trimmed mission
# allocates at its peak.
_REFERENCE_FEED = (gpu_budget._REFERENCE_FEED_WIDTH, gpu_budget._REFERENCE_FEED_HEIGHT)
# fm380 (measured): the stage-scoped sensing teardown returns the ~0.95 GB shelf-camera render product
# proactively at the collect->couple boundary -- before the long couple->transfer->return creep span --
# so the creep peak lands that much lower. DEFAULT ON (sim-robust-vram-headroom-default-on-v1).
STAGE_SCOPED_TEARDOWN_SAVING_MIB = 948
# Stated headroom kept free for RTX/Vulkan allocator churn + fragmentation + whispr's own VRAM creep
# (it grew 2.18 -> 2.44 GB in a day). Not a mission allocation; a margin against the ceiling.
DEFAULT_MARGIN_MIB = 400

# Back-compat module constant, now DERIVED (reference single-feed mission, default-on trims, margin)
# rather than a hand-set 9500. ~= 9630 - 948 + 400 = 9082.
DEFAULT_MIN_FREE_MIB = int(round(
    gpu_budget.estimate_mission_mib([_REFERENCE_FEED])[0]
    - STAGE_SCOPED_TEARDOWN_SAVING_MIB + DEFAULT_MARGIN_MIB))


def _stage_scoped_teardown_enabled(config, env):
    """Minimal mirror of ``sensing_sources.stage_scoped_teardown_enabled`` so the guard stays decoupled
    from the camera stack. ``env`` is ``RESTOCK_STAGE_SCOPED_SENSING`` (or None). Kept in parity by
    ``test_vram_guard.py``. DEFAULT ON; env falsey => OFF; else ``config['sensing_teardown']`` opt-out."""
    if env is not None:
        return env.strip().lower() not in ('', '0', 'false', 'off', 'no')
    block = config.get('sensing_teardown') if isinstance(config, dict) else None
    if block is None:
        return True
    if block is False or block is True:
        return bool(block)
    if isinstance(block, dict):
        return bool(block.get('stage_scoped', True))
    return True


def required_min_free_mib(config=None, *, margin_mib=DEFAULT_MARGIN_MIB,
                          teardown_env=None):
    """Derive the minimum FREE VRAM (MiB) to boot, and a breakdown dict explaining it.

    = the mission's own peak footprint (gpu_budget estimate for the configured feeds, EXCLUDING
    resident services) - the default-on footprint trims + ``margin_mib``. A config with more/larger
    feeds (e.g. the opt-in mast camera) raises it; opting the teardown out raises it (the mission then
    genuinely needs the full footprint free). Pure (no subprocess)."""
    feeds = gpu_budget.feeds_from_config(config) if config else [_REFERENCE_FEED]
    mission, per_feed = gpu_budget.estimate_mission_mib(feeds)
    teardown_on = _stage_scoped_teardown_enabled(config or {}, teardown_env)
    trim = STAGE_SCOPED_TEARDOWN_SAVING_MIB if teardown_on else 0
    peak = mission - trim
    min_free = int(round(peak + float(margin_mib)))
    return min_free, {
        'mission_peak_mib': round(mission, 1),
        'per_feed_mib': [round(f, 1) for f in per_feed],
        'stage_scoped_teardown': teardown_on,
        'teardown_saving_mib': trim,
        'effective_peak_mib': round(peak, 1),
        'margin_mib': float(margin_mib),
        'min_free_mib': min_free,
    }


def parse_smi(csv_text):
    """Parse the first GPU row of `nvidia-smi --query-gpu=memory.free,memory.used,memory.total
    --format=csv,noheader,nounits` into (free_mib, used_mib, total_mib) ints. Raises ValueError on
    malformed input."""
    for line in csv_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            raise ValueError(f'unexpected nvidia-smi row: {line!r}')
        try:
            return int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
        except ValueError:
            raise ValueError(f'non-numeric nvidia-smi memory fields: {line!r}')
    raise ValueError('no GPU row in nvidia-smi output')


def holders_text(smi_full_text, *, top=3):
    """Name the biggest resident GPU consumers (e.g. whispr) from full ``nvidia-smi`` text, so the
    refusal says WHAT holds the memory. Reuses ``gpu_guard.parse_processes``; '' when none/unparsable."""
    try:
        from twin.gpu_guard import parse_processes
        procs = sorted(parse_processes(smi_full_text), key=lambda p: p.mem_mib, reverse=True)
    except Exception:
        return ''
    if not procs:
        return ''
    return ', '.join(f'{p.name} (pid {p.pid}, {p.mem_mib} MiB)' for p in procs[:top])


def decide(free_mib, min_free_mib=DEFAULT_MIN_FREE_MIB, *, breakdown=None, holders=''):
    """Return (ok, message). ok=False => refuse to launch (not enough free VRAM). The message states
    the DERIVED minimum, its breakdown, and the resident holders (what to free)."""
    basis = ''
    if breakdown:
        basis = (f' [min = mission peak {breakdown["mission_peak_mib"]} MiB'
                 f' - teardown {breakdown["teardown_saving_mib"]} MiB'
                 f' + margin {breakdown["margin_mib"]:.0f} MiB]')
    if free_mib >= min_free_mib:
        return True, f'VRAM_GUARD OK: free={free_mib} MiB >= min {min_free_mib} MiB{basis}'
    held = f' Holding VRAM now: {holders}.' if holders else ''
    return False, (
        f'VRAM_GUARD REFUSE: free={free_mib} MiB < min {min_free_mib} MiB{basis} -- a full run would '
        f'over-subscribe the card and crash the render loop (run-285).{held} Free non-protected VRAM '
        f'first (NEVER whispr); or lower the mission footprint (stage-scoped teardown / vram_headroom '
        f'are default-on) before re-launching.')


def query_smi():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free,memory.used,memory.total',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30, check=True)
    return out.stdout


def query_smi_full():
    """Full nvidia-smi text (process table) for holder naming; '' if unavailable (never fatal)."""
    try:
        out = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30, check=True)
        return out.stdout
    except Exception:
        return ''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', help='mission-runtime config JSON; derives the minimum from its feeds '
                                     '+ the default-on trims')
    ap.add_argument('--margin-mib', type=float, default=DEFAULT_MARGIN_MIB,
                    help='stated safety headroom kept free above the mission peak')
    ap.add_argument('--min-free-mib', type=int, default=None,
                    help='override the derived minimum with an explicit value (discouraged)')
    ap.add_argument('--sample', action='store_true',
                    help='just print used/free/total (for periodic logging), always exit 0')
    args = ap.parse_args(argv)
    try:
        free, used, total = parse_smi(query_smi())
    except Exception as exc:  # nvidia-smi missing/odd -> do not block the run on the guard itself
        print(f'VRAM_GUARD UNAVAILABLE: {exc}')
        return 0
    if args.sample:
        print(f'VRAM_SAMPLE used={used} free={free} total={total} MiB')
        return 0
    config = json.loads(Path(args.config).read_text()) if args.config else None
    env = os.environ.get('RESTOCK_STAGE_SCOPED_SENSING')
    derived, breakdown = required_min_free_mib(config, margin_mib=args.margin_mib, teardown_env=env)
    min_free = args.min_free_mib if args.min_free_mib is not None else derived
    holders = holders_text(query_smi_full())
    ok, msg = decide(free, min_free, breakdown=None if args.min_free_mib is not None else breakdown,
                     holders=holders)
    print(msg + f' (used={used}, total={total})', file=sys.stderr if not ok else sys.stdout)
    return 0 if ok else 6


if __name__ == '__main__':
    raise SystemExit(main())
