#!/usr/bin/env python3
"""fm360: cheap GPU VRAM guard so a run is never lost SILENTLY to over-subscription again.

run-285 segfaulted inside the Isaac render loop at ~13.14 GB VRAM on a 12 GiB card once the run got
long enough (full collect), with whispr's ~2.2 GB always-on competing. This guard, called pre-launch
by the orchestrator, refuses to boot when the card does not have enough FREE VRAM for the mission's
render budget -- a one-second CPU check, not a 20-minute wasted boot.

Reads `nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv,noheader,nounits`.
Pure parse/decision helpers are unit-tested; main() runs nvidia-smi and exits nonzero on refusal.
"""
import argparse
import subprocess
import sys

DEFAULT_MIN_FREE_MIB = 9500   # mission render budget at 640x360 fits well under this headroom


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


def decide(free_mib, min_free_mib=DEFAULT_MIN_FREE_MIB):
    """Return (ok, message). ok=False => refuse to launch (not enough free VRAM)."""
    if free_mib >= min_free_mib:
        return True, f'VRAM_GUARD OK: free={free_mib} MiB >= min {min_free_mib} MiB'
    return False, (f'VRAM_GUARD REFUSE: free={free_mib} MiB < min {min_free_mib} MiB -- a full run would '
                   f'over-subscribe the card and crash the render loop (run-285). Free VRAM first.')


def query_smi():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free,memory.used,memory.total',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30, check=True)
    return out.stdout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min-free-mib', type=int, default=DEFAULT_MIN_FREE_MIB)
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
    ok, msg = decide(free, args.min_free_mib)
    print(msg + f' (used={used}, total={total})', file=sys.stderr if not ok else sys.stdout)
    return 0 if ok else 6


if __name__ == '__main__':
    raise SystemExit(main())
