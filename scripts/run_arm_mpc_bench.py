#!/usr/bin/env python3
"""Guarded GPU runner for the cuRobo MPC follow-and-avoid bench (RUNBOOK §18).

Reuses the existing ``fm-plan-mpc-curobo-v1`` container (cuRobo V2, Apache-2.0 --
scout build spec (a): do NOT rebuild the arm stack) and runs
``scripts/twin/tools/arm_mpc_run.py`` inside it against the store world + the
moving-person proxy, driving the mission arm's own ``shelfbot_curobo.yml``.

GPU etiquette (the mission lanes hold priority on /tmp/aile-gpu.lock):
  * flock on /tmp/aile-gpu.lock. Default NON-BLOCKING -- if the mission lane holds it,
    this exits 3 immediately (never barges). With --wait-lock-s N it instead BLOCKS up
    to N seconds for the lock to free (the `flock -w N` shape the integration
    orchestrator uses), so a background runner can wait politely for a free window
    rather than idling a human at the prompt. On timeout it exits 3.
  * Hard-refuses beside the live Blender GUI / MCP socket (the AGENTS.md rule).
  * A modest VRAM-headroom check (MPC needs ~0.22 GB; default require 1500 MiB free).
    There is NO --force; the VRAM guard is never overridden.
  * Wall-clock cap (default 600 s; refuses > 900 s = the 15-min budget) kills a wedged run.

No Isaac Sim boots: cuRobo MPC is pure cuRobo + torch, so this is the cheapest GPU
consumer in the fleet and safe to slot into a short free window.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # blender-supermarket worktree
LOCK = '/tmp/aile-gpu.lock'
IMAGE = 'fm-plan-mpc-curobo-v1:latest'
HOME = os.path.expanduser('~')
DEFAULT_MPC_DIR = os.path.join(HOME, 'ur10/mpc_curobo')
DEFAULT_SHELFBOT = os.path.join(HOME, 'workspaces/shelfbot_newgripper')


def _gpu_free_mib() -> float:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=10)
    return float(out.stdout.strip().splitlines()[0])


def _blender_live() -> bool:
    """True if a Blender process or the MCP socket (127.0.0.1:9876) is present."""
    for comm in Path('/proc').glob('[0-9]*/comm'):
        try:
            if 'blender' in comm.read_text().lower():
                return True
        except (FileNotFoundError, PermissionError):
            pass
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(('127.0.0.1', 9876)) == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario', default='all',
                    help="follow|person|blocked|unreachable|all (default: all)")
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--out', default=str(REPO / 'export' / 'arm_mpc_bench' / 'metrics.json'),
                    help='metrics JSON (host path; written via the /out mount)')
    ap.add_argument('--mpc-dir', default=DEFAULT_MPC_DIR,
                    help='plan-mpc-curobo-v1 mpc_curobo dir (gate + window guard reuse)')
    ap.add_argument('--shelfbot', default=DEFAULT_SHELFBOT,
                    help='shelfbot_newgripper repo (declares package shelfbot + the cuRobo yml)')
    ap.add_argument('--wait-lock-s', type=int, default=0,
                    help='block up to this many seconds for /tmp/aile-gpu.lock to free '
                         '(0 = non-blocking, the default; e.g. 3600 for a polite background wait)')
    ap.add_argument('--min-free-mb', type=int, default=1500,
                    help='refuse unless at least this many MiB VRAM are free (default: %(default)s)')
    ap.add_argument('--timeout-s', type=int, default=600,
                    help='wall-clock cap (<= 900 s = 15-min budget; default: %(default)s)')
    ap.add_argument('--image', default=IMAGE)
    args = ap.parse_args(argv)

    if args.timeout_s > 900:
        print('refuse: timeout exceeds the 15-min GPU budget (900 s)', file=sys.stderr)
        return 2
    mpc_dir = Path(args.mpc_dir).resolve()
    shelfbot = Path(args.shelfbot).resolve()
    for p, what in ((mpc_dir, 'mpc_curobo dir'), (shelfbot, 'shelfbot repo')):
        if not p.is_dir():
            print(f'refuse: {what} not found at {p}', file=sys.stderr)
            return 2
    yml = shelfbot / 'description' / 'shelfbot_curobo.yml'
    urdf = shelfbot / 'description' / 'shelfbot_curobo_isaac.urdf'
    if not yml.exists():
        print(f'refuse: {yml} not found', file=sys.stderr)
        return 2

    out_host = Path(args.out).resolve()
    out_host.parent.mkdir(parents=True, exist_ok=True)

    # --- lock: default non-blocking (never barge); --wait-lock-s blocks politely ----- #
    lock_fh = open(LOCK, 'a')
    if args.wait_lock_s > 0:
        # Bounded blocking acquire == `flock -w N`: block up to N s, then give up.
        def _timeout(signum, frame):
            raise TimeoutError
        old = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(args.wait_lock_s)
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)   # blocks until free or the alarm fires
        except (TimeoutError, InterruptedError):
            print(f'GPU_LOCK_TIMEOUT: /tmp/aile-gpu.lock still held after {args.wait_lock_s}s',
                  file=sys.stderr)
            return 3
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('GPU_LOCK_HELD: /tmp/aile-gpu.lock is held by another lane; re-run when free',
                  file=sys.stderr)
            return 3

    try:
        if _blender_live():
            print('refuse: protected Blender GUI / MCP socket is live (hard rule)', file=sys.stderr)
            return 2
        free = _gpu_free_mib()
        if free < args.min_free_mb:
            print(f'refuse: only {free:.0f} MiB VRAM free (< {args.min_free_mb}); '
                  f'the mission lanes have priority -- re-run in a freer window', file=sys.stderr)
            return 2

        cmd = [
            'docker', 'run', '--rm', '--gpus', 'all',
            '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'HOME=/tmp',
            '-v', f'{REPO}:/store:ro',
            '-v', f'{mpc_dir}:/mpc:ro',
            '-v', f'{shelfbot}:/ros_ws/src/shelfbot:ro',
            '-v', f'{out_host.parent}:/out:rw',
            args.image,
            'python3', '/store/scripts/twin/tools/arm_mpc_run.py',
            '--scenario', args.scenario, '--steps', str(args.steps),
            '--mpc-dir', '/mpc',
            '--robot', '/ros_ws/src/shelfbot/description/shelfbot_curobo.yml',
            '--limits', '/ros_ws/src/shelfbot/description/shelfbot_curobo_isaac.urdf',
            '--out', f'/out/{out_host.name}',
        ]
        print('free VRAM: %.0f MiB; running:\n  %s' % (free, ' '.join(cmd)), file=sys.stderr)
        try:
            proc = subprocess.run(cmd, timeout=args.timeout_s)
        except subprocess.TimeoutExpired:
            print(f'FAILED: bench exceeded {args.timeout_s}s wall cap (killed)', file=sys.stderr)
            return 4
        return proc.returncode
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


if __name__ == '__main__':
    raise SystemExit(main())
