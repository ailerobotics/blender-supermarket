#!/usr/bin/env python3
"""Out-of-process finalizer for a NATIVELY-crashed restock mission (sim-robust-vram-headroom-v1).

The Isaac restock mission runs the simulator in-process; a native fault (coldwin13: SIGSEGV, exit
139) kills the python process before the ``finally`` that writes ``mission.json`` can run. This tool
runs from OUTSIDE that process (the shell parent / the orchestrator) AFTER the mission exits: when the
process died by a signal (or wall-cap timeout) and ``mission.json`` is absent, it reconstructs a
``state="crashed"`` verdict from the durable run artifacts (``mission-progress.jsonl``, else
``nav-trace.jsonl`` + ``bt_events.jsonl``) plus the last sampled VRAM headroom (``vram-headroom.json``),
so the record-first teardown and the evidence tooling still have a verdict to grade.

If ``mission.json`` already exists (a clean success or failure), this is a NO-OP -- the normal path is
byte-identical. Pure stdlib (no Isaac/ROS), so it is safe to run in any plain ``python3``.

Usage::

    finalize_crashed_run.py --run-dir <dir> --exit-code <rc>      # rc from the mission process
    finalize_crashed_run.py --run-dir <dir> --signal 11           # or an explicit signal number

``--exit-code`` follows the shell convention: a process killed by signal N exits 128+N (139 = SIGSEGV),
and GNU ``timeout`` exits 124 on its own TERM. A clean exit (0/1) with ``mission.json`` present is a
no-op; a clean exit with NO mission.json still records a crashed verdict (the process vanished without
writing one).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin.restock import crash_record  # noqa: E402


def signal_from_exit_code(rc):
    """Map a process exit code to a crash signal number, or None when it is not a signal death.

    Shell convention: killed-by-signal N => 128+N. GNU timeout => 124 (its own TERM; treated as a
    wall-cap timeout, signal None). A clean 0/1 => None."""
    if rc is None:
        return None
    rc = int(rc)
    if rc > 128:
        return rc - 128
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True, type=Path, help='the mission run directory')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--exit-code', type=int, default=None,
                   help='the mission process exit code (128+signal for a signal death; 124 timeout)')
    g.add_argument('--signal', type=int, default=None, help='an explicit crash signal number')
    args = ap.parse_args(argv)

    if not args.run_dir.is_dir():
        print(f'finalize_crashed_run: {args.run_dir} is not a directory', file=sys.stderr)
        return 2

    signal = args.signal if args.signal is not None else signal_from_exit_code(args.exit_code)
    written = crash_record.write_if_missing(args.run_dir, signal, log=lambda m: print(m, flush=True))
    if written is None:
        return 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
