#!/usr/bin/env python3
"""Dry-run the warm-session reset decision + sequence against a recorded session state (CPU, no GPU).

Proves the warm-session reset path offline: it loads a recorded between-runs session state (the
prev/next run config, the live stage prim census, and the post-boot vs live VRAM readings), runs the
exact production decision + sequence (``twin.restock.warm_session``), and prints the verdict -- reuse
the live session for the next run, or fall back to a cold boot and WHY. No Isaac, no ROS, no GPU.

The reset STEPS are dry (no stage is actually re-seated); ``--inject-fault <step>`` makes a named step
RAISE so the dry-run can also demonstrate the fail-closed cold-boot fallback (e.g. a refusing preflight
or an absent warm_reset provider). Exit code: 0 when the decision reuses the session, 1 on a cold-boot
fallback -- flip the expectation with ``--expect cold``.

Usage:
  scripts/check_warm_session_reset.py --state scripts/twin/tests/fixtures/warm_session_state.json
  scripts/check_warm_session_reset.py --state <fixture> --inject-fault preflight_verify --expect cold
  scripts/check_warm_session_reset.py --state <fixture> --leak-vram 900     # simulate a VRAM leak
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin.restock import warm_session as ws  # noqa: E402


def load_state(path: Path) -> dict:
    return json.loads(path.read_text())


def build_from_state(state: dict, *, inject_fault=None, leak_vram=0.0):
    """Turn a recorded state dict into the decision inputs + dry reset steps."""
    stage = state.get('stage') or {}
    stage_state = ws.StageState(
        present_prims=frozenset(stage.get('present_prims', [])),
        expected_prims=frozenset(stage.get('expected_prims', [])),
        dirty=bool(stage.get('dirty', False)),
        note=stage.get('note', ''))
    vram = state.get('vram') or {}
    used_now = vram.get('used_mib_now')
    if used_now is not None:
        used_now = float(used_now) + float(leak_vram)
    baseline = vram.get('post_boot_baseline_mib')
    tol = float(vram.get('tol_mib', ws.DEFAULT_VRAM_TOL_MIB))

    def _make_step(name):
        def _fn():
            if inject_fault and name == inject_fault:
                raise RuntimeError(f'injected fault at {name}')
        return ws.SequenceStep(name, _fn)

    steps = [_make_step(n) for n in state.get('reset_steps', ['reset_stage_and_reseed', 'preflight_verify'])]
    return stage_state, used_now, baseline, tol, steps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--state', type=Path, required=True,
                        help='recorded warm-session state JSON (see fixtures/warm_session_state.json)')
    parser.add_argument('--inject-fault', default=None,
                        help='make the named reset step RAISE (demonstrate the cold-boot fallback)')
    parser.add_argument('--leak-vram', type=float, default=0.0,
                        help='MiB to add to the live reading (simulate GPU memory not returning)')
    parser.add_argument('--expect', choices=('reuse', 'cold'), default='reuse',
                        help='expected verdict for the exit code (default: reuse)')
    parser.add_argument('--json', action='store_true', help='print the machine-readable record only')
    args = parser.parse_args(argv)

    state = load_state(args.state)
    stage_state, used_now, baseline, tol, steps = build_from_state(
        state, inject_fault=args.inject_fault, leak_vram=args.leak_vram)

    logs = []
    outcome = ws.plan_and_run_reset(
        state.get('prev_config', {}), state.get('next_config', {}), steps,
        stage_state=stage_state, vram_used_mib=used_now,
        post_boot_baseline_mib=baseline, vram_tol_mib=tol,
        on_log=logs.append)

    record = outcome.record()
    if args.json:
        print(json.dumps(record, indent=2))
    else:
        print(f'# Warm-session reset dry-run against {args.state}')
        print(f'#   source_run={state.get("source_run", "?")}  '
              f'inject_fault={args.inject_fault}  leak_vram={args.leak_vram:g} MiB')
        print()
        for line in logs:
            print('  ' + line)
        print()
        print('decision checks:')
        for c in record['decision']['checks']:
            print(f'  [{"PASS" if c["ok"] else "FAIL"}] {c["name"]}'
                  + (f' -- {c["detail"]}' if c['detail'] else ''))
        print()
        verdict = 'REUSE live session' if outcome.reused else 'COLD BOOT fallback'
        print(f'VERDICT: {verdict}')
        print(f'  reason: {outcome.reason}')
        if outcome.sequence is not None:
            print(f'  reset sequence: {"ok" if outcome.sequence.ok else "refused"} '
                  f'(steps {list(outcome.sequence.completed)}, {outcome.sequence.wall_seconds}s)')

    reused_expected = (args.expect == 'reuse')
    return 0 if outcome.reused == reused_expected else 1


if __name__ == '__main__':
    raise SystemExit(main())
