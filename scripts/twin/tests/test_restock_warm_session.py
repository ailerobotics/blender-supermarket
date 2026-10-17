"""Warm-session retry control flow + provider wiring (CPU; no Isaac/ROS/GPU).

`run_warm_session` re-runs the mission inside one live Isaac session (saving the per-attempt
re-boot + separate commissioning) and MUST compose with the downstream --start-at shortcut and
never bypass the baseline lifetime/session guard. These exercise the pure control flow with a fake
run_once and a fake reset provider, plus the create() wiring that resolves config['warm_reset'] onto
bundle.warm_reset. No renderer/physics is involved -- the native in-session re-seat is GPU-pending.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_mobile_restock import run_warm_session  # noqa: E402


def _runner(states):
    """A run_once that yields `states` in order; each call persists nothing (test stub)."""
    seq = iter(states)

    def run_once():
        state = next(seq)
        mission = object()
        return mission, {'state': state, 'reason': None if state == 'complete' else f'{state} fail'}
    return run_once


def test_default_single_attempt_is_one_run_byte_identical():
    calls = []
    run_once = _runner(['failed'])
    # warm_attempts 1 (the default): exactly one run_once, no reset consulted, no warm_* keys.
    _, summary = run_warm_session(
        run_once, warm_attempts=1, warm_reset=lambda **k: calls.append(k))
    assert summary['state'] == 'failed'
    assert calls == []                          # the reset provider is never consulted
    assert 'warm_attempts' not in summary and 'warm_note' not in summary


def test_retries_a_failed_attempt_and_stops_on_complete():
    resets = []
    run_once = _runner(['failed', 'failed', 'complete', 'failed'])  # 3rd succeeds; 4th never reached

    def warm_reset(*, attempt, now):
        resets.append((attempt, now))
    written = {}
    _, summary = run_warm_session(run_once, warm_attempts=4, warm_reset=warm_reset,
                                  write_attempts=lambda recs: written.update(n=len(recs), recs=recs))
    assert summary['state'] == 'complete'
    assert resets == [(1, 1), (2, 2)]           # re-seat after attempt 1 and 2, not after the win
    assert written['n'] == 3                    # three attempts recorded (1 fail, 2 fail, 3 win)
    assert [r['state'] for r in summary['warm_attempts']] == ['failed', 'failed', 'complete']


def test_exhausts_the_budget_when_every_attempt_fails():
    resets = []
    run_once = _runner(['failed', 'failed', 'failed'])
    _, summary = run_warm_session(run_once, warm_attempts=3,
                                  warm_reset=lambda **k: resets.append(k))
    assert summary['state'] == 'failed'
    assert len(resets) == 2                      # reset between the 3 attempts, not after the last
    assert len(summary['warm_attempts']) == 3


def test_no_provider_means_a_single_attempt_not_a_silent_retry():
    run_once = _runner(['failed', 'complete'])   # a 2nd would win, but no provider to re-seat
    _, summary = run_warm_session(run_once, warm_attempts=3, warm_reset=None)
    assert summary['state'] == 'failed'          # stopped after attempt 1
    assert 'no config.warm_reset provider' in summary['warm_note']


def test_a_refusing_provider_stops_without_bypassing_the_guard():
    # The provider raises (baseline expired / session unbound / re-seat failed): the loop stops and
    # records the refusal; it must NOT swallow it and re-run anyway.
    run_once = _runner(['failed', 'complete'])

    def refusing(*, attempt, now):
        raise ValueError('baseline age exceeds lifetime on the sim clock')
    _, summary = run_warm_session(run_once, warm_attempts=3, warm_reset=refusing)
    assert summary['state'] == 'failed'
    assert 'warm re-run refused' in summary['warm_stop']
    assert 'baseline age exceeds lifetime' in summary['warm_stop']


def test_zero_or_negative_warm_attempts_floors_to_one_run():
    for n in (0, -5, None):
        run_once = _runner(['failed'])
        _, summary = run_warm_session(run_once, warm_attempts=n, warm_reset=lambda **k: None)
        assert summary['state'] == 'failed'
        assert 'warm_attempts' not in summary


def test_create_wires_config_warm_reset_onto_the_bundle():
    # The create() seam resolves config['warm_reset'] (module:function) and attaches the returned
    # per-attempt callable to bundle.warm_reset; absent config leaves it None. Reuses the existing
    # isaac_bundle create()-wiring fixtures (test-only, not physical evidence).
    import types
    from scripts.twin.restock.isaac_bundle import create
    from scripts.twin.tests.test_isaac_bundle import base_config, MANIFEST, SCENARIO, CreateWiringTests

    runtime = CreateWiringTests()._fixture_runtime({})
    built = {}
    mod = types.ModuleType('warm_reset_probe_mod')

    def build(*, app, assembly, parts, plan, config, output):
        built['called'] = True
        assert 'port' in parts and plan is not None       # provider gets the live scene handles
        return lambda **k: None
    mod.build = build
    sys.modules['warm_reset_probe_mod'] = mod
    try:
        bundle = create(app=object(), manifest=MANIFEST, scenario=SCENARIO,
                        config=base_config(warm_reset='warm_reset_probe_mod:build'),
                        output='/tmp/x', _assemble=lambda **k: object(), _runtime=runtime)
        assert built.get('called') is True
        assert callable(bundle.warm_reset)
        # Without the key, no provider is wired (warm retry stays unavailable).
        bundle2 = create(app=object(), manifest=MANIFEST, scenario=SCENARIO, config=base_config(),
                         output='/tmp/x', _assemble=lambda **k: object(), _runtime=runtime)
        assert bundle2.warm_reset is None
    finally:
        sys.modules.pop('warm_reset_probe_mod', None)
