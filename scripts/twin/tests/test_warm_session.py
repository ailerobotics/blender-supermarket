"""Warm-session reset decision, sequence, fallback + consecutive-run control flow (CPU; no GPU).

The warm-session path keeps one live Isaac session across consecutive runs to skip the fixed
~215 s cold boot (measured; data/sim-warm-session-reset-v1/boot-vs-mission.txt). These exercise the
pure safety core (twin.restock.warm_session) -- the conservative reuse-vs-cold-boot decision and its
reasons, the fail-closed reset sequence, the combined plan+run gate -- plus the launcher's
consecutive-run control flow and the offline dry-run harness. No renderer/physics/ROS is involved.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import warm_session as ws  # noqa: E402


def _clean_stage():
    prims = frozenset({'/Supermarket', '/Rollwagen', '/mobile_robot', '/World/handling_unit'})
    return ws.StageState(present_prims=prims, expected_prims=prims)


# --------------------------------------------------------------------------- decision

def test_reuse_when_every_check_passes():
    d = ws.decide_reset({'nav_mode': 'sensor'}, {'nav_mode': 'sensor'},
                        stage_state=_clean_stage(), vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert d.reuse and not d.cold_boot
    assert d.reason == 'warm reuse: all reset checks passed'
    assert all(c.ok for c in d.checks)


def test_host_key_change_forces_cold_boot():
    # cart_cage_place is a HOST key: the live host stack booted with the old value cannot re-apply it.
    d = ws.decide_reset({'cart_cage_place': True}, {'cart_cage_place': False},
                        stage_state=_clean_stage(), vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert d.cold_boot
    assert 'config_host_stable' in d.reason and 'cart_cage_place' in d.reason


def test_boot_fixed_render_change_forces_cold_boot():
    d = ws.decide_reset({'render_width': 960}, {'render_width': 640},
                        stage_state=_clean_stage(), vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert d.cold_boot and 'render_width' in d.reason


def test_runtime_key_change_is_allowed():
    # stage_timeout_s is a RUNTIME key re-read per mission; changing it does not need a relaunch.
    changed = ws.host_level_changes({'stage_timeout_s': 120}, {'stage_timeout_s': 240})
    assert changed == []
    d = ws.decide_reset({'stage_timeout_s': 120}, {'stage_timeout_s': 240},
                        stage_state=_clean_stage(), vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert d.reuse


def test_unknown_changed_key_is_conservatively_a_cold_boot():
    # A key not registered in KEY_CONSUMER (and not a launcher-ignore key): we cannot prove it is
    # RUNTIME-only, so a change forces a cold boot rather than a silent reuse.
    changed = ws.host_level_changes({'mystery_key': 1}, {'mystery_key': 2})
    assert changed == ['mystery_key']


def test_launcher_bookkeeping_keys_are_ignored():
    # output / launch env / stage bench keys differ per run and are not simulator inputs.
    changed = ws.host_level_changes(
        {'output': '/a', '_launch_environment': {'X': '1'}, 'start_at': 'couple'},
        {'output': '/b', '_launch_environment': {'X': '2'}, 'start_at': 'transfer'})
    assert changed == []


def test_dirty_stage_forces_cold_boot():
    s = ws.StageState(present_prims={'x'}, expected_prims={'x'}, dirty=True, note='physx faulted')
    d = ws.decide_reset({}, {}, stage_state=s, vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert d.cold_boot and 'stage_clean' in d.reason and 'physx faulted' in d.reason


def test_leaked_and_missing_prims_force_cold_boot():
    s = ws.StageState(present_prims={'/Supermarket', '/ghost'}, expected_prims={'/Supermarket', '/Rollwagen'})
    assert s.leaked() == ('/ghost',)
    assert s.missing() == ('/Rollwagen',)
    d = ws.decide_reset({}, {}, stage_state=s, vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert d.cold_boot and 'leaked prims: /ghost' in d.reason and 'missing prims: /Rollwagen' in d.reason


def test_missing_stage_census_cannot_prove_clean():
    d = ws.decide_reset({}, {}, stage_state=None, vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert d.cold_boot and 'no stage census' in d.reason


def test_vram_over_tolerance_forces_cold_boot():
    d = ws.decide_reset({}, {}, stage_state=_clean_stage(),
                        vram_used_mib=8900, post_boot_baseline_mib=8200, vram_tol_mib=400)
    assert d.cold_boot and 'vram_returned' in d.reason and '+700' in d.reason


def test_vram_check_skipped_when_no_reading():
    d = ws.decide_reset({}, {}, stage_state=_clean_stage(),
                        vram_used_mib=None, post_boot_baseline_mib=None)
    vram = [c for c in d.checks if c.name == 'vram_returned'][0]
    assert vram.ok and 'check skipped' in vram.detail
    assert d.reuse


def test_explicit_doubt_forces_cold_boot():
    d = ws.decide_reset({}, {}, stage_state=_clean_stage(), vram_used_mib=8300,
                        post_boot_baseline_mib=8200, extra_doubts=['operator flagged recommission'])
    assert d.cold_boot and 'operator flagged recommission' in d.reason


# --------------------------------------------------------------------------- sequence

def test_sequence_runs_all_steps_in_order():
    order = []
    steps = [ws.SequenceStep(n, (lambda n=n: order.append(n))) for n in ('a', 'b', 'c')]
    res = ws.run_reset_sequence(steps, clock=iter([0.0, 2.0]).__next__)
    assert res.ok and order == ['a', 'b', 'c']
    assert res.completed == ('a', 'b', 'c') and res.failed_step is None
    assert res.wall_seconds == 2.0


def test_sequence_stops_at_first_refusal_and_names_it():
    order = []

    def boom():
        raise ValueError('baseline age exceeds lifetime on the sim clock')
    steps = [ws.SequenceStep('reset', lambda: order.append('reset')),
             ws.SequenceStep('preflight', boom),
             ws.SequenceStep('never', lambda: order.append('never'))]
    res = ws.run_reset_sequence(steps)
    assert not res.ok and res.failed_step == 'preflight'
    assert 'never' not in order
    assert res.completed == ('reset',)
    assert 'baseline age exceeds lifetime' in res.reason and "'preflight'" in res.reason


# --------------------------------------------------------------------------- plan_and_run_reset

def test_plan_cold_decision_does_not_touch_the_stage():
    touched = []
    steps = [ws.SequenceStep('reset', lambda: touched.append('reset'))]
    out = ws.plan_and_run_reset(
        {'cart_cage_place': 1}, {'cart_cage_place': 2}, steps,
        stage_state=_clean_stage(), vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert out.cold_boot and touched == []           # the sequence never ran
    assert out.sequence is None
    assert 'cart_cage_place' in out.reason


def test_plan_reuse_runs_sequence_and_reuses_on_success():
    ran = []
    steps = [ws.SequenceStep('reset', lambda: ran.append('reset')),
             ws.SequenceStep('preflight', lambda: ran.append('preflight'))]
    out = ws.plan_and_run_reset(
        {}, {}, steps, stage_state=_clean_stage(), vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert out.reused and ran == ['reset', 'preflight']
    rec = out.record()
    assert rec['reused'] and rec['sequence']['completed'] == ['reset', 'preflight']


def test_plan_reuse_then_sequence_refusal_is_not_reused():
    def boom():
        raise RuntimeError('no config.warm_reset provider wired')
    steps = [ws.SequenceStep('reset', boom)]
    out = ws.plan_and_run_reset(
        {}, {}, steps, stage_state=_clean_stage(), vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert out.cold_boot and out.sequence is not None and not out.sequence.ok
    assert 'no config.warm_reset provider wired' in out.reason


# --------------------------------------------------- consecutive-run launcher control flow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
from run_mobile_restock import run_consecutive_warm, warm_run_count  # noqa: E402


# --------------------------------------------------- --warm-attempts / --consecutive-runs routing

def test_warm_attempts_runs_consecutive_only_with_a_provider():
    # sim-warm-reset-provider-v1: --warm-attempts 3 with a provider wired => 3 consecutive missions;
    # without a provider it floors to 1 (the legacy run_warm_session retry path keeps the old meaning).
    assert warm_run_count(warm_attempts=3, consecutive_runs=1, has_provider=True) == 3
    assert warm_run_count(warm_attempts=3, consecutive_runs=1, has_provider=False) == 1


def test_consecutive_runs_flag_is_independent_of_the_provider():
    assert warm_run_count(warm_attempts=1, consecutive_runs=2, has_provider=False) == 2
    assert warm_run_count(warm_attempts=1, consecutive_runs=2, has_provider=True) == 2


def test_routing_takes_the_larger_of_the_two_and_floors_to_one():
    assert warm_run_count(warm_attempts=5, consecutive_runs=2, has_provider=True) == 5
    assert warm_run_count(warm_attempts=2, consecutive_runs=4, has_provider=True) == 4
    for n in (0, -3, None):
        assert warm_run_count(warm_attempts=n, consecutive_runs=n, has_provider=True) == 1


def _reuse_outcome():
    d = ws.ResetDecision(reuse=True, reason='ok', checks=())
    return ws.WarmOutcome(reused=True, reason='ok', decision=d,
                          sequence=ws.SequenceResult(ok=True, reason='ok', completed=('reset',)))


def _cold_outcome(reason):
    d = ws.ResetDecision(reuse=False, reason=reason, checks=())
    return ws.WarmOutcome(reused=False, reason=reason, decision=d)


def _runner(states):
    seq = iter(states)

    def run_once(index=1):
        state = next(seq)
        return object(), {'state': state, 'reason': None if state == 'complete' else f'{state} fail'}
    return run_once


def test_single_run_is_byte_identical_and_never_consults_between():
    calls = []
    _, summary, records = run_consecutive_warm(
        _runner(['complete']), runs=1, between=lambda *a: calls.append(a) or _reuse_outcome())
    assert summary['state'] == 'complete'
    assert calls == []                       # between never consulted
    assert 'warm_runs' not in summary        # no consecutive bookkeeping on a single run
    assert [r['run'] for r in records] == [1]


def test_runs_continue_regardless_of_outcome():
    # Unlike run_warm_session, a COMPLETE run does not stop the loop -- both runs execute.
    indices = []

    def run_once(index=1):
        indices.append(index)
        return object(), {'state': 'complete', 'reason': None}
    written = {}
    _, summary, records = run_consecutive_warm(
        run_once, runs=2, between=lambda *a: _reuse_outcome(),
        write_records=lambda recs: written.update(n=len(recs)))
    assert indices == [1, 2]
    assert len(summary['warm_runs']) == 2 and written['n'] == 2
    assert records[0]['warm']['reused'] is True   # the gate record is attached between runs


def test_cold_boot_fallback_stops_the_warm_loop_with_a_reason():
    indices = []

    def run_once(index=1):
        indices.append(index)
        return object(), {'state': 'complete', 'reason': None}
    _, summary, records = run_consecutive_warm(
        run_once, runs=3,
        between=lambda prev, i: _cold_outcome('cold boot: VRAM not returned (+900, tol 400)'))
    assert indices == [1]                                  # run 2 never started on this session
    assert summary['warm_fallback'].startswith('cold boot:')
    assert records[-1]['warm']['reused'] is False


def test_zero_or_negative_runs_floors_to_one():
    for n in (0, -2, None):
        _, summary, records = run_consecutive_warm(
            _runner(['failed']), runs=n, between=lambda *a: _reuse_outcome())
        assert summary['state'] == 'failed' and 'warm_runs' not in summary and len(records) == 1


# ------------------------------------------------------------------- dry-run harness

import check_warm_session_reset as dry  # noqa: E402


FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'warm_session_state.json'


def test_dry_run_fixture_reuses_clean_session():
    assert dry.main(['--state', str(FIXTURE), '--json']) == 0


def test_dry_run_injected_preflight_fault_falls_back_to_cold():
    assert dry.main(['--state', str(FIXTURE), '--inject-fault', 'preflight_verify',
                     '--expect', 'cold', '--json']) == 0


def test_dry_run_vram_leak_falls_back_to_cold():
    assert dry.main(['--state', str(FIXTURE), '--leak-vram', '900',
                     '--expect', 'cold', '--json']) == 0


def test_dry_run_build_from_state_matches_fixture_numbers():
    import json
    state = json.loads(FIXTURE.read_text())
    stage_state, used_now, baseline, tol, steps = dry.build_from_state(state)
    assert used_now == 8360.0 and baseline == 8200.0 and tol == 400.0
    assert [s.name for s in steps] == state['reset_steps']
    assert stage_state.leaked() == () and stage_state.missing() == ()
