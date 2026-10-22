"""estimate_sim_speed: replay a run's per-tick wall cost under the sim_speed profile (sim-video-speedup-v1)."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import sim_speed as S
from twin.tools import estimate_sim_speed as E

DT = 1 / 60


def _rows(stages):
    """nav-trace rows: per stage (action, ticks, render_every, plain_s, rendered_s)."""
    rows, wall, tick = [{'tick': 0, 'wall': 0.0, 'action': None, 'rendered': False}], 0.0, 0
    for action, ticks, every, plain, rendered in stages:
        for i in range(ticks):
            tick += 1
            hit = bool(every) and i % every == 0
            wall += rendered if hit else plain
            rows.append({'tick': tick, 'wall': round(wall, 6), 'action': action, 'rendered': hit})
    return rows


def test_tick_costs_split_plain_and_rendered_and_skip_gaps():
    rows = [{'tick': 1, 'wall': 0.0, 'action': 'couple'},
            {'tick': 2, 'wall': 0.04, 'action': 'couple', 'rendered': False},
            {'tick': 3, 'wall': 0.74, 'action': 'couple', 'rendered': True},
            {'tick': 5, 'wall': 9.00, 'action': 'couple', 'rendered': False},     # gap: not a tick delta
            {'tick': 6, 'wall': 9.05, 'action': 'park'}]
    costs = E.tick_costs(rows)
    assert costs['couple']['plain'] == pytest.approx([0.04])
    assert costs['couple']['rendered'] == pytest.approx([0.70])
    assert costs['park']['plain'] == pytest.approx([0.05])


def test_nav_stage_repriced_at_the_floor_and_the_walkthrough_stride():
    rows = _rows([('couple', 600, 10, 0.040, 0.700)])                 # recorded: render every 10 ticks
    result = E.estimate(rows, {}, floor_s=0.30)                       # no sim_speed block -> profile max
    couple = result['stages']['couple']
    assert result['profile'] == 'max'
    assert couple['decode_paused'] is True
    assert couple['rendered'] == 60 and couple['est_rendered'] == 30  # 3 Hz sim -> stride 20
    assert couple['est_rendered_ms'] == pytest.approx(300.0)
    saving = (E.PROBE_MS + E.BT_SNAPSHOT_MS) / 1e3
    expected = 570 * (0.040 - saving) + 30 * 0.30
    assert couple['est_wall_s'] == pytest.approx(expected, abs=0.1)
    assert couple['est_ratio'] > couple['ratio']
    assert result['total']['speedup'] == pytest.approx(couple['wall_s'] / couple['est_wall_s'], rel=0.01)


def test_host_coupled_stage_keeps_its_cameras_unless_ground_truth():
    rows = _rows([('transfer', 120, 6, 0.050, 0.700)])
    live = E.estimate(rows, {'sim_speed': {'profile': 'max'}})['stages']['transfer']
    assert live['decode_paused'] is False
    assert live['est_rendered'] == live['rendered']                   # stride untouched off the nav stages
    assert live['est_rendered_ms'] == live['rendered_ms']
    gt = E.estimate(rows, {'sim_speed': {'profile': 'max'}, 'transfer_ground_truth': {'enabled': True}},
                    floor_s=0.30)['stages']['transfer']
    assert gt['decode_paused'] is True and gt['est_rendered_ms'] == pytest.approx(300.0)
    off = E.estimate(rows, {'sim_speed': {'profile': 'off'}})['stages']['transfer']
    assert off['est_wall_s'] == pytest.approx(off['wall_s'], abs=0.1)  # profile off re-prices nothing


def test_measured_reads_the_tick_profiler_lines():
    t = [0.0]
    lines = []
    prof = S.TickProfiler(1.0, clock=lambda: t[0], emit=lines.append)
    sim = 0.0
    for action, cost in (('couple', 0.030), ('transfer', 0.050)):
        for _ in range(121):
            prof.begin()
            t[0] += cost
            prof.mark('step')
            sim += DT
            prof.end(sim, rendered=False, action=action)
        prof.flush()
    summary = E.measured(['2026-10-22T00:00:00Z [1,000ms] ' + line for line in lines] + ['unrelated'])
    assert set(summary) == {'couple', 'transfer'}
    assert summary['couple']['ms']['step'] == pytest.approx(30.0, abs=0.1)
    assert summary['transfer']['ms']['step'] == pytest.approx(50.0, abs=0.1)
    assert summary['couple']['ratio'] == pytest.approx(DT / 0.030, rel=0.02)


def test_cli_reads_a_run_dir_and_writes_json(tmp_path, capsys):
    import json
    run = tmp_path / 'run-001'
    run.mkdir()
    (run / 'nav-trace.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in _rows([('park', 60, 20, 0.04, 0.7)])) + '\n')
    out = tmp_path / 'est.json'
    assert E.main(['--run', str(run), '--json', str(out)]) == 0
    assert 'speedup x' in capsys.readouterr().out
    assert json.loads(out.read_text())['run-001']['stages']['park']['decode_paused'] is True
    assert E.main([]) == 2                                            # nothing to report
