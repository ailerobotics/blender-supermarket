"""sim-robust-vram-headroom-v1: crash-safe mission recording (twin.restock.crash_record). The
incremental ProgressWriter + the out-of-process finalizer reconstruct a state=crashed mission.json
from a TRUNCATED progress log, and from nav-trace.jsonl + bt_events.jsonl alone (the coldwin13 shape).
CPU only, stdlib only (no Isaac/ROS)."""
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import crash_record as C


# ---------------------------------------------------------------- ProgressWriter + scan
def test_progress_writer_then_scan(tmp_path):
    w = C.ProgressWriter(tmp_path)
    w.begin('collect', 0.0)
    w.succeeded('collect', 50.0)
    w.begin('couple', 51.0)
    w.succeeded('couple', 120.0)
    w.begin('transfer', 121.0, attempt=1)
    p = C.scan_progress(tmp_path)
    assert p.last_stage == 'transfer'
    assert p.stages_done == ('collect', 'couple')
    assert p.stages_pending == ('verify', 'return')   # after transfer in the ladder
    assert p.sim_time == 121.0
    assert 'progress' in p.sources


def test_scan_tolerates_a_truncated_last_line(tmp_path):
    # Simulate a native crash mid-append: valid lines + a half-written trailing line.
    log = tmp_path / C.PROGRESS_LOG
    w = C.ProgressWriter(tmp_path)
    w.begin('collect', 0.0)
    w.succeeded('collect', 40.0)
    w.begin('couple', 41.0)
    with log.open('a') as f:
        f.write('{"event": "succeeded", "stage": "couple", "sim_tim')  # truncated, no newline
    p = C.scan_progress(tmp_path)
    assert p.last_stage == 'couple'          # recovered despite the corrupt tail
    assert p.stages_done == ('collect',)
    assert p.sim_time == 41.0


def test_write_if_missing_builds_crashed_record(tmp_path):
    w = C.ProgressWriter(tmp_path)
    w.begin('collect', 0.0)
    w.succeeded('collect', 40.0)
    w.begin('transfer', 200.0)
    out = C.write_if_missing(tmp_path, 11)          # SIGSEGV
    assert out is not None and out.name == 'mission.json'
    rec = json.loads(out.read_text())
    assert rec['state'] == 'crashed'
    assert rec['crash_signal'] == 11
    assert rec['crash_signal_name'] == 'SIGSEGV'
    assert rec['last_stage'] == 'transfer'
    assert rec['sim_time'] == 200.0
    assert rec['stage_ladder']['current'] == 'transfer'
    assert 'collect' in rec['stage_ladder']['done']
    assert rec['physics_verified'] is False


def test_write_if_missing_is_noop_when_mission_json_exists(tmp_path):
    (tmp_path / 'mission.json').write_text('{"state": "complete"}\n')
    w = C.ProgressWriter(tmp_path)
    w.begin('transfer', 1.0)
    assert C.write_if_missing(tmp_path, 11) is None
    # The clean verdict is untouched (byte-identical success path).
    assert json.loads((tmp_path / 'mission.json').read_text())['state'] == 'complete'


# ---------------------------------------------------------------- legacy recovery (coldwin13 shape)
def _coldwin13_like(tmp_path):
    """A run dir with NO mission-progress.jsonl -- only nav-trace.jsonl + bt_events.jsonl, as a
    native-crash run from before the progress writer leaves (coldwin13)."""
    nav = tmp_path / C.NAV_TRACE
    with nav.open('w') as f:
        for (action, t) in (('collect', 10.0), ('couple', 120.0), ('outbound', 180.0),
                            ('park', 220.0), ('transfer', 241.55), ('transfer', 241.717)):
            f.write(json.dumps({'t': t, 'action': action, 'x': -2.0, 'coupled': True}) + '\n')
    bt = tmp_path / C.BT_EVENTS
    with bt.open('w') as f:
        for stage, t in (('collect', 10.0), ('couple', 120.0), ('outbound', 180.0), ('park', 220.0)):
            f.write(json.dumps({'t': t, 'type': 'stage_begin',
                                'data': {'stage': stage, 'sim_time': t}}) + '\n')
            f.write(json.dumps({'t': t, 'type': 'stage_succeeded',
                                'data': {'stage': stage, 'sim_time': t}}) + '\n')
        f.write(json.dumps({'t': 241.5, 'type': 'stage_begin',
                            'data': {'stage': 'transfer', 'sim_time': 241.53, 'attempt': 1}}) + '\n')
        f.write(json.dumps({'t': 241.55, 'type': 'stage_failed',
                            'data': {'stage': 'transfer',
                                     'reason': 'collision planning or execution failed'}}) + '\n')
    return tmp_path


def test_recovers_from_nav_and_bt_without_progress_log(tmp_path):
    _coldwin13_like(tmp_path)
    p = C.scan_progress(tmp_path)
    assert p.last_stage == 'transfer'
    assert p.stages_done == ('collect', 'couple', 'outbound', 'park')
    assert p.stages_pending == ('verify', 'return')
    assert abs(p.sim_time - 241.717) < 1e-6        # max across nav + bt
    assert p.last_reason == 'collision planning or execution failed'
    assert 'bt' in p.sources and 'nav' in p.sources


def test_last_vram_included_when_present(tmp_path):
    _coldwin13_like(tmp_path)
    (tmp_path / C.VRAM_FILE).write_text(json.dumps(
        {'used_mib': 10470, 'free_mib': 1438, 'headroom_mib': 1438, 'peak_used_mib': 10549,
         'fired_sheds': ['capture_rate'], 'ceiling_reached': False}) + '\n')
    out = C.write_if_missing(tmp_path, 11)
    rec = json.loads(out.read_text())
    assert rec['last_vram']['free_mib'] == 1438
    assert rec['last_vram']['peak_used_mib'] == 10549
    assert rec['last_fault'] == 'collision planning or execution failed'


def test_timeout_has_no_signal(tmp_path):
    w = C.ProgressWriter(tmp_path)
    w.begin('outbound', 5.0)
    out = C.write_if_missing(tmp_path, None)         # wall-cap timeout, not a signal
    rec = json.loads(out.read_text())
    assert rec['state'] == 'crashed'
    assert rec['crash_signal'] is None
    assert 'timeout' in rec['reason'].lower()


def test_signal_name():
    assert C.signal_name(11) == 'SIGSEGV'
    assert C.signal_name(6) == 'SIGABRT'
    assert C.signal_name(None) is None
    assert C.signal_name(99) == 'SIG99'


# ---------------------------------------------------------------- real coldwin13 artifacts (if present)
def test_real_coldwin13_run_dir_if_available():
    run = Path('/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin13')
    if not (run / C.NAV_TRACE).is_file():
        pytest.skip('coldwin13 run dir not present on this host')
    p = C.scan_progress(run)
    assert p.last_stage == 'transfer'
    assert p.sim_time is not None and p.sim_time > 240.0
    # Build the record without writing into the (external, read-only) run dir.
    rec = C.build_crashed_summary(p, 11)
    assert rec['state'] == 'crashed' and rec['last_stage'] == 'transfer'
    assert rec['stage_ladder']['pending'] == ['verify', 'return']
