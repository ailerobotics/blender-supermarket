"""CPU tests for twin.demo.cut_qa and scripts/check_cut_ranges.py (synthetic sources only)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from twin.demo import cut_qa as Q

cv2 = pytest.importorskip('cv2')

SCRIPTS = Path(__file__).resolve().parents[3]


def _checker():
    spec = importlib.util.spec_from_file_location('check_cut_ranges', SCRIPTS / 'check_cut_ranges.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _real_run(tmp_path, *, n=80, unlatch_at=None, teleport_at=None, still=()):
    """A real-kind source: n frames at 1/3 s, a square that moves every frame except ``still`` frames,
    a coupled tow with an optional unlatch and an optional base teleport."""
    run = tmp_path / 'run-x'
    (run / 'mission-frames').mkdir(parents=True)
    frames, rows = [], []
    x = 0
    for i in range(n):
        if i not in still:
            x += 2
        im = np.full((48, 240, 3), 120, np.uint8)
        im[10:30, x:x + 20] = 250
        Image.fromarray(im).save(run / 'mission-frames' / f'{i:06d}.png')
        frames.append({'file': f'{i:06d}.png', 'simulation_time': 0.05 + i / 3.0})
    for k in range(n * 4):
        t = k / 12.0
        coupled = unlatch_at is None or t < unlatch_at / 3.0
        bx = 0.1 * t + (5.0 if teleport_at is not None and t >= teleport_at / 3.0 else 0.0)
        rows.append({'t': t, 'world_t': t + 0.05, 'x': bx, 'y': 0.0, 'yaw_deg': 0.0, 'action': 'return',
                     'coupled': coupled, 'cart': [bx - 1.2, 0.0, 0.0, 0.2], 'carton': [bx - 1.0, 0.0, 1.0, 0.0]})
    (run / 'mission-frames' / 'frames.json').write_text(json.dumps({'frames': frames}))
    (run / 'nav-trace.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    (run / 'bt_events.jsonl').write_text('')
    return run


def test_unlatch_is_terminal_and_cut_half_a_second_before(tmp_path):
    run = _real_run(tmp_path, unlatch_at=50)
    rec = Q.scan(run)
    s = rec['stages']['return']
    f = s['first_fault']
    assert f['kind'] == 'unlatch' and 49 <= f['frame'] <= 51
    assert s['keep'] == [[0, f['frame'] - Q.MARGIN_FRAMES - 1]]       # nothing after the unlatch
    assert all(c['reason'] for c in s['cuts'])


def test_teleport_is_a_position_jump(tmp_path):
    run = _real_run(tmp_path, teleport_at=40)
    rec = Q.scan(run)
    kinds = {f['kind'] for f in rec['stages']['return']['faults']}
    assert 'position_jump' in kinds
    keep = rec['stages']['return']['keep']
    assert keep and keep[0][1] < 40 - Q.MARGIN_FRAMES


def test_stall_is_spliced_not_ended(tmp_path):
    run = _real_run(tmp_path, n=90, still=range(20, 70))     # 50 static frames > 2 s of output
    rec = Q.scan(run)
    s = rec['stages']['return']
    assert s['stalls'] and s['primary'] == [0, 89]
    assert len(s['keep']) == 2
    (a0, b0), (a1, b1) = s['keep']
    assert a0 == 0 and b1 == 89 and a1 - b0 - 1 > 0                   # interior dropped
    assert b0 - s['stalls'][0][0] + 1 == Q.MARGIN_FRAMES             # 0.5 s of the hold kept


def test_leading_fault_moves_the_start():
    faults = [{'kind': 'image_jump', 'frame': 2, 'end': 2, 'detail': '', 'channel': 'image'}]
    primary, keep, _, _ = Q.stage_ranges((0, 100), faults, [])
    assert primary == [2 + Q.MARGIN_FRAMES + 1, 100] and keep == [primary]


def test_checker_spec_and_ranges():
    chk = _checker()
    cr = {'margin_frames': 8, 'sources': {'s': {'n_frames': 100, 'stages': {
        'collect': {'stage_span': [0, 59], 'keep': [[0, 20], [40, 50]], 'faults': [
            {'kind': 'carton_tilt', 'frame': 59, 'end': 99, 'detail': 'tilt'}],
            'cuts': [{'start': 21, 'end': 39, 'reason': 'stall'}, {'start': 51, 'end': 59, 'reason': 'tilt'}]},
        'couple': {'stage_span': [60, 99], 'keep': [], 'faults': [], 'cuts': []}}}}}
    assert chk.check_ranges(cr) == []
    ok = {'segments': [{'stage': 'collect', 'source': 's', 'frame_start': 40, 'frame_end': 50}]}
    assert chk.check_spec(cr, ok) == []
    over = {'segments': [{'stage': 'collect', 'source': 's', 'frame_start': 0, 'frame_end': 45}]}
    assert chk.check_spec(cr, over) and 'stall' in chk.check_spec(cr, over)[0]
    cr['sources']['s']['stages']['collect']['keep'].append([55, 58])     # inside the tilt margin
    assert any('within 8 frames' in b for b in chk.check_ranges(cr))
