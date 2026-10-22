"""CPU tests for the multi-run compile (twin.demo.stitch).

Synthetic sources only: real-run dirs built by the sibling walkthrough-cut fixture (make_run) and a
hand-built choreographed source (a decision_log.json + a tiny composited mp4). No Isaac; the encode
tests skip when ffmpeg is absent.
"""
import json
import math

import numpy as np
import pytest
from PIL import Image

from twin.demo import panel as PANEL
from twin.demo import run_evidence as E
from twin.demo import stitch as S
from twin.demo import walkthrough_cut as W
from twin.demo.tests import test_walkthrough_cut as TW


def _real(tmp_path, name):
    """A synthetic real-run dir under a fresh parent (make_run does not create its parent)."""
    parent = tmp_path / name
    parent.mkdir(parents=True, exist_ok=True)
    return TW.make_run(parent)


# ---------------- continuity rewrite (pure) ----------------
def _tree():
    """A seven-stage tree where every stage node starts idle with two idle steps."""
    return {'name': 'restock mission', 'status': 'idle', 'feedback': '', 'kind': 'Sequence',
            'children': [{'name': f'stage:{s}', 'status': 'idle', 'feedback': '', 'kind': 'StageRetry',
                          'children': [{'name': f'{s}:a', 'status': 'idle', 'feedback': 'x'},
                                       {'name': f'{s}:b', 'status': 'idle', 'feedback': 'y'}]}
                         for s in E.STAGES]}


def test_rewrite_tree_makes_ladder_global():
    tree = S.rewrite_tree(_tree(), 'outbound')
    by = {n['name']: n for n in tree['children']}
    # collect, couple -> done; their steps all success.
    for s in ('collect', 'couple'):
        assert by[f'stage:{s}']['status'] == 'success'
        assert all(st['status'] == 'success' for st in by[f'stage:{s}']['children'])
    # outbound (the active stage) is forced at least running, with a running leaf.
    assert by['stage:outbound']['status'] == 'running'
    assert by['stage:outbound']['children'][0]['status'] == 'running'
    # park..return idle with cleared feedback.
    for s in ('park', 'transfer', 'verify', 'return'):
        assert by[f'stage:{s}']['status'] == 'idle'
        assert all(st['status'] == 'idle' and st['feedback'] == '' for st in by[f'stage:{s}']['children'])
    assert tree['status'] == 'running'


def test_rewrite_tree_passes_through_active_stage_progress():
    tree = _tree()
    active = next(n for n in tree['children'] if n['name'] == 'stage:transfer')
    active['status'] = 'success'                    # the source already ran this stage
    active['children'][0]['status'] = 'success'
    active['children'][1]['status'] = 'success'
    S.rewrite_tree(tree, 'transfer', final=True)
    keep = next(n for n in tree['children'] if n['name'] == 'stage:transfer')
    assert keep['status'] == 'success'              # not clobbered back to running
    assert tree['status'] == 'success'              # final frame


def test_rewrite_entry_sets_clock_frame_state():
    entry = {'tree': _tree(), 'header': {}, 'chips': []}
    S.rewrite_entry(entry, 'couple', global_frame=42, clock=2.8, final=False)
    assert entry['header']['frame'] == 42
    assert entry['header']['clock'] == pytest.approx(2.8)
    assert entry['header']['mission_state'] == 'running'
    assert entry['header']['stage_title'].endswith('Couple')
    assert S.rewrite_entry({'tree': _tree(), 'header': {}}, 'return', global_frame=1, clock=1.0,
                           final=True)['header']['mission_state'] == 'complete'


# ---------------- spec / detection ----------------
def test_load_run_lenient_carved_segment_without_mission_json(tmp_path):
    """A helper's carved single-stage clip has mission-frames + bt_events + nav-trace but no
    mission.json; the compiler must still load it (empty event log, panel from its bt_events)."""
    from twin.demo import bt_timeline as BT
    run_dir = _real(tmp_path, 'seg')
    (run_dir / 'mission.json').unlink()
    run = S.load_run_lenient(run_dir)
    assert run.frames and run.events == [] and run.mission['state'] == 'running'
    assert isinstance(run.clock_offset, float)
    BT.BtTimeline.from_run(run)                      # bt panel still reconstructs


def test_render_carved_segment_source(tmp_path):
    ffmpeg = W.find_ffmpeg()
    if ffmpeg is None:
        pytest.skip('no ffmpeg on this host')
    run_dir = _real(tmp_path, 'seg')
    (run_dir / 'mission.json').unlink()             # carved clip: no mission.json
    spec = S.CompileSpec(mission_root=tmp_path, fps=15, xfade_frames=0, segments=[
        S.Segment('collect', str(run_dir), 0, 4),
    ])
    out = tmp_path / 'out.mp4'
    summary = S.render(spec, out, ffmpeg=ffmpeg, tmp_dir=tmp_path / 'tmp')
    assert out.is_file() and summary['output_frames'] == 5


def test_detect_kind_and_spec_from_json(tmp_path):
    real = TW.make_run(tmp_path)
    assert S.detect_kind(real) == 'real'
    choreo = tmp_path / 'demo-x'
    choreo.mkdir()
    (choreo / 'decision_log.json').write_text('{"frames": []}')
    assert S.detect_kind(choreo) == 'choreo'
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps({'mission_root': str(tmp_path), 'fps': 12, 'xfade_frames': 4,
                                     'segments': [{'stage': 'couple', 'source': real.name,
                                                   'frame_start': 0, 'frame_end': 3}]}))
    spec = S.CompileSpec.from_json(spec_path)
    assert spec.fps == 12 and spec.xfade_frames == 4 and len(spec.segments) == 1
    assert spec.segments[0].stage == 'couple'


# ---------------- a hand-built choreographed source ----------------
def _choreo_source(tmp_path, n=6, active='outbound'):
    """A demo dir: decision_log.json with n entries (active stage running) + a matching mp4."""
    ffmpeg = W.find_ffmpeg()
    d = tmp_path / 'demo-choreo'
    d.mkdir()
    frames = []
    for i in range(n):
        tree = _tree()
        node = next(x for x in tree['children'] if x['name'] == f'stage:{active}')
        node['status'] = 'running'
        node['children'][0]['status'] = 'running'
        frames.append({'scene_file': f'{i:06d}.png',
                       'panel': {'tree': tree, 'header': {'stage_title': active, 'clock': i, 'frame': i},
                                 'chips': [('coupled', 'yes'), ('base x,y', '1.0, 2.0 m')]}})
    (d / 'decision_log.json').write_text(json.dumps({'meta': {'frames': n}, 'frames': frames}))
    # Encode n distinct 1920x1080 frames so the crop has real pixels.
    import subprocess
    raw = b''.join((np.full((S.OUT_H, S.OUT_W, 3), 30 + i * 20, dtype=np.uint8)).tobytes() for i in range(n))
    cmd = [ffmpeg, '-nostdin', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{S.OUT_W}x{S.OUT_H}', '-r', '15', '-i', 'pipe:0', '-c:v', 'libx264', '-pix_fmt',
           'yuv420p', str(d / 'walkthrough.mp4')]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(raw)
    assert p.returncode == 0
    return d


# ---------------- end to end ----------------
def test_render_two_real_sources_is_continuous(tmp_path):
    ffmpeg = W.find_ffmpeg()
    if ffmpeg is None:
        pytest.skip('no ffmpeg on this host')
    a = _real(tmp_path, 'a')
    b = _real(tmp_path, 'b')
    # a supplies couple, b supplies return -> two different sources -> one crossfade.
    spec = S.CompileSpec(mission_root=tmp_path, fps=15, xfade_frames=4, segments=[
        S.Segment('couple', str(a), 0, 5),
        S.Segment('return', str(b), 0, 5, labels=['x']),
    ])
    out = tmp_path / 'out' / 'wk.mp4'
    summary = S.render(spec, out, ffmpeg=ffmpeg, tmp_dir=tmp_path / 'tmp')
    assert out.is_file() and out.stat().st_size > 1000
    assert out.with_suffix('.table.json').is_file()
    # 6 + 6 frames minus the ~4-frame dissolve overlap.
    assert summary['output_frames'] == 12 - 4
    stages = [r['stage'] for r in summary['segments']]
    assert stages == ['couple', 'return']
    assert summary['segments'][0]['source'] == a.name and summary['segments'][1]['source'] == b.name
    with pytest.raises(FileExistsError):
        S.render(spec, out, ffmpeg=ffmpeg, tmp_dir=tmp_path / 'tmp')


def test_same_source_neighbours_do_not_crossfade(tmp_path):
    ffmpeg = W.find_ffmpeg()
    if ffmpeg is None:
        pytest.skip('no ffmpeg on this host')
    a = _real(tmp_path, 'a')
    spec = S.CompileSpec(mission_root=tmp_path, fps=15, xfade_frames=4, segments=[
        S.Segment('couple', str(a), 0, 5),
        S.Segment('outbound', str(a), 6, 11),
    ])
    out = tmp_path / 'out.mp4'
    summary = S.render(spec, out, ffmpeg=ffmpeg, tmp_dir=tmp_path / 'tmp')
    assert summary['output_frames'] == 12          # no dissolve within one source


def test_min_seconds_holds_a_short_stage(tmp_path):
    ffmpeg = W.find_ffmpeg()
    if ffmpeg is None:
        pytest.skip('no ffmpeg on this host')
    a = _real(tmp_path, 'a')
    spec = S.CompileSpec(mission_root=tmp_path, fps=15, xfade_frames=0, segments=[
        S.Segment('park', str(a), 20, 20, min_seconds=1.0),
    ])
    out = tmp_path / 'out.mp4'
    summary = S.render(spec, out, ffmpeg=ffmpeg, tmp_dir=tmp_path / 'tmp')
    assert summary['output_frames'] == 15          # one frame held to 1.0 s at 15 fps
    assert summary['segments'][0]['held_frames'] == 14


def test_render_choreo_source(tmp_path):
    ffmpeg = W.find_ffmpeg()
    if ffmpeg is None:
        pytest.skip('no ffmpeg on this host')
    d = _choreo_source(tmp_path, n=6, active='outbound')
    spec = S.CompileSpec(mission_root=tmp_path, fps=15, xfade_frames=0, segments=[
        S.Segment('outbound', d.name, 0, 5, kind='choreo'),
    ])
    out = tmp_path / 'out.mp4'
    summary = S.render(spec, out, ffmpeg=ffmpeg, tmp_dir=tmp_path / 'tmp')
    assert out.is_file() and summary['output_frames'] == 6
    assert summary['segments'][0]['kind'] == 'choreo'


def test_panel_renders_from_a_rewritten_entry():
    entry = {'tree': _tree(), 'header': {'stage_title': 'x', 'clock': 1.0, 'frame': 1},
             'chips': [('base x,y', '1.0, 2.0 m')]}
    S.rewrite_entry(entry, 'transfer', global_frame=3, clock=0.2, final=False)
    img = PANEL.render_panel(entry, size=(S.PANEL_W, S.OUT_H))
    assert img.size == (S.PANEL_W, S.OUT_H)


def test_composite_frame_size_and_panel_placement():
    scene = Image.new('RGB', (960, 720), (200, 200, 200))
    panel = Image.new('RGB', (S.PANEL_W, S.OUT_H), (10, 10, 10))
    out = S.composite_frame(scene, panel)
    assert out.size == (S.OUT_W, S.OUT_H)
    # panel occupies the right strip.
    assert out.getpixel((S.OUT_W - 5, S.OUT_H // 2)) == (10, 10, 10)
