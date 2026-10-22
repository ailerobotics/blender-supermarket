"""CPU tests for the demo cut + run checker (twin.demo.run_evidence / run_check / walkthrough_cut).

Synthetic run dirs only: a mission.json event log, a nav-trace with/without the RESTOCK_TRACE_POSES
fields, and overhead frames drawn so the carton sits (or not) where the chase camera model puts it.
"""
import json
import math
import sys
import types

import numpy as np
import pytest
from PIL import Image, ImageDraw

from twin.demo import run_check as C
from twin.demo import run_evidence as E
from twin.demo import walkthrough_cut as W

CART0 = (-5.73, 15.25, 0.0)
SCENARIO = {
    'handling_unit': {'dimensions_m': [0.6, 0.4, 0.16]},
    'initial_staging': {'cart_body_pose': list(CART0), 'cart_load_center_body_xy': [0.18, 0.0],
                        'cart_support_z_world': 0.9325,
                        'unit_yaws_world': {'source': math.pi, 'cart': -math.pi / 2, 'shelf': 0.0}},
    'transfer_bay': {'center_xy': [-2.0, 16.42], 'dimensions_m': [0.7, 0.5, 0.73], 'top_z': 0.73},
}


def _events(stages_ok, *, shortcut=(), fail_at=None, extra_evidence=None):
    ev, seq = [{'sequence': 1, 'state': 'reserved', 'evidence': {}}], 1
    for s in shortcut:
        seq += 1
        ev.append({'sequence': seq, 'state': s + '_shortcut',
                   'evidence': {'shortcut': True, 'source': 'start_phase_shortcut'}})
    for s in stages_ok:
        seq += 1
        ev.append({'sequence': seq, 'state': s, 'evidence': {}})
        seq += 1
        evidence = {'source': 'sensor', 'observation_id': f'physics:{seq}', 'simulation_time': 1.0}
        evidence.update((extra_evidence or {}).get(s, {}))
        ev.append({'sequence': seq, 'state': s + '_succeeded', 'evidence': evidence})
    if fail_at:
        ev.append({'sequence': seq + 1, 'state': fail_at, 'evidence': {}})
        ev.append({'sequence': seq + 2, 'state': 'failed', 'evidence': {'reason': 'boom [x.py:1 f]'}})
    return ev


def _plan_trace(*, poses=False, carton_offset=(0.0, 0.0), frozen=False, on_bay=True):
    """couple (robot drives east to the hitch) -> outbound tow east -> park -> transfer -> verify ->
    return; one row per 0.1 s."""
    rows, t = [], 0.0
    hitch = 1.27

    def add(action, x, y, yaw, coupled, lin=0.2):
        nonlocal t
        row = {'tick': len(rows) + 1, 't': round(t, 3), 'action': action, 'x': x, 'y': y,
               'yaw_deg': math.degrees(yaw), 'lin': lin, 'coupled': coupled, 'parked': lin == 0,
               'vel': {'rv': lin, 'rw': 0.0}}
        if poses:
            if coupled or action in ('outbound', 'park'):
                cart = (x - hitch, y, yaw)
            else:
                cart = CART0
            carton = (cart[0] + 0.18 + carton_offset[0], cart[1] + carton_offset[1], 1.0125)
            if frozen:
                carton = (CART0[0] + 0.18, CART0[1], 1.0125)
            if action in ('verify', 'return') and on_bay:
                carton = (-2.0, 16.42, 0.81)
            row.update(world_t=round(t + 0.55, 4), cart=[cart[0], cart[1], math.degrees(cart[2]), 0.2],
                       carton=[carton[0], carton[1], carton[2], -90.0])
        rows.append(row)
        t += 0.1
    for i in range(20):                       # couple: approach + latch
        add('couple', CART0[0] + hitch, CART0[1], 0.0, i >= 15, 0.0 if i >= 15 else 0.02)
    for i in range(40):                       # outbound: 4 m east
        add('outbound', CART0[0] + hitch + 0.1 * i, CART0[1], 0.0, True)
    end_x = CART0[0] + hitch + 3.9
    add('park', end_x, CART0[1], 0.0, True, 0.0)
    for _ in range(10):
        add('transfer', end_x, CART0[1], 0.0, True, 0.0)
    for _ in range(5):
        add('verify', end_x, CART0[1], 0.0, True, 0.0)
    for i in range(10):
        add('return', end_x - 0.1 * i, CART0[1], 0.0, i < 3)
    return rows


def _draw_frames(run_dir, trace, *, carton_visible=True, offset=0.55):
    """One 960x720 overhead frame per 3 trace rows; the carton drawn warm-white at its projection."""
    frames_dir = run_dir / 'mission-frames'
    frames_dir.mkdir()
    cam = E.ChaseCamera()
    model = E.CartModel()
    frames = []
    for i, row in enumerate(trace[::3]):
        img = Image.new('RGB', (960, 720), (229, 228, 224))
        if carton_visible:
            if row['action'] in ('verify', 'return'):
                center, yaw = (-2.0, 16.42), 0.0
                top = 0.73 + 0.16
                size = (0.6, 0.4)
            else:
                cart = ((row['x'] - 1.27, row['y'], 0.0) if row['coupled'] or row['action'] != 'couple'
                        else CART0)
                center, yaw = (cart[0] + 0.18, cart[1]), -math.pi / 2
                top = 0.9325 + 0.16
                size = (0.6, 0.4)
            pts = cam.project((row['x'], row['y'], math.radians(row['yaw_deg'])),
                              E.rect_corners(center, yaw, size, top))
            ImageDraw.Draw(img).polygon(pts, fill=(252, 249, 235))
        name = f'{i:06d}.png'
        img.save(frames_dir / name)
        frames.append({'render_id': str(i), 'simulation_time': row['t'] + offset, 'shape': [720, 960, 3],
                       'file': name, 'sha256': ''})
    (frames_dir / 'frames.json').write_text(json.dumps({'nominal_fps': 30, 'frames': frames}))


def make_run(tmp_path, *, state='complete', poses=False, carton_offset=(0.0, 0.0), frozen=False,
             frames=True, carton_visible=True, shortcut=(), fail_at=None, on_bay=True, mission_extra=None,
             extra_evidence=None):
    run_dir = tmp_path / 'run-900'
    run_dir.mkdir()
    stages = [s for s in E.STAGES if s not in shortcut]
    if fail_at:
        stages = stages[:stages.index(fail_at)]
    mission = {'state': state, 'reason': '' if state == 'complete' else 'boom', 'degraded': [],
               'physics_verified': False,
               'events': _events(stages, shortcut=shortcut, fail_at=fail_at, extra_evidence=extra_evidence)}
    if shortcut:
        mission['shortcut'] = {'start_at': stages[0], 'skipped': list(shortcut)}
    mission.update(mission_extra or {})
    (run_dir / 'mission.json').write_text(json.dumps(mission))
    trace = _plan_trace(poses=poses, carton_offset=carton_offset, frozen=frozen, on_bay=on_bay)
    if fail_at:                                        # a failed run's trace stops in the failing stage
        keep = E.STAGES[:E.STAGES.index(fail_at) + 1]
        trace = [r for r in trace if r['action'] in keep]
    for ev in mission['events']:                       # stamp success times from the trace
        if ev['state'].endswith('_succeeded'):
            stage = E.event_stage(ev['state'])
            ts = [r['t'] for r in trace if r['action'] == stage]
            if ts:
                ev['evidence']['simulation_time'] = ts[-1]
    (run_dir / 'mission.json').write_text(json.dumps(mission))
    with (run_dir / 'nav-trace.jsonl').open('w') as f:
        for r in trace:
            f.write(json.dumps(r) + '\n')
    (tmp_path / 'run-900.console.log').write_text(
        'CLOCK_DIAG tick=1 stage_time=0.0167 world_current_time=0.5667 wall=0.12\n')
    (run_dir / 'mission-scenario.json').write_text(json.dumps(SCENARIO))
    if frames:
        _draw_frames(run_dir, trace, carton_visible=carton_visible)
    return run_dir


# ---------------- run_evidence ----------------
def test_stage_status_order_and_labels(tmp_path):
    run = E.load_run(make_run(tmp_path, shortcut=('collect',),
                              mission_extra={'verify_mode': 'ground_truth'},
                              extra_evidence={'transfer': {'evidence': [{'source': 'ground_truth_place_check',
                                                                         'verify_mode': 'ground_truth'}]}}))
    status = E.stage_status(run)
    assert status['collect'] == 'shortcut' and all(status[s] == 'succeeded' for s in E.STAGES[1:])
    assert E.stage_order(run) == list(E.STAGES)
    texts = {(l.stage, l.text) for l in E.workaround_labels(run)}
    assert ('collect', 'start_at=couple (collect synthesized, not performed)') in texts
    assert (None, 'verify_mode=ground_truth') in texts
    assert ('transfer', 'source=ground_truth_place_check') in texts
    assert ('transfer', 'verify_mode=ground_truth') in texts
    assert run.clock_offset == pytest.approx(0.55) and 'CLOCK_DIAG' in run.clock_source


def test_failed_stage_and_result_line(tmp_path):
    run = E.load_run(make_run(tmp_path, state='failed', shortcut=('collect',), fail_at='transfer'))
    status = E.stage_status(run)
    assert status['transfer'] == 'failed' and status['verify'] == 'missing'
    assert E.result_line(run) == 'MISSION FAILED at TRANSFER'
    assert E.short_reason('x: y [a.py:1 f | b.py:2 g]') == 'x: y'


def test_chase_camera_projection_matches_recorded_run356():
    # run-356 frame 640: base (-2.079, 15.254, -0.5 deg) coupled; the carton appears centred ~(255, 360).
    cam = E.ChaseCamera()
    base = (-2.079, 15.254, math.radians(-0.5))
    carton_xy = E.compose(E.compose(base, (-1.27, 0.0)) + (base[2],), (0.18, 0.0))
    (u, v), = cam.project(base, [(carton_xy[0], carton_xy[1], 1.0925)])
    assert abs(u - 255) < 15 and abs(v - 360) < 15
    assert cam.for_size(480, 360).fx == pytest.approx(215.0)


# ---------------- run_check ----------------
def _by_name(checks):
    return {c.name: c for c in checks}


def test_complete_run_with_pose_trace_passes(tmp_path):
    run = E.load_run(make_run(tmp_path, poses=True))
    assert run.has_pose_trace and run.clock_source == 'nav-trace world_t'
    checks = _by_name(C.run_checks(run))
    assert checks['stages_in_order'].result == C.PASS
    assert checks['mission_succeeded'].result == C.PASS
    assert checks['carton_on_cart'].result == C.PASS and 'pose trace' in checks['carton_on_cart'].evidence
    assert checks['carton_on_bay'].result == C.PASS
    assert C.overall(list(checks.values())) in (C.PASS, C.WARN)


def test_carton_off_deck_fails_from_poses(tmp_path):
    checks = _by_name(C.run_checks(E.load_run(make_run(tmp_path, poses=True, carton_offset=(0.0, 0.9)))))
    assert checks['carton_on_cart'].result == C.FAIL


def test_frozen_carton_pose_is_unverified_not_pass(tmp_path):
    checks = _by_name(C.run_checks(E.load_run(make_run(tmp_path, poses=True, frozen=True))))
    assert checks['carton_on_cart'].result == C.UNVERIFIED


def test_carton_not_on_bay_fails_from_poses(tmp_path):
    checks = _by_name(C.run_checks(E.load_run(make_run(tmp_path, poses=True, on_bay=False))))
    assert checks['carton_on_bay'].result == C.FAIL


def test_frame_analysis_sees_carton_on_deck_and_bay(tmp_path):
    run = E.load_run(make_run(tmp_path))
    assert not run.has_pose_trace
    checks = _by_name(C.run_checks(run))
    assert checks['carton_on_cart'].result == C.PASS and 'frame analysis' in checks['carton_on_cart'].evidence
    assert checks['carton_on_bay'].result == C.PASS


def test_frame_analysis_fails_without_carton(tmp_path):
    checks = _by_name(C.run_checks(E.load_run(make_run(tmp_path, carton_visible=False))))
    assert checks['carton_on_cart'].result == C.FAIL
    assert checks['carton_on_bay'].result == C.FAIL


def test_shortcut_is_warn_and_failure_fails_overall(tmp_path):
    checks = C.run_checks(E.load_run(make_run(tmp_path, poses=True, shortcut=('collect',))))
    assert _by_name(checks)['stages_in_order'].result == C.WARN
    assert C.overall(checks) == C.WARN


def test_failed_run_table(tmp_path):
    run = E.load_run(make_run(tmp_path, state='failed', fail_at='transfer'))
    checks = C.run_checks(run)
    table = C.format_table(run, checks)
    assert 'OVERALL' in table and C.overall(checks) == C.FAIL
    assert _by_name(checks)['carton_on_bay'].evidence.startswith('transfer never succeeded')


# ---------------- walkthrough_cut ----------------
def test_plan_covers_every_stage_and_time_lapses(tmp_path):
    run = E.load_run(make_run(tmp_path, shortcut=('collect',)))
    plan = W.build_plan(run)
    kinds = [(s.kind, s.card) for s in plan.shots]
    assert kinds[0] == ('card', 'title') and kinds[-1] == ('card', 'end')
    assert ('card', 'shortcut') in kinds
    shown = {s.stage for s in plan.shots if s.kind in ('frame', 'still')}
    assert shown == set(E.STAGES[1:])
    for stage in E.STAGES[1:]:      # every performed stage stays up long enough to read its caption
        assert sum(s.duration for s in plan.shots if s.stage == stage) >= W.MIN_STAGE_S - 1e-9
    couple = [s for s in plan.shots if s.stage == 'couple' and s.kind == 'frame']
    assert max(s.speed for s in couple) >= W.SLOW_SPEED     # the creeping latch is time-lapsed


def test_unreached_card_on_failure(tmp_path):
    plan = W.build_plan(E.load_run(make_run(tmp_path, state='failed', fail_at='transfer')))
    assert any(s.card == 'unreached' for s in plan.shots)
    assert plan.unreached == ['verify', 'return']


def test_render_video_end_to_end(tmp_path):
    ffmpeg = W.find_ffmpeg()
    if ffmpeg is None:
        pytest.skip('no ffmpeg on this host')
    run = E.load_run(make_run(tmp_path, poses=True, shortcut=('collect',)))
    plan = W.build_plan(run, checks=C.run_checks(run))
    out = tmp_path / 'demo.mp4'
    summary = W.render_video(plan, out, ffmpeg=ffmpeg)
    assert out.is_file() and out.stat().st_size > 1000
    assert summary['output_frames'] == round(sum(s.duration for s in plan.shots) * W.FPS)
    assert json.loads(out.with_suffix('.cut.json').read_text())['result'] == 'MISSION COMPLETE'
    with pytest.raises(FileExistsError):
        W.render_video(plan, out, ffmpeg=ffmpeg)


def test_cards_render(tmp_path):
    run = E.load_run(make_run(tmp_path, state='failed', shortcut=('collect',), fail_at='transfer'))
    plan = W.build_plan(run, checks=C.run_checks(run))
    r = W.Renderer(plan)
    for shot in plan.shots:
        img = r.card(shot) if shot.kind == 'card' else r.footage(shot)
        assert img.size == (W.OUT_W, W.OUT_H)


# ---------------- the port's opt-in pose trace ----------------
def test_port_trace_poses_reads_carton_prim(monkeypatch):
    from twin.restock.isaac_port import IsaacPort

    class Matrix:
        def __getitem__(self, i):
            return [[0.0, -1.0, 0, 0], [1.0, 0.0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]][i]

        def ExtractTranslation(self):
            return (-5.55, 15.25, 1.0125)

    class Cache:
        def GetLocalToWorldTransform(self, prim):
            return Matrix()

    class Prim:
        def __bool__(self):
            return True

        def IsValid(self):
            return True

    pxr = types.ModuleType('pxr')
    pxr.UsdGeom = types.SimpleNamespace(XformCache=Cache)
    monkeypatch.setitem(sys.modules, 'pxr', pxr)
    port = object.__new__(IsaacPort)
    port._trace_carton_path = '/RestockScenario/HandlingUnit'
    port.world = types.SimpleNamespace(stage=types.SimpleNamespace(GetPrimAtPath=lambda p: Prim()))
    cart = types.SimpleNamespace(x=-5.73, y=15.25, yaw=0.0)
    out = port._trace_poses(12.3456, cart, 0.2)
    assert out['world_t'] == 12.3456 and out['cart'][:2] == [-5.73, 15.25]
    assert out['carton'] == [-5.55, 15.25, 1.0125, -90.0]
    port.world = types.SimpleNamespace(stage=None)          # a failed read records None, never raises
    assert port._trace_poses(1.0, cart, 0.2)['carton'] is None


def test_bench_caption_and_staging_mode_labels(tmp_path):
    run_dir = make_run(tmp_path, shortcut=('collect', 'couple', 'outbound', 'park'),
                       mission_extra={'start_at': 'transfer', 'start_staging_mode': 'recorded_bay_park',
                                      'walkthrough_caption': 'BENCH RUN start_at=transfer: collect, couple, '
                                                             'outbound, park SYNTHESIZED, not performed'})
    run = E.load_run(run_dir)
    texts = [l.text for l in E.workaround_labels(run)]
    assert 'start_staging_mode=recorded_bay_park' in texts
    assert any(t.startswith('caption=BENCH RUN start_at=transfer') for t in texts)
    assert not any(t == 'start_at=transfer' for t in texts)       # said per stage, not twice
    check = _by_name(C.run_checks(run))['carton_on_cart']
    assert check.result == C.UNVERIFIED and 'synthesized' in check.evidence
