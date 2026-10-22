"""CPU tests for the recorded decision/reason tree (twin.demo.bt_timeline) and the --bt-panel cut.

Synthetic run dirs only: a bt_events.jsonl the tree is replayed from, a nav-trace with the
RESTOCK_TRACE_POSES cart/carton fields the reasons are measured from, a mission.json event log and a
handful of overhead frames. No Isaac, no ffmpeg (the encode test skips when ffmpeg is absent).
"""
import json
import math

import pytest
from PIL import Image

from twin.demo import bt_timeline as B
from twin.demo import panel as panel_mod
from twin.demo import run_check as C
from twin.demo import run_evidence as E
from twin.demo import walkthrough_cut as W

CART0 = (-5.73, 15.25, 0.0)
BAY = (-2.0, 16.42)
SCENARIO = {
    'handling_unit': {'dimensions_m': [0.6, 0.4, 0.16]},
    'initial_staging': {'cart_body_pose': list(CART0), 'cart_load_center_body_xy': [0.18, 0.0],
                        'cart_support_z_world': 0.9325,
                        'unit_yaws_world': {'cart': -math.pi / 2}},
    'transfer_bay': {'center_xy': list(BAY), 'dimensions_m': [0.7, 0.5, 0.73], 'top_z': 0.73},
}

# Per-stage step plans the synthetic bt_events announce (labels carry the ground-truth tags verbatim).
STEPS = {'couple': ['drive→cart', 'dock drawbar', 'attach'], 'outbound': ['tow→shelf'],
         'park': ['stationarize'], 'transfer': ['perceive slot [scripted, ground truth]',
         'place on shelf', 'verify placed [scripted, ground truth]'], 'verify': ['observe inventory'],
         'return': ['tow→cart park', 'brake cart', 'uncouple', 'back away', 'drive→home']}
KIND = {'couple': ['nav', 'perceive', 'couple'], 'outbound': ['nav'], 'park': ['manip'],
        'transfer': ['perceive', 'manip', 'verify'], 'verify': ['perceive'],
        'return': ['nav', 'manip', 'uncouple', 'nav', 'nav']}


def _bt_events(*, fail_stage=None, fail_reason=''):
    """A shortcut-collect run that performs couple..verify then returns; optionally fails a stage at
    its first step. One event stream in the shape run_mobile_restock writes."""
    ev = [{'t': 0.0, 'type': 'stage_shortcut', 'data': {'stage': 'collect', 'sim_time': 0.0}}]
    order = ['couple', 'outbound', 'park', 'transfer', 'verify', 'return']
    t = 1.0
    for stage in order:
        ev.append({'t': t, 'type': 'stage_begin', 'data': {'stage': stage, 'attempt': 1, 'steps': STEPS[stage]}})
        for i, label in enumerate(STEPS[stage]):
            ev.append({'t': t, 'type': 'stage_step',
                       'data': {'stage': stage, 'attempt': 1, 'step': i, 'label': label, 'kind': KIND[stage][i]}})
            t += 1.0
            if fail_stage == stage and i == 0:
                ev.append({'t': t, 'type': 'stage_failed',
                           'data': {'stage': stage, 'reason': fail_reason}})
                return ev
        ev.append({'t': t, 'type': 'stage_succeeded', 'data': {'stage': stage}})
    return ev


def _trace(events):
    """One pose row per 0.5 s spanning the events, carton on the bay once verify is reached."""
    tmax = events[-1]['t'] + 1.0
    rows, t = [], 0.0
    while t <= tmax:
        stage = 'couple'
        for e in events:
            if e['t'] <= t and e['type'] in ('stage_begin', 'stage_step'):
                stage = e['data']['stage']
        coupled = stage in ('outbound', 'park', 'transfer', 'verify') or (stage == 'couple' and t > 3)
        bx = -5.73 + 1.27 + min(1.0, t / tmax) * 2.4          # drift east toward the bay
        cart = (bx - 1.27, 15.25, 0.0)
        if stage in ('verify', 'return'):
            carton = (BAY[0], BAY[1], 0.81)
        else:
            carton = (cart[0] + 0.18, cart[1], 1.0125)
        rows.append({'tick': len(rows) + 1, 't': round(t, 3), 'action': stage, 'x': round(bx, 3),
                     'y': 15.25, 'yaw_deg': 0.0, 'lin': 0.0 if stage in ('park', 'verify') else 0.2,
                     'coupled': coupled, 'parked': stage in ('park', 'verify'),
                     'vel': {'rv': 0.2, 'rw': 0.0}, 'world_t': round(t + 0.55, 4),
                     'cart': [cart[0], cart[1], 0.0, 0.0], 'carton': [carton[0], carton[1], carton[2], -90.0]})
        t += 0.5
    return rows


def _make_run(tmp_path, *, fail_stage=None, fail_reason='', state='failed'):
    run_dir = tmp_path / 'run-bt'
    run_dir.mkdir()
    events = _bt_events(fail_stage=fail_stage, fail_reason=fail_reason)
    trace = _trace(events)
    mission_events = [{'sequence': 1, 'state': 'collect_shortcut', 'evidence': {'source': 'start_phase_shortcut'}}]
    reached = ['couple', 'outbound', 'park', 'transfer', 'verify', 'return']
    if fail_stage:
        reached = reached[:reached.index(fail_stage)]
    seq = 1
    for s in reached:
        seq += 1
        mission_events.append({'sequence': seq, 'state': s + '_succeeded', 'evidence': {'source': 'sensor'}})
    mission = {'state': state, 'reason': fail_reason, 'degraded': [], 'physics_verified': False,
               'shortcut': {'start_at': 'couple', 'skipped': ['collect']}, 'events': mission_events}
    (run_dir / 'mission.json').write_text(json.dumps(mission))
    (run_dir / 'mission-scenario.json').write_text(json.dumps(SCENARIO))
    with (run_dir / 'bt_events.jsonl').open('w') as f:
        for e in events:
            f.write(json.dumps(e) + '\n')
    with (run_dir / 'nav-trace.jsonl').open('w') as f:
        for r in trace:
            f.write(json.dumps(r) + '\n')
    frames_dir = run_dir / 'mission-frames'
    frames_dir.mkdir()
    fr = []
    for i, r in enumerate(trace[::2]):
        Image.new('RGB', (960, 720), (229, 228, 224)).save(frames_dir / f'{i:06d}.png')
        fr.append({'file': f'{i:06d}.png', 'simulation_time': r['t'] + 0.55})
    (frames_dir / 'frames.json').write_text(json.dumps({'nominal_fps': 30, 'frames': fr}))
    return run_dir


# ---------------- replay ----------------
def test_replay_statuses_and_recorded_labels(tmp_path):
    run = E.load_run(_make_run(tmp_path, state='complete'))
    tl = B.BtTimeline.from_run(run)
    assert tl.shortcut_stages == {'collect'}
    assert [lab for lab, _ in tl.labels['transfer']] == STEPS['transfer']        # recorded labels, tags kept
    assert tl.labels['couple'][0][1] == 'nav' and tl.labels['couple'][2][1] == 'couple'  # recorded kinds
    early = tl.state_at(1.5)                                     # couple step 0 running
    assert early['collect']['status'] == 'shortcut'
    assert early['couple']['status'] == 'running' and early['couple']['active_step'] == 0
    assert early['couple']['steps'][0]['status'] == 'running' and early['outbound']['status'] == 'idle'
    late = tl.state_at(999)
    assert all(late[s]['status'] == 'success' for s in ('couple', 'outbound', 'park', 'transfer', 'verify'))


def test_shortcut_stage_is_idle_not_green(tmp_path):
    """A synthesized (shortcut) stage renders muted (idle), never success-green that would imply it ran."""
    run = E.load_run(_make_run(tmp_path, state='complete'))
    tl = B.BtTimeline.from_run(run)
    sr = {}
    B.seed_reasons(tl, run, E.cart_model(run), sr)
    entry = B.panel_entry(tl, run, E.cart_model(run), t=1.5, header_stage='couple', frame_index=0, step_reasons=sr)
    collect = next(n for n in entry['tree']['children'] if n['name'] == 'stage:collect')
    assert collect['status'] == 'idle' and 'synthesized' in collect['feedback']


# ---------------- reasons measured from the recorded poses ----------------
def test_reasons_are_measured_from_the_trace(tmp_path):
    run = E.load_run(_make_run(tmp_path, state='complete'))
    model, tl = E.cart_model(run), B.BtTimeline.from_run(run)
    reads = B.build_reads(run, model, 1.2)
    assert reads['carton_src'] == 'trace'                       # pose came from the recorded trace
    couple = B.compute_reason(tl, run, model, 'couple', 0, reads)
    assert 'hitch gap' in couple and 'mm' in couple
    # place delta: the carton sits on the bay by verify, so the measured error is small.
    verify_reads = B.build_reads(run, model, tl.stage_end['verify'])
    place = B.compute_reason(tl, run, model, 'transfer', 1, verify_reads)
    assert 'placed Δ' in place
    err = M_error(verify_reads['carton_xyz'], B.bay_target(model))
    assert err < 50                                            # carton on the bay target within 5 cm


def M_error(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b))) * 1000.0


def test_idle_step_has_no_reason_no_lookahead(tmp_path):
    run = E.load_run(_make_run(tmp_path, state='complete'))
    model, tl = E.cart_model(run), B.BtTimeline.from_run(run)
    sr = {}
    B.seed_reasons(tl, run, model, sr)
    entry = B.panel_entry(tl, run, model, t=1.2, header_stage='couple', frame_index=0, step_reasons=sr)
    transfer = next(n for n in entry['tree']['children'] if n['name'] == 'stage:transfer')
    assert all(s['status'] == 'idle' and s['feedback'] == '' for s in transfer['children'])  # not run yet


# ---------------- failure honesty ----------------
def test_failed_step_shows_recorded_fault(tmp_path):
    run = E.load_run(_make_run(tmp_path, fail_stage='transfer',
                               fail_reason="StationAckStreamStalled: stamp frozen [x.py:1]"))
    model, tl = E.cart_model(run), B.BtTimeline.from_run(run)
    sr = {}
    B.seed_reasons(tl, run, model, sr)
    entry = B.panel_entry(tl, run, model, t=tl.stage_end['transfer'], header_stage='transfer',
                          frame_index=0, step_reasons=sr)
    transfer = next(n for n in entry['tree']['children'] if n['name'] == 'stage:transfer')
    assert transfer['status'] == 'failure'
    failed = [s for s in transfer['children'] if s['status'] == 'failure']
    assert failed and 'StationAckStreamStalled' in failed[0]['feedback'] and '[x.py:1]' not in failed[0]['feedback']


def test_return_no_progress_keeps_planner_reason(tmp_path):
    reason = ("return stuck with no base progress for 217s wall (> 120s budget; "
              "last reason 'planning_stop'); failing fast")
    run = E.load_run(_make_run(tmp_path, fail_stage='return', fail_reason=reason))
    tl = B.BtTimeline.from_run(run)
    assert tl.stage_fail_reason['return'] == 'no base progress · planning_stop'


def test_mission_state_is_running_until_terminal(tmp_path):
    run = E.load_run(_make_run(tmp_path, fail_stage='return', fail_reason="x (last reason 'planning_stop')"))
    model, tl = E.cart_model(run), B.BtTimeline.from_run(run)
    sr = {}
    B.seed_reasons(tl, run, model, sr)
    assert B.panel_entry(tl, run, model, t=2.0, header_stage='couple', frame_index=0,
                         step_reasons=sr)['header']['mission_state'] == 'running'
    assert B.panel_entry(tl, run, model, t=tl.last_event_t, header_stage='return', frame_index=0,
                         step_reasons=sr)['header']['mission_state'] == 'failed'


# ---------------- the panel entry drives the demo-004 renderer ----------------
def test_panel_entry_renders(tmp_path):
    run = E.load_run(_make_run(tmp_path, state='complete'))
    model, tl = E.cart_model(run), B.BtTimeline.from_run(run)
    sr = {}
    B.seed_reasons(tl, run, model, sr)
    entry = B.panel_entry(tl, run, model, t=5.0, header_stage='transfer', frame_index=3, step_reasons=sr)
    img = panel_mod.render_panel(entry, size=(480, 1080))
    assert img.size == (480, 1080)
    assert entry['chips'] and entry['header']['stage_title'] == '5 · Transfer'


# ---------------- the --bt-panel cut ----------------
def test_bt_panel_plan_and_frames(tmp_path):
    run = E.load_run(_make_run(tmp_path, fail_stage='return', fail_reason="x (last reason 'planning_stop')"))
    plan = W.build_plan(run, checks=C.run_checks(run), bt_panel=True)
    assert plan.bt_panel and plan.bt is not None
    r = W.Renderer(plan)
    assert (r.out_w, r.out_h) == (W.BT_OUT_W, W.BT_OUT_H)
    frame_shot = next(s for s in plan.shots if s.kind in ('frame', 'still') and s.stage == 'couple')
    assert r.footage_bt(frame_shot).size == (W.BT_OUT_W, W.BT_OUT_H)
    card_shot = next(s for s in plan.shots if s.kind == 'card')
    assert r.card_scaled(card_shot).size == (W.BT_OUT_W, W.BT_OUT_H)


def test_default_cut_unaffected_by_bt_panel_flag(tmp_path):
    """Without --bt-panel the cut stays the original 1280x720 route-panel layout."""
    run = E.load_run(_make_run(tmp_path, state='complete'))
    plan = W.build_plan(run)
    assert not plan.bt_panel and plan.bt is None
    r = W.Renderer(plan)
    assert (r.out_w, r.out_h) == (W.OUT_W, W.OUT_H)
    frame_shot = next(s for s in plan.shots if s.kind in ('frame', 'still'))
    assert r.footage(frame_shot).size == (W.OUT_W, W.OUT_H)


def test_bt_panel_render_video_end_to_end(tmp_path):
    ffmpeg = W.find_ffmpeg()
    if ffmpeg is None:
        pytest.skip('no ffmpeg on this host')
    run = E.load_run(_make_run(tmp_path, fail_stage='return', fail_reason="x (last reason 'planning_stop')"))
    plan = W.build_plan(run, checks=C.run_checks(run), bt_panel=True)
    out = tmp_path / 'bt.mp4'
    summary = W.render_video(plan, out, ffmpeg=ffmpeg)
    assert out.is_file() and out.stat().st_size > 1000
    assert summary['bt_panel'] is True and summary['size'] == [W.BT_OUT_W, W.BT_OUT_H]
