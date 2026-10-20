"""CPU tests for the choreographed restock demo: tree structure, value extraction from
scene reads, choreography timeline invariants, panel render and the composite encode."""
import json
import math
import os

import numpy as np
import pytest

from twin.demo import choreography as C
from twin.demo import decision_log as DL
from twin.demo import measures as M
from twin.demo import panel as P
from twin.demo import tree_model as T


# ---------- tree_model ----------
def test_structure_matches_pr153_plans():
    struct = dict((s, steps) for s, steps in T.build_structure())
    assert tuple(struct) == ('collect', 'couple', 'outbound', 'park', 'transfer', 'verify', 'return')
    assert len(struct['collect']) == 7
    assert [k for _, k in struct['couple']] == ['nav', 'perceive', 'couple']
    # return uses the uncouple-at-parking plan (5 legs incl. uncouple).
    labels = [lbl for lbl, _ in struct['return']]
    assert 'uncouple' in [k for _, k in struct['return']]
    assert len(struct['return']) == 5 and 'back away' in labels


def test_annotate_rollup_and_shape():
    states = {'collect': {'status': T.SUCCESS, 'steps': [{'status': T.SUCCESS} for _ in range(7)]},
              'couple': {'status': T.RUNNING, 'steps': [{'status': T.RUNNING, 'reason': 'hitch gap 20 mm'}]}}
    tree = T.annotate(states)
    assert tree['name'] == 'restock mission'
    assert tree['status'] == T.RUNNING  # a failure-free in-progress mission
    names = [c['name'] for c in tree['children']]
    assert names[0] == 'stage:collect' and len(names) == 7
    couple = tree['children'][1]
    assert couple['children'][0]['feedback'] == 'hitch gap 20 mm'


def test_rollup_failure_dominates():
    assert T._roll_up([T.SUCCESS, T.FAILURE, T.RUNNING]) == T.FAILURE
    assert T._roll_up([T.SUCCESS, T.SUCCESS]) == T.SUCCESS
    assert T._roll_up([T.IDLE, T.IDLE]) == T.IDLE


# ---------- measures ----------
def test_base_to_pose_zero_and_yaw():
    r = M.base_to_pose((1.0, 2.0), 0.0, (1.0, 2.0), 0.0, label='pickup')
    assert r['dist_m'] == pytest.approx(0.0) and r['yaw_err_rad'] == pytest.approx(0.0)
    r2 = M.base_to_pose((0.0, 0.0), 0.0, (3.0, 4.0), math.pi, label='x')
    assert r2['dist_m'] == pytest.approx(5.0)
    assert r2['yaw_err_rad'] == pytest.approx(math.pi)


def test_hitch_gap():
    r = M.hitch_gap((0.0, 0.0), (0.02, 0.0), 0.0, math.radians(2))
    assert r['gap_m'] == pytest.approx(0.02)
    assert r['yaw_err_rad'] == pytest.approx(math.radians(2))
    assert 'mm' in r['reason']


def test_lane_progress_monotonic_and_remaining():
    wp = [(0, 0), (10, 0)]
    start = M.lane_progress((0, 0), wp, lane_id='L')
    mid = M.lane_progress((5, 0), wp, lane_id='L')
    end = M.lane_progress((10, 0), wp, lane_id='L')
    assert start['frac'] == pytest.approx(0.0)
    assert mid['frac'] == pytest.approx(0.5)
    assert end['frac'] == pytest.approx(1.0)
    assert mid['remaining_m'] == pytest.approx(5.0)
    assert end['total_m'] == pytest.approx(10.0)


def test_forward_clearance_corridor():
    # Obstacle 2 m ahead on-axis, radius 0.5, within look-ahead -> surface gap 1.5 m ahead.
    r = M.forward_clearance((0, 0), 0.0, [(2.0, 0.0, 0.5)], robot_radius=0.65, brake_m=0.3, lookahead=3.0)
    assert r['clearance_m'] == pytest.approx(1.5)
    assert 'ahead' in r['reason']
    # Obstacle far to the side (outside the forward corridor) -> clear.
    side = M.forward_clearance((0, 0), 0.0, [(1.0, 2.0, 0.2)], robot_radius=0.65, brake_m=0.3, lookahead=3.0)
    assert side['clearance_m'] == math.inf
    # Obstacle beyond the look-ahead window -> clear.
    far = M.forward_clearance((0, 0), 0.0, [(5.0, 0.0, 0.3)], robot_radius=0.65, brake_m=0.3, lookahead=2.5)
    assert far['clearance_m'] == math.inf
    # Behind the base -> clear.
    behind = M.forward_clearance((0, 0), 0.0, [(-1.0, 0.0, 0.3)], robot_radius=0.65, brake_m=0.3)
    assert behind['clearance_m'] == math.inf
    # No obstacles -> clear (inf).
    none = M.forward_clearance((0, 0), 0.0, [], robot_radius=0.65, brake_m=1.0)
    assert none['clearance_m'] == math.inf


def test_box_surface_gap():
    dims = (0.6, 0.4, 0.16)
    # Point at the carton centre -> inside -> gap 0 (sealed).
    assert M.box_surface_gap((0, 0, 0), (0, 0, 0), 0.0, dims) == pytest.approx(0.0)
    # Point 0.5 m out along +x: surface is at 0.3 -> gap 0.2.
    assert M.box_surface_gap((0.5, 0, 0), (0, 0, 0), 0.0, dims) == pytest.approx(0.2)
    # 17 mm from centre is still inside (all half-extents > 17 mm) -> sealed.
    assert M.box_surface_gap((0.017, 0, 0), (0, 0, 0), 0.0, dims) == pytest.approx(0.0)
    # Yaw 90deg swaps x/y extents: 0.35 m along world +x hits the 0.2 half-extent -> gap 0.15.
    assert M.box_surface_gap((0.35, 0, 0), (0, 0, 0), math.pi / 2, dims) == pytest.approx(0.15)


def test_placement_and_slot_and_cage():
    assert M.placement_error((0, 0, 0), (0, 0, 0.01))['error_mm'] == pytest.approx(10.0)
    sf = M.slot_free_depth(4.9, 4.826, 0.47)
    assert 0 < sf['slot_free_m'] < 0.47
    # Cage clearance to the TOP RAIL is positive when the carton fits under it.
    ch = M.cage_clearance(1.09, 1.65)
    assert ch['clearance_m'] == pytest.approx(1.65 - 1.09)
    assert ch['clearance_m'] > 0 and 'clearance' in ch['reason']


# ---------- choreography ----------
def test_timeline_invariants():
    frames, meta = C.build_timeline(fps=8)
    assert meta['frames'] == len(frames) > 100
    # Time strictly increasing.
    ts = [f['t'] for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    # Stages appear in canonical order (first occurrence).
    order, seen = [], set()
    for f in frames:
        if f['stage'] not in seen:
            seen.add(f['stage'])
            order.append(f['stage'])
    assert order == ['collect', 'couple', 'outbound', 'park', 'transfer', 'verify', 'return']
    # All poses finite.
    for f in frames:
        for key in ('base', 'cart'):
            assert all(math.isfinite(v) for v in f[key])
        assert all(math.isfinite(v) for v in f['carton'])


def test_carton_modes_progress_and_coupling():
    frames, _ = C.build_timeline(fps=8)
    modes = [f['carton_mode'] for f in frames]
    # support -> (carried) -> cage -> shelf ordering of first appearances.
    first = {}
    for i, m in enumerate(modes):
        first.setdefault(m, i)
    assert first['support'] < first['cage'] < first['shelf']
    # Coupled during outbound, not coupled at the very end (uncoupled at parking).
    outbound = [f for f in frames if f['stage'] == 'outbound']
    assert all(f['coupled'] for f in outbound)
    assert frames[-1]['coupled'] is False
    # Cart is left behind near the parking area (it trails the west-facing base at the
    # PR154 cart_parking base goal) and clearly west of where the robot drives home empty.
    assert frames[-1]['cart'][0] == pytest.approx(C.CART_PARKING[0], abs=1.6)
    assert frames[-1]['cart'][0] < C.HOME[0] - 1.0


def test_articulation_bounded_while_coupled():
    frames, _ = C.build_timeline(fps=8)
    for f in frames:
        if not f['coupled']:
            continue
        art = (f['cart'][2] - f['base'][2] + math.pi) % (2 * math.pi) - math.pi
        assert abs(art) <= C.MAX_ARTICULATION + 1e-6


def test_final_frame_all_success():
    frames, _ = C.build_timeline(fps=8)
    last = frames[-1]['node_states']
    for stage, st in last.items():
        assert st['status'] == T.SUCCESS
        assert all(x['status'] == T.SUCCESS for x in st['steps'])


# ---------- decision_log (value extraction from reads) ----------
def _reads(base_xy=(-5.73, 16.0), base_yaw=0.0):
    return {
        'base_xy': base_xy, 'base_yaw': base_yaw,
        'cart_yaw': 0.0, 'hitch_xy': (-6.3, 16.0), 'dock_xy': (-6.32, 16.0),
        'carton_xyz': (-3.05, 4.83, 0.81), 'carton_face_xyz': (-5.73, 17.0, 0.68),
        'pad_xyz': (-3.05, 4.83, 0.81), 'carton_top_z': 1.09, 'cage_top_z': 1.65,
        'cage_target_xyz': (-5.65, 15.25, 1.01), 'shelf_target_xyz': (-3.05, 4.826, 0.81),
        'obstacles': [(-3.0, 3.0, 0.3)], 'brake_m': 0.3, 'articulation_rad': math.radians(3),
    }


def test_compute_reason_every_node_nonempty():
    reads = _reads()
    for stage, steps in T.build_structure():
        for i in range(len(steps)):
            reason = DL.compute_reason(stage, i, reads)
            assert isinstance(reason, str) and reason, f'empty reason for {stage}:{i}'


def test_reason_is_measured_not_constant():
    # Moving the base changes the drive→pick distance reason -> proves it reads the scene.
    r_here = DL.compute_reason('collect', 0, _reads(base_xy=(-5.73, 16.0)))
    r_far = DL.compute_reason('collect', 0, _reads(base_xy=(-5.73, 13.0)))
    assert r_here != r_far
    assert '0.00 m' in r_here or '0 m' in r_here


def test_build_entry_shape_and_chips():
    frames, _ = C.build_timeline(fps=6)
    reads = _reads()
    reasons = {}
    f = frames[len(frames) // 3]
    reasons[f"{f['stage']}:{f['step']}"] = DL.compute_reason(f['stage'], f['step'], reads)
    chips = DL.compute_chips(reads, f)
    entry = DL.build_entry(f, reasons, chips)
    assert entry['tree']['name'] == 'restock mission'
    assert entry['header']['frame'] == f['frame']
    assert any(lbl == 'coupled' for lbl, _ in entry['chips'])


# ---------- panel render ----------
def test_panel_renders_nonblank():
    frames, _ = C.build_timeline(fps=6)
    reads = _reads()
    f = frames[len(frames) // 2]
    reasons = {f"{f['stage']}:{f['step']}": DL.compute_reason(f['stage'], f['step'], reads)}
    entry = DL.build_entry(f, reasons, DL.compute_chips(reads, f))
    img = P.render_panel(entry, size=(480, 1080))
    assert img.size == (480, 1080)
    arr = np.asarray(img)
    assert arr.std() > 3.0  # not a flat fill


# ---------- arm IK (skipped when the imported mobile URDF is unavailable) ----------
_MANIFEST = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile/mobile_manifest.json'


def _urdf():
    if os.path.isfile(_MANIFEST):
        u = json.load(open(_MANIFEST)).get('mobile_urdf')
        if u and os.path.isfile(u):
            return u
    return None


@pytest.mark.skipif(_urdf() is None, reason='imported mobile URDF not available')
def test_arm_ik_reaches_carton_targets():
    from twin.demo.arm_ik import ArmIK
    arm = ['ur_shoulder_pan_joint', 'ur_shoulder_lift_joint', 'ur_elbow_joint',
           'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']
    ik = ArmIK(_urdf(), arm, base_link='base_link', tip_link='gripper_pad_tcp')
    stow = np.zeros(6)
    for tgt in ([0.0, 1.1, 0.44], [0.77, 0.0, 0.77], [0.0, 1.17, 0.57]):
        q = ik.solve(np.array(tgt), stow)
        p = ik.fk(q)[:3, 3]
        assert np.linalg.norm(np.array(tgt) - p) < 0.01  # within 10 mm


# ---------- compositor ----------
def test_composite_frame_size():
    from twin.demo import compositor as CO
    scene = P.Image.new('RGB', (960, 720), (10, 40, 10))
    pan = P.Image.new('RGB', (480, 1080), (20, 20, 30))
    out = CO.composite_frame(scene, pan, out_size=(1920, 1080), panel_w=480)
    assert out.size == (1920, 1080)
