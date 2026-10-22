"""GROUND-TRUTH COLLECT (collect-gt) (RESTOCK_COLLECT_GROUND_TRUTH, default off) -- CPU proof.

Cold runs 343-348 died because the fixed source camera never produced a backstock_00 carton target,
so no /sensors packet ever reached the host ('no initial sensor frame' at the 120 s wall grace). These
tests prove, through the production code, that with the switch ON:
  * the Isaac bridge serves a packet on the FIRST parked collect tick, from the carton asset's pose,
    labelled 'ground_truth', at the SAME pose the camera measured when it worked (run-326), never
    consulting the camera provider;
  * the packet passes the Isaac-side integrity gate and the host controller's provenance + fm449 gates
    into the reach, with collect_mode recorded;
  * after the carton is held the packet keeps the cart target (place re-registration) but claims no
    source observation;
  * the completed collect's ground-truth place check becomes labelled stage evidence the runtime and
    mission accept (collect only, never verify);
  * mission.json is labelled; and with the switch OFF every path is unchanged.
"""
import math
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock_manipulation import ground_truth_collect as gtc  # noqa: E402
from twin.restock_manipulation.controller import TransferController  # noqa: E402
from twin.restock_manipulation.types import Observation, ProductProfile, Sensors, Target  # noqa: E402

ENV = 'RESTOCK_COLLECT_GROUND_TRUTH'


# ---------------------------------------------------------------- pure helpers

def test_switch_is_explicit_opt_in(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert gtc.enabled() is False                                   # default off
    for off in ('', '0', 'false', 'no', 'off', 'garbage'):
        assert gtc.enabled({ENV: off}) is False
    for on in ('1', 'true', 'YES', ' on '):
        assert gtc.enabled({ENV: on}) is True


def test_mission_labels_off_on_and_shortcut():
    s = {'state': 'complete', 'note': 'n.'}
    assert gtc.label_summary(dict(s), env={}) == s                  # off => summary untouched
    on = gtc.label_summary(dict(s), env={ENV: '1'})
    assert on['collect_mode'] == 'ground_truth' and 'GROUND-TRUTH COLLECT' in on['note']
    assert on['walkthrough_caption'] == gtc.CAPTION
    skipped = gtc.label_summary(dict(s), {'start_at': 'couple', 'skipped': ['collect']}, env={ENV: '1'})
    assert skipped == s                                             # collect never ran -> no label
    bench = gtc.label_summary(dict(s, walkthrough_caption='BENCH RUN'), env={ENV: '1'})
    assert bench['walkthrough_caption'] == 'BENCH RUN | ' + gtc.CAPTION


def test_walkthrough_caption_joins_only_when_collect_runs():
    assert gtc.join_caption(None, {}, env={}) is None               # off => byte-identical (no caption)
    assert gtc.join_caption('BENCH', {}, env={}) == 'BENCH'
    assert gtc.join_caption(None, {}, env={ENV: '1'}) == gtc.CAPTION
    assert gtc.join_caption(None, {'start_at': 'collect'}, env={ENV: '1'}) == gtc.CAPTION
    assert gtc.join_caption('BENCH', {'start_at': 'couple'}, env={ENV: '1'}) == 'BENCH'   # collect skipped


SCENARIO_STAGING = {   # mission-scenario.json initial_staging (the warm shortcut's slot fields)
    'cart_body_pose': [-5.73, 15.25, 0], 'cart_load_center_body_xy': [0.18, 0.0],
    'cart_support_z_world': 0.9325, 'unit_yaws_world': {'source': math.pi, 'cart': -math.pi / 2, 'shelf': 0.0}}


def test_warm_slot_reads_the_shortcut_staging_fields():
    slot = gtc.warm_slot({'initial_staging': SCENARIO_STAGING})
    assert slot['load_center_body_xy'] == (0.18, 0.0) and slot['support_z_world'] == 0.9325
    assert slot['yaw_cart_rel'] == pytest.approx(-math.pi / 2)
    yawed = dict(SCENARIO_STAGING, cart_body_pose=[-5.73, 15.25, 0.3])       # cart-RELATIVE yaw
    assert gtc.warm_slot({'initial_staging': yawed})['yaw_cart_rel'] == pytest.approx(-math.pi / 2 - 0.3)


@pytest.mark.parametrize('broken', [
    {}, {'initial_staging': None}, {'initial_staging': dict(SCENARIO_STAGING, cart_load_center_body_xy=[0.1])},
    {'initial_staging': dict(SCENARIO_STAGING, cart_support_z_world=float('nan'))},
    {'initial_staging': dict(SCENARIO_STAGING, unit_yaws_world={})}])
def test_warm_slot_fails_closed(broken):
    with pytest.raises(ValueError):
        gtc.warm_slot(broken)


def _T(x, y, z, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1.]])


def test_carton_top_pose_matches_the_camera_pose_measured_in_run_326():
    # Scenario staging: carton centre (-5.73, 17.1, 0.6+0.08), world yaw pi; pickup base (-5.73, 16.0, 0)
    # with ur_base_link 0.499 up. run-326's source camera measured pose=(-0.001, 1.1, 0.261, 0,0,1,0).
    pose = gtc.carton_top_pose(_T(-5.73, 17.1, 0.68, math.pi), _T(-5.73, 16.0, 0.499, 0.0),
                               (0., 0., 0.), 0.16)
    assert pose[:3] == pytest.approx((-0.001, 1.1, 0.261), abs=2e-3)
    yaw = 2 * math.atan2(pose[5], pose[6])
    assert abs(abs(yaw) - math.pi) < 1e-6                           # same yaw the camera reported
    assert math.isclose(sum(v * v for v in pose[3:]), 1.0, abs_tol=1e-9)


def test_carton_top_pose_is_expressed_in_a_yawed_arm_base():
    yb = math.radians(-5.0)
    pose = gtc.carton_top_pose(_T(-5.73, 17.1, 0.68, math.pi), _T(-5.73, 16.0, 0.499, yb),
                               (0., 0., 0.), 0.16)
    # position rotates into the base frame (distance preserved); orientation stays the WORLD yaw that the
    # bridge's _map_pose_to_base re-expresses through the chassis, exactly like the camera pose_station.
    assert math.hypot(pose[0], pose[1]) == pytest.approx(1.1, abs=1e-9)
    assert pose[0] == pytest.approx(1.1 * math.sin(yb), abs=1e-9)    # R(-yb) @ (0, 1.1)
    assert pose[2] == pytest.approx(0.261, abs=1e-9)


def test_slot_pose_is_the_support_point_with_the_slot_world_yaw():
    slot = gtc.warm_slot({'initial_staging': SCENARIO_STAGING})
    # robot at place_approach (-4.5, 15.4, pi), arm base 0.499 up; cart at its dock (-5.73, 15.25, 0)
    pose = gtc.slot_pose(_T(-5.73, 15.25, 0.0, 0.0), _T(-4.5, 15.4, 0.499, math.pi), slot)
    assert pose[:3] == pytest.approx((1.05, 0.15, 0.9325 - 0.499), abs=1e-9)   # in front, shelf height
    assert 2 * math.atan2(pose[5], pose[6]) == pytest.approx(-math.pi / 2)    # WORLD yaw = cart - 90 deg


def test_carton_top_pose_rejects_bad_transforms():
    with pytest.raises(ValueError):
        gtc.carton_top_pose(np.eye(3), np.eye(4), (0, 0, 0), .16)
    bad = np.eye(4); bad[0, 3] = float('nan')
    with pytest.raises(ValueError):
        gtc.carton_top_pose(bad, np.eye(4), (0, 0, 0), .16)


def _place_check(**kw):
    rec = {'source': 'ground_truth_place_check', 'location': 'rollwagen',
           'verify_mode': 'ground_truth_place', 'simulation_time': 12.5,
           'carton_pose_station': (-1.2, 0., -.19, 0, 0, 0, 1), 'support_pose_station': (-1.2, 0, -.27, 0, 0, 0, 1),
           'xy_offset': .01, 'bottom_gap': .002, 'collect_mode': 'ground_truth'}
    rec.update(kw)
    return rec


def test_stage_evidence_from_place_check_is_honest_and_complete():
    ev = gtc.stage_evidence_from_place_check(_place_check(), sku='juice_orange', quantity=1,
                                             expected_location='rollwagen')
    assert ev['source'] == 'ground_truth' and ev['ground_truth'] is True          # never 'sensor'
    assert ev['sku'] == 'juice_orange' and ev['quantity'] == 1 and ev['location'] == 'rollwagen'
    assert ev['observation_id'] == 'ground_truth_place:12.500' and ev['simulation_time'] == 12.5
    assert ev['verify_mode'] == 'ground_truth_place' and ev['collect_mode'] == 'ground_truth'
    assert ev['xy_offset'] == .01 and ev['bottom_gap'] == .002


@pytest.mark.parametrize('bad', [
    dict(location='G3_back_s2'), dict(simulation_time=float('nan')), dict(simulation_time=True),
    dict(source='sensor')])
def test_stage_evidence_from_place_check_refuses_mismatch(bad):
    with pytest.raises(ValueError):
        gtc.stage_evidence_from_place_check(_place_check(**bad), sku='juice_orange', quantity=1,
                                            expected_location='rollwagen')


# ---------------------------------------------------------------- host controller

class _Motion:
    def __init__(self):
        self.moves = []
    def begin(self, pose, joints, **kw):
        self.moves.append(pose)
    def poll(self):
        return 'succeeded'
    def begin_joint_goal(self, goal, measured, slow=False, goal_tol=None):
        self.moves.append(goal)
    def stop(self):
        pass
    def update_payload(self, held, profile):
        return True
    def last_fault_was_sim_deadline(self):
        return False


PROFILE = ProductProfile('sku-box', (.10, .10, .10), (1., 0., 0., 0.))
JOB = dict(job_id='job-1', sku='sku-box', quantity=1, source_bin='backroom', shelf_id='shelf-a')


def _controller(op='collect', **kw):
    c = TransferController(_Motion(), {'sku-box': PROFILE}, fixture_mode=True, **kw)
    c.start(op, JOB)
    return c


def _packet(c, t, obs_source='ground_truth', obs_location=None, **kw):
    obs = Observation(f'{obs_source}:{t:.3f}', obs_source, t, 'sku-box', obs_location or c.source,
                      (.4, 0., .2, 0., 0., 0., 1.), 1.0)
    tgt = Target(f'cart:{t}', 'test_fixture', t, c.destination, (.5, .2, .1, 0., 0., 0., 1.), True, (.3, .3, .4))
    import time
    values = dict(simulation_time=t, observations=(obs,), target=tgt, parked=True, contact=c.held,
                  feedback_time=t, collision_time=t, joint_positions=(0.,) * 6,
                  scanned_locations=(c.source, c.destination), scan_time=t,
                  valid_until_steady=time.monotonic() + 100.)
    values.update(kw)
    return Sensors(**values)


def test_controller_switch_reads_the_shared_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert _controller().collect_ground_truth is False
    monkeypatch.setenv(ENV, '1')
    assert _controller().collect_ground_truth is True


def test_provenance_accepts_ground_truth_only_for_the_collect_source(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    on, off = _controller(collect_ground_truth=True), _controller()
    gt_src = NS(source='ground_truth', observation_id='g', simulation_time=1., frame='ur_base_link',
                location=on.source)
    assert on._provenance(gt_src, 1.1) is True
    assert off._provenance(gt_src, 1.1) is False                    # default: refused (no silent trust)
    assert on._provenance(replace_ns(gt_src, location=on.destination), 1.1) is False   # not the cart
    tr = _controller('transfer', collect_ground_truth=True)
    assert tr._provenance(replace_ns(gt_src, location=tr.source), 1.1) is False        # collect only


def replace_ns(ns, **kw):
    d = dict(vars(ns)); d.update(kw)
    return NS(**d)


def test_ground_truth_pick_enters_the_reach_through_the_fm449_gate_and_is_labelled(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    c = _controller(collect_ground_truth=True, require_collect_source_evidence=True)
    r = c.step(.1, _packet(c, 1.0, collect_source_clean=False))     # no clean camera frame exists
    assert r.status != 'failed' and c.phase == 'approach'
    labels = [e for e in c.evidence if e.get('source') == 'ground_truth_collect']
    assert labels and labels[0]['collect_mode'] == 'ground_truth'
    assert c.evidence[0]['source'] == 'ground_truth'                # the pick itself stays labelled


def test_without_the_switch_a_ground_truth_pick_never_enters_the_reach(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    c = _controller(require_collect_source_evidence=True)
    for i in range(3):
        c.step(.1, _packet(c, 1.0 + i * .1))
        assert c.phase == 'observe'                                 # provenance refuses it
    assert c.evidence == []


def test_camera_pick_is_still_gated_by_fm449_when_the_switch_is_on(monkeypatch):
    # A camera ('sensor') observation does not get the bypass -- only the labelled ground-truth pick does.
    monkeypatch.delenv(ENV, raising=False)
    c = _controller(collect_ground_truth=True, require_collect_source_evidence=True)
    c.collect_source_evidence_wait_ticks = 2
    r = None
    for i in range(5):
        r = c.step(.1, _packet(c, 1.0 + i * .1, obs_source='test_fixture', collect_source_clean=False))
        if r.status == 'failed':
            break
    assert r.status == 'failed' and r.reason == 'collect_source_evidence_unavailable'


def _run_to(c, phase, t0):
    for i in range(80):
        if c.phase == phase:
            return t0 + i * .1
        contact = c.held or (c.phase == 'grasp' and c.issued)
        if c.phase == 'release' and c.issued:
            contact = False
        r = c.step(.1, _packet(c, t0 + i * .1, obs_source='test_fixture', contact=contact))
        assert r.status != 'failed', r.reason
    pytest.fail('phase not reached')


def test_completed_ground_truth_collect_place_check_carries_collect_mode(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    c = _controller(collect_ground_truth=True, require_collect_source_evidence=True)
    c.step(.1, _packet(c, 1.0))                                     # ground-truth pick -> approach
    t = _run_to(c, 'verify', 1.1)
    on = (.5, .2, .15, 0., 0., 0., 1.)                              # resting on the deck target
    c.step(.1, _packet(c, t, obs_source='test_fixture', placed_pose=on))
    r = c.step(.1, _packet(c, t + .1, obs_source='test_fixture', placed_pose=on))
    assert r.status == 'succeeded'
    last = r.evidence[-1]
    assert last['source'] == 'ground_truth_place_check' and last['collect_mode'] == 'ground_truth'
    ev = gtc.stage_evidence_from_place_check(last, sku='sku-box', quantity=1, expected_location='rollwagen')
    assert ev['collect_mode'] == 'ground_truth' and ev['source'] == 'ground_truth'


# ---------------------------------------------------------------- Isaac-side packet integrity gate

def _ros_runtime(packet, op='collect'):
    from twin.restock.ros_manipulation import RosManipulationRuntime
    calls = []
    graph = NS(start=lambda *a: calls.append('start'), publish_sensors=lambda s: calls.append('sensor'),
               poll=lambda *a: None, close=lambda: calls.append('hold'))
    rt = RosManipulationRuntime(graph, lambda *a: packet, prepare_station=lambda *a: calls.append('calibrate'))
    rt.start(op, dict(job_id='j', source_bin='back', shelf_id='shelf'))
    return rt, calls


def test_integrity_gate_accepts_ground_truth_collect_observation_only_when_enabled(monkeypatch):
    packet = NS(simulation_time=1., observations=[NS(source='ground_truth')], target=NS(source='sensor'))
    monkeypatch.setenv(ENV, '1')
    rt, calls = _ros_runtime(packet)
    assert rt.step(.01, 1.).status == 'running' and 'sensor' in calls and 'hold' not in calls
    rt, calls = _ros_runtime(packet, op='transfer')                 # collect only
    assert rt.step(.01, 1.).status == 'failed' and calls[-1] == 'hold'
    monkeypatch.delenv(ENV)
    rt, calls = _ros_runtime(packet)                                # default: fail-closed as before
    assert rt.step(.01, 1.).status == 'failed' and calls[-1] == 'hold'


def test_integrity_gate_still_refuses_fixtures_with_the_switch_on(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    packet = NS(simulation_time=1., observations=[NS(source='test_fixture')], target=None)
    rt, calls = _ros_runtime(packet)
    assert rt.step(.01, 1.).status == 'failed' and calls[-1] == 'hold'


# ---------------------------------------------------------------- runtime + mission acceptance

def _collect_adapter(evidence):
    from twin.restock.runtime import Packet, RuntimeAdapter
    from twin.restock.inventory import Job
    p0 = Packet('frame-0', 'test_fixture', .1, None, None, None, None, True, False, 0., True)
    ticks = [0]
    def advance(dt):
        ticks[0] += 1
        return replace(p0, observation_id=f'frame-{ticks[0]}', simulation_time=ticks[0] * .1)
    port = NS(advance=advance, controls=lambda c: None, stop=lambda: None)
    nav = NS(start=lambda goal, coupled, **kw: None, cancel=lambda: None,
             step=lambda dt, p: NS(status='succeeded', linear=0., angular=0., reason=''))
    mobile = NS(park=lambda: None, hold=lambda: None, release=lambda: None, command=lambda *a, **k: None,
                attach=lambda obs: None, coupling_state=lambda o: {'healthy': True})

    class M:
        cart_location = 'rollwagen'
        def start(s, a, j): pass
        def warm(s, dt, p): return True
        def step(s, dt, p):
            return NS(status='succeeded', reason='', controls=(), phase='verify', evidence=evidence)
    poses = {k: [0, 0, 0] for k in ('home', 'pickup', 'cart_dock', 'shelf_approach')}
    adapter = RuntimeAdapter(nav, mobile, M(), port, poses, fixture_mode=True)
    adapter.start('collect', Job('j', 'milk', 1, 'back', 'shelf'))
    return adapter


def _drive(adapter):
    r = None
    for _ in range(8):
        r = adapter._step(.1)
        if r.status in ('succeeded', 'failed'):
            break
    return r


def test_runtime_converts_the_ground_truth_place_check_into_labelled_collect_evidence(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    r = _drive(_collect_adapter(({'source': 'ground_truth', 'location': 'backstock_00'},
                                 _place_check(simulation_time=.1))))
    assert r.status == 'succeeded'
    ev = r.evidence
    assert ev['loaded'] is True and ev['source'] == 'ground_truth' and ev['ground_truth'] is True
    assert ev['sku'] == 'milk' and ev['location'] == 'rollwagen' and ev['collect_mode'] == 'ground_truth'


def test_runtime_refuses_a_ground_truth_place_check_at_the_wrong_location(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    adapter = _collect_adapter((_place_check(location='G3_back_s2', simulation_time=.1),))
    with pytest.raises(ValueError):
        _drive(adapter)


def test_runtime_place_check_conversion_is_off_without_the_switch(monkeypatch):
    """Unset => byte-identical: a collect ending on the fm389 place check keeps the pre-existing refusal."""
    monkeypatch.delenv(ENV, raising=False)
    adapter = _collect_adapter(({'source': 'ground_truth', 'location': 'backstock_00'},
                                _place_check(simulation_time=.1)))
    with pytest.raises(ValueError, match='transfer evidence mismatch'):
        _drive(adapter)


def test_runtime_sensor_collect_evidence_is_unchanged():
    r = _drive(_collect_adapter(({'source': 'test_fixture', 'sku': 'milk', 'location': 'rollwagen',
                                  'quantity': 1, 'observation_id': 'o', 'simulation_time': .1},)))
    assert r.status == 'succeeded' and r.evidence['source'] == 'test_fixture' and 'ground_truth' not in r.evidence


def _mission(state):
    from twin.restock.mission import Mission
    m = Mission.__new__(Mission)
    m.fixture_mode, m.time, m.max_evidence_age_s, m.state = False, 12.6, 1.0, state
    return m


def test_mission_accepts_labelled_ground_truth_evidence_for_collect_only(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    ev = dict(gtc.stage_evidence_from_place_check(_place_check(), sku='juice_orange', quantity=1,
                                                  expected_location='rollwagen'), loaded=True)
    _mission('collect')._validate(ev)                               # accepted, labelled
    with pytest.raises(ValueError):
        _mission('collect')._validate(dict(ev, ground_truth=False))   # unlabelled -> sensor-only rule
    with pytest.raises(ValueError):
        _mission('transfer')._validate(dict(ev, released=True))     # not this lane's stage
    with pytest.raises(ValueError):
        _mission('verify')._validate(ev)                            # the stock-commit gate stays sensor-only
    monkeypatch.delenv(ENV)
    with pytest.raises(ValueError, match='requires sensor evidence'):
        _mission('collect')._validate(ev)                           # switch off -> sensor-only, as before


# ---------------------------------------------------------------- Isaac bridge on a real USD stage

pxr = pytest.importorskip('pxr')
from pxr import Gf, Usd, UsdGeom  # noqa: E402


def _xf(stage, path, xyz, yaw_deg=0.0):
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    if yaw_deg:
        x.AddRotateZOp().Set(yaw_deg)
    return x


def _box(stage, path, center, half):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(2.0)
    x = UsdGeom.Xformable(cube.GetPrim())
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    x.AddScaleOp().Set(Gf.Vec3f(*half))


def _stage(base_yaw_deg=0.0, base_xy=(-5.73, 16.0)):
    """The cold-collect start as scene.py + mission-scenario.json author it (backroom dock)."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mesh = UsdGeom.Mesh.Define(stage, '/RestockScenario/HandlingUnit')   # exactly scene.py's authoring
    x, y, z = .3, .2, .08
    mesh.CreatePointsAttr([(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
                           (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    mesh.CreateExtentAttr([(-x, -y, -z), (x, y, z)])
    mesh.AddTranslateOp().Set(Gf.Vec3d(-5.73, 17.1, 0.6 + z))          # support_center_xy, unit_bottom_z
    mesh.AddRotateZOp().Set(180.0)                                      # unit_yaws_world.source = pi
    mesh.GetPrim().SetCustomDataByKey('restock:sku', 'juice_orange')
    _xf(stage, '/World/base_link', (*base_xy, 0.24), base_yaw_deg)  # pickup pose by default
    _xf(stage, '/World/ur_base_link', (*base_xy, 0.499), base_yaw_deg)
    _xf(stage, '/Rollwagen', (-5.73, 15.25, 0.0))                       # cart_body_pose
    UsdGeom.Xform.Define(stage, '/Rollwagen/chassis')
    _box(stage, '/Rollwagen/chassis/deck', (0.0, 0.0, 0.19995), (0.4, 0.34, 0.02505))   # deck top 0.225
    _box(stage, '/Rollwagen/chassis/shelf', (0.0, 0.0, 0.92), (0.38, 0.32, 0.0125))     # upper shelf 0.9325
    return stage


def _bridge(stage, monkeypatch, *, held=False, parked=True, provider=None, slot=None):
    from twin.restock_manipulation.isaac_factory import CollectTargetBridge
    monkeypatch.setenv(ENV, '1')
    cam = NS(get_current_frame=lambda: {'distance_to_image_plane': np.zeros((240, 424))},
             get_intrinsics_matrix=lambda: np.array([[300., 0, 212.], [0, 300., 120.], [0, 0, 1.]]))
    cp = {'location': 'rollwagen', 'load_center_body_xy': (0.08, 0.0), 'support_z_world': 0.9325,
          'free_dimensions': (0.65, 0.45, 0.3)}
    b = CollectTargetBridge(
        stage=stage, arm_camera=cam, arm_camera_path='/arm', base_path='/World/ur_base_link',
        profiles={'juice_orange': ProductProfile(sku='juice_orange', dimensions=(.6, .4, .16),
                                                 tcp_quaternion=(.5, .5, .5, .5))},
        motion=NS(parked=lambda: parked, collision_time=9.0),
        contact=NS(read=lambda: held, path='/RestockPanelGripper'), joints=lambda: (0.,) * 6,
        source_location='backstock_00', mobile_base_path='/World/base_link', cart_placement=cp,
        cart_root='/Rollwagen', cart_support_path='/Rollwagen/chassis/deck', verify_place=True,
        gt_place_slot=slot)
    b._arm_base_from_camera = lambda: np.eye(4)
    b.provider = provider
    return b


def _never(now):
    raise AssertionError('the source camera provider must not be consulted under ground-truth collect')


def test_first_parked_tick_yields_a_labelled_packet_at_the_camera_measured_pose(monkeypatch):
    b = _bridge(_stage(), monkeypatch, provider=_never)
    assert b.collect_ground_truth is True
    packet = b.capture(0.567, 'backstock_00', 'rollwagen')          # run-348's MANIP_START tick
    assert packet is not None, 'no packet => the host would again fail no initial sensor frame'
    (obs,) = packet.observations
    assert obs.source == 'ground_truth' and obs.location == 'backstock_00' and obs.sku == 'juice_orange'
    assert obs.frame == 'ur_base_link' and obs.simulation_time == 0.567
    assert obs.pose[:3] == pytest.approx((-0.001, 1.1, 0.261), abs=2e-3)   # run-326 camera pose
    assert abs(abs(2 * math.atan2(obs.pose[5], obs.pose[6])) - math.pi) < 1e-6
    tgt = packet.target                                             # the cart deck placement target
    assert tgt is not None and tgt.location == 'rollwagen'
    assert tgt.pose[:3] == pytest.approx((0.0, -0.75, 0.225 - 0.499), abs=1e-3)   # deck top centre
    assert packet.placed_pose[:3] == pytest.approx((0.0, 1.1, 0.181), abs=1e-3)    # carton centre


def test_packet_feeds_the_real_host_controller_into_the_reach(monkeypatch):
    b = _bridge(_stage(), monkeypatch, provider=_never)
    packet = b.capture(1.0, 'backstock_00', 'rollwagen')
    import time
    packet = replace(packet, valid_until_steady=time.monotonic() + 100., collision_time=1.0)
    top = ProductProfile('juice_orange', (.6, .4, .16), (1., 0., 0., 0.), grasp_mode='top', max_reach=1.3)
    c = TransferController(_Motion(), {'juice_orange': top}, require_collect_source_evidence=True)
    c.start('collect', dict(job_id='j', sku='juice_orange', quantity=1, source_bin='backstock_00',
                            shelf_id='G3_back_s2'))
    r = c.step(.1, packet)
    assert r.status != 'failed', r.reason
    assert c.phase == 'approach'
    assert any(e.get('collect_mode') == 'ground_truth' for e in c.evidence)


def test_after_the_carton_is_held_no_source_claim_but_the_cart_target_stays(monkeypatch):
    b = _bridge(_stage(), monkeypatch, held=True, provider=_never)
    packet = b.capture(2.0, 'backstock_00', 'rollwagen')
    assert packet is not None and packet.observations == ()
    assert packet.target is not None and packet.target.location == 'rollwagen'
    assert packet.contact is True
    b.contact = NS(read=lambda: False, path='/RestockPanelGripper')     # released on the deck later
    later = b.capture(2.1, 'backstock_00', 'rollwagen')
    assert later.observations == ()                                     # latched: never re-claimed


def test_base_not_parked_yields_no_packet(monkeypatch):
    assert _bridge(_stage(), monkeypatch, parked=False, provider=_never).capture(
        1.0, 'backstock_00', 'rollwagen') is None


def test_yawed_dock_keeps_the_camera_convention(monkeypatch):
    b = _bridge(_stage(base_yaw_deg=-5.0), monkeypatch, provider=_never)
    (obs,) = b.capture(1.0, 'backstock_00', 'rollwagen').observations
    assert math.hypot(obs.pose[0], obs.pose[1]) == pytest.approx(1.1, abs=2e-3)
    # _map_pose_to_base re-expresses the world yaw through the chassis: pi - (-5 deg)
    yaw = math.degrees(2 * math.atan2(obs.pose[5], obs.pose[6])) % 360.0
    assert yaw == pytest.approx(185.0, abs=1e-3)


def test_switch_off_keeps_the_camera_path(monkeypatch):
    from twin.restock_manipulation.isaac_factory import CollectTargetBridge  # noqa: F401
    b = _bridge(_stage(), monkeypatch, provider=lambda now: None)
    monkeypatch.delenv(ENV)
    b.collect_ground_truth = gtc.enabled()                               # as a fresh build would read it
    assert b.collect_ground_truth is False
    assert b.capture(1.0, 'backstock_00', 'rollwagen') is None           # camera provider: nothing sensed


def test_place_target_is_the_warm_slot_not_the_deck(monkeypatch):
    slot = gtc.warm_slot({'initial_staging': SCENARIO_STAGING})
    b = _bridge(_stage(), monkeypatch, provider=_never, slot=slot)        # robot at pickup
    tgt = b.capture(1.0, 'backstock_00', 'rollwagen').target
    # shelf support point at cart body (+0.18, 0) -> world (-5.55, 15.25, 0.9325) -> pickup ur_base_link
    assert tgt.pose[:3] == pytest.approx((0.18, -0.75, 0.9325 - 0.499), abs=1e-6)
    assert 2 * math.atan2(tgt.pose[5], tgt.pose[6]) == pytest.approx(-math.pi / 2)   # base yaw 0 here
    assert tgt.source == 'sensor' and tgt.location == 'rollwagen' and tgt.clear


def test_place_target_reregistered_at_place_approach_is_base_relative(monkeypatch):
    # After the fm390 reposition the base faces the cart mouth (yaw pi); the re-read target must give
    # the wall policy a BASE-relative yaw (-90 - 180 = +90 deg) so the carton lands at -90 cart-relative.
    slot = gtc.warm_slot({'initial_staging': SCENARIO_STAGING})
    b = _bridge(_stage(base_yaw_deg=180.0, base_xy=(-4.5, 15.4)), monkeypatch, held=True, slot=slot)
    tgt = b.capture(9.0, 'backstock_00', 'rollwagen').target
    assert tgt.pose[:3] == pytest.approx((1.05, 0.15, 0.4335), abs=1e-6)
    yaw = math.degrees(2 * math.atan2(tgt.pose[5], tgt.pose[6])) % 360.0
    assert yaw == pytest.approx(90.0, abs=1e-6)


SHELFBOT_ROOT = Path.home() / 'workspaces' / 'shelfbot_newgripper'


@pytest.mark.skipif(not (SHELFBOT_ROOT / 'shelfbot' / 'motion_planning' / 'geometry.py').is_file(),
                    reason='shelfbot (gripper) source not on this machine')
def test_real_wall_policy_places_at_the_ik_proven_slot_contact(monkeypatch):
    """Closed loop with the REAL shelfbot side-grip wall policy: the ground-truth pick (pickup) and the
    warm-slot place target (re-read at place_approach) produce the contact points the actual-URDF IK
    probe (scripts/check_restock_reach.check, robot at place_approach) proves reachable:
    pick contact base_link (0, 0.897, 0.44), place contact base_link (0.847, 0.15, 0.7725)."""
    from twin.restock_manipulation.wall_policy import WallCartonPolicy
    tool = NS(validate_profile=lambda p: None, panel_size=(0.30, 0.20))
    policy = WallCartonPolicy(SHELFBOT_ROOT, tool, walls={'backstock_00': 3, 'rollwagen': 3})
    prof = ProductProfile('juice_orange', (.6, .4, .16), (.5, .5, .5, .5), tcp_surface_offset=.00275,
                          approach=.12, max_reach=1.3, grasp_mode='side')
    slot = gtc.warm_slot({'initial_staging': SCENARIO_STAGING})
    pick_bridge = _bridge(_stage(), monkeypatch, provider=_never, slot=slot)
    pick = pick_bridge.capture(1.0, 'backstock_00', 'rollwagen')
    place_bridge = _bridge(_stage(base_yaw_deg=180.0, base_xy=(-4.5, 15.4)), monkeypatch, held=True, slot=slot)
    target = place_bridge.capture(9.0, 'backstock_00', 'rollwagen').target
    poses = policy.build(pick.observations[0], target, prof)
    to_base_link = 0.499 - 0.24                                          # ur_base_link -> base_link z
    assert poses['contact'][:3] == pytest.approx((0.0, 0.89725, 0.44 - to_base_link), abs=2e-3)
    assert poses['place'][:3] == pytest.approx((0.84725, 0.15, 0.7725 - to_base_link), abs=2e-3)
    for name in ('approach', 'contact', 'lift', 'preplace', 'place', 'retreat'):
        assert math.dist(poses[name][:3], (0, 0, 0)) <= prof.max_reach, name   # controller _pose_ok gate


def _create_params(monkeypatch, tmp_path, scenario):
    from twin.restock_manipulation.isaac_factory import create
    eye = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    path = tmp_path / 'mission-scenario.json'
    import json
    path.write_text(json.dumps(scenario))
    config = {'scenario': str(path), 'manipulation': {
        'profiles': [{'sku': 'juice_orange', 'dimensions': [.6, .4, .16], 'tcp_quaternion': [.5, .5, .5, .5],
                      'grasp_mode': 'side'}],
        'stations': {'source': {'slots': [{'id': 's0'}], 'base_from_camera': eye},
                     'shelf': {'slots': [{'id': 'd0'}], 'base_from_camera': eye}},
        'shelfbot_root': '/x'}}
    built = {}

    def builder(*, app, assembly, params, output):
        built['params'] = params
        return NS(controller=NS(cart_location='rollwagen'))
    create(app=object(), assembly=object(), plan=NS(job_sku='juice_orange'), config=config,
           output=tmp_path, _builder=builder)
    return built['params']


def test_create_loads_the_warm_slot_from_the_runtime_scenario_only_when_enabled(monkeypatch, tmp_path):
    scenario = {'initial_staging': SCENARIO_STAGING}
    monkeypatch.delenv(ENV, raising=False)
    assert _create_params(monkeypatch, tmp_path, scenario).gt_place_slot == {}      # off: untouched
    monkeypatch.setenv(ENV, '1')
    slot = _create_params(monkeypatch, tmp_path, scenario).gt_place_slot
    assert slot['load_center_body_xy'] == (0.18, 0.0) and slot['support_z_world'] == 0.9325
    with pytest.raises(ValueError):                                                  # fail closed at boot
        _create_params(monkeypatch, tmp_path, {'initial_staging': {}})


def test_replay_cli_reads_the_recorded_launch_and_failure_signature(tmp_path):
    """scripts/check_collect_ground_truth.py (the recorded-run replay): the launch args and the cold-collect
    failure signature are parsed from the run's own console/host logs, never assumed."""
    import json
    import check_collect_ground_truth as replay
    run = tmp_path / 'run-9'
    run.mkdir()
    (run / 'mission.json').write_text(json.dumps({'state': 'failed', 'reason': 'StationAckStreamStalled: x'}))
    (tmp_path / 'run-9-manip.txt').write_text('restock_manip: no initial sensor frame\n')
    console = ("Passing the following args to the base kit application:  ['--execute', '--scenario', '/s.json', "
               "'--runtime-config', '/r.json', '--output', '/o']\n"
               'COLLECT_CAPTURE calls=1 provider_error=ValueError:unknown or undecoded label in calibrated slot\n'
               'COLLECT_CAPTURE calls=11 target=None parked=True\n'
               'COLLECT_CAPTURE calls=12 sku=juice_orange roi=(1, 2, 3, 4) pose=(-0.001, 1.1, 0.261, 0.0, 0.0, 1.0, 0.0)\n')
    assert replay.launch_args(console) == {'--scenario': '/s.json', '--runtime-config': '/r.json'}
    sig = replay.signature(run, console)
    assert sig['camera_target_frames'] == 1 and sig['first_camera_pose'][:3] == (-0.001, 1.1, 0.261)
    assert sig['camera_provider_errors'] == {'ValueError:unknown or undecoded label in calibrated slot': 1}
    assert sig['camera_target_none_logged'] == 1 and sig['host_no_initial_sensor_frame']
    assert sig['mission'].startswith('failed: StationAckStreamStalled')
    with pytest.raises(ValueError):
        replay.launch_args('no launch line here')
