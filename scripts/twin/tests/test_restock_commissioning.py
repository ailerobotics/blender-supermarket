"""CPU tests for the empty-support commissioning choreography (persist/load/refuse).

Analytic RGB-D fixtures only; never a renderer or physics demonstration. The baseline
path proves bookkeeping and the fail-closed boot refusal, not perception accuracy.
"""
import json
import numpy as np
import pytest

from scripts.twin.restock_camera.geometry import Intrinsics, project, transform_points
from scripts.twin.restock_camera.sensed import polygon_mask
from scripts.twin.restock_camera import commissioning as C


def quad(center, size):
    return np.asarray(center) + np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]) * [*np.asarray(size)/2, 0]


def pix(points, k, camera):
    uv, valid = project(transform_points(np.linalg.inv(camera), points), k)
    assert valid.all(), 'analytic fixture outside view'
    return uv


def empty_frame():
    """A render-faithful empty-support RGB-D frame with two decodable QR markers."""
    cv2 = pytest.importorskip('cv2')
    scale = 2
    k = Intrinsics(1280*scale, 960*scale, 640.*scale, 640.*scale, 640.*scale-.5, 480.*scale-.5)
    camera = np.diag([1.,-1.,-1.,1.]); camera[2,3] = 1.
    depth = np.full((k.height,k.width), .8)
    support = pix(quad([0,0,.2],[.70,.50]), k, camera)
    witness = pix(quad([-.48,0,.2],[.08,.25]), k, camera)
    a = pix(quad([-.43,-.17,.2],[.06,.06])[[3,2,1,0]], k, camera)
    b = pix(quad([-.43,.17,.2],[.06,.06])[[3,2,1,0]], k, camera)
    gray = np.full(depth.shape, 255, np.uint8)
    for code, dest in [('support:a',a),('support:b',b)]:
        qr = cv2.copyMakeBorder(cv2.QRCodeEncoder_create().encode(code),4,4,4,4,cv2.BORDER_CONSTANT,value=255)
        qr = cv2.resize(qr,(300,300),interpolation=cv2.INTER_NEAREST)
        _,corners,_ = cv2.QRCodeDetector().detectAndDecode(qr)
        homography = cv2.getPerspectiveTransform(corners[0].astype(np.float32),dest.astype(np.float32))
        gray = np.minimum(gray, cv2.warpPerspective(qr,homography,(k.width,k.height),
                                                    flags=cv2.INTER_NEAREST,borderValue=255))
    frame = dict(rgb=np.repeat(gray[:,:,None],3,axis=2), distance_to_image_plane=depth,
                 rendering_frame=1, rendering_time=1.)
    return k, camera, depth, support, witness, frame


class StepClock:
    """Deterministic steady clock: each call returns start, then advances by step.

    Injecting it makes the commission() acquisition-to-use gate independent of real decode
    wall time — commission reads the clock once before processing and once after, so a
    two-reading StepClock reproduces any elapsed exactly, on any machine load.
    """
    def __init__(self, start=100., step=0.):
        self.t, self.step = float(start), float(step)

    def __call__(self):
        value = self.t
        self.t += self.step
        return value


def context(k, camera, location='backstock_00', camera_id='source', steady_clock=None):
    from scripts.twin.restock_camera.binding import RenderedSupportContext, SupportSlot
    slot = SupportSlot(location,'deck','support:a',.06,'bolted1',np.ones((k.height,k.width),bool),
                       auxiliary_id='support:b',auxiliary_offset_xy=(0,.34))
    return RenderedSupportContext(camera_id=camera_id,intrinsics=k,slots=[slot],
        measure_context=lambda frame,now:(camera,now,'park1',True),
        clearance_for_slot=lambda loc,view,frame,now:None,
        depth_error_m=.0002,camera_revision='lens1',
        steady_clock=steady_clock or StepClock())            # deterministic by default


def commissioned_dir(tmp_path, location='backstock_00'):
    k, camera, depth, support, witness, frame = empty_frame()
    ctx = context(k, camera, location)
    records = C.commission_and_persist(ctx, frame, 1., {location:(support,witness)},
        dataset_dir=tmp_path, capture_poses={location:[-5.73,16,0]},
        raw_frames={location:dict(rgb=frame['rgb'],depth=depth,support_uv=support,witness_uv=witness,base_from_camera=camera)})
    return tmp_path, records, (k, camera, depth, support, witness, frame)


# --- persistence --------------------------------------------------------------

def test_geometry_json_roundtrip_is_lossless_and_keeps_tuple_types(tmp_path):
    _, records, _ = commissioned_dir(tmp_path)
    geometry = records[0].geometry
    restored = C.geometry_from_json(C.geometry_to_json(geometry))
    assert restored == geometry                                   # frozen dataclass equality
    assert isinstance(restored.required_xy, tuple) and restored.required_xy == (.62,.42)
    assert isinstance(restored.intrinsics, Intrinsics)
    assert type(restored.baseline_frame_id) is int
    assert isinstance(restored.polygon_support, tuple) and isinstance(restored.polygon_support[0], tuple)


def test_persisted_record_carries_provenance_matching_the_capture(tmp_path):
    directory, records, _ = commissioned_dir(tmp_path)
    data = json.loads((directory / 'backstock_00.json').read_text())
    prov = data['provenance']
    assert data['source'] == C.SOURCE and data['schema'] == C.SCHEMA
    assert prov['station'] == 'backstock_00' and prov['capture_pose'] == [-5.73,16,0]
    assert prov['baseline_stamp'] == records[0].geometry.baseline_stamp
    assert prov['calibration_id'] == records[0].geometry.calibration_id
    manifest = json.loads((directory / C.MANIFEST).read_text())
    assert manifest['stations'] == {'backstock_00':'backstock_00.json'}
    assert (directory / 'backstock_00.npz').is_file()             # raw-frame audit sidecar


# --- loader -------------------------------------------------------------------

def test_loader_picks_up_baselines_by_station(tmp_path):
    directory, records, _ = commissioned_dir(tmp_path)
    loaded = C.load_baselines(directory)
    assert set(loaded) == {'backstock_00'}
    assert loaded['backstock_00'].geometry == records[0].geometry
    assert C.load_baseline(directory, 'backstock_00').station == 'backstock_00'


def test_apply_baselines_attaches_tracker_so_mission_capture_validates(tmp_path):
    directory, _, (k, camera, depth, support, witness, frame) = commissioned_dir(tmp_path)
    loaded = C.load_baselines(directory)
    boot = context(k, camera)                                     # fresh context: trackers are None
    assert boot.slots[0].tracker is None
    attached = C.apply_baselines(boot, loaded, required_locations=['backstock_00'])
    assert attached == ['backstock_00'] and boot.slots[0].tracker is not None
    # A later loaded capture must validate against the RELOADED baseline.
    loaded_depth = depth.copy(); loaded_depth[polygon_mask(support,k)] -= .16
    frame.update(distance_to_image_plane=loaded_depth, rendering_frame=2, rendering_time=2.)
    result = boot(frame, 2.)
    assert len(result.slots) == 1
    assert result.slots[0].free_dimensions[2] == 0                # foreground seen, not empty


# --- commissioning freshness bound (deterministic, no real decode wall time) --

def test_commission_uses_its_own_freshness_bound_not_the_live_half_second(tmp_path):
    """A full-res commission that elapses 0.575 s (integration's native measurement) passes
    under commissioning_freshness_s but would trip the 0.5 s live budget. The injected clock
    makes this exact regardless of machine load — no dependence on real decode wall time."""
    k, camera, depth, support, witness, frame = empty_frame()
    # Clock advances 0.575 s between the pre-processing reading and the post-processing check.
    ctx = context(k, camera, steady_clock=StepClock(start=100., step=0.575))
    records = C.commission_and_persist(ctx, frame, 1., {'backstock_00':(support,witness)},
        dataset_dir=tmp_path, commissioning_freshness_s=5.)
    assert records and records[0].commissioning_freshness_s == 5.
    data = json.loads((tmp_path / 'backstock_00.json').read_text())
    assert data['commissioning_freshness_s'] == 5. and data['provenance']['commissioning_freshness_s'] == 5.
    # The same 0.575 s elapse under the 0.5 s live budget is refused.
    ctx2 = context(k, camera, steady_clock=StepClock(start=100., step=0.575))
    with pytest.raises(ValueError, match='freshness expired after processing'):
        ctx2.commission(frame, 1., {'backstock_00':(support,witness)}, commissioning_freshness_s=.5)


def test_commission_rejects_an_unbounded_freshness():
    k, camera, depth, support, witness, frame = empty_frame()
    ctx = context(k, camera, steady_clock=StepClock())
    with pytest.raises(ValueError, match='at least the 0.5 s live budget'):
        ctx.commission(frame, 1., {'backstock_00':(support,witness)}, commissioning_freshness_s=.1)


# --- choreography over several stations ---------------------------------------

def test_run_commissioning_covers_source_and_shelf_into_one_manifest(tmp_path):
    """A commissioning run over both the collect source and the shelf target, as the
    assignment requires at minimum, persists both into one union manifest and reloads."""
    source_k, source_cam, _, support, witness, source_frame = empty_frame()
    shelf_k, shelf_cam, _, s_support, s_witness, shelf_frame = empty_frame()
    shelf_frame.update(rendering_frame=7, rendering_time=1.)
    source_ctx = context(source_k, source_cam, location='backstock_00', camera_id='source')
    shelf_ctx = context(shelf_k, shelf_cam, location='G3_back_s2', camera_id='shelf')

    def capture(ctx):
        if ctx is source_ctx:
            return dict(frame=source_frame, now=1.,
                        empty_polygons={'backstock_00':(support,witness)},
                        capture_poses={'backstock_00':[-5.73,16,0]})
        return dict(frame=shelf_frame, now=1.,
                    empty_polygons={'G3_back_s2':(s_support,s_witness)},
                    capture_poses={'G3_back_s2':[-3.05,6.0,np.pi]})

    records = C.run_commissioning([source_ctx, shelf_ctx], capture, dataset_dir=tmp_path)
    assert {r.station for r in records} == {'backstock_00','G3_back_s2'}
    loaded = C.load_baselines(tmp_path)
    assert set(loaded) == {'backstock_00','G3_back_s2'}
    assert loaded['G3_back_s2'].geometry.camera_id == 'shelf'
    manifest = json.loads((tmp_path / C.MANIFEST).read_text())
    assert set(manifest['stations']) == {'backstock_00','G3_back_s2'}
    assert manifest['provenance']['G3_back_s2']['capture_pose'] == [-3.05,6.0,np.pi]


# --- refusal (fail closed, never fabricate) -----------------------------------

def test_missing_station_baseline_is_refused_at_boot(tmp_path):
    directory, _, (k, camera, depth, support, witness, frame) = commissioned_dir(tmp_path)
    loaded = C.load_baselines(directory)
    shelf_ctx = context(k, camera, location='shelf')
    with pytest.raises(C.CommissioningBaselineMissing, match="station 'shelf' has no commissioned"):
        C.apply_baselines(shelf_ctx, loaded, required_locations=['shelf'])


def test_load_from_uncommissioned_dir_refuses(tmp_path):
    with pytest.raises(C.CommissioningBaselineMissing, match='no commissioning manifest'):
        C.load_baselines(tmp_path / 'never')
    directory, *_ = commissioned_dir(tmp_path)
    with pytest.raises(C.CommissioningBaselineMissing, match="station 'G3_back_s2' has no"):
        C.load_baseline(directory, 'G3_back_s2')


def test_baseline_cannot_be_fabricated_from_a_hand_built_record(tmp_path):
    directory, records, _ = commissioned_dir(tmp_path)
    data = json.loads((directory / 'backstock_00.json').read_text())
    # A record that claims to be something other than an observed commission is refused.
    forged = dict(data, source='config_default')
    with pytest.raises(C.CommissioningBaselineInvalid, match='not a commissioned empty-support'):
        C.BaselineRecord.from_json(forged)
    # Tampering the provenance stamp without the matching geometry is refused.
    tampered = json.loads(json.dumps(data)); tampered['provenance']['baseline_stamp'] = 999.
    with pytest.raises(C.CommissioningBaselineInvalid, match='provenance does not match'):
        C.BaselineRecord.from_json(tampered)


def test_persist_refuses_geometry_without_genuine_provenance(tmp_path):
    _, records, _ = commissioned_dir(tmp_path)
    from dataclasses import replace
    blank = replace(records[0].geometry, calibration_id='')
    with pytest.raises(C.CommissioningBaselineInvalid, match='genuine commissioning provenance'):
        C.persist_baseline(tmp_path / 'x', C.BaselineRecord(station='backstock_00', geometry=blank))


# --- CLI plan (CPU, no capture) -----------------------------------------------

def _design_config():
    """Minimal camera_config-shaped dict with source + shelf supports."""
    def cam(fx):
        return dict(intrinsics=dict(width=2560,height=1920,fx=fx,fy=fx,cx=1280.,cy=960.,near=.15,far=3.5))
    def support(cid):
        return dict(id=cid,camera_id=cid,revision='bolted-pair-v2',required_xy_m=[.62,.42],
                    markers=[dict(id=f'support:{cid}:0',side_m=.06),dict(id=f'support:{cid}:1',side_m=.06)])
    return dict(cameras=dict(source=cam(2000.),shelf=cam(1050.),cart=cam(1500.)),
                supports=dict(source=support('source'),shelf=support('shelf'),cart=support('cart')),
                required_depth_error_bound_m=.0002)


def _scenario():
    return dict(job=dict(source_bin='backstock_00',shelf_id='G3_back_s2'),
                initial_staging=dict(robot_base_pose=[-5.73,16,0],shelf_robot_pose=[-3.05,6.0,np.pi]))


def test_cli_plan_resolves_source_and_shelf_stations_and_poses():
    from scripts import run_restock_commissioning as cli
    plan = cli.commissioning_plan(_design_config(), _scenario(), stations=cli.DEFAULT_STATIONS)
    stations = {s['station']: s for s in plan['stations']}
    assert set(stations) == {'backstock_00','G3_back_s2'}
    assert stations['backstock_00']['camera_id'] == 'source'
    assert stations['backstock_00']['capture_pose'] == [-5.73,16,0]
    assert stations['G3_back_s2']['camera_id'] == 'shelf'
    assert stations['G3_back_s2']['capture_pose'] == [-3.05,6.0,np.pi]
    assert stations['G3_back_s2']['marker_id'] == 'support:shelf:0'
    assert stations['G3_back_s2']['intrinsics']['fx'] == 1050.


def test_cli_plan_refuses_a_station_the_design_does_not_cover():
    from scripts import run_restock_commissioning as cli
    with pytest.raises(ValueError, match='no support in the camera design maps to station'):
        cli.commissioning_plan(_design_config(), _scenario(), stations=['nonexistent_bin'])


def _commission_both(tmp_path):
    """Commission backstock_00 (source) and G3_back_s2 (shelf) into one dataset dir."""
    source_k, source_cam, _, support, witness, source_frame = empty_frame()
    shelf_k, shelf_cam, _, s_support, s_witness, shelf_frame = empty_frame()
    source_ctx = context(source_k, source_cam, location='backstock_00', camera_id='source')
    shelf_ctx = context(shelf_k, shelf_cam, location='G3_back_s2', camera_id='shelf')

    def capture(ctx):
        if ctx is source_ctx:
            return dict(frame=source_frame, now=1., empty_polygons={'backstock_00':(support,witness)})
        return dict(frame=shelf_frame, now=1., empty_polygons={'G3_back_s2':(s_support,s_witness)})

    C.run_commissioning([source_ctx, shelf_ctx], capture, dataset_dir=tmp_path)
    return (source_k, source_cam), (shelf_k, shelf_cam)


def test_apply_to_contexts_attaches_across_cameras_and_enforces_required(tmp_path):
    (source_k, source_cam), (shelf_k, shelf_cam) = _commission_both(tmp_path)
    records = C.load_baselines(tmp_path)
    boot = {'source': context(source_k, source_cam, location='backstock_00', camera_id='source'),
            'shelf': context(shelf_k, shelf_cam, location='G3_back_s2', camera_id='shelf')}
    attached = C.apply_to_contexts(boot, records, required_locations=['backstock_00','G3_back_s2'])
    assert set(attached) == {'backstock_00','G3_back_s2'}
    assert all(ctx.slots[0].tracker is not None for ctx in boot.values())
    # A required station with no baseline refuses at boot.
    with pytest.raises(C.CommissioningBaselineMissing, match='no commissioned empty-support baseline'):
        C.apply_to_contexts(boot, records, required_locations=['rollwagen'])


def test_build_sensing_commissions_contexts_from_the_dataset_dir(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from scripts.twin.restock_manipulation import isaac_sensors
    from scripts.twin.restock_camera.sensing import build_sensing
    (source_k, source_cam), (shelf_k, shelf_cam) = _commission_both(tmp_path)
    providers = {'source': context(source_k, source_cam, location='backstock_00', camera_id='source'),
                 'shelf': context(shelf_k, shelf_cam, location='G3_back_s2', camera_id='shelf')}
    monkeypatch.setattr(isaac_sensors, 'IsaacCartonSensors',
                        lambda camera, **kw: NS(slots=providers[kw['camera_id']].slots, capture=None,
                                                sensed_target_bboxes=(), evidence=()))
    assembly = NS(sensing_sources=dict(
        cameras={'source': object(), 'shelf': object()}, context_providers=providers,
        location_cameras={'backstock_00': 'source', 'G3_back_s2': 'shelf'}, profiles={},
        pose_bridge_root='fixture', motion=NS(), contact=NS(), recording_camera='cam'))
    assert all(p.slots[0].tracker is None for p in providers.values())
    build_sensing(app=None, assembly=assembly, plan=None,
                  config={'commissioning_dir': str(tmp_path),
                          'commissioning_required_locations': ['backstock_00', 'G3_back_s2']},
                  output=None)
    assert all(p.slots[0].tracker is not None for p in providers.values())
    # A config that requires an uncommissioned station fails closed at boot.
    with pytest.raises(C.CommissioningBaselineMissing):
        build_sensing(app=None, assembly=assembly, plan=None,
                      config={'commissioning_dir': str(tmp_path),
                              'commissioning_required_locations': ['rollwagen']}, output=None)


def test_apply_refuses_baseline_for_a_different_camera_or_support(tmp_path):
    directory, _, (k, camera, *_) = commissioned_dir(tmp_path)
    loaded = C.load_baselines(directory)
    from scripts.twin.restock_camera.binding import RenderedSupportContext, SupportSlot
    # Same station location, but a context on a DIFFERENT camera revision.
    slot = SupportSlot('backstock_00','deck','support:a',.06,'bolted1',np.ones((k.height,k.width),bool),
                       auxiliary_id='support:b',auxiliary_offset_xy=(0,.34))
    other = RenderedSupportContext(camera_id='source',intrinsics=k,slots=[slot],
        measure_context=lambda frame,now:(camera,now,'park1',True),
        clearance_for_slot=lambda *a:None,depth_error_m=.0002,camera_revision='lens2')
    with pytest.raises(C.CommissioningBaselineInvalid, match='was commissioned on camera'):
        C.apply_baselines(other, loaded, required_locations=['backstock_00'])
