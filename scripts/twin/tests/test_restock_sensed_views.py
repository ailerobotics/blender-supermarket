"""Analytic RGB-D fixtures, never sensor/physics demonstration evidence."""
from dataclasses import replace
import numpy as np
import pytest

from scripts.twin.restock_camera.geometry import Intrinsics, project, transform_points, pose_matrix
from scripts.twin.restock_camera.sensed import (
    FrameEvidence, ClearanceEvidence, RegisteredSupport, calibrate_rigid_support,
    register_marker, polygon_mask)


def pixels(points, k, camera):
    uv,valid = project(transform_points(np.linalg.inv(camera),points),k)
    assert valid.all(), 'analytic fixture outside view'
    return uv


def quad(center, size):
    return np.asarray(center)+np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])*[*np.asarray(size)/2,0]


def marker_pixels(center, side, k, camera):
    return pixels(quad(center,[side,side])[[3,2,1,0]],k,camera)


def fixture(scale=1):
    k = Intrinsics(1280*scale,960*scale,640.*scale,640.*scale,640.*scale-.5,480.*scale-.5)
    camera = np.diag([1.,-1.,-1.,1.])
    camera[2,3] = 1.
    depth = np.full((k.height,k.width),.8)
    support = pixels(quad([0,0,.2],[.70,.50]),k,camera)
    witness = pixels(quad([-.48,0,.2],[.08,.25]),k,camera)
    a = marker_pixels([-.43,-.17,.2],.06,k,camera)
    b = marker_pixels([-.43,.17,.2],.06,k,camera)
    return k,camera,depth,support,witness,a,b


def registration(k,camera,depth,a,b, *, stamp=1.,frame_id=1,epoch='park1',now=None):
    return register_marker(depth,a,k,camera,marker_id='support:a',decoded_id='support:a',
        support_id='deck',revision='bolted1',side_m=.06,
        evidence=FrameEvidence('camera',frame_id,stamp,stamp,epoch,'lens1'),
        now=stamp if now is None else now,parked=True,depth_error_m=.0002,
        auxiliary_pixels=b,auxiliary_id='support:b',decoded_auxiliary_id='support:b',
        auxiliary_offset_xy=(0,.34))


def commissioned():
    k,camera,depth,support,witness,a,b = fixture()
    reg = registration(k,camera,depth,a,b)
    model = calibrate_rigid_support(depth,support,witness,k,camera,reg,
        required_xy=(.62,.42),calibration_id='empty-1',now=1.,parked=True)
    return RegisteredSupport(model),(k,camera,depth,support,witness,a,b)


def clearance(stamp=2.,epoch='park1'):
    return ClearanceEvidence('deck','measured-volume',stamp,epoch,(.62,.42),.25,stamp)


def test_full_measured_quad_does_not_shrink_to_image_rectangle():
    tracker,(k,camera,depth,_,_,a,b) = commissioned()
    reg = registration(k,camera,depth,a,b,stamp=2.,frame_id=2)
    view = tracker.prepare(depth,k,camera,reg,clearance(),now=2.,parked=True)
    assert view.geometry.required_xy == (.62,.42)
    assert view.geometry.baseline_stamp == 1.
    assert view.registration.evidence.stamp == 2.
    assert view.uncertainty_m < .01
    assert view.mask(k).sum() > 10000
    np.testing.assert_allclose(view.center_base,[0,0,.2],atol=.002)


def test_loaded_support_moves_only_with_fresh_markers_and_witness():
    tracker,(k,camera,depth,support,_,a,b) = commissioned()
    loaded = depth.copy()
    # A real foreground changes depth; the dated support model is not called empty.
    loaded[polygon_mask(support,k)] -= .16
    move = pose_matrix([.2,.1,.8014567751836435])
    current_camera = move @ camera
    reg = registration(k,current_camera,loaded,a,b,stamp=54.,frame_id=2,epoch='park2')
    view = tracker.prepare(loaded,k,current_camera,reg,clearance(54.,'park2'),now=54.,parked=True)
    assert view.geometry.baseline_stamp == 1. and view.registration.evidence.stamp == 54.
    assert view.yaw_rad == pytest.approx(.8014567751836435,abs=.002)
    assert not np.isclose(view.support_z,1.-loaded[480,640])
    with pytest.raises(ValueError,match='different frame'):
        view.validate(loaded,k,current_camera,replace(reg.evidence,epoch='park3'),54.,True)


@pytest.mark.parametrize('fault',['stale','witness_hidden','unknown_depth','expired','revision','moving','clearance','identity'])
def test_support_refusals(fault):
    tracker,(k,camera,depth,_,witness,a,b) = commissioned()
    now = 602. if fault == 'expired' else 2.
    reg = registration(k,camera,depth,a,b,stamp=now,frame_id=2)
    proof = clearance(now)
    if fault == 'stale': reg = replace(reg,evidence=replace(reg.evidence,stamp=0.,transform_stamp=0.))
    if fault == 'revision': reg = replace(reg,revision='unbolted2')
    if fault == 'identity': reg = replace(reg,support_id='other')
    if fault == 'clearance': proof = replace(proof,source='ground_truth')
    if fault == 'witness_hidden': depth[polygon_mask(witness,k)] -= .10
    if fault == 'unknown_depth': depth[480,640] = np.nan
    with pytest.raises(ValueError):
        tracker.prepare(depth,k,camera,reg,proof,now=now,parked=fault != 'moving')


def test_occupied_source_cannot_be_commissioned_as_empty():
    k,camera,depth,support,witness,a,b = fixture()
    reg = registration(k,camera,depth,a,b)
    depth[polygon_mask(support,k)] -= .16
    with pytest.raises(ValueError):
        calibrate_rigid_support(depth,support,witness,k,camera,reg,
            required_xy=(.62,.42),calibration_id='not-empty',now=1.,parked=True)


def test_insufficient_footprint_is_not_inflated():
    k,camera,depth,_,witness,a,b = fixture()
    reg = registration(k,camera,depth,a,b)
    support = pixels(quad([0,0,.2],[.61,.41]),k,camera)
    with pytest.raises(ValueError,match='full required footprint'):
        calibrate_rigid_support(depth,support,witness,k,camera,reg,
            required_xy=(.62,.42),calibration_id='small',now=1.,parked=True)


def test_explicit_invalidation_and_replayed_frame_refused():
    tracker,(k,camera,depth,_,_,a,b) = commissioned()
    reg = registration(k,camera,depth,a,b,stamp=2.,frame_id=2)
    tracker.prepare(depth,k,camera,reg,clearance(),now=2.,parked=True)
    with pytest.raises(ValueError,match='monotonic'):
        tracker.prepare(depth,k,camera,reg,clearance(),now=2.,parked=True)
    tracker.invalidate('marker bracket was disturbed')
    with pytest.raises(ValueError,match='invalidated'):
        tracker.prepare(depth,k,camera,registration(k,camera,depth,a,b,stamp=3.,frame_id=3),
                        clearance(3.),now=3.,parked=True)


def test_commissioning_provenance_survives_only_explicit_persistent_clock():
    k,camera,depth,support,witness,a,b = fixture()
    reg = registration(k,camera,depth,a,b,stamp=90.,frame_id=100)
    reg = replace(reg,evidence=replace(reg.evidence,session_id='commission',
                  steady_domain='process-boot-uuid',acquired_steady=1000.))
    g = calibrate_rigid_support(depth,support,witness,k,camera,reg,
        required_xy=(.62,.42),calibration_id='immutable-empty',now=90.,parked=True)
    tracker = RegisteredSupport(g)
    current = registration(k,camera,depth,a,b,stamp=.1,frame_id=1)
    current = replace(current,evidence=replace(current.evidence,session_id='mission',
                      steady_domain='process-boot-uuid',acquired_steady=1010.))
    view = tracker.prepare(depth,k,camera,current,None,now=.1,parked=True)
    assert view.geometry.baseline_stamp == 90.
    assert view.geometry.baseline_session_id == 'commission'
    assert view.geometry.baseline_acquired_steady == 1000.
    assert view.clearance is None
    for changed in (replace(current.evidence,steady_domain='other-process'),
                    replace(current.evidence,acquired_steady=999.),
                    replace(current.evidence,acquired_steady=1601.)):
        with pytest.raises(ValueError):
            RegisteredSupport(g).prepare(depth,k,camera,replace(current,evidence=changed),None,now=.1,parked=True)


def test_lifetime_covers_measured_mission_but_keeps_live_freshness():
    tracker,(k,camera,depth,_,_,a,b) = commissioned()
    reg = registration(k,camera,depth,a,b,stamp=159.1,frame_id=200)
    view = tracker.prepare(depth,k,camera,reg,None,now=159.1,parked=True)
    assert view.geometry.baseline_stamp == 1.
    with pytest.raises(ValueError,match='stale'):
        RegisteredSupport(view.geometry).prepare(depth,k,camera,reg,None,now=159.7,parked=True)
    short = replace(view.geometry,lifetime_s=100.)
    with pytest.raises(ValueError,match='expired'):
        RegisteredSupport(short).prepare(depth,k,camera,reg,None,now=159.1,parked=True)
    with pytest.raises(ValueError,match='bounded'):
        RegisteredSupport(replace(view.geometry,lifetime_s=100000.))
    almost_expired = replace(view.geometry,lifetime_s=158.2)
    current = RegisteredSupport(almost_expired).prepare(depth,k,camera,reg,None,now=159.1,parked=True)
    with pytest.raises(ValueError,match='expired'):
        current.validate(depth,k,camera,reg.evidence,159.3,True)


def test_no_clearance_is_not_implicit_vacancy():
    from scripts.twin.restock_manipulation.isaac_sensors import SlotView
    tracker,(k,camera,depth,_,_,a,b) = commissioned()
    reg = registration(k,camera,depth,a,b,stamp=2.,frame_id=2)
    view = tracker.prepare(depth,k,camera,reg,None,now=2.,parked=True)
    slot = SlotView.from_support('deck',view,np.ones(depth.shape,bool))
    assert slot.free_dimensions == (.62,.42,0.)


def test_oblique_quad_retains_physical_footprint():
    from scripts.twin.restock_camera.geometry import look_at
    k,_,_,_,_,_,_ = fixture(scale=2)
    camera = look_at([0,-.25,1.],[0,0,.2])
    v,u = np.mgrid[:k.height,:k.width]
    rays = np.stack([(u-k.cx)/k.fx,(v-k.cy)/k.fy,np.ones_like(u)],axis=-1)
    direction = rays @ camera[:3,:3].T
    depth = (.2-camera[2,3])/direction[:,:,2]
    a = marker_pixels([-.43,-.17,.2],.06,k,camera)
    b = marker_pixels([-.43,.17,.2],.06,k,camera)
    reg = registration(k,camera,depth,a,b)
    g = calibrate_rigid_support(depth,pixels(quad([0,0,.2],[.70,.50]),k,camera),
        pixels(quad([-.48,0,.2],[.08,.25]),k,camera),k,camera,reg,
        required_xy=(.62,.42),calibration_id='oblique',now=1.,parked=True)
    view = RegisteredSupport(g).prepare(depth,k,camera,
        registration(k,camera,depth,a,b,stamp=2.,frame_id=2),clearance(),now=2.,parked=True)
    np.testing.assert_allclose(view.center_base,[0,0,.2],atol=.002)
    assert view.geometry.required_xy == (.62,.42)


def test_executable_context_decodes_real_qr_pixels_before_clearance(monkeypatch):
    cv2 = pytest.importorskip('cv2')
    from scripts.twin.restock_camera.binding import RenderedSupportContext, SupportSlot
    k,camera,depth,support,witness,a,b = fixture(scale=2)
    gray = np.full(depth.shape,255,np.uint8)
    for code,destination in [('support:a',a),('support:b',b)]:
        qr = cv2.QRCodeEncoder_create().encode(code)
        qr = cv2.copyMakeBorder(qr,4,4,4,4,cv2.BORDER_CONSTANT,value=255)
        qr = cv2.resize(qr,(300,300),interpolation=cv2.INTER_NEAREST)
        decoded,corners,_ = cv2.QRCodeDetector().detectAndDecode(qr)
        assert decoded == code
        homography = cv2.getPerspectiveTransform(corners[0].astype(np.float32),destination.astype(np.float32))
        image = cv2.warpPerspective(qr,homography,(k.width,k.height),flags=cv2.INTER_NEAREST,borderValue=255)
        gray = np.minimum(gray,image)
    frame = dict(rgb=np.repeat(gray[:,:,None],3,axis=2),distance_to_image_plane=depth,
                 rendering_frame=1,rendering_time=1.)
    calls = []
    def clearance_callback(location,view,frame,now):
        calls.append((location,view.registration.evidence.stamp,view.geometry.baseline_stamp))
        return None
    slot = SupportSlot('cart','deck','support:a',.06,'bolted1',np.ones(depth.shape,bool),
                       auxiliary_id='support:b',auxiliary_offset_xy=(0,.34))
    binding = RenderedSupportContext(camera_id='camera',intrinsics=k,slots=[slot],
        measure_context=lambda frame,now:(camera,now,'park1',True),
        clearance_for_slot=clearance_callback,depth_error_m=.0002,camera_revision='lens1',
        steady_clock=lambda:0.)      # deterministic commission freshness, independent of decode wall time
    baseline = binding.commission(frame,1.,{'cart':(support,witness)})[0]
    assert not calls
    loaded = depth.copy()
    loaded[polygon_mask(support,k)] -= .16
    frame.update(distance_to_image_plane=loaded,rendering_frame=2,rendering_time=2.)
    current = binding(frame,2.)
    assert calls == [('cart',2.,1.)]
    assert current.slots[0].free_dimensions[2] == 0
    assert baseline.baseline_stamp == 1.
    # Inventory reuses exactly the marker decoder's frame result; it must not
    # pay for an independent decode or silently accept changed pixels.
    from types import SimpleNamespace as NS
    from scripts.twin.restock_manipulation import isaac_sensors
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr(isaac_sensors,'load_pose_bridge',lambda root:FixturePoseBridge())
    camera_stub=NS(get_current_frame=lambda **kw:frame,
        get_intrinsics_matrix=lambda:np.array([[k.fx,0,k.cx],[0,k.fy,k.cy],[0,0,1.]]),
        add_distance_to_image_plane_to_frame=lambda:None)
    sensor=isaac_sensors.IsaacCartonSensors(camera_stub,pose_bridge_root='fixture',profiles={},
                                          context_provider=lambda f,now:current)
    def forbidden_decode(*args,**kwargs):
        pytest.fail('support and inventory decoded the same acquired pixels twice')
    sensor.decoder=NS(decode=forbidden_decode)
    motion=NS(parked=lambda:True,collision_time=2.,joint_positions=lambda:(0.,)*6,
              observe_source=lambda observation:None)
    packet=sensor.capture(2.,'cart','cart',motion,NS(read=lambda:False))
    assert not packet.observations and not packet.target.clear
    assert sensor.evidence[0]['qr_rgb_sha256']==current.qr_observation.rgb_sha256
    # Removing real identity pixels refuses; geometry/config never supplies ID.
    frame.update(rgb=np.full_like(frame['rgb'],255),rendering_frame=3,rendering_time=3.)
    with pytest.raises(ValueError,match='not decoded'):
        binding(frame,3.)


def test_station_binding_aggregates_only_synchronized_assigned_cameras(monkeypatch):
    from types import SimpleNamespace as NS
    from scripts.twin.restock_camera.binding import StationSensors
    from scripts.twin.restock_manipulation import isaac_sensors
    from scripts.twin.restock_manipulation.types import Sensors, Target
    target = Target('depth:2','sensor',2.,'shelf',(0,0,.2,0,0,0,1),True,(.62,.42,.25))
    packets = {
        'cart':Sensors(2.,(),None,True,False,2.,2.,(0.,)*6,(),2.),
        'shelf':Sensors(2.,(),target,True,False,2.,2.,(0.,)*6,('shelf',),2.)}
    def adapter(camera,**kw):
        key = kw['camera_id']
        return NS(slots=(NS(location=key),),capture=lambda *args:packets[key],
                  sensed_target_bboxes=({'camera_id':key},),evidence=({'camera_id':key},))
    monkeypatch.setattr(isaac_sensors,'IsaacCartonSensors',adapter)
    binding = StationSensors(cameras={'cart':object(),'shelf':object()},profiles={},
        pose_bridge_root='fixture',context_providers={'cart':object(),'shelf':object()},
        location_cameras={'cart':'cart','shelf':'shelf'})
    result = binding.capture(2.,'cart','shelf',None,None)
    assert result.target is target and result.scanned_locations == ('shelf',)
    assert len(binding.sensed_target_bboxes) == 2
    assert not hasattr(binding,'mapper_ready')
    packets['cart'] = replace(packets['cart'],scan_time=1.9)
    with pytest.raises(ValueError,match='not synchronized'):
        binding.capture(2.,'cart','shelf',None,None)
    assert binding.sensed_target_bboxes == ()
    packets['cart'] = replace(packets['cart'],scan_time=2.,target=target)
    with pytest.raises(ValueError,match='one current'):
        binding.capture(2.,'cart','shelf',None,None)
