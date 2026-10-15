"""Approach-occlusion tolerance for the fixed-camera support registration (CPU; no Isaac/GPU).

During a grasp the reaching arm occludes the FIXED source camera's view of the support it grasps
toward, so a fresh full-footprint registration is briefly unavailable. The capture must keep
publishing the last clean registration (flagged occluded) instead of refusing and starving the host
sensor-stream watchdog -- but only within a bounded window, never when the visible markers
contradict the held pose, and never for observe/commissioning (which stay strict). Analytic
fixtures; not a renderer/physics demonstration.
"""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.twin.restock_camera.sensed import polygon_mask
from .test_restock_sensed_views import commissioned, registration, clearance


def _clean(tracker, k, camera, depth, a, b, *, now, frame_id, window=20.):
    reg = registration(k, camera, depth, a, b, stamp=now, frame_id=frame_id)
    return tracker.prepare(depth, k, camera, reg, clearance(now), now=now, parked=True,
                           occlusion_window_s=window)


def _occluded_depth(depth, support, k):
    occ = depth.copy()
    occ[polygon_mask(support, k)] = np.nan   # arm hides the support footprint; markers stay visible
    return occ


# --- prepare-level hold --------------------------------------------------------

def test_footprint_occlusion_holds_the_last_clean_registration():
    tracker,(k,camera,depth,support,_,a,b) = commissioned()
    clean = _clean(tracker, k, camera, depth, a, b, now=2., frame_id=2)
    assert not clean.occluded

    occ = _occluded_depth(depth, support, k)
    reg = registration(k, camera, occ, a, b, stamp=3., frame_id=3)   # markers still register
    held = tracker.prepare(occ, k, camera, reg, None, now=3., parked=True, occlusion_window_s=20.)
    assert held.occluded
    np.testing.assert_allclose(held.center_base, clean.center_base, atol=1e-9)
    assert held.support_z == clean.support_z and held.yaw_rad == clean.yaw_rad


def test_strict_mode_still_refuses_occlusion_when_window_is_none():
    tracker,(k,camera,depth,support,_,a,b) = commissioned()
    _clean(tracker, k, camera, depth, a, b, now=2., frame_id=2)
    occ = _occluded_depth(depth, support, k)
    reg = registration(k, camera, occ, a, b, stamp=3., frame_id=3)
    with pytest.raises(ValueError):           # occlusion_window_s=None => historical strict refusal
        tracker.prepare(occ, k, camera, reg, None, now=3., parked=True, occlusion_window_s=None)


def test_first_frame_cannot_hold_without_an_established_clean_view():
    tracker,(k,camera,depth,support,_,a,b) = commissioned()
    occ = _occluded_depth(depth, support, k)          # occluded from the very first observe frame
    reg = registration(k, camera, occ, a, b, stamp=2., frame_id=2)
    with pytest.raises(ValueError):                   # self.good is None -> strict refusal (observe)
        tracker.prepare(occ, k, camera, reg, None, now=2., parked=True, occlusion_window_s=20.)


def test_hold_refused_beyond_the_configurable_window():
    tracker,(k,camera,depth,support,_,a,b) = commissioned()
    _clean(tracker, k, camera, depth, a, b, now=2., frame_id=2)
    occ = _occluded_depth(depth, support, k)
    reg = registration(k, camera, occ, a, b, stamp=25., frame_id=3)   # 23 s after the clean view
    with pytest.raises(ValueError, match='occlusion window'):
        tracker.prepare(occ, k, camera, reg, None, now=25., parked=True, occlusion_window_s=20.)


def test_hold_refused_when_visible_markers_contradict_the_held_pose():
    tracker,(k,camera,depth,support,_,a,b) = commissioned()
    clean = _clean(tracker, k, camera, depth, a, b, now=2., frame_id=2)
    occ = _occluded_depth(depth, support, k)
    reg = registration(k, camera, occ, a, b, stamp=3., frame_id=3)
    shifted = np.asarray(reg.base_from_support).copy()
    shifted[0,3] += 0.20                               # the support appears to have moved 20 cm
    moved = replace(reg, base_from_support=tuple(map(tuple, shifted)))
    with pytest.raises(ValueError, match='contradict'):
        tracker.prepare(occ, k, camera, moved, None, now=3., parked=True, occlusion_window_s=20.)


def test_a_disturbed_support_can_no_longer_be_held():
    tracker,(k,camera,depth,support,_,a,b) = commissioned()
    _clean(tracker, k, camera, depth, a, b, now=2., frame_id=2)
    tracker.invalidate('marker bracket was disturbed')
    occ = _occluded_depth(depth, support, k)
    reg = registration(k, camera, occ, a, b, stamp=3., frame_id=3)
    with pytest.raises(ValueError):                    # invalidate() clears the held good view
        tracker.prepare(occ, k, camera, reg, None, now=3., parked=True, occlusion_window_s=20.)


def test_held_view_still_obeys_the_baseline_lifetime():
    tracker,(k,camera,depth,support,_,a,b) = commissioned()
    _clean(tracker, k, camera, depth, a, b, now=2., frame_id=2)
    occ = _occluded_depth(depth, support, k)
    reg = registration(k, camera, occ, a, b, stamp=602., frame_id=3)   # past the 600 s lifetime
    with pytest.raises(ValueError, match='expired or reversed'):
        tracker.prepare(occ, k, camera, reg, None, now=602., parked=True, occlusion_window_s=10000.)


# --- end to end over a real commissioned source: /sensors keeps flowing under occlusion ----

class _Camera:
    def __init__(self, frame, k):
        self.frame, self.k = frame, k

    def get_current_frame(self, **kw):
        return self.frame

    def get_intrinsics_matrix(self):
        k = self.k
        return np.array([[k.fx,0,k.cx],[0,k.fy,k.cy],[0,0,1.]])

    def add_distance_to_image_plane_to_frame(self):
        pass


def _source_rig(tmp_path, window=20.):
    from scripts.twin.restock_manipulation import isaac_sensors
    from scripts.twin.restock_camera import commissioning as C
    from .test_restock_commissioning import empty_frame, context
    from .test_restock_manipulation import FixturePoseBridge
    k, camera_pose, depth, support, witness, frame = empty_frame()
    commissioned_ctx = context(k, camera_pose, location='backstock_00', camera_id='source')
    C.commission_and_persist(commissioned_ctx, frame, 1., {'backstock_00': (support, witness)},
                             dataset_dir=tmp_path)
    boot = context(k, camera_pose, location='backstock_00', camera_id='source')
    C.apply_baselines(boot, C.load_baselines(tmp_path), required_locations=['backstock_00'])
    boot.occlusion_window_s = window                  # enable approach-occlusion tolerance
    camera = _Camera(frame, k)
    adapter = isaac_sensors.IsaacCartonSensors(camera, pose_bridge_root='fixture', profiles={},
                                               context_provider=boot, camera_id='source',
                                               steady_clock=lambda: 0.)
    motion = NS(parked=lambda: True, collision_time=1., joint_positions=lambda: (0.,)*6,
                observe_source=lambda o: None)
    return isaac_sensors, adapter, camera, motion, NS(read=lambda: False), (k, witness, frame, camera_pose)


def _frame_with_unknown_label(frame, k, camera_pose, *, frame_id, stamp):
    """Paint an extra QR for a SKU that is not in the profiles at the slot centre: it lands in the
    calibrated slot's label mask and decodes as an unknown (or undecoded) label -> capture's
    'unknown or undecoded label in calibrated slot' site (isaac_sensors.py). The deck support
    markers stay where they are, so the support still registers."""
    import cv2
    from .test_restock_commissioning import quad, pix
    gray = frame['rgb'][:,:,0].copy()
    code = cv2.copyMakeBorder(cv2.QRCodeEncoder_create().encode('mystery-sku'),4,4,4,4,cv2.BORDER_CONSTANT,value=255)
    code = cv2.resize(code,(220,220),interpolation=cv2.INTER_NEAREST)
    _,corners,_ = cv2.QRCodeDetector().detectAndDecode(code)
    dest = pix(quad([0,0,.2],[.12,.12]),k,camera_pose)
    homography = cv2.getPerspectiveTransform(corners[0].astype(np.float32),dest.astype(np.float32))
    gray = np.minimum(gray, cv2.warpPerspective(code,homography,(k.width,k.height),
                                                flags=cv2.INTER_NEAREST,borderValue=255))
    return dict(frame, rgb=np.repeat(gray[:,:,None],3,axis=2), rendering_frame=frame_id, rendering_time=stamp)


def test_capture_keeps_publishing_under_approach_occlusion(tmp_path, monkeypatch):
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path)

    clean = adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    assert adapter.evidence[0]['registration_occluded'] is False

    # The reaching arm occludes the support footprint depth; the markers remain decodable.
    occ = frame['distance_to_image_plane'].copy()
    occ[polygon_mask(witness, k)] = np.nan  # arm hides the independent witness patch
    camera.frame = dict(frame, distance_to_image_plane=occ, rendering_frame=2, rendering_time=1.1)
    motion.collision_time = 1.1
    packet = adapter.capture(1.10, 'backstock_00', 'backstock_00', motion, contact)   # no raise
    assert packet is not None and packet.parked
    assert adapter.evidence[0]['registration_occluded'] is True   # held registration, flagged


def test_capture_refuses_once_occlusion_exceeds_the_window(tmp_path, monkeypatch):
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path, window=0.5)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    occ = frame['distance_to_image_plane'].copy()
    occ[polygon_mask(witness, k)] = np.nan  # arm hides the independent witness patch
    camera.frame = dict(frame, distance_to_image_plane=occ, rendering_frame=2, rendering_time=1.55)
    motion.collision_time = 1.55
    with pytest.raises(ValueError, match='occlusion window'):      # 0.55 s since clean > 0.5 s window
        adapter.capture(1.55, 'backstock_00', 'backstock_00', motion, contact)


# --- undecoded/unknown label in a calibrated slot (gripper over the carton top) -----------------

def test_undecoded_label_in_slot_is_held_within_the_window(tmp_path, monkeypatch):
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)   # clean read seeds window
    assert adapter.evidence[0]['registration_occluded'] is False

    # The descending gripper occludes the carton top; its label decodes as an unknown SKU.
    camera.frame = _frame_with_unknown_label(frame, k, camera_pose, frame_id=2, stamp=1.10)
    motion.collision_time = 1.10
    packet = adapter.capture(1.10, 'backstock_00', 'backstock_00', motion, contact)   # no raise
    assert packet is not None and packet.parked
    assert adapter.evidence[0]['registration_occluded'] is True


def test_undecoded_label_refused_without_a_prior_clean_read(tmp_path, monkeypatch):
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path)
    # First ever frame already shows an undecoded label: nothing clean to hold -> strict refusal.
    camera.frame = _frame_with_unknown_label(frame, k, camera_pose, frame_id=2, stamp=1.10)
    motion.collision_time = 1.10
    with pytest.raises(ValueError, match='unknown or undecoded'):
        adapter.capture(1.10, 'backstock_00', 'backstock_00', motion, contact)


def test_undecoded_label_refused_once_the_window_expires(tmp_path, monkeypatch):
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path, window=0.5)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)   # clean read at t=1.0
    camera.frame = _frame_with_unknown_label(frame, k, camera_pose, frame_id=2, stamp=1.55)
    motion.collision_time = 1.55
    with pytest.raises(ValueError, match='unknown or undecoded'):            # 0.55 s > 0.5 s window
        adapter.capture(1.55, 'backstock_00', 'backstock_00', motion, contact)


def test_strict_mode_refuses_an_undecoded_label_every_frame(tmp_path, monkeypatch):
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    # occlusion_window_s=None => the slot label must decode every frame (historical behaviour).
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path, window=None)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    camera.frame = _frame_with_unknown_label(frame, k, camera_pose, frame_id=2, stamp=1.10)
    motion.collision_time = 1.10
    with pytest.raises(ValueError, match='unknown or undecoded'):
        adapter.capture(1.10, 'backstock_00', 'backstock_00', motion, contact)
