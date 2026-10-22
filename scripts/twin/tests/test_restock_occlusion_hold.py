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


def test_hold_tolerates_a_moved_support_camera_by_re_expressing_the_pose():
    # run-101: as the arm reaches, the support camera's base-relative pose changes; the hold must
    # re-express the held registration from live TF, not bail with 'support camera moved'.
    tracker,(k,camera,depth,support,_,a,b) = commissioned()
    clean = _clean(tracker, k, camera, depth, a, b, now=2., frame_id=2)
    assert not clean.occluded
    moved = camera.copy()
    moved[0,3] += 0.01                                  # camera shifted 1 cm relative to the base
    e = registration(k, camera, depth, a, b, stamp=3., frame_id=3).evidence
    held = tracker.hold(depth, k, moved, e, now=3., parked=True, occlusion_window_s=20.,
                        current_registration=None)
    assert held.occluded
    # base_from_support was re-expressed into the moved camera frame (not the frozen clean pose).
    np.testing.assert_allclose(np.asarray(held.base_from_camera), moved, atol=1e-12)
    assert not np.allclose(np.asarray(held.registration.base_from_support),
                           np.asarray(clean.registration.base_from_support))


def test_hold_accepts_the_repeated_held_frame_between_renders():
    # C1 (fm448 / run-331 cold collect): the source renders only every sensing_stride ticks, so
    # between renders hold() is re-polled with the SAME last frame. The held path re-expresses the
    # registration via live TF (consumes no fresh pixels), so an equal (non-regressing) frame_id must
    # HOLD, not raise 'rigid-support registration needs a new monotonic frame'.
    tracker,(k,camera,depth,support,_,a,b) = commissioned()
    _clean(tracker, k, camera, depth, a, b, now=2., frame_id=2)   # clean view seeds the hold
    e = registration(k, camera, depth, a, b, stamp=3., frame_id=3).evidence
    first = tracker.hold(depth, k, camera, e, now=3., parked=True, occlusion_window_s=20.,
                         current_registration=None)
    assert first.occluded
    # Re-poll with the IDENTICAL frame (same id/stamp) between renders: non-regressing -> holds.
    repeat = tracker.hold(depth, k, camera, e, now=3., parked=True, occlusion_window_s=20.,
                          current_registration=None)
    assert repeat.occluded
    np.testing.assert_allclose(repeat.center_base, first.center_base, atol=1e-12)


def test_hold_rejects_a_regressing_frame_id():
    # A frame_id that goes BACKWARDS (even with an advancing stamp) is still a regression and refuses.
    tracker,(k,camera,depth,support,_,a,b) = commissioned()
    _clean(tracker, k, camera, depth, a, b, now=5., frame_id=5)   # last_frame = frame 5
    regress = registration(k, camera, depth, a, b, stamp=6., frame_id=4).evidence
    with pytest.raises(ValueError, match='monotonic'):
        tracker.hold(depth, k, camera, regress, now=6., parked=True, occlusion_window_s=20.,
                     current_registration=None)


def test_hold_repeat_refused_once_the_frame_ages_past_the_window():
    # Relaxing the monotonic gate to non-regressing does NOT widen the occlusion window: a held frame
    # aged past the window since the last clean view still refuses.
    tracker,(k,camera,depth,support,_,a,b) = commissioned()
    _clean(tracker, k, camera, depth, a, b, now=2., frame_id=2)   # clean view at now=2
    e = registration(k, camera, depth, a, b, stamp=3., frame_id=3).evidence
    assert tracker.hold(depth, k, camera, e, now=3., parked=True, occlusion_window_s=2.0,
                        current_registration=None).occluded
    late = registration(k, camera, depth, a, b, stamp=4.5, frame_id=4).evidence   # 2.5 s after clean
    with pytest.raises(ValueError, match='occlusion window'):
        tracker.hold(depth, k, camera, late, now=4.5, parked=True, occlusion_window_s=2.0,
                     current_registration=None)


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


def _frame_with_unknown_label(frame, k, camera_pose, *, frame_id, stamp, text='mystery-sku'):
    """Paint an extra QR for a SKU that is not in the profiles at the slot centre: it lands in the
    calibrated slot's label mask and decodes as an unknown (or undecoded) label -> capture's
    'unknown or undecoded label in calibrated slot' site (isaac_sensors.py). The deck support
    markers stay where they are, so the support still registers. ``text`` overrides the painted code
    (e.g. an unregistered 'support:...' marker, which fm466 must skip rather than refuse)."""
    import cv2
    from .test_restock_commissioning import quad, pix
    gray = frame['rgb'][:,:,0].copy()
    code = cv2.copyMakeBorder(cv2.QRCodeEncoder_create().encode(text),4,4,4,4,cv2.BORDER_CONSTANT,value=255)
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


def test_unregistered_support_marker_in_slot_is_skipped_not_refused(tmp_path, monkeypatch):
    """fm466: a support's AUXILIARY registration marker (support:source:1) that the baseline did not
    register (it was marginal at a dimmer commission scale) decodes in the mission and lands in the
    calibrated slot's all-support label mask. A code in the reserved 'support:' namespace is a
    registration marker, never inventory, so capture must SKIP it -- NOT raise 'unknown or undecoded
    label in calibrated slot' as a genuine mystery SKU would (run-343 cold-collect trip). Inject the
    extra decoded marker through the decoder (a synthetic warp cannot render a cleanly decodable extra
    QR) so the candidate carries a real decoded 'support:*' text at the slot centre."""
    from dataclasses import replace
    from .test_restock_manipulation import FixturePoseBridge
    from scripts.twin.restock_camera.qr import FixedTileQR, QRDetection
    from .test_restock_commissioning import quad, pix
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path)
    corners = tuple(map(tuple, pix(quad([0, 0, .2], [.06, .06]), k, camera_pose)))

    class _Injecting:   # the real tiled decode (registers support:a/support:b) + an extra decoded marker
        def __init__(self): self.real = FixedTileQR('source')
        def decode(self, rgb, evidence, *, now, parked, roi=None):
            q = self.real.decode(rgb, evidence, now=now, parked=parked, roi=roi)
            return replace(q, detections=q.detections + (QRDetection('support:c', corners),))
    adapter.decoder = _Injecting()
    # First ever frame (no prior clean read): a genuine unknown SKU would strictly refuse here; the
    # unregistered 'support:' marker must be skipped instead, so the capture completes without the raise.
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)   # no 'unknown or undecoded'


def test_tile_seam_unreadable_partial_of_a_decoded_qr_is_dropped(tmp_path, monkeypatch):
    """fm467: the tiled decoder can see one physical QR in two tiles -- decoded whole in one, and an
    UNREADABLE partial quad at the adjacent tile seam. The partial overlaps the decoded quad, so it is
    the same QR seen twice and must be dropped, NOT raised as an unknown label (run-344 cold-collect
    trip on the bottom-right support marker's seam partial). A genuine occlusion unreadable quad (which
    overlaps NO decode) is still kept + handled by the window, as the mystery-sku test above asserts."""
    from dataclasses import replace
    from .test_restock_manipulation import FixturePoseBridge
    from scripts.twin.restock_camera.qr import FixedTileQR, QRDetection
    from .test_restock_commissioning import quad, pix
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path)
    corners = tuple(map(tuple, pix(quad([0, 0, .2], [.06, .06]), k, camera_pose)))

    class _Injecting:   # a decoded SKU-ish marker + an UNREADABLE partial of that SAME quad (tile seam)
        def __init__(self): self.real = FixedTileQR('source')
        def decode(self, rgb, evidence, *, now, parked, roi=None):
            q = self.real.decode(rgb, evidence, now=now, parked=parked, roi=roi)
            return replace(q, detections=q.detections + (QRDetection('support:c', corners),),
                           unreadable_corners=q.unreadable_corners + (np.asarray(corners),))
    adapter.decoder = _Injecting()
    # The unreadable partial overlaps the decoded quad -> dropped; no 'unknown or undecoded' raise even
    # on the first (never-clean) frame. Without the fm467 drop the kept partial (code='') would refuse.
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)


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


# --- sustained hold through the reach: a FRESH failed render-tick frame must engage the hold ----
# (run-100: the moving arm makes render-tick frames fail decode/registration; PR77 identical-frame
#  reuse cannot cover it, so /sensors must come from engaging the hold on the fresh failed frame.)

def _fail_decode_next(adapter, exc=ValueError('QR decoder refuses replayed camera frame/session')):
    """Make the boot context's decoder raise once (simulating the occluded arm garbling the frame)."""
    decoder = adapter.context_provider.decoder
    real = decoder.decode
    state = {'armed': True}
    def decode(*a, **kw):
        if state['armed']:
            state['armed'] = False
            raise exc
        return real(*a, **kw)
    decoder.decode = decode
    return state


def test_fresh_render_tick_decode_failure_engages_the_hold(tmp_path, monkeypatch):
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)   # clean observe seeds good

    # A FRESH frame (new id/stamp) whose decode RAISES under occlusion: hold must engage (no reuse
    # key matches it) and /sensors still publishes with the occluded flag.
    _fail_decode_next(adapter)
    camera.frame = dict(frame, rendering_frame=2, rendering_time=1.05)
    motion.collision_time = 1.05
    packet = adapter.capture(1.05, 'backstock_00', 'backstock_00', motion, contact)   # no raise
    assert packet is not None and packet.parked
    assert adapter.evidence[0]['registration_occluded'] is True


def test_sustained_hold_over_a_multi_tick_reach(tmp_path, monkeypatch):
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)

    # Each successive FRESH render-tick frame fails decode (the arm stays over the markers); every
    # one must hold and keep /sensors flowing -- not just an identical-frame reuse.
    for i, t in enumerate([1.05, 1.10, 1.15, 1.20], start=2):
        _fail_decode_next(adapter)
        camera.frame = dict(frame, rendering_frame=i, rendering_time=t)
        motion.collision_time = t
        packet = adapter.capture(t, 'backstock_00', 'backstock_00', motion, contact)
        assert packet is not None and adapter.evidence[0]['registration_occluded'] is True


def test_held_frame_stale_past_half_second_still_publishes_within_window(tmp_path, monkeypatch):
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path, window=20.)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    _fail_decode_next(adapter)
    camera.frame = dict(frame, rendering_frame=2, rendering_time=1.05)   # held render-tick frame
    motion.collision_time = 1.05
    adapter.capture(1.05, 'backstock_00', 'backstock_00', motion, contact)
    # Re-present that held frame 0.65 s later (> the 0.5 s live budget, < the 20 s hold window):
    # the frame-age / freshness gates must accept held output for the window.
    motion.collision_time = 1.70
    packet = adapter.capture(1.70, 'backstock_00', 'backstock_00', motion, contact)   # no raise
    assert packet is not None and adapter.evidence[0]['registration_occluded'] is True


def test_held_frame_refused_once_stale_past_the_window(tmp_path, monkeypatch):
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path, window=0.5)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    _fail_decode_next(adapter)
    camera.frame = dict(frame, rendering_frame=2, rendering_time=1.05)
    motion.collision_time = 1.05
    adapter.capture(1.05, 'backstock_00', 'backstock_00', motion, contact)
    motion.collision_time = 1.70
    with pytest.raises(ValueError):            # 0.65 s stale > 0.5 s window -> refuse
        adapter.capture(1.70, 'backstock_00', 'backstock_00', motion, contact)


def test_decode_failure_with_seeded_baseline_holds_on_the_commissioned_prior(tmp_path, monkeypatch):
    # Option B (fm443/fm444): the commissioned baseline carries the world-constant camera_from_support,
    # so apply_baselines SEEDS tracker.good. A first-frame decode failure under occlusion then re-expresses
    # that commissioned prior through live TF (hold) instead of strict-refusing -- the whole point of
    # seeding the hold from the baseline (pre-Option-B this was strict: there was no prior to fall back on).
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    _, adapter, camera, motion, contact, (k, witness, frame, camera_pose) = _source_rig(tmp_path)
    assert adapter.context_provider.slots[0].tracker.good is not None   # Option B seeded the prior at boot
    _fail_decode_next(adapter)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)   # holds on the seed, no raise


def test_decode_failure_without_a_seeded_prior_is_strict(tmp_path, monkeypatch):
    # Safety invariant preserved for the genuinely-no-prior case: a LEGACY baseline lacking
    # camera_from_support leaves tracker.good None (no seed), so a first-frame decode failure with nothing
    # clean to hold strict-refuses, exactly as before Option B.
    from dataclasses import replace
    from scripts.twin.restock_camera import commissioning as C
    from .test_restock_commissioning import empty_frame, context
    from .test_restock_manipulation import FixturePoseBridge
    from scripts.twin.restock_manipulation import isaac_sensors
    monkeypatch.setattr('scripts.twin.restock_manipulation.isaac_sensors.load_pose_bridge',
                        lambda root: FixturePoseBridge())
    k, camera_pose, depth, support, witness, frame = empty_frame()
    ctx = context(k, camera_pose, location='backstock_00', camera_id='source')
    C.commission_and_persist(ctx, frame, 1., {'backstock_00': (support, witness)}, dataset_dir=tmp_path)
    loaded = C.load_baselines(tmp_path)
    legacy = {loc: replace(rec, geometry=replace(rec.geometry, camera_from_support=None))
              for loc, rec in loaded.items()}            # strip the pose -> no Option-B seed
    boot = context(k, camera_pose, location='backstock_00', camera_id='source')
    C.apply_baselines(boot, legacy, required_locations=['backstock_00'])
    boot.occlusion_window_s = 20.
    assert boot.slots[0].tracker.good is None             # unseeded: nothing to hold
    adapter = isaac_sensors.IsaacCartonSensors(_Camera(frame, k), pose_bridge_root='fixture', profiles={},
                                               context_provider=boot, camera_id='source',
                                               steady_clock=lambda: 0.)
    motion = NS(parked=lambda: True, collision_time=1., joint_positions=lambda: (0.,)*6,
                observe_source=lambda o: None)
    _fail_decode_next(adapter)
    with pytest.raises(ValueError):
        adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, NS(read=lambda: False))
