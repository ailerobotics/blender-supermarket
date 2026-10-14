"""CPU tests for the fixed-shelf transfer providers: slot target + post-placement verify.

No Isaac/ROS, no GPU. Fake adapters return canned ``Sensors`` packets for the extraction /
provenance / tolerance cases; the end-to-end case wraps a real fixed-shelf
``IsaacCartonSensors`` whose shelf context carries the RELOADED ``G3_back_s2`` baseline.
"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.twin.restock_camera.shelf_target import (
    build_shelf_target_provider, build_shelf_verification_provider,
    ShelfSlotTarget, ShelfVerification)
from scripts.twin.restock_manipulation.types import Sensors, Observation, Target


def _evidence(location='G3_back_s2', calibration_id='sess1:shelf:7:shelf'):
    return dict(location=location, baseline_calibration_id=calibration_id, baseline_stamp=1.,
                baseline_session_id='sess1', baseline_acquired_steady=None, current_frame_id=7,
                current_stamp=2., epoch='park1', registration_uncertainty_m=.003,
                clearance_observation_id='clr:1', qr_processing_seconds=.08, qr_rgb_sha256='abc123',
                provenance='dated_rigid_geometry_with_fresh_marker_witness_and_depth')


def _adapter(observations=(), target=None, *, evidence=None, camera_id='shelf'):
    packet = Sensors(2., tuple(observations), target, True, False, 2., 2., (0.,)*6, (), 2.)
    captured = {}
    def capture(now, source, destination, motion, contact):
        captured['args'] = (now, source, destination, motion, contact)
        return packet
    return NS(capture=capture, evidence=tuple(evidence or [_evidence()]), camera_id=camera_id), captured


def _slot(location='G3_back_s2', clear=True, source='sensor', yaw=0.):
    return Target('camera:7:slot:G3_back_s2', source, 2., location,
                  (-3.1, 6.05, 1.21, 0., 0., math.sin(yaw/2), math.cos(yaw/2)), clear, (.62, .42, .30))


def _carton(location='G3_back_s2', sku='juice_orange', source='sensor', pose=None):
    return Observation('camera:7:qr:0', source, 2., sku, location,
                       pose or (-3.1, 6.05, 1.33, 0., 0., 0., 1.), 1.)


# --- transfer slot target provider --------------------------------------------

def test_slot_provider_returns_station_frame_pose_free_dims_clear_stamp_provenance():
    adapter, captured = _adapter(target=_slot())
    provider = build_shelf_target_provider(adapter, motion=NS(), contact=NS())
    slot = provider(2.)
    assert isinstance(slot, ShelfSlotTarget)
    assert slot.location == 'G3_back_s2' and slot.station_frame == 'map'
    assert slot.pose_station == (-3.1, 6.05, 1.21, 0., 0., 0., 1.)
    assert slot.free_dimensions == (.62, .42, .30) and slot.clear is True and slot.stamp == 2.
    assert slot.provenance['camera_id'] == 'shelf' and slot.provenance['source'] == 'sensor'
    assert slot.provenance['calibration_id'] == 'sess1:shelf:7:shelf'
    assert slot.provenance['baseline_session_id'] == 'sess1' and slot.provenance['frame_id'] == 7
    assert slot.provenance['epoch'] == 'park1' and slot.provenance['qr_rgb_sha256'] == 'abc123'
    # The provider wraps capture(now, 'G3_back_s2', 'G3_back_s2', motion, contact).
    assert captured['args'][:3] == (2., 'G3_back_s2', 'G3_back_s2')


def test_slot_provider_reports_non_clear_slot_for_the_consumer_to_gate():
    # Sensing is faithful: an occupied slot is still reported, with clear=False — the consumer
    # (controller) refuses to place into a non-clear slot; the provider does not hide it.
    adapter, _ = _adapter(target=_slot(clear=False))
    slot = build_shelf_target_provider(adapter, motion=NS(), contact=NS())(2.)
    assert slot is not None and slot.clear is False


def test_slot_provider_returns_none_when_no_slot_is_sensed():
    adapter, _ = _adapter(target=None)
    assert build_shelf_target_provider(adapter, motion=NS(), contact=NS())(2.) is None


def test_slot_provider_ignores_a_target_at_another_location():
    adapter, _ = _adapter(target=_slot(location='rollwagen'))
    assert build_shelf_target_provider(adapter, motion=NS(), contact=NS())(2.) is None


def test_slot_provider_refuses_a_fixture_sourced_target():
    adapter, _ = _adapter(target=_slot(source='test_fixture'))
    with pytest.raises(ValueError, match='sensor-sourced, never a fixture'):
        build_shelf_target_provider(adapter, motion=NS(), contact=NS())(2.)


# --- post-placement verification provider -------------------------------------

def test_verify_confirms_expected_carton_within_tolerance():
    adapter, captured = _adapter(observations=[_carton(pose=(-3.1, 6.05, 1.33, 0., 0., 0., 1.))])
    verify = build_shelf_verification_provider(adapter, motion=NS(), contact=NS())
    result = verify(2., 'juice_orange', (-3.11, 6.06, 1.33, 0., 0., 0., 1.))
    assert isinstance(result, ShelfVerification)
    assert result.present is True and result.within_tolerance is True
    assert result.observed_sku == 'juice_orange' and result.observed_pose[:3] == (-3.1, 6.05, 1.33)
    assert result.position_error_m < .03 and result.yaw_error_rad < .01
    assert result.stamp == 2. and result.provenance['calibration_id'] == 'sess1:shelf:7:shelf'
    assert captured['args'][:3] == (2., 'G3_back_s2', 'G3_back_s2')


def test_verify_absent_carton_is_present_false_never_fabricated():
    adapter, _ = _adapter(observations=[])
    result = build_shelf_verification_provider(adapter, motion=NS(), contact=NS())(2., 'juice_orange',
                                                                                   (-3.1, 6.05, 1.33, 0., 0., 0., 1.))
    assert result.present is False and result.within_tolerance is False
    assert result.observed_sku is None and result.observed_pose is None and result.provenance is None


def test_verify_wrong_sku_distinguishes_a_mis_stock_from_empty():
    adapter, _ = _adapter(observations=[_carton(sku='soda_cola')])
    result = build_shelf_verification_provider(adapter, motion=NS(), contact=NS())(2., 'juice_orange',
                                                                                   (-3.1, 6.05, 1.33, 0., 0., 0., 1.))
    assert result.present is False and result.within_tolerance is False
    assert result.observed_sku == 'soda_cola'            # a wrong carton is surfaced, not hidden


def test_verify_position_out_of_tolerance_is_present_but_not_within():
    adapter, _ = _adapter(observations=[_carton(pose=(-3.1, 6.05, 1.33, 0., 0., 0., 1.))])
    verify = build_shelf_verification_provider(adapter, motion=NS(), contact=NS())
    result = verify(2., 'juice_orange', (-3.1, 6.15, 1.33, 0., 0., 0., 1.))   # 10 cm off
    assert result.present is True and result.within_tolerance is False
    assert result.position_error_m > .03


def test_verify_yaw_out_of_tolerance_is_present_but_not_within():
    adapter, _ = _adapter(observations=[_carton(pose=(-3.1, 6.05, 1.33, 0., 0., math.sin(.3), math.cos(.3)))])
    verify = build_shelf_verification_provider(adapter, motion=NS(), contact=NS())
    result = verify(2., 'juice_orange', (-3.1, 6.05, 1.33, 0., 0., 0., 1.))   # ~0.6 rad yaw error
    assert result.present is True and result.within_tolerance is False
    assert result.yaw_error_rad > .26


def test_verify_refuses_a_fixture_sourced_carton():
    adapter, _ = _adapter(observations=[_carton(source='test_fixture')])
    with pytest.raises(ValueError, match='sensor-sourced, never a fixture'):
        build_shelf_verification_provider(adapter, motion=NS(), contact=NS())(2., 'juice_orange',
                                                                              (-3.1, 6.05, 1.33, 0., 0., 0., 1.))


def test_verify_refuses_more_than_one_carton_in_the_slot():
    adapter, _ = _adapter(observations=[_carton(), _carton()])
    with pytest.raises(ValueError, match='more than one handling unit'):
        build_shelf_verification_provider(adapter, motion=NS(), contact=NS())(2., 'juice_orange',
                                                                              (-3.1, 6.05, 1.33, 0., 0., 0., 1.))


# --- build_sensing wiring -----------------------------------------------------

def test_build_sensing_exposes_shelf_providers_when_configured(monkeypatch):
    from scripts.twin.restock_manipulation import isaac_sensors
    from scripts.twin.restock_camera.sensing import build_sensing
    def fake(camera, **kw):
        adapter, _ = _adapter(target=_slot(), observations=[_carton()], camera_id=kw['camera_id'])
        return NS(capture=adapter.capture, evidence=adapter.evidence, camera_id=kw['camera_id'], slots=())
    monkeypatch.setattr(isaac_sensors, 'IsaacCartonSensors', fake)
    sources = dict(cameras={'shelf': object()}, context_providers={'shelf': object()},
                   location_cameras={'G3_back_s2': 'shelf'}, profiles={},
                   pose_bridge_root='fixture', motion=NS(), contact=NS(), recording_camera='cam')
    built = build_sensing(app=None, assembly=NS(sensing_sources=sources), plan=None,
                          config={'transfer_shelf_location': 'G3_back_s2'}, output=None)
    assert 'shelf_target_provider' in built and 'shelf_verification_provider' in built
    assert built['shelf_target_provider'](2.).location == 'G3_back_s2'
    assert built['shelf_verification_provider'](2., 'juice_orange', (-3.1, 6.05, 1.33, 0., 0., 0., 1.)).present
    # Absent unless configured.
    plain = build_sensing(app=None, assembly=NS(sensing_sources=sources), plan=None, config={}, output=None)
    assert 'shelf_target_provider' not in plain and 'shelf_verification_provider' not in plain


def test_build_sensing_refuses_shelf_location_without_an_assigned_camera(monkeypatch):
    from scripts.twin.restock_manipulation import isaac_sensors
    from scripts.twin.restock_camera.sensing import build_sensing
    monkeypatch.setattr(isaac_sensors, 'IsaacCartonSensors',
                        lambda camera, **kw: NS(capture=None, evidence=(), camera_id=kw['camera_id'], slots=()))
    sources = dict(cameras={'shelf': object()}, context_providers={'shelf': object()},
                   location_cameras={'G3_back_s2': 'shelf'}, profiles={},
                   pose_bridge_root='fixture', motion=NS(), contact=NS(), recording_camera='cam')
    with pytest.raises(ValueError, match='has no assigned camera'):
        build_sensing(app=None, assembly=NS(sensing_sources=sources), plan=None,
                      config={'transfer_shelf_location': 'nonexistent_slot'}, output=None)


# --- end to end over a real commissioned (reloaded-baseline) shelf context -----

def test_slot_provider_over_a_real_commissioned_shelf_reports_the_slot(tmp_path, monkeypatch):
    """Wrap a real fixed-shelf IsaacCartonSensors whose shelf context carries the RELOADED
    G3_back_s2 baseline. An empty-support frame still yields the slot target (pose + free dims)
    in the station frame, and the reloaded baseline's calibration_id appears in the evidence."""
    from scripts.twin.restock_manipulation import isaac_sensors
    from scripts.twin.restock_camera import commissioning as C
    from .test_restock_commissioning import empty_frame, context
    from .test_restock_manipulation import FixturePoseBridge
    k, camera, depth, support, witness, frame = empty_frame()
    # Commission G3_back_s2 on the shelf camera, persist, reload, apply to a fresh shelf context.
    commissioned = context(k, camera, location='G3_back_s2', camera_id='shelf')
    C.commission_and_persist(commissioned, frame, 1., {'G3_back_s2': (support, witness)},
                             dataset_dir=tmp_path)
    boot = context(k, camera, location='G3_back_s2', camera_id='shelf')
    C.apply_baselines(boot, C.load_baselines(tmp_path), required_locations=['G3_back_s2'])
    baseline_cal = boot.slots[0].tracker.geometry.calibration_id

    monkeypatch.setattr(isaac_sensors, 'load_pose_bridge', lambda root: FixturePoseBridge())
    camera_stub = NS(get_current_frame=lambda **kw: frame,
                     get_intrinsics_matrix=lambda: np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1.]]),
                     add_distance_to_image_plane_to_frame=lambda: None)
    adapter = isaac_sensors.IsaacCartonSensors(camera_stub, pose_bridge_root='fixture', profiles={},
                                               context_provider=boot, camera_id='shelf',
                                               steady_clock=lambda: 0.)
    motion = NS(parked=lambda: True, collision_time=1., joint_positions=lambda: (0.,)*6,
                observe_source=lambda observation: None)
    provider = build_shelf_target_provider(adapter, motion, NS(read=lambda: False))
    slot = provider(1.)
    assert slot is not None and slot.location == 'G3_back_s2' and slot.station_frame == 'map'
    assert len(slot.pose_station) == 7 and np.isfinite(slot.pose_station).all()
    assert len(slot.free_dimensions) == 3
    assert slot.provenance['calibration_id'] == baseline_cal       # reloaded baseline is in the path
