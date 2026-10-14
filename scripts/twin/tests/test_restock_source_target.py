"""CPU tests for the fixed-source collect-target provider. No Isaac/ROS, no GPU."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.twin.restock_camera.source_target import build_source_target_provider, SourceTarget
from scripts.twin.restock_manipulation.types import Sensors, Observation


def _evidence(location='backstock_00', calibration_id='sess1:source:7:source'):
    return dict(location=location, baseline_calibration_id=calibration_id, baseline_stamp=1.,
                baseline_session_id='sess1', baseline_acquired_steady=None, current_frame_id=7,
                current_stamp=2., epoch='park1', registration_uncertainty_m=.003,
                clearance_observation_id=None, qr_processing_seconds=.08, qr_rgb_sha256='abc123',
                provenance='dated_rigid_geometry_with_fresh_marker_witness_and_depth')


def _adapter(observations, *, evidence=None, camera_id='source'):
    packet = Sensors(2., tuple(observations), None, True, False, 2., 2., (0.,)*6, (), 2.)
    captured = {}
    def capture(now, source, destination, motion, contact):
        captured['args'] = (now, source, destination, motion, contact)
        return packet
    return NS(capture=capture, evidence=tuple(evidence or [_evidence()]), camera_id=camera_id), captured


def _carton(location='backstock_00', sku='juice_orange', source='sensor'):
    return Observation('camera:7:qr:0', source, 2., sku, location, (-.42,4.9,.81,0.,0.,.38,.92), 1.)


# --- provider extraction / provenance -----------------------------------------

def test_provider_returns_station_frame_pose_sku_stamp_and_baseline_provenance():
    adapter, captured = _adapter([_carton()])
    provider = build_source_target_provider(adapter, motion=NS(), contact=NS())
    target = provider(2.)
    assert isinstance(target, SourceTarget)
    assert target.sku == 'juice_orange' and target.location == 'backstock_00'
    assert target.station_frame == 'map' and target.pose_station == (-.42,4.9,.81,0.,0.,.38,.92)
    assert target.stamp == 2.
    assert target.provenance['camera_id'] == 'source' and target.provenance['source'] == 'sensor'
    assert target.provenance['calibration_id'] == 'sess1:source:7:source'
    assert target.provenance['baseline_session_id'] == 'sess1' and target.provenance['frame_id'] == 7
    assert target.provenance['epoch'] == 'park1' and target.provenance['qr_rgb_sha256'] == 'abc123'
    # The provider wraps capture(now, 'backstock_00', 'backstock_00', motion, contact).
    assert captured['args'][:3] == (2., 'backstock_00', 'backstock_00')


def test_provider_returns_none_when_no_handling_unit_is_sensed():
    adapter, _ = _adapter([])                                 # empty source station this tick
    provider = build_source_target_provider(adapter, motion=NS(), contact=NS())
    assert provider(2.) is None                               # absence is never a target


def test_provider_refuses_a_fixture_sourced_observation():
    adapter, _ = _adapter([_carton(source='test_fixture')])
    provider = build_source_target_provider(adapter, motion=NS(), contact=NS())
    with pytest.raises(ValueError, match='sensor-sourced, never a fixture'):
        provider(2.)


def test_provider_refuses_more_than_one_unit_at_the_source():
    adapter, _ = _adapter([_carton(), _carton()])
    provider = build_source_target_provider(adapter, motion=NS(), contact=NS())
    with pytest.raises(ValueError, match='more than one handling unit'):
        provider(2.)


def test_provider_ignores_observations_at_other_locations():
    adapter, _ = _adapter([_carton(location='rollwagen')], evidence=[_evidence()])
    provider = build_source_target_provider(adapter, motion=NS(), contact=NS())
    assert provider(2.) is None


# --- build_sensing wiring -----------------------------------------------------

def test_build_sensing_exposes_source_target_provider_when_configured(monkeypatch):
    from scripts.twin.restock_manipulation import isaac_sensors
    from scripts.twin.restock_camera.sensing import build_sensing
    adapters = {}
    def fake(camera, **kw):
        adapter, _ = _adapter([_carton()] if kw['camera_id'] == 'source' else [])
        adapter = NS(capture=adapter.capture, evidence=adapter.evidence, camera_id=kw['camera_id'], slots=())
        adapters[kw['camera_id']] = adapter
        return adapter
    monkeypatch.setattr(isaac_sensors, 'IsaacCartonSensors', fake)
    sources = dict(cameras={'source': object()}, context_providers={'source': object()},
                   location_cameras={'backstock_00': 'source'}, profiles={},
                   pose_bridge_root='fixture', motion=NS(), contact=NS(), recording_camera='cam')
    built = build_sensing(app=None, assembly=NS(sensing_sources=sources), plan=None,
                          config={'collect_source_location': 'backstock_00'}, output=None)
    assert 'source_target_provider' in built
    target = built['source_target_provider'](2.)
    assert target.sku == 'juice_orange' and target.station_frame == 'map'
    # Absent unless configured.
    plain = build_sensing(app=None, assembly=NS(sensing_sources=sources), plan=None, config={}, output=None)
    assert 'source_target_provider' not in plain


# --- end to end over a real commissioned (reloaded-baseline) source context ----

def test_provider_over_a_real_commissioned_source_returns_none_on_empty(tmp_path, monkeypatch):
    """Wrap a real fixed-source IsaacCartonSensors whose source context carries the RELOADED
    backstock_00 baseline. An empty source frame yields no handling unit -> None, and the
    reloaded baseline's calibration_id appears in the capture evidence (baseline in the path)."""
    from scripts.twin.restock_manipulation import isaac_sensors
    from scripts.twin.restock_camera import commissioning as C
    from .test_restock_commissioning import empty_frame, context
    from .test_restock_manipulation import FixturePoseBridge
    k, camera, depth, support, witness, frame = empty_frame()
    # Commission backstock_00, persist, reload, and apply to a fresh source context.
    commissioned = context(k, camera, location='backstock_00', camera_id='source')
    C.commission_and_persist(commissioned, frame, 1., {'backstock_00': (support, witness)},
                             dataset_dir=tmp_path)
    boot = context(k, camera, location='backstock_00', camera_id='source')
    C.apply_baselines(boot, C.load_baselines(tmp_path), required_locations=['backstock_00'])
    baseline_cal = boot.slots[0].tracker.geometry.calibration_id

    monkeypatch.setattr(isaac_sensors, 'load_pose_bridge', lambda root: FixturePoseBridge())
    camera_stub = NS(get_current_frame=lambda **kw: frame,
                     get_intrinsics_matrix=lambda: np.array([[k.fx,0,k.cx],[0,k.fy,k.cy],[0,0,1.]]),
                     add_distance_to_image_plane_to_frame=lambda: None)
    adapter = isaac_sensors.IsaacCartonSensors(camera_stub, pose_bridge_root='fixture', profiles={},
                                               context_provider=boot, camera_id='source')
    motion = NS(parked=lambda: True, collision_time=1., joint_positions=lambda: (0.,)*6,
                observe_source=lambda observation: None)
    provider = build_source_target_provider(adapter, motion, NS(read=lambda: False))
    assert provider(1.) is None                               # empty support: no pick carton
    evidence = next(e for e in adapter.evidence if e['location'] == 'backstock_00')
    assert evidence['baseline_calibration_id'] == baseline_cal   # reloaded baseline is in the path
