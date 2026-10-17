"""Offline tests for the GPU-detector bridge (pure decode + handler; no rclpy)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans.detector_bridge import HumanSafetyBridge, safety_state_from_json
from twin.humans.perception import DistanceStatus, Zone


def _state_json(zone, dist_base, stamp, tracks=None):
    return json.dumps({
        'schema_version': 1, 'frame_id': 'base_link', 'stamp': stamp,
        'evidence_stamp': stamp, 'heartbeat_seq': 3, 'zone': zone,
        'distance_status': 'known' if dist_base is not None else 'none',
        'nearest_distance': None if dist_base is None else max(0.0, dist_base - 1.3),
        'nearest_distance_base': dist_base, 'nearest_track_id': 1 if tracks else None,
        'any_person_visible': bool(tracks), 'reasons': ['test'],
        'cameras': [], 'tracks': tracks or []})


def test_decode_core_fields():
    s = safety_state_from_json(_state_json('STOP', 1.4, 10.0))
    assert s.zone == Zone.STOP and s.stamp == 10.0
    assert s.nearest_distance_base == 1.4
    assert s.distance_status == DistanceStatus.KNOWN and s.heartbeat_seq == 3


def test_decode_none_distance_is_inf():
    s = safety_state_from_json(_state_json('CLEAR', None, 1.0))
    assert s.nearest_distance_base == float('inf')
    assert s.distance_status == DistanceStatus.NONE


def test_bridge_drives_reaction_and_detections():
    br = HumanSafetyBridge(state_timeout_s=0.7)
    tracks = [{'id': 1, 'position': [1.2, 0.0, 0.9], 'confirmed': True,
               'nearest_point': [0.95, 0.0, 0.9]}]
    out, dets = br.on_json(_state_json('STOP', 1.2, 0.0, tracks), now=0.0)
    assert out.base_scale == 0.0 and out.arm_hold is True
    assert len(dets) == 1 and dets[0].source == 'human_safety' and dets[0].frame == 'base_link'


def test_bridge_fails_safe_when_stale():
    br = HumanSafetyBridge(state_timeout_s=0.5)
    br.on_json(_state_json('CLEAR', None, 0.0), now=0.0)
    out, dets = br.tick(now=1.0)                 # 1.0 s later, > 0.5 s timeout
    # ReceivedSafety synthesises a fresh STOP when the detector goes stale, so the
    # reaction stops the base and holds the arm (absence of evidence is never clearance).
    assert out.zone == 'STOP' and out.base_scale == 0.0 and out.arm_hold
    assert dets == []
