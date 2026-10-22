"""Offline tests for the detection bridge (needs numpy; no ROS/USD)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans.config import DetectionConfig
from twin.humans.detection import (Detection, GroundTruthSafety, ReceivedSafety,
                                   base_frame_point, camera_sees, make_provider,
                                   nearest_point_toward_base, to_nav_detections)
from twin.humans.perception import SafetyState, Zone


def test_base_frame_point_rotation():
    # base at origin facing +x: a world point ahead stays ahead
    p = base_frame_point((2.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    assert np.allclose(p, [2.0, 0.0, 1.0])
    # base facing +y: a world point at +x is to the robot's right (-y in base)
    p = base_frame_point((1.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2))
    assert np.allclose(p, [0.0, -1.0, 0.0], atol=1e-9)


def test_camera_sees_cone():
    cam = (0.0, 0.0, 0.0)                      # at origin looking +x, 0.6 rad half-cone
    assert camera_sees((3.0, 0.0, 0.0), cam, 6.0, 0.6)       # dead ahead
    assert not camera_sees((0.0, 3.0, 0.0), cam, 6.0, 0.6)   # 90 deg off axis
    assert not camera_sees((7.0, 0.0, 0.0), cam, 6.0, 0.6)   # beyond range


def test_nearest_point_is_closer_than_centre():
    pos = np.array([3.0, 0.0, 1.0])
    near = nearest_point_toward_base(pos, 0.3)
    assert math.hypot(near[0], near[1]) < math.hypot(pos[0], pos[1])
    assert math.isclose(math.hypot(near[0], near[1]), 2.7, abs_tol=1e-6)


def _gt(people_fn, cams=None):
    cams = cams or {'restock_camera_wrist': (0.0, 0.0, 0.0)}
    return GroundTruthSafety(
        config=DetectionConfig(source='ground_truth', cameras=tuple(cams)),
        people=people_fn, base_pose=lambda: (0.0, 0.0, 0.0),
        camera_poses=lambda: cams)


def test_ground_truth_far_person_is_clear():
    gt = _gt(lambda now: [('p1', (5.5, 0.0, 0.9), 0.25)])   # within 6 m range, far ring
    st = gt.state(0.0)
    assert st.zone == Zone.SLOW                              # any visible person -> >= SLOW
    assert st.any_person_visible


def test_ground_truth_close_person_stops():
    # person 1.5 m from base (inside reach+stop trigger) -> STOP
    gt = _gt(lambda now: [('p1', (1.5, 0.0, 0.9), 0.25)])
    st = gt.state(0.0)
    assert st.zone == Zone.STOP


def test_ground_truth_unseen_person_is_not_tracked():
    # person behind the camera (out of cone) -> no track, CLEAR
    gt = _gt(lambda now: [('p1', (-3.0, 0.0, 0.9), 0.25)])
    st = gt.state(0.0)
    assert st.zone == Zone.CLEAR
    assert not st.any_person_visible


def test_ground_truth_emits_shared_detection_contract():
    gt = _gt(lambda now: [('p1', (2.0, 0.5, 0.9), 0.25)])
    dets = gt.detections(0.0)
    assert len(dets) == 1 and isinstance(dets[0], Detection)
    assert dets[0].source == 'human_ground_truth' and dets[0].frame == 'map'
    assert dets[0].position == (2.0, 0.5) and dets[0].z == 0.9 and dets[0].radius == 0.25
    assert len(dets[0].position) == 2             # matches the dynobs 2-tuple contract


def test_human_detections_feed_the_real_dynobs_layer():
    """A ground-truth human detection is accepted by the shared nav obstacle layer
    (no second detector): the same Detection drives DynamicObstacleLayer.update."""
    pytest.importorskip('twin.navigation.dynamic')
    from twin.navigation.dynamic import (Detection as NavDetection,
                                         DynamicObstacleLayer)
    gt = _gt(lambda now: [('p1', (2.0, 0.5, 0.9), 0.25)])
    nav_dets = to_nav_detections(gt.detections(0.0))
    assert nav_dets and all(isinstance(d, NavDetection) for d in nav_dets)
    assert nav_dets[0].position == (2.0, 0.5) and nav_dets[0].source == 'human_ground_truth'
    layer = DynamicObstacleLayer()
    layer.update(nav_dets, stamp=0.0)             # the layer's isinstance gate accepts them
    layer.update(to_nav_detections(gt.detections(0.5)), stamp=0.5)
    assert layer.tracks                            # the human became a tracked obstacle


def test_to_nav_detections_rejects_base_frame():
    # A detector detection is base_link-relative; converting without TF is refused.
    base = Detection(position=(1.0, 0.0), radius=0.25, source='human_safety',
                     frame='base_link')
    with pytest.raises(ValueError):
        to_nav_detections([base])


def test_received_safety_stale_fails_to_stop():
    rc = ReceivedSafety(config=DetectionConfig(source='detector'), state_timeout_s=0.5)
    assert rc.state(0.0).zone == Zone.STOP                  # nothing received yet
    fresh = SafetyState(stamp=1.0, zone=Zone.CLEAR, nearest_distance=float('inf'),
                        nearest_distance_base=float('inf'),
                        distance_status=None, nearest_track_id=None,
                        any_person_visible=False)
    rc.on_state(fresh, now=1.0)
    assert rc.state(1.2).zone == Zone.CLEAR                 # fresh -> passthrough
    assert rc.state(2.0).zone == Zone.STOP                  # stale -> STOP


def test_make_provider_dispatches_by_source():
    assert isinstance(make_provider(DetectionConfig(source='detector')), ReceivedSafety)
    gt = make_provider(DetectionConfig(source='ground_truth'),
                       people=lambda now: [], base_pose=lambda: (0.0, 0.0, 0.0),
                       camera_poses=lambda: {})
    assert isinstance(gt, GroundTruthSafety)
