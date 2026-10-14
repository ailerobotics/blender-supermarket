"""CPU tests for the Isaac-side rig-masked depth core (no rclpy / no Isaac)."""
import numpy as np
import pytest

from types import SimpleNamespace as NS

from twin.restock.masked_depth import (
    parse_instance_frame, rig_instance_ids, mask_rig_depth, MaskedDepthPublisher,
    bbox_payload, ArmCameraDepthBridge)


def _frame(depth, ids, id_to_path):
    return {'distance_to_image_plane': np.asarray(depth, np.float32),
            'instance_id_segmentation': {'data': np.asarray(ids),
                                         'info': {'idToLabels': id_to_path}}}


def test_rig_ids_prefix_match_excludes_siblings_and_background():
    mapping = {0: 'BACKGROUND', 1: '/World/robot/base', 2: '/World/robot/arm/tool',
               3: '/World/robot_dock/post', 4: '/World/Cart/chassis', 5: '',
               6: '/World/Store/shelf_G3'}
    ids = rig_instance_ids(mapping, ('/World/robot', '/World/Cart'))
    # robot + its descendants and the cart match; the robot_dock sibling, empty and store do not.
    assert ids == {1, 2, 4}


def test_rig_ids_exact_equality_matches():
    assert rig_instance_ids({7: '/World/robot'}, ('/World/robot',)) == {7}


def test_rig_ids_requires_a_prefix():
    with pytest.raises(ValueError):
        rig_instance_ids({1: '/World/robot'}, ())


def test_mask_invalidates_rig_and_bad_depth_only():
    depth = [[1.0, 2.0], [3.0, 4.0]]
    ids = [[1, 9], [1, 9]]          # id 1 is rig, id 9 is environment
    out = mask_rig_depth(depth, ids, {1}, invalid=0.0)
    assert out.tolist() == [[0.0, 2.0], [0.0, 4.0]]      # rig pixels zeroed, env kept


def test_mask_invalidates_nonfinite_and_nonpositive():
    depth = np.array([[np.nan, -1.0], [0.0, 5.0]], np.float32)
    ids = np.zeros((2, 2), int)     # nothing is rig
    out = mask_rig_depth(depth, ids, set(), invalid=0.0)
    assert out.tolist() == [[0.0, 0.0], [0.0, 5.0]]


def test_mask_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        mask_rig_depth(np.zeros((2, 2), np.float32), np.zeros((2, 3), int), set())


def test_parse_instance_frame_normalizes_ids_and_dict_values():
    frame = _frame([[1.0]], [[2]], {'0': 'BACKGROUND', '2': {'class': '/World/robot/x'}})
    ids, mapping = parse_instance_frame(frame)
    assert ids.tolist() == [[2]]
    assert mapping == {0: 'BACKGROUND', 2: '/World/robot/x'}


def test_parse_instance_frame_requires_segmentation():
    with pytest.raises(ValueError):
        parse_instance_frame({'distance_to_image_plane': np.zeros((2, 2), np.float32)})


class _FakeCamera:
    def __init__(self, frame, k):
        self._frame, self._k = frame, k

    def get_current_frame(self):
        return self._frame

    def get_intrinsics_matrix(self):
        return self._k


def test_bbox_payload_valid():
    assert bbox_payload((10, 20, 30, 45)) == [10.0, 20.0, 30.0, 45.0]


def test_bbox_payload_rejects_degenerate_or_nonfinite():
    for bad in [(10, 20, 10, 45), (10, 20, 30, 20), (1, 2, 3), (float('nan'), 0, 1, 1)]:
        with pytest.raises(ValueError):
            bbox_payload(bad)


def test_masked_frame_end_to_end_without_ros():
    depth = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    ids = [[1, 1, 0], [0, 2, 0]]
    mapping = {0: 'BACKGROUND', 1: '/World/robot/base', 2: '/World/Cart/wheel'}
    k = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    pub = MaskedDepthPublisher.__new__(MaskedDepthPublisher)   # skip ROS init
    pub.camera = _FakeCamera(_frame(depth, ids, mapping), k)
    pub.rig_prefixes = ('/World/robot', '/World/Cart')
    pub.invalid = 0.0
    pub._id_cache = (None, frozenset())
    masked, intrinsics = pub.masked_frame(pub.camera.get_current_frame())
    assert masked.tolist() == [[0.0, 0.0, 3.0], [4.0, 0.0, 6.0]]   # robot(1,1), cart(2) zeroed
    assert masked.dtype == np.float32
    assert np.allclose(intrinsics, k)


class _RecordingDepth:
    def __init__(self, frame):
        self.camera = NS(get_current_frame=lambda: frame)
        self.calls = []

    def publish_frame(self, frame, stamp_s):
        self.calls.append((frame, stamp_s))


class _Recorder:
    def __init__(self):
        self.calls = []

    def publish(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _bridge(frame, sensors, target_location=None):
    depth, bbox, tf = _RecordingDepth(frame), _Recorder(), _Recorder()
    br = ArmCameraDepthBridge(depth_publisher=depth, bbox_publisher=bbox, tf_publisher=tf,
                             sensors=sensors, base_from_optical=lambda: ((1.0, 2.0, 3.0),
                             (0.0, 0.0, 0.0, 1.0)), target_location=target_location)
    return br, depth, bbox, tf


def test_bridge_publishes_depth_tf_bbox_on_one_stamp():
    frame = {'rendering_time': 4.25}
    sensors = NS(sensed_target_bboxes=({'location': 'backstock_00', 'roi': (1, 2, 3, 4)},))
    br, depth, bbox, tf = _bridge(frame, sensors, target_location='backstock_00')
    stamp = br.publish(now=99.0)
    assert stamp == 4.25                                   # stamp taken from the frame, not now
    assert depth.calls == [(frame, 4.25)]
    assert tf.calls and tf.calls[0][1] == {'parent': 'ur_base_link', 'child': 'restock_camera_arm'}
    assert bbox.calls and bbox.calls[0][0][0] == (1, 2, 3, 4)


def test_bridge_republishes_cached_bbox_when_perception_empty():
    frame = {'rendering_time': 1.0}
    sensors = NS(sensed_target_bboxes=({'location': 'backstock_00', 'roi': (5, 6, 7, 8)},))
    br, depth, bbox, tf = _bridge(frame, sensors, target_location='backstock_00')
    br.publish(now=1.0)
    sensors.sensed_target_bboxes = ()                      # perception produced none this tick
    br.publish(now=1.1)
    assert [c[0][0] for c in bbox.calls] == [(5, 6, 7, 8), (5, 6, 7, 8)]   # cached, still fresh


def test_bridge_filters_bbox_by_target_location():
    frame = {'rendering_time': 2.0}
    sensors = NS(sensed_target_bboxes=({'location': 'rollwagen', 'roi': (0, 0, 2, 2)},
                                       {'location': 'backstock_00', 'roi': (9, 9, 11, 11)}))
    br, depth, bbox, tf = _bridge(frame, sensors, target_location='backstock_00')
    br.publish(now=2.0)
    assert bbox.calls[0][0][0] == (9, 9, 11, 11)           # the source target, not the cart


def test_bridge_skips_bbox_until_first_detection():
    frame = {'rendering_time': 3.0}
    sensors = NS(sensed_target_bboxes=())
    br, depth, bbox, tf = _bridge(frame, sensors, target_location='backstock_00')
    br.publish(now=3.0)
    assert bbox.calls == [] and depth.calls and tf.calls   # depth+TF still publish


def test_mat_to_quat_identity_and_180x():
    from twin.restock.masked_depth import mat_to_quat_xyzw
    assert np.allclose(mat_to_quat_xyzw(np.eye(3)), [0, 0, 0, 1])
    # Rx(180) = diag(1,-1,-1) -> quaternion (1,0,0,0) xyzw
    assert np.allclose(mat_to_quat_xyzw(np.diag([1.0, -1.0, -1.0])), [1, 0, 0, 1e-16], atol=1e-9) \
        or np.allclose(np.abs(mat_to_quat_xyzw(np.diag([1.0, -1.0, -1.0]))), [1, 0, 0, 0], atol=1e-9)


def test_base_from_optical_identity_is_the_flip():
    from twin.restock.masked_depth import base_from_optical_transform
    t, q = base_from_optical_transform(np.eye(4), np.eye(4))
    assert np.allclose(t, [0, 0, 0])
    # optical frame in base == 180deg about X (USD cam -> ROS optical)
    assert np.allclose(np.abs(q), [1, 0, 0, 0], atol=1e-9)


def test_base_from_optical_translation_and_norm():
    from twin.restock.masked_depth import base_from_optical_transform
    cam = np.array([[1, 0, 0, 2.0], [0, 1, 0, -1.0], [0, 0, 1, 0.5], [0, 0, 0, 1]])
    t, q = base_from_optical_transform(cam, np.eye(4))
    assert np.allclose(t, [2.0, -1.0, 0.5])              # optical origin == camera origin
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9           # unit quaternion


def test_base_from_optical_moving_base_subtracts_base_pose():
    from twin.restock.masked_depth import base_from_optical_transform
    base = np.array([[1, 0, 0, 1.0], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]])
    cam = np.array([[1, 0, 0, 3.0], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]])
    t, _ = base_from_optical_transform(cam, base)
    assert np.allclose(t, [2.0, 0.0, 0.0])               # camera is 2 m ahead of the base origin
