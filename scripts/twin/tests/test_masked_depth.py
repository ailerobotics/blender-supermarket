"""CPU tests for the Isaac-side rig-masked depth core (no rclpy / no Isaac)."""
import numpy as np
import pytest

from twin.restock.masked_depth import (
    parse_instance_frame, rig_instance_ids, mask_rig_depth, MaskedDepthPublisher)


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
