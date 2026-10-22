"""CPU tests for the Isaac-side rig-masked depth core (no omni.graph / no Isaac).

The OmniGraph publisher (ArmDepthOmniGraphBridge) is validated only by the guarded GPU
shakedown; here we cover every pure piece it relies on.
"""
import numpy as np
import pytest

from twin.restock.masked_depth import (
    parse_instance_frame, rig_instance_ids, mask_rig_depth, bbox_payload,
    select_target_bbox, masked_depth_from_frame, mat_to_quat_xyzw, base_from_optical_transform)


def _frame(depth, ids, id_to_path):
    return {'distance_to_image_plane': np.asarray(depth, np.float32),
            'instance_id_segmentation': {'data': np.asarray(ids),
                                         'info': {'idToLabels': id_to_path}}}


# --- rig identity + masking -----------------------------------------------------

def test_rig_ids_prefix_match_excludes_siblings_and_background():
    mapping = {0: 'BACKGROUND', 1: '/World/robot/base', 2: '/World/robot/arm/tool',
               3: '/World/robot_dock/post', 4: '/World/Cart/chassis', 5: '',
               6: '/World/Store/shelf_G3'}
    assert rig_instance_ids(mapping, ('/World/robot', '/World/Cart')) == {1, 2, 4}


def test_rig_ids_exact_equality_matches():
    assert rig_instance_ids({7: '/World/robot'}, ('/World/robot',)) == {7}


def test_rig_ids_requires_a_prefix():
    with pytest.raises(ValueError):
        rig_instance_ids({1: '/World/robot'}, ())


def test_mask_invalidates_rig_and_bad_depth_only():
    out = mask_rig_depth([[1.0, 2.0], [3.0, 4.0]], [[1, 9], [1, 9]], {1}, invalid=0.0)
    assert out.tolist() == [[0.0, 2.0], [0.0, 4.0]]


def test_mask_invalidates_nonfinite_and_nonpositive():
    depth = np.array([[np.nan, -1.0], [0.0, 5.0]], np.float32)
    out = mask_rig_depth(depth, np.zeros((2, 2), int), set(), invalid=0.0)
    assert out.tolist() == [[0.0, 0.0], [0.0, 5.0]]


def test_mask_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        mask_rig_depth(np.zeros((2, 2), np.float32), np.zeros((2, 3), int), set())


def test_parse_instance_frame_normalizes_ids_and_dict_values():
    ids, mapping = parse_instance_frame(_frame([[1.0]], [[2]],
                                               {'0': 'BACKGROUND', '2': {'class': '/World/robot/x'}}))
    assert ids.tolist() == [[2]] and mapping == {0: 'BACKGROUND', 2: '/World/robot/x'}


def test_parse_instance_frame_requires_segmentation():
    with pytest.raises(ValueError):
        parse_instance_frame({'distance_to_image_plane': np.zeros((2, 2), np.float32)})


def test_masked_depth_from_frame_masks_and_caches_rig_ids():
    mapping = {0: 'BACKGROUND', 1: '/World/robot/base', 2: '/World/Cart/wheel'}
    frame = _frame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1, 1, 0], [0, 2, 0]], mapping)
    prefixes = ('/World/robot', '/World/Cart')
    masked, cache = masked_depth_from_frame(frame, prefixes, (None, frozenset()))
    assert masked.tolist() == [[0.0, 0.0, 3.0], [4.0, 0.0, 6.0]] and masked.dtype == np.float32
    assert cache[0] is frame['instance_id_segmentation']['info'] and cache[1] == {1, 2}
    _, cache2 = masked_depth_from_frame(frame, prefixes, cache)  # same seg-info object reused
    assert cache2 is cache


# --- grasp-target bbox selection ------------------------------------------------

def test_select_bbox_prefers_target_location():
    boxes = ({'location': 'rollwagen', 'roi': (0, 0, 2, 2)},
             {'location': 'backstock_00', 'roi': (9, 9, 11, 11)})
    assert select_target_bbox(boxes, 'backstock_00', None) == (9, 9, 11, 11)


def test_select_bbox_caches_previous_when_empty():
    assert select_target_bbox((), 'backstock_00', (5, 6, 7, 8)) == (5, 6, 7, 8)
    assert select_target_bbox(None, 'backstock_00', None) is None


def test_select_bbox_falls_back_to_first_when_no_location_match():
    boxes = ({'location': 'elsewhere', 'roi': (1, 2, 3, 4)},)
    assert select_target_bbox(boxes, 'backstock_00', None) == (1, 2, 3, 4)


def test_bbox_payload_valid_and_rejects_degenerate():
    assert bbox_payload((10, 20, 30, 45)) == [10.0, 20.0, 30.0, 45.0]
    for bad in [(10, 20, 10, 45), (10, 20, 30, 20), (1, 2, 3), (float('nan'), 0, 1, 1)]:
        with pytest.raises(ValueError):
            bbox_payload(bad)


# --- optical transform convention (kept for the optical-prim orientation) -------

def test_mat_to_quat_identity_and_180x():
    assert np.allclose(mat_to_quat_xyzw(np.eye(3)), [0, 0, 0, 1])
    assert np.allclose(np.abs(mat_to_quat_xyzw(np.diag([1.0, -1.0, -1.0]))), [1, 0, 0, 0], atol=1e-9)


def test_base_from_optical_identity_is_the_flip():
    t, q = base_from_optical_transform(np.eye(4), np.eye(4))
    assert np.allclose(t, [0, 0, 0]) and np.allclose(np.abs(q), [1, 0, 0, 0], atol=1e-9)


def test_base_from_optical_moving_base_subtracts_base_pose():
    base = np.array([[1, 0, 0, 1.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    cam = np.array([[1, 0, 0, 3.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    t, _ = base_from_optical_transform(cam, base)
    assert np.allclose(t, [2.0, 0.0, 0.0])
