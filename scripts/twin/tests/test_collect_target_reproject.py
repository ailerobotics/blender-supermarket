"""Pure reprojection of the fixed-source collect target into the arm image."""
import math

import numpy as np
import pytest

from scripts.twin.restock.collect_target import reproject_carton_roi


DIMS = (0.6, 0.4, 0.16)                       # juice_orange handling unit
INTR = (1500., 1500., 480., 270.)             # fx,fy,cx,cy
SHAPE = (540, 960)                            # height, width


def test_centered_carton_projects_to_a_centered_roi():
    # Arm camera optical frame == base frame (identity); carton 1.5 m straight ahead (+Z).
    roi = reproject_carton_roi((0, 0, 1.5, 0, 0, 0, 1), DIMS, np.eye(4), INTR, SHAPE)
    assert roi is not None
    u1, v1, u2, v2 = roi
    cu, cv = (u1 + u2) / 2., (v1 + v2) / 2.
    assert abs(cu - INTR[2]) < 2 and abs(cv - INTR[3]) < 2      # centered on principal point
    assert 0 <= u1 < u2 <= SHAPE[1] and 0 <= v1 < v2 <= SHAPE[0]
    assert (u2 - u1) > 5 and (v2 - v1) > 5                       # non-degenerate box


def test_behind_camera_returns_none_not_a_fabricated_box():
    assert reproject_carton_roi((0, 0, -1.0, 0, 0, 0, 1), DIMS, np.eye(4), INTR, SHAPE) is None


def test_off_axis_target_is_clamped_into_the_image():
    # Carton far to the side but still in front: the ROI clamps to the image bounds.
    roi = reproject_carton_roi((3.0, 0, 1.0, 0, 0, 0, 1), DIMS, np.eye(4), INTR, SHAPE)
    if roi is not None:
        u1, v1, u2, v2 = roi
        assert 0 <= u1 < u2 <= SHAPE[1] and 0 <= v1 < v2 <= SHAPE[0]


def test_translated_base_from_camera_shifts_the_roi_consistently():
    # Camera offset +0.2 m in base x: the carton's image column shifts, box stays valid+in-frame.
    bfc = np.eye(4); bfc[0, 3] = 0.2
    roi = reproject_carton_roi((0, 0, 1.5, 0, 0, 0, 1), DIMS, bfc, INTR, SHAPE)
    assert roi is not None
    u1, v1, u2, v2 = roi
    assert 0 <= u1 < u2 <= SHAPE[1] and 0 <= v1 < v2 <= SHAPE[0]
    # moving the camera +x shifts the target to smaller u (left) vs the centered case
    assert (u1 + u2) / 2. < INTR[2]


def test_yaw_rotation_changes_the_box_footprint():
    straight = reproject_carton_roi((0, 0, 1.2, 0, 0, 0, 1), DIMS, np.eye(4), INTR, SHAPE)
    rot = reproject_carton_roi((0, 0, 1.2, 0, 0, math.sin(math.pi/4), math.cos(math.pi/4)),
                               DIMS, np.eye(4), INTR, SHAPE)
    assert straight is not None and rot is not None
    assert straight != rot                     # 90-degree yaw swaps the projected extents
