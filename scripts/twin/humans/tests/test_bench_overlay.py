"""Offline tests for the demonstrative overview view (numpy + PIL; no Isaac/USD)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans import bench_overlay as bo
from twin.humans.config import PersonSpec
from twin.restock_camera.geometry import look_at


def test_overview_look_at_frames_the_path():
    base = (0.0, 0.0, 0.0)
    eye, target = bo.overview_look_at(base, (1.0, 0.0), near_m=0.9, far_m=6.5,
                                      side_m=5.0, height_m=3.2, target_z=0.9)
    # target is the midpoint of the crossing in front of the base, at chest height
    assert math.isclose(target[0], 3.7, abs_tol=1e-6) and abs(target[1]) < 1e-9
    assert math.isclose(target[2], 0.9, abs_tol=1e-6)
    # eye sits side_m to the LEFT of the forward axis (+y here) and height_m up
    assert math.isclose(eye[1], 5.0, abs_tol=1e-6) and math.isclose(eye[2], 3.2, abs_tol=1e-6)


def test_person_box_visible_in_front_and_none_behind():
    intr = dict(width=960, height=540, fx=620., fy=620., cx=480., cy=270., near=0.1, far=40.)
    cam = look_at((0.0, 0.0, 1.0), (5.0, 0.0, 1.0))        # at origin looking +x
    box = bo.person_image_box((3.0, 0.0, 0.0), radius=0.3, height=1.75,
                              world_from_optical=cam, intrinsics=intr)
    assert box is not None
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 < 960 and 0 <= y0 < y1 < 540
    # a person behind the camera projects to nothing
    assert bo.person_image_box((-3.0, 0.0, 0.0), 0.3, 1.75, cam, intr) is None


def test_draw_overlay_shape_and_banner_colour():
    rgb = np.full((540, 960, 3), 40, dtype=np.uint8)
    out = bo.draw_overlay(rgb, 'STOP', 0.0, True, [(400, 200, 500, 400)], t=4.5)
    assert out.shape == (540, 960, 3) and out.dtype == np.uint8
    # the top banner is painted the STOP colour
    assert tuple(out[2, 2]) == bo.ZONE_RGB['STOP']
    # the box region is modified vs the flat input
    assert not np.array_equal(out[200:400, 400:500], rgb[200:400, 400:500])


def test_frame_span_from_people():
    p = PersonSpec(person_id='x', role='shopper', waypoints=((6.5, 0.0), (0.9, 0.0)),
                   speed_mps=0.9)
    near, far = bo.frame_span_from_people((p,), base_xy=(0.0, 0.0))
    assert math.isclose(near, 0.9, abs_tol=1e-6) and math.isclose(far, 6.5, abs_tol=1e-6)
