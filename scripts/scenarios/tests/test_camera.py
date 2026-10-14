"""Camera intrinsics, extrinsics and projection behave like Blender's pin-hole."""
import math

import camera as cam


def test_lookat_orthonormal():
    c2w = cam.look_at((2, -3, 1.5), (0, 0, 1))
    cols = [(c2w[0][i], c2w[1][i], c2w[2][i]) for i in range(3)]
    for v in cols:
        assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-9
    # forward (-Z column) points from eye toward target
    z = cols[2]
    fwd = (-z[0], -z[1], -z[2])
    to_t = (0 - 2, 0 + 3, 1 - 1.5)
    n = math.sqrt(sum(x * x for x in to_t))
    dot = sum(a * b for a, b in zip(fwd, (to_t[0] / n, to_t[1] / n, to_t[2] / n)))
    assert dot > 0.99


def test_intrinsics_scale_with_width():
    k1 = cam.intrinsics(30, 640, 480)
    k2 = cam.intrinsics(30, 1280, 480)
    assert abs(k2["fx"] - 2 * k1["fx"]) < 1e-6
    assert k1["cx"] == 320 and k1["cy"] == 240


def test_center_projects_to_principal_point():
    pose = cam.CameraPose("c", eye=(0, -4, 1.0), target=(0, 0, 1.0),
                          lens_mm=30, width=640, height=480)
    c2w, K = pose.c2w(), pose.K()
    # a point straight ahead on the optical axis lands at the principal point
    pr = cam.project(c2w, K, (0, 0, 1.0))
    assert pr is not None
    u, v, depth = pr
    assert abs(u - K["cx"]) < 1e-3 and abs(v - K["cy"]) < 1e-3
    assert abs(depth - 4.0) < 1e-3


def test_behind_camera_returns_none():
    pose = cam.CameraPose("c", eye=(0, -4, 1.0), target=(0, 0, 1.0),
                          lens_mm=30, width=640, height=480)
    assert cam.project(pose.c2w(), pose.K(), (0, -8, 1.0)) is None


def test_box_projection_and_clip():
    pose = cam.CameraPose("c", eye=(0, -4, 1.0), target=(0, 0, 1.0),
                          lens_mm=30, width=640, height=480)
    corners = [(x, y, z) for x in (-0.2, 0.2) for y in (-0.1, 0.1) for z in (0.9, 1.1)]
    box = cam.project_box(pose.c2w(), pose.K(), corners)
    clipped = cam.clip_box(box, 640, 480)
    assert clipped is not None
    x0, y0, x1, y1 = clipped
    assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480


def test_to_json_shape():
    pose = cam.CameraPose("c", eye=(1, 2, 3), target=(0, 0, 1), lens_mm=24,
                          width=256, height=192)
    j = pose.to_json()
    assert len(j["extrinsics"]["camera_to_world"]) == 4
    assert set(("fx", "fy", "cx", "cy")) <= set(j["intrinsics"])
