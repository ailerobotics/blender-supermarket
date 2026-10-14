"""CPU geometry tests for the arm eye-in-hand camera framing (collect + transfer).

No Isaac/USD. The mount geometry is taken from the mission URDF
(`sim-restock-host-v2/refined/mobile.urdf`): front_camera_frame is fixed to base_link at
(0.46,0,0.35) and ur_base_link is base_link+(0,0,0.26), so the camera frame is at
(0.46,0,0.09) in ur_base_link with identity rotation. Targets are the measured carton /
slot poses in ur_base_link. These reproduce the run-75 `reproject-None behind-camera` and
prove the look-at re-aim frames both the source bin and the shelf slot.
"""
import math

import numpy as np

from scripts.twin.restock_camera.arm_mount import arm_camera_look_at_mount

EYE = (0.46, 0.0, 0.09)                     # front_camera_frame origin in ur_base_link
SRC = (0.002, 1.1, 0.261)                   # source carton (ur_base_link), run-75 measured
SHELF = (0.0, 0.821, 0.31)                  # shelf slot (ur_base_link), from reach artifacts
DIMS = (0.6, 0.4, 0.16)                     # juice_orange handling unit
INTR = (305., 305., 64., 64.)               # arm camera intrinsics observed in run-75 (128x128)
SHAPE = (128, 128)
# The forward-looking matrix isaac_factory authored on the arm_camera prim (run-75), row-major.
HARDCODED_GF = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], float)
FLIP = np.diag([1., -1., -1., 1.])


def _base_from_camera_from_prim(gf_rowmajor, eye):
    """Reproduce isaac_factory._arm_base_from_camera: inv(base)@(cam@diag) with frame rot identity."""
    m_prim_std = np.asarray(gf_rowmajor, float).T
    t = np.eye(4); t[:3, :3] = m_prim_std[:3, :3]; t[:3, 3] = eye
    return t @ FLIP


def _reproject(base_from_camera, pose_xyz, yaw, intr=INTR, shape=SHAPE):
    """Self-contained copy of reproject_carton_roi's pinhole (integration module not on main)."""
    x, y, z = pose_xyz
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy, dz = DIMS[0] / 2, DIMS[1] / 2, DIMS[2]
    corners = np.array([[x + c * ax - s * ay, y + s * ax + c * ay, z - az]
                        for ax in (-dx, dx) for ay in (-dy, dy) for az in (0., dz)])
    c2b = np.linalg.inv(base_from_camera)
    p = corners @ c2b[:3, :3].T + c2b[:3, 3]
    front = p[p[:, 2] > 1e-6]
    if len(front) < 4:
        return None, p[:, 2]
    fx, fy, cx, cy = intr
    uv = front[:, :2] / front[:, 2, None] * [fx, fy] + [cx, cy]
    lo = np.maximum(np.floor(uv.min(0)).astype(int), [0, 0])
    hi = np.minimum(np.ceil(uv.max(0)).astype(int) + 1, [shape[1], shape[0]])
    if hi[0] <= lo[0] or hi[1] <= lo[1]:
        return None, p[:, 2]
    return (int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])), p[:, 2]


def test_current_forward_mount_puts_both_targets_behind_the_camera():
    b2c = _base_from_camera_from_prim(HARDCODED_GF, EYE)
    # view direction is base +X (forward); reproduces run-75's behind-camera verdict.
    assert np.allclose(b2c[:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-9)
    for pose, yaw in [(SRC, math.pi), (SHELF, 0.)]:
        roi, zs = _reproject(b2c, pose, yaw)
        assert roi is None and (zs < 0).all()      # every carton corner behind the optical axis
    # Matches the logged origin-in-camera for the source (run-75: (-1.101,-0.171,-0.460)).
    c2b = np.linalg.inv(b2c)
    cic = np.asarray(SRC) @ c2b[:3, :3].T + c2b[:3, 3]
    assert np.allclose(cic, [-1.100, -0.171, -0.458], atol=2e-3)


def test_look_at_mount_frames_both_source_and_shelf():
    mount = arm_camera_look_at_mount(EYE, [SRC, SHELF])
    # view direction now points mostly base +Y (left) and slightly down — toward the targets.
    assert mount.view_dir[1] > 0.8 and abs(mount.view_dir[0]) < 0.6
    for pose, yaw in [(SRC, math.pi), (SHELF, 0.)]:
        roi, zs = _reproject(mount.base_from_camera, pose, yaw)
        assert roi is not None                      # a real ROI reaches the host (was None)
        assert (zs > 0).all()                       # every carton corner now in front of the camera
        u1, v1, u2, v2 = roi
        assert 0 <= u1 < u2 <= SHAPE[1] and 0 <= v1 < v2 <= SHAPE[0]


def test_look_at_both_directions_fit_within_the_arm_camera_fov():
    # A single fixed mount can serve both stages: the two look directions are ~8 deg apart,
    # inside the ~24 deg HFOV, so aiming at their centroid keeps both well inside the frame.
    eye = np.asarray(EYE)
    fs = (np.asarray(SRC) - eye); fs /= np.linalg.norm(fs)
    fh = (np.asarray(SHELF) - eye); fh /= np.linalg.norm(fh)
    sep = math.degrees(math.acos(np.clip(fs @ fh, -1, 1)))
    hfov = 2 * math.degrees(math.atan(INTR[2] / INTR[0]))
    assert sep < hfov / 2                           # both within a half-FOV of the aim centroid


def test_usd_transform_op_reconstructs_base_from_camera_through_the_factory_chain():
    mount = arm_camera_look_at_mount(EYE, [SRC, SHELF])
    # Authoring usd_transform_op on the prim and running isaac_factory's _arm_base_from_camera
    # (frame rotation identity, prim at frame origin) must recover base_from_camera exactly.
    recovered = _base_from_camera_from_prim(mount.usd_transform_op, EYE)
    assert np.allclose(recovered, mount.base_from_camera, atol=1e-9)


def test_single_target_and_centroid_aim():
    one = arm_camera_look_at_mount(EYE, SRC)
    many = arm_camera_look_at_mount(EYE, [SRC, SRC])
    assert np.allclose(one.base_from_camera, many.base_from_camera)
    assert np.allclose(one.aim_point, SRC)
