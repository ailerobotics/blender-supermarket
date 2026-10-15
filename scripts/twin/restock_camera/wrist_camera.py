"""Eye-in-hand wrist RGB-D camera for the mobile UR10 (camera lane).

The robot's only camera was base-front mounted (`front_camera_frame`, looking +X), so the arm's
side-grasp workspace was never in the collision depth — the planner was blind to neighbouring
cartons there (see `data/sim-restock-camera-v2/arm-framing-report.md`). This adds a real
eye-in-hand camera rigidly mounted on the UR wrist (`ur_tool0`): because it rides the tool, the
grasp region is in frame at any base/arm pose, which the fixed body camera could never achieve.

Roles split cleanly: the fixed `source`/`shelf` station cameras stay decode-grade (QR); this wrist
camera is the **collision / masked-depth** camera (frame `restock_camera_wrist`) — it does not
decode, so it uses a wide field at a modest depth-camera resolution, not decode-grade pixels.

This module is the single source of the calibrated mount: the frame offset on `ur_tool0`, the
optical `parent_from_optical`, the USD prim transform, the intrinsics, and the design/selection
helpers. `mobile_assets` authors the URDF frame + USD camera from here; `restock_camera.config`
records it in the camera design; integration's `native_cameras` / masked-depth bridge select it
via :func:`collision_camera`. Pure NumPy; `usd_camera_transform` is reused from `geometry`.
"""
import math
from dataclasses import asdict

import numpy as np

from .geometry import Intrinsics, usd_camera_transform

WRIST_PARENT_LINK = 'ur_tool0'          # UR flange tool frame the mount rides
WRIST_FRAME = 'restock_camera_wrist'    # TF frame the masked-depth bridge publishes/selects
# Mount frame origin (the optical centre) in ur_tool0: offset to the +Y side of the tool and set
# back along -Z so the camera clears the gripper body and looks across the grasp. Axis-aligned with
# ur_tool0 (identity rotation on the URDF joint); the optical rotation is carried in parent_from_optical.
WRIST_MOUNT_XYZ = (0.0, 0.12, -0.06)
GRASP_POINT_TOOL0 = (-0.0001, -0.0586, 0.2039)   # gripper_tcp in ur_tool0 (mission URDF gripper chain)
_UP_HINT = (1.0, 0.0, 0.0)              # ur_tool0 +X as the image-right reference (perp to the view)

# Collision / masked-depth intrinsics: WIDE field, modest resolution. Depth does not need the
# decode-grade resolution of the fixed source/shelf cameras; it needs to cover the grasp workspace
# at the ~0.32 m wrist standoff. 424x240 @ 75-deg HFOV (~47-deg VFOV) is a native depth-camera mode.
WRIST_WIDTH, WRIST_HEIGHT, WRIST_HFOV_DEG = 424, 240, 75.0
WRIST_REVISION = 'eye-in-hand-wrist-v1'


def wrist_camera_intrinsics(near=0.05, far=2.0):
    """Wide-field, modest-resolution :class:`Intrinsics` for the wrist collision camera."""
    fx = (WRIST_WIDTH / 2.) / math.tan(math.radians(WRIST_HFOV_DEG) / 2.)
    return Intrinsics(width=WRIST_WIDTH, height=WRIST_HEIGHT, fx=fx, fy=fx,
                      cx=WRIST_WIDTH / 2., cy=WRIST_HEIGHT / 2., near=near, far=far)


def _optical_rotation():
    """Optical rotation (columns right/down/forward) that aims +Z at the grasp point."""
    eye = np.asarray(WRIST_MOUNT_XYZ, float)
    forward = np.asarray(GRASP_POINT_TOOL0, float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, _UP_HINT); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack([right, down, forward])


def wrist_parent_from_optical():
    """4x4 ``restock_camera_wrist`` <- optical. The frame is the optical centre (zero translation)
    and is axis-aligned with ``ur_tool0``, so this is purely the optical rotation."""
    transform = np.eye(4)
    transform[:3, :3] = _optical_rotation()
    return transform


def wrist_usd_camera_transform():
    """Row-major 4x4 for the USD ``Camera`` prim op on the wrist frame (Gf.Matrix4d)."""
    return [[float(v) for v in row] for row in usd_camera_transform(wrist_parent_from_optical()).T]


def wrist_camera_design():
    """The wrist camera design entry (recorded in the camera calibration data / camera_config)."""
    return dict(id='wrist', parent=WRIST_FRAME, parent_from_optical=wrist_parent_from_optical().tolist(),
                intrinsics=asdict(wrist_camera_intrinsics()), revision=WRIST_REVISION,
                role='collision', frame=WRIST_FRAME, decode_grade=False)


def collision_camera(design):
    """The camera the masked-depth bridge selects as the collision camera, from the design.

    Returns the single ``role == 'collision'`` camera spec (its ``frame`` is ``restock_camera_wrist``).
    Refuses if the design has none or more than one — the bridge must select exactly one depth camera.
    """
    cameras = design['cameras'] if isinstance(design, dict) else design
    collision = [c for c in cameras.values() if c.get('role') == 'collision']
    if len(collision) != 1:
        raise ValueError(f'camera design must declare exactly one collision camera, found {len(collision)}')
    return collision[0]


def grasp_region_in_image(point_tool0, intrinsics=None):
    """Project a point given in ``ur_tool0`` into the wrist image: ``(u, v, z)``.

    The wrist camera and the gripper are both rigid on the tool, so a grasp-region point expressed in
    ``ur_tool0`` projects to a fixed image location independent of the arm/base pose — which is why the
    eye-in-hand view frames the side-grasp workspace the fixed body camera could not.
    """
    k = intrinsics or wrist_camera_intrinsics()
    camera_from_tool = np.linalg.inv(wrist_tool0_from_optical())
    p = np.asarray(point_tool0, float) @ camera_from_tool[:3, :3].T + camera_from_tool[:3, 3]
    return (p[0] / p[2] * k.fx + k.cx, p[1] / p[2] * k.fy + k.cy, p[2])


def wrist_tool0_from_optical():
    """4x4 ``ur_tool0`` <- optical: the mount translation on the tool plus the optical rotation."""
    transform = wrist_parent_from_optical()
    transform[:3, 3] = WRIST_MOUNT_XYZ
    return transform


def grasp_region_depth_coverage(depth, points_tool0, *, intrinsics=None, window_px=16):
    """Fraction of the grasp-region windows (around each tool-frame point) with valid depth.

    This is the pure assertion core of the native proof: given a rendered wrist ``depth`` image
    (metres, non-finite where nothing was returned), project each grasp-region point (in
    ``ur_tool0``) into the image and measure, over a ``window_px`` square around it, the fraction of
    pixels whose depth is finite and within the camera's ``[near, far]`` clip range. The native
    renderer (integration, under the GPU lock) supplies ``depth`` at the collect pose; a high
    coverage there proves the side-grasp region is actually in the wrist camera's depth (what the
    fixed body camera could never show). Points outside the image contribute zero coverage.
    """
    k = intrinsics or wrist_camera_intrinsics()
    depth = np.asarray(depth, float)
    if depth.ndim != 2:
        raise ValueError('wrist depth must be a 2-D image')
    result = {}
    for i, point in enumerate(np.atleast_2d(np.asarray(points_tool0, float))):
        u, v, z = grasp_region_in_image(point, k)
        if not (z > 0):
            result[i] = 0.0
            continue
        u0, u1 = int(round(u)) - window_px, int(round(u)) + window_px + 1
        v0, v1 = int(round(v)) - window_px, int(round(v)) + window_px + 1
        u0, v0 = max(u0, 0), max(v0, 0)
        u1, v1 = min(u1, depth.shape[1]), min(v1, depth.shape[0])
        if u1 <= u0 or v1 <= v0:
            result[i] = 0.0
            continue
        patch = depth[v0:v1, u0:u1]
        valid = np.isfinite(patch) & (patch >= k.near) & (patch <= k.far)
        result[i] = float(valid.mean())
    return result
