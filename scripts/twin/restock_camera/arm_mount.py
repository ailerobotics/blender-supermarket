"""Aim the arm eye-in-hand camera mount at the restock targets (camera lane).

The arm camera rides a FIXED robot body frame (``front_camera_frame`` on ``base_link``). In
run-75 it was authored looking straight forward (optical +Z = base +X), but at BOTH the collect
and transfer docking poses the target (source bin / shelf slot) sits to the robot's LEFT
(base +Y, ~1 m) — so every reprojection put the carton behind the optical axis (camera-frame
z < 0) and ``CollectTargetBridge._reproject`` returned ``roi=None`` (``behind-camera``).

This re-aims the mount with a look-at toward the target(s), reusing the existing camera-lane
geometry (:func:`geometry.look_at` + :func:`geometry.usd_camera_transform`). The two look
directions (source bin, shelf slot) are only ~8 deg apart from the fixed camera origin — well
inside the arm camera FOV — so a single fixed re-aim frames both stages. Integration authors
:attr:`ArmCameraMount.usd_transform_op` on the ``front_camera_frame/arm_camera`` prim (the
value ``isaac_factory`` currently hardcodes) and the live ``_arm_base_from_camera`` then equals
:attr:`ArmCameraMount.base_from_camera`.

Pure NumPy, no Isaac/USD. See ``data/sim-restock-camera-v2/arm-framing-report.md``.
"""
import math
from dataclasses import dataclass

import numpy as np

from .geometry import Intrinsics, look_at, usd_camera_transform, rigid

# Collision / masked-depth intrinsics for the re-aimed arm camera. This camera feeds the host
# nvblox collision depth and the grasp-target bbox — it does NOT decode QR, so it does NOT need the
# decode-grade resolution of the fixed source/shelf cameras (those stay untouched). What it DOES
# need is a WIDE field: the run-75 128x128 / 23.8-deg field covers only ~0.50 m at the ~1.2 m grasp
# standoff, so the 0.60x0.40 carton overflowed the frame (ROI clamped to full width). The required
# half-angles to enclose all 8 carton corners of BOTH the collect bin and the transfer slot (at the
# centroid look-at) are 25.3 deg horizontal, 10.0 deg vertical. 424x240 at 75-deg HFOV (~47-deg VFOV)
# clears both with margin at a modest depth-camera resolution (a native RealSense-class depth mode).
COLLISION_CAMERA_WIDTH = 424
COLLISION_CAMERA_HEIGHT = 240
COLLISION_CAMERA_HFOV_DEG = 75.0


def collision_camera_intrinsics(near=0.1, far=2.5):
    """Wide-field, modest-resolution :class:`Intrinsics` for the arm collision/masked-depth camera."""
    fx = (COLLISION_CAMERA_WIDTH / 2.) / math.tan(math.radians(COLLISION_CAMERA_HFOV_DEG) / 2.)
    return Intrinsics(width=COLLISION_CAMERA_WIDTH, height=COLLISION_CAMERA_HEIGHT, fx=fx, fy=fx,
                      cx=COLLISION_CAMERA_WIDTH / 2., cy=COLLISION_CAMERA_HEIGHT / 2., near=near, far=far)


@dataclass(frozen=True)
class ArmCameraMount:
    base_from_camera: np.ndarray   # camera-base_link <- optical (what _arm_base_from_camera must return)
    usd_transform_op: list         # row-major 4x4 for Gf.Matrix4d on front_camera_frame/arm_camera
    aim_point: tuple               # the look-at point, in the camera base frame
    view_dir: tuple                # optical +Z (view direction), in the camera base frame


def arm_camera_look_at_mount(camera_origin, targets, *, frame_rotation=None):
    """Mount transform that aims the fixed arm camera at ``targets``.

    ``camera_origin`` is the mount frame origin in the camera base frame (``ur_base_link``);
    ``targets`` is one ``(x,y,z[,...])`` point or a sequence of them (the mount aims at their
    centroid so a single fixed mount frames every stage). ``frame_rotation`` is the mount
    frame's 3x3 rotation in the base frame (default identity — ``front_camera_frame`` is
    axis-aligned with ``ur_base_link``); it only affects the authored prim transform, not the
    resulting ``base_from_camera``.
    """
    origin = np.asarray(camera_origin, float)
    pts = np.atleast_2d(np.asarray(targets, float))[:, :3]
    if pts.shape[0] == 0:
        raise ValueError('at least one target point is required to aim the arm camera')
    aim = pts.mean(axis=0)
    base_from_camera = look_at(origin, aim)                     # ur_base <- optical, columns [right,down,fwd]
    frame_rot = np.eye(3) if frame_rotation is None else np.asarray(frame_rotation, float)
    # Prim sits at the mount frame origin; its local rotation (frame -> optical) is R_frame^T @ R_optical.
    parent_from_optical = np.eye(4)
    parent_from_optical[:3, :3] = frame_rot.T @ base_from_camera[:3, :3]
    usd_op = usd_camera_transform(parent_from_optical).T        # transpose => Gf.Matrix4d row-major
    return ArmCameraMount(
        base_from_camera=base_from_camera,
        usd_transform_op=[[float(v) for v in row] for row in usd_op],
        aim_point=tuple(float(v) for v in aim),
        view_dir=tuple(float(v) for v in (base_from_camera[:3, :3] @ [0., 0., 1.])))
