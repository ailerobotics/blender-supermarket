"""Reproject the fixed-source collect target into the arm eye-in-hand camera image.

The collect stage senses the pick carton with the FIXED ``source`` station camera against
its commissioned baseline (``twin.restock_camera.source_target``); the carton pose comes back
in the robot base (``ur_base_link``) frame. The arm camera is masked-depth/fine-alignment only,
so the host needs the grasp-target box expressed in the ARM camera image that backs the masked
depth. This module reprojects the base-frame carton volume through the live arm camera pose +
intrinsics into that image ROI, reusing exactly the projection of
``IsaacCartonSensors._object_bbox`` (catalog carton box anchored to the sensed pose).

Pure NumPy, no Isaac/USD: the live arm ``base_from_camera`` and intrinsics are read by the
integration caller (``isaac_factory``) and passed in, so the geometry is CPU-tested here.
See ``data/mobile-restock-v2/report.md`` section "Collect sensing contract".
"""
from __future__ import annotations

import math

import numpy as np


def reproject_carton_roi(pose_base, dimensions, base_from_camera, intrinsics, depth_shape):
    """ROI ``(u1,v1,u2,v2)`` of a base-frame carton in the arm camera image, or ``None``.

    ``pose_base`` = ``(x,y,z, qx,qy,qz,qw)`` carton pose in the camera's base frame (the
    ``source`` target's ``pose_station``, which is the ur_base_link frame). ``dimensions`` =
    ``(dx,dy,dz)`` catalog carton size (m). ``base_from_camera`` = 4x4 mapping the arm camera
    optical frame to that base (live pose, same convention as ``camera.get_intrinsics_matrix``).
    ``intrinsics`` = ``(fx,fy,cx,cy)``. ``depth_shape`` = ``(height,width)`` of the masked depth.

    The 8 carton corners (top face at the sensed pose, extruded down by ``dz``, matching
    ``IsaacCartonSensors._object_bbox``) are projected with the pinhole; the ROI is clamped to
    the image (the host carves the target within the visible region). Returns ``None`` when the
    target is wholly behind the camera or projects entirely outside the image — a box is never
    fabricated; the bridge then republishes its last good box for the 0.5 s host freshness.
    """
    pose = np.asarray(pose_base, float)
    x, y, z = pose[:3]
    yaw = 2 * math.atan2(float(pose[5]), float(pose[6]))
    c, s = math.cos(yaw), math.sin(yaw)
    dx2, dy2, dz = dimensions[0] / 2., dimensions[1] / 2., dimensions[2]
    corners = np.array([[x + c * ddx - s * ddy, y + s * ddx + c * ddy, z - ddz]
                        for ddx in (-dx2, dx2) for ddy in (-dy2, dy2) for ddz in (0., dz)])
    camera_from_base = np.linalg.inv(np.asarray(base_from_camera, float))
    p = corners @ camera_from_base[:3, :3].T + camera_from_base[:3, 3]
    front = p[p[:, 2] > 1e-6]
    if len(front) < 4:                       # wholly (or almost) behind the arm camera
        return None
    fx, fy, cx, cy = intrinsics
    uv = front[:, :2] / front[:, 2, None] * [fx, fy] + [cx, cy]
    height, width = int(depth_shape[0]), int(depth_shape[1])
    lo = np.floor(uv.min(axis=0)).astype(int)
    hi = np.ceil(uv.max(axis=0)).astype(int) + 1
    lo = np.maximum(lo, [0, 0])
    hi = np.minimum(hi, [width, height])
    if hi[0] <= lo[0] or hi[1] <= lo[1]:     # projects entirely outside the image
        return None
    return (int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1]))
