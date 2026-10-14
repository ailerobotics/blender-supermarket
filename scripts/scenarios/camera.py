"""Pin-hole camera model shared by the scene builder and the annotator.

Camera poses are authored *here* (pure Python) and pushed into Blender as an
explicit ``matrix_world``; the annotator projects ground-truth geometry with the
same matrix and intrinsics. Nothing is read back from Blender, so the rendered
pixels and the projected boxes are guaranteed consistent.

Conventions (Blender native):
  * the camera looks down its local **-Z**, local **+Y** is up, local **+X** right;
  * ``sensor_fit = HORIZONTAL`` with ``sensor_width = 36 mm`` so intrinsics are a
    pure function of (lens_mm, width_px): fx = fy = lens * W / 36, principal point
    at the image centre;
  * image coordinates are pixels with origin at the top-left, u right, v down;
  * depth is camera-space planar Z (metres) = the Blender/Cycles ``Depth`` pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
SENSOR_MM = 36.0


def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _norm(a):
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def look_at(eye: Vec3, target: Vec3, up: Vec3 = (0.0, 0.0, 1.0)) -> List[List[float]]:
    """Camera-to-world 4x4 matrix for a camera at ``eye`` looking at ``target``.

    Matches Blender's ``Vector(target-eye).to_track_quat('-Z','Y')`` orientation.
    """
    fwd = _norm(_sub(target, eye))      # camera looks along -Z, so -Z == fwd
    z = (-fwd[0], -fwd[1], -fwd[2])
    x = _norm(_cross(up, z))
    if _dot(x, x) < 1e-9:               # gaze parallel to up: pick a stable right
        x = _norm(_cross((0.0, 1.0, 0.0), z))
    y = _cross(z, x)
    return [
        [x[0], y[0], z[0], eye[0]],
        [x[1], y[1], z[1], eye[1]],
        [x[2], y[2], z[2], eye[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def intrinsics(lens_mm: float, width: int, height: int) -> dict:
    f = lens_mm * width / SENSOR_MM
    return dict(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                width=width, height=height, lens_mm=lens_mm, sensor_mm=SENSOR_MM,
                model="pinhole")


def _world_to_cam(c2w: Sequence[Sequence[float]], p: Vec3) -> Vec3:
    # inverse of a rigid transform: R^T (p - t)
    t = (c2w[0][3], c2w[1][3], c2w[2][3])
    d = _sub(p, t)
    return (
        c2w[0][0] * d[0] + c2w[1][0] * d[1] + c2w[2][0] * d[2],
        c2w[0][1] * d[0] + c2w[1][1] * d[1] + c2w[2][1] * d[2],
        c2w[0][2] * d[0] + c2w[1][2] * d[1] + c2w[2][2] * d[2],
    )


def project(c2w, K: dict, p: Vec3) -> Optional[Tuple[float, float, float]]:
    """World point -> (u, v, depth). None if behind the camera. May fall off-image."""
    xc, yc, zc = _world_to_cam(c2w, p)
    depth = -zc                      # points in front have zc < 0
    if depth <= 1e-6:
        return None
    u = K["cx"] + K["fx"] * (xc / depth)
    v = K["cy"] - K["fy"] * (yc / depth)
    return (u, v, depth)


def project_box(c2w, K, corners: Sequence[Vec3]) -> Optional[Tuple[float, float, float, float]]:
    """Axis-aligned 2D bbox (x0,y0,x1,y1) enclosing the 8 projected corners.

    Returns None if every corner is behind the camera. The box is clamped to the
    image inside ``annotate`` once visibility is decided; here it is raw pixels.
    """
    us, vs = [], []
    for c in corners:
        pr = project(c2w, K, c)
        if pr is None:
            continue
        us.append(pr[0]); vs.append(pr[1])
    if not us:
        return None
    return (min(us), min(vs), max(us), max(vs))


def clip_box(box, W: int, H: int) -> Optional[Tuple[float, float, float, float]]:
    """Clamp a pixel box to the image; None if it lies fully outside."""
    x0, y0, x1, y1 = box
    x0, x1 = max(0.0, min(x0, x1)), min(float(W), max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(float(H), max(y0, y1))
    if x1 - x0 <= 0.5 or y1 - y0 <= 0.5:
        return None
    return (x0, y0, x1, y1)


@dataclass
class CameraPose:
    """A named, fully-specified camera: intrinsics + pose, JSON-serialisable."""
    name: str
    eye: Vec3
    target: Vec3
    lens_mm: float
    width: int
    height: int
    up: Vec3 = (0.0, 0.0, 1.0)

    def c2w(self) -> List[List[float]]:
        return look_at(self.eye, self.target, self.up)

    def K(self) -> dict:
        return intrinsics(self.lens_mm, self.width, self.height)

    def to_json(self) -> dict:
        c2w = self.c2w()
        return dict(name=self.name, intrinsics=self.K(),
                    extrinsics=dict(camera_to_world=c2w,
                                    eye=list(self.eye), target=list(self.target),
                                    up=list(self.up)),
                    convention=dict(forward="-Z", up="+Y", right="+X",
                                    image_origin="top-left",
                                    depth="camera-space planar Z, metres"))
