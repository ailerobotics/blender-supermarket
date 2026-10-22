"""Pure helpers for the DEMONSTRATIVE humans-bench view.

The default bench records the robot-eye candidate camera, which is aimed at the
shelf support and does not frame the person (the reaction still fires on the
person's ground-truth position). For evidence a reviewer can SEE, this module adds
a third-person OVERVIEW camera that frames the robot and the scripted person
together, draws the ground-truth person box on each frame, and overlays the safety
state (CLEAR/SLOW/STOP, base scale, arm HOLD/run).

Everything here is numpy + PIL only (no Isaac/USD), so the overview pose, the
ground-truth box projection and the drawing are unit-tested offline. The Isaac-side
camera authoring + render lives in ``gpu_bench.run``.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

Pose2D = Tuple[float, float, float]

# Zone -> banner/box RGB (matches the CPU demo's top-down SVG palette).
ZONE_RGB = {'CLEAR': (46, 125, 50), 'SLOW': (249, 168, 37), 'STOP': (198, 40, 40)}


def overview_look_at(base_xy: Sequence[float], forward_xy: Sequence[float],
                     near_m: float, far_m: float, *, side_m: float = 5.0,
                     height_m: float = 3.2, target_z: float = 0.9):
    """Eye/target for a third-person camera framing the robot + the person path.

    The person walks along ``forward_xy`` from ``near_m`` to ``far_m`` in front of
    the robot ``base_xy``; the camera sits ``side_m`` to the left and ``height_m``
    up, looking at the middle of that segment, so both the robot and the whole
    crossing are in frame.
    """
    bx, by = float(base_xy[0]), float(base_xy[1])
    fx, fy = float(forward_xy[0]), float(forward_xy[1])
    n = math.hypot(fx, fy) or 1.0
    fx, fy = fx / n, fy / n
    mid = (float(near_m) + float(far_m)) / 2.0
    tx, ty = bx + fx * mid, by + fy * mid
    lx, ly = -fy, fx                                   # left normal of forward
    eye = (tx + lx * side_m, ty + ly * side_m, float(height_m))
    target = (tx, ty, float(target_z))
    return eye, target


def person_image_box(person_xyz: Sequence[float], radius: float, height: float,
                     world_from_optical, intrinsics) -> Optional[Tuple[int, int, int, int]]:
    """Project a person's AABB into the overview image; ``None`` if not visible.

    ``world_from_optical`` is the camera's ``look_at`` matrix (parent = world);
    ``intrinsics`` a ``twin.restock_camera.geometry.Intrinsics`` or its kwargs.
    """
    from twin.restock_camera.geometry import Intrinsics
    k = intrinsics if isinstance(intrinsics, Intrinsics) else Intrinsics(**intrinsics)
    cx, cy = float(person_xyz[0]), float(person_xyz[1])
    r = float(radius)
    corners = [(cx + dx, cy + dy, z)
               for dx in (-r, r) for dy in (-r, r) for z in (0.0, float(height))]
    m = np.asarray(world_from_optical, float)
    opt_from_world = np.linalg.inv(m)
    pts = np.asarray(corners, float) @ opt_from_world[:3, :3].T + opt_from_world[:3, 3]
    front = pts[pts[:, 2] > k.near]
    if len(front) == 0:
        return None
    uv = front[:, :2] / front[:, 2, None] * [k.fx, k.fy] + [k.cx, k.cy]
    x0, y0, x1, y1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
    x0c, y0c = max(0.0, x0), max(0.0, y0)
    x1c, y1c = min(k.width - 1.0, x1), min(k.height - 1.0, y1)
    if x1c <= x0c or y1c <= y0c:
        return None                                    # fully off-frame
    return (int(x0c), int(y0c), int(x1c), int(y1c))


def draw_overlay(rgb, zone: str, base_scale: float, arm_hold: bool,
                 boxes: Sequence[Tuple[int, int, int, int]], *, t: Optional[float] = None,
                 source: str = 'ground_truth'):
    """Draw the GT person boxes + a safety-state banner on an RGB uint8 frame."""
    from PIL import Image, ImageDraw
    arr = np.asarray(rgb)[:, :, :3].astype('uint8').copy()
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    color = ZONE_RGB.get(zone, (120, 120, 120))
    for (x0, y0, x1, y1) in boxes:
        d.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=3)
        d.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], outline=color, width=2)
        d.text((x0 + 3, max(0, y0 - 12)), 'person (' + source + ')', fill=(255, 255, 255))
    w = img.width
    d.rectangle([0, 0, w, 22], fill=color)
    tpart = ('t=%.1fs  ' % t) if t is not None else ''
    label = f"{tpart}{zone}   base x{base_scale:.2f}   arm {'HOLD' if arm_hold else 'run'}"
    d.text((6, 5), label, fill=(255, 255, 255))
    return np.asarray(img)


def frame_span_from_people(people, base_xy: Sequence[float]) -> Tuple[float, float]:
    """(near, far) distances of the people's waypoints from the base, for framing."""
    bx, by = float(base_xy[0]), float(base_xy[1])
    dists = [math.hypot(wx - bx, wy - by)
             for p in people for (wx, wy) in p.waypoints]
    if not dists:
        return (0.9, 6.5)
    return (min(dists), max(dists))
