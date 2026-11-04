"""Recorded-frames bridge: drive the route-alerts runner from a SAVED run's camera.

The brief's primary evidence path is recorded frames, not a live GPU session: a saved
mission run under ``.../mission/<run>/`` carries ``mission-frames/frames.json`` (an
ordered index of PNG frames, each with its ``simulation_time``) plus the per-frame
PNGs and ``mission.json`` beside them. This module reads that index and the PNGs and
turns them into :class:`~twin.route_alerts.frames.RouteFrame` objects stamped with the
run's REAL mission timestamps, so the production :class:`RouteAlertsRunner` can be
driven over a recorded drive with no simulator and no GPU.

Two honest limitations, labelled wherever they surface (see docs/route_alerts.md and
the v1 report's option-A follow-up):

* The recorded ``mission-frames`` are the cinematic chase-camera RGB (no depth, not a
  calibrated floor camera). The faithful "mobile floor-facing RGB-D camera on the
  route" run is the integration lane's follow-up. So for an *evidence replay* the
  hazards are SEEDED (:func:`seed_obstruction`, :func:`seed_spill`): the RGB background
  and the timestamps are the run's real recorded data; the seeded obstruction's depth +
  floor homography are synthetic (the chase cam supplies neither). The detectors and the
  whole findings/act chain are the identical production code.
* store-vision's spill/obstruction trackers are tuned for a FIXED camera; a replay over
  a drifting chase cam is soundest on the depth-protrusion obstruction path (it fits a
  floor plane per frame) and over a near-static dwell window.

Nothing here is imported by the live mission; it is a CPU replay + evidence tool.
``numpy`` is required (same as :mod:`.frames`); the PNG decode lazily imports OpenCV
then Pillow, so the index parser + the pure seeders import without an image backend.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np


@dataclass(frozen=True)
class RecordedFrameRef:
    """One entry of a saved run's ``mission-frames/frames.json``."""

    seq: int
    file: str
    simulation_time: float
    shape: tuple  # (h, w, c)


def load_frame_index(run_dir) -> List[RecordedFrameRef]:
    """Parse ``<run_dir>/mission-frames/frames.json`` into ordered frame refs.

    Accepts either the run dir (``.../<run>``) or the ``mission-frames`` dir itself.
    Raises ``FileNotFoundError`` with a clear message when the index is absent.
    """
    frames_dir = _frames_dir(run_dir)
    index = frames_dir / "frames.json"
    if not index.is_file():
        raise FileNotFoundError(f"no recorded-frames index at {index}")
    data = json.loads(index.read_text())
    refs = []
    for i, fr in enumerate(data.get("frames", [])):
        shape = tuple(int(v) for v in fr.get("shape", []))
        refs.append(RecordedFrameRef(
            seq=int(fr.get("seq", i)),
            file=str(fr["file"]),
            simulation_time=float(fr["simulation_time"]),
            shape=shape,
        ))
    if not refs:
        raise ValueError(f"recorded-frames index {index} lists no frames")
    return refs


def _frames_dir(run_dir) -> Path:
    p = Path(run_dir)
    if p.name == "mission-frames":
        return p
    return p / "mission-frames"


def _default_image_loader(path: Path) -> np.ndarray:
    """Decode a PNG to HxWx3 uint8 BGR (store-vision convention). Lazy backends."""
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError(f"OpenCV could not decode {path}")
        return np.ascontiguousarray(img[:, :, :3].astype(np.uint8))
    except ImportError:
        pass
    from PIL import Image  # type: ignore

    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB -> BGR


def iter_recorded_frames(run_dir, *, stride: int = 1, start: int = 0,
                         limit: Optional[int] = None,
                         image_loader: Optional[Callable[[Path], np.ndarray]] = None
                         ) -> Iterator[tuple]:
    """Yield ``(rgb_bgr, simulation_time, seq)`` for a saved run's recorded frames.

    ``stride`` sub-samples (the drive renders faster than a floor detector needs),
    ``start`` skips a prefix, ``limit`` caps the count. ``image_loader(path)->HxWx3``
    overrides the default OpenCV/Pillow decode (used by tests to avoid a backend).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    frames_dir = _frames_dir(run_dir)
    refs = load_frame_index(run_dir)
    load = image_loader or _default_image_loader
    emitted = 0
    for ref in refs[start::stride]:
        if limit is not None and emitted >= limit:
            return
        rgb = load(frames_dir / ref.file)
        yield np.ascontiguousarray(rgb), ref.simulation_time, ref.seq
        emitted += 1


# --------------------------------------------------------------------------- #
# Evidence seeding (NOT production): put a known spill + obstruction on recorded
# pixels so the identical detectors confirm them, with the run's real timestamps.
# Pure numpy; store-vision is never imported here.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeededFloor:
    """The synthetic floor calibration for a seeded recorded-frames replay.

    ``image_to_floor`` is a 3x3 image-pixel -> store-metre homography and
    ``floor_polygon`` the watched floor band (Nx2 image pixels). Both are synthetic
    (the chase cam is uncalibrated); they are consistent with the seeded geometry so
    the detectors' store-frame output lands where the hazard was placed.
    """

    image_to_floor: np.ndarray
    floor_polygon: np.ndarray
    aisle: dict


def floor_calibration(shape, *, aisle_x=(0.0, 4.0), aisle_y=(0.0, 6.0),
                      floor_top_frac: float = 0.5, aisle_id: str = "route-aisle",
                      aisle_min_width_m: float = 0.9) -> SeededFloor:
    """A simple linear image->floor homography over the lower ``floor`` band.

    The bottom image row maps to ``aisle_y[0]`` (near the camera) and the band top to
    ``aisle_y[1]`` (far); columns map linearly across ``aisle_x``. Returns the
    homography, the watched-floor polygon, and a matching :class:`AisleSpec`-kwargs
    dict so an obstruction footprint can block the corridor.
    """
    h, w = int(shape[0]), int(shape[1])
    v_top = float(h) * float(floor_top_frac)
    x0, x1 = float(aisle_x[0]), float(aisle_x[1])
    y0, y1 = float(aisle_y[0]), float(aisle_y[1])
    sx = (x1 - x0) / max(w - 1, 1)
    sy = (y1 - y0) / max(h - 1 - v_top, 1.0)
    # X = x0 + sx*u ;  Y = y0 + sy*((h-1) - v)
    itf = np.array([
        [sx, 0.0, x0],
        [0.0, -sy, sy * (h - 1) + y0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    poly = np.array([[0, v_top], [w - 1, v_top], [w - 1, h - 1], [0, h - 1]],
                    dtype=np.float64)
    aisle = dict(x0=x0, y0=y0, x1=x1, y1=y1, axis="y", min_width_m=aisle_min_width_m)
    return SeededFloor(image_to_floor=itf, floor_polygon=poly, aisle={aisle_id: aisle})


def seed_obstruction(shape, *, box_px, floor_depth_m: float = 3.0,
                     box_depth_m: float = 2.0) -> np.ndarray:
    """Synthetic depth HxW (float32): a flat floor with a box protruding toward the cam.

    ``box_px`` is ``(u0, v0, u1, v1)`` image pixels. Floor pixels read ``floor_depth_m``;
    box pixels read ``box_depth_m`` (closer), so ``floor_z - depth`` clears store-vision's
    ``min_object_height_m`` and :func:`occupancy_from_depth` marks the box occupied.
    """
    h, w = int(shape[0]), int(shape[1])
    depth = np.full((h, w), float(floor_depth_m), dtype=np.float32)
    u0, v0, u1, v1 = (int(round(v)) for v in box_px)
    depth[max(v0, 0):min(v1, h), max(u0, 0):min(u1, w)] = float(box_depth_m)
    return depth


def seed_spill(rgb, *, center_px, radius_px: int = 36,
               color=(18, 22, 30)) -> np.ndarray:
    """Composite a dark spill blob onto a COPY of ``rgb`` (BGR uint8). Returns the copy.

    A high-contrast ellipse on the floor band is a clean change from the learned
    clean-floor model, so store-vision's spill detector confirms it. Uses OpenCV when
    present, else a numpy disc.
    """
    out = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).copy())
    cx, cy = int(center_px[0]), int(center_px[1])
    bgr = tuple(int(c) for c in color)
    try:
        import cv2

        cv2.ellipse(out, (cx, cy), (int(radius_px * 1.3), int(radius_px)), 0, 0, 360,
                    bgr, -1)
        cv2.ellipse(out, (cx, cy), (int(radius_px * 1.3), int(radius_px)), 0, 0, 360,
                    tuple(min(c + 30, 255) for c in bgr), 2)
        return out
    except ImportError:
        h, w = out.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        disc = ((xx - cx) ** 2) / float(radius_px * 1.3) ** 2 + \
               ((yy - cy) ** 2) / float(radius_px) ** 2 <= 1.0
        out[disc] = np.array(bgr, dtype=np.uint8)
        return out
