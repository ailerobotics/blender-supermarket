"""The camera-stream bridge type: a mission frame -> a store-vision ``Frame``.

``RouteFrame`` is what the mission hands the runner each sensed tick: one camera's
RGB (``HxWx3`` uint8, BGR -- OpenCV convention, as store-vision expects) and optional
depth (``HxW`` float32 metres), plus the per-camera floor calibration (watched-floor
polygon in image pixels + a 3x3 image->store homography). The calibration can live
in :class:`~.config.RouteAlertsConfig.floors` instead; a frame-level value overrides
it so a moving camera can supply a per-pose homography.

store-vision is imported lazily in :meth:`RouteFrame.to_store_frame` so this module
imports without it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RouteFrame:
    """One camera observation offered to the route-alerts runner."""

    camera_id: str
    seq: int
    rgb: np.ndarray
    capture_time: float
    depth: np.ndarray | None = None
    zone_id: str | None = None
    #: 3x3 image->store homography (overrides the config floor's, if given).
    image_to_floor: np.ndarray | None = None
    #: Nx2 image-pixel polygon of the watched floor (overrides the config floor's).
    floor_polygon: np.ndarray | None = None
    #: People / cart detections to MASK OUT (GDPR-by-design: findings are
    #: people-masked, no biometrics). Each item is a store-vision ``Detection``
    #: (passed through) or a dict ``{"bbox": (x, y, w, h), "label": "person",
    #: "score": ..., "mask": <HxW bool, optional>}``. The humans lane's detections
    #: feed straight in. Empty => nothing masked.
    agents: list = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("RouteFrame.rgb must be HxWx3")
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        self.rgb = np.ascontiguousarray(rgb)
        if self.depth is not None:
            d = np.asarray(self.depth, dtype=np.float32)
            if d.shape[:2] != self.rgb.shape[:2]:
                raise ValueError("RouteFrame.depth must match rgb HxW")
            self.depth = d
        if self.image_to_floor is not None:
            self.image_to_floor = np.asarray(self.image_to_floor, dtype=np.float64).reshape(3, 3)

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        h, w = self.rgb.shape[:2]
        return (w, h)

    def _agent_detections(self):
        """Convert :attr:`agents` to store-vision ``Detection`` objects (lazy)."""
        from storevision.core.types import BBox, Detection

        out = []
        for ag in self.agents:
            if isinstance(ag, Detection):
                out.append(ag)
                continue
            if isinstance(ag, dict):
                bb = ag["bbox"]
                out.append(Detection(
                    camera_id=self.camera_id, time=float(self.capture_time),
                    bbox=bb if isinstance(bb, BBox) else BBox(*(float(v) for v in bb)),
                    label=str(ag.get("label", "person")), score=float(ag.get("score", 1.0)),
                    mask=ag.get("mask")))
                continue
            # (x, y, w, h) tuple
            out.append(Detection(camera_id=self.camera_id, time=float(self.capture_time),
                                 bbox=BBox(*(float(v) for v in ag)), label="person"))
        return out

    def to_store_frame(self):
        """Build a store-vision ``core.types.Frame`` (lazy import).

        People/cart detections are folded into ``meta['agents']`` so the detectors
        mask them out (GDPR-by-design); any agents already in ``meta`` are kept.
        """
        from storevision.core.types import Frame

        meta = dict(self.meta)
        if self.agents:
            meta["agents"] = list(meta.get("agents", [])) + self._agent_detections()
        return Frame(
            camera_id=self.camera_id,
            seq=int(self.seq),
            rgb=self.rgb,
            capture_time=float(self.capture_time),
            depth=self.depth,
            meta=meta,
        )
