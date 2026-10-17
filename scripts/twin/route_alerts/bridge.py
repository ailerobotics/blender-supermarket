"""Drive the read-only store-vision detectors from the mission camera stream.

:class:`StoreVisionBridge` builds the enabled store-vision analyzers once (spill,
obstruction -- both classical CPU by default; the optional products-recognizer-fed
strays detector is driven separately), a minimal synthetic ``StoreConfig`` + a
persisted ``AnalyzerContext``, and on each :meth:`process` call turns the offered
:class:`~.frames.RouteFrame` objects into store-vision ``Event`` findings plus the
confirmed obstruction footprints + passability layer.

Every store-vision import is lazy and wrapped: if store-vision (or a GPU model it
needs) cannot be imported, the bridge records :attr:`disabled_reason` and returns
empty outputs forever -- it never raises into the mission tick.

Fixed-camera caveat (honest limitation, see docs/route_alerts.md): store-vision's
spill/obstruction build a clean-floor background + a persistence tracker tuned for a
FIXED camera. On the robot they are soundest on the depth-protrusion path (obstruction
fits a floor plane per frame) and while the base is slowed/stopped at a hazard (the nav
gate stops for obstacles, giving the tracker consistent frames). The camera->floor
homography is treated as the rigid mount transform; mapping base-frame footprints into
the global map frame is the caller's job via the frame pose.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from .config import RouteAlertsConfig
from .frames import RouteFrame


def _accepted(tuning: dict, cls) -> dict:
    """Keep only ``tuning`` keys that ``cls`` (a dataclass config) actually accepts,
    so a tunable meant for one detector is silently ignored by the other."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in tuning.items() if k in names}


@dataclass
class BridgeOutput:
    """What one :meth:`StoreVisionBridge.process` call produced."""

    events: list = field(default_factory=list)  # store-vision core.types.Event
    obstructions: list = field(default_factory=list)  # obstruction.passability.Obstruction
    passability: dict = field(default_factory=dict)  # PassabilityLayer.to_dict()

    def is_empty(self) -> bool:
        return not (self.events or self.obstructions)


class StoreVisionBridge:
    """Hold the enabled analyzers + synthetic store, process route frames."""

    def __init__(self, config: RouteAlertsConfig):
        self.config = config
        self.disabled_reason: str | None = None
        self._built = False
        self._spill = None
        self._obstruction = None
        self._store = None
        self._state: dict = {}
        self._tick = 0
        self._floor_cfgs: dict = {}  # camera_id -> FloorRegionConfig

    # ------------------------------------------------------------- build
    def _floor_for_frame(self, frame: RouteFrame):
        """A store-vision ``FloorRegionConfig`` for this camera (cached).

        Prefers the frame's own calibration, else ``config.floors[camera_id]``, else
        a default full-image floor with an identity homography (pixels==store units,
        valid for the near-nadir synthetic bench and any caller that pre-rectifies).
        """
        import numpy as np
        from storevision.spill.config import FloorRegionConfig

        cam = frame.camera_id
        if cam in self._floor_cfgs:
            return self._floor_cfgs[cam]
        raw = self.config.floors.get(cam, {})
        w, h = frame.size
        polygon = frame.floor_polygon
        if polygon is None and raw.get("polygon") is not None:
            polygon = np.asarray(raw["polygon"], dtype=np.float64)
        if polygon is None:
            polygon = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
        itf = frame.image_to_floor
        if itf is None and raw.get("image_to_floor") is not None:
            itf = np.asarray(raw["image_to_floor"], dtype=np.float64).reshape(3, 3)
        zone = frame.zone_id or raw.get("zone_id")
        fcfg = FloorRegionConfig(camera_id=cam, polygon=polygon, image_to_floor=itf, zone_id=zone)
        self._floor_cfgs[cam] = fcfg
        return fcfg

    def _lazy_build(self, frames: list[RouteFrame]) -> bool:
        """Import store-vision + build the analyzers. Returns False (and sets
        ``disabled_reason``) if anything is missing; the bridge then no-ops."""
        if self._built:
            return self.disabled_reason is None
        self._built = True
        try:
            from storevision.core.config import CameraSpec, FloorMap, StoreConfig

            floors = {f.camera_id: self._floor_for_frame(f) for f in frames}
            cameras = [
                CameraSpec(id=f.camera_id, kind=("rgbd" if f.depth is not None else "rgb"),
                           size=(f.size[0], f.size[1]), role="aisle")
                for f in frames
            ]
            self._store = StoreConfig(
                name=self.config.store_name,
                floor=FloorMap(width=100.0, depth=100.0, origin=(0.0, 0.0)),
                cameras=cameras,
            )
            if self.config.spill.enabled:
                self._spill = self._build_spill(floors)
            if self.config.obstruction.enabled:
                self._obstruction = self._build_obstruction(floors)
        except Exception as exc:  # pragma: no cover - env-dependent
            self.disabled_reason = f"{type(exc).__name__}: {exc}"
            self._spill = self._obstruction = None
            return False
        return True

    def _build_spill(self, floors):
        from storevision.spill.analyzer import build_analyzer
        from storevision.spill.config import SpillConfig

        pref = self.config.spill
        kw = dict(floors=floors, use_open_vocab=pref.gpu,
                  open_vocab_device=pref.device if pref.gpu else "cpu")
        kw.update(_accepted(pref.tuning, SpillConfig))
        cfg = SpillConfig(**kw)
        return build_analyzer(config=cfg, use_open_vocab=pref.gpu)

    def _build_obstruction(self, floors):
        from storevision.obstruction.analyzer import build_analyzer
        from storevision.obstruction.config import AisleSpec, ObstructionConfig

        pref = self.config.obstruction
        aisles = {aid: AisleSpec(aisle_id=aid, **kw) for aid, kw in self.config.aisles.items()}
        kw = dict(floors=floors, aisles=aisles, use_depth=True, use_open_vocab=pref.gpu,
                  open_vocab_device=pref.device if pref.gpu else "cpu")
        kw.update(_accepted(pref.tuning, ObstructionConfig))
        cfg = ObstructionConfig(**kw)
        return build_analyzer(config=cfg, use_open_vocab=pref.gpu)

    # ------------------------------------------------------------- process
    def process(self, frames: list[RouteFrame], now: float) -> BridgeOutput:
        """Run the enabled detectors over this tick's frames; aggregate findings."""
        out = BridgeOutput()
        if self.disabled_reason is not None:
            return out
        if not frames:
            return out
        if not self._lazy_build(frames):
            return out
        self._tick += 1
        store_frames = [f.to_store_frame() for f in frames]
        try:
            from storevision.core.analyzer import AnalyzerContext

            ctx = AnalyzerContext(store=self._store, tick=self._tick, now=now, state=self._state)
            for analyzer in (self._spill, self._obstruction):
                if analyzer is None:
                    continue
                res = analyzer.process(store_frames, ctx)
                out.events.extend(res.events)
            if self._obstruction is not None:
                out.obstructions = list(self._obstruction.obstructions.values())
                out.passability = self._obstruction.layer.to_dict()
        except Exception as exc:  # pragma: no cover - env-dependent
            self.disabled_reason = f"process: {type(exc).__name__}: {exc}"
            return BridgeOutput()
        return out

    @property
    def store(self):
        return self._store
