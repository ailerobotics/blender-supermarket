"""Route-alerts configuration -- pure dataclasses, no store-vision / numpy import.

A mission opts in by carrying a ``route_alerts`` object in its runtime-config JSON
(or ``--route-alerts`` on the launcher, which injects the defaults). Absent => the
whole subsystem is dormant and the run is byte-identical.

Per-detector GPU/CPU path (``DetectorPref.gpu``):

* ``spill``       -- classical CPU is the base pipeline (OpenCV only); ``gpu=True``
                     adds the store-vision OWLv2 + SAM 2 open-vocab refine
                     (``store-vision[spill]``: torch/transformers), which SHARES the
                     GPU with the running mission. Default CPU.
* ``obstruction`` -- classical CPU (depth protrusion + texture) is the base pipeline
                     and also what feeds navigation; ``gpu=True`` adds the OWLv2
                     open-vocab object classifier (label quality only). Default CPU.
* ``strays``      -- SEMANTIC: needs SKU-per-slot observations from the store-vision
                     products recognizer (OWLv2 / SigLIP, GPU). No classical CPU
                     equivalent, so it only runs when a ``slot_observation`` provider
                     is wired; ``gpu`` records that the recognizer is GPU-backed.

``chiller_door`` has no dedicated per-frame store-vision detector today (openclose is
a semantic checklist, not a pixel detector); the runner leaves a pluggable slot and
the report proposes one. It is accepted in config but inert without a provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class DetectorPref:
    """Per-detector enable + GPU-path preference.

    ``tuning`` is an opaque passthrough merged into the store-vision detector config
    constructor (only the keys it accepts). Production leaves it empty and uses the
    fixed-camera defaults (``persist_seconds`` ~6-8 s); a caller that wants faster
    reaction on the drive -- or the short synthetic bench -- can set e.g.
    ``{"persist_seconds": 0.4, "warmup_frames": 2}``. It never changes GPU/CPU choice.
    """

    enabled: bool = False
    gpu: bool = False
    #: open-vocab torch device when ``gpu`` (store-vision honours "cuda"/"cpu").
    device: str = "cuda"
    tuning: dict = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj) -> "DetectorPref":
        if obj is None:
            return cls()
        if isinstance(obj, bool):
            return cls(enabled=obj)
        if isinstance(obj, DetectorPref):
            return obj
        if isinstance(obj, dict):
            return cls(
                enabled=bool(obj.get("enabled", True)),
                gpu=bool(obj.get("gpu", False)),
                device=str(obj.get("device", "cuda")),
                tuning=dict(obj.get("tuning", {})),
            )
        raise ValueError(f"invalid detector preference: {obj!r}")


@dataclass(frozen=True)
class RouteAlertsConfig:
    """Everything the route-alerts runner needs, all configuration.

    * ``spill`` / ``obstruction`` / ``strays`` -- per-detector preferences.
    * ``cadence`` -- run the detectors every Nth offered frame (>=1); the drive
      renders faster than a floor detector needs (store-vision's own
      ``min_hz``/``max_hz`` further rate-limit inside the analyzers).
    * ``staff_tasks`` -- also materialise de-duplicated store-vision staff tasks
      from the findings (CLEANING for spill/obstruction, STRAY for cold-chain).
    * ``floors`` -- per-camera floor region: ``{camera_id: {polygon, image_to_floor,
      zone_id}}`` (image pixels + a 3x3 image->store homography). Required for a
      camera to be watched; a frame can also carry its own calibration.
    * ``aisles`` -- store-frame aisle corridors for the passability layer
      (``obstruction.config.AisleSpec`` kwargs). Optional; without them obstruction
      still emits per-object footprints + ``obstruction.detected`` events.
    * ``obstruction_source`` / ``footprint_pad_m`` -- the ``source`` tag and extra
      plan-view padding (m) for the dynobs ``Detection`` objects fed to navigation.
    * ``store_name`` -- cosmetic label for the synthetic store config.
    """

    spill: DetectorPref = DetectorPref(enabled=True, gpu=False)
    obstruction: DetectorPref = DetectorPref(enabled=True, gpu=False)
    strays: DetectorPref = DetectorPref(enabled=False, gpu=True)
    cadence: int = 1
    staff_tasks: bool = True
    floors: dict = field(default_factory=dict)
    aisles: dict = field(default_factory=dict)
    obstruction_source: str = "route_obstruction"
    footprint_pad_m: float = 0.0
    store_name: str = "route-alerts"

    def __post_init__(self) -> None:
        if type(self.cadence) is not int or self.cadence < 1:
            raise ValueError("route-alerts cadence must be a positive int")
        if not (self.footprint_pad_m >= 0):
            raise ValueError("footprint_pad_m must be >= 0")
        if not isinstance(self.floors, dict) or not isinstance(self.aisles, dict):
            raise ValueError("floors/aisles must be objects")

    @property
    def any_enabled(self) -> bool:
        return self.spill.enabled or self.obstruction.enabled or self.strays.enabled

    @classmethod
    def from_obj(cls, obj) -> "RouteAlertsConfig":
        """Build from a mission config value (``True`` => defaults, or a dict)."""
        if obj is True:
            return cls()
        if obj in (None, False):
            raise ValueError("route-alerts config absent")
        if isinstance(obj, RouteAlertsConfig):
            return obj
        if not isinstance(obj, dict):
            raise ValueError(f"invalid route_alerts config: {type(obj).__name__}")
        base = cls()
        return replace(
            base,
            spill=DetectorPref.from_obj(obj.get("spill", base.spill)),
            obstruction=DetectorPref.from_obj(obj.get("obstruction", base.obstruction)),
            strays=DetectorPref.from_obj(obj.get("strays", base.strays)),
            cadence=int(obj.get("cadence", base.cadence)),
            staff_tasks=bool(obj.get("staff_tasks", base.staff_tasks)),
            floors=dict(obj.get("floors", base.floors)),
            aisles=dict(obj.get("aisles", base.aisles)),
            obstruction_source=str(obj.get("obstruction_source", base.obstruction_source)),
            footprint_pad_m=float(obj.get("footprint_pad_m", base.footprint_pad_m)),
            store_name=str(obj.get("store_name", base.store_name)),
        )
