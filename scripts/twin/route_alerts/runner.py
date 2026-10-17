"""The route-alerts runner: one call per sensed drive tick, wired by the mission.

``RouteAlertsRunner.observe(frames, now)`` is the whole subsystem from the mission's
point of view. It:

1. rate-limits by ``config.cadence`` (the drive renders faster than a floor detector
   needs);
2. runs the store-vision detectors over the frames (:class:`.bridge.StoreVisionBridge`);
3. optionally runs the semantic strays / cold-chain detector when the caller wires the
   products-recognizer-fed providers (GPU path; off by default);
4. mirrors every finding into ``robot.route_alert.*`` events and (optionally)
   de-duplicated staff tasks;
5. turns confirmed obstruction footprints into ``twin.navigation.dynamic.Detection``
   objects -- transformed into the map frame by the robot pose when given -- for the
   caller to pass to ``SensorNavigation.step(detections=...)`` (the dynobs seam);
6. appends a JSONL evidence record per tick when a sink directory is set.

Never raises into the mission tick: a store-vision import/runtime failure disables the
runner (``disabled_reason``) and every later call returns an empty result.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .bridge import StoreVisionBridge
from .config import RouteAlertsConfig
from .costmap import obstructions_to_detections
from .events import build_staff_tasks, to_robot_events
from .frames import RouteFrame


@dataclass
class RouteAlertsResult:
    """Outcome of one :meth:`RouteAlertsRunner.observe` call."""

    ran: bool = False  # False when skipped by cadence / disabled
    events: list = field(default_factory=list)  # store-vision findings
    robot_events: list = field(default_factory=list)  # robot.route_alert.* mirrors
    detections: list = field(default_factory=list)  # dynobs Detection for navigation
    tasks: list = field(default_factory=list)  # store-vision StaffTask
    passability: dict = field(default_factory=dict)
    disabled_reason: str | None = None


def _transform_detection(det, pose):
    """Map a base-frame ``Detection`` into the map frame by ``pose=(x,y,theta)``."""
    from twin.navigation.dynamic import Detection

    x, y, th = pose
    bx, by = det.position
    c, s = math.cos(th), math.sin(th)
    return Detection(position=(x + c * bx - s * by, y + s * bx + c * by),
                     radius=det.radius, source=det.source)


class RouteAlertsRunner:
    """Opt-in mobile store sensor. Construct only when ``route_alerts`` is configured."""

    def __init__(self, config, *, sink_dir=None, robot_id="shelfbot",
                 strays_detector=None, slot_observation_provider=None):
        self.config = RouteAlertsConfig.from_obj(config)
        self.robot_id = robot_id
        self.bridge = StoreVisionBridge(self.config)
        self._offer = 0
        self._latest_detections: list = []
        self._task_gen = None
        # Semantic strays / cold-chain: supply BOTH a detector (duck-typed
        # ``.process(observations)->[Event]``, e.g. store-vision StrayDetector) and a
        # ``slot_observation_provider(frames, now)->[SlotObservation]`` from the
        # products recognizer (GPU). Absent => strays stays off.
        self._strays_detector = strays_detector
        self._slot_observation_provider = slot_observation_provider
        self._sink = None
        if sink_dir is not None:
            d = Path(sink_dir)
            d.mkdir(parents=True, exist_ok=True)
            self._sink = (d / "route-alerts.jsonl").open("a", buffering=1)

    @property
    def disabled_reason(self):
        return self.bridge.disabled_reason

    def observe(self, frames, now: float, *, robot_pose=None) -> RouteAlertsResult:
        """Process one tick's frames. ``frames`` is a list of :class:`RouteFrame`."""
        self._offer += 1
        if self._offer % self.config.cadence != 0:
            return RouteAlertsResult(ran=False)
        if not isinstance(frames, (list, tuple)):
            frames = [frames]
        frames = [f for f in frames if isinstance(f, RouteFrame)]
        out = self.bridge.process(frames, now)

        events = list(out.events)
        events.extend(self._run_strays(frames, now))

        robot_events = to_robot_events(events, robot_id=self.robot_id) if events else []

        detections = obstructions_to_detections(
            out.obstructions, source=self.config.obstruction_source,
            pad_m=self.config.footprint_pad_m)
        if robot_pose is not None:
            detections = [_transform_detection(d, robot_pose) for d in detections]
        self._latest_detections = detections

        tasks = []
        if self.config.staff_tasks and events and self.bridge.store is not None \
                and self.disabled_reason is None:
            try:
                tasks = self._staff_tasks(events, now)
            except Exception as exc:  # pragma: no cover - env-dependent
                self.bridge.disabled_reason = f"tasks: {type(exc).__name__}: {exc}"

        result = RouteAlertsResult(
            ran=True, events=events, robot_events=robot_events, detections=detections,
            tasks=tasks, passability=out.passability, disabled_reason=self.disabled_reason)
        self._write(now, result)
        return result

    def _run_strays(self, frames, now):
        if not (self.config.strays.enabled and self._strays_detector
                and self._slot_observation_provider):
            return []
        try:
            observations = self._slot_observation_provider(frames, now)
            return list(self._strays_detector.process(observations))
        except Exception as exc:  # pragma: no cover - env-dependent
            self.bridge.disabled_reason = f"strays: {type(exc).__name__}: {exc}"
            return []

    def _staff_tasks(self, events, now):
        if self._task_gen is None:
            from storevision.tasks.generation import TaskGenerator
            from storevision.tasks.store import StaffTaskStore
            # The generator's store is a StaffTaskStore (open-task dedup/fold), NOT the
            # StoreConfig; persisted across calls so a still-open hazard folds, not dupes.
            self._task_store = StaffTaskStore()
            self._task_gen = TaskGenerator(self._task_store)
        return build_staff_tasks(events, self._task_store, now=now, generator=self._task_gen)

    def latest_detections(self):
        """The most recent map-frame dynobs detections (for a pull-based nav provider)."""
        return list(self._latest_detections)

    def _write(self, now, result: RouteAlertsResult):
        if self._sink is None:
            return
        rec = {
            "t": now,
            "events": [e.to_dict() for e in result.events],
            "robot_events": [e.to_dict() for e in result.robot_events],
            "detections": [{"position": list(d.position), "radius": d.radius, "source": d.source}
                           for d in result.detections],
            "tasks": [t.to_dict() for t in result.tasks],
            "passability": result.passability,
            "disabled_reason": result.disabled_reason,
        }
        self._sink.write(json.dumps(rec, allow_nan=False) + "\n")

    def close(self):
        if self._sink is not None:
            try:
                self._sink.close()
            finally:
                self._sink = None
