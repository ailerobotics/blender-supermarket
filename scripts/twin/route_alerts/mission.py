"""Opt-in mission hook: run the route-alerts sensor while the robot drives.

This is the ONLY code the live mission calls. It is attached to the
``RuntimeAdapter`` (and built by ``isaac_bundle.create``) ONLY when the mission
config carries a ``route_alerts`` section; absent, the adapter never references it
and the run is byte-identical.

The hook is defensive by construction: every step is wrapped, and the FIRST failure
(no camera frame available, store-vision not importable in the live runtime, a
detector raising) disables the hook for the rest of the run with a recorded reason --
it NEVER raises into the mission tick and NEVER changes navigation or actuation on its
own. The obstruction detections it produces are exposed via :meth:`latest_detections`
for the navigation dynamic-obstacle layer (owned by ``sim-restock-dynobs-v1``) to pull
through its existing ``SensorNavigation.step(detections=...)`` seam -- this hook adds
no layer and no gate of its own.
"""
from __future__ import annotations


def build_from_config(config, *, output_dir=None, frame_builder=None, robot_id="shelfbot"):
    """Construct a :class:`RouteAlertsMissionHook` from a mission ``config`` dict.

    Returns ``None`` when ``config`` carries no ``route_alerts`` section (the caller
    then attaches nothing). Never raises: a malformed section yields a disabled hook.
    """
    section = (config or {}).get("route_alerts") if isinstance(config, dict) else None
    if not section:
        return None
    try:
        from .runner import RouteAlertsRunner

        sink = None
        if output_dir is not None:
            from pathlib import Path
            sink = Path(output_dir) / "route-alerts"
        runner = RouteAlertsRunner(section, robot_id=robot_id, sink_dir=sink)
        return RouteAlertsMissionHook(runner, frame_builder=frame_builder)
    except Exception as exc:  # pragma: no cover - defensive
        return _DisabledHook(f"build: {type(exc).__name__}: {exc}")


class _DisabledHook:
    """A hook that does nothing (records why). Keeps the adapter call site uniform."""

    def __init__(self, reason):
        self.disabled_reason = reason

    def on_navigate(self, port, packet, now):
        return None

    def latest_detections(self):
        return []

    def close(self):
        pass


class RouteAlertsMissionHook:
    """Drive the runner from the mission's camera stream during the drive."""

    def __init__(self, runner, *, frame_builder=None):
        self._runner = runner
        # frame_builder(port, packet) -> RouteFrame | None. Pulls the robot's floor/
        # aisle-facing camera frame + its calibration from the live port. Provided by
        # the integration (it owns the camera rig); absent => the hook self-disables
        # (it must not fabricate a frame or a homography).
        self._frame_builder = frame_builder
        self._disabled_reason = None
        if frame_builder is None:
            self._disabled_reason = "no frame_builder wired (integration must supply the nav camera tap)"

    @property
    def disabled_reason(self):
        return self._disabled_reason or getattr(self._runner, "disabled_reason", None)

    def on_navigate(self, port, packet, now):
        """Called each driving tick. Returns the tick's route-alerts result or None."""
        if self._disabled_reason is not None:
            return None
        try:
            frame = self._frame_builder(port, packet)
            if frame is None:
                return None
            pose = _packet_pose(packet)
            return self._runner.observe([frame], float(now), robot_pose=pose)
        except Exception as exc:  # pragma: no cover - defensive
            self._disabled_reason = f"on_navigate: {type(exc).__name__}: {exc}"
            return None

    def latest_detections(self):
        """Map-frame dynobs ``Detection`` objects from the latest tick (for navigation)."""
        try:
            return self._runner.latest_detections()
        except Exception:  # pragma: no cover - defensive
            return []

    def close(self):
        try:
            self._runner.close()
        except Exception:  # pragma: no cover - defensive
            pass


def recording_camera_frame_builder(config, *, agents_provider=None):
    """Default ``frame_builder`` for the live mission: tap the drive camera.

    Reads the robot's floor/aisle-facing camera RGB(+depth) from the live port each
    driving tick and builds a :class:`~.frames.RouteFrame`, tagging it with the
    camera id the ``route_alerts.floors`` calibration is keyed by (``route_alerts.camera``,
    default ``"route"``). The watched-floor homography therefore comes from config, not
    guessed from pose.

    GDPR-by-design: findings must be people-masked. ``agents_provider(port, packet)``
    supplies the people/cart detections to mask (from the ``sim-restock-humans-v1``
    detection stream). With no provider wired the builder returns ``None`` every tick
    so NOTHING is analysed -- it never runs an un-masked detector over people.
    """
    section = (config or {}).get("route_alerts", {}) if isinstance(config, dict) else {}
    camera_id = section.get("camera", "route")
    counter = {"n": 0}

    def _build(port, packet):
        import numpy as np
        from .frames import RouteFrame

        if agents_provider is None:
            return None  # refuse to analyse without a people-mask source (GDPR)
        cam = getattr(port, "recording_camera", None) or getattr(port, "nav_camera", None)
        if cam is None:
            return None
        frame = cam.get_current_frame()
        if not isinstance(frame, dict) or "rgb" not in frame:
            return None
        rgb = np.asarray(frame["rgb"])
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return None  # camera has not produced a real RGBA image yet
        depth = frame.get("depth")
        if depth is None:
            depth = frame.get("distance_to_image_plane")
        now = float(getattr(packet, "simulation_time", 0.0))
        counter["n"] += 1
        agents = agents_provider(port, packet) or []
        return RouteFrame(camera_id=camera_id, seq=counter["n"], rgb=rgb[:, :, :3],
                          depth=None if depth is None else np.asarray(depth),
                          capture_time=now, agents=list(agents))

    return _build


def _packet_pose(packet):
    """Best-effort (x, y, theta) map pose from the mission packet; None if absent.

    When present the runner maps base-frame footprints into the map frame; absent,
    the footprints are assumed already in the map frame.
    """
    for attr in ("pose", "map_pose", "robot_pose"):
        p = getattr(packet, attr, None)
        if p is not None and len(p) == 3:
            return (float(p[0]), float(p[1]), float(p[2]))
    return None
