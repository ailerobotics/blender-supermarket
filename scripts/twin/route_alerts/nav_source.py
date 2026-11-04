"""The ONE live navigation seam for the smart ``act`` mode.

The brief is explicit: an obstruction the robot sees on the route must feed
navigation ONLY through the moving-obstacle costmap layer the
``sim-restock-dynobs-v1`` lane owns -- never a parallel layer or gate. That layer is
fed per navigate tick by a ``source(now) -> [Detection]`` callable through
``SensorNavigation.step(detections=...)`` (the same seam the humans/scenario feed
uses, see :mod:`twin.navigation.mission_source`). This module turns the route-alerts
mission hook into exactly that callable.

Two invariants keep this honest and safe:

* It returns whatever :meth:`RouteAlertsMissionHook.latest_detections` returns, which
  is EMPTY in the findings-only working default and the confirmed obstruction
  footprints only in ``act`` mode. So wiring this source in findings mode (or before
  any frame is analysed) feeds the layer nothing and the run is byte-identical.
* It adds no layer and no gate. The dynobs layer still decides how a detection raises
  cost; this is purely the detection feed.

The builder is pure (no store-vision, no Isaac), so the whole act-mode chain --
hook -> source -> ``DynamicObstacleLayer`` -> planner detour -- is CPU-testable.
"""
from __future__ import annotations

from typing import Callable, List


def source_from_hook(hook) -> Callable[[float], list]:
    """Return ``source(now) -> [Detection]`` pulling the hook's latest detections.

    ``hook`` is a :class:`~twin.route_alerts.mission.RouteAlertsMissionHook` (or any
    object exposing ``latest_detections()``). The returned callable never raises: a
    hook that self-disabled returns ``[]`` and the layer is fed nothing.
    """
    def source(now: float) -> List:
        try:
            return list(hook.latest_detections())
        except Exception:  # pragma: no cover - defensive
            return []

    return source
