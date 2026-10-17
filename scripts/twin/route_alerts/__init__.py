"""Route alerts: the mission robot as a mobile store sensor (opt-in, default off).

While the robot tows the cart down the aisle its cameras see the floor and
shelves. This package bridges that camera stream to the read-only ``store-vision``
detectors (spill, obstruction, strays / cold-chain) so the drive produces the same
findings + staff tasks a fixed store camera would, and feeds the obstruction
passability footprints to navigation THROUGH the moving-obstacle costmap layer the
`sim-restock-dynobs-v1` lane owns (``twin.navigation.dynamic``) -- never a parallel
layer.

Design invariants (see docs/route_alerts.md):

* **Opt-in / byte-identical when absent.** Nothing here runs unless a mission
  config carries a ``route_alerts`` section (CLI ``--route-alerts``). With it
  absent the mission is byte-identical: no import, no detector, no extra render.
* **store-vision is read-only.** Its analyzers are imported and driven in place;
  every import is lazy so this package imports (and its pure seams test) without
  store-vision installed. If store-vision cannot be imported in the live runtime,
  the runner disables itself with a logged reason and never crashes the mission.
* **Add-only navigation coupling.** Obstruction footprints become
  ``twin.navigation.dynamic.Detection`` objects handed to the EXISTING
  ``SensorNavigation.step(detections=...)`` seam; the dynobs layer only ever ADDS
  occupancy, so a route alert can turn a stop into motion, never the reverse.

The GPU/CPU path per detector is stated in :mod:`.detectors` and the report.
"""
from .config import DetectorPref, RouteAlertsConfig
from .costmap import obstructions_to_detections
from .events import build_staff_tasks, to_robot_events
from .frames import RouteFrame
from .runner import RouteAlertsResult, RouteAlertsRunner

__all__ = [
    "DetectorPref",
    "RouteAlertsConfig",
    "RouteFrame",
    "RouteAlertsRunner",
    "RouteAlertsResult",
    "obstructions_to_detections",
    "to_robot_events",
    "build_staff_tasks",
]
