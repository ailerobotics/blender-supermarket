"""Build a deterministic default crowd from the mission's own goal poses.

The brief asks for scripted people whose paths cross the robot's route and
approach it near the shelf station and the backstock dock. Rather than hard-code
world coordinates (which drift when the store is regenerated), this derives the
paths from the mission poses the runtime already computes
(``twin.restock.scenario.poses_for_job``): ``pickup`` (backstock), ``cart_dock``,
``shelf_approach`` (shelf station) and ``home``.

Pure: takes the poses mapping, returns :class:`PersonSpec` tuples. Tests assert the
generated paths really do cross the outbound route and come close to the two
stations, so the "approaches the robot" requirement is checked, not asserted.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

from .config import PersonSpec
from .paths import CYCLE, PINGPONG, Point


def _xy(pose: Sequence[float]) -> Point:
    if len(pose) < 2 or not all(math.isfinite(v) for v in pose[:2]):
        raise ValueError('pose must carry a finite (x, y)')
    return (float(pose[0]), float(pose[1]))


def _unit(dx: float, dy: float) -> Tuple[float, float]:
    n = math.hypot(dx, dy)
    if n == 0:
        return (1.0, 0.0)
    return (dx / n, dy / n)


def _crossing(a: Point, b: Point, *, half_span: float, offset: float) -> Tuple[Point, Point]:
    """A segment perpendicular to AB through the point ``offset`` of the way from
    A to B, extending ``half_span`` either side. Walking it crosses route AB."""
    mx, my = a[0] + (b[0] - a[0]) * offset, a[1] + (b[1] - a[1]) * offset
    ux, uy = _unit(b[0] - a[0], b[1] - a[1])
    px, py = -uy, ux                      # left normal
    return ((mx - px * half_span, my - py * half_span),
            (mx + px * half_span, my + py * half_span))


def default_people(poses: Dict[str, Sequence[float]], *,
                   shopper_speed: float = 1.1, staff_speed: float = 0.8) -> Tuple[PersonSpec, ...]:
    """Three scripted people derived from the mission poses.

    * ``shopper_aisle`` -- crosses the outbound route (pickup -> shelf) near its
      middle, repeatedly (ping-pong): intrudes on the robot's travel lane.
    * ``staff_shelf`` -- paces a short beat right next to the shelf station, so it
      enters the arm's reach envelope while the robot is transferring.
    * ``shopper_dock`` -- loops past the backstock dock (pickup/cart_dock), so a
      person is present where the robot collects and couples.
    """
    pickup, cart_dock = _xy(poses['pickup']), _xy(poses['cart_dock'])
    shelf, home = _xy(poses['shelf_approach']), _xy(poses['home'])

    # 1) Aisle shopper crossing the robot's outbound lane near its midpoint.
    cross_a, cross_b = _crossing(pickup, shelf, half_span=2.5, offset=0.5)
    shopper_aisle = PersonSpec(
        person_id='shopper_aisle', role='shopper',
        waypoints=(cross_a, cross_b), speed_mps=shopper_speed, mode=PINGPONG,
        dwell_s=1.0, start_offset_s=0.0)

    # 2) Staff pacing beside the shelf station, stepping toward it (into reach).
    ux, uy = _unit(shelf[0] - home[0], shelf[1] - home[1])  # toward shelf from home side
    near_shelf = (shelf[0] - ux * 1.0, shelf[1] - uy * 1.0)
    beside_shelf = (shelf[0] - uy * 2.0, shelf[1] + ux * 2.0)
    staff_shelf = PersonSpec(
        person_id='staff_shelf', role='staff',
        waypoints=(beside_shelf, near_shelf), speed_mps=staff_speed, mode=PINGPONG,
        dwell_s=1.5, start_offset_s=2.0)

    # 3) Shopper looping past the backstock dock / coupling area.
    dock_cross_a, dock_cross_b = _crossing(pickup, cart_dock, half_span=2.0, offset=0.5)
    shopper_dock = PersonSpec(
        person_id='shopper_dock', role='shopper',
        waypoints=(dock_cross_a, pickup, dock_cross_b), speed_mps=shopper_speed,
        mode=CYCLE, dwell_s=0.5, start_offset_s=1.0)

    return (shopper_aisle, staff_shelf, shopper_dock)
