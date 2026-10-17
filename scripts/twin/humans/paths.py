"""Pure, deterministic scripted paths for simulated people.

A :class:`ScriptedPath` walks a person along a piecewise-linear poly-line of
world waypoints at a constant ground speed, optionally dwelling at each vertex.
``pose_at(t)`` is a deterministic function of the elapsed simulation time, so the
whole crowd replays bit-for-bit across runs and every rule is unit-tested with no
simulator, no ROS and no GPU. The Isaac spawner (:mod:`twin.humans.isaac_people`)
teleports each character to ``pose_at(sim_time)`` every physics step; nothing here
imports USD or Isaac.

Coordinates are metres in the mission ``map`` frame (Z up); ``pose_at`` returns
``(x, y, yaw)`` with ``yaw`` the heading along the current segment (held through a
dwell and at the path ends).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point = Tuple[float, float]
Pose = Tuple[float, float, float]

# Loop modes.
PINGPONG = 'pingpong'   # walk to the last waypoint, reverse, repeat (aisle traffic)
CYCLE = 'cycle'         # after the last waypoint return straight to the first, repeat
ONCE = 'once'           # walk once and hold at the last waypoint forever
MODES = (PINGPONG, CYCLE, ONCE)


def _finite(v: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True)
class ScriptedPath:
    """A constant-speed walk along ``waypoints`` with optional per-vertex dwell."""

    waypoints: Tuple[Point, ...]
    speed_mps: float
    mode: str = PINGPONG
    dwell_s: float = 0.0
    start_offset_s: float = 0.0

    def __post_init__(self):
        if len(self.waypoints) < 2:
            raise ValueError('a scripted path needs at least two waypoints')
        for p in self.waypoints:
            if len(p) != 2 or not all(_finite(v) for v in p):
                raise ValueError('each waypoint must be a finite (x, y)')
        if not _finite(self.speed_mps) or self.speed_mps <= 0:
            raise ValueError('speed_mps must be finite and positive')
        if not _finite(self.dwell_s) or self.dwell_s < 0:
            raise ValueError('dwell_s must be finite and nonnegative')
        if not _finite(self.start_offset_s) or self.start_offset_s < 0:
            raise ValueError('start_offset_s must be finite and nonnegative')
        if self.mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        # A degenerate path (every waypoint identical) has no heading and no motion.
        if self._total_length() <= 0:
            raise ValueError('path has zero length; waypoints must differ')

    # -- geometry ------------------------------------------------------------
    def _legs(self) -> List[Tuple[Point, Point, float]]:
        """The ordered (start, end, length) segments one full traversal visits.

        ``pingpong`` appends the reversed interior; ``cycle`` appends the closing
        leg back to the first waypoint; ``once`` is just the forward legs.
        """
        wp = list(self.waypoints)
        pts = list(wp)
        if self.mode == PINGPONG:
            pts = wp + wp[-2::-1]            # forward then back (no repeated vertex)
        elif self.mode == CYCLE:
            pts = wp + [wp[0]]               # close the loop
        legs = []
        for a, b in zip(pts, pts[1:]):
            legs.append((a, b, math.hypot(b[0] - a[0], b[1] - a[1])))
        return legs

    def _total_length(self) -> float:
        wp = self.waypoints
        fwd = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(wp, wp[1:]))
        if self.mode == PINGPONG:
            return 2 * fwd
        if self.mode == CYCLE:
            close = math.hypot(wp[0][0] - wp[-1][0], wp[0][1] - wp[-1][1])
            return fwd + close
        return fwd

    def _period(self) -> float:
        """Seconds for one full traversal including dwell at every interior vertex.

        A dwell is applied at the arrival of each leg; for a closed/ping-pong cycle
        that is one dwell per leg, so motion + dwell tile the time axis exactly.
        """
        legs = self._legs()
        travel = sum(length for _, _, length in legs) / self.speed_mps
        return travel + self.dwell_s * len(legs)

    @staticmethod
    def _heading(a: Point, b: Point) -> float:
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def pose_at(self, t: float) -> Pose:
        """World ``(x, y, yaw)`` at elapsed simulation time ``t`` (seconds)."""
        if not _finite(t):
            raise ValueError('time must be finite')
        legs = self._legs()
        first_heading = self._heading(*legs[0][:2])
        te = t - self.start_offset_s
        if te <= 0:                           # not yet started: wait at the first waypoint
            x, y = self.waypoints[0]
            return (x, y, first_heading)
        if self.mode == ONCE:
            period = self._period()
            if te >= period:
                x, y = legs[-1][1]
                return (x, y, self._heading(*legs[-1][:2]))
        else:
            te = math.fmod(te, self._period())
        # Walk the legs, consuming travel-then-dwell on each.
        last_heading = first_heading
        for a, b, length in legs:
            heading = self._heading(a, b) if length > 0 else last_heading
            leg_travel = length / self.speed_mps
            if te <= leg_travel:
                f = 0.0 if leg_travel == 0 else te / leg_travel
                return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, heading)
            te -= leg_travel
            if te <= self.dwell_s:            # dwelling at b, facing the way we arrived
                return (b[0], b[1], heading)
            te -= self.dwell_s
            last_heading = heading
        # Floating-point spillover at the exact period boundary: hold the final vertex.
        x, y = legs[-1][1]
        return (x, y, last_heading)

    def sample(self, duration_s: float, dt: float) -> List[Tuple[float, Pose]]:
        """``[(t, (x, y, yaw))]`` over ``[0, duration_s]`` at step ``dt`` (helper)."""
        if not _finite(duration_s) or duration_s < 0 or not _finite(dt) or dt <= 0:
            raise ValueError('duration_s >= 0 and dt > 0 required')
        n = int(round(duration_s / dt))
        return [(i * dt, self.pose_at(i * dt)) for i in range(n + 1)]


def min_distance_to_point(path: ScriptedPath, point: Point, duration_s: float,
                          dt: float = 0.1) -> float:
    """Closest the path's position comes to ``point`` over ``[0, duration_s]``.

    A coarse time-grid minimum used by tests/planning to assert a person actually
    approaches a station; not a continuous optimum.
    """
    best = math.inf
    for _t, (x, y, _yaw) in path.sample(duration_s, dt):
        best = min(best, math.hypot(x - point[0], y - point[1]))
    return best


def min_distance_between(a: ScriptedPath, b: ScriptedPath, duration_s: float,
                         dt: float = 0.1) -> float:
    """Closest two scripted people come to each other over ``[0, duration_s]``."""
    if not _finite(duration_s) or duration_s < 0 or not _finite(dt) or dt <= 0:
        raise ValueError('duration_s >= 0 and dt > 0 required')
    n = int(round(duration_s / dt))
    best = math.inf
    for i in range(n + 1):
        t = i * dt
        xa, ya, _ = a.pose_at(t)
        xb, yb, _ = b.pose_at(t)
        best = min(best, math.hypot(xa - xb, ya - yb))
    return best


def crosses_segment(path: ScriptedPath, seg: Tuple[Point, Point], duration_s: float,
                    clearance_m: float, dt: float = 0.1) -> bool:
    """True if the path's position comes within ``clearance_m`` of line ``seg``.

    Used to assert a scripted person actually intrudes on the robot's planned
    route (approximated by a straight segment between two mission poses).
    """
    (x0, y0), (x1, y1) = seg
    sx, sy = x1 - x0, y1 - y0
    seg_len2 = sx * sx + sy * sy
    n = int(round(duration_s / dt))
    for i in range(n + 1):
        px, py, _ = path.pose_at(i * dt)
        if seg_len2 == 0:
            d = math.hypot(px - x0, py - y0)
        else:
            u = max(0.0, min(1.0, ((px - x0) * sx + (py - y0) * sy) / seg_len2))
            d = math.hypot(px - (x0 + u * sx), py - (y0 + u * sy))
        if d <= clearance_m:
            return True
    return False
