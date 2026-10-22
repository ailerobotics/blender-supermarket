"""Moving-obstacle layer: track detections and predict their swept path.

This layer turns a stream of per-frame obstacle DETECTIONS (map-frame centroids
with a physical radius -- from the robot depth/navigation cameras through an
nvblox dynamic/people layer, a detector node, or the deterministic scenario) into
extra conservative occupancy points that are UNIONED into `Observation.obstacles`.
Every existing gate (reference, articulated, Nav2) already stops for and plans
around `Observation.obstacles` using the full towed-cart swept envelope, so a
moving obstacle is avoided with the cart footprint for free once its predicted
cells are in that tuple.

Three invariants keep this safe:

* It only ADDS occupancy. A detection never clears a raw sensor return, never
  shortens a stopping distance, and never relabels an unknown cell free. Removing
  this layer's output can only turn a stop into motion, never the reverse.
* A detection's CURRENT position is always emitted the moment the track is
  confirmed; velocity prediction only extends the blocked region FORWARD along the
  track's motion, so a missed velocity estimate degrades to the static case.
* It is pure and deterministic given its inputs and configuration: the same
  detection stream yields byte-identical emitted points, so an evidence run
  replays exactly. When no layer is attached the observation is untouched.

This is a simulation routing aid, not a certified dynamic-safety function.
"""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Detection:
    """One map-frame obstacle observation for a single frame.

    `position` is the (x, y) centroid in the map frame; `radius` is the obstacle's
    physical plan-view radius (its own extent, NOT the robot clearance, which the
    gate adds separately). `source` records the detector/layer of origin for the
    evidence log; it never relaxes a check.
    """
    position: tuple
    radius: float
    source: str = 'dynamic_detection'

    def __post_init__(self):
        if (len(self.position) != 2 or not all(math.isfinite(v) for v in self.position)
                or not math.isfinite(self.radius) or self.radius < 0
                or not isinstance(self.source, str) or not self.source):
            raise ValueError('invalid dynamic detection')


@dataclass(frozen=True)
class TrackedObstacle:
    """A confirmed track: smoothed position, velocity and extent in the map frame."""
    track_id: int
    position: tuple
    velocity: tuple
    radius: float
    stamp: float
    hits: int
    source: str

    @property
    def speed(self):
        return math.hypot(self.velocity[0], self.velocity[1])


@dataclass(frozen=True)
class DynamicObstacleConfig:
    """Detection-to-costmap thresholds and behaviours, all configuration.

    * `association_distance`: max map-frame gap (m) to tie a detection to an
      existing track's predicted position between frames.
    * `moving_speed`: estimated speed (m/s) at or above which a track is treated as
      MOVING and its forward path predicted; below it only the current extent is
      emitted (a near-stationary obstacle is still fully blocked).
    * `prediction_horizon`: how far ahead (s) a moving track's path is blocked.
    * `prediction_step`: time step (s) sampling that predicted path.
    * `velocity_gain`: 0..1 low-pass blend of the new finite-difference velocity
      into the retained estimate (1 = no smoothing).
    * `confirm_frames`: consecutive associated frames before a track is emitted
      (debounces one-frame detector noise into the costmap).
    * `drop_after`: a track older than this (s, no association) is forgotten.
    * `point_spacing`: emitted-point spacing (m); keep <= the planner/costmap
      resolution so the blocked region has no gaps.
    * `max_tracks`: cap on simultaneously retained tracks (oldest dropped first).
    """
    association_distance: float = 0.6
    moving_speed: float = 0.10
    prediction_horizon: float = 1.5
    prediction_step: float = 0.3
    velocity_gain: float = 0.5
    confirm_frames: int = 2
    drop_after: float = 0.5
    point_spacing: float = 0.20
    max_tracks: int = 32

    def __post_init__(self):
        values = (self.association_distance, self.moving_speed, self.prediction_horizon,
                  self.prediction_step, self.velocity_gain, self.drop_after, self.point_spacing)
        if (not all(math.isfinite(v) for v in values)
                or self.association_distance <= 0 or self.moving_speed < 0
                or self.prediction_horizon < 0 or self.prediction_step <= 0
                or not 0 < self.velocity_gain <= 1 or self.drop_after <= 0
                or self.point_spacing <= 0
                or type(self.confirm_frames) is not int or self.confirm_frames < 1
                or type(self.max_tracks) is not int or self.max_tracks < 1):
            raise ValueError('invalid dynamic-obstacle configuration')


def _disk_points(cx, cy, radius, spacing):
    """Map-frame points covering a disk of `radius`, spaced <= `spacing`.

    The gate treats every obstacle point as zero-size and adds the robot clearance
    separately, so the obstacle's own extent must be represented by a filled set of
    points no farther apart than the costmap resolution.
    """
    if radius <= 0:
        return [(cx, cy)]
    points = [(cx, cy)]
    rings = max(1, math.ceil(radius / spacing))
    for ri in range(1, rings + 1):
        r = radius * ri / rings
        count = max(4, math.ceil(2 * math.pi * r / spacing))
        for k in range(count):
            a = 2 * math.pi * k / count
            points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return points


def predicted_points(obstacle, config):
    """Conservative occupancy points for one tracked obstacle.

    Always covers the current extent; a moving track additionally blocks its
    predicted swept path over the configured horizon. Returns a list of (x, y).
    """
    cx, cy = obstacle.position
    r = obstacle.radius
    points = _disk_points(cx, cy, r, config.point_spacing)
    if obstacle.speed >= config.moving_speed and config.prediction_horizon > 0:
        vx, vy = obstacle.velocity
        steps = max(1, math.ceil(config.prediction_horizon / config.prediction_step))
        for s in range(1, steps + 1):
            t = config.prediction_horizon * s / steps
            points.extend(_disk_points(cx + vx * t, cy + vy * t, r, config.point_spacing))
    return points


class DynamicObstacleLayer:
    """Stateful tracker + predictor: detections in, costmap points out.

    Call `update(detections, stamp)` once per navigation frame with the current
    map-frame detections, then `apply(observation)` to fold the predicted points
    into `Observation.obstacles` before gating. The two steps are split so the
    emitted points and the active tracks can be logged for evidence.
    """

    def __init__(self, config=None):
        self.config = config or DynamicObstacleConfig()
        if not isinstance(self.config, DynamicObstacleConfig):
            raise ValueError('DynamicObstacleLayer requires a DynamicObstacleConfig')
        self._tracks = {}  # track_id -> TrackedObstacle
        self._next_id = 0
        self._last_stamp = None

    @property
    def tracks(self):
        """Confirmed tracks after the most recent update (deterministic order)."""
        return tuple(t for _, t in sorted(self._tracks.items())
                     if t.hits >= self.config.confirm_frames)

    def update(self, detections, stamp):
        """Associate this frame's detections to tracks and refresh velocities.

        Detections are matched greedily to the nearest existing track's PREDICTED
        position within `association_distance`; matched tracks update position and a
        low-passed velocity, unmatched detections seed new tracks, and tracks not
        seen within `drop_after` are forgotten. Returns the confirmed tracks.
        """
        if not math.isfinite(stamp):
            raise ValueError('dynamic update stamp must be finite')
        if self._last_stamp is not None and stamp < self._last_stamp:
            raise ValueError('dynamic update stamp must not move backward')
        dets = tuple(detections)
        for d in dets:
            if not isinstance(d, Detection):
                raise ValueError('update requires Detection objects')
        dt = 0.0 if self._last_stamp is None else stamp - self._last_stamp
        cfg = self.config

        # Predicted current position of each existing track (deterministic order).
        existing = sorted(self._tracks.items())
        predicted = {tid: (t.position[0] + t.velocity[0] * dt,
                           t.position[1] + t.velocity[1] * dt)
                     for tid, t in existing}
        # Greedy nearest-neighbour association over all (detection, track) pairs.
        pairs = []
        for di, d in enumerate(dets):
            for tid, _ in existing:
                px, py = predicted[tid]
                dist = math.hypot(d.position[0] - px, d.position[1] - py)
                if dist <= cfg.association_distance:
                    pairs.append((dist, di, tid))
        pairs.sort()
        matched_det, matched_track, assignment = set(), set(), {}
        for _, di, tid in pairs:
            if di in matched_det or tid in matched_track:
                continue
            matched_det.add(di)
            matched_track.add(tid)
            assignment[tid] = di

        updated = {}
        for tid, t in existing:
            if tid in assignment:
                d = dets[assignment[tid]]
                if dt > 1e-6:
                    raw = ((d.position[0] - t.position[0]) / dt,
                           (d.position[1] - t.position[1]) / dt)
                    g = cfg.velocity_gain
                    vel = (t.velocity[0] * (1 - g) + raw[0] * g,
                           t.velocity[1] * (1 - g) + raw[1] * g)
                else:
                    vel = t.velocity
                updated[tid] = TrackedObstacle(tid, d.position, vel, d.radius,
                                               stamp, t.hits + 1, d.source)
            elif stamp - t.stamp <= cfg.drop_after:
                updated[tid] = t  # keep briefly; no re-detection yet (not aged out)

        for di, d in enumerate(dets):
            if di not in matched_det:
                updated[self._next_id] = TrackedObstacle(
                    self._next_id, d.position, (0.0, 0.0), d.radius, stamp, 1, d.source)
                self._next_id += 1

        # Cap retained tracks: drop the oldest (smallest stamp, then id) first.
        if len(updated) > cfg.max_tracks:
            keep = sorted(updated.values(), key=lambda t: (t.stamp, t.track_id),
                          reverse=True)[:cfg.max_tracks]
            updated = {t.track_id: t for t in keep}

        self._tracks = updated
        self._last_stamp = stamp
        return self.tracks

    def emitted_points(self):
        """All costmap points for the currently confirmed tracks (deterministic)."""
        points = []
        for t in self.tracks:
            points.extend(predicted_points(t, self.config))
        return tuple(points)

    def apply(self, observation):
        """Union the emitted points into `observation.obstacles` (never removes)."""
        extra = self.emitted_points()
        if not extra:
            return observation
        return replace(observation, obstacles=tuple(observation.obstacles) + extra)
