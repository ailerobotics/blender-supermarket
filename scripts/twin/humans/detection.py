"""Detection bridge: feed the vendored fleet safety machine from the simulation.

Two providers, selected EXPLICITLY by ``DetectionConfig.source`` (never silently):

``detector``  -- the real fleet detector runs on the mission camera topics
    (``/restock_sim/camera/{rgb,depth,camera_info}``) and publishes its
    ``/human_safety/state`` JSON exactly as on the rig; :class:`ReceivedSafety`
    deserialises the latest message and applies the same staleness rule the gate
    uses. This is the production port; running the neural detector here needs the
    fleet package + model + ROS at runtime (and competes with Isaac for the one
    GPU), so it is wired at GPU time, not on the CPU path.

``ground_truth`` -- :class:`GroundTruthSafety` projects the KNOWN scripted-people
    poses into the configured cameras (cone visibility), lifts each visible person
    to the robot base frame, and runs the fleet :class:`SafetyEvaluator` on those
    tracks. The zone/hysteresis/dwell/health judgement is the fleet's, unchanged;
    only the *source* of the person positions is ground truth. Reported as
    ground-truth evidence, not a detection measurement.

Pure Python + numpy; no ROS or USD import here, so both providers are unit-tested
offline. The camera/base world poses are supplied by the caller (the GPU runtime
reads them from the stage; tests pass them directly).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import DetectionConfig
from .perception import (CameraHealth, DistanceStatus, SafetyEvaluator, SafetyParams,
                         SafetyState, Track, Zone)

Pose2D = Tuple[float, float, float]            # (x, y, yaw) in the map frame


@dataclass(frozen=True)
class Detection:
    """A detected dynamic obstacle in the shared navigation contract.

    The first three fields match the dynamic-obstacle navigation lane's landed
    ``Detection(position, radius, source)`` (``scripts/twin/navigation/dynamic.py``,
    ``sim-restock-dynobs-v1``) EXACTLY -- same names, order and semantics -- so a
    human detection is consumed as a dynamic obstacle by the navigation obstacle
    layer and as an event by the behaviour-tree lane (``sim-restock-bt-v1``) without
    a second detector. ``position`` is the ``(x, y)`` plan-view centroid (a 2-tuple;
    the nav layer validates ``len == 2``) in the frame named by ``frame`` (the map
    frame for the published obstacle contract), ``radius`` the person's footprint
    radius in metres, and ``source`` the detection provenance (``'human_safety'``
    for the real detector, ``'human_ground_truth'`` for the explicit GT fallback).
    ``z`` (body-centre height), ``person_id`` and ``distance_base_m`` are optional
    extras, ignored by consumers that only read ``position``/``radius``/``source``.
    """

    position: Tuple[float, float]
    radius: float
    source: str
    frame: str = 'map'
    z: float = 0.0
    person_id: str = ''
    distance_base_m: float = float('inf')


def base_frame_point(world_xyz: Sequence[float], base_pose: Pose2D) -> np.ndarray:
    """Transform a world point into the robot base frame (2D yaw + translation)."""
    x, y, yaw = base_pose
    dx, dy = world_xyz[0] - x, world_xyz[1] - y
    c, s = math.cos(-yaw), math.sin(-yaw)
    z = world_xyz[2] if len(world_xyz) > 2 else 0.0
    return np.array([c * dx - s * dy, s * dx + c * dy, z], dtype=float)


def camera_sees(person_world: Sequence[float], camera_pose: Pose2D,
                max_range: float, fov_halfangle: float) -> bool:
    """Cone visibility: person within ``max_range`` and the camera's yaw cone."""
    cx, cy, cyaw = camera_pose
    dx, dy = person_world[0] - cx, person_world[1] - cy
    rng = math.hypot(dx, dy)
    if rng > max_range or rng == 0.0:
        return rng == 0.0                       # standing on the lens counts as seen
    bearing = math.atan2(dy, dx)
    diff = (bearing - cyaw + math.pi) % (2 * math.pi) - math.pi
    return abs(diff) <= fov_halfangle


def nearest_point_toward_base(position_base: np.ndarray, radius: float) -> np.ndarray:
    """A body point ``radius`` nearer the base than the centre (never past it)."""
    px, py, pz = position_base
    horiz = math.hypot(px, py)
    if horiz <= 1e-6:
        return position_base.copy()
    f = max(0.0, (horiz - radius)) / horiz
    return np.array([px * f, py * f, pz], dtype=float)


class _IdRegistry:
    """Stable integer track ids per person string id (the evaluator keys on int)."""

    def __init__(self):
        self._ids: Dict[str, int] = {}

    def get(self, person_id: str) -> int:
        if person_id not in self._ids:
            self._ids[person_id] = len(self._ids) + 1
        return self._ids[person_id]


@dataclass
class GroundTruthSafety:
    """Explicit ground-truth provider; runs the real safety machine on known poses.

    ``people(now)`` -> ``[(person_id, (x,y,z) world, radius_m)]``
    ``base_pose()`` -> ``(x, y, yaw)`` robot base in the map frame
    ``camera_poses()`` -> ``{frame: (x, y, yaw)}`` configured cameras in the map frame
    """

    config: DetectionConfig
    people: Callable[[float], List[Tuple[str, Sequence[float], float]]]
    base_pose: Callable[[], Pose2D]
    camera_poses: Callable[[], Dict[str, Pose2D]]
    params: SafetyParams = None

    def __post_init__(self):
        if self.config.source != 'ground_truth':
            raise ValueError('GroundTruthSafety requires detection source=ground_truth')
        self.evaluator = SafetyEvaluator(self.params or SafetyParams())
        self._ids = _IdRegistry()
        self.provenance = 'ground_truth'

    def _visible(self, now: float):
        """``[(person_id, world_xyz, radius, seen_by)]`` for people in view now."""
        cams = self.camera_poses()
        out = []
        for person_id, world_xyz, radius in self.people(now):
            seen_by = [f for f in self.config.cameras
                       if f in cams and camera_sees(
                           world_xyz, cams[f], self.config.gt_visible_range_m,
                           self.config.gt_fov_halfangle_rad)]
            if seen_by:
                out.append((person_id, tuple(map(float, world_xyz)), float(radius), seen_by))
        return out

    def detections(self, now: float) -> List[Detection]:
        """Shared ``Detection(position, radius, source)`` list in the map frame."""
        base = self.base_pose()
        out = []
        for person_id, world_xyz, radius, _seen in self._visible(now):
            pos = base_frame_point(world_xyz, base)
            out.append(Detection(
                position=(float(world_xyz[0]), float(world_xyz[1])), radius=radius,
                source='human_ground_truth', frame='map',
                z=float(world_xyz[2] if len(world_xyz) > 2 else 0.0),
                person_id=person_id,
                distance_base_m=float(math.hypot(pos[0], pos[1]))))
        return out

    def state(self, now: float) -> SafetyState:
        base = self.base_pose()
        tracks: List[Track] = []
        any_presence = False
        for person_id, world_xyz, radius, seen_by in self._visible(now):
            any_presence = True
            pos = base_frame_point(world_xyz, base)
            tracks.append(Track(
                id=self._ids.get(person_id), position=pos,
                velocity=np.zeros(3), covariance=np.eye(6) * 1e-3, stamp=now,
                hits=5, age=5, time_since_update=0.0, confirmed=True,
                nearest_point=nearest_point_toward_base(pos, radius),
                cameras=tuple(seen_by)))
        health = [CameraHealth(camera=f, present_in_graph=True, frame_age=0.0,
                               detector_alive=True, positioned=True, fresh=True)
                  for f in self.config.cameras]
        return self.evaluator.evaluate(tracks, health, now, any_presence=any_presence,
                                       source_stamp=now, latency=0.0)


@dataclass
class ReceivedSafety:
    """Holds the latest ``SafetyState`` published by the real detector node.

    The node subscribes to ``config.state_topic`` and hands each decoded state to
    :meth:`on_state`. ``state(now)`` returns it, or a fail-safe STOP state when no
    fresh message has arrived within ``state_timeout_s`` -- absence of evidence is
    never clearance.
    """

    config: DetectionConfig
    state_timeout_s: float = 0.7

    def __post_init__(self):
        if self.config.source != 'detector':
            raise ValueError('ReceivedSafety requires detection source=detector')
        self._latest: Optional[SafetyState] = None
        self._arrival: Optional[float] = None
        self.provenance = 'detector'

    def on_state(self, state: SafetyState, now: float) -> None:
        self._latest, self._arrival = state, now

    def detections(self, now: float) -> List[Detection]:
        """Shared ``Detection`` list from the detector's tracks (base frame).

        The real detector publishes tracks in the robot base frame, so these
        positions are ``base_link``-relative (``frame='base_link'``); the obstacle
        layer reprojects to the map frame with TF before unioning them. Empty when
        the last state is missing or stale (fail toward no-clearance: the stale case
        is handled as STOP by :meth:`state`).
        """
        if (self._latest is None or self._arrival is None
                or now - self._arrival > self.state_timeout_s):
            return []
        out = []
        for t in getattr(self._latest, 'tracks', []) or []:
            if not getattr(t, 'confirmed', False) or t.position is None:
                continue
            p = np.asarray(t.position, dtype=float)
            out.append(Detection(
                position=(float(p[0]), float(p[1])), radius=0.25,
                source='human_safety', frame='base_link',
                z=float(p[2] if len(p) > 2 else 0.0), person_id=str(t.id),
                distance_base_m=float(math.hypot(p[0], p[1]))))
        return out

    def state(self, now: float) -> SafetyState:
        if (self._latest is None or self._arrival is None
                or now - self._arrival > self.state_timeout_s):
            reason = ('no detector state received yet' if self._latest is None
                      else f'detector state stale ({now - self._arrival:.2f}s) -> STOP')
            return SafetyState(stamp=now, zone=Zone.STOP, nearest_distance=float('inf'),
                               nearest_distance_base=float('inf'),
                               distance_status=DistanceStatus.NONE, nearest_track_id=None,
                               any_person_visible=False, reasons=[reason])
        return self._latest


def to_nav_detections(dets: Sequence['Detection']):
    """Adapt human :class:`Detection` objects to the nav layer's own ``Detection``.

    The dynamic-obstacle layer (``twin.navigation.dynamic.DynamicObstacleLayer``)
    accepts its OWN ``Detection(position, radius, source)`` by ``isinstance``, so a
    consumer that wants to fold human detections into the costmap in-process (rather
    than over the ``/humans/detections`` topic) passes them through here. Only
    ``frame == 'map'`` detections are converted -- the layer is map-frame; a
    base-frame detector detection must be reprojected (TF) by the caller first. The
    import is deferred so this pure module stays importable without the nav package.
    """
    from twin.navigation.dynamic import Detection as NavDetection
    out = []
    for d in dets:
        if d.frame != 'map':
            raise ValueError(
                f'to_nav_detections needs map-frame detections, got {d.frame!r}; '
                'reproject base-frame detector detections with TF first')
        out.append(NavDetection(position=tuple(d.position), radius=d.radius,
                                source=d.source))
    return out


def make_provider(config: DetectionConfig, **kwargs):
    """Build the provider named by ``config.source``. Never picks a fallback for you."""
    if config.source == 'ground_truth':
        return GroundTruthSafety(config=config, **kwargs)
    if config.source == 'detector':
        return ReceivedSafety(config=config,
                              state_timeout_s=kwargs.get('state_timeout_s', 0.7))
    raise ValueError(f'unknown detection source {config.source!r}')
