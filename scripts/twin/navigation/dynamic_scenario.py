"""Deterministic scripted dynamic obstacles for the dynamic-obstacle evidence run.

These are scenario ground truth, not a detector: `scenario_detections` turns the
scripted actor poses at a sim time into `Detection`s exactly as a perception layer
(nvblox dynamic/people, a detector node) would publish them, so the navigation
dynamic-obstacle layer can be exercised end to end on CPU with no Blender, no
Isaac and no GPU. The actor poses are fully determined by the (time, x, y)
keyframes, so an evidence run replays byte-for-byte.

Three canned actors cover the briefed cases, using a moving box / cart proxy
until the people assets from `sim-restock-humans-v1` land (then share them):

* `crossing_person` -- a shopper-sized disk that walks across the aisle, pausing
  in the lane, then leaves (detected only while present).
* `rolled_cart` -- a cart-sized disk rolled in from the side that comes to rest
  blocking part of the aisle and stays.
* `dropped_box` -- a box that appears on the route at a drop time and stays put.

Nothing here commands the robot or asserts safety; it only produces detections.
"""
from dataclasses import dataclass
import math

from .dynamic import Detection


@dataclass(frozen=True)
class Keyframe:
    t: float
    x: float
    y: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.t, self.x, self.y)):
            raise ValueError('keyframe values must be finite')


@dataclass(frozen=True)
class ScriptedActor:
    """A disk obstacle following piecewise-linear keyframes over sim time.

    `keyframes` must be time-sorted and non-empty. The actor is DETECTED only
    within [appear, disappear]; outside that window it contributes nothing (a box
    that has not been dropped, or a shopper who has left the camera's view). Before
    the first / after the last keyframe the position holds (clamped).
    """
    name: str
    radius: float
    keyframes: tuple
    appear: float = -math.inf
    disappear: float = math.inf
    source: str = 'scenario_ground_truth'

    def __post_init__(self):
        kfs = tuple(self.keyframes)
        if (not isinstance(self.name, str) or not self.name
                or not math.isfinite(self.radius) or self.radius <= 0
                or not kfs or any(not isinstance(k, Keyframe) for k in kfs)
                or list(kfs) != sorted(kfs, key=lambda k: k.t)
                or any(a.t == b.t for a, b in zip(kfs, kfs[1:]))
                or self.appear > self.disappear
                or not isinstance(self.source, str) or not self.source):
            raise ValueError('invalid scripted actor')
        object.__setattr__(self, 'keyframes', kfs)

    def active_at(self, t):
        return self.appear <= t <= self.disappear

    def position_at(self, t):
        kfs = self.keyframes
        if t <= kfs[0].t:
            return (kfs[0].x, kfs[0].y)
        if t >= kfs[-1].t:
            return (kfs[-1].x, kfs[-1].y)
        for a, b in zip(kfs, kfs[1:]):
            if a.t <= t <= b.t:
                s = (t - a.t) / (b.t - a.t)
                return (a.x + (b.x - a.x) * s, a.y + (b.y - a.y) * s)
        raise AssertionError('unreachable')  # pragma: no cover

    def detection_at(self, t):
        if not self.active_at(t):
            return None
        return Detection(self.position_at(t), self.radius, source=self.source)


@dataclass(frozen=True)
class DynamicScenario:
    name: str
    description: str
    actors: tuple

    def __post_init__(self):
        actors = tuple(self.actors)
        if (not isinstance(self.name, str) or not self.name
                or not isinstance(self.description, str) or not self.description
                or not actors or any(not isinstance(a, ScriptedActor) for a in actors)
                or len({a.name for a in actors}) != len(actors)):
            raise ValueError('invalid dynamic scenario')
        object.__setattr__(self, 'actors', actors)

    def detections_at(self, t):
        return tuple(d for d in (a.detection_at(t) for a in self.actors) if d is not None)


def scenario_detections(actors, t):
    """Detections for all active scripted actors at sim time `t` (deterministic)."""
    return tuple(d for d in (a.detection_at(t) for a in actors) if d is not None)


# -- canned actors ---------------------------------------------------------------
# Defaults assume an aisle whose centre line runs along +x at y ~= 0; the caller
# overrides the geometry to match the scene. Speeds/times are in sim seconds.

def crossing_person(*, lane_x=4.0, from_y=2.2, to_y=-2.2, radius=0.35,
                    enter=2.0, cross_duration=6.0, dwell=1.5, name='crossing_person'):
    """A shopper walking across the aisle at `lane_x`, pausing mid-lane, then gone."""
    mid = enter + cross_duration / 2
    return ScriptedActor(name, radius, (
        Keyframe(enter, lane_x, from_y),
        Keyframe(mid, lane_x, 0.0),
        Keyframe(mid + dwell, lane_x, 0.0),
        Keyframe(mid + dwell + cross_duration / 2, lane_x, to_y)),
        appear=enter, disappear=mid + dwell + cross_duration / 2)


def rolled_cart(*, lane_x=4.0, from_y=2.4, rest_y=0.6, radius=0.5,
                enter=1.0, roll_duration=3.0, name='rolled_cart'):
    """A cart rolled in from the side that stops blocking part of the aisle and stays."""
    return ScriptedActor(name, radius, (
        Keyframe(enter, lane_x, from_y),
        Keyframe(enter + roll_duration, lane_x, rest_y)),
        appear=enter)


def dropped_box(*, at_x=4.0, at_y=0.0, radius=0.3, drop=3.0, name='dropped_box'):
    """A box that appears on the route at `drop` and stays put."""
    return ScriptedActor(name, radius, (Keyframe(drop, at_x, at_y),), appear=drop)
