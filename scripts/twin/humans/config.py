"""Opt-in configuration for the simulated-people + human-safety layer.

Parsed from the ``humans`` block of the mission runtime config (the JSON passed to
``run_mobile_restock.py --runtime-config``). When the block is absent,
:func:`HumansConfig.from_config` returns a disabled config and every integration
hook is inert, so the mission is byte-identical to a run without this package.

Nothing here imports USD, ROS or Isaac; it is a plain, validated data model so the
whole configuration surface is exercised in offline tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

from .paths import MODES, PINGPONG, Point, ScriptedPath

ROLES = ('shopper', 'staff')


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True)
class PersonSpec:
    """One scripted simulated person: identity, role, path and body size."""

    person_id: str
    role: str
    waypoints: Tuple[Point, ...]
    speed_mps: float
    mode: str = PINGPONG
    dwell_s: float = 0.0
    start_offset_s: float = 0.0
    height_m: float = 1.75
    radius_m: float = 0.25
    # Optional Isaac People character asset reference (USD). None -> the spawner
    # falls back to a parametric capsule stand-in (no network, deterministic).
    character_usd: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.person_id, str) or not self.person_id:
            raise ValueError('person_id must be a non-empty string')
        if self.role not in ROLES:
            raise ValueError(f'role must be one of {ROLES}')
        if self.mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        for name, lo in (('height_m', 0.5), ('radius_m', 0.05)):
            v = getattr(self, name)
            if not _finite(v) or v < lo:
                raise ValueError(f'{name} must be finite and >= {lo}')
        # Reuse ScriptedPath's own validation for waypoints/speed/dwell/offset.
        self.path()

    def path(self) -> ScriptedPath:
        return ScriptedPath(waypoints=tuple(tuple(map(float, p)) for p in self.waypoints),
                            speed_mps=float(self.speed_mps), mode=self.mode,
                            dwell_s=float(self.dwell_s),
                            start_offset_s=float(self.start_offset_s))

    @classmethod
    def from_dict(cls, d: dict) -> 'PersonSpec':
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f'unknown person fields: {sorted(unknown)}')
        wp = tuple(tuple(map(float, p)) for p in d['waypoints'])
        return cls(person_id=d['person_id'], role=d['role'], waypoints=wp,
                   speed_mps=float(d['speed_mps']), mode=d.get('mode', PINGPONG),
                   dwell_s=float(d.get('dwell_s', 0.0)),
                   start_offset_s=float(d.get('start_offset_s', 0.0)),
                   height_m=float(d.get('height_m', 1.75)),
                   radius_m=float(d.get('radius_m', 0.25)),
                   character_usd=d.get('character_usd'))

    def to_dict(self) -> dict:
        return {'person_id': self.person_id, 'role': self.role,
                'waypoints': [list(p) for p in self.waypoints],
                'speed_mps': self.speed_mps, 'mode': self.mode, 'dwell_s': self.dwell_s,
                'start_offset_s': self.start_offset_s, 'height_m': self.height_m,
                'radius_m': self.radius_m, 'character_usd': self.character_usd}


@dataclass(frozen=True)
class DetectionConfig:
    """How detections are produced and published (piece 2).

    ``source`` selects the detection provider:

    * ``detector`` -- the real fleet detector runs on the mission camera topics and
      the bridge consumes its ``/human_safety/*`` output (requires the fleet package,
      ROS and the model to be available at runtime);
    * ``ground_truth`` -- an EXPLICIT fallback that projects the known people poses
      into the cameras and feeds the real safety machine. Never selected silently;
      it must be named in the config and is reported as ground-truth evidence.
    """

    source: str = 'detector'
    cameras: Tuple[str, ...] = ('restock_camera_wrist',)
    state_topic: str = '/human_safety/state'
    # Max range (m) a ground-truth person is considered "visible" to a camera.
    gt_visible_range_m: float = 6.0
    # Half-angle (rad) of the camera cone a GT person must fall within to be "seen".
    gt_fov_halfangle_rad: float = 0.60

    def __post_init__(self):
        if self.source not in ('detector', 'ground_truth'):
            raise ValueError("detection source must be 'detector' or 'ground_truth'")
        if not self.cameras:
            raise ValueError('at least one camera frame is required')
        for name, lo in (('gt_visible_range_m', 0.0), ('gt_fov_halfangle_rad', 0.0)):
            v = getattr(self, name)
            if not _finite(v) or v <= lo:
                raise ValueError(f'{name} must be finite and positive')

    @classmethod
    def from_dict(cls, d: dict) -> 'DetectionConfig':
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f'unknown detection fields: {sorted(unknown)}')
        kw = dict(d)
        if 'cameras' in kw:
            kw['cameras'] = tuple(kw['cameras'])
        return cls(**kw)

    def to_dict(self) -> dict:
        return {'source': self.source, 'cameras': list(self.cameras),
                'state_topic': self.state_topic,
                'gt_visible_range_m': self.gt_visible_range_m,
                'gt_fov_halfangle_rad': self.gt_fov_halfangle_rad}


@dataclass(frozen=True)
class ReactionConfig:
    """Thresholds for the slow/stop base + pause/resume arm reaction (piece 3).

    Every value here is a provisional PLACEHOLDER, not a certified separation
    distance (the same posture as the fleet safety layer). The owner decisions on
    the real numbers are OPEN -- see the report.
    """

    slow_base_scale: float = 0.35       # base speed multiplier while zone == SLOW
    stop_base_scale: float = 0.0        # base speed multiplier while zone == STOP
    # Extra costmap inflation (m) the nav layer may add per zone. 0 disables; the
    # navigation obstacle layer (sim-restock-dynobs-v1) owns whether it is consumed.
    slow_inflation_m: float = 0.0
    stop_inflation_m: float = 0.0
    # The arm pauses when a person is within reach_radius + arm_hold_margin of the
    # base, or whenever the published zone is STOP. reach_radius mirrors the fleet
    # SafetyParams default (UR10 reach).
    reach_radius_m: float = 1.30
    arm_hold_margin_m: float = 0.30
    # Continuous healthy-CLEAR time required before the base resumes full speed and
    # the arm un-pauses (defence-in-depth hysteresis over the camera-side dwell).
    resume_clear_dwell_s: float = 2.0
    # If no fresh safety state arrives for this long, fail toward STOP (base) /
    # hold (arm): absence of evidence is never clearance.
    state_timeout_s: float = 0.7

    def __post_init__(self):
        for name in ('slow_base_scale', 'stop_base_scale'):
            v = getattr(self, name)
            if not _finite(v) or not 0.0 <= v <= 1.0:
                raise ValueError(f'{name} must be in [0, 1]')
        if self.stop_base_scale > self.slow_base_scale:
            raise ValueError('stop_base_scale must not exceed slow_base_scale')
        for name in ('slow_inflation_m', 'stop_inflation_m'):
            v = getattr(self, name)
            if not _finite(v) or v < 0.0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('reach_radius_m', 'arm_hold_margin_m', 'resume_clear_dwell_s',
                     'state_timeout_s'):
            v = getattr(self, name)
            if not _finite(v) or v < 0.0:
                raise ValueError(f'{name} must be finite and nonnegative')

    @classmethod
    def from_dict(cls, d: dict) -> 'ReactionConfig':
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f'unknown reaction fields: {sorted(unknown)}')
        return cls(**d)

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


@dataclass(frozen=True)
class HumansConfig:
    """The whole opt-in layer. ``enabled=False`` -> every hook is a no-op."""

    enabled: bool = False
    people: Tuple[PersonSpec, ...] = ()
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    reaction: ReactionConfig = field(default_factory=ReactionConfig)
    # Backend for authoring/animating the characters: 'teleport' (default, cheap,
    # deterministic, proven in data/human-detect-isaac-sim-v1) or 'anim_people'
    # (omni.anim.people skeletal walk, used only if the extension is installed).
    backend: str = 'teleport'
    seed: int = 0

    def __post_init__(self):
        if self.backend not in ('teleport', 'anim_people'):
            raise ValueError("backend must be 'teleport' or 'anim_people'")
        ids = [p.person_id for p in self.people]
        if len(ids) != len(set(ids)):
            raise ValueError('person_id values must be unique')
        # An enabled config MAY omit people: the default crowd is then derived from
        # the mission poses at author time (twin.humans.integration.resolve_people),
        # which raises if no poses are available to derive them from.

    @classmethod
    def from_config(cls, config: Optional[dict]) -> 'HumansConfig':
        """Parse the ``humans`` block of a runtime config; absent -> disabled."""
        block = (config or {}).get('humans') if isinstance(config, dict) else None
        if not block:
            return cls(enabled=False)
        if not isinstance(block, dict):
            raise ValueError('humans config block must be an object')
        known = {'enabled', 'people', 'detection', 'reaction', 'backend', 'seed'}
        unknown = set(block) - known
        if unknown:
            raise ValueError(f'unknown humans fields: {sorted(unknown)}')
        people = tuple(PersonSpec.from_dict(p) for p in block.get('people', []))
        detection = DetectionConfig.from_dict(block['detection']) \
            if 'detection' in block else DetectionConfig()
        reaction = ReactionConfig.from_dict(block['reaction']) \
            if 'reaction' in block else ReactionConfig()
        return cls(enabled=bool(block.get('enabled', False)), people=people,
                   detection=detection, reaction=reaction,
                   backend=block.get('backend', 'teleport'),
                   seed=int(block.get('seed', 0)))

    def to_dict(self) -> dict:
        return {'enabled': self.enabled,
                'people': [p.to_dict() for p in self.people],
                'detection': self.detection.to_dict(),
                'reaction': self.reaction.to_dict(),
                'backend': self.backend, 'seed': self.seed}

    def with_people(self, people: Tuple[PersonSpec, ...]) -> 'HumansConfig':
        return replace(self, people=tuple(people))
