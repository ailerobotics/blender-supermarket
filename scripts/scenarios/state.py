"""Scenario state model: entities and a frame timeline, all pure Python.

A :class:`Scenario` is a fully-resolved, deterministic description of one labelled
sequence — every entity that ever appears, when it is present, and (for movers) its
path. ``state_at(frame)`` returns the exact world state for a frame; the Blender
builder realises it and the annotator projects it, so both agree frame by frame.

Nothing here imports ``bpy``; the whole simulation (depletion schedules, people
paths, spill onsets) is unit-tested offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from camera import CameraPose

Vec3 = Tuple[float, float, float]

# Annotation classes. ``is_hazard`` / ``is_distractor`` drive the floor-hazard GT:
# distractors are annotated so an evaluator can confirm they are *not* flagged.
HAZARD_CLASSES = {"water", "milk", "juice", "oil", "debris", "fallen_product"}
DISTRACTOR_CLASSES = {"shadow", "reflection", "floor_marking", "paper_sign"}


@dataclass
class Entity:
    """One annotated or structural object in the scene."""
    id: str
    kind: str                 # product | person | spill | distractor | fixture | floor
    cls: str                  # SKU id, hazard class, distractor class, "person", ...
    pass_index: int           # unique object index -> IndexOB mask channel
    center: Vec3
    size: Vec3                # world-axis-aligned full extents
    rot_z: float = 0.0
    annotate: bool = True     # emit ground truth for this entity
    present_from: Optional[int] = None   # inclusive; None = from frame 0
    present_to: Optional[int] = None     # inclusive; None = to last frame
    path: List[Tuple[int, Vec3, float]] = field(default_factory=list)  # (frame, center, rot_z)
    # Free-form, scenario-specific ground truth attached to this entity.
    meta: Dict = field(default_factory=dict)

    def present_at(self, frame: int) -> bool:
        if self.present_from is not None and frame < self.present_from:
            return False
        if self.present_to is not None and frame > self.present_to:
            return False
        return True

    def pose_at(self, frame: int) -> Tuple[Vec3, float]:
        if not self.path:
            return self.center, self.rot_z
        kf = self.path
        if frame <= kf[0][0]:
            return kf[0][1], kf[0][2]
        if frame >= kf[-1][0]:
            return kf[-1][1], kf[-1][2]
        for (f0, c0, r0), (f1, c1, r1) in zip(kf, kf[1:]):
            if f0 <= frame <= f1:
                a = (frame - f0) / (f1 - f0) if f1 > f0 else 0.0
                c = (c0[0] + (c1[0] - c0[0]) * a,
                     c0[1] + (c1[1] - c0[1]) * a,
                     c0[2] + (c1[2] - c0[2]) * a)
                # shortest-arc interpolation keeps heading smooth
                dr = (r1 - r0 + 3.14159265) % (2 * 3.14159265) - 3.14159265
                return c, r0 + dr * a
        return self.center, self.rot_z

    def corners_at(self, frame: int) -> List[Vec3]:
        (cx, cy, cz), _ = self.pose_at(frame)
        sx, sy, sz = self.size
        return [(cx + dx, cy + dy, cz + dz)
                for dx in (-sx / 2, sx / 2)
                for dy in (-sy / 2, sy / 2)
                for dz in (-sz / 2, sz / 2)]


@dataclass
class Scenario:
    name: str
    kind: str                 # depletion | spills | people | misplaced | multiview
    seed: int
    fps: int
    n_frames: int
    cameras: List[CameraPose]
    entities: List[Entity]
    params: Dict = field(default_factory=dict)      # the knobs this run was built with
    ground_truth: Dict = field(default_factory=dict)  # scenario-level GT (slots, events...)
    with_depth: bool = False  # request a depth pass + 16-bit depth output

    def state_at(self, frame: int) -> List[Tuple[Entity, Vec3, float]]:
        """Present entities at ``frame`` with their resolved (center, rot_z)."""
        out = []
        for e in self.entities:
            if not e.present_at(frame):
                continue
            c, r = e.pose_at(frame)
            out.append((e, c, r))
        return out

    def manifest(self) -> dict:
        return dict(
            name=self.name, kind=self.kind, seed=self.seed, fps=self.fps,
            n_frames=self.n_frames, with_depth=self.with_depth,
            cameras=[c.name for c in self.cameras], params=self.params,
            n_entities=len(self.entities), ground_truth=self.ground_truth)
