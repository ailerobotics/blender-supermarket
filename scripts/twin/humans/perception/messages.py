# SPDX-License-Identifier: Apache-2.0
"""ROS-free data model shared by the detector, fusion, safety and viz layers.

Everything here is a plain dataclass so the decision logic can be exercised in
offline unit tests with no ROS, no torch and no GPU. The ROS nodes serialise
these to/from JSON on ``std_msgs/String`` topics (see ``json_util``) rather than
custom ROS messages, because this repository is not colcon-built (no
``package.xml`` anywhere) and a custom .msg would force a build step into the
otherwise script-based workflow and the offline test path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Zone(str, Enum):
    """Safety zones, ordered by severity. ``CLEAR`` < ``SLOW`` < ``STOP``."""

    CLEAR = "CLEAR"
    SLOW = "SLOW"
    STOP = "STOP"

    @property
    def severity(self) -> int:
        return {"CLEAR": 0, "SLOW": 1, "STOP": 2}[self.value]


class Method(str, Enum):
    """How a 3D position was obtained."""

    DEPTH = "depth"            # robust depth inside the person region
    MONOCULAR = "monocular"    # ground-plane / height-prior from a known pose
    PRESENCE = "presence"      # person seen but no trustworthy position


@dataclass
class Detection2D:
    """A single person detected in one image."""

    bbox: tuple  # (x0, y0, x1, y1) pixels
    score: float
    # (K, 3) array of (u, v, confidence) COCO keypoints, or None for box-only.
    keypoints: Optional[np.ndarray] = None

    def bbox_center(self) -> tuple:
        x0, y0, x1, y1 = self.bbox
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


@dataclass
class Detection3D:
    """One person from one camera, lifted (or not) to the robot base frame.

    ``position`` and ``nearest_point`` are (3,) arrays in the base frame, or
    None when the camera has no trustworthy pose (presence-only). ``sigma`` is a
    1-sigma isotropic position uncertainty in metres; monocular estimates carry
    an honestly larger value than depth estimates.
    """

    camera: str
    stamp: float                 # image acquisition time (seconds, ROS clock)
    score: float
    method: Method
    present: bool = True
    position: Optional[np.ndarray] = None      # body centre, base frame
    nearest_point: Optional[np.ndarray] = None  # nearest credible body point
    sigma: float = 0.0
    n_depth_valid: int = 0       # valid depth samples used (depth method)

    @property
    def positioned(self) -> bool:
        return self.position is not None


@dataclass
class Track:
    """A fused person track in the base frame (constant-velocity state)."""

    id: int
    position: np.ndarray          # (3,)
    velocity: np.ndarray          # (3,)
    covariance: np.ndarray        # (6, 6) state covariance
    stamp: float
    hits: int = 0
    age: int = 0
    time_since_update: float = 0.0
    confirmed: bool = False
    nearest_point: Optional[np.ndarray] = None
    cameras: tuple = ()           # roles that contributed at last update


@dataclass
class CameraHealth:
    """Per-camera liveness as judged by fusion."""

    camera: str
    present_in_graph: bool
    frame_age: float = float("inf")   # seconds between image stamp and clock
    detector_alive: bool = False
    positioned: bool = False          # contributes a 3D position (vs presence)
    fresh: bool = False               # frame_age within budget
    note: str = ""


class DistanceStatus(str, Enum):
    """Honesty of ``nearest_distance``. UNKNOWN is NOT 'far' (report §7 #5)."""

    KNOWN = "known"                       # a calibrated camera localised a person
    UNKNOWN_PRESENCE_ONLY = "unknown"     # someone visible but only on a presence-only camera
    NONE = "none"                         # no person anywhere


@dataclass
class SafetyState:
    """The published judgement. Fails toward STOP.

    ``evidence_stamp`` is the OLDEST positioned-detection stamp that sizes the
    current STOP ring (NOT the publication time, and NOT the freshest frame) — the
    timestamp the robot-side gate uses for its own staleness check, kept
    conservative so the latency it derives is never understated (human-safety-s1
    §7). It is ``None`` when nobody is being localised. ``heartbeat_seq`` advances
    every publish so the gate can detect a wedged publisher.
    """

    stamp: float                      # publication time (wall clock)
    zone: Zone
    nearest_distance: float           # to the reach envelope, metres (inf if none/unknown)
    nearest_distance_base: float      # to the base origin, metres (inf if none/unknown)
    distance_status: DistanceStatus
    nearest_track_id: Optional[int]
    any_person_visible: bool          # includes presence-only cameras
    reasons: list = field(default_factory=list)
    health: list = field(default_factory=list)   # list[CameraHealth]
    tracks: list = field(default_factory=list)    # list[Track]
    evidence_stamp: Optional[float] = None        # oldest positioned-detection stamp sizing the STOP ring
    heartbeat_seq: int = 0
    # DIAGNOSTIC: the body-CENTRE distances (what the zone used before the
    # nearest-point fix), kept alongside the nearest-point distances above for
    # offline comparison. The zone is sized from ``nearest_distance`` /
    # ``nearest_distance_base``; consumers should not read these.
    nearest_distance_centre: float = float("inf")       # centre -> reach envelope, m
    nearest_distance_base_centre: float = float("inf")  # centre -> base origin, m
