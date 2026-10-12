"""Wire-neutral contracts. Positions and xyzw orientations use the arm base frame."""
from dataclasses import dataclass
from typing import Protocol

Pose = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class ProductProfile:
    sku: str
    # Only rigid, sealed boxes with a horizontal suctionable top are supported.
    dimensions: tuple[float, float, float]
    tcp_quaternion: tuple[float, float, float, float]
    tcp_surface_offset: float = 0.0
    approach: float = 0.12
    max_reach: float = 1.1


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source: str
    simulation_time: float
    sku: str
    location: str
    pose: Pose                    # observed top contact surface
    confidence: float
    quantity: int = 1              # one sealed handling unit, not contents count
    family: str = 'sealed_box'
    frame: str = 'base_link'


@dataclass(frozen=True)
class Target:
    observation_id: str
    source: str
    simulation_time: float
    location: str
    pose: Pose                    # observed support surface
    clear: bool
    free_dimensions: tuple[float, float, float]
    frame: str = 'base_link'


@dataclass(frozen=True)
class Sensors:
    simulation_time: float
    observations: tuple[Observation, ...]
    target: Target | None
    parked: bool
    contact: bool
    feedback_time: float
    collision_time: float
    joint_positions: tuple[float, ...]
    scanned_locations: tuple[str, ...] = ()
    scan_time: float = -1.0
    # Driver must reject stale/absent joint/contact/odom data before making a packet.


@dataclass(frozen=True)
class Result:
    status: str
    phase: str
    reason: str = ''
    controls: tuple[dict, ...] = ()
    evidence: tuple[dict, ...] = ()


class MotionPort(Protocol):
    """Collision-aware planner + measured trajectory completion, never animation.

    begin receives base-frame TCP pose and measured start joints. update_payload
    must synchronously update the collision world and return its acknowledgement.
    Motion failure/cancellation must stop the simulation controller.
    """
    def begin(self, pose: Pose, joints: tuple[float, ...]) -> None: ...
    def poll(self) -> str: ...      # running/succeeded/failed
    def stop(self) -> None: ...
    def update_payload(self, held: bool, profile: ProductProfile) -> bool: ...
