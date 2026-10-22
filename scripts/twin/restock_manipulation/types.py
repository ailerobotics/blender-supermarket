"""Wire-neutral contracts. Positions and xyzw orientations use the arm base frame."""
from dataclasses import dataclass
from typing import Protocol

Pose = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class ProductProfile:
    sku: str
    # Only rigid sealed cartons; the CAD slice selects a measured top or wall seal.
    dimensions: tuple[float, float, float]
    tcp_quaternion: tuple[float, float, float, float]
    tcp_surface_offset: float = 0.0
    approach: float = 0.12
    max_reach: float = 1.1
    mass_kg: float = 0.5
    grasp_mode: str = 'top'


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
    frame: str = 'ur_base_link'
    unit_kind: str = 'handling_unit'
    retail_quantity: int | None = None  # sealed contents have not been counted


@dataclass(frozen=True)
class Target:
    observation_id: str
    source: str
    simulation_time: float
    location: str
    pose: Pose                    # observed support surface
    clear: bool
    free_dimensions: tuple[float, float, float]
    frame: str = 'ur_base_link'


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
    valid_until_steady: float | None = None
    # fm389 (sim-only, opt-in): the measured world pose of the carried/placed handling unit, in the
    # controller frame (ur_base_link), as (x, y, z, *quat) of the carton CENTRE. None unless the driver
    # computes it (RESTOCK_VERIFY_PLACE) -- a controlled-asset GROUND-TRUTH read for the PLACE physics
    # gate only (NOT inventory, which the mission `verify` stage still commits from sensor evidence). Lets
    # the collect `verify` phase refuse an empty-gripper 'success' when the carton is not resting on the
    # cart deck (run-300e placed it on the floor). Default None => byte-identical.
    placed_pose: tuple[float, ...] | None = None
    # fm400 (reposition handshake): True once the base has driven the carried carton to place_approach,
    # carried across the Isaac-adapter -> host-controller ROS boundary on the sensor packet (asdict ->
    # JSON -> Sensors(**data)). The host controller's TRANSIT phase resumes to preplace on this OR its
    # own self.base_repositioned (the in-process fixture path). Default False => byte-identical.
    base_repositioned: bool = False
    # fm449 (collect-source evidence gate): True when the FIXED source camera delivered, THIS tick at park,
    # a CLEAN synchronized frame WITH a decoded label for the calibrated source slot (NOT a held/seeded
    # republish) -- the Isaac-side clean verdict (restock_camera.source_target SourceTarget.clean) carried to
    # the host controller across the same asdict -> JSON -> Sensors(**data) boundary as base_repositioned. The
    # collect observe gate (require_collect_source_evidence) permits the reach only on this. Default False =>
    # byte-identical (the gate waits / fails fast, exactly as when the driver never sets it).
    collect_source_clean: bool = False
    # Staged-start clock sync (twin.restock.staged_start): the host /clock minus this packet's clock at
    # capture. A start-phase shortcut steps physics before the first mission tick, so /clock leads the
    # packet clock by a constant (1.55 s on start_at=transfer); the host packet-freshness check subtracts
    # it and the Isaac side moves collision_time into the packet clock. Default 0.0 => byte-identical.
    clock_offset: float = 0.0
    # Driver must reject stale/absent joint/contact/odom data before making a packet.


@dataclass(frozen=True)
class Result:
    status: str
    phase: str
    reason: str = ''
    controls: tuple[dict, ...] = ()
    evidence: tuple[dict, ...] = ()
    valid_until_steady: float | None = None


class MotionPort(Protocol):
    """Collision-aware planner + measured trajectory completion, never animation.

    begin receives base-frame TCP pose and measured start joints. update_payload
    must synchronously update the collision world and return its acknowledgement.
    Motion failure/cancellation must stop the simulation controller.
    """
    def begin(self, pose: Pose, joints: tuple[float, ...], straight_line: bool = False, joint_space: bool = False) -> None: ...
    def poll(self) -> str: ...      # running/succeeded/failed
    def stop(self) -> None: ...
    def update_payload(self, held: bool, profile: ProductProfile) -> bool: ...
