"""Fixed-shelf transfer sensing: slot target + post-placement verification (camera lane).

The transfer stage places the pick carton into a shelf slot and then confirms the placement,
both sensed with the FIXED ``shelf`` station camera against its commissioned ``G3_back_s2``
empty-support baseline — the calibrated, pose-stable, commissionable view, not the arm
eye-in-hand camera (demoted to masked depth + fine alignment). This mirrors
:mod:`twin.restock_camera.source_target` for the shelf side:

- :func:`build_shelf_target_provider` → ``shelf_target_provider(now) -> ShelfSlotTarget | None``
  gives the target slot's support-surface pose in the shelf station (``map``) frame, its free
  dimensions and its observed-vacancy flag, with the baseline provenance and acquisition stamp.
  The arm receives a target REPROJECTED from this by integration (``isaac_factory``); that
  reprojection is not this module's job. The consumer must require ``clear`` before placing.
- :func:`build_shelf_verification_provider` → ``verify(now, expected_sku, expected_pose) ->
  ShelfVerification`` captures again AFTER placement and reports whether the expected carton is
  sensed in the slot and whether its pose is within the position/yaw tolerance of where it was
  commanded. It never credits a placement from a fixture — the carton identity and pose come only
  from the fixed camera's decode + depth against the baseline.

Both go through the single fixed ``shelf`` ``IsaacCartonSensors.capture(now, 'G3_back_s2', ...)``,
so both inherit the live 0.5 s acquisition-to-use freshness bound the capture enforces; the
returned ``stamp`` lets the consumer re-validate freshness at the moment of use (base parked).

Contract: ``data/sim-restock-camera-v2/shelf-contract.md``. No Isaac/ROS imports: the adapter is
duck-typed (``capture``, ``evidence``, ``camera_id``).
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ShelfSlotTarget:
    """A shelf slot sensed by the fixed shelf camera, in the station (map) frame."""
    location: str                 # mission shelf id, e.g. 'G3_back_s2'
    pose_station: tuple           # support-surface center (x,y,z, qx,qy,qz,qw) in the station frame
    free_dimensions: tuple        # (x, y, z) free footprint + measured vertical clearance
    clear: bool                   # the slot is observed empty (ready to receive); consumer must gate on it
    stamp: float                  # acquisition simulation_time of the measuring frame
    provenance: dict              # baseline linkage; the consumer asserts sensor provenance
    station_frame: str = 'map'


@dataclass(frozen=True)
class ShelfVerification:
    """Post-placement verification of one shelf slot against the commissioned baseline."""
    location: str
    present: bool                 # the EXPECTED carton is sensed in the slot
    within_tolerance: bool        # present AND its pose is within the position + yaw tolerance
    observed_sku: str | None      # what was sensed (distinguishes empty from a wrong carton)
    observed_pose: tuple | None   # sensed carton pose in the station frame
    position_error_m: float | None
    yaw_error_rad: float | None
    stamp: float | None           # acquisition simulation_time of the measuring frame
    provenance: dict | None       # baseline linkage when a carton was sensed


def _slot_evidence(adapter, location):
    for item in adapter.evidence:
        if item['location'] == location:
            return item
    raise ValueError(f'no support evidence for shelf station {location!r} after capture')


def _provenance(adapter, evidence, source):
    return dict(camera_id=adapter.camera_id, source=source,
                calibration_id=evidence['baseline_calibration_id'],
                baseline_session_id=evidence['baseline_session_id'],
                baseline_stamp=evidence['baseline_stamp'],
                frame_id=evidence['current_frame_id'], epoch=evidence['epoch'],
                registration_uncertainty_m=evidence['registration_uncertainty_m'],
                qr_rgb_sha256=evidence['qr_rgb_sha256'])


def _yaw(pose):
    """Yaw about +z from an (x,y,z, qx,qy,qz,qw) pose (the convention the capture authors)."""
    return 2. * math.atan2(pose[5], pose[6])


def _wrap(angle):
    return (angle + math.pi) % (2. * math.pi) - math.pi


def build_ground_truth_shelf_target_provider(location, target_xyz, *,
                                             free_dimensions=(0.6, 0.6, 0.5), station_frame='map'):
    """fm461 demo fallback: a GROUND-TRUTH transfer-destination provider that returns the AUTHORED bay
    support pose every tick, with no fixed shelf camera / commissioned baseline (marker decode is blocked
    at the bay; follow-up). The target is honestly tagged ``provenance['source'] == 'ground_truth'`` so the
    isaac bridge labels the Target source 'ground_truth' (not 'sensor') and the controller's
    transfer_ground_truth gate -- and only that gate -- accepts it; the placement is then verified from the
    sim-measured carton pose vs this bay target (controller ground_truth_place_check). NEVER use this to
    certify real inventory -- it is a sim-only demo verify, labelled as such end-to-end."""
    pose = (float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]), 0.0, 0.0, 0.0, 1.0)
    fd = tuple(float(v) for v in free_dimensions)

    def ground_truth_shelf_target_provider(now):
        return ShelfSlotTarget(location=location, pose_station=pose, free_dimensions=fd, clear=True,
                               stamp=float(now),
                               provenance={'source': 'ground_truth', 'location': location,
                                           'note': 'fm461 authored bay pose; sim ground-truth demo verify'},
                               station_frame=station_frame)
    return ground_truth_shelf_target_provider


def build_shelf_target_provider(adapter, motion, contact, *, shelf_location='G3_back_s2',
                                station_frame='map'):
    """Return ``shelf_target_provider(now) -> ShelfSlotTarget | None`` over a fixed-shelf adapter.

    ``adapter`` is the commissioned fixed-``shelf`` ``IsaacCartonSensors`` (its ``shelf`` context
    carries the ``G3_back_s2`` baseline); ``motion``/``contact`` are the integration ports, closed
    over so the transfer stage calls the provider with ``now`` alone. The provider never
    fabricates: it returns ``None`` when the shelf camera senses no slot this tick, and the slot
    pose/free dimensions/vacancy come only from the fixed camera's depth against the baseline.
    """
    def shelf_target_provider(now):
        packet = adapter.capture(now, shelf_location, shelf_location, motion, contact)
        target = packet.target
        if target is None or target.location != shelf_location:
            return None                       # slot not sensed this tick; absence is never a target
        if target.source != 'sensor':
            raise ValueError('transfer slot target must be sensor-sourced, never a fixture')
        evidence = _slot_evidence(adapter, shelf_location)
        return ShelfSlotTarget(location=shelf_location, pose_station=tuple(target.pose),
                               free_dimensions=tuple(target.free_dimensions), clear=bool(target.clear),
                               stamp=float(target.simulation_time),
                               provenance=_provenance(adapter, evidence, target.source),
                               station_frame=station_frame)
    return shelf_target_provider


def build_shelf_verification_provider(adapter, motion, contact, *, shelf_location='G3_back_s2',
                                      station_frame='map'):
    """Return ``verify(now, expected_sku, expected_pose, *, position_tol_m, yaw_tol_rad)``.

    The post-placement check: capture the fixed shelf camera again and report whether the
    EXPECTED carton is sensed in the target slot and whether its sensed pose is within tolerance
    of ``expected_pose`` (the commanded placement, in the station frame). Absence yields
    ``present=False`` (never a fabricated success); a wrong-SKU carton yields ``present=False``
    with ``observed_sku`` set so the consumer can distinguish an empty slot from a mis-stock.
    ``within_tolerance`` is only ever true when the expected carton is present.
    """
    def verify(now, expected_sku, expected_pose, *, position_tol_m=0.03, yaw_tol_rad=0.26):
        packet = adapter.capture(now, shelf_location, shelf_location, motion, contact)
        cartons = [o for o in packet.observations if o.location == shelf_location]
        if not cartons:
            return ShelfVerification(location=shelf_location, present=False, within_tolerance=False,
                                     observed_sku=None, observed_pose=None, position_error_m=None,
                                     yaw_error_rad=None, stamp=None, provenance=None)
        if len(cartons) != 1:
            raise ValueError(f'shelf station {shelf_location!r} shows more than one handling unit')
        obs = cartons[0]
        if obs.source != 'sensor':
            raise ValueError('verification carton must be sensor-sourced, never a fixture')
        evidence = _slot_evidence(adapter, shelf_location)
        provenance = _provenance(adapter, evidence, obs.source)
        present = obs.sku == expected_sku
        position_error = math.dist(obs.pose[:3], tuple(expected_pose[:3]))
        yaw_error = abs(_wrap(_yaw(obs.pose) - _yaw(expected_pose)))
        within = bool(present and position_error <= position_tol_m and yaw_error <= yaw_tol_rad)
        return ShelfVerification(location=shelf_location, present=bool(present), within_tolerance=within,
                                 observed_sku=obs.sku, observed_pose=tuple(obs.pose),
                                 position_error_m=float(position_error), yaw_error_rad=float(yaw_error),
                                 stamp=float(obs.simulation_time), provenance=provenance)
    return verify
