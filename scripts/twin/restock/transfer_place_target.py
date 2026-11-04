"""sim-service-place-target-v1: resolve the TRANSFER place target as ONE explicit function of
the shelf-staging flow switch.

coldwinint1 (start_at=couple, side bay, carton pre-staged) proved couple + the straight tow into
the sales-floor side bay + park, then FAILED at transfer: the place target resolved to the fm461
demo-fallback BACKROOM bay ``[-2.0, 16.424, 0.73]`` even though ``shelf_staging``+``cart_side_bay``
parked the free base at the sales-floor side bay and the service leg services the sales-floor
ENDCAP alone. The bay target is 9.7 m from where the robot parked -- nothing reachable to place.

Root cause: ``transfer_ground_truth.target`` is read FLAT by every orchestrate-side consumer (the
camera-lane ``shelf_target_provider``, the ground-truth place verification, the scripted-transfer
target) with NO dependence on the flow switch. A key filed for the legacy backroom demo silently
drove the re-sequenced sales-floor run to the wrong place.

Fix at the source: with ``shelf_staging`` on, the service leg's place is the sales-floor ENDCAP --
the PROVEN shelf place target ``scenario.initial_staging.target_surface_pose_world`` serviced from
the ``shelf_approach`` mission pose. :func:`resolve_transfer_place_target` makes that selection an
explicit function of the switch and FAILS LOUDLY (named :class:`TransferPlaceTargetError`) when the
resolved -- or a mis-filed configured -- place is farther than the robot can drive to its service
pose and reach (the exact coldwinint1 trap). Legacy (``shelf_staging`` off) returns the configured
target unchanged => byte-identical.

Pure (no Isaac/ROS/GPU); CPU-tested in ``scripts/twin/tests/test_transfer_place_target.py`` and
proven end-to-end by ``scripts/check_service_leg_carry.py`` (route bay -> endcap -> bay + reach).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# Arm base (ur_base_link) z on the mobile base: BASE_ATT 0.24 + arm_mount 0.26 (matches
# scripts/check_service_leg_carry.py UB_Z and the mobile manifest arm_mount_z).
ARM_BASE_Z = 0.50
# UR10 working envelope default (ros_host_config profile max_reach / handling_unit.max_reach_m).
DEFAULT_MAX_REACH_M = 1.30


class TransferPlaceTargetError(ValueError):
    """Loud, NAMED failure: the resolved transfer place target is inconsistent with the active
    flow -- it sits farther than the robot can drive to its service pose and reach. Raised at boot
    (orchestrate side) BEFORE the GPU run drives the base to an unreachable place (coldwinint1)."""


@dataclass(frozen=True)
class TransferPlaceTarget:
    flow: str                              # 'shelf_staging' | 'legacy'
    location: Optional[str]                # shelf/bay slot id (e.g. 'G3_back_s2')
    target_xyz: Optional[Tuple[float, float, float]]   # world place target (x, y, z)
    service_pose: Optional[Tuple[float, float, float]]  # (x, y, yaw) the base drives to, to reach it
    reach_m: Optional[float]               # arm-base -> carton-rest distance from the service pose
    source: str                            # 'shelf_staging_endcap' | 'configured_bay'
    overridden: bool                       # True when the resolver SELECTED the endcap over a
                                           # different configured target (the flow-switch override)


def shelf_staging_flow_active(config) -> bool:
    """The re-sequenced sales-floor flow (sim-shelf-staging-v1 / sim-service-leg-carry-v1): the
    coupled train tows the cart to a staging/side-bay pose, uncouples, and the free base services
    the endcap ALONE. ``cart_side_bay`` is an extension of it; the place target is the endcap either
    way, so the switch is ``shelf_staging``."""
    return bool(isinstance(config, dict) and config.get('shelf_staging'))


def _as_xyz(seq):
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _endcap_from_scenario(scenario):
    staging = (scenario or {}).get('initial_staging') or {}
    pose = staging.get('target_surface_pose_world')
    if not pose or len(pose) < 3:
        raise TransferPlaceTargetError(
            'shelf_staging transfer needs scenario.initial_staging.target_surface_pose_world '
            '(the proven sales-floor endcap place target); none found in the scenario')
    return _as_xyz(pose)


def _carton_half_height(scenario):
    dims = ((scenario or {}).get('handling_unit') or {}).get('dimensions_m')
    if dims and len(dims) >= 3:
        return float(dims[2]) / 2.0
    return 0.0


def _max_reach(scenario, override):
    if override is not None:
        return float(override)
    unit = (scenario or {}).get('handling_unit') or {}
    return float(unit.get('max_reach_m', DEFAULT_MAX_REACH_M))


def _reach_from(service_pose, target_xyz, *, carton_half=0.0):
    """3-D arm-base -> carton-rest distance -- the same reach envelope test the live_pose transfer
    observe gate applies (REACH_ENVELOPE_FAIL). The carton rests ON the slot surface, so the arm
    reaches the carton CENTRE = surface_z + carton_half (matches check_service_leg_carry)."""
    sx, sy = float(service_pose[0]), float(service_pose[1])
    tx, ty, tz = target_xyz
    return math.dist((sx, sy, ARM_BASE_Z), (tx, ty, tz + carton_half))


def _service_pose(config):
    poses = config.get('mission_poses') or {}
    pose = poses.get('shelf_approach')
    if not pose or len(pose) < 2:
        raise TransferPlaceTargetError(
            "shelf_staging transfer needs a 'shelf_approach' mission pose (the endcap service "
            "pose the free base drives to, to place); none found in mission_poses")
    yaw = float(pose[2]) if len(pose) >= 3 else 0.0
    return (float(pose[0]), float(pose[1]), yaw)


def resolve_transfer_place_target(config, scenario, *, max_reach_m=None) -> TransferPlaceTarget:
    """Resolve the transfer PLACE target as one explicit function of the shelf-staging flow switch.

    shelf_staging ON  -> the sales-floor ENDCAP (scenario.initial_staging.target_surface_pose_world),
                         serviced from the ``shelf_approach`` pose. Raises
                         :class:`TransferPlaceTargetError` when the endcap -- or a configured target
                         that disagrees with it -- is farther than the robot can drive-and-reach.
    shelf_staging OFF -> the configured ``transfer_ground_truth.target`` unchanged (byte-identical).
    """
    gt_raw = config.get('transfer_ground_truth')
    gt = gt_raw if isinstance(gt_raw, dict) else {}
    # The configured ground-truth target is an ACTIVE place target only when the block is enabled;
    # a disabled block is the camera-sensed path (sensing.py reads it only when enabled), so a stale
    # target under a disabled block must NOT be treated as the place (nor trip the reach guard).
    gt_enabled = bool(gt.get('enabled', False) if isinstance(gt_raw, dict) else gt_raw)
    configured = gt.get('target') if gt_enabled else None
    conf_xyz = _as_xyz(configured) if configured and len(configured) >= 3 else None
    location = (gt.get('location') or config.get('transfer_shelf_location')
                or ((scenario or {}).get('job') or {}).get('shelf_id'))

    if not shelf_staging_flow_active(config):
        return TransferPlaceTarget(
            flow='legacy', location=location, target_xyz=conf_xyz, service_pose=None,
            reach_m=None, source='configured_bay', overridden=False)

    # --- shelf_staging flow: the service leg services the sales-floor endcap alone ---
    endcap = _endcap_from_scenario(scenario)
    service = _service_pose(config)
    max_reach = _max_reach(scenario, max_reach_m)
    carton_half = _carton_half_height(scenario)

    endcap_reach = _reach_from(service, endcap, carton_half=carton_half)
    if endcap_reach > max_reach:
        # sim-service-carry-drive-to-shelf-v1: name the trip 'unreachable place target: <dist> m >
        # <reach> m' and raise it HERE, at PLAN TIME (boot, orchestrate side) BEFORE any GPU boot or
        # arm command -- not as a mid-transfer 'unreachable or undersized target' emergency stop or an
        # 'observe' phase timeout. Move shelf_approach toward the endcap (never raise the reach bound).
        raise TransferPlaceTargetError(
            f'unreachable place target: {endcap_reach:.3f} m > {max_reach:.2f} m -- the shelf_staging '
            f'endcap place {tuple(round(v, 3) for v in endcap)} is {endcap_reach:.3f} m from the '
            f'service pose {tuple(round(v, 3) for v in service[:2])} arm base, past the {max_reach:.2f} m '
            f'UR10 reach; the service leg cannot place there. Move shelf_approach closer to the endcap '
            f'(never the reach bound), or check scenario.initial_staging.target_surface_pose_world.')

    # Loud guard against the coldwinint1 mis-file: a configured target the service leg could NEVER
    # reach (the fm461 BACKROOM bay) must not silently drive the base 9.7 m away. Name it.
    overridden = False
    if conf_xyz is not None:
        conf_reach = _reach_from(service, conf_xyz, carton_half=carton_half)
        if conf_reach > max_reach:
            raise TransferPlaceTargetError(
                f'shelf_staging is ON but transfer_ground_truth.target '
                f'{tuple(round(v, 3) for v in conf_xyz)} is {conf_reach:.3f} m from the endcap '
                f'service pose {tuple(round(v, 3) for v in service[:2])} -- farther than the robot '
                f'can drive-and-reach ({max_reach:.2f} m). The service leg services the sales-floor '
                f'endcap {tuple(round(v, 3) for v in endcap)}; set transfer_ground_truth.target to '
                f'the endcap (scenario.initial_staging.target_surface_pose_world).')
        overridden = math.dist(conf_xyz, endcap) > 1e-6

    return TransferPlaceTarget(
        flow='shelf_staging', location=location, target_xyz=endcap, service_pose=service,
        reach_m=endcap_reach, source='shelf_staging_endcap', overridden=overridden)


# sim-verify-region-endcap-v1: the injected / configured ground-truth VERIFY bay centre must sit
# within this of the resolved place target under shelf_staging. Farther means the verify stage still
# frames a DIFFERENT surface (the legacy backroom bay, ~9.7 m from the sales-floor endcap) and would
# false-fail after the service leg placed at the endcap. Half a metre comfortably spans an endcap-sized
# support while catching the backroom mis-file -- the verify sibling of the place-target reach guard.
DEFAULT_VERIFY_CENTER_TOL_M = 0.5


class VerifyRegionError(TransferPlaceTargetError):
    """Loud, NAMED failure: the configured post-place VERIFY ground-truth region disagrees with the
    resolved place target under shelf_staging -- it still frames the backroom bay, so the verify stage
    would check the wrong surface after the service leg placed at the endcap. Raised at boot (orchestrate
    side) BEFORE the GPU run, the verify sibling of :class:`TransferPlaceTargetError` (coldwinint1)."""


@dataclass(frozen=True)
class VerifyRegion:
    flow: str                              # 'shelf_staging' | 'legacy'
    bay: Optional[dict]                    # {center_xy, size_xy, top_z} to WRITE into verify_ground_truth,
                                           # or None to leave the config's verify region exactly as authored
    source: str                            # 'shelf_staging_endcap' | 'configured' | 'unchanged'
    center_offset_m: Optional[float]       # distance from the configured verify centre to the place target
    overridden: bool                       # True when the resolver injected the endcap bay over a bay-less /
                                           # scenario.transfer_bay default


def _verify_block(config):
    """``(enabled, block)`` for ``verify_ground_truth`` -- a bool or dict, like ground_truth_verify."""
    block = config.get('verify_ground_truth') if isinstance(config, dict) else None
    if isinstance(block, bool):
        return block, {}
    if isinstance(block, dict):
        return bool(block.get('enabled')), block
    return False, {}


def resolve_verify_region(config, scenario, place_target, *,
                          center_tol_m=DEFAULT_VERIFY_CENTER_TOL_M) -> VerifyRegion:
    """Resolve the post-place VERIFY ground-truth region as ONE explicit function of the shelf-staging
    flow switch -- the verify sibling of :func:`resolve_transfer_place_target`.

    shelf_staging ON (verify enabled, a resolved place target)
        -> the verify bay must FRAME the place target (the sales-floor endcap). A configured
           ``verify_ground_truth.bay`` centred farther than ``center_tol_m`` from it (the legacy
           backroom bay) is REFUSED by name (:class:`VerifyRegionError`). A bay-less block gets the
           endcap bay injected -- otherwise ``ground_truth_verify.from_config`` reads
           ``scenario.transfer_bay`` (the backroom) and false-fails the verify after an endcap place.
    shelf_staging OFF / verify disabled / no resolved place target
        -> ``bay=None`` so the config's verify region is left EXACTLY as authored => byte-identical.
    """
    enabled, block = _verify_block(config)
    if (not shelf_staging_flow_active(config) or not enabled
            or place_target is None or place_target.target_xyz is None):
        flow = 'shelf_staging' if shelf_staging_flow_active(config) else 'legacy'
        return VerifyRegion(flow=flow, bay=None, source='unchanged',
                            center_offset_m=None, overridden=False)

    endcap = place_target.target_xyz
    configured_bay = block.get('bay') if isinstance(block.get('bay'), dict) else None
    center = configured_bay.get('center_xy') if configured_bay else None
    if center and len(center) >= 2:
        offset = math.dist((float(center[0]), float(center[1])), (endcap[0], endcap[1]))
        if offset > center_tol_m:
            raise VerifyRegionError(
                f'shelf_staging is ON but verify_ground_truth.bay is centred at '
                f'{tuple(round(float(v), 3) for v in center[:2])} -- {offset:.3f} m from the '
                f'service-leg place target (the sales-floor endcap '
                f'{tuple(round(v, 3) for v in endcap[:2])}, farther than the {center_tol_m:.2f} m '
                f'tolerance). The verify stage would check the wrong surface (the legacy backroom bay) '
                f'after an endcap place; centre verify_ground_truth.bay on the endcap place target '
                f'(scenario.initial_staging.target_surface_pose_world).')
        # A configured bay already consistent with the endcap -- keep the lane's authored sizing.
        return VerifyRegion(flow='shelf_staging', bay=None, source='configured',
                            center_offset_m=offset, overridden=False)

    # No explicit (consistent) bay: inject the endcap-centred bay so from_config does not fall back to
    # scenario.transfer_bay (the backroom). Place and verify then agree by construction.
    bay = verify_bay_for_place_target(endcap, scenario)
    return VerifyRegion(flow='shelf_staging', bay=bay, source='shelf_staging_endcap',
                        center_offset_m=0.0, overridden=True)


def verify_bay_for_place_target(place_target_xyz, scenario, *, fallback_size_xy=(0.7, 0.5)):
    """The ground-truth VERIFY bay centred ON the resolved place target, so the place and its
    post-place verification agree BY CONSTRUCTION in the shelf_staging flow.

    The ground-truth verifier (``twin.restock.ground_truth_verify.from_config``) otherwise reads
    ``scenario.transfer_bay`` -- the legacy BACKROOM bay -- so after the service leg places at the
    sales-floor endcap the verify stage would check the backroom and false-fail. Size/height come
    from the scenario support geometry (``transfer_bay.dimensions_m`` if present, else
    ``initial_staging.support_dimensions_m``); the top is the place-target z. Returns a ``bay``
    override block for ``verify_ground_truth``."""
    cx, cy, cz = (float(place_target_xyz[0]), float(place_target_xyz[1]), float(place_target_xyz[2]))
    sc = scenario or {}
    bay = sc.get('transfer_bay') or {}
    size = (bay.get('dimensions_m')
            or (sc.get('initial_staging') or {}).get('support_dimensions_m')
            or list(fallback_size_xy))
    return {'center_xy': [cx, cy], 'size_xy': [float(size[0]), float(size[1])], 'top_z': cz}
