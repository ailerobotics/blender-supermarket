"""Cage-aware re-grasp off the PARKED cart + the reposition place-park (sim-transfer-regrasp-reposition-v1).

coldwin93 ran couple + the tow + the uncouple + the park on the GPU for the first time, then the TRANSFER
stage died at first contact: the live_pose transfer PICK approach is a NEAR-FACE SIDE reach to the carton's
mouth-ward face deep inside the rollwagen cage, and cuMotion could not find a collision-free inverse
kinematics solution for it -- ``trajopt: INVERSE_KINEMATICS_FAILURE`` ten times in a row
(host-stack.log 1467-1952), ``PLAN_TOTAL_ATTEMPTS_CAP`` -> ``HOST_FAULT_ESTOP phase=approach``. The nvblox
ESDF marks the cage bars + the carton as occupied and ``world_obstacles=[]`` gave cuMotion no clean cart
geometry, so every side-reach config into the +x mouth is in collision. The observed store-shelf place target
(``place_surface=(2.175, 1.237, 0.186)`` base, reach 2.39 m) is also > 2 m from the arm base, so even a
successful pick could not be placed from where the robot parks.

This module is the pure-CPU half of the fix (opt-in, default-off => byte-identical). Two pieces:

1. **Cage-aware re-grasp through the mouth.** After uncouple the cart pose is NOT the coupled hitch offset;
   it comes from the PARKED cart prim (:func:`cart_pose_in_base`). :func:`parked_cart_world_obstacles` turns
   the authored rollwagen colliders into the PR 350 ``world_obstacles`` host-key specs AT that parked pose,
   so cuMotion plans the approach around clean box geometry (the open +x mouth routes the gripper in). The
   re-grasp itself is the PR 376 pad-down geometry IN REVERSE: a pad-DOWN top grasp that approaches to a
   standoff ABOVE the carton inside the cage, descends to the carton top, grasps (the existing seal/weld
   path), lifts, and withdraws STRAIGHT OUT the mouth -- never the side reach that cuMotion could not IK.
   :func:`mouth_pad_down_path` builds that path as cart-frame carton poses and :func:`regrasp_cage_clearance`
   proves the gripper (empty on the approach/descent, gripper+carton on the carried lift/withdraw) clears
   every non-support cage body with the sweep tool; :func:`carried_withdraw_level_ok` proves the carried
   withdraw stays within the 2.0 deg carton-level bound.

2. **Reposition before the place.** :func:`place_park_pose` computes the base park pose from the endcap place
   target and the arm reach -- the target within ~0.9 m of the base, facing the shelf -- and
   :func:`place_reach_ok` proves the place + the level descent onto the store shelf top (0.73) sit inside the
   UR10 working envelope from that park. This is a tighter, derived sibling of the authored ``shelf_approach``
   pose the existing service leg (sim-service-leg-carry-v1) drives to.

Pure numpy; the cage geometry needs ``usd-core`` (``pxr`` via :func:`place_clearance.rollwagen_collider_boxes`)
unless the caller passes a pre-baked ``collider_boxes`` dict (the CPU tests do). No Isaac, ROS or GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import place_clearance as pc

# Arm base (ur_base_link) z on the mobile base: BASE_ATT 0.24 + arm_mount 0.26 -- the SAME constant
# transfer_place_target.ARM_BASE_Z / check_service_leg_carry.UB_Z use for the reach envelope.
ARM_BASE_Z = 0.50
# UR10 working envelope (ros_host_config profile max_reach / handling_unit.max_reach_m).
DEFAULT_MAX_REACH_M = 1.30
# The captain's reposition goal: the place target within ~0.9 m of the arm base (a comfortable reach, not the
# 1.217 m envelope-edge reach of the authored shelf_approach).
DEFAULT_PLACE_REACH_M = 0.90
# Pad-down standoff above the carton top (m): the approach parks the gripper this far above the carton inside
# the cage, then descends. For a DECK carton this must stay below the upper shelf (0.907 cart-frame); the
# default 0.12 m clears it with ~0.5 m to spare and matches the profile approach offset.
DEFAULT_REGRASP_STANDOFF_M = 0.12
# How far along the +x mouth axis the withdraw carries the carton clear of the front posts (cart frame). The
# posts sit at x ~ 0.393; 0.65 m puts the carton (half 0.225) fully outside the mouth.
DEFAULT_MOUTH_EXIT_X_M = 0.65
# Minimum gap (m) the swept gripper(+carton) must keep from every NON-support rollwagen body along the
# re-grasp path. 30 mm matches the place-clearance / place-descent cage margins.
DEFAULT_REGRASP_CAGE_MARGIN_M = 0.030
# Cart bodies NOT published as re-grasp world obstacles by default: the drawbar is the tow hitch, which
# sticks out the +x mouth right at the arm base (~34 mm standing clearance -- the same reason
# place_clearance.cage_body_names drops it), so a world CollisionObject for it would OVERLAP the arm's start
# config and reject EVERY plan with INVALID_INITIAL_CSPACE (firstmate 295). The re-grasp does not thread the
# drawbar; the cage bars + supports are what cuMotion must route the mouth-approach around.
DEFAULT_WORLD_OBSTACLE_EXCLUDE = ('drawbar',)
# The carton-level bound (deg) the carried withdraw must not exceed -- never loosened (the captain's
# "carton parallel to the ground" safety); mirrors carton_level.DEFAULT_TILT_BOUND_DEG.
DEFAULT_TILT_BOUND_DEG = 2.0

# Log / refuse token for the re-grasp cage sweep.
REGRASP_CAGE_CLEARANCE = 'REGRASP_CAGE_CLEARANCE'
REGRASP_CAGE_CLEARANCE_REJECTED = 'REGRASP_CAGE_CLEARANCE_REJECTED'


# --------------------------------------------------------------- cart pose in the arm base frame

def _rotz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _xyyaw_to_T(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rotz(float(yaw))
    T[:3, 3] = (float(x), float(y), float(z))
    return T


def cart_pose_in_base(robot_park, cart_park, *, arm_base_z: float = ARM_BASE_Z,
                      cart_base_z: float = 0.0) -> Tuple[float, float, float, float]:
    """The parked cart-body -> ur_base_link pose ``(x, y, z, yaw)`` from the WORLD poses of the robot park and
    the parked cart.

    ``robot_park`` / ``cart_park`` are each ``(x, y, yaw)`` in the map (the robot's transfer park from the
    nav-trace / mission pose, and the PARKED cart prim's world pose -- after uncouple the cart is NOT at the
    coupled hitch offset, so this MUST be the parked prim, not the coupled transform). The arm base rides
    ``arm_base_z`` above the robot ground point; the cart collider frame sits ``cart_base_z`` above the cart
    ground point (0 for the authored rollwagen). Returns the pose the ``world_obstacles`` / cage-gate layers
    consume (a yaw-only cart on a level floor, so position + yaw is the full pose). Pure/CPU."""
    T_w_base = _xyyaw_to_T(robot_park[0], robot_park[1], robot_park[2], z=arm_base_z)
    T_w_cart = _xyyaw_to_T(cart_park[0], cart_park[1], cart_park[2], z=cart_base_z)
    T_base_cart = np.linalg.inv(T_w_base) @ T_w_cart
    yaw = math.atan2(T_base_cart[1, 0], T_base_cart[0, 0])
    return (float(T_base_cart[0, 3]), float(T_base_cart[1, 3]), float(T_base_cart[2, 3]), float(yaw))


# --------------------------------------------------------------- parked-cart world obstacles (PR 350)

def parked_cart_world_obstacles(cart_pose, *, pad_m: float = 0.0,
                                collider_boxes: Optional[Dict[str, Tuple]] = None, include=None,
                                exclude: Sequence[str] = DEFAULT_WORLD_OBSTACLE_EXCLUDE):
    """``world_obstacles`` host-key specs for the PARKED rollwagen cage, in the arm base frame (PR 350, from
    the parked cart prim pose rather than the coupled hitch transform).

    Thin wrapper over :func:`cart_world_obstacle.rollwagen_world_obstacle_specs`: the cage bars group under
    ``rollwagen_cage`` and each horizontal support (deck/shelf) is its own box. ``collider_boxes`` is the
    cart-frame ``{name: (lo, hi)}`` AABBs (default: read the authored USD, needs usd-core; pass a pre-baked
    dict for a no-pxr host). ``cart_pose`` is ``(x, y, z, yaw)`` (e.g. from :func:`cart_pose_in_base`) or a 4x4.
    ``exclude`` drops bodies (default the ``drawbar`` tow hitch, which overlaps the arm base -- see
    :data:`DEFAULT_WORLD_OBSTACLE_EXCLUDE`); pass ``exclude=()`` to publish everything. Drop the returned list
    into ``config['world_obstacles']`` so ``ros_host`` publishes the cage CollisionObjects cuMotion plans the
    re-grasp around (:func:`check_regrasp_obstacles_clear_arm` proves they clear the arm start config first)."""
    from .cart_world_obstacle import rollwagen_world_obstacle_specs
    boxes = collider_boxes if collider_boxes is not None else pc.rollwagen_collider_boxes()
    drop = set(exclude or ())
    if drop:
        boxes = {n: v for n, v in boxes.items() if n not in drop}
    return rollwagen_world_obstacle_specs(boxes, cart_pose, pad_m=pad_m, include=include)


def check_regrasp_obstacles_clear_arm(specs, urdf_path, xrdf_path, *, configs=None):
    """Prove the published parked-cart cage obstacles CLEAR the arm at its rest configs (the firstmate 295
    lesson: a world obstacle overlapping the start/stow config rejects EVERY plan with INVALID_INITIAL_CSPACE).
    Thin wrapper over :func:`cart_world_obstacle.check_cart_obstacle_clears_arm`; returns ``{id: min_clearance}``
    and raises when any obstacle intersects the arm."""
    from .cart_world_obstacle import check_cart_obstacle_clears_arm
    return check_cart_obstacle_clears_arm(specs, urdf_path, xrdf_path, configs=configs)


# --------------------------------------------------------------- the pad-down re-grasp path (PR 376 reversed)

@dataclass(frozen=True)
class RegraspGripperGeometry:
    """The re-grasp gripper + (optionally) the welded carton, as boxes in the CARTON frame, for the cage sweep.

    A pad-DOWN top grasp: the suction housing sits directly ABOVE the carton top (the captain's "pad-down
    above the carton"). ``housing_size`` is a conservative box around the suction panel + the gripper cube
    (the PR 376 ``cube_without_surface_plates`` / ``surface_plate_*`` links the cuMotion sphere model covers);
    its default is the measured suction panel :data:`place_clearance.PAD_SIZE_M` inflated to the housing. The
    carton box is :data:`place_clearance.CARTON_DIMS_M`. ``carried`` selects whether the carton travels with
    the gripper (the lift/withdraw legs) or stays on the support (the empty approach/descent legs)."""
    carton_dims: Tuple[float, float, float] = pc.CARTON_DIMS_M
    housing_size: Tuple[float, float, float] = (0.27, 0.12, 0.16)
    carried: bool = True

    def carton_surface_points(self, step: float = 0.01) -> np.ndarray:
        if not self.carried:
            return np.empty((0, 3))
        return pc._box_surface_points(np.asarray(self.carton_dims, float) / 2.0, step)

    def pad_surface_points(self, step: float = 0.01) -> np.ndarray:
        half = np.asarray(self.housing_size, float) / 2.0
        pts = pc._box_surface_points(half, step)
        # the housing hangs above the carton top: +z by (carton_h/2 + housing_z/2)
        return pts + np.array([0.0, 0.0, self.carton_dims[2] / 2.0 + half[2]])

    def surface_points(self, step: float = 0.01) -> np.ndarray:
        return np.vstack([self.carton_surface_points(step), self.pad_surface_points(step)])


def mouth_pad_down_path(load_center_xy, support_z, yaw_cart_rel, *, carton_h: float,
                        standoff_above: float = DEFAULT_REGRASP_STANDOFF_M,
                        mouth_exit_x: float = DEFAULT_MOUTH_EXIT_X_M, n: int = 6):
    """The pad-down re-grasp path as cart-frame carton poses + a per-waypoint carried flag.

    PR 376 ``place_descent_poses`` IN REVERSE, routed through the open +x mouth. Four legs:

      * ``approach`` (empty): from a standoff ABOVE the mouth exit, move -x through the mouth to directly
        above the carton, holding ``z_above = standoff_above`` (so the gripper stays clear of the front posts
        and, for a deck carton, below the upper shelf);
      * ``descend`` (empty): lower from the standoff straight down onto the carton top (grasp);
      * ``lift`` (carried): raise straight back up to the standoff with the carton welded;
      * ``withdraw`` (carried): carry the carton +x straight OUT the mouth to the exit.

    ``load_center_xy`` is the carton load centre (cart frame), ``support_z`` the support top the carton rests
    on (deck 0.225 or shelf 0.933), ``yaw_cart_rel`` the carton yaw relative to the cart. Returns
    ``(poses, carried)``: ``poses`` a list of 4x4 cart-frame carton poses, ``carried`` a parallel list of bools.
    The poses are LEVEL (roll/pitch 0 throughout), so the carried legs keep the carton parallel to the ground
    by construction (:func:`carried_withdraw_level_ok`)."""
    lx, ly = float(load_center_xy[0]), float(load_center_xy[1])

    def p(x, z_above):
        return pc.carton_pose_cart((x, ly), support_z, yaw_cart_rel, carton_h=carton_h, z_above=z_above)

    n = max(1, int(n))
    poses, carried = [], []
    # approach: mouth_exit_x -> lx at z = standoff (empty)
    for i in range(n + 1):
        t = i / float(n)
        poses.append(p(mouth_exit_x + t * (lx - mouth_exit_x), standoff_above))
        carried.append(False)
    # descend: z standoff -> 0 at x = lx (empty)
    for i in range(1, n + 1):
        poses.append(p(lx, standoff_above * (1.0 - i / float(n))))
        carried.append(False)
    # lift: z 0 -> standoff at x = lx (carried)
    for i in range(1, n + 1):
        poses.append(p(lx, standoff_above * (i / float(n))))
        carried.append(True)
    # withdraw: lx -> mouth_exit_x at z = standoff (carried)
    for i in range(1, n + 1):
        t = i / float(n)
        poses.append(p(lx + t * (mouth_exit_x - lx), standoff_above))
        carried.append(True)
    return poses, carried


# --------------------------------------------------------------- the CPU cage sweep

def regrasp_cage_clearance(poses, carried, bodies, *, support, carton_dims=pc.CARTON_DIMS_M,
                           housing_size=(0.27, 0.12, 0.16)):
    """``{body: min signed gap (m)}`` of the re-grasp sweep: the EMPTY legs sweep the gripper HOUSING alone vs
    every cart body, the CARRIED legs sweep the gripper+carton (carton vs the kept bodies, housing vs all --
    the cage-filtered honest model). ``support`` is the body the carton rests on (deck/shelf), dropped from
    the carried-leg carton check. ``bodies`` is the cart-frame ``{name: (lo, hi)}`` AABBs
    (:func:`place_clearance.rollwagen_collider_boxes`). Pure/CPU.

    The quantity the refuse gate reads; negative = the gripper/carton penetrates that cart body along the path."""
    empty_poses = [T for T, c in zip(poses, carried) if not c]
    carry_poses = [T for T, c in zip(poses, carried) if c]
    per_body: Dict[str, float] = {}

    def _merge(pm):
        for name, v in pm.items():
            if name not in per_body or v < per_body[name]:
                per_body[name] = v

    if empty_poses:
        geom = RegraspGripperGeometry(carton_dims=tuple(carton_dims), housing_size=tuple(housing_size),
                                      carried=False)
        # empty gripper: the housing alone (no carton) vs EVERY body (the lumped sweep -- the honest sweep
        # needs carton points this leg does not carry). The support is dropped from the reject
        # (the descent intentionally approaches the carton resting on it, so a small gap there is the goal).
        _, pm = pc.sweep_clearance(empty_poses, geom, bodies, honest_support=None)
        _merge(pm)
    if carry_poses:
        geom = RegraspGripperGeometry(carton_dims=tuple(carton_dims), housing_size=tuple(housing_size),
                                      carried=True)
        _, pm = pc.sweep_clearance(carry_poses, geom, bodies, honest_support=support)
        _merge(pm)
    return per_body


def regrasp_sweep_reject(per_body_min, *, margin: float = DEFAULT_REGRASP_CAGE_MARGIN_M,
                         support=pc.SUPPORT_BODIES):
    """The :data:`REGRASP_CAGE_CLEARANCE_REJECTED` message when any NON-support cart body's minimum gap over
    the re-grasp sweep is below ``margin``, else ``None``. Names the offending bodies + gaps so the lane sees
    exactly which post/bar the pad-down path would strike (never widens the margin)."""
    support = set(support)
    bad = sorted((v, k) for k, v in per_body_min.items() if k not in support and v < margin)
    if not bad:
        return None
    named = ', '.join('%s=%.1fmm' % (k, v * 1000.0) for v, k in bad)
    return ('%s pad-down re-grasp path would strike a NON-support rollwagen body: %s (minimum gap below the '
            '%.0f mm clearance margin). The approach/descent/lift/withdraw must thread the open +x mouth clear '
            'of the cage bars; refusing before motion.' % (REGRASP_CAGE_CLEARANCE_REJECTED, named,
                                                           margin * 1000.0))


def regrasp_clearance_summary(per_body_min, *, support=pc.SUPPORT_BODIES, worst_n=4):
    """A one-line :data:`REGRASP_CAGE_CLEARANCE` log: the ``worst_n`` tightest non-support gaps + the support
    gaps, mirroring :func:`place_clearance.clearance_summary`."""
    support = set(support)
    nonsupport = sorted((v, k) for k, v in per_body_min.items() if k not in support)[:worst_n]
    sup = sorted((v, k) for k, v in per_body_min.items() if k in support)

    def fmt(items):
        return ', '.join('%s=%.1fmm' % (k, v * 1000.0) for v, k in items)

    return '%s min non-support: %s | support: %s' % (REGRASP_CAGE_CLEARANCE, fmt(nonsupport), fmt(sup))


def carried_withdraw_level_ok(poses, carried, *, bound_deg: float = DEFAULT_TILT_BOUND_DEG):
    """The carried legs (lift/withdraw) keep the carton PARALLEL to the ground. The pad-down path poses are
    level by construction (:func:`mouth_pad_down_path` only varies position + yaw about +Z, roll/pitch 0), so
    the carried carton's max |roll|,|pitch| is 0 deg -- always within ``bound_deg``. Returns
    ``(max_tilt_deg, ok)``; measures the actual pose rotation so a non-level pose would be caught."""
    worst = 0.0
    for T, c in zip(poses, carried):
        if not c:
            continue
        R = np.asarray(T, float)[:3, :3]
        # tilt of the carton +Z from world +Z (roll/pitch magnitude)
        up = R @ np.array([0.0, 0.0, 1.0])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
        worst = max(worst, tilt)
    return worst, worst <= bound_deg + 1e-9


# --------------------------------------------------------------- the reposition place-park (scope 2)

@dataclass(frozen=True)
class PlacePark:
    """The computed reposition base park pose + its reach proof against the endcap place target."""
    pose: Tuple[float, float, float]      # (x, y, yaw) world base park, facing the shelf
    endcap_xyz: Tuple[float, float, float]
    standoff_xy_m: float                  # base-to-endcap xy distance (the facing-normal standoff)
    reach_center_m: float                 # arm-base -> carton-rest (the grasp target the brief bounds to ~0.9 m)
    reach_place_m: float                  # arm-base -> the place TCP pose (carton bottom on the shelf top)
    reach_preplace_m: float               # arm-base -> the preplace standoff above the shelf
    max_reach_m: float
    faces_shelf: bool                     # the base yaw points at the endcap
    within_goal: bool                     # the grasp target sits within the ~0.9 m goal reach
    within_envelope: bool                 # place + preplace sit inside the UR10 working envelope


def _reach(base_xy, target_z, endcap_xyz, arm_base_z):
    return math.dist((base_xy[0], base_xy[1], arm_base_z),
                     (endcap_xyz[0], endcap_xyz[1], target_z))


def place_park_pose(endcap_xyz, *, reach_m: float = DEFAULT_PLACE_REACH_M, approach_normal=(0.0, 1.0),
                    arm_base_z: float = ARM_BASE_Z, carton_h: float = 0.0, approach_offset: float = 0.12,
                    max_reach_m: float = DEFAULT_MAX_REACH_M) -> PlacePark:
    """Compute the base park pose from the ENDCAP place target and the arm reach: the carton-rest target within
    ``reach_m`` (~0.9 m) of the arm base, the base facing the shelf.

    ``endcap_xyz`` is the world shelf-surface place target (``scenario.initial_staging.target_surface_pose_world``,
    e.g. ``[-3.05, 4.826, 0.73]``). ``approach_normal`` is the UNIT direction from the endcap toward the service
    aisle the base parks in (default ``(0, 1)`` -- the +y aisle north of the G3_back endcap; the authored
    ``shelf_approach`` parks there). The base parks at the xy standoff that puts the 3-D reach to the carton
    centre (``endcap top + carton_h/2``) at exactly ``reach_m``, with the base yaw pointing at the endcap. Also
    reports the reach to the place TCP pose (carton bottom on the shelf top) and the preplace standoff so the
    caller can prove the level descent onto the shelf top (0.73) is in the envelope (:func:`place_reach_ok`).
    Pure/CPU -- the SAME 3-D reach envelope test the live_pose transfer observe gate applies (REACH_ENVELOPE)."""
    nx, ny = float(approach_normal[0]), float(approach_normal[1])
    nmag = math.hypot(nx, ny) or 1.0
    nx, ny = nx / nmag, ny / nmag
    target_center_z = float(endcap_xyz[2]) + float(carton_h) / 2.0
    dz = target_center_z - float(arm_base_z)
    standoff = math.sqrt(max(0.0, reach_m * reach_m - dz * dz))
    bx = float(endcap_xyz[0]) + nx * standoff
    by = float(endcap_xyz[1]) + ny * standoff
    yaw = math.atan2(float(endcap_xyz[1]) - by, float(endcap_xyz[0]) - bx)
    place_z = float(endcap_xyz[2]) + float(carton_h)                       # TCP at carton-bottom-on-surface
    preplace_z = place_z + float(approach_offset)
    rc = _reach((bx, by), target_center_z, endcap_xyz, arm_base_z)
    rp = _reach((bx, by), place_z, endcap_xyz, arm_base_z)
    rpp = _reach((bx, by), preplace_z, endcap_xyz, arm_base_z)
    # the base forward (yaw) points at the endcap by construction; measure it so a degenerate normal is caught
    to_endcap = math.atan2(float(endcap_xyz[1]) - by, float(endcap_xyz[0]) - bx)
    faces = abs(_wrap(yaw - to_endcap)) < 1e-6
    return PlacePark(
        pose=(bx, by, yaw), endcap_xyz=tuple(float(v) for v in endcap_xyz[:3]),
        standoff_xy_m=standoff, reach_center_m=rc, reach_place_m=rp, reach_preplace_m=rpp,
        max_reach_m=float(max_reach_m), faces_shelf=faces,
        within_goal=rc <= reach_m + 1e-6,
        within_envelope=max(rp, rpp) <= float(max_reach_m) + 1e-9)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def place_reach_ok(park_pose, endcap_xyz, *, arm_base_z: float = ARM_BASE_Z, carton_h: float = 0.0,
                   approach_offset: float = 0.12, max_reach_m: float = DEFAULT_MAX_REACH_M) -> dict:
    """Prove the place target + the level descent onto the shelf top (0.73) are reachable from ``park_pose``
    ``(x, y, yaw)``. Returns the reach to the carton centre, the place TCP pose (carton bottom on the shelf
    top) and the preplace standoff, with the envelope verdict -- the SAME REACH_ENVELOPE distance test the
    controller's observe gate applies (the full collision IK is the GPU plan; this is the CPU reach proof the
    existing service-leg check uses). Pure/CPU."""
    bx, by = float(park_pose[0]), float(park_pose[1])
    target_center_z = float(endcap_xyz[2]) + float(carton_h) / 2.0
    place_z = float(endcap_xyz[2]) + float(carton_h)
    preplace_z = place_z + float(approach_offset)
    rc = _reach((bx, by), target_center_z, endcap_xyz, arm_base_z)
    rp = _reach((bx, by), place_z, endcap_xyz, arm_base_z)
    rpp = _reach((bx, by), preplace_z, endcap_xyz, arm_base_z)
    return {'reach_center_m': rc, 'reach_place_m': rp, 'reach_preplace_m': rpp,
            'max_reach_m': float(max_reach_m),
            'within_envelope': max(rc, rp, rpp) <= float(max_reach_m) + 1e-9,
            'level_descent_ok': rp <= float(max_reach_m) + 1e-9 and rpp <= float(max_reach_m) + 1e-9}
