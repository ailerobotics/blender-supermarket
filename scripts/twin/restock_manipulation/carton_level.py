"""Keep the carried carton PARALLEL TO THE GROUND (captain's standing rule, 2026-11-03).

The handling unit is welded rigidly to the gripper by the SIM_ATTACH_FALLBACK fixed joint
for the whole carry (collect lift -> place-on-cart; transfer lift -> shelf place), so the
carton's attitude is the tool's attitude: a level tool means a level carton. In coldwin16
the carried carton swung well off level during the fold (data/sim-carton-level-carry-v1),
which the captain saw in the whole-run video.

This module is the PLANNER/EXECUTOR-LEVEL fix, not a post-hoc correction. It is pure
(numpy + the Isaac-free ``twin.urdf_probe`` FK) so the whole thing is CPU unit-testable and
the arm-carry bench can assert ``max |roll|,|pitch|`` under a bound for every carried
segment before any GPU boot. Three pieces:

  * MEASURE -- decompose a carton/tool orientation (a world quaternion, a rotation matrix, or
    the logged gripper approach axis) into roll / pitch / total tilt from level.
  * LEVEL -- re-solve the wrist joints of a carried joint goal so the carton's up-axis points
    straight up (a level-orientation waypoint), and interpolate a move as a sequence of such
    leveled waypoints with a level-hold at both ends (before lift-off, before set-down). The
    carton's up-axis in the tool frame is FIXED at the weld (the carton was level on the
    shelf), so holding that axis vertical holds the carton level.
  * TRIP -- a stated bound and a named verdict so a carried move whose measured tilt exceeds
    the bound is reported as ``CARTON_TILT`` in the log and FAILS rather than silently
    continuing (the captain's rule: never show a visually wrong moment).

Everything here is opt-in: ``resolve_level_config`` returns None for an absent/false config,
and every caller leaves the mission byte-identical when it is off.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

# The log token the executor prints when a carried carton leaves level by more than the bound.
CARTON_TILT = 'CARTON_TILT'
# The log token for a SUB-GRACE tilt spike that is reported but does NOT fail the move
# (sim-carton-tilt-transient-debounce-v1): a one-to-few-tick tracking/impulse excursion.
CARTON_TILT_TRANSIENT = 'CARTON_TILT_TRANSIENT'
# The log token the planner prints when a carried move's PREDICTED executed path (replayed before
# dispatch) would leave level by more than the bound (sim-carry-level-lift-v1). The caller FAILS the
# move by name BEFORE executing it, rather than streaming a visibly tilted carry -- the captain's rule
# that a visually wrong moment never reaches the camera. This is the PLAN-time counterpart of the
# execution-time CARTON_TILT trip: CARTON_TILT catches a tilt the live arm actually reached;
# CARTON_LEVEL_PLAN_REJECTED catches one the plan WOULD reach, so the arm never starts the move.
CARTON_LEVEL_PLAN_REJECTED = 'CARTON_LEVEL_PLAN_REJECTED'
# Default bound (deg) on |roll| and |pitch| while carrying -- the captain's "parallel to the
# ground"; 2 deg is the brief's stated threshold for naming a move off-level.
DEFAULT_TILT_BOUND_DEG = 2.0
# Default SUSTAINED-tilt grace window (sim-seconds AT 1/60 physics step). A carried-carton tilt must
# exceed the bound CONTINUOUSLY for this long before the CARTON_TILT trip FAILS the move; a shorter
# spike logs CARTON_TILT_TRANSIENT (peak + duration) and proceeds. Why 0.15 s / 9 physics ticks at
# 1/60 (sim-carton-tilt-transient-debounce-v1):
#   * the observed transient is one-to-few ticks -- collectacc1 tripped at the LIFT re-plan restart
#     (a move_group SIGABRT respawn + carton re-attach + plan re-delivery, then the arm re-accelerated
#     from the settled hold along the re-sent trajectory). The joint->carton-tilt sensitivity is ~1:1
#     (CPU-measured at carry_lift_shape), so a ~2 deg wrist/elbow tracking excursion momentarily crosses
#     the 2.00 deg bound; the CPU-reconstructed leveled lift PATH is level to <0.01 deg throughout, so
#     the 2.01 deg was an EXECUTION transient, not a planned-pose tilt;
#   * 9 ticks covers "a few ticks" with margin while staying under the 1/3 s walkthrough capture
#     interval, so a debounced spike cannot yield a visibly-tilted RECORDED frame (the captain's
#     "nothing visually wrong on camera"); and
#   * 0.15 s is ~2 orders of magnitude below the multi-second carry, so a GENUINELY off-level carton
#     (which stays tilted the whole carry) still trips within 9 ticks -- effectively immediately at
#     mission scale. The window is a FRAME-COUNT budget: scale it onto a raised physics step with
#     twin.restock.physics_step.freshness_scale (identical frame margin, re-expressed on the running
#     clock). 0.0 => legacy IMMEDIATE trip on the first crossing tick (byte-identical to pre-debounce).
DEFAULT_TILT_GRACE_S = 0.15
# The arm joints (ARM order) a level-orientation waypoint is allowed to re-solve. The wrist
# cluster sets the tool attitude; pan/lift/elbow carry the position the carry plan chose.
DEFAULT_LEVEL_JOINTS = ('ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint')
# Default per-waypoint wrist-step continuity threshold (rad) for the densified pre-place approach
# (sim-preplace-wrist-continuity-v1). Each leg-2 waypoint re-solves the wrist cluster for level-in-world;
# the re-solve must be BRANCH-CONTINUOUS -- the per-joint wrist step from the previous waypoint must stay
# at or below this. coldwin57 reached 97 % of the approach at 0.00 deg world tilt, then the per-waypoint
# level solve jumped the commanded wrist_2 from ~1.63 to ~0.404 rad (a 1.3 rad step between consecutive
# waypoints to a FAR but also-level branch) because ``level_config`` ranked the lowest-tilt branch with no
# travel penalty; the soft wrist (eff 400, speed cap 0.05 rad/s) could not follow the step, so the
# trajectory host tripped its tracking guard (TRAJ_HOST_TRACK_MISS worst_j=4 ref 1.63 vs meas 0.33). At the
# active densification (``preplace_max_segment_rad`` default 0.25 rad per position step) a continuous
# level branch moves each wrist well under 0.15 rad per waypoint, so 0.15 passes a healthy approach and
# refuses a branch jump before motion.
DEFAULT_MAX_WRIST_STEP_RAD = 0.15
# sim-preplace-approach-seeded-v1: bounds on the seeded interior solve's LOCAL densification. A segment that
# cannot meet both the step clamp and the level gate is bisected; ``_SEEDED_MAX_DEPTH`` caps the bisection
# depth per base segment (2^6 = 64 sub-steps is far finer than any real level-manifold feature) and
# ``_SEEDED_MAX_WAYPOINTS`` caps the total waypoint count so a genuine branch gap refuses promptly (named)
# instead of densifying unboundedly. Both are generous safety nets, never reached on a healthy approach.
_SEEDED_MAX_DEPTH = 6
_SEEDED_MAX_WAYPOINTS = 600
# sim-preplace-pan-first-seeded-leg2-v1: the most carry-connected (least carry-travel) standoff branches
# whose leg-2 continuity gate is validated before giving up. The carry-connected branch ranks near the
# front, so the healthy path validates ~1; the cap bounds a pathological input from running the expensive
# seeded leg-2 plan over every far (heavily densified) branch and blowing the CPU search budget.
_CARRY_CONNECTED_MAX_VALIDATE = 8
WORLD_UP = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------- held chassis attitude (sim-preplace-leg2-level-v1)
# The carton-level FK is rooted at ``ur_base_link``, which the whole module treated as world-aligned. During
# the collect place the base is NOT level: the station parking brake pins the chassis at a held attitude
# (``chassis_station_pitch_hold_pitch_deg`` / ``_roll_deg``, default pitch -1.17 deg), so ``ur_base_link`` +Z
# is tilted from world +Z and the shoulder_pan YAW axis is off vertical -- a pan sweep then carries the welded
# carton on a cone and the WORLD tilt is NOT pan-invariant even though the base-frame tilt is (coldwin56
# leg 2: the executed held-wrist carton rose 1.64 -> 2.02 across the pan descent). ``base_R`` is the 3x3
# world<-base rotation of the held chassis; when supplied, every leveling residual and tilt measurement is
# taken in the WORLD frame (``base_R @ up_world_base`` vs world +Z) so "parallel to the ground" is the
# captain's world rule, not the base frame. Default ``None`` => identity => byte-identical (the historical
# base-frame behaviour).

def base_rotation_rp(roll_deg, pitch_deg):
    """The 3x3 world<-base rotation of a chassis held at ``roll_deg`` / ``pitch_deg`` (yaw is irrelevant to
    tilt-from-vertical, so it is omitted): ``Ry(pitch) @ Rx(roll)``. Pure/CPU. Returns ``None`` for a level
    chassis (both ~0) so callers stay byte-identical."""
    r = math.radians(float(roll_deg)); p = math.radians(float(pitch_deg))
    if abs(r) < 1e-12 and abs(p) < 1e-12:
        return None
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(r), -math.sin(r)], [0.0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0], [-math.sin(p), 0.0, math.cos(p)]])
    return Ry @ Rx


def _apply_base(base_R, up_base):
    """``up_base`` expressed in world: ``base_R @ up_base`` (or ``up_base`` verbatim when ``base_R`` is
    None/identity)."""
    if base_R is None:
        return up_base
    return np.asarray(base_R, float) @ np.asarray(up_base, float)


# --------------------------------------------------------------------- decompose

def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError('cannot normalise a zero-length vector')
    return v / n


def _wrap(a):
    """Wrap an angle (rad) to (-pi, pi] so a +/-2*pi joint branch is not counted as a large
    displacement when ranking level solutions by nearness to the reach shape."""
    from twin.restock_manipulation.transfer_carry import wrap_angle
    return wrap_angle(float(a))


def up_axis_rpy(up):
    """(roll_deg, pitch_deg, tilt_deg) of a carton whose world up-axis is ``up`` (3-vector).

    ``tilt`` is the unambiguous angle of the up-axis from world +Z. ``pitch`` is the lean along
    world +X (nose up/down), ``roll`` the lean along world +Y; both are yaw-free (a box that only
    yaws stays level), signed, and match ``twin.restock.isaac_port.body_roll_pitch_deg`` for small
    attitudes (validated in the tests). A perfectly level carton reads (0, 0, 0)."""
    ux, uy, uz = _unit(up)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, uz))))
    pitch = math.degrees(math.atan2(ux, uz))
    roll = math.degrees(math.atan2(-uy, uz))
    return roll, pitch, tilt


def rotation_up(R):
    """The up-axis (local +Z column) of a 3x3 rotation matrix ``R`` in world."""
    R = np.asarray(R, float)
    if R.shape != (3, 3):
        raise ValueError('rotation matrix must be 3x3')
    return R[:, 2].copy()


def quat_rpy(wxyz):
    """(roll_deg, pitch_deg, tilt_deg) from a world quaternion ``wxyz`` (Isaac order w,x,y,z),
    treating the body's local +Z as its up-axis. The roll/pitch sign convention matches
    ``up_axis_rpy``; use this for the live carton body read."""
    w, x, y, z = (float(v) for v in wxyz)
    up = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return up_axis_rpy(up)


def forward_down_tilt(forward):
    """Angle (deg) of the gripper approach axis ``forward`` (SEAL_DIAG ``forward_world``) from
    world-down -- the exact ``tilt_deg_from_down`` the seal reader logs. ~0 for a top-down grasp,
    ~90 for a level side grasp. The measurement notes why this alone is not the carton tilt."""
    f = _unit(forward)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(f, [0.0, 0.0, -1.0]))))))


def angle_between_deg(a, b):
    """Angle (deg) between two 3-vectors -- e.g. the gripper approach axis now vs at the weld."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(_unit(a), _unit(b)))))))


# -------------------------------------------------------------------- FK + attitude

def tool_rotation(robot, config, tool_link='ur_tool0', root='ur_base_link'):
    """World 3x3 rotation of ``tool_link`` at arm joint ``config`` (a dict joint_name->rad, or a
    6-vector in ARM order), via the Isaac-free URDF FK. Pure/CPU."""
    return tool_transform(robot, config, tool_link, root)[:3, :3]


def tool_transform(robot, config, tool_link='ur_tool0', root='ur_base_link'):
    """World 4x4 transform of ``tool_link`` at arm joint ``config`` (dict or 6-vector in ARM order),
    via the Isaac-free URDF FK. Pure/CPU. ``tool_rotation`` / ``tool_position`` are views of this."""
    from twin.urdf_probe import link_world_transforms
    cfg = _as_joint_map(config)
    world = link_world_transforms(robot, cfg, root=root)
    if tool_link not in world:
        raise ValueError(f'tool link {tool_link!r} not in the URDF FK ({len(world)} links)')
    return np.asarray(world[tool_link], float)


def tool_position(robot, config, tool_link='ur_tool0', root='ur_base_link'):
    """World position (3-vector) of ``tool_link`` origin at arm joint ``config``. Pure/CPU."""
    return tool_transform(robot, config, tool_link, root)[:3, 3].copy()


# The log token the controller prints when the welded carton reaches the cart deck during the collect
# set-down and the contact-triggered weld release fires (sim-place-descent-residual-v3).
PLACE_CONTACT_RELEASE = 'PLACE_CONTACT_RELEASE'


# --------------------------------------------- cached tool-chain FK (sim-preplace-select-live-cost-v1)
# The reaction-branch standoff selector ran ~25-28 s on the real coldwin60 inputs (coldwin60-select-cost),
# NOT the ~0.08 s the coldwin57 fixture measured -- the fixture's standoff target IS its seed's own TCP, so
# level_place_ik's first seed reaches it in zero iterations, while the real target (the place load-centre,
# ~0.5 m off the seed TCP) forces the full damped-least-squares descent. Profiling: 8 branches x a ~31-seed
# multi-start, up-to-200-iteration DLS, each iteration a residual FK + a 6-column finite-difference Jacobian
# (7 FK/iter) = ~61 000 FK calls, each ~0.39 ms through ``link_world_transforms`` -- which rebuilds the
# WHOLE 36-link mobile robot tree (base + cart + arm) though only ``tool_link`` is needed.
#
# This cached chain evaluates ONLY the root->tool_link kinematic chain, with each joint's constant origin
# transform precomputed once, so a single-joint FINITE-DIFFERENCE perturbation (the Jacobian columns)
# recomputes only that joint and two cached prefix/suffix products. The arithmetic is the SAME
# finite-difference Jacobian the slow path computes (same ``step_fd``, same residual, same matmul order up
# to floating-point associativity), so a ``fast=True`` solve lands within ~1e-4 rad of the slow solve
# (asserted in test_preplace_branch_select_bounded_v1) -- not a different method, just the same math without
# the whole-tree FK. Scoped to the opt-in ``fast=`` selector path; every other caller is byte-identical.
_CHAIN_CACHE = {}


def _axis_rot4(axis, q):
    """4x4 homogeneous rotation of ``q`` rad about the unit ``axis`` (Rodrigues). Matches
    ``urdf_probe.joint_transform``'s axis block to < 1e-12."""
    c = math.cos(q); s = math.sin(q); C = 1.0 - c
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s, 0.0],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s, 0.0],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


class _KinChain:
    """The ordered root->tool_link joint chain with constant per-joint origin transforms precomputed, for
    the selector's many FK evaluations. ``tool(cfg)`` returns the same 4x4 as ``tool_transform`` (same
    matmul order); ``perturbed_tool`` gives a single-joint finite-difference neighbour cheaply via the
    cached prefix/suffix products. Revolute/continuous joints only (the UR arm chain); a prismatic or an
    unreachable tool raises so the caller falls back to the slow whole-tree FK."""

    def __init__(self, robot, tool_link, root):
        from twin.urdf_probe import rpy_to_matrix
        by_child = {j.child: j for j in robot.joints}
        chain = []
        cur = tool_link
        while cur != root and cur in by_child:
            j = by_child[cur]
            chain.append(j)
            cur = j.parent
        if not chain or cur != root:
            raise KeyError('tool link %r not reachable from %r via the joint tree' % (tool_link, root))
        chain.reverse()
        self.joints = chain
        self.names = [j.name for j in chain]
        self.index = {j.name: k for k, j in enumerate(chain)}
        self.origin = []
        self.axis = []
        self.movable = []
        for j in chain:
            if j.jtype == 'prismatic':
                raise KeyError('prismatic joint %r in the tool chain is not supported by the fast FK'
                               % j.name)
            T = np.eye(4)
            T[:3, :3] = rpy_to_matrix(j.origin_rpy)
            T[:3, 3] = j.origin_xyz
            self.origin.append(T)
            a = np.asarray(j.axis, float)
            n = float(np.linalg.norm(a))
            self.axis.append(a / n if n else a)
            self.movable.append(j.jtype in ('revolute', 'continuous'))
        self.n = len(chain)

    def _mat(self, k, q):
        if not self.movable[k]:
            return self.origin[k]
        return self.origin[k] @ _axis_rot4(self.axis[k], q)

    def evaluate(self, cfg):
        """Return ``(T_tool, prefix, suffix)`` at ``cfg`` (a joint map). ``prefix[k]`` is the product of
        joint matrices 0..k-1, ``suffix[k]`` of k..n-1, so ``prefix[k] @ _mat(k, q') @ suffix[k+1]`` is the
        tool transform with only joint ``k`` changed to ``q'``."""
        mats = [self._mat(k, float(cfg.get(self.names[k], 0.0))) for k in range(self.n)]
        prefix = [np.eye(4)] * (self.n + 1)
        for k in range(self.n):
            prefix[k + 1] = prefix[k] @ mats[k]
        suffix = [np.eye(4)] * (self.n + 1)
        for k in range(self.n - 1, -1, -1):
            suffix[k] = mats[k] @ suffix[k + 1]
        return prefix[self.n], prefix, suffix

    def perturbed_tool(self, prefix, suffix, jname, q_new):
        k = self.index[jname]
        return prefix[k] @ self._mat(k, q_new) @ suffix[k + 1]


def _chain_for(robot, tool_link, root):
    """A cached :class:`_KinChain` for ``(robot, tool_link, root)`` or ``None`` when the chain cannot be
    built by the fast FK (caller then uses the slow whole-tree FK, byte-identical)."""
    key = (id(robot), tool_link, root)
    hit = _CHAIN_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        chain = _KinChain(robot, tool_link, root)
    except KeyError:
        chain = None
    _CHAIN_CACHE[key] = chain
    return chain


def place_descent_drop_m(robot, waypoints, *, tool_link='ur_tool0', root='ur_base_link'):
    """The vertical TCP drop (m, world +Z) of a planned place descent -- the height the welded carton
    centre falls from the leveled standoff (``waypoints[0]``) to the set-down (``waypoints[-1]``). Both
    descent legs are leveled, so the weld offset is constant and the carton-centre z tracks the tool z with
    a fixed offset: the carton falls by exactly this TCP drop. Pure/CPU. Returns max(0, z0 - z1)."""
    if waypoints is None or len(waypoints) < 2:
        raise ValueError('place_descent_drop_m needs at least two waypoints (standoff + set-down)')
    z0 = float(tool_position(robot, list(waypoints[0]), tool_link, root)[2])
    z1 = float(tool_position(robot, list(waypoints[-1]), tool_link, root)[2])
    return max(0.0, z0 - z1)


def place_contact_release_z(start_carton_z, drop_m, gap_m):
    """The carton-BOTTOM-frame contact-release height: ``start_carton_z - drop_m + gap_m``. ``start_carton_z``
    is the live carton centre z when the place descent is issued (carton at the standoff), ``drop_m`` the
    planned descent drop (:func:`place_descent_drop_m`) EXPRESSED AS A DROP TO THE CARTON-BOTTOM-AT-SUPPORT
    set-down, and ``gap_m`` the carton-bottom-to-support gap at which to release. Pure/CPU.

    WARNING (sim-release-threshold-carton-frame-v1): the production place descent waypoints target the carton
    REFERENCE (centre) at the support top rather than the carton BOTTOM at the support top, so the measured
    ``place_descent_drop_m`` drop is one carton HALF-height larger than a drop-to-bottom-at-support, and this
    value then lands one half-height BELOW the measured carton CENTRE (``Sensors.placed_pose[2]``). The
    controller must compare the measured carton CENTRE against :func:`place_contact_release_z_centre`, which
    adds the carton half-height so the trigger fires at true contact. This bottom-frame helper is retained for
    the deck set-down geometry tests (``test_place_support_z`` passes a drop already reduced to the
    carton-bottom-at-support set-down, where no half-height correction is wanted)."""
    if not all(math.isfinite(v) for v in (start_carton_z, drop_m, gap_m)):
        raise ValueError('place_contact_release_z needs finite start_carton_z, drop_m, gap_m')
    return float(start_carton_z) - float(drop_m) + float(gap_m)


def place_contact_release_z_centre(start_carton_z, drop_m, gap_m, carton_height_m):
    """sim-release-threshold-carton-frame-v1: the carton-CENTRE z (ur_base_link) at which the contact-triggered
    weld release should fire, in the SAME frame as the measured carton pose. ``Sensors.placed_pose`` is the
    carton CENTRE (see ``types.py``), so the geometric release threshold must be expressed in the carton-centre
    frame too: ``place_contact_release_z(start, drop, gap) + carton_height_m / 2``.

    WHY (coldwin87): ``PLACE_CONTACT_RELEASE_ARM phase=place start_carton_z=0.6422 descent_drop=0.0907
    gap=0.0120 -> release at carton_z<=0.5635``. The welded carton rested on the shelf at carton centre
    ~0.6422 (the ``PLACE_RELEASE_TICK`` plateau), which is ``half - gap`` = ``0.09 - 0.012`` = ``0.078`` m
    ABOVE the 0.5635 threshold -- the legacy bottom-frame threshold compared the carton BOTTOM height against
    the carton CENTRE. With PR 373's startup-stall suppression on, the stall trigger could not fire either, so
    the stiff place drive pressed the welded carton into the shelf to ``PATH_TOLERANCE_VIOLATED`` ->
    ``HOST_FAULT_ESTOP``. Adding the carton half-height makes the geometric trigger fire when the carton BOTTOM
    reaches ``support_top + gap`` (carton CENTRE at ``support_top + half + gap``) -- the SAME true-contact
    height the descent-stall trigger detects: cw87 -> ``0.6422 - 0.0907 + 0.18/2 + 0.012 = 0.6535``, i.e. at
    the measured rest instead of 0.078 m below it. Pure/CPU; the caller fires once ``live_carton_centre_z <=
    this``."""
    if not (math.isfinite(float(carton_height_m)) and float(carton_height_m) >= 0):
        raise ValueError('place_contact_release_z_centre needs a finite non-negative carton_height_m')
    return place_contact_release_z(start_carton_z, drop_m, gap_m) + float(carton_height_m) / 2.0


def _as_joint_map(config):
    """Accept either a {joint_name: rad} map or a 6-vector in ARM_JOINTS order."""
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    if isinstance(config, dict):
        return {k: float(v) for k, v in config.items()}
    vals = [float(v) for v in config]
    if len(vals) != len(ARM_JOINTS):
        raise ValueError(f'joint vector must be {len(ARM_JOINTS)} values in ARM order')
    return dict(zip(ARM_JOINTS, vals))


def carton_up_in_tool(tool_R_weld):
    """The carton's up-axis expressed in the TOOL frame, fixed at the weld. The carton rested
    LEVEL on the shelf, so its world up-axis was +Z at the weld; in the tool frame that is
    ``tool_R_weld^T @ [0,0,1]``. Holding this tool-frame axis vertical keeps the carton level."""
    R = np.asarray(tool_R_weld, float)
    return _unit(R.T @ WORLD_UP)


def carried_up_world(robot, config, up_in_tool, tool_link='ur_tool0', root='ur_base_link', base_R=None):
    """World up-axis of the carried carton at ``config``: ``base_R @ tool_R(config) @ up_in_tool`` (the held
    chassis attitude ``base_R`` folded in; None => the base frame, byte-identical)."""
    return _apply_base(base_R, tool_rotation(robot, config, tool_link, root) @ _unit(up_in_tool))


def carried_rpy(robot, config, up_in_tool, tool_link='ur_tool0', root='ur_base_link', base_R=None):
    """(roll_deg, pitch_deg, tilt_deg) of the carried carton at arm ``config``, in the WORLD frame when the
    held chassis attitude ``base_R`` is given (else the base frame)."""
    return up_axis_rpy(carried_up_world(robot, config, up_in_tool, tool_link, root, base_R=base_R))


# --------------------------------------------------------------------- leveling

@dataclass(frozen=True)
class LevelConfig:
    """Resolved opt-in carton-level settings. ``bound_deg`` is the carried |roll|/|pitch| bound
    (the CARTON_TILT trip threshold and the leveling target); ``level_goals`` applies the
    level-orientation waypoint to carried joint goals; ``trip`` arms the executor CARTON_TILT
    trip; ``joints`` is the wrist set a level waypoint may re-solve; ``tool_link`` is the frame
    whose up-axis is held vertical."""
    bound_deg: float = DEFAULT_TILT_BOUND_DEG
    level_goals: bool = True
    trip: bool = True
    level_weld: bool = True
    level_place: bool = True
    joints: tuple = DEFAULT_LEVEL_JOINTS
    tool_link: str = 'ur_tool0'
    # sim-carry-level-lift-v1: plan the carried LIFT (and the tuck/pan carry) as a LEVELED WAYPOINT
    # PATH rather than a single leveled-ENDPOINT joint goal. ``level_goals`` alone levels only the goal
    # pose, so the executor's joint-space path from the grasp pose to that goal passes through a tilted
    # configuration (coldwin25: the lift to carry_lift_shape -- CPU-reconstructed ~11 deg mid-path to the
    # leveled endpoint, ~97 deg to the raw shape -- while BOTH endpoints are level). With ``level_path``
    # on, the move is decomposed into ``path_waypoints`` leveled waypoints (the wrist cluster re-solved
    # at every waypoint), so the carton stays parallel to the ground THROUGHOUT; the predicted executed
    # tilt is replayed before dispatch and the move is CARTON_LEVEL_PLAN_REJECTED if it exceeds the bound.
    # Default ON when carton_level is enabled (the repo default carton_level is None => off => byte-
    # identical); ``level_path: false`` restores the single leveled-endpoint goal.
    level_path: bool = True
    path_waypoints: int = 12
    # sim-carton-level-preplace-v2: GENTLED peak joint speed / accel (rad/s, rad/s^2) for the LOADED
    # collect-deck level PLACE set-down (the straight-line descent from the leveled standoff to the
    # full-DOF leveled set-down). The descent's COMMANDED straight joint line is already LEVEL (CPU-
    # reconstructed <0.05 deg), but at the standard 0.08 rad/s the welded carton's DIFFERENTIAL joint
    # tracking during the heavy shoulder/elbow set-down reached ~2.2 deg off level mid-descent
    # (collectlevel3: CARTON_TILT at phase=place, sustained 0.167 s > the 0.15 s grace). This is an
    # EXECUTION transient of the loaded set-down (not a planned-pose tilt and not the pre-place segment
    # PR 298 covered), so the lever is to TIME-PARAMETERISE the same level path slower: a representative
    # differential-lag tracking model calibrated to reproduce the 2.2 deg at 0.08 rad/s predicts ~0.7 deg
    # at 0.02 rad/s (data/sim-carton-level-preplace-v2). Lower ceilings are strictly more conservative
    # for every host check; the bound and grace are UNCHANGED. Scoped to the collect-deck level place
    # descent (controller), so every other move is byte-identical. The integration lane tunes these from
    # the new per-segment CARTON_SEGMENT_TILT log if the in-sim residual differs from the model.
    place_descent_speed: float = 0.02
    place_descent_accel: float = 0.04
    # sim-carton-tilt-transient-debounce-v1: SUSTAINED-tilt grace (sim-seconds AT 1/60). The trip FAILS
    # only when the measured tilt exceeds the bound CONTINUOUSLY for this long; a shorter spike logs
    # CARTON_TILT_TRANSIENT and proceeds. 0 => legacy immediate trip. The controller rebases this onto
    # the running physics step (freshness_scale) so it is a constant FRAME-COUNT budget.
    tilt_grace_s: float = DEFAULT_TILT_GRACE_S
    # sim-carton-level-southdock-v1: loaded gravity-droop FEEDFORWARD for the level place IK. At the PR-307
    # south-dock place the sensed deck sits far enough forward that the welded far-extended arm's quasi-static
    # gravity droop pitched the carton +2.07 deg even with a geometrically-level command and PR 302's 0.02
    # rad/s gentling (coldwin31, phase=place) -- a torque/stiffness effect, not velocity lag, so going slower
    # cannot fix it. When set, the collect-deck level place/preplace IK solves for the COMMANDED config whose
    # EXECUTED (drooped) pose is level, so the physical carton lands parallel to the ground. A
    # ``GravityCompensation`` or None (off => byte-identical; the geometric level solve + the CARTON_TILT trip
    # own it). The bound and grace are UNCHANGED.
    gravity_compensation: object = None
    # sim-setdown-shelf-contact-v1: APPROACH-FROM-ABOVE for the collect place-on-cart set-down. The legacy
    # place descent interpolates the TCP on a SINGLE straight line from the pre-place standoff (which the
    # fixed collect_place_preplace_shape parks OUT IN FRONT of the cart, the held carton overhanging the
    # rollwagen's +x open "mouth" edge) down+in to the support load-centre. That diagonal drags the carton
    # ACROSS the shelf's front-top (mouth) edge while the carton bottom is already BELOW the shelf top, so
    # the carton's bottom-front rides the /Rollwagen/chassis/shelf edge and pivots off level (coldwin33/34/35:
    # CARTON_TILT +2.06 deg, a CONTACT pivot -- d(executed)/d(commanded) ~ 0, so NOT droop). With this on, the
    # set-down is routed as an L-path through the CLEAR opening above the support: first move horizontally (at
    # the standoff height, which clears the shelf top) to DIRECTLY ABOVE the load-centre, then descend
    # VERTICALLY onto the support -- so the carton stays clear of the shelf top/edge until it is laterally
    # inside the shelf footprint, then lands FLAT (face-on-surface, not edge). ``place_above_clearance_m`` is
    # the minimum carton-above-support height of the horizontal-leg via-point. The per-waypoint reach + level +
    # droop-cancel IK is unchanged (still runs at every waypoint of BOTH legs), so the descent stays level and
    # the CARTON_TILT reject gate still owns an un-levelable path. Default OFF => the single straight line =>
    # byte-identical; scoped to the collect-deck level place descent only.
    place_from_above: bool = False
    place_above_clearance_m: float = 0.08
    # sim-setdown-shelf-contact-v1: DEPTH bias (m) for the from-above set-down. The side-grip suction panel
    # hangs ~37 mm BELOW the carton's bottom face and sits on the carton's +x (mouth) side, so at the
    # config load-centre the panel STRADDLES the shelf's +x (mouth) edge (CPU-reconstructed pad cart-x
    # [0.255, 0.415] vs the shelf edge at +0.38): descending, the panel's inboard half lands on the shelf
    # top while its outboard half is over the open mouth -> an ASYMMETRIC edge contact that pivots the
    # carton even on a vertical descent. This shifts the set-down target DEEPER into the cart (along the
    # horizontal approach direction, away from the mouth) by this much, so the whole panel lands flat on the
    # support surface, clear of the mouth edge. Config-independent (uses the descent's own geometry, not the
    # cart frame). Only applied when ``place_from_above`` is on; 0 => no shift. The carton has ample room to
    # go deeper (shelf back at cart-x -0.38, carton back at -0.003). An alternative is to shift the external
    # load-centre instead (see the PR hand-off); this code bias makes the fix self-contained.
    place_depth_bias_m: float = 0.0
    # sim-place-descent-residual-v1: the collect set-down CONTACT-PIVOT fix. coldwin31..40 tripped
    # CARTON_TILT +2.02..+2.07 deg at the collect set-down, UNCHANGED across three droop-compensation
    # generations (d(exec)/d(cmd) ~ 0) -- a GEOMETRIC pivot, not droop: the gripper suction pad (a ROBOT
    # link, NOT the carton, so the ``cage_collision: filtered`` carton-vs-cage filter does not cover it)
    # jams flush on ``/Rollwagen/chassis/post_2`` at the fixed ``collect_place_preplace_shape`` standoff,
    # which parks the held carton OVERHANGING the +x mouth (coldwin39 cart (0.243, -0.185): pad gap 0.1 mm
    # to post_2; the PLACE target (0.18, 0) itself clears post_2 by 90 mm). ``preplace_above_place`` is the
    # REMEDY: re-target the collect preplace standoff to DIRECTLY ABOVE the place load-centre (raised by
    # ``preplace_standoff_m``) instead of the over-the-mouth fixed shape, so the from_above vertical descent
    # keeps the pad clear of the front posts (CPU-proven >= 89 mm). ``place_clearance_check`` arms the
    # ``place_clearance`` CPU sweep as a GATE (log ``PLACE_CLEARANCE`` + refuse by name if any NON-support
    # Rollwagen body is within ``place_clearance_margin_m``), so a geometry that would strike a post fails
    # fast and named instead of the opaque CARTON_TILT pivot. All default OFF => byte-identical. The bound,
    # grace, droop stiffness and tow lanes are UNCHANGED; the from_above descent is KEPT.
    preplace_above_place: bool = False
    preplace_standoff_m: float = 0.12
    place_clearance_check: bool = False
    place_clearance_margin_m: float = 0.030
    # sim-place-preplace-above-support-v1: derive the collect pre-place standoff from the RESOLVED support
    # top so the from-above descent onto the SELECTED support is POSITIVE. coldwin48 selected the Rollwagen
    # UPPER shelf (support top 0.9707 world) but the fixed, DECK-tuned ``collect_place_preplace_shape``
    # parked the carried carton bottom 34 mm BELOW the shelf top (carton centre 1.027 world): the place
    # target (0.957,0.064,0.538 ur_base) sat ABOVE the fixed standoff (TCP 0.528 ur_base), so
    # ``place_descent_drop_m`` read 0.0, the contact-triggered weld release fired in mid-air below the shelf
    # (PLACE_CONTACT_RELEASE_ARM DESCENT_DROP=0.0000) and the carton fell (verify_place: carton not resting,
    # bottom_gap -0.809). With ``preplace_above_support`` on, the collect preplace standoff is re-targeted
    # DIRECTLY ABOVE the place load-centre, raised so the carton bottom sits at support_top + the clearance;
    # the standoff pose is solved by the SAME reach+level IK (``level_place_ik``) + droop model (K 2850) as
    # the deck above-place path, seeded from the fixed shape, and fed to ``place_from_above`` -- so the
    # descent length is the clearance and the release fires at the true support top. The clearance defaults
    # to ``place_above_clearance_m`` (the from-above via clearance -- standoff height and the descent's own
    # above-clearance then match); set ``preplace_support_clearance_m`` to override it. Default OFF => the
    # fixed shape is used unchanged (byte-identical). Distinct from ``preplace_above_place`` (the DECK v1
    # remedy the preflight REFUSES because its above-place IK drifted the carton into the cage mouth posts,
    # coldwin41 -- a deck-only hazard the higher, open upper shelf does not have).
    preplace_above_support: bool = False
    preplace_support_clearance_m: object = None   # None => fall back to place_above_clearance_m
    # sim-place-descent-residual-v2: the DROOP-AWARE clearance gate. The v1 ``place_clearance_check`` sweeps
    # the carton as an IDEALISED level box at the configured load-centre -- the WRONG quantity (coldwin41:
    # the above-place IK actually drifts the carton ~93 mm toward the +x mouth, swinging the welded pad into
    # post_2 at ~2 mm; the level sweep read 89 mm clear). With ``place_clearance_drooped`` on, the gate
    # evaluates the EXECUTED pose: it applies the loaded ``gravity_compensation`` droop to the planned
    # place-descent configs and sweeps the FK gripper pad + carton (not the idealised box) against every
    # Rollwagen body, refusing before motion if any non-support body is within ``place_clearance_margin_m``.
    # Needs the FK model + a gravity_compensation; default OFF => byte-identical (the level v1 gate / the
    # CARTON_TILT trip own it).
    place_clearance_drooped: bool = False
    # sim-place-descent-residual-v2: CLOSED-LOOP carton-level BACKSTOP during the collect place descent
    # (brief part 3). A bounded, sim-only, opt-in wrist servo: at the controller rate it reads the in-sim
    # carton attitude (the same FK-from-measured-joints the CARTON_TILT trip reads) and servos the wrist
    # cluster to hold the carton level, with the correction clamped to +-``closed_loop_authority_deg`` total
    # and rate-limited to ``closed_loop_rate_deg_per_s``. It NEVER substitutes for the geometric clearance
    # (parts 1-2 keep the pad off the posts); it only bounds a RESIDUAL droop-model error so the executed
    # carton cannot drift past ~1.5 deg. Default OFF => byte-identical. The bound/grace are UNCHANGED; this
    # is labelled sim-only like CAGE_COLLISION. NOTE (honesty): the PR 318/326 open-loop wrist FEEDFORWARD
    # was empirically INEFFECTIVE (executed ~= uncompensated droop); this is the closed-loop attempt, proven
    # offline by replaying the coldwin39/41 executed descents, and shipped opt-in so a residual model error
    # can be caught without risking the byte-identical default path.
    closed_loop_level: bool = False
    closed_loop_authority_deg: float = 3.0
    closed_loop_rate_deg_per_s: float = 30.0
    closed_loop_gain: float = 0.6
    # sim-place-descent-residual-v3: the CONTACT-TRIGGERED weld release -- the primary fix for the
    # REACH-INDEPENDENT ~2.0 deg collect set-down residual. Characterisation (part 1, coldwin42-deck.txt +
    # the cw39/cw42 anchors): the carried carton is LEVEL through lift/tuck/transit/preplace and steps to
    # ~2.0 deg ONLY at the set-down, with NO pad/post SEAL contact; the step is REACH-INDEPENDENT
    # (cw42 2.00 @ reach 0.839 m == cw39 2.02 @ 1.086 m) and MODEL-BLIND (the droop model predicts
    # reach-DEPENDENT 1.50 vs 2.16 and the closed loop corrects to 0.26). That is the signature of the arm
    # PRESSING the leveled welded carton onto the cart deck: the CARTON_TILT grader reads FK of the EXECUTED
    # joints, which deflect ~2 deg as the position-controlled arm drives into the rigid deck reaction (no FK
    # model sees a contact). No arm control can level a carton pressed into a non-level/rigid plane -- the
    # only fix is to NOT press it. With ``place_contact_release`` on, the controller watches the welded
    # carton descend (its measured centre z, ``sensors.placed_pose``) and, the instant it reaches within
    # ``place_contact_gap_m`` of the planned set-down height (deck contact), RELEASES the weld (transition to
    # the release phase: gripper-open + detach) instead of continuing to press. The carton is then never
    # driven into the deck, so the executed-joint deflection never builds and the carton settles flat under
    # gravity. The release happens AT contact, inside the 0.15 s grace, so the CARTON_TILT trip (which only
    # grades the WELDED/carried carton) never sustains. Scoped to the collect deck place (``_collect_deck_place``);
    # INERT when ``sensors.placed_pose`` is unavailable (verify_place off / real rig) -> the legacy place->level
    # gate->release owns it. Default OFF => byte-identical.
    place_contact_release: bool = False
    place_contact_gap_m: float = 0.012
    # sim-place-descent-residual-v3 (part 3, always-ship): the SENSOR-based closed loop. The v2 closed loop
    # feeds the servo the MODEL-predicted executed attitude (commanded + droop model), so it is blind to the
    # true executed attitude (it corrected to 0.26 while the sim read 2.00). With ``closed_loop_sensor`` on,
    # the pre-dispatch correction is ANCHORED on the MEASURED carton attitude (FK of the live measured joints
    # at the descent start -- the exact quantity the CARTON_TILT grader reads), so a real starting offset the
    # droop model misses is folded into the correction. Same bounded authority / rate limit. This does NOT
    # fix the contact-conform residual (a servo cannot win against a rigid contact -- that is the
    # contact-release's job); it removes the model blindness about the executed attitude and is cheap, so it
    # ships opt-in. Requires ``closed_loop_level``; default OFF => byte-identical.
    closed_loop_sensor: bool = False
    # sim-preplace-level-gate-v1: the carried PRE-PLACE APPROACH carton-level GATE + a carried-carton
    # DYNAMICS cap. coldwin50 tripped CARTON_TILT at the HIGH from-above preplace standoff (pitch +7.64 deg,
    # base rock-steady). The integration-worker forensics (coldwin50-preplace.txt) pinned it: cw49 and cw50
    # commanded the SAME standoff IK branch (every joint within 0.003 rad) and its STATIC droop-executed tilt
    # is only ~1.6 deg -- the 7.64 is run-to-run DYNAMIC deflection during a coarse, FAST preplace sweep (a
    # 9-point cuMotion move_group plan, pan -1.35 rad + wrist_2 +1.64 rad), which no static model sees. Two
    # gaps caused it: (1) the carried APPROACH to the standoff levelled only the GEOMETRIC interior wrists
    # (``interpolate_level``), so the EXECUTED (drooped) interior still pitched ~1.8 deg, and the approach plan
    # gate replayed the GEOMETRIC path (``predict_path_tilt``, droop-blind) so it never saw it; (2) the move
    # was a coarse fast sweep, so the welded carton deflected dynamically. ``preplace_level_gate`` (default
    # OFF => byte-identical) closes both: the approach is planned as a DROOP-COMPENSATED, DENSIFIED leveled
    # waypoint path (every interior waypoint wrist-droop-cancelled like the place descent, and split so each
    # segment's max per-joint step is <= ``preplace_max_segment_rad`` -- the 132-point lesson, no coarse 9-pt
    # sweeps), the predicted EXECUTED (droop-aware) tilt is gated to ``preplace_gate_bound_deg`` (the bound
    # minus a margin) and the move is CARTON_LEVEL_PLAN_REJECTED before any motion if it exceeds the true
    # ``bound_deg``, the carried approach is time-capped at ``preplace_approach_speed``/``preplace_approach_accel``
    # (lower peak joint velocity => less dynamic deflection), and when ``closed_loop_level`` is also on the
    # bounded sensor servo is extended to the approach. The standoff endpoint IK keeps the droop-cancel it
    # already has (``level_place_ik(gravity_comp=...)`` -> executed-level), but now PREFERS the branch with the
    # smallest predicted EXECUTED tilt + most clearance (``solve_level_standoff``). The 2.0 deg bound / 0.15 s
    # grace / carton dims / droop stiffness / tow lanes are ALL UNCHANGED. Scoped to the collect preplace
    # approach (``_preplace_above_place_goal`` engaged), so every other move is byte-identical.
    preplace_level_gate: bool = False
    preplace_gate_bound_deg: float = 1.0
    preplace_max_segment_rad: float = 0.25
    preplace_approach_speed: float = 0.05
    preplace_approach_accel: float = 0.1
    # sim-preplace-wrist-continuity-v1: BRANCH-CONTINUITY threshold (rad) for the WORLD-level pre-place
    # approach wrist re-solve. Each densified approach waypoint re-solves the wrist cluster for
    # level-in-world; the solve is seeded from the PREVIOUS waypoint's solution and chooses, among the level
    # branches, the one with the SMALLEST per-joint wrist travel -- never a lower-tilt FAR branch. A planned
    # per-waypoint wrist step above this is a branch jump the soft wrist cannot track (coldwin57), so the
    # plan is CARTON_LEVEL_PLAN_REJECTED before motion with the offending waypoint and step. Scoped to the
    # geometric WORLD-level approach (``plan_level_world_approach``), so every other move is byte-identical.
    preplace_max_wrist_step_rad: float = DEFAULT_MAX_WRIST_STEP_RAD
    # sim-preplace-wrist-continuity-v1: the REACTION-NATURAL STANDOFF BRANCH selector (the primary coldwin57
    # fix). coldwin57's leg-2 COMMAND held wrist_2 continuously at the HIGH level branch (1.63), but the soft
    # wrist_2 drive DIVERGED monotonically DOWN to the LOW branch (~0.33-0.40) under the pan descent and the
    # executor faulted on the executed-vs-commanded divergence (both branches are world-level, so the carton
    # stayed level -- the fault is purely the wrist_2 drive divergence). With ``preplace_reaction_branch`` on,
    # the collect pre-place standoff is chosen as the reach+level branch with the MINIMUM |wrist_2|
    # (``select_reaction_natural_standoff``) -- the bottom of the equivalent-orientation family the drive
    # drifts toward (== the observed settling point, a smaller leg-1 sweep from carry than the 1.63 branch) --
    # so leg 1 drives to THAT branch and leg 2 holds it with no divergence. HONEST: the gravity/geometry
    # holding torque is ~0.5-1.0 N*m on EVERY branch, far below the soft-wrist URDF effort limit (56 N*m), so no
    # branch saturates and torque does NOT select the settling branch (minimum torque would pick the UPHILL
    # wrist_2~2.79 branch); the torque is reported only for the margin + the honest model comparison. Scoped
    # to the geometric WORLD-level standoff (``preplace_level_gate`` + held chassis attitude); default OFF =>
    # byte-identical (the prior single ``level_place_ik`` standoff is used). ``preplace_soft_effort_nm`` is
    # the soft-wrist effort limit used only to REPORT the margin (never relaxed, never used to stiffen).
    preplace_reaction_branch: bool = False
    preplace_soft_effort_nm: float = 56.0
    # sim-preplace-pan-first-seeded-leg2-v1: prefer the CARRY-CONNECTED pre-place standoff branch (requires
    # preplace_reaction_branch). coldwin72 -- the first run with every physical fix stacked -- was stopped
    # before motion by the continuity gate in the PAN-FIRST leg 2 (PR 366), where the arm extends the
    # wrists/lift/elbow to the standoff AFTER the pan sweep: with shoulder_pan held at the STANDOFF value and
    # the wrist_1 self-collision cap (0.65) in force, the reaction-natural standoff (min |wrist_2| = 0.668)
    # sits on a DIFFERENT carton-level basin than the carry/tucked shape (wrist_2 ~0), so the leg-2 wrist_2
    # holds ~0 for nine waypoints then JUMPS 0.300 rad in one step (> the 0.15 rad continuity limit) --
    # CARTON_LEVEL_PLAN_REJECTED. No continuous monotonic <=0.15 level path connects the two basins at the
    # standoff pan (CPU-proven: the crossing tilt ridge minimum is ~2.44 deg > the 2.0 bound, for EVERY
    # wrist_1 cap). The fix is to choose the standoff on the branch CONNECTED to the carry shape (wrist_2 ~
    # the carry wrist_2), where leg 2 is a monotonic extension with no cross-branch sweep. The soft-wrist
    # drift that originally pushed PR 345 to the 0.668 settling branch is now held by the PR 359 firm wrist
    # hold, so the ~0 branch is admissible again. When ON, ``select_reaction_natural_standoff`` enumerates
    # reach+level+self-clear branches seeded from the carry shape + a wrist grid (the low-wrist_2 family the
    # 7-seed sweep misses), ranks by total wrist travel from the carry shape, and returns the FIRST branch
    # whose leg-2 world-level plan the continuity gate ACCEPTS (step <= preplace_max_wrist_step_rad AND tilt
    # <= bound_deg) -- the gate stays the authority. Default OFF => the min-|wrist_2| settling branch is used
    # (byte-identical). Requires preplace_reaction_branch (it is that selector's branch rule).
    preplace_branch_prefer_carry_connected: bool = False
    # sim-preplace-branch-descent-feasible-v1: the carry-connected standoff selector must choose only among
    # standoffs whose PLACE DESCENT is ALSO within the carton-level bound, not just among those whose leg-2
    # continuity gate accepts (coldwin79). The selector (``select_carry_connected_standoff``) ranked candidate
    # standoffs only by carry travel + the leg-2 continuity gate and greedily returned the FIRST leg-2-accepted
    # branch; it NEVER evaluated whether the later place descent FROM that standoff keeps the carton level. The
    # candidate set and each branch's leg-2 continuity result depend on the run's measured carry pose / sensed
    # place target (which vary a few mm / mrad run-to-run), so different runs greedily accept different
    # low-carry-travel branches: coldwin76/77/78 happened to accept a ~0.59 wrist_2 branch whose place descent
    # is level (~1.8 deg), but coldwin79 accepted the low wrist_2=0.052 branch whose descent would roll
    # 3.43 deg (> the 2.0 bound), so the CARTON_LEVEL_PLACE_DESCENT gate refused it BEFORE motion and the run
    # e-stopped -- making the set-down a lottery between runs. With ``preplace_descent_feasible`` on, the
    # controller passes the selector a ``descent_predictor`` (the SAME ``plan_compensated_place_descent`` the
    # place-descent gate runs, from each CANDIDATE standoff to the resolved support), the selector DISCARDS any
    # candidate whose predicted descent exceeds ``bound_deg``, and -- among the leg-2-accepted AND
    # descent-feasible branches -- chooses DETERMINISTICALLY by the stable key (least carry travel, then
    # smallest predicted descent tilt, then smallest leg-2 max wrist step, then wrist_2 closest to the carry
    # value), independent of the search time / candidate-enumeration order. The carry-travel-primary key (NOT
    # the smallest-descent-tilt-primary the brief sketched) is what preserves the carry-connected design AND
    # keeps coldwin76's choice unchanged while discarding coldwin79's infeasible branch. Requires
    # ``preplace_branch_prefer_carry_connected`` (it extends that selector's ranking). Default OFF => the
    # selector takes no ``descent_predictor`` => byte-identical. The 2.0 deg bound is UNTOUCHED.
    preplace_descent_feasible: bool = False
    # sim-preplace-branch-select-bounded-v1: CPU time-box (seconds) for the reaction-branch standoff
    # search (``select_reaction_natural_standoff``). The search is pure CPU and now cheap (~0.08 s on the
    # coldwin57 fixture), so this is a safety net: a search still running after the budget REFUSES the
    # preplace before motion (BranchSelectBudgetExceeded -> named _fail) instead of blocking the host
    # executor past the Isaac 5 sim-s HOST_STATUS_WATCHDOG (coldwin59). Default 2.0 s (>20x the measured
    # cost, a fraction of the 5 s watchdog). Measured on time.perf_counter (a CPU/wall budget, NOT sim).
    preplace_branch_select_budget_s: float = 2.0
    # sim-preplace-approach-seeded-v1: BRANCH-SEEDED interior solve for the WORLD-level pre-place approach
    # (the primary coldwin62 fix). The legacy ``plan_level_world_approach`` levels every interior waypoint's
    # wrist cluster INDEPENDENTLY (``continuous_level_solve`` ranks the min-travel branch, but the free
    # Gauss-Newton descent from the position geometry can still fall into a FAR level basin), so the
    # per-waypoint wrist_2 dips onto a different (lower) level branch at the interior and snaps back near the
    # standoff -- coldwin62: carry-side w2 ~0.03 -> DIP ~0.19 (wp4) -> JUMP 0.62 (wp5, +0.433 rad) -> standoff
    # 0.66, refused by the continuity guard before any motion (14th run to die here; the base lock / contact
    # release / placement check have never run). PR 348's ``anchor='auto'`` only chooses between two
    # ENDPOINT-anchored plans; the INTERIOR leveler still picks branches independently, so the dip survives.
    # With ``preplace_seeded_interior`` on, the approach interior is solved SEEDED from the start->standoff
    # wrist interpolation (the standoff branch family) and warm-continuous from the previous waypoint, with
    # the per-waypoint wrist step CLAMPED to ``preplace_max_wrist_step_rad`` and the executed tilt held within
    # ``preplace_gate_bound_deg``; a segment that cannot meet both is DENSIFIED LOCALLY (bisected) before the
    # plan is refused, and only a genuine branch gap no densification bridges is CARTON_LEVEL_PLAN_REJECTED by
    # name. So wrist_2 sweeps MONOTONICALLY carry->standoff with every step <= the continuity threshold and the
    # guard accepts. Scoped to the geometric WORLD-level approach (``plan_level_world_approach`` anchor
    # ``'seeded'``, engaged only on the collect preplace world-level lane). Default OFF => ``anchor='auto'`` =>
    # byte-identical. The 2.0 deg bound / 0.15 s grace / carton dims / standoff choice are UNCHANGED.
    preplace_seeded_interior: bool = False
    # sim-place-release-on-contact-v1: the CONTACT-GATED place-descent weld release (the primary fix for the
    # coldwin68 press-to-757-N*m hang; supersedes the fixed carton_z gate above when on). coldwin68's pre-place
    # hold-firm held the welded carton rigid into a ~1.46 deg roll so its edge met cage post
    # /Rollwagen/chassis/post_2 at the DESCENT START (carton_z frozen at 0.6257, 0.075 m above the computed
    # ``release_z`` 0.5624), so the fixed ``place_contact_release`` z gate never fired and the stiff eff2000
    # place drive pressed the blocked carton past the 600 N*m station full-brake guard. With
    # ``place_release_on_contact`` on, ``place_release.PlaceReleaseGate`` releases the weld the instant the
    # support is ACTUALLY touched -- whichever of (arm effort rising ``place_release_effort_margin_nm`` over its
    # pre-contact hold baseline for ``place_release_effort_consecutive`` ticks, the descent STALLING
    # ``place_release_stall_m`` for ``place_release_stall_ticks`` ticks, or the legacy carton_z threshold) fires
    # FIRST -- so the press never reaches the guard. The release is GATED on the touched body being the selected
    # SUPPORT: a hit-prim path containing a ``place_support_tokens`` substring (shelf/deck) is the support
    # surface; a flush (<= ``place_obstruction_gap_mm``) contact on ANY OTHER body is an OBSTRUCTION -- the
    # controller does NOT press/release but logs ``PLACE_OBSTRUCTED prim=...`` and either does up to
    # ``place_reapproach_max`` bounded lateral re-approaches (shifting the set-down ``place_reapproach_shift_m``)
    # or fails by name. Reuses the PR 334 cancel->wait-terminal->detach release serialization. Scoped to the
    # collect deck place (``_collect_deck_place``). Default OFF => byte-identical (the legacy z gate owns it).
    place_release_on_contact: bool = False
    place_release_effort_margin_nm: float = 100.0
    place_release_effort_consecutive: int = 3
    place_release_stall_m: float = 0.001
    place_release_stall_ticks: int = 5
    # sim-place-verify-rest-tolerance-v1: gate the STALL release trigger on the carton being near the
    # planned set-down height. None (default) => the stall trigger is unchanged (byte-identical). A finite
    # positive margin (metres) makes a descent stall count as 'support met' ONLY when the carton centre z
    # is within ``release_z + this`` -- so the render-paced STARTUP stall (carton_z frozen at the standoff
    # before the descent trajectory moves) no longer fires the release ~0.06 m above the shelf. coldwin83
    # and coldwin85 BOTH released on that startup stall at the standoff height, leaving the carton cradled
    # above the shelf; cw83's slipped free and rested (verify passed) while cw85's was carried off by the
    # retreat and fell to the floor (verify: carton not resting on deck, xy=1.219 bottom_gap=-0.954).
    place_release_stall_height_margin_m: object = None
    place_release_contact_gap_mm: object = None   # None => the seal-gap contact trigger is off
    place_support_tokens: tuple = ('shelf', 'deck')
    place_obstruction_gap_mm: float = 2.0
    place_reapproach_max: int = 0                 # 0 => fail fast by name on obstruction (no re-approach)
    place_reapproach_shift_m: float = 0.03
    # sim-place-release-cancel-handshake-v1 (coldwin74/75): when the contact-triggered weld release fires
    # the release phase must STOP the in-flight place descent and reach a TERMINAL, settled result before
    # it detaches (PR 334 serialization). The default path CANCELS the ExecuteTrajectory (``motion.cancel``),
    # but the place descent is a DRIVEN trajectory on the sim FollowJointTrajectory executor: an action
    # cancel is BOTH dropped in the goal-accept race (``self.handle is None`` when the stall fires, so
    # ``cancel_goal_async`` never goes out -- cw75 line 24459) AND unsafe (the sim trajectory_host treats a
    # goal cancel as a FAULT and latches the global ``/manipulation/stop``). ``cancel_outcome`` then stays
    # None for the whole budget and the run e-stops with the carton still welded on the shelf (cw74 2.0 s,
    # cw75 8.0 s). With ``release_cancel_handshake`` on, the release phase STOPS the descent on its OWN
    # channel -- it emits the contact-hold control so trajectory_host SUCCEEDS holding the current pose
    # (terminal, no fault, no global stop) and marks the port cancel-requested so ``cancel_outcome`` reads
    # the resolved execute future -- and, if the terminal still never arrives but the arm is PROVABLY
    # stopped (measured max joint speed <= ``release_cancel_stopped_vel`` rad/s for
    # ``release_cancel_stopped_ticks`` consecutive ticks), logs ``RELEASE_CANCEL_STOPPED_FALLBACK`` and
    # proceeds to the gripper-open + detach from the stopped pose. NEVER detaches into a moving arm. Default
    # off => the legacy ``motion.cancel`` path => byte-identical.
    release_cancel_handshake: bool = False
    release_cancel_stopped_vel: float = 0.02       # rad/s: max measured joint speed that counts as stopped
    release_cancel_stopped_ticks: int = 3          # consecutive stopped ticks before the fallback proceeds
    # sim-release-open-reissue-v1 (coldwin84): the gripper-open that releases the weld at the end of the
    # release phase is a ONE-SHOT control. coldwin84 (a byte-identical re-run of the clean coldwin83) lost
    # that single open on the LAST-VALUE /manipulation/status relay -- the Isaac side ran ZERO
    # ``cad_tool.command(open)`` (0 ``SEAL_DIAG command:open``; the next no-control release tick's higher
    # ``sequence`` overwrote the open-carrying status before ``IsaacRosGraph.poll`` read it) -- so the weld
    # stayed engaged and the run e-stopped (``release did not detach ... within 8.0s``). Five earlier runs
    # on the identical config released cleanly: the release was a dice roll on relay timing. With
    # ``release_reissue`` on, the release phase RE-EMITS the gripper-open control (which on the Isaac side
    # is BOTH the open AND the ``SIM_ATTACH_FALLBACK`` weld detach) on EVERY tick until the per-tick
    # feedback (``RESTOCK_CONTACT_STATUS_FEEDBACK`` / PR 358: ``gripper_enabled`` + ``gripper_status_time``
    # + the ``s.contact`` latch) confirms the gripper OPEN and the carton NOT held (seal released), then
    # proceeds to the cuMotion detach; if unconfirmed within ``release_confirm_timeout_s`` sim it FAILS BY
    # NAME (``RELEASE_NOT_CONFIRMED``) -- never restarting the stage with the carton welded. The SAME
    # one-shot-resilience PR 356 gave the contact ENABLE (``contact_reissue_suction_each_tick``). Default
    # off => the single-shot open + the 8 s ``release_detach_timeout_s`` path => byte-identical. See
    # ``release_reissue.ReleaseReissueGate`` + docs/examples/release-open-reissue-fragment.json.
    release_reissue: bool = False
    release_confirm_timeout_s: float = 8.0         # sim seconds: bounded window for the release to confirm
    release_reissue_log_every: int = 30            # emit one RELEASE_REISSUE line per this many ticks
    # sim-release-threshold-carton-frame-v1 (coldwin85/coldwin86 fell to the floor after release): after the
    # weld detach, PROVE the carton is FREE of the gripper and RESTING before retreating, then clear the
    # gripper straight up before any lateral / pan retreat motion. coldwin85 and coldwin86 released the weld
    # at the set-down and the retreat's first (near-horizontal) motion swept the barely-rested carton off the
    # shelf (verify: carton not resting on deck, cw85 bottom_gap=-0.954, cw86 -0.815); coldwin83 on the same
    # config happened to rest. With ``release_free_check`` on, ``release_free.ReleaseFreeGate`` HOLDs (no
    # retreat motion) until the measured carton CENTRE z is stable and the carton BOTTOM is resting
    # (``|bottom_gap| <= release_free_rest_gap_m``) for ``release_free_settle_ticks`` consecutive ticks, THEN
    # raises the gripper straight up by ``release_free_lift_dz_m`` and verifies the carton did NOT follow (rose
    # less than ``release_free_follow_frac`` of the lift); it FAILS BY NAME (``RELEASE_NOT_FREE``) when the
    # carton never rests within ``release_free_settle_window_ticks`` or follows the gripper up. Needs a
    # measured ``placed_pose`` (RESTOCK_VERIFY_PLACE); absent => the gate is skipped (real rig). Default off =>
    # the detach transitions straight to retreat => byte-identical.
    release_free_check: bool = False
    release_free_settle_ticks: int = 5             # consecutive stable+resting ticks before the clearance lift
    release_free_stable_eps_m: float = 0.005       # max carton-centre-z change per tick that counts as stable
    release_free_rest_gap_m: float = 0.03          # |carton bottom - support top| that counts as resting
    release_free_settle_window_ticks: int = 60     # bounded hold before refusing RELEASE_NOT_FREE (not at rest)
    release_free_lift_dz_m: float = 0.06           # pure vertical clearance lift before the lateral retreat
    release_free_follow_frac: float = 0.5          # carton rose >= this fraction of the lift => followed => refuse
    # sim-place-seat-geometry-v1 (brief parts 2+3): the HONEST seat-gated place release. The collect set-down
    # release must fire ONLY when a measured contact (stall/effort) is within ``place_seat_tol_m`` of the
    # CORRECTED modelled seat (support TOP + carton half, carton-centre frame). coldwin87's "solid stall" was
    # 14 cm ABOVE the true seat (0.9325 world): the carton is filtered against the whole rollwagen cage so it
    # rests on NOTHING there -- the arm jams a side-grip link on the cage -- so the stall is an OBSTRUCTION,
    # not a shelf contact. PR 374 wrongly added a half-height to calibrate the threshold TO that stall (and it
    # then sat ABOVE the descent start -> fired at the standoff). With ``place_seat_gate`` on, the release gate
    # (``place_release.PlaceReleaseGate``) is armed with the seat centre z + tolerance: a contact AT the seat
    # releases; a contact well ABOVE it is ``PLACE_OBSTRUCTED`` (bounded lateral re-approach / fail by name); a
    # descent reaching/passing the seat with NO contact is a named fault; and the release threshold is the
    # ABSOLUTE seat (support_top + half) -- NOT PR 374's relative ``placed_pose - drop + gap + half`` -- so it
    # can never sit above the descent start (asserted by name at arm time). The seat's support top is the live
    # cart-tracked ``target_surface[2]`` (the collect place target z, in ur_base_link); use the opt-in
    # factory env ``RESTOCK_PLACE_SUPPORT_FLAT_TOP=1`` to resolve it as the slab FLAT TOP FACE instead of the
    # tilted world-AABB max-z CORNER (+7 mm). Requires ``place_release_on_contact``. Default off => byte-
    # identical (PR 374's relative threshold + the startup-stall margin own the set-down). See
    # ``place_seat_geometry.classify_set_down`` (the shared predicate) and docs/examples/place-seat-geometry-fragment.json.
    place_seat_gate: bool = False
    place_seat_tol_m: float = 0.02
    # sim-place-pad-down-arrival-v1: the opt-in THREE-STAGE set-down that separates the carried APPROACH
    # from the gripper REORIENTATION. coldwin92 proved the last contradiction of the single-standoff design
    # with both cage gates live: every carry-connected standoff whose carried approach keeps the carton
    # level ends on the LOW-wrist_2 CARRY branch, whose straight place descent rakes the +x mouth posts and
    # is correctly refused; the only cage-clear descent holds the HIGH-wrist_2 PAD-DOWN branch, which the
    # carried approach cannot reach level (the fallback tilted the carton 13.96 deg -> CARTON_LEVEL_PLAN_
    # REJECTED). CPU-proven resolution (data/sim-place-pad-down-arrival-v1): an in-place reorientation
    # directly over the seat rakes post_2 at EVERY height (the post is tall; raising to clear it degenerates
    # the level IK), but reorienting in the OPEN APPROACH CORRIDOR -- ``pad_down_arrival_shift_m`` toward the
    # arm base, ``pad_down_arrival_height_m`` above the seat -- keeps the housing clear of the mouth posts
    # (at the default 0.20 m / 0.10 m, in the controller gripper_pad_tcp frame: >= +66 mm reorient,
    # >= +103 mm descent) while the carton stays level (<= 0.04 deg reorient, <= 1.25 deg descent) AND
    # branch-continuous (reorient wrist step <= 0.080 rad, the soft wrist tracks it) at every cw64-cw92 park.
    # When on, the collect-deck place runs: (a) carried approach to the
    # SHIFTED above-shelf standoff on the carry-connected arrival branch (the existing selector, no straight-
    # descent feasibility requirement); (b) a short, slow, tilt-gated + cage-gated in-place reorientation of
    # the wrists from the arrival (low wrist_2) orientation to the selected PAD-DOWN (high wrist_2) branch,
    # carton kept within ``bound_deg`` throughout (``place_reorient``); (c) the PR 376 pad-down descent to
    # the seat (place_from_above L-path) with the descent cage gate + the PR 375 seat gate for release.
    # Requires ``place_from_above`` (the L-path returns the pad-down carton over the seat) and the wired
    # descent cage gate (``_place_descent_cage_gate``; the reorientation refuses by name without a cage
    # criterion). Default OFF => the single-standoff place is unchanged (byte-identical). The 2.0 deg bound /
    # grace / carton dims / seat are UNTOUCHED. See place_reorient.py + docs/examples/place-pad-down-arrival-fragment.json.
    place_pad_down_arrival: bool = False
    pad_down_arrival_shift_m: float = 0.20
    pad_down_arrival_height_m: float = 0.10
    pad_down_reorient_waypoints: int = 12
    pad_down_reorient_speed: float = 0.04
    pad_down_reorient_accel: float = 0.08

    def validate(self):
        if not (math.isfinite(self.bound_deg) and self.bound_deg > 0):
            raise ValueError('carton_level.bound_deg must be a finite positive number of degrees')
        if not (math.isfinite(self.tilt_grace_s) and self.tilt_grace_s >= 0):
            raise ValueError('carton_level.tilt_grace_s must be a finite non-negative number of seconds')
        if not (isinstance(self.path_waypoints, int) and self.path_waypoints >= 2):
            raise ValueError('carton_level.path_waypoints must be an integer >= 2')
        for nm, v in (('place_descent_speed', self.place_descent_speed),
                      ('place_descent_accel', self.place_descent_accel)):
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f'carton_level.{nm} must be a finite positive number')
        if not (math.isfinite(self.place_above_clearance_m) and self.place_above_clearance_m >= 0):
            raise ValueError('carton_level.place_above_clearance_m must be a finite non-negative number')
        if self.preplace_support_clearance_m is not None and not (
                math.isfinite(float(self.preplace_support_clearance_m))
                and float(self.preplace_support_clearance_m) > 0):
            raise ValueError('carton_level.preplace_support_clearance_m must be a finite positive '
                             'number of metres (the collect pre-place carton-bottom clearance above the '
                             'resolved support top), or omitted to default to place_above_clearance_m')
        if not (math.isfinite(self.place_depth_bias_m) and self.place_depth_bias_m >= 0):
            raise ValueError('carton_level.place_depth_bias_m must be a finite non-negative number')
        if not (math.isfinite(self.preplace_standoff_m) and self.preplace_standoff_m > 0):
            raise ValueError('carton_level.preplace_standoff_m must be a finite positive number of metres')
        if not (math.isfinite(self.place_clearance_margin_m) and self.place_clearance_margin_m >= 0):
            raise ValueError('carton_level.place_clearance_margin_m must be a finite non-negative number')
        if not (math.isfinite(self.closed_loop_authority_deg) and self.closed_loop_authority_deg > 0):
            raise ValueError('carton_level.closed_loop_authority_deg must be a finite positive number of degrees')
        if not (math.isfinite(self.closed_loop_rate_deg_per_s) and self.closed_loop_rate_deg_per_s > 0):
            raise ValueError('carton_level.closed_loop_rate_deg_per_s must be a finite positive deg/s')
        if not (math.isfinite(self.closed_loop_gain) and 0 < self.closed_loop_gain <= 2.0):
            raise ValueError('carton_level.closed_loop_gain must be a finite number in (0, 2]')
        if not (math.isfinite(self.place_contact_gap_m) and self.place_contact_gap_m > 0):
            raise ValueError('carton_level.place_contact_gap_m must be a finite positive number of metres')
        if self.closed_loop_sensor and not self.closed_loop_level:
            raise ValueError('carton_level.closed_loop_sensor requires closed_loop_level to be on '
                             '(the sensor anchor refines the closed-loop correction)')
        # sim-place-release-on-contact-v1
        for nm, v in (('place_release_effort_margin_nm', self.place_release_effort_margin_nm),
                      ('place_release_stall_m', self.place_release_stall_m),
                      ('place_obstruction_gap_mm', self.place_obstruction_gap_mm),
                      ('place_reapproach_shift_m', self.place_reapproach_shift_m)):
            if not (math.isfinite(float(v)) and float(v) > 0):
                raise ValueError(f'carton_level.{nm} must be a finite positive number')
        for nm, v in (('place_release_effort_consecutive', self.place_release_effort_consecutive),
                      ('place_release_stall_ticks', self.place_release_stall_ticks)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f'carton_level.{nm} must be a positive integer')
        if isinstance(self.place_reapproach_max, bool) or not isinstance(self.place_reapproach_max, int) \
                or self.place_reapproach_max < 0:
            raise ValueError('carton_level.place_reapproach_max must be a non-negative integer '
                             '(0 => fail by name on obstruction with no re-approach)')
        # sim-place-release-cancel-handshake-v1
        if not (math.isfinite(float(self.release_cancel_stopped_vel)) and float(self.release_cancel_stopped_vel) > 0):
            raise ValueError('carton_level.release_cancel_stopped_vel must be a finite positive rad/s '
                             '(the max measured joint speed the stopped-arm release fallback tolerates)')
        if isinstance(self.release_cancel_stopped_ticks, bool) \
                or not isinstance(self.release_cancel_stopped_ticks, int) \
                or self.release_cancel_stopped_ticks < 1:
            raise ValueError('carton_level.release_cancel_stopped_ticks must be a positive integer '
                             '(consecutive stopped ticks before the release fallback proceeds)')
        if self.place_release_contact_gap_mm is not None and not (
                math.isfinite(float(self.place_release_contact_gap_mm))
                and float(self.place_release_contact_gap_mm) > 0):
            raise ValueError('carton_level.place_release_contact_gap_mm must be None or a finite positive '
                             'number of millimetres (the support-contact seal-gap release trigger)')
        # sim-place-verify-rest-tolerance-v1
        if self.place_release_stall_height_margin_m is not None and not (
                math.isfinite(float(self.place_release_stall_height_margin_m))
                and float(self.place_release_stall_height_margin_m) > 0):
            raise ValueError('carton_level.place_release_stall_height_margin_m must be None or a finite '
                             'positive number of metres (the stall release trigger only counts within '
                             'release_z + this of the planned set-down height)')
        # sim-release-open-reissue-v1
        if not (math.isfinite(float(self.release_confirm_timeout_s)) and float(self.release_confirm_timeout_s) > 0):
            raise ValueError('carton_level.release_confirm_timeout_s must be a finite positive number of '
                             'sim seconds (the bounded window for the re-issued release to confirm)')
        if isinstance(self.release_reissue_log_every, bool) \
                or not isinstance(self.release_reissue_log_every, int) \
                or self.release_reissue_log_every < 1:
            raise ValueError('carton_level.release_reissue_log_every must be a positive integer '
                             '(RELEASE_REISSUE is logged once per this many ticks)')
        # sim-release-threshold-carton-frame-v1
        for nm, v in (('release_free_stable_eps_m', self.release_free_stable_eps_m),
                      ('release_free_rest_gap_m', self.release_free_rest_gap_m),
                      ('release_free_lift_dz_m', self.release_free_lift_dz_m)):
            if not (math.isfinite(float(v)) and float(v) > 0):
                raise ValueError(f'carton_level.{nm} must be a finite positive number of metres')
        if not (math.isfinite(float(self.release_free_follow_frac))
                and 0 < float(self.release_free_follow_frac) <= 1.0):
            raise ValueError('carton_level.release_free_follow_frac must be a finite number in (0, 1] '
                             '(carton rose >= this fraction of the clearance lift => followed the gripper)')
        for nm, v in (('release_free_settle_ticks', self.release_free_settle_ticks),
                      ('release_free_settle_window_ticks', self.release_free_settle_window_ticks)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f'carton_level.{nm} must be a positive integer')
        if self.release_free_settle_window_ticks < self.release_free_settle_ticks:
            raise ValueError('carton_level.release_free_settle_window_ticks must be >= '
                             'release_free_settle_ticks (the bounded hold must allow the settle streak)')
        if (isinstance(self.place_support_tokens, str) or not hasattr(self.place_support_tokens, '__iter__')
                or not tuple(self.place_support_tokens)
                or not all(isinstance(t, str) and t for t in self.place_support_tokens)):
            raise ValueError('carton_level.place_support_tokens must be a non-empty list of non-empty '
                             'substrings identifying the support surface in a hit-prim path (e.g. shelf, deck)')
        # sim-place-seat-geometry-v1
        if not (math.isfinite(float(self.place_seat_tol_m)) and float(self.place_seat_tol_m) > 0):
            raise ValueError('carton_level.place_seat_tol_m must be a finite positive number of metres '
                             '(a measured contact counts as a true set-down only within this of the '
                             'corrected modelled seat)')
        if self.place_seat_gate and not self.place_release_on_contact:
            raise ValueError('carton_level.place_seat_gate requires place_release_on_contact to be on '
                             '(the honest seat gate arms the contact-gated place release)')
        # sim-place-pad-down-arrival-v1
        for nm, v in (('pad_down_arrival_shift_m', self.pad_down_arrival_shift_m),
                      ('pad_down_arrival_height_m', self.pad_down_arrival_height_m),
                      ('pad_down_reorient_speed', self.pad_down_reorient_speed),
                      ('pad_down_reorient_accel', self.pad_down_reorient_accel)):
            if not (math.isfinite(float(v)) and float(v) > 0):
                raise ValueError(f'carton_level.{nm} must be a finite positive number')
        if isinstance(self.pad_down_reorient_waypoints, bool) \
                or not isinstance(self.pad_down_reorient_waypoints, int) \
                or self.pad_down_reorient_waypoints < 2:
            raise ValueError('carton_level.pad_down_reorient_waypoints must be an integer >= 2')
        if self.place_pad_down_arrival and not self.place_from_above:
            raise ValueError('carton_level.place_pad_down_arrival requires place_from_above to be on '
                             '(the L-path descent returns the reoriented pad-down carton over the seat; '
                             'a straight-line descent from the shifted standoff would rake the cage)')
        if self.place_pad_down_arrival and not self.level_place:
            raise ValueError('carton_level.place_pad_down_arrival requires level_place to be on '
                             '(the three-stage set-down keeps the carton level throughout)')
        # sim-preplace-level-gate-v1
        if not (math.isfinite(self.preplace_gate_bound_deg)
                and 0 < self.preplace_gate_bound_deg <= self.bound_deg):
            raise ValueError('carton_level.preplace_gate_bound_deg must be a finite number of degrees in '
                             '(0, bound_deg] (the preplace-approach executed-tilt target, the bound minus a margin)')
        if not (math.isfinite(self.preplace_max_segment_rad) and self.preplace_max_segment_rad > 0):
            raise ValueError('carton_level.preplace_max_segment_rad must be a finite positive number of radians '
                             '(the max per-joint step per densified approach segment)')
        for nm, v in (('preplace_approach_speed', self.preplace_approach_speed),
                      ('preplace_approach_accel', self.preplace_approach_accel)):
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f'carton_level.{nm} must be a finite positive number')
        if not (math.isfinite(self.preplace_max_wrist_step_rad) and self.preplace_max_wrist_step_rad > 0):
            raise ValueError('carton_level.preplace_max_wrist_step_rad must be a finite positive number of '
                             'radians (the per-waypoint wrist-step continuity threshold for the world-level '
                             'pre-place approach)')
        if not (math.isfinite(self.preplace_soft_effort_nm) and self.preplace_soft_effort_nm > 0):
            raise ValueError('carton_level.preplace_soft_effort_nm must be a finite positive N*m (the '
                             'soft-wrist effort limit used to report the holding-torque margin)')
        if not (math.isfinite(self.preplace_branch_select_budget_s)
                and self.preplace_branch_select_budget_s > 0):
            raise ValueError('carton_level.preplace_branch_select_budget_s must be a finite positive '
                             'number of seconds (the CPU time-box for the reaction-branch standoff search)')
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        bad = [j for j in self.joints if j not in ARM_JOINTS]
        if bad or not self.joints:
            raise ValueError(f'carton_level.joints must be a non-empty subset of ARM_JOINTS; bad={bad}')
        if self.gravity_compensation is not None:
            if not isinstance(self.gravity_compensation, GravityCompensation):
                raise ValueError('carton_level.gravity_compensation must resolve to a GravityCompensation')
            self.gravity_compensation.validate()
        return self

    def preplace_support_clearance(self):
        """The effective collect pre-place carton-bottom clearance above the resolved support top
        (sim-place-preplace-above-support-v1): ``preplace_support_clearance_m`` when set, else
        ``place_above_clearance_m`` so the standoff height and the from-above descent's own
        ``above_clearance_m`` match (one via height, one clean vertical descent of this length)."""
        if self.preplace_support_clearance_m is not None:
            return float(self.preplace_support_clearance_m)
        return float(self.place_above_clearance_m)


def resolve_level_config(spec):
    """Normalise a ``carton_level`` config value into a ``LevelConfig`` or ``None`` (off =>
    byte-identical). Accepts absent/None/False/'0'/'off' => None; True/'on'/{} => defaults; a
    mapping with optional ``bound_deg``, ``tilt_grace_s``, ``level_goals``, ``trip``, ``level_weld``,
    ``level_place``, ``level_path``, ``place_from_above``, ``path_waypoints``, ``place_descent_speed``,
    ``place_descent_accel``, ``place_above_clearance_m``, ``place_depth_bias_m``, ``preplace_above_place``,
    ``preplace_standoff_m``, ``place_clearance_check``, ``place_clearance_margin_m``, ``place_clearance_drooped``,
    ``closed_loop_level``, ``closed_loop_authority_deg``, ``closed_loop_rate_deg_per_s``, ``closed_loop_gain``,
    ``place_contact_release``, ``place_contact_gap_m``, ``closed_loop_sensor``, ``preplace_above_support``,
    ``preplace_support_clearance_m``, the sim-preplace-level-gate-v1 keys ``preplace_level_gate`` /
    ``preplace_gate_bound_deg`` / ``preplace_max_segment_rad`` / ``preplace_approach_speed`` /
    ``preplace_approach_accel``, ``joints``, ``tool_link``.
    Raises on a malformed value so a typo fails loudly at boot."""
    if spec is None or spec is False:
        return None
    if isinstance(spec, str):
        v = spec.strip().lower()
        if v in ('', '0', 'off', 'false', 'none'):
            return None
        if v in ('1', 'on', 'true'):
            return LevelConfig().validate()
        raise ValueError(f'carton_level: unknown value {spec!r}')
    if spec is True:
        return LevelConfig().validate()
    if not isinstance(spec, dict):
        raise ValueError('carton_level must be a bool, string, or mapping')
    kw = {}
    if 'bound_deg' in spec and spec['bound_deg'] is not None:
        kw['bound_deg'] = float(spec['bound_deg'])
    if 'tilt_grace_s' in spec and spec['tilt_grace_s'] is not None:
        kw['tilt_grace_s'] = float(spec['tilt_grace_s'])
    if 'path_waypoints' in spec and spec['path_waypoints'] is not None:
        kw['path_waypoints'] = int(spec['path_waypoints'])
    for fnum in ('place_descent_speed', 'place_descent_accel', 'place_above_clearance_m',
                 'place_depth_bias_m', 'preplace_standoff_m', 'place_clearance_margin_m',
                 'closed_loop_authority_deg', 'closed_loop_rate_deg_per_s', 'closed_loop_gain',
                 'place_contact_gap_m', 'preplace_gate_bound_deg', 'preplace_max_segment_rad',
                 'preplace_approach_speed', 'preplace_approach_accel', 'preplace_max_wrist_step_rad',
                 'preplace_soft_effort_nm', 'preplace_branch_select_budget_s',
                 'pad_down_arrival_shift_m', 'pad_down_arrival_height_m',
                 'pad_down_reorient_speed', 'pad_down_reorient_accel'):
        if fnum in spec and spec[fnum] is not None:
            kw[fnum] = float(spec[fnum])
    for flag in ('level_goals', 'trip', 'level_weld', 'level_place', 'level_path', 'place_from_above',
                 'preplace_above_place', 'place_clearance_check', 'place_clearance_drooped',
                 'closed_loop_level', 'place_contact_release', 'closed_loop_sensor',
                 'preplace_above_support', 'preplace_level_gate', 'preplace_reaction_branch',
                 'preplace_seeded_interior', 'preplace_branch_prefer_carry_connected',
                 'preplace_descent_feasible',
                 'place_release_on_contact', 'release_cancel_handshake', 'release_reissue',
                 'release_free_check', 'place_seat_gate', 'place_pad_down_arrival'):
        if flag in spec and spec[flag] is not None:
            kw[flag] = bool(spec[flag])
    # sim-place-release-on-contact-v1 / sim-place-release-cancel-handshake-v1 / sim-release-open-reissue-v1
    for fnum in ('place_release_effort_margin_nm', 'place_release_stall_m', 'place_obstruction_gap_mm',
                 'place_reapproach_shift_m', 'release_cancel_stopped_vel', 'release_confirm_timeout_s',
                 'release_free_stable_eps_m', 'release_free_rest_gap_m', 'release_free_lift_dz_m',
                 'release_free_follow_frac', 'place_seat_tol_m'):
        if spec.get(fnum) is not None:
            kw[fnum] = float(spec[fnum])
    for fint in ('place_release_effort_consecutive', 'place_release_stall_ticks', 'place_reapproach_max',
                 'release_cancel_stopped_ticks', 'release_reissue_log_every',
                 'release_free_settle_ticks', 'release_free_settle_window_ticks',
                 'pad_down_reorient_waypoints'):
        if spec.get(fint) is not None:
            kw[fint] = int(spec[fint])
    if spec.get('place_release_contact_gap_mm') is not None:
        kw['place_release_contact_gap_mm'] = float(spec['place_release_contact_gap_mm'])
    # sim-place-verify-rest-tolerance-v1 (nullable metres; absent => None => stall trigger unchanged)
    if spec.get('place_release_stall_height_margin_m') is not None:
        kw['place_release_stall_height_margin_m'] = float(spec['place_release_stall_height_margin_m'])
    if 'place_support_tokens' in spec and spec['place_support_tokens'] is not None:
        _pst = spec['place_support_tokens']
        # Pass the value through (even a malformed one) so validate() rejects a string / empty / non-list
        # by name, instead of silently splitting a string into characters or ignoring an empty list.
        kw['place_support_tokens'] = (_pst if isinstance(_pst, str)
                                      else tuple(str(t) for t in _pst) if hasattr(_pst, '__iter__')
                                      else _pst)
    # sim-place-preplace-above-support-v1: nullable clearance (None => place_above_clearance_m).
    if spec.get('preplace_support_clearance_m') is not None:
        kw['preplace_support_clearance_m'] = float(spec['preplace_support_clearance_m'])
    if spec.get('joints'):
        kw['joints'] = tuple(str(j) for j in spec['joints'])
    if spec.get('tool_link'):
        kw['tool_link'] = str(spec['tool_link'])
    # sim-carton-level-southdock-v1: loaded gravity-droop feedforward for the level place IK (off => None).
    if spec.get('gravity_compensation') is not None:
        kw['gravity_compensation'] = resolve_gravity_compensation(
            spec['gravity_compensation'], tool_link=kw.get('tool_link', LevelConfig.tool_link))
    return LevelConfig(**kw).validate()


# Wrist seeds (rad, ABSOLUTE targets for the first adjustable joint) the multi-start sweep restarts
# from. The first adjustable joint (wrist_1) dominates the carton attitude, so sweeping it around the
# circle lands a seed in the basin of every level branch; the goal's own wrists (offset 0) are tried
# first so a pose that is already near level converges fast and nearest to the plan.
_W1_SEED_GRID = tuple(np.linspace(-math.pi, math.pi, 12))
_W2_SEED_GRID = (0.0, math.pi / 2, math.pi)


def level_config(robot, goal_config, up_in_tool, *, joints=DEFAULT_LEVEL_JOINTS,
                 tool_link='ur_tool0', root='ur_base_link', limits=None,
                 max_iter=80, tol_deg=0.05, accept_deg=None, base_R=None, fast=False):
    """Re-solve ``joints`` of ``goal_config`` so the carried carton's up-axis points world-up (a
    level-orientation waypoint). Returns a new joint map. Position (pan/lift/elbow) is untouched;
    only the wrist cluster moves, so the tool stays where the carry plan put it and the carton is
    level.

    sim-carton-tilt-trip-margin-v1: the original single-start Gauss-Newton minimised the 2-D
    horizontal residual ``up_world x world_up``, which VANISHES at BOTH the upright pole (up_world =
    +Z, tilt 0) AND the inverted pole (up_world = -Z, tilt 180). From a carried fold whose start tilt
    is already past ~90 deg -- which the collect carry fold (carry_lift_shape, tuck/pan,
    collect_place_preplace_shape) is for the real top-grasp up-axis -- the descent fell into the
    inverted basin and returned "no level solution (residual tilt 180 deg)" even though a clear
    wrist-only level branch existed (coldwin24 / coldwin24b CARTON_TILT at the collect lift). The fix:
      * a full **3-D** orientation residual ``up_world - world_up`` that is zero ONLY at the upright
        pole and strongly NON-zero (and repulsive) at the inverted pole, and
      * **multi-start** restarts over the attitude wrists so a wrist-IK local minimum cannot hide an
        existing level branch.
    Returns the lowest-tilt in-``limits`` solution (preferring a fully level one, then the one nearest
    the planned goal). Succeeds when the best reaches ``tol_deg`` OR, when ``accept_deg`` is given, when
    it is within ``accept_deg`` (level as far as the clear wrist region allows -- the caller's carried
    bound). Raises only when no in-limits config gets under that acceptance (genuinely un-levelable in
    the clear region -- caller fails honestly / the CARTON_TILT trip owns it)."""
    cfg = _as_joint_map(goal_config)
    adj = [j for j in joints if j in cfg]
    if not adj:
        raise ValueError('no adjustable joints present in the goal config')
    u = _unit(up_in_tool)
    accept = float(tol_deg if accept_deg is None else accept_deg)
    # sim-preplace-select-live-cost-v1: the cached tool-chain FK (same finite-difference arithmetic, only
    # the root->tool_link chain instead of the whole-robot tree) when fast; None => the slow whole-tree FK.
    chain = _chain_for(robot, tool_link, root) if fast else None

    def up_world(cfgmap):
        return _apply_base(base_R, tool_rotation(robot, cfgmap, tool_link, root) @ u)

    def tilt_deg(cfgmap):
        uw = up_world(cfgmap)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(uw[2])))))

    def clamp(cfgmap):
        if limits:
            for j in adj:
                if j in limits:
                    lo, hi = limits[j]
                    cfgmap[j] = min(max(cfgmap[j], lo), hi)
        return cfgmap

    def _tilt_uw(uw):
        return math.degrees(math.acos(max(-1.0, min(1.0, float(uw[2])))))

    def descend(start):
        m = clamp(dict(start))
        step_fd = 1e-4
        for _ in range(int(max_iter)):
            if chain is not None:
                T, prefix, suffix = chain.evaluate(m)
                uw = _apply_base(base_R, T[:3, :3] @ u)
                if _tilt_uw(uw) <= tol_deg:
                    break
                r0 = uw - WORLD_UP
                J = np.zeros((3, len(adj)))
                for k, j in enumerate(adj):
                    Tp = chain.perturbed_tool(prefix, suffix, j, m[j] + step_fd)
                    uwp = _apply_base(base_R, Tp[:3, :3] @ u)
                    J[:, k] = (uwp - WORLD_UP - r0) / step_fd
            else:
                if tilt_deg(m) <= tol_deg:
                    break
                r0 = up_world(m) - WORLD_UP  # 3-D residual: zero only upright, repulsive at the 180 pole
                J = np.zeros((3, len(adj)))
                for k, j in enumerate(adj):
                    m2 = dict(m)
                    m2[j] = m[j] + step_fd
                    J[:, k] = (up_world(m2) - WORLD_UP - r0) / step_fd
            lam = 1e-6
            dq = -np.linalg.solve(J.T @ J + lam * np.eye(len(adj)), J.T @ r0)
            mx = float(np.max(np.abs(dq))) if dq.size else 0.0
            if mx > 0.3:  # bound the step so FK stays in its linear regime
                dq *= 0.3 / mx
            for k, j in enumerate(adj):
                m[j] = m[j] + float(dq[k])
            clamp(m)
        if chain is not None:
            T, _, _ = chain.evaluate(m)
            return m, _tilt_uw(_apply_base(base_R, T[:3, :3] @ u))
        return m, tilt_deg(m)

    # Seed combos: the goal itself first (offset 0 -> the fast, nearest-to-plan path for an
    # already-near-level pose), then wrist_1 swept around the circle (x wrist_2 fallbacks) so a seed
    # lands in the upright basin of every branch.
    seeds = [{}]
    for a in _W1_SEED_GRID:
        seeds.append({adj[0]: a})
    if len(adj) > 1:
        for a in _W1_SEED_GRID[::2]:
            for b in _W2_SEED_GRID:
                seeds.append({adj[0]: a, adj[1]: b})

    best = None  # (rank_key, cfgmap, tilt)
    for seed in seeds:
        start = dict(cfg)
        start.update(seed)   # seed entries are ABSOLUTE joint targets
        sol, t = descend(start)
        if t <= accept:
            disp = sum((sol[j] - cfg[j]) ** 2 for j in adj)
            key = (0 if t <= tol_deg else 1, disp)   # prefer fully level, then nearest the plan
            if best is None or key < best[0]:
                best = (key, sol, t)
            if key[0] == 0 and disp < 1.0:
                break        # a fully level solution close to the plan -- good enough, stop early
    if best is not None:
        return best[1]
    raise ValueError(f'no level-orientation solution for this carried pose within {accept:.2f} deg '
                     f'(best in-limits tilt stays above it with joints {adj})')


def interpolate_level(robot, start_config, goal_config, up_in_tool, *, n=8,
                      joints=DEFAULT_LEVEL_JOINTS, tool_link='ur_tool0', root='ur_base_link',
                      limits=None):
    """A carried move as ``n+1`` waypoints from ``start`` to ``goal``, each LEVELED so roll/pitch
    stay ~0 THROUGHOUT (not only at the endpoints). The position joints (everything not in
    ``joints``) are linearly interpolated start->goal; the wrist cluster is re-solved per waypoint
    to keep the carton level. Includes a level-hold at both ends (the first and last waypoints are
    the leveled start and goal -- the level-hold before lift-off and before set-down). Returns a
    list of joint maps (ARM order keys)."""
    s = _as_joint_map(start_config)
    g = _as_joint_map(goal_config)
    keys = list(s)
    pos_keys = [k for k in keys if k not in joints]
    out = []
    for i in range(int(n) + 1):
        a = i / float(n)
        wp = {k: (1 - a) * s[k] + a * g[k] for k in keys}
        # seed the wrists from the previous leveled waypoint for continuity
        if out:
            for j in joints:
                if j in wp:
                    wp[j] = out[-1][j]
        for k in pos_keys:
            wp[k] = (1 - a) * s[k] + a * g[k]
        out.append(level_config(robot, wp, up_in_tool, joints=joints, tool_link=tool_link,
                                root=root, limits=limits))
    return out


# ------------------------------------------------ leveled WAYPOINT path (sim-carry-level-lift-v1)

def _as_vec(cfg):
    """A config map -> 6-vector in ARM order."""
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    return [float(cfg[j]) for j in ARM_JOINTS]


def predict_path_tilt(robot, waypoints, up_in_tool, *, tool_link='ur_tool0', root='ur_base_link',
                      interior=8, base_R=None):
    """Replay the carton attitude along an EXECUTED leveled waypoint path and return
    ``(max_abs_roll_deg, max_abs_pitch_deg, max_tilt_deg)``.

    ``waypoints`` is a list of 6-vectors (ARM order) the host executor will drive through, each a
    rest-to-rest cubic Hermite segment (zero waypoint velocities => a joint-space smoothstep that visits
    exactly the straight joint segment q[i]->q[i+1]). The set of configs the executor visits between two
    waypoints is therefore the straight joint segment, so linear-interior sampling here is EXACT for the
    executor's max tilt (the smoothstep only reparameterises time, not the joint-value path). ``interior``
    samples per segment catches a tilt that the leveled ENDPOINTS hide but the interpolation passes through
    (the wrist-branch transition). Pure/CPU."""
    return predict_path_tilt_executed(robot, waypoints, up_in_tool, tool_link=tool_link, root=root,
                                      interior=interior, gravity_comp=None, base_R=base_R)


def predict_path_tilt_executed(robot, waypoints, up_in_tool, *, tool_link='ur_tool0',
                               root='ur_base_link', interior=8, gravity_comp=None, base_R=None):
    """``predict_path_tilt`` with the loaded gravity droop applied (sim-carton-level-southdock-v1).

    When ``gravity_comp`` is a ``GravityCompensation`` each sampled COMMANDED config ``q`` is evaluated at
    its EXECUTED pose ``q + gravity_comp.droop(q)`` -- so the prediction is the carton attitude the PHYSICAL
    welded carton will actually reach (the quantity the GPU CARTON_TILT trip and the captain's "parallel to
    the ground" rule both read), NOT the geometric attitude of the pre-tilted command. ``None`` => the
    geometric replay (identical to ``predict_path_tilt``). Pure/CPU."""
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    pts = [list(w) for w in waypoints]

    def executed(q):
        if gravity_comp is None:
            return q
        cfg = dict(zip(ARM_JOINTS, q))
        dq = gravity_comp.droop(robot, cfg, tool_link=tool_link, root=root)
        return [q[i] + dq.get(ARM_JOINTS[i], 0.0) for i in range(6)]

    mr = mp = mt = 0.0
    for i in range(len(pts) - 1):
        for k in range(interior + 1):
            a = k / float(interior)
            q = [(1 - a) * pts[i][j] + a * pts[i + 1][j] for j in range(6)]
            roll, pitch, tilt = carried_rpy(robot, executed(q), up_in_tool, tool_link, root, base_R=base_R)
            mr = max(mr, abs(roll)); mp = max(mp, abs(pitch)); mt = max(mt, tilt)
    return mr, mp, mt


@dataclass(frozen=True)
class LevelPathPlan:
    """A leveled carried-move plan (sim-carry-level-lift-v1).

    * ``waypoints`` -- the ARM-order 6-vectors the host executor drives through (leveled at every
      waypoint); the first is the measured start, the last the leveled goal.
    * ``max_roll`` / ``max_pitch`` / ``max_tilt`` -- the PREDICTED executed attitude extremes (replayed
      over the waypoints AND their straight-segment interiors), so the caller can reject a plan whose
      executed path would leave level before any motion streams.
    * ``waypoints_n`` -- the segment count actually chosen (the lowest-tilt candidate).
    * ``max_wrist_step`` -- the largest per-joint wrist step between consecutive waypoints of the planned
      approach (sim-preplace-wrist-continuity-v1); ``worst_step_wp`` is the 1-based waypoint index where it
      occurs. Default 0.0 for plans that do not track it (e.g. the droop-compensated lift/descent paths).
    """
    waypoints: tuple
    max_roll: float
    max_pitch: float
    max_tilt: float
    waypoints_n: int
    max_wrist_step: float = 0.0
    worst_step_wp: int = 0

    def reject(self, bound_deg):
        """The CARTON_LEVEL_PLAN_REJECTED message when the predicted executed path exceeds ``bound_deg``
        on |roll| or |pitch|, else ``None``. Never widens the bound: the message names the predicted
        peak and the segment count so the lane sees exactly how far over it was."""
        worst = max(self.max_roll, self.max_pitch)
        if worst <= float(bound_deg):
            return None
        return (f'{CARTON_LEVEL_PLAN_REJECTED} carried move would tilt the carton off level: predicted '
                f'peak roll={self.max_roll:.2f} pitch={self.max_pitch:.2f} deg over the '
                f'{self.waypoints_n}-waypoint leveled path exceeds bound {float(bound_deg):.2f} deg '
                f'-- refusing to execute a visibly tilted carry')

    def reject_wrist_step(self, max_step_rad, level_joints=DEFAULT_LEVEL_JOINTS):
        """The CARTON_LEVEL_PLAN_REJECTED message when the planned approach has a per-waypoint wrist step
        above ``max_step_rad`` (sim-preplace-wrist-continuity-v1), else ``None``. A branch jump the soft
        wrist cannot track (coldwin57's class) is refused before motion rather than streamed into a
        tracking fault. Cheap safety: the branch-continuous re-solve already avoids a jump, so this only
        ever fires if a genuinely discontinuous level branch is the sole option at a waypoint.

        sim-preplace-select-live-cost-v1: the message now also NAMES the jump -- the before/after waypoint
        joint vectors, which wrist stepped, and the standoff (final waypoint) -- so the exact branch jump is
        captured in the log even though the plan is refused before any CARTON_TICK streams (the coldwin61
        evidence ask: the joints were otherwise unrecoverable after the fact)."""
        if self.max_wrist_step <= float(max_step_rad):
            return None
        detail = ''
        wp = int(self.worst_step_wp)
        if 1 <= wp < len(self.waypoints):
            from twin.restock_manipulation.cad_tool import ARM_JOINTS
            before = self.waypoints[wp - 1]
            after = self.waypoints[wp]
            idx = {j: k for k, j in enumerate(ARM_JOINTS)}
            jmax = max((j for j in level_joints if j in idx),
                       key=lambda j: abs(after[idx[j]] - before[idx[j]]), default=None)
            jn = jmax if jmax is not None else '?'
            detail = (' joint=%s wp%d=%s wp%d=%s standoff=%s'
                      % (jn, wp - 1, [round(float(v), 4) for v in before],
                         wp, [round(float(v), 4) for v in after],
                         [round(float(v), 4) for v in self.waypoints[-1]]))
        return (f'{CARTON_LEVEL_PLAN_REJECTED} carried approach has a {self.max_wrist_step:.3f} rad wrist '
                f'step at waypoint {self.worst_step_wp}/{self.waypoints_n} (continuity threshold '
                f'{float(max_step_rad):.3f} rad) -- a branch jump the soft wrist cannot track; refusing '
                f'before motion rather than streaming it into a TRAJ_HOST tracking fault' + detail)


def plan_level_path(robot, start_config, goal_config, up_in_tool, *, bound_deg=DEFAULT_TILT_BOUND_DEG,
                    joints=DEFAULT_LEVEL_JOINTS, tool_link='ur_tool0', root='ur_base_link', limits=None,
                    waypoints=12, candidates=(8, 12, 16), interior=8):
    """Plan a carried move from ``start_config`` to ``goal_config`` as a LEVELED WAYPOINT PATH that keeps
    the welded carton parallel to the ground THROUGHOUT (not only at the endpoints).

    The goal is first leveled (``level_config``; raises if it has no in-``limits`` level solution), then
    the move is decomposed into leveled waypoints via ``interpolate_level``. Because a leveled waypoint's
    wrist branch can jump between adjacent waypoints (the carton up-axis manifold is constrained by the
    clear carry wrist region, so a fully level + self-clear + continuous path need not exist), the
    straight-segment INTERIOR between two level waypoints can still tilt; the segment count that minimises
    the predicted interior tilt is chosen from ``candidates`` (|| plus the requested ``waypoints``). The
    returned ``LevelPathPlan`` carries the predicted executed extremes so the caller can
    CARTON_LEVEL_PLAN_REJECTED a plan still over the bound. Raises ValueError only when the GOAL itself
    has no in-limits level solution (genuinely un-levelable -- the caller falls back / fails honestly).
    Pure/CPU."""
    leveled_goal = level_config(robot, list(goal_config), up_in_tool, joints=joints, tool_link=tool_link,
                                root=root, limits=limits, accept_deg=bound_deg)
    gvec = _as_vec(leveled_goal)
    return _select_level_path(robot, start_config, gvec, up_in_tool, bound_deg=bound_deg, joints=joints,
                              tool_link=tool_link, root=root, limits=limits, waypoints=waypoints,
                              candidates=candidates, interior=interior, pin_goal=False, pin_start=False)


def plan_level_path_to_goal(robot, start_config, leveled_goal, up_in_tool, *,
                            bound_deg=DEFAULT_TILT_BOUND_DEG, joints=DEFAULT_LEVEL_JOINTS,
                            tool_link='ur_tool0', root='ur_base_link', limits=None,
                            waypoints=12, candidates=(8, 12, 16), interior=8):
    """Plan a carried APPROACH from ``start_config`` to an ALREADY-LEVELED goal as a LEVELED WAYPOINT PATH
    that keeps the welded carton parallel to the ground THROUGHOUT (not only at the endpoints), WITHOUT
    re-leveling the goal (sim-carton-level-preplace-v1).

    This is the counterpart of ``plan_level_path`` for the carried move INTO the PR 290 place-on-cart
    standoff: ``plan_level_path`` levels its goal wrist-only, which for a reach-authored place standoff
    (``collect_place_preplace_shape`` leveled by the FULL-DOF ``level_place_ik``, PR 290) would pick a
    different wrist branch and move the standoff off its reach-validated TCP. Here the goal is passed in
    already leveled (the PR 290 full-DOF solution) and PINNED verbatim as the final waypoint; only the
    INTERIOR waypoints re-solve the wrist cluster for level, and the FIRST waypoint is pinned to the
    measured start (so the path begins exactly where the arm is -- ``begin_joint_path`` requires it, and
    any genuine start-pose tilt is reported honestly rather than hidden). Returns a ``LevelPathPlan``
    whose predicted executed extremes the caller checks against the bound (CARTON_LEVEL_PLAN_REJECTED).
    Raises ValueError only when an INTERIOR waypoint has no in-limits level solution. Pure/CPU."""
    gvec = _as_vec(_as_joint_map(leveled_goal))
    return _select_level_path(robot, start_config, gvec, up_in_tool, bound_deg=bound_deg, joints=joints,
                              tool_link=tool_link, root=root, limits=limits, waypoints=waypoints,
                              candidates=candidates, interior=interior, pin_goal=True, pin_start=True)


def _select_level_path(robot, start_config, gvec, up_in_tool, *, bound_deg, joints, tool_link, root,
                       limits, waypoints, candidates, interior, pin_goal, pin_start):
    """Shared candidate sweep for ``plan_level_path`` / ``plan_level_path_to_goal``: decompose the move to
    ``gvec`` into leveled waypoints at several segment counts and return the FEWEST-waypoint plan whose
    predicted executed tilt stays under ``bound_deg`` (else the lowest-tilt attempt, so the caller's reject
    gate fires honestly). ``pin_start``/``pin_goal`` overwrite the first/last waypoint with the exact
    measured start / given goal (used when the goal must be preserved verbatim, e.g. the PR 290 standoff)."""
    svec = _as_vec(_as_joint_map(start_config))
    tried = []
    seen = set()
    # Try the requested count first, then the fallbacks, in ascending order so a satisfying plan with
    # the FEWEST waypoints wins (fewer rest-to-rest segments => a shorter executed carry).
    order = sorted({int(n) for n in (int(waypoints),) + tuple(candidates) if int(n) >= 2})
    for n in order:
        if n in seen:
            continue
        seen.add(n)
        wps = interpolate_level(robot, start_config, gvec, up_in_tool, n=n, joints=joints,
                                tool_link=tool_link, root=root, limits=limits)
        vecs = [_as_vec(w) for w in wps]
        if pin_start:
            vecs[0] = list(svec)
        if pin_goal:
            vecs[-1] = list(gvec)
        mr, mp, mt = predict_path_tilt(robot, vecs, up_in_tool, tool_link=tool_link, root=root,
                                       interior=interior)
        plan = LevelPathPlan(waypoints=tuple(tuple(v) for v in vecs),
                             max_roll=mr, max_pitch=mp, max_tilt=mt, waypoints_n=n)
        if max(mr, mp) <= float(bound_deg):
            return plan           # fewest waypoints that keep the predicted carry under the bound
        tried.append(plan)
    # None kept it under the bound: return the lowest-tilt attempt so the caller's reject gate fires
    # honestly with the true minimal residual (never silently execute a tilt).
    return min(tried, key=lambda p: max(p.max_roll, p.max_pitch))


def _descent_tcp_targets(start_tcp, tgt, n, *, approach_from_above=False, above_clearance_m=0.08,
                         depth_bias_m=0.0):
    """The sequence of ``n`` TCP targets (3-vectors) from ``start_tcp`` to ``tgt`` for the set-down.

    Default (``approach_from_above`` False): a single straight line start->tgt (byte-identical to the
    legacy ``tgt_i = (1-a)*start_tcp + a*tgt``). With ``approach_from_above``: an L-path through the clear
    opening ABOVE the support -- first a (near-)horizontal leg at the standoff height to DIRECTLY ABOVE the
    load-centre (``via`` xy = ``tgt`` xy, z = max(start_tcp z, tgt z + clearance)), then a VERTICAL leg down
    to ``tgt`` -- so the carton stays clear of the shelf top/edge until it is laterally inside the footprint,
    then descends straight down and lands flat. Arc-length parameterised so the point density is uniform
    along the two legs. ``depth_bias_m`` shifts the (biased) target DEEPER along the horizontal approach
    direction ``unit(tgt_xy - start_tcp_xy)`` (away from the mouth) so the overhanging gripper panel lands
    clear of the mouth edge. Pure/CPU."""
    start_tcp = np.asarray(start_tcp, float)[:3]
    tgt = np.asarray(tgt, float)[:3]
    if not approach_from_above:
        return [((1.0 - i / float(n)) * start_tcp + (i / float(n)) * tgt) for i in range(1, n + 1)]
    if depth_bias_m:
        horiz = np.array([tgt[0] - start_tcp[0], tgt[1] - start_tcp[1], 0.0])
        hn = float(np.linalg.norm(horiz))
        if hn > 1e-9:
            tgt = tgt + (horiz / hn) * float(depth_bias_m)
    via = np.array([tgt[0], tgt[1], max(float(start_tcp[2]), float(tgt[2]) + float(above_clearance_m))])
    l1 = float(np.linalg.norm(via - start_tcp))
    l2 = float(np.linalg.norm(tgt - via))
    total = l1 + l2
    out = []
    for i in range(1, n + 1):
        if total <= 1e-9:
            out.append(tgt.copy()); continue
        s = (i / float(n)) * total
        if s <= l1 and l1 > 1e-9:
            out.append(start_tcp + (via - start_tcp) * (s / l1))
        elif l2 > 1e-9:
            out.append(via + (tgt - via) * ((s - l1) / l2))
        else:
            out.append(via.copy())
    out[-1] = tgt.copy()   # pin the endpoint exactly on the support target
    return out


def plan_compensated_place_descent(robot, start_config, target_pos, seed_goal, up_in_tool, gc, *,
                                   bound_deg=DEFAULT_TILT_BOUND_DEG, joints=None, tool_link='ur_tool0',
                                   root='ur_base_link', limits=None, waypoints=8, interior=8,
                                   approach_from_above=False, above_clearance_m=0.08, depth_bias_m=0.0):
    """Plan the loaded place-on-cart SET-DOWN descent as a PER-POINT wrist-compensated leveled path
    (sim-carton-level-place-descent-v1).

    The place descent lowers the WELDED carton from the (measured) leveled standoff ``start_config``
    down onto the sensed deck at ``target_pos`` (tool origin in ``root``). The far-extended loaded arm
    droops under gravity, and PR 318's wrist droop-cancel (``GravityCompensation.wrist_feedforward`` via
    ``level_place_ik(gravity_comp=gc)``) cancels it -- but the place descent applied it only at the
    single ENDPOINT of a 2-point straight-line ``begin_joint_goal`` (and only when a gravity_compensation
    was configured, which the default ``carton_level: true`` host config is NOT). Routed only to the
    endpoint, the pre-rotation does not FOLLOW THE EXTENSION through the insert, so the welded carton
    pitches off level as the arm extends into the deepest pose (coldwin34 +2.06 deg at the set-down
    dwell -- the trip site).

    Here the TCP target is interpolated from the standoff TCP to ``target_pos`` and EVERY waypoint is
    solved reach+level+wrist-droop-cancel against ITS OWN loaded droop, so the executed welded carton
    stays parallel to the ground at every interpolated point, not only the endpoint. With
    ``approach_from_above`` (sim-setdown-shelf-contact-v1) the TCP target follows an L-path -- horizontal
    to directly above the support load-centre, then vertical down -- instead of the single diagonal
    straight line, so the carton descends through the CLEAR opening above the support and lands flat
    rather than dragging its bottom-front across the rollwagen shelf's mouth edge (the coldwin33/34/35
    +2.06 deg CONTACT pivot). ``above_clearance_m`` sets the via-point height above the support. The
    first waypoint is pinned to ``start_config`` verbatim (``begin_joint_path`` requires the path to
    start exactly where the arm is, and the standoff's own small executed droop is reported honestly);
    ``seed_goal`` seeds the first reach+level IK and each solve seeds the next for a continuous branch.
    ``gc`` must be a ``GravityCompensation`` (the caller supplies a default when the config omits one).

    Returns a ``LevelPathPlan`` whose ``max_*`` are the PREDICTED EXECUTED extremes under ``gc`` (the
    droop-aware replay -- what the GPU carton and the CARTON_TILT trip read, NOT the pre-tilted
    command's geometric attitude). Raises ValueError when a waypoint has no in-``limits`` reach+level
    solution (the caller falls back to the unleveled straight place / the CARTON_TILT trip owns it).
    Pure/CPU."""
    if gc is None:
        raise ValueError('plan_compensated_place_descent requires a GravityCompensation')
    start = _as_joint_map(start_config)
    svec = _as_vec(start)
    start_tcp = tool_position(robot, start, tool_link, root)
    tgt = np.asarray(target_pos, float)[:3]
    n = max(2, int(waypoints))
    vecs = [list(svec)]
    seed = _as_joint_map(seed_goal)
    tcp_targets = _descent_tcp_targets(start_tcp, tgt, n, approach_from_above=approach_from_above,
                                       above_clearance_m=above_clearance_m, depth_bias_m=depth_bias_m)
    for tgt_i in tcp_targets:
        sol = level_place_ik(robot, tgt_i, up_in_tool, seed, joints=joints, tool_link=tool_link,
                             root=root, limits=limits, gravity_comp=gc)
        vecs.append(_as_vec(sol))
        seed = sol
    mr, mp, mt = predict_path_tilt_executed(robot, vecs, up_in_tool, tool_link=tool_link, root=root,
                                            interior=interior, gravity_comp=gc)
    return LevelPathPlan(waypoints=tuple(tuple(v) for v in vecs), max_roll=mr, max_pitch=mp,
                         max_tilt=mt, waypoints_n=n)


# ----------------------------------- carried pre-place APPROACH gate + dynamics (sim-preplace-level-gate-v1)

def densify_count(start_config, goal_config, max_segment_rad):
    """The FEWEST leveled-approach segments ``n`` (>= 2) whose straight joint interpolation keeps each
    segment's max per-joint step ``|dq|`` at or below ``max_segment_rad`` -- the "no coarse 9-point sweeps"
    densification (sim-preplace-level-gate-v1). A fast preplace sweep (e.g. pan -1.35 rad, wrist_2 +1.64 rad)
    split into many short level segments both keeps the welded carton level THROUGHOUT (each short segment is
    wrist-droop-cancelled) and lowers the executor's per-segment peak joint velocity/acceleration (less
    dynamic deflection of the welded carton). Pure/CPU."""
    s = _as_vec(_as_joint_map(start_config))
    g = _as_vec(_as_joint_map(goal_config))
    if not (math.isfinite(float(max_segment_rad)) and float(max_segment_rad) > 0):
        raise ValueError('max_segment_rad must be a finite positive number of radians')
    dmax = max((abs(g[i] - s[i]) for i in range(len(s))), default=0.0)
    return max(2, int(math.ceil(dmax / float(max_segment_rad))))


def wrist_first_leg_durations(carry, standoff, *, speed=0.05, accel=0.1, max_segment_rad=0.25):
    """The planned SIM durations (seconds) of the two wrist-first preplace legs at the active caps
    (sim-preplace-leg2-level-v1, part 2). Leg 1 sweeps every joint except shoulder_pan from ``carry`` to
    ``standoff`` while the pan holds; leg 2 moves the pan to ``standoff``. Each leg is densified
    (:func:`densify_count` at ``max_segment_rad``) and timed with the EXACT executor rest-to-rest
    segment-time formula (:func:`_waypoints_segment_times`, the same ``ros_adapter`` the real move runs), so
    the sum is the planned preplace duration the phase budget must cover. Returns ``(leg1_s, leg2_s)``.
    Pure/CPU."""
    c = _as_vec(_as_joint_map(carry))
    g = _as_vec(_as_joint_map(standoff))
    leg1_goal = list(g); leg1_goal[0] = c[0]            # pan held at carry through leg 1

    def leg_time(a, b):
        n = densify_count(a, b, max_segment_rad)
        wps = [[(1 - t / float(n)) * a[i] + (t / float(n)) * b[i] for i in range(6)] for t in range(n + 1)]
        return float(sum(_waypoints_segment_times(wps, speed, accel)))

    return leg_time(c, leg1_goal), leg_time(leg1_goal, g)


def preplace_phase_budget(carry, standoff, *, speed=0.05, accel=0.1, max_segment_rad=0.25,
                          margin_s=90.0, floor_s=120.0):
    """The wrist-first preplace PHASE budget (SIM seconds): ``max(floor_s, leg1 + leg2 + margin_s)``
    (sim-preplace-leg2-level-v1, part 2). The two-leg wrist-first sequence legitimately needs ~150 s at the
    caps -- well over the legacy hard-coded 120 s phase_timeout (coldwin56 timed out at 58 % of leg 2) --
    so the budget is derived from the planned leg durations (:func:`wrist_first_leg_durations`) plus a
    configurable margin, never below ``floor_s`` (so a short plan never shrinks the historical 120 s).
    Returns ``(budget_s, plan_s, leg1_s, leg2_s)``. Pure/CPU."""
    l1, l2 = wrist_first_leg_durations(carry, standoff, speed=speed, accel=accel,
                                       max_segment_rad=max_segment_rad)
    plan = l1 + l2
    return max(float(floor_s), plan + float(margin_s)), plan, l1, l2


def plan_compensated_approach(robot, start_config, leveled_goal, up_in_tool, gc, *, joints=None,
                              tool_link='ur_tool0', root='ur_base_link', limits=None,
                              bound_deg=DEFAULT_TILT_BOUND_DEG, max_segment_rad=0.25, interior=8,
                              level_joints=DEFAULT_LEVEL_JOINTS):
    """Plan the carried PRE-PLACE APPROACH from ``start_config`` (the measured carry/tuck pose) to an
    ALREADY-LEVELED standoff ``leveled_goal`` as a DROOP-COMPENSATED, DENSIFIED leveled waypoint path
    (sim-preplace-level-gate-v1).

    The counterpart of :func:`plan_level_path_to_goal` for the standoff approach, but it closes the two gaps
    coldwin50 exposed:

      * **droop-compensated interior** -- ``plan_level_path_to_goal`` levels the interior wrists only
        GEOMETRICALLY (``interpolate_level`` -> ``level_config``), so the EXECUTED (drooped) interior still
        pitches ~1.8 deg at the far-extended near-standoff configs, and its gate replays the GEOMETRIC path
        (droop-blind). Here each interior waypoint is geometrically leveled THEN wrist-droop-cancelled against
        its OWN loaded droop (``gc.wrist_feedforward``, exactly like the place descent), so the EXECUTED carton
        stays parallel to the ground along the whole approach, not just the endpoints;
      * **densified** -- the position joints interpolate over ``densify_count(start, goal, max_segment_rad)``
        segments (the 132-point lesson), so the fast pan/wrist sweep is split into short level segments (lower
        executor per-segment peak joint velocity + less dynamic deflection, and the leveled interior is sampled
        finely).

    The first waypoint is pinned to ``start_config`` verbatim (``begin_joint_path`` requires the path to start
    exactly where the arm is, and any genuine start-pose droop is reported honestly) and the last to
    ``leveled_goal`` (the executed-level standoff the caller already solved). ``gc`` must be a
    ``GravityCompensation``. Returns a ``LevelPathPlan`` whose ``max_*`` are the PREDICTED EXECUTED (droop-aware)
    extremes -- the quantity the caller gates against ``bound_deg`` (CARTON_LEVEL_PLAN_REJECTED). Raises
    ValueError only when an interior waypoint has no in-``limits`` reach+level+droop-cancel solution (the caller
    falls back / the CARTON_TILT trip owns it). Pure/CPU."""
    if gc is None:
        raise ValueError('plan_compensated_approach requires a GravityCompensation')
    start = _as_joint_map(start_config)
    svec = _as_vec(start)
    gvec = _as_vec(_as_joint_map(leveled_goal))
    n = densify_count(svec, gvec, max_segment_rad)
    vecs = [list(svec)]
    seed = None
    for i in range(1, n + 1):
        a = i / float(n)
        base = {j: (1 - a) * svec[k] + a * gvec[k] for k, j in enumerate(_ARM_ORDER())}
        if seed is not None:
            # continuity: seed the level solve from the previous leveled waypoint's wrists
            for j in level_joints:
                if j in base and j in seed:
                    base[j] = seed[j]
        glev = level_config(robot, base, up_in_tool, joints=level_joints, tool_link=tool_link,
                            root=root, limits=limits, accept_deg=bound_deg)
        comp = gc.wrist_feedforward(robot, glev, up_in_tool, tool_link=tool_link, root=root,
                                    limits=limits, accept_deg=bound_deg)
        vecs.append(_as_vec(comp))
        seed = comp
    vecs[-1] = list(gvec)   # pin the executed-level standoff verbatim
    mr, mp, mt = predict_path_tilt_executed(robot, vecs, up_in_tool, tool_link=tool_link, root=root,
                                            interior=interior, gravity_comp=gc)
    return LevelPathPlan(waypoints=tuple(tuple(v) for v in vecs), max_roll=mr, max_pitch=mp,
                         max_tilt=mt, waypoints_n=n)


def level_branches(robot, config, up_in_tool, *, joints=DEFAULT_LEVEL_JOINTS, tool_link='ur_tool0',
                   root='ur_base_link', limits=None, accept_deg=DEFAULT_TILT_BOUND_DEG, base_R=None,
                   tol_deg=0.05, max_iter=80):
    """Enumerate the DISTINCT level-orientation wrist branches for ``config`` (sim-preplace-wrist-continuity-v1).

    Same multi-start Gauss-Newton descent as :func:`level_config`, but instead of returning the single best
    branch this collects EVERY distinct in-``limits`` solution whose tilt is within ``accept_deg`` (deduped by
    the rounded re-solved joint values). Returns a list of ``(cfgmap, tilt_deg)``. The caller ranks them (by
    continuity to a previous waypoint, by holding torque, ...). Pure/CPU."""
    cfg = _as_joint_map(config)
    adj = [j for j in joints if j in cfg]
    if not adj:
        raise ValueError('no adjustable joints present in the config')
    u = _unit(up_in_tool)
    accept = float(accept_deg)
    seeds = [{}]
    for a in _W1_SEED_GRID:
        seeds.append({adj[0]: a})
    if len(adj) > 1:
        for a in _W1_SEED_GRID[::2]:
            for b in _W2_SEED_GRID:
                seeds.append({adj[0]: a, adj[1]: b})
    out = []
    seen = set()
    for seed in seeds:
        start = dict(cfg)
        start.update(seed)
        sol, t = _level_descend(robot, u, adj, start, tool_link=tool_link, root=root, limits=limits,
                                base_R=base_R, max_iter=max_iter, tol_deg=tol_deg)
        if t <= accept:
            key = tuple(round(float(sol[j]), 3) for j in adj)
            if key not in seen:
                seen.add(key)
                out.append((sol, t))
    return out


def _level_descend(robot, u, adj, start, *, tool_link='ur_tool0', root='ur_base_link', limits=None,
                   base_R=None, max_iter=80, tol_deg=0.05, fast=False):
    """One multi-start Gauss-Newton descent of a single seed to a level-orientation config, shared by
    :func:`level_branches` and :func:`continuous_level_solve`. ``u`` is the unit carton up-axis in the tool
    frame, ``adj`` the adjustable (wrist) joints. Returns ``(cfgmap, tilt_deg)``. The math is IDENTICAL to the
    descent inside :func:`level_config` (3-D orientation residual, step-bounded, clamped to ``limits``).
    ``fast`` routes the FK through the cached tool-chain (sim-preplace-select-live-cost-v1)."""
    chain = _chain_for(robot, tool_link, root) if fast else None

    def up_world(cfgmap):
        return _apply_base(base_R, tool_rotation(robot, cfgmap, tool_link, root) @ u)

    def tilt_deg(cfgmap):
        uw = up_world(cfgmap)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(uw[2])))))

    def clamp(cfgmap):
        if limits:
            for j in adj:
                if j in limits:
                    lo, hi = limits[j]
                    cfgmap[j] = min(max(cfgmap[j], lo), hi)
        return cfgmap

    m = clamp(dict(start))
    step_fd = 1e-4
    for _ in range(int(max_iter)):
        if chain is not None:
            T, prefix, suffix = chain.evaluate(m)
            uw = _apply_base(base_R, T[:3, :3] @ u)
            if math.degrees(math.acos(max(-1.0, min(1.0, float(uw[2]))))) <= tol_deg:
                break
            r0 = uw - WORLD_UP
            J = np.zeros((3, len(adj)))
            for k, j in enumerate(adj):
                Tp = chain.perturbed_tool(prefix, suffix, j, m[j] + step_fd)
                J[:, k] = (_apply_base(base_R, Tp[:3, :3] @ u) - WORLD_UP - r0) / step_fd
        else:
            if tilt_deg(m) <= tol_deg:
                break
            r0 = up_world(m) - WORLD_UP
            J = np.zeros((3, len(adj)))
            for k, j in enumerate(adj):
                m2 = dict(m)
                m2[j] = m[j] + step_fd
                J[:, k] = (up_world(m2) - WORLD_UP - r0) / step_fd
        dq = -np.linalg.solve(J.T @ J + 1e-6 * np.eye(len(adj)), J.T @ r0)
        mx = float(np.max(np.abs(dq))) if dq.size else 0.0
        if mx > 0.3:
            dq *= 0.3 / mx
        for k, j in enumerate(adj):
            m[j] = m[j] + float(dq[k])
        clamp(m)
    return m, tilt_deg(m)


def continuous_level_solve(robot, base_config, up_in_tool, prev_wrists, *, joints=DEFAULT_LEVEL_JOINTS,
                           tool_link='ur_tool0', root='ur_base_link', limits=None,
                           accept_deg=DEFAULT_TILT_BOUND_DEG, base_R=None, tol_deg=0.05, max_iter=80,
                           fast=False):
    """BRANCH-CONTINUOUS level-orientation solve (sim-preplace-wrist-continuity-v1).

    Enumerate the level wrist branches for ``base_config`` (multi-start, like :func:`level_config`) and return
    the accepted (tilt <= ``accept_deg``) branch with the SMALLEST per-joint wrist travel from ``prev_wrists``
    (the previous waypoint's solution), NOT the lowest-tilt branch. ``level_config`` ranks by lowest tilt then
    nearest the GOAL, which let coldwin57's leg-2 re-solve prefer a FAR but slightly-more-level branch over the
    near one -- the class of jump this refuses. Returns ``(sol_map, tilt_deg, max_wrist_step)`` where
    ``max_wrist_step = max_j |sol[j] - prev_wrists[j]|`` over the re-solved joints. Raises ValueError when no
    in-``limits`` level branch exists within ``accept_deg``. The caller enforces the per-waypoint step
    threshold (a jump still returns the min-travel branch + its step, so the caller can refuse by name).
    Pure/CPU."""
    cfg = _as_joint_map(base_config)
    adj = [j for j in joints if j in cfg]
    if not adj:
        raise ValueError('no adjustable joints present in the config')
    u = _unit(up_in_tool)
    accept = float(accept_deg)
    prev = {j: float(prev_wrists[j]) for j in adj if j in prev_wrists}
    # Seed from the previous waypoint FIRST (so a continuous branch is enumerated even if the grid misses
    # it), then the wrist_1 circle sweep x wrist_2 fallbacks (so a near-branch local minimum cannot hide an
    # existing level branch -- but the RANKING is min-travel, not min-tilt).
    seeds = []
    if prev:
        seeds.append(dict(prev))
    seeds.append({})
    for a in _W1_SEED_GRID:
        seeds.append({adj[0]: a})
    if len(adj) > 1:
        for a in _W1_SEED_GRID[::2]:
            for b in _W2_SEED_GRID:
                seeds.append({adj[0]: a, adj[1]: b})
    best = None   # (max_step, cfgmap, tilt)
    for seed in seeds:
        start = dict(cfg)
        start.update(seed)
        sol, t = _level_descend(robot, u, adj, start, tool_link=tool_link, root=root, limits=limits,
                                base_R=base_R, max_iter=max_iter, tol_deg=tol_deg, fast=fast)
        if t <= accept:
            step = max((abs(sol[j] - prev[j]) for j in adj if j in prev), default=0.0)
            if best is None or step < best[0]:
                best = (step, sol, t)
    if best is None:
        raise ValueError('no level-orientation branch for this carried pose within '
                         f'{accept:.2f} deg (joints {adj})')
    return best[1], best[2], best[0]


def plan_level_world_approach(robot, start_config, leveled_goal, up_in_tool, *, joints=None,
                              tool_link='ur_tool0', root='ur_base_link', limits=None,
                              bound_deg=DEFAULT_TILT_BOUND_DEG, max_segment_rad=0.25, interior=8,
                              level_joints=DEFAULT_LEVEL_JOINTS, base_R=None,
                              max_wrist_step_rad=DEFAULT_MAX_WRIST_STEP_RAD, anchor='start', fast=False,
                              level_accept_deg=None):
    """Plan the carried pre-place APPROACH from ``start_config`` to an ALREADY-LEVELED standoff
    ``leveled_goal`` as a GEOMETRIC, DENSIFIED, WORLD-level waypoint path (sim-preplace-leg2-level-v1).

    The droop-free counterpart of :func:`plan_compensated_approach`: coldwin56 proved the K=2850 droop
    feed-forward is a PHANTOM at the preplace (the GPU executed joints matched the commanded within
    0.003 rad -> the modelled ~0.016 rad shoulder/elbow sag never materialised), so the gc path pre-tilted
    the standoff wrist_1 by ~+0.031 rad and left the welded carton ~1.8-2.0 deg off. Here each densified
    interior waypoint is GEOMETRICALLY re-levelled in the WORLD frame (``base_R`` folds in the held chassis
    attitude so the carton tracks level in world as the shoulder_pan sweeps on its tilted cone), with NO droop
    pre-tilt.

    sim-preplace-wrist-continuity-v1: the per-waypoint re-solve is BRANCH-CONTINUOUS
    (:func:`continuous_level_solve`): each waypoint's wrist solve is seeded from the previous waypoint and,
    among the level branches, the one with the SMALLEST per-joint wrist travel is chosen -- never a lower-tilt
    FAR branch. The largest per-waypoint wrist step (and its waypoint index) is recorded on the returned plan
    (``max_wrist_step`` / ``worst_step_wp``) so the caller's gate can refuse a plan with a branch jump (the
    soft wrist cannot track one) before motion. A jump is NOT refused here (the plan is still returned so the
    caller reports it by name); only a genuinely un-levelable waypoint raises ValueError.

    The first waypoint is pinned to ``start_config`` verbatim (``begin_joint_path`` requires it) and the last
    to ``leveled_goal`` (the already-world-level standoff); the final pin's step from the last interior
    waypoint is included in ``max_wrist_step``. Returns a ``LevelPathPlan`` whose ``max_*`` are the predicted
    WORLD-frame extremes. Pure/CPU.

    sim-preplace-select-live-cost-v1 -- ``anchor`` addresses the coldwin61 branch jump. The legacy FORWARD
    pass (``anchor='start'``, the DEFAULT -- byte-identical to the prior behaviour) seeds each interior
    waypoint from the previous one STARTING at ``start_config``, so the min-travel chain walks the START-side
    branch and, when the selected standoff sits on a different wrist branch than the one the interior drifted
    onto, the FINAL pin to ``leveled_goal`` jumps (coldwin61: 0.439 rad at waypoint 10/10 ->
    CARTON_LEVEL_PLAN_REJECTED before motion). ``anchor='goal'`` solves the interior BACKWARD from
    ``leveled_goal`` -- seeding each waypoint from its GOAL-side neighbour -- so the interior holds the SELECTED
    standoff's branch and the TERMINAL pin's step is ~0; any residual branch mismatch then surfaces at the
    START pin (where the live arm actually is) instead of at the standoff the soft wrist must hold through
    leg 2. ``anchor='auto'`` (what the controller uses) plans BOTH and chooses GATE-AWARE: it keeps the
    legacy forward plan whenever that already passes both gates (wrist step <= ``max_wrist_step_rad`` AND
    predicted tilt <= ``bound_deg``) -- so an already-acceptable plan is NEVER replaced -- and switches to the
    goal-anchored plan only when it is acceptable and forward is not; if neither passes it returns the
    smaller-wrist-step plan (the caller reports it either way). ``fast`` routes the per-waypoint solves
    through the cached tool-chain FK.

    NOTE (sim-preplace-select-live-cost-v1 CPU finding): when ``start_config`` and ``leveled_goal`` are on
    genuinely DIFFERENT wrist branches (a discrete gap no densification bridges), neither anchor removes the
    jump -- it only MOVES which waypoint crosses it. The real remedy there is to choose a standoff branch
    reachable from the carry pose (the reaction-natural standoff is meant to be exactly that); the reject
    message now names the before/after joints so the next run captures which joint and how far.

    sim-preplace-approach-seeded-v1 -- ``anchor='seeded'`` is the coldwin62 fix for the INTERIOR (not just the
    endpoints). ``'start'`` / ``'goal'`` / ``'auto'`` re-solve each interior waypoint's wrist cluster by a
    FREE level descent (``continuous_level_solve``) which, even min-travel-ranked, can fall into a FAR level
    basin -- so wrist_2 DIPS onto a lower branch at the interior and snaps back near the standoff (coldwin62:
    ~0.03 -> 0.19 -> 0.62 -> 0.66, a 0.433 rad jump refused before motion). ``'seeded'`` instead seeds each
    interior waypoint's wrist solve from the START->STANDOFF wrist INTERPOLATION (the standoff branch family)
    and warm-continuous from the previous waypoint, CLAMPS the per-waypoint wrist step to ``max_wrist_step_rad``
    and holds the executed tilt within ``level_accept_deg`` (the gate bound; ``None`` => ``bound_deg``), and
    DENSIFIES LOCALLY (bisects a base segment, up to ``_SEEDED_MAX_DEPTH`` / ``_SEEDED_MAX_WAYPOINTS``) when a
    segment cannot meet both -- so wrist_2 sweeps MONOTONICALLY carry->standoff with every step <= the
    continuity threshold. Only a genuine branch gap no densification bridges leaves a step > the clamp on the
    returned plan, which the caller's ``reject_wrist_step`` names and refuses. The first waypoint is pinned to
    ``start_config`` and the last to ``leveled_goal`` exactly; the waypoint count is variable (it grows with
    local densification). Scoped to the collect preplace world-level lane; ``'auto'`` stays the default so
    every other caller is byte-identical."""
    start = _as_joint_map(start_config)
    svec = _as_vec(start)
    goal_map = _as_joint_map(leveled_goal)
    gvec = _as_vec(goal_map)
    adj = [j for j in level_joints if j in start]
    n = densify_count(svec, gvec, max_segment_rad)

    def _interp_base(i):
        a = i / float(n)
        return {j: (1 - a) * svec[k] + a * gvec[k] for k, j in enumerate(_ARM_ORDER())}

    def _solve(direction):
        sols = [None] * (n + 1)
        sols[0] = dict(start)        # start pinned (begin_joint_path requires it)
        sols[n] = dict(goal_map)     # the selected world-level standoff, pinned verbatim
        rng = range(n - 1, 0, -1) if direction == 'goal' else range(1, n)
        prev = dict(goal_map) if direction == 'goal' else dict(start)
        for i in rng:
            base = _interp_base(i)
            for j in adj:
                base[j] = prev[j]
            sol, _tilt, _step = continuous_level_solve(
                robot, base, up_in_tool, prev, joints=level_joints, tool_link=tool_link, root=root,
                limits=limits, accept_deg=bound_deg, base_R=base_R, fast=fast)
            sols[i] = sol
            prev = sol
        ms, wp = 0.0, 0
        for i in range(1, n + 1):
            step = max((abs(sols[i][j] - sols[i - 1][j]) for j in adj), default=0.0)
            if step > ms:
                ms, wp = step, i
        return sols, ms, wp

    def _solve_seeded():
        """sim-preplace-approach-seeded-v1: branch-seeded interior solve with local densification.

        Each interior waypoint's wrist cluster is solved by a level descent SEEDED from the START->STANDOFF
        wrist interpolation (so the solution lands on the standoff branch family, not a far level basin),
        warm-continuous from the previous accepted waypoint. A waypoint is accepted only when its per-joint
        wrist step from the previous accepted waypoint is <= ``max_wrist_step_rad`` AND its executed tilt is
        within the level acceptance; otherwise the span to it is bisected (local densification) and retried.
        The start and goal are pinned verbatim. Returns the accepted list of joint maps (variable length)."""
        clamp = float(max_wrist_step_rad)
        level_tol = float(bound_deg if level_accept_deg is None else level_accept_deg)
        u = _unit(up_in_tool)
        order = _ARM_ORDER()

        def _interp_full(t):
            # pan/lift/elbow AND the wrist cluster interpolated: the interpolated wrists seed the level
            # descent onto the standoff branch family (wrist_2 ramps start->standoff monotonically).
            return {j: (1 - t) * svec[k] + t * gvec[k] for k, j in enumerate(order)}

        def _solve_at(t):
            sol, tilt = _level_descend(robot, u, adj, _interp_full(t), tool_link=tool_link, root=root,
                                       limits=limits, base_R=base_R, max_iter=80, tol_deg=0.05, fast=fast)
            return sol, tilt

        sols = [dict(start)]
        # Pending target fractions (ascending), goal last. A failed segment inserts its midpoint ahead.
        pending = [i / float(n) for i in range(1, n)] + [1.0]
        prev, prevt = dict(start), 0.0
        while pending:
            t = pending[0]
            is_goal = abs(t - 1.0) < 1e-12
            if is_goal:
                sol, tilt = dict(goal_map), 0.0   # goal pinned verbatim (already the executed-level standoff)
            else:
                sol, tilt = _solve_at(t)
            step = max((abs(sol[j] - prev[j]) for j in adj), default=0.0)
            level_ok = is_goal or tilt <= level_tol
            if level_ok and step <= clamp + 1e-9:
                sols.append(sol); prev, prevt = sol, t; pending.pop(0)
                continue
            # Need to densify this span. Give up (accept the offending step, so the caller refuses by name)
            # only when the span is already finer than the depth cap or the total waypoint budget is spent.
            span = t - prevt
            min_span = 1.0 / (float(n) * float(2 ** _SEEDED_MAX_DEPTH))
            if span <= min_span or len(sols) >= _SEEDED_MAX_WAYPOINTS:
                sols.append(sol); prev, prevt = sol, t; pending.pop(0)
                continue
            pending.insert(0, prevt + 0.5 * span)
        return sols

    def _build(direction):
        if direction == 'seeded':
            sols = _solve_seeded()
        else:
            sols, _ms, _wp = _solve(direction)
        vecs = [_as_vec(s) for s in sols]
        order = _ARM_ORDER()
        adj_idx = [order.index(j) for j in adj]
        ms, wp = 0.0, 0
        for i in range(1, len(vecs)):
            step = max((abs(vecs[i][k] - vecs[i - 1][k]) for k in adj_idx), default=0.0)
            if step > ms:
                ms, wp = step, i
        mr, mp, mt = predict_path_tilt_executed(robot, vecs, up_in_tool, tool_link=tool_link, root=root,
                                                interior=interior, gravity_comp=None, base_R=base_R)
        return LevelPathPlan(waypoints=tuple(tuple(v) for v in vecs), max_roll=mr, max_pitch=mp,
                             max_tilt=mt, waypoints_n=len(vecs) - 1, max_wrist_step=ms, worst_step_wp=wp)

    if anchor != 'auto':
        return _build(anchor)
    # 'auto': plan both and choose GATE-AWARE so a forward plan that already passes BOTH gates is never
    # replaced (no regression); switch to the goal-anchored plan only when it is acceptable and forward is
    # not; else keep whichever has the smaller wrist step (both fail -- the caller reports it either way).
    fwd = _build('start')
    gol = _build('goal')

    def _ok(p):
        return (p.max_wrist_step <= float(max_wrist_step_rad)
                and max(p.max_roll, p.max_pitch) <= float(bound_deg))

    if _ok(fwd):
        return fwd
    if _ok(gol):
        return gol
    return gol if gol.max_wrist_step < fwd.max_wrist_step else fwd


def _ARM_ORDER():
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    return ARM_JOINTS


def _waypoints_segment_times(waypoints, speed, accel):
    """The per-segment durations the simulation executor (``ros_adapter._waypoints_goal``) assigns to a
    rest-to-rest leveled waypoint path, mirrored VERBATIM so a CPU proof reads the real executed timing:
    ``seg = max(1.0, 1.5*dmax/speed, sqrt(6*dmax/accel))`` with ``speed``/``accel`` clamped to the host
    ceilings (0.08 rad/s, 0.15 rad/s^2). Pure/CPU."""
    speed = min(float(speed), 0.08)
    accel = min(float(accel), 0.15)
    pts = [[float(v) for v in w] for w in waypoints]
    out = []
    for i in range(1, len(pts)):
        dmax = max((abs(b - a) for a, b in zip(pts[i - 1], pts[i])), default=0.0)
        out.append(max(1.0, 1.5 * dmax / speed, math.sqrt(6 * dmax / accel)) if dmax > 0 else 1.0)
    return out


def path_peak_joint_speed(waypoints, *, speed=0.05, accel=0.1):
    """``(peak_joint_speed, peak_joint_accel)`` (rad/s, rad/s^2) the simulation executor drives a rest-to-rest
    leveled ``waypoints`` path at, under the ``_waypoints_goal`` timing. For each segment the rest-to-rest
    cubic peaks are ``speed=1.5*dmax/seg`` and ``accel=6*dmax/seg^2``; the maxima over the path are returned.
    Densifying the path (smaller per-segment ``dmax``) and lowering the ``speed``/``accel`` caps both lower
    these peaks -- the dynamics lever against the welded carton's run-to-run deflection. Pure/CPU."""
    pts = [[float(v) for v in w] for w in waypoints]
    if len(pts) < 2:
        return 0.0, 0.0
    times = _waypoints_segment_times(pts, speed, accel)
    pk_v = pk_a = 0.0
    for i, seg in enumerate(times):
        dmax = max((abs(b - a) for a, b in zip(pts[i], pts[i + 1])), default=0.0)
        if seg > 0 and dmax > 0:
            pk_v = max(pk_v, 1.5 * dmax / seg)
            pk_a = max(pk_a, 6.0 * dmax / (seg * seg))
    return pk_v, pk_a


def timed_path_peak_joint_speed(times, positions):
    """``(peak_joint_speed, peak_joint_accel)`` of an EXTERNALLY-timed joint path (e.g. a recorded cuMotion
    move_group preplace sweep): ``times`` (N,) seconds-from-start, ``positions`` (N, J). The per-segment
    rest-to-rest cubic peaks for the step ``dq`` over duration ``d`` are ``1.5*|dq|/d`` and ``6*|dq|/d^2`` --
    the same reconstruction the executor uses -- so a COARSE fast sweep (few points, big ``dq``, short ``d``)
    reads a high peak and the densified + capped approach reads a low one. Pure/CPU (numpy-free)."""
    t = [float(x) for x in times]
    Q = [[float(v) for v in row] for row in positions]
    pk_v = pk_a = 0.0
    for i in range(len(t) - 1):
        d = t[i + 1] - t[i]
        if d <= 0:
            continue
        dmax = max((abs(b - a) for a, b in zip(Q[i], Q[i + 1])), default=0.0)
        pk_v = max(pk_v, 1.5 * dmax / d)
        pk_a = max(pk_a, 6.0 * dmax / (d * d))
    return pk_v, pk_a


def solve_level_standoff(robot, target_pos, up_in_tool, seed_config, gc, *, joints=None,
                         tool_link='ur_tool0', root='ur_base_link', limits=None, accept_deg=0.3):
    """Solve the collect pre-place STANDOFF so the welded carton's EXECUTED (drooped) pose is level AND
    prefer the IK branch with the SMALLEST predicted executed tilt + the most clearance
    (sim-preplace-level-gate-v1, brief part 1).

    ``level_place_ik(gravity_comp=gc)`` already returns an executed-level standoff, but its internal ranking
    prefers the clear-wrist branch NEAREST the seed, not the one whose EXECUTED tilt is smallest. Here the
    wrist-flip / elbow branches are searched explicitly: each candidate is solved geometrically level + reach
    (``level_place_ik(gravity_comp=None)`` from a wrist-flip/elbow seed) then droop-cancelled onto the wrist
    (``gc.wrist_feedforward``), its predicted executed tilt (``droop_induced_rpy``) measured, and the branch
    with the smallest executed tilt is returned (tie-break: in the clear carry wrist region, then nearest the
    seed). Returns ``(solution_map, (roll, pitch, tilt))`` -- the ARM-order command and its predicted EXECUTED
    attitude. Raises ValueError when no branch yields an in-``limits`` reach+level solution (the caller falls
    back to the fixed shape). Pure/CPU."""
    if gc is None:
        raise ValueError('solve_level_standoff requires a GravityCompensation')
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    cfg = _as_joint_map(seed_config)
    w1, w3 = 'ur_wrist_1_joint', 'ur_wrist_3_joint'
    # Branch seeds: the seed itself FIRST (the nominal reach-validated branch), then the flange-roll
    # (wrist_3) flip sweep x a small wrist_1 sweep -- the DOFs that select the attitude branch without moving
    # the reach (level_place_ik re-solves the rest). The nominal seed is tried first and, when it is already
    # comfortably executed-level (<= accept_deg) on a clear branch, returned immediately, so the common path
    # is one IK; the extra branches only matter when a run's nominal branch droops more than accept_deg.
    seeds = [{}]
    if w3 in cfg:
        for a in _W3_SEED_GRID[::2]:
            seeds.append({w3: a})
    if w3 in cfg and w1 in cfg:
        for a in _W3_SEED_GRID[::4]:
            for b in (-math.pi / 2, math.pi / 2):
                seeds.append({w3: a, w1: cfg[w1] + b})
    best = None   # (rank_key, cfgmap, rpy)
    seen = set()
    for seed in seeds:
        start = dict(cfg)
        start.update(seed)
        try:
            geom = level_place_ik(robot, target_pos, up_in_tool, start, joints=joints,
                                  tool_link=tool_link, root=root, limits=limits, gravity_comp=None)
        except ValueError:
            continue
        cmd = gc.wrist_feedforward(robot, geom, up_in_tool, tool_link=tool_link, root=root,
                                   limits=limits, accept_deg=DEFAULT_TILT_BOUND_DEG)
        cmdv = tuple(round(float(cmd[j]), 4) for j in ARM_JOINTS)
        if cmdv in seen:
            continue
        seen.add(cmdv)
        roll, pitch, tilt = droop_induced_rpy(robot, [cmd[j] for j in ARM_JOINTS], gc,
                                              tool_link=tool_link, root=root, up_in_tool=up_in_tool)
        worst = max(abs(roll), abs(pitch))
        disp = sum((_wrap(cmd[j] - cfg.get(j, 0.0))) ** 2 for j in ARM_JOINTS)
        key = (round(worst, 4), 0 if _in_clear_wrist(cmd) else 1, disp)
        if best is None or key < best[0]:
            best = (key, dict(cmd), (roll, pitch, tilt))
        if worst <= float(accept_deg) and _in_clear_wrist(cmd):
            break   # already comfortably executed-level on a clear branch -- stop searching (fast path)
    if best is None:
        raise ValueError('no executed-level standoff branch at target %s'
                         % list(np.round(np.asarray(target_pos, float)[:3], 4)))
    return best[1], best[2]


# ------------------------------------------- loaded gravity-droop model (sim-carton-level-southdock-v1)
# Default EFFECTIVE droop stiffness (N*m/rad). The welded carton's attitude is the arm's attitude; at a
# FAR-EXTENDED, LOADED place pose the heavy shoulder/elbow pitch chain carries a large gravity torque that
# the finite servo cannot fully reject, so the physical (executed) joints droop BELOW the commanded level
# config and the carton pitches off level EVEN THOUGH the commanded config is geometrically level. This is
# a QUASI-STATIC effect (set by torque/stiffness, NOT joint speed), which is why time-parameterising the
# descent slower (PR 302's place_descent_speed) could not save the bigger south-dock place: at 0.02 rad/s
# the welded carton still pitched +2.07 deg (coldwin31, phase=place) -- the SAME droop the 0.08 rad/s
# collectlevel3 saw (2.22 deg). 2850 N*m/rad is the ONE free parameter, pinned by matching the coldwin31
# executed peak (model: south place TCP shoulder_lift gravity torque 58 N*m -> pitch 2.03 deg ~ the observed
# 2.07; east cart 49 N*m -> 1.86 deg, which collectlevel4 saw as a sub-grace transient and PASSED). The
# integration lane re-pins it from the new per-segment CARTON_SEGMENT_TILT log if the in-sim residual differs.
DEFAULT_DROOP_STIFFNESS = 2850.0
# The UR pitch-chain links whose gravity torque drives the carton droop, in ARM order with the serial-chain
# child link each arm joint carries downstream. Masses/COMs are read from the real mission URDF.
_ARM_CHAIN_LINKS = ('ur_shoulder_link', 'ur_upper_arm_link', 'ur_forearm_link',
                    'ur_wrist_1_link', 'ur_wrist_2_link', 'ur_wrist_3_link')
_GRAVITY = np.array([0.0, 0.0, -9.81])


def arm_gravity_torques(robot, config, *, carton_mass=0.0, tool_link='ur_tool0', root='ur_base_link'):
    """Static gravity torque (N*m) on each of the six arm revolute joints at ``config`` (dict or ARM
    6-vector), from the real URDF link masses/COMs plus an optional welded ``carton_mass`` lumped at
    ``tool_link``. Pure/CPU. Returns a dict joint_name -> torque. The sign convention is the torque the
    joint must RESIST (so a positive value droops the joint toward negative under finite stiffness)."""
    from twin.urdf_probe import link_world_transforms
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    cfg = _as_joint_map(config)
    jmap = {j.name: j for j in robot.joints}
    links = robot.links
    world = link_world_transforms(robot, cfg, root=root)
    masses, coms = {}, {}
    for name, link in links.items():
        inert = getattr(link, 'inertial', None)
        if inert is not None and inert.mass and name in world:
            T = np.asarray(world[name], float)
            masses[name] = float(inert.mass)
            coms[name] = T[:3, :3] @ np.asarray(inert.origin_xyz, float) + T[:3, 3]
    extra = []
    if carton_mass and tool_link in world:
        extra.append((float(carton_mass), np.asarray(world[tool_link], float)[:3, 3]))
    out = {}
    for idx, jn in enumerate(ARM_JOINTS):
        joint = jmap.get(jn)
        if joint is None or joint.child not in world:
            out[jn] = 0.0
            continue
        Tc = np.asarray(world[joint.child], float)
        origin = Tc[:3, 3]
        axis = Tc[:3, :3] @ np.asarray(joint.axis, float)
        torque = 0.0
        for ln in _ARM_CHAIN_LINKS[idx:]:
            if ln in masses:
                torque += float(np.dot(axis, np.cross(coms[ln] - origin, masses[ln] * _GRAVITY)))
        for m, p in extra:
            torque += float(np.dot(axis, np.cross(p - origin, m * _GRAVITY)))
        out[jn] = torque
    return out


@dataclass(frozen=True)
class GravityCompensation:
    """Resolved loaded gravity-droop feedforward settings (sim-carton-level-southdock-v1). Off => None.

    ``stiffness`` is the effective droop stiffness (N*m/rad); ``carton_mass`` the welded handling-unit mass
    lumped at the tool for the torque; ``max_pretilt_rad`` clamps each joint's modelled droop so a
    mis-calibrated stiffness can never model a large attitude offset (the droop is < 0.04 rad, so a small
    clamp is ample headroom). ``droop(robot, cfg)`` is the per-joint EXECUTED-minus-COMMANDED position offset
    the finite servo leaves under gravity: delta_q = -torque/stiffness (clamped), ARM-order dict.

    ``correct_joints`` (sim-carton-level-setdown-v1) is the joint set the droop-cancelling pre-tilt is
    placed ON -- the faithfully-TRACKING joints. coldwin33 proved the original full-DOF pre-tilt is wrong:
    the level place IK spread the pre-tilt onto the HEAVY shoulder_lift/elbow pitch joints (shoulder_lift
    +0.020 rad, elbow +0.015 rad), but at the far-extended loaded set-down those joints are gravity-pinned
    -- the GPU moved the executed carton by only 0.01 deg for a +2.04 deg command pre-tilt (set-down
    2.07->2.06), i.e. d(executed)/d(command) ~ 0 on the heavy joints. The wrist, which carries ~2 N*m, DOES
    track its command faithfully (coldwin33 lift/tuck/preplace executed matched the geometric plan within
    0.02 deg). So the droop-cancellation must be realised through the WRIST (the correct_joints), not the
    heavy joints that sag back regardless of the fine setpoint. Default = the wrist cluster."""
    stiffness: float = DEFAULT_DROOP_STIFFNESS
    carton_mass: float = 0.45
    max_pretilt_rad: float = 0.12
    tool_link: str = 'ur_tool0'
    correct_joints: tuple = DEFAULT_LEVEL_JOINTS

    def validate(self):
        if not (math.isfinite(self.stiffness) and self.stiffness > 0):
            raise ValueError('gravity_compensation.stiffness must be a finite positive N*m/rad')
        if not (math.isfinite(self.carton_mass) and self.carton_mass >= 0):
            raise ValueError('gravity_compensation.carton_mass must be a finite non-negative kg')
        if not (math.isfinite(self.max_pretilt_rad) and self.max_pretilt_rad > 0):
            raise ValueError('gravity_compensation.max_pretilt_rad must be a finite positive rad')
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        bad = [j for j in self.correct_joints if j not in ARM_JOINTS]
        if bad or not self.correct_joints:
            raise ValueError(f'gravity_compensation.correct_joints must be a non-empty subset of ARM_JOINTS; bad={bad}')
        return self

    def droop(self, robot, config, *, tool_link=None, root='ur_base_link'):
        """ARM-order dict of EXECUTED-minus-COMMANDED joint offsets under the loaded gravity torque at
        ``config``. A joint that must resist a positive torque droops negative: delta_q = -tau/stiffness,
        clamped to +/- max_pretilt_rad. This is the SAG the wrist feedforward must cancel (its magnitude
        still sizes the correction; it is NOT pre-commanded onto the heavy joints -- see ``wrist_feedforward``)."""
        tl = tool_link or self.tool_link
        tau = arm_gravity_torques(robot, config, carton_mass=self.carton_mass, tool_link=tl, root=root)
        lim = float(self.max_pretilt_rad)
        return {j: max(-lim, min(lim, -tau[j] / self.stiffness)) for j in tau}

    def wrist_feedforward(self, robot, command, up_in_tool, *, tool_link=None, root='ur_base_link',
                          limits=None, accept_deg=None, base_R=None):
        """sim-carton-level-setdown-v1: realise the droop-cancellation on the TRACKING ``correct_joints``.

        Given a geometrically-level COMMAND (reach + level, e.g. the output of the gc-free ``level_place_ik``),
        predict the loaded joint SAG (``droop``), form the config the heavy joints will actually SETTLE at
        (``command + sag``), then re-solve ONLY the ``correct_joints`` (the wrist) so the carton is level AT
        THAT SETTLED config. The returned command keeps every non-correct joint at its geometric value (the
        heavy joints sag from there -- a disturbance the fine setpoint cannot null, coldwin33) and pre-rotates
        the wrist to counter the sag-induced carton pitch. Because the wrist tracks its command faithfully, the
        physical welded carton lands level. The correction follows the extension automatically: the sag is
        ``tau(command)/stiffness`` and ``tau`` grows with reach, so a deeper set-down gets a larger wrist
        pre-rotation (the per-waypoint gravity-torque solve the lane asked for). Returns an ARM-order joint map.
        Raises ValueError (via ``level_config``) only when no in-``limits`` wrist solution levels the sag -- the
        caller then falls through / the CARTON_TILT trip owns it (never a silent tilt)."""
        tl = tool_link or self.tool_link
        cmd = _as_joint_map(command)
        sag = self.droop(robot, cmd, tool_link=tl, root=root)
        settled = {j: cmd[j] + sag.get(j, 0.0) for j in cmd}
        wrist_sol = level_config(robot, settled, up_in_tool, joints=self.correct_joints, tool_link=tl,
                                 root=root, limits=limits, accept_deg=accept_deg, base_R=base_R)
        out = dict(cmd)
        for j in self.correct_joints:
            if j in wrist_sol:
                out[j] = wrist_sol[j]
        return out


def resolve_gravity_compensation(spec, *, tool_link='ur_tool0'):
    """Normalise a ``gravity_compensation`` value into a ``GravityCompensation`` or ``None`` (off =>
    byte-identical). Accepts absent/None/False/'0'/'off' => None; True/'on'/{} => defaults; a mapping with
    optional ``stiffness`` (or ``stiffness_nm_per_rad``), ``carton_mass`` (or ``carton_mass_kg``),
    ``max_pretilt_rad``. Raises on a malformed value so a typo fails loudly at boot."""
    if spec is None or spec is False:
        return None
    if isinstance(spec, str):
        v = spec.strip().lower()
        if v in ('', '0', 'off', 'false', 'none'):
            return None
        if v in ('1', 'on', 'true'):
            return GravityCompensation(tool_link=tool_link).validate()
        raise ValueError(f'gravity_compensation: unknown value {spec!r}')
    if spec is True:
        return GravityCompensation(tool_link=tool_link).validate()
    if not isinstance(spec, dict):
        raise ValueError('gravity_compensation must be a bool, string, or mapping')
    kw = {'tool_link': tool_link}
    for key, aliases in (('stiffness', ('stiffness', 'stiffness_nm_per_rad')),
                         ('carton_mass', ('carton_mass', 'carton_mass_kg')),
                         ('max_pretilt_rad', ('max_pretilt_rad',))):
        for a in aliases:
            if a in spec and spec[a] is not None:
                kw[key] = float(spec[a])
                break
    if spec.get('tool_link'):
        kw['tool_link'] = str(spec['tool_link'])
    # sim-carton-level-setdown-v1: the tracking joint set the droop-cancel pre-tilt is placed on (default wrist).
    if spec.get('correct_joints'):
        kw['correct_joints'] = tuple(str(j) for j in spec['correct_joints'])
    return GravityCompensation(**kw).validate()


# ----------------------------------------------------- level-place IK (reach + level)

# Seeds (ABSOLUTE wrist targets) the level-place IK restarts from. For a TOP-DOWN grasp the carton's
# up-axis is FIXED off the tool approach axis, so a ~pi flip of wrist_3 (the flange roll about the
# approach axis) turns an inverted hold upright WITHOUT moving the tool origin (tool0 sits on the
# wrist_3 axis); the sweep lands a seed in the upright basin of every branch so the position+level
# descent cannot hide the holdable level solution behind a reach-authored (inverted) start.
_W3_SEED_GRID = tuple(np.linspace(-math.pi, math.pi, 9))


def level_place_ik(robot, target_pos, up_in_tool, seed_config, *, joints=None,
                   tool_link='ur_tool0', root='ur_base_link', limits=None, max_iter=200,
                   pos_tol=1e-3, tilt_tol_deg=0.05, w_ori=1.0, gravity_comp=None, base_R=None,
                   fast=False):
    """Re-plan a carried place pose that is BOTH at ``target_pos`` (tool origin in ``root``) AND level
    (carton up-axis world +Z) -- the fix PR 287 flagged as a follow-up: wrist-ONLY leveling of a
    reach-authored place shape holds the attitude but shifts the TCP ~0.2-0.5 m and loses the deck
    reach. This is a FULL-DOF (shoulder + elbow + wrist) damped-least-squares IK over a stacked
    residual ``[tool_pos - target_pos (3); w_ori*(up_world - world_up) (3)]`` with multi-start restarts
    over the attitude wrists, so it moves whatever joints are needed to reach the deck AND stay parallel
    to the ground. ``seed_config`` is the reach-validated place shape (dict or 6-vector); its TCP is the
    reach the solution must preserve (pass ``target_pos = tool_position(robot, seed_config)``). Returns
    a joint map (ARM order). Prefers a solution that is level, reaches, is in ``limits``, in the clear
    carry wrist region, and nearest the seed. Raises ValueError when no in-limits pose reaches both
    within tolerance (genuinely unreachable-while-level -- the caller fails honestly / the CARTON_TILT
    trip owns it).

    sim-carton-level-setdown-v1: when ``gravity_comp`` is a ``GravityCompensation`` the pose is first solved
    GEOMETRICALLY (reach + level, no droop), then the droop-cancelling pre-tilt is routed onto the TRACKING
    ``gravity_comp.correct_joints`` (the wrist) by ``GravityCompensation.wrist_feedforward``. This corrects
    the sim-carton-level-southdock-v1 original, which let the full-DOF solve spread the pre-tilt onto the
    HEAVY shoulder_lift/elbow pitch joints: coldwin33 showed those are gravity-pinned at the far-extended
    loaded set-down (the GPU moved the executed carton 0.01 deg for a +2.04 deg command pre-tilt, 2.07->2.06),
    so the wrist -- which tracks faithfully -- must carry the correction. The wrist pre-rotation follows the
    extension (the sag is ``tau/stiffness`` and ``tau`` grows with reach). ``None`` => the geometric level
    solve (byte-identical)."""
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    if gravity_comp is not None:
        geom = level_place_ik(robot, target_pos, up_in_tool, seed_config, joints=joints,
                              tool_link=tool_link, root=root, limits=limits, max_iter=max_iter,
                              pos_tol=pos_tol, tilt_tol_deg=tilt_tol_deg, w_ori=w_ori, gravity_comp=None,
                              base_R=base_R, fast=fast)
        return gravity_comp.wrist_feedforward(robot, geom, up_in_tool, tool_link=tool_link, root=root,
                                              limits=limits, accept_deg=DEFAULT_TILT_BOUND_DEG, base_R=base_R)
    # sim-preplace-select-live-cost-v1: the cached tool-chain FK (same finite-difference arithmetic on just
    # the root->tool_link chain) when fast and no droop model is requested; None => the slow whole-tree FK.
    chain = _chain_for(robot, tool_link, root) if fast else None
    cfg = _as_joint_map(seed_config)
    adj = list(joints) if joints is not None else list(ARM_JOINTS)
    adj = [j for j in adj if j in cfg]
    if not adj:
        raise ValueError('no adjustable joints present in the seed config')
    u = _unit(up_in_tool)
    tgt = np.asarray(target_pos, float)[:3]
    idx = [ARM_JOINTS.index(j) for j in adj]

    def _executed(cfgmap):
        """The physically-executed config under the loaded gravity droop (commanded + droop), or the
        commanded config verbatim when no compensation is requested."""
        if gravity_comp is None:
            return cfgmap
        dq = gravity_comp.droop(robot, cfgmap, tool_link=tool_link, root=root)
        return {j: cfgmap[j] + dq.get(j, 0.0) for j in cfgmap}

    def state(cfgmap):
        T = tool_transform(robot, _executed(cfgmap), tool_link, root)
        return T[:3, 3], _apply_base(base_R, T[:3, :3] @ u)

    def residual(cfgmap):
        p, uw = state(cfgmap)
        return np.concatenate([(p - tgt), w_ori * (uw - WORLD_UP)])

    def clamp(cfgmap):
        if limits:
            for j in adj:
                if j in limits:
                    lo, hi = limits[j]
                    cfgmap[j] = min(max(cfgmap[j], lo), hi)
        return cfgmap

    def _resid_from_T(T):
        p = T[:3, 3]
        uw = _apply_base(base_R, T[:3, :3] @ u)
        return np.concatenate([(p - tgt), w_ori * (uw - WORLD_UP)])

    def descend(start):
        m = clamp(dict(start))
        step_fd = 1e-5
        for _ in range(int(max_iter)):
            if chain is not None:
                T, prefix, suffix = chain.evaluate(m)
                r0 = _resid_from_T(T)
                tilt = math.degrees(math.acos(max(-1.0, min(1.0,
                       float(_apply_base(base_R, T[:3, :3] @ u)[2])))))
                if np.linalg.norm(r0[:3]) <= pos_tol and tilt <= tilt_tol_deg:
                    break
                J = np.zeros((6, len(adj)))
                for k, j in enumerate(adj):
                    Tp = chain.perturbed_tool(prefix, suffix, j, m[j] + step_fd)
                    J[:, k] = (_resid_from_T(Tp) - r0) / step_fd
            else:
                r0 = residual(m)
                if np.linalg.norm(r0[:3]) <= pos_tol and tilt_from(m) <= tilt_tol_deg:
                    break
                J = np.zeros((6, len(adj)))
                for k, j in enumerate(adj):
                    m2 = dict(m)
                    m2[j] = m[j] + step_fd
                    J[:, k] = (residual(m2) - r0) / step_fd
            dq = -np.linalg.solve(J.T @ J + 1e-6 * np.eye(len(adj)), J.T @ r0)
            mx = float(np.max(np.abs(dq))) if dq.size else 0.0
            if mx > 0.3:
                dq *= 0.3 / mx
            for k, j in enumerate(adj):
                m[j] = m[j] + float(dq[k])
            clamp(m)
        return m

    def tilt_from(cfgmap):
        _, uw = state(cfgmap)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(uw[2])))))

    def in_limits(cfgmap):
        if not limits:
            return True
        return all(lo - 1e-9 <= cfgmap[j] <= hi + 1e-9
                   for j in adj if j in limits for lo, hi in [limits[j]])

    # Seeds: the reach shape first (fast, nearest the plan), then the wrist_3 flange-roll sweep (which
    # flips an inverted hold upright without moving the origin) x the wrist_1 attitude sweep.
    seeds = [{}]
    w3 = 'ur_wrist_3_joint'
    w1 = 'ur_wrist_1_joint'
    if w3 in cfg:
        for a in _W3_SEED_GRID:
            seeds.append({w3: a})
    if w3 in cfg and w1 in cfg:
        for a in _W3_SEED_GRID[::2]:
            for b in (-math.pi / 2, 0.0, math.pi / 2):
                seeds.append({w3: a, w1: cfg[w1] + b})

    # wrist_3 is the flange ROLL about the tool approach axis: rolling it does NOT move the tool
    # origin and (for a top grasp) flips an inverted hold upright, while leaving every upstream link
    # -- and thus the reach shape's self-collision / cage / deck-reach validation -- untouched. So a
    # wrist_3 change is near-free; weight it ~0 in the "nearest the reach shape" metric so the solver
    # prefers the minimal flange-roll level solution ON the validated arm branch over a far branch
    # that happens to be numerically closer in raw joint space.
    _w3 = 'ur_wrist_3_joint'
    weight = {j: (0.02 if j == _w3 else 1.0) for j in adj}

    best = None  # (rank, cfgmap)
    for seed in seeds:
        start = dict(cfg)
        start.update(seed)
        sol = descend(start)
        r = residual(sol)
        pos_err = float(np.linalg.norm(r[:3]))
        tilt = tilt_from(sol)
        if pos_err <= pos_tol and tilt <= tilt_tol_deg and in_limits(sol):
            disp = sum(weight[j] * (_wrap(sol[j] - cfg[j])) ** 2 for j in adj)
            rank = (0 if _in_clear_wrist(sol) else 1, disp)
            if best is None or rank < best[0]:
                best = (rank, sol)
            if rank[0] == 0 and disp < 0.25:
                break
    if best is not None:
        return best[1]
    raise ValueError(f'no level place solution at target {tgt.round(4).tolist()} within '
                     f'{pos_tol:.3f} m / {tilt_tol_deg:.2f} deg (joints {adj})')


def _in_clear_wrist(cfgmap, wrist1_cap=0.65, wrist2_min=0.0):
    """Whether a config stays in the bench-proven clear carry wrist region (wrist_1 <= cap,
    wrist_2 >= min, wrapped to (-pi, pi]) so the level pose is not driven into the
    ur_wrist_2-vs-forearm self-collision stop. Used only to RANK level-place solutions."""
    from twin.restock_manipulation.transfer_carry import wrap_angle
    return (wrap_angle(cfgmap.get('ur_wrist_1_joint', 0.0)) <= wrist1_cap + 1e-6
            and wrap_angle(cfgmap.get('ur_wrist_2_joint', 0.0)) >= wrist2_min - 1e-6)


def level_preserve_tcp(robot, config, up_in_tool, *, joints=None, tool_link='ur_tool0',
                       root='ur_base_link', limits=None, **kw):
    """Re-plan ``config`` to keep the carton level WHILE preserving its tool TCP position (the reach).
    The in-place counterpart of ``level_place_ik`` for a reach-authored place SHAPE whose TCP is
    already correct but whose attitude is off level (e.g. collect_place_preplace_shape holds a
    top-grasped carton inverted). Returns a level joint map at the same TCP; raises if none exists."""
    target = tool_position(robot, config, tool_link, root)
    return level_place_ik(robot, target, up_in_tool, config, joints=joints, tool_link=tool_link,
                          root=root, limits=limits, **kw)


# --------------------------------------------------------------------- trip

def tilt_trip(roll_deg, pitch_deg, bound_deg=DEFAULT_TILT_BOUND_DEG):
    """Trip verdict for a carried carton at measured ``roll_deg``/``pitch_deg``. Returns ``None``
    when both are within ``bound_deg`` (level enough), else a ``(message, worst_abs_deg)`` pair; the
    message is the ``CARTON_TILT`` log line the executor prints before failing the move -- naming the
    tilt instead of silently continuing."""
    worst = max(abs(float(roll_deg)), abs(float(pitch_deg)))
    if worst <= float(bound_deg):
        return None
    msg = (f'{CARTON_TILT} carried carton off level: roll={float(roll_deg):+.2f} '
           f'pitch={float(pitch_deg):+.2f} deg exceeds bound {float(bound_deg):.2f} deg')
    return msg, worst


# ----------------------------------------------------- sustained-tilt debounce (grace window)

@dataclass(frozen=True)
class TiltVerdict:
    """The decision for one carried tick (sim-carton-tilt-transient-debounce-v1).

    * ``trip`` -- the ``CARTON_TILT`` log/fault message when the tilt has exceeded the bound
      CONTINUOUSLY for the whole grace window (the caller FAILS the move), else ``None``. The
      message keeps the legacy ``CARTON_TILT carried carton off level: ...`` prefix (so downstream
      reason-string matching is unchanged) and appends the sustained duration + peak.
    * ``transient`` -- the ``CARTON_TILT_TRANSIENT`` log line for a SUB-grace spike that just ENDED
      (returned to within the bound) -- the caller LOGS it and PROCEEDS, else ``None``.

    A tick that is still exceeding but within the grace, or a plainly level tick, yields both
    ``None`` (proceed silently)."""
    trip: object = None        # str | None
    transient: object = None   # str | None


class TiltDebounce:
    """Debounce the CARTON_TILT trip over a SUSTAINED sim-time grace window.

    Fed the measured carton (roll, pitch) each carried tick with the tick's sim time, it fails the
    move only when the tilt stays over the bound CONTINUOUSLY for ``grace_s`` sim-seconds -- so a
    one-to-few-tick tracking/impulse spike (collectacc1's lift re-plan restart) is reported as
    ``CARTON_TILT_TRANSIENT`` with its peak + duration but does NOT abort, while a genuinely off-level
    carry (tilted for the whole multi-second carry) still trips within the window. Any tick within the
    bound RESETS the continuous-exceedance timer (the window must be unbroken).

    ``grace_s <= 0`` reproduces the legacy IMMEDIATE trip on the first crossing tick, so a run with the
    grace disabled is byte-identical to the pre-debounce controller. PURE / CPU unit-tested; the
    controller passes it the physics-step-scaled grace so the window is a constant frame-count budget.
    """

    def __init__(self, bound_deg=DEFAULT_TILT_BOUND_DEG, grace_s=DEFAULT_TILT_GRACE_S):
        self.bound_deg = float(bound_deg)
        self.grace_s = max(0.0, float(grace_s))
        self.reset()

    def reset(self):
        """Clear the continuous-exceedance state (call on a phase/move boundary)."""
        self._start = None          # sim time the current unbroken exceedance began (None => level now)
        self._peak = 0.0            # max worst(|roll|,|pitch|) seen in the current exceedance
        self._peak_rp = (0.0, 0.0)  # (roll, pitch) at the peak
        self._tripped = False       # latched: once sustained-tripped, stay tripped (avoid re-arming)

    @property
    def exceeding(self):
        return self._start is not None

    def observe(self, now, roll_deg, pitch_deg):
        """Advance the debounce one tick. Returns a ``TiltVerdict``. ``now`` is the sim time (seconds)."""
        roll = float(roll_deg)
        pitch = float(pitch_deg)
        worst = max(abs(roll), abs(pitch))
        now = float(now)
        if worst <= self.bound_deg:
            # Within the bound -- end any ongoing exceedance (report it as a sub-grace transient).
            if self._start is not None and not self._tripped:
                dur = max(0.0, now - self._start)
                transient = (f'{CARTON_TILT_TRANSIENT} carried carton spiked off level: peak={self._peak:.2f}'
                             f' deg (roll={self._peak_rp[0]:+.2f} pitch={self._peak_rp[1]:+.2f}) for'
                             f' {dur:.3f}s (<= grace {self.grace_s:.3f}s) -- reported, not a fault')
                self._start = None
                self._peak = 0.0
                self._peak_rp = (0.0, 0.0)
                return TiltVerdict(trip=None, transient=transient)
            self._start = None
            self._peak = 0.0
            self._peak_rp = (0.0, 0.0)
            return TiltVerdict()
        # Exceeding the bound this tick.
        if self._start is None:
            self._start = now
            self._peak = worst
            self._peak_rp = (roll, pitch)
        elif worst > self._peak:
            self._peak = worst
            self._peak_rp = (roll, pitch)
        dur = max(0.0, now - self._start)
        if dur >= self.grace_s and not self._tripped:
            self._tripped = True
            base = tilt_trip(roll, pitch, self.bound_deg)[0]
            msg = (f'{base} (sustained {dur:.3f}s > grace {self.grace_s:.3f}s, peak {self._peak:.2f} deg)'
                   if self.grace_s > 0 else base)
            return TiltVerdict(trip=msg, transient=None)
        return TiltVerdict()


# --------------------------------------------------------------- weld attach level

def weld_attach_level_check(carton_world_wxyz, bound_deg=DEFAULT_TILT_BOUND_DEG):
    """Point 3: CHECK the attach transform at the weld. The SIM_ATTACH_FALLBACK joint captures the
    carton's CURRENT world pose relative to the gripper body (zero correction, zero impulse), so a
    carton that is level in the world at the weld instant is welded level -- and because the carton
    then rigidly follows the tool, a level tool keeps it level. This reads the carton body's world
    quaternion ``carton_world_wxyz`` (Isaac w,x,y,z) at the weld and returns
    ``(roll_deg, pitch_deg, tilt_deg, trip)`` where ``trip`` is the ``CARTON_TILT`` message (or None)
    if the attach is already off level beyond ``bound_deg``. Re-authoring the relative rotation to
    FORCE level is deliberately NOT done: that would yank the carton and pivot the base (the runs
    208-212 impulse) -- the level-hold before lift-off is what makes the tool level at the weld."""
    roll, pitch, tilt = quat_rpy(carton_world_wxyz)
    trip = tilt_trip(roll, pitch, bound_deg)
    return roll, pitch, tilt, (trip[0] if trip else None)


# ------------------------------------------- droop-induced tilt predictor (sim-place-descent-residual-v2)

def reach_xy(robot, config, *, tool_link='ur_tool0', root='ur_base_link'):
    """Horizontal reach (m) of the tool origin at arm ``config`` -- the single quantity the loaded-droop
    tilt scales with (``droop_induced_rpy`` below). Pure/CPU."""
    p = tool_position(robot, config, tool_link, root)
    return float(math.hypot(p[0], p[1]))


def droop_induced_rpy(robot, commanded, gravity_comp, *, tool_link='ur_tool0', root='ur_base_link',
                      up_in_tool=None):
    """(roll_deg, pitch_deg, tilt_deg) the welded carton ACTUALLY reaches under the loaded gravity sag at a
    geometrically-level ``commanded`` config -- the EXECUTED attitude the GPU CARTON_TILT trip reads, NOT
    the (level) attitude of the command (sim-place-descent-residual-v2).

    This is the honest predictor validated against the measured runs: with the DEFAULT K=2850 model it
    reproduces coldwin39's collect set-down EXECUTED pitch (model 2.16 vs measured 2.02 deg at the real
    place reach 1.086 m). It is computed FRAME-INDEPENDENTLY from the sag-induced change in the TOOL
    rotation (``R_exec @ R_cmd^T`` applied to world-up), so it needs no captured weld up-axis when the
    command is level; pass ``up_in_tool`` to measure a command that is itself slightly off level. The
    open-loop wrist feedforward (PR 318/326) did NOT change this executed attitude in sim (d(exec)/d(cmd) ~
    0), so this predictor deliberately reads the UNCOMPENSATED droop of the geometric command -- the
    quantity that actually tripped. Pure/CPU."""
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    cmd = _as_joint_map(commanded)
    dq = gravity_comp.droop(robot, cmd, tool_link=tool_link, root=root)
    execq = {j: cmd[j] + dq.get(j, 0.0) for j in cmd}
    if up_in_tool is None:
        dR = tool_rotation(robot, execq, tool_link, root) @ tool_rotation(robot, cmd, tool_link, root).T
        return up_axis_rpy(dR @ WORLD_UP)
    return carried_rpy(robot, execq, up_in_tool, tool_link, root)


# ------------------------------------------- closed-loop carton-level backstop (sim-place-descent-residual-v2)

@dataclass(frozen=True)
class ClosedLoopConfig:
    """Resolved closed-loop carton-level backstop settings (brief part 3). Off => the controller never
    builds a servo. ``authority_deg`` caps the total wrist correction (the servo can bend the carton by at
    most +-this many degrees, so it can never fight the planner hard); ``rate_deg_per_s`` rate-limits the
    correction per second (no step kick); ``gain`` is the proportional loop gain (0 < g <= 2); ``joints``
    is the wrist cluster the correction is spread over; ``target_deg`` is the attitude the servo holds
    (0 = level)."""
    authority_deg: float = 3.0
    rate_deg_per_s: float = 30.0
    gain: float = 0.6
    joints: tuple = DEFAULT_LEVEL_JOINTS
    tool_link: str = 'ur_tool0'

    @classmethod
    def from_level(cls, lc):
        """Build from a resolved ``LevelConfig`` (its ``closed_loop_*`` fields), or ``None`` when the
        backstop is off."""
        if lc is None or not getattr(lc, 'closed_loop_level', False):
            return None
        return cls(authority_deg=lc.closed_loop_authority_deg,
                   rate_deg_per_s=lc.closed_loop_rate_deg_per_s,
                   gain=lc.closed_loop_gain, joints=lc.joints, tool_link=lc.tool_link)


class CartonLevelServo:
    """A bounded, sim-only closed-loop wrist servo that holds the carried carton level during the collect
    place descent (sim-place-descent-residual-v2, brief part 3).

    Each controller tick it is fed the CURRENT measured arm config (the in-sim state the CARTON_TILT trip
    already reads) and returns a per-joint wrist CORRECTION to add to the commanded trajectory. The
    correction is the damped-least-squares wrist delta that nulls the measured carton roll/pitch, scaled by
    ``gain``, rate-limited to ``rate_deg_per_s`` per call and accumulated under a hard +-``authority_deg``
    clamp. So it can only ever bend the carton attitude by the authority -- it is a BACKSTOP for a residual
    droop-model error, never the primary clearance (parts 1-2 keep the pad off the posts) and never able to
    drive the arm into the posts.

    PURE / CPU. The controller applies the returned offset to the live wrist command; ``replay`` proves the
    convergence offline by stepping a recorded executed descent through the servo with the plan's own
    kinematics (the droop applied to the commanded path). NOTE: labelled sim-only like CAGE_COLLISION and
    opt-in; the open-loop feedforward it backs up was ineffective in sim, so this is an opt-in attempt whose
    only guaranteed property is the OFFLINE convergence + the hard authority/rate bounds."""

    def __init__(self, robot, up_in_tool, cfg: ClosedLoopConfig, *, root='ur_base_link'):
        self.robot = robot
        self.up = _unit(up_in_tool)
        self.cfg = cfg
        self.root = root
        self.joints = tuple(cfg.joints)
        self.reset()

    def reset(self):
        """Clear the accumulated correction (call at the start of each guarded descent)."""
        self._accum = {j: 0.0 for j in self.joints}

    @property
    def accumulated(self):
        return dict(self._accum)

    def _carton_rpy(self, cfgmap):
        return carried_rpy(self.robot, cfgmap, self.up, self.cfg.tool_link, self.root)

    def step(self, measured, dt):
        """Advance the servo one tick and return the ARM-order wrist CORRECTION dict to ADD to the commanded
        config. ``measured`` is the current arm config (dict or 6-vector); ``dt`` is the tick's sim-seconds.

        The correction is clamped so each wrist's accumulated offset stays within +-authority and moves by
        at most rate*dt this tick."""
        cmd = _as_joint_map(measured)
        roll, pitch, _ = self._carton_rpy(cmd)
        # residual to null: the horizontal components of the carton up-axis (the quantity up_axis_rpy reads).
        err = np.array([math.radians(roll), math.radians(pitch)])
        adj = [j for j in self.joints if j in cmd]
        if not adj or float(np.linalg.norm(err)) < 1e-9:
            return {j: self._accum.get(j, 0.0) for j in adj}
        # numeric Jacobian of (roll,pitch) [rad] wrt each wrist joint at the current measured config.
        step_fd = 1e-4
        J = np.zeros((2, len(adj)))
        for k, j in enumerate(adj):
            m2 = dict(cmd); m2[j] = cmd[j] + step_fd
            r2, p2, _ = self._carton_rpy(m2)
            J[:, k] = (np.array([math.radians(r2), math.radians(p2)]) - err) / step_fd
        # DLS wrist delta that reduces the residual, scaled by the loop gain.
        dq = -self.cfg.gain * np.linalg.solve(J.T @ J + 1e-6 * np.eye(len(adj)), J.T @ err)
        auth = math.radians(self.cfg.authority_deg)
        rate = math.radians(self.cfg.rate_deg_per_s) * max(0.0, float(dt))
        for k, j in enumerate(adj):
            target = self._accum.get(j, 0.0) + float(dq[k])
            # rate-limit the change this tick, then hard-clamp the accumulated correction to the authority.
            delta = max(-rate, min(rate, target - self._accum.get(j, 0.0)))
            self._accum[j] = max(-auth, min(auth, self._accum.get(j, 0.0) + delta))
        return {j: self._accum.get(j, 0.0) for j in adj}

    def correct_path(self, commanded_path, gravity_comp, *, dt=1.0 / 60.0, tool_link=None, root=None):
        """Return the CLOSED-LOOP-corrected commanded waypoints for ``commanded_path`` (list of ARM-order
        6-vectors): step the servo waypoint-by-waypoint reading the predicted executed attitude (the loaded
        droop applied to each commanded config) and add the bounded accumulated wrist correction to each
        commanded config. The correction never exceeds the authority / rate bounds, so the returned path is
        the original descent with a bounded wrist bend toward level. Used by the controller to apply the
        backstop to the planned place descent (opt-in). Pure/CPU."""
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        tl = tool_link or self.cfg.tool_link
        rt = root or self.root
        self.reset()
        out = []
        for vec in commanded_path:
            cmd = dict(zip(ARM_JOINTS, [float(v) for v in vec]))
            dq0 = gravity_comp.droop(self.robot, cmd, tool_link=tl, root=rt)
            ex0 = {j: cmd[j] + dq0.get(j, 0.0) for j in cmd}
            corr = self.step(ex0, dt)
            cmd_c = dict(cmd)
            for j, v in corr.items():
                cmd_c[j] = cmd[j] + v
            out.append([cmd_c[j] for j in ARM_JOINTS])
        return out

    def correct_path_sensor_anchored(self, commanded_path, gravity_comp, measured_start, *,
                                     dt=1.0 / 60.0, tool_link=None, root=None):
        """sim-place-descent-residual-v3 (part 3): the SENSOR-ANCHORED closed-loop correction. Identical to
        :meth:`correct_path`, but the per-waypoint executed-attitude ESTIMATE the servo reads is shifted by
        the MEASURED-minus-model up-axis residual at the descent start -- the live FK-of-measured-joints
        attitude (``measured_start``, ARM-order 6-vector or dict, the exact quantity the CARTON_TILT grader
        reads) minus the droop model's prediction for the same config. So a real starting tilt the droop
        model misses is folded into every executed estimate, and the bounded wrist correction accounts for
        it. Falls back to the plain model ``correct_path`` when ``measured_start`` is unusable (None / wrong
        length). The authority / rate bounds are unchanged (the shift only moves the target the servo nulls).
        Pure/CPU."""
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        tl = tool_link or self.cfg.tool_link
        rt = root or self.root
        if measured_start is None:
            return self.correct_path(commanded_path, gravity_comp, dt=dt, tool_link=tl, root=rt)
        try:
            meas = _as_joint_map(measured_start)
        except (ValueError, TypeError):
            return self.correct_path(commanded_path, gravity_comp, dt=dt, tool_link=tl, root=rt)
        # measured carton up-axis vs the model's predicted executed up-axis at the SAME start config.
        meas_up = carried_up_world(self.robot, meas, self.up, tl, rt)
        dq_s = gravity_comp.droop(self.robot, meas, tool_link=tl, root=rt)
        model_start = {j: meas.get(j, 0.0) + dq_s.get(j, 0.0) for j in meas}
        model_up = carried_up_world(self.robot, model_start, self.up, tl, rt)
        # residual rotation (small-angle) measured relative to model, applied additively to each estimate.
        mr, mp, _ = up_axis_rpy(meas_up)
        kr, kp, _ = up_axis_rpy(model_up)
        d_roll, d_pitch = mr - kr, mp - kp
        self.reset()
        out = []
        for vec in commanded_path:
            cmd = dict(zip(ARM_JOINTS, [float(v) for v in vec]))
            dq0 = gravity_comp.droop(self.robot, cmd, tool_link=tl, root=rt)
            ex0 = {j: cmd[j] + dq0.get(j, 0.0) for j in cmd}
            # step the servo against the SENSOR-anchored attitude (model executed + the measured residual).
            corr = self._step_with_bias(ex0, dt, d_roll, d_pitch)
            cmd_c = dict(cmd)
            for j, v in corr.items():
                cmd_c[j] = cmd[j] + v
            out.append([cmd_c[j] for j in ARM_JOINTS])
        return out

    def _step_with_bias(self, measured, dt, bias_roll_deg, bias_pitch_deg):
        """:meth:`step`, but the carton roll/pitch error the servo nulls is shifted by a fixed measured-vs-model
        bias (deg). Used by the sensor-anchored correction so the model estimate carries the real start offset."""
        cmd = _as_joint_map(measured)
        roll, pitch, _ = self._carton_rpy(cmd)
        roll += float(bias_roll_deg)
        pitch += float(bias_pitch_deg)
        err = np.array([math.radians(roll), math.radians(pitch)])
        adj = [j for j in self.joints if j in cmd]
        if not adj or float(np.linalg.norm(err)) < 1e-9:
            return {j: self._accum.get(j, 0.0) for j in adj}
        step_fd = 1e-4
        J = np.zeros((2, len(adj)))
        for k, j in enumerate(adj):
            m2 = dict(cmd); m2[j] = cmd[j] + step_fd
            r2, p2, _ = self._carton_rpy(m2)
            J[:, k] = (np.array([math.radians(r2 + bias_roll_deg),
                                 math.radians(p2 + bias_pitch_deg)]) - err) / step_fd
        dq = -self.cfg.gain * np.linalg.solve(J.T @ J + 1e-6 * np.eye(len(adj)), J.T @ err)
        auth = math.radians(self.cfg.authority_deg)
        rate = math.radians(self.cfg.rate_deg_per_s) * max(0.0, float(dt))
        for k, j in enumerate(adj):
            target = self._accum.get(j, 0.0) + float(dq[k])
            delta = max(-rate, min(rate, target - self._accum.get(j, 0.0)))
            self._accum[j] = max(-auth, min(auth, self._accum.get(j, 0.0) + delta))
        return {j: self._accum.get(j, 0.0) for j in adj}

    def replay_measured(self, executed_configs, *, dt=1.0 / 60.0, tool_link=None, root=None):
        """OFFLINE proof (part 3, SENSOR feed): step a sequence of MEASURED/EXECUTED arm configs
        (``executed_configs`` = list of ARM-order 6-vectors -- real joint-encoder feedback, NOT a model) through
        the servo and return ``(measured_rpy, servoed_rpy)`` per step: the carton (roll, pitch, tilt) the grader
        would read with NO servo vs. with the bounded wrist correction the servo commands from reading that same
        measured attitude. Proves the sensor-fed loop drives the EXECUTED carton tilt DOWN (toward level) within
        the authority and that the correction stays bounded. Pure/CPU."""
        tl = tool_link or self.cfg.tool_link
        rt = root or self.root
        self.reset()
        measured_rpy, servoed_rpy = [], []
        for vec in executed_configs:
            ex = _as_joint_map(vec)
            measured_rpy.append(carried_rpy(self.robot, ex, self.up, tl, rt))
            corr = self.step(ex, dt)
            servoed = dict(ex)
            for j, v in corr.items():
                servoed[j] = ex.get(j, 0.0) + v
            servoed_rpy.append(carried_rpy(self.robot, servoed, self.up, tl, rt))
        return measured_rpy, servoed_rpy

    def replay(self, commanded_path, gravity_comp, *, dt=1.0 / 60.0, tool_link=None, root=None):
        """OFFLINE proof: step a recorded/planned place descent (``commanded_path`` = list of ARM-order
        6-vectors) through the servo with the plan's own kinematics and the loaded droop, and return
        ``(executed_rpy_without_servo, executed_rpy_with_servo)`` -- each a list of (roll, pitch, tilt) per
        waypoint. ``executed = commanded + servo_correction + droop(commanded + correction)``. Proves the
        servo drives the EXECUTED carton tilt down (within the authority) vs the uncompensated droop.
        Pure/CPU."""
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        tl = tool_link or self.cfg.tool_link
        rt = root or self.root
        self.reset()
        no_servo, with_servo = [], []
        for vec in commanded_path:
            cmd = dict(zip(ARM_JOINTS, [float(v) for v in vec]))
            # uncompensated executed (what actually tripped)
            dq0 = gravity_comp.droop(self.robot, cmd, tool_link=tl, root=rt)
            ex0 = {j: cmd[j] + dq0.get(j, 0.0) for j in cmd}
            no_servo.append(carried_rpy(self.robot, ex0, self.up, tl, rt))
            # servo reads the executed attitude and corrects the wrist command, then the arm re-droops
            corr = self.step(ex0, dt)
            cmd_c = dict(cmd)
            for j, v in corr.items():
                cmd_c[j] = cmd[j] + v
            dq1 = gravity_comp.droop(self.robot, cmd_c, tool_link=tl, root=rt)
            ex1 = {j: cmd_c[j] + dq1.get(j, 0.0) for j in cmd_c}
            with_servo.append(carried_rpy(self.robot, ex1, self.up, tl, rt))
        return no_servo, with_servo


# ---------------- reaction-natural standoff branch selector (sim-preplace-wrist-continuity-v1)


class BranchSelectBudgetExceeded(RuntimeError):
    """The reaction-branch standoff search did not finish inside its CPU time-box
    (sim-preplace-branch-select-bounded-v1). Raised by :func:`select_reaction_natural_standoff` so the
    controller REFUSES the preplace before motion with a named reason, instead of blocking the host
    executor past the Isaac HOST_STATUS_WATCHDOG (coldwin59)."""


def wrist_hold_torque_peak(robot, waypoints, soft_joints, *, carton_mass=0.45, tool_link='ur_tool0',
                           root='ur_base_link'):
    """Peak static-gravity holding torque (N*m) on the soft wrist ``soft_joints`` over a leg's ``waypoints``
    (ARM-order 6-vectors). Returns ``(peak_abs_nm, peak_joint, per_joint_peak_nm)``. Uses the geometry+mass
    model :func:`arm_gravity_torques` (the REAL URDF link masses/COMs + the welded carton), NOT the phantom
    droop stiffness -- the "gravity/reaction torque model (geometry and masses, not the phantom K)" the lane
    asked for. Pure/CPU."""
    soft = list(soft_joints)
    per = {j: 0.0 for j in soft}
    for wp in waypoints:
        tau = arm_gravity_torques(robot, list(wp), carton_mass=carton_mass, tool_link=tool_link, root=root)
        for j in soft:
            per[j] = max(per[j], abs(float(tau.get(j, 0.0))))
    pj = max(per, key=per.get) if per else None
    return (per[pj] if pj is not None else 0.0), pj, per


def enumerate_level_place_branches(robot, target_pos, up_in_tool, seed_config, *, joints=None,
                                   tool_link='ur_tool0', root='ur_base_link', limits=None, base_R=None,
                                   fast=False):
    """Enumerate the DISTINCT reach+level standoff branches at ``target_pos`` (sim-preplace-wrist-continuity-v1).

    Runs :func:`level_place_ik` (geometric, world frame, no droop) from several wrist_2 seeds spanning the
    allowed range, so BOTH the high-wrist_2 (~1.63) and low-wrist_2 (~0.4) equivalent-orientation branches
    are found, deduped by the rounded wrist_2 (the branch discriminant for a top-grasped level carton).
    Returns a list of ARM-order joint maps (each reaches ``target_pos`` AND is level). ``fast`` routes each
    :func:`level_place_ik` through the cached tool-chain FK (sim-preplace-select-live-cost-v1). Pure/CPU."""
    cfg = _as_joint_map(seed_config)
    w2 = 'ur_wrist_2_joint'
    seeds = [dict(cfg)]
    if w2 in cfg:
        lo, hi = 0.0, math.pi
        if limits and w2 in limits:
            lo = max(lo, float(limits[w2][0]))
            hi = min(hi, float(limits[w2][1]))
        if hi > lo:
            for v in np.linspace(lo, hi, 7):
                s = dict(cfg)
                s[w2] = float(v)
                seeds.append(s)
    out = []
    seen = set()
    for s in seeds:
        try:
            sol = level_place_ik(robot, target_pos, up_in_tool, s, joints=joints, tool_link=tool_link,
                                 root=root, limits=limits, gravity_comp=None, base_R=base_R, fast=fast)
        except ValueError:
            continue
        key = round(_wrap(float(sol.get(w2, 0.0))), 2)
        if key in seen:
            continue
        seen.add(key)
        out.append(sol)
    return out


# sim-preplace-pan-first-seeded-leg2-v1: the wrist_2 seed values (ABSOLUTE targets, rad) the carry-connected
# enumeration sweeps. ``level_place_ik`` already multi-starts internally over wrist_3 (flange roll) x wrist_1,
# returning the branch NEAREST the seed, so sweeping only the seed wrist_2 (the branch discriminant for a
# top-grasped level carton) spans the equivalent-orientation family cheaply. 5 values x 2 seed bases
# (carry + the reach shape) = 10 level_place_ik calls (~3 s on the real cw72 inputs, comparable to the
# min-|wrist_2| selector's 7-call enumerate), far fewer than a (wrist_2, wrist_3) grid.
_CARRY_CONNECTED_W2_SEEDS = (0.0, 0.2, 0.4, 0.6, 0.8)


def enumerate_carry_connected_branches(robot, target_pos, up_in_tool, seed_config, carry_config, *,
                                       joints=None, tool_link='ur_tool0', root='ur_base_link',
                                       limits=None, base_R=None, fast=True):
    """Enumerate reach+level standoff branches spanning the LOW-wrist_2 family connected to the carry shape
    (sim-preplace-pan-first-seeded-leg2-v1).

    The 7-seed wrist_2 sweep in :func:`enumerate_level_place_branches` (seeded only from the reach shape,
    deduped by wrist_2 alone) misses the carry-connected low-wrist_2 branches at the pan-first standoff
    (coldwin72: it reported min wrist_2 0.668, while reach+level+self-clear branches at wrist_2 ~0.09 /
    wrist_3 ~-0.12 exist -- the ones whose leg-2 the continuity gate accepts). Here :func:`level_place_ik`
    (which already multi-starts over wrist_3 x wrist_1 internally, returning the branch nearest the seed) is
    seeded from BOTH the carry shape and the reach shape, each with wrist_2 swept over
    ``_CARRY_CONNECTED_W2_SEEDS``, so both low- and mid-wrist_2 families are found. Returns a list of
    ARM-order joint maps (reach+level, in ``limits``), deduped by the rounded (wrist_2, wrist_3). Pure/CPU."""
    cfg = _as_joint_map(seed_config)
    carry = _as_joint_map(carry_config)
    w2, w3 = 'ur_wrist_2_joint', 'ur_wrist_3_joint'
    lo2, hi2 = 0.0, math.pi
    if limits and w2 in limits:
        lo2 = max(lo2, float(limits[w2][0]))
        hi2 = min(hi2, float(limits[w2][1]))
    seeds = [dict(carry), dict(cfg)]
    for base in (carry, cfg):
        if w2 in base:
            for a in _CARRY_CONNECTED_W2_SEEDS:
                if lo2 - 1e-9 <= a <= hi2 + 1e-9:
                    s = dict(base)
                    s[w2] = float(a)
                    seeds.append(s)
    out = []
    seen = set()
    for s in seeds:
        try:
            sol = level_place_ik(robot, target_pos, up_in_tool, s, joints=joints, tool_link=tool_link,
                                 root=root, limits=limits, gravity_comp=None, base_R=base_R, fast=fast)
        except ValueError:
            continue
        key = (round(_wrap(float(sol.get(w2, 0.0))), 2), round(_wrap(float(sol.get(w3, 0.0))), 2))
        if key in seen:
            continue
        seen.add(key)
        out.append(sol)
    return out


def select_carry_connected_standoff(robot, target_pos, up_in_tool, seed_config, carry_config, *,
                                    joints=None, level_joints=DEFAULT_LEVEL_JOINTS, soft_joints=None,
                                    tool_link='ur_tool0', root='ur_base_link', limits=None, base_R=None,
                                    carton_mass=0.45, max_segment_rad=0.25,
                                    bound_deg=DEFAULT_TILT_BOUND_DEG,
                                    max_wrist_step_rad=DEFAULT_MAX_WRIST_STEP_RAD,
                                    seeded_interior=False, level_accept_deg=None, effort_limit=56.0,
                                    descent_predictor=None, budget_s=None, clock=None, fast=True):
    """Choose the pre-place standoff on the branch CONNECTED to the carry/tucked shape, so the PAN-FIRST
    leg 2 (carry -> standoff) is a monotonic extension the continuity gate accepts
    (sim-preplace-pan-first-seeded-leg2-v1, the coldwin72 fix).

    coldwin72 -- the first run with every physical fix stacked (pick sealed, pan swept the cage clear with
    the base locked, wrist held) -- was stopped before motion by the continuity gate in the re-ordered
    PAN-FIRST leg 2 (PR 366): the reaction-natural standoff (min |wrist_2| = 0.668) sits on a DIFFERENT
    carton-level basin than the carry/tucked shape (wrist_2 ~0), separated at the standoff pan by the
    wrist_1 self-collision cap. The leg-2 wrist_2 therefore holds ~0 for nine waypoints then JUMPS 0.300 rad
    (> the 0.15 rad continuity limit the soft wrist can track) -> CARTON_LEVEL_PLAN_REJECTED. CPU-proven: no
    continuous monotonic <=0.15 level path connects the two basins at the standoff pan (the crossing tilt
    ridge minimum is ~2.44 deg > the 2.0 bound, for every wrist_1 cap), so the PR 349 seeding cannot help.
    The PR 359 firm wrist hold now covers the soft-wrist drift that originally pushed PR 345 to the 0.668
    settling branch, so the carry-connected (wrist_2 ~0) branch is admissible again.

    Among the reach+level+self-clear branches (:func:`enumerate_carry_connected_branches`, seeded from the
    carry shape so the low-wrist_2 family is found), rank by total wrist travel from the carry shape and
    return the FIRST branch whose leg-2 world-level plan (:func:`plan_level_world_approach`, the SAME anchor
    the controller drives -- ``'seeded'`` when ``seeded_interior`` else ``'auto'``) the continuity gate
    ACCEPTS: every per-waypoint wrist step <= ``max_wrist_step_rad`` AND predicted tilt <= ``bound_deg``. The
    leg-2 start is the carry tucked wrists at the branch's (standoff) shoulder_pan -- the pan-first leg-1
    end. The gate stays the authority; a branch whose leg-2 gate rejects is skipped. ``budget_s`` time-boxes
    the search on ``clock`` (default ``time.perf_counter``) as in :func:`select_reaction_natural_standoff`.

    sim-preplace-branch-descent-feasible-v1 (coldwin79) -- ``descent_predictor``. The leg-2 continuity gate
    alone is NOT sufficient: a branch whose leg 2 is monotonic can still have a PLACE DESCENT (standoff ->
    support) that rolls the carton past ``bound_deg``, which the later CARTON_LEVEL_PLACE_DESCENT gate refuses
    BEFORE motion -- so the run e-stops (coldwin79 accepted the low wrist_2=0.052 branch whose descent rolls
    3.43 deg). When ``descent_predictor`` is supplied (a callable ``standoff_map -> predicted descent tilt
    deg``, the SAME ``plan_compensated_place_descent`` the place gate runs, raising ValueError / returning a
    non-finite value when the descent has no in-limits level solution), the selector evaluates the predicted
    place-descent tilt for every leg-2-accepted candidate, DISCARDS any whose descent exceeds ``bound_deg``,
    and chooses DETERMINISTICALLY by the stable key ``(carry_travel, descent_tilt, leg2_max_wrist_step,
    |wrist_2 - carry_wrist_2|)`` -- least carry travel first (so the choice preserves the carry-connected
    design AND stays identical on a run whose greedy branch was already descent-feasible), then the remaining
    fields break any degenerate carry-travel tie. The ranking is independent of the search time / candidate
    enumeration order. ``descent_predictor=None`` => the legacy greedy first-leg-2-accepted choice, with no
    descent evaluation => byte-identical.

    Returns ``(standoff_map, report)`` with ``report['reason'] = 'carry-connected'`` plus
    ``chosen_wrist_2``, ``carry_wrist_travel``, ``leg2_max_wrist_step``, ``leg2_max_tilt_deg``,
    ``chosen_peak_torque_nm`` / ``chosen_peak_joint`` / ``chosen_margin_nm`` (for the log), ``branches``
    (ranked; each carries ``descent_tilt_deg`` + ``descent_ok`` when a ``descent_predictor`` was used),
    ``descent_feasible_gate`` (bool: whether the descent filter ran), ``chosen_descent_tilt_deg``,
    ``effort_limit_nm``, ``select_seconds``. Raises ValueError when no carry-connected branch yields a
    gate-accepted leg-2 (and, with a ``descent_predictor``, a descent-feasible one); the caller falls back to
    the single ``level_place_ik`` standoff / fails honestly. Pure/CPU."""
    if clock is None:
        clock = time.perf_counter
    t0 = clock()

    def _tick():
        if budget_s is not None and (clock() - t0) > budget_s:
            raise BranchSelectBudgetExceeded(
                'carry-connected standoff search exceeded the %.3f s budget (%.3f s elapsed)'
                % (float(budget_s), clock() - t0))

    soft = tuple(soft_joints) if soft_joints is not None else tuple(level_joints)
    carry = _as_joint_map(carry_config)
    lj = [j for j in level_joints if j in carry]
    branches = enumerate_carry_connected_branches(
        robot, target_pos, up_in_tool, seed_config, carry_config, joints=joints, tool_link=tool_link,
        root=root, limits=limits, base_R=base_R, fast=fast)
    if not branches:
        raise ValueError('no reach+level standoff branch at target %s (carry-connected enumeration)'
                         % list(np.round(np.asarray(target_pos, float)[:3], 4)))

    def _travel(std):
        return sum(abs(_wrap(float(std[j]) - float(carry[j]))) for j in lj if j in std)

    # sim-preplace-branch-descent-feasible-v1: when a descent_predictor is given, sort by a FULLY
    # DETERMINISTIC key so the ranking never depends on the enumerate's dict/set order (the legacy single
    # ``_travel`` key leaves equal-travel ties to the enumeration order). The off path keeps ``_travel`` alone
    # so it is byte-identical.
    carry_w2 = float(carry.get('ur_wrist_2_joint', 0.0))

    def _w3(std):
        return round(_wrap(float(std.get('ur_wrist_3_joint', 0.0))), 6)

    if descent_predictor is not None:
        ranked = sorted(branches, key=lambda std: (round(_travel(std), 9),
                                                   round(_wrap(float(std.get('ur_wrist_2_joint', 0.0))), 6),
                                                   _w3(std)))
    else:
        ranked = sorted(branches, key=_travel)
    anchor = 'seeded' if seeded_interior else 'auto'
    lacc = level_accept_deg if seeded_interior else None
    report_branches = []
    chosen = chosen_plan = None
    chosen_descent = None
    # The chosen candidate's stable-key tuple (with a descent_predictor): (carry_travel, descent_tilt,
    # leg2_max_wrist_step, |wrist_2 - carry_wrist_2|). Least carry travel first, so the choice stays identical
    # when the run's greedy branch was already descent-feasible, and the remaining fields break a degenerate
    # carry-travel tie deterministically.
    chosen_key = None
    # Validate the leg-2 gate of the CLOSEST (least carry travel) branches first. WITHOUT a descent_predictor
    # (byte-identical legacy path) STOP at the first the gate accepts -- the carry-connected branch is by
    # construction a near branch, so this is ~one leg-2 plan on the healthy path. WITH a descent_predictor,
    # also evaluate the place descent from each leg-2-accepted candidate and keep walking only while a later
    # candidate could still beat the best feasible one (travel <= the best feasible travel): once a candidate's
    # travel exceeds the best feasible travel it can never win the carry-travel-primary key, so the walk stops
    # (early exit) -- cost ~the same as the legacy first-accepted on the healthy path. Bound the validations
    # (``_CARRY_CONNECTED_MAX_VALIDATE``) so a pathological input cannot run the expensive seeded plan over
    # every far (high-travel, heavily densified) branch and blow the heartbeat budget.
    for std in ranked[:_CARRY_CONNECTED_MAX_VALIDATE]:
        _tick()
        travel = _travel(std)
        if (descent_predictor is not None and chosen is not None
                and travel > float(chosen_key[0]) + 1e-9):
            break   # no later (higher-travel) candidate can beat the carry-travel-primary key
        # leg-2 start: the carry tucked wrists/lift/elbow at THIS branch's shoulder_pan (the leg-1 end --
        # pan-first leg 1 holds the carry shape and sweeps shoulder_pan to the standoff pan).
        start = dict(carry)
        start['ur_shoulder_pan_joint'] = float(std['ur_shoulder_pan_joint'])
        rec = {'wrist_2': float(std.get('ur_wrist_2_joint', 0.0)), 'carry_wrist_travel': travel,
               'leg2_max_wrist_step': None, 'leg2_max_tilt_deg': None, 'accepted': False, 'chosen': False}
        if descent_predictor is not None:
            rec['descent_tilt_deg'] = None
            rec['descent_ok'] = None
        try:
            plan = plan_level_world_approach(
                robot, start, std, up_in_tool, level_joints=level_joints, tool_link=tool_link, root=root,
                limits=limits, bound_deg=bound_deg, max_segment_rad=max_segment_rad, base_R=base_R,
                max_wrist_step_rad=max_wrist_step_rad, anchor=anchor, fast=fast, level_accept_deg=lacc)
        except ValueError:
            report_branches.append(rec)
            continue
        step_ok = plan.reject_wrist_step(max_wrist_step_rad, level_joints=level_joints) is None
        tilt_ok = plan.reject(bound_deg) is None
        leg2_accepted = step_ok and tilt_ok
        leg2_step = float(plan.max_wrist_step)
        rec['leg2_max_wrist_step'] = leg2_step
        rec['leg2_max_tilt_deg'] = float(max(plan.max_roll, plan.max_pitch))
        rec['accepted'] = leg2_accepted
        if descent_predictor is None:
            report_branches.append(rec)
            if leg2_accepted:
                rec['chosen'] = True
                chosen = std
                chosen_plan = plan
                break
            continue
        # --- descent-feasibility filter (sim-preplace-branch-descent-feasible-v1) ---
        descent_tilt = None
        descent_ok = False
        if leg2_accepted:
            _tick()
            try:
                dt = descent_predictor(std)
                descent_tilt = float(dt)
                descent_ok = math.isfinite(descent_tilt) and descent_tilt <= float(bound_deg) + 1e-9
            except (ValueError, KeyError):
                descent_tilt = None   # no in-limits level descent from this standoff -> infeasible
                descent_ok = False
        rec['descent_tilt_deg'] = descent_tilt
        rec['descent_ok'] = descent_ok
        report_branches.append(rec)
        if leg2_accepted and descent_ok:
            key = (round(travel, 9), round(descent_tilt, 6), round(leg2_step, 6),
                   round(abs(_wrap(float(std.get('ur_wrist_2_joint', 0.0)) - carry_w2)), 6))
            if chosen_key is None or key < chosen_key:
                chosen_key = key
                chosen = std
                chosen_plan = plan
                chosen_descent = descent_tilt
    if chosen is None:
        if descent_predictor is not None:
            n_leg2 = sum(1 for b in report_branches if b.get('accepted'))
            raise ValueError('no carry-connected standoff branch whose pan-first leg-2 the continuity gate '
                             'accepts AND whose place descent stays within the %.2f deg bound; ranked %d '
                             'reach+level branches, validated %d (%d leg-2-accepted, all descent-infeasible)'
                             % (float(bound_deg), len(ranked), len(report_branches), n_leg2))
        raise ValueError('no carry-connected standoff branch whose pan-first leg-2 approach the continuity '
                         'gate accepts (step <= %.3f rad AND tilt <= %.2f deg); ranked %d reach+level '
                         'branches, validated %d' % (float(max_wrist_step_rad), float(bound_deg),
                                                     len(ranked), len(report_branches)))
    # mark the chosen branch in the ranked report (descent path only; the legacy path marked it inline at the
    # break to stay byte-identical)
    if descent_predictor is not None:
        for b in report_branches:
            b['chosen'] = (b['carry_wrist_travel'] == _travel(chosen)
                           and b['wrist_2'] == float(chosen.get('ur_wrist_2_joint', 0.0)))
    # torque for the chosen (reported only, like the min-|wrist_2| selector)
    chosen_start = dict(carry)
    chosen_start['ur_shoulder_pan_joint'] = float(chosen['ur_shoulder_pan_joint'])
    peak, pj, per = wrist_hold_torque_peak(robot, [_as_vec(chosen_start), _as_vec(chosen)], soft,
                                           carton_mass=carton_mass, tool_link=tool_link, root=root)
    report = {'reason': 'carry-connected', 'branches': report_branches,
              'chosen_wrist_2': float(chosen.get('ur_wrist_2_joint', 0.0)),
              'carry_wrist_travel': _travel(chosen),
              'leg2_max_wrist_step': float(chosen_plan.max_wrist_step),
              'leg2_max_tilt_deg': float(max(chosen_plan.max_roll, chosen_plan.max_pitch)),
              'chosen_peak_torque_nm': peak, 'chosen_peak_joint': pj,
              'chosen_margin_nm': float(effort_limit) - peak, 'effort_limit_nm': float(effort_limit),
              'descent_feasible_gate': descent_predictor is not None,
              'chosen_descent_tilt_deg': chosen_descent,
              'select_seconds': clock() - t0, 'full_plan': False}
    return dict(chosen), report


def select_reaction_natural_standoff(robot, target_pos, up_in_tool, seed_config, carry_config, *,
                                     joints=None, level_joints=DEFAULT_LEVEL_JOINTS, soft_joints=None,
                                     tool_link='ur_tool0', root='ur_base_link', limits=None, base_R=None,
                                     carton_mass=0.45, max_segment_rad=0.25, bound_deg=DEFAULT_TILT_BOUND_DEG,
                                     effort_limit=56.0, full_plan=False, budget_s=None, clock=None, fast=True):
    """Choose the pre-place standoff wrist BRANCH the soft wrist_2 drive naturally SETTLES on -- the
    reaction-natural branch (sim-preplace-wrist-continuity-v1, the primary coldwin57 fix).

    coldwin57's leg-2 COMMAND held wrist_2 continuously at the HIGH level branch (1.63), but the soft wrist_2
    drive DIVERGED monotonically DOWN to the LOW branch (~0.33-0.40) under the pan descent, and the executor
    faulted on the executed-vs-commanded divergence (TRAJ_HOST_TRACK_MISS worst_j=4, 1.3 rad). Both branches
    are world-level, so the carton stayed level (peak 0.29 deg) -- the only failure is the wrist_2 drive
    divergence. The fix is to COMMAND the branch the drive naturally holds: among the distinct reach+level
    standoff branches (:func:`enumerate_level_place_branches`, all world-level and in the clear wrist region),
    pick the one with the MINIMUM |wrist_2| -- the bottom of the equivalent-orientation family the drive
    drifts toward, with the least remaining down-drift room (most stable) -- so leg 1 drives to it and leg 2
    holds it with no divergence (chosen wrist_2 ~0.32 == the observed settling point, and a smaller leg-1
    wrist sweep from the carry pose than the 1.63 branch).

    The gravity/geometry holding-torque model (:func:`wrist_hold_torque_peak`, the REAL URDF masses, NOT the
    phantom K) is computed and REPORTED per branch, but it is NOT the selector: measured on the coldwin57
    URDF the peak soft-wrist holding torque is ~0.5-1.0 N*m on EVERY branch -- all far below ``effort_limit``
    (56 N*m URDF), so no branch saturates and the torque does not predict the drift; worse, the minimum-torque
    branch (wrist_2 ~2.79, peak ~0.15 N*m) is the FARTHEST from the carry pose and UPHILL from where the drive
    drifts, so minimising torque would pick the wrong branch. The honest discriminator is the observed
    settling DIRECTION (the drive drifts toward low wrist_2), hence the min-|wrist_2| rule; the torque is
    reported only so the lane sees the margin and the model's inapplicability.

    sim-preplace-branch-select-bounded-v1 -- CHEAP + TIME-BOXED (coldwin59). The ``full_plan=False`` DEFAULT
    validates each branch's leg-2 feasibility with the CHEAP endpoint level-solve (``level_config`` of the
    leg-2 start at the carry pan; the standoff end is already reach+level from the enumerate) and reports the
    holding torque over the 2-point ``[leg-2 start, standoff]`` sweep -- NOT the full densified
    ``plan_level_world_approach`` the selector used to run per branch (8 branches x ~1.7 s each = ~14 s, which
    blocked the host executor for >5 s at the preplace transition and tripped the Isaac HOST_STATUS_WATCHDOG).
    The CHOSEN standoff is unchanged: the branch is decided by the min-|wrist_2| primary key, which the full
    densified leg-2 plan never affected (it only filtered and ranked); the standoff returned is the enumerate
    branch config verbatim either way, so the selection result is byte-identical to the old inline computation
    on the coldwin57 fixture (asserted in ``test_preplace_branch_select_bounded_v1``) at ~0.08 s instead of
    ~14 s. A mid-leg-2 infeasibility that the full plan alone would have caught is still owned by the
    downstream world-level approach gate (``_issue_gated_preplace_approach`` re-plans + rejects before motion).
    ``full_plan=True`` restores the old densified per-branch path (kept for the parity reference).

    ``budget_s`` time-boxes the whole search on the CPU clock ``clock`` (default ``time.perf_counter`` -- the
    search is pure CPU, never a sim event): a search still running after ``budget_s`` raises
    :class:`BranchSelectBudgetExceeded` so the caller REFUSES before motion rather than starving the heartbeat.
    ``None`` => no time-box (legacy).

    sim-preplace-select-live-cost-v1 -- ``fast`` (default True) routes every ``level_place_ik`` /
    ``level_config`` solve through the cached tool-chain FK (the SAME finite-difference arithmetic, only the
    root->tool_link chain instead of the whole 36-link mobile-robot tree that ``link_world_transforms``
    rebuilt each call). coldwin60's select took ~25-28 s on the real place target (NOT the ~0.08 s the
    coldwin57 fixture measured, whose standoff target IS its seed's own TCP so the descent runs zero
    iterations); the cached chain brings it to ~1.5 s with the chosen standoff within ~1e-4 rad of the slow
    FK -- comfortably inside a small ``budget_s``. ``fast=False`` restores the whole-tree FK (the reference
    the identity test compares against).

    Returns ``(standoff_map, report)`` where ``report`` is a dict with ``branches`` (per-branch
    ``{wrist_2, peak_torque_nm, peak_joint, per_joint_nm, peak_tilt_deg, leg1_wrist_sweep_rad, chosen}``,
    sorted by |wrist_2|), ``chosen_wrist_2``, ``chosen_peak_torque_nm``, ``chosen_peak_joint``,
    ``chosen_margin_nm`` (effort_limit - peak), ``min_torque_wrist_2`` (the branch minimum-torque WOULD pick,
    for the honest comparison), ``effort_limit_nm``, ``select_seconds`` (the measured CPU search time) and
    ``full_plan``. Raises ValueError when no reach+level branch yields a valid leg-2 start (the caller falls
    back to the single ``level_place_ik`` standoff). Pure/CPU."""
    if clock is None:
        clock = time.perf_counter
    t0 = clock()

    def _tick():
        if budget_s is not None and (clock() - t0) > budget_s:
            raise BranchSelectBudgetExceeded(
                'reaction-branch standoff search exceeded the %.3f s budget (%.3f s elapsed, %s path)'
                % (float(budget_s), clock() - t0, 'full' if full_plan else 'cheap'))

    soft = tuple(soft_joints) if soft_joints is not None else tuple(level_joints)
    carry = _as_joint_map(carry_config)
    carry_pan = float(carry['ur_shoulder_pan_joint'])
    carry_w2 = float(carry.get('ur_wrist_2_joint', 0.0))
    branches = enumerate_level_place_branches(robot, target_pos, up_in_tool, seed_config, joints=joints,
                                              tool_link=tool_link, root=root, limits=limits, base_R=base_R,
                                              fast=fast)
    if not branches:
        raise ValueError('no reach+level standoff branch at target %s'
                         % list(np.round(np.asarray(target_pos, float)[:3], 4)))
    scored = []
    for std in branches:
        _tick()
        # leg-2 start = this standoff with pan at the carry pan, wrists re-leveled in world at the carry pan
        # (the leg-1 end). The CHEAP default validates feasibility at that endpoint; full_plan replays the
        # whole densified leg 2 (pan -> standoff pan) ON this branch via the continuous solve.
        start = dict(std)
        start['ur_shoulder_pan_joint'] = carry_pan
        try:
            start = level_config(robot, start, up_in_tool, joints=level_joints, tool_link=tool_link,
                                 root=root, limits=limits, accept_deg=bound_deg, base_R=base_R, fast=fast)
            if full_plan:
                plan = plan_level_world_approach(robot, start, std, up_in_tool, level_joints=level_joints,
                                                 tool_link=tool_link, root=root, limits=limits,
                                                 bound_deg=bound_deg, max_segment_rad=max_segment_rad,
                                                 base_R=base_R)
                tau_waypoints = plan.waypoints
                peak_tilt = plan.max_tilt
            else:
                tau_waypoints = [_as_vec(start), _as_vec(std)]
                peak_tilt = None
        except ValueError:
            continue
        peak, pj, per = wrist_hold_torque_peak(robot, tau_waypoints, soft, carton_mass=carton_mass,
                                               tool_link=tool_link, root=root)
        w2 = float(std.get('ur_wrist_2_joint', 0.0))
        scored.append({'std': std, 'wrist_2': w2, 'abs_w2': abs(_wrap(w2)),
                       'peak_torque_nm': peak, 'peak_joint': pj, 'per_joint_nm': per,
                       'peak_tilt_deg': peak_tilt,
                       'leg1_wrist_sweep_rad': abs(_wrap(w2 - carry_w2))})
    if not scored:
        raise ValueError('no reach+level standoff branch produced a valid leg-2 sweep')
    # Primary: the settling branch (minimum |wrist_2|); tie-break by holding torque then leg-1 sweep.
    scored.sort(key=lambda s: (round(s['abs_w2'], 3), s['peak_torque_nm'], s['leg1_wrist_sweep_rad']))
    chosen = scored[0]
    min_torque = min(scored, key=lambda s: s['peak_torque_nm'])
    report = {'branches': [], 'chosen_wrist_2': chosen['wrist_2'],
              'chosen_peak_torque_nm': chosen['peak_torque_nm'], 'chosen_peak_joint': chosen['peak_joint'],
              'chosen_margin_nm': float(effort_limit) - chosen['peak_torque_nm'],
              'min_torque_wrist_2': min_torque['wrist_2'], 'effort_limit_nm': float(effort_limit),
              'select_seconds': clock() - t0, 'full_plan': bool(full_plan)}
    for s in scored:
        report['branches'].append({'wrist_2': s['wrist_2'], 'peak_torque_nm': s['peak_torque_nm'],
                                   'peak_joint': s['peak_joint'],
                                   'per_joint_nm': {j: float(s['per_joint_nm'].get(j, 0.0)) for j in soft},
                                   'peak_tilt_deg': s['peak_tilt_deg'],
                                   'leg1_wrist_sweep_rad': s['leg1_wrist_sweep_rad'],
                                   'chosen': s is chosen})
    return dict(chosen['std']), report
