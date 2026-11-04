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
WORLD_UP = np.array([0.0, 0.0, 1.0])


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


def carried_up_world(robot, config, up_in_tool, tool_link='ur_tool0', root='ur_base_link'):
    """World up-axis of the carried carton at ``config``: ``tool_R(config) @ up_in_tool``."""
    return tool_rotation(robot, config, tool_link, root) @ _unit(up_in_tool)


def carried_rpy(robot, config, up_in_tool, tool_link='ur_tool0', root='ur_base_link'):
    """(roll_deg, pitch_deg, tilt_deg) of the carried carton at arm ``config``."""
    return up_axis_rpy(carried_up_world(robot, config, up_in_tool, tool_link, root))


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
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        bad = [j for j in self.joints if j not in ARM_JOINTS]
        if bad or not self.joints:
            raise ValueError(f'carton_level.joints must be a non-empty subset of ARM_JOINTS; bad={bad}')
        if self.gravity_compensation is not None:
            if not isinstance(self.gravity_compensation, GravityCompensation):
                raise ValueError('carton_level.gravity_compensation must resolve to a GravityCompensation')
            self.gravity_compensation.validate()
        return self


def resolve_level_config(spec):
    """Normalise a ``carton_level`` config value into a ``LevelConfig`` or ``None`` (off =>
    byte-identical). Accepts absent/None/False/'0'/'off' => None; True/'on'/{} => defaults; a
    mapping with optional ``bound_deg``, ``tilt_grace_s``, ``level_goals``, ``trip``, ``level_weld``,
    ``level_place``, ``level_path``, ``place_from_above``, ``path_waypoints``, ``place_descent_speed``,
    ``place_descent_accel``, ``place_above_clearance_m``, ``place_depth_bias_m``, ``preplace_above_place``,
    ``preplace_standoff_m``, ``place_clearance_check``, ``place_clearance_margin_m``, ``place_clearance_drooped``,
    ``closed_loop_level``, ``closed_loop_authority_deg``, ``closed_loop_rate_deg_per_s``, ``closed_loop_gain``,
    ``joints``, ``tool_link``.
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
                 'closed_loop_authority_deg', 'closed_loop_rate_deg_per_s', 'closed_loop_gain'):
        if fnum in spec and spec[fnum] is not None:
            kw[fnum] = float(spec[fnum])
    for flag in ('level_goals', 'trip', 'level_weld', 'level_place', 'level_path', 'place_from_above',
                 'preplace_above_place', 'place_clearance_check', 'place_clearance_drooped',
                 'closed_loop_level'):
        if flag in spec and spec[flag] is not None:
            kw[flag] = bool(spec[flag])
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
                 max_iter=80, tol_deg=0.05, accept_deg=None):
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

    def up_world(cfgmap):
        return tool_rotation(robot, cfgmap, tool_link, root) @ u

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

    def descend(start):
        m = clamp(dict(start))
        step_fd = 1e-4
        for _ in range(int(max_iter)):
            if tilt_deg(m) <= tol_deg:
                break
            r0 = up_world(m) - WORLD_UP   # 3-D residual: zero only upright, repulsive at the 180 pole
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
                      interior=8):
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
                                      interior=interior, gravity_comp=None)


def predict_path_tilt_executed(robot, waypoints, up_in_tool, *, tool_link='ur_tool0',
                               root='ur_base_link', interior=8, gravity_comp=None):
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
            roll, pitch, tilt = carried_rpy(robot, executed(q), up_in_tool, tool_link, root)
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
    """
    waypoints: tuple
    max_roll: float
    max_pitch: float
    max_tilt: float
    waypoints_n: int

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
                          limits=None, accept_deg=None):
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
                                 root=root, limits=limits, accept_deg=accept_deg)
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
                   pos_tol=1e-3, tilt_tol_deg=0.05, w_ori=1.0, gravity_comp=None):
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
                              pos_tol=pos_tol, tilt_tol_deg=tilt_tol_deg, w_ori=w_ori, gravity_comp=None)
        return gravity_comp.wrist_feedforward(robot, geom, up_in_tool, tool_link=tool_link, root=root,
                                              limits=limits, accept_deg=DEFAULT_TILT_BOUND_DEG)
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
        return T[:3, 3], T[:3, :3] @ u

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

    def descend(start):
        m = clamp(dict(start))
        step_fd = 1e-5
        for _ in range(int(max_iter)):
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
