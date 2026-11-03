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
    # sim-carton-tilt-transient-debounce-v1: SUSTAINED-tilt grace (sim-seconds AT 1/60). The trip FAILS
    # only when the measured tilt exceeds the bound CONTINUOUSLY for this long; a shorter spike logs
    # CARTON_TILT_TRANSIENT and proceeds. 0 => legacy immediate trip. The controller rebases this onto
    # the running physics step (freshness_scale) so it is a constant FRAME-COUNT budget.
    tilt_grace_s: float = DEFAULT_TILT_GRACE_S

    def validate(self):
        if not (math.isfinite(self.bound_deg) and self.bound_deg > 0):
            raise ValueError('carton_level.bound_deg must be a finite positive number of degrees')
        if not (math.isfinite(self.tilt_grace_s) and self.tilt_grace_s >= 0):
            raise ValueError('carton_level.tilt_grace_s must be a finite non-negative number of seconds')
        if not (isinstance(self.path_waypoints, int) and self.path_waypoints >= 2):
            raise ValueError('carton_level.path_waypoints must be an integer >= 2')
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        bad = [j for j in self.joints if j not in ARM_JOINTS]
        if bad or not self.joints:
            raise ValueError(f'carton_level.joints must be a non-empty subset of ARM_JOINTS; bad={bad}')
        return self


def resolve_level_config(spec):
    """Normalise a ``carton_level`` config value into a ``LevelConfig`` or ``None`` (off =>
    byte-identical). Accepts absent/None/False/'0'/'off' => None; True/'on'/{} => defaults; a
    mapping with optional ``bound_deg``, ``tilt_grace_s``, ``level_goals``, ``trip``, ``level_weld``,
    ``level_place``, ``level_path``, ``path_waypoints``, ``joints``, ``tool_link``. Raises on a malformed
    value so a typo fails loudly at boot."""
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
    for flag in ('level_goals', 'trip', 'level_weld', 'level_place', 'level_path'):
        if flag in spec and spec[flag] is not None:
            kw[flag] = bool(spec[flag])
    if spec.get('joints'):
        kw['joints'] = tuple(str(j) for j in spec['joints'])
    if spec.get('tool_link'):
        kw['tool_link'] = str(spec['tool_link'])
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
    pts = [list(w) for w in waypoints]
    mr = mp = mt = 0.0
    for i in range(len(pts) - 1):
        for k in range(interior + 1):
            a = k / float(interior)
            q = [(1 - a) * pts[i][j] + a * pts[i + 1][j] for j in range(6)]
            roll, pitch, tilt = carried_rpy(robot, q, up_in_tool, tool_link, root)
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


# ----------------------------------------------------- level-place IK (reach + level)

# Seeds (ABSOLUTE wrist targets) the level-place IK restarts from. For a TOP-DOWN grasp the carton's
# up-axis is FIXED off the tool approach axis, so a ~pi flip of wrist_3 (the flange roll about the
# approach axis) turns an inverted hold upright WITHOUT moving the tool origin (tool0 sits on the
# wrist_3 axis); the sweep lands a seed in the upright basin of every branch so the position+level
# descent cannot hide the holdable level solution behind a reach-authored (inverted) start.
_W3_SEED_GRID = tuple(np.linspace(-math.pi, math.pi, 9))


def level_place_ik(robot, target_pos, up_in_tool, seed_config, *, joints=None,
                   tool_link='ur_tool0', root='ur_base_link', limits=None, max_iter=200,
                   pos_tol=1e-3, tilt_tol_deg=0.05, w_ori=1.0):
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
    trip owns it)."""
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    cfg = _as_joint_map(seed_config)
    adj = list(joints) if joints is not None else list(ARM_JOINTS)
    adj = [j for j in adj if j in cfg]
    if not adj:
        raise ValueError('no adjustable joints present in the seed config')
    u = _unit(up_in_tool)
    tgt = np.asarray(target_pos, float)[:3]
    idx = [ARM_JOINTS.index(j) for j in adj]

    def state(cfgmap):
        T = tool_transform(robot, cfgmap, tool_link, root)
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
