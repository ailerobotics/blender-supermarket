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
# Default bound (deg) on |roll| and |pitch| while carrying -- the captain's "parallel to the
# ground"; 2 deg is the brief's stated threshold for naming a move off-level.
DEFAULT_TILT_BOUND_DEG = 2.0
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
    from twin.urdf_probe import link_world_transforms
    cfg = _as_joint_map(config)
    world = link_world_transforms(robot, cfg, root=root)
    if tool_link not in world:
        raise ValueError(f'tool link {tool_link!r} not in the URDF FK ({len(world)} links)')
    return np.asarray(world[tool_link][:3, :3], float)


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
    joints: tuple = DEFAULT_LEVEL_JOINTS
    tool_link: str = 'ur_tool0'

    def validate(self):
        if not (math.isfinite(self.bound_deg) and self.bound_deg > 0):
            raise ValueError('carton_level.bound_deg must be a finite positive number of degrees')
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        bad = [j for j in self.joints if j not in ARM_JOINTS]
        if bad or not self.joints:
            raise ValueError(f'carton_level.joints must be a non-empty subset of ARM_JOINTS; bad={bad}')
        return self


def resolve_level_config(spec):
    """Normalise a ``carton_level`` config value into a ``LevelConfig`` or ``None`` (off =>
    byte-identical). Accepts absent/None/False/'0'/'off' => None; True/'on'/{} => defaults; a
    mapping with optional ``bound_deg``, ``level_goals``, ``trip``, ``level_weld``, ``joints``,
    ``tool_link``. Raises on a malformed value so a typo fails loudly at boot."""
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
    for flag in ('level_goals', 'trip', 'level_weld'):
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
