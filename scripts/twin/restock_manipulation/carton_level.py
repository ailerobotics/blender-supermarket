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


def level_config(robot, goal_config, up_in_tool, *, joints=DEFAULT_LEVEL_JOINTS,
                 tool_link='ur_tool0', root='ur_base_link', limits=None,
                 max_iter=60, tol_deg=0.05):
    """Re-solve ``joints`` of ``goal_config`` so the carried carton's up-axis points world-up (a
    level-orientation waypoint). Returns a new joint map. Position (pan/lift/elbow) is untouched;
    only the wrist cluster moves, so the tool stays where the carry plan put it and the carton is
    level. Damped Gauss-Newton on the 2-D tilt residual (up_world cross world-up) over the chosen
    joints, with optional per-joint ``limits`` {name:(lo,hi)}. Raises if it cannot reach ``tol_deg``
    (no level solution for this position with these joints -- caller fails honestly / re-plans)."""
    cfg = _as_joint_map(goal_config)
    joints = [j for j in joints if j in cfg]
    if not joints:
        raise ValueError('no adjustable joints present in the goal config')
    u = _unit(up_in_tool)

    def up_world(cfgmap):
        return tool_rotation(robot, cfgmap, tool_link, root) @ u

    def residual(cfgmap):
        # tilt residual = world-up x up_world projected to the two horizontal axes (0 when level).
        uw = up_world(cfgmap)
        r = np.cross(uw, WORLD_UP)
        return r[:2]

    def tilt_deg(cfgmap):
        uw = up_world(cfgmap)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(uw[2])))))

    step_fd = 1e-4
    for _ in range(int(max_iter)):
        if tilt_deg(cfg) <= tol_deg:
            return cfg
        r0 = residual(cfg)
        J = np.zeros((2, len(joints)))
        for k, j in enumerate(joints):
            cfg2 = dict(cfg)
            cfg2[j] = cfg[j] + step_fd
            J[:, k] = (residual(cfg2) - r0) / step_fd
        # damped least squares  dq = -(J^T J + lambda I)^-1 J^T r
        lam = 1e-6
        JTJ = J.T @ J + lam * np.eye(len(joints))
        dq = -np.linalg.solve(JTJ, J.T @ r0)
        # bound the step so FK stays in its linear regime
        mx = float(np.max(np.abs(dq))) if dq.size else 0.0
        if mx > 0.3:
            dq *= 0.3 / mx
        for k, j in enumerate(joints):
            v = cfg[j] + float(dq[k])
            if limits and j in limits:
                lo, hi = limits[j]
                v = min(max(v, lo), hi)
            cfg[j] = v
    if tilt_deg(cfg) <= max(tol_deg, 1e-3):
        return cfg
    raise ValueError(f'no level-orientation solution for this carried pose '
                     f'(residual tilt {tilt_deg(cfg):.2f} deg > {tol_deg} with joints {joints})')


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
