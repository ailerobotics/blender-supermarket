"""CPU carton-level proof: for every CARRIED segment of the restock arm fold, show that the
unleveled joint move tilts the welded carton off level (reproducing coldwin16) and that the
level-orientation constraint (twin.restock_manipulation.carton_level) holds |roll|,|pitch|
under the bound THROUGHOUT -- with no GPU boot.

Honest scope (state with every result):
  * REAL: the robot kinematics (the mission ``mobile-import.urdf`` forward kinematics), the
    carton-is-welded-to-the-tool relationship, and the real ``carton_level`` leveling solver +
    trip are exercised exactly as the mission would use them.
  * REPRESENTATIVE: the carried joint GOALS. The live_pose mission plans preplace/place with
    cuMotion and does not log the per-tick arm joints, so the exact carried configs are not
    recoverable from coldwin16. The segments below are plausible lift/tuck/pan/preplace/place/
    retreat folds anchored at the measured grasp attitude (a side grip, tool +Z ~ world +Y); the
    point proven is that the leveling MECHANISM keeps the carton level across the carry under the
    real kinematics, which the integration lane then confirms in-sim.

Pure: numpy + the Isaac-free urdf_probe FK + carton_level. CPU unit-tested under
scripts/twin/tests/test_carton_level_bench.py.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from twin.urdf_probe import parse_urdf
from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.transfer_carry import wrap_angle

# The grasp/weld attitude measured in coldwin16: a SIDE grip, SEAL_DIAG forward ~ [0,1,0] (the
# tool approach axis points world +Y, horizontal), carton resting level. A representative arm
# config that reaches tool +Z = world +Y (verified by FK in the tests).
WELD_CONFIG = (0.0, -math.pi / 2, math.pi / 2, 0.0, 0.0, 0.0)

# sim-carton-tilt-trip-margin-v1: the coldwin24 collect grasp is TOP-DOWN, not the coldwin16 side
# grip. SEAL_DIAG logged forward_world ~ [-0.04, 0, -0.999] (the suction plates point straight DOWN
# onto the carton top). The suction-plate joint's local +Z expressed in the tool0 frame is [0, 1, 0]
# (URDF-fixed), so a grasp whose plate forward points world-down has tool0 +Y pointing world-down,
# which makes the level carton's up-axis (world +Z) read [0, -1, 0] in the tool frame. This arm
# config realises exactly that (FK-verified in the tests: plate forward world ~ [0,0,-1], up_in_tool
# ~ [0,-1,0], carton level at the weld) -- the honest reconstruction of the failing collect grasp.
TOP_GRASP_WELD = (-math.pi, -2.2, 2.2, -math.pi, 0.0, 0.0)

# The REAL carried joint goals the collect carry drives to (restock_manipulation/ros_host_config.json),
# loaded from that file so the bench can never drift from the shipped config.
_HOST_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'restock_manipulation', 'ros_host_config.json')


def collect_carry_shapes(host_config_path=_HOST_CONFIG):
    """The fixed carried joint goals of the real COLLECT carry, read from the host config:
    lift (carry_lift_shape), tuck/pan (cart_pan_joint0 + cart_tuck_shape) and the place-on-cart
    standoff (collect_place_preplace_shape). These are the moves the CARTON_TILT trip guards and the
    ones coldwin24/24b tripped at. Returns [(name, goal6)]."""
    with open(host_config_path) as f:
        c = json.load(f)
    lift = tuple(float(v) for v in c['carry_lift_shape'])
    tuck = (float(c['cart_pan_joint0'] if c.get('cart_pan_joint0') is not None else 0.0),) \
        + tuple(float(v) for v in c['cart_tuck_shape'])
    preplace = tuple(float(v) for v in c['collect_place_preplace_shape'])
    return [('lift', lift), ('pan', tuck), ('place_preplace', preplace)]


def run_collect(urdf_path, *, bound_deg=cl.DEFAULT_TILT_BOUND_DEG, weld=TOP_GRASP_WELD,
                joints=cl.DEFAULT_LEVEL_JOINTS, tool_link='ur_tool0', host_config_path=_HOST_CONFIG):
    """Reproduce the coldwin24 collect-carry tilt on the REAL kinematics + REAL carry shapes + the
    REAL top-down grasp up-axis, then prove the (fixed) leveling cancels it. Returns (up_in_tool,
    [(name, before_tilt_deg, after_tilt_deg, after_wrists, in_clear)]). ``before`` is the raw fixed
    shape (what the controller drove when the leveling silently failed -> the carton off level); the
    carried bound is the leveling acceptance (clear wrist region)."""
    robot = parse_urdf(urdf_path)
    u = cl.carton_up_in_tool(cl.tool_rotation(robot, list(weld), tool_link))
    limits = clear_wrist_limits()
    out = []
    for name, goal in collect_carry_shapes(host_config_path):
        before = cl.carried_rpy(robot, list(goal), u, tool_link)[2]
        try:
            lev = cl.level_config(robot, list(goal), u, joints=joints, tool_link=tool_link,
                                  limits=limits, accept_deg=bound_deg)
            after = cl.carried_rpy(robot, lev, u, tool_link)[2]
            w = (wrap_angle(lev['ur_wrist_1_joint']), wrap_angle(lev['ur_wrist_2_joint']),
                 wrap_angle(lev['ur_wrist_3_joint']))
            clear = w[0] <= 0.65 + 1e-6 and w[1] >= -1e-6
        except ValueError:
            after, w, clear = float('nan'), None, False
        out.append((name, before, after, w, clear))
    return u, out


@dataclass
class PlaceResult:
    """The level place-on-cart proof (sim-carton-level-place-on-cart-v1): the reach-authored place
    standoff holds the top-grasped carton inverted; the re-planned level place reaches the SAME deck
    TCP and stays parallel to the ground through the straight-line descent."""
    preplace_before_tilt: float          # reach shape, unleveled (the ~180 deg inverted hold)
    preplace_after_tilt: float           # level standoff
    preplace_tcp_shift_mm: float         # reach preserved (~0)
    preplace_wrist3_roll_deg: float      # the near-free flange roll that levels it
    preplace_arm_shift_deg: float        # max |change| of the non-wrist_3 joints (branch preserved)
    setdown_after_tilt: float            # level set-down on the deck
    setdown_pos_err_mm: float            # reaches the sensed deck target
    descent_max_tilt: float              # max |roll|,|pitch| across the straight-line descent
    preplace_clear: bool
    setdown_clear: bool

    def passes(self, bound_deg, strict_deg=0.1):
        return (self.preplace_after_tilt <= strict_deg and self.setdown_after_tilt <= strict_deg
                and self.descent_max_tilt <= strict_deg and self.preplace_clear and self.setdown_clear
                and self.preplace_tcp_shift_mm <= 1.0 and self.setdown_pos_err_mm <= 5.0)


def run_place(urdf_path, *, bound_deg=cl.DEFAULT_TILT_BOUND_DEG, weld=TOP_GRASP_WELD,
              descent_m=0.12, n=20, host_config_path=_HOST_CONFIG, tool_link='gripper_pad_tcp'):
    """CPU proof of the LEVEL place-on-cart move on the REAL kinematics + the REAL reach-authored
    standoff (collect_place_preplace_shape) + the REAL top-down grasp up-axis. Shows the standoff holds
    the carton ~180 deg off level, that the re-planned level place reaches the SAME deck TCP at ~0 deg
    (a near-free wrist_3 flange roll, same arm branch), and that the single straight-line descent to a
    level set-down on the deck stays parallel to the ground throughout. ``tool_link`` is the port's IK
    reach frame (gripper_pad_tcp, the frame the sensed deck poses are in); its rotation equals ur_tool0's,
    so the carton attitude is identical. Returns a PlaceResult."""
    robot = parse_urdf(urdf_path)
    u = cl.carton_up_in_tool(cl.tool_rotation(robot, list(weld), tool_link))
    limits = clear_wrist_limits()
    limits['ur_elbow_joint'] = (0.1, math.pi)
    with open(host_config_path) as f:
        preplace = [float(v) for v in json.load(f)['collect_place_preplace_shape']]

    before = cl.carried_rpy(robot, preplace, u, tool_link)[2]
    lev = cl.level_preserve_tcp(robot, preplace, u, tool_link=tool_link, limits=limits)
    levv = [lev[j] for j in ARM_JOINTS]
    after_pp = cl.carried_rpy(robot, levv, u, tool_link)[2]
    tcp0 = cl.tool_position(robot, preplace, tool_link)
    tcp1 = cl.tool_position(robot, levv, tool_link)
    tcp_shift = float(np.linalg.norm(tcp1 - tcp0)) * 1000.0
    w3_roll = abs(math.degrees(wrap_angle(levv[5] - preplace[5])))
    arm_shift = max(abs(math.degrees(wrap_angle(levv[k] - preplace[k]))) for k in range(5))

    target = tcp0.copy()
    target[2] -= float(descent_m)                       # vertical descent to the deck set-down
    sd = cl.level_place_ik(robot, target, u, levv, tool_link=tool_link, limits=limits)
    sdv = [sd[j] for j in ARM_JOINTS]
    after_sd = cl.carried_rpy(robot, sdv, u, tool_link)[2]
    pos_err = float(np.linalg.norm(cl.tool_position(robot, sdv, tool_link) - target)) * 1000.0

    dmax = 0.0
    for i in range(int(n) + 1):
        a = i / float(n)
        wp = [(1 - a) * levv[k] + a * sdv[k] for k in range(6)]
        roll, pitch, _ = cl.carried_rpy(robot, wp, u, tool_link)
        dmax = max(dmax, abs(roll), abs(pitch))

    return PlaceResult(
        preplace_before_tilt=before, preplace_after_tilt=after_pp,
        preplace_tcp_shift_mm=tcp_shift, preplace_wrist3_roll_deg=w3_roll,
        preplace_arm_shift_deg=arm_shift, setdown_after_tilt=after_sd, setdown_pos_err_mm=pos_err,
        descent_max_tilt=dmax, preplace_clear=_in_clear_wrist(lev), setdown_clear=_in_clear_wrist(sd))


# The coldwin25 collect grasp/weld pose (ARM order): the measured arm config where the carton welds
# level and the lift begins, reconstructed from the run's final grasp-track creep endpoint
# (data/sim-carry-level-lift-v1). The EXECUTED-path proof anchors the real carry here, not the
# representative WELD_CONFIG, so it replays the ACTUAL failing lift.
COLDWIN25_GRASP = (1.2655, -1.2755, 2.0905, -0.8149, 1.2621, 0.0)


@dataclass
class ExecutedPathResult:
    """Replay of a REAL carried move (sim-carry-level-lift-v1): the single-goal joint path (what the
    executor drives for a single leveled-ENDPOINT begin_joint_goal) tilts the carton mid-path, while the
    leveled WAYPOINT path (begin_joint_path) keeps |roll|,|pitch| under the bound THROUGHOUT."""
    name: str
    single_goal_max_tilt: float      # straight joint segment start->leveled goal (both ends level)
    single_goal_max_roll: float
    single_goal_max_pitch: float
    path_max_tilt: float             # leveled waypoint path (predicted executed)
    path_max_roll: float
    path_max_pitch: float
    waypoints_n: int
    single_series: list = field(default_factory=list)   # [(frac, roll, pitch, tilt)]
    path_series: list = field(default_factory=list)      # [(frac, roll, pitch, tilt)]

    def passes(self, bound_deg):
        return max(self.path_max_roll, self.path_max_pitch) <= bound_deg


def _series(robot, waypoints, u, tool_link, interior):
    """Sample (frac, roll, pitch, tilt) along the straight-segment interiors of a waypoint list."""
    pts = [list(w) for w in waypoints]
    out = []
    total = len(pts) - 1
    for i in range(total):
        for k in range(interior + 1):
            if i > 0 and k == 0:
                continue
            a = k / float(interior)
            q = [(1 - a) * pts[i][j] + a * pts[i + 1][j] for j in range(6)]
            roll, pitch, tilt = cl.carried_rpy(robot, q, u, tool_link)
            out.append(((i + a) / total, roll, pitch, tilt))
    return out


def run_executed_paths(urdf_path, *, bound_deg=cl.DEFAULT_TILT_BOUND_DEG, weld=COLDWIN25_GRASP,
                       tool_link='ur_tool0', host_config_path=_HOST_CONFIG, interior=20):
    """Replay the REAL collect carry (lift, then the pan fold to the cart) on the REAL mission kinematics
    + REAL carry shapes + the measured coldwin25 grasp up-axis. For each move it returns an
    ExecutedPathResult comparing the single-goal joint path (the current single begin_joint_goal: both
    ENDPOINTS leveled but the straight joint segment between them tilts the carton) against the leveled
    WAYPOINT path (the fix). This is the brief's EXECUTED-path proof -- it replays the PATH, not just the
    endpoints."""
    robot = parse_urdf(urdf_path)
    u = cl.carton_up_in_tool(cl.tool_rotation(robot, list(weld), tool_link))
    limits = clear_wrist_limits()
    c = json.load(open(host_config_path))
    lift = [float(v) for v in c['carry_lift_shape']]
    pan = [float(c['cart_pan_joint0'])] + [float(v) for v in c['cart_tuck_shape']]
    # lift starts at the grasp pose; pan starts at the leveled lift goal.
    lift_leveled_goal = [cl.level_config(robot, lift, u, limits=limits, accept_deg=bound_deg)[j]
                         for j in ARM_JOINTS]
    moves = [('lift', list(weld), lift), ('pan', lift_leveled_goal, pan)]
    out = []
    for name, start, goal in moves:
        leveled_goal = [cl.level_config(robot, goal, u, limits=limits, accept_deg=bound_deg)[j]
                        for j in ARM_JOINTS]
        single = _series(robot, [start, leveled_goal], u, tool_link, interior)
        plan = cl.plan_level_path(robot, start, goal, u, bound_deg=bound_deg, tool_link=tool_link,
                                  limits=limits)
        path = _series(robot, [list(w) for w in plan.waypoints], u, tool_link, interior)
        out.append(ExecutedPathResult(
            name=name,
            single_goal_max_tilt=max(r[3] for r in single),
            single_goal_max_roll=max(abs(r[1]) for r in single),
            single_goal_max_pitch=max(abs(r[2]) for r in single),
            path_max_tilt=plan.max_tilt, path_max_roll=plan.max_roll, path_max_pitch=plan.max_pitch,
            waypoints_n=plan.waypoints_n, single_series=single, path_series=path))
    return u, out


# ----------------------------------- preplace APPROACH executed path (sim-carton-level-preplace-v1)
# The REAL collectlevel1 collect carry, reconstructed from the saved run's host-stack TRAJ_HOST_PLAN
# dense trajectories (ARM order). The grasp fixes the carton up-axis at the weld; the preplace START is
# the pan/tuck carry pose and the preplace GOAL is the PR 290 full-DOF leveled place-on-cart standoff
# (the executed endpoint). Both ENDPOINTS are level, but the single begin_joint_goal joint-space path
# between them passed through a pitched config and tripped CARTON_TILT at ~2.25 deg pitch.
COLLECTLEVEL1_GRASP = (1.1923, -1.2833, 2.1071, -0.8249, 1.1932, -0.0001)
COLLECTLEVEL1_PREPLACE_START = (1.2672, -2.27, 1.0, 0.65, 0.0017, 0.6187)
COLLECTLEVEL1_PREPLACE_GOAL = (0.093, -1.9103, 2.0154, -0.1061, 1.664, -0.0006)


def run_preplace_approach(urdf_path, *, bound_deg=cl.DEFAULT_TILT_BOUND_DEG,
                          grasp=COLLECTLEVEL1_GRASP, start=COLLECTLEVEL1_PREPLACE_START,
                          goal=COLLECTLEVEL1_PREPLACE_GOAL, tool_link='ur_tool0', interior=40):
    """Replay the REAL collectlevel1 carry/tuck -> place-on-cart standoff APPROACH on the REAL mission
    kinematics + the measured grasp up-axis, comparing the single leveled-ENDPOINT joint goal (what the
    controller drove -- both endpoints level but the straight joint segment between them tilts the carton
    off level mid-path, tripping CARTON_TILT at the pre-place transition) against the leveled WAYPOINT path
    to the SAME PR 290 standoff goal (the fix: the goal pinned verbatim, interior wrists re-solved level).
    Returns an ExecutedPathResult. This is the brief's EXECUTED-path proof for the pre-place segment."""
    robot = parse_urdf(urdf_path)
    u = cl.carton_up_in_tool(cl.tool_rotation(robot, list(grasp), tool_link))
    limits = clear_wrist_limits()
    s, g = list(start), list(goal)
    single = _series(robot, [s, g], u, tool_link, interior)
    plan = cl.plan_level_path_to_goal(robot, s, g, u, bound_deg=bound_deg, tool_link=tool_link,
                                      limits=limits)
    path = _series(robot, [list(w) for w in plan.waypoints], u, tool_link, interior)
    return u, ExecutedPathResult(
        name='preplace_approach',
        single_goal_max_tilt=max(r[3] for r in single),
        single_goal_max_roll=max(abs(r[1]) for r in single),
        single_goal_max_pitch=max(abs(r[2]) for r in single),
        path_max_tilt=plan.max_tilt, path_max_roll=plan.max_roll, path_max_pitch=plan.max_pitch,
        waypoints_n=plan.waypoints_n, single_series=single, path_series=path)


def executed_table(results, bound_deg=cl.DEFAULT_TILT_BOUND_DEG):
    rows = [('move', 'single|roll|', 'single|pitch|', 'single_tilt',
             'path|roll|', 'path|pitch|', 'path_tilt', 'n', 'verdict')]
    for r in results:
        rows.append((r.name, f'{r.single_goal_max_roll:.2f}', f'{r.single_goal_max_pitch:.2f}',
                     f'{r.single_goal_max_tilt:.2f}', f'{r.path_max_roll:.2f}', f'{r.path_max_pitch:.2f}',
                     f'{r.path_max_tilt:.2f}', str(r.waypoints_n),
                     'PASS' if r.passes(bound_deg) else 'FAIL'))
    w = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return '\n'.join('  '.join(c.ljust(w[i]) for i, c in enumerate(row)) for row in rows)


def _vec(cfg):
    """Config map -> 6-vector in ARM order."""
    return [float(cfg[j]) for j in ARM_JOINTS]


@dataclass
class SegmentResult:
    name: str
    before_max_roll: float
    before_max_pitch: float
    before_max_tilt: float
    after_max_roll: float
    after_max_pitch: float
    after_max_tilt: float
    after_in_clear_wrist: bool
    before_rpy: list = field(default_factory=list)   # [(frac, roll, pitch, tilt)] unleveled
    after_rpy: list = field(default_factory=list)    # [(frac, roll, pitch, tilt)] leveled

    def passes(self, bound_deg):
        return self.after_max_roll <= bound_deg and self.after_max_pitch <= bound_deg


def carried_segments(weld=WELD_CONFIG):
    """Representative (name, start, goal) joint moves for the welded carry, anchored at ``weld``.
    Each goal perturbs the wrists/shoulder the way a fold/preplace/place would, so the UNLEVELED
    move tilts the carton -- the condition the level constraint must cancel."""
    W = dict(zip(ARM_JOINTS, [float(v) for v in weld]))

    def goal(**d):
        g = dict(W)
        for k, v in d.items():
            g['ur_' + k + '_joint'] = v
        return g

    lift = goal(shoulder_lift=-1.2, elbow=1.3)                    # raise off the shelf
    tuck = goal(shoulder_lift=-1.0, elbow=1.1, wrist_1=0.6, wrist_2=0.3)  # fold the carton in
    pan = goal(shoulder_pan=-1.4, shoulder_lift=-1.0, elbow=1.1, wrist_1=0.6, wrist_2=0.3)  # swing to cart
    preplace = goal(shoulder_pan=-1.4, shoulder_lift=-1.6, elbow=1.4, wrist_1=0.4, wrist_2=-0.2)
    place = goal(shoulder_pan=-1.4, shoulder_lift=-1.9, elbow=1.7, wrist_1=0.2, wrist_2=-0.2)
    # The CARRIED segments only: lift -> tuck -> pan -> preplace -> place. retreat/return happen
    # AFTER the gripper releases (carton no longer held), so they are not level-constrained.
    return [
        ('lift', W, lift),
        ('tuck', lift, tuck),
        ('pan', tuck, pan),
        ('preplace', pan, preplace),
        ('place', preplace, place),
    ]


def _in_clear_wrist(cfg, wrist1_cap=0.65, wrist2_min=0.0):
    """Does a leveled config stay in the bench-proven clear wrist carry region (transfer_carry)?
    Reported so a level waypoint that would need the self-colliding wrist branch is flagged for the
    integration lane rather than silently shipped."""
    v = _vec(cfg)
    return wrap_angle(v[3]) <= wrist1_cap and wrap_angle(v[4]) >= wrist2_min


def clear_wrist_limits(wrist1_cap=0.65, wrist2_min=0.0):
    """Per-joint limits that keep a leveled waypoint inside the bench-proven clear carry wrist
    region (transfer_carry): wrist_1 <= cap, wrist_2 >= min. Passed to the leveling solver so the
    level-orientation constraint cannot drive the wrist into the ur_wrist_2-vs-forearm stop."""
    big = 2 * math.pi
    return {'ur_wrist_1_joint': (-big, wrist1_cap), 'ur_wrist_2_joint': (wrist2_min, big)}


def run(urdf_path, *, bound_deg=cl.DEFAULT_TILT_BOUND_DEG, n=10, weld=WELD_CONFIG,
        joints=cl.DEFAULT_LEVEL_JOINTS, tool_link='ur_tool0', respect_clear_wrist=True):
    """Run the proof over all carried segments. Returns (up_in_tool, [SegmentResult...]). When
    ``respect_clear_wrist`` the leveling is bounded to the clear carry wrist region so the fix is
    proven COMPATIBLE with the existing self-collision wrist guard (a segment with no level solution
    in that region raises -- an honest 'this pose cannot be both level and self-clear' for the lane)."""
    robot = parse_urdf(urdf_path)
    Rw = cl.tool_rotation(robot, weld, tool_link)
    u = cl.carton_up_in_tool(Rw)
    limits = clear_wrist_limits() if respect_clear_wrist else None
    results = []
    for name, start, goal in carried_segments(weld):
        s = cl._as_joint_map(start)
        g = cl._as_joint_map(goal)
        # BEFORE: raw linear joint interpolation (no orientation constraint)
        before = []
        for i in range(n + 1):
            a = i / float(n)
            wp = {k: (1 - a) * s[k] + a * g[k] for k in s}
            roll, pitch, tilt = cl.carried_rpy(robot, wp, u, tool_link)
            before.append((a, roll, pitch, tilt))
        # AFTER: leveled waypoints (the level-orientation path constraint)
        after = []
        leveled = cl.interpolate_level(robot, start, goal, u, n=n, joints=joints,
                                       tool_link=tool_link, limits=limits)
        clear = True
        for i, wp in enumerate(leveled):
            roll, pitch, tilt = cl.carried_rpy(robot, wp, u, tool_link)
            after.append((i / float(n), roll, pitch, tilt))
            clear = clear and _in_clear_wrist(wp)
        results.append(SegmentResult(
            name=name,
            before_max_roll=max(abs(r[1]) for r in before),
            before_max_pitch=max(abs(r[2]) for r in before),
            before_max_tilt=max(r[3] for r in before),
            after_max_roll=max(abs(r[1]) for r in after),
            after_max_pitch=max(abs(r[2]) for r in after),
            after_max_tilt=max(r[3] for r in after),
            after_in_clear_wrist=clear,
            before_rpy=before, after_rpy=after))
    return u, results


def table(results, bound_deg=cl.DEFAULT_TILT_BOUND_DEG):
    rows = [('segment', 'before|roll|', 'before|pitch|', 'before_tilt',
             'after|roll|', 'after|pitch|', 'after_tilt', 'clear_wrist', 'verdict')]
    for r in results:
        rows.append((r.name, f'{r.before_max_roll:.1f}', f'{r.before_max_pitch:.1f}',
                     f'{r.before_max_tilt:.1f}', f'{r.after_max_roll:.2f}',
                     f'{r.after_max_pitch:.2f}', f'{r.after_max_tilt:.2f}',
                     'yes' if r.after_in_clear_wrist else 'NO',
                     'PASS' if r.passes(bound_deg) else 'FAIL'))
    w = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return '\n'.join('  '.join(c.ljust(w[i]) for i, c in enumerate(row)) for row in rows)


def all_pass(results, bound_deg=cl.DEFAULT_TILT_BOUND_DEG):
    return bool(results) and all(r.passes(bound_deg) for r in results)


# -------------------------------------- place-on-cart SET-DOWN descent (sim-carton-level-preplace-v2)
# The REAL collectlevel3 collect place-on-cart descent, reconstructed from the saved run's host-stack
# TRAJ_HOST_PLAN (the npoints=2 goal_pos_tol=0.120 plan at host-stack.log L15978, converted from MoveIt
# alphabetical [elbow, slift, span, w1, w2, w3] order to ARM order). The grasp fixes the carton up-axis
# at the weld (the lift-path start, ARM order). The descent is the ONE carried segment still driven as a
# SINGLE 2-point straight-line begin_joint_goal (every other carried move is a leveled waypoint path).
PLACE_DESCENT_GRASP = (1.1994, -1.2682, 2.0862, -0.8177, 1.2002, 0.0)
PLACE_DESCENT_STANDOFF = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)
PLACE_DESCENT_SETDOWN = (-0.0218, -1.4813, 1.7927, -0.3113, 1.6079, 0.0001)

# Representative per-joint first-order tracking-lag time constants (s), ARM order. The welded carton's
# attitude is the arm's attitude, and this descent keeps it level by a LARGE OPPOSING swing of the pitch
# chain (shoulder_lift +24.6 deg while elbow -12.7 and wrist_1 -11.9 deg, netting a constant level sum).
# A uniform lag leaves that sum level (the carton stays level); the tilt comes from the DIFFERENTIAL lag
# of a heavy shoulder/elbow against the light wrist. tau_shoulder_lift was CALIBRATED (bisection) so this
# model reproduces the collectlevel3 executed peak (2.22 deg pitch) on the ungentled 0.08 rad/s descent;
# elbow is half the shoulder, the wrists an order lighter. This is a REPRESENTATIVE execution model (the
# run does not log per-tick joints) whose ONE free parameter is pinned to the one observed GPU data point;
# the integration lane confirms the residual in-sim via the new CARTON_SEGMENT_TILT log.
DEFAULT_PLACE_LAG = (0.30, 0.7115, 0.35575, 0.05, 0.05, 0.05)


def _descent_duration(dmax, speed, accel):
    """The rest-to-rest 2-point duration ros_adapter._straight_line_goal sizes for a move of max joint
    delta ``dmax`` under peak-speed ``speed`` / peak-accel ``accel`` ceilings (identical formula)."""
    return max(2.0, 1.5 * dmax / speed, math.sqrt(6 * dmax / accel))


def _smoothstep(x):
    x = min(1.0, max(0.0, x))
    return 3 * x * x - 2 * x * x * x


def tracking_series(robot, start, goal, u, *, speed, accel, tau, tool_link='ur_tool0', dt=0.004):
    """Replay the EXECUTED carton attitude of a straight-line 2-point joint descent under a per-joint
    first-order tracking lag. The COMMANDED path is the rest-to-rest smoothstep start->goal (all joints
    share one normalised profile, so the command stays on the straight joint line -- i.e. level); the
    MEASURED joints follow it through an independent first-order lag per joint (``tau``), so a heavy joint
    drifts off the level-cancelling sum. Returns ``(commanded_series, executed_series)`` as lists of
    ``(t, roll, pitch, tilt)``. Pure/CPU; ``tau`` is ARM order."""
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    tau = np.asarray(tau, float)
    D = goal - start
    dmax = float(np.max(np.abs(D))) if D.size else 0.0
    T = _descent_duration(dmax, speed, accel)
    n = int(T / dt) + 1
    qm = start.copy()
    cmd_series, exe_series = [], []
    for k in range(n):
        t = k * dt
        qc = start + D * _smoothstep(t / T)
        qm = qm + (qc - qm) * (dt / np.maximum(tau, 1e-6))
        cr, cp, ct = cl.carried_rpy(robot, qc.tolist(), u, tool_link)
        er, ep, et = cl.carried_rpy(robot, qm.tolist(), u, tool_link)
        cmd_series.append((t, cr, cp, ct))
        exe_series.append((t, er, ep, et))
    return cmd_series, exe_series


@dataclass
class PlaceDescentResult:
    """The place-on-cart SET-DOWN descent proof (sim-carton-level-preplace-v2): the COMMANDED straight
    joint line is level, the UNGENTLED executed path (tracking lag) tilts ~2.22 deg (reproducing the
    collectlevel3 CARTON_TILT), the GENTLED (time-parameterised slower) executed path stays under 1 deg."""
    commanded_max_pitch: float
    commanded_max_tilt: float
    ungentled_speed: float
    ungentled_max_pitch: float
    gentled_speed: float
    gentled_max_pitch: float
    ungentled_duration: float
    gentled_duration: float
    commanded_series: list = field(default_factory=list)
    ungentled_series: list = field(default_factory=list)
    gentled_series: list = field(default_factory=list)


def run_place_descent(urdf_path, *, bound_deg=cl.DEFAULT_TILT_BOUND_DEG,
                      grasp=PLACE_DESCENT_GRASP, standoff=PLACE_DESCENT_STANDOFF,
                      setdown=PLACE_DESCENT_SETDOWN, tau=DEFAULT_PLACE_LAG,
                      ungentled_speed=0.08, ungentled_accel=0.15,
                      gentled_speed=0.02, gentled_accel=0.04, tool_link='ur_tool0'):
    """Replay the REAL collectlevel3 place-on-cart SET-DOWN descent on the REAL mission kinematics +
    measured grasp up-axis. Proves (1) the COMMANDED single-2-point straight joint line standoff->setdown
    is LEVEL (so the trip is NOT a planned-pose tilt and NOT the pre-place segment PR 298 covered), and
    (2) the EXECUTED path under the representative differential tracking lag reproduces the ~2.22 deg
    CARTON_TILT at the standard 0.08 rad/s but stays under the bound when the SAME level path is
    time-parameterised slower (carton_level.place_descent_speed). Returns a PlaceDescentResult."""
    robot = parse_urdf(urdf_path)
    u = cl.carton_up_in_tool(cl.tool_rotation(robot, list(grasp), tool_link))
    cmd, exe_u = tracking_series(robot, standoff, setdown, u, speed=ungentled_speed,
                                 accel=ungentled_accel, tau=tau, tool_link=tool_link)
    _, exe_g = tracking_series(robot, standoff, setdown, u, speed=gentled_speed,
                               accel=gentled_accel, tau=tau, tool_link=tool_link)
    dmax = max(abs(b - a) for a, b in zip(standoff, setdown))
    return PlaceDescentResult(
        commanded_max_pitch=max(abs(r[2]) for r in cmd),
        commanded_max_tilt=max(r[3] for r in cmd),
        ungentled_speed=ungentled_speed,
        ungentled_max_pitch=max(abs(r[2]) for r in exe_u),
        gentled_speed=gentled_speed,
        gentled_max_pitch=max(abs(r[2]) for r in exe_g),
        ungentled_duration=_descent_duration(dmax, ungentled_speed, ungentled_accel),
        gentled_duration=_descent_duration(dmax, gentled_speed, gentled_accel),
        commanded_series=cmd, ungentled_series=exe_u, gentled_series=exe_g)


# -------------------------------------------- SOUTH-DOCK loaded-droop place (sim-carton-level-southdock-v1)
# The REAL collect place-on-cart geometry at the PR-307 SOUTH dock (coldwin31) and the pre-307 EAST cart
# (collectlevel4), reconstructed from each run's host-stack: the grasp fixes the carton up-axis at the weld;
# the raw fixed-shape standoff is collect_place_preplace_shape; the sensed place TCP is the deck+approach
# pose the REPOSITION_REREGISTER logged (``place=(...)`` in ur_base_link, the frame the level place IK reaches).
SOUTHDOCK_GRASP = (1.1994, -1.2682, 2.0862, -0.8177, 1.2002, 0.0)
SOUTHDOCK_RAW_STANDOFF = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)
# place TCP in ur_base_link (REPOSITION_REREGISTER place=...): coldwin31 south dock, collectlevel4 east cart.
SOUTHDOCK_PLACE_TCP = (1.043, 0.055, 0.435)
EASTCART_PLACE_TCP = (0.97, 0.131, 0.442)


def _southdock_limits():
    big = 2 * math.pi
    return {'ur_wrist_1_joint': (-big, 0.65), 'ur_wrist_2_joint': (0.0, big),
            'ur_elbow_joint': (0.1, math.pi)}


@dataclass
class DroopPlaceResult:
    """Loaded gravity-droop place proof for ONE deck geometry (sim-carton-level-southdock-v1): the
    geometrically-level set-down command (current) DROOPS off level under the welded far-extended arm's
    gravity torque (reproducing coldwin31's +2.07 deg at the south dock / collectlevel4's sub-bound 1.86 at
    the east cart), while the gravity-FEEDFORWARD-compensated command lands the EXECUTED carton level."""
    name: str
    place_tcp: tuple
    shoulder_lift_torque: float       # gravity torque at the set-down (N*m) -- the dominant droop driver
    setdown_slift_swing: float        # the heavy-joint swing the (current) descent needs from the standoff
    current_cmd_tilt: float           # commanded (geometric) set-down tilt -- level by construction
    current_exec_pitch: float         # EXECUTED pitch under the droop model (the GPU trip quantity)
    current_exec_tilt: float
    comp_cmd_tilt: float              # compensated command is intentionally PRE-TILTED (not geometric level)
    comp_exec_pitch: float            # EXECUTED pitch after compensation -- level
    comp_exec_tilt: float
    descent_exec_max: float           # max |roll|,|pitch| over the compensated descent (standoff->setdown)
    standoff_exec_tilt: float         # executed tilt at the descent START (uncompensated standoff, pre-existing)
    setdown_clear: bool

    def passes(self, bound_deg, margin_deg=1.0):
        """The SET-DOWN (the coldwin31 trip site -- the far-extended loaded dwell) executes level within the
        brief's <=1 deg margin; the whole descent stays under the real bound. The descent START is the
        unchanged preplace standoff, whose (pre-existing, already-passing) droop caps the descent max."""
        return (self.comp_exec_pitch <= margin_deg and self.comp_exec_tilt <= margin_deg
                and self.descent_exec_max <= bound_deg and self.setdown_clear
                and self.comp_exec_tilt < self.current_exec_tilt)


def run_southdock_place(urdf_path, *, geometry='south', stiffness=cl.DEFAULT_DROOP_STIFFNESS,
                        carton_mass=0.45, standoff_above_m=0.09, tool_link='gripper_pad_tcp',
                        grasp=SOUTHDOCK_GRASP):
    """CPU proof of the loaded gravity-droop place on the REAL mission kinematics + the REAL fixed-shape
    standoff + the measured top-down grasp up-axis, for the PR-307 south dock (``geometry='south'``) or the
    pre-307 east cart (``'east'``). Shows (1) the geometrically-level set-down command droops the welded
    carton off level under the far-extended arm's gravity torque (the quasi-static mechanism, speed-
    independent), and (2) the gravity-FEEDFORWARD-compensated level place IK (``carton_level.
    gravity_compensation``) lands the EXECUTED carton level and keeps the whole descent under the bound.
    Returns a DroopPlaceResult. Pure/CPU."""
    place = SOUTHDOCK_PLACE_TCP if geometry == 'south' else EASTCART_PLACE_TCP
    robot = parse_urdf(urdf_path)
    u = cl.carton_up_in_tool(cl.tool_rotation(robot, list(grasp), tool_link))
    limits = _southdock_limits()
    gc = cl.GravityCompensation(stiffness=stiffness, carton_mass=carton_mass, tool_link=tool_link).validate()
    # The fixed-shape standoff, leveled (preserve its TCP) -- the descent START + the IK seed.
    seed = [cl.level_preserve_tcp(robot, list(SOUTHDOCK_RAW_STANDOFF), u, tool_link=tool_link,
                                  limits=limits)[j] for j in ARM_JOINTS]
    tgt = np.asarray(place, float)
    # CURRENT: geometric level set-down at the sensed place; droops under load.
    cur = [cl.level_place_ik(robot, tgt, u, seed, tool_link=tool_link, limits=limits)[j] for j in ARM_JOINTS]
    # COMPENSATED: solve the COMMANDED config whose EXECUTED (drooped) pose is level at the sensed place.
    comp = [cl.level_place_ik(robot, tgt, u, seed, tool_link=tool_link, limits=limits,
                              gravity_comp=gc)[j] for j in ARM_JOINTS]

    def exec_rpy(q):
        dq = gc.droop(robot, dict(zip(ARM_JOINTS, q)), tool_link=tool_link)
        return cl.carried_rpy(robot, [q[i] + dq[ARM_JOINTS[i]] for i in range(6)], u, tool_link)

    cur_cmd = cl.carried_rpy(robot, cur, u, tool_link)
    cur_exe = exec_rpy(cur)
    comp_cmd = cl.carried_rpy(robot, comp, u, tool_link)
    comp_exe = exec_rpy(comp)
    # compensated descent: standoff (seed) -> compensated set-down, executed attitude throughout
    dmax = 0.0
    for i in range(41):
        a = i / 40.0
        q = [(1 - a) * seed[k] + a * comp[k] for k in range(6)]
        r, p, _ = exec_rpy(q)
        dmax = max(dmax, abs(r), abs(p))
    tau = cl.arm_gravity_torques(robot, cur, carton_mass=carton_mass, tool_link=tool_link)
    return DroopPlaceResult(
        name=geometry, place_tcp=tuple(place),
        shoulder_lift_torque=tau['ur_shoulder_lift_joint'],
        setdown_slift_swing=cur[1] - seed[1],
        current_cmd_tilt=cur_cmd[2], current_exec_pitch=abs(cur_exe[1]), current_exec_tilt=cur_exe[2],
        comp_cmd_tilt=comp_cmd[2], comp_exec_pitch=abs(comp_exe[1]), comp_exec_tilt=comp_exe[2],
        descent_exec_max=dmax, standoff_exec_tilt=exec_rpy(seed)[2],
        setdown_clear=cl._in_clear_wrist(dict(zip(ARM_JOINTS, comp))))


def southdock_table(results, bound_deg=cl.DEFAULT_TILT_BOUND_DEG):
    rows = [('geometry', 'place_x', 'slift_torque', 'cur_cmd', 'cur_exec_pitch',
             'comp_cmd', 'comp_exec_pitch', 'descent_exec', 'verdict')]
    for r in results:
        rows.append((r.name, f'{r.place_tcp[0]:.3f}', f'{r.shoulder_lift_torque:.1f}',
                     f'{r.current_cmd_tilt:.2f}', f'{r.current_exec_pitch:.2f}',
                     f'{r.comp_cmd_tilt:.2f}', f'{r.comp_exec_pitch:.2f}', f'{r.descent_exec_max:.2f}',
                     'PASS' if r.passes(bound_deg) else 'FAIL'))
    w = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return '\n'.join('  '.join(c.ljust(w[i]) for i, c in enumerate(row)) for row in rows)


# ---------------------------------------- TWO-REGIME executed model (sim-carton-level-setdown-v1)
# The sim-carton-level-southdock-v1 "executed = command + droop(command)" model is CIRCULAR for the
# compensated command: level_place_ik solves the command so command+droop is level, and the predictor
# then evaluates command+droop -> always ~0. It therefore cannot express WHY coldwin33 still tripped at
# 2.06 deg with gravity_compensation on. The honest model, calibrated to BOTH GPU points (coldwin31 2.07
# UNCOMPENSATED and coldwin33 2.06 PR311-COMPENSATED), is TWO-REGIME:
#   * HEAVY pitch joints (shoulder_lift, elbow) are gravity-PINNED at the far-extended loaded set-down --
#     they SETTLE at geometric_level - tau/K regardless of a sub-degree command pre-tilt (coldwin33: the
#     executed carton moved 0.01 deg for a +2.04 deg heavy pre-tilt, d(exec)/d(cmd) ~ 0);
#   * the WRIST (and pan) TRACK their command faithfully (coldwin33 lift/tuck/preplace executed matched the
#     geometric plan within 0.02 deg -- the servo holds the light joints at setpoint).
# Under this model, a pre-tilt placed on the HEAVY joints (PR311's full-DOF solve) does NOT realise -> the
# carton stays ~2 deg off level (reproducing 2.06); a pre-tilt placed on the WRIST (the fix) DOES realise
# -> level. K=2850 still sizes the sag the wrist must cancel.
_HEAVY_PITCH_JOINTS = ('ur_shoulder_lift_joint', 'ur_elbow_joint')


@dataclass
class SetdownTwoRegimeResult:
    """Two-regime executed set-down for ONE geometry: the UNCOMPENSATED geometric command, the PR311-style
    HEAVY-routed pre-tilt (reproduces the coldwin33 trip), and the sim-carton-level-setdown-v1 WRIST-routed
    feedforward (the fix). ``*_exec_pitch`` are the EXECUTED |pitch| under the two-regime model."""
    name: str
    place_tcp: tuple
    uncomp_exec_pitch: float          # geometric-level command -> executed (the pre-comp droop)
    heavy_exec_pitch: float           # PR311 heavy-routed pre-tilt -> executed (still off level: the bug)
    wrist_exec_pitch: float           # the fix: wrist-routed pre-tilt -> executed (level)
    wrist_exec_tilt: float
    wrist_reach_err_mm: float         # executed set-down reach vs the sensed deck (place goal_tol 120 mm)
    wrist_pretilt_joint: str          # which joint carries the wrist fix's pre-tilt
    wrist_clear: bool

    def passes(self, margin_deg=1.0, reach_tol_m=0.12):
        """The fix lands the EXECUTED set-down level within the brief's <= 1 deg margin, strictly better than
        BOTH the uncompensated droop AND the PR311 heavy-routed command, within the place reach tolerance,
        in the clear wrist region."""
        return (self.wrist_exec_tilt <= margin_deg and self.wrist_exec_pitch <= margin_deg
                and self.wrist_exec_pitch < self.uncomp_exec_pitch
                and self.wrist_exec_pitch < self.heavy_exec_pitch
                and self.wrist_reach_err_mm <= reach_tol_m * 1000.0 and self.wrist_clear)


def run_setdown_tworegime(urdf_path, *, geometry='south', stiffness=cl.DEFAULT_DROOP_STIFFNESS,
                          carton_mass=0.45, tool_link='gripper_pad_tcp', grasp=SOUTHDOCK_GRASP):
    """CPU proof under the TWO-REGIME executed model (sim-carton-level-setdown-v1): the set-down executed
    pitch for the UNCOMPENSATED command (reproduces coldwin31/collectlevel4), the PR311 HEAVY-routed pre-tilt
    (reproduces coldwin33's residual 2.06 -- the pre-tilt on the gravity-pinned joints does not realise), and
    the WRIST-routed feedforward (the fix -> level). Pure/CPU. Returns a SetdownTwoRegimeResult."""
    import numpy as np
    place = SOUTHDOCK_PLACE_TCP if geometry == 'south' else EASTCART_PLACE_TCP
    robot = parse_urdf(urdf_path)
    u = cl.carton_up_in_tool(cl.tool_rotation(robot, list(grasp), tool_link))
    limits = _southdock_limits()
    gc = cl.GravityCompensation(stiffness=stiffness, carton_mass=carton_mass, tool_link=tool_link).validate()
    seed = [cl.level_preserve_tcp(robot, list(SOUTHDOCK_RAW_STANDOFF), u, tool_link=tool_link,
                                  limits=limits)[j] for j in ARM_JOINTS]
    tgt = np.asarray(place, float)
    geom = [cl.level_place_ik(robot, tgt, u, seed, tool_link=tool_link, limits=limits)[j] for j in ARM_JOINTS]
    wrist = [cl.level_place_ik(robot, tgt, u, seed, tool_link=tool_link, limits=limits,
                               gravity_comp=gc)[j] for j in ARM_JOINTS]
    # PR311-style heavy-routed pre-tilt: pre-command every joint by -droop (= +tau/K), which lands the
    # correction on the HEAVY pitch joints (the full-DOF solve's minimum-norm choice, coldwin33).
    sag_geom = gc.droop(robot, dict(zip(ARM_JOINTS, geom)), tool_link=tool_link)
    heavy = [geom[i] - sag_geom[ARM_JOINTS[i]] for i in range(6)]

    def executed_two_regime(cmd):
        return [(geom[i] + sag_geom[ARM_JOINTS[i]]) if ARM_JOINTS[i] in _HEAVY_PITCH_JOINTS else cmd[i]
                for i in range(6)]

    def pitch(cmd):
        return abs(cl.carried_rpy(robot, executed_two_regime(cmd), u, tool_link)[1])
    wrist_exec = cl.carried_rpy(robot, executed_two_regime(wrist), u, tool_link)
    reach_mm = 1000.0 * float(np.linalg.norm(cl.tool_position(robot, executed_two_regime(wrist), tool_link) - tgt))
    dj = {ARM_JOINTS[i]: wrist[i] - geom[i] for i in range(6)}
    pre_joint = max(dj, key=lambda j: abs(dj[j]))
    return SetdownTwoRegimeResult(
        name=geometry, place_tcp=tuple(place),
        uncomp_exec_pitch=pitch(geom), heavy_exec_pitch=pitch(heavy),
        wrist_exec_pitch=abs(wrist_exec[1]), wrist_exec_tilt=wrist_exec[2],
        wrist_reach_err_mm=reach_mm, wrist_pretilt_joint=pre_joint,
        wrist_clear=cl._in_clear_wrist(dict(zip(ARM_JOINTS, wrist))))


def setdown_tworegime_table(results, bound_deg=cl.DEFAULT_TILT_BOUND_DEG, margin_deg=1.0):
    rows = [('geometry', 'place_x', 'uncomp_exec', 'heavy_exec(PR311)', 'wrist_exec(fix)',
             'fix_pretilt_joint', 'reach_mm', 'verdict')]
    for r in results:
        rows.append((r.name, f'{r.place_tcp[0]:.3f}', f'{r.uncomp_exec_pitch:.2f}',
                     f'{r.heavy_exec_pitch:.2f}', f'{r.wrist_exec_pitch:.2f}',
                     r.wrist_pretilt_joint.replace('ur_', '').replace('_joint', ''),
                     f'{r.wrist_reach_err_mm:.0f}', 'PASS' if r.passes(margin_deg) else 'FAIL'))
    w = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return '\n'.join('  '.join(c.ljust(w[i]) for i, c in enumerate(row)) for row in rows)
