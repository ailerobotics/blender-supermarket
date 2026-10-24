"""SIM-ONLY scripted TRANSFER fallback -- ``RESTOCK_TRANSFER_MODE=scripted`` (default off => never built).

The guaranteed way past the transfer stage for the first walkthrough video while the real-arm transfer
(cuMotion pick off the rollwagen, place on the bay) is still being made reliable. With the switch on, the
transfer stage does NOT warm the station or dispatch the host manipulation job; this module moves the
carton from where it actually rests on the rollwagen to the authored ground-truth bay target along a
smooth scripted path shaped like a pick-and-place:

    reach_high -> approach -> contact -> [grasp] -> lift -> extract (out of the cage mouth)
    -> carry_high -> above_place -> place -> [release] -> retreat -> retreat_high -> stow

* Path: joint keyframes from a 6-DoF damped-least-squares IK on the mission's own imported URDF
  (``twin.urdf_probe`` FK) at the shelfbot side-grip stations (wall 3, the host's grip; near-face yaw
  choice as in live_pose); straight tool segments by continuation IK; every keyframe move timed
  rest-to-rest inside the host trajectory executor's speed/acceleration ceilings (its own Hermite checks).
* Arm (``RESTOCK_SCRIPTED_TRANSFER_ARM``): ``still`` (default) sends NO arm command -- the host executor
  keeps holding the arm, so nothing can trip its tracking watchdog or the Isaac arm heartbeat -- and the
  carton follows the planned tool path. ``executor`` (opt-in) sends each arm segment as one
  FollowJointTrajectory goal to that same executor (the path every cuMotion move takes); the carton follows
  the MEASURED tool. Driving the arm from Isaac directly is deliberately not offered: the executor
  idle-holds its last target, so a side-channel move trips the Isaac command envelope and its e-stop.
* Carton: kinematic with its colliders off while carried; at release it is set EXACTLY on the target (bay
  top + half height) and left kinematic (no dynamic re-add, the fm379 VRAM concern), so the ground-truth
  placement check passes by construction.
* Labels: transfer evidence ``transfer_mode=scripted`` + ``verify_mode=ground_truth`` + ``arm_mode`` /
  ``arm_follow`` / ``arm_fallback_reason``, the stage reason strings ('SCRIPTED TRANSFER ...'), mission.json
  transfer_mode / workarounds / note / walkthrough_caption (``transfer_mode.stamp_summary``) and the
  burned-in caption (``transfer_caption``).

Everything except ``IsaacScriptedDriver`` and the ``ExecutorArm`` subprocess is pure numpy and CPU-tested.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..restock_manipulation.ground_truth_collect import SOURCE as GT_SOURCE
from ..restock_manipulation.transfer_carry import (DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN,
                                                   clear_branch, in_clear_region, wrist_flip_seed)
from ..restock_manipulation.transfer_mode import SCRIPTED, transfer_mode
from .mission import Result

ARM_JOINTS = ('ur_shoulder_pan_joint', 'ur_shoulder_lift_joint', 'ur_elbow_joint',
              'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint')
GRASP_QUAT = (.5, .5, .5, .5)          # shelfbot side-grip grasp_quat (x, y, z, w)
HANDLING_UNIT_PATH = '/RestockScenario/HandlingUnit'


# ---------------------------------------------------------------- pose math (x, y, z, w quaternions)

def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def yaw_quat(yaw):
    return (0., 0., math.sin(yaw / 2.), math.cos(yaw / 2.))


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_T(xyz, quat):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(quat)
    T[:3, 3] = xyz
    return T


def T_yaw(T):
    return math.atan2(T[1, 0], T[0, 0])


def rot_err(R_target, R):
    """Rotation vector taking R to R_target (world frame)."""
    E = R_target @ R.T
    c = max(-1.0, min(1.0, (np.trace(E) - 1.0) / 2.0))
    angle = math.acos(c)
    if angle < 1e-9:
        return np.zeros(3)
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    s = math.sin(angle)
    if s < 1e-6:                                    # ~pi: use the symmetric part
        B = (E + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(B), 0.0, None))
        axis = axis / (np.linalg.norm(axis) or 1.0)
        return axis * angle
    return v / (2.0 * s) * angle


def side_grip(centre, yaw, *, dims, wall, contact_margin, standoff, grasp_quat=GRASP_QUAT):
    """shelfbot ``wall_grip`` + ``approach_contact`` (zero pad offset) for a carton CENTRE + yaw, all in one
    frame: returns (contact_T, approach_T, outward_xy). The pad meets the gripped wall at centre height;
    ``outward_xy`` points from the carton centre toward the pad (the approach/retreat direction)."""
    angle = yaw + wall * (math.pi / 2.0)
    half = (dims[0] if wall % 2 == 0 else dims[1]) / 2.0
    d = half + contact_margin
    c, s = math.cos(angle), math.sin(angle)
    q = quat_mul(yaw_quat(angle), grasp_quat)
    contact = pose_T((centre[0] - d * c, centre[1] - d * s, centre[2]), q)
    approach = pose_T((centre[0] - (d + standoff) * c, centre[1] - (d + standoff) * s, centre[2]), q)
    return contact, approach, np.array([-c, -s])


# ---------------------------------------------------------------- 6-DoF IK on the imported URDF

class ArmIK:
    """Position + orientation damped-least-squares IK of ``tip_link`` in ``base_link`` (numeric Jacobian
    on the twin.urdf_probe FK of the mission's own imported URDF). Pure numpy."""

    def __init__(self, urdf_path, *, arm_joints=ARM_JOINTS, base_link='ur_base_link',
                 tip_link='gripper_pad_tcp'):
        from twin import urdf_probe as U
        self._U = U
        robot = U.parse_urdf(str(urdf_path))
        self.arm = list(arm_joints)
        child_joint = {j.child: j for j in robot.joints}
        chain, link = [], tip_link
        for _ in range(64):
            j = child_joint.get(link)
            if j is None:
                raise ValueError(f'broken URDF chain above {link!r}')
            chain.append(j)
            link = j.parent
            if link == base_link:
                break
        else:
            raise ValueError(f'{tip_link!r} is not below {base_link!r}')
        self.chain = chain[::-1]
        by_name = {j.name: j for j in robot.joints}
        missing = [n for n in self.arm if n not in by_name]
        if missing:
            raise ValueError(f'arm joints missing from the URDF: {missing}')
        self.lower = np.array([by_name[n].lower if by_name[n].lower is not None else -2 * math.pi
                               for n in self.arm], float)
        self.upper = np.array([by_name[n].upper if by_name[n].upper is not None else 2 * math.pi
                               for n in self.arm], float)
        self._index = {n: i for i, n in enumerate(self.arm)}

    def fk(self, q):
        T = np.eye(4)
        for j in self.chain:
            i = self._index.get(j.name)
            T = T @ self._U.joint_transform(j, 0.0 if i is None else float(q[i]))
        return T

    def error(self, T_target, q):
        T = self.fk(q)
        return np.concatenate([T_target[:3, 3] - T[:3, 3], rot_err(T_target[:3, :3], T[:3, :3])])

    def solve(self, T_target, q0, *, iters=400, lam=0.05, eps=1e-6, step=0.25, rot_weight=0.5):
        """Returns (q, position_error_m, rotation_error_rad) from the seed q0."""
        q = np.clip(np.asarray(q0, float), self.lower, self.upper)
        W = np.diag([1., 1., 1., rot_weight, rot_weight, rot_weight])
        for _ in range(iters):
            e = self.error(T_target, q)
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 5e-3:
                break
            J = np.zeros((6, 6))
            for i in range(6):
                dq = q.copy()
                dq[i] += eps
                J[:, i] = -(self.error(T_target, dq) - e) / eps
            Jw, ew = W @ J, W @ e
            dqv = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam ** 2 * np.eye(6), ew)
            n = np.max(np.abs(dqv))
            if n > step:
                dqv *= step / n
            q = np.clip(q + dqv, self.lower, self.upper)
        e = self.error(T_target, q)
        return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))


# ---------------------------------------------------------------- plan

@dataclass
class Keyframe:
    name: str
    tcp: np.ndarray                 # 4x4 TCP pose in the arm base frame (ur_base_link)
    duration: float                 # seconds to reach this keyframe from the previous one
    carried: bool                   # carton attached to the tool on arrival
    line: bool = False              # straight TCP line from the previous keyframe (continuation IK)
    q: np.ndarray | None = None     # joint solution at the keyframe (None => no IK solution)
    path: list = field(default_factory=list)   # joint waypoints from the previous keyframe, ending at q
    ik_error: tuple = (float('nan'), float('nan'))
    samples: list = field(default_factory=list)   # rest-to-rest [(t, q, v)] from the previous keyframe


@dataclass
class TransferPlan:
    keyframes: list
    q_start: np.ndarray
    carton_start_base: np.ndarray   # 4x4
    carton_goal_base: np.ndarray    # 4x4 (carton centre on the target, chosen yaw)
    grasp_offset: np.ndarray        # 4x4 TCP -> carton (rigid while carried)
    ik_complete: bool               # every keyframe solved => a joint path exists (arm can follow)
    choice: dict = field(default_factory=dict)
    wrist_guard: tuple | None = None   # (wrist1_cap, wrist2_min) when the carry wrist guard is on, else None

    @property
    def duration(self):
        return sum(k.duration for k in self.keyframes) + 2 * DWELL_S

    def wrist1_within_cap(self):
        """With the guard on, True iff every MOVING joint waypoint (keyframe paths; the parked 'stow' pose is
        excluded) commands ur_wrist_1 at or below the clear cap -- i.e. no sent sample sweeps the arm through
        the ur_wrist_1-vs-ur_forearm stop. Guard off => True. The companion to in_clear_region that tests the
        RAW commanded angle, so a far-2*pi branch that wraps 'clear' does not pass."""
        if self.wrist_guard is None:
            return True
        cap = self.wrist_guard[0]
        for k in self.keyframes:
            if k.name == 'stow':
                continue
            for q in k.path:
                if float(q[3]) > cap + 1e-6:
                    return False
        return True


DWELL_S = 0.6        # grasp / release dwell (s)
_POS_TOL, _ROT_TOL = 4e-3, 0.04   # IK acceptance (m, rad)
_LINE_STEP = 0.02                 # continuation step along a straight TCP line (m)
_JUMP_MAX = 0.35                  # max joint change per continuation step (rad) -- stays on one branch


def plan_transfer(ik, q_start, carton_base, goal_centre_base, goal_yaw_base, *, dims, wall=3,
                  contact_margin=0.00275, standoff=0.15, lift=0.03, clear=0.15, extract_margin=0.06,
                  time_scale=1.0, wrist_guard=True, wrist1_cap=DEFAULT_WRIST1_CAP,
                  wrist2_min=DEFAULT_WRIST2_MIN, place_approach='above', insert_lift=0.015):
    """Keyframes for the scripted pick (carton pose ``carton_base``, 4x4 in ur_base_link) and place (carton
    centre ``goal_centre_base`` at yaw ``goal_yaw_base``). The carton and the target are symmetric under a
    180 deg yaw: the representatives that minimise the worst grip-station reach are used (live_pose's
    near-face rule), so the gripped wall faces the arm at both ends. Straight segments (approach, contact,
    lift, extract, raise, place, retreat) are solved by continuation IK so the TCP really moves on a line;
    the extract stops where the arm runs out of reach (recorded as extract_m vs extract_wanted_m).

    ``wrist_guard`` (default on): every seeded keyframe IK must land in the bench-proven CLEAR wrist region
    (wrist_1 <= ``wrist1_cap``, wrist_2 >= ``wrist2_min``; the live_pose ``RESTOCK_TRANSFER_CARRY=wrist_guard``
    predicate), re-seeded from its WRIST-FLIPPED twin when the nearest branch is not clear. The TCP poses are
    unchanged (same tool path, so the still-mode carton path is byte-identical), but the arm follows the clear
    branch -- without it the scripted executor pick picks the ur_gripper-vs-ur_forearm self-colliding branch
    (wrist_1 ~1.37, wrist_2 ~-1.27) and the host trajectory executor faults on the >1.3 rad tracking error
    (warm5, 2026-10-22). The place half already resolves clear; the guard carries the PICK half onto it too
    via the continuation IK seeded from the clear ``reach_high``.

    ``place_approach`` (default ``'above'``, byte-identical): the carton is lowered onto the target from
    ``clear`` above it -- right for the open backroom bay, but under a shelf board (the sales-floor endcap
    slot G3_back_s2 has only ~0.25 m to the s3 board) that descent drives the carton through the board.
    ``'front'`` (opt-in, RESTOCK_SCRIPTED_TRANSFER_PLACE=front) keeps the keyframe names and moves the
    pre-place pose IN FRONT of the slot -- the carton just outside it along the gripped wall's outward
    normal, ``insert_lift`` above its final height -- reached by a straight TCP line (a joint-space swing
    there overshoots the turning carton's corners into the upper board's front edge), so ``place`` becomes a
    level slide into the slot. That line is not continuation-IK feasible on the mission arm, so the plan is
    task-space only (``ik_complete`` False): built for the still arm (warm8's mode), where the carton follows
    the planned tool path; an ``executor`` arm falls back to still, as for any plan without a joint path."""
    if place_approach not in PLACE_MODES:
        raise ValueError(f'place_approach {place_approach!r} unknown (known: {", ".join(PLACE_MODES)})')
    guard = (wrist1_cap, wrist2_min) if wrist_guard else None
    q_start = np.asarray(q_start, float)
    c0 = carton_base[:3, 3]
    psi0 = T_yaw(carton_base)
    best = None
    for f0 in (0.0, math.pi):
        for f1 in (0.0, math.pi):
            pc, pa, po = side_grip(c0, psi0 + f0, dims=dims, wall=wall, contact_margin=contact_margin,
                                   standoff=standoff)
            gc, ga, go = side_grip(goal_centre_base, goal_yaw_base + f1, dims=dims, wall=wall,
                                   contact_margin=contact_margin, standoff=standoff)
            worst = max(np.linalg.norm(T[:3, 3]) for T in (pc, pa, gc, ga))
            if best is None or worst < best[0] - 1e-9:
                best = (worst, f0, f1, pc, pa, po, gc, ga, go)
    worst, f0, f1, pick_c, pick_a, out0, _place_c, _place_a, out1 = best
    # the gripped carton keeps its pick-time pose relative to the tool, so the place TCP is the goal carton
    # pose (f1 representative) composed with the inverse grasp offset (f0 representative at the pick)
    start_rep = carton_base @ pose_T((0., 0., 0.), yaw_quat(f0))
    offset = np.linalg.inv(pick_c) @ start_rep
    goal = pose_T(goal_centre_base, yaw_quat(goal_yaw_base + f1))
    place_tcp = goal @ np.linalg.inv(offset)
    up = lambda T, dz: _shift(T, (0., 0., dz))
    depth = (dims[1] if wall % 2 == 1 else dims[0]) + extract_margin
    s = float(time_scale)
    choice = dict(source_flipped=bool(f0), target_flipped=bool(f1), worst_station_reach=float(worst),
                  extract_wanted_m=float(depth))
    lifted = up(pick_c, lift)
    plan_k = [Keyframe('reach_high', up(pick_a, clear), 3.0 * s, False),
              Keyframe('approach', pick_a, 2.0 * s, False, line=True),
              Keyframe('contact', pick_c, 1.5 * s, False, line=True),
              Keyframe('lift', lifted, 1.0 * s, True, line=True),
              Keyframe('extract', _shift(lifted, (out0[0] * depth, out0[1] * depth, 0.)), 2.5 * s, True,
                       line=True)]
    retreat = _shift(place_tcp, (out1[0] * standoff, out1[1] * standoff, 0.))
    if place_approach == PLACE_FRONT:
        # slide in level, insert_lift above the slot (clear of its front ticket rail); the release sets the
        # carton exactly on the target, as in every mode
        pre_place = up(_shift(place_tcp, (out1[0] * depth, out1[1] * depth, 0.)), insert_lift)
        place_at = up(place_tcp, insert_lift)
        choice['place_approach'] = PLACE_FRONT
    else:
        pre_place, place_at = up(place_tcp, clear), place_tcp
    tail = [Keyframe('above_place', pre_place, 4.0 * s, True, line=place_approach == PLACE_FRONT),
            Keyframe('place', place_at, 2.0 * s, True, line=True),
            Keyframe('retreat', retreat, 1.5 * s, False, line=True),
            Keyframe('retreat_high', up(retreat, clear), 1.0 * s, False, line=True)]
    prev_q, prev_T, ik_complete = q_start, ik.fk(q_start), True
    keyframes = []
    for k in plan_k + [None] + tail:
        if k is None:                           # raise straight up from wherever the extract ended
            k = Keyframe('carry_high', up(keyframes[-1].tcp, clear), 1.5 * s, True, line=True)
        keyframes.append(k)
        if not ik_complete:
            continue
        if k.line:
            path, T_end, perr, rerr = _line_path(ik, prev_T, k.tcp, prev_q, adaptive=(k.name == 'extract'))
            if path:
                k.tcp, k.path, k.q = T_end, path, path[-1]
        else:
            q, perr, rerr = _solve_seeded(ik, k.tcp, prev_q, guard=guard)
            if perr <= _POS_TOL and rerr <= _ROT_TOL:
                k.q, k.path = q, [q]
        k.ik_error = (perr, rerr)
        if k.q is None:
            ik_complete = False
            continue
        prev_q, prev_T = k.q, k.tcp
    if keyframes[4].q is not None:
        choice['extract_m'] = float(np.linalg.norm(keyframes[4].tcp[:2, 3] - keyframes[3].tcp[:2, 3]))
    keyframes.append(Keyframe('stow', ik.fk(q_start), 3.0 * s, False, q=q_start.copy(), path=[q_start.copy()],
                              ik_error=(0., 0.)))
    if ik_complete:                             # rest-to-rest timing inside the executor's own limits
        prev = q_start
        for k in keyframes:
            k.samples = time_keyframe(prev, k.path, minimum=k.duration, limits=(ik.lower, ik.upper))
            k.duration = k.samples[-1][0] if k.samples else 0.0
            prev = k.q
    else:                                       # no joint path: the carton follows the task-space tool path
        for k in keyframes:
            k.q, k.path = None, []
    return TransferPlan(keyframes, q_start, carton_base, goal, offset, ik_complete, choice,
                        wrist_guard=guard)


def _line_path(ik, T_from, T_to, q_from, *, adaptive=False):
    """Continuation IK along the straight TCP line T_from -> T_to (orientation interpolated along the
    geodesic). Returns (joint path, end pose reached, pos_err, rot_err); an empty path => infeasible.
    ``adaptive`` keeps the feasible prefix (the line is cut where the arm runs out of reach)."""
    dist = float(np.linalg.norm(T_to[:3, 3] - T_from[:3, 3]))
    n = max(1, int(math.ceil(dist / _LINE_STEP)))
    path, q, T_end, err = [], np.asarray(q_from, float), T_from, (0.0, 0.0)
    for i in range(1, n + 1):
        T = _interp_T(T_from, T_to, i / n)
        qn, perr, rerr = ik.solve(T, q, iters=150)
        if perr > _POS_TOL or rerr > _ROT_TOL or np.max(np.abs(qn - q)) > _JUMP_MAX:
            if adaptive and path:
                break
            return [], T_from, perr, rerr
        path.append(qn)
        q, T_end, err = qn, T, (perr, rerr)
    return path, T_end, err[0], err[1]


def _shift(T, d):
    out = T.copy()
    out[:3, 3] = out[:3, 3] + np.asarray(d, float)
    return out


def _solve_seeded(ik, T, prev, *, guard=None):
    """IK from the previous keyframe, then from pan-retargeted variants of it; the converged solution
    closest to the previous configuration wins (keeps the swing on one branch). With ``guard`` =
    (wrist1_cap, wrist2_min) set, a WRIST-FLIPPED twin of each seed is also tried and an in-clear-region
    solution is preferred over a colliding one (the same clear carry branch the live_pose wrist guard uses),
    so the scripted pick never seeds the ur_gripper-vs-ur_forearm self-collision branch."""
    prev = np.asarray(prev, float)
    target_pan = math.atan2(T[1, 3], T[0, 3])
    seeds = [prev]
    for pan in (target_pan, target_pan - 2 * math.pi, target_pan + 2 * math.pi):
        if ik.lower[0] <= pan <= ik.upper[0]:
            s = prev.copy()
            s[0] = pan
            seeds.append(s)
    if guard is not None:                               # also seed the clear (wrist-flipped) branch
        seeds = seeds + [np.array(wrist_flip_seed(s)) for s in list(seeds)]
    best = None
    for seed in seeds:
        q, perr, rerr = ik.solve(T, seed)
        ok = perr <= _POS_TOL and rerr <= _ROT_TOL
        # With the guard on, COMMAND the clear wrist_1 branch: an IK solve can unwrap wrist_1 onto a far
        # 2*pi branch (raw ~+4.1) that reads as clear under in_clear_region's wrap but makes the joint sweep
        # up through the ~0.69 self-collision stop (the execretest executor-pick jam). clear_branch shifts it
        # to the equivalent in-limit branch <= cap -- identical tool pose, swept path below the stop.
        if guard is not None and ok:
            cb = clear_branch(q, guard[0], ik.lower, ik.upper)
            if cb is not None:
                q = np.array(cb)
        # Prefer a reachable, IN-CLEAR-REGION branch; among equals, the one closest to the previous config.
        clear_penalty = (guard is not None and ok and not in_clear_region(q, guard[0], guard[1]))
        key = (not ok, clear_penalty, float(np.linalg.norm(q - prev)) if ok else perr + rerr)
        if best is None or key < best[0]:
            best = (key, q, perr, rerr)
    return best[1], best[2], best[3]


def _smooth(u):
    u = min(1.0, max(0.0, u))
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _interp_T(Ta, Tb, w):
    """Pose interpolation (lerp position, rotation along the shortest geodesic)."""
    out = np.eye(4)
    out[:3, 3] = (1 - w) * Ta[:3, 3] + w * Tb[:3, 3]
    v = rot_err(Tb[:3, :3], Ta[:3, :3]) * w
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        out[:3, :3] = Ta[:3, :3]
    else:
        k = v / angle
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rv = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
        out[:3, :3] = Rv @ Ta[:3, :3]
    return out


# ---------------------------------------------------------------- timing (the arm executor's own limits)

EXECUTOR_MAX_SPEED = 0.3                                  # trajectory_host --max-speed default (rad/s)
EXECUTOR_MAX_ACCEL = EXECUTOR_MAX_SPEED * (.5 / .15)      # trajectory_host's acceleration ceiling for it
TIMING_MARGIN = 0.8          # profile at 80% of both ceilings; the Hermite check below has the final say
SAMPLE_DT = 0.2              # trajectory point spacing (s)
# One FollowJointTrajectory goal per arm segment; the grasp / release dwells sit between them.
SEGMENTS = (('pick', ('reach_high', 'approach', 'contact')),
            ('carry', ('lift', 'extract', 'carry_high', 'above_place', 'place')),
            ('retreat', ('retreat', 'retreat_high', 'stow')))


def hermite_violation(points, limits=None, *, max_speed=EXECUTOR_MAX_SPEED,
                      max_acceleration=EXECUTOR_MAX_ACCEL):
    """trajectory_host.Plan's acceptance rules on [(t, q, v)] points -- start at rest at t=0, end at rest,
    and every cubic-Hermite segment inside the joint limits, speed and acceleration ceilings. Returns the
    first violation (str) or None. Mirrors Plan exactly (the CPU test checks both agree)."""
    if (len(points) < 2 or points[0][0] != 0 or np.max(np.abs(points[0][2])) > 1e-3
            or np.max(np.abs(points[-1][2])) > 1e-3):
        return 'plan must start at rest at zero, end at rest and contain two points'
    for (t0, q0, v0), (t1, q1, v1) in zip(points, points[1:]):
        d = t1 - t0
        if not d > 0:
            return 'trajectory time not increasing'
        a = (2 * (q0 - q1) + d * (v0 + v1)) / d ** 3
        b = (3 * (q1 - q0) - d * (2 * v0 + v1)) / d ** 2
        for j in range(6):
            if limits is not None:
                ts = {0., d}
                ts.update(float(r.real) for r in np.roots([3 * a[j], 2 * b[j], v0[j]])
                          if abs(r.imag) < 1e-10 and 0 < r.real < d)
                if any(not limits[0][j] <= ((a[j] * t + b[j]) * t + v0[j]) * t + q0[j] <= limits[1][j]
                       for t in ts):
                    return f'interpolated joint limit violation (joint {j})'
            st = {0., d}
            if abs(a[j]) > 1e-15 and 0 < -b[j] / (3 * a[j]) < d:
                st.add(-b[j] / (3 * a[j]))
            if max(abs(3 * a[j] * t * t + 2 * b[j] * t + v0[j]) for t in st) > max_speed + 1e-9:
                return f'interpolated speed limit violation (joint {j})'
            if max(abs(2 * b[j]), abs(6 * a[j] * d + 2 * b[j])) > max_acceleration + 1e-9:
                return f'interpolated acceleration limit violation (joint {j})'
    return None


def time_keyframe(q_from, path, *, minimum=0.0, limits=None, max_speed=EXECUTOR_MAX_SPEED,
                  max_acceleration=EXECUTOR_MAX_ACCEL, dt=SAMPLE_DT):
    """Rest-to-rest samples [(t, q, v)] (t from 0) from ``q_from`` through the joint waypoints ``path``:
    a cosine profile on the path's max-norm arc length, at most ``TIMING_MARGIN`` of the executor's speed
    and acceleration ceilings, slowed until every Hermite segment passes ``hermite_violation``. [] when
    there is no motion. ``minimum`` is a floor on the duration (s)."""
    pts = np.vstack([np.asarray(q_from, float)] + [np.asarray(q, float) for q in path])
    seg = np.max(np.abs(np.diff(pts, axis=0)), axis=1) if len(pts) > 1 else np.zeros(0)
    length = float(seg.sum())
    if length < 1e-6:
        return []
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    v, a = TIMING_MARGIN * max_speed, TIMING_MARGIN * max_acceleration
    T = max(float(minimum), math.pi * length / (2 * v), math.pi * math.sqrt(length / (2 * a)))
    for _ in range(16):
        samples = _cosine_samples(pts, cum, length, T, max(2, int(math.ceil(T / dt))))
        if hermite_violation(samples, limits, max_speed=max_speed, max_acceleration=max_acceleration) is None:
            return samples
        T *= 1.3
    raise ValueError('scripted transfer: no executor-feasible timing for the joint path')


def _cosine_samples(pts, cum, length, T, n):
    def q_at(t):
        s = length * (0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, t / T))))
        i = int(min(len(cum) - 2, max(0, np.searchsorted(cum, s, side='right') - 1)))
        span = cum[i + 1] - cum[i]
        return pts[i] + (0.0 if span <= 0 else (s - cum[i]) / span) * (pts[i + 1] - pts[i])
    h = T / n / 2.0
    out = []
    for k in range(n + 1):
        t = T * k / n
        out.append((t, q_at(t), np.zeros(6) if k in (0, n) else (q_at(t + h) - q_at(t - h)) / (2 * h)))
    return out


def segment_points(keyframes, names):
    """The keyframes' rest-to-rest samples for one executor segment, concatenated on one clock."""
    out = []
    for k in keyframes:
        if k.name not in names or not k.samples:
            continue
        t0 = out[-1][0] if out else 0.0
        out.extend((t0 + t, q, v) for t, q, v in (k.samples[1:] if out else k.samples))
    return out


def _sample_q(samples, t):
    """Joint position at time t (linear between the dense samples)."""
    ts = [x[0] for x in samples]
    i = int(min(len(ts) - 2, max(0, np.searchsorted(ts, t, side='right') - 1)))
    span = ts[i + 1] - ts[i]
    u = 0.0 if span <= 0 else min(1.0, max(0.0, (t - ts[i]) / span))
    return (1 - u) * samples[i][1] + u * samples[i + 1][1]


# ---------------------------------------------------------------- stage runner

ARM_STILL, ARM_EXECUTOR = 'still', 'executor'
ARM_MODES = (ARM_STILL, ARM_EXECUTOR)
ARM_ENV = 'RESTOCK_SCRIPTED_TRANSFER_ARM'
# Place approach (plan_transfer ``place_approach``): 'above' = the default top-down descent onto the bay;
# 'front' = opt-in slide-in under an upper shelf board (the sales-floor endcap slot G3_back_s2).
PLACE_ABOVE, PLACE_FRONT = 'above', 'front'
PLACE_MODES = (PLACE_ABOVE, PLACE_FRONT)
PLACE_ENV = 'RESTOCK_SCRIPTED_TRANSFER_PLACE'
PLACE_LABEL = 'transfer_place=front_insert'
PLACE_CAPTION = 'TRANSFER PLACE: scripted front insert into the shelf slot (under the upper board)'
PLACE_NOTE = (' TRANSFER_PLACE=front_insert: the scripted carton path slides into the shelf slot from in '
              'front of it (straight line under the upper board) instead of descending from above.')


def scripted_place_approach(environ=None):
    """``above`` (default) or ``front``: env RESTOCK_SCRIPTED_TRANSFER_PLACE. Unknown => ValueError (fail loud
    at boot). Only consulted when RESTOCK_TRANSFER_MODE=scripted builds the scripted transfer."""
    env = os.environ if environ is None else environ
    raw = env.get(PLACE_ENV) or PLACE_ABOVE
    value = str(raw).strip().lower()
    if value not in PLACE_MODES:
        raise ValueError(f'{PLACE_ENV}={raw!r}: unknown scripted-transfer place approach '
                         f'(known: {", ".join(PLACE_MODES)})')
    return value


def _front_place_active(environ=None):
    try:
        return transfer_mode(environ) == SCRIPTED and scripted_place_approach(environ) == PLACE_FRONT
    except ValueError:
        return False


def place_caption(base=None, environ=None):
    """The walkthrough caption with the front-insert place named (scripted mode + PLACE=front only);
    otherwise ``base`` unchanged (byte-identical)."""
    if not _front_place_active(environ) or (base and PLACE_CAPTION in base):
        return base
    return f'{base} | {PLACE_CAPTION}' if base else PLACE_CAPTION


def stamp_place_summary(summary, environ=None):
    """Record the front-insert place in a mission.json summary dict (in place; returned): ``transfer_place_mode``
    + the ``transfer_place=front_insert`` workaround + note + caption. Off => untouched (byte-identical)."""
    if not _front_place_active(environ):
        return summary
    summary['transfer_place_mode'] = 'front_insert'
    workarounds = summary.setdefault('workarounds', [])
    if PLACE_LABEL not in workarounds:
        workarounds.append(PLACE_LABEL)
    note = summary.get('note') or ''
    if PLACE_NOTE not in note:
        summary['note'] = note + PLACE_NOTE
    summary['walkthrough_caption'] = place_caption(summary.get('walkthrough_caption'), environ)
    return summary


EXECUTOR_START_TOL = 0.05     # validate_trajectory: first waypoint vs measured joints (rad)
EXECUTOR_SETTLE_S = 60.0      # sim seconds past a segment's end for the executor's terminal settle
# IK-plan latency budget (WALL seconds) between begin() (which runs the whole transfer IK) and the arm
# goal being sent. Reported every run; over-budget is logged LOUD, not fatal: the executor now re-stamps
# the trajectory origin to the first post-feedback stamp (trajectory_host.anchored_start), so a slow IK no
# longer races the execution clock -- but a creeping latency is still worth surfacing. Env override:
# RESTOCK_SCRIPTED_TRANSFER_IK_BUDGET_S (0/negative disables the warning; the latency is still reported).
IK_PLAN_BUDGET_DEFAULT_S = 120.0


def ik_plan_budget_seconds(environ=None):
    """``RESTOCK_SCRIPTED_TRANSFER_IK_BUDGET_S`` (default 120 s). <=0 disables the over-budget warning.
    Unparseable => ValueError (fail loud at boot). Pure."""
    env = os.environ if environ is None else environ
    raw = env.get('RESTOCK_SCRIPTED_TRANSFER_IK_BUDGET_S')
    if raw is None or str(raw).strip() == '':
        return IK_PLAN_BUDGET_DEFAULT_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f'RESTOCK_SCRIPTED_TRANSFER_IK_BUDGET_S={raw!r} is not a number of seconds')
    if not math.isfinite(value):
        raise ValueError('RESTOCK_SCRIPTED_TRANSFER_IK_BUDGET_S must be finite')
    return value


def ik_latency_report(latency_s, budget_s):
    """One line describing the begin()->send IK-plan latency and whether it is over budget. Returns
    ``(over_budget: bool, line: str)``. ``budget_s <= 0`` disables the verdict (never over). Pure, so the
    report is CPU-testable without a live transfer."""
    latency_s = float(latency_s)
    over = budget_s > 0 and latency_s > float(budget_s)
    verdict = (f'OVER budget {budget_s:g}s' if over else
               (f'within budget {budget_s:g}s' if budget_s > 0 else 'budget disabled'))
    return over, f'ik_plan_latency={latency_s:.1f}s ({verdict})'


class ScriptedTransfer:
    """The transfer stage, scripted. ``begin()`` once (robot parked at the bay park), then ``step()`` each
    mission tick until it returns a terminal Result. The driver supplies the measured state + carton
    actuation; ``arm`` (an ExecutorArm, or None) moves the arm.

    * ``still`` (arm None): no arm command at all -- the host trajectory executor keeps holding the arm, so
      its tracking watchdog and the Isaac arm heartbeat stay exactly as in any parked stage. The carton
      follows the PLANNED tool path (FK of the planned joint path x grasp offset).
    * ``executor``: each arm segment (pick / carry / retreat) is ONE FollowJointTrajectory goal to the
      mission's own executor (trajectory_host), the same path every cuMotion move takes, timed inside its
      speed/acceleration ceilings; the carton follows the MEASURED tool. A goal the executor REJECTS
      (nothing latched) falls back to ``still`` for the rest of the timeline; an accepted goal that aborts
      (the executor has e-stopped) fails the stage honestly.
    """

    def __init__(self, driver, *, ik, target_xyz_world, target_yaw_world=0.0, dims, location, sku,
                 quantity=1, wall=3, contact_margin=0.00275, standoff=0.15, time_scale=1.0,
                 drop_gap=0.002, xy_tol=0.25, z_tol=0.12, arm=None, place_approach=PLACE_ABOVE,
                 ik_plan_budget_s=IK_PLAN_BUDGET_DEFAULT_S):
        self.driver, self.ik, self.arm = driver, ik, arm
        self.ik_plan_budget_s = float(ik_plan_budget_s)
        self.ik_plan_latency_s = None       # begin()->plan WALL latency (the transfer IK), reported below
        self.place_approach = place_approach
        self.target_xyz_world = tuple(float(v) for v in target_xyz_world)
        self.target_yaw_world = float(target_yaw_world)
        self.dims = tuple(float(v) for v in dims)
        self.location, self.sku, self.quantity = location, sku, int(quantity)
        self.wall, self.contact_margin, self.standoff = int(wall), float(contact_margin), float(standoff)
        self.time_scale, self.drop_gap = float(time_scale), float(drop_gap)
        self.xy_tol, self.z_tol = float(xy_tol), float(z_tol)
        self.arm_mode = ARM_EXECUTOR if arm is not None else ARM_STILL
        self.fallback_reason = None
        self.plan = None
        self._guard_fail = None
        self.t = 0.0
        self.state = 'idle'
        self.phase = None

    # -- lifecycle
    def begin(self, job=None, now=None):
        d = self.driver
        if job is not None:                         # evidence identity = the mission's own job
            self.sku = getattr(job, 'sku', self.sku)
            self.location = getattr(job, 'shelf_id', self.location)
            self.quantity = int(getattr(job, 'quantity', self.quantity))
        self.base_world = np.asarray(d.base_world_T(), float)
        world_to_base = np.linalg.inv(self.base_world)
        carton_world = np.asarray(d.carton_world_T(), float)
        carton_base = world_to_base @ carton_world
        gx, gy, gz = self.target_xyz_world
        goal_world = np.array([gx, gy, gz + self.dims[2] / 2.0 + self.drop_gap, 1.0])
        goal_base = (world_to_base @ goal_world)[:3]
        goal_yaw_base = self.target_yaw_world - T_yaw(self.base_world)
        _ik_t0 = time.perf_counter()
        self.plan = plan_transfer(self.ik, d.arm_joint_positions(), carton_base, goal_base, goal_yaw_base,
                                  dims=self.dims, wall=self.wall, contact_margin=self.contact_margin,
                                  standoff=self.standoff, place_approach=self.place_approach)
        # IK-plan latency: the WALL time spent building the joint path here (the dominant begin()->send
        # cost). Reported every run; the executor's feedback-clock re-anchor (trajectory_host) means a slow
        # plan no longer races the execution clock, but a creeping latency is surfaced LOUD when over budget.
        self.ik_plan_latency_s = time.perf_counter() - _ik_t0
        over, line = ik_latency_report(self.ik_plan_latency_s, self.ik_plan_budget_s)
        print('SCRIPTED_TRANSFER %s %s' % ('WARNING ik-plan slow:' if over else 'ik-plan:', line), flush=True)
        self.goal_world = self.base_world @ self.plan.carton_goal_base
        self.t, self.state, self.carried, self.released = 0.0, 'running', False, False
        self.offset = self.plan.grasp_offset
        if self.arm_mode == ARM_EXECUTOR and not self.plan.ik_complete:
            self._fallback('no IK solution for every keyframe')
        # Clear-branch wrist guard: if the guard is on but a moving keyframe still commands ur_wrist_1 past
        # the carry cap (no clear-branch solution exists for the pick), the executor MUST NOT stream it --
        # that sweeps the arm through the ur_wrist_1-vs-ur_forearm stop (the execretest jam). Fail fast with
        # a named reason on the first step (arm held), never a silent still degrade.
        self._guard_fail = None
        if (self.arm_mode == ARM_EXECUTOR and self.plan.ik_complete
                and not self.plan.wrist1_within_cap()):
            self._guard_fail = ('the carry wrist guard found no clear-branch solution: ur_wrist_1 exceeds the '
                                '%.2f rad cap on a moving keyframe (would sweep the ur_wrist_1-vs-ur_forearm '
                                'stop)' % self.plan.wrist_guard[0])
        self._segments = self._timeline()
        self._exec = dict(index=0, step='send', clock=0.0, budget=0.0)
        print('SCRIPTED_TRANSFER begin arm=%s ik_complete=%s duration=%.1fs choice=%s carton_world=%s '
              'goal_world=%s ik=%s' % (self.arm_mode, self.plan.ik_complete, self._segments[-1][1],
                                       self.plan.choice, tuple(round(v, 3) for v in carton_world[:3, 3]),
                                       tuple(round(v, 3) for v in self.goal_world[:3, 3]),
                                       {k.name: tuple(round(e, 4) for e in k.ik_error)
                                        for k in self.plan.keyframes}), flush=True)
        return self.plan

    def _fallback(self, reason):
        if self.fallback_reason is None:
            self.fallback_reason = reason
        if self.arm_mode == ARM_EXECUTOR:
            print('SCRIPTED_TRANSFER arm=executor -> still (carton-only path from here): %s' % reason,
                  flush=True)
        self.arm_mode = ARM_STILL

    def _fail_executor(self, name, kind, detail):
        """A MID-TRANSFER executor refusal (pre-send infeasibility, host rejection, abort or settle timeout).
        Never degrade silently to the still carton-only path: once the host executor has e-stopped, still
        mode sends NO arm command and the Isaac arm-command heartbeat starves into the misleading 'arm
        command heartbeat lost' crash (warm5, 2026-10-22). Fail the stage FAST with a NAMED reason -- so the
        mission reports the true cause and tears down before the wall-backstop watchdog can trip -- while
        keeping the arm command stream alive by holding the current joints through the executor (best effort;
        a genuinely faulted executor rejects the hold too, but the fast failure still beats the backstop)."""
        if self.arm is not None:
            self.arm.hold(np.asarray(self.driver.arm_joint_positions(), float))
        self.fallback_reason = f'executor {kind}: {detail}'
        self.state = 'failed'
        self.fail_reason = (f'SCRIPTED TRANSFER executor {kind} the {name} trajectory: {detail} '
                            f'-- failed fast (arm held, no silent still fallback)')
        print('SCRIPTED_TRANSFER executor %s %s: %s -- failing fast, arm held (no silent still fallback)'
              % (name, kind, detail), flush=True)

    def _timeline(self):
        """[(t0, t1, kind, a, b)] with kind 'move' (keyframe a -> b) or 'grasp'/'release' dwell; the still
        clock (time_scale stretches it)."""
        segs, t, prev = [], 0.0, None
        for k in self.plan.keyframes:
            a = prev if prev is not None else Keyframe('start', self.ik.fk(self.plan.q_start), 0., False,
                                                       q=self.plan.q_start.copy())
            dur = k.duration * self.time_scale
            segs.append((t, t + dur, 'move', a, k))
            t += dur
            if k.name == 'contact':
                segs.append((t, t + DWELL_S, 'grasp', k, k))
                t += DWELL_S
            elif k.name == 'place':
                segs.append((t, t + DWELL_S, 'release', k, k))
                t += DWELL_S
            prev = k
        return segs

    def _segment_start(self, name):
        first = dict(SEGMENTS)[name][0]
        return next(t0 for t0, _t1, kind, _a, b in self._segments if kind == 'move' and b.name == first)

    def _pose_at(self, a, b, elapsed, span):
        """Planned tool pose (base frame) ``elapsed`` s into the move a -> b: FK of b's timed joint samples,
        or the smoothed task-space tool path when there is no joint path."""
        if b.samples:
            return self.ik.fk(_sample_q(b.samples, elapsed / max(self.time_scale, 1e-9)))
        return _interp_T(a.tcp, b.tcp, _smooth(elapsed / max(span, 1e-9)))

    # -- tick
    def step(self, dt, packet=None):
        if self.state != 'running':
            return self._terminal(packet)
        if self.arm_mode == ARM_EXECUTOR:
            r = self._step_executor(float(dt))
            if r is not None:
                return r
            if self.state != 'running':
                return self._terminal(packet)
            if self.arm_mode == ARM_EXECUTOR:
                return Result('running', reason=f'SCRIPTED TRANSFER (workaround): {self.phase}')
        return self._step_still(float(dt), packet)

    def _grasp(self):
        if not self.carried:
            self.driver.attach_carton()     # kinematic + colliders off; pose driven from here on
            self.carried = True
            print('SCRIPTED_TRANSFER grasp (kinematic carton attach, arm=%s)' % self.arm_mode, flush=True)

    def _release(self):
        if not self.released:
            self.driver.place_carton(self.goal_world)   # EXACT ground-truth target; stays kinematic
            self.carried, self.released = False, True
            print('SCRIPTED_TRANSFER release carton placed at %s'
                  % (tuple(round(v, 4) for v in self.goal_world[:3, 3]),), flush=True)

    def _step_still(self, dt, packet):
        # The arm does not move in still mode, so only the CARRIED stretch plays (grasp dwell -> release
        # dwell); the pre-grasp reach and the post-release retreat would be dead air.
        carried = [(t0, t1) for t0, t1, kind, _a, _b in self._segments if kind in ('grasp', 'release')]
        self.t = max(self.t, carried[0][0]) + dt
        if self.t > carried[-1][1]:
            self._release()
            self._finish()
            return self._terminal(packet)
        for t0, t1, kind, a, b in self._segments:
            if self.t <= t1:
                break
        if kind == 'move':
            if self.carried:
                tcp = self._pose_at(a, b, self.t - t0, t1 - t0)
                self.driver.set_carton_world_T(self.base_world @ tcp @ self.plan.grasp_offset)
            self.phase = b.name
        elif kind == 'grasp':
            self._grasp()
            self.phase = 'grasp'
        else:
            self._release()
            self.phase = 'release'
        return Result('running', reason=f'SCRIPTED TRANSFER (workaround): {self.phase}')

    def _step_executor(self, dt):
        """One tick of the event-driven executor timeline; None when the caller should continue in still
        mode (fallback) or report the generic running status."""
        d, ex = self.driver, self._exec
        if self._guard_fail is not None:
            reason, self._guard_fail = self._guard_fail, None
            self._fail_executor('pick', 'infeasible', reason)
            return None
        ex['clock'] += dt
        if ex['index'] >= len(SEGMENTS):
            self._finish()
            return None
        name, names = SEGMENTS[ex['index']]
        if ex['step'] == 'send':
            points = segment_points(self.plan.keyframes, names)
            measured = np.asarray(d.arm_joint_positions(), float)
            if points and float(np.max(np.abs(measured - points[0][1]))) > EXECUTOR_START_TOL:
                self._fail_executor(name, 'start-mismatch',
                                    f'arm not at the start (max joint error '
                                    f'{float(np.max(np.abs(measured - points[0][1]))):.3f} rad)')
                return None
            if points:
                points[0] = (0.0, measured, np.zeros(6))     # re-seed to the live state (fm357)
                why = hermite_violation(points, (self.ik.lower, self.ik.upper))
                if why is not None:
                    self._fail_executor(name, 'infeasible',
                                        f'trajectory not executor-feasible after re-seed: {why}')
                    return None
                self.arm.send(name, points)
                ex.update(step='wait', clock=0.0, budget=points[-1][0] + EXECUTOR_SETTLE_S)
                print('SCRIPTED_TRANSFER executor %s sent: %d points %.1fs' % (name, len(points), points[-1][0]),
                      flush=True)
            else:
                ex.update(step='dwell', clock=0.0)
            self.phase = name
            return None
        if ex['step'] == 'wait':
            if self.carried:
                d.set_carton_world_T(np.asarray(d.tool_world_T(), float) @ self.offset)
            result = self.arm.poll()
            if result is None:
                if ex['clock'] > ex['budget']:
                    self._fail_executor(name, 'settle-timeout',
                                        f'did not finish within {ex["budget"]:.0f} sim s')
                return None
            status = result.get('status')
            if status == 'succeeded':
                ex.update(step='dwell', clock=0.0)
                return None
            if status == 'rejected':
                self._fail_executor(name, 'rejected', result.get('detail'))
                return None
            self._fail_executor(name, 'aborted', result.get('detail'))
            return None
        # dwell after a segment: grasp after pick, release after carry, then the next segment
        if name == 'pick':
            if not self.carried:
                tool = np.asarray(d.tool_world_T(), float)
                self.offset = np.linalg.inv(tool) @ np.asarray(d.carton_world_T(), float)
            self._grasp()
            self.phase = 'grasp'
        elif name == 'carry':
            self._release()
            self.phase = 'release'
        if name == 'retreat' or ex['clock'] >= DWELL_S:
            ex.update(index=ex['index'] + 1, step='send', clock=0.0)
        return None

    def _finish(self):
        d = self.driver
        carton = np.asarray(d.carton_world_T(), float)
        c, g = carton[:3, 3], self.goal_world[:3, 3]
        top = self.target_xyz_world[2]
        xy = float(math.hypot(c[0] - g[0], c[1] - g[1]))
        bottom_gap = float((c[2] - self.dims[2] / 2.0) - top)
        self.check = dict(xy_offset=xy, bottom_gap=bottom_gap, carton_world=tuple(float(v) for v in c),
                          target_world=self.target_xyz_world)
        ok = self.released and xy <= self.xy_tol and abs(bottom_gap) <= self.z_tol
        self.state = 'succeeded' if ok else 'failed'
        if not ok:
            self.fail_reason = ('SCRIPTED TRANSFER ground-truth placement check failed: xy_offset=%.3f '
                                'bottom_gap=%.3f released=%s' % (xy, bottom_gap, self.released))
        print('SCRIPTED_TRANSFER %s ground-truth placement xy_offset=%.4f bottom_gap=%.4f carton=%s target=%s '
              'arm=%s fallback=%r' % (self.state.upper(), xy, bottom_gap, tuple(round(v, 4) for v in c),
                                     self.target_xyz_world, self.arm_mode, self.fallback_reason), flush=True)

    def _terminal(self, packet):
        if self.state == 'succeeded':
            return Result('succeeded', self.evidence_for(packet))
        if self.state == 'failed':
            return Result('failed', reason=getattr(self, 'fail_reason', 'SCRIPTED TRANSFER failed'))
        return Result('running', reason='SCRIPTED TRANSFER (workaround)')

    @property
    def arm_follow(self):
        """True when the arm physically executed every segment (executor mode, no fallback)."""
        return self.arm_mode == ARM_EXECUTOR and self.fallback_reason is None

    def evidence_for(self, packet):
        """The transfer success evidence: stamped by the CURRENT physics packet (its id + clock, so the
        mission freshness gate measures this acceptance) and honestly labelled -- the placement is checked
        against the SIM GROUND TRUTH, so source='ground_truth' + ground_truth=True (vr1's convention; the
        Mission accepts it only when transfer_ground_truth is enabled), never 'sensor'."""
        return dict(observation_id=f'ground_truth:transfer:{packet.observation_id}', source=GT_SOURCE,
                    ground_truth=True, simulation_time=packet.simulation_time, released=True,
                    sku=self.sku, quantity=self.quantity, location=self.location,
                    transfer_mode=SCRIPTED, verify_mode='ground_truth', scripted=True,
                    arm_mode=self.arm_mode, arm_follow=self.arm_follow,
                    arm_fallback_reason=self.fallback_reason, ik_complete=bool(self.plan.ik_complete),
                    ik_plan_latency_s=(round(self.ik_plan_latency_s, 3)
                                       if self.ik_plan_latency_s is not None else None),
                    near_face=dict(self.plan.choice), **self.check)


# ---------------------------------------------------------------- arm executor client (host ROS)

class ExecutorArm:
    """Sends one scripted joint trajectory per segment to the mission's own FollowJointTrajectory executor
    (trajectory_host, /restock_sim/arm_controller/follow_joint_trajectory) through a short-lived client
    subprocess under the HOST overlay ROS env -- the same spawn recipe as control_ipc.HostControlPublisher
    (Isaac's bundled rclpy cannot reach the host DDS). ``poll()`` -> None while running, else
    dict(status='succeeded'|'rejected'|'aborted', detail=str)."""

    ACTION = '/restock_sim/arm_controller/follow_joint_trajectory'

    def __init__(self, *, runtime_env, workdir, python='python3', domain=88, _popen=None):
        import subprocess
        self.runtime_env, self.python, self.domain = runtime_env, python, int(domain)
        self.workdir = Path(workdir)
        self._popen = _popen or subprocess.Popen
        self._proc = None
        self._log = None

    def send(self, name, points):
        import json
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError('scripted arm segment already running')
        self.workdir.mkdir(parents=True, exist_ok=True)
        goal = self.workdir / f'scripted-arm-{name}.json'
        goal.write_text(json.dumps({'action': self.ACTION, 'names': list(ARM_JOINTS),
                                    'points': [{'t': float(t), 'q': [float(x) for x in q],
                                                'v': [float(x) for x in v]} for t, q, v in points]}))
        self._log = self.workdir / f'scripted-arm-{name}.log'
        client = Path(__file__).with_name('scripted_arm_client.py')
        command = (f'set +u; source {self.runtime_env}; '
                   f'export ROS_DOMAIN_ID={self.domain} ROS_LOCALHOST_ONLY=1 ROS_STATIC_PEERS= '
                   f'ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp '
                   f'FASTDDS_BUILTIN_TRANSPORTS=UDPv4 SKIP_DEFAULT_XML=1; '
                   f'unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE '
                   f'ROS_DISCOVERY_SERVER CYCLONEDDS_URI RMW_FASTRTPS_USE_QOS_FROM_XML; '
                   f'exec {self.python} {client} {goal}')
        clean_env = {k: os.environ[k] for k in ('HOME', 'USER', 'LOGNAME', 'TERM', 'DISPLAY')
                     if k in os.environ}
        clean_env['PATH'] = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
        with open(self._log, 'w') as log:
            self._proc = self._popen(['bash', '-c', command], env=clean_env, stdout=log, stderr=log)

    def hold(self, measured):
        """Keep-alive: command the arm to HOLD its current measured joints as ONE immediate rest-to-rest
        goal, so the host executor keeps streaming and the Isaac arm-command heartbeat does not starve while
        a failing transfer winds down. Fired on a mid-transfer refusal instead of going silent; a genuinely
        e-stopped executor rejects the hold too, which is fine (the stage still fails fast, below the wall
        backstop). Never raises -- a keep-alive that cannot be sent must not mask the real failure."""
        q = [float(x) for x in measured]
        try:
            self.send('hold', [(0.0, q, [0.0] * 6), (SAMPLE_DT, q, [0.0] * 6)])
        except Exception as exc:                        # a stuck prior proc / spawn failure: best effort only
            print('SCRIPTED_TRANSFER executor hold keep-alive could not be sent: %s' % exc, flush=True)

    def poll(self):
        if self._proc is None or self._proc.poll() is None:
            return None
        import json
        text = self._log.read_text() if self._log is not None and self._log.exists() else ''
        for line in reversed(text.splitlines()):
            if line.startswith('SCRIPTED_ARM_RESULT '):
                try:
                    return json.loads(line[len('SCRIPTED_ARM_RESULT '):])
                except ValueError:
                    break
        # no result line: the client never got a goal accepted (env / server / crash) => nothing latched
        return dict(status='rejected', detail=f'client exited {self._proc.returncode} without a result '
                                              f'(log {self._log})')


# ---------------------------------------------------------------- Isaac side (lazy imports)

class IsaacScriptedDriver:
    """Measured state + carton actuation for ScriptedTransfer inside the live Isaac process. Never
    commands the arm (the host trajectory executor owns it). The carton is driven as a KINEMATIC body via
    its USD transform (the runtime USD-authoring pattern the sim-attach weld and RESTOCK_RELEASE_KINEMATIC
    already use); poses are read from USD like the mission's own /tf relay."""

    def __init__(self, *, stage, gate, base_path, tool_path, carton_path=HANDLING_UNIT_PATH):
        self.stage, self.gate = stage, gate
        self.base_path, self.tool_path, self.carton_path = base_path, tool_path, carton_path

    def _world(self, path):
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError(f'prim not found: {path}')
        return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), float).T

    def base_world_T(self):
        return self._world(self.base_path)

    def tool_world_T(self):
        return self._world(self.tool_path)

    def carton_world_T(self):
        return self._world(self.carton_path)

    def arm_joint_positions(self):
        return self.gate.measured()

    def _colliders(self, enabled):
        from pxr import UsdPhysics, Usd
        for p in Usd.PrimRange(self.stage.GetPrimAtPath(self.carton_path)):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(bool(enabled))

    def attach_carton(self):
        from pxr import UsdPhysics, UsdGeom
        prim = self.stage.GetPrimAtPath(self.carton_path)
        rb = (UsdPhysics.RigidBodyAPI(prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)
              else UsdPhysics.RigidBodyAPI.Apply(prim))
        rb.CreateKinematicEnabledAttr(True)
        self._colliders(False)
        # One translate + orient op stack (the authored stack is translate + rotateZ, or whatever PhysX
        # rewrote it to), keeping the current pose, so every later write is a plain pose set.
        xf = UsdGeom.Xformable(prim)
        names = [op.GetOpName() for op in xf.GetOrderedXformOps()]
        if names != ['xformOp:translate', 'xformOp:orient']:
            T = self.carton_world_T()
            xf.ClearXformOpOrder()
            xf.AddTranslateOp()
            xf.AddOrientOp()
            self.set_carton_world_T(T)

    def set_carton_world_T(self, T_world):
        from pxr import Gf, UsdGeom
        prim = self.stage.GetPrimAtPath(self.carton_path)
        parent = prim.GetParent()
        P = (np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(parent), float).T
             if parent and parent.GetPath().pathString != '/' else np.eye(4))
        L = np.linalg.inv(P) @ np.asarray(T_world, float)
        ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        t, o = ops['xformOp:translate'], ops['xformOp:orient']
        t.Set(Gf.Vec3d(float(L[0, 3]), float(L[1, 3]), float(L[2, 3])))
        q = _R_to_quat(L[:3, :3])                    # (x, y, z, w)
        cls = Gf.Quatf if o.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd
        o.Set(cls(float(q[3]), float(q[0]), float(q[1]), float(q[2])))

    def place_carton(self, T_world):
        self.set_carton_world_T(T_world)
        self._colliders(True)


def _R_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, .25 * s)
    if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] >= R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 1] + R[1, 0]) / s, .25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, .25 * s, (R[1, 0] - R[0, 1]) / s)


def scripted_arm_mode(config=None, environ=None):
    """``still`` (default) or ``executor``: env RESTOCK_SCRIPTED_TRANSFER_ARM, else
    config['scripted_transfer']['arm']. Unknown => ValueError (fail loud at boot)."""
    env = os.environ if environ is None else environ
    opts = (config.get('scripted_transfer') or {}) if isinstance(config, dict) else {}
    raw = env.get(ARM_ENV) or opts.get('arm') or ARM_STILL
    value = str(raw).strip().lower()
    if value not in ARM_MODES:
        raise ValueError(f'{ARM_ENV}={raw!r}: unknown scripted-transfer arm mode (known: {", ".join(ARM_MODES)})')
    return value


def build_scripted_transfer(*, assembly, config, scenario, output=None, environ=None):
    """The scripted transfer runner when ``RESTOCK_TRANSFER_MODE=scripted``, else None (byte-identical).
    Target = config['transfer_ground_truth']['target'] (the authored bay top); dims from the scenario."""
    if transfer_mode(environ) != SCRIPTED:
        return None
    gt = config.get('transfer_ground_truth') if isinstance(config, dict) else None
    target = (gt or {}).get('target') if isinstance(gt, dict) else None
    if not target or len(target) < 3 or not gt.get('enabled'):
        raise ValueError("RESTOCK_TRANSFER_MODE=scripted needs config transfer_ground_truth "
                         "{'enabled': true, 'target': [x, y, z]} -- the ground-truth bay top; 'enabled' is what "
                         "makes the Mission accept the labelled ground-truth transfer evidence")
    unit = scenario.get('handling_unit') or {}
    dims = unit.get('dimensions_m') or (0.6, 0.4, 0.16)
    opts = dict(config.get('scripted_transfer') or {})
    env = os.environ if environ is None else environ
    imported = Path(output) / 'mobile-import.urdf' if output is not None else None
    urdf = opts.get('urdf') or (imported if imported is not None and imported.is_file()
                                else _mission_urdf(config))
    ik = ArmIK(urdf)
    arm_mode = scripted_arm_mode(config, env)
    place_approach = scripted_place_approach(env)
    arm = None
    if arm_mode == ARM_EXECUTOR:
        md = config.get('masked_depth') or {}
        runtime_env = opts.get('runtime_env') or md.get('runtime_env')
        if not runtime_env:
            raise ValueError('scripted transfer arm=executor needs a host overlay env '
                             "(config['scripted_transfer']['runtime_env'] or masked_depth.runtime_env)")
        arm = ExecutorArm(runtime_env=runtime_env, python=opts.get('python') or md.get('python', 'python3'),
                          workdir=Path(output or '.') / 'scripted-transfer', domain=int(md.get('domain') or 88))
    stock = (getattr(assembly, 'evidence', None) or {}).get('composition', {}).get('stock')
    driver = IsaacScriptedDriver(stage=assembly.stage, gate=assembly.command_gate,
                                 base_path=assembly.paths['ur_base_link'],
                                 tool_path=assembly.paths['gripper_pad_tcp'],
                                 carton_path=opts.get('carton_path') or stock or HANDLING_UNIT_PATH)
    job = scenario.get('job') or {}
    runner = ScriptedTransfer(
        driver, ik=ik, target_xyz_world=target[:3], target_yaw_world=float(opts.get('target_yaw', 0.0)),
        dims=dims, location=(gt.get('location') or job.get('shelf_id')), sku=job.get('sku'),
        quantity=int(job.get('quantity', 1)), wall=int(opts.get('wall', 3)),
        contact_margin=float(unit.get('tcp_surface_offset_m', 0.00275)),
        standoff=float(opts.get('standoff', 0.15)),
        time_scale=float(opts.get('time_scale', env.get('RESTOCK_SCRIPTED_TRANSFER_TIME_SCALE', 1.0))),
        arm=arm, place_approach=place_approach,
        ik_plan_budget_s=float(opts.get('ik_plan_budget_s', ik_plan_budget_seconds(env))))
    print('TRANSFER_MODE scripted: transfer stage = scripted %s + kinematic carton path to the ground-truth '
          'target %s%s (urdf=%s)' % ('arm (host executor)' if arm is not None else 'carton path, arm still',
                                     tuple(target[:3]),
                                     '' if place_approach == PLACE_ABOVE else ' place=front (slide-in)', urdf),
          flush=True)
    return runner


def _mission_urdf(config):
    import json
    manifest = json.loads(Path(config['mobile_manifest']).read_text())
    path = manifest.get('mobile_urdf')
    if isinstance(path, dict):
        path = path.get('path') or path.get('urdf')
    if not path:
        raise ValueError('mobile manifest has no mobile_urdf for the scripted-transfer IK')
    p = Path(path)
    return p if p.is_absolute() else Path(config['mobile_manifest']).parent / p
