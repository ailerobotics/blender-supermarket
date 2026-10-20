"""Asynchronous simulated MoveIt/cuMotion port and ROS restock bridge.

Imports ROS lazily. Uses seeded collision-checking IK, plan-only MoveGroup, then
ExecuteTrajectory. Executor spinning belongs to the caller; no nested spin or GPU
boot. Existing dynamic_pick's measured-start/joint-window checks are retained;
its analytic surrogate and cell-specific calibration are deliberately not loaded.
"""
from dataclasses import asdict
import json
import math
import os
import time
import xml.etree.ElementTree as ET

from .types import Observation, Sensors, Target

from .cad_tool import ARM_JOINTS

JOINTS = ARM_JOINTS

# Sim-time settle margin added to a trajectory's own sim duration for the execution deadline (the
# node-clock bound on ExecuteTrajectory). Covers start handshake + terminal settle in SIM seconds.
EXECUTE_SETTLE_MARGIN_S = 25.0
# fm367: SIM-ONLY multiplier on EVERY sim-clock move/settle deadline's added budget (the trajectory's
# own sim duration + the settle margin). Default 1.0 = byte-identical real-rig; the sim profile sets
# 1.5 so a render-paced settle can never miss its sim deadline by a hair and end the mission (run-294:
# the straight insert/retreat settled 10 ms past its await_settle sim deadline at RELEASE). Clamped
# sane; WALL deadlines (time.monotonic) are the wall sweep's job and are NOT scaled here.
MOVE_SIM_DEADLINE_SCALE = max(1.0, min(5.0, float(os.environ.get('RESTOCK_MOVE_SIM_DEADLINE_SCALE', 1.0))))


def sim_execute_budget(duration, scale=None):
    """Sim-clock execution budget for one trajectory: (its own sim ``duration`` + EXECUTE_SETTLE_MARGIN_S),
    multiplied by MOVE_SIM_DEADLINE_SCALE (sim-only; 1.0 => byte-identical). Added to self._sim_now() at
    dispatch to form the node-clock execution deadline. Pure + importable so the scale is CPU-testable
    without a ROS node (the inline _poll() dispatch cannot be reached on CPU)."""
    return (duration + EXECUTE_SETTLE_MARGIN_S) * (MOVE_SIM_DEADLINE_SCALE if scale is None else scale)
# Bounded re-plan of the SAME goal on a MoveIt plan rejection. MoveIt's FCL ValidateSolution is
# ground truth and cuMotion is stochastic, so re-planning draws a fresh seed; a few retries turn the
# seed-variant base/plate collision (INVALID_MOTION_PLAN, run-129/134 ~half of seeds) into a
# near-certain pass at ~1 s of planning per retry (firstmate standing-authority lever).
REPLAN_MAX = 12
# Bounded recovery from a move_group PROCESS crash (SIGABRT double-free in the cuMotion result
# handling, run-266/268): the action server VANISHES mid-move. The launch now respawns move_group
# instead of tearing the host stack down, so the planning client waits for the respawned server and
# re-sends the SAME plan goal (a fresh plan -> execute) up to this many times, turning an abort during
# a stage into a logged retry instead of a lost 20-30 min GPU run. Distinct from REPLAN_MAX, which
# reacts to a DELIVERED plan rejection; here no result is ever delivered because the server died.
MOVE_GROUP_CRASH_RETRY_MAX = 3
# Wall cap on how long a single crash-retry holds the move open waiting for move_group to respawn and
# re-advertise its action servers. The launch respawns with a short delay and the CumotionPlanner
# plugin re-init blocks on the (still-live) nvblox ESDF service, so a healthy respawn re-advertises in
# a few seconds; this generous bound makes an UNRECOVERABLE crash (respawn exhausted, or the launch
# itself down) surface as a clean motion failure instead of an unbounded hold.
RESPAWN_WAIT_CAP_S = max(5.0, min(300.0, float(os.environ.get('RESTOCK_RESPAWN_WAIT_CAP_S', 45.0))))  # fm362
# Fail FAST on a persistently-failing plan (firstmate run-216/279): an IK-infeasible goal churns the
# 12 replans + every seed, each resetting a fresh per-attempt deadline, so the move could retry well
# past the host's 30 s /sensors wall backstop -- which then fired as the mission reason instead of a
# clean "planning failed" (the preplace IK failure manifested as a 30 s stall). Cap the TOTAL planning
# wall time for a single move below the backstop so an unplannable pose surfaces as a motion failure
# fast. Legit stochastic retries succeed in ~1 s each and never approach this.
MOVE_WALL_TIMEOUT_S = max(5.0, min(300.0, float(os.environ.get('RESTOCK_MOVE_WALL_TIMEOUT_S', 25.0))))  # fm362: the ik/plan-accept WALL deadline default
PLAN_TOTAL_WALL_CAP_S = max(5.0, min(300.0, float(os.environ.get('RESTOCK_PLAN_WALL_CAP_S', 22.0))))  # fm362 sim-only env widen
# fm364/run-293: host /sensors stream wall backstop (RosTransferBridge default). Sim-only env widen;
# default 30 = byte-identical real-rig. The long carton-loaded collect's held sensing starves the
# stream past 30 s wall -> sim sets 180.
SENSOR_STREAM_WALL_BACKSTOP_S = max(5.0, min(600.0, float(os.environ.get('RESTOCK_SENSOR_STREAM_WALL_BACKSTOP_S', 30.0))))
# fm366: RosTransferBridge startup-grace wall bound (ros_host passes config, default 5 s). Sim-only env.
STARTUP_GRACE_S = max(1.0, min(120.0, float(os.environ.get('RESTOCK_STARTUP_GRACE_S', 5.0))))
# fm385/run-300c: the host /sensors PER-FRAME bound, in SIM/node seconds. During a long render-paced
# move (e.g. retreat) the collect capture holds (motion-completion feedback stale) and emits no fresh
# station snapshot; a 1 s SIM bound then trips mid-retreat even with the render alive and no OOM (run-300c
# reached sim 241 past the release peak, zero OOM, then this gate fired). SIM-only widen tolerates the
# held-sensing move; the WALL backstop (SENSOR_STREAM_WALL_BACKSTOP_S) stays the true dead-stream detector
# (a frozen sim clock never advances this bound). Default 1.0 = byte-identical real-rig (sim == wall).
SENSOR_WATCHDOG_S = max(0.5, min(120.0, float(os.environ.get('RESTOCK_SENSOR_WATCHDOG_S', 1.0))))


def env_wins(env_var, passed, module_default):
    """fm363: the SIM-ONLY env override WINS over any value a caller (ros_host config) passes to the
    ctor, so a widen can never again be silently moot (run-290: RosMotionPort(plan_wall_cap_s=config[60])
    shadowed RESTOCK_PLAN_WALL_CAP_S). Env set -> the env value; env unset -> the passed value (or the
    module default). The module_default already clamps the env, so a set env reaches here as that value."""
    if os.environ.get(env_var) is not None:
        return float(module_default)
    return float(passed if passed is not None else module_default)
# fm345 (c): the tight terminal goal-gate position tolerance (matches trajectory_host GOAL_POSITION_TOL);
# the per-move PLACE relaxation widens it only for the carton-loaded extend, reset to this otherwise.
GOAL_POSITION_TOL_DEFAULT = 0.01


def _should_replan(replan_attempts, has_plan_goal):
    """On a MoveIt plan rejection, re-plan the SAME goal (a fresh stochastic cuMotion seed) up to
    REPLAN_MAX times before falling back to the next IK seed / failing. Pure for CPU testing."""
    return replan_attempts < REPLAN_MAX and has_plan_goal


def _should_crash_retry(crash_retries, has_plan_goal):
    """After a move_group PROCESS crash (its action server vanished mid-move, run-266/268), re-send the
    SAME plan goal to the respawned server up to MOVE_GROUP_CRASH_RETRY_MAX times. Distinct from
    _should_replan: that reacts to a DELIVERED plan rejection; here the server itself died, so there is
    no result to react to -- the launch respawns move_group and the whole move is re-planned. A re-send
    needs a stored plan goal (the straight-line contact descent has none -> normal failure path). Pure
    for CPU testing."""
    return has_plan_goal and crash_retries < MOVE_GROUP_CRASH_RETRY_MAX


def planner_velocity_scale(urdf, *, command_limit=.15, scale_cap=.05):
    """Keep planner URDF-scaled speeds below the simulator's absolute gate.

    Reserve ten percent for interpolation; the executor still checks exact cubic
    extrema and rejects any overshoot. This is not trajectory certification.

    ``command_limit`` is the absolute joint-speed gate (rad/s); ``scale_cap`` is an
    extra ceiling on the URDF-velocity SCALE (fraction) regardless of command_limit.
    Both default to the conservative real-rig profile (0.15 rad/s, 5% scale); the SIM
    execution profile passes a single raised ``arm_max_speed`` for both so the grasp
    completes within the render-paced wall budget (see data/mobile-restock-v2 report).
    """
    robot=ET.parse(urdf).getroot()
    rates={joint.attrib['name']:float(joint.find('limit').attrib['velocity'])
           for joint in robot.findall('joint') if joint.attrib['name'] in JOINTS}
    if (set(rates)!=set(JOINTS) or not math.isfinite(command_limit) or command_limit<=0
            or not math.isfinite(scale_cap) or scale_cap<=0
            or any(not math.isfinite(v) or v<=0 for v in rates.values())):
        raise ValueError('actual six-joint URDF velocity limits required')
    return min(scale_cap,.9*command_limit/max(rates.values()))


def require_sim_domain():
    if os.environ.get('ROS_DOMAIN_ID') != '88':
        raise RuntimeError('simulation requires ROS_DOMAIN_ID=88')
    if os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE') != 'LOCALHOST':
        raise RuntimeError('simulation requires ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST')
    if os.environ.get('ROS_STATIC_PEERS', ''):
        raise RuntimeError('simulation forbids ROS_STATIC_PEERS')


def validate_trajectory(trajectory, start, limits):
    """Reject malformed, reordered, nonfinite, discontinuous or unbounded plans."""
    names = list(trajectory.joint_names)
    if len(names) != 6 or set(names) != set(JOINTS) or not trajectory.points:
        raise ValueError('trajectory missing joints/waypoints')
    order = [names.index(j) for j in JOINTS]
    previous = -1.0
    for index, point in enumerate(trajectory.points):
        if len(point.positions) != 6:
            raise ValueError('invalid waypoint width')
        q = [point.positions[i] for i in order]
        if any(not math.isfinite(v) or not lo <= v <= hi for v, (lo, hi) in zip(q, limits)):
            raise ValueError('waypoint outside joint limits')
        stamp = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
        if not math.isfinite(stamp) or stamp <= previous:
            raise ValueError('trajectory time not increasing')
        if index == 0 and max(abs(a-b) for a, b in zip(q, start)) > .05:
            raise ValueError('trajectory start differs from measured joints')
        previous = stamp
    last = trajectory.points[-1]
    if last.velocities and (len(last.velocities) != 6 or
                            any(not math.isfinite(v) or abs(v) > 1e-3 for v in last.velocities)):
        raise ValueError('trajectory does not end at rest')


# fm357: the host plans from the joints measured at begin-time, then hands the trajectory to MoveIt's
# ExecuteTrajectory, which validates the FIRST waypoint against the CURRENT /joint_states
# (allowed_start_tolerance). Under render pacing the plan->validate gap is ~hundreds of seconds, and a
# gravity-loaded joint DRIFTS in that window (run-280 shoulder_pan; run-282 ur_elbow), so the start no
# longer matches and the move ABORTS ('start point deviates ...'). The structural fix (below): settle
# the arm to ~rest, then re-seed the first waypoint to the LIVE measured state immediately before the
# execute dispatch. Both are pure/CPU-testable; gated on a live-state provider (absent => byte-identical).
SETTLE_VEL_DEFAULT = 0.05  # rad/s: max per-joint speed for the arm to count as settled before executing
# fm369: SIM-ONLY best-effort settle sub-budget (sim seconds). The await_settle hold waits for the arm to
# reach SETTLE_VEL_DEFAULT before dispatching execute; at a gravity/contact-jittery pose the arm can never
# settle and the hold runs the whole ~40 s execute deadline, then faults and ends the mission (run-296: the
# post-release RETREAT-start never dropped below 0.05 rad/s). When the arm has been held this many sim
# seconds without settling, dispatch ANYWAY: the reseed re-anchors the first waypoint to the live pose, so
# the start stays valid and MoveIt's ExecuteTrajectory start-tolerance is the real gate. +inf (env unset)
# => byte-identical (hold to the execute deadline exactly as before).
SETTLE_WAIT_S = max(0.5, float(os.environ.get('RESTOCK_SETTLE_WAIT_S', 'inf')))
# fm370: SIM-ONLY settle-velocity THRESHOLD override (rad/s). Default SETTLE_VEL_DEFAULT (0.05) =
# byte-identical real-rig; the sim raises it (0.15) so a soft-drive arm jittering a little at a loaded
# pose counts as settled on the NORMAL path instead of always timing out into fm369 best-effort.
# Independent of SETTLE_WAIT_S.
SETTLE_VEL = max(0.001, min(2.0, float(os.environ.get('RESTOCK_SETTLE_VEL', SETTLE_VEL_DEFAULT))))


def arm_settled(velocities, tol=SETTLE_VEL_DEFAULT):
    """True when every measured joint velocity magnitude is within ``tol`` (arm at rest, not drifting)."""
    vs = list(velocities)
    return bool(vs) and all(math.isfinite(v) and abs(v) <= tol for v in vs)


def reseed_first_waypoint(joint_names, planned_first_positions, live_by_name):
    """Return the first waypoint's positions with each joint overwritten by its LIVE measured value
    (looked up by name), in the trajectory's own joint order. A joint absent from ``live_by_name``
    keeps its planned value. So MoveIt's ExecuteTrajectory start-state check matches the current arm
    state after the render-paced plan->validate drift. Pure/CPU-testable."""
    return [float(live_by_name.get(n, p)) for n, p in zip(joint_names, planned_first_positions)]


class RosMotionPort:
    def __init__(self, node, *, joint_limits, payload_sync, emergency_stop,
                 group='ur_manipulator', tcp='gripper_pad_tcp', namespace='/restock_sim',
                 pipeline='isaac_ros_cumotion', timeout=MOVE_WALL_TIMEOUT_S, velocity_scale=.04, pose_goal=False,
                 stow_config=None, max_velocity_scale=.05, plan_wall_cap_s=PLAN_TOTAL_WALL_CAP_S,
                 live_arm_state=None, settle_vel=SETTLE_VEL_DEFAULT, settle_wait_s=SETTLE_WAIT_S):
        require_sim_domain()
        # fm357 structural fix: live_arm_state() -> ({joint: pos}, {joint: vel}, fresh:bool) | None gives
        # the latest measured arm state; when present, each arm execute SETTLES the arm (<= settle_vel)
        # then RE-SEEDS the trajectory's first waypoint to the live state before ExecuteTrajectory, so
        # the render-paced plan->validate drift no longer aborts the move. None => byte-identical (no
        # settle/re-seed; the original direct execute dispatch).
        self._live_arm_state = live_arm_state if callable(live_arm_state) else None
        # fm370 / fm369: both SIM-ONLY (env wins over any config-passed value; default = real-rig).
        self._settle_vel = env_wins('RESTOCK_SETTLE_VEL', settle_vel, SETTLE_VEL)
        self._settle_wait_s = env_wins('RESTOCK_SETTLE_WAIT_S', settle_wait_s, SETTLE_WAIT_S)
        self._settle_wait_start = None   # fm369: sim-time the current await_settle hold began
        self._pending_exec = None
        # Total planning-wall cap for a single move (ik/plan request/result). Real-rig default 22 s
        # (PLAN_TOTAL_WALL_CAP_S); the sim profile raises it (firstmate 293) because a plan that
        # move_group completes in ~1.4 s can take >20 s to DELIVER its ActionResult to this port under
        # render pacing (run-234/237/239), so a 22 s cap cut off a succeeded plan. The 30 s /sensors
        # wall backstop stays the genuine hang catch; real-rig path untouched. Recorded with the other
        # render-pacing wall-window relaxations.
        # fm363: env WINS over the ros_host config value (run-290 shadow fix).
        self.plan_wall_cap_s = env_wins('RESTOCK_PLAN_WALL_CAP_S', plan_wall_cap_s, PLAN_TOTAL_WALL_CAP_S)
        if node.context.get_domain_id() != 88 or not node.get_parameter('use_sim_time').value:
            raise RuntimeError('node must use domain 88 and simulation clock')
        if namespace != '/restock_sim':
            raise ValueError('planner endpoints must be under /restock_sim')
        if (len(joint_limits) != 6 or any(len(x) != 2 or not all(map(math.isfinite, x))
                                       or x[0] >= x[1] for x in joint_limits)):
            raise ValueError('six finite URDF joint-limit pairs required')
        if not callable(payload_sync) or not callable(emergency_stop):
            raise ValueError('payload collision acknowledgement and stop callbacks required')
        # Upper bound on cuMotion's max_velocity_scaling_factor. Default 0.05 is the conservative
        # real-rig ceiling; the SIM execution profile raises it (to arm_max_speed) so the planner may
        # plan the faster grasp that fits the render-paced wall budget. Still a hard sanity gate.
        if (not math.isfinite(max_velocity_scale) or not 0<max_velocity_scale<=1
                or not math.isfinite(velocity_scale) or not 0<velocity_scale<=max_velocity_scale):
            raise ValueError('bounded planner velocity scaling required')
        from rclpy.action import ActionClient
        from moveit_msgs.action import MoveGroup, ExecuteTrajectory
        from moveit_msgs.srv import GetPositionIK
        self.node, self.group, self.tcp = node, group, tcp
        # fm363: env WINS over any caller-passed timeout too (ros_host does not pass it today, but keep
        # the ik/plan-accept deadline env-authoritative so it can never be silently shadowed either).
        self.pipeline, self.limits = pipeline, joint_limits
        self.timeout = env_wins('RESTOCK_MOVE_WALL_TIMEOUT_S', timeout, MOVE_WALL_TIMEOUT_S)
        self.velocity_scale=velocity_scale
        # pose_goal=True plans to the cartesian TCP pose with a MoveGroup pose-goal (cuMotion does
        # collision-aware IK against its own XRDF model) instead of KDL GetPositionIK -> c-space
        # target. The KDL/SRDF path returned configs cuMotion's XRDF flagged as self-collision
        # (run-80: planToCSpaceTarget 'self collision detected'); one collision model removes that
        # disagreement. Kept behind this switch so a regression is a one-flag revert.
        self.pose_goal=bool(pose_goal)
        # Transport stow config (6 arm joints): the first/central KDL IK seed + the nearest-branch
        # nominal, so solutions land in-branch and in-limits. None -> seed only with the measured
        # start and normalise to 0.
        if stow_config is not None and (len(stow_config) != 6
                or any(not math.isfinite(q) or not lo <= q <= hi for q, (lo, hi) in zip(stow_config, joint_limits))):
            raise ValueError('stow_config must be six in-limit joint values')
        self.stow = tuple(float(q) for q in stow_config) if stow_config is not None else None
        self.payload_sync, self.emergency_stop = payload_sync, emergency_stop
        self.ik = node.create_client(GetPositionIK, namespace + '/compute_ik')
        self.plan = ActionClient(node, MoveGroup, namespace + '/move_action')
        self.execute = ActionClient(node, ExecuteTrajectory, namespace + '/execute_trajectory')
        self.state, self.handle, self.future = 'idle', None, None
        self.generation = 0
        # fm367: True when the move that just returned 'failed' did so on a SIM-clock move/settle
        # deadline miss (a render-pacing transient, safe to re-attempt) rather than a collision/plan
        # rejection (a port raise). The controller reads it after a 'failed' poll to retry ONLY the
        # former with a fresh deadline. A wall-clock deadline miss (the wall sweep's job) stays False.
        self._last_fault_sim_deadline = False
        # fm345 smaller-problems-later: per-move PLACE relaxations for the carton-loaded extend.
        # (b) _move_speed_scale slows the cuMotion joint-space plan (preplace/place) to ~half so the
        # soft arm drive (3e5/8000/400) can track the min-jerk timing it lagged in run-270 (sagged
        # 0.567 rad short). (c) a latched side-channel topic carries a WIDER terminal goal-gate
        # tolerance to the trajectory_host for those moves (the FJT goal_tolerance is unreachable
        # through MoveIt's ExecuteTrajectory), reset to the tight default on every other move so the
        # contact-descent precision is untouched. TRANSIENT_LOCAL so the executor sees the latest
        # value even though it subscribes after this publisher is created.
        from rclpy.qos import QoSProfile, DurabilityPolicy
        from std_msgs.msg import Float64
        self._move_speed_scale = 1.0
        self._goal_tol_pub = node.create_publisher(
            Float64, namespace + '/arm_controller/goal_position_tol',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # Deadlines: IK/planning are compute-bound -> WALL clock. Trajectory EXECUTION is paced by the
        # simulator: under render pacing executing D sim-seconds takes far more WALL time, so a wall
        # execution deadline false-fails a healthy grasp (run-93). The execute deadline is therefore
        # measured on the NODE/SIM clock (use_sim_time). deadline_clock selects which per action.
        self._sim_now = lambda: self.node.get_clock().now().nanoseconds * 1e-9
        self.deadline_clock = time.monotonic

    def _publish_goal_tol(self, goal_tol):
        """fm345 (c): latch the trajectory_host terminal goal-gate position tolerance for the NEXT
        move. goal_tol=None restores the tight default (GOAL_POSITION_TOL); a wider value is set only
        for the carton-loaded PLACE extend. Published BEFORE planning so it latches well before the
        executor receives the FJT goal (planning takes ~1 s). The executor clamps it to a sane band."""
        from std_msgs.msg import Float64
        tol = GOAL_POSITION_TOL_DEFAULT if goal_tol is None else float(goal_tol)
        self._goal_tol_pub.publish(Float64(data=tol))

    def begin(self, pose, joints, straight_line=False, joint_space=False, wrist1_cap=None,
              wrist2_min=None, slow=False, goal_tol=None, near_start=False):
        from moveit_msgs.srv import GetPositionIK
        if self.state not in ('idle', 'succeeded', 'failed'):
            raise RuntimeError('motion already active')
        if not self.ik.service_is_ready() or not self.plan.server_is_ready() or not self.execute.server_is_ready():
            raise RuntimeError('simulated IK/planner/executor unavailable')
        if (len(pose) != 7 or not all(map(math.isfinite, pose)) or len(joints) != 6
                or any(not math.isfinite(q) or not lo <= q <= hi for q, (lo, hi) in zip(joints, self.limits))):
            raise ValueError('invalid pose or measured start')
        self.start = tuple(joints)
        # fm345 (b)+(c): slow the carton-loaded PLACE extend to ~half and widen its terminal goal gate;
        # both reset to the full-speed / tight-tolerance default on every non-place move.
        self._move_speed_scale = 0.5 if slow else 1.0
        self._publish_goal_tol(goal_tol)
        # fm347 (3): pan<->preplace IK-branch consistency. The joint-space IK below normally unwraps to
        # the branch nearest the STOW nominal (_nearest_branch), which let the cart PLACE pose flip
        # shoulder_pan to a far branch (run-273: pan left j0 at -1.822 but preplace solved ~2.09 -> a
        # ~3.9 rad base rotation the soft drive stalled on). With near_start the PLACE moves seed the
        # measured (pan) config FIRST and branch to the config NEAREST the measured start, so the
        # preplace stays on the pan branch (small joint move, cart-cage clearance preserved).
        self._near_start = bool(near_start)
        self._last_fault_sim_deadline = False   # fm367: fresh deadline-fault signal per move
        self._replan_attempts = 0          # fresh-seed re-plan budget for this move (see REPLAN_MAX)
        self._server_crash_retries = 0     # move_group-respawn re-send budget (see MOVE_GROUP_CRASH_RETRY_MAX)
        self._awaiting_server_respawn = False  # True while holding the move open for a respawned move_group
        self._plan_goal = None             # the MoveGroup goal to re-send on a plan rejection
        self._plan_started_wall = time.monotonic()   # total planning-wall cap anchor (see PLAN_TOTAL_WALL_CAP_S)
        # Straight-line contact descent (firstmate run-191). The contact move is BY DEFINITION a short
        # straight descent INTO the target carton, so a collision planner that still holds the carton
        # (and its support face) as obstacles rejects it near contact (INVALID_MOTION_PLAN) whenever the
        # seed lands close -- the exact failure that stalled runs 190-and-earlier. For THIS move only,
        # bypass the planner: KDL-IK the contact pose to a c-space target (collision-avoidance OFF, since
        # touching the carton is the intent), branch it to the config NEAREST the measured approach start
        # (so the move is a few-mm straight descent, not a joint swing), and execute a seeded rest-to-rest
        # trajectory to it straight through the trajectory host -- the SAME executor path as the approach,
        # not a cuMotion plan. Every free-space move keeps cuMotion. Forces the KDL IK path even under
        # pose_goal, since the host trajectory interpolates to an explicit joint target.
        self._straight_line = bool(straight_line)
        # Per-move goal type (firstmate msg 284/285): cuMotion TASK-SPACE IK is systemically broken in
        # this XRDF, but the pick's task-space poses happen to solve; the cart PLACE poses do not, so
        # the caller forces those through the seeded-KDL -> JOINT-SPACE path (joint_space=True) -- the
        # reliable path -- while the proven pick keeps its task-space goal. A global pose_goal=False
        # instead made the PICK approach's joint-space trajectory oscillate (run-226), so route PER MOVE.
        if self.pose_goal and not self._straight_line and not joint_space:
            self.generation += 1
            self.handle = None
            self._plan_goal = self._pose_plan_goal(pose, joints, wrist1_cap=wrist1_cap, wrist2_min=wrist2_min)
            self._send(self.plan, self._plan_goal, 'plan_accept')
            self.deadline_clock = time.monotonic
            self.deadline = time.monotonic() + self.timeout
            return
        if self._straight_line:
            # Seed IK from the measured approach config only -- the arm is already at the pose just above
            # the pad, so this is the closest possible seed and keeps q on the start branch.
            self._goal_pose = tuple(float(v) for v in pose)
            self._seeds = [tuple(joints)]
            self._seed_idx = 0
            self._send_ik(self._seeds[0])
            return
        # Joint-space path (host verdict 2026-10-14: cuMotion task-space IK is systemically broken in
        # this XRDF; KDL compute_ik + cuMotion joint-space planning work). Harden branch selection:
        # try several KDL IK seeds (transport stow first -- a central in-limits config -- so KDL does
        # not return a wrapped branch like run-80's joint2=-4.45), normalise each solution to the
        # nearest in-limit branch, and if cuMotion rejects a c-space target as self-collision, try
        # the next seed's solution before failing.
        self._goal_pose = tuple(float(v) for v in pose)
        # fm347 (3): near_start -> try the MEASURED (pan) config as the FIRST IK seed so KDL returns the
        # branch nearest the pan (not a stow-nominal branch); the normal seeds remain as fallbacks.
        self._seeds = ([tuple(joints)] + self._seed_configs(joints)
                       if self._near_start else self._seed_configs(joints))
        self._seed_idx = 0
        self._send_ik(self._seeds[0])

    def _seed_configs(self, measured):
        """IK seeds, best-branch-first: transport stow (central, in-limits) then the measured start
        then mild stow perturbations, so KDL returns a near in-branch solution rather than a wrapped
        one. De-duplicated; always at least the measured start."""
        seeds = []
        if self.stow is not None:
            seeds.append(tuple(float(v) for v in self.stow))
        seeds.append(tuple(float(v) for v in measured))
        if self.stow is not None:
            for d in (0.4, -0.4):
                seeds.append(tuple(min(hi, max(lo, s + d)) for s, (lo, hi) in zip(self.stow, self.limits)))
        out = []
        for s in seeds:
            if s not in out:
                out.append(s)
        return out

    def _nearest_branch(self, q):
        """Shift each joint by multiples of 2*pi to the branch nearest the stow nominal (or 0), then
        keep it inside the XRDF joint limits -- avoids KDL's far wrapped branches that cuMotion
        flags as self-collision. Returns None if a joint has no in-limit branch."""
        nominal = self.stow if self.stow is not None else (0.,) * 6
        out = []
        for v, (lo, hi), nom in zip(q, self.limits, nominal):
            v2 = v + 2 * math.pi * round((nom - v) / (2 * math.pi))
            while v2 < lo:
                v2 += 2 * math.pi
            while v2 > hi:
                v2 -= 2 * math.pi
            if not lo <= v2 <= hi:
                return None
            out.append(v2)
        return out

    def _nearest_to(self, q, ref):
        """Shift each joint by multiples of 2*pi to the branch nearest ``ref`` (the measured start),
        clamped inside the XRDF limits. Used for the straight-line contact descent so the executed move
        stays on the approach branch -- a few-mm descent, never a wrapped-branch swing. None if a joint
        has no in-limit branch."""
        out = []
        for v, (lo, hi), r in zip(q, self.limits, ref):
            v2 = v + 2 * math.pi * round((r - v) / (2 * math.pi))
            while v2 < lo:
                v2 += 2 * math.pi
            while v2 > hi:
                v2 -= 2 * math.pi
            if not lo <= v2 <= hi:
                return None
            out.append(v2)
        return out

    def _straight_line_goal(self, q):
        """A seeded rest-to-rest 2-point joint trajectory from the measured start to ``q`` for the host
        executor. A single cubic-Hermite segment with zero endpoint velocities is a smoothstep (no
        overshoot, so positions stay within [start, q] and thus within limits); the duration is sized so
        the segment's peak speed/accel stay well under the host's conservative interpolation ceilings,
        making it pass both validate_trajectory and the trajectory_host Plan checks regardless of the
        mission's --max-speed profile."""
        from moveit_msgs.msg import RobotTrajectory
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        dmax = max((abs(b - a) for a, b in zip(self.start, q)), default=0.0)
        # Rest-to-rest cubic peaks: speed = 1.5*dmax/T, accel = 6*dmax/T^2. Target <=0.08 rad/s and
        # <=0.15 rad/s^2 (under both the sim 0.3/1.0 and real-rig 0.15/0.5 host ceilings); floor 2 s.
        T = max(2.0, 1.5 * dmax / 0.08, math.sqrt(6 * dmax / 0.15))
        sec = int(T)
        nanosec = int(round((T - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        start = JointTrajectoryPoint()
        start.positions = [float(v) for v in self.start]
        start.velocities = [0.0] * 6
        start.time_from_start = Duration(sec=0, nanosec=0)
        finish = JointTrajectoryPoint()
        finish.positions = [float(v) for v in q]
        finish.velocities = [0.0] * 6
        finish.time_from_start = Duration(sec=sec, nanosec=nanosec)
        jt = JointTrajectory()
        jt.joint_names = list(JOINTS)
        jt.points = [start, finish]
        rt = RobotTrajectory()
        rt.joint_trajectory = jt
        return rt

    def _send_ik(self, seed):
        from moveit_msgs.srv import GetPositionIK
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group
        req.ik_request.ik_link_name = self.tcp
        # Collision-avoidance OFF only for the straight-line contact descent: the contact pose places the
        # pad against the carton, which an avoid_collisions IK would reject as the very collision we are
        # deliberately driving into. Every other move keeps it on.
        req.ik_request.avoid_collisions = not getattr(self, '_straight_line', False)
        req.ik_request.timeout.sec = 2
        req.ik_request.robot_state.joint_state.name = list(JOINTS)
        req.ik_request.robot_state.joint_state.position = list(seed)
        req.ik_request.pose_stamped.header.frame_id = 'ur_base_link'
        req.ik_request.pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        p = req.ik_request.pose_stamped.pose
        p.position.x, p.position.y, p.position.z = map(float, self._goal_pose[:3])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, self._goal_pose[3:])
        self.generation += 1
        self.handle = None
        self.future, self.state = self.ik.call_async(req), 'ik'
        self.deadline_clock = time.monotonic
        self.deadline = time.monotonic() + self.timeout

    def _next_seed(self):
        """Advance to the next IK seed; returns True if one was dispatched, False if exhausted."""
        if self._seed_idx + 1 < len(self._seeds):
            self._seed_idx += 1
            self._send_ik(self._seeds[self._seed_idx])
            return True
        return False

    def _pose_plan_goal(self, pose, joints, wrist1_cap=None, wrist2_min=None):
        """MoveGroup plan request to a cartesian TCP pose (position+orientation goal constraints in
        ur_base_link); cuMotion does its own collision-aware IK, so a single collision model (XRDF)
        decides feasibility. Mirrors begin_joint_goal's request fields; plan_only (the port executes
        a validated trajectory separately). wrist1_cap (fm327, sim-only) adds a GOAL joint constraint
        ur_wrist_1_joint <= wrist1_cap so cuMotion IKs a carry pose (the lift) BELOW the ur_wrist_2-vs-
        ur_forearm self-collision stop (~0.70) even before the planner catches that pair. wrist2_min
        (fm403, sim-only) adds the companion LOWER bound ur_wrist_2_joint >= wrist2_min so the IK cannot
        pick the w2-negative self-colliding branch (the bench showed w1-cap alone leaves w2 jamming)."""
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, JointConstraint
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose as GPose
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name, req.pipeline_id = self.group, self.pipeline
        req.allowed_planning_time = 5.0
        req.num_planning_attempts = 1
        req.max_velocity_scaling_factor = self.velocity_scale
        req.max_acceleration_scaling_factor = .05
        req.start_state.joint_state.name = list(JOINTS)
        req.start_state.joint_state.position = list(joints)
        pc = PositionConstraint()
        pc.header.frame_id = 'ur_base_link'
        pc.link_name = self.tcp
        pc.weight = 1.0
        bv = BoundingVolume()
        bv.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.004, .004, .004]))
        tp = GPose()
        tp.position.x, tp.position.y, tp.position.z = map(float, pose[:3])
        tp.orientation.w = 1.0
        bv.primitive_poses.append(tp)
        pc.constraint_region = bv
        oc = OrientationConstraint()
        oc.header.frame_id = 'ur_base_link'
        oc.link_name = self.tcp
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = map(float, pose[3:])
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = .02
        oc.weight = 1.0
        c = Constraints()
        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        if wrist1_cap is not None:
            _low = -2.0   # cap only the UPPER bound at wrist1_cap; generous lower bound
            c.joint_constraints.append(JointConstraint(
                joint_name='ur_wrist_1_joint', position=(_low + float(wrist1_cap)) / 2.,
                tolerance_above=(float(wrist1_cap) - _low) / 2.,
                tolerance_below=(float(wrist1_cap) - _low) / 2., weight=1.))
        if wrist2_min is not None:
            _high = 2.0   # fm403: bound only the LOWER side at wrist2_min; generous upper bound
            c.joint_constraints.append(JointConstraint(
                joint_name='ur_wrist_2_joint', position=(float(wrist2_min) + _high) / 2.,
                tolerance_above=(_high - float(wrist2_min)) / 2.,
                tolerance_below=(_high - float(wrist2_min)) / 2., weight=1.))
        req.goal_constraints = [c]
        goal.planning_options.plan_only = True
        return goal

    def _send(self, client, goal, state):
        generation = self.generation
        future = client.send_goal_async(goal)
        # If cancellation races goal acceptance, cancel the late handle too.
        def accepted(f):
            if generation != self.generation:
                handle = f.result()
                if handle and handle.accepted:
                    handle.cancel_goal_async()
        future.add_done_callback(accepted)
        self.future, self.state = future, state

    def begin_joint_goal(self, goal_joints, measured_joints, slow=False, goal_tol=None):
        """Collision-planned arm stow; never an unchecked joint command."""
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import Constraints, JointConstraint
        if self.state not in ('idle','succeeded','failed'):
            raise RuntimeError('motion already active')
        if not self.plan.server_is_ready() or not self.execute.server_is_ready():
            raise RuntimeError('simulated planner/executor unavailable')
        for positions in (goal_joints,measured_joints):
            if len(positions) != 6 or any(not math.isfinite(q) or not lo <= q <= hi
                    for q,(lo,hi) in zip(positions,self.limits)):
                raise ValueError('invalid joint goal or measured start')
        self.start=tuple(measured_joints)
        # fm345 (c): tuck/pan keep the tight terminal goal gate -- reset the latched tolerance so a
        # wider PLACE value from a prior move cannot carry into these folds (begin_joint_goal moves
        # are not the carton-loaded place extend). goal_tol lets a specific joint-goal move relax it:
        # run-279 the carton-loaded fixed-shape PREPLACE (a ~1.88 rad shoulder_pan swing) settled 0.176
        # rad short on joint0 only (the executor held the plan end but the acceleration-mode drive did
        # not close the last step on the high-inertia extended-arm yaw swing; no contact -- armbench
        # reached the shape, no mid-track-miss), so the tight terminal gate rejected an otherwise-arrived
        # pose. The straight-line insert (place) re-solves to the SENSED place, correcting the offset.
        self._publish_goal_tol(goal_tol)
        self._near_start = False   # fm347 (3): begin_joint_goal moves use their own goal, never the near_start IK branch
        # Reset the planning anchors for THIS move, exactly as begin() does (firstmate 292 / run-238):
        # without a fresh _plan_started_wall the total-planning-wall cap carried the stale anchor from
        # the previous pose-goal move (lift), so the tuck/pan/stow joint goal tripped
        # 'planning failed (exhausted within wall cap)' the instant it entered planning
        # (PLAN_WALL_DIAG elapsed=49.6 s on a plan that had just started). A joint goal does not re-plan
        # a stored pose, so _plan_goal stays None (poll's REPLAN path is a no-op for it).
        self._last_fault_sim_deadline=False   # fm367: fresh deadline-fault signal per move
        self._replan_attempts=0
        self._server_crash_retries=0
        self._awaiting_server_respawn=False
        self._plan_goal=None
        self._straight_line=False
        self._plan_started_wall=time.monotonic()
        goal=MoveGroup.Goal()
        req=goal.request
        req.group_name=self.group
        req.pipeline_id=self.pipeline
        req.allowed_planning_time=5.
        req.num_planning_attempts=1
        # ``slow`` gentles a joint move that CARRIES the welded carton (the staged tuck/pan fold): the
        # 0.45 kg carton on the long side-grip lever displaced wrist1 ~1.0 rad during the aggressive fold
        # (run-241/246 TRAJ_HOST_TRACK_MISS), because the carton's inertial torque during a fast
        # shoulder/elbow swing outruns the wrist drive. A 0.35x velocity + 0.4x acceleration keeps the
        # carton's inertia low so the arm tracks the (body-clear) plan; free-space/empty moves keep full.
        req.max_velocity_scaling_factor=self.velocity_scale*(0.35 if slow else 1.0)
        req.max_acceleration_scaling_factor=(.02 if slow else .05)
        req.start_state.joint_state.name=list(JOINTS)
        req.start_state.joint_state.position=list(self.start)
        constraints=Constraints()
        for name,position in zip(JOINTS,goal_joints):
            constraints.joint_constraints.append(JointConstraint(joint_name=name,
                position=float(position),tolerance_above=.001,tolerance_below=.001,weight=1.))
        req.goal_constraints=[constraints]
        goal.planning_options.plan_only=True
        self.generation+=1
        self.handle=None
        self.deadline_clock=time.monotonic
        self.deadline=time.monotonic()+self.timeout
        self._send(self.plan,goal,'plan_accept')

    def _pending_plan_success(self):
        """True iff the current planning future has ALREADY DELIVERED a usable (successful) result
        that poll() has not consumed yet, so the plan cap must NOT declare the move 'exhausted'
        (firstmate 290 / run-234 false trip): move_group logged the plan succeeded in ~1 s, but the
        /sensors-paced poll() was behind, so the 22 s wall cap fired on a plan that had in fact
        succeeded. Read-only, per-state: a done IK solution, an accepted goal, or a SUCCEEDED plan
        result are all forward progress that the next poll() will consume. A hang (future not done)
        or a FAILED delivered result is NOT progress -- the cap still surfaces those (run-220)."""
        f = self.future
        if f is None or not f.done():
            return False
        try:
            r = f.result()
        except Exception:
            return False                                 # a result future that raised is not progress
        if self.state == 'plan_result':
            return r is not None and r.status == 4 and r.result.error_code.val == 1
        if self.state == 'ik':
            return r is not None and r.error_code.val == 1
        if self.state == 'plan_accept':
            return bool(r and r.accepted)
        return False

    #: PLANNING states where the planning-wall cap can fire. Draining poll() in these advances the
    #: plan request/accept/result -> execute independent of /sensors. EXECUTION states are sim-clock
    #: bounded (never the wall cap) and the 'contact' descent has its own suction choreography, so the
    #: watchdog does not drain those (see _watchdog).
    _DRAIN_STATES = ('ik', 'plan_accept', 'plan_result')

    def drain(self):
        """Advance the motion state machine through PLANNING if a plan is mid-flight, WITHOUT the
        sensor stream (firstmate 290 option b / run-237). The port only advances in poll(), which the
        controller drives from the /sensors callback; under render pacing /sensors is ~1.7-3.5 s apart
        (and the lift SUSPENDS sensing), so the plan_accept -> plan_result -> get_result -> execute
        chain can take many seconds per step and the move_group result (which arrived in ~1 s) is not
        consumed before the planning-wall cap fires on a plan that in fact succeeded. The bridge
        watchdog calls this on its own 0.1 s timer so the plan is requested + consumed within 0.1 s of
        arrival, independent of /sensors -- deterministic, not dice. No-op outside a planning state;
        poll() is idempotent so this and the controller's own poll() never conflict (single-threaded
        executor). Execution is left to the controller's poll() (sim-clock bounded, not the wall cap)."""
        if getattr(self, 'state', None) in self._DRAIN_STATES:
            self.poll()

    def plan_wall_diag(self):
        """One-line state snapshot for the plan-cap fire (firstmate 290/292, run-237): distinguishes a
        result DELIVERED-but-not-consumed (future done -> the fix should have caught it) from one
        NOT-delivered in the cap (future not done -> move_group succeeded but the action result never
        reached the port). Read-only."""
        f = getattr(self, 'future', None)
        done = bool(f is not None and f.done())
        status = None
        if done:
            try:
                r = f.result()
                status = getattr(r, 'status', None)
            except Exception as exc:
                status = 'raised:%r' % exc
        elapsed = time.monotonic() - getattr(self, '_plan_started_wall', time.monotonic())
        return ('PLAN_WALL_DIAG state=%s future_done=%s result_status=%s elapsed=%.2f replans=%s'
                % (getattr(self, 'state', None), done, status, elapsed,
                   getattr(self, '_replan_attempts', None)))

    def planning_wall_exceeded(self):
        """True once a single move has spent more than the total planning-wall cap in a PLANNING
        phase (ik / plan request / plan result) WITHOUT a usable result yet. Read-only (no state
        advance): the bridge watchdog checks it on its own 0.1 s timer so a planning hang surfaces as
        a motion failure within the cap EVEN WHEN /sensors has stalled and poll() is no longer being
        driven (run-220: the preplace PLANNING_FAILED was masked by the 30 s /sensors wall backstop
        because poll() stopped being called). A SUCCEEDED plan result already delivered but not yet
        consumed by the sensor-paced poll() does NOT count as exhausted (firstmate 290; run-234
        false-tripped on a plan move_group had already completed). Execution phases are excluded -- a
        render-paced execute legitimately runs long."""
        return (self.state in ('ik', 'plan_accept', 'plan_result')
                and not self._pending_plan_success()
                and time.monotonic() - getattr(self, '_plan_started_wall', time.monotonic())
                > getattr(self, 'plan_wall_cap_s', PLAN_TOTAL_WALL_CAP_S))

    def failed_result_pending(self):
        """True when an EXECUTE action RESULT has arrived FAILED (status != SUCCEEDED) but poll() has
        not consumed it. Read-only. The bridge watchdog checks it so a trajectory_host GOAL_MISS abort
        during a reach that suspends sensing -- /sensors stalls and poll() stops being driven --
        surfaces as a motion failure within ~0.1 s instead of hanging to the wall cap (run-224: the
        lift GOAL_MISS aborted the executor but the controller never saw it).

        ONLY 'execute_result' (a terminal execution abort), NOT 'plan_result': a PLANNING failure has
        the REPLAN/next-seed retry logic in poll() (cuMotion is stochastic, run-229 preplace planning
        failed once then would re-plan), so letting the watchdog fire on the first failed plan result
        preempted those retries and killed a recoverable plan. A genuine planning HANG is still bounded
        by planning_wall_exceeded; exhausted retries still raise a clean 'planning failed' in poll()."""
        f = self.future
        if f is None or self.state != 'execute_result' or not f.done():
            return False
        try:
            return f.result().status != 4            # GoalStatus.STATUS_SUCCEEDED
        except Exception:
            return True                              # a result future that raised is a failure too

    def ack_failed_result(self):
        """fm387: the bridge watchdog consumed a failed/broken motion result (failed_result_pending or a
        drain() raise) and routed it to the controller's bounded move retry. Reset the port to idle so
        the 0.1 s watchdog does NOT re-fire the same stale result (failed_result_pending needs
        'execute_result'; drain() is a no-op outside _DRAIN_STATES) before the controller re-issues the
        move next tick. The re-issue (begin/begin_joint_goal) bumps the generation and sets up a fresh
        plan+execute; generation++ here invalidates any late action callback from the abandoned move."""
        self.generation += 1
        self.state, self.handle, self.future = 'idle', None, None

    def _server_action_for_state(self):
        """The MoveGroup/ExecuteTrajectory action client whose server must be live for the current
        state, or None outside a planning/execution state. Used to notice a move_group PROCESS crash
        (the server vanishes) distinctly from a slow-but-alive server."""
        if self.state in ('plan_accept', 'plan_result'):
            return self.plan
        if self.state in ('execute_accept', 'execute_result'):
            return self.execute
        return None

    def _handle_server_crash(self):
        """Recover a move_group PROCESS crash (SIGABRT double-free, run-266/268) without failing the
        move. While the action server for the current state is down, hold the move open (the launch
        respawns move_group); once it is back, re-send the SAME plan goal as a bounded, logged retry
        (a fresh plan -> execute). Returns 'running' while recovering, or None to let poll() proceed
        normally. No-op when there is no stored plan goal or the crash-retry budget is spent -- the
        normal wall-cap/failure path then owns it, so an unrecoverable hang still surfaces."""
        server = self._server_action_for_state()
        if server is None:
            return None
        if not server.server_is_ready():
            # Server down: crashed, or in the respawn gap. Keep the move alive for the respawn only if
            # we have a goal to re-send and crash-retry budget remains.
            if not _should_crash_retry(getattr(self, '_server_crash_retries', 0),
                                       getattr(self, '_plan_goal', None) is not None):
                return None                     # no goal / budget spent -> normal failure path owns it
            now = time.monotonic()
            if not getattr(self, '_awaiting_server_respawn', False):
                self._awaiting_server_respawn = True
                self._respawn_wait_started = now
                print('MOVE_GROUP_CRASH_RETRY state=%s action server lost; awaiting respawn (retry %d/%d)'
                      % (self.state, getattr(self, '_server_crash_retries', 0) + 1,
                         MOVE_GROUP_CRASH_RETRY_MAX), flush=True)
            if now - getattr(self, '_respawn_wait_started', now) > RESPAWN_WAIT_CAP_S:
                # move_group did not come back within the cap -> unrecoverable; fail the move cleanly
                # (poll() wraps this into stop() -> 'failed') instead of holding the move open forever.
                raise RuntimeError('move_group did not respawn within %.0fs' % RESPAWN_WAIT_CAP_S)
            # Hold the deadline ahead of the respawn wait so the move does not fail while down.
            self.deadline_clock = time.monotonic
            self.deadline = now + self.timeout
            return 'running'
        if getattr(self, '_awaiting_server_respawn', False):
            # Respawned server is back -> re-plan the SAME goal from scratch (plan -> execute). Reset
            # the per-move planning anchors for a full fresh budget; count the crash retry.
            self._awaiting_server_respawn = False
            self._server_crash_retries = getattr(self, '_server_crash_retries', 0) + 1
            self._replan_attempts = 0
            self.handle = None
            self.generation += 1
            self.deadline_clock = time.monotonic
            self.deadline = time.monotonic() + self.timeout
            self._plan_started_wall = time.monotonic()
            print('MOVE_GROUP_CRASH_RETRY resending plan goal after respawn (retry %d/%d)'
                  % (self._server_crash_retries, MOVE_GROUP_CRASH_RETRY_MAX), flush=True)
            self._send(self.plan, self._plan_goal, 'plan_accept')
            return 'running'
        return None

    def _live_measured_start(self):
        """fm368: the live measured arm pose as an ordered six-tuple in JOINTS order, or None when no
        fresh live state is available (real rig / not yet fresh). Used to re-anchor a post-move_group-
        respawn plan whose start drifted from the stale self.start -- await_settle reseeds the first
        waypoint to this same live state before execute, so accepting against it is consistent."""
        if self._live_arm_state is None:
            return None
        live = self._live_arm_state()
        if live is None:
            return None
        pos_by_name, _vel_by_name, fresh = live
        if not fresh or not pos_by_name:
            return None
        try:
            return tuple(float(pos_by_name[j]) for j in JOINTS)
        except (KeyError, TypeError, ValueError):
            return None

    def last_fault_was_sim_deadline(self):
        """fm367: True iff the most recent 'failed' poll was a SIM-clock move/settle deadline miss (a
        render-pacing transient the controller may re-attempt), False for a collision/plan rejection or
        a wall-clock miss. Cleared at the start of every move (begin/begin_joint_goal)."""
        return bool(getattr(self, '_last_fault_sim_deadline', False))

    def poll(self):
        try:
            return self._poll()
        except Exception:
            # fm367: a port RAISE is a collision/plan-rejection/crash fault, never a sim-deadline miss ->
            # clear the flag so the controller's deadline retry does not fire (fail-fast preserved).
            self._last_fault_sim_deadline = False
            self.stop()
            return 'failed'

    def _poll(self):
        from moveit_msgs.action import MoveGroup, ExecuteTrajectory
        from moveit_msgs.msg import Constraints, JointConstraint
        if self.state in ('succeeded', 'failed'):
            return self.state
        # A move_group SIGABRT (double-free in cuMotion result handling, run-266/268) makes its action
        # server vanish mid-move. Checked BEFORE the deadline so the move is held open for the launch's
        # respawn and re-sent, rather than timing out to 'failed'. Inert when the server is live.
        crashed = self._handle_server_crash()
        if crashed is not None:
            return crashed
        if self.deadline_clock() > self.deadline:
            # fm362 evidence: this is the ONLY path that returns 'failed' (-> 'collision planning or
            # execution failed'), and it is silent. run-289's retreat died here with move_group all
            # SUCCEEDED, so name the fired deadline: which clock (wall MOVE_WALL_TIMEOUT / sim settle),
            # the overrun, and the state, so a wall-gate trip is unambiguous next run.
            _clock = 'wall' if self.deadline_clock is time.monotonic else 'sim'
            print('MOVE_DEADLINE_FAULT clock=%s over=%.2fs state=%s straight=%s'
                  % (_clock, self.deadline_clock() - self.deadline, self.state,
                     getattr(self, '_straight_line', False)), flush=True)
            # fm367: a SIM-clock miss is a render-pacing transient (not a collision), so flag it so the
            # controller can re-issue the SAME move with a fresh deadline (bounded). A WALL miss is the
            # wall sweep's job and the planner's fail-fast -- do NOT flag it for the deadline retry.
            self._last_fault_sim_deadline = (_clock == 'sim')
            self.stop()
            return 'failed'
        # fm357: before dispatching ExecuteTrajectory, hold until the arm is settled, then re-seed the
        # first waypoint to the live measured state. This state has NO pending future (the plan already
        # resolved), so it is handled before the future-done check below. Bounded by self.deadline so a
        # never-settling arm still faults.
        if self.state == 'await_settle':
            from moveit_msgs.action import ExecuteTrajectory
            live = self._live_arm_state() if self._live_arm_state is not None else None
            waited = (self._sim_now() - self._settle_wait_start
                      if self._settle_wait_start is not None else 0.0)
            path, maxvel = 'settled', 0.0
            if live is not None:
                pos_by_name, vel_by_name, fresh = live
                if fresh and vel_by_name:
                    maxvel = max((abs(v) for v in vel_by_name.values()), default=0.0)
                    if not arm_settled(vel_by_name.values(), self._settle_vel):
                        # fm369 best-effort: a gravity/contact-jittery loaded pose may never reach
                        # settle_vel (run-296 RETREAT). After settle_wait_s sim-seconds stop holding and
                        # dispatch ANYWAY -- the reseed below re-anchors the first waypoint to the live
                        # pose so the start stays valid; MoveIt's ExecuteTrajectory start-tolerance is the
                        # real gate. settle_wait_s = +inf (env unset) => hold to the execute deadline,
                        # byte-identical to the original fm357 behaviour.
                        if waited < self._settle_wait_s:
                            return 'running'   # still drifting; hold (arm drive keeps the pose)
                        path = 'besteffort'
                if fresh:
                    jt = self._pending_exec.joint_trajectory
                    if jt.points:
                        jt.points[0].positions = reseed_first_waypoint(
                            list(jt.joint_names), list(jt.points[0].positions), pos_by_name)
                        # start at rest at the live state: SIX explicit zero velocities -- trajectory_host
                        # requires every waypoint to carry exactly 6 velocities with the first ~0 (run-284
                        # regressed: clearing to [] -> 'six explicit waypoint velocities required' -> the
                        # FJT goal was rejected and the APPROACH aborted).
                        jt.points[0].velocities = [0.0] * 6
            # fm370: one line at EVERY await_settle exit -- which path (settled|besteffort), the exit joint
            # speed, the sim-seconds waited -- so run-297's log shows, per move, which path it took.
            print('AWAIT_SETTLE_EXIT path=%s maxvel=%.3f waited=%.2fs'
                  % (path, maxvel, waited), flush=True)
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = self._pending_exec
            self._pending_exec = None
            self._settle_wait_start = None
            self._send(self.execute, goal, 'execute_accept')
            return 'running'
        if not self.future.done():
            return 'running'
        result = self.future.result()
        if self.state == 'ik':
            if result is None or result.error_code.val != 1:
                if self._next_seed():          # this seed had no IK solution; try the next
                    return 'running'
                raise RuntimeError('IK failed')
            js = result.solution.joint_state
            q = [js.position[list(js.name).index(j)] for j in JOINTS]
            if not all(math.isfinite(v) for v in q):
                raise ValueError('IK returned a nonfinite joint')
            if getattr(self, '_straight_line', False):
                # Contact descent: branch q to the config NEAREST the measured start (not the stow
                # nominal), so the executed move is the few-mm straight descent, then hand a seeded
                # rest-to-rest trajectory straight to the host executor -- no cuMotion plan, no
                # collision-world rejection. See begin()'s straight-line note.
                q = self._nearest_to(q, self.start)
                if q is None:
                    raise ValueError('contact IK outside joint limits')
                from moveit_msgs.action import ExecuteTrajectory
                goal = ExecuteTrajectory.Goal()
                goal.trajectory = self._straight_line_goal(q)
                validate_trajectory(goal.trajectory.joint_trajectory, self.start, self.limits)
                end = goal.trajectory.joint_trajectory.points[-1].time_from_start
                duration = end.sec + end.nanosec * 1e-9
                # Execution is sim-paced, exactly like the cuMotion plan path: bound on the NODE/SIM clock.
                # fm367: the sim budget (duration + settle margin) is MOVE_SIM_DEADLINE_SCALE-scaled.
                self.deadline_clock = self._sim_now
                self.deadline = self._sim_now() + sim_execute_budget(duration)
                if self._live_arm_state is None:
                    self._send(self.execute, goal, 'execute_accept')
                else:                                   # fm357: settle + re-seed before dispatch
                    self._pending_exec = goal.trajectory
                    self._settle_wait_start = self._sim_now()   # fm369: start the settle sub-budget clock
                    self.state = 'await_settle'
                return 'running'
            # fm347 (3): the cart PLACE moves (near_start) branch to the config NEAREST the measured pan
            # start, so shoulder_pan stays on the pan branch (no ~3.9 rad flip to the stow-nominal
            # branch); every other move keeps the stow-nominal unwrap.
            q = (self._nearest_to(q, self.start) if getattr(self, '_near_start', False)
                 else self._nearest_branch(q))   # unwrap to the nearest in-limit branch
            if q is None:
                if self._next_seed():            # no in-limit branch for this solution; try next seed
                    return 'running'
                raise ValueError('IK outside joint limits')
            goal = MoveGroup.Goal()
            req = goal.request
            req.group_name, req.pipeline_id = self.group, self.pipeline
            req.allowed_planning_time = 5.0
            req.num_planning_attempts = 1
            # fm345 (b): this is the seeded-KDL -> cuMotion JOINT-space plan used by the cart PLACE
            # moves (preplace/place); _move_speed_scale (0.5 for those, 1.0 otherwise) halves the
            # planned speed so the soft drive tracks the min-jerk timing it lagged in run-270.
            req.max_velocity_scaling_factor = self.velocity_scale * self._move_speed_scale
            req.max_acceleration_scaling_factor = .05 * self._move_speed_scale
            req.start_state.joint_state.name = list(JOINTS)
            req.start_state.joint_state.position = list(self.start)
            constraints = Constraints()
            for name, position in zip(JOINTS, q):
                constraints.joint_constraints.append(JointConstraint(
                    joint_name=name, position=float(position), tolerance_above=.001,
                    tolerance_below=.001, weight=1.0))
            req.goal_constraints = [constraints]
            goal.planning_options.plan_only = True
            self._plan_goal = goal          # remember for a fresh-seed re-plan on rejection
            self._send(self.plan, goal, 'plan_accept')
        elif self.state in ('plan_accept', 'execute_accept'):
            if not result or not result.accepted:
                raise RuntimeError('action rejected')
            self.handle = result
            self.future = result.get_result_async()
            self.state = 'plan_result' if self.state == 'plan_accept' else 'execute_result'
        elif self.state == 'plan_result':
            if result.status != 4 or result.result.error_code.val != 1:
                # MoveIt's FCL ValidateSolution is ground truth and cuMotion is stochastic, so FIRST
                # re-plan the SAME goal (fresh seed) up to REPLAN_MAX times -- this turns the
                # seed-variant base/plate collision (INVALID_MOTION_PLAN, ~half of seeds) into a
                # near-certain pass. Then fall back to the next IK seed (a wrapped-branch self-collision
                # needs a different c-space target), then fail.
                # Fail fast: once the cumulative planning wall time for this move crosses the cap, stop
                # retrying (seed/replan) and surface a clean motion failure before the /sensors backstop.
                if time.monotonic() - getattr(self, '_plan_started_wall', time.monotonic()) > getattr(self, 'plan_wall_cap_s', PLAN_TOTAL_WALL_CAP_S):
                    raise RuntimeError('planning failed (exhausted within wall cap)')
                if _should_replan(self._replan_attempts, getattr(self, '_plan_goal', None) is not None):
                    self._replan_attempts += 1
                    self.deadline_clock = time.monotonic
                    self.deadline = time.monotonic() + self.timeout   # fresh planning budget per retry
                    self._send(self.plan, self._plan_goal, 'plan_accept')
                    return 'running'
                if getattr(self, '_seeds', None) and self._next_seed():
                    return 'running'
                raise RuntimeError('planning failed')
            # Plan round-trip latency (request -> result consumed here), WALL. firstmate 293: if this
            # ever exceeds ~5 s on a plan move_group completes in ~1.4 s, the ActionResult was starved in
            # the host's single-threaded executor -> land the MultiThreadedExecutor + callback-group fix.
            print('PLAN_DELIVERY elapsed=%.2f'
                  % (time.monotonic() - getattr(self, '_plan_started_wall', time.monotonic())), flush=True)
            trajectory = result.result.planned_trajectory
            try:
                validate_trajectory(trajectory.joint_trajectory, self.start, self.limits)
            except ValueError as exc:
                # fm368: a move_group SIGABRT respawn (run-266/268/295) starves the in-flight plan -- the
                # action server vanishes with no clean deregistration, so _handle_server_crash's
                # server_is_ready() misses it; the WALL deadline fires -> STAGED_MOVE_RETRY re-plans. The
                # respawned move_group plans from the arm's TRUE current pose via its re-subscribed state
                # monitor, which can differ from the pre-crash self.start by >0.05 rad, so validate rejects
                # an otherwise-valid plan on the START check alone and poll()'s except latches
                # emergency_stop (run-295 died here: PLAN_DELIVERY succeeded, then validate start-drift ->
                # 'host emergency stop'). await_settle already RE-SEEDS the first waypoint to the live
                # measured state before execute, so re-anchor self.start to the live measured joints and
                # re-validate: accept iff ONLY the start drifted and the plan starts at the real arm (every
                # malformed / out-of-limits / nonmonotonic / not-at-rest check still hard-fails). No-op on
                # the real rig (no live_arm_state) and whenever self.start already matched -> byte-identical.
                live_start = self._live_measured_start()
                if str(exc) != 'trajectory start differs from measured joints' or live_start is None:
                    raise
                validate_trajectory(trajectory.joint_trajectory, live_start, self.limits)
                self.start = live_start
                print('MOVE_GROUP_RESPAWN_RESEED start re-anchored to live measured joints '
                      'after post-respawn plan start-drift', flush=True)
            end=trajectory.joint_trajectory.points[-1].time_from_start
            duration=end.sec+end.nanosec*1e-9
            if duration>110.:
                raise ValueError('trajectory exceeds bounded execution budget')
            # Execution is simulator-paced, so bound it on the NODE/SIM clock: the trajectory's own
            # sim duration plus a settle margin. A wall bound here false-failed a healthy grasp whose
            # render-paced execution legitimately took many wall seconds per sim second (run-93).
            # fm367: the sim budget (duration + settle margin) is MOVE_SIM_DEADLINE_SCALE-scaled.
            self.deadline_clock=self._sim_now
            self.deadline=self._sim_now()+sim_execute_budget(duration)
            if self._live_arm_state is None:
                goal = ExecuteTrajectory.Goal()
                goal.trajectory = trajectory
                self._send(self.execute, goal, 'execute_accept')
            else:                                       # fm357: settle + re-seed before dispatch
                self._pending_exec = trajectory
                self._settle_wait_start = self._sim_now()   # fm369: start the settle sub-budget clock
                self.state = 'await_settle'
        elif self.state == 'execute_result':
            if result.status != 4 or result.result.error_code.val != 1:
                raise RuntimeError('execution failed')
            self.state = 'succeeded'
        return 'succeeded' if self.state == 'succeeded' else 'running'

    def stop(self):
        self.generation += 1
        if self.handle is not None:
            self.handle.cancel_goal_async()
        self.state = 'failed'
        self.emergency_stop()  # immediate task-owned simulator joint hold

    def update_payload(self, held, profile):
        # Must update BOTH MoveIt and cuMotion collision attachment, not PhysX
        # teleport/parenting. The simulator separately supplies physical contact.
        return self.payload_sync(held, profile) is True


def sensors_from_json(value):
    data = json.loads(value)
    if type(data.get('parked')) is not bool or type(data.get('contact')) is not bool:
        raise ValueError('parked/contact must be boolean')
    if data.get('target') is not None and type(data['target'].get('clear')) is not bool:
        raise ValueError('target clearance must be boolean')
    if (not isinstance(data.get('scanned_locations'), list)
            or not all(isinstance(x, str) for x in data['scanned_locations'])):
        raise ValueError('scanned_locations must be a list of observed ROI identifiers')
    data['observations'] = tuple(Observation(**o) for o in data['observations'])
    data['target'] = Target(**data['target']) if data['target'] else None
    return Sensors(**data)


class RosTransferBridge:
    """Attach to an externally spun sim-time node; publish controls and evidence.

    /restock_sim/sensors is a complete synchronized derived-sensor packet (Sensors
    JSON), never a list of scene prims. Raw RGB-D preprocessing uses DepthPerception.
    External integration supplies contact/odom/joints/map timestamps and full ROI
    scan coverage. Missing packets trigger a watchdog and immediate stop.

    Sensor-stream bounds, after the first /sensors frame arms the tight per-frame watchdog:

    - The per-frame bound is measured in NODE-CLOCK (ROS) time, not wall time. The node runs
      under use_sim_time, so a 1 s bound is 1 s of *simulation* time. On real hardware sim
      time == wall time, so this keeps exactly today's 1 s semantics; in a render-paced
      simulation it scales correctly -- each collect tick renders the full-resolution source
      camera, masks depth and captures, so consecutive /sensors are 1.7-3.5 s apart in WALL
      time (measured, run-74) but only a small step apart in SIMULATION time. A wall-time 1 s
      bound false-trips on that render latency; a node-clock 1 s bound does not, because the
      sim clock barely advances across one render tick.
    - A wall-time BACKSTOP (30 s since the last frame) still hard-stops if the sim clock
      freezes: a frozen clock stops the node-clock bound from ever advancing, so without the
      backstop a stalled simulator would hang the job forever. The watchdog timer itself stays
      on STEADY_TIME so it keeps firing when the sim clock is frozen.
    - Arm-on-first-frame and the 5 s startup grace (a job that never produces a first frame)
      stay in WALL time, unchanged: cold first-ESDF incorporation is ~1 s and unachievable
      from a cold job start, and before the first frame the arm has acted on nothing, so the
      moving-arm protection is not weakened.
    """
    def __init__(self, node, controller, *, watchdog_seconds=1.0, startup_grace_seconds=5.0,
                 wall_backstop_seconds=None):
        # fm364/run-293: the host /sensors stream wall backstop is the THIRD wall gate the fm362 audit
        # missed (effective value is this ctor default, not the module constants it widened). run-293 did
        # the FULL collect manip incl retreat_tuck, then the held-sensing /sensors stream starved past the
        # backstop during the long carton-loaded fold. Default now the sim-only env-overridable module
        # constant (RESTOCK_SENSOR_STREAM_WALL_BACKSTOP_S, default 30 = byte-identical real-rig; sim 180).
        if wall_backstop_seconds is None:
            wall_backstop_seconds = SENSOR_STREAM_WALL_BACKSTOP_S
        require_sim_domain()
        if node.context.get_domain_id() != 88 or not node.get_parameter('use_sim_time').value:
            raise RuntimeError('bridge requires isolated simulation node')
        from std_msgs.msg import String, Bool
        from rclpy.clock import Clock, ClockType
        self.node, self.controller = node, controller
        self.String, self.Bool = String, Bool
        self.last_packet_wall = time.monotonic()
        self.last_packet_sim = None   # node-clock (sim) time of the last frame; gates the per-frame bound
        self.last_sim = None
        self.sequence = 0
        self.status = node.create_publisher(String, '/restock_sim/manipulation/status', 10)
        self.gripper = node.create_publisher(Bool, '/restock_sim/gripper/enable', 10)
        # Contact-terminated move seam: the controller raises a 'contact' control from measured
        # s.contact during the grasp descent; publish it to trajectory_host, which then succeeds the
        # active goal holding the current pose (firstmate key contact-move-termination).
        self.arm_contact = node.create_publisher(Bool, '/restock_sim/arm_contact', 10)
        self.subscriptions = [
            node.create_subscription(String, '/restock_sim/manipulation/job', self._job, 10),
            node.create_subscription(String, '/restock_sim/sensors', self._sensors, 10)]
        # Split sensor-stream bound. The tight per-frame watchdog only ARMS after the FIRST
        # /sensors frame (a moving arm mid-grasp must never stall); before the first frame a
        # separate, looser STARTUP GRACE bounds a job that never produces sensors at all. Host-lane
        # measurement (data/sim-restock-host-v2/gate7-watchdog-timing-report.md): cold first ESDF
        # incorporation is ~1.0-1.05 s on this GPU, so a 1 s bound from a cold job start is
        # unachievable; the approach warm-up + this grace make the first frame feasible without
        # weakening the moving-arm protection (before the first frame the arm has acted on nothing).
        self.watchdog_seconds = env_wins('RESTOCK_SENSOR_WATCHDOG_S', watchdog_seconds, SENSOR_WATCHDOG_S)
        self.startup_grace_seconds = env_wins('RESTOCK_STARTUP_GRACE_S', startup_grace_seconds, STARTUP_GRACE_S)
        self.wall_backstop_seconds = wall_backstop_seconds
        self.first_frame = False
        self.job_start_wall = None
        self.timer = node.create_timer(.1, self._watchdog,
                                      clock=Clock(clock_type=ClockType.STEADY_TIME))

    def _emit(self, result):
        self.sequence += 1
        data = asdict(result)
        data['sequence'] = self.sequence
        data['job_id'] = getattr(self.controller, 'job', {}).get('job_id')
        self.status.publish(self.String(data=json.dumps(data, allow_nan=False)))
        for command in result.controls:
            if command['kind'] == 'gripper':
                self.gripper.publish(self.Bool(data=command['enabled']))
            elif command['kind'] == 'contact':
                self.arm_contact.publish(self.Bool(data=True))

    def _job(self, message):
        try:
            data = json.loads(message.data)
            self.controller.start(data['operation'], data['job'])
            self.last_sim = None
            self.last_packet_sim = None
            self.first_frame = False
            self.job_start_wall = time.monotonic()
            self.last_packet_wall = time.monotonic()
        except (ValueError, KeyError, TypeError) as exc:
            self.status.publish(self.String(data=json.dumps({'status': 'rejected', 'reason': str(exc)})))

    def _sensors(self, message):
        if self.controller.phase in ('idle', 'failed', 'succeeded'):
            return
        try:
            sensors = sensors_from_json(message.data)
            now = self.node.get_clock().now().nanoseconds * 1e-9
            if not 0 <= now - sensors.simulation_time <= self.controller.max_age:
                raise ValueError('packet simulation timestamp stale or future')
            dt = .01 if self.last_sim is None else sensors.simulation_time - self.last_sim
            self.last_sim = sensors.simulation_time
            self._emit(self.controller.step(dt, sensors))
            self.last_packet_wall = time.monotonic()
            self.last_packet_sim = now         # node-clock stamp the per-frame bound measures against
            self.first_frame = True            # arms the tight per-frame watchdog from here on
        except (ValueError, KeyError, TypeError) as exc:
            self._emit(self.controller._fail('invalid sensor packet: ' + str(exc)))

    def _watchdog(self):
        if self.controller.phase in ('idle', 'failed', 'succeeded'):
            return
        wall_now = time.monotonic()
        # A planning hang must surface as a planning failure within the total planning-wall cap even
        # when /sensors has stalled and poll() is no longer driven (run-220/281): this 0.1 s timer is
        # independent of the sensor stream, so check it here BEFORE the 30 s sensor backstop can mask it.
        motion = getattr(self.controller, 'motion', None)
        # Drain the planning state machine independent of /sensors FIRST (firstmate 290 option b): a
        # move_group result that has arrived is requested + consumed here within 0.1 s, so the cap
        # below sees the advanced (post-planning) state instead of firing on a /sensors-paced stall.
        # NOT during 'contact' -- its straight-line descent + suction choreography is driven only by
        # the controller's own step()/poll() (run-202/203 freeze-settle), never pre-advanced here.
        if (motion is not None and self.controller.phase != 'contact'
                and callable(getattr(motion, 'drain', None))):
            try:
                motion.drain()
            except Exception:
                # fm387: route through the bounded move retry (not a direct _fail) -- a drain-time plan
                # raise is run-to-run cuMotion variance; a persistent one exhausts the budget -> _fail.
                self._emit(self.controller.retry_or_fail_move('arm motion failed (watchdog drain)'))
                return
        if motion is not None and getattr(motion, 'planning_wall_exceeded', lambda: False)():
            diag = getattr(motion, 'plan_wall_diag', lambda: '')()
            if diag:
                print(diag, flush=True)
            self._emit(self.controller._fail('planning failed (exhausted within wall cap)'))
            return
        # A trajectory_host GOAL_MISS/abort during the LIFT suspends sensing, so /sensors stalls and
        # poll() stops draining the action result -- surface the failed result here independently
        # (run-224 hung ~20 min to the wall cap instead of failing when the lift missed its goal).
        if motion is not None and getattr(motion, 'failed_result_pending', lambda: False)():
            # fm387: a terminal EXECUTE goal-miss (run-300d LIFT, after a stochastic move_group crash)
            # is transient variance -- route through the bounded move retry (re-issue a fresh plan+exec)
            # instead of a direct _fail; a persistent miss exhausts the budget -> _fail. ack_failed_result
            # (inside retry_or_fail_move) clears the port so this 0.1 s check does not re-fire meanwhile.
            self._emit(self.controller.retry_or_fail_move('arm motion failed (goal tolerance not reached)'))
            return
        if not self.first_frame:
            # No /sensors yet: the looser startup grace (not the 1 s per-frame bound) applies, so a
            # warm-up + cold first ESDF has time to produce the first frame. Stays in WALL time; a
            # job that never produces a frame still hard-stops with a distinct reason.
            if self.job_start_wall is not None and wall_now - self.job_start_wall > self.startup_grace_seconds:
                self._emit(self.controller._fail('no initial sensor frame'))
            return
        # Per-frame bound in NODE-CLOCK (sim) time: on real hardware sim == wall so the 1 s
        # semantics are unchanged, and under a render-paced /clock it does not false-trip on the
        # 1.7-3.5 s wall gap between captures (the sim clock barely advances across a render tick).
        sim_now = self.node.get_clock().now().nanoseconds * 1e-9
        if self.last_packet_sim is not None and sim_now - self.last_packet_sim > self.watchdog_seconds:
            self._emit(self.controller._fail('sensor stream watchdog expired'))
            return
        # Wall-time backstop: a frozen sim clock never advances sim_now, so the node-clock bound
        # above can never trip; this still hard-stops a stalled simulator.
        if wall_now - self.last_packet_wall > self.wall_backstop_seconds:
            self._emit(self.controller._fail(
                'sensor stream wall backstop expired (gap=%.1fs > %.1fs)'
                % (wall_now - self.last_packet_wall, self.wall_backstop_seconds)))


class NvbloxWorldBridge:
    """Concrete task-only nvblox ESDF service client for IsaacCadMotion.

    The nvblox mapper/TF must use ur_base_link during parked manipulation and
    receive rendered depth. map_stamp is its last incorporated sensor timestamp,
    supplied by integration's mapper acknowledgement, never wall-clock 'now'.
    Missing nvblox_msgs is an explicit runtime dependency, not a fixture fallback.
    """
    def __init__(self, node, motion, origin, size):
        require_sim_domain()
        if node.context.get_domain_id() != 88 or not node.get_parameter('use_sim_time').value:
            raise RuntimeError('nvblox client must use isolated simulation clock/domain')
        from nvblox_msgs.srv import EsdfAndGradients
        self.service_type = EsdfAndGradients
        self.node, self.motion = node, motion
        self.origin, self.size = tuple(origin), tuple(size)
        self.client = node.create_client(EsdfAndGradients, '/restock_sim/nvblox_node/get_esdf_and_gradient')
        self.future = None

    def request(self, map_stamp):
        if self.future is not None and not self.future.done():
            return
        if not math.isfinite(map_stamp) or map_stamp < 0 or not self.client.service_is_ready():
            raise RuntimeError('sensor mapper or incorporated-frame acknowledgement unavailable')
        from geometry_msgs.msg import Point, Vector3
        request = self.service_type.Request()
        request.update_esdf, request.use_aabb = True, True
        request.frame_id = 'ur_base_link'
        request.aabb_min_m = Point(x=float(self.origin[0]), y=float(self.origin[1]), z=float(self.origin[2]))
        request.aabb_size_m = Vector3(x=float(self.size[0]), y=float(self.size[1]), z=float(self.size[2]))
        self.stamp = map_stamp
        self.future = self.client.call_async(request)

    def poll(self):
        if self.future is None or not self.future.done():
            return False
        import numpy as np
        response = self.future.result()
        self.future = None
        if response is None:
            raise RuntimeError('nvblox ESDF request failed')
        array = response.esdf_and_gradients
        dims = {dimension.label:int(dimension.size) for dimension in array.layout.dim}
        shape = tuple(dims[axis] for axis in ('x','y','z'))
        grid = np.asarray(array.data, np.float32).reshape(shape)
        origin = (response.origin_m.x, response.origin_m.y, response.origin_m.z)
        self.motion.update_esdf(grid, origin, self.stamp)
        return True
