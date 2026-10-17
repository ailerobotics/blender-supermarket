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
# Bounded re-plan of the SAME goal on a MoveIt plan rejection. MoveIt's FCL ValidateSolution is
# ground truth and cuMotion is stochastic, so re-planning draws a fresh seed; a few retries turn the
# seed-variant base/plate collision (INVALID_MOTION_PLAN, run-129/134 ~half of seeds) into a
# near-certain pass at ~1 s of planning per retry (firstmate standing-authority lever).
REPLAN_MAX = 12
# Fail FAST on a persistently-failing plan (firstmate run-216/279): an IK-infeasible goal churns the
# 12 replans + every seed, each resetting a fresh per-attempt deadline, so the move could retry well
# past the host's 30 s /sensors wall backstop -- which then fired as the mission reason instead of a
# clean "planning failed" (the preplace IK failure manifested as a 30 s stall). Cap the TOTAL planning
# wall time for a single move below the backstop so an unplannable pose surfaces as a motion failure
# fast. Legit stochastic retries succeed in ~1 s each and never approach this.
PLAN_TOTAL_WALL_CAP_S = 22.0


def _should_replan(replan_attempts, has_plan_goal):
    """On a MoveIt plan rejection, re-plan the SAME goal (a fresh stochastic cuMotion seed) up to
    REPLAN_MAX times before falling back to the next IK seed / failing. Pure for CPU testing."""
    return replan_attempts < REPLAN_MAX and has_plan_goal


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


class RosMotionPort:
    def __init__(self, node, *, joint_limits, payload_sync, emergency_stop,
                 group='ur_manipulator', tcp='gripper_pad_tcp', namespace='/restock_sim',
                 pipeline='isaac_ros_cumotion', timeout=25.0, velocity_scale=.04, pose_goal=False,
                 stow_config=None, max_velocity_scale=.05):
        require_sim_domain()
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
        self.pipeline, self.limits, self.timeout = pipeline, joint_limits, timeout
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
        # Deadlines: IK/planning are compute-bound -> WALL clock. Trajectory EXECUTION is paced by the
        # simulator: under render pacing executing D sim-seconds takes far more WALL time, so a wall
        # execution deadline false-fails a healthy grasp (run-93). The execute deadline is therefore
        # measured on the NODE/SIM clock (use_sim_time). deadline_clock selects which per action.
        self._sim_now = lambda: self.node.get_clock().now().nanoseconds * 1e-9
        self.deadline_clock = time.monotonic

    def begin(self, pose, joints, straight_line=False, joint_space=False):
        from moveit_msgs.srv import GetPositionIK
        if self.state not in ('idle', 'succeeded', 'failed'):
            raise RuntimeError('motion already active')
        if not self.ik.service_is_ready() or not self.plan.server_is_ready() or not self.execute.server_is_ready():
            raise RuntimeError('simulated IK/planner/executor unavailable')
        if (len(pose) != 7 or not all(map(math.isfinite, pose)) or len(joints) != 6
                or any(not math.isfinite(q) or not lo <= q <= hi for q, (lo, hi) in zip(joints, self.limits))):
            raise ValueError('invalid pose or measured start')
        self.start = tuple(joints)
        self._replan_attempts = 0          # fresh-seed re-plan budget for this move (see REPLAN_MAX)
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
            self._plan_goal = self._pose_plan_goal(pose, joints)
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
        self._seeds = self._seed_configs(joints)
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

    def _pose_plan_goal(self, pose, joints):
        """MoveGroup plan request to a cartesian TCP pose (position+orientation goal constraints in
        ur_base_link); cuMotion does its own collision-aware IK, so a single collision model (XRDF)
        decides feasibility. Mirrors begin_joint_goal's request fields; plan_only (the port executes
        a validated trajectory separately)."""
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
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

    def begin_joint_goal(self, goal_joints, measured_joints):
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
        # Reset the planning anchors for THIS move, exactly as begin() does (firstmate 292 / run-238):
        # without a fresh _plan_started_wall the total-planning-wall cap carried the stale anchor from
        # the previous pose-goal move (lift), so the tuck/pan/stow joint goal tripped
        # 'planning failed (exhausted within wall cap)' the instant it entered planning
        # (PLAN_WALL_DIAG elapsed=49.6 s on a plan that had just started). A joint goal does not re-plan
        # a stored pose, so _plan_goal stays None (poll's REPLAN path is a no-op for it).
        self._replan_attempts=0
        self._plan_goal=None
        self._straight_line=False
        self._plan_started_wall=time.monotonic()
        goal=MoveGroup.Goal()
        req=goal.request
        req.group_name=self.group
        req.pipeline_id=self.pipeline
        req.allowed_planning_time=5.
        req.num_planning_attempts=1
        req.max_velocity_scaling_factor=self.velocity_scale
        req.max_acceleration_scaling_factor=.05
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
                > PLAN_TOTAL_WALL_CAP_S)

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

    def poll(self):
        try:
            return self._poll()
        except Exception:
            self.stop()
            return 'failed'

    def _poll(self):
        from moveit_msgs.action import MoveGroup, ExecuteTrajectory
        from moveit_msgs.msg import Constraints, JointConstraint
        if self.state in ('succeeded', 'failed'):
            return self.state
        if self.deadline_clock() > self.deadline:
            self.stop()
            return 'failed'
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
                self.deadline_clock = self._sim_now
                self.deadline = self._sim_now() + duration + EXECUTE_SETTLE_MARGIN_S
                self._send(self.execute, goal, 'execute_accept')
                return 'running'
            q = self._nearest_branch(q)          # unwrap to the nearest in-limit branch
            if q is None:
                if self._next_seed():            # no in-limit branch for this solution; try next seed
                    return 'running'
                raise ValueError('IK outside joint limits')
            goal = MoveGroup.Goal()
            req = goal.request
            req.group_name, req.pipeline_id = self.group, self.pipeline
            req.allowed_planning_time = 5.0
            req.num_planning_attempts = 1
            req.max_velocity_scaling_factor = self.velocity_scale
            req.max_acceleration_scaling_factor = .05
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
                if time.monotonic() - getattr(self, '_plan_started_wall', time.monotonic()) > PLAN_TOTAL_WALL_CAP_S:
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
            trajectory = result.result.planned_trajectory
            validate_trajectory(trajectory.joint_trajectory, self.start, self.limits)
            end=trajectory.joint_trajectory.points[-1].time_from_start
            duration=end.sec+end.nanosec*1e-9
            if duration>110.:
                raise ValueError('trajectory exceeds bounded execution budget')
            # Execution is simulator-paced, so bound it on the NODE/SIM clock: the trajectory's own
            # sim duration plus a settle margin. A wall bound here false-failed a healthy grasp whose
            # render-paced execution legitimately took many wall seconds per sim second (run-93).
            self.deadline_clock=self._sim_now
            self.deadline=self._sim_now()+duration+EXECUTE_SETTLE_MARGIN_S
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = trajectory
            self._send(self.execute, goal, 'execute_accept')
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
                 wall_backstop_seconds=30.0):
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
        self.watchdog_seconds = watchdog_seconds
        self.startup_grace_seconds = startup_grace_seconds
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
                self._emit(self.controller._fail('arm motion failed (watchdog drain)'))
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
            self._emit(self.controller._fail('arm motion failed (goal tolerance not reached)'))
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
            self._emit(self.controller._fail('sensor stream wall backstop expired'))


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
