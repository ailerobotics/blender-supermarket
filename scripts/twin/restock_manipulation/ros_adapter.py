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

from .types import Observation, Sensors, Target

from .cad_tool import ARM_JOINTS

JOINTS = ARM_JOINTS


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
                 pipeline='isaac_ros_cumotion', timeout=25.0):
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
        from rclpy.action import ActionClient
        from moveit_msgs.action import MoveGroup, ExecuteTrajectory
        from moveit_msgs.srv import GetPositionIK
        self.node, self.group, self.tcp = node, group, tcp
        self.pipeline, self.limits, self.timeout = pipeline, joint_limits, timeout
        self.payload_sync, self.emergency_stop = payload_sync, emergency_stop
        self.ik = node.create_client(GetPositionIK, namespace + '/compute_ik')
        self.plan = ActionClient(node, MoveGroup, namespace + '/move_action')
        self.execute = ActionClient(node, ExecuteTrajectory, namespace + '/execute_trajectory')
        self.state, self.handle, self.future = 'idle', None, None
        self.generation = 0

    def begin(self, pose, joints):
        from moveit_msgs.srv import GetPositionIK
        if self.state not in ('idle', 'succeeded', 'failed'):
            raise RuntimeError('motion already active')
        if not self.ik.service_is_ready() or not self.plan.server_is_ready() or not self.execute.server_is_ready():
            raise RuntimeError('simulated IK/planner/executor unavailable')
        if (len(pose) != 7 or not all(map(math.isfinite, pose)) or len(joints) != 6
                or any(not math.isfinite(q) or not lo <= q <= hi for q, (lo, hi) in zip(joints, self.limits))):
            raise ValueError('invalid pose or measured start')
        self.start = tuple(joints)
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group
        req.ik_request.ik_link_name = self.tcp
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec = 2
        req.ik_request.robot_state.joint_state.name = list(JOINTS)
        req.ik_request.robot_state.joint_state.position = list(joints)
        req.ik_request.pose_stamped.header.frame_id = 'ur_base_link'
        req.ik_request.pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        p = req.ik_request.pose_stamped.pose
        p.position.x, p.position.y, p.position.z = map(float, pose[:3])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, pose[3:])
        self.generation += 1
        self.handle = None
        self.future, self.state = self.ik.call_async(req), 'ik'
        self.deadline = time.monotonic() + self.timeout

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
        goal=MoveGroup.Goal()
        req=goal.request
        req.group_name=self.group
        req.pipeline_id=self.pipeline
        req.allowed_planning_time=5.
        req.num_planning_attempts=1
        req.max_velocity_scaling_factor=.05
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
        self.deadline=time.monotonic()+self.timeout
        self._send(self.plan,goal,'plan_accept')

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
        if time.monotonic() > self.deadline:
            self.stop()
            return 'failed'
        if not self.future.done():
            return 'running'
        result = self.future.result()
        if self.state == 'ik':
            if result is None or result.error_code.val != 1:
                raise RuntimeError('IK failed')
            js = result.solution.joint_state
            q = [js.position[list(js.name).index(j)] for j in JOINTS]
            if any(not math.isfinite(v) or not lo <= v <= hi for v, (lo, hi) in zip(q, self.limits)):
                raise ValueError('IK outside joint limits')
            goal = MoveGroup.Goal()
            req = goal.request
            req.group_name, req.pipeline_id = self.group, self.pipeline
            req.allowed_planning_time = 5.0
            req.num_planning_attempts = 1
            req.max_velocity_scaling_factor = .05
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
            self._send(self.plan, goal, 'plan_accept')
        elif self.state in ('plan_accept', 'execute_accept'):
            if not result or not result.accepted:
                raise RuntimeError('action rejected')
            self.handle = result
            self.future = result.get_result_async()
            self.state = 'plan_result' if self.state == 'plan_accept' else 'execute_result'
        elif self.state == 'plan_result':
            if result.status != 4 or result.result.error_code.val != 1:
                raise RuntimeError('planning failed')
            trajectory = result.result.planned_trajectory
            validate_trajectory(trajectory.joint_trajectory, self.start, self.limits)
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
    scan coverage. Missing packets trigger a wall-time watchdog and immediate stop.
    """
    def __init__(self, node, controller, *, watchdog_seconds=1.0):
        require_sim_domain()
        if node.context.get_domain_id() != 88 or not node.get_parameter('use_sim_time').value:
            raise RuntimeError('bridge requires isolated simulation node')
        from std_msgs.msg import String, Bool
        from rclpy.clock import Clock, ClockType
        self.node, self.controller = node, controller
        self.String, self.Bool = String, Bool
        self.last_packet_wall = time.monotonic()
        self.last_sim = None
        self.sequence = 0
        self.status = node.create_publisher(String, '/restock_sim/manipulation/status', 10)
        self.gripper = node.create_publisher(Bool, '/restock_sim/gripper/enable', 10)
        self.subscriptions = [
            node.create_subscription(String, '/restock_sim/manipulation/job', self._job, 10),
            node.create_subscription(String, '/restock_sim/sensors', self._sensors, 10)]
        self.watchdog_seconds = watchdog_seconds
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

    def _job(self, message):
        try:
            data = json.loads(message.data)
            self.controller.start(data['operation'], data['job'])
            self.last_sim = None
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
        except (ValueError, KeyError, TypeError) as exc:
            self._emit(self.controller._fail('invalid sensor packet: ' + str(exc)))

    def _watchdog(self):
        if (self.controller.phase not in ('idle', 'failed', 'succeeded')
                and time.monotonic() - self.last_packet_wall > self.watchdog_seconds):
            self._emit(self.controller._fail('sensor stream watchdog expired'))


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
