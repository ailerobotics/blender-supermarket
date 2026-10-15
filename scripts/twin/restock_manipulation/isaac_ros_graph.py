"""Isaac native ROS bridge: no Python rclpy, cuRobo or nvblox imports.

Construct inside integration's guarded app after enabling isaacsim.ros2.bridge.
Host Jazzy runs ros_host separately. Existing clock, TF, joint trajectory and
RGB-D graphs remain integration-owned; this graph adds manipulation messages.
"""
from dataclasses import asdict
import json
import time

from .ros_adapter import require_sim_domain
from ..restock_camera.freshness import require_fresh


TOPICS = {'sensors': '/restock_sim/sensors',
          'job': '/restock_sim/manipulation/job',
          'status': '/restock_sim/manipulation/status',
          'gripper': '/restock_sim/gripper/enable',
          'stop': '/restock_sim/manipulation/stop'}


class IsaacMeasuredFeedback:
    """Planner-free sensor adapter for the real imported articulation.

    Integration sets collision_time only from mapper incorporation acknowledgement,
    and resets it to -1 after docking/frame changes. No depth-receipt shortcut.
    """
    def __init__(self, articulation):
        from .cad_tool import ARM_JOINTS
        self.robot = articulation
        self.indices = [articulation.get_dof_index(name) for name in ARM_JOINTS]
        self.collision_time = -1.
        self.source = None

    def observe_source(self, observation):
        if observation.source != 'sensor' or observation.frame != 'ur_base_link':
            raise ValueError('source must be sensor-derived in current arm frame')
        self.source = observation

    def joint_positions(self):
        import numpy as np
        joints = np.asarray(self.robot.get_joint_positions(joint_indices=self.indices),float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError('invalid measured arm encoders')
        return tuple(joints)

    def parked(self):
        import numpy as np
        linear = np.asarray(self.robot.get_linear_velocity(),float)
        angular = np.asarray(self.robot.get_angular_velocity(),float)
        return bool(np.isfinite(linear).all() and np.isfinite(angular).all()
                    and np.linalg.norm(linear)<.005 and np.linalg.norm(angular)<.005)


class IsaacRosGraph:
    def __init__(self, *, contact, hold, root='/RestockManipulationROS', timeout=5.,
                 wall_backstop=180., host_control=None):
        require_sim_domain()
        import omni.graph.core as og
        import omni.usd
        if omni.usd.get_context().get_stage().GetPrimAtPath(root):
            raise ValueError('refusing to overwrite an existing ROS graph')
        self.og, self.contact, self.hold = og, contact, hold
        self.root, self.timeout = root, timeout
        # Node-clock (sim-time) status bound + a generous wall backstop for a frozen sim clock.
        self.wall_backstop = wall_backstop
        self.last_receive_sim = None
        # Isaac->host PUBLISHES (job, sensors) go through the host-DDS subprocess, since the
        # OmniGraph ROS2 publisher does not reach the host UDPv4 DDS (run-59). The host->Isaac
        # SUBSCRIBERS (status, stop, gripper) stay on OmniGraph, which discovers the host writers.
        self.host_control = host_control
        self.active, self.last_status = False, ''
        self.last_sequence = -1
        self.faulted, self.job_id = False, None
        self.last_receive = time.monotonic()
        for key, topic in TOPICS.items():
            publish = key in ('sensors', 'job')
            # job/sensors are published out-of-Isaac via the control subprocess; /manipulation/status
            # is forwarded back by it too (the OmniGraph subscriber delivered it too slowly for the
            # 5 s manip watchdog). stop + gripper stay on OmniGraph subscribers.
            if host_control is not None and key in ('sensors', 'job', 'status'):
                continue
            node_type = 'ROS2Publisher' if publish else 'ROS2Subscriber'
            tick_type = 'OnImpulseEvent' if publish else 'OnPlaybackTick'
            tick_port = 'execOut' if publish else 'tick'
            graph, _, _, _ = og.Controller.edit(
                {'graph_path': root+'/'+key, 'evaluator_name': 'execution'}, {
                og.Controller.Keys.CREATE_NODES: [
                    ('tick','omni.graph.action.'+tick_type),
                    ('context','isaacsim.ros2.bridge.ROS2Context'),
                    ('wire','isaacsim.ros2.bridge.'+node_type)],
                og.Controller.Keys.SET_VALUES: [
                    ('context.inputs:useDomainIDEnvVar',False),
                    ('context.inputs:domain_id',88),
                    ('wire.inputs:topicName',topic),
                    ('wire.inputs:messagePackage','std_msgs'),
                    ('wire.inputs:messageSubfolder','msg'),
                    ('wire.inputs:messageName','Bool' if key in ('stop','gripper') else 'String')],
                og.Controller.Keys.CONNECT: [
                    ('tick.outputs:'+tick_port,'wire.inputs:execIn'),
                    ('context.outputs:context','wire.inputs:context')]})
            # MessageName materializes dynamic data attributes in the native node.
            og.Controller.evaluate_sync(graph)

    def _publish(self, key, data):
        if self.host_control is not None and key in ('sensors', 'job'):
            self.host_control.publish(TOPICS[key], data)  # host-DDS subprocess (reaches the host)
            return
        og = self.og
        path = self.root+'/'+key
        og.Controller.attribute(path+'/wire.inputs:data').set(data)
        og.Controller.attribute(path+'/tick.state:enableImpulse').set(True)
        og.Controller.evaluate_sync(path)

    def start(self, operation, job):
        if self.active or self.faulted:
            raise RuntimeError('transfer active or explicit fault recovery required')
        self._publish('job', json.dumps({'operation':operation,'job':job},allow_nan=False))
        self.active = True
        self.job_id = job['job_id']
        self.last_receive = time.monotonic()
        self.last_receive_sim = None   # re-seeded from the first poll of this job

    def publish_sensors(self, sensors):
        self._publish('sensors',json.dumps(asdict(sensors),allow_nan=False))

    def check_health(self):
        """Check global stop every physics tick without consuming job replies.

        Integration also calls this while navigating or warming the mapper,
        when the manipulation state machine does not call poll().
        """
        if self.faulted:
            self.hold()
            raise RuntimeError('explicit fault recovery required')
        og = self.og
        if og.Controller.attribute(self.root+'/stop/wire.outputs:data').get():
            self.hold()
            self.faulted = True
            self.active = False
            raise RuntimeError('host emergency stop')

    def _status_fault(self, reason):
        self.hold()
        self.faulted = True
        self.active = False
        return {'status':'failed','reason':reason}

    def poll(self, simulation_time=None):
        """Call every physics step, including steps without a fresh image.

        ``simulation_time`` (the node/sim clock, seconds) node-clocks the status watchdog.
        The host goes legitimately silent during a long render-paced cuMotion multi-seed IK
        retry: many WALL seconds pass while the sim clock barely advances across each render
        tick, so a wall-time 5 s bound false-trips and kills the mission mid-grasp (run-82).
        Measuring the bound in SIM time keeps exactly the 5 s semantics on real hardware
        (sim == wall) without false-tripping under render pacing; a generous WALL backstop
        still hard-stops a genuinely dead host when the sim clock is frozen (the sim bound can
        never advance to trip then). Called with no sim clock => legacy wall behaviour.
        """
        try:
            self.check_health()
        except RuntimeError as exc:
            return {'status':'failed','reason':str(exc)}
        og=self.og
        # /manipulation/status arrives via the control subprocess (host-side latency) when a
        # host_control relay is wired; otherwise the OmniGraph subscriber.
        host_control = getattr(self, 'host_control', None)
        if host_control is not None:
            raw = host_control.read(TOPICS['status'])
        else:
            raw = og.Controller.attribute(self.root+'/status/wire.outputs:data').get()
        result = None
        if raw and raw != self.last_status:
            result = json.loads(raw)
            sequence = result.get('sequence')
            if (not self.active or result.get('job_id') != self.job_id or
                    type(sequence) is not int or sequence <= self.last_sequence):
                result = None
            if result is None:
                raw = ''
        if result is not None:
            if result.get('status')!='failed' and any(c.get('kind')=='gripper' for c in result.get('controls',[])):
                try:
                    deadline=result.get('valid_until_steady')
                    if deadline is None:
                        raise ValueError('missing acquisition deadline for gripper command')
                    require_fresh(deadline,time.monotonic)
                except (ValueError,TypeError) as exc:
                    self.hold()
                    self.faulted,self.active=True,False
                    return dict(status='failed',reason=str(exc))
            self.last_sequence = result['sequence']
            self.last_status, self.last_receive = raw, time.monotonic()
            if simulation_time is not None:
                self.last_receive_sim = simulation_time
            for control in result.get('controls',[]):
                if control.get('kind') == 'gripper':
                    self.contact.command(control['enabled'])
            if result.get('status') in ('succeeded','failed','rejected'):
                self.active = False
        if self.active:
            wall_now = time.monotonic()
            if simulation_time is None:
                if wall_now - self.last_receive > self.timeout:
                    return self._status_fault('host process/status watchdog')
            else:
                last_sim = getattr(self, 'last_receive_sim', None)
                if last_sim is None:
                    self.last_receive_sim = simulation_time   # seed from the first poll of the job
                elif simulation_time - last_sim > self.timeout:
                    return self._status_fault('host process/status watchdog')
                elif wall_now - self.last_receive > getattr(self, 'wall_backstop', 180.):
                    return self._status_fault('host process/status wall backstop')
        return result

    def close(self):
        self.hold()  # leave any physically held payload sealed
        self.active = False
        self.faulted = True
        if self.host_control is not None:
            try:
                self.host_control.close()  # terminate the out-of-Isaac control publisher subprocess
            except Exception:
                pass
