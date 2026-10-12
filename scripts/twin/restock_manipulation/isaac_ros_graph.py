"""Isaac native ROS bridge: no Python rclpy, cuRobo or nvblox imports.

Construct inside integration's guarded app after enabling isaacsim.ros2.bridge.
Host Jazzy runs ros_host separately. Existing clock, TF, joint trajectory and
RGB-D graphs remain integration-owned; this graph adds manipulation messages.
"""
from dataclasses import asdict
import json
import time

from .ros_adapter import require_sim_domain


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
    def __init__(self, *, contact, hold, root='/RestockManipulationROS', timeout=5.):
        require_sim_domain()
        import omni.graph.core as og
        import omni.usd
        if omni.usd.get_context().get_stage().GetPrimAtPath(root):
            raise ValueError('refusing to overwrite an existing ROS graph')
        self.og, self.contact, self.hold = og, contact, hold
        self.root, self.timeout = root, timeout
        self.active, self.last_status = False, ''
        self.last_sequence = -1
        self.faulted, self.job_id = False, None
        self.last_receive = time.monotonic()
        for key, topic in TOPICS.items():
            publish = key in ('sensors', 'job')
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

    def publish_sensors(self, sensors):
        self._publish('sensors',json.dumps(asdict(sensors),allow_nan=False))

    def poll(self):
        """Call every physics step, including steps without a fresh image."""
        if self.faulted:
            self.hold()
            return {'status':'failed','reason':'explicit fault recovery required'}
        og = self.og
        if og.Controller.attribute(self.root+'/stop/wire.outputs:data').get():
            self.hold()
            self.faulted = True
            self.active = False
            return {'status':'failed','reason':'host emergency stop'}
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
            self.last_sequence = result['sequence']
            self.last_status, self.last_receive = raw, time.monotonic()
            for control in result.get('controls',[]):
                if control.get('kind') == 'gripper':
                    self.contact.command(control['enabled'])
            if result.get('status') in ('succeeded','failed','rejected'):
                self.active = False
        if self.active and time.monotonic()-self.last_receive > self.timeout:
            self.hold()
            self.faulted = True
            self.active = False
            return {'status':'failed','reason':'host process/status watchdog'}
        return result

    def close(self):
        self.hold()  # leave any physically held payload sealed
        self.active = False
        self.faulted = True
