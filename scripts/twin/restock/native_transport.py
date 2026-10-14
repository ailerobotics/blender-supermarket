"""Isaac5.1 native ROS clock/joint/TF transport and guarded drive targets.

Construct after enabling isaacsim.ros2.bridge. No rclpy import, hardware endpoint,
joint-state teleport or planner fallback. Host execution supplies stamped arm
commands; all completion feedback comes from the physical articulation.
"""
import math
import time

import numpy as np

from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.restock_manipulation.ros_adapter import require_sim_domain


class ArmCommandGate:
    def __init__(self, robot, limits, apply, *, max_speed=.15, max_error=.15):
        self.robot,self.limits,self.apply=robot,tuple(limits),apply
        self.indices=[robot.get_dof_index(name) for name in ARM_JOINTS]
        if len(self.limits)!=6:
            raise ValueError('six measured CAD joint limits required')
        self.max_speed,self.max_error=max_speed,max_error
        self.last_stamp=-1.
        self.last_receive=time.monotonic()
        self.started=False
        self.faulted=False
        self.hold_target=None

    def measured(self):
        positions=np.asarray(self.robot.get_joint_positions(joint_indices=self.indices),float)
        if positions.shape!=(6,) or not np.isfinite(positions).all():
            raise ValueError('invalid arm feedback')
        return positions

    def hold(self):
        if self.hold_target is None:
            self.hold_target=self.measured()
        self.faulted=True
        self.apply(self.hold_target,np.zeros(6),self.indices)

    def command(self, *, names, positions, velocities, efforts, stamp, now):
        if self.faulted:
            self.hold()
            return
        try:
            if (len(names)!=6 or set(names)!=set(ARM_JOINTS) or len(positions)!=6
                    or len(velocities)!=6 or len(efforts)!=0
                    or not math.isfinite(stamp) or not 0 <= now-stamp <= .2):
                raise ValueError('invalid or stale simulated arm command')
            if stamp<=self.last_stamp:
                return  # reordered/duplicate commands cannot move the arm
            order=[names.index(name) for name in ARM_JOINTS]
            q=np.asarray(positions,float)[order]
            v=np.asarray(velocities,float)[order]
            if (not np.isfinite(q).all() or not np.isfinite(v).all()
                    or any(not lo<=value<=hi for value,(lo,hi) in zip(q,self.limits))
                    or np.max(np.abs(v))>self.max_speed
                    or np.max(np.abs(q-self.measured()))>self.max_error):
                raise ValueError('arm target exceeds limits or tracking envelope')
            self.apply(q,v,self.indices)
            self.last_stamp=stamp
            self.last_receive=time.monotonic()
            self.started=True
        except Exception:
            self.hold()
            raise

    def watchdog(self):
        if self.started and time.monotonic()-self.last_receive>.5:
            self.hold()
            raise RuntimeError('arm command heartbeat lost')
        if self.faulted:
            self.hold()


class NativeTransport:
    def __init__(self, *, stage, articulation_path, arm_base_path, tf_paths,
                 command_gate, root='/RestockTransport', robot=None, runtime_env=None,
                 python='python3'):
        require_sim_domain()
        import omni.graph.core as og
        from pxr import Sdf
        if stage.GetPrimAtPath(root):
            raise ValueError('refusing to replace native transport graph')
        for path in (articulation_path,arm_base_path,*tf_paths):
            if not stage.GetPrimAtPath(path):
                raise ValueError('native transport frame missing: '+path)
        self.og,self.root,self.command_gate=og,root,command_gate
        self.stage,self.tf_paths=stage,list(tf_paths)
        self.arm_base_path=arm_base_path
        self.robot=robot            # physics articulation for /joint_states (names+positions)
        # Host-DDS relay for the sim-time backbone: /clock, /tf, /restock_sim/joint_states. These are
        # OmniGraph publishers below too, but OmniGraph publishers are invisible to the host UDPv4 DDS
        # (run-69: /clock had 0 publishers to the host, so host use_sim_time fell back to wall time).
        # Relaying them out-of-Isaac (like masked depth + the control plane) gives the whole host
        # stack ONE render-paced simulation clock + TF + joints.
        self._relay=None
        if runtime_env:
            from .control_ipc import HostControlPublisher
            self._relay=HostControlPublisher(
                topics=('/clock', '/tf', '/restock_sim/joint_states'),
                runtime_env=runtime_env, python=python, node_name='restock_transport_pub')
        # Last Isaac stage time published on /clock (set each tick). The station/command stamps
        # must use THIS clock domain -- never the mission-relative Packet.simulation_time, which is
        # zeroed at the first mission action -- so the host ack's /clock-validated 0.5 s freshness
        # guard sees one epoch. Read it at the moment of publish via a stage_clock seam.
        self.stage_time=0.
        keys=og.Controller.Keys
        nodes=[('tick','omni.graph.action.OnImpulseEvent'),
               ('context','isaacsim.ros2.bridge.ROS2Context'),
               ('clock','isaacsim.ros2.bridge.ROS2PublishClock'),
               ('joints','isaacsim.ros2.bridge.ROS2PublishJointState'),
               ('tf','isaacsim.ros2.bridge.ROS2PublishTransformTree'),
               ('command','isaacsim.ros2.bridge.ROS2SubscribeJointState')]
        values=[('context.inputs:useDomainIDEnvVar',False),('context.inputs:domain_id',88),
                ('clock.inputs:topicName','/clock'),
                ('joints.inputs:topicName','/restock_sim/joint_states'),
                ('joints.inputs:targetPrim',[Sdf.Path(articulation_path)]),
                ('tf.inputs:topicName','/tf'),('tf.inputs:parentPrim',[Sdf.Path(arm_base_path)]),
                ('tf.inputs:targetPrims',[Sdf.Path(p) for p in tf_paths]),
                ('command.inputs:topicName','/restock_sim/arm_joint_command')]
        connections=[]
        for name in ('clock','joints','tf','command'):
            connections.extend([('tick.outputs:execOut',name+'.inputs:execIn'),
                                ('context.outputs:context',name+'.inputs:context')])
        og.Controller.edit({'graph_path':root,'evaluator_name':'execution'},
            {keys.CREATE_NODES:nodes,keys.SET_VALUES:values,keys.CONNECT:connections})

    def add_frames(self, paths):
        """Include actual optical mount prims after camera construction.

        TF stays relative to the measured moving arm base; cart-mounted frames
        retain their independently articulated pose. No object/support poses
        are published as perceived measurements.
        """
        from pxr import Sdf
        paths=list(paths)
        if any(not self.stage.GetPrimAtPath(path) for path in paths):
            raise ValueError('native optical TF frame missing')
        self.tf_paths=list(dict.fromkeys([*self.tf_paths,*paths]))
        self.og.Controller.attribute(self.root+'/tf.inputs:targetPrims').set(
            [Sdf.Path(path) for path in self.tf_paths])

    def tick(self, now):
        og=self.og
        self.stage_time=float(now)   # the /clock value this tick; station stamps read it
        for name in ('clock','joints','tf'):
            og.Controller.attribute(self.root+'/'+name+'.inputs:timeStamp').set(float(now))
        og.Controller.attribute(self.root+'/tick.state:enableImpulse').set(True)
        og.Controller.evaluate_sync(self.root)
        get=lambda name:og.Controller.attribute(self.root+'/command.outputs:'+name).get()
        names=list(get('jointNames') or [])
        if names:
            self.command_gate.command(names=names,positions=get('positionCommand'),
                velocities=get('velocityCommand'),efforts=get('effortCommand'),
                stamp=float(get('timeStamp')),now=now)
        self.command_gate.watchdog()
        self._publish_relay(now)

    def _publish_relay(self, now):
        """Ship the sim-time backbone (/clock, /joint_states, /tf) to the host-DDS relay subprocess
        so the whole host stack shares the render-paced simulation clock (OmniGraph can't reach it)."""
        if self._relay is None:
            return
        import json
        import numpy as np
        from pxr import UsdGeom, Usd
        self._relay.publish('/clock', json.dumps({'t': float(now)}))
        if self.robot is not None:
            pos = [float(v) for v in np.asarray(self.robot.get_joint_positions(), float)]
            self._relay.publish('/restock_sim/joint_states',
                                json.dumps({'t': float(now), 'names': list(self.robot.dof_names), 'pos': pos}))
        # Each tf frame relative to the arm base, via USD world transforms (ExtractRotationQuat gives
        # the correct USD-convention quaternion; frame_ids are the prim names, as OmniGraph used).
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        parent_inv = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.arm_base_path)).GetInverse()
        parent_name = self.arm_base_path.rsplit('/', 1)[-1]
        rows = []
        for path in self.tf_paths:
            rel = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path)) * parent_inv
            t = rel.ExtractTranslation()
            q = rel.ExtractRotationQuat()
            im = q.GetImaginary()
            rows.append([parent_name, path.rsplit('/', 1)[-1], float(t[0]), float(t[1]), float(t[2]),
                         float(im[0]), float(im[1]), float(im[2]), float(q.GetReal())])
        self._relay.publish('/tf', json.dumps({'t': float(now), 'tf': rows}))

    def close(self):
        self.command_gate.hold()
        if self._relay is not None:
            try:
                self._relay.close()   # terminate the out-of-Isaac sim-clock/TF/joints relay subprocess
            except Exception:
                pass
