"""Isaac5.1 native ROS clock/joint/TF transport and guarded drive targets.

Construct after enabling isaacsim.ros2.bridge. No rclpy import, hardware endpoint,
joint-state teleport or planner fallback. Host execution supplies stamped arm
commands; all completion feedback comes from the physical articulation.
"""
import os
import math
import time

import numpy as np

from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.restock_manipulation.ros_adapter import require_sim_domain


def _frame_id(path):
    """ROS TF frame_id for a USD prim path = the prim NAME, matching Isaac's ROS2PublishTransformTree
    (e.g. '/MobileRobot/front_camera_frame/arm_camera/restock_camera_arm' -> 'restock_camera_arm')."""
    return str(path).rsplit('/', 1)[-1]


# Arm-command heartbeat bounds. Primary bound is SIM time (does not false-trip under render pacing);
# the wall bound is only a frozen-sim backstop, far larger than the worst render-paced command gap.
ARM_HEARTBEAT_SIM_S = 0.5
ARM_HEARTBEAT_WALL_BACKSTOP_S = max(5.0, min(300.0, float(os.environ.get('RESTOCK_ARM_HEARTBEAT_WALL_BACKSTOP_S', 30.0))))  # fm362 sim-only env widen


def joint_states_payload(now, names, positions, velocities):
    """The /restock_sim/joint_states relay JSON. Measured positions AND velocities both ride along:
    the host FollowJointTrajectory executor (trajectory_host) indexes msg.velocity by the arm joints,
    so omitting velocity yields an empty array and an index-out-of-bounds fault (run-87)."""
    return {'t': float(now), 'names': list(names),
            'pos': [float(v) for v in positions], 'vel': [float(v) for v in velocities]}


def tf_payload(parent_path, child_entries, stamp):
    """Assemble the /tf relay JSON: every child published relative to the single parent (the arm
    base), frame_ids = prim names, so nvblox resolves restock_camera_arm <- ur_base_link. Pure (no
    USD): child_entries is a list of (child_path, (tx,ty,tz), (qx,qy,qz,qw))."""
    parent = _frame_id(parent_path)
    tf = [[parent, _frame_id(cp), float(t[0]), float(t[1]), float(t[2]),
           float(q[0]), float(q[1]), float(q[2]), float(q[3])] for cp, t, q in child_entries]
    return {'t': float(stamp), 'tf': tf}


class ArmCommandGate:
    def __init__(self, robot, limits, apply, *, max_speed=.15, max_error=.15, freshness_s=ARM_HEARTBEAT_SIM_S):
        self.robot,self.limits,self.apply=robot,tuple(limits),apply
        self.indices=[robot.get_dof_index(name) for name in ARM_JOINTS]
        if len(self.limits)!=6:
            raise ValueError('six measured CAD joint limits required')
        self.max_speed,self.max_error=max_speed,max_error
        # Arm-command sim-clock freshness + heartbeat bound. Real-rig default 0.5 s (ARM_HEARTBEAT_SIM_S).
        # The SIM profile relaxes it (firstmate 288, render-pacing relaxation): during the preplace
        # (sensing-held big swing + recording load the host-DDS /clock relay) the host sim clock lags the
        # ISAAC /clock ~0.5 s, so a 0.5 s bound false-rejected a fresh arm command at now-stamp=0.5000
        # (run-230). A frozen stream is still hard-caught by the 30 s wall backstop.
        self.freshness_s=float(freshness_s)
        self.last_stamp=-1.
        # Host-domain clock estimate = the greatest arm-command stamp ever received. The freshness gate
        # compares an incoming stamp against THIS (both in the host /clock domain) rather than against
        # ISAAC's live now, so the host<->ISAAC /clock-relay lag cancels instead of being tolerated
        # (firstmate 291 same-clock fix; run-230/235/244 tripped a WIDENING sim bound on pure relay lag,
        # e.g. now-stamp=2.5000 at approach once the heavier cuMotion load grew the lag). The newest
        # command has stamp==_host_now (age 0 -> always fresh); only a genuinely old resend is rejected.
        self._host_now=float('-inf')
        self.last_receive=time.monotonic()
        self.started=False
        self.faulted=False
        self.hold_target=None
        # Temporary (non-faulting) contact hold: freeze the arm at the measured joints while the sim
        # attach-fallback settles the link to ~0 velocity before welding, so the weld snaps no moving
        # link (firstmate run-209). Distinct from hold()/faulted, which is a terminal e-stop; this is
        # released (begin/end_contact_hold) and the arm resumes host commands for lift/couple/tow/park.
        self._contact_hold=False
        self._contact_hold_target=None

    def begin_contact_hold(self):
        if not self.faulted:
            self._contact_hold_target=self.measured()
            self._contact_hold=True
            self.apply(self._contact_hold_target,np.zeros(6),self.indices)

    def end_contact_hold(self):
        self._contact_hold=False
        self._contact_hold_target=None

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
        if self._contact_hold:
            # Temporary contact-settle freeze: hold the measured joints, ignore the host's continued
            # over-press, so the link decelerates to ~0 before the stationary weld. Released shortly.
            self.apply(self._contact_hold_target,np.zeros(6),self.indices)
            return
        try:
            # This is a POSITION+VELOCITY command (``apply`` drives only q, v); the effort field is
            # never commanded and is NOT validated. trajectory_host publishes no effort, but Isaac's
            # ROS2SubscribeJointState zero- or torque-fills effort to the joint count (run-89 saw
            # eff=6 all-zero; run-90 saw eff=6 non-zero measured torques), so gating on effort at all
            # wrongly rejected the executed grasp command. Validate only names, position, velocity and
            # the stamp; position/velocity limits + tracking envelope are enforced below.
            if math.isfinite(stamp):
                self._host_now=max(self._host_now,stamp)   # advance the host-domain clock estimate
            # Freshness in the HOST domain (self._host_now - stamp), NOT ISAAC now - stamp: the newest
            # command is age 0 and always fresh regardless of the relay lag; a genuinely old resend
            # (stamp far behind the latest host stamp) is still rejected. ``now`` is kept for the diag.
            host_age = self._host_now-stamp if math.isfinite(stamp) else float('nan')
            if (len(names)!=6 or set(names)!=set(ARM_JOINTS) or len(positions)!=6
                    or len(velocities)!=6
                    or not math.isfinite(stamp) or not -0.05 <= host_age <= self.freshness_s):
                print(f'ARM_CMD_REJECT names={len(names)} setok={set(names)==set(ARM_JOINTS)} '
                      f'pos={len(positions)} vel={len(velocities)} eff={len(efforts)} '
                      f'stamp={stamp!r} host_now={self._host_now!r} host_age={host_age:.4f} '
                      f'isaac_now={now!r} relay_lag={now-stamp if math.isfinite(stamp) else float("nan"):.4f}',
                      flush=True)  # diagnostic: which sub-condition rejected the executed arm command
                raise ValueError('invalid or stale simulated arm command')
            if stamp<=self.last_stamp:
                return  # reordered/duplicate commands cannot move the arm
            order=[names.index(name) for name in ARM_JOINTS]
            q=np.asarray(positions,float)[order]
            v=np.asarray(velocities,float)[order]
            track_err=float(np.max(np.abs(q-self.measured())))
            vel_mag=float(np.max(np.abs(v)))
            if (not np.isfinite(q).all() or not np.isfinite(v).all()
                    or any(not lo<=value<=hi for value,(lo,hi) in zip(q,self.limits))
                    or vel_mag>self.max_speed
                    or track_err>self.max_error):
                # Diagnostic: which sub-condition rejected the command (the Isaac-side command-vs-
                # measured tracking envelope mirrors trajectory_host's TRACKING_TOL; under render pacing
                # the ~0.15 rad control lag can trip a too-tight max_error, run-115).
                print(f'ARM_CMD_ENVELOPE track_err={track_err:.4f} vel={vel_mag:.4f} '
                      f'max_error={self.max_error} max_speed={self.max_speed} '
                      f'limit_ok={all(lo<=value<=hi for value,(lo,hi) in zip(q,self.limits))}',flush=True)
                raise ValueError('arm target exceeds limits or tracking envelope')
            self.apply(q,v,self.indices)
            self.last_stamp=stamp
            self.last_receive=time.monotonic()
            self.started=True
        except Exception:
            self.hold()
            raise

    def watchdog(self, now):
        # Arm-command heartbeat. With the freshness gate now in the HOST /clock domain (self._host_now),
        # a sim-time "now - last_stamp" heartbeat would re-introduce the very relay lag the same-clock fix
        # removes (ISAAC now runs ahead of the host stamps), so the heartbeat is the generous WALL
        # backstop ONLY: a genuinely dead/stalled arm stream stops advancing last_receive and faults
        # after ARM_HEARTBEAT_WALL_BACKSTOP_S. Under render pacing arm commands are 1.7-3.5 s apart in
        # WALL but the backstop is far larger, so it never false-trips a healthy render-paced grasp.
        if self.started and time.monotonic()-self.last_receive>ARM_HEARTBEAT_WALL_BACKSTOP_S:
            self.hold()
            raise RuntimeError('arm command heartbeat lost')
        if self.faulted:
            self.hold()


class NativeTransport:
    def __init__(self, *, stage, articulation_path, arm_base_path, tf_paths,
                 command_gate, root='/RestockTransport', robot=None, runtime_env=None,
                 python='python3', base_path=None):
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
        # The URDF root link (base_link) is MoveIt's planning frame, but nvblox/the relay root the
        # tree at ur_base_link; without base_link <- ur_base_link in /tf, move_group cannot transform
        # the collision object to its planning frame and the approach plan fails ("base_link ... does
        # not exist"). Publish that rigid link each tick so the host tree reaches base_link too.
        self.base_path = base_path if (base_path and stage.GetPrimAtPath(base_path)) else None
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
        self.command_gate.watchdog(now)
        self._publish_relay(now)

    def _publish_relay(self, now):
        """Ship the sim-time backbone (/clock, /joint_states, /tf) to the host-DDS relay subprocess
        so the whole host stack shares the render-paced simulation clock (OmniGraph can't reach it).
        Publishes /tf BEFORE the masked depth for this stamp: this runs inside isaac_port.advance,
        which precedes the manipulation runtime's sensing_bridge.publish on the same tick."""
        if self._relay is None:
            return
        import json
        import numpy as np
        from pxr import UsdGeom, Usd
        self._relay.publish('/clock', json.dumps({'t': float(now)}))
        if self.robot is not None:
            pos = [float(v) for v in np.asarray(self.robot.get_joint_positions(), float)]
            # Velocity MUST ride along: the host FollowJointTrajectory executor (trajectory_host)
            # indexes msg.velocity by the arm joints, so a names+position-only message (empty
            # velocity) raised 'index out of bounds, size 0' and emergency-stopped the mission at the
            # grasp (run-87). The relay decode reads 'vel' (control_ipc), defaulting to [] when absent.
            vel = [float(v) for v in np.asarray(self.robot.get_joint_velocities(), float)]
            self._relay.publish('/restock_sim/joint_states',
                                json.dumps(joint_states_payload(now, self.robot.dof_names, pos, vel)))
        # Each tf frame relative to the arm base, via USD world transforms (ExtractRotationQuat gives
        # the correct USD-convention quaternion). Pure frame-id + payload assembly is tf_payload().
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        parent_inv = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.arm_base_path)).GetInverse()
        entries = []
        for path in self.tf_paths:
            rel = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path)) * parent_inv
            t = rel.ExtractTranslation()
            q = rel.ExtractRotationQuat()
            im = q.GetImaginary()
            entries.append((path, (t[0], t[1], t[2]), (im[0], im[1], im[2], q.GetReal())))
        payload = tf_payload(self.arm_base_path, entries, now)
        if self.base_path:                           # base_link <- ur_base_link, MoveIt's planning frame
            rel = (cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.arm_base_path))
                   * cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.base_path)).GetInverse())
            t = rel.ExtractTranslation()
            q = rel.ExtractRotationQuat()
            im = q.GetImaginary()
            base = tf_payload(self.base_path, [(self.arm_base_path, (t[0], t[1], t[2]),
                                                (im[0], im[1], im[2], q.GetReal()))], now)
            payload['tf'].extend(base['tf'])
        self._relay.publish('/tf', json.dumps(payload))

    def close(self):
        self.command_gate.hold()
        if self._relay is not None:
            try:
                self._relay.close()   # terminate the out-of-Isaac sim-clock/TF/joints relay subprocess
            except Exception:
                pass
