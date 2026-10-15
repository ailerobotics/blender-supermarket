"""Simulation-only FollowJointTrajectory executor, on host ROS Python.

Interpolates supplied plans; never plans a path or changes measured joint states.
Every cubic segment is checked analytically for position, speed and acceleration
extrema. Completion requires fresh physical feedback. Any execution fault latches
a stop until the task-owned process is restarted.
"""
import argparse
import math
import threading
import time

import numpy as np

from twin.restock_manipulation.cad_tool import ARM_JOINTS, inspect_tool
from twin.restock_manipulation.ros_adapter import require_sim_domain, validate_trajectory

# Measured-feedback staleness watchdog bounds. The primary bound is NODE/SIM time (use_sim_time),
# so it does not false-trip under render pacing; the wall bound is only a frozen-sim backstop and is
# deliberately far larger than the worst render-paced /joint_states wall gap (~1.7-3.5 s).
FEEDBACK_SIM_STALE_S = 0.2
FEEDBACK_WALL_BACKSTOP_S = 30.0


def feedback_stale(sim_now, feedback_sim_stamp, wall_now, wall_received):
    """True if measured joint feedback is stale. Primary bound is SIM time (does not false-trip under
    render pacing); the wall bound is only a frozen-sim backstop (sim can't advance to trip then)."""
    return (sim_now - feedback_sim_stamp > FEEDBACK_SIM_STALE_S
            or wall_now - wall_received > FEEDBACK_WALL_BACKSTOP_S)


class Plan:
    def __init__(self, trajectory, measured, limits, max_speed=.15, max_acceleration=.5):
        validate_trajectory(trajectory, measured, limits)
        order=[list(trajectory.joint_names).index(n) for n in ARM_JOINTS]
        self.times=[]
        positions=[]
        velocities=[]
        for p in trajectory.points:
            if len(p.velocities)!=6 or p.effort:
                raise ValueError('six explicit waypoint velocities and no effort required')
            q=np.asarray(p.positions,float)[order]
            v=np.asarray(p.velocities,float)[order]
            if not np.isfinite(v).all():
                raise ValueError('nonfinite planned velocity')
            self.times.append(p.time_from_start.sec+p.time_from_start.nanosec*1e-9)
            positions.append(q)
            velocities.append(v)
        if len(positions)<2 or self.times[0]!=0 or np.max(np.abs(velocities[0]))>1e-3:
            raise ValueError('plan must start at rest at zero and contain two points')
        self.end=positions[-1]
        self.segments=[]
        for i in range(len(positions)-1):
            duration=self.times[i+1]-self.times[i]
            q0,q1=positions[i:i+2]
            v0,v1=velocities[i:i+2]
            # Cubic Hermite coefficients in seconds from the segment start.
            a=(2*(q0-q1)+duration*(v0+v1))/duration**3
            b=(3*(q1-q0)-duration*(2*v0+v1))/duration**2
            for j,(lo,hi) in enumerate(limits):
                times={0.,duration}
                roots=np.roots([3*a[j],2*b[j],v0[j]])
                times.update(float(r.real) for r in roots
                             if abs(r.imag)<1e-10 and 0<r.real<duration)
                if any(not lo<=((a[j]*t+b[j])*t+v0[j])*t+q0[j]<=hi for t in times):
                    raise ValueError('interpolated joint limit violation')
                speed_times={0.,duration}
                if abs(a[j])>1e-15 and 0 < -b[j]/(3*a[j]) < duration:
                    speed_times.add(-b[j]/(3*a[j]))
                if max(abs(3*a[j]*t*t+2*b[j]*t+v0[j]) for t in speed_times)>max_speed+1e-9:
                    raise ValueError('interpolated speed limit violation')
                if max(abs(2*b[j]),abs(6*a[j]*duration+2*b[j]))>max_acceleration+1e-9:
                    raise ValueError('interpolated acceleration limit violation')
            self.segments.append((a,b,v0,q0))

    def sample(self, elapsed):
        if not math.isfinite(elapsed) or elapsed<0:
            raise ValueError('invalid trajectory clock')
        if elapsed>=self.times[-1]:
            return self.end.copy(),np.zeros(6)
        i=max(0,int(np.searchsorted(self.times,elapsed,side='right'))-1)
        t=elapsed-self.times[i]
        a,b,c,d=self.segments[i]
        return ((a*t+b)*t+c)*t+d,(3*a*t+2*b)*t+c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf',required=True)
    args=parser.parse_args()
    require_sim_domain()
    limits=inspect_tool(args.urdf).arm_limits
    import rclpy
    from rclpy.action import ActionServer, GoalResponse, CancelResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from control_msgs.action import FollowJointTrajectory
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool

    rclpy.init()
    node=Node('restock_trajectory_executor',parameter_overrides=[Parameter('use_sim_time',value=True)])
    if node.context.get_domain_id()!=88:
        raise RuntimeError('trajectory executor requires simulation domain 88')
    lock=threading.RLock()
    state=dict(feedback=None,received=0.,fault=None,active=False,target=None,plan=None,start=None)
    group=ReentrantCallbackGroup()
    command=node.create_publisher(JointState,'/restock_sim/arm_joint_command',10)
    stop=node.create_publisher(Bool,'/restock_sim/manipulation/stop',10)
    now=lambda:node.get_clock().now().nanoseconds*1e-9

    def fault(reason):
        state['fault']=reason
        state['plan']=None
        print(f'TRAJ_HOST_FAULT {reason}',flush=True)  # diagnostic: the stop source is otherwise unlogged
        stop.publish(Bool(data=True))

    def feedback(msg):
        with lock:
            try:
                if len(set(msg.name))!=len(msg.name):
                    raise ValueError('duplicate feedback joints')
                order=[list(msg.name).index(n) for n in ARM_JOINTS]
                q=np.asarray(msg.position,float)[order]
                v=np.asarray(msg.velocity,float)[order]
                stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
                if not np.isfinite(q).all() or not np.isfinite(v).all() or not 0<=now()-stamp<=.2:
                    raise ValueError('stale/nonfinite joint feedback')
                if state['feedback'] is not None and stamp<=state['feedback'][0]:
                    return
                state['feedback']=(stamp,q,v)
                state['received']=time.monotonic()
                if state['target'] is None:
                    state['target']=q.copy()
            except Exception as exc:
                fault(str(exc))

    def tick():
        with lock:
            if state['fault']:
                stop.publish(Bool(data=True))
                return
            if state['feedback'] is None:
                return
            # Node-clock the feedback-staleness bound. Measured in NODE/SIM time (``now()`` is the
            # sim clock, use_sim_time) so it does not false-trip under render pacing, where Isaac
            # publishes /joint_states every ~1.7-3.5 s WALL (a full-res capture render tick) but only
            # a small step apart in SIM time. A wall-time 0.5 s bound here emergency-stopped the
            # mission at the render-paced collect/grasp (run-86); the SIM bound keeps the same
            # tracking freshness. A GENEROUS wall backstop still hard-stops a genuinely frozen sim
            # clock (a dead Isaac), where the sim bound can never advance to trip.
            if feedback_stale(now(), state['feedback'][0], time.monotonic(), state['received']):
                fault('measured joint feedback lost')
                return
            q,v=state['target'],np.zeros(6)
            if state['plan'] is not None:
                q,v=state['plan'].sample(now()-state['start'])
            if np.max(np.abs(q-state['feedback'][1]))>.15:
                fault('trajectory tracking error exceeds .15 rad')
                return
            msg=JointState()
            msg.header.stamp=node.get_clock().now().to_msg()
            msg.name=list(ARM_JOINTS)
            msg.position=q.tolist()
            msg.velocity=v.tolist()
            command.publish(msg)

    def goal(request):
        with lock:
            if state['fault'] or state['active'] or state['feedback'] is None:
                return GoalResponse.REJECT
            try:
                if request.multi_dof_trajectory.points:
                    raise ValueError('multi-DOF commands unsupported')
                if request.trajectory.header.stamp.sec or request.trajectory.header.stamp.nanosec:
                    raise ValueError('only immediate simulation goals supported')
                # This bounded executor declares its fixed tolerance policy explicitly.
                if request.path_tolerance or request.goal_tolerance:
                    raise ValueError('custom tolerance policy unsupported')
                Plan(request.trajectory,state['feedback'][1],limits)
                state['active']=True
                return GoalResponse.ACCEPT
            except ValueError:
                return GoalResponse.REJECT

    def execute(handle):
        result=FollowJointTrajectory.Result()
        try:
            with lock:
                plan=Plan(handle.request.trajectory,state['feedback'][1],limits)
                state['plan'],state['start']=plan,now()
                started=state['start']
            wall_deadline=time.monotonic()+min(300.,20.+plan.times[-1]*5)
            while rclpy.ok():
                with lock:
                    if handle.is_cancel_requested:
                        fault('planned trajectory canceled')
                        handle.canceled()
                        result.error_code=result.PATH_TOLERANCE_VIOLATED
                        result.error_string=state['fault']
                        return result
                    elapsed=now()-started
                    if elapsed<0 or time.monotonic()>wall_deadline:
                        fault('trajectory clock/timeout failure')
                    if state['fault']:
                        raise RuntimeError(state['fault'])
                    measured=state['feedback']
                    desired,desired_velocity=plan.sample(elapsed)
                    feedback_message=FollowJointTrajectory.Feedback()
                    feedback_message.header.stamp=node.get_clock().now().to_msg()
                    feedback_message.joint_names=list(ARM_JOINTS)
                    feedback_message.desired.positions=desired.tolist()
                    feedback_message.desired.velocities=desired_velocity.tolist()
                    feedback_message.actual.positions=measured[1].tolist()
                    feedback_message.actual.velocities=measured[2].tolist()
                    feedback_message.error.positions=(desired-measured[1]).tolist()
                    feedback_message.error.velocities=(desired_velocity-measured[2]).tolist()
                    handle.publish_feedback(feedback_message)
                    if elapsed>=plan.times[-1]:
                        if (now()-measured[0]<=.2 and time.monotonic()-state['received']<=.5
                                and np.max(np.abs(measured[1]-plan.end))<=.01
                                and np.max(np.abs(measured[2]))<=.02):
                            state['target']=plan.end.copy()
                            state['plan']=None
                            handle.succeed()
                            result.error_code=result.SUCCESSFUL
                            return result
                        if elapsed>plan.times[-1]+2.:
                            raise RuntimeError('measured goal tolerance not reached')
                time.sleep(.01)
            raise RuntimeError('executor shutdown')
        except Exception as exc:
            with lock:
                fault(str(exc))
            handle.abort()
            result.error_code=result.PATH_TOLERANCE_VIOLATED
            result.error_string=str(exc)
            return result
        finally:
            with lock:
                state['active']=False

    node.create_subscription(JointState,'/restock_sim/joint_states',feedback,10,callback_group=group)
    node.create_timer(.02,tick,callback_group=group)
    server=ActionServer(node,FollowJointTrajectory,'/restock_sim/arm_controller/follow_joint_trajectory',
        execute_callback=execute,goal_callback=goal,
        cancel_callback=lambda _:CancelResponse.ACCEPT,callback_group=group)
    executor=MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        with lock:
            fault('executor stopped')
        server.destroy()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
