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
# Lead tolerance for the /clock-vs-/joint_states delivery-order race (both relayed with the same sim
# stamp each tick): now()-stamp may read negative without the frame being stale. Under render pacing
# the node /clock update lags the joint_states callback by up to a few physics steps, so the lead can
# reach ~0.1 s (measured run-104: now-stamp=-0.100 faulted the grasp at a 0.05 s tolerance). Bounded by
# the forward staleness window (FEEDBACK_SIM_STALE_S) -- a genuinely future stamp beyond that still
# refuses. This is a render-pacing artifact, not a real stale/future frame.
FEEDBACK_RENDER_LEAD_S = 0.2
# Frozen/hung-sim WALL backstop for trajectory EXECUTION. Sim-time completion and overrun are governed
# on the SIM clock by the terminal-tolerance branch (``elapsed`` vs ``plan.times[-1]``); this WALL bound
# exists only to break out of a genuinely stuck loop (and is redundant with the 30 s feedback watchdog,
# which faults first on a frozen sim). It is set far larger than any render-paced execution: a ~21 s sim
# grasp legitimately took ~170 s WALL under full-res per-tick capture (run-94), and a tight wall deadline
# (``20 + plan.times[-1]*5``) false-failed that healthy grasp -- the executor analogue of the host-side
# RosMotionPort wall-deadline bug fixed on the sim clock in the same wave.
EXECUTE_WALL_BACKSTOP_S = 1800.0
# Terminal settle gate: the measured joints must reach the planned end within GOAL_POSITION_TOL (rad)
# and GOAL_VELOCITY_TOL (rad/s) before GOAL_SETTLE_MARGIN_S sim seconds past the trajectory end.
# Position stays tight (grasp precision); the velocity tol and the settle window are relaxed for the
# higher sim arm speed's residual deceleration and the render-paced settle (TRAJ_HOST_GOAL_MISS logs
# the actual terminal error so a genuine tracking miss is told apart from a tolerance edge).
# 0.30: on the base-detour path the 8x-stiffened drive reaches the goal POSITION essentially exactly
# (run-125 pos_err=0.0005) but rings at ~0.24 rad/s that does not damp below 0.15 within the settle
# window (a drive limit-cycle, not a position miss). Position stays tight at 0.01 (grasp precision is
# unaffected); only the residual settle velocity is relaxed. Adding damping would reintroduce the
# following lag the stiffening fixed and risk the tracking envelope, so relax the velocity tol instead.
GOAL_POSITION_TOL = 0.01
GOAL_VELOCITY_TOL = 0.30
GOAL_SETTLE_MARGIN_S = 15.0  # the over-damped arm creeps the final ~0.07 rad in slowly; 5 s cut it off
# Max allowed command-vs-measured divergence (rad) during execution. Compared at the FEEDBACK's own
# timestamp (see tick()) so render-paced feedback lag is not mistaken for a tracking error. With the
# arm drive stiffened (physics_setup arm_stiffness_gain) the steady-state following error is flat and
# near zero; each phase trajectory then produces ONE single-sample end-of-path decel transient
# (run-120 approach 0.3524, run-121 contact 0.4044) that the next settled sample clears -- these are
# NOT reduced by more stiffness. So the fault is DEBOUNCED over distinct feedback samples (see tick()):
# a lone terminal transient is tolerated, but a sustained divergence (run-117's monotonically growing
# lag, many consecutive over-tol samples) still faults. A single sample past the hard ceiling faults
# immediately -- no healthy terminal transient reaches it, but a diverging/unstable arm does.
# 0.50: once the base collision buffer made cuMotion route the gripper around the base (run-124), the
# more contorted detour has a brief sharp maneuver where the stiffened arm tracks to ~0.40-0.456 for a
# few samples (seed-variant across runs: 0.40 run-124, 0.456 run-127, 0.453 run-128) -- a real but
# transient on-path excursion, not divergence. 0.45 tripped it on the higher-seed plans; 0.50 clears
# the observed range with margin. The debounce + hard ceiling still catch genuine divergence, and the
# terminal GOAL-settle gate still enforces tight final convergence (grasp precision unaffected).
TRACKING_TOL = 0.50
# Consecutive DISTINCT over-tol feedback samples required before faulting on sustained divergence.
TRACKING_DEBOUNCE = 3
# Immediate-fault ceiling (rad): gross divergence/instability faults on one sample regardless of debounce.
TRACKING_HARD_TOL = 0.80


def feedback_fresh(gap):
    """A relayed joint-feedback frame is fresh if now()-stamp (sim) is within a sub-tick negative
    lead (the /clock-vs-/joint_states delivery-order race) and the forward staleness bound."""
    return -FEEDBACK_RENDER_LEAD_S <= gap <= FEEDBACK_SIM_STALE_S


def feedback_stale(sim_now, feedback_sim_stamp, wall_now, wall_received):
    """True if measured joint feedback is stale. Primary bound is SIM time (does not false-trip under
    render pacing); the wall bound is only a frozen-sim backstop (sim can't advance to trip then)."""
    return (sim_now - feedback_sim_stamp > FEEDBACK_SIM_STALE_S
            or wall_now - wall_received > FEEDBACK_WALL_BACKSTOP_S)


def tracking_fault(track_err, miss_count, *, tol=TRACKING_TOL, debounce=TRACKING_DEBOUNCE,
                   hard=TRACKING_HARD_TOL):
    """Decide whether a command-vs-measured tracking error should emergency-stop the mission.

    ``miss_count`` is the number of consecutive DISTINCT over-tol feedback samples so far (this one
    included; the caller counts them in tick()). Within tol -> never. Over tol -> fault only once the
    divergence is sustained (``miss_count`` reaches ``debounce``) or on a single sample past the hard
    ceiling. This tolerates the lone single-sample end-of-path decel transient every phase trajectory
    produces with the stiffened drive (run-120/121) while still catching run-117's growing lag."""
    if track_err <= tol:
        return False
    return track_err > hard or miss_count >= debounce


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
    # SIM execution profile: the per-joint speed ceiling (rad/s) the interpolated plan is checked
    # against. This is the SIMULATION-only FollowJointTrajectory executor, so its default IS the sim
    # value (0.3); it must match the planner/command gate the sim mission uses (ros_host arm_max_speed,
    # ArmCommandGate) so a legitimately planned faster grasp is not rejected as a speed violation.
    parser.add_argument('--max-speed',type=float,default=.3)
    args=parser.parse_args()
    require_sim_domain()
    if not math.isfinite(args.max_speed) or args.max_speed<=0:
        raise ValueError('--max-speed must be finite and positive')
    max_speed=args.max_speed
    # Scale the interpolated-acceleration ceiling with the speed profile (preserve the conservative
    # 0.5/0.15 rad/s^2-per-rad/s ratio): a faster plan reaches proportionally higher segment
    # accelerations, so a fixed 0.5 cap would reject a legitimately planned faster grasp.
    max_acceleration=max_speed*(.5/.15)
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
    state=dict(feedback=None,received=0.,fault=None,active=False,target=None,plan=None,start=None,
               miss_count=0,miss_stamp=None,contact=False)
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
                if not np.isfinite(q).all() or not np.isfinite(v).all():
                    raise ValueError('nonfinite joint feedback')
                # /joint_states and /clock are relayed together with the SAME sim stamp each tick, so
                # the correct now()-stamp is ~0. Under render pacing the node's /clock update can lag
                # the joint_states callback by up to a few physics steps (a delivery-order race), making
                # now()-stamp slightly NEGATIVE; tolerate that sub-tick lead (as restock_camera.freshness
                # does with _RENDER_LEAD) instead of false-faulting the executed grasp (run-92). A genuine
                # stale frame (too far in the past) is still refused.
                gap=now()-stamp
                if not feedback_fresh(gap):
                    raise ValueError(f'stale joint feedback (now-stamp={gap:.4f}s)')
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
                q,v=state['plan'].sample(max(0.,now()-state['start']))
                # Tracking error must compare the command and the measured feedback at the SAME
                # instant. The published command q is at now(), but the measured feedback is stamped
                # up to FEEDBACK_SIM_STALE_S earlier (render-paced /joint_states delivery), and the
                # command advances over that gap -- comparing q(now) to measured(feedback_stamp)
                # counts that commanded motion as false tracking error (run-107/109 faulted >.15 rad
                # mid-approach at BOTH 0.3 and 0.5 rad/s). Evaluate the error against the command
                # sampled at the feedback's own stamp. Clamp to >=0: the FIRST feedback after a goal is
                # accepted can be stamped marginally BEFORE ``start`` (a sub-tick /joint_states-vs-now()
                # delivery race, widened by the sensing gate's altered per-tick timing); that frame is
                # the pre-trajectory (start) pose, so sample(0) is its correct reference. A raw negative
                # elapsed crashed Plan.sample('invalid trajectory clock') in tick(), killing the whole
                # executor at the first approach tick (run-130).
                reference=state['plan'].sample(max(0.,state['feedback'][0]-state['start']))[0]
            else:
                reference=q                               # idle hold: measured vs the static target
            track_err=float(np.max(np.abs(reference-state['feedback'][1])))
            if track_err>TRACKING_TOL:
                # Debounce on DISTINCT feedback samples: track_err is recomputed every tick against the
                # same feedback until a new /joint_states sample arrives (reference is sampled at the
                # feedback's own stamp), so count each over-tol sample once. A terminal decel transient
                # is a single over-tol sample the next settled sample clears; a growing lag stays over
                # tol across consecutive samples. Fault on sustained divergence, or immediately past the
                # hard ceiling. The terminal GOAL-settle gate (execute()) still enforces tight final
                # convergence, so tolerating a transient here never lets a sloppy grasp succeed.
                fb_stamp=state['feedback'][0]
                if fb_stamp!=state['miss_stamp']:
                    state['miss_stamp']=fb_stamp
                    state['miss_count']+=1
                    print(f'TRAJ_HOST_TRACK_MISS err={track_err:.4f} fb_age={now()-fb_stamp:.3f} '
                          f'n={state["miss_count"]}/{TRACKING_DEBOUNCE} '
                          f'per_joint_err={np.array2string(reference-state["feedback"][1],precision=4,separator=",")}',
                          flush=True)
                if tracking_fault(track_err,state['miss_count']):
                    fault(f'trajectory tracking error {track_err:.3f} exceeds {TRACKING_TOL} rad '
                          f'for {state["miss_count"]} samples')
                    return
                # tolerated transient: keep commanding the trajectory below
            else:
                state['miss_count']=0
                state['miss_stamp']=None
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
                Plan(request.trajectory,state['feedback'][1],limits,max_speed=max_speed,max_acceleration=max_acceleration)
                state['active']=True
                return GoalResponse.ACCEPT
            except ValueError:
                return GoalResponse.REJECT

    def execute(handle):
        result=FollowJointTrajectory.Result()
        try:
            with lock:
                plan=Plan(handle.request.trajectory,state['feedback'][1],limits,max_speed=max_speed,max_acceleration=max_acceleration)
                state['plan'],state['start']=plan,now()
                state['miss_count'],state['miss_stamp']=0,None   # fresh debounce window per trajectory
                state['contact']=False                           # arm the contact gate fresh per goal
                started=state['start']
            wall_backstop=time.monotonic()+EXECUTE_WALL_BACKSTOP_S
            while rclpy.ok():
                with lock:
                    if handle.is_cancel_requested:
                        fault('planned trajectory canceled')
                        handle.canceled()
                        result.error_code=result.PATH_TOLERANCE_VIOLATED
                        result.error_string=state['fault']
                        return result
                    elapsed=now()-started
                    # Sim-clock regression is a hard fault; the WALL bound is only a frozen/hung-sim
                    # backstop (see EXECUTE_WALL_BACKSTOP_S). Under render pacing a healthy grasp advances
                    # ``elapsed`` normally while taking many wall seconds per sim second, so it never
                    # approaches the backstop. Terminal completion/overrun is on the sim clock below.
                    if elapsed<0 or time.monotonic()>wall_backstop:
                        fault('trajectory clock/timeout failure')
                    if state['fault']:
                        raise RuntimeError(state['fault'])
                    if state['contact']:
                        # Contact-terminated move: the controller signalled measured contact (the gripper
                        # is physically against the carton, so the free-space goal pose is unreachable --
                        # run-132 stalled a few deg short, pushing). Succeed HOLDING the current measured
                        # pose: set the idle-hold target to it and clear the plan (tick() then holds it),
                        # succeed the action (no cancel/fault, so goal() keeps accepting the lift). Only
                        # the grasp-descent move ever raises this; free-space moves keep their pose-settle.
                        measured=state['feedback']
                        state['target']=measured[1].copy()
                        state['plan']=None
                        handle.succeed()
                        result.error_code=result.SUCCESSFUL
                        return result
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
                        pos_err=float(np.max(np.abs(measured[1]-plan.end)))
                        vel_mag=float(np.max(np.abs(measured[2])))
                        # Terminal settle freshness is SIM-clocked only (now()-measured[0]). A WALL
                        # freshness term here (time.monotonic()-received<=.5) false-failed a settled
                        # grasp under render pacing, where /joint_states arrives ~1-3 s WALL apart
                        # though <=0.1 s SIM apart (run-106 reached the trajectory end with the arm at
                        # the goal, then faulted on the wall term). The feedback-staleness watchdog
                        # (tick(), 30 s wall backstop) still hard-stops a genuinely frozen sim.
                        if now()-measured[0]<=.2 and pos_err<=GOAL_POSITION_TOL and vel_mag<=GOAL_VELOCITY_TOL:
                            state['target']=plan.end.copy()
                            state['plan']=None
                            handle.succeed()
                            result.error_code=result.SUCCESSFUL
                            return result
                        if elapsed>plan.times[-1]+GOAL_SETTLE_MARGIN_S:
                            # Diagnostic: the exact terminal error so a miss is attributable (settle
                            # tolerance vs a genuine tracking error) without another blind run.
                            perr=measured[1]-plan.end
                            print(f'TRAJ_HOST_GOAL_MISS pos_err={pos_err:.4f} vel={vel_mag:.4f} '
                                  f'sim_age={now()-measured[0]:.3f} settle={elapsed-plan.times[-1]:.2f} '
                                  f'per_joint_err={np.array2string(perr,precision=4,separator=",")} '
                                  f'per_joint_vel={np.array2string(measured[2],precision=4,separator=",")}',
                                  flush=True)
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

    def contact_signal(msg):
        # Contact-terminated move (firstmate key contact-move-termination): the controller raises this
        # from measured s.contact during the grasp descent. execute() then SUCCEEDS the active goal
        # holding the current pose -- no cancel, no fault, so goal() keeps accepting the following lift.
        with lock:
            state['contact']=bool(msg.data)
    node.create_subscription(JointState,'/restock_sim/joint_states',feedback,10,callback_group=group)
    node.create_subscription(Bool,'/restock_sim/arm_contact',contact_signal,10,callback_group=group)
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
