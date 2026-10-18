"""Numerical execution checks without ROS, GPU or a synthetic planner."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from twin.restock.trajectory_host import Plan
from twin.restock_manipulation.cad_tool import ARM_JOINTS


def trajectory(end=.1,duration=2.,velocity=0.):
    def point(q,v,t):
        return NS(positions=[q]*6,velocities=[v]*6,effort=[],
                  time_from_start=NS(sec=int(t),nanosec=round((t-int(t))*1e9)))
    return NS(joint_names=list(ARM_JOINTS),points=[point(0,0,0),point(end,velocity,duration)])


def test_interpolation_and_terminal_hold():
    plan=Plan(trajectory(),[0]*6,[(-1,1)]*6)
    q,v=plan.sample(1)
    assert np.allclose(q,.05)
    assert np.allclose(v,.075)
    assert np.allclose(plan.sample(5)[0],.1)
    assert np.allclose(plan.sample(5)[1],0)


def test_sample_rejects_negative_clock_but_start_is_clamped():
    # Plan.sample rejects a negative clock; tick() must clamp the reference-sample time to >=0, since
    # the first feedback after a goal is accepted can be stamped marginally before the trajectory
    # start (a sub-tick /joint_states-vs-now() race, widened by the sensing gate). A raw negative
    # crashed the whole executor (run-130). The clamped time (0) yields the trajectory's start pose.
    plan=Plan(trajectory(),[0]*6,[(-1,1)]*6)
    with pytest.raises(ValueError,match='invalid trajectory clock'):
        plan.sample(-1e-6)
    assert np.allclose(plan.sample(max(0.,-1e-6))[0],0.)   # clamped: start pose, no crash


def test_interior_speed_violation_despite_rest_endpoints():
    with pytest.raises(ValueError,match='speed'):
        Plan(trajectory(.14,1),[0]*6,[(-1,1)]*6)


def test_acceleration_violation_despite_valid_peak_speed():
    with pytest.raises(ValueError,match='acceleration'):
        Plan(trajectory(.01,.2),[0]*6,[(-1,1)]*6)


def test_interior_joint_limit_violation():
    t=trajectory(.09,2)
    t.points.insert(1,NS(positions=[.09]*6,velocities=[.1]*6,effort=[],
                        time_from_start=NS(sec=1,nanosec=0)))
    with pytest.raises(ValueError,match='joint limit'):
        Plan(t,[0]*6,[(-.1,.1)]*6,max_speed=1,max_acceleration=10)


def test_requires_measured_start_and_velocities():
    with pytest.raises(ValueError,match='measured'):
        Plan(trajectory(),[.2]*6,[(-1,1)]*6)
    t=trajectory()
    t.points[0].velocities=[]
    with pytest.raises(ValueError,match='explicit'):
        Plan(t,[0]*6,[(-1,1)]*6)


def test_reorders_joint_names():
    t=trajectory()
    t.joint_names.reverse()
    t.points[-1].positions=[.01,.02,.03,.04,.05,.06]
    assert np.allclose(Plan(t,[0]*6,[(-1,1)]*6).end,[.06,.05,.04,.03,.02,.01])


def test_feedback_watchdog_is_node_clocked_not_wall():
    # Node-clock the feedback-staleness bound: a render-paced /joint_states WALL gap (~3.5 s, far over
    # the old 0.5 s wall bound that emergency-stopped run-86) must NOT be stale while the SIM gap is
    # tiny; a real SIM-time gap, or a frozen-sim wall backstop, still is.
    from twin.restock.trajectory_host import (feedback_stale, FEEDBACK_SIM_STALE_S,
                                              FEEDBACK_WALL_BACKSTOP_S)
    # render-paced: 3.5 s wall since last feedback, but only 0.016 s sim -> fresh
    assert not feedback_stale(sim_now=10.016, feedback_sim_stamp=10.0,
                              wall_now=100.0, wall_received=96.5)
    # genuine sim-time staleness -> stale
    assert feedback_stale(sim_now=10.0+FEEDBACK_SIM_STALE_S+0.01, feedback_sim_stamp=10.0,
                          wall_now=100.0, wall_received=99.99)
    # frozen sim clock: sim gap 0 but wall gap past the backstop -> stale
    assert feedback_stale(sim_now=10.0, feedback_sim_stamp=10.0,
                          wall_now=100.0+FEEDBACK_WALL_BACKSTOP_S+1, wall_received=100.0)


def test_feedback_fresh_tolerates_subtick_clock_lead():
    # /clock and /joint_states are relayed with the same sim stamp; the node's /clock update can lag
    # the joint_states callback, so now()-stamp reads slightly negative (a delivery-order race, not
    # staleness). run-92 false-faulted on exactly 0<=now()-stamp. Tolerate the sub-tick lead; still
    # refuse a genuinely stale (too-old) frame and anything past the forward bound.
    from twin.restock.trajectory_host import (feedback_fresh, FEEDBACK_RENDER_LEAD_S,
                                             FEEDBACK_SIM_STALE_S)
    assert feedback_fresh(0.0)
    assert feedback_fresh(-FEEDBACK_RENDER_LEAD_S + 1e-6)     # small negative lead -> fresh
    assert feedback_fresh(FEEDBACK_SIM_STALE_S - 1e-6)        # within forward staleness -> fresh
    assert not feedback_fresh(-FEEDBACK_RENDER_LEAD_S - 0.01) # too far in the "future" -> refused
    assert not feedback_fresh(FEEDBACK_SIM_STALE_S + 0.01)    # genuinely stale -> refused


def test_tracking_fault_debounces_single_sample_transients():
    # Each phase trajectory produces ONE single-sample end-of-path decel transient (run-120 approach
    # 0.3524, run-121 contact 0.4044) that the next settled sample clears. The debounce must tolerate a
    # lone transient while still faulting on sustained divergence (run-117's growing lag) and on gross
    # instability immediately.
    from twin.restock.trajectory_host import (tracking_fault, TRACKING_TOL, TRACKING_DEBOUNCE,
                                             TRACKING_HARD_TOL)
    # within tolerance -> never faults, whatever the count
    assert not tracking_fault(TRACKING_TOL - 1e-6, TRACKING_DEBOUNCE + 5)
    # a lone over-tol transient (1st distinct sample) below the hard ceiling -> tolerated
    assert not tracking_fault(TRACKING_TOL + 0.02, 1)
    assert 1 < TRACKING_DEBOUNCE  # the tolerance above is meaningful: one sample is under the debounce
    # sustained over-tol divergence reaching the debounce count -> faults
    assert tracking_fault(TRACKING_TOL + 0.01, TRACKING_DEBOUNCE)
    assert not tracking_fault(TRACKING_TOL + 0.01, TRACKING_DEBOUNCE - 1)  # not yet sustained
    # gross divergence past the hard ceiling faults on a single sample regardless of debounce
    assert tracking_fault(TRACKING_HARD_TOL + 0.01, 1)


def test_executor_exit_is_honest_and_bounded_tolerant():
    # run-266: executor.spin() returned mid-move under render pacing with no tracking fault, and the
    # generic fault('executor stopped') emergency-stopped a healthy grasp while hiding the cause.
    from twin.restock.trajectory_host import executor_exit_action
    # A callback already latched the real cause -> stop, no new (masking) fault.
    assert executor_exit_action(raised_exc='RuntimeError("x")', prior_fault='measured joint feedback lost',
                                context_ok=True, transient_budget=5) == ('stop', None)
    # Clean rclpy shutdown (no exception, context closed) -> stop, no fault.
    assert executor_exit_action(raised_exc=None, prior_fault=None,
                                context_ok=False, transient_budget=5) == ('stop', None)
    # run-266: ExternalShutdownException tore the context down mid-move -> stop (NOT a futile re-spin),
    # no new masking fault; the real cause was already logged.
    assert executor_exit_action(raised_exc='ExternalShutdownException()', prior_fault=None,
                                context_ok=False, transient_budget=5) == ('stop', None)
    # Transient spin exit with budget left -> re-enter spin() instead of stopping a healthy move.
    assert executor_exit_action(raised_exc='RCLError()', prior_fault=None,
                                context_ok=True, transient_budget=3) == ('retry', None)
    assert executor_exit_action(raised_exc=None, prior_fault=None,
                                context_ok=True, transient_budget=1) == ('retry', None)
    # Budget exhausted -> stop with the HONEST cause, not 'executor stopped'.
    act, reason = executor_exit_action(raised_exc='RCLError(boom)', prior_fault=None,
                                       context_ok=True, transient_budget=0)
    assert act == 'stop' and 'RCLError(boom)' in reason and 'failed repeatedly' in reason
    act, reason = executor_exit_action(raised_exc=None, prior_fault=None,
                                       context_ok=True, transient_budget=0)
    assert act == 'stop' and 'without shutdown or tracking fault' in reason
