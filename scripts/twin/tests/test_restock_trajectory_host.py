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
