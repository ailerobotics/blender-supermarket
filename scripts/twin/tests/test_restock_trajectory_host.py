"""Numerical execution checks without ROS, GPU or a synthetic planner."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from twin.restock.trajectory_host import (
    Plan, goal_reached, classify_goal_miss, GOAL_POSITION_TOL, GOAL_VELOCITY_TOL)
from twin.restock_manipulation.cad_tool import ARM_JOINTS


def test_goal_reached_requires_fresh_settled_within_tolerance():
    # Settled at the goal: fresh sim age, zero error, zero velocity -> reached.
    assert goal_reached(0.0, 0.0, 0.0, GOAL_POSITION_TOL)
    # A per-joint miss at the tight tolerance is NOT reached: execff's shoulder_lift ended ~0.10 rad short,
    # which must keep failing the gate. This tolerance is NOT loosened for the loaded pick -- that miss is a
    # physical block (the scripted pick jams short of the carton), named by classify_goal_miss, not a
    # gravity sag a wider tolerance or feedforward could close (sim-robust-executor-shoulder-drive-v1).
    assert not goal_reached(0.0, 0.0981, 0.0, GOAL_POSITION_TOL)
    assert goal_reached(0.0, GOAL_POSITION_TOL, 0.0, GOAL_POSITION_TOL)          # exactly at tol: reached
    assert not goal_reached(0.0, GOAL_POSITION_TOL + 1e-6, 0.0, GOAL_POSITION_TOL)


def test_goal_reached_rejects_stale_or_moving():
    assert not goal_reached(0.5, 0.0, 0.0, GOAL_POSITION_TOL)                     # stale sim age (>0.2)
    assert not goal_reached(0.0, 0.0, GOAL_VELOCITY_TOL + 0.01, GOAL_POSITION_TOL)  # still moving
    # A controller-latched wider tolerance (the loaded-place side channel) is honoured as the pos_tol arg.
    assert goal_reached(0.0, 0.05, 0.0, 0.08)


# ---- classify_goal_miss (sim-robust-executor-shoulder-drive-v1): name the physical-block pick miss ----
GENERIC = 'measured goal tolerance not reached'
# The execff terminal signature, arm order [pan, lift, elbow, w1, w2, w3]: only shoulder_lift (index 1)
# misses, frozen 0.10 rad short while the other five reached the goal (~1e-5 rad).
EXECFF_ERR = [6.3e-5, 0.1002, 3.6e-5, -1.0e-5, -4.3e-5, -3.5e-5]
EXECFF_VEL = [0.0079, -0.001, 0.0142, -0.021, -0.0117, 0.0024]


def test_classify_names_the_physical_block_from_the_joint_signature():
    reason = classify_goal_miss(EXECFF_ERR, EXECFF_VEL, GOAL_POSITION_TOL)
    assert reason != GENERIC
    assert 'pick physically blocked' in reason
    assert 'ur_shoulder_lift_joint' in reason        # the single frozen joint is named
    assert '0.1002' in reason                          # the measured shortfall is reported
    assert 'not gravity sag' in reason                 # the correction is stated explicitly
    # No surface feed => the reason says the gripper->surface reading is unavailable, not a fabricated gap.
    assert 'unavailable' in reason


def test_classify_enriches_with_surface_gap_and_flush_when_fed():
    reason = classify_goal_miss(EXECFF_ERR, EXECFF_VEL, GOAL_POSITION_TOL, surface=(39.8, 5.3))
    assert 'pick physically blocked' in reason
    assert '39.8 mm' in reason and '5.3 deg' in reason   # the measured gap + flush angle appear
    assert 'unavailable' not in reason


def test_classify_true_multijoint_miss_stays_generic():
    # Two joints past tolerance (a genuine tracking error, not a single blocked joint) -> generic.
    err = [0.05, 0.1002, 3.6e-5, -1.0e-5, -4.3e-5, -3.5e-5]
    assert classify_goal_miss(err, EXECFF_VEL, GOAL_POSITION_TOL) == GENERIC


def test_classify_marginal_single_joint_miss_stays_generic():
    # One joint just past the tight tol but well under the gross-miss floor (settle/tolerance edge) -> generic.
    err = [6.3e-5, 0.015, 3.6e-5, -1.0e-5, -4.3e-5, -3.5e-5]
    assert classify_goal_miss(err, [0.0] * 6, GOAL_POSITION_TOL) == GENERIC


def test_classify_single_far_joint_still_moving_stays_generic():
    # Far from goal but NOT frozen (still decelerating) -> not yet a block; stay generic.
    vel = [0.0, 0.4, 0.0, 0.0, 0.0, 0.0]
    assert classify_goal_miss(EXECFF_ERR, vel, GOAL_POSITION_TOL) == GENERIC


def test_classify_honours_a_widened_pos_tol():
    # Under a controller-latched wider tolerance the same small errors are all within tol -> no miss class.
    assert classify_goal_miss(EXECFF_ERR, EXECFF_VEL, 0.2) == GENERIC


def test_classify_survives_garbled_surface_without_crashing():
    # A malformed surface reading must not crash the fault path, nor suppress the joint-only naming.
    for bad in [(float('nan'), 2.0), ('x',), None, (1.0,)]:
        reason = classify_goal_miss(EXECFF_ERR, EXECFF_VEL, GOAL_POSITION_TOL, surface=bad)
        assert 'pick physically blocked' in reason


def _ring(t_end, *, amp0, omega, sigma, dt=0.1):
    """A residual-oscillation terminal-sample stream ``(t, pos_err, vel_mag)`` for the settle gate.

    A single ringing joint at ``pos(t)=A(t) sin(omega t)``, ``A(t)=amp0 e^{-sigma t}`` (``sigma>0``
    decaying, ``0`` sustained, ``<0`` diverging); ``pos_err=|pos|`` and ``vel_mag=|pos'|``. A pure
    sinusoid NEVER has |pos| and |vel| both near zero at one instant (pos~0 at a zero crossing is where
    |vel| peaks), so the instantaneous both-small terminal gate cannot fire for such a ring -- the
    coldwin2 ur_wrist_3 case the envelope gate exists to resolve."""
    out, t = [], 0.0
    while t <= t_end + 1e-9:
        A = amp0 * np.exp(-sigma * t)
        pos = abs(A * np.sin(omega * t))
        vel = abs(A * (omega * np.cos(omega * t) - sigma * np.sin(omega * t)))
        out.append((t, pos, vel))
        t += dt
    return out


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


def _transfer1_plan():
    """A Plan shaped like transfer1's pick: ur_wrist_1 (joint 3) swings from the -0.7 start to a far
    value, with plan.sample(11) ~ the logged reference at t=11 (ref_q[3]=0.6066). A symmetric zero-end-
    velocity cubic over [0, 22] has its half-time sample at the (start+end)/2 midpoint, so end = 2*ref11
    - start reproduces the logged reference at t=11. Feasible inside the executor's 0.3 rad/s ceiling."""
    start = np.array([-0.0, -2.5001, 1.7002, -0.7, -1.5708, -0.0])
    ref11 = np.array([0.14, -2.6979, 1.8492, 0.6037, -0.8621, 0.8493])   # transfer1 ref_q at t~=10.986
    end = 2 * ref11 - start
    def point(q, t):
        return NS(positions=list(q), velocities=[0.0] * 6, effort=[],
                  time_from_start=NS(sec=int(t), nanosec=round((t - int(t)) * 1e9)))
    traj = NS(joint_names=list(ARM_JOINTS), points=[point(start, 0.0), point(end, 22.0)])
    return Plan(traj, start, [(-2 * np.pi, 2 * np.pi)] * 6, max_speed=.3, max_acceleration=1.0), start, ref11


def test_transfer1_first_reference_is_the_start_pose_not_the_clock_burst():
    # transfer1 (2026-10-22, RESTOCK_SCRIPTED_TRANSFER_ARM=executor): the executor pick faulted at the
    # FIRST evaluated tracking sample with meas_q EXACTLY the plan's t=0 start pose and ref_q at plan
    # t~=11 (ur_wrist_1 0.6066 vs -0.7 => 1.31 > 1.3 rad). fb_age=0.000 proved now() and the feedback
    # stamp are the SAME clock, so this was NOT a stale frame: start was anchored to now() at execute
    # install, ~11 sim s BEFORE the first feedback the tracker evaluated (the mission burst-advanced the
    # sim between accept and first feedback), and feedback_stamp-start charged that 11 s as arm progress.
    from twin.restock.trajectory_host import (anchored_start, reference_time, tracking_fault,
                                             TRACKING_TOL)
    plan, start, ref11 = _transfer1_plan()
    assert np.allclose(plan.sample(0.0)[0], start)              # t=0 reference is the arm's start pose
    install_now = 100.0                                         # node /clock latched at execute install
    burst = 11.0                                                # sim seconds the mission advanced before
    first_fb_stamp = install_now + burst                       #   the first feedback the tracker sees
    measured = start                                           # arm physically still parked at the start

    # OLD behaviour (start = now() at install): reference is sampled 11 s into the plan while the arm is
    # at t=0 => the logged false fault, reproduced here.
    old_ref = plan.sample(reference_time(first_fb_stamp, install_now))[0]
    assert np.isclose(reference_time(first_fb_stamp, install_now), 11.0)
    assert np.allclose(old_ref, ref11, atol=5e-3)
    old_err = float(np.max(np.abs(old_ref - measured)))
    assert old_err > TRACKING_TOL                              # > 1.3 rad: the transfer1 TRAJ_HOST_FAULT
    assert tracking_fault(old_err, 3)

    # FIX (start anchored to the first post-install feedback stamp): the first evaluated reference is
    # plan.sample(0) = the start pose, so the error against the (correct, parked) arm is ~0, no fault.
    start_anchor = anchored_start(True, None, first_fb_stamp)
    assert start_anchor == first_fb_stamp
    new_ref = plan.sample(reference_time(first_fb_stamp, start_anchor))[0]
    assert np.allclose(new_ref, start)
    new_err = float(np.max(np.abs(new_ref - measured)))
    assert new_err < 1e-9 and not tracking_fault(new_err, 3)


def test_two_clock_offset_is_absorbed_by_feedback_anchor():
    # The staged-start two-sim-clock offset (staged_start.py: /clock vs the mission packet clock, 1.55 s
    # on start_at=transfer) and any render-pacing burst mean now() at execute install can lead/lag the
    # measured-feedback clock by an ARBITRARY amount. Anchoring the trajectory origin to the first
    # post-install feedback stamp makes the first evaluated reference plan.sample(0) = the start pose for
    # EVERY offset, so the arm path no longer depends on the two clocks agreeing.
    from twin.restock.trajectory_host import anchored_start, reference_time
    plan, start, _ = _transfer1_plan()
    for offset in (-5.0, -1.55, 0.0, 0.05, 1.55, 11.0, 200.0):
        install_now = 50.0
        first_fb_stamp = install_now + offset
        anchor = anchored_start(True, None, first_fb_stamp)
        assert anchor == first_fb_stamp                        # origin = the feedback clock, not now()
        assert reference_time(first_fb_stamp, anchor) == 0.0   # first sample is exactly t=0
        assert np.allclose(plan.sample(reference_time(first_fb_stamp, anchor))[0], start)
    # idempotent + clamp contract: once anchored it never re-stamps; a sub-tick-early feedback clamps to 0
    assert anchored_start(True, 50.0, 999.0) == 50.0           # already set -> unchanged
    assert anchored_start(False, None, 10.0) is None           # no active plan -> stays pending
    assert reference_time(49.999, 50.0) == 0.0                 # feedback a hair before start -> sample(0)
    assert reference_time(60.0, 50.0) == 10.0


def _coldlive2_collect_plan():
    """The exact 2-point collect plan from coldlive sample 2 (2026-10-23 host-stack.log TRAJ_HOST_PLAN):
    a near-stationary ~0.005 rad nudge over 2.0 s, trivially inside every executor limit. Reordered from
    the logged joint_names (elbow, shoulder_lift, shoulder_pan, wrist_1, wrist_2, wrist_3) into ARM_JOINTS.
    Zero end-velocity (the dump logs positions only); the point of the test is the CLOCK, not the shape."""
    log_names = ['ur_elbow_joint', 'ur_shoulder_lift_joint', 'ur_shoulder_pan_joint',
                 'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']
    p0 = [2.1921, -1.3433, 1.1623, -0.8487, 1.1631, 0.0]
    p1 = [2.1864, -1.3391, 1.1646, -0.8472, 1.1654, 0.0]
    order = [log_names.index(n) for n in ARM_JOINTS]
    start = np.asarray(p0, float)[order]
    def point(q, t):
        return NS(positions=list(q), velocities=[0.0] * 6, effort=[],
                  time_from_start=NS(sec=int(t), nanosec=round((t - int(t)) * 1e9)))
    traj = NS(joint_names=list(log_names), points=[point(p0, 0.0), point(p1, 2.0)])
    # measured joints are supplied in ARM_JOINTS order (the executor reorders the trajectory to match)
    return Plan(traj, start, [(-2 * np.pi, 2 * np.pi)] * 6, max_speed=.3, max_acceleration=1.0), start


def test_coldlive2_two_point_plan_is_valid():
    # The collect plan that faulted was VALID: a tiny 2-point nudge well inside the executor's limits.
    # If Plan() accepts it, the 'trajectory clock/timeout failure' was NOT a plan-validity or zero-/
    # sub-period-duration problem -- it was purely the execute-loop clock check (test below).
    plan, start = _coldlive2_collect_plan()
    assert plan.times == [0.0, 2.0]                       # 2 s, far above one control period
    assert np.allclose(plan.sample(0.0)[0], start)        # t=0 = the measured start pose
    assert np.allclose(plan.sample(2.0)[1], 0.0)          # ends at rest


def test_execute_clock_tolerates_subtick_feedback_lead_coldlive2():
    # coldlive sample 2 reproduction: start is anchored to a measured-feedback stamp (26.533) that LEADS
    # the node /clock (install 26.5167) by anchor_lag 0.017 s -- the /clock-vs-/joint_states render-pacing
    # delivery race. The next execute-loop now() still reads ~26.516, so now()-start = -0.017 < 0. The old
    # hard ``elapsed<0`` fired 'trajectory clock/timeout failure' twice on a valid plan; execute_clock must
    # clamp this sub-tick lead to the start pose (0.0) with NO fault.
    from twin.restock.trajectory_host import execute_clock, FEEDBACK_RENDER_LEAD_S
    start = 26.533
    wall_now, wall_backstop = 1000.0, 1e9               # nowhere near the wall backstop
    elapsed, fault = execute_clock(26.516 - start, wall_now, wall_backstop)
    assert fault is None
    assert elapsed == 0.0                                # held at the start pose, not a crash/fault
    # the whole tolerated band, right up to the render-lead bound
    e, f = execute_clock(-FEEDBACK_RENDER_LEAD_S + 1e-9, wall_now, wall_backstop)
    assert f is None and e == 0.0


def test_execute_clock_normal_forward_is_identity():
    # The healthy case stays byte-identical: a normal forward elapsed is returned unchanged with no fault,
    # so the tracking/terminal logic below is unaffected for every trajectory that already worked.
    from twin.restock.trajectory_host import execute_clock
    for e in (0.0, 0.005, 1.0, 2.0, 123.4):
        elapsed, fault = execute_clock(e, wall_now=10.0, wall_backstop=1e9)
        assert fault is None and elapsed == e


def test_execute_clock_faults_genuine_regression_by_name():
    # A real backward sim clock (beyond the render-lead band) is still a hard fault -- a stale/future frame
    # could never have anchored start past the lead (feedback_fresh refuses it), so anything past -lead is
    # a genuine regression, failed BY NAME (distinct from the wall timeout) so the cause stays legible.
    from twin.restock.trajectory_host import execute_clock, FEEDBACK_RENDER_LEAD_S
    elapsed, fault = execute_clock(-FEEDBACK_RENDER_LEAD_S - 0.05, wall_now=10.0, wall_backstop=1e9)
    assert elapsed is None and fault == 'trajectory clock regression'


def test_execute_clock_wall_backstop_still_times_out():
    # The frozen/hung-sim WALL backstop is unchanged: past it we still fault 'trajectory clock/timeout
    # failure' even for an otherwise-fine forward elapsed, and it takes precedence over a stale elapsed so
    # a genuinely stuck loop reports the timeout rather than the regression.
    from twin.restock.trajectory_host import execute_clock, FEEDBACK_RENDER_LEAD_S
    elapsed, fault = execute_clock(1.0, wall_now=100.0, wall_backstop=99.0)
    assert elapsed is None and fault == 'trajectory clock/timeout failure'
    # wall timeout wins over a simultaneously-negative elapsed
    elapsed, fault = execute_clock(-FEEDBACK_RENDER_LEAD_S - 1.0, wall_now=100.0, wall_backstop=99.0)
    assert elapsed is None and fault == 'trajectory clock/timeout failure'


def test_execute_clock_stalled_sim_frame_holds_without_fault():
    # A single stalled sim frame under the live_pose feeds (now() does not advance for a tick, so elapsed
    # sits flat at the anchored start or a hair before it) must HOLD the start pose, not fault -- only a
    # genuinely frozen sim that runs past the wall backstop fails. Mirrors feedback_fresh's stall handling.
    from twin.restock.trajectory_host import execute_clock
    # frame stalled exactly at the anchor (elapsed 0) and a sub-tick before it: both hold, no fault
    for e in (0.0, -0.01, -0.1):
        elapsed, fault = execute_clock(e, wall_now=50.0, wall_backstop=1e9)
        assert fault is None and elapsed == max(0.0, e)


def test_execute_clock_lead_bound_matches_feedback_freshness():
    # Invariant the whole fix rests on: the clamp bound is EXACTLY feedback_fresh's render lead, so a
    # feedback frame leading now() by more than the bound is refused upstream (never anchors start) and a
    # within-bound negative elapsed can only be the delivery race -- the clamp can never mask a regression.
    from twin.restock.trajectory_host import execute_clock, feedback_fresh, FEEDBACK_RENDER_LEAD_S
    lead = FEEDBACK_RENDER_LEAD_S
    # a frame that would produce elapsed just past -lead is exactly the frame feedback_fresh rejects
    assert not feedback_fresh(-lead - 0.01)
    assert execute_clock(-lead - 0.01, 0.0, 1e9)[1] == 'trajectory clock regression'
    # a frame within the lead is fresh upstream and clamped (not faulted) here
    assert feedback_fresh(-lead + 1e-9)
    assert execute_clock(-lead + 1e-9, 0.0, 1e9) == (0.0, None)


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


def test_reject_reason_names_each_race_and_accepts_when_clear():
    # sim-robust-pick-plan-rejection-v1: the goal callback must never reject SILENTLY. reject_reason
    # names which of the three startup/timing conditions fired (so seconddeath's feedback-None race is
    # attributable in host-stack.log instead of surfacing as 'collision planning or execution failed').
    from twin.restock.trajectory_host import reject_reason
    # all clear -> accept (None)
    assert reject_reason(None, False, True) is None
    # the actual seconddeath cause: first goal arrived before the first /joint_states sample
    r = reject_reason(None, False, False)
    assert r is not None and 'feedback' in r and 'race' in r
    # a previous goal still executing
    r = reject_reason(None, True, True)
    assert r is not None and 'previous goal' in r
    # a latched fault wins over everything and carries the cause
    r = reject_reason('trajectory tracking error 1.6 exceeds 1.3 rad', True, False)
    assert r is not None and 'fault latched' in r and 'tracking error' in r


def test_active_is_stale_only_when_handle_terminated():
    # (c) a NEW goal must not be rejected on a dead active flag (a prior handle that aborted/canceled
    # without execute()'s finally clearing active), but a LIVE goal's flag must be respected.
    from twin.restock.trajectory_host import active_is_stale
    assert active_is_stale(True, False) is True      # active set, handle terminated -> stale, clear it
    assert active_is_stale(True, True) is False       # active set, handle live -> genuine, keep it
    assert active_is_stale(False, False) is False     # not active -> nothing to clear
    assert active_is_stale(False, True) is False


def test_reset_clears_rearms_latched_fault_never_a_live_goal():
    # (d) op-start re-arm (PR 210 parity): a reset clears a LATCHED fault + a STALE active flag, but
    # NEVER a live execution, and is a no-op when there is nothing to clear (byte-identical normal path).
    from twin.restock.trajectory_host import reset_clears
    assert reset_clears(None, False, False) == (False, False)          # fault-free, idle -> no-op
    assert reset_clears('feedback lost', False, False) == (True, False)  # re-arm a latched fault
    assert reset_clears('feedback lost', True, False) == (True, True)    # + clear a stale active flag
    assert reset_clears(None, True, True) == (False, False)              # live goal, no fault -> untouched
    assert reset_clears('x', True, True) == (True, False)                # fault cleared; live active kept


def test_abort_active_clears_forces_only_a_live_goal_and_clears_fault():
    # sim-robust-movegroup-double-death-v1 (gap 2): a mid-op move_group respawn+reseed. move_group crashes
    # mid-execution leaving its FollowJointTrajectory goal LIVE on the executor, so the reseeded resend would
    # be rejected 'a previous goal is still executing' (coldwin4 death 2). abort_active REQUESTS the live
    # goal to self-terminate (only when active AND its handle is live) and clears a latched fault so the
    # resend is accepted -- where reset_clears (PR 229 op-start) deliberately never touches a live goal.
    from twin.restock.trajectory_host import abort_active_clears, reset_clears
    assert abort_active_clears(True, True, None) == (True, False)    # live goal -> force-abort it
    assert abort_active_clears(True, True, 'x') == (True, True)       # + clear a latched fault
    assert abort_active_clears(True, False, None) == (False, False)  # stale active -> active_is_stale owns it
    assert abort_active_clears(False, False, 'x') == (False, True)   # idle + fault -> just re-arm the fault
    assert abort_active_clears(False, False, None) == (False, False)  # idle, fault-free -> no-op
    # Decisive contrast with the op-start reset: on a LIVE goal, reset_clears refuses to clear it, abort does.
    assert reset_clears('x', True, True)[1] is False                 # op-start reset keeps a live goal
    assert abort_active_clears(True, True, 'x')[0] is True            # abort_active terminates it


def test_handle_live_tolerates_none_and_duck_types():
    from twin.restock.trajectory_host import _handle_live
    from types import SimpleNamespace as NS
    assert _handle_live(None) is False
    assert _handle_live(NS(is_active=True)) is True
    assert _handle_live(NS(is_active=False)) is False

    class _Raises:
        @property
        def is_active(self):
            raise RuntimeError('torn down')
    assert _handle_live(_Raises()) is False   # a torn-down handle reports not-live, never raises


def test_first_feedback_grace_default_and_env_override(monkeypatch):
    from twin.restock import trajectory_host
    import importlib
    monkeypatch.delenv('RESTOCK_FIRST_FEEDBACK_GRACE_S', raising=False)
    assert trajectory_host.first_feedback_grace_s() == 8.0   # bounded default waits out the race
    monkeypatch.setenv('RESTOCK_FIRST_FEEDBACK_GRACE_S', '0')
    assert trajectory_host.first_feedback_grace_s() == 0.0   # 0 => legacy immediate decision
    monkeypatch.setenv('RESTOCK_FIRST_FEEDBACK_GRACE_S', '999')
    assert trajectory_host.first_feedback_grace_s() == 60.0  # clamped sane
    monkeypatch.setenv('RESTOCK_FIRST_FEEDBACK_GRACE_S', '-5')
    assert trajectory_host.first_feedback_grace_s() == 0.0


def test_settle_envelope_accepts_bounded_decaying_oscillation_by_name():
    # coldwin2 (2026-10-23): the welded-carton collect-place left ur_wrist_3 ringing at 0.72 rad/s at a
    # 3.4 mm-accurate pose for the whole 30 s settle window; the instantaneous both-small gate can NEVER
    # fire for such a ring (pos~0 only at the velocity peak), so PATH_TOLERANCE_VIOLATED aborted a place
    # whose carton was on the deck within tolerance. A DECAYING ring whose amplitude falls BELOW the
    # position tolerance and whose velocity envelope decays IS a successful place -- accepted by name.
    from twin.restock.trajectory_host import settle_decision
    # amplitude starts above tol (orbit too big for the box early) and decays well below it; vel peak
    # starts ~1.0 rad/s (never both-small with pos) and decays.
    s = _ring(10.0, amp0=0.20, omega=5.0, sigma=0.35)
    verdict, reason = settle_decision(s, pos_tol=0.12, vel_tol=0.70, final=True)
    assert verdict == 'settled', reason
    assert 'decaying' in reason and 'tight pose' in reason


def test_settle_envelope_fails_sustained_limit_cycle_by_name():
    # A non-decaying (sustained) ring is NOT a settle: effort-saturation limit cycle that never damps.
    # With amplitude above the position tolerance it is also never within the pose box -- fail by name,
    # never silently accepted. (This is coldwin2 IF its wrist_3 ring did not decay within the window.)
    from twin.restock.trajectory_host import settle_decision
    s = _ring(30.0, amp0=0.20, omega=5.0, sigma=0.0)   # constant amplitude 0.20 > 0.12 tol
    verdict, reason = settle_decision(s, pos_tol=0.12, vel_tol=0.70, final=True)
    assert verdict == 'miss'
    assert 'not bounded' in reason or 'sustained' in reason


def test_settle_envelope_fails_bounded_but_sustained_by_name():
    # A ring that stays WITHIN the position box but whose velocity envelope never decays is still a
    # limit cycle, not a settle -- named "sustained", not accepted. (Amplitude 0.08 < 0.12 tol, flat.)
    from twin.restock.trajectory_host import settle_decision
    s = _ring(30.0, amp0=0.08, omega=9.0, sigma=0.0)
    verdict, reason = settle_decision(s, pos_tol=0.12, vel_tol=0.70, final=True)
    assert verdict == 'miss' and 'sustained' in reason


def test_settle_envelope_fails_diverging_by_name_and_early():
    # A growing oscillation (instability) must fail by name, and may fail BEFORE the settle budget ends
    # (final=False) rather than wait the full window.
    from twin.restock.trajectory_host import settle_decision
    s = _ring(12.0, amp0=0.05, omega=6.0, sigma=-0.15)   # amplitude grows
    verdict, reason = settle_decision(s, pos_tol=0.12, vel_tol=0.70, final=False)
    assert verdict == 'diverging' and 'diverging' in reason


def test_settle_envelope_is_pending_until_enough_window():
    # The gate must not settle (or fault) on a lucky early dip: with too few samples / too short a
    # window it returns 'pending' and keeps observing.
    from twin.restock.trajectory_host import settle_decision, SETTLE_MIN_SAMPLES, SETTLE_MIN_WINDOW_S
    short = _ring(SETTLE_MIN_WINDOW_S * 0.3, amp0=0.20, omega=5.0, sigma=0.35, dt=0.1)
    assert len(short) < SETTLE_MIN_SAMPLES or (short[-1][0] - short[0][0]) < SETTLE_MIN_WINDOW_S
    assert settle_decision(short, pos_tol=0.12, vel_tol=0.70, final=False) == ('pending', None)


def test_settle_envelope_matches_coldwin2_orbit_never_fits_box():
    # Reconstruct the coldwin2 phase-space fact: while the orbit amplitude stays ABOVE the position tol
    # (wrist_3 swung > the 0.12 rad box for the whole 30 s window -- the only way the instantaneous gate
    # never fired over ~300 samples), NO single sample has both pos_err<=tol AND vel_mag<=vel_tol, so the
    # instantaneous gate is structurally unable to pass OR fail it with a reason -- the envelope gate is
    # the only path that classifies it (here: a sustained limit cycle, failed by name, not silently).
    from twin.restock.trajectory_host import settle_decision
    pos_tol, vel_tol = 0.12, 0.70
    s = _ring(30.0, amp0=0.20, omega=5.0, sigma=0.0)   # sustained, amplitude 0.20 > 0.12 the whole time
    assert not any(p <= pos_tol and v <= vel_tol for _, p, v in s), \
        'instantaneous gate should be unable to fire for an above-tol ring'
    verdict, reason = settle_decision(s, pos_tol=pos_tol, vel_tol=vel_tol, final=True)
    assert verdict == 'miss' and ('not bounded' in reason or 'sustained' in reason)

    # And once such a ring DECAYS below the box, the envelope accepts it as a settled place by name.
    decaying = _ring(12.0, amp0=0.20, omega=5.0, sigma=0.35)
    assert settle_decision(decaying, pos_tol=pos_tol, vel_tol=vel_tol, final=True)[0] == 'settled'


def test_settle_accept_vel_never_below_instantaneous_tol():
    # The envelope accept ceiling is clamped at/above the instantaneous velocity tol, so the envelope
    # can never settle a residual the instantaneous gate would reject on a tighter velocity bound.
    from twin.restock.trajectory_host import SETTLE_ACCEPT_VEL, GOAL_VELOCITY_TOL
    assert SETTLE_ACCEPT_VEL >= GOAL_VELOCITY_TOL


def test_settle_position_tolerance_is_not_loosened():
    # Explicit guard on the deliverable constraint: a decaying ring whose amplitude never falls below
    # the position tolerance is NOT accepted (position is never loosened), even though it is decaying.
    from twin.restock.trajectory_host import settle_decision
    # decays slowly; stays above 0.12 for the whole 8 s window (amp 0.20 -> 0.20*e^{-0.05*8}=0.134)
    s = _ring(8.0, amp0=0.20, omega=5.0, sigma=0.05)
    assert max(p for _, p, _ in s[len(s)//2:]) > 0.12   # recent half still swings past the 0.12 box
    verdict, reason = settle_decision(s, pos_tol=0.12, vel_tol=0.70, final=True)
    assert verdict == 'miss' and 'not bounded' in reason


def test_settle_envelope_switch_is_backward_compatible():
    # RESTOCK_SETTLE_ENVELOPE defaults ON; =0/false/off/no disables it, restoring the legacy
    # instantaneous-only terminal gate (the execute() budget-expiry branch then raises the unchanged
    # 'measured goal tolerance not reached'). Lock the switch parsing so the opt-out stays stable.
    import importlib, os
    import twin.restock.trajectory_host as th
    try:
        for val, expected in (('0', False), ('false', False), ('OFF', False), ('no', False),
                              ('1', True), ('true', True), ('', True)):
            os.environ['RESTOCK_SETTLE_ENVELOPE'] = val
            importlib.reload(th)
            assert th.SETTLE_ENVELOPE is expected, (val, th.SETTLE_ENVELOPE)
        os.environ.pop('RESTOCK_SETTLE_ENVELOPE', None)
        importlib.reload(th)
        assert th.SETTLE_ENVELOPE is True   # absent => default ON
    finally:
        os.environ.pop('RESTOCK_SETTLE_ENVELOPE', None)
        importlib.reload(th)
