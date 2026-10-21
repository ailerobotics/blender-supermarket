"""CPU-only fixtures: planner and contact here are tests, never live evidence."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.twin.restock_manipulation import (
    Observation, ProductProfile, Sensors, Target, TransferController)
from scripts.twin.restock_manipulation.perception import DepthPerception
from scripts.twin.restock_manipulation.ros_adapter import (
    JOINTS, require_sim_domain, validate_trajectory)


class FixtureMotion:
    def __init__(self):
        self.moves, self.payloads, self.stops = [], [], 0
        self.stows = []
        self.outcome, self.attach_ok = 'succeeded', True
        # fm367: when True, a 'failed' outcome is reported to the controller as a SIM-clock deadline
        # miss (retriable); default False => a collision/plan-rejection fault (fails immediately).
        self.sim_deadline_fault = False
        self.polls = 0
        # fm372: inject an arbitrary exception class/message from poll() when outcome=='raise'
        # (default None => the legacy plan-exhaust RuntimeError, so existing tests are unchanged).
        # raise_times=N raises it N times then returns 'succeeded' (to prove a bounded retry recovers).
        self.raise_exc = None
        self.raise_times = None
        self._raised_count = 0

    def begin(self, pose, joints, straight_line=False, joint_space=False, wrist1_cap=None,
              wrist2_min=None, slow=False, goal_tol=None, near_start=False):
        self.moves.append(pose)
        self.joint_space_moves = getattr(self, 'joint_space_moves', [])
        self.joint_space_moves.append(joint_space)
        self.wrist1_caps = getattr(self, 'wrist1_caps', [])
        self.wrist1_caps.append(wrist1_cap)
        self.wrist2_mins = getattr(self, 'wrist2_mins', [])
        self.wrist2_mins.append(wrist2_min)
        # fm345: per-move PLACE slow + widened goal-gate tolerance (recorded for assertions).
        self.slows = getattr(self, 'slows', [])
        self.slows.append(slow)
        self.goal_tols = getattr(self, 'goal_tols', [])
        self.goal_tols.append(goal_tol)
        # fm347 (3): per-move IK-branch-near-the-pan flag for the PLACE moves (recorded for assertions).
        self.near_starts = getattr(self, 'near_starts', [])
        self.near_starts.append(near_start)

    def poll(self):
        self.polls += 1
        if self.outcome == 'raise':
            if self.raise_times is not None and self._raised_count >= self.raise_times:
                return 'succeeded'
            self._raised_count += 1
            raise self.raise_exc or RuntimeError('planning failed (exhausted within wall cap)')
        return self.outcome

    def last_fault_was_sim_deadline(self):
        return self.sim_deadline_fault

    def begin_joint_goal(self, goal, measured, slow=False, goal_tol=None):
        self.stows.append((goal,measured,slow))

    def stop(self):
        self.stops += 1

    def update_payload(self, held, profile):
        self.payloads.append(held)
        return self.attach_ok


PROFILE = ProductProfile('sku-box', (.10, .10, .10), (1., 0., 0., 0.))
JOB = dict(job_id='job-1', sku='sku-box', quantity=1, source_bin='backroom', shelf_id='shelf-a')


def make_controller(operation='transfer', fixture_mode=True, **kwargs):
    motion = FixtureMotion()
    controller = TransferController(motion, {'sku-box': PROFILE}, fixture_mode=fixture_mode, **kwargs)
    controller.start(operation, JOB)
    return controller, motion


def packet(c, t=1., **kwargs):
    observation = Observation(f'source-{t}', 'test_fixture', t, 'sku-box', c.source,
                              (.4, 0., .2, 0., 0., 0., 1.), .99)
    target = Target(f'target-{t}', 'test_fixture', t, c.destination,
                    (.5, .2, .1, 0., 0., 0., 1.), True, (.3, .3, .4))
    import time
    values = dict(simulation_time=t, observations=(observation,), target=target,
                  parked=True, contact=c.held, feedback_time=t, collision_time=t,
                  joint_positions=(0.,)*6, scanned_locations=(c.source, c.destination), scan_time=t,
                  valid_until_steady=time.monotonic() + 100.)  # fresh acquisition deadline by default
    values.update(kwargs)
    return Sensors(**values)


def run_to(c, phase, t0=1.0):
    for i in range(80):
        if c.phase == phase:
            return t0 + i*.1
        contact = c.held or (c.phase == 'grasp' and c.issued)
        if c.phase == 'release' and c.issued:
            contact = False
        result = c.step(.1, packet(c, t0+i*.1, contact=contact))
        assert result.status != 'failed', result.reason
    pytest.fail('phase not reached')


def test_grasp_requires_stable_confirmed_attachment_before_lift():
    # fm395 (A): with grasp_confirm_ticks=3 the grasp must NOT advance to lift on a single transient
    # contact -- it needs 3 consecutive confirmed-attachment ticks (never lift on a one-tick weld).
    c, m = make_controller('collect')
    c.grasp_confirm_ticks = 3
    t = run_to(c, 'grasp')
    c.step(.1, packet(c, t, contact=False))                 # issue the gripper enable (not issued -> issued)
    r = c.step(.1, packet(c, t + .1, contact=True))         # confirm 1/3
    assert c.phase == 'grasp' and r.status == 'running'
    c.step(.1, packet(c, t + .2, contact=True))             # confirm 2/3
    assert c.phase == 'grasp'
    c.step(.1, packet(c, t + .3, contact=True))             # confirm 3/3 -> lift
    assert c.phase == 'lift'

    # A contact that DROPS before the streak completes must reset it (never advance on a flicker).
    c2, _ = make_controller('collect'); c2.grasp_confirm_ticks = 3
    t2 = run_to(c2, 'grasp')
    c2.step(.1, packet(c2, t2, contact=False))              # issue
    c2.step(.1, packet(c2, t2 + .1, contact=True))          # 1/3
    c2.step(.1, packet(c2, t2 + .2, contact=False))         # drop -> reset
    c2.step(.1, packet(c2, t2 + .3, contact=True))          # 1/3 again
    c2.step(.1, packet(c2, t2 + .4, contact=True))          # 2/3
    assert c2.phase == 'grasp'                              # still not lifted (only 2 since the reset)


def test_grasp_retries_then_fails_honestly_never_lifting_air():
    # fm395 (A): when no attachment ever confirms, the grasp bounded-retries (re-seat via contact) then
    # FAILS collect honestly -- it must NEVER advance to lift without a held carton (run-306 lifted air).
    c, m = make_controller('collect')
    c.grasp_retry_max = 2
    c.grasp_wait_ticks = 3
    t = run_to(c, 'grasp')
    retries = 0
    result = None
    for i in range(200):
        r = c.step(.1, packet(c, t + i * .1, contact=False))   # the seal/weld never forms
        assert c.phase != 'lift', 'must never lift without a confirmed carton attachment'
        if r.status == 'failed':
            result = r
            break
    assert result is not None and 'did not confirm a held carton' in result.reason
    assert c.phase == 'failed'
    assert m.payloads == []                                 # update_payload(True) never issued -> no false attach


def test_grasp_flicker_does_not_livelock_retries_or_fails_within_budget():
    # fm399 (run-309 regression): a FLICKERING contact (True/False/True...) must NOT live-lock the
    # grasp. The old wait budget reset on every contact tick, so a flicker never reached the
    # consecutive confirm threshold AND never accrued the retry/fail budget -- the phase looped until
    # the coarse 120 s phase_timeout. The monotonic budget retries then fails honestly, never lifting.
    c, m = make_controller('collect')
    c.grasp_confirm_ticks = 3        # needs 3 CONSECUTIVE; a flicker never delivers 3 in a row
    c.grasp_retry_max = 2
    c.grasp_wait_ticks = 6
    t = run_to(c, 'grasp')
    c.step(.1, packet(c, t, contact=False))                 # issue the gripper enable
    result = None
    for i in range(1, 400):
        r = c.step(.1, packet(c, t + i * .1, contact=(i % 2 == 1)))   # perfect T/F/T/F flicker
        assert c.phase != 'lift', 'must never lift without a stable confirmed attachment'
        if r.status == 'failed':
            result = r
            break
    # It fails via the HONEST grasp reason (bounded retries exhausted), NOT by the coarse phase_timeout.
    assert result is not None, 'flicker live-locked (no honest failure within the retry budget)'
    assert 'did not confirm a held carton' in result.reason
    assert c._grasp_retries == c.grasp_retry_max
    assert m.payloads == []


def test_collect_source_evidence_gate_off_is_byte_identical_reach_proceeds():
    # fm449 (a): flag OFF (default) => observe advances into the reach exactly as today, even with no
    # collect-source evidence provider wired.
    c, m = make_controller('collect')
    assert c.require_collect_source_evidence is False
    r = c.step(.1, packet(c, 1.0))
    assert c.phase == 'approach' and r.status != 'failed'   # left observe into the reach


def test_collect_source_evidence_gate_permits_reach_on_a_clean_capture():
    # fm449 (b): flag ON + a clean source capture present this tick => observe advances into the reach.
    c, m = make_controller('collect', require_collect_source_evidence=True)
    c.set_collect_source_evidence(lambda now: True)
    r = c.step(.1, packet(c, 1.0))
    assert c.phase == 'approach' and r.status != 'failed'


def test_collect_source_evidence_gate_fails_fast_when_no_clean_capture_arrives():
    # fm449 (c): flag ON + no clean capture within the bounded wait => fail fast with the named reason,
    # never entering the reach.
    c, m = make_controller('collect', require_collect_source_evidence=True)
    c.collect_source_evidence_wait_ticks = 3
    c.set_collect_source_evidence(lambda now: False)
    result = None
    for i in range(10):
        result = c.step(.1, packet(c, 1.0 + i * .1))
        assert c.phase != 'approach', 'must never begin the reach without a clean source capture'
        if result.status == 'failed':
            break
    assert c.phase == 'failed' and result.reason == 'collect_source_evidence_unavailable'
    assert m.moves == []                                   # no reach move was ever issued


def test_collect_source_evidence_gate_host_path_reads_sensors_field():
    # fm449 host path: no injected provider (the controller runs in the host process); the clean verdict
    # rides Sensors.collect_source_clean, set Isaac-side. clean=True permits the reach; clean=False holds.
    c, m = make_controller('collect', require_collect_source_evidence=True)
    assert c._collect_source_evidence is None                 # host path: no in-process provider
    r = c.step(.1, packet(c, 1.0, collect_source_clean=True))
    assert c.phase == 'approach' and r.status != 'failed'     # Isaac-set clean field permits the reach


def test_collect_source_evidence_gate_host_path_not_clean_holds_then_fails():
    # fm449 host path: Sensors.collect_source_clean False => holds in observe, then fails fast at the bound.
    c, m = make_controller('collect', require_collect_source_evidence=True)
    c.collect_source_evidence_wait_ticks = 3
    result = None
    for i in range(10):
        result = c.step(.1, packet(c, 1.0 + i * .1, collect_source_clean=False))
        assert c.phase != 'approach'
        if result.status == 'failed':
            break
    assert c.phase == 'failed' and result.reason == 'collect_source_evidence_unavailable'


def test_collect_source_evidence_gate_holds_in_observe_during_the_bounded_wait():
    # fm449: while awaiting the clean capture the controller stays in observe (running), not failed,
    # until the bounded budget is exhausted.
    c, m = make_controller('collect', require_collect_source_evidence=True)
    c.collect_source_evidence_wait_ticks = 5
    c.set_collect_source_evidence(lambda now: False)
    for i in range(5):
        r = c.step(.1, packet(c, 1.0 + i * .1))
        assert c.phase == 'observe' and r.status == 'running'


def test_collect_source_evidence_gate_held_or_seeded_prior_does_not_count():
    # fm449 (d): a HELD/seeded prior (the provider reports not-clean) does NOT satisfy the gate -- the
    # reach stays blocked until a genuine clean-this-tick True arrives. Models the camera-lane verdict
    # where an occluded/held SourceTarget yields clean=False (see test_restock_source_target).
    c, m = make_controller('collect', require_collect_source_evidence=True)
    c.collect_source_evidence_wait_ticks = 20
    state = {'clean': False}
    c.set_collect_source_evidence(lambda now: state['clean'])
    for i in range(4):                                     # held/seeded prior: not clean -> stay in observe
        c.step(.1, packet(c, 1.0 + i * .1))
        assert c.phase == 'observe'
    assert m.moves == []                                   # no reach begun on the held prior
    state['clean'] = True                                  # a genuine clean capture arrives
    c.step(.1, packet(c, 1.5))
    assert c.phase == 'approach'


def test_collect_source_evidence_gate_does_not_arm_for_transfer():
    # fm449: the gate is scoped to collect (the source is the fixed camera station). A transfer run
    # (source = cart, controlled asset) is byte-identical even with the flag on.
    c, m = make_controller('transfer', require_collect_source_evidence=True)
    c.set_collect_source_evidence(lambda now: False)       # would block collect; must not touch transfer
    r = c.step(.1, packet(c, 1.0))
    assert c.phase == 'approach' and r.status != 'failed'


def test_lift_passes_carry_wrist1_cap_to_motion_fm402():
    # fm402 regression guard for the runs 300d/305/311 jam: with carry_wrist1_cap set, the 'lift' move
    # MUST carry that wrist_1 cap to the planner so the loaded-lift IK stays below the wrist_2-vs-forearm
    # self-collision stop; an absent/None cap on the lift is the silent un-cap that jammed those runs.
    c, m = make_controller('collect', carry_wrist1_cap=0.65)
    t = run_to(c, 'lift')
    c.step(.1, packet(c, t, contact=True))     # issue the lift move
    assert 0.65 in [x for x in m.wrist1_caps if x is not None], m.wrist1_caps
    assert any(x is None for x in m.wrist1_caps)   # approach/contact moves pass None (lift-only cap)

    # And with the cap UNSET (the run-311 config), the lift carries NO cap -> IK free to cross the stop.
    c2, m2 = make_controller('collect')
    t2 = run_to(c2, 'lift')
    c2.step(.1, packet(c2, t2, contact=True))
    assert all(x is None for x in m2.wrist1_caps)


def test_lift_routes_to_joint_space_clear_shape_when_configured_fm403():
    # fm403: with carry_lift_shape set, the loaded LIFT is a FIXED begin_joint_goal to the clear-region
    # carry shape (w1<=0.65, w2>=0), NOT a task-space IK that can land in the wrist_2-vs-forearm
    # self-colliding branch (runs 300d/305/311 jammed even with wrist_1 capped). The shape must be the
    # joint goal; no task-space begin() is issued for the lift.
    shape = [1.2675, -2.27, 1.0, 0.65, 0.8, -1.2545]
    c, m = make_controller('collect', carry_lift_shape=shape)
    moves_before = len(m.moves)
    t = run_to(c, 'lift')
    moves_at_lift = len(m.moves)
    c.step(.1, packet(c, t, contact=True))     # issue the lift
    assert any(list(g[0]) == shape for g in m.stows), m.stows   # lift went through begin_joint_goal
    assert len(m.moves) == moves_at_lift                        # NO task-space begin() added by the lift


def test_task_space_lift_fallback_carries_both_wrist_guards_fm403():
    # fm403: when carry_lift_shape is UNSET, the task-space lift fallback carries BOTH guards -- the
    # wrist_1 cap AND the wrist_2 lower bound -- so the IK cannot pick the w2-negative self-colliding branch.
    c, m = make_controller('collect', carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    t = run_to(c, 'lift')
    c.step(.1, packet(c, t, contact=True))
    assert 0.65 in [x for x in m.wrist1_caps if x is not None], m.wrist1_caps
    assert 0.0 in [x for x in m.wrist2_mins if x is not None], m.wrist2_mins


def test_collision_stale_pauses_held_arm_within_max_age_then_trips_fm419():
    # fm419 (run-320): a COLLISION-only staleness while the carton is HELD (the weld/attach ESDF
    # re-incorporation lag) PAUSES the arm (holds pose, issues NO new move) while the stamp age
    # (now - collision_time) is within collision_stale_tol_s, then trips honestly once the age passes
    # the bound. Replays run-320's lift trace: collision_time frozen, so the age grows 0.52 -> 0.60 ...
    # until it exceeds the bound. With tol 0.65 s, the run-320 0.60 s stall is PAUSED (not an e-stop).
    c, m = make_controller('collect')
    c.collision_stale_tol_s = 0.65
    t = run_to(c, 'lift')
    assert c.held
    m.outcome = 'running'                      # lift stays in-flight -> phase stays lift, held True
    c.step(.1, packet(c, t, contact=True))     # issue the lift
    frozen = t - 0.5                           # collision_time frozen 0.5 s behind -> age grows with now
    moves_before = len(m.stows) + len(m.moves)
    # ages 0.517..0.600 (<= 0.65) -> PAUSE: running, no NEW move issued, `issued` cleared for a re-plan
    for now in (t + 0.017, t + 0.05, t + 0.10):
        r = c.step(.1, packet(c, now, contact=True, collision_time=frozen))
        assert r.status == 'running' and r.reason == '', (now, r.reason)
        assert c.issued is False
    assert len(m.stows) + len(m.moves) == moves_before   # paused: the lift was NOT re-issued while stale
    # a FRESH stamp resumes: _advance re-issues the held lift (begin_joint_goal to the carry shape)
    c.step(.1, packet(c, t + 0.12, contact=True))
    assert c.issued is True
    # if instead the stamp stays frozen until the age passes the bound (0.66 > 0.65) -> honest trip
    r = c.step(.1, packet(c, frozen + 0.66, contact=True, collision_time=frozen))
    assert r.status == 'failed' and 'collision_dt=' in r.reason, r.reason


def test_collision_stale_zero_tolerance_pre_grasp_fm419():
    # pre-attach (not held): collision staleness is a real fault (the parked mapper must be fresh before
    # manipulating) -> zero tolerance, fail immediately naming collision, even within the max age.
    c, _ = make_controller('collect')
    c.collision_stale_tol_s = 0.8
    r = c.step(.1, packet(c, 5.0, collision_time=4.4))   # observe phase, not held; age 0.6 < tol
    assert r.status == 'failed' and 'collision_dt=' in r.reason, r.reason


def test_collision_tolerance_off_by_default_zero_byte_identical_fm419():
    # 0 s tolerance => no pause, trip exactly as before (byte-identical to the pre-fm415 invariant).
    c, m = make_controller('collect')
    c.collision_stale_tol_s = 0.0             # off
    t = run_to(c, 'lift'); m.outcome = 'running'
    c.step(.1, packet(c, t, contact=True))
    r = c.step(.1, packet(c, t + .1, contact=True, collision_time=(t + .1) - 0.6))   # age 0.6
    assert r.status == 'failed' and 'collision_dt=' in r.reason, r.reason   # zero tol -> no pause


def test_future_lead_collision_stamp_is_fresh_no_pause_fm423():
    # fm423 (b) / run-321 cold collect: a relayed collision stamp can arrive marginally AHEAD of the
    # controller's node `now` (a render-pacing clock-relay lead). It is FRESH (within render_lead), so
    # the held lift must NOT trip the invariant and must NOT enter COLLISION_STALE_PAUSE -- the bug that
    # held the lift indefinitely and starved the 90 s arm-command heartbeat ('arm command heartbeat lost').
    import contextlib, io
    c, m = make_controller('collect')
    assert c.render_lead == 0.2                 # default lead matches trajectory_host's relay race
    t = run_to(c, 'lift'); assert c.held
    m.outcome = 'running'
    c.step(.1, packet(c, t, contact=True))      # issue the lift
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # collision stamp 16.7 ms (one physics step) in the FUTURE -> age -0.0167, within -render_lead
        r = c.step(.1, packet(c, t + .1, contact=True, collision_time=(t + .1) + 0.0167))
    assert r.status == 'running' and r.reason == '', r.reason
    assert c.issued is True                      # lift proceeds (not paused / not failed)
    assert 'COLLISION_STALE_PAUSE' not in buf.getvalue()
    assert 'CONTROLLER_INVARIANT_TRIP' not in buf.getvalue()


def test_future_feedback_stamp_within_render_lead_is_fresh_fm423():
    # The same render-lead tolerance applies to feedback_time (same relay family). A marginally-future
    # feedback stamp pre-grasp is fresh, not a fault.
    c, _ = make_controller('collect')
    r = c.step(.1, packet(c, 5.0, feedback_time=5.0 + 0.1, collision_time=5.0 + 0.1))  # 0.1 s lead < 0.2
    assert r.status != 'failed', r.reason


def test_wildly_future_collision_stamp_still_trips_fm423():
    # A stamp beyond render_lead in the future is a real clock anomaly: it must TRIP honestly (never a
    # pause -- a persistent future lead must not hold the arm forever).
    c, m = make_controller('collect')
    t = run_to(c, 'lift'); assert c.held
    m.outcome = 'running'
    c.step(.1, packet(c, t, contact=True))
    r = c.step(.1, packet(c, t + .1, contact=True, collision_time=(t + .1) + 0.5))   # 0.5 s future >> lead
    assert r.status == 'failed' and 'collision_dt=' in r.reason, r.reason


def test_contact_stall_wall_fail_fast_fm425():
    # fm425 (b) / run-322 cold: the contact descent crept CONTACT_STEP with contact=False and the
    # arm-command stream lapsed into the 90 s host heartbeat watchdog ('arm command heartbeat lost', no
    # named cause). A WALL budget now fails the contact phase fast with a named reason BEFORE the watchdog.
    clock = [0.0]
    c, m = make_controller('collect', steady_clock=lambda: clock[0],
                           contact_straight_line=True, contact_overpress_step_m=0.003,
                           contact_overpress_cap_m=0.02)
    c.contact_stuck_wall_s = 5.0
    t = run_to(c, 'contact')                       # reach the contact phase (progress mark set at wall 0)
    # Press with contact never latching; hold the wall clock just under budget -> still running.
    clock[0] = 4.9
    r = c.step(.1, packet(c, t, contact=False))
    assert r.status == 'running', r.reason
    # Past the budget with no contact progress -> fail fast with a named reason (not the heartbeat).
    clock[0] = 6.0
    r = c.step(.1, packet(c, t + .1, contact=False))
    assert r.status == 'failed' and 'contact stalled' in r.reason and 'heartbeat' in r.reason, r.reason


def test_contact_stall_wall_budget_default_off_is_byte_identical_fm425():
    # Default (contact_stuck_wall_s == 0) => the WALL budget is inert: a long stall does not fail here
    # (the existing cap / host watchdog still backstop), so absent config is byte-identical.
    clock = [0.0]
    c, m = make_controller('collect', steady_clock=lambda: clock[0],
                           contact_straight_line=True, contact_overpress_step_m=0.003,
                           contact_overpress_cap_m=0.02)
    assert c.contact_stuck_wall_s == 0.0
    t = run_to(c, 'contact')
    clock[0] = 10_000.0                            # absurd wall elapsed
    r = c.step(.1, packet(c, t, contact=False))
    assert r.status != 'failed' or 'contact stalled' not in r.reason, r.reason


def test_collision_stale_pause_keeps_stream_alive_and_wall_fail_fast_fm427():
    # fm427 (A1) / run-323 cold lift: the COLLISION_STALE_PAUSE held the post-attach lift on a genuinely-
    # stale collision stamp (the benign weld/ESDF re-incorporation lag) by returning EMPTY controls -- a
    # SILENT hold that starved the 90 s heartbeat. With collision_pause_wall_s>0 the pause now (1) re-
    # commands a HOLD at the CURRENT measured joints each tick (stream alive, stationary) and (2) fails
    # fast with a named reason past the WALL budget.
    clock = [0.0]
    c, m = make_controller('collect', steady_clock=lambda: clock[0])
    c.collision_stale_tol_s = 0.8
    c.collision_pause_wall_s = 5.0
    t = run_to(c, 'lift'); assert c.held
    m.outcome = 'running'
    c.step(.1, packet(c, t, contact=True))                     # issue the lift
    stows_before = len(m.stows)
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # a genuinely-STALE collision stamp (0.6 s old): max_age(0.5) < 0.6 <= tol(0.8) -> PAUSE
        r = c.step(.1, packet(c, t + .1, contact=True, collision_time=(t + .1) - 0.6))
    assert r.status == 'running', r.reason
    assert 'COLLISION_STALE_PAUSE' in buf.getvalue()
    assert len(m.stows) > stows_before                         # a HOLD command was issued (stream fed)
    goal, measured, _slow = m.stows[-1]
    assert goal == (0.,) * 6 and measured == (0.,) * 6          # hold at the CURRENT joints (stationary)
    assert c.phase == 'lift'                                    # did NOT advance the held move
    # past the wall budget with the stamp still stale -> fail fast with a named reason (not the watchdog)
    clock[0] = 6.0
    r = c.step(.1, packet(c, t + .2, contact=True, collision_time=(t + .2) - 0.6))
    assert r.status == 'failed' and 'collision-stale pause' in r.reason and 'heartbeat' in r.reason, r.reason


def test_held_lift_collision_grace_rides_then_fails_fast_fm430():
    # fm430 (2a) fallback: the post-attach LIFT to the fixed clear-region carry_lift_shape may RIDE a
    # stale collision world (proceed, not pause) for a bounded WALL grace, then fail fast named. Scoped to
    # phase 'lift' + carry_lift_shape set + collision-sole staleness + held.
    clock = [0.0]
    c, m = make_controller('collect', steady_clock=lambda: clock[0],
                           carry_lift_shape=(1.2675, -2.27, 1.0, 0.65, 0.8, -1.2545))
    c.collision_stale_tol_s = 0.8
    c.lift_collision_grace_wall_s = 5.0
    t = run_to(c, 'lift'); assert c.held
    m.outcome = 'running'
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # stale collision stamp (0.6 s old): past the 0.5 max_age -> would pause; the grace RIDES it
        r = c.step(.1, packet(c, t, contact=True, collision_time=t - 0.6))
    assert r.status == 'running', r.reason
    assert 'LIFT_COLLISION_GRACE' in buf.getvalue()
    assert 'COLLISION_STALE_PAUSE' not in buf.getvalue()   # grace proceeds, never the hold path
    assert c.phase == 'lift'
    # past the wall grace with the stamp still stale -> fail fast named
    clock[0] = 6.0
    r = c.step(.1, packet(c, t + .1, contact=True, collision_time=(t + .1) - 0.6))
    assert r.status == 'failed' and 'held-lift collision grace exceeded' in r.reason, r.reason


def test_held_lift_collision_grace_default_off_pauses_fm430():
    # Default (lift_collision_grace_wall_s == 0) => the fm419(a) pause path, byte-identical (no grace).
    c, m = make_controller('collect', carry_lift_shape=(1.2675, -2.27, 1.0, 0.65, 0.8, -1.2545))
    c.collision_stale_tol_s = 0.8
    assert c.lift_collision_grace_wall_s == 0.0
    t = run_to(c, 'lift'); assert c.held
    m.outcome = 'running'
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = c.step(.1, packet(c, t, contact=True, collision_time=t - 0.6))
    assert r.status == 'running', r.reason
    assert 'COLLISION_STALE_PAUSE' in buf.getvalue()       # the legacy pause, not the grace
    assert 'LIFT_COLLISION_GRACE' not in buf.getvalue()


def test_held_lift_collision_grace_scoped_to_lift_only_fm430():
    # The grace must NOT apply outside the lift (e.g. at contact/grasp) -- a collision-stale there still
    # pauses (fm419) or trips, never rides. Here with grace on but phase != lift.
    c, m = make_controller('collect', carry_lift_shape=(1.2675, -2.27, 1.0, 0.65, 0.8, -1.2545))
    c.collision_stale_tol_s = 0.8
    c.lift_collision_grace_wall_s = 5.0
    t = run_to(c, 'grasp')   # held becomes True only after grasp->lift; at grasp pre-weld it is not held
    # drive a held state at a non-lift phase by forcing phase after a hold; simplest: assert grace inert
    # when not in lift via a direct check that a stale collision at 'grasp' does not print the grace.
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c.step(.1, packet(c, t, contact=False))            # issue gripper
        c.step(.1, packet(c, t + .1, contact=False, collision_time=(t + .1) - 0.6))
    assert 'LIFT_COLLISION_GRACE' not in buf.getvalue()     # never grace outside the lift


def test_collision_stale_pause_hold_swallows_motion_active_raise_fm427():
    # fm427 (A1) / run-324 cold REGRESSION: the keep-alive calls begin_joint_goal, which the REAL host
    # (RosMotionPort) raises 'motion already active' when the lift trajectory is still active -- and that
    # active move is ALREADY feeding the arm-command stream, so the hold is unnecessary. The raise MUST be
    # swallowed (an uncaught RuntimeError here killed the cold step -> heartbeat lost); the WALL budget
    # stays the backstop. (FixtureMotion does not model the guard, which is why the CPU suite missed it.)
    clock = [0.0]
    c, m = make_controller('collect', steady_clock=lambda: clock[0])
    c.collision_stale_tol_s = 0.8
    c.collision_pause_wall_s = 5.0
    t = run_to(c, 'lift'); assert c.held
    m.outcome = 'running'
    c.step(.1, packet(c, t, contact=True))
    def boom(*a, **k):
        raise RuntimeError('motion already active')
    m.begin_joint_goal = boom
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = c.step(.1, packet(c, t + .1, contact=True, collision_time=(t + .1) - 0.6))
    assert r.status == 'running', r.reason                     # swallowed, NOT crashed
    assert 'hold skipped' in buf.getvalue()
    clock[0] = 6.0                                             # wall budget still fails fast past the bound
    r = c.step(.1, packet(c, t + .2, contact=True, collision_time=(t + .2) - 0.6))
    assert r.status == 'failed' and 'collision-stale pause' in r.reason, r.reason


def test_collision_stale_pause_default_off_is_byte_identical_fm427():
    # collision_pause_wall_s == 0 (default) => the legacy empty-control pause: NO hold command issued,
    # NO wall fail-fast (the pause still holds pose and resumes on a fresh stamp, exactly as before).
    c, m = make_controller('collect')
    c.collision_stale_tol_s = 0.8
    assert c.collision_pause_wall_s == 0.0
    t = run_to(c, 'lift'); assert c.held
    m.outcome = 'running'
    c.step(.1, packet(c, t, contact=True))
    stows_before = len(m.stows)
    r = c.step(.1, packet(c, t + .1, contact=True, collision_time=(t + .1) - 0.6))
    assert r.status == 'running' and r.controls == ()          # empty-control pause (legacy)
    assert len(m.stows) == stows_before                        # NO hold command (byte-identical)


def test_invariant_trip_names_the_failed_condition_fm412():
    # fm412 (run-317/318): the top-of-step invariant is a 5-condition OR; its reason must name the ACTUAL
    # failed field + value so a run that dies here is self-diagnosing (both runs died here ambiguously).
    c, _ = make_controller('collect')
    r = c.step(.1, packet(c, 1.0, parked=False))
    assert r.status == 'failed' and 'parked=False' in r.reason, r.reason
    c2, _ = make_controller('collect')
    r2 = c2.step(.1, packet(c2, 5.0, collision_time=1.0))   # collision 4s old >> max_age -> stale
    assert r2.status == 'failed' and 'collision_dt=' in r2.reason, r2.reason


def _confirm_grasp_with_carton(c, z0, t0=1.0):
    """Drive to grasp and confirm the attachment with a MEASURED carton pose (physics confirm), latching
    z0 as the lift-probe baseline. Returns the next packet time. Uses no source observation at confirm."""
    t = run_to(c, 'grasp', t0)
    c.step(.1, packet(c, t, contact=False, placed_pose=(.4, 0., z0, 0., 0., 0., 1.)))       # issue gripper
    r = c.step(.1, packet(c, t + .1, contact=True, placed_pose=(.4, 0., z0, 0., 0., 0., 1.)))  # confirm -> lift
    assert c.phase == 'lift', (c.phase, r.reason)
    return t + .2


def test_grasp_confirm_is_physics_and_advances_when_carton_tracks_lift_fm410():
    # fm410 (run-316): the grasp confirms on the ATTACHMENT, then the LIFT is the physics probe -- with the
    # measured carton risen >= grasp_track_m it advances to carry. Crucially the lift packets carry NO source
    # observation (observations=()), modelling the arm occluding the fixed source cam during grasp/hold: that
    # must never block or fail the confirm/advance (run-316's stall was the source going occluded there).
    c, m = make_controller('collect', grasp_track_m=0.03)
    z0 = 0.2
    t = _confirm_grasp_with_carton(c, z0)
    c.step(.1, packet(c, t, contact=True, observations=(), placed_pose=(.4, 0., z0 + .05, 0., 0., 0., 1.)))   # issue lift
    r = c.step(.1, packet(c, t + .1, contact=True, observations=(), placed_pose=(.4, 0., z0 + .05, 0., 0., 0., 1.)))  # complete
    assert c.phase == 'preplace', (c.phase, r.reason)   # carton tracked the tool up -> advanced to carry
    assert r.status == 'running'


def test_grasp_confirm_fails_when_carton_does_not_track_the_lift_fm410():
    # A carton that stayed put through the lift (placed_pose flat) was never truly held: FAIL honestly rather
    # than fold a phantom load through the cart. This is the physics gate the source-camera confirm lacked.
    c, m = make_controller('collect', grasp_track_m=0.03)
    z0 = 0.2
    t = _confirm_grasp_with_carton(c, z0)
    c.step(.1, packet(c, t, contact=True, placed_pose=(.4, 0., z0, 0., 0., 0., 1.)))              # issue lift
    r = c.step(.1, packet(c, t + .1, contact=True, placed_pose=(.4, 0., z0 + .005, 0., 0., 0., 1.)))  # barely moved
    assert r.status == 'failed'
    assert 'did not track' in r.reason, r.reason


def test_grasp_track_inert_without_threshold_or_placed_pose_byte_identical_fm410():
    # Off (grasp_track_m unset) => no physics gate, advance exactly as before (byte-identical legacy fm395).
    c, _ = make_controller('collect')
    assert run_to(c, 'preplace') and c.phase == 'preplace'
    # ON but the driver provides NO placed_pose (the real rig / fixture) => baseline None => gate inert, so a
    # real-rig run that cannot measure the carton is never blocked by this sim-only confirm.
    c2, _ = make_controller('collect', grasp_track_m=0.03)
    assert run_to(c2, 'preplace') and c2.phase == 'preplace'


def test_grasp_stable_hold_still_confirms_under_monotonic_budget():
    # fm399: the monotonic wait must NOT pre-empt a genuine stable hold: with wait_ticks >> confirm_ticks
    # a steady contact confirms and lifts well inside the budget (no spurious retry).
    c, m = make_controller('collect')
    c.grasp_confirm_ticks = 3
    c.grasp_retry_max = 2
    c.grasp_wait_ticks = 40
    t = run_to(c, 'grasp')
    c.step(.1, packet(c, t, contact=False))                 # issue
    for i in range(1, 5):
        c.step(.1, packet(c, t + i * .1, contact=True))
    assert c.phase == 'lift'
    assert c._grasp_retries == 0


@pytest.mark.parametrize('operation', ['collect', 'transfer'])
def test_success_needs_two_new_observations_and_source_scan(operation):
    c, m = make_controller(operation)
    t = run_to(c, 'verify')
    obs = Observation('placed-1', 'test_fixture', t, 'sku-box', c.destination,
                      (.5, .2, .2, 0., 0., 0., 1.), .99)
    assert c.step(.1, packet(c, t, observations=(obs,), scanned_locations=())).status == 'running'
    assert c.step(.1, packet(c, t+.1, observations=(replace(obs, simulation_time=t+.1),))).status == 'running'
    # A new id on the same acquisition must not double count.
    assert c.step(.1, packet(c, t+.2, observations=(replace(obs, observation_id='renamed', simulation_time=t+.1),))).status == 'running'
    result = c.step(.1, packet(c, t+.3, observations=(replace(obs, observation_id='placed-2', simulation_time=t+.3),)))
    assert result.status == 'succeeded'
    assert len(m.moves) == 6 and m.payloads == [True, False]
    assert all(e['source'] == 'test_fixture' for e in result.evidence)


def test_live_rejects_fixture_and_ground_truth():
    c, m = make_controller(fixture_mode=False, phase_timeout=.2)
    s = packet(c)
    for source in ('test_fixture', 'ground_truth'):
        r = c.step(.1, replace(s, observations=(replace(s.observations[0], source=source),)))
        assert r.status == 'running'
    assert c.step(.1, s).status == 'failed'
    assert not m.moves


def test_staged_cart_swing_routes_lift_through_tuck_pan_and_retreat_through_tuck():
    # firstmate 292: lift -> TUCK -> PAN -> preplace, and retreat -> RETREAT_TUCK -> stow, as joint-space
    # via points so the welded carton clears the robot body. TUCK/retreat_tuck fold to the STOW shape at
    # the measured shoulder-pan; PAN rotates joint0 to cart_pan_joint0.
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    tuck_shape = (-2.3, 1.0, 2.1066, 0.8, -1.2545)        # the high carry shape (joints 1-5)
    c, m = make_controller(transport_joints=stow, staged_cart_swing=True, cart_pan_joint0=-0.8605,
                           cart_tuck_shape=tuck_shape)
    t = run_to(c, 'tuck')                                 # lift transitions to tuck, not preplace
    before = len(m.stows)
    c.step(.1, packet(c, t))                              # issue the tuck joint goal
    assert m.stows[before][0] == (0.0,) + tuck_shape      # fold at measured joint0 (=0), carry shape
    assert m.stows[before][2] is True                     # tuck carries the carton -> slow fold
    c.step(.1, packet(c, t+.1))                           # poll succeeds -> pan
    assert c.phase == 'pan'
    c.step(.1, packet(c, t+.2))                           # issue the pan joint goal
    assert m.stows[-1][0] == (-0.8605,) + tuck_shape      # pan to the cart side at the carry shape
    c.step(.1, packet(c, t+.3))
    assert c.phase == 'preplace'                          # then extend to the mirrored preplace
    # Return leg: retreat -> retreat_tuck -> stow.
    t = run_to(c, 'retreat_tuck', t0=t+1.0)
    n = len(m.stows)
    c.step(.1, packet(c, t))
    assert m.stows[n][0] == (0.0,) + tuck_shape           # fold at the measured pan on the way back
    assert m.stows[n][2] is False                         # retreat_tuck is empty-handed -> full speed
    c.step(.1, packet(c, t+.1))
    assert c.phase == 'stow'


def _staged_controller():
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    tuck_shape = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
    return make_controller(transport_joints=stow, staged_cart_swing=True,
                           cart_pan_joint0=-0.8605, cart_tuck_shape=tuck_shape)


def test_reposition_before_place_holds_in_transit_until_base_repositioned():
    # fm390: with reposition_before_place, the carry fold routes lift -> tuck -> TRANSIT (not pan); the
    # arm HOLDS in TRANSIT (no further joint goal) until the adapter sets base_repositioned, then proceeds
    # to preplace at the new base pose. The pan (station-pose joint0 swing) is skipped -- base nav replaces it.
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    tuck_shape = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
    c, m = make_controller(transport_joints=stow, staged_cart_swing=True, cart_pan_joint0=-0.8605,
                           cart_tuck_shape=tuck_shape, reposition_before_place=True)
    t = run_to(c, 'tuck')
    c.step(.1, packet(c, t))                      # issue the tuck fold
    c.step(.1, packet(c, t + .1))                 # tuck succeeds -> TRANSIT (not pan)
    assert c.phase == 'transit'
    n = len(m.stows) + len(m.moves)
    r = c.step(.1, packet(c, t + .2))             # HOLD: no new arm goal while the base drives
    assert c.phase == 'transit' and r.status == 'running'
    assert len(m.stows) + len(m.moves) == n       # nothing issued during the hold
    c.base_repositioned = True                    # adapter drove the base to place_approach
    c.step(.1, packet(c, t + .3))
    assert c.phase == 'preplace'                  # resumes the place at the new pose


def test_transit_resumes_on_sensor_carried_base_repositioned_fm400():
    # fm400 handshake, host side: the host controller runs in a SEPARATE process, so the adapter cannot
    # set its in-process self.base_repositioned -- it carries arrival on the Sensors packet
    # (s.base_repositioned). TRANSIT must resume to preplace on the sensor flag even though
    # self.base_repositioned stays False (the run-310 gap: arrival never reached the host controller).
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    tuck_shape = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
    c, m = make_controller(transport_joints=stow, staged_cart_swing=True, cart_pan_joint0=-0.8605,
                           cart_tuck_shape=tuck_shape, reposition_before_place=True)
    t = run_to(c, 'tuck')
    c.step(.1, packet(c, t)); c.step(.1, packet(c, t + .1))      # tuck -> TRANSIT
    assert c.phase == 'transit'
    c.step(.1, packet(c, t + .2))                                # still held; no sensor flag yet
    assert c.phase == 'transit' and c.base_repositioned is False
    c.step(.1, packet(c, t + .3, base_repositioned=True))        # arrival carried ON THE SENSOR PACKET
    assert c.phase == 'preplace'                                 # resumes even though self.base_repositioned is False
    assert c.base_repositioned is False


def test_reposition_reregisters_deck_target_in_new_base_frame():
    # fm390 PERCEIVE: after the base reaches place_approach, the controller re-reads the live deck target
    # in the NEW base frame and rebuilds the place-side poses from it -- the observe-time poses are in the
    # stale pickup frame. A fresh target at a different position must re-anchor target_surface + place.
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    tuck_shape = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
    c, m = make_controller(transport_joints=stow, staged_cart_swing=True, cart_pan_joint0=-0.8605,
                           cart_tuck_shape=tuck_shape, reposition_before_place=True)
    t = run_to(c, 'tuck')
    c.step(.1, packet(c, t)); c.step(.1, packet(c, t + .1))      # tuck -> TRANSIT
    assert c.phase == 'transit'
    fresh = Target('deck-new', 'test_fixture', t + .2, c.destination,
                   (.55, -.05, .12, 0., 0., 0., 1.), True, (.3, .3, .4))
    c.base_repositioned = True
    r = c.step(.1, packet(c, t + .2, target=fresh))
    assert r.status == 'running' and c.phase == 'preplace'
    assert c.target_surface[:3] == (.55, -.05, .12)              # re-anchored to the fresh deck read
    assert c.poses['place'][:2] == (.55, -.05)                   # place rebuilt in the new frame


def test_reposition_fails_if_deck_not_clear_at_place_approach():
    # fm390: the PERCEIVE step must FAIL (never place blindly) if the deck is not clear from the new pose.
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    tuck_shape = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
    c, m = make_controller(transport_joints=stow, staged_cart_swing=True, cart_pan_joint0=-0.8605,
                           cart_tuck_shape=tuck_shape, reposition_before_place=True)
    t = run_to(c, 'tuck')
    c.step(.1, packet(c, t)); c.step(.1, packet(c, t + .1))
    assert c.phase == 'transit'
    blocked = Target('deck-occ', 'test_fixture', t + .2, c.destination,
                     (.5, .2, .1, 0., 0., 0., 1.), False, (.3, .3, .4))
    c.base_repositioned = True
    r = c.step(.1, packet(c, t + .2, target=blocked))
    assert r.status == 'failed' and 'deck not clear' in r.reason


def test_reposition_full_place_path_reaches_verify():
    # fm390 end-to-end (controller tier): collect with reposition_before_place drives observe -> ... ->
    # lift -> tuck -> TRANSIT -> (base drives) -> PERCEIVE re-register -> preplace -> place -> release ->
    # retreat -> retreat_tuck -> stow -> verify. The deck target is held constant (parked at place_approach)
    # so the preplace re-validation holds. Proves the place-side sequence completes in reposition mode.
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    tuck_shape = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
    c, m = make_controller('collect', transport_joints=stow, staged_cart_swing=True,
                           cart_pan_joint0=-0.8605, cart_tuck_shape=tuck_shape,
                           reposition_before_place=True)
    t0 = 1.0
    for i in range(120):
        if c.phase == 'verify':
            break
        if c.phase == 'transit':
            c.base_repositioned = True      # the adapter drove the base to place_approach
        contact = c.held or (c.phase == 'grasp' and c.issued)
        if c.phase == 'release' and c.issued:
            contact = False
        extra = {'joint_positions': stow} if c.phase == 'stow' else {}  # stow confirms transport feedback
        r = c.step(.1, packet(c, t0 + i * .1, contact=contact, **extra))
        assert r.status != 'failed', (c.phase, r.reason)
    assert c.phase == 'verify'


def test_reposition_fails_if_deck_target_not_reacquired():
    # fm390: a missing/mislocated deck read at place_approach must FAIL, not fall through to a blind place.
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    tuck_shape = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
    c, m = make_controller(transport_joints=stow, staged_cart_swing=True, cart_pan_joint0=-0.8605,
                           cart_tuck_shape=tuck_shape, reposition_before_place=True)
    t = run_to(c, 'tuck')
    c.step(.1, packet(c, t)); c.step(.1, packet(c, t + .1))
    assert c.phase == 'transit'
    c.base_repositioned = True
    r = c.step(.1, packet(c, t + .2, target=None))
    assert r.status == 'failed' and 'not reacquired' in r.reason


def test_reposition_before_place_off_keeps_pan_byte_identical():
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    tuck_shape = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
    c, m = make_controller(transport_joints=stow, staged_cart_swing=True, cart_pan_joint0=-0.8605,
                           cart_tuck_shape=tuck_shape)        # reposition_before_place defaults off
    t = run_to(c, 'tuck')
    c.step(.1, packet(c, t)); c.step(.1, packet(c, t + .1))
    assert c.phase == 'pan'                        # unchanged: tuck -> pan


def test_staged_move_retries_then_succeeds_on_transient():
    # fm358: a staged move (run-283 pan) failed once at the plan->execute handoff after a normal plan.
    # A transient 'failed' must RETRY (fresh plan) rather than fail the stage.
    c, m = _staged_controller()
    t = run_to(c, 'pan')
    c.step(.1, packet(c, t))                 # issue the pan move
    n = len(m.stows)
    m.outcome = 'failed'
    c.step(.1, packet(c, t + .1))            # poll fails -> retry (re-issue next tick), stays in pan
    assert c.phase == 'pan'
    c.step(.1, packet(c, t + .2))            # re-plan the pan move
    assert len(m.stows) == n + 1             # a fresh begin_joint_goal was issued
    m.outcome = 'succeeded'
    c.step(.1, packet(c, t + .3))            # now it succeeds -> transition
    assert c.phase == 'preplace'


def test_staged_move_retries_on_plan_exhaust_raise():
    # run-290: a staged move's cuMotion planning can RAISE 'planning failed (exhausted within wall cap)'
    # on a stochastic bad streak; it must be caught and retried (fresh budget), not fail the mission.
    c, m = _staged_controller()
    t = run_to(c, 'pan')
    c.step(.1, packet(c, t))                 # issue the pan move
    n = len(m.stows)
    m.outcome = 'raise'                      # poll() raises the plan-exhaust RuntimeError
    c.step(.1, packet(c, t + .1))            # caught -> retry, stays in pan (not failed)
    assert c.phase == 'pan'
    c.step(.1, packet(c, t + .2))            # re-plan with a fresh budget
    assert len(m.stows) == n + 1
    m.outcome = 'succeeded'
    c.step(.1, packet(c, t + .3))
    assert c.phase == 'preplace'


def test_staged_move_retry_exhausts_then_fails():
    # The retry is BOUNDED: a persistently failing staged move still fails the stage (not an infinite loop).
    c, m = _staged_controller()
    t = run_to(c, 'pan')
    c.step(.1, packet(c, t))                 # issue
    m.outcome = 'failed'
    result = None
    for i in range(12):
        result = c.step(.1, packet(c, t + (i + 1) * .1))
        if c.phase == 'failed':
            break
    assert c.phase == 'failed'
    assert 'staged cart-swing pan move failed' in (result.reason or '')


def test_no_staged_swing_keeps_direct_lift_to_preplace():
    c, m = make_controller(transport_joints=(0.,)*6)      # staged off by default
    run_to(c, 'preplace')
    assert c.phase == 'preplace'                          # lift -> preplace directly, no tuck/pan


def test_place_moves_request_near_start_ik_branch_but_pick_moves_do_not():
    # fm347 (3): the cart PLACE pose moves (preplace/place) must seed+branch the IK to the measured PAN
    # config so shoulder_pan stays on the pan branch -- run-273 flipped j0 ~3.9 rad (pan -1.822 vs
    # preplace IK ~2.09) and the soft drive stalled. Pick moves (approach/contact/lift) must NOT.
    c, m = make_controller(transport_joints=(0.11, -2.5, 1.7, -0.7, -1.5708, 0.0),
                           staged_cart_swing=True, cart_pan_joint0=-1.822,
                           cart_tuck_shape=(-2.3, 1.0, 0.65, 0.8, -1.2545))
    t = run_to(c, 'preplace')                             # through approach/contact/grasp/lift/tuck/pan
    assert m.near_starts and not any(m.near_starts)       # every begin() so far is a pick move -> False
    c.step(.1, packet(c, t))                              # issue the preplace pose move
    assert m.near_starts[-1] is True                      # preplace stays on the pan branch
    assert m.slows[-1] is True and m.goal_tols[-1] == 0.12  # and keeps the fm345 place relaxations


def test_transport_stow_is_planned_before_fresh_inventory_verification():
    c,m=make_controller(transport_joints=(0.,)*6)
    t=run_to(c,'stow')
    assert c.step(.1,packet(c,t)).status=='running'
    assert m.stows==[((0.,)*6,(0.,)*6,False)]   # stow is empty-handed -> full speed (slow=False)
    assert c.step(.1,packet(c,t+.1)).status=='running'
    assert c.phase=='verify' and not c.verify_ids


def test_stow_executor_success_cannot_override_wrong_measured_joints():
    c,m=make_controller(transport_joints=(.2,)*6)
    t=run_to(c,'stow')
    c.step(.1,packet(c,t))
    result=c.step(.1,packet(c,t+.1))
    assert result.status=='failed'
    assert 'joint feedback' in result.reason


@pytest.mark.parametrize('change,reason', [
    ({'parked': False}, 'base moving'),
    ({'feedback_time': 0}, 'stale'),
    ({'collision_time': 0}, 'stale'),
    ({'joint_positions': (float('nan'),)*6}, 'feedback'),
])
def test_feedback_gates(change, reason):
    c, m = make_controller()
    result = c.step(.1, packet(c, **change))
    assert result.status == 'failed' and reason in result.reason and m.stops == 1


@pytest.mark.parametrize('field,value', [('clear', False), ('free_dimensions', (.05,.05,.05)),
                                        ('pose', (2.,0.,0.,0.,0.,0.,1.))])
def test_target_refusal(field, value):
    c, m = make_controller()
    s = packet(c)
    result = c.step(.1, replace(s, target=replace(s.target, **{field: value})))
    assert result.status == 'failed' and not m.moves


def test_lost_grasp_stops_without_release():
    c, m = make_controller()
    t = run_to(c, 'lift')
    r = c.step(.1, packet(c, t, contact=False))
    assert r.reason == 'lost grasp'
    assert r.controls[-1] == {'kind': 'gripper', 'enabled': True}
    with pytest.raises(ValueError, match='recovery'):
        c.start('transfer', JOB)


def _fresh():
    import time
    return time.monotonic() + 100.   # a future steady-clock acquisition deadline (require_fresh passes)


def test_contact_descent_enables_suction_then_terminates_on_latch():
    # Suction-grasp choreography + contact-terminated move (firstmate keys). In the 'contact' phase:
    # (1) issue the descent move; (2) enable suction ONCE a fresh valid_until_steady is present (the
    # grip command carries an acquisition deadline); (3) s.contact is the grip LATCH (needs suction
    # on), so once it latches, raise {'kind':'contact'} to terminate the move holding the touch pose;
    # (4) advance to grasp. Free-space moves never enable suction or signal contact.
    c, m = make_controller()
    t = run_to(c, 'contact')
    m.outcome = 'running'
    r = c.step(.1, packet(c, t, contact=False, valid_until_steady=_fresh()))        # issue the move
    assert r.controls[0]['kind'] == 'move'
    r = c.step(.1, packet(c, t+.1, contact=False, valid_until_steady=_fresh()))     # suction enabled
    assert {'kind': 'gripper', 'enabled': True} in r.controls
    r = c.step(.1, packet(c, t+.2, contact=False, valid_until_steady=_fresh()))     # running, no latch
    assert r.status == 'running' and not any(x.get('kind') == 'contact' for x in r.controls)
    r = c.step(.1, packet(c, t+.3, contact=True, valid_until_steady=_fresh()))      # latch -> terminate
    assert r.status == 'running' and c.phase == 'contact'
    assert any(x.get('kind') == 'contact' for x in r.controls)
    m.outcome = 'succeeded'                                                          # host succeeded+held
    c.step(.1, packet(c, t+.4, contact=True, valid_until_steady=_fresh()))
    assert c.phase == 'grasp'


def test_contact_suction_waits_for_a_fresh_acquisition_deadline():
    # The grip command must carry an acquisition deadline; under the sensing gate the first held ticks
    # may lack one (run-137). Do not emit the gripper until valid_until_steady is present.
    c, m = make_controller()
    t = run_to(c, 'contact')
    m.outcome = 'running'
    c.step(.1, packet(c, t, contact=False, valid_until_steady=_fresh()))            # issue the move
    r = c.step(.1, packet(c, t+.1, contact=False, valid_until_steady=None))         # no deadline yet
    assert not any(x.get('kind') == 'gripper' for x in r.controls)                  # suction withheld


def test_free_space_move_never_enables_suction_or_signals_contact():
    # The approach (free-space) move keeps its exact pose-settle: never enables suction, never signals
    # contact, even if a (spurious) contact is measured mid-flight.
    c, m = make_controller()
    run_to(c, 'approach')
    m.outcome = 'running'
    c.step(.1, packet(c, 50., contact=False, valid_until_steady=_fresh()))          # issue approach
    r = c.step(.1, packet(c, 50.1, contact=True, valid_until_steady=_fresh()))
    assert r.status == 'running' and c.phase == 'approach'
    assert not any(x.get('kind') in ('contact', 'gripper') for x in r.controls)


def test_planning_failure_and_missing_attachment():
    c, m = make_controller()
    c.step(.1, packet(c))
    m.outcome = 'failed'
    assert c.step(.1, packet(c, 1.1)).status == 'failed'
    c, m = make_controller()
    t = run_to(c, 'grasp')
    c.step(.1, packet(c, t))
    m.attach_ok = False
    assert 'attachment' in c.step(.1, packet(c, t+.1, contact=True)).reason


def test_failed_verification_never_credits_stock():
    c, m = make_controller(phase_timeout=2)
    t = run_to(c, 'verify')
    r = c.step(2.1, packet(c, t))
    assert r.status == 'failed' and 'timeout' in r.reason
    assert len(r.evidence) == 1


def test_direct_place_verify_succeeds_when_carton_rests_on_deck():
    # fm389: with a measured carton pose threaded (s.placed_pose), the collect verify phase proves the
    # carton rests on the SENSED support directly (controlled-asset ground truth) -- the cart destination
    # has no post-release camera. target_surface = packet() target (.5,.2,.1); a 0.10-tall carton resting
    # on it has centre z = .1 + .05. Two distinct on-deck frames -> succeeded, with labelled evidence.
    c, m = make_controller('collect')
    t = run_to(c, 'verify')
    on = (.5, .2, .15, 0., 0., 0., 1.)
    assert c.step(.1, packet(c, t, placed_pose=on)).status == 'running'
    r = c.step(.1, packet(c, t + .1, placed_pose=on))
    assert r.status == 'succeeded'
    assert any(e.get('source') == 'ground_truth_place_check' for e in r.evidence)


def test_direct_place_verify_hard_fails_when_carton_fell_to_the_floor():
    # run-300e: carton released off the east edge, settled on the floor (bottom well BELOW the support
    # top). The direct gate must refuse the empty-gripper 'success' -- not let it propagate into couple.
    c, m = make_controller('collect')
    t = run_to(c, 'verify')
    c.target_surface = (.5, .2, .3, 0., 0., 0., 1.)       # a deck-height support
    floor = (.5, .2, .05, 0., 0., 0., 1.)                 # carton on the floor (bottom ~0, 0.3 below)
    r = None
    for i in range(4):
        r = c.step(.1, packet(c, t + i * .1, placed_pose=floor))
    assert r.status == 'failed' and 'place verification failed' in r.reason
    assert not any(e.get('source') == 'ground_truth_place_check' for e in c.evidence)


def test_direct_place_verify_hard_fails_when_carton_left_on_the_upper_shelf():
    # fm390 wants deck vs shelf vs floor distinguished. With the DECK as the target, a carton settled on
    # the upper SHELF (bottom well ABOVE the deck top) is a mis-place -> hard-fail, same as the floor case.
    c, m = make_controller('collect')
    t = run_to(c, 'verify')
    c.target_surface = (.5, .2, .225, 0., 0., 0., 1.)     # deck top
    shelf = (.5, .2, 1.0, 0., 0., 0., 1.)                 # carton on the upper shelf (bottom ~0.95)
    r = None
    for i in range(4):
        r = c.step(.1, packet(c, t + i * .1, placed_pose=shelf))
    assert r.status == 'failed' and 'place verification failed' in r.reason


def test_placed_pose_absent_keeps_the_camera_two_frame_verify_byte_identical():
    # Unset RESTOCK_VERIFY_PLACE => placed_pose None => the existing sensor verify path is untouched.
    c, m = make_controller('collect')
    t = run_to(c, 'verify')
    assert c.step(.1, packet(c, t)).status == 'running'   # no placed_pose, no ground-truth branch taken
    assert not any(x[0] == 'place_gt' for x in c.verify_ids)


def test_ambiguous_duplicate_and_unreachable_source_wait_without_motion():
    for mutate in (lambda o: (replace(o, confidence=.2),), lambda o: (o,o),
                   lambda o: (replace(o, pose=(3.,0.,0.,0.,0.,0.,1.)),)):
        c, m = make_controller()
        s = packet(c)
        assert c.step(.1, replace(s, observations=mutate(s.observations[0]))).status == 'running'
        assert not m.moves


def test_target_move_tolerated_during_locked_reach_but_revalidated_at_preplace():
    # Observe-then-execute lock (key approach-sensing-occlusion): the placement target is NOT
    # re-validated from the fixed camera during the locked reach (approach/contact/lift) -- the
    # reaching arm occludes it, so a changed/absent target there is expected, not a fault. It IS
    # re-validated again at preplace, once lift has cleared the view.
    c, m = make_controller()
    c.step(.1, packet(c))
    assert c.phase == 'approach'
    moved = replace(packet(c, 1.1).target, pose=(.7, .2, .1, 0., 0., 0., 1.))
    assert c.step(.1, replace(packet(c, 1.1), target=moved)).status == 'running'  # tolerated while locked
    # Fresh controller (clean clock): the target IS re-validated again at preplace, after lift clears.
    c2, _ = make_controller()
    t = run_to(c2, 'preplace')
    s = replace(packet(c2, t), target=replace(packet(c2, t).target, pose=(.7, .2, .1, 0., 0., 0., 1.)))
    assert c2.step(.1, s).reason == 'target clearance lost'                        # re-validated after lift


def test_replayed_sensor_time_and_invalid_configuration():
    c, _ = make_controller()
    c.step(.1, packet(c, 2))
    assert c.step(.1, packet(c, 1)).status == 'failed'
    with pytest.raises(ValueError):
        c.start('transfer', dict(JOB, quantity=2))


def trajectory(points=None, names=JOINTS):
    return NS(joint_names=names, points=points or [
        NS(positions=[0.]*6, velocities=[], time_from_start=NS(sec=0,nanosec=0)),
        NS(positions=[.1]*6, velocities=[0.]*6, time_from_start=NS(sec=1,nanosec=0))])


def test_trajectory_validates_all_waypoints():
    traj = trajectory()
    validate_trajectory(traj, [0.]*6, [(-1,1)]*6)
    traj.points[-1].positions[2] = 2
    with pytest.raises(ValueError, match='limits'):
        validate_trajectory(traj, [0.]*6, [(-1,1)]*6)


@pytest.mark.parametrize('mutation', [
    lambda t: setattr(t, 'joint_names', ['wrong']*6),
    lambda t: setattr(t.points[0], 'positions', [.2]*6),
    lambda t: setattr(t.points[-1], 'velocities', [.1]*6),
    lambda t: setattr(t.points[-1], 'time_from_start', NS(sec=0,nanosec=0)),
    lambda t: setattr(t.points[-1], 'positions', [float('nan')]*6),
])
def test_bad_trajectory(mutation):
    t = trajectory()
    mutation(t)
    with pytest.raises(ValueError):
        validate_trajectory(t, [0.]*6, [(-1,1)]*6)


def test_ros_isolation(monkeypatch):
    for domain in ('0', '42', '83', ''):
        monkeypatch.setenv('ROS_DOMAIN_ID', domain)
        with pytest.raises(RuntimeError):
            require_sim_domain()
    monkeypatch.setenv('ROS_DOMAIN_ID', '88')
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    monkeypatch.delenv('ROS_STATIC_PEERS', raising=False)
    require_sim_domain()


class FixturePoseBridge:
    """Test pinhole math only; separate explicit upstream test below."""
    CameraIntrinsics = staticmethod(lambda *k: k)

    @staticmethod
    def backproject_mask(mask, depth, k):
        v, u = np.nonzero(mask)
        fx, fy, cx, cy = k
        z = depth[v,u]
        return np.column_stack(((u-cx)*z/fx, (v-cy)*z/fy, z))

    @staticmethod
    def estimate_pose(points, min_points):
        return NS(translation=points.mean(0), normal=np.array([0.,0.,1.]))


def depth_args():
    return dict(depth_m=np.full((10,10), .5), mask=np.ones((10,10), bool),
                intrinsics=(100.,100.,5.,5.), base_from_camera=np.eye(4),
                classification=([('sku-box', .8)], 'OK', .8, .1),
                observation_id='rgbd-1', simulation_time=1., location='rollwagen',
                rgb_time=1., depth_time=1., transform_time=1.)


def test_depth_pose_and_uncertainty():
    p = DepthPerception(FixturePoseBridge())
    args = depth_args()
    obs = p.observe(**args)
    assert obs.source == 'sensor' and obs.pose[2] == pytest.approx(.5)
    for change in ({'classification':([], 'UNKNOWN', .2, .01)}, {'depth_time':.8},
                   {'depth_m':np.zeros((10,10))}, {'base_from_camera':np.zeros((4,4))}):
        with pytest.raises(ValueError):
            p.observe(**dict(args, **change))


def test_target_depth_coverage_and_occupancy():
    p = DepthPerception(FixturePoseBridge())
    args = depth_args()
    for k in ('classification', 'rgb_time', 'depth_time', 'transform_time'):
        args.pop(k)
    args.update(support_z=.5, free_dimensions=(.3,.3,.4))
    assert p.target(**args).clear
    depth = args['depth_m'].copy()
    depth[0,0] = .6
    assert not p.target(**dict(args, depth_m=depth)).clear
    depth[:] = float('nan')
    with pytest.raises(ValueError):
        p.target(**dict(args, depth_m=depth))


def test_actual_ros_message_request_and_async_progression(monkeypatch):
    """Uses installed message types but fake transport: no ROS graph or GPU."""
    pytest.importorskip('rclpy')
    from concurrent.futures import Future
    from moveit_msgs.msg import RobotTrajectory
    from moveit_msgs.srv import GetPositionIK
    from trajectory_msgs.msg import JointTrajectoryPoint
    from scripts.twin.restock_manipulation.ros_adapter import RosMotionPort
    import rclpy.action

    monkeypatch.setenv('ROS_DOMAIN_ID', '88')
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    monkeypatch.delenv('ROS_STATIC_PEERS', raising=False)

    def ready(value):
        f = Future()
        f.set_result(value)
        return f

    class Handle:
        accepted = True
        cancelled = False
        def __init__(self, result):
            self.result = result
        def get_result_async(self):
            return ready(self.result)
        def cancel_goal_async(self):
            self.cancelled = True
            return ready(None)

    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = list(JOINTS)
    for index in (0, 1):
        point = JointTrajectoryPoint()
        point.positions = [float(index)*.1]*6
        point.time_from_start.sec = index
        trajectory.joint_trajectory.points.append(point)
    plan_result = NS(status=4, result=NS(error_code=NS(val=1), planned_trajectory=trajectory))
    execute_result = NS(status=4, result=NS(error_code=NS(val=1)))
    clients = []

    class Client:
        def __init__(self, node, action_type, name):
            self.name, self.goals = name, []
            clients.append(self)
        def server_is_ready(self):
            return True
        def send_goal_async(self, goal):
            self.goals.append(goal)
            return ready(Handle(plan_result if self.name.endswith('move_action') else execute_result))

    monkeypatch.setattr(rclpy.action, 'ActionClient', Client)
    response = GetPositionIK.Response()
    response.error_code.val = 1
    response.solution.joint_state.name = list(JOINTS)
    response.solution.joint_state.position = [.1]*6
    requests = []
    def request(req):
        requests.append(req)
        return ready(response)
    from builtin_interfaces.msg import Time
    node = NS(context=NS(get_domain_id=lambda:88),
              get_parameter=lambda name:NS(value=True),
              get_clock=lambda:NS(now=lambda:NS(to_msg=lambda:Time())),
              create_client=lambda *args:NS(service_is_ready=lambda:True, call_async=request))
    stopped = []
    port = RosMotionPort(node, joint_limits=[(-1,1)]*6,
                         payload_sync=lambda *args:False, emergency_stop=lambda:stopped.append(True))
    port.begin((.4,0.,.2,1.,0.,0.,0.), (0.,)*6)
    for _ in range(6):
        status = port.poll()
        if status == 'succeeded':
            break
    assert status == 'succeeded'
    assert requests[0].ik_request.avoid_collisions
    assert clients[0].goals[0].planning_options.plan_only
    assert list(clients[0].goals[0].request.start_state.joint_state.position) == [0.]*6
    assert clients[0].goals[0].request.pipeline_id == 'isaac_ros_cumotion'
    assert len(clients[1].goals) == 1
    assert not port.update_payload(True, PROFILE)
    port.stop()
    assert stopped


def _respawn_adapter(monkeypatch, *, first_wp, live_pos, limits=None):
    """fm368 harness: a RosMotionPort with fake transport whose plan returns a trajectory STARTING at
    first_wp (rad, all joints), and a live_arm_state reading live_pos (None disables it -> real-rig
    path). Models a post-move_group-respawn plan whose start drifted from the recorded self.start.
    Returns (port, stopped)."""
    pytest.importorskip('rclpy')
    from concurrent.futures import Future
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from scripts.twin.restock_manipulation.ros_adapter import RosMotionPort
    import rclpy.action

    monkeypatch.setenv('ROS_DOMAIN_ID', '88')
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    monkeypatch.delenv('ROS_STATIC_PEERS', raising=False)

    def ready(value):
        f = Future(); f.set_result(value); return f

    class Handle:
        accepted = True
        cancelled = False
        def __init__(self, result):
            self.result = result
        def get_result_async(self):
            return ready(self.result)
        def cancel_goal_async(self):
            self.cancelled = True
            return ready(None)

    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = list(JOINTS)
    for index, val in enumerate((first_wp, first_wp + 0.02)):
        point = JointTrajectoryPoint()
        point.positions = [float(val)] * 6
        point.time_from_start.sec = index
        trajectory.joint_trajectory.points.append(point)
    plan_result = NS(status=4, result=NS(error_code=NS(val=1), planned_trajectory=trajectory))
    execute_result = NS(status=4, result=NS(error_code=NS(val=1)))

    class Client:
        def __init__(self, node, action_type, name):
            self.name = name
        def server_is_ready(self):
            return True
        def send_goal_async(self, goal):
            return ready(Handle(plan_result if self.name.endswith('move_action') else execute_result))

    monkeypatch.setattr(rclpy.action, 'ActionClient', Client)
    from builtin_interfaces.msg import Time
    node = NS(context=NS(get_domain_id=lambda: 88),
              get_parameter=lambda name: NS(value=True),
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=0, to_msg=lambda: Time())),
              create_client=lambda *args: NS(service_is_ready=lambda: True,
                                             call_async=lambda req: ready(None)),
              create_publisher=lambda *a, **k: NS(publish=lambda *a, **k: None))
    stopped = []
    live = None
    if live_pos is not None:
        live = lambda: ({j: float(live_pos) for j in JOINTS},
                        {j: 0.0 for j in JOINTS}, True)
    port = RosMotionPort(node, joint_limits=(limits or [(-1, 1)] * 6),
                         payload_sync=lambda *a: True,
                         emergency_stop=lambda: stopped.append(True),
                         live_arm_state=live)
    return port, stopped


def _drive_to_terminal(port):
    status = None
    for _ in range(8):
        status = port.poll()
        if status in ('succeeded', 'failed'):
            break
    return status


def test_move_group_respawn_start_drift_recovers_against_live_state(monkeypatch, capsys):
    # fm368: a post-respawn plan whose first waypoint is the arm's TRUE pose (0.2) differs from the stale
    # recorded start (0.0) by >0.05 rad. With a live measured state matching the plan start, the move
    # RECOVERS (re-anchors self.start, settles, executes) instead of latching emergency_stop (run-295).
    port, stopped = _respawn_adapter(monkeypatch, first_wp=0.2, live_pos=0.2)
    port.begin_joint_goal([0.3] * 6, [0.0] * 6)
    assert _drive_to_terminal(port) == 'succeeded'
    assert not stopped
    assert port.start == pytest.approx((0.2,) * 6)
    assert 'MOVE_GROUP_RESPAWN_RESEED' in capsys.readouterr().out


def test_move_group_respawn_drift_hard_fails_without_live_state(monkeypatch):
    # Byte-identical real-rig path: no live_arm_state -> the start-drift validate raises as before ->
    # poll's except latches emergency_stop and the move fails. run-295's actual behaviour, UNCHANGED.
    port, stopped = _respawn_adapter(monkeypatch, first_wp=0.2, live_pos=None)
    port.begin_joint_goal([0.3] * 6, [0.0] * 6)
    assert _drive_to_terminal(port) == 'failed'
    assert stopped


def test_move_group_respawn_recovery_still_hard_fails_bad_plan(monkeypatch):
    # The recovery is start-drift ONLY: an out-of-limits waypoint still hard-fails even WITH a live state,
    # so a collision/garbage plan is never silently accepted.
    port, stopped = _respawn_adapter(monkeypatch, first_wp=2.0, live_pos=2.0)   # 2.0 outside (-1, 1)
    port.begin_joint_goal([0.3] * 6, [0.0] * 6)
    assert _drive_to_terminal(port) == 'failed'
    assert stopped


def test_move_group_respawn_recovery_rejects_start_differing_from_live_too(monkeypatch):
    # Not a blanket accept: if the plan start differs from the LIVE state as well (0.2 vs 0.9), the
    # re-validate against live still raises start-drift -> the move hard-fails.
    port, stopped = _respawn_adapter(monkeypatch, first_wp=0.2, live_pos=0.9)
    port.begin_joint_goal([0.3] * 6, [0.0] * 6)
    assert _drive_to_terminal(port) == 'failed'
    assert stopped


def test_move_group_respawn_happy_path_untouched(monkeypatch, capsys):
    # Byte-identical happy path: plan start already matches self.start -> validate(self.start) passes and
    # the live-state fallback never engages (no RESPAWN_RESEED print), the move succeeds.
    port, stopped = _respawn_adapter(monkeypatch, first_wp=0.0, live_pos=0.0)
    port.begin_joint_goal([0.3] * 6, [0.0] * 6)
    assert _drive_to_terminal(port) == 'succeeded'
    assert not stopped
    assert 'MOVE_GROUP_RESPAWN_RESEED' not in capsys.readouterr().out


def _settle_adapter(monkeypatch, *, vel, settle_wait_s=None, settle_vel=None, live_pos=0.1):
    """fm369/fm370 harness: a RosMotionPort whose fake plan resolves to a 1 s trajectory starting at the
    measured start (0.0, so validate passes) and whose live_arm_state reports every joint at ``live_pos``
    (the drifted pose the reseed must adopt) moving at ``vel`` rad/s. The node sim clock is a MUTABLE
    container the test advances to drive the settle sub-budget. The sim knobs reach the port via ctor
    params (env unset => env_wins keeps the passed value). Returns (port, clk, trajectory)."""
    pytest.importorskip('rclpy')
    from concurrent.futures import Future
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from scripts.twin.restock_manipulation.ros_adapter import RosMotionPort
    import rclpy.action

    monkeypatch.setenv('ROS_DOMAIN_ID', '88')
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    monkeypatch.delenv('ROS_STATIC_PEERS', raising=False)
    monkeypatch.delenv('RESTOCK_SETTLE_WAIT_S', raising=False)
    monkeypatch.delenv('RESTOCK_SETTLE_VEL', raising=False)

    def ready(value):
        f = Future(); f.set_result(value); return f

    class Handle:
        accepted = True
        cancelled = False
        def __init__(self, result):
            self.result = result
        def get_result_async(self):
            return ready(self.result)
        def cancel_goal_async(self):
            self.cancelled = True
            return ready(None)

    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = list(JOINTS)
    for index in range(2):
        point = JointTrajectoryPoint()
        point.positions = [0.0] * 6               # planned start == measured start 0.0 -> validate passes
        point.time_from_start.sec = index         # 1 s trajectory -> exec budget (1+25)*scale
        trajectory.joint_trajectory.points.append(point)
    plan_result = NS(status=4, result=NS(error_code=NS(val=1), planned_trajectory=trajectory))
    execute_result = NS(status=4, result=NS(error_code=NS(val=1)))

    class Client:
        def __init__(self, node, action_type, name):
            self.name = name
        def server_is_ready(self):
            return True
        def send_goal_async(self, goal):
            return ready(Handle(plan_result if self.name.endswith('move_action') else execute_result))

    monkeypatch.setattr(rclpy.action, 'ActionClient', Client)
    from builtin_interfaces.msg import Time
    clk = {'ns': 0}
    node = NS(context=NS(get_domain_id=lambda: 88),
              get_parameter=lambda name: NS(value=True),
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=clk['ns'], to_msg=lambda: Time())),
              create_client=lambda *args: NS(service_is_ready=lambda: True,
                                             call_async=lambda req: ready(None)),
              create_publisher=lambda *a, **k: NS(publish=lambda *a, **k: None))
    live = lambda: ({j: float(live_pos) for j in JOINTS}, {j: float(vel) for j in JOINTS}, True)
    kwargs = {}
    if settle_wait_s is not None:
        kwargs['settle_wait_s'] = settle_wait_s
    if settle_vel is not None:
        kwargs['settle_vel'] = settle_vel
    port = RosMotionPort(node, joint_limits=[(-1, 1)] * 6, payload_sync=lambda *a: True,
                         emergency_stop=lambda: None, live_arm_state=live, **kwargs)
    return port, clk, trajectory


def _advance_to_await_settle(port):
    for _ in range(8):
        port.poll()
        if port.state == 'await_settle':
            return True
    return False


def test_await_settle_besteffort_dispatches_after_subbudget(monkeypatch, capsys):
    # fm369: a non-settling arm (0.30 rad/s >> 0.05) with a 6 s settle sub-budget HOLDS until 6 sim-seconds
    # pass, then dispatches BEST-EFFORT (run-296 RETREAT would have otherwise run the ~26 s exec deadline
    # into a fault). The reseed still re-anchors the first waypoint to the live pose (0.1), and the exit
    # log names the besteffort path.
    port, clk, trajectory = _settle_adapter(monkeypatch, vel=0.30, settle_wait_s=6.0, live_pos=0.1)
    port.begin_joint_goal([0.3] * 6, [0.0] * 6)
    assert _advance_to_await_settle(port)
    clk['ns'] = int(3e9)                 # 3 s < 6 s sub-budget -> still holding
    assert port.poll() == 'running'
    assert port.state == 'await_settle'
    clk['ns'] = int(7e9)                 # 7 s > 6 s sub-budget (and < 26 s exec deadline) -> best-effort
    assert _drive_to_terminal(port) == 'succeeded'
    out = capsys.readouterr().out
    assert 'AWAIT_SETTLE_EXIT path=besteffort' in out
    assert trajectory.joint_trajectory.points[0].positions == pytest.approx((0.1,) * 6)  # reseeded to live


def test_await_settle_byte_identical_holds_when_subbudget_unset(monkeypatch, capsys):
    # Byte-identical real-rig: settle_wait_s defaults to +inf, so a non-settling arm HOLDS exactly as the
    # original fm357 behaviour (never best-effort), bounded only by the execute deadline as before.
    port, clk, _ = _settle_adapter(monkeypatch, vel=0.30, live_pos=0.1)   # no settle_wait_s -> inf default
    port.begin_joint_goal([0.3] * 6, [0.0] * 6)
    assert _advance_to_await_settle(port)
    clk['ns'] = int(10e9)                # 10 s (< 26 s exec deadline) -> still holding, no best-effort
    assert port.poll() == 'running'
    assert port.state == 'await_settle'
    assert 'AWAIT_SETTLE_EXIT' not in capsys.readouterr().out


def test_await_settle_vel_threshold_knob_counts_jitter_as_settled(monkeypatch, capsys):
    # fm370: with the settle-velocity threshold raised to 0.15, an arm jittering at 0.12 rad/s counts as
    # SETTLED on the NORMAL path (no best-effort, no hold) and dispatches immediately.
    port, clk, _ = _settle_adapter(monkeypatch, vel=0.12, settle_vel=0.15, live_pos=0.1)
    port.begin_joint_goal([0.3] * 6, [0.0] * 6)
    assert _drive_to_terminal(port) == 'succeeded'
    out = capsys.readouterr().out
    assert 'AWAIT_SETTLE_EXIT path=settled' in out
    assert 'path=besteffort' not in out


def test_await_settle_vel_default_still_holds_jitter(monkeypatch, capsys):
    # The 0.15 threshold is opt-in: at the real-rig default (0.05) the SAME 0.12 rad/s jitter is NOT
    # settled, so it holds (byte-identical) rather than dispatching.
    port, clk, _ = _settle_adapter(monkeypatch, vel=0.12, live_pos=0.1)   # default settle_vel 0.05
    port.begin_joint_goal([0.3] * 6, [0.0] * 6)
    assert _advance_to_await_settle(port)
    clk['ns'] = int(10e9)
    assert port.poll() == 'running'
    assert 'AWAIT_SETTLE_EXIT' not in capsys.readouterr().out


def test_await_settle_settled_arm_normal_path_logs_settled(monkeypatch, capsys):
    # A genuinely settled arm (0.0 rad/s) dispatches on the normal settled path and logs path=settled.
    port, clk, trajectory = _settle_adapter(monkeypatch, vel=0.0, settle_wait_s=6.0, live_pos=0.1)
    port.begin_joint_goal([0.3] * 6, [0.0] * 6)
    assert _drive_to_terminal(port) == 'succeeded'
    out = capsys.readouterr().out
    assert 'AWAIT_SETTLE_EXIT path=settled' in out
    assert trajectory.joint_trajectory.points[0].positions == pytest.approx((0.1,) * 6)  # reseeded to live


def test_settle_env_knobs_win_over_passed_value(monkeypatch):
    # fm369/fm370 env-win: env unset keeps the ctor-passed value (byte-identical real-rig default); env set
    # wins. Mirrors the fm363 env_wins contract (module default already reflects the env at import).
    from scripts.twin.restock_manipulation.ros_adapter import env_wins, SETTLE_VEL_DEFAULT
    monkeypatch.delenv('RESTOCK_SETTLE_VEL', raising=False)
    assert env_wins('RESTOCK_SETTLE_VEL', SETTLE_VEL_DEFAULT, SETTLE_VEL_DEFAULT) == pytest.approx(0.05)
    monkeypatch.setenv('RESTOCK_SETTLE_VEL', '0.15')
    assert env_wins('RESTOCK_SETTLE_VEL', SETTLE_VEL_DEFAULT, 0.15) == pytest.approx(0.15)
    monkeypatch.delenv('RESTOCK_SETTLE_WAIT_S', raising=False)
    assert env_wins('RESTOCK_SETTLE_WAIT_S', float('inf'), float('inf')) == float('inf')
    monkeypatch.setenv('RESTOCK_SETTLE_WAIT_S', '6')
    assert env_wins('RESTOCK_SETTLE_WAIT_S', float('inf'), 6.0) == pytest.approx(6.0)


def test_ensemble_calls_rgb_models_and_rejects_unknown():
    from scripts.twin.restock_manipulation.perception import EnsembleObservationPipeline
    bounds = np.array([[0,0,10,10]])
    detector = NS(predict=lambda *args, **kwargs:[NS(boxes=[NS(conf=.99, xyxy=bounds)])])
    classification = lambda crop: ([('sku-box', .8)], 'OK', .8, .1)
    pipeline = EnsembleObservationPipeline(detector, classification, DepthPerception(FixturePoseBridge()),
                                          lambda rgb, depth, bounds:np.ones(depth.shape, bool),
                                          lambda bounds:'rollwagen')
    args = depth_args()
    for key in ('classification','observation_id','location','mask'):
        args.pop(key)
    observations, rejected = pipeline.observe(np.zeros((10,10,3), np.uint8), frame_id='frame-1', **args)
    assert len(observations) == 1 and not rejected
    pipeline.classify_crop = lambda crop:([], 'UNKNOWN', .2, .01)
    observations, rejected = pipeline.observe(np.zeros((10,10,3), np.uint8), frame_id='frame-2', **args)
    assert not observations and 'uncertain SKU' in rejected[0]['reason']


def test_source_pose_change_tolerated_during_locked_approach():
    # Observe-then-execute lock: once the approach is planned the reaching arm occludes/degrades the
    # fixed source camera, so a changed or absent source observation during approach/contact is
    # EXPECTED occlusion, not a fault. The grasp proceeds; a genuine miss is caught at contact and by
    # the executor's joint-tracking/heartbeat e-stops, not by per-tick fixed-camera re-validation.
    c, _ = make_controller()
    c.step(.1, packet(c))
    assert c.phase == 'approach'
    s = packet(c, 1.1)
    moved = replace(s.observations[0], pose=(.45,0.,.2,0.,0.,0.,1.))
    assert c.step(.1, replace(s, observations=(moved,))).status == 'running'


def test_ros_packet_rejects_string_booleans_and_scan_coverage():
    import json
    from dataclasses import asdict
    from scripts.twin.restock_manipulation.ros_adapter import sensors_from_json
    c, _ = make_controller()
    data = asdict(packet(c))
    parsed = sensors_from_json(json.dumps(data))
    assert parsed.observations[0].sku == 'sku-box'
    for key, value in (('parked', 'false'), ('contact', 1), ('scanned_locations', 'rollwagen')):
        with pytest.raises(ValueError):
            sensors_from_json(json.dumps(dict(data, **{key:value})))


# ---------------------------------------------------------------------------
# fm372: close the whole "a collect move poll() can raise/fail and kill the mission" family.
# Every non-collision, non-joint-limit raise at any collect move phase -> bounded fresh-seed retry.
# Collisions (returned 'failed', not sim-deadline) and joint-limit raises still hard-fail. Bounded.
# ---------------------------------------------------------------------------

# The real move-poll collect phases: tuck/pan (+ stow) are begin_joint_goal (staged budget);
# preplace/place/retreat are begin (general / deadline budget). 'release' is a gripper detach, not a
# move poll, so it is not injectable here.
_FM372_MOVE_PHASES = ['tuck', 'pan', 'preplace', 'place', 'retreat', 'stow']
_FM372_RETRYABLE_EXC = [
    RuntimeError('planning failed (exhausted within wall cap)'),  # plan-exhaust
    ValueError('trajectory does not end at rest'),                # validate (bad stochastic seed)
    RuntimeError('execution failed'),                             # goal-tolerance / track miss
    RuntimeError('measured-feedback staleness watchdog'),         # feedback-lost
    Exception('an unforeseen move-path error'),                   # any other class
]


def _issue_move(c, m, phase, t0=1.0):
    """Drive a staged-swing controller to a move phase and ISSUE that move (begin/begin_joint_goal),
    so the NEXT step calls poll(). Returns the sim time for that first poll step."""
    t = run_to(c, phase, t0=t0)
    c.step(.1, packet(c, t))        # issue the move
    return t + .1


@pytest.mark.parametrize('phase', _FM372_MOVE_PHASES)
@pytest.mark.parametrize('exc', _FM372_RETRYABLE_EXC,
                         ids=lambda e: type(e).__name__ + '-' + str(e).split()[0])
def test_fm372_move_poll_raise_retries_every_class_every_phase(phase, exc, capsys):
    c, m = _staged_controller()
    tp = _issue_move(c, m, phase)
    m.outcome, m.raise_exc = 'raise', exc
    r = c.step(.1, packet(c, tp))                 # poll() raises -> bounded fresh-seed retry
    assert r.status != 'failed', (phase, exc, r.reason)
    assert c.phase == phase                       # stays in the phase, mission alive
    assert not c.issued                           # re-issue armed for the next tick
    out = capsys.readouterr().out
    assert 'MOVE_RETRY phase=%s cls=%s' % (phase, type(exc).__name__) in out


@pytest.mark.parametrize('phase', ['pan', 'retreat'])   # one staged, one general
def test_fm372_joint_limit_raise_hard_fails_not_retried(phase):
    for msg in ('waypoint outside joint limits', 'IK outside joint limits',
                'contact IK outside joint limits'):
        c, m = _staged_controller()
        tp = _issue_move(c, m, phase)
        m.outcome, m.raise_exc = 'raise', ValueError(msg)
        r = c.step(.1, packet(c, tp))
        assert r.status == 'failed', (phase, msg)
        assert 'joint-limit' in c.reason and m.stops >= 1


def test_fm372_collision_returned_failed_still_hard_fails_general():
    # A COLLISION is a RETURNED 'failed' (not sim-deadline, not a raise) -> fail-fast, unchanged.
    c, m = _staged_controller()
    tp = _issue_move(c, m, 'retreat')
    m.outcome, m.sim_deadline_fault = 'failed', False
    r = c.step(.1, packet(c, tp))
    assert r.status == 'failed'
    assert c.reason == 'collision planning or execution failed'


def test_fm372_sim_deadline_returned_failed_retries_general():
    # A sim-deadline fault is a RETURNED 'failed' flagged retriable -> re-issue (existing fm367 path).
    c, m = _staged_controller()
    tp = _issue_move(c, m, 'retreat')
    m.outcome, m.sim_deadline_fault = 'failed', True
    r = c.step(.1, packet(c, tp))
    assert r.status != 'failed'
    assert c.phase == 'retreat' and not c.issued


def test_fm372_persistent_raise_exhausts_then_fails_bounded(monkeypatch):
    # A persistent retryable raise must EXHAUST the (env-tunable) max and _fail -- never loop forever.
    monkeypatch.setenv('RESTOCK_MOVE_DEADLINE_RETRY_MAX', '2')
    c, m = _staged_controller()
    tp = _issue_move(c, m, 'retreat')                       # general path -> deadline budget
    m.outcome, m.raise_exc = 'raise', ValueError('trajectory does not end at rest')
    t, failed = tp, False
    for _ in range(40):                                     # bounded: must fail well within this
        r = c.step(.1, packet(c, t)); t += .1
        if r.status == 'failed':
            failed = True
            break
    assert failed and 'does not end at rest' in c.reason
    assert c._move_deadline_retries == 2                    # exactly the env max, then fail


def test_fm372_bounded_retry_recovers_after_transient_raises(monkeypatch):
    monkeypatch.setenv('RESTOCK_STAGED_MOVE_RETRY_MAX', '4')
    c, m = _staged_controller()
    tp = _issue_move(c, m, 'pan')                           # staged path
    m.outcome, m.raise_exc, m.raise_times = 'raise', ValueError('trajectory does not end at rest'), 2
    t = tp
    for _ in range(12):
        if c.phase == 'preplace':
            break
        c.step(.1, packet(c, t)); t += .1
    assert c.phase == 'preplace'                            # 2 raises (within 4) then succeeded


def test_fm372_retry_max_env_tunable(monkeypatch):
    monkeypatch.setenv('RESTOCK_STAGED_MOVE_RETRY_MAX', '4')
    monkeypatch.setenv('RESTOCK_MOVE_DEADLINE_RETRY_MAX', '4')
    c, _ = _staged_controller()
    assert c._staged_move_retry_max == 4 and c._move_deadline_retry_max == 4


def test_fm372_retry_max_default_byte_identical():
    c, _ = _staged_controller()
    assert c._staged_move_retry_max == 2 and c._move_deadline_retry_max == 2


def test_fm372_retry_max_env_clamped_and_malformed_safe(monkeypatch):
    monkeypatch.setenv('RESTOCK_STAGED_MOVE_RETRY_MAX', '999')
    monkeypatch.setenv('RESTOCK_MOVE_DEADLINE_RETRY_MAX', 'not-an-int')
    c, _ = _staged_controller()
    assert c._staged_move_retry_max == 8          # clamped to [0, 8]
    assert c._move_deadline_retry_max == 2        # malformed -> module default
