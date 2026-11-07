"""sim-motion-active-guard-v1: serialize every re-issued arm goal behind the active motion's terminal.

Twice a recovery path that re-issues an arm goal CRASHED the host instead of recovering, both with the same
``RuntimeError: motion already active`` from the motion port's in-flight guard, surfaced by the controller's
generic adapter-failure handler as a ``HOST_FAULT_ESTOP``:

  * coldwin73 -- the carried-preplace DRIFT RE-CAPTURE (sim-preplace-wrist2-hold-v1) re-issued the preplace
    joint goal while the preplace move was still in flight.
  * coldwin90 -- the place set-down RE-APPROACH (sim-place-release-on-contact-v1), after the seat gate
    correctly refused an obstructed descent (PLACE_OBSTRUCTED, bounded re-approach 1/2), re-descended via
    the pre-place standoff while the obstructed place descent was still in flight.

The fix is ONE serialization helper (``TransferController._serialize_reissue``) beneath both opt-in paths:
before a re-issued goal it brings the active motion to a TERMINAL (not-in-flight) state on the channel that
actually terminates -- the PR 368 contact-hold stop for a driven descent, a PR 334 cancel + bounded
wait-terminal for an action goal -- confirms the handle is terminal, then begins; if it does not terminate
within a bounded sim window it FAILS BY NAME (``REISSUE_MOTION_STILL_ACTIVE``) with the handle state, never
re-issuing into an in-flight motion. A recovery is a real second attempt, never a crash.

CPU-only: no Isaac / ROS / GPU.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_motion_active_guard_v1.py
"""
import os
from types import SimpleNamespace as NS

import pytest

from twin.restock_manipulation import ProductProfile, TransferController
from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.restock_manipulation.controller import _REISSUE_TERMINAL_STATES

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='dataset mobile URDF not present')

PROFILE = ProductProfile('sku-box', (.10, .10, .10), (1., 0., 0., 0.))
IN_FLIGHT = 'execute_result'          # a representative ros_adapter._PLAN_IN_FLIGHT_STATES member


# ===================================================================== stub motion ports (no ROS / no GPU)

class GuardedMotion:
    """A motion port that MODELS the real RosMotionPort in-flight guard (begin*() raises 'motion already
    active' unless the state is terminal) AND the cancel/contact-hold serialization protocol the controller
    drives to drain an in-flight move. ``settle_ticks`` scripts how many cancel_outcome() polls return None
    before the move settles terminal (0 => settles on the first poll); ``never_settles`` keeps it in flight
    forever (the bounded-fail path)."""

    def __init__(self, *, state=IN_FLIGHT, settle_ticks=0, never_settles=False, settled_outcome='cancelled'):
        self.state = state
        self.begins = []                     # (kind, goal) for every begin / begin_joint_goal / begin_joint_path
        self.cancels = 0
        self.settles = 0
        self.outcome_polls = 0
        self._cancel_requested = False
        self._settle_ticks = int(settle_ticks)
        self._never_settles = bool(never_settles)
        self._settled_outcome = settled_outcome
        self.stops = 0
        self.payloads = []

    # --- the moves the controller issues (the real in-flight guard) ---
    def _guard(self):
        if self.state not in _REISSUE_TERMINAL_STATES:
            raise RuntimeError('motion already active')

    def begin(self, pose, joints, **k):
        self._guard()
        self.state = IN_FLIGHT
        self.begins.append(('begin', tuple(pose)))

    def begin_joint_goal(self, goal, measured, **k):
        self._guard()
        self.state = IN_FLIGHT
        self.begins.append(('joint_goal', tuple(goal)))

    def begin_joint_path(self, waypoints, measured, **k):
        self._guard()
        self.state = IN_FLIGHT
        self.begins.append(('joint_path', None))

    def poll(self):
        return 'running'

    def last_fault_was_sim_deadline(self):
        return False

    def take_exec_abort_reason(self):
        return None

    def reset_crash_budget(self):
        pass

    # --- the serialization contract ---
    def cancel(self):
        self.cancels += 1
        self._cancel_requested = True

    def settle_in_place(self):
        self.settles += 1
        self._cancel_requested = True

    def cancel_outcome(self):
        self.outcome_polls += 1
        if not self._cancel_requested:
            return None
        if self._never_settles:
            return None
        if self._settle_ticks > 0:
            self._settle_ticks -= 1
            return None
        self.state = 'succeeded' if self._settled_outcome == 'succeeded' else 'failed'
        self._cancel_requested = False
        return self._settled_outcome

    def stop(self):
        self.stops += 1
        self.state = 'failed'

    def update_payload(self, held, profile):
        self.payloads.append((held, self.state))
        return True


class _ErrorMotion(GuardedMotion):
    def cancel_outcome(self):
        self.outcome_polls += 1
        if not self._cancel_requested:
            return None
        self.state = 'failed'
        self._cancel_requested = False
        return 'error'


class _NoCancelMotion:
    """A motion port with NO cancel protocol at all (the _serialize_reissue 'not cancellable' fail path)."""

    def __init__(self, state=IN_FLIGHT):
        self.state = state


# ===================================================================== A) the helper _serialize_reissue

def _guard_host(motion, *, timeout_s=2.0, phase='preplace'):
    """A bare host exposing exactly what _serialize_reissue / _reissue_reset / _reissue_fail_reason touch,
    so the helper runs on CPU with no controller construction."""
    c = NS()
    c.motion = motion
    c.phase = phase
    c._reissue_stop_issued = False
    c._reissue_sim_start = None
    c.reissue_settle_timeout_s = timeout_s
    c._serialize_reissue = TransferController._serialize_reissue.__get__(c, TransferController)
    c._reissue_reset = TransferController._reissue_reset.__get__(c, TransferController)
    c._reissue_fail_reason = TransferController._reissue_fail_reason.__get__(c, TransferController)
    return c


def test_helper_ready_when_already_terminal():
    """A port already terminal (nothing to stop) -> 'ready' on the first call, no cancel/settle issued."""
    for state in _REISSUE_TERMINAL_STATES:
        m = GuardedMotion(state=state)
        c = _guard_host(m)
        verdict, payload = c._serialize_reissue(0.0, driven=False)
        assert verdict == 'ready' and payload == ()
        assert m.cancels == 0 and m.settles == 0


def test_helper_action_goal_terminates_after_cancel():
    """An in-flight ACTION goal: the first call cancels (no contact control) and holds; once the cancel
    reaches a terminal result the next call is 'ready'."""
    m = GuardedMotion(settle_ticks=0)
    c = _guard_host(m)
    v1, p1 = c._serialize_reissue(0.0, driven=False)
    assert v1 == 'hold' and p1 == ()                         # action-cancel emits no control
    assert m.cancels == 1 and m.settles == 0
    v2, p2 = c._serialize_reissue(0.1, driven=False)
    assert v2 == 'ready' and p2 == ()
    assert m.state in _REISSUE_TERMINAL_STATES


def test_helper_driven_descent_terminates_on_contact_hold():
    """An in-flight DRIVEN descent: the stop is the PR 368 contact-hold (settle_in_place + a {'kind':'contact'}
    control so trajectory_host succeeds), NOT an ExecuteTrajectory cancel; it drains to terminal."""
    m = GuardedMotion(settle_ticks=0, settled_outcome='succeeded')
    c = _guard_host(m)
    v1, p1 = c._serialize_reissue(0.0, driven=True)
    assert v1 == 'hold' and p1 == ({'kind': 'contact'},)     # the contact-hold control
    assert m.settles == 1 and m.cancels == 0                  # settle_in_place, never cancel()
    v2, p2 = c._serialize_reissue(0.1, driven=True)
    assert v2 == 'ready'
    assert m.state in _REISSUE_TERMINAL_STATES


def test_helper_holds_then_fails_by_name_when_never_terminal():
    """The motion never terminates -> after the bounded sim window the helper FAILS BY NAME
    (REISSUE_MOTION_STILL_ACTIVE) and names the handle state; it never returns 'ready'."""
    m = GuardedMotion(never_settles=True)
    c = _guard_host(m, timeout_s=0.25)
    v, _ = c._serialize_reissue(0.0, driven=False)
    assert v == 'hold'                                        # stop issued, settling
    last = None
    for t in (0.1, 0.2, 0.3, 0.4):
        last = c._serialize_reissue(t, driven=False)
        if last[0] == 'fail':
            break
    verdict, reason = last
    assert verdict == 'fail'
    assert reason.startswith('REISSUE_MOTION_STILL_ACTIVE')
    assert 'handle state=%s' % IN_FLIGHT in reason            # the handle state is named
    assert m.begins == []                                     # never re-issued into the in-flight motion


def test_helper_fails_by_name_on_broken_terminal():
    """A cancel that resolves to a broken ('error') terminal fails by name, not 'ready'."""
    m = _ErrorMotion()
    c = _guard_host(m)
    assert c._serialize_reissue(0.0, driven=False)[0] == 'hold'
    verdict, reason = c._serialize_reissue(0.1, driven=False)
    assert verdict == 'fail' and reason.startswith('REISSUE_MOTION_STILL_ACTIVE')
    assert 'broken result' in reason


def test_helper_fails_by_name_when_port_not_cancellable():
    """A port with no cancel protocol cannot be serialized -> fail by name rather than risk the begin
    (never silently re-issue into an in-flight motion)."""
    c = _guard_host(_NoCancelMotion())
    verdict, reason = c._serialize_reissue(0.0, driven=True)
    assert verdict == 'fail' and reason.startswith('REISSUE_MOTION_STILL_ACTIVE')
    assert 'not cancellable' in reason


def test_helper_resets_substate_on_every_terminal_verdict():
    """After 'ready' or 'fail' the shared cancel->settle sub-state is reset, so the next re-issue starts
    fresh (no stale stop_issued leaks a second re-issue into a hold)."""
    m = GuardedMotion(settle_ticks=0)
    c = _guard_host(m)
    c._serialize_reissue(0.0, driven=False)                  # stop issued
    assert c._reissue_stop_issued is True
    c._serialize_reissue(0.1, driven=False)                  # -> ready
    assert c._reissue_stop_issued is False and c._reissue_sim_start is None


# ===================================================================== B) coldwin73 drift re-capture replay

W2 = ARM_JOINTS.index('ur_wrist_2_joint')
CMD = 0.6678


def _goal(w2=CMD):
    g = [0.0] * 6
    g[W2] = w2
    return tuple(g)


def _meas(w2):
    m = [0.0] * 6
    m[W2] = w2
    return tuple(m)


def _sensors(w2):
    return NS(joint_positions=_meas(w2))


def _recapture_controller(motion):
    c = TransferController(motion, {'sku-box': PROFILE}, fixture_mode=True,
                           preplace_convergence_stall_s=30.0,
                           preplace_drift_recapture_rad=0.05, preplace_drift_recapture_max=3)
    c.start('collect', dict(job_id='j', sku='sku-box', quantity=1, source_bin='b', shelf_id='s'))
    c.phase, c.issued = 'preplace', True
    c._preplace_recapture_logged = False
    c._preplace_convergence.arm(_goal())
    return c


def _drive_to_drift(c):
    c._handle_preplace_convergence(0.0, _sensors(0.0))        # capture start
    for t in (1.0, 2.0, 3.0):
        c._handle_preplace_convergence(t, _sensors(0.667))    # converge (closest approach)


def test_cw73_reissue_would_crash_without_serialization():
    """Proof of the coldwin73 mechanism: re-issuing begin_joint_goal directly while the move is IN FLIGHT
    raises the exact 'motion already active' the host handler e-stopped on."""
    m = GuardedMotion(state=IN_FLIGHT)
    with pytest.raises(RuntimeError, match='motion already active'):
        m.begin_joint_goal(_goal(), _meas(0.0))


def test_cw73_drift_recapture_serializes_then_reissues_cleanly():
    """With the fix: the drift re-capture first drains the in-flight preplace move (cancel + wait-terminal),
    THEN re-issues the preplace goal -- a real second attempt, no 'motion already active' crash."""
    m = GuardedMotion(state=IN_FLIGHT, settle_ticks=0)
    c = _recapture_controller(m)
    _drive_to_drift(c)
    # tick 4: DriftRecapture fires while the preplace move is in flight -> serialize (cancel), hold, NO begin.
    handled, res = c._handle_preplace_convergence(4.0, _sensors(-0.326))
    assert handled and res.status == 'running'
    assert c._recapture_reissue_pending is True
    assert m.cancels == 1 and m.begins == []                  # stop issued, nothing re-issued yet
    # tick 5: the cancel settled terminal -> the preplace goal is re-issued cleanly (no raise).
    handled, res = c._handle_preplace_convergence(4.1, _sensors(-0.33))
    assert handled
    assert c._recapture_reissue_pending is False
    assert len(m.begins) == 1 and m.begins[0][0] == 'joint_goal'
    assert m.begins[0][1][W2] == pytest.approx(CMD)           # re-driven toward the armed wrist_2 target
    assert c.issued is True


def test_cw73_drift_recapture_never_reobserves_while_settling():
    """While the stop is settling the guard is NOT re-observed, so the per-joint re-capture budget is spent
    exactly once per episode (no runaway re-capture while waiting for the terminal)."""
    m = GuardedMotion(state=IN_FLIGHT, settle_ticks=3)        # three settling ticks before terminal
    c = _recapture_controller(m)
    _drive_to_drift(c)
    c._handle_preplace_convergence(4.0, _sensors(-0.326))     # fire + issue the cancel
    recaptures = sum(1 for e in c.evidence if 'phase_drift_recapture' in e)
    for t in (4.1, 4.2, 4.3):                                 # settling: held, NOT re-observed
        handled, res = c._handle_preplace_convergence(t, _sensors(-0.33))
        assert handled and res.status == 'running' and m.begins == []
    assert sum(1 for e in c.evidence if 'phase_drift_recapture' in e) == recaptures
    # terminal reached -> the single re-issue fires.
    c._handle_preplace_convergence(4.4, _sensors(-0.33))
    assert len(m.begins) == 1


def test_cw73_drift_recapture_fails_by_name_if_move_never_terminates():
    """If the in-flight preplace move never terminates, the re-capture fails by name
    (REISSUE_MOTION_STILL_ACTIVE) rather than crashing or re-issuing blindly."""
    m = GuardedMotion(state=IN_FLIGHT, never_settles=True)
    c = _recapture_controller(m)
    c.reissue_settle_timeout_s = 0.25
    _drive_to_drift(c)
    c._handle_preplace_convergence(4.0, _sensors(-0.326))     # cancel issued
    res = None
    for t in (4.1, 4.2, 4.3, 4.4):
        _handled, res = c._handle_preplace_convergence(t, _sensors(-0.33))
        if res.status == 'failed':
            break
    assert res.status == 'failed'
    assert 'REISSUE_MOTION_STILL_ACTIVE' in c.reason
    assert m.begins == []


# ===================================================================== C) coldwin90 place re-approach replay

POST = '/Rollwagen/chassis/post_2'
_SL_STOW = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
_SL_TUCK = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
_SL_PREPLACE_SHAPE = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)


def _packet(c, t, *, contact, placed_pose, seal_gap_mm=None, seal_hit_prims=None):
    import time
    from twin.restock_manipulation import Observation, Sensors, Target
    observation = Observation(f'src-{t}', 'test_fixture', t, 'sku-box', c.source,
                              (.4, 0., .2, 0., 0., 0., 1.), .99)
    target = Target(f'tgt-{t}', 'test_fixture', t, c.destination, (.5, .2, .1, 0., 0., 0., 1.),
                    True, (.3, .3, .4))
    return Sensors(simulation_time=t, observations=(observation,), target=target, parked=True,
                   contact=contact, feedback_time=t, collision_time=t, joint_positions=(0.,) * 6,
                   scanned_locations=(c.source, c.destination), scan_time=t,
                   valid_until_steady=time.monotonic() + 100., placed_pose=placed_pose,
                   seal_gap_mm=seal_gap_mm, seal_hit_prims=tuple(seal_hit_prims or ()))


def _place_controller(motion, reapproach_max=2):
    lc = cl.resolve_level_config({'place_release_on_contact': True, 'place_reapproach_max': reapproach_max,
                                  'place_reapproach_shift_m': 0.03, 'trip': False})
    c = TransferController(
        motion, {'sku-box': PROFILE}, fixture_mode=True, transport_joints=_SL_STOW,
        staged_cart_swing=True, cart_pan_joint0=-0.8605, cart_tuck_shape=_SL_TUCK,
        reposition_before_place=True, collect_place_preplace_shape=_SL_PREPLACE_SHAPE,
        collect_place_wrist_guard=True, collect_place_straight_line=True, carton_level=lc,
        fk_urdf=URDF, carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    c.start('collect', dict(job_id='j', sku='sku-box', quantity=1, source_bin='b', shelf_id='s'))
    c.phase, c.held, c.issued = 'place', True, True
    c.poses = {'preplace': (0.5, 0.2, 0.32, 0., 0., 0., 1.), 'place': (0.5, 0.2, 0.2, 0., 0., 0., 1.)}
    c.target_surface = (0.5, 0.2, 0.1, 0., 0., 0., 1.)
    c.target_pose = (0.5, 0.2, 0.2, 0., 0., 0., 1.)
    c._capture_level_up((0.0,) * 6)
    motion.state = IN_FLIGHT                                  # the place descent is executing (in flight)
    return c


def test_place_obstructed_owes_a_serialized_reissue():
    """_place_obstructed arms the re-approach serialization: it sets _reapproach_reissue_pending and
    transitions to preplace with the obstructed descent STILL in flight (the stop is owed to the preplace
    entry, not fired here)."""
    m = GuardedMotion(state=IN_FLIGHT)
    c = _place_controller(m, reapproach_max=2)
    dec = NS(hit_prim=POST, reason='met post')
    res = c._place_obstructed(0.1, dec)
    assert res.status == 'running'
    assert c.phase == 'preplace' and c.issued is False
    assert c._reapproach_reissue_pending is True
    assert m.state == IN_FLIGHT                               # descent not yet stopped (owed)
    assert m.cancels == 0 and m.settles == 0


@need_urdf
def test_cw90_reapproach_gets_a_real_second_descent_never_crashes():
    """coldwin90 end to end: an obstructed place descent triggers the bounded re-approach, which re-descends
    via the pre-place standoff. The in-flight descent is stopped (contact-hold) and brought terminal BEFORE
    the preplace re-issue, so begin*() never raises 'motion already active' -- a real second attempt."""
    m = GuardedMotion(state=IN_FLIGHT, settle_ticks=0, settled_outcome='succeeded')
    c = _place_controller(m, reapproach_max=2)
    # tick 1: the descending carton meets the non-support post -> PLACE_OBSTRUCTED -> re-approach -> preplace.
    r = c.step(.05, _packet(c, 1.0, contact=True, placed_pose=(0.5, 0.2, 0.2, 0., 0., 0., 1.),
                            seal_gap_mm=0.0, seal_hit_prims=[POST]))
    assert r.status == 'running' and c.phase == 'preplace' and c._reapproach_reissue_pending is True
    # ticks 2..: the preplace entry serializes the in-flight descent (contact-hold stop), then re-issues the
    # preplace goal -- NEVER a RuntimeError, and a real second begin fires.
    begins_before = len(m.begins)
    for t in (1.05, 1.10, 1.15, 1.20):
        r = c.step(.05, _packet(c, t, contact=True, placed_pose=(0.5, 0.2, 0.3, 0., 0., 0., 1.)))
        assert r.status != 'failed', r.reason
        if len(m.begins) > begins_before:
            break
    assert m.settles >= 1                                     # the descent was stopped on the contact channel
    assert len(m.begins) == begins_before + 1                 # the re-descend began cleanly (no crash)
    assert c._reapproach_reissue_pending is False


@need_urdf
def test_cw90_reapproach_fails_by_name_if_descent_never_terminates():
    """If the obstructed descent never reaches a terminal result, the re-approach fails by name
    (REISSUE_MOTION_STILL_ACTIVE) -- it never re-issues into the in-flight descent and never crashes."""
    m = GuardedMotion(state=IN_FLIGHT, never_settles=True)
    c = _place_controller(m, reapproach_max=2)
    c.reissue_settle_timeout_s = 0.15
    c.step(.05, _packet(c, 1.0, contact=True, placed_pose=(0.5, 0.2, 0.2, 0., 0., 0., 1.),
                        seal_gap_mm=0.0, seal_hit_prims=[POST]))
    r = None
    for t in (1.05, 1.10, 1.15, 1.20, 1.25):
        r = c.step(.05, _packet(c, t, contact=True, placed_pose=(0.5, 0.2, 0.3, 0., 0., 0., 1.)))
        if r.status == 'failed':
            break
    assert r.status == 'failed' and 'REISSUE_MOTION_STILL_ACTIVE' in r.reason
    assert m.begins == []                                     # never re-issued into the in-flight descent


# ============================== C') sim-place-reapproach-counter-terminates-v1: the re-descend does NOT reset

def _place_controller_wide_bound(motion, reapproach_max=2):
    """Like _place_controller but with a generous level bound, so the place descent PLAN is ACCEPTED and the
    descent-plan block runs to completion (reaching the release-arm region where the stale reset used to
    live) instead of being rejected for tilt -- lets the re-descend path be exercised on CPU."""
    lc = cl.resolve_level_config({'place_release_on_contact': True, 'place_reapproach_max': reapproach_max,
                                  'place_reapproach_shift_m': 0.03, 'trip': False,
                                  'bound_deg': 10.0, 'preplace_gate_bound_deg': 9.0})
    c = TransferController(
        motion, {'sku-box': PROFILE}, fixture_mode=True, transport_joints=_SL_STOW,
        staged_cart_swing=True, cart_pan_joint0=-0.8605, cart_tuck_shape=_SL_TUCK,
        reposition_before_place=True, collect_place_preplace_shape=_SL_PREPLACE_SHAPE,
        collect_place_wrist_guard=True, collect_place_straight_line=True, carton_level=lc,
        fk_urdf=URDF, carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    c.start('collect', dict(job_id='j', sku='sku-box', quantity=1, source_bin='b', shelf_id='s'))
    c.phase, c.held, c.issued = 'place', True, False
    c.poses = {'preplace': (0.5, 0.2, 0.32, 0., 0., 0., 1.), 'place': (0.5, 0.2, 0.2, 0., 0., 0., 1.)}
    c.target_surface = (0.5, 0.2, 0.1, 0., 0., 0., 1.)
    c.target_pose = (0.5, 0.2, 0.2, 0., 0., 0., 1.)
    c._capture_level_up((0.0,) * 6)
    c._place_reoriented = True                               # past the reorient leg -> the descent plan runs
    return c


@need_urdf
def test_place_descent_replan_does_not_reset_the_reapproach_counter(capsys):
    """coldwin101b root cause: the place descent-plan block re-runs on EVERY bounded lateral re-approach (the
    place->preplace->reorient->descent cycle). It used to zero ``_place_reapproach_count`` there, so every
    PLACE_OBSTRUCTED re-incremented from 0 and the attempt line stuck at ``1/2`` forever -- the budget never
    exhausted and the run died on the re-issue-timeout e-stop. With the reset moved to start() (per op), a
    re-descend must PRESERVE a counter already advanced this stage."""
    m = GuardedMotion(state='idle')                          # terminal -> the descent begin() is allowed
    c = _place_controller_wide_bound(m, reapproach_max=2)
    c._place_reapproach_count = 1                            # one re-approach already spent this place stage
    r = c.step(.05, _packet(c, 1.0, contact=True, placed_pose=(0.5, 0.2, 0.3, 0., 0., 0., 1.)))
    out = capsys.readouterr().out
    assert 'PLACE_CONTACT_RELEASE_ARM' in out                # the descent-plan block ran past where the reset was
    assert r.status == 'running' and len(m.begins) == 1      # the re-descend was issued
    assert c._place_reapproach_count == 1                    # NOT reset to 0 by the re-descend plan


# ===================================================================== D) byte-identical when inert

def test_no_pending_never_touches_the_cancel_protocol():
    """On a normal preplace entry (no re-approach / no drift re-capture pending) the serialization is never
    engaged: a port that is already terminal is begun directly, and cancel/settle are never called."""
    m = GuardedMotion(state='succeeded')                      # a just-completed lift -> terminal
    c = _guard_host(m)
    verdict, payload = c._serialize_reissue(0.0, driven=True)
    assert verdict == 'ready' and payload == ()
    assert m.cancels == 0 and m.settles == 0 and m.outcome_polls == 0


def test_recapture_off_controller_never_serializes():
    """With drift re-capture off, _handle_preplace_convergence never sets a re-capture pending, so the
    serialization helper is never reached (byte-identical to the pre-fix terminal-stall path)."""
    m = GuardedMotion(state='idle')
    c = TransferController(m, {'sku-box': PROFILE}, fixture_mode=True,
                          preplace_convergence_stall_s=30.0)   # recapture off (no rad/max)
    c.start('collect', dict(job_id='j', sku='sku-box', quantity=1, source_bin='b', shelf_id='s'))
    c.phase, c.issued = 'preplace', True
    c._preplace_convergence.arm(_goal())
    c._handle_preplace_convergence(0.0, _sensors(0.0))
    assert c._recapture_reissue_pending is False
    assert m.cancels == 0 and m.settles == 0
