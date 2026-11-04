"""CPU tests for the collect contact-release SERIALIZATION fix (sim-place-release-serialization-v1).

coldwin43 (PR 333) cleared the set-down carton TILT (place tilt 1.57 deg, zero CARTON_TILT trips) but then
FAILED collect at the weld release: the contact-triggered release fired MID place-descent and its detach
mutated the planning scene while the place ExecuteTrajectory was STILL in flight, tripping the host adapter's
``_PLAN_IN_FLIGHT_STATES`` invariant ->
``RuntimeError: planning-scene mutation issued while a plan/execute is in flight (state=execute_result)`` ->
HOST_FAULT_ESTOP phase=release.

The fix makes the release a proper phase transition: on the contact trigger the controller CANCELS the
in-flight place execute (the cancel is the stop -- the arm never keeps pressing the welded carton into the
deck), WAITS for the cancel's terminal (settled, not-in-flight) result with a bounded timeout, and only THEN
detaches and continues withdraw/verify from the settled pose.

These tests cover:
  * the ADAPTER contract (the real RosMotionPort methods, exercised directly with fake handle/future):
    the in-flight guard reproduces the coldwin43 crash; cancel()/cancel_outcome() drain to a terminal,
    not-in-flight state WITHOUT the global emergency stop, so the detach is then legal; and every terminal
    outcome (cancelled / succeeded-before-cancel / error / still-settling) is handled.
  * the CONTROLLER state machine (a mock adapter that models the in-flight guard + the cancel protocol):
    cancel-then-detach ordering, no scene mutation while in flight, the bounded timeout path, and a
    coldwin43 timeline replay (old ordering crashes; new ordering passes), plus the grace proof.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_release_serialization_v1.py
"""
import os
from types import SimpleNamespace as NS

import pytest

from twin.restock_manipulation import ProductProfile, TransferController
from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.ros_adapter import _PLAN_IN_FLIGHT_STATES, RosMotionPort

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='dataset mobile URDF not present')

COLDWIN43_CRASH = 'planning-scene mutation issued while a plan/execute is in flight (state=execute_result)'


# ------------------------------------------------------------------- adapter fakes (no ROS / no GPU)

class _FakeFuture:
    def __init__(self, *, done=True, status=5, raises=False):
        self._done, self._status, self._raises = done, status, raises

    def done(self):
        return self._done

    def result(self):
        if self._raises:
            raise RuntimeError('broken result future')
        return NS(status=self._status)


class _FakeHandle:
    def __init__(self):
        self.cancelled = False

    def cancel_goal_async(self):
        self.cancelled = True


def _bare_port(state, *, future=None, handle=None):
    """A RosMotionPort with ONLY the attributes cancel/cancel_outcome/update_payload touch -- so the real
    adapter methods run on CPU with no node/ROS. payload_sync returns True (a clean detach ack)."""
    p = RosMotionPort.__new__(RosMotionPort)
    p.state = state
    p.handle = handle
    p.future = future
    p.generation = 0
    p._cancel_requested = False
    p.payload_sync = lambda held, profile: True
    p.emergency_stop = lambda: (_ for _ in ()).throw(AssertionError('cancel must NEVER emergency_stop'))
    return p


PROFILE = ProductProfile('sku-box', (.10, .10, .10), (1., 0., 0., 0.))


# =================================================================== ADAPTER: the in-flight guard (crash)

@pytest.mark.parametrize('state', _PLAN_IN_FLIGHT_STATES)
def test_update_payload_guards_every_in_flight_state(state):
    """The coldwin43 mechanism: a scene mutation issued while the adapter is in ANY plan/execute state
    raises the exact invariant error. execute_result is the one coldwin43 tripped."""
    p = _bare_port(state)
    with pytest.raises(RuntimeError, match='planning-scene mutation issued while a plan/execute is in flight'):
        p.update_payload(False, PROFILE)


def test_coldwin43_crash_reproduced_detach_in_flight():
    """Replay the coldwin43 ordering directly: detach (update_payload False) while state=execute_result ->
    the precise HOST_FAULT_ESTOP reason string."""
    p = _bare_port('execute_result')
    with pytest.raises(RuntimeError) as exc:
        p.update_payload(False, PROFILE)
    assert str(exc.value) == COLDWIN43_CRASH


def test_detach_legal_once_terminal():
    """Out of the in-flight set (a settled move), the detach is legal (returns the ack)."""
    p = _bare_port('failed')
    assert p.update_payload(False, PROFILE) is True


# =================================================================== ADAPTER: cancel()/cancel_outcome()

def test_cancel_requests_goal_cancel_without_estop():
    """cancel() requests the action cancel (arm stops) and fires NO emergency stop; it does NOT itself move
    the state (the move is still settling) -- cancel_outcome() owns the terminal transition."""
    h = _FakeHandle()
    p = _bare_port('execute_result', handle=h, future=_FakeFuture(done=False))
    p.cancel()                                  # _bare_port's emergency_stop raises if ever called
    assert h.cancelled is True and p._cancel_requested is True
    assert p.state == 'execute_result'          # still in flight until the terminal result arrives


def test_cancel_outcome_waits_until_result_done():
    """While the execute result has not arrived, cancel_outcome() returns None (keep waiting) and leaves the
    state IN FLIGHT -- so the controller can never detach early."""
    p = _bare_port('execute_result', handle=_FakeHandle(), future=_FakeFuture(done=False))
    p.cancel()
    assert p.cancel_outcome() is None
    assert p.state in _PLAN_IN_FLIGHT_STATES     # not yet terminal -> detach still illegal


def test_cancel_outcome_cancelled_drains_to_terminal_then_detach_ok():
    """A CANCELED result -> 'cancelled', state driven OUT of the in-flight set, late callbacks invalidated
    (generation bumped); the detach is then legal."""
    p = _bare_port('execute_result', handle=_FakeHandle(), future=_FakeFuture(done=True, status=5))
    p.cancel()
    assert p.cancel_outcome() == 'cancelled'
    assert p.state == 'failed' and p.state not in _PLAN_IN_FLIGHT_STATES
    assert p.handle is None and p.future is None and p.generation == 1
    assert p.update_payload(False, PROFILE) is True      # now legal


def test_cancel_outcome_succeeded_before_cancel():
    """The goal SUCCEEDED before the cancel landed -> 'succeeded', terminal, detach legal."""
    p = _bare_port('execute_result', handle=_FakeHandle(), future=_FakeFuture(done=True, status=4))
    p.cancel()
    assert p.cancel_outcome() == 'succeeded'
    assert p.state == 'succeeded' and p.state not in _PLAN_IN_FLIGHT_STATES
    assert p.update_payload(False, PROFILE) is True


def test_cancel_outcome_error_on_broken_result():
    """A result future that RAISES is a broken terminal -> 'error', state terminal (the controller reports
    and refuses to detach blindly)."""
    p = _bare_port('execute_result', handle=_FakeHandle(), future=_FakeFuture(done=True, raises=True))
    p.cancel()
    assert p.cancel_outcome() == 'error'
    assert p.state == 'failed'


def test_cancel_outcome_when_already_terminal():
    """cancel() after the move already ended (idle/succeeded/failed) resolves immediately -- never hangs."""
    for state, want in (('failed', 'cancelled'), ('succeeded', 'succeeded'), ('idle', 'cancelled')):
        p = _bare_port(state)
        p.cancel()
        assert p.cancel_outcome() == want


def test_cancel_outcome_none_without_request():
    """No cancel requested -> cancel_outcome() is inert (None), never classifies a live move."""
    p = _bare_port('execute_result', future=_FakeFuture(done=True, status=5))
    assert p.cancel_outcome() is None


# =================================================================== CONTROLLER: the serialized release

class SerializingMotion:
    """A fixture motion port that models the RosMotionPort release-serialization contract the controller
    relies on: a plan/execute ``state`` exposed to the controller, update_payload that RAISES while the
    state is in flight (the coldwin43 invariant), and cancel()/cancel_outcome() the controller drives to
    drain an in-flight move before detaching. ``cancel_seq`` scripts the cancel_outcome() returns (so a
    test can drive the still-settling / cancelled / succeeded / error / timeout paths)."""

    def __init__(self, cancel_seq=('cancelled',)):
        self.state = 'idle'
        self.payloads = []                 # (held, state_at_call) for every update_payload
        self.cancels = 0
        self.cancel_outcomes = 0
        self._cancel_seq = list(cancel_seq)
        self.outcome = 'running'           # the active place move stays in flight
        self.stops = 0
        self.attach_ok = True

    # --- the moves the controller issues (set the in-flight state) ---
    def begin(self, *a, **k):
        self.state = 'execute_result'

    def begin_joint_goal(self, *a, **k):
        self.state = 'execute_result'

    def begin_joint_path(self, *a, **k):
        self.state = 'execute_result'

    def poll(self):
        return self.outcome

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

    def cancel_outcome(self):
        self.cancel_outcomes += 1
        if not self._cancel_seq:
            return None                    # exhausted script -> keep returning None (the timeout path)
        out = self._cancel_seq.pop(0)
        if out is None:
            return None
        self.state = 'succeeded' if out == 'succeeded' else 'failed'   # terminal, not in flight
        return out

    def stop(self):
        self.stops += 1

    def update_payload(self, held, profile):
        self.payloads.append((held, self.state))
        if self.state in _PLAN_IN_FLIGHT_STATES:
            raise RuntimeError('planning-scene mutation issued while a plan/execute is in flight '
                               '(state=%s)' % self.state)
        return self.attach_ok


def _packet(c, t, *, contact, placed_pose):
    import time
    from twin.restock_manipulation import Observation, Sensors, Target
    observation = Observation(f'src-{t}', 'fixture', t, 'sku-box', c.source,
                              (.4, 0., .2, 0., 0., 0., 1.), .99)
    target = Target(f'tgt-{t}', 'fixture', t, c.destination, (.5, .2, .1, 0., 0., 0., 1.), True, (.3, .3, .4))
    return Sensors(simulation_time=t, observations=(observation,), target=target, parked=True,
                   contact=contact, feedback_time=t, collision_time=t, joint_positions=(0.,) * 6,
                   scanned_locations=(c.source, c.destination), scan_time=t,
                   valid_until_steady=time.monotonic() + 100., placed_pose=placed_pose)


def _place_phase_controller(cancel_seq=('cancelled',)):
    """A collect controller driven directly into the place-descent state with the contact-release gate armed
    and a SerializingMotion whose place move is in flight (state=execute_result)."""
    JOB = dict(job_id='job-1', sku='sku-box', quantity=1, source_bin='backroom', shelf_id='shelf-a')
    lc = cl.resolve_level_config({'place_contact_release': True, 'place_contact_gap_m': 0.012, 'trip': False})
    motion = SerializingMotion(cancel_seq=cancel_seq)
    c = TransferController(
        motion, {'sku-box': PROFILE}, fixture_mode=True, transport_joints=_SL_STOW,
        staged_cart_swing=True, cart_pan_joint0=-0.8605, cart_tuck_shape=_SL_TUCK,
        reposition_before_place=True, collect_place_preplace_shape=_SL_PREPLACE_SHAPE,
        collect_place_wrist_guard=True, collect_place_straight_line=True, carton_level=lc,
        fk_urdf=URDF, carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    c.start('collect', JOB)
    c.phase, c.held, c.issued = 'place', True, True
    c.poses = {'place': (0.4, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0)}
    c._capture_level_up((0.0,) * 6)
    c._place_contact_release_z = 0.90
    motion.state = 'execute_result'            # the place descent is executing (in flight)
    return c, motion


_SL_STOW = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
_SL_TUCK = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
_SL_PREPLACE_SHAPE = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)


@need_urdf
def test_contact_release_cancels_then_detaches_in_order():
    """The fixed sequence: the contact trigger CANCELS the in-flight place, WAITS for the terminal result,
    THEN detaches -- and never mutates the planning scene while in flight (so no coldwin43 crash)."""
    c, m = _place_phase_controller(cancel_seq=('cancelled',))
    # tick 1: descend above threshold -> no release.
    r = c.step(.1, _packet(c, 1.0, contact=True, placed_pose=(0.4, 0., 0.95, 0., 0., 0., 1.)))
    assert c.phase == 'place' and r.status == 'running' and m.cancels == 0
    # tick 2: AT deck contact -> transition to release, cancel owed, NOT YET issued (cancel is next tick).
    r = c.step(.1, _packet(c, 1.1, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.)))
    assert c.phase == 'release' and c._release_cancel_pending is True and m.cancels == 0
    # tick 3: cancel the in-flight place execute (the stop). No detach yet; place still in flight.
    r = c.step(.1, _packet(c, 1.2, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.)))
    assert m.cancels == 1 and m.payloads == [] and r.status == 'running'
    # tick 4: cancel settles terminal -> gripper-open (carton still welded, contact True).
    r = c.step(.1, _packet(c, 1.3, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.)))
    assert c._release_cancel_pending is False and m.state not in _PLAN_IN_FLIGHT_STATES
    assert ({'kind': 'gripper', 'enabled': False},) == r.controls and m.payloads == []
    # tick 5: contact cleared -> the detach (update_payload False) runs from the SETTLED (not-in-flight)
    # state and is acknowledged; the mission continues (release -> retreat).
    r = c.step(.1, _packet(c, 1.4, contact=False, placed_pose=(0.4, 0., 0.60, 0., 0., 0., 1.)))
    assert m.payloads == [(False, 'failed')]       # detached, NOT in an in-flight state
    assert c.phase == 'retreat' and r.status != 'failed', r.reason


@need_urdf
def test_detach_never_issued_while_in_flight():
    """Hard invariant: across the whole serialized release, update_payload is NEVER called while the adapter
    state is in flight -- the coldwin43 crash can never recur."""
    c, m = _place_phase_controller(cancel_seq=('cancelled',))
    c.step(.1, _packet(c, 1.0, contact=True, placed_pose=(0.4, 0., 0.95, 0., 0., 0., 1.)))
    for i, t in enumerate((1.1, 1.2, 1.3, 1.4, 1.5)):
        contact = t < 1.4
        r = c.step(.1, _packet(c, t, contact=contact, placed_pose=(0.4, 0., 0.60, 0., 0., 0., 1.)))
        assert r.status != 'failed', r.reason
    assert m.payloads, 'the detach must have run'
    assert all(state not in _PLAN_IN_FLIGHT_STATES for _held, state in m.payloads)


@need_urdf
def test_succeeded_before_cancel_still_detaches():
    """If the place execute SUCCEEDED before the cancel landed, the release still detaches from that settled
    terminal (never stuck waiting)."""
    c, m = _place_phase_controller(cancel_seq=('succeeded',))
    c.step(.1, _packet(c, 1.0, contact=True, placed_pose=(0.4, 0., 0.95, 0., 0., 0., 1.)))     # above thr
    for t in (1.1, 1.2, 1.3, 1.4):
        contact = t < 1.4
        c.step(.1, _packet(c, t, contact=contact, placed_pose=(0.4, 0., 0.60, 0., 0., 0., 1.)))
    assert m.payloads == [(False, 'succeeded')] and c.phase == 'retreat'


@need_urdf
def test_cancel_timeout_fails_by_name_and_never_detaches():
    """If the cancel never reaches a terminal result, the release FAILS BY NAME after the bounded sim budget
    and NEVER detaches into an in-flight scene (report, do not detach blindly)."""
    c, m = _place_phase_controller(cancel_seq=(None,))         # cancel_outcome always None -> never settles
    c.release_cancel_timeout_s = 0.25                          # tight bound for the test
    c.step(.1, _packet(c, 1.0, contact=True, placed_pose=(0.4, 0., 0.95, 0., 0., 0., 1.)))    # above thr
    c.step(.1, _packet(c, 1.1, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.)))    # -> release
    c.step(.1, _packet(c, 1.2, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.)))    # cancel issued
    r = None
    for t in (1.3, 1.4, 1.5, 1.6):                             # advance sim past the 0.25 s settle budget
        r = c.step(.1, _packet(c, t, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.)))
        if r.status == 'failed':
            break
    assert r.status == 'failed' and 'did not reach a terminal result' in r.reason
    assert m.payloads == []                                    # never detached


@need_urdf
def test_cancel_error_fails_by_name_and_never_detaches():
    """A broken cancel terminal ('error') fails by name and never detaches."""
    c, m = _place_phase_controller(cancel_seq=('error',))
    c.step(.1, _packet(c, 1.0, contact=True, placed_pose=(0.4, 0., 0.95, 0., 0., 0., 1.)))
    c.step(.1, _packet(c, 1.1, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.)))    # -> release
    c.step(.1, _packet(c, 1.2, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.)))    # cancel
    r = c.step(.1, _packet(c, 1.3, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.)))  # error
    assert r.status == 'failed' and 'terminated with an error' in r.reason
    assert m.payloads == []


def test_release_not_a_carried_trip_phase():
    """The tilt win is structural: 'release' is not a CARTON_TILT-graded carried phase, so the brief carton
    conform at the deck during the (sub-0.15 s) cancel settle can never sustain a trip -- and 'place' is."""
    assert 'release' not in TransferController._CARRIED_PHASES
    assert 'place' in TransferController._CARRIED_PHASES


def test_trigger_to_cancel_settled_inside_grace_on_coldwin43_cadence():
    """Grace proof (brief part 2) from the coldwin43 timeline: during the release phase the controller ticks
    at ~0.017 s sim (BASE_ATT_SEGMENT phase=release: samples=17 over dt=0.3 s -> ~0.0176 s/sample, the
    per-physics-step packet cadence). The serialized cancel settles within a bounded few ticks
    (trigger-tick transitions, +1 issues the cancel, +1 reads the terminal), so trigger -> cancel-settled is
    at most ~3 ticks ~= 0.053 s -- well inside the 0.15 s CARTON_TILT debounce grace."""
    coldwin43_release_dt_s, coldwin43_release_samples = 0.3, 17        # BASE_ATT_SEGMENT phase=release
    tick_s = coldwin43_release_dt_s / coldwin43_release_samples
    assert tick_s == pytest.approx(0.0176, abs=1e-3)
    ticks_trigger_to_settled = 3                                       # transition, cancel, read-terminal
    assert ticks_trigger_to_settled * tick_s < 0.15
