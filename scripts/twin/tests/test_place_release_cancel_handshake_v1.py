"""CPU tests for the contact-release CANCEL HANDSHAKE fix (sim-place-release-cancel-handshake-v1).

coldwin74/75 both carried the carton through pick, pre-place and a level descent onto the cart shelf and
the contact-gated weld release fired when the support was met (``PLACE_CONTACT_RELEASE trigger=stall``).
The release phase then CANCELLED the in-flight place ExecuteTrajectory and WAITED for a terminal result
(``RESTOCK_RELEASE_CANCEL_SETTLE_S``) before detaching -- but the place descent is a DRIVEN trajectory on
the sim FollowJointTrajectory executor: the action cancel was dropped in the goal-accept race
(``self.handle is None`` at the stall, so ``cancel_goal_async`` never went out) AND the sim trajectory_host
treats a goal cancel as a FAULT that latches the global stop. ``cancel_outcome()`` stayed None for the full
budget (cw74 2.0 s, cw75 8.0 s) and the run e-stopped with the carton still welded on the shelf.

The fix (opt-in ``carton_level.release_cancel_handshake``) STOPS the descent on its OWN channel -- the
release phase emits the contact-hold control so trajectory_host SUCCEEDS holding the current pose (terminal,
no fault, no global stop) and marks the port cancel-requested (``settle_in_place``, no ``cancel_goal_async``)
so ``cancel_outcome`` reads the resolved execute future -- and, if the terminal STILL never arrives but the
arm is PROVABLY stopped (measured max joint speed <= ``release_cancel_stopped_vel`` for
``release_cancel_stopped_ticks`` consecutive ticks), logs ``RELEASE_CANCEL_STOPPED_FALLBACK`` and proceeds to
the gripper-open + detach. NEVER detaches into a moving arm. Default off => byte-identical (the legacy
``motion.cancel`` path; covered by test_place_release_serialization_v1.py).

These tests cover:
  * the ADAPTER contract: ``settle_in_place`` arms ``cancel_outcome`` WITHOUT sending a goal cancel, and
    ``force_terminal_settled`` drains an in-flight state to terminal (so the detach is legal) with no
    emergency stop, and is a no-op when already terminal.
  * the CONTROLLER handshake: the contact-hold is emitted (not ``cancel()``); a normal terminal detaches
    unchanged; a hung cancel with a provably-stopped arm proceeds via the fallback; a MOVING arm never
    detaches; and a faithful cw75-trace replay (OFF reproduces the e-stop timeout, ON completes the release).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_release_cancel_handshake_v1.py
"""
import os
import time
from types import SimpleNamespace as NS

import pytest

from twin.restock_manipulation import Observation, ProductProfile, Sensors, Target, TransferController
from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.ros_adapter import _PLAN_IN_FLIGHT_STATES, RosMotionPort

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='dataset mobile URDF not present')

PROFILE = ProductProfile('sku-box', (.10, .10, .10), (1., 0., 0., 0.))

_SL_STOW = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
_SL_TUCK = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
_SL_PREPLACE_SHAPE = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)

# The cw75 arm config frozen through the release window (host-stack.log CARTON_TICK at 245.500:
# q=[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]).
CW75_Q = (-0.4323, -1.4634, 1.4546, 0.0388, 0.6004, -0.0351)


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
    p = RosMotionPort.__new__(RosMotionPort)
    p.state = state
    p.handle = handle
    p.future = future
    p.generation = 0
    p._cancel_requested = False
    p.payload_sync = lambda held, profile: True
    p.emergency_stop = lambda: (_ for _ in ()).throw(AssertionError('handshake must NEVER emergency_stop'))
    return p


# =================================================================== ADAPTER: settle_in_place / force_terminal

def test_settle_in_place_arms_outcome_without_goal_cancel():
    """settle_in_place arms cancel_outcome (sets _cancel_requested) but NEVER calls cancel_goal_async -- the
    descent is stopped on the contact-hold channel, not by an (unsafe, dropped) ExecuteTrajectory cancel."""
    h = _FakeHandle()
    p = _bare_port('execute_result', handle=h, future=_FakeFuture(done=False))
    p.settle_in_place()
    assert p._cancel_requested is True
    assert h.cancelled is False                       # no goal cancel was sent
    assert p.state == 'execute_result'                # still in flight until the terminal arrives
    assert p.cancel_outcome() is None                 # now armed: reads the (not-yet-done) future


def test_settle_in_place_then_terminal_reads_succeeded():
    """Once the contact-hold makes the execute future resolve SUCCEEDED, the armed cancel_outcome reports it
    and drains out of the in-flight set so the detach is legal."""
    p = _bare_port('execute_result', handle=_FakeHandle(), future=_FakeFuture(done=True, status=4))
    p.settle_in_place()
    assert p.cancel_outcome() == 'succeeded'
    assert p.state == 'succeeded' and p.state not in _PLAN_IN_FLIGHT_STATES
    assert p.update_payload(False, PROFILE) is True


def test_force_terminal_settled_drains_in_flight_without_estop():
    """The stopped-arm fallback: force_terminal_settled moves an in-flight state OUT of the in-flight set
    (so the detach's guard passes), bumps the generation to invalidate late callbacks, and NEVER e-stops."""
    p = _bare_port('execute_result', handle=_FakeHandle(), future=_FakeFuture(done=False))
    p.force_terminal_settled()
    assert p.state == 'succeeded' and p.state not in _PLAN_IN_FLIGHT_STATES
    assert p.handle is None and p.future is None and p.generation == 1
    assert p.update_payload(False, PROFILE) is True        # the detach is now legal


@pytest.mark.parametrize('state', ('idle', 'succeeded', 'failed'))
def test_force_terminal_settled_noop_when_terminal(state):
    """A no-op when already terminal: state and generation are untouched."""
    p = _bare_port(state)
    p.force_terminal_settled()
    assert p.state == state and p.generation == 0


# ------------------------------------------------------------------- controller fixture (handshake path)

class HandshakeMotion:
    """A fixture motion port modelling the handshake contract: the place move sets an in-flight ``state``;
    ``settle_in_place`` arms the terminal read WITHOUT a goal cancel; ``cancel_outcome`` returns a scripted
    sequence (None => still settling); ``force_terminal_settled`` drains an in-flight state terminal; and
    ``update_payload`` RAISES while in flight (the coldwin43 invariant)."""

    def __init__(self, cancel_seq=('succeeded',)):
        self.state = 'idle'
        self.payloads = []
        self.cancels = 0                 # the legacy motion.cancel() path (must stay 0 on the handshake)
        self.settles = 0
        self.forced = 0
        self.cancel_outcomes = 0
        self._cancel_seq = list(cancel_seq)
        self._cancel_requested = False

    def begin(self, *a, **k):
        self.state = 'execute_result'

    def begin_joint_goal(self, *a, **k):
        self.state = 'execute_result'

    def begin_joint_path(self, *a, **k):
        self.state = 'execute_result'

    def poll(self):
        return 'running'

    def last_fault_was_sim_deadline(self):
        return False

    def take_exec_abort_reason(self):
        return None

    def reset_crash_budget(self):
        pass

    def cancel(self):
        self.cancels += 1
        self._cancel_requested = True

    def settle_in_place(self):
        self.settles += 1
        self._cancel_requested = True

    def force_terminal_settled(self):
        self.forced += 1
        if self.state in _PLAN_IN_FLIGHT_STATES:
            self.state = 'succeeded'

    def cancel_outcome(self):
        self.cancel_outcomes += 1
        if not self._cancel_seq:
            return None
        out = self._cancel_seq.pop(0)
        if out is None:
            return None
        self.state = 'succeeded' if out == 'succeeded' else 'failed'
        return out

    def stop(self):
        pass

    def update_payload(self, held, profile):
        self.payloads.append((held, self.state))
        if self.state in _PLAN_IN_FLIGHT_STATES:
            raise RuntimeError('planning-scene mutation issued while a plan/execute is in flight '
                               '(state=%s)' % self.state)
        return True


def _packet(c, t, *, contact, placed_pose, q=(0.,) * 6):
    observation = Observation(f'src-{t}', 'fixture', t, 'sku-box', c.source,
                              (.4, 0., .2, 0., 0., 0., 1.), .99)
    target = Target(f'tgt-{t}', 'fixture', t, c.destination, (.5, .2, .1, 0., 0., 0., 1.), True, (.3, .3, .4))
    return Sensors(simulation_time=t, observations=(observation,), target=target, parked=True,
                   contact=contact, feedback_time=t, collision_time=t, joint_positions=tuple(q),
                   scanned_locations=(c.source, c.destination), scan_time=t,
                   valid_until_steady=time.monotonic() + 100., placed_pose=placed_pose)


def _place_phase_controller(cancel_seq=('succeeded',), *, handshake=True,
                            stopped_vel=0.02, stopped_ticks=3):
    JOB = dict(job_id='job-1', sku='sku-box', quantity=1, source_bin='backroom', shelf_id='shelf-a')
    spec = {'place_contact_release': True, 'place_contact_gap_m': 0.012, 'trip': False,
            'release_cancel_handshake': handshake, 'release_cancel_stopped_vel': stopped_vel,
            'release_cancel_stopped_ticks': stopped_ticks}
    lc = cl.resolve_level_config(spec)
    motion = HandshakeMotion(cancel_seq=cancel_seq)
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
    motion.state = 'execute_result'
    return c, motion


def _arm_into_release(c, m, *, q=(0.,) * 6):
    """Drive the controller to the contact trigger and the first (cancel-issuing) release tick. Returns the
    result of the tick that issued the hold-and-settle."""
    c.step(.1, _packet(c, 1.0, contact=True, placed_pose=(0.4, 0., 0.95, 0., 0., 0., 1.), q=q))   # above thr
    c.step(.1, _packet(c, 1.1, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.), q=q))   # -> release
    return c.step(.1, _packet(c, 1.2, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.), q=q))


# =================================================================== CONTROLLER: the handshake release

@need_urdf
def test_handshake_emits_contact_hold_not_cancel():
    """On the handshake path the first release tick emits the contact-HOLD control (stop the descent on its
    own channel) and arms settle_in_place -- NOT the legacy ExecuteTrajectory cancel()."""
    c, m = _place_phase_controller(cancel_seq=('succeeded',))
    r = _arm_into_release(c, m)
    assert c.phase == 'release' and c._release_cancel_pending is True
    assert m.settles == 1 and m.cancels == 0                       # hold-and-settle, no goal cancel
    assert ({'kind': 'contact'},) == r.controls


@need_urdf
def test_handshake_terminal_detaches_in_order():
    """With the contact-hold terminal ('succeeded'), the release detaches from the settled state, never while
    in flight."""
    c, m = _place_phase_controller(cancel_seq=('succeeded',))
    _arm_into_release(c, m)
    # next tick: cancel_outcome -> terminal -> gripper-open (carton still welded, contact True)
    r = c.step(.1, _packet(c, 1.3, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.)))
    assert c._release_cancel_pending is False and m.state not in _PLAN_IN_FLIGHT_STATES
    assert ({'kind': 'gripper', 'enabled': False},) == r.controls and m.payloads == []
    # contact cleared -> detach from the settled state
    r = c.step(.1, _packet(c, 1.4, contact=False, placed_pose=(0.4, 0., 0.60, 0., 0., 0., 1.)))
    assert m.payloads == [(False, 'succeeded')] and c.phase == 'retreat'


@need_urdf
def test_handshake_stopped_arm_fallback_proceeds_when_cancel_hangs():
    """The cw75 class: cancel_outcome NEVER terminal, but the arm is PROVABLY stopped (constant joints) ->
    RELEASE_CANCEL_STOPPED_FALLBACK forces the terminal and the release detaches."""
    c, m = _place_phase_controller(cancel_seq=(None,), stopped_vel=0.02, stopped_ticks=3)
    _arm_into_release(c, m, q=CW75_Q)
    r = None
    for t in (1.3, 1.4, 1.5, 1.6, 1.7, 1.8):        # arm frozen at CW75_Q -> stopped for N ticks
        r = c.step(.1, _packet(c, t, contact=(t < 1.6), placed_pose=(0.4, 0., 0.60, 0., 0., 0., 1.), q=CW75_Q))
        if c.phase == 'retreat':
            break
    assert m.forced == 1, 'the stopped-arm fallback must force the terminal'
    assert m.payloads and all(state not in _PLAN_IN_FLIGHT_STATES for _h, state in m.payloads)
    assert c.phase == 'retreat' and r.status != 'failed', r.reason


@need_urdf
def test_handshake_moving_arm_never_detaches():
    """Safety invariant: cancel_outcome never terminal AND the arm is still MOVING (joints advance each tick)
    -> the fallback never fires; the release FAILS BY NAME at the budget and NEVER detaches."""
    c, m = _place_phase_controller(cancel_seq=(None,), stopped_vel=0.02, stopped_ticks=3)
    c.release_cancel_timeout_s = 0.35
    # moving arm: wrist_1 advances 0.05 rad per 0.1 s tick = 0.5 rad/s >> 0.02 threshold
    q0 = list(CW75_Q)
    c.step(.1, _packet(c, 1.0, contact=True, placed_pose=(0.4, 0., 0.95, 0., 0., 0., 1.), q=q0))
    c.step(.1, _packet(c, 1.1, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.), q=q0))
    r = None
    for i, t in enumerate((1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8)):
        q = list(CW75_Q)
        q[3] += 0.05 * (i + 1)                      # arm keeps moving
        r = c.step(.1, _packet(c, t, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.), q=q))
        if r.status == 'failed':
            break
    assert r.status == 'failed' and 'never read stopped' in r.reason
    assert m.forced == 0 and m.payloads == []       # never detached into a moving arm


@need_urdf
def test_handshake_cancelled_terminal_unchanged():
    """A normal terminal cancel outcome ('cancelled') detaches exactly as before (the fallback is not
    needed)."""
    c, m = _place_phase_controller(cancel_seq=('cancelled',))
    _arm_into_release(c, m)
    c.step(.1, _packet(c, 1.3, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.)))
    c.step(.1, _packet(c, 1.4, contact=False, placed_pose=(0.4, 0., 0.60, 0., 0., 0., 1.)))
    assert m.payloads == [(False, 'failed')] and c.phase == 'retreat' and m.forced == 0


# =================================================================== cw75 TRACE REPLAY REGRESSION

@need_urdf
def test_cw75_replay_off_reproduces_estop():
    """Faithful to the cw75 failure: the place descent is in flight, the stall release fires, the cancel
    NEVER reaches a terminal (dropped goal cancel). With the handshake OFF the release times out and FAILS BY
    NAME with the carton still welded (NO detach) -- the coldwin75 HOST_FAULT_ESTOP phase=release."""
    c, m = _place_phase_controller(cancel_seq=(None,), handshake=False)
    c.release_cancel_timeout_s = 0.25
    _arm_into_release(c, m, q=CW75_Q)
    assert m.cancels == 1 and m.settles == 0        # OFF uses the legacy cancel() path
    r = None
    for t in (1.3, 1.4, 1.5, 1.6, 1.7):
        r = c.step(.1, _packet(c, t, contact=True, placed_pose=(0.4, 0., 0.89, 0., 0., 0., 1.), q=CW75_Q))
        if r.status == 'failed':
            break
    assert r.status == 'failed' and 'did not reach a terminal result' in r.reason
    assert m.payloads == []                          # carton still welded, e-stopped


@need_urdf
def test_cw75_replay_on_completes_release():
    """Same cw75 timeline with the handshake ON: the stall release emits the contact-hold, and with the arm
    stopped at the traced config the stopped-arm fallback proceeds to detach -- the release COMPLETES (the
    run no longer e-stops with the carton welded on the shelf)."""
    c, m = _place_phase_controller(cancel_seq=(None,), handshake=True, stopped_vel=0.02, stopped_ticks=3)
    c.release_cancel_timeout_s = 2.0
    r = _arm_into_release(c, m, q=CW75_Q)
    assert m.settles == 1 and m.cancels == 0 and ({'kind': 'contact'},) == r.controls
    for t in (1.3, 1.4, 1.5, 1.6, 1.7, 1.8):
        r = c.step(.1, _packet(c, t, contact=(t < 1.6), placed_pose=(0.4, 0., 0.60, 0., 0., 0., 1.), q=CW75_Q))
        if c.phase == 'retreat':
            break
    assert c.phase == 'retreat' and r.status != 'failed', r.reason
    assert m.payloads and all(state not in _PLAN_IN_FLIGHT_STATES for _h, state in m.payloads)
