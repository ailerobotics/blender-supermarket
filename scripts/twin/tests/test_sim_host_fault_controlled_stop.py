"""sim-host-fault-controlled-stop-v1: a host fault that strikes while a carton is WELDED must be a
CONTROLLED stop, not a scene-wrecking slack-and-drive.

Replays the coldwin33 fault path on CPU with a fake host/port (no Isaac, ROS or GPU). coldwin33: a
carried-carton CARTON_TILT trip of 2.06 deg (0.06 over the captain-fixed 2.00 deg bound) fired the
host emergency stop at the place set-down while the arm was welded to the carton and extended over
the cart deck with the carton interpenetrating the deck. The collect RE-BEGIN then drove the base --
whose ``command()`` released every brake -- and PhysX resolved the penetration by flinging the base
and the cart (roll 138 deg, pitch 47 deg, z 2.11 m); collect re-began twice on the flipped robot and
the run ended with the generic 'arm command heartbeat lost'.

These prove the fix:
  * brake-hold: ``MobileController.command`` refuses to release any brake / drive while the welded
    controlled-stop latch is set;
  * no slack while welded: the controlled stop holds the arm (drive energised) and keeps the FULL
    base brake regardless of phase; the heartbeat watchdog does NOT raise during the deliberate hold;
  * retry refusal: a collect re-begin is refused while the controlled stop is active AND when the
    base attitude is outside the sane band (a flipped base);
  * named reason: the terminal reason names the FIRST trip (CARTON_TILT 2.06), not the downstream
    'arm command heartbeat lost'.

Run (BT tier needs py_trees from the task venv; the rest is pure):
  PYTHONPATH=/home/manishsaini/venvs/sim-restock-bt/lib/python3.12/site-packages:scripts \
    python3 -m pytest scripts/twin/tests/test_sim_host_fault_controlled_stop.py -q
"""
from pathlib import Path
from types import SimpleNamespace as NS
import math
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import controlled_stop, native_transport
from twin.restock.isaac_port import IsaacPort
from twin.restock.runtime import RuntimeAdapter
from twin.mobile_runtime import MobileController


# The verbatim coldwin33 terminal reasons (mission.json stage_timeline).
CARTON_TILT_REASON = ('host emergency stop (host fault: CARTON_TILT carried carton off level: '
                      'roll=-0.09 pitch=+2.06 deg exceeds bound 2.00 deg '
                      '(sustained 0.150s > grace 0.150s, peak 2.06 deg))')
HEARTBEAT_REASON = ('RuntimeError: arm command heartbeat lost [isaac_port.py:574 advance | '
                    'native_transport.py:425 tick | native_transport.py:334 watchdog]')

# A level chassis on this rig reads ~(roll 0, pitch -1.17, z 0.24) -- the caster-lift static tilt.
LEVEL = (-0.001, -1.173, 0.2400)
# The coldwin33 fling.
FLIPPED = (138.0, 47.0, 2.11)


# ============================================================ controlled_stop pure decision logic
def test_attitude_band_passes_level_rejects_flip_and_nonfinite():
    band = controlled_stop.AttitudeBand.from_nominal(*LEVEL)
    assert controlled_stop.attitude_ok(*LEVEL, band)
    # a mild, physically-plausible drift still passes
    assert controlled_stop.attitude_ok(5.0, 3.0, 0.27, band)
    # the coldwin33 fling is firmly rejected
    assert not controlled_stop.attitude_ok(*FLIPPED, band)
    # each axis alone, just over its tolerance, is rejected
    assert not controlled_stop.attitude_ok(LEVEL[0] + 21.0, LEVEL[1], LEVEL[2], band)
    assert not controlled_stop.attitude_ok(LEVEL[0], LEVEL[1] + 21.0, LEVEL[2], band)
    assert not controlled_stop.attitude_ok(LEVEL[0], LEVEL[1], LEVEL[2] + 0.16, band)
    # fail safe: a non-finite / bool reading is NOT ok
    assert not controlled_stop.attitude_ok(float('nan'), 0.0, 0.24, band)
    assert not controlled_stop.attitude_ok(0.0, float('inf'), 0.24, band)
    assert not controlled_stop.attitude_ok(True, 0.0, 0.24, band)


def test_prefer_first_trip_keeps_carton_tilt_over_heartbeat():
    # The core naming rule: a real first trip wins over a later downstream symptom.
    assert controlled_stop.prefer_first_trip(CARTON_TILT_REASON, HEARTBEAT_REASON) == CARTON_TILT_REASON
    # the full coldwin33 chain (tilt, then two heartbeat losses) still names the tilt
    chain = [CARTON_TILT_REASON, HEARTBEAT_REASON, HEARTBEAT_REASON]
    first = chain[0]
    for later in chain[1:]:
        first = controlled_stop.prefer_first_trip(first, later)
    assert first == CARTON_TILT_REASON


def test_prefer_first_trip_leaves_normal_chains_unchanged():
    # Two non-trip reasons: the normal last-reason behaviour is preserved (byte-identical).
    assert controlled_stop.prefer_first_trip('blocked_no_route_timeout', 'goal_timeout') == 'goal_timeout'
    assert controlled_stop.prefer_first_trip('', 'goal_timeout') == 'goal_timeout'
    # a real trip followed by ANOTHER real trip keeps the latest (never masks a second genuine trip)
    second = 'host emergency stop (host fault: CARTON_TILT ... 2.40 deg)'
    assert controlled_stop.prefer_first_trip(CARTON_TILT_REASON, second) == second


def test_is_trip_reason_and_symptom():
    assert controlled_stop.is_trip_reason(CARTON_TILT_REASON)
    assert not controlled_stop.is_trip_reason(HEARTBEAT_REASON)
    assert controlled_stop.is_downstream_symptom(HEARTBEAT_REASON)
    assert not controlled_stop.is_downstream_symptom(CARTON_TILT_REASON)


def test_decide_welded_stop_holds_and_never_drives_while_interpenetrating():
    # Welded + the carton interpenetrating the deck (coldwin33 at the trip): hold the arm energised,
    # keep the base brake, NEVER drive, and do NOT release the weld.
    d = controlled_stop.decide_welded_stop(welded=True, interpenetrating=True, supported=False,
                                           retreat_clear=False, trip_reason=CARTON_TILT_REASON)
    assert d.hold_arm and d.keep_base_brake and not d.allow_base_drive and not d.release_weld
    assert not d.retreat and d.reason == CARTON_TILT_REASON
    # a clear retreat exists -> lift/back off along the approach (still no base drive, weld kept)
    d2 = controlled_stop.decide_welded_stop(welded=True, interpenetrating=True, supported=False,
                                            retreat_clear=True)
    assert d2.retreat and not d2.allow_base_drive and not d2.release_weld


def test_decide_welded_stop_releases_only_when_supported_or_clear():
    # supported on a surface -> may release the weld
    assert controlled_stop.decide_welded_stop(welded=True, interpenetrating=False, supported=True,
                                              retreat_clear=False).release_weld
    # clear of the deck (not interpenetrating) -> may release the weld
    assert controlled_stop.decide_welded_stop(welded=True, interpenetrating=False, supported=False,
                                              retreat_clear=False).release_weld
    # nothing welded -> no controlled stop (the normal failure path is untouched)
    d = controlled_stop.decide_welded_stop(welded=False, interpenetrating=True, supported=False,
                                           retreat_clear=False)
    assert not d.hold_arm and not d.keep_base_brake and d.allow_base_drive and not d.release_weld


# ============================================================ MobileController brake-hold latch
class _FakeDrive:
    def __init__(self):
        self.target_velocity = None

    def GetTargetVelocityAttr(self):
        drive = self

        class _Attr:
            def Set(self, v):
                drive.target_velocity = v
        return _Attr()


def _fake_mobile(welded_hold):
    """A duck object exercising the REAL MobileController.command/park/latch_welded_hold guard
    without pxr: it records every brake-release call so we can prove none fire while latched."""
    released = []
    fake = NS(_welded_hold=welded_hold, drives=[_FakeDrive(), _FakeDrive()], released=released)
    for name in ('release_brake', 'release_base_full_brake', 'release', 'release_cart_brake'):
        fake.__dict__[name] = (lambda n: (lambda *a, **k: released.append(n)))(name)
    fake.park = MobileController.park.__get__(fake, type(fake))
    fake.command = MobileController.command.__get__(fake, type(fake))
    fake.latch_welded_hold = MobileController.latch_welded_hold.__get__(fake, type(fake))
    fake.release_welded_hold = MobileController.release_welded_hold.__get__(fake, type(fake))
    return fake


def test_command_latched_never_releases_a_brake_and_parks():
    fake = _fake_mobile(welded_hold=True)
    out = fake.command(0.3, 0.0, NS(age=0.0))  # a drive request during the controlled stop
    assert out == (0.0, 0.0)                    # parked
    assert fake.released == []                  # NO brake released -> base stays planted
    assert all(d.target_velocity == 0.0 for d in fake.drives)  # wheels held at zero


def test_command_unlatched_releases_brakes_as_before():
    # Baseline: WITHOUT the latch, command() releases every brake (observation None -> park after).
    fake = _fake_mobile(welded_hold=False)
    fake.command(0.3, 0.0, None)
    assert set(fake.released) == {'release_brake', 'release_base_full_brake', 'release', 'release_cart_brake'}


def test_latch_toggle():
    fake = _fake_mobile(welded_hold=False)
    fake.latch_welded_hold()
    assert fake._welded_hold is True
    fake.command(0.3, 0.0, NS(age=0.0))
    assert fake.released == []            # latched -> no release
    fake.release_welded_hold()
    assert fake._welded_hold is False


# ============================================================ ArmCommandGate heartbeat suppression
def _arm_gate(monkeypatch, wall):
    monkeypatch.setattr(native_transport.time, 'monotonic', lambda: wall[0])
    robot = NS(get_dof_index=lambda name: 0,
               get_joint_positions=lambda joint_indices=None: np.zeros(6))
    applied = []
    gate = native_transport.ArmCommandGate(robot, [(-10., 10.)] * 6,
                                           lambda q, v, i: applied.append((q, v)))
    gate.applied = applied
    return gate


def test_heartbeat_suppressed_during_controlled_stop_still_holds_no_raise(monkeypatch):
    wall = [0.0]
    gate = _arm_gate(monkeypatch, wall)
    gate.started = True
    gate._host_now = 10.0
    gate.suppress_heartbeat = True      # the controlled stop is holding the arm deliberately
    wall[0] = 100.0
    gate.watchdog(10.0 + native_transport.ARM_HEARTBEAT_SIM_BACKSTOP_S + 1.0)  # host stamps frozen
    assert gate.faulted                 # still HELD energised (not slack)...
    assert gate.applied                 # ...the hold target was applied...
    # ...but no 'arm command heartbeat lost' raised over the true trip.


def test_heartbeat_fires_when_not_suppressed(monkeypatch):
    wall = [0.0]
    gate = _arm_gate(monkeypatch, wall)
    gate.started = True
    gate._host_now = 10.0
    assert gate.suppress_heartbeat is False
    with pytest.raises(RuntimeError, match='arm command heartbeat lost'):
        gate.watchdog(10.0 + native_transport.ARM_HEARTBEAT_SIM_BACKSTOP_S + 1.0)


# ============================================================ IsaacPort controlled-stop wiring
class _FakeMobilePort:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        # record every brake method call (release_brake / engage_base_full_brake / engage_brake /
        # release_base_full_brake / latch_welded_hold ...)
        def rec(*a, **k):
            self.calls.append(name)
        return rec


def _port_stub():
    """A duck object the IsaacPort controlled-stop methods are bound onto (no Isaac construction)."""
    mobile = _FakeMobilePort()
    gate = NS(suppress_heartbeat=False)
    holds = []
    port = NS(_welded_stop=False, _welded_stop_reason=None, _carton_welded=True,
              mobile=mobile, command_gate=gate, hold_arm=lambda: holds.append('hold'), holds=holds)
    for name in ('enter_welded_controlled_stop', '_apply_base_brake', 'carton_welded', 'base_attitude'):
        setattr(port, name, getattr(IsaacPort, name).__get__(port, type(port)))
    port._BASE_LOOP_PHASES = IsaacPort._BASE_LOOP_PHASES
    return port


def test_enter_welded_controlled_stop_pins_base_holds_arm_suppresses_heartbeat():
    port = _port_stub()
    port.enter_welded_controlled_stop(CARTON_TILT_REASON)
    assert port._welded_stop is True and port._welded_stop_reason == CARTON_TILT_REASON
    # latched the mobile welded-hold + engaged the FULL base brake (base stays planted)
    assert 'latch_welded_hold' in port.mobile.calls
    assert 'engage_base_full_brake' in port.mobile.calls
    assert 'release_brake' in port.mobile.calls          # swapped off the yaw-lock loop-relief brake
    assert port.command_gate.suppress_heartbeat is True  # deliberate hold is not a heartbeat death
    assert port.holds == ['hold']                        # arm held energised (not slack)


def test_apply_base_brake_forces_full_lock_during_controlled_stop():
    port = _port_stub()
    # Before the controlled stop, a WELDED carton resting at 'place' takes the YAW-LOCK-ONLY brake
    # (translation free) -- the exact relief that let the fling start.
    port._apply_base_brake(True, True, 'place')
    assert 'engage_brake' in port.mobile.calls and 'engage_base_full_brake' not in port.mobile.calls
    # Once the controlled stop is latched, the SAME phase takes the FULL lock instead (base planted).
    port.mobile.calls.clear()
    port._welded_stop = True
    port._apply_base_brake(True, True, 'place')
    assert 'engage_base_full_brake' in port.mobile.calls and 'engage_brake' not in port.mobile.calls


def test_base_attitude_reads_roll_pitch_z():
    port = _port_stub()
    # identity quaternion (w=1) at z=0.24 -> level
    port.robot_body = NS(get_world_pose=lambda: (np.array([0., 0., 0.24]), np.array([1., 0., 0., 0.])),
                         get_linear_velocity=lambda: np.zeros(3),
                         get_angular_velocity=lambda: np.zeros(3))
    roll, pitch, z = port.base_attitude()
    assert abs(roll) < 1e-6 and abs(pitch) < 1e-6 and abs(z - 0.24) < 1e-9
    assert port.carton_welded() is True


# ============================================================ RuntimeAdapter fault note + retry gate
def _adapter_stub(*, welded, attitude):
    """Bind the three new RuntimeAdapter methods onto a duck self with a fake port."""
    entered = []
    port = NS(carton_welded=lambda: welded,
              base_attitude=lambda: attitude,
              enter_welded_controlled_stop=lambda reason: entered.append(reason))
    a = NS(port=port, _welded_controlled_stop=False, _first_trip_reason=None,
           _attitude_band=controlled_stop.AttitudeBand.from_nominal(*LEVEL), entered=entered)
    for name in ('_note_manip_failure', 'manip_retry_gate', 'first_trip_reason'):
        setattr(a, name, getattr(RuntimeAdapter, name).__get__(a, type(a)))
    return a


def test_note_manip_failure_enters_controlled_stop_and_records_first_trip():
    a = _adapter_stub(welded=True, attitude=LEVEL)
    a._note_manip_failure(CARTON_TILT_REASON)
    assert a._welded_controlled_stop is True
    assert a.entered == [CARTON_TILT_REASON]           # controlled stop entered at the base/weld owner
    assert a.first_trip_reason() == CARTON_TILT_REASON  # first trip preserved


def test_note_manip_failure_not_welded_is_inert():
    a = _adapter_stub(welded=False, attitude=LEVEL)
    a._note_manip_failure('reach stalled: approach')
    assert a._welded_controlled_stop is False and a.entered == []
    # a non-welded, non-trip reason does NOT claim the first-trip slot
    assert a.first_trip_reason() is None


def test_manip_retry_gate_refuses_while_controlled_stop_active():
    a = _adapter_stub(welded=True, attitude=LEVEL)
    a._note_manip_failure(CARTON_TILT_REASON)       # latches the controlled stop
    refusal = a.manip_retry_gate()
    assert refusal is not None and 'welded controlled stop' in refusal


def test_manip_retry_gate_refuses_flipped_base_allows_level():
    # Not welded, but the base is flipped (a prior fling): the gate refuses on attitude.
    flipped = _adapter_stub(welded=False, attitude=FLIPPED)
    refusal = flipped.manip_retry_gate()
    assert refusal is not None and 'out of sane band' in refusal
    # A level base with nothing welded -> allowed (byte-identical normal retry).
    level = _adapter_stub(welded=False, attitude=LEVEL)
    assert level.manip_retry_gate() is None


def test_manip_retry_gate_inert_without_band_or_attitude():
    a = _adapter_stub(welded=False, attitude=LEVEL)
    a._attitude_band = None
    assert a.manip_retry_gate() is None             # no band captured -> gate inert


# ============================================================ coldwin33 replay through StageRetry (BT tier)
py_trees = pytest.importorskip('py_trees')
from py_trees.common import Status  # noqa: E402
from twin.restock.bt.leaves import StageRetry  # noqa: E402
from twin.restock.bt.events import Event  # noqa: E402


class _FakeRun:
    def __init__(self, ctx, action='collect'):
        self.ctx = ctx
        self.action = action
        self.fail_reason = None
        self.resets = 0

    def reset_for_retry(self):
        self.resets += 1


class _FakeDecorated:
    def __init__(self):
        self.status = Status.FAILURE
        self.stopped = []

    def stop(self, status):
        self.stopped.append(status)


class _FakeCtx:
    def __init__(self, adapter):
        self.mission = NS(adapter=adapter, terminal=False, reason='',
                          fail=self._fail)
        self.reactions = NS(reobserve=lambda: None, replan=lambda: None, wait=lambda: None)
        self.emitted = []
        self.logs = []

    def _fail(self, reason):
        self.mission.terminal = True
        self.mission.reason = reason

    def emit_stage_failure(self, event):
        self.emitted.append(event)

    def log(self, etype, data, action=None):
        self.logs.append((etype, data, action))


def _stage_retry(adapter):
    ctx = _FakeCtx(adapter)
    run = _FakeRun(ctx)
    dec = _FakeDecorated()
    node = StageRetry.__new__(StageRetry)
    node.run = run
    node.max_retries = 2
    node.retries = 0
    node._first_trip = None
    node.decorated = dec
    node.feedback_message = ''
    return node, run, ctx


def test_coldwin33_replay_aborts_immediately_and_names_carton_tilt():
    # Attempt 1 fails with the CARTON_TILT trip while welded -> the adapter latched the controlled
    # stop in step() (replayed here). StageRetry must ABORT immediately (no re-begin drive) and name
    # the CARTON_TILT first trip -- never the downstream heartbeat loss, and never a retry on the
    # flipped base.
    adapter = _adapter_stub(welded=True, attitude=LEVEL)
    adapter._note_manip_failure(CARTON_TILT_REASON)   # what runtime.step did at the fault
    node, run, ctx = _stage_retry(adapter)
    run.fail_reason = CARTON_TILT_REASON
    status = node.update()
    assert status == Status.FAILURE
    assert run.resets == 0                            # NO re-begin -> no drive -> no fling
    assert ctx.mission.terminal and ctx.mission.reason == CARTON_TILT_REASON  # named first trip


def test_normal_collect_grasp_failure_still_retries():
    # A non-welded grasp failure on a level base retries exactly as before (byte-identical).
    adapter = _adapter_stub(welded=False, attitude=LEVEL)
    node, run, ctx = _stage_retry(adapter)
    run.fail_reason = 'reach stalled: grasp did not seal'
    status = node.update()
    assert status == Status.RUNNING and run.resets == 1 and node.retries == 1


def test_downstream_heartbeat_after_tilt_still_names_tilt():
    # Even if a later attempt surfaces the heartbeat loss, the preserved first trip wins at abort.
    adapter = _adapter_stub(welded=True, attitude=LEVEL)
    adapter._note_manip_failure(CARTON_TILT_REASON)
    node, run, ctx = _stage_retry(adapter)
    node._first_trip = CARTON_TILT_REASON             # first attempt already recorded the tilt
    run.fail_reason = HEARTBEAT_REASON                # a later attempt reports the downstream symptom
    node.update()
    assert ctx.mission.reason == CARTON_TILT_REASON
