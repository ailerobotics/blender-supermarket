"""sim-robust-esdf-boot-gate (v1 gate, v2 facts/proof): name a never-acknowledged nvblox ESDF
incorporation at BOOT, on the sim clock, BEFORE the first arm motion streams -- instead of the
opaque ~40 s mid-stage collision_dt invariant trip, AND absorb the one-tick boot race the GPU
proof actually exercised.

v2 established from the three run dirs (reading (i), see data/sim-robust-esdf-boot-gate-v2/report.md):
  * meshuris1 (FAILED, stale cc-bench1 baseline, gate OFF): collision_time read -1 at the FIRST observe
    step (now=39) and the collision_dt invariant e-stopped it (collision_dt = now - (-1) = 40) BEFORE
    the (one-tick-late) incorporation could land.
  * esdfgate1 (SAME stale baseline, gate ON): held EXACTLY ONE tick at now=39, then ran to completion
    at sim 174 with ZERO collision_dt trips -- only possible with continuous REAL incorporation. So the
    stale baseline incorporates one tick late; it is NOT a never-ack reproduction, and the gate's hold
    absorbed that one-tick race.
  * meshuris0 (fresh commission, gate OFF): incorporated during warm-up (fresh collision_time by now=39)
    -> transfer + verify complete, no trip.
The genuine fail-fast TRIP needs a real never-incorporation (RESTOCK_ESDF_FORCE_NEVER_ACK, or a
disabled masked-depth feed/mapper), modelled here on the IncorporationAckState source.

CPU-only: the pure EsdfBootGate, the REAL TransferController.step seam driven with test clocks, and
the REAL IncorporationAckState (no Isaac / ROS / GPU). See
scripts/twin/restock_manipulation/esdf_boot_gate.py and scripts/twin/restock_host/incorporation_ack.py.
"""
import contextlib
import io
import os

import pytest

from scripts.twin.restock_manipulation import esdf_boot_gate
from scripts.twin.restock_manipulation.esdf_boot_gate import EsdfBootGate, HOLD, PROCEED
from scripts.twin.restock_host.incorporation_ack import IncorporationAckState

# Reuse the controller CPU fixtures (FixtureMotion / make_controller / packet / run_to).
from scripts.twin.tests.test_restock_manipulation import make_controller, packet, run_to

GRACE = 15.0
PARKED_SINCE = 1.667   # meshuris0/1 parked_since; a valid ack is a stamp strictly after it


# --------------------------------------------------------------------------- pure gate: disabled
def test_disabled_grace_always_proceeds_byte_identical():
    gate = EsdfBootGate(0.0)
    assert not gate.enabled
    # Even a permanently-stuck -1 stamp, far past any plausible grace, proceeds untouched.
    assert gate.observe(0.0, esdf_boot_gate.NOT_READY) is PROCEED
    assert gate.observe(1000.0, esdf_boot_gate.NOT_READY) is PROCEED


def test_negative_grace_is_disabled():
    assert not EsdfBootGate(-5.0).enabled


# ------------------------------------------------------------- pure gate: meshuris1 stuck stamp
def test_stuck_stamp_holds_through_grace_then_fails_by_name():
    gate = EsdfBootGate(GRACE)
    op_start = 39.0   # the meshuris1 transfer op first stepped observe at now=39
    # Within the grace the op is HELD (no motion), never the opaque collision_dt invariant.
    assert gate.observe(op_start, esdf_boot_gate.NOT_READY) is HOLD
    assert gate.observe(op_start + 5.0, esdf_boot_gate.NOT_READY) is HOLD
    assert gate.observe(op_start + GRACE, esdf_boot_gate.NOT_READY) is HOLD   # boundary: still within
    # Once the sim-clock grace is exceeded with the stamp still stuck, fail fast by name.
    reason = gate.observe(op_start + GRACE + 0.001, esdf_boot_gate.NOT_READY)
    assert isinstance(reason, str)
    assert 'nvblox ESDF incorporation never acknowledged' in reason
    assert 'boot grace' in reason
    assert 'collision_time stuck at -1.000' in reason
    assert 'stale/uncommissioned station baseline' in reason
    # It fails BOUNDED BY THE GRACE (~op_start+15), not the ~40 s the collision_dt invariant would take.
    assert op_start + GRACE < op_start + GRACE + 0.001 < op_start + 40.0


# --------------------------------------------------------------- pure gate: meshuris0 valid ack
def test_valid_ack_disarms_and_stays_disarmed():
    gate = EsdfBootGate(GRACE)
    # meshuris0: the first stepped tick already carries a real integration stamp (102.650 > parked).
    assert gate.observe(103.250, 102.650) is PROCEED
    assert gate._satisfied
    # A later transient -1 must NOT re-arm the gate -- post-ack freshness is the collision_dt invariant's job.
    assert gate.observe(104.0, esdf_boot_gate.NOT_READY) is PROCEED
    # A valid ack arriving mid-grace (not on the first tick) also disarms before the grace expires.
    g2 = EsdfBootGate(GRACE)
    assert g2.observe(39.0, esdf_boot_gate.NOT_READY) is HOLD
    assert g2.observe(42.0, 41.9) is PROCEED
    assert g2._satisfied


def test_reset_rearms_for_a_new_op():
    gate = EsdfBootGate(GRACE)
    assert gate.observe(39.0, 38.9) is PROCEED and gate._satisfied
    gate.reset()
    assert not gate._satisfied and gate._op_start is None
    assert gate.observe(39.0, esdf_boot_gate.NOT_READY) is HOLD


# ------------------------------------------------------------- pure gate: SIM clock, not wall
def test_grace_is_sim_clock_not_wall():
    gate = EsdfBootGate(GRACE)
    op_start = 39.0
    gate.observe(op_start, esdf_boot_gate.NOT_READY)
    # Many repeated observations at a FROZEN sim clock (as wall time would pass) never trip: the gate
    # keys only on sim `now`, so no amount of wall-time / call-count advances it.
    for _ in range(1000):
        assert gate.observe(op_start + 5.0, esdf_boot_gate.NOT_READY) is HOLD
    # Advancing the SIM clock past the grace is the only thing that trips it.
    assert isinstance(gate.observe(op_start + GRACE + 0.1, esdf_boot_gate.NOT_READY), str)


# =============================================================== real TransferController.step seam
def _collect_controller(grace):
    # Collect op: the gate short-circuits before any collect-specific observe logic, so the op choice
    # is immaterial; collect keeps the fixture simple.
    return make_controller('collect', esdf_boot_ack_grace_s=grace)


def test_controller_holds_then_fails_named_on_stuck_stamp():
    # meshuris1 through the REAL controller: collision_time stuck at -1 from the first observe step.
    c, m = _collect_controller(GRACE)
    assert c.phase == 'observe'
    # First steps within the grace HOLD: running, phase unchanged, and NO arm motion streamed.
    r = c.step(1.0, packet(c, 1.0, collision_time=-1.0))
    assert r.status == 'running' and c.phase == 'observe'
    r = c.step(9.0, packet(c, 10.0, collision_time=-1.0))     # now=10, within grace
    assert r.status == 'running' and c.phase == 'observe'
    assert m.moves == [] and m.stows == []                     # the first arm motion never streamed
    # Past the sim-clock grace with the stamp still stuck: fail fast by name, NOT the collision_dt invariant.
    r = c.step(7.0, packet(c, 17.0, collision_time=-1.0))      # now=17 > op_start(1)+15
    assert r.status == 'failed'
    assert 'nvblox ESDF incorporation never acknowledged' in r.reason
    # The gate owns the failure, NOT the opaque collision_dt invariant (its trip reason never appears).
    assert 'base moving or stale/missing feedback/collision world' not in r.reason
    assert 'collision_dt=' not in r.reason
    assert m.moves == []                                        # still no arm motion


def test_controller_disarms_on_valid_ack_and_runs_normally():
    # meshuris0 through the REAL controller: a fresh collision_time at the op -> gate silent, run proceeds.
    c, m = _collect_controller(GRACE)
    reached = run_to(c, 'approach', t0=1.0)   # fresh packets (collision_time=t) -> gate disarms on step 1
    assert reached is not None
    assert c._esdf_boot_gate._satisfied       # disarmed by the first valid ack
    assert c.phase == 'approach'              # advanced exactly as a healthy run


def test_controller_key_off_is_byte_identical_legacy_invariant():
    # Gate OFF (grace 0): a stuck -1 stamp takes the LEGACY path -- the collision_dt invariant trips at
    # the first observe step, exactly as before this change (proving byte-identical when off).
    c, m = _collect_controller(0.0)
    r = c.step(1.0, packet(c, 1.0, collision_time=-1.0))
    assert r.status == 'failed'
    assert 'collision_dt' in r.reason                          # the pre-existing opaque invariant
    assert 'nvblox ESDF incorporation never acknowledged' not in r.reason


# ========================================================= v2: once-per-op sentinel->valid transition
def test_just_satisfied_is_a_one_shot_transition():
    # The controller consumes this one-shot to log ESDF_BOOT_GATE_READY exactly once per op.
    gate = EsdfBootGate(GRACE)
    assert not gate.consume_just_satisfied()                 # nothing observed yet
    assert gate.observe(39.0, esdf_boot_gate.NOT_READY) is HOLD
    assert not gate.consume_just_satisfied()                 # a HOLD is not the transition
    assert gate.observe(42.0, 41.9) is PROCEED               # the first valid ack: sentinel -> valid
    assert gate.consume_just_satisfied()                     # fires exactly once, on that step
    assert not gate.consume_just_satisfied()                 # and is then consumed
    assert gate.observe(43.0, 42.9) is PROCEED               # already satisfied, no new transition
    assert not gate.consume_just_satisfied()
    gate.reset()                                             # a fresh op re-arms + clears the one-shot
    assert not gate.consume_just_satisfied()


def test_disabled_gate_signals_no_transition_byte_identical_console():
    # Gate OFF must never fire the READY transition, so a gate-off run stays byte-identical on the console.
    off = EsdfBootGate(0.0)
    assert off.observe(1.0, 5.0) is PROCEED
    assert not off.consume_just_satisfied()
    assert off.observe(2.0, esdf_boot_gate.NOT_READY) is PROCEED
    assert not off.consume_just_satisfied()


# =============================================== v2 reading (i): any finite >=0 is the invariant handoff
def test_any_finite_nonneg_stamp_disarms_handing_off_to_invariant():
    # Reading (i): the arm-host chain only ever produces a finite >=0 collision_time from a REAL nvblox
    # integration (v2 traced acknowledge -> station snapshot ready -> StationClient -> feedback mirror ->
    # hold_last_good_collision -> sync_packet_clock; every hop passes -1 through untouched, no default /
    # wall / fabricated >=0). So >=0 is the correct handoff point to the collision_dt freshness invariant:
    # ANY finite >=0 disarms, and a later transient -1 never re-arms (post-handoff staleness is the
    # invariant's job, not the gate's).
    gate = EsdfBootGate(GRACE)
    assert gate.observe(39.0, 1000.0) is PROCEED             # even a far-future stamp = "a world exists" -> disarm
    assert gate._satisfied and gate.consume_just_satisfied()
    assert gate.observe(40.0, esdf_boot_gate.NOT_READY) is PROCEED   # a transient -1 does NOT re-arm
    assert not gate.consume_just_satisfied()                 # no new transition
    # A NaN / non-finite stamp is NOT a valid ack: keep HOLDing, never a spurious disarm.
    g2 = EsdfBootGate(GRACE)
    assert g2.observe(39.0, float('nan')) is HOLD
    assert not g2._satisfied


# ======================================= v2 (a): one-tick boot race -> hold once, proceed, invariant silent
def test_controller_one_tick_hold_then_proceeds_invariant_silent():
    # The esdfgate1 shape through the REAL controller: a single HOLD at the op's first observe step
    # (stuck -1), then the incorporation lands one step later -> PROCEED + ONE READY line, and the
    # collision_dt invariant stays SILENT across the rest of the run (run_to asserts no failure).
    c, m = _collect_controller(GRACE)
    r = c.step(.1, packet(c, 1.0, collision_time=-1.0))      # one-tick boot race: stuck -1
    assert r.status == 'running' and c.phase == 'observe'
    assert m.moves == [] and m.stows == []                   # no motion streamed during the HOLD
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = c.step(.1, packet(c, 1.1))                       # fresh stamp lands -> PROCEED + READY
    assert r.status != 'failed'
    out = buf.getvalue()
    assert out.count('ESDF_BOOT_GATE_READY') == 1            # the handoff logged exactly once
    assert 'collision_time=1.100' in out and 'now=1.100' in out and 'source=' in out
    reached = run_to(c, 'approach', t0=1.2)                  # invariant silent => reaches approach, no fail
    assert reached is not None and c.phase == 'approach'
    assert not c._esdf_boot_gate.consume_just_satisfied()    # the transition fired once, already consumed


def test_controller_logs_ready_once_and_silent_when_off():
    # Gate ON: the healthy case (fresh ack on step 1, no HOLD) still logs READY once with value+now+source.
    c, m = _collect_controller(GRACE)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c.step(.1, packet(c, 1.0))                           # fresh ack on the first step
    out = buf.getvalue()
    assert out.count('ESDF_BOOT_GATE_READY') == 1
    assert 'collision_time=1.000' in out and 'now=1.000' in out and 'source=' in out
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        run_to(c, 'approach', t0=1.1)                        # no second READY on later steps
    assert 'ESDF_BOOT_GATE_READY' not in buf2.getvalue()
    # Gate OFF: never logs READY (byte-identical console).
    c2, _ = _collect_controller(0.0)
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        c2.step(.1, packet(c2, 1.0))
    assert 'ESDF_BOOT_GATE_READY' not in buf3.getvalue()


# ============================ v2 (b): genuine never-incorporation at the REAL ack source (TRIP reproduction)
def _active_ack_state():
    s = IncorporationAckState()
    s.epoch, s.active, s.parked_since = 1, True, 1.0
    s.last_heartbeat = 2.0          # keep the liveness watchdog happy so we isolate the ack behaviour
    return s


def test_force_never_ack_env_default_off_is_byte_identical():
    # Default (unset): acknowledge advances collision_time and the station reaches 'ready' as today.
    assert 'RESTOCK_ESDF_FORCE_NEVER_ACK' not in os.environ
    s = _active_ack_state()
    assert not s.force_never_ack
    s.acknowledge(2.0, 1)
    assert s.collision_time == 2.0
    snap = s.snapshot(2.1, endpoints_ready=True, process_alive=True)
    assert snap['state'] == 'ready' and snap['collision_time'] == 2.0


def test_force_never_ack_reproduces_stuck_sentinel_for_the_trip(monkeypatch):
    # Proof switch ON: acknowledge REFUSES through the real reject path, collision_time stays at the -1
    # sentinel, and the station never reaches 'ready' -> the arm host keeps reading -1 -> the gate HOLDs
    # through the grace and TRIPs by name. This is the genuine never-incorporation the stale baseline is NOT.
    monkeypatch.setenv('RESTOCK_ESDF_FORCE_NEVER_ACK', '1')
    s = _active_ack_state()
    assert s.force_never_ack
    with pytest.raises(ValueError, match='FORCE_NEVER_ACK'):
        s.acknowledge(2.0, 1)
    assert s.collision_time == -1.0                          # never leaves the not-ready sentinel
    snap = s.snapshot(2.1, endpoints_ready=True, process_alive=True)
    assert snap['state'] == 'warming'                        # active but never ready
    assert snap['collision_time'] == -1.0                    # the wire the StationClient reads stays -1

    # Feed that stuck -1 through the REAL controller gate: HOLD through the grace, then named TRIP, and
    # NEVER the opaque collision_dt invariant (its reason never appears) and never any arm motion.
    c, m = _collect_controller(GRACE)
    assert c.step(.1, packet(c, 1.0, collision_time=snap['collision_time'])).status == 'running'
    assert c.step(.1, packet(c, 10.0, collision_time=snap['collision_time'])).status == 'running'
    r = c.step(.1, packet(c, 17.0, collision_time=snap['collision_time']))   # now=17 > op_start(1)+15
    assert r.status == 'failed'
    assert 'nvblox ESDF incorporation never acknowledged' in r.reason
    assert 'collision_dt=' not in r.reason and 'base moving or stale' not in r.reason
    assert m.moves == []
