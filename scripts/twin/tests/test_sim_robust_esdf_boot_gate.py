"""sim-robust-esdf-boot-gate-v1: name a never-acknowledged nvblox ESDF incorporation at BOOT,
on the sim clock, BEFORE the first arm motion streams -- instead of the opaque ~40 s mid-stage
collision_dt invariant trip (meshuris1).

The two staged transfer-only runs are modelled from the logged incorporation-ack signal:
  * meshuris1 (FAILED): the mirrored station collision_time stays at the -1 not-ready sentinel from
    the first tick (nvblox never incorporated against a stale cc-bench1 baseline); the controller
    collision_dt invariant only discovered it at the first observe step now=39 (collision_dt=40).
  * meshuris0 (PASSED): collision_time leaves -1 for a real integration stamp (resp_stamp 0.000 ->
    102.650, dt 0.600) -> transfer + verify complete, no collision_dt trip.

CPU-only: the pure EsdfBootGate and the REAL TransferController.step seam driven with test clocks;
no Isaac / ROS / GPU. See scripts/twin/restock_manipulation/esdf_boot_gate.py and the report at
data/sim-robust-esdf-boot-gate-v1/report.md.
"""
from scripts.twin.restock_manipulation import esdf_boot_gate
from scripts.twin.restock_manipulation.esdf_boot_gate import EsdfBootGate, HOLD, PROCEED

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
