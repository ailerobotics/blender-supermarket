"""CPU proof for the human-aware RUN/SLOW/STOP/BACKSTOP arm authority
(sim-arm-reactive-replan-v1, PR-A). No ROS / Isaac / GPU.

Three tiers, all driving a SCRIPTED person distance:
  1. the ported humangate decision core (ArmHumanGate) — approach/slow/stop/dwell/
     lost-close/heartbeat/stale/backstop, adapted from ur10-setup/humangate;
  2. the ReactiveArmSupervisor — the scripted person drives RUN -> SLOW -> STOP ->
     resume, preempting a running leg via the e-stop-free cancel path and driving
     the reversible arm-command-gate pause;
  3. the REAL TransferController.step seam — the supervisor blocks + pauses the arm
     on a person and releases on a clear dwell, and is byte-identical when off;
  4. ArmCommandGate.set_paused — the reversible, non-latching freeze.
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation.arm_human_gate import (  # noqa: E402
    ArmGateConfig, ArmHumanGate, ArmSafetyFrame, BACKSTOP, RUN, SLOW, STOP,
    frame_from_safety_state)
from twin.restock_manipulation.arm_backstop import ArmBackstopParams, ArmBackstopWatcher  # noqa: E402
from twin.restock_manipulation.reactive_arm import ReactiveArmSupervisor  # noqa: E402
from twin.restock_manipulation import Sensors, ProductProfile, TransferController  # noqa: E402
from twin.restock import native_transport  # noqa: E402


# ----------------------------------------------------------------- scripted person helpers
def person(zone, dist=None, *, stamp, health_ok=True, flags=()):
    """A vendored-SafetyState-shaped object (what the detection provider returns)."""
    return NS(zone=zone, nearest_distance=dist, evidence_stamp=stamp,
              health_ok=health_ok, health_flags=frozenset(flags))


def drive_gate(gate, script):
    """Feed (t, zone, dist) samples with a monotonic heartbeat; return [(t, action)]."""
    out, seq = [], 0
    for t, zone, dist in script:
        seq += 1
        gate.on_state(ArmSafetyFrame(stamp=t, zone=zone, nearest_distance_m=dist,
                                     heartbeat_seq=seq), t, t)
        out.append((t, gate.decide(t, t).action))
    return out


# =========================================================== tier 1: the gate core
def test_clean_start_runs_immediately_without_startup_dwell():
    g = ArmHumanGate(ArmGateConfig(clear_dwell_s=2.0))
    g.on_state(ArmSafetyFrame(stamp=0.0, zone="CLEAR", heartbeat_seq=1), 0.0, 0.0)
    d = g.decide(0.0, 0.0)
    assert d.action == RUN and d.speed_fraction == 1.0


def test_require_initial_dwell_blocks_first_run():
    g = ArmHumanGate(ArmGateConfig(clear_dwell_s=2.0, require_initial_dwell=True))
    g.on_state(ArmSafetyFrame(stamp=0.0, zone="CLEAR", heartbeat_seq=1), 0.0, 0.0)
    assert g.decide(0.0, 0.0).action == STOP   # must earn the first RUN through the dwell


def test_person_approach_slow_then_stop_then_resume():
    g = ArmHumanGate(ArmGateConfig(clear_dwell_s=1.0))
    actions = dict(drive_gate(g, [
        (0.0, "CLEAR", None),      # clean start -> RUN
        (1.0, "SLOW", 0.8),        # person in the SLOW ring
        (1.5, "STOP", 0.3),        # person in the STOP ring
        (2.0, "SLOW", 0.8),        # tracked leaving
        (2.1, "CLEAR", None),      # clear, dwell starts
        (2.5, "CLEAR", None),
        (3.2, "CLEAR", None),      # dwell complete -> RUN
    ]))
    assert actions[0.0] == RUN
    assert actions[1.0] == SLOW
    assert actions[1.5] == STOP
    assert actions[2.1] == STOP    # still dwelling
    assert actions[3.2] == RUN     # resumed only after the clear dwell


def test_lost_close_never_resumes_until_tracked_pass():
    g = ArmHumanGate(ArmGateConfig(clear_dwell_s=0.5))
    # person close then VANISHES (STOP -> CLEAR directly): provenance LOST_CLOSE, no resume.
    acts = dict(drive_gate(g, [
        (0.0, "CLEAR", None), (0.5, "STOP", 0.2), (0.6, "CLEAR", None),
        (1.2, "CLEAR", None), (2.0, "CLEAR", None)]))
    assert acts[2.0] == STOP       # never resumes on a lost-close departure


def test_heartbeat_timeout_is_stop():
    g = ArmHumanGate(ArmGateConfig())
    g.on_state(ArmSafetyFrame(stamp=0.0, zone="CLEAR", heartbeat_seq=1), 0.0, 0.0)
    assert g.decide(0.0, 0.0).action == RUN
    # no new frame for > heartbeat_timeout_s -> STOP (fail toward stop)
    assert g.decide(0.5, 0.5).action == STOP


def test_stale_evidence_is_stop():
    g = ArmHumanGate(ArmGateConfig(state_max_age_s=0.2))
    # evidence stamp is 0.5 s old relative to now_wall -> stale -> STOP
    g.on_state(ArmSafetyFrame(stamp=0.0, zone="CLEAR", heartbeat_seq=1), 0.5, 0.5)
    assert g.decide(0.5, 0.5).action == STOP


def test_unknown_distance_slow_never_runs():
    g = ArmHumanGate(ArmGateConfig(clear_dwell_s=0.5))
    # a present person with UNKNOWN distance is SLOW (never treated as far/clear)
    acts = dict(drive_gate(g, [(0.0, "CLEAR", None), (0.5, "SLOW", None)]))
    assert acts[0.5] == SLOW


def test_none_state_fails_toward_stop():
    g = ArmHumanGate(ArmGateConfig())
    frame = frame_from_safety_state(None, now_wall=1.0, heartbeat_seq=1)
    g.on_state(frame, 1.0, 1.0)
    assert g.decide(1.0, 1.0).action == STOP   # absence of evidence is never clearance


def test_sustained_stop_escalates_to_backstop_when_arm_not_stopped():
    g = ArmHumanGate(ArmGateConfig(backstop_deadline_s=0.3))
    g.on_state(ArmSafetyFrame(stamp=0.0, zone="CLEAR", heartbeat_seq=1), 0.0, 0.0)
    g.decide(0.0, 0.0)
    g.on_state(ArmSafetyFrame(stamp=0.1, zone="STOP", nearest_distance_m=0.2, heartbeat_seq=2), 0.1, 0.1)
    assert g.decide(0.1, 0.1, driver_stopped=False).action == STOP
    g.on_state(ArmSafetyFrame(stamp=0.6, zone="STOP", nearest_distance_m=0.2, heartbeat_seq=3), 0.6, 0.6)
    # STOP demanded for > deadline and the arm is NOT confirmed stopped -> BACKSTOP
    assert g.decide(0.6, 0.6, driver_stopped=False).action == BACKSTOP


# =========================================================== tier 1b: the backstop watcher
def test_backstop_watcher_cancels_on_gate_backstop():
    w = ArmBackstopWatcher(ArmBackstopParams(startup_grace_s=0.05))
    r = w.update(1.0, last_state_arrival=1.0, last_gate_arrival=1.0, state_zone="STOP",
                 gate_action=BACKSTOP, observed_scaling=0.0, gate_demand="STOP")
    assert r and w.tripped


def test_backstop_watcher_latches_on_state_heartbeat_loss():
    w = ArmBackstopWatcher(ArmBackstopParams(startup_grace_s=0.05, state_heartbeat_timeout_s=0.3))
    assert w.update(1.0, last_state_arrival=0.5, last_gate_arrival=1.0, state_zone="CLEAR",
                    gate_action=RUN, observed_scaling=1.0) is not None
    assert w.tripped
    # latched: keeps cancelling even once the stream recovers
    assert w.update(1.1, last_state_arrival=1.1, last_gate_arrival=1.1, state_zone="CLEAR",
                    gate_action=RUN, observed_scaling=1.0) is not None


# =========================================================== tier 2: the supervisor
class FakeMotion:
    """A motion port stub exposing the e-stop-free cancel path, recording calls."""

    def __init__(self, settle_ticks=0):
        self.cancels = 0
        self.stops = 0
        self._settle_ticks = settle_ticks   # ticks of 'still settling' before terminal
        self._since_cancel = None

    def cancel(self):
        self.cancels += 1
        self._since_cancel = 0

    def cancel_outcome(self):
        if self._since_cancel is None:
            return None
        if self._since_cancel < self._settle_ticks:
            self._since_cancel += 1
            return None
        self._since_cancel = None
        return "cancelled"

    def stop(self):
        self.stops += 1


def make_supervisor(**cfg):
    pauses = []
    sup = ReactiveArmSupervisor(
        gate_config=ArmGateConfig(clear_dwell_s=cfg.pop("clear_dwell_s", 1.0)),
        backstop_params=ArmBackstopParams(startup_grace_s=0.05),
        pause_sink=pauses.append, enable_backstop=cfg.pop("enable_backstop", True))
    sup.reset(0.0)
    return sup, pauses


def test_supervisor_run_allows_leg():
    sup, pauses = make_supervisor()
    m = FakeMotion()
    d = sup.supervise(now=0.0, now_wall=0.0, phase="lift", motion=m, leg_running=True,
                      state=person("CLEAR", stamp=0.0))
    assert not d.block and not d.reissue and d.action == RUN
    assert m.cancels == 0 and pauses == []


def test_supervisor_slow_preempts_and_reissues_slow():
    sup, _ = make_supervisor()
    m = FakeMotion()
    d = sup.supervise(now=1.0, now_wall=1.0, phase="lift", motion=m, leg_running=True,
                      state=person("SLOW", 0.8, stamp=1.0))
    assert d.slow and d.reissue and m.cancels == 1    # cancel the running leg, re-issue SLOW


def test_supervisor_stop_cancels_pauses_and_blocks_then_resumes():
    sup, pauses = make_supervisor(clear_dwell_s=0.5)
    m = FakeMotion()
    # clean run first
    sup.supervise(now=0.0, now_wall=0.0, phase="lift", motion=m, leg_running=True,
                  state=person("CLEAR", stamp=0.0))
    # person enters STOP ring while a leg runs
    d = sup.supervise(now=1.0, now_wall=1.0, phase="lift", motion=m, leg_running=True,
                      state=person("STOP", 0.2, stamp=1.0))
    assert d.block and d.action == STOP
    assert m.cancels == 1 and sup.paused and pauses[-1] is True
    # person leaves through the SLOW ring (TRACKED leaving, not lost-close), then clears; the
    # resume dwell accumulates; still blocked until it completes.
    sup.supervise(now=1.05, now_wall=1.05, phase="lift", motion=m, leg_running=False,
                  state=person("SLOW", 0.8, stamp=1.05))
    for t in (1.1, 1.3):
        d = sup.supervise(now=t, now_wall=t, phase="lift", motion=m, leg_running=False,
                          state=person("CLEAR", stamp=t))
        assert d.block
    # after the dwell -> RUN: release the pause, allow the re-issue
    d = sup.supervise(now=1.7, now_wall=1.7, phase="lift", motion=m, leg_running=False,
                      state=person("CLEAR", stamp=1.7))
    assert not d.block and d.action == RUN and not sup.paused and pauses[-1] is False


def test_supervisor_settle_waits_across_ticks_before_reissue():
    sup, _ = make_supervisor()
    m = FakeMotion(settle_ticks=2)      # cancel takes 2 ticks to settle
    # SLOW on a running leg -> cancel requested, but blocks while settling
    d = sup.supervise(now=1.0, now_wall=1.0, phase="lift", motion=m, leg_running=True,
                      state=person("SLOW", 0.8, stamp=1.0))
    assert d.block and not d.reissue and m.cancels == 1
    d = sup.supervise(now=1.05, now_wall=1.05, phase="lift", motion=m, leg_running=True,
                      state=person("SLOW", 0.8, stamp=1.05))
    assert d.block and m.cancels == 1   # still settling, not re-cancelled
    d = sup.supervise(now=1.1, now_wall=1.1, phase="lift", motion=m, leg_running=True,
                      state=person("SLOW", 0.8, stamp=1.1))
    assert d.reissue and d.slow         # settled -> re-issue SLOW


def test_supervisor_inert_outside_replannable_phases():
    sup, pauses = make_supervisor()
    m = FakeMotion()
    # a person in STOP, but the phase is a delicate frozen one (grasp): no preempt, no pause
    d = sup.supervise(now=1.0, now_wall=1.0, phase="grasp", motion=m, leg_running=True,
                      state=person("STOP", 0.2, stamp=1.0))
    assert not d.block and m.cancels == 0 and not sup.paused


# =========================================================== tier 3: the controller seam
PROFILE = ProductProfile("sku-box", (.10, .10, .10), (1., 0., 0., 0.))
JOB = dict(job_id="job-1", sku="sku-box", quantity=1, source_bin="backroom", shelf_id="shelf-a")


class StepMotion:
    """Minimal motion port for the controller seam: records cancels; polls 'running'."""

    def __init__(self):
        self.cancels = 0
        self.outcome = "running"
        self._cancelling = False

    def cancel(self):
        self.cancels += 1
        self._cancelling = True

    def cancel_outcome(self):
        if not self._cancelling:
            return None
        self._cancelling = False
        return "cancelled"

    def begin(self, *a, **k):
        pass

    def begin_joint_goal(self, *a, **k):
        pass

    def poll(self):
        return self.outcome

    def stop(self):
        pass

    def update_payload(self, held, profile):
        return True


def _sensors(now, joints=(0.,) * 6):
    return Sensors(simulation_time=now, observations=(), target=None, parked=True, contact=False,
                   feedback_time=now, collision_time=now, joint_positions=joints)


_POSE = (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)


def _controller(**kwargs):
    m = StepMotion()
    c = TransferController(m, {"sku-box": PROFILE}, fixture_mode=True, **kwargs)
    c.start("transfer", JOB)
    # park at a free-space re-plannable leg that is already running (so when the gate ALLOWS, _advance
    # just polls 'running' -- give it the pose map the general move path reads).
    c.phase = "approach"
    c.issued = True
    c.last_time = None
    c.elapsed = 0.0
    c.poses = {ph: _POSE for ph in ("approach", "contact", "lift", "preplace", "place", "retreat")}
    return c, m


def test_controller_off_is_byte_identical_no_supervisor():
    c, _ = _controller()   # reactive_arm_human defaults False
    assert c._reactive_arm is None


def test_controller_seam_blocks_and_pauses_on_person_then_resumes():
    pauses = []
    script = {"zone": "CLEAR", "dist": None}

    def provider(now_wall):
        return person(script["zone"], script["dist"], stamp=now_wall)

    c, m = _controller(reactive_arm_human=True,
                       reactive_arm_config={"clear_dwell_s": 0.5,
                                            "backstop": {"startup_grace_s": 0.05}})
    c.set_arm_pause_sink(pauses.append)
    c.set_human_detection_provider(provider)

    # clear scene: the leg runs (not blocked), arm not paused
    r = c.step(0.1, _sensors(0.1))
    assert r.status == "running" and not c._reactive_arm.paused

    # a person enters the STOP ring: the seam blocks the leg and pauses the arm
    script.update(zone="STOP", dist=0.25)
    r = c.step(0.1, _sensors(0.2))
    assert r.status == "running"               # held (no fail, no advance)
    assert c._reactive_arm.paused and pauses[-1] is True
    assert m.cancels == 1                       # the running leg was preempted (e-stop-free cancel)
    assert any("reactive_arm" in e and e["reactive_arm"]["action"] == "STOP" for e in c.evidence)

    # the person leaves THROUGH the SLOW ring (tracked departure, not lost-close), then clears; after
    # the clear dwell the arm resumes (pause released).
    script.update(zone="SLOW", dist=0.8)
    c.step(0.1, _sensors(0.25))
    script.update(zone="CLEAR", dist=None)
    for t in (0.3, 0.5, 0.9, 1.1):
        c.step(0.1, _sensors(t))
    assert not c._reactive_arm.paused and pauses[-1] is False


def test_controller_seam_stays_stopped_while_person_present():
    def provider(now_wall):
        return person("STOP", 0.2, stamp=now_wall)

    c, m = _controller(reactive_arm_human=True,
                       reactive_arm_config={"clear_dwell_s": 0.5,
                                            "backstop": {"startup_grace_s": 0.05}})
    c.set_arm_pause_sink(lambda _p: None)
    c.set_human_detection_provider(provider)
    for t in (0.1, 0.2, 0.3, 0.4):
        r = c.step(0.1, _sensors(t))
        assert r.status == "running"           # perpetually held, never advances
    assert c._reactive_arm.paused


# =========================================================== tier 4: ArmCommandGate.set_paused
def _arm_gate():
    pos = {"q": np.zeros(6)}
    robot = NS(get_dof_index=lambda name: 0,
               get_joint_positions=lambda joint_indices=None: pos["q"])
    applied = []
    gate = native_transport.ArmCommandGate(robot, [(-10., 10.)] * 6,
                                           lambda q, v, i: applied.append((np.asarray(q), np.asarray(v))))
    gate.applied = applied
    return gate, pos


def test_set_paused_holds_and_drops_commands_without_latching():
    gate, _ = _arm_gate()
    gate.set_paused(True)
    assert gate._paused and gate.suppress_heartbeat and not gate.faulted
    before = len(gate.applied)
    # a host command while paused is DROPPED (the hold is re-applied, never the command)
    gate.command(names=list(native_transport.ARM_JOINTS),
                 positions=[1.0] * 6, velocities=[0.0] * 6, efforts=[0.0] * 6, stamp=1.0, now=1.0)
    assert len(gate.applied) == before + 1
    assert np.allclose(gate.applied[-1][0], 0.0)        # held at measured, not the commanded 1.0
    assert not gate.faulted                              # never latches


def test_set_paused_watchdog_does_not_fault_while_paused():
    gate, _ = _arm_gate()
    gate.started = True
    gate._host_now = 0.0
    gate.set_paused(True)
    gate.watchdog(10_000.0)                              # a huge sim gap: would fault if not paused
    assert not gate.faulted


def test_set_paused_release_re_seats_a_clean_ramp():
    gate, _ = _arm_gate()
    gate.set_paused(True)
    gate.started = True
    gate.set_paused(False)
    assert not gate._paused and not gate.suppress_heartbeat
    assert gate.last_stamp == -1. and not gate.started   # re-seated for a clean first-command ramp


def test_set_paused_idempotent_and_off_is_noop():
    gate, _ = _arm_gate()
    assert not gate._paused
    gate.set_paused(False)                               # no-op when already unpaused
    assert not gate._paused and not gate.suppress_heartbeat


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
