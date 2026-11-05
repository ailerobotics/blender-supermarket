"""CPU proof for the execution-time corridor-watch re-plan (sim-arm-reactive-replan-v1,
PR-B). No ROS / Isaac / GPU.

Proves, against a MOCK nvblox ESDF (the merged ``esdf_source.AnalyticBoxEsdfSource``
with a moving obstacle) and the merged ``collision_gate.PerceptionCollisionGate``
(reused verbatim):
  1. the pure ``ReactiveReplanController`` — continue / preempt / bounded / HOLD / reset;
  2. the ``ReactiveArmSupervisor`` corridor watch — a planted obstacle ENTERING the
     remaining swept corridor mid-move causes a preempt (e-stop-free cancel) + re-plan
     (re-issue), the arm CONTINUES once it clears, and a persistent obstacle exhausts the
     bounded budget into a self-recovering HOLD;
  3. the REAL ``TransferController.step`` seam with ``reactive_arm_replan`` on;
  4. a moving-obstacle timeline replay showing the preempt/re-plan/continue timeline.
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation import reactive_replan as rr  # noqa: E402
from twin.restock_manipulation.reactive_arm import ReactiveArmSupervisor  # noqa: E402
from twin.restock_manipulation.collision_gate import PerceptionCollisionGate  # noqa: E402
from twin.restock_manipulation.esdf_source import AnalyticBoxEsdfSource  # noqa: E402
from twin.restock_manipulation import Sensors, ProductProfile, TransferController  # noqa: E402
from twin.restock_manipulation.reactive_arm import ArmDirective  # noqa: E402
from twin.restock_manipulation import arm_human_gate as hg  # noqa: E402


# A fixed corridor slab in the forward arm workspace (ur_base_link metres), and obstacle boxes
# that are FAR from it or INTRUDE within the margin.
CORRIDOR = [((0.30, 0.40, 0.30), (0.40, 0.80, 0.50))]
FAR_BOX = ((0.30, 1.20, 0.30), (0.40, 1.30, 0.50))     # ~0.40 m clear of the corridor
NEAR_BOX = ((0.30, 0.82, 0.30), (0.40, 0.92, 0.50))    # ~0.02 m from the corridor -> intrusion
GOAL = (0.35, 0.80, 0.40, 0.0, 0.0, 0.0, 1.0)


def corridor_fn(measured, goal_pose, held):
    return list(CORRIDOR)


class FakeMotion:
    def __init__(self, settle_ticks=0):
        self.cancels = 0
        self._settle_ticks = settle_ticks
        self._since = None

    def cancel(self):
        self.cancels += 1
        self._since = 0

    def cancel_outcome(self):
        if self._since is None:
            return None
        if self._since < self._settle_ticks:
            self._since += 1
            return None
        self._since = None
        return "cancelled"


# =========================================================== tier 1: the decide core
def test_replan_continue_when_clear():
    c = rr.ReactiveReplanController()
    c.new_leg("lift")
    d = c.observe(0.0, leg_running=True, intruded=False)
    assert d.outcome == rr.CONTINUE and d.replans_used == 0


def test_replan_preempts_on_intrusion():
    c = rr.ReactiveReplanController(min_replan_interval_s=0.3, max_replans=3)
    c.new_leg("lift")
    d = c.observe(1.0, leg_running=True, intruded=True)
    assert d.outcome == rr.PREEMPT and d.replans_used == 1


def test_replan_rate_limited_between_preempts():
    c = rr.ReactiveReplanController(min_replan_interval_s=0.5, max_replans=3)
    c.new_leg("lift")
    assert c.observe(1.0, leg_running=True, intruded=True).outcome == rr.PREEMPT
    # within the interval: do not re-cancel yet (the just-issued re-plan may route clear)
    assert c.observe(1.2, leg_running=True, intruded=True).outcome == rr.CONTINUE
    # past the interval, still intruded -> preempt again
    assert c.observe(1.6, leg_running=True, intruded=True).outcome == rr.PREEMPT


def test_replan_bounded_then_holds():
    c = rr.ReactiveReplanController(min_replan_interval_s=0.0, max_replans=2)
    c.new_leg("lift")
    assert c.observe(1.0, leg_running=True, intruded=True).outcome == rr.PREEMPT
    assert c.observe(1.1, leg_running=True, intruded=True).outcome == rr.PREEMPT
    d = c.observe(1.2, leg_running=True, intruded=True)
    assert d.outcome == rr.HOLD and d.replans_used == 2


def test_replan_new_leg_resets_budget():
    c = rr.ReactiveReplanController(min_replan_interval_s=0.0, max_replans=1)
    c.new_leg("lift")
    assert c.observe(1.0, leg_running=True, intruded=True).outcome == rr.PREEMPT
    assert c.observe(1.1, leg_running=True, intruded=True).outcome == rr.HOLD
    c.new_leg("retreat")   # a fresh leg -> budget resets
    assert c.observe(2.0, leg_running=True, intruded=True).outcome == rr.PREEMPT


def test_replan_inert_when_not_running_or_not_watched():
    c = rr.ReactiveReplanController()
    c.new_leg("lift")
    assert c.observe(1.0, leg_running=False, intruded=True).outcome == rr.CONTINUE
    assert c.observe(1.0, leg_running=True, intruded=True, watched=False).outcome == rr.CONTINUE


# =========================================================== tier 2: supervisor corridor watch
def make_supervisor(boxes_or_callable, *, margin_m=0.05, max_replans=3, min_interval=0.3):
    source = AnalyticBoxEsdfSource(boxes_or_callable, voxel_size=0.02)
    gate = PerceptionCollisionGate(margin_m=margin_m, phases=frozenset({"lift"}),
                                   max_snapshot_age_s=1e9, logger=lambda *a, **k: None)
    replan = rr.ReactiveReplanController(min_replan_interval_s=min_interval, max_replans=max_replans)
    sup = ReactiveArmSupervisor(enable_human=False, replannable_phases=frozenset({"lift"}),
                                esdf_source=source, corridor_fn=corridor_fn,
                                collision_gate=gate, replan_controller=replan)
    sup.reset(0.0)
    return sup


def _sv(sup, now, motion):
    return sup.supervise(now=now, now_wall=now, phase="lift", motion=motion, leg_running=True,
                         measured=(0.0,) * 6, goal_pose=GOAL)


def test_supervisor_preempts_and_reissues_when_obstacle_enters_corridor():
    def boxes(now):
        return [NEAR_BOX if now >= 1.0 else FAR_BOX]
    sup = make_supervisor(boxes)
    m = FakeMotion()
    # clear: the leg runs
    d = _sv(sup, 0.5, m)
    assert not d.block and not d.reissue and d.action == hg.RUN and m.cancels == 0
    # obstacle enters the corridor mid-move: preempt (e-stop-free cancel) + re-plan (re-issue)
    d = _sv(sup, 1.0, m)
    assert d.reissue and d.action == "REPLAN" and m.cancels == 1


def test_supervisor_continues_after_obstacle_leaves():
    def boxes(now):
        # intrudes only for a window, then moves out
        return [NEAR_BOX if 1.0 <= now < 1.5 else FAR_BOX]
    sup = make_supervisor(boxes, min_interval=0.3, max_replans=3)
    m = FakeMotion()
    assert _sv(sup, 0.5, m).action == hg.RUN                      # clear
    assert _sv(sup, 1.0, m).reissue                               # intrusion -> preempt+replan
    # obstacle gone -> the arm CONTINUES
    d = _sv(sup, 1.6, m)
    assert not d.block and not d.reissue and d.action == hg.RUN


def test_supervisor_bounded_replans_then_self_recovering_hold():
    def boxes(now):
        return [NEAR_BOX if now < 3.0 else FAR_BOX]      # persistent, then clears at t=3
    sup = make_supervisor(boxes, min_interval=0.3, max_replans=2)
    m = FakeMotion()
    outcomes = [_sv(sup, t, m).action for t in (1.0, 1.4, 1.8, 2.2)]
    assert outcomes[0] == "REPLAN" and outcomes[1] == "REPLAN"    # two bounded re-plans
    assert "REPLAN_HOLD" in outcomes[2:]                          # budget exhausted -> HOLD
    assert m.cancels == 2                                         # exactly max_replans preempts
    # obstacle clears -> the HOLD self-recovers (CONTINUE)
    d = _sv(sup, 3.2, m)
    assert not d.block and d.action == hg.RUN


def test_supervisor_replan_inert_without_source():
    # no esdf_source/corridor_fn wired -> replan_enabled False -> continue (byte-identical)
    sup = ReactiveArmSupervisor(enable_human=False, replannable_phases=frozenset({"lift"}),
                                replan_controller=rr.ReactiveReplanController())
    assert not sup.replan_enabled
    d = _sv(sup, 1.0, FakeMotion())
    assert not d.block and not d.reissue


# =========================================================== tier 3: the controller seam
PROFILE = ProductProfile("sku-box", (.10, .10, .10), (1., 0., 0., 0.))
JOB = dict(job_id="job-1", sku="sku-box", quantity=1, source_bin="backroom", shelf_id="shelf-a")
_POSE = GOAL


class StepMotion:
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


def _sensors(now):
    return Sensors(simulation_time=now, observations=(), target=None, parked=True, contact=True,
                   feedback_time=now, collision_time=now, joint_positions=(0.,) * 6)


def test_controller_seam_preempts_on_execution_time_intrusion():
    def boxes(now):
        return [NEAR_BOX if now >= 0.2 else FAR_BOX]
    m = StepMotion()
    c = TransferController(m, {"sku-box": PROFILE}, fixture_mode=True,
                           reactive_arm_replan=True,
                           reactive_arm_replan_config={"margin_m": 0.05, "max_replans": 3,
                                                       "min_replan_interval_s": 0.3,
                                                       "max_snapshot_age_s": 1e9})
    c.start("transfer", JOB)
    c.phase = "lift"
    c.held = True
    c.issued = True
    c.last_time = None
    c.elapsed = 0.0
    c.poses = {"lift": _POSE}
    # inject the mock ESDF seam (what ros_host.build_corridor_fn wires at boot)
    c.set_arm_clearance_source(esdf_source=AnalyticBoxEsdfSource(boxes, voxel_size=0.02),
                               corridor_fn=corridor_fn)
    # replan scope defaults to the human replannable set, which includes 'lift'
    c._reactive_arm.replannable_phases = frozenset({"lift"})
    c._reactive_arm._collision_gate.phases = frozenset({"lift"})
    # clear tick: no preempt
    c.step(0.1, _sensors(0.1))
    assert m.cancels == 0 and c.issued
    # obstacle enters the corridor: the seam preempts (e-stop-free cancel) and re-plans the
    # remaining leg (issued is cleared then _advance re-issues it against the fresh ESDF -> the
    # move continues, never e-stops). The observable preempt evidence is the cancel.
    r = c.step(0.1, _sensors(0.25))
    assert m.cancels == 1 and r.status == "running" and c.issued
    assert any(e.get("reactive_arm", {}).get("action") == "REPLAN" for e in c.evidence)


def test_controller_replan_off_is_byte_identical():
    c = TransferController(StepMotion(), {"sku-box": PROFILE}, fixture_mode=True)
    assert c._reactive_arm is None     # neither human nor replan enabled -> no supervisor


# =========================================================== tier 4: moving-obstacle timeline replay
def test_moving_obstacle_timeline_shows_preempt_replan_continue():
    """A planted obstacle slides INTO the swept corridor at t=1.0 and OUT at t=2.0; record the
    per-tick reactive action to show the preempt -> re-plan -> continue timeline."""
    def boxes(now):
        # obstacle y-centre marches from far (1.25) toward the corridor and back out
        if now < 1.0 or now >= 2.0:
            return [FAR_BOX]
        return [NEAR_BOX]
    sup = make_supervisor(boxes, min_interval=0.4, max_replans=5)
    m = FakeMotion()
    timeline = []
    t = 0.0
    while t <= 2.6 + 1e-9:
        d = _sv(sup, round(t, 2), m)
        timeline.append((round(t, 2), d.action, d.reissue, d.block))
        t += 0.2
    actions = [a for (_, a, _, _) in timeline]
    # before 1.0: running clear
    assert all(a == hg.RUN for (tt, a, _, _) in timeline if tt < 1.0)
    # during [1.0, 2.0): at least one preempt/re-plan fired
    assert any(a == "REPLAN" and reiss for (tt, a, reiss, _) in timeline if 1.0 <= tt < 2.0)
    # after the obstacle leaves: back to running (continuation)
    assert timeline[-1][1] == hg.RUN and not timeline[-1][3]
    assert m.cancels >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
