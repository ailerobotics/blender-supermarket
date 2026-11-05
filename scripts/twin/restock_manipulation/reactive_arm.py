"""Reactive arm supervisor — human-aware slow/stop authority for the simulated
restock arm (sim-arm-reactive-replan-v1, P1).

This is the single controller seam for the reactive arm layer. It owns the ported
human-safety authority (:class:`~arm_human_gate.ArmHumanGate` +
:class:`~arm_backstop.ArmBackstopWatcher`), fed by the vendored
``scripts/twin/humans`` evaluator, and translates its RUN/SLOW/STOP/BACKSTOP
decision into a per-tick directive the controller executes with the EXISTING,
e-stop-free preempt path (``motion.cancel()`` / ``motion.cancel_outcome()``, the
PR-334 / sim-place-release-serialization primitive) — never the global emergency
stop. It also drives the reversible arm-command-gate pause
(``ArmCommandGate.set_paused``) through an injected sink, edge-triggered.

Mapping (the honest C1 segmented-execution mapping, human-dynamic-plan-s1 §2):
  * RUN      -> allow the arm leg to run / be issued at full speed.
  * SLOW     -> preempt a running re-plannable leg and re-issue it SLOW (reduced
                velocity/accel); a not-yet-issued leg is issued slow.
  * STOP     -> preempt (cancel) a running re-plannable leg, PAUSE the arm command
                gate, and BLOCK new legs; resume (re-plan from the live pose) only
                after the gate's own CLEAR dwell restores RUN.
  * BACKSTOP -> same terminal hold as STOP, additionally latched by the independent
                watcher until RUN is re-earned.

Scope: the human layer acts only on the free-space re-plannable arm phases
(approach/lift/retreat and the staged tuck/pan/stow). The delicate, frozen phases
(contact/grasp/preplace/place/release) are left untouched — the pre-place planner
is owned by another lane and must not be edited. Per the perception-architecture
scout (sim-perception-architecture-s1 §4) the gate is the SINGLE motion authority
for people; the nvblox map is never the stop.

Disabled (the controller does not construct a supervisor when the HOST key is off)
=> byte-identical. PURE control logic: no ROS/Isaac/GPU; the motion port, pause
sink and detection provider are injected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, FrozenSet, Optional

from . import arm_human_gate as hg
from . import reactive_replan as rr
from .arm_backstop import ArmBackstopParams, ArmBackstopWatcher
from .arm_human_gate import ArmGateConfig, ArmHumanGate

# The free-space re-plannable arm phases the reactive layer may preempt/slow.
# Everything else (contact/grasp/preplace/place/release/observe/...) is frozen.
DEFAULT_REPLANNABLE_PHASES: FrozenSet[str] = frozenset(
    {"approach", "lift", "retreat", "tuck", "pan", "retreat_tuck", "stow"})


def build_corridor_fn(motion_port):
    """PR-B mission wiring: build ``(esdf_source, corridor_fn)`` from the sibling
    ``gated_motion.GatedMotionPort`` for the execution-time corridor watch, or ``(None, None)`` when
    the motion port does not expose the perceived-clearance seam (``perception_collision_gate`` off).

    ``corridor_fn(measured_joints, goal_pose, held) -> [AABB]`` is the REMAINING swept corridor from
    the current-TCP FK to the goal, reusing the port's own ``_tcp`` FK + ``leg_corridor`` (same
    envelope + sampling as the dispatch gate), and ``esdf_source`` is the SAME live nvblox source
    cuMotion plans against."""
    source = getattr(motion_port, "esdf_source", None)
    leg_corridor = getattr(motion_port, "leg_corridor", None)
    tcp = getattr(motion_port, "_tcp", None)
    if source is None or not callable(leg_corridor) or not callable(tcp):
        return None, None

    def corridor_fn(measured, goal_pose, held):
        return leg_corridor(tcp(measured), goal_pose, held=held)

    return source, corridor_fn


@dataclass(frozen=True)
class ArmDirective:
    """What the controller must do this tick for the reactive arm layer."""

    block: bool          # do NOT advance/issue an arm leg this tick (hold)
    reissue: bool        # clear ``issued`` so the leg re-plans from the live pose
    slow: bool           # issue/re-issue the leg SLOW (reduced velocity/accel)
    fail: Optional[str]  # non-None => fail the op by this named reason
    action: str          # RUN/SLOW/STOP/BACKSTOP (for logging/evidence)
    reason: str
    speed_fraction: Optional[float]

    @staticmethod
    def allow(action: str = hg.RUN, reason: str = "", speed: Optional[float] = 1.0,
              slow: bool = False) -> "ArmDirective":
        return ArmDirective(block=False, reissue=False, slow=slow, fail=None,
                            action=action, reason=reason, speed_fraction=speed)


class ReactiveArmSupervisor:
    """Human-aware slow/stop authority. Constructed only when the HOST key enables
    it; one per controller op, ``reset()`` at ``start``."""

    def __init__(self, *,
                 gate_config: Optional[ArmGateConfig] = None,
                 backstop_params: Optional[ArmBackstopParams] = None,
                 replannable_phases: Optional[FrozenSet[str]] = None,
                 pause_sink: Optional[Callable[[bool], None]] = None,
                 detection_provider: Optional[Callable[[float], object]] = None,
                 enable_backstop: bool = True,
                 enable_human: bool = True,
                 esdf_source: object = None,
                 corridor_fn: Optional[Callable] = None,
                 collision_gate: object = None,
                 replan_controller: object = None) -> None:
        self.enable_human = bool(enable_human)
        self.gate = ArmHumanGate(gate_config)
        self.backstop = ArmBackstopWatcher(backstop_params) if (enable_human and enable_backstop) else None
        self.replannable_phases = (frozenset(replannable_phases)
                                   if replannable_phases is not None
                                   else DEFAULT_REPLANNABLE_PHASES)
        self._pause_sink = pause_sink
        self._detection_provider = detection_provider
        # PR-B: execution-time corridor-watch re-plan (sibling perceived-clearance seams).
        self._esdf_source = esdf_source
        self._corridor_fn = corridor_fn
        self._collision_gate = collision_gate
        self._replan = replan_controller
        # per-op reactive state
        self._paused = False
        self._cancel_pending = False
        self._seq = 0
        self._last_gate_arrival: Optional[float] = None
        self._t0: Optional[float] = None

    @property
    def replan_enabled(self) -> bool:
        return (self._esdf_source is not None and self._corridor_fn is not None
                and self._collision_gate is not None and self._replan is not None)

    # ---- lifecycle --------------------------------------------------------

    def reset(self, now: float = 0.0) -> None:
        self.gate = ArmHumanGate(self.gate.cfg)
        if self.backstop is not None:
            self.backstop.reset(now)
        if self._replan is not None:
            self._replan.reset()
        self._set_pause(False)
        self._cancel_pending = False
        self._seq = 0
        self._last_gate_arrival = None
        self._t0 = now

    def set_pause_sink(self, sink: Optional[Callable[[bool], None]]) -> None:
        self._pause_sink = sink

    def set_detection_provider(self, provider: Optional[Callable[[float], object]]) -> None:
        self._detection_provider = provider

    def set_clearance_source(self, *, esdf_source=None, corridor_fn=None,
                             collision_gate=None, replan_controller=None) -> None:
        """PR-B: inject the perceived-clearance seams (sibling sim-rgbd-arm-collision-aware-v1).
        ``esdf_source`` is an ``esdf_source.EsdfSource`` (``snapshot(now)``); ``corridor_fn`` builds
        the REMAINING swept corridor ``(measured_joints, goal_pose, held) -> [AABB]`` (current TCP
        FK -> goal); ``collision_gate`` is a ``PerceptionCollisionGate`` (its ``evaluate_corridor``
        is reused verbatim); ``replan_controller`` is a :class:`reactive_replan.ReactiveReplanController`.
        Absent => the corridor watch is inert (byte-identical)."""
        if esdf_source is not None:
            self._esdf_source = esdf_source
        if corridor_fn is not None:
            self._corridor_fn = corridor_fn
        if collision_gate is not None:
            self._collision_gate = collision_gate
        if replan_controller is not None:
            self._replan = replan_controller

    # ---- the per-tick decision -------------------------------------------

    def supervise(self, *, now: float, now_wall: float, phase: str, motion,
                  leg_running: bool, driver_stopped: Optional[bool] = None,
                  state: object = None, measured=None, goal_pose=None,
                  held: bool = False) -> ArmDirective:
        """Decide the reactive directive for this controller tick.

        ``state`` is an optional pre-fetched vendored ``SafetyState`` (used by the CPU tests); when
        None the injected ``detection_provider`` is consulted. ``measured``/``goal_pose``/``held``
        drive the PR-B execution-time corridor watch (the remaining-leg swept corridor is FK'd from
        ``measured`` to ``goal_pose``). The reactive layer is inert (``allow``) outside the
        re-plannable free-space phases so delicate/other-lane-owned moves are untouched."""
        if self._t0 is None:
            self._t0 = now
        if driver_stopped is None:
            # The arm is "confirmed stopped" once it is held paused (set last tick):
            # a cancelled + paused leg is at rest, so a sustained STOP need not
            # escalate to BACKSTOP merely because no external feedback is wired.
            driver_stopped = self._paused

        # --- human gate: ingest the freshest safety frame + decide (EVERY tick, so the gate's
        #     resume dwell + heartbeat continuity span phase boundaries). Optional (enable_human). ---
        if self.enable_human:
            if state is None and self._detection_provider is not None:
                try:
                    state = self._detection_provider(now_wall)
                except Exception as exc:   # noqa: BLE001 — provider failure fails toward STOP
                    state = None
                    self.gate.inject_self_fault("detection provider error: %s" % exc)
            self._seq += 1
            frame = hg.frame_from_safety_state(state, now_wall, heartbeat_seq=self._seq)
            self.gate.on_state(frame, now, now_wall)
            decision = self.gate.decide(now, now_wall, driver_stopped=driver_stopped)
            self._last_gate_arrival = now
            backstop_reason = None
            if self.backstop is not None:
                observed_scaling = 0.0 if (self._paused or self._cancel_pending) \
                    else (decision.speed_fraction if decision.speed_fraction is not None else 0.0)
                backstop_reason = self.backstop.update(
                    now,
                    last_state_arrival=now,              # we just ingested a frame this tick
                    last_gate_arrival=self._last_gate_arrival,
                    state_zone=frame.zone,
                    gate_action=decision.action,
                    observed_scaling=observed_scaling,
                    gate_demand=decision.demand)
            action = hg.BACKSTOP if backstop_reason else decision.action
            reason = backstop_reason or decision.reason
            speed = decision.speed_fraction
        else:
            action, reason, speed = hg.RUN, "human gate disabled", 1.0

        # Outside the free-space re-plannable phases the layer does NOT preempt a delicate /
        # other-lane-owned in-progress move; it keeps the gate warm (above) but takes no action.
        if phase not in self.replannable_phases:
            self._cancel_pending = False
            self._set_pause(False)
            return ArmDirective.allow(action=action, reason="phase %s not reactive-replannable "
                                      "(gate=%s)" % (phase, action), speed=speed)

        # --- translate the decision into a controller directive -----------
        if action == hg.RUN:
            # clear any human preempt + release the pause; the leg runs / may be issued.
            self._cancel_pending = False
            self._set_pause(False)
            # PR-B: execution-time corridor watch — a NEW obstacle in the remaining swept corridor
            # of a RUNNING leg preempts + re-plans (bounded; HOLD when it cannot route clear).
            rd = self._corridor_watch(now=now, phase=phase, motion=motion, leg_running=leg_running,
                                      measured=measured, goal_pose=goal_pose, held=held)
            if rd is not None:
                return rd
            return ArmDirective.allow(action=hg.RUN, reason=reason, speed=speed)

        if action == hg.SLOW:
            # Slow the arm: preempt a running leg so it re-plans SLOW; a not-yet-
            # issued leg is simply issued slow. Never pause for SLOW.
            self._set_pause(False)
            if leg_running:
                settled = self._drive_preempt(motion)
                if not settled:
                    return ArmDirective(block=True, reissue=False, slow=True, fail=None,
                                        action=hg.SLOW, reason="SLOW: settling preempt — " + reason,
                                        speed_fraction=speed)
                return ArmDirective(block=False, reissue=True, slow=True, fail=None,
                                    action=hg.SLOW, reason=reason,
                                    speed_fraction=speed)
            return ArmDirective.allow(action=hg.SLOW, reason=reason,
                                      speed=speed, slow=True)

        # STOP or BACKSTOP: preempt a running leg (cancel -> settle), pause the arm
        # command gate, and BLOCK any new leg until RUN is re-earned.
        if leg_running:
            self._drive_preempt(motion)   # request cancel; settle polled across ticks
        else:
            # no active trajectory to cancel: ensure no stale preempt lingers
            self._cancel_pending = False
        self._set_pause(True)
        return ArmDirective(block=True, reissue=False, slow=False, fail=None,
                            action=action, reason=reason,
                            speed_fraction=speed)

    # ---- PR-B: execution-time corridor watch -----------------------------

    def _corridor_watch(self, *, now, phase, motion, leg_running, measured, goal_pose, held):
        """Watch the REMAINING swept corridor of a running free-space leg against the live ESDF.
        Returns a preempt/HOLD :class:`ArmDirective`, or ``None`` to continue (clear / inert).

        Reuses the merged ``PerceptionCollisionGate.evaluate_corridor`` so an execution intrusion
        uses the SAME margin logic as the dispatch gate. Only a FINITE observed clearance below the
        margin counts as an intrusion -- an unobserved/stale corridor is the dispatch gate's concern
        (the sibling's unknown_action), not an execution preempt (which must react to a NEW mapped
        obstacle, not to the ESDF momentarily lagging)."""
        if not self.replan_enabled or phase not in self.replannable_phases:
            return None
        # a phase IS a leg: reset the per-leg re-plan budget on a phase change (our own preempt
        # re-issues keep the phase, so a storming obstacle still terminates at max_replans).
        self._replan.new_leg(phase)
        if not leg_running or measured is None or goal_pose is None:
            return None
        try:
            snapshot = self._esdf_source.snapshot(now)
            boxes = self._corridor_fn(measured, goal_pose, held)
            verdict = self._collision_gate.evaluate_corridor(
                phase=phase, boxes=boxes, snapshot=snapshot, now=now, held=held)
        except Exception as exc:   # noqa: BLE001 — a watch failure must never crash the move
            print('REACTIVE_REPLAN_WATCH skipped (%s: %s)' % (type(exc).__name__, exc), flush=True)
            return None
        intruded = (math.isfinite(verdict.min_clearance)
                    and verdict.min_clearance < verdict.margin_m)
        rd = self._replan.observe(now, leg_running=True, intruded=intruded, watched=True)
        if rd.outcome == rr.CONTINUE:
            return None
        if rd.outcome == rr.HOLD:
            # bounded re-plans exhausted: HOLD in place (pause) and keep watching -- self-recovers
            # if the obstacle leaves the corridor (then observe() returns CONTINUE). Never e-stops.
            self._set_pause(True)
            return ArmDirective(block=True, reissue=False, slow=False, fail=None,
                                action="REPLAN_HOLD", reason=rd.reason,
                                speed_fraction=0.0)
        # PREEMPT: cancel the running leg -> settle -> re-issue (it re-plans against the current
        # ESDF, now carrying the obstacle, and routes the remaining leg clear).
        settled = self._drive_preempt(motion)
        if not settled:
            return ArmDirective(block=True, reissue=False, slow=False, fail=None,
                                action="REPLAN", reason="settling preempt — " + rd.reason,
                                speed_fraction=0.0)
        return ArmDirective(block=False, reissue=True, slow=False, fail=None,
                            action="REPLAN", reason=rd.reason, speed_fraction=None)

    # ---- preempt-settle (shared e-stop-free primitive) -------------------

    def _drive_preempt(self, motion) -> bool:
        """Request / advance an e-stop-free cancel of the in-flight leg. Returns
        True once the cancel has reached a terminal (settled, not-in-flight)
        outcome so the controller may re-issue; False while still settling."""
        if motion is None:
            return True
        cancel = getattr(motion, "cancel", None)
        outcome = getattr(motion, "cancel_outcome", None)
        if not callable(cancel) or not callable(outcome):
            # a thin stub without the cancel path: nothing to settle
            return True
        if not self._cancel_pending:
            cancel()
            self._cancel_pending = True
        result = outcome()
        if result is None:
            return False           # still settling
        self._cancel_pending = False
        return True

    # ---- pause sink edge-trigger -----------------------------------------

    def _set_pause(self, paused: bool) -> None:
        if paused == self._paused:
            return
        self._paused = paused
        if self._pause_sink is not None:
            self._pause_sink(paused)

    @property
    def paused(self) -> bool:
        return self._paused
