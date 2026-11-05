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

from dataclasses import dataclass
from typing import Callable, FrozenSet, Optional

from . import arm_human_gate as hg
from .arm_backstop import ArmBackstopParams, ArmBackstopWatcher
from .arm_human_gate import ArmGateConfig, ArmHumanGate

# The free-space re-plannable arm phases the reactive layer may preempt/slow.
# Everything else (contact/grasp/preplace/place/release/observe/...) is frozen.
DEFAULT_REPLANNABLE_PHASES: FrozenSet[str] = frozenset(
    {"approach", "lift", "retreat", "tuck", "pan", "retreat_tuck", "stow"})


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
                 enable_backstop: bool = True) -> None:
        self.gate = ArmHumanGate(gate_config)
        self.backstop = ArmBackstopWatcher(backstop_params) if enable_backstop else None
        self.replannable_phases = (frozenset(replannable_phases)
                                   if replannable_phases is not None
                                   else DEFAULT_REPLANNABLE_PHASES)
        self._pause_sink = pause_sink
        self._detection_provider = detection_provider
        # per-op reactive state
        self._paused = False
        self._cancel_pending = False
        self._seq = 0
        self._last_gate_arrival: Optional[float] = None
        self._t0: Optional[float] = None

    # ---- lifecycle --------------------------------------------------------

    def reset(self, now: float = 0.0) -> None:
        self.gate = ArmHumanGate(self.gate.cfg)
        if self.backstop is not None:
            self.backstop.reset(now)
        self._set_pause(False)
        self._cancel_pending = False
        self._seq = 0
        self._last_gate_arrival = None
        self._t0 = now

    def set_pause_sink(self, sink: Optional[Callable[[bool], None]]) -> None:
        self._pause_sink = sink

    def set_detection_provider(self, provider: Optional[Callable[[float], object]]) -> None:
        self._detection_provider = provider

    # ---- the per-tick decision -------------------------------------------

    def supervise(self, *, now: float, now_wall: float, phase: str, motion,
                  leg_running: bool, driver_stopped: Optional[bool] = None,
                  state: object = None) -> ArmDirective:
        """Decide the reactive directive for this controller tick.

        ``state`` is an optional pre-fetched vendored ``SafetyState`` (used by the
        CPU tests); when None the injected ``detection_provider`` is consulted. The
        reactive layer is inert (``allow``) outside the re-plannable free-space
        phases so delicate/other-lane-owned moves are untouched."""
        if self._t0 is None:
            self._t0 = now
        if driver_stopped is None:
            # The arm is "confirmed stopped" once it is held paused (set last tick):
            # a cancelled + paused leg is at rest, so a sustained STOP need not
            # escalate to BACKSTOP merely because no external feedback is wired.
            driver_stopped = self._paused

        # --- ingest the freshest safety frame + decide (EVERY tick, so the gate's
        #     resume dwell + heartbeat continuity span phase boundaries) ---------
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

        # --- independent backstop watcher ---------------------------------
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

        # Outside the free-space re-plannable phases the human layer does NOT
        # preempt a delicate / other-lane-owned in-progress move; it keeps the gate
        # warm (above) but takes no action and holds no stale pause.
        if phase not in self.replannable_phases:
            self._cancel_pending = False
            self._set_pause(False)
            return ArmDirective.allow(action=action, reason="phase %s not reactive-replannable "
                                      "(gate=%s)" % (phase, action), speed=decision.speed_fraction)

        # --- translate the decision into a controller directive -----------
        if action == hg.RUN:
            # clear any preempt + release the pause; the leg runs / may be issued.
            self._cancel_pending = False
            self._set_pause(False)
            return ArmDirective.allow(action=hg.RUN, reason=reason,
                                      speed=decision.speed_fraction)

        if action == hg.SLOW:
            # Slow the arm: preempt a running leg so it re-plans SLOW; a not-yet-
            # issued leg is simply issued slow. Never pause for SLOW.
            self._set_pause(False)
            if leg_running:
                settled = self._drive_preempt(motion)
                if not settled:
                    return ArmDirective(block=True, reissue=False, slow=True, fail=None,
                                        action=hg.SLOW, reason="SLOW: settling preempt — " + reason,
                                        speed_fraction=decision.speed_fraction)
                return ArmDirective(block=False, reissue=True, slow=True, fail=None,
                                    action=hg.SLOW, reason=reason,
                                    speed_fraction=decision.speed_fraction)
            return ArmDirective.allow(action=hg.SLOW, reason=reason,
                                      speed=decision.speed_fraction, slow=True)

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
                            speed_fraction=decision.speed_fraction)

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
