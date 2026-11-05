"""Independent backstop watcher for the simulated restock arm
(sim-arm-reactive-replan-v1, P1).

A near-verbatim port of the real cell's independent backstop-watcher decision
core (``ur10-setup/humangate_driver/watcher_core.py``). It is a SEPARATE check
with its OWN copy of the safety-frame + gate-decision heartbeats, so a hang in
the primary :class:`~arm_human_gate.ArmHumanGate` path cannot disable it. It asks
for a goal CANCEL (the arm decelerates to a hold) on ANY of:

  * the safety-frame heartbeat is lost (it watches the frame stream itself);
  * the gate's decision heartbeat is lost (the primary path died);
  * the gate explicitly decided BACKSTOP;
  * a STOP is demanded but the observed speed scaling did NOT fall to a near-stop
    within a deadline (the slow/stop lever did not take effect).

Everything fails toward cancel. Once tripped it latches until an operator reset.
PURE: standard library only, all time passed in (sim/monotonic seconds).
Disabled (not constructed when the HOST key is off) => byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

ACTION_BACKSTOP = "BACKSTOP"
ACTION_STOP = "STOP"
ZONE_STOP = "STOP"


@dataclass
class ArmBackstopParams:
    # Its own safety-frame heartbeat timeout. Slightly looser than the gate's own
    # (0.3 s) so, healthy, the gate acts first; the watcher is the independent net.
    state_heartbeat_timeout_s: float = 0.5
    # No gate decision within this -> primary path dead -> cancel.
    gate_heartbeat_timeout_s: float = 0.5
    # After a STOP is demanded, the scaling must fall below stop_scaling_threshold
    # within this deadline, else the lever did not take effect -> cancel.
    stop_effect_deadline_s: float = 0.7
    stop_scaling_threshold: float = 0.05
    # Grace after startup before "never seen the gate/frame" can trip.
    startup_grace_s: float = 3.0

    def validate(self) -> "ArmBackstopParams":
        for name in ("state_heartbeat_timeout_s", "gate_heartbeat_timeout_s",
                     "stop_effect_deadline_s", "startup_grace_s"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if not (0.0 <= self.stop_scaling_threshold < 1.0):
            raise ValueError("stop_scaling_threshold must be in [0, 1)")
        return self


class ArmBackstopWatcher:
    """Stateful watcher core. Call update() each tick; cancel when it returns a
    reason. Latches once tripped."""

    def __init__(self, params: Optional[ArmBackstopParams] = None,
                 start_time: float = 0.0) -> None:
        self.p = (params or ArmBackstopParams()).validate()
        self._t0 = start_time
        self._stop_demand_since: Optional[float] = None
        self.tripped: bool = False
        self.reason: str = ""

    def reset(self, start_time: float = 0.0) -> None:
        self.tripped = False
        self.reason = ""
        self._stop_demand_since = None
        self._t0 = start_time

    def update(
        self,
        now: float,
        *,
        last_state_arrival: Optional[float],
        last_gate_arrival: Optional[float],
        state_zone: Optional[str],
        gate_action: Optional[str],
        observed_scaling: Optional[float],
        gate_demand: Optional[str] = None,
    ) -> Optional[str]:
        """Return a cancel reason if the watcher should cancel this tick (newly or
        still tripped), else None."""
        if self.tripped:
            return self.reason

        past_grace = (now - self._t0) >= self.p.startup_grace_s

        if gate_action == ACTION_BACKSTOP:
            return self._trip("gate decided BACKSTOP")

        # "Never received past grace" cancels (harmless if no goal) but does NOT
        # latch, so it recovers the moment the stream starts.
        if last_state_arrival is None:
            if past_grace:
                return "no human-safety frame received yet (past startup grace)"
        elif (now - last_state_arrival) > self.p.state_heartbeat_timeout_s:
            return self._trip(
                f"safety-frame heartbeat lost ({(now - last_state_arrival) * 1e3:.0f} ms silent)")

        if last_gate_arrival is None:
            if past_grace:
                return "gate has not decided yet (past startup grace)"
        elif (now - last_gate_arrival) > self.p.gate_heartbeat_timeout_s:
            return self._trip(
                f"gate decision heartbeat lost "
                f"({(now - last_gate_arrival) * 1e3:.0f} ms silent) — primary path dead")

        # STOP demanded but the lever did not take effect. Deliberately excludes a
        # STOP-while-dwelling (action STOP but demand CLEAR): cancelling there would
        # discard the very hold the resume dwell is counting against.
        stop_demanded = (state_zone == ZONE_STOP) or (gate_demand == ACTION_STOP)
        if stop_demanded:
            if self._stop_demand_since is None:
                self._stop_demand_since = now
            effected = (observed_scaling is not None
                        and observed_scaling <= self.p.stop_scaling_threshold)
            if (not effected
                    and (now - self._stop_demand_since) > self.p.stop_effect_deadline_s):
                sc = "unknown" if observed_scaling is None else f"{observed_scaling:.3f}"
                return self._trip(
                    f"STOP demanded but scaling still {sc} > "
                    f"{self.p.stop_scaling_threshold:.2f} after "
                    f"{self.p.stop_effect_deadline_s:.2f}s — slow/stop lever ineffective")
        else:
            self._stop_demand_since = None

        return None

    def _trip(self, reason: str) -> str:
        self.tripped = True
        self.reason = reason
        return reason
