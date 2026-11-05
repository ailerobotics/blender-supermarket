"""Execution-time reactive re-plan decide core for the simulated restock arm
(sim-arm-reactive-replan-v1, PR-B).

The perceived-clearance gate from ``sim-rgbd-arm-collision-aware-v1`` gates a leg at
DISPATCH (before ``begin``). THIS is the execution-time complement its
``README_perception.md`` names: while a free-space re-plannable leg EXECUTES, re-query
the live nvblox ESDF along the gripper(+carton) swept corridor of the REMAINING leg;
when a NEW obstacle enters it below a margin, PREEMPT the running goal through the
e-stop-free ``motion.cancel()`` path (never the global emergency stop), RE-PLAN the
remaining leg with cuMotion against the current ESDF, and resume. Bounded re-plan count
with a HOLD when it cannot route clear; rate-limited so a transient reading does not
churn. People are generic dynamic volumes for routing (they re-route the plan); the
human gate owns the STOP (scout sim-perception-architecture-s1 §4).

This module is the PURE decide core (ported from ``~/ur10/dynamic_pick``
``replan_controller_core.ReplanController`` — the segmented re-plan/preempt shape, here
triggered by corridor intrusion rather than a moving pick target). All time is passed
in; the ESDF query / corridor build / cancel live in the supervisor. The clearance
verdict comes from the merged ``collision_gate.PerceptionCollisionGate.evaluate_corridor``
(reused verbatim), so a reactive intrusion uses the SAME margin/unobserved logic as the
dispatch gate. Disabled (not constructed) => byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# decide outcomes
CONTINUE = "continue"   # leg proceeds unchanged (clear, or not a watched leg)
PREEMPT = "preempt"     # cancel + re-plan the remaining leg against the current ESDF
HOLD = "hold"           # bounded re-plans exhausted / cannot route clear -> hold, named fail
OUTCOMES = (CONTINUE, PREEMPT, HOLD)


@dataclass(frozen=True)
class ReplanDecision:
    outcome: str
    reason: str
    replans_used: int


class ReactiveReplanController:
    """Bounded, rate-limited execution-time re-plan trigger.

    :param min_replan_interval_s: minimum SIM seconds between successive preempts (a
        transient intrusion reading does not churn the planner).
    :param max_replans: total preempt+re-plans permitted per leg before the controller
        HOLDs and fails the op by name (the "cannot find a path" bound the brief requires).
    """

    def __init__(self, *, min_replan_interval_s: float = 0.5, max_replans: int = 3):
        if not (min_replan_interval_s >= 0):
            raise ValueError("min_replan_interval_s must be >= 0")
        if not (isinstance(max_replans, int) and max_replans >= 0):
            raise ValueError("max_replans must be a non-negative int")
        self.min_replan_interval_s = float(min_replan_interval_s)
        self.max_replans = int(max_replans)
        self._replans = 0
        self._last_replan_t: Optional[float] = None
        self._active_leg = None   # identity of the leg currently being watched (phase, issue id)

    def reset(self) -> None:
        self._replans = 0
        self._last_replan_t = None
        self._active_leg = None

    def new_leg(self, leg_id) -> None:
        """A fresh leg began (new phase, or a re-issue that is NOT one of our preempts): reset
        the per-leg re-plan budget. Our own preempt re-issues keep the same ``leg_id`` so the
        budget persists across them (a storming obstacle still terminates at max_replans)."""
        if leg_id != self._active_leg:
            self._active_leg = leg_id
            self._replans = 0
            self._last_replan_t = None

    def observe(self, now: float, *, leg_running: bool, intruded: bool,
                watched: bool = True) -> ReplanDecision:
        """Decide for this tick.

        :param leg_running: a leg is in flight (``issued``).
        :param intruded: the REMAINING swept corridor is blocked by a (new) obstacle below the
            margin — i.e. the reused dispatch-gate verdict is ``blocked`` for the remaining leg.
        :param watched: this phase is a watched free-space re-plannable leg.
        """
        if not watched or not leg_running or not intruded:
            return ReplanDecision(CONTINUE, "clear" if watched else "phase not watched",
                                  self._replans)
        # a new obstacle is in the remaining corridor.
        if self._replans >= self.max_replans:
            return ReplanDecision(
                HOLD, "execution-time re-plan budget exhausted (%d/%d): the arm could not route "
                      "the remaining leg clear of the obstacle in the swept corridor — holding"
                      % (self._replans, self.max_replans), self._replans)
        if (self._last_replan_t is not None
                and (now - self._last_replan_t) < self.min_replan_interval_s):
            # rate-limited: do not re-cancel yet (the just-issued re-plan may already route clear
            # against the fresher map). Hold position for this tick by preempting-in-progress is
            # the supervisor's settle concern; here we simply do not trigger a NEW preempt.
            return ReplanDecision(CONTINUE, "rate-limited: awaiting the prior re-plan", self._replans)
        self._replans += 1
        self._last_replan_t = now
        return ReplanDecision(
            PREEMPT, "obstacle entered the remaining swept corridor below margin — preempting and "
                     "re-planning the remaining leg against the current ESDF (re-plan %d/%d)"
                     % (self._replans, self.max_replans), self._replans)

    @property
    def replans_used(self) -> int:
        return self._replans
