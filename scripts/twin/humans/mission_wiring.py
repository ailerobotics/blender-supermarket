"""Opt-in mission wiring for the human-safety REACTION (base slow/stop + arm pause).

The detection feed into navigation already exists (``sim-restock-dynobs-v1``, PR #130:
``twin.navigation.mission_source.build_dynamic_obstacle_source`` authors the scripted
people into the mission stage and feeds ``Detection`` into the costmap, so the base
already SLOWS/avoids via inflation). What is still missing is the rest of the reaction
contract: scaling the base velocity by the reaction's ``base_scale`` and PAUSING the arm
(reversibly) when a person is in the arm envelope. This module is that mechanism, add-only
and default-off, so the mission is byte-identical when ``humans.enabled`` is absent.

It deliberately does NOT edit the lane-owned mission files; it provides the mechanism and
NAMES the one-line touch points the owning lanes wire at integration-boot time:

* **scene** (``twin.restock.isaac_bundle.create``, lane sim-restock-downstream-v1): the
  dynobs source already authors the people; when humans is enabled WITHOUT the nav
  dynamic-obstacle source, call ``twin.humans.integration.author_people(assembly.stage,
  config, plan.poses)`` once so the actors are still present.
* **per-step hook** (``twin.restock.isaac_port.IsaacPort.after_step``, currently unwired,
  lane sim-restock-downstream-v1): build :class:`MissionReaction` once and call
  ``reaction.step(now)`` each tick; multiply the nav velocity command by
  ``reaction.base_scale`` in ``twin.restock.runtime.RuntimeAdapter._step`` (base path,
  lane sim-restock-dynobs-v1).
* **arm gate** (``twin.restock.native_transport.ArmCommandGate``, lane
  sim-restock-planner-obstacles-v1): add a REVERSIBLE ``set_paused(bool)`` checked in
  ``command()`` (the existing ``hold()`` latches ``faulted`` permanently, so it cannot
  resume); pass that callable in as :class:`ReversibleArmPause`'s ``pause`` sink.

Pure Python; no USD/ROS/Isaac import, so the whole mechanism is CPU-tested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import HumansConfig
from .integration import MissionSafety, build_safety
from .reaction import ReactionOutput


class ReversibleArmPause:
    """Edge-triggered, RELEASABLE arm pause driving an injected ``pause(bool)`` sink.

    ``update(arm_hold)`` calls ``pause(True)`` on the rising edge (arm enters hold) and
    ``pause(False)`` on the falling edge (person clears + resume dwell elapsed, decided
    upstream by :class:`~twin.humans.reaction.ReactionController`). Idempotent: the sink
    is only called on a change, so wiring it to a real arm gate every tick is cheap and
    does not re-issue a hold while already held.
    """

    def __init__(self, pause: Callable[[bool], None]):
        self._pause = pause
        self._paused: Optional[bool] = None

    @property
    def paused(self) -> bool:
        return bool(self._paused)

    def update(self, arm_hold: bool) -> None:
        arm_hold = bool(arm_hold)
        if arm_hold != self._paused:
            self._pause(arm_hold)
            self._paused = arm_hold


@dataclass
class MissionReaction:
    """Per-tick reaction hook for the mission loop: steps people, updates the safety
    machine, drives the arm pause, and exposes the base-velocity scale.

    ``people`` is the :class:`~twin.humans.isaac_people.PeopleStage` (or ``None`` when the
    actors are authored/stepped elsewhere, e.g. by the dynobs source). ``arm_pause`` is a
    :class:`ReversibleArmPause` (or ``None`` to not gate the arm). ``base_scale`` holds the
    latest multiplier the base-velocity touch point reads.
    """

    safety: MissionSafety
    people: object = None
    arm_pause: Optional[ReversibleArmPause] = None
    base_scale: float = 1.0
    last: Optional[ReactionOutput] = None

    def step(self, now: float) -> ReactionOutput:
        if self.people is not None:
            self.people.update(now)
        out = self.safety.update(now)
        self.base_scale = out.base_scale
        if self.arm_pause is not None:
            self.arm_pause.update(out.arm_hold)
        self.last = out
        return out

    def scale_velocity(self, linear: float, angular: float):
        """Convenience for the base path: scale a (v, omega) command by base_scale."""
        return linear * self.base_scale, angular * self.base_scale


def build_mission_reaction(config: Optional[dict], *, people_positions=None,
                           base_pose=None, camera_poses=None, people=None,
                           pause: Optional[Callable[[bool], None]] = None
                           ) -> Optional[MissionReaction]:
    """Assemble the per-tick reaction hook if ``humans.enabled``, else ``None``.

    Builds the humans :class:`MissionSafety` (detection provider + reaction mechanism)
    via :func:`twin.humans.integration.build_safety`; wires the reversible arm pause to
    ``pause`` when given. Returns ``None`` (inert, byte-identical) when humans is off.
    """
    humans = HumansConfig.from_config(config)
    if not humans.enabled:
        return None
    safety = build_safety(config, people_positions=people_positions,
                          base_pose=base_pose, camera_poses=camera_poses)
    if safety is None:
        return None
    arm_pause = ReversibleArmPause(pause) if pause is not None else None
    return MissionReaction(safety=safety, people=people, arm_pause=arm_pause)
