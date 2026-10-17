"""Thin opt-in glue the scene / mission-loop / behaviour-tree owners call.

Everything funnels through :func:`build` so the three contested touch points stay
one guarded line each in the files their lanes own, and nothing happens at all when
the ``humans`` config block is absent:

1. **Scene composition** (``twin.restock.physics_setup.assemble``, owned by the
   integration lane) -- after the stage is open::

       people = humans.integration.author_people(stage, config, poses)

2. **Mission step loop** (the simulator port's ``advance``) -- once per physics
   step, with the mission-relative sim time::

       humans.integration.step_people(people, sim_time)
       react = safety.update(sim_time)          # react.base_scale / react.arm_hold

3. **Base + arm command path** (navigation config owned by
   ``sim-restock-dynobs-v1``; arm planner by ``sim-restock-planner-obstacles-v1``;
   the reactive subtree by ``sim-restock-bt-v1``) -- scale the nav velocity by
   ``react.base_scale`` and gate the arm on ``react.arm_hold``. The
   :class:`~twin.humans.reaction.ReactionController` is the *mechanism*; the BT lane
   calls it from its ``human detected`` subtree and consumes the shared
   ``Detection(position, radius, source)`` list from ``safety.detections(now)``.

This module imports USD only inside :func:`author_people`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import HumansConfig
from .detection import Detection, make_provider
from .people import default_people
from .reaction import ReactionController, ReactionOutput

Pose2D = Tuple[float, float, float]


def resolve_people(humans: HumansConfig, poses: Optional[Dict[str, Sequence[float]]]):
    """The configured people, or the default crowd derived from mission poses."""
    if humans.people:
        return humans.people
    if poses is None:
        raise ValueError('humans enabled with no people and no poses to derive them')
    return default_people(poses)


def author_people(stage, config: Optional[dict], poses: Optional[Dict[str, Sequence[float]]],
                  *, physical: bool = False):
    """Author the scripted people into ``stage`` if enabled; else return ``None``.

    ``config`` is the raw runtime-config dict; the ``humans`` block drives it.
    Returns a :class:`~twin.humans.isaac_people.PeopleStage` (step it each frame) or
    ``None`` when the layer is off.
    """
    humans = HumansConfig.from_config(config)
    if not humans.enabled:
        return None
    from .isaac_people import PeopleStage               # USD import deferred
    stage_people = PeopleStage(humans.with_people(tuple(resolve_people(humans, poses))))
    stage_people.author(stage, physical=physical)
    return stage_people


def step_people(people, sim_time: float) -> None:
    """Teleport the people to their scripted poses at ``sim_time`` (no-op if None)."""
    if people is not None:
        people.update(sim_time)


@dataclass
class MissionSafety:
    """Bundles the detection provider and the reaction mechanism for the loop/BT."""

    provider: object                 # GroundTruthSafety | ReceivedSafety
    reaction: ReactionController
    provenance: str

    def update(self, now: float) -> ReactionOutput:
        return self.reaction.update(self.provider.state(now), now)

    def state(self, now: float):
        return self.provider.state(now)

    def detections(self, now: float) -> List[Detection]:
        return self.provider.detections(now)


def build_safety(config: Optional[dict], *, people_positions=None, base_pose=None,
                 camera_poses=None) -> Optional[MissionSafety]:
    """Assemble the detection provider + reaction mechanism if enabled, else None.

    For the ground-truth provider the caller supplies the live accessors
    (``people_positions(now)`` -> ``[(id, (x,y,z), radius)]``, ``base_pose()`` and
    ``camera_poses()``, all reading the stage). For the detector provider only the
    reaction config is needed; the node feeds ``provider.on_state(...)``.
    """
    humans = HumansConfig.from_config(config)
    if not humans.enabled:
        return None
    det = humans.detection
    if det.source == 'ground_truth':
        if None in (people_positions, base_pose, camera_poses):
            raise ValueError('ground-truth detection needs people/base/camera pose accessors')
        provider = make_provider(det, people=people_positions, base_pose=base_pose,
                                 camera_poses=camera_poses)
    else:
        provider = make_provider(det, state_timeout_s=humans.reaction.state_timeout_s)
    return MissionSafety(provider=provider,
                         reaction=ReactionController(humans.reaction),
                         provenance=getattr(provider, 'provenance', det.source))
