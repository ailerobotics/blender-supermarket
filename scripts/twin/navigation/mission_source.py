"""Mission-side moving-obstacle detection source for the sensor nav route.

The dynamic-obstacle layer (``navigation.dynamic.DynamicObstacleLayer``, wired into
``SensorNavigation`` by the factory when ``navigation.dynamic_obstacles`` is set) is
inert until something feeds it ``Detection(position, radius, source)`` per frame. In a
live mission that feed comes from this module: ``NavigationBridge`` calls the returned
``source(now)`` once per navigate tick and passes the result into
``SensorNavigation.step(detections=...)``.

Design (opt-in, default-off):

* Absent ``navigation.dynamic_obstacle_source`` => the bundle never builds a source and
  the run is byte-identical; the layer, even if configured, stays inert.
* :func:`build_dynamic_obstacle_source` REUSES the ``sim-restock-humans-v1`` scripted
  person asset (the decision's "person asset if it is cheap to reuse"): the ``humans``
  config block authors one or more deterministic scripted actors, and this source turns
  each actor's KNOWN pose at ``now`` into a map-frame ``Detection``. Because the detection
  position comes from the same :class:`~twin.humans.isaac_people.PeopleStage` scripted
  path that renders the visible actor, pixels and the costmap agree by construction.
* The ``source`` provenance is ``'scenario_scripted_actor'`` -- honest: this is a
  deterministic scripted-actor ground truth, NOT a perception detection. The perception
  path (the humans lane's CUDA detector / ground-truth projection via
  ``humans.integration.build_safety(...).detections`` + ``detection.to_nav_detections``)
  plugs into the SAME ``source(now)`` seam when that lane's GPU detector run lands; see
  :func:`humans_detection_source`.

``world_positions`` is a pure helper (USD-free), so the detection feed is fully
CPU-testable. Authoring/stepping the visible prim needs a live USD stage and is
best-effort: if it is unavailable the detections still flow (the base still avoids the
obstacle), they are just not rendered.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from .dynamic import Detection

SCRIPTED_SOURCE = 'scenario_scripted_actor'


def build_dynamic_obstacle_source(*, app=None, assembly=None, plan=None, config=None,
                                  output=None) -> Optional[Callable[[float], List[Detection]]]:
    """Return a ``source(now) -> [Detection]`` for the sensor route, or ``None``.

    Resolved by ``twin.restock.isaac_bundle`` as
    ``navigation.dynamic_obstacle_source = "twin.navigation.mission_source:build_dynamic_obstacle_source"``.
    Returns ``None`` (no feed, byte-identical) when the ``humans`` block is absent/disabled,
    so an operator cannot accidentally arm a detection feed with no scripted actor to back it.
    """
    from twin.humans.config import HumansConfig
    from twin.humans.integration import resolve_people
    from twin.humans.isaac_people import PeopleStage

    humans = HumansConfig.from_config(config)
    if not humans.enabled:
        return None
    poses = getattr(plan, 'poses', None)
    people = tuple(resolve_people(humans, poses))
    if not people:
        return None
    radii = {p.person_id: float(p.radius_m) for p in people}
    stage_people = PeopleStage(humans.with_people(people))

    # Visible actor: author the scripted people into the live stage (best-effort). The
    # detections do NOT depend on this -- world_positions is pure -- so a stage that is
    # absent (CPU tests) or an authoring failure degrades to "detected but not rendered"
    # rather than breaking the mission tick.
    authored = False
    stage = getattr(assembly, 'stage', None)
    if stage is not None:
        try:
            stage_people.author(stage)
            authored = True
        except Exception as exc:  # pragma: no cover - needs a live USD stage
            print(f'DYNOBS visible actor author skipped: {type(exc).__name__}: {exc}',
                  flush=True)

    def source(now: float) -> List[Detection]:
        # Teleport the visible prim to its scripted pose at the mission clock, in lockstep
        # with the detection it produces (one render frame behind, within the layer's
        # confirm window). Pure position read backs the detection regardless.
        if authored:
            try:
                stage_people.update(now)
            except Exception:  # pragma: no cover - needs a live USD stage
                pass
        out: List[Detection] = []
        for pid, (x, y, _z) in stage_people.world_positions(now).items():
            out.append(Detection(position=(float(x), float(y)),
                                  radius=radii[pid], source=SCRIPTED_SOURCE))
        return out

    return source


def humans_detection_source(safety) -> Callable[[float], List[Detection]]:
    """Adapt the humans lane's ``MissionSafety`` to the ``source(now)`` seam.

    ``safety.detections(now)`` returns the humans lane's own ``Detection`` dataclass
    (perception-labelled, e.g. ``'human_ground_truth'`` / ``'human_safety'``); this wraps
    it through ``detection.to_nav_detections`` so the SAME navigate seam consumes the
    perception path once that lane's detector/frame GPU run lands. Not exercised by the
    deterministic scenario evidence run (which uses the scripted source above).
    """
    from twin.humans.detection import to_nav_detections

    def source(now: float) -> List[Detection]:
        return list(to_nav_detections(safety.detections(now)))

    return source
