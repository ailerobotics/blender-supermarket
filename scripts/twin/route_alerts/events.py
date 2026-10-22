"""Publish findings: store-vision events -> robot-tagged events + staff tasks.

The detectors emit store-vision ``core.types.Event`` objects (``spill.detected``,
``obstruction.aisle_blocked``, ``strays.coldchain``, ...). This module:

* keeps those store-vision events intact (the store-vision event stream), and
* mirrors each into a ``robot.route_alert.<original_type>`` event -- a
  robot-originated finding carrying the original type + payload + the robot/camera
  context, which is the vocabulary the mission artifact and the ``sim-restock-bt-v1``
  behaviour-tree lane consume; and
* optionally materialises de-duplicated store-vision staff tasks from the findings
  (store-vision ``tasks.TaskGenerator``: CLEANING for spill/obstruction, STRAY for
  cold-chain), so the drive raises the same human work a fixed camera would.

store-vision is imported lazily; nothing here runs unless the runner calls it.
"""
from __future__ import annotations

ROBOT_PREFIX = "robot.route_alert."


def to_robot_events(events, *, robot_id: str | None = None):
    """Mirror store-vision findings as ``robot.route_alert.*`` events (lazy import).

    Returns NEW events; the inputs are not mutated. The original type is preserved
    in ``payload['store_vision_type']`` so a consumer can recover the source class.
    """
    from storevision.core.ids import new_event_id
    from storevision.core.types import Event

    out = []
    for ev in events:
        payload = dict(ev.payload or {})
        payload["store_vision_type"] = ev.type
        payload["sensor"] = "mobile_robot"
        if robot_id is not None:
            payload["robot_id"] = robot_id
        out.append(
            Event(
                event_id=new_event_id("ra"),
                time=ev.time,
                type=ROBOT_PREFIX + ev.type,
                severity=ev.severity,
                location=ev.location,
                camera_id=ev.camera_id,
                payload=payload,
                evidence=list(ev.evidence or []),
            )
        )
    return out


def build_staff_tasks(events, store, *, now: float | None = None, generator=None):
    """De-duplicated store-vision staff tasks from the findings (lazy import).

    Pass a persistent ``generator`` (``storevision.tasks.TaskGenerator``) across
    calls so an unchanged, still-open hazard folds into its existing task instead of
    raising a duplicate; absent, a fresh generator is built per call. Returns the
    list of newly created / materially updated :class:`StaffTask` objects.
    """
    if generator is None:
        from storevision.tasks.generation import TaskGenerator

        # ``store`` here is the StaffTaskStore (open-task dedup), not the StoreConfig.
        generator = TaskGenerator(store)
    tasks = []
    for ev in events:
        task = generator.ingest(ev, now=now)
        if task is not None:
            tasks.append(task)
    return tasks
