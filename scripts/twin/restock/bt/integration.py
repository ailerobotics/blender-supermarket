"""Glue the behaviour-tree orchestrator into the live run (``run_mobile_restock.py``), opt-in.

This is the ONLY place the BT layer touches the run harness, and it is gated: nothing here runs
unless the runtime-config sets ``"bt": true`` (default off -- the state machine stays production).
The touch point in ``run_mobile_restock.run_once`` is a two-line branch into :func:`run_once_bt`.

The ROS event subscriber and the adapter reactions are best-effort: on a runtime without rclpy
(CPU) the subscriber is simply absent, and the reactions fall back to the adapter's existing
payload-preserving primitives. The humans / dynamic-obstacle lanes publish their detections as
``std_msgs/String`` JSON onto :data:`events.EVENT_TOPIC`; we parse and enqueue them here.
"""
from __future__ import annotations

import json

from .driver import build_bus, run_mission_tree
from .events import EVENT_TOPIC, Event
from .context import AdapterReactions


def make_reactions(bundle):
    """Adapter-backed reactions for a live run. ``recommission`` is delegated to the stations lane's
    in-session provider if the bundle exposes one (``bundle.recommission``), else refused."""
    return AdapterReactions(getattr(bundle, 'adapter', bundle),
                            recommission_provider=getattr(bundle, 'recommission', None))


def start_event_subscriber(queue):
    """Subscribe to :data:`EVENT_TOPIC` and push parsed events onto ``queue``. Returns a callable
    that tears the subscriber down, or ``None`` if ROS is unavailable (CPU runs). Best-effort: a
    malformed message is logged and dropped, never raised onto the mission thread."""
    try:
        import threading
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except Exception:
        return None

    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init(args=None)

    class _Sub(Node):
        def __init__(self):
            super().__init__('restock_bt_events')
            self.create_subscription(String, EVENT_TOPIC, self._on, 10)

        def _on(self, msg):
            try:
                queue.push(Event.from_json(msg.data))
            except Exception as exc:
                self.get_logger().warning(f'dropped bad BT event: {exc}')

    node = _Sub()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    def shutdown():
        try:
            executor.shutdown()
            node.destroy_node()
        finally:
            if owns_rclpy and rclpy.ok():
                rclpy.shutdown()
    return shutdown


def run_once_bt(db, scenario, bundle, config, output_dir, *, mission_factory, is_running=None):
    """Drive one mission attempt through the behaviour tree, writing the live view into
    ``output_dir``. Returns ``(mission, summary)`` with the SAME shape ``run_once`` returns, so the
    warm-session loop and mission.json persistence are unchanged.

    ``mission_factory()`` builds a fresh ``Mission`` (state ``'new'``) exactly as ``run_once`` does
    -- the tree calls ``start(begin=False)`` on it. The live-view files land in ``output_dir``:
    ``bt_view.html`` (phone), ``bt_snapshot.json`` (polled), ``bt_events.jsonl`` (timeline)."""
    from .view import SnapshotWriter

    mission = mission_factory()
    bus, queue = build_bus(scenario)
    # Write the live view FIRST so bt_view.html exists even if a later step fails.
    writer = SnapshotWriter(output_dir, title=f"restock {scenario.get('job', {}).get('job_id', '')}")
    # The ROS event subscriber is OPT-IN (config 'bt_ros_events'): it creates a second rclpy node
    # inside the Isaac process, which already runs its own ROS2 bridge -- doing so unconditionally
    # SIGABRTs the simulator (proven in the sim-restock-bt-v1 GPU proof). A scripted-scenario run
    # needs no ROS at all (events come from the ScenarioEventSource), so default off keeps the
    # mission stable; the humans/dynobs lanes enable it only when they publish live detections.
    shutdown = start_event_subscriber(queue) if config.get('bt_ros_events') else None
    try:
        ctx, root, summary = run_mission_tree(
            mission, bus, reactions=make_reactions(bundle), snapshot=writer,
            is_running=is_running)
    finally:
        if shutdown is not None:
            shutdown()

    summary.setdefault('note', '')
    if mission.shortcut is not None:
        summary['note'] += (' START-PHASE SHORTCUT: stages ' + ', '.join(mission.shortcut['skipped'])
                            + ' were synthesized (robot pose, braked base, carton attach), not '
                              'performed; see events *_shortcut.')
    summary['bt_view'] = str((output_dir / 'bt_view.html'))
    try:
        early = dict(summary, events=db.events(scenario['job']['job_id']))
        (output_dir / 'mission.json').write_text(json.dumps(early, indent=2, allow_nan=False) + '\n')
    except Exception:
        pass
    return mission, summary
