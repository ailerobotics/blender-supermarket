"""Bridge the real GPU detector's ``/human_safety/state`` into the reaction mechanism.

The fleet detector (``cams/human_safety``, keypoint R-CNN on CUDA) runs on the
mission camera topics with ``params/human_safety.restock.yaml`` and publishes its
judgement as JSON on ``/human_safety/state`` (schema in the fleet ``json_util``).
This module:

* :func:`safety_state_from_json` -- PURE decode of that JSON into the fields the
  reaction mechanism reads (unit-tested offline, no ROS);
* :class:`HumanSafetyBridge` -- PURE per-message handler: decode -> feed
  :class:`~twin.humans.detection.ReceivedSafety` -> run
  :class:`~twin.humans.reaction.ReactionController`, returning the reaction output
  and the shared ``Detection`` list (unit-tested offline);
* :func:`run_node` -- the thin rclpy node (import guarded) that wires the above to
  the live topics, publishes the reaction + ``Detection`` array on the event topic
  the behaviour-tree lane consumes, and records a JSONL trace. Reached only at GPU
  time; nothing here imports rclpy at module load.
"""
from __future__ import annotations

import json
import math
from typing import List, Optional, Tuple

import numpy as np

from .config import DetectionConfig, ReactionConfig
from .detection import Detection, ReceivedSafety
from .perception import DistanceStatus, SafetyState, Track, Zone
from .reaction import ReactionController, ReactionOutput


def _inf_if_none(x) -> float:
    return float('inf') if x is None else float(x)


def safety_state_from_json(payload: str) -> SafetyState:
    """Decode a fleet ``/human_safety/state`` JSON string into a ``SafetyState``.

    Only the fields the reaction mechanism and the shared ``Detection`` contract
    need are reconstructed; unknown/extra fields are ignored so a schema bump that
    only adds fields keeps working.
    """
    d = json.loads(payload)
    zone = Zone(d['zone'])
    status = DistanceStatus(d.get('distance_status', 'none'))
    tracks = []
    for t in d.get('tracks', []) or []:
        pos = t.get('position')
        tracks.append(Track(
            id=int(t.get('id', 0)),
            position=None if pos is None else np.asarray(pos, dtype=float),
            velocity=np.zeros(3), covariance=np.eye(6), stamp=float(t.get('stamp', d['stamp'])),
            confirmed=bool(t.get('confirmed', True)),
            nearest_point=(None if t.get('nearest_point') is None
                           else np.asarray(t['nearest_point'], dtype=float))))
    return SafetyState(
        stamp=float(d['stamp']), zone=zone,
        nearest_distance=_inf_if_none(d.get('nearest_distance')),
        nearest_distance_base=_inf_if_none(d.get('nearest_distance_base')),
        distance_status=status, nearest_track_id=d.get('nearest_track_id'),
        any_person_visible=bool(d.get('any_person_visible', False)),
        reasons=list(d.get('reasons', [])),
        evidence_stamp=d.get('evidence_stamp'),
        heartbeat_seq=int(d.get('heartbeat_seq', 0)), tracks=tracks)


class HumanSafetyBridge:
    """Pure per-message handler: detector state -> reaction output + detections."""

    def __init__(self, reaction_config: Optional[ReactionConfig] = None,
                 state_timeout_s: float = 0.7):
        self.received = ReceivedSafety(config=DetectionConfig(source='detector'),
                                       state_timeout_s=state_timeout_s)
        self.reaction = ReactionController(reaction_config)

    def on_json(self, payload: str, now: float) -> Tuple[ReactionOutput, List[Detection]]:
        self.received.on_state(safety_state_from_json(payload), now)
        return self.tick(now)

    def tick(self, now: float) -> Tuple[ReactionOutput, List[Detection]]:
        state = self.received.state(now)
        out = self.reaction.update(state, now)
        return out, self.received.detections(now)


def run_node(*, state_topic='/human_safety/state', event_topic='/humans/reaction',
             detection_topic='/humans/detections', reaction_config=None,
             trace_path=None, tick_hz=15.0):  # pragma: no cover - requires rclpy + a live graph
    """Thin rclpy node wiring :class:`HumanSafetyBridge` to the live topics.

    Subscribes ``state_topic`` (std_msgs/String JSON from the fleet detector),
    publishes the reaction (base_scale / arm_hold) on ``event_topic`` and the shared
    ``Detection`` list on ``detection_topic``, and optionally appends a JSONL trace.

    ROS 2 hygiene (fleet rule): to receive the mission's ``/human_safety/state`` and
    camera topics this node must join the MISSION graph -- export the mission's
    ``ROS_DOMAIN_ID`` (88) before launching it against a live run. For ISOLATED lane
    testing (a synthetic publisher, no mission graph) export this lane's own
    ``ROS_DOMAIN_ID=104`` and ``ROS_LOCALHOST_ONLY=1`` so lanes don't cross-talk.
    Always stop this node when done and remove only YOUR OWN stale
    ``/dev/shm/fastrtps_*`` segments afterwards (never a live one, never a rig's or
    another lane's) -- FastDDS exhausts its shared-memory ports otherwise.
    """
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    bridge = HumanSafetyBridge(reaction_config=reaction_config)
    trace = open(trace_path, 'a') if trace_path else None

    class _Bridge(Node):
        def __init__(self):
            super().__init__('humans_safety_bridge')
            self.create_subscription(String, state_topic, self._on_state, 10)
            self.event_pub = self.create_publisher(String, event_topic, 10)
            self.det_pub = self.create_publisher(String, detection_topic, 10)
            self.create_timer(1.0 / tick_hz, self._tick)

        def _now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def _on_state(self, msg):
            try:
                bridge.on_json(msg.data, self._now())
            except Exception as exc:
                self.get_logger().warn(f'bad state message: {exc!r}')

        def _tick(self):
            now = self._now()
            out, dets = bridge.tick(now)
            self.event_pub.publish(String(data=json.dumps(out.to_log())))
            self.det_pub.publish(String(data=json.dumps(
                [{'position': list(x.position), 'radius': x.radius, 'source': x.source,
                  'frame': x.frame, 'z': x.z} for x in dets])))
            if trace is not None:
                trace.write(json.dumps({'t': now, **out.to_log(),
                                        'n_detections': len(dets)}) + '\n')
                trace.flush()

    rclpy.init()
    node = _Bridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if trace is not None:
            trace.close()
