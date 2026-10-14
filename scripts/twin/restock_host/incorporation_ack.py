"""Host-side nvblox mapper-incorporation acknowledgement publisher.

Co-launched with the ``--host-stack`` planning stack (``twin.restock_host.launch``).
Unlike the full station supervisor (which launches its own planner per parked epoch),
this node does NOT start any child process -- the planner/mapper stack is already up as
its launch siblings. It only:

  1. speaks the parked-station command protocol on ``/restock_sim/station/command``
     (prepare / heartbeat / stop, epoch + sequence validated);
  2. polls the live nvblox ``get_esdf_and_gradient`` service to read NVIDIA's real
     incorporation stamp and advances ``collision_time`` ONLY through a successful,
     fresh ESDF integration (``preflight.incorporated_time``) -- input receipt is never
     an acknowledgement, and warm-up / empty / stale responses never advance it;
  3. publishes the acknowledgement snapshot (carrying ``collision_time``) on
     ``/restock_sim/station/status`` as a ``std_msgs/String`` JSON object.

That status object is the contract the mission consumes to sync
``IsaacMeasuredFeedback.collision_time`` (``0 <= now - collision_time <= 0.5``): without
a host-published ack the first manip job has no fresh ESDF stamp and dies on the Isaac
process/status watchdog. The nvblox launch runs ``update_esdf_rate_hz: 0.0`` (on-request
only), so this node's service poll is what actually integrates pending masked depth.

The epoch/acknowledgement state machine below is a self-contained copy of the parked
mapper lifecycle contract (the integration lane keeps the mirror-side copy); the two
agree on the JSON wire fields, not on a shared import.
"""
import argparse
import json
import math
import os
import time


class IncorporationAckState:
    """Fresh parked epochs and real mapper acknowledgements; no receipt shortcuts.

    Replicates the wire contract the mission's station client validates: a reply is
    trusted only with a matching ``epoch``, a fresh ``stamp``, ``state == 'ready'`` and a
    fresh ``collision_time``. collision_time is advanced only by acknowledge() with a
    NVIDIA ESDF integration stamp strictly after the parked epoch and never going
    backwards.
    """
    def __init__(self, *, wall_clock=time.monotonic):
        self.wall_clock = wall_clock
        self.epoch = 0
        self.sequence = -1
        self.parked_since = -1.
        self.last_heartbeat = -1.
        self.wall_heartbeat = wall_clock()
        self.collision_time = -1.
        self.active = False
        self.fault = None
        self.camera_frame = None

    def accept(self, command, now):
        if self.fault:
            raise RuntimeError('station fault requires explicit recovery')
        epoch = command.get('epoch')
        stamp = command.get('stamp')
        seq = command.get('sequence')
        action = command.get('action')
        if (type(epoch) is not int or epoch <= 0 or type(seq) is not int or seq < 0
                or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
                or not math.isfinite(now) or not 0 <= now - stamp <= .5):
            raise ValueError('invalid station epoch, sequence or timestamp')
        if epoch < self.epoch or (epoch == self.epoch and seq <= self.sequence):
            return 'ignored'
        if action == 'prepare':
            if epoch <= self.epoch or command.get('parked') is not True:
                raise ValueError('new measured parked epoch required')
            frame = command.get('camera_frame')
            if not isinstance(frame, str) or not frame.startswith('restock_camera_'):
                raise ValueError('calibrated mapper camera frame required')
            self.epoch, self.parked_since, self.camera_frame = epoch, stamp, frame
            self.active = True
            self.collision_time = -1.
        elif action in ('heartbeat', 'stop'):
            if epoch != self.epoch or not self.active:
                raise ValueError('station command has no active matching epoch')
            if action == 'heartbeat' and command.get('parked') is not True:
                raise ValueError('base/cart motion invalidates parked mapper')
            if action == 'stop':
                self.active = False
                self.collision_time = -1.
        else:
            raise ValueError('unsupported station action')
        self.sequence = seq
        self.last_heartbeat = stamp
        self.wall_heartbeat = self.wall_clock()
        return action

    def acknowledge(self, stamp, epoch):
        if (not self.active or epoch != self.epoch or not math.isfinite(stamp)
                or stamp <= self.parked_since or stamp < self.collision_time):
            raise ValueError('mapper acknowledgement does not belong to current parked epoch')
        self.collision_time = stamp

    def snapshot(self, now, *, endpoints_ready, process_alive):
        if self.active and (not process_alive or now - self.last_heartbeat > .5
                            or self.wall_clock() - self.wall_heartbeat > 1.):
            self.fault = 'station process or parked heartbeat lost'
        ready = (self.active and not self.fault and endpoints_ready
                 and 0 <= now - self.collision_time <= .5 and self.collision_time > self.parked_since)
        return dict(epoch=self.epoch, sequence=self.sequence, stamp=now,
                    state='failed' if self.fault else ('ready' if ready else ('warming' if self.active else 'idle')),
                    reason=self.fault or '', collision_time=self.collision_time if ready else -1.,
                    parked_since=self.parked_since, camera_frame=self.camera_frame)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Workspace bounds (ur_base_link frame) scope the ESDF query AABB. The stack launch
    # has already validated the full config; this co-launched child takes only the bounds.
    p.add_argument('--workspace-min', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'))
    p.add_argument('--workspace-max', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'))
    args = p.parse_args()
    workspace_min, workspace_max = args.workspace_min, args.workspace_max

    from .config import isolated_environment, NS, FRAME
    from .preflight import incorporated_time
    from .readiness import missing_endpoints
    os.environ.update(isolated_environment(os.environ))

    import rclpy
    from rclpy.parameter import Parameter
    from rclpy.action import get_action_server_names_and_types_by_node
    from std_msgs.msg import String, Bool
    from nvblox_msgs.srv import EsdfAndGradients

    rclpy.init()
    node = rclpy.create_node('restock_incorporation_ack', namespace=NS,
                             parameter_overrides=[Parameter('use_sim_time', value=True)])
    if node.context.get_domain_id() != 88:
        raise RuntimeError('incorporation ack requires domain 88')
    state = IncorporationAckState()
    publisher = node.create_publisher(String, NS + '/station/status', 10)
    stop = node.create_publisher(Bool, NS + '/manipulation/stop', 10)
    service = node.create_client(EsdfAndGradients, NS + '/nvblox_node/get_esdf_and_gradient')
    pending = None
    pending_epoch = None
    pending_wall = 0.
    last_discovery = 0.
    endpoints_ready = False
    diag_left = 6  # log the first few incorporation rejections for diagnosis
    now = lambda: node.get_clock().now().nanoseconds * 1e-9

    # The ESDF query AABB only needs to incorporate depth and read NVIDIA's integration
    # stamp, not return the whole workspace: a large grid (millions of voxels) serializes
    # slowly enough that the response stamp is stale on arrival (now - stamp > 0.5). Clamp
    # the query to a focused box (<= 1.6 m/axis) centred in the workspace, kept in bounds.
    query_min, query_size = [], []
    for a, b in zip(workspace_min, workspace_max):
        centre, half = (a + b) / 2.0, min((b - a) / 2.0, 0.8)
        query_min.append(centre - half)
        query_size.append(2.0 * half)

    def command(message):
        try:
            state.accept(json.loads(message.data), now())
        except Exception as exc:
            state.fault = str(exc)
            stop.publish(Bool(data=True))

    def tick():
        nonlocal pending, pending_epoch, pending_wall, last_discovery, endpoints_ready, diag_left
        try:
            stamp = now()
            if state.active and not state.fault:
                if time.monotonic() - last_discovery > .5:
                    actions = []
                    for name, namespace in node.get_node_names_and_namespaces():
                        actions.extend(get_action_server_names_and_types_by_node(node, name, namespace))
                    missing = missing_endpoints(node.get_service_names_and_types(), actions)
                    # The mapper-incorporation stamp depends on the planner/mapper graph, not
                    # the downstream external arm controller (arm_controller/follow_joint_
                    # trajectory is provided outside this launch), so its absence must not
                    # gate collision_time -- the same exclusion the stack probe/verify apply.
                    endpoints_ready = not [m for m in missing if 'follow_joint_trajectory' not in m]
                    last_discovery = time.monotonic()
                if pending is not None and pending.done():
                    response = pending.result()
                    pending = None
                    # Warm-up / empty / stale responses never advance the acknowledgement.
                    try:
                        incorporated = incorporated_time(response, now=stamp, parked_since=state.parked_since)
                        state.acknowledge(incorporated, pending_epoch)
                    except (ValueError, AttributeError) as exc:
                        if diag_left > 0:
                            diag_left -= 1
                            try:
                                h = response.header
                                rs = h.stamp.sec + h.stamp.nanosec * 1e-9
                                node.get_logger().warn(
                                    f'incorp reject: {exc}; success={getattr(response, "success", "?")} '
                                    f'frame={h.frame_id!r} resp_stamp={rs:.3f} now={stamp:.3f} '
                                    f'dt={stamp - rs:.3f} parked_since={state.parked_since:.3f} '
                                    f'ndata={len(response.esdf_and_gradients.data)}')
                            except Exception as e2:
                                node.get_logger().warn(f'incorp reject: {exc}; (no header: {e2})')
                if pending is not None and time.monotonic() - pending_wall > 2.:
                    pending.cancel()
                    pending = None
                if pending is None and endpoints_ready and service.service_is_ready():
                    request = EsdfAndGradients.Request()
                    request.update_esdf = True
                    request.visualize_esdf = False
                    request.use_aabb = True
                    request.frame_id = FRAME
                    for axis, mn, sz in zip(('x', 'y', 'z'), query_min, query_size):
                        setattr(request.aabb_min_m, axis, float(mn))
                        setattr(request.aabb_size_m, axis, float(sz))
                    pending = service.call_async(request)
                    pending_epoch = state.epoch
                    pending_wall = time.monotonic()
            # The launch's OnProcessExit guard tears this node down with the stack, so a
            # live ack node implies the planner/mapper processes are still alive.
            result = state.snapshot(stamp, endpoints_ready=endpoints_ready, process_alive=True)
            if state.fault:
                stop.publish(Bool(data=True))
            publisher.publish(String(data=json.dumps(result, allow_nan=False)))
        except Exception as exc:
            state.fault = str(exc)
            stop.publish(Bool(data=True))

    node.create_subscription(String, NS + '/station/command', command, 10)
    node.create_timer(.05, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
