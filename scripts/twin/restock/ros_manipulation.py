"""Dispatch real camera packets to the host planner through native Isaac ROS.

No ROS Python import enters Isaac. The graph owns gripper actuation and its
latched heartbeat stop. Camera capture and mapper acknowledgement remain real
runtime dependencies supplied by the guarded simulator bundle.
"""
import math

from twin.restock_manipulation.types import Result


class RosManipulationRuntime:
    cart_location = 'rollwagen'

    def __init__(self, graph, capture, *, prepare_station):
        self.graph = graph
        self.capture = capture
        self.prepare_station = prepare_station
        self.last_sensor_time = -1.
        self.latest = Result('running','idle')

    def start(self, operation, job):
        if operation not in ('collect','transfer'):
            raise ValueError('unsupported transfer operation')
        self.source = job['source_bin'] if operation == 'collect' else self.cart_location
        self.destination = self.cart_location if operation == 'collect' else job['shelf_id']
        # Invalidate old ROIs, current arm-frame TF and mapper acknowledgement.
        self.preparation = self.prepare_station(operation, job)
        self.operation,self.job=operation,job
        self.dispatched=False
        self.last_sensor_time = -1.
        self.latest = Result('running','observe')
        if self.preparation is None:
            self.graph.start(operation,job)
            self.dispatched=True

    def step(self, dt, simulation_time):
        try:
            if not self.dispatched:
                if not self.preparation.ready(simulation_time):
                    return Result('running','prepare_station')
                self.graph.start(self.operation,self.job)
                self.dispatched=True
            packet = self.capture(simulation_time,self.source,self.destination)
            if packet is not None:
                stamp = packet.simulation_time
                if (not math.isfinite(stamp) or stamp <= self.last_sensor_time or
                        not 0 <= simulation_time-stamp <= .5):
                    raise ValueError('stale or repeated camera packet')
                items = list(packet.observations)+([packet.target] if packet.target else [])
                if any(item.source != 'sensor' for item in items):
                    raise ValueError('production manipulation requires sensor provenance')
                self.graph.publish_sensors(packet)
                self.last_sensor_time = stamp
            reply = self.graph.poll()
            if reply is not None:
                status = reply.get('status')
                if status not in ('running','succeeded','failed','rejected'):
                    raise ValueError('invalid host transfer status')
                self.latest = Result('failed' if status == 'rejected' else status,
                    reply.get('phase','host'),reply.get('reason',''),controls=(),
                    evidence=tuple(reply.get('evidence',())))
                if self.latest.status == 'failed':
                    self.stop()
            return self.latest
        except Exception as exc:
            self.stop()
            self.latest = Result('failed','host',str(exc))
            return self.latest

    def stop(self):
        self.graph.close()  # hold; never unseal a held object on failure
