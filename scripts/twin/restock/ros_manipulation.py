"""Dispatch real camera packets to the host planner through native Isaac ROS.

No ROS Python import enters Isaac. The graph owns gripper actuation and its
latched heartbeat stop. Camera capture and mapper acknowledgement remain real
runtime dependencies supplied by the guarded simulator bundle.
"""
import math

from twin.restock_manipulation.types import Result


class RosManipulationRuntime:
    cart_location = 'rollwagen'

    def __init__(self, graph, capture, *, prepare_station, sensing_bridge=None, feedback=None,
                 station=None, station_camera_frame='restock_camera_arm'):
        self.graph = graph
        self.capture = capture
        self.prepare_station = prepare_station
        # Measured sensing/manipulation feedback whose collision_time is mirrored each step from
        # the parked-station ack (StationClient over /restock_sim/station/command|status), so the
        # sensing capture's collision_time gate sees a real, fresh nvblox-incorporation stamp --
        # never a depth-receipt or fabricated value. ``station`` is the StationClient; its ack is
        # the host-launched twin.restock_host.incorporation_ack. ``station_camera_frame`` is the
        # masked-depth optical frame whose ESDF the ack integrates (restock_camera_arm).
        self.feedback = feedback
        self.station = station
        self.station_camera_frame = station_camera_frame
        self._station_started = False
        # Optional Isaac-side arm masked-depth/TF/bbox bridge for the host nvblox (option B).
        # None keeps the legacy behaviour; publishes every step so the host's 0.5 s bbox
        # freshness never lapses and nvblox gets continuous depth while the station maps.
        self.sensing_bridge = sensing_bridge
        self.last_sensor_time = -1.
        self.latest = Result('running','idle')

    def start(self, operation, job):
        if operation not in ('collect','transfer'):
            raise ValueError('unsupported transfer operation')
        print(f'MANIP_START operation={operation}', flush=True)  # execute phase reached (diagnostic)
        self.source = job['source_bin'] if operation == 'collect' else self.cart_location
        self.destination = self.cart_location if operation == 'collect' else job['shelf_id']
        if self.sensing_bridge is not None:
            # The grasp target to carve from the ESDF: the pick source for collect, the
            # place destination for transfer.
            self.sensing_bridge.target_location = (self.source if operation == 'collect'
                                                   else self.destination)
        # Invalidate old ROIs, current arm-frame TF and mapper acknowledgement.
        self.preparation = self.prepare_station(operation, job)
        self.operation,self.job=operation,job
        self.dispatched=False
        self._station_started = False
        self.last_sensor_time = -1.
        self.latest = Result('running','observe')
        if self.preparation is None:
            self.graph.start(operation,job)
            self.dispatched=True

    def step(self, dt, simulation_time, parked=None):
        try:
            # The port's robust net-pose-displacement parked (Packet.parked) overrides the flappy
            # instantaneous velocity for the station gate AND the capture's motion.parked().
            if parked is not None and hasattr(self.feedback, 'set_parked'):
                self.feedback.set_parked(parked)
            # Drive the parked-station ack and mirror its collision_time onto the feedback BEFORE
            # this tick's capture reads it (the sensing capture gates on motion.collision_time
            # freshness). begin() prepares the station on the first step; later steps heartbeat.
            if self.station is not None and self.feedback is not None:
                if not self._station_started:
                    self.station.begin(self.station_camera_frame, simulation_time)
                    self._station_started = True
                else:
                    self.station.tick(simulation_time, parked=bool(self.feedback.parked()))
                self.feedback.collision_time = self.station.collision_time
            if not self.dispatched:
                if not self.preparation.ready(simulation_time):
                    return Result('running','prepare_station')
                self.graph.start(self.operation,self.job)
                self.dispatched=True
            if self.sensing_bridge is not None:
                # Continuous masked depth + info + TF + (cached) target bbox for the host
                # nvblox/object_segmenter EVERY sensing tick, independent of carton verification.
                self.sensing_bridge.publish(simulation_time)
            # A carton-verification (capture) failure must NOT stop the bridge or the mission:
            # the bridge keeps publishing so the host ESDF builds; the perception gate is just
            # not satisfied yet, and the stage timeout still bounds a genuinely stuck capture.
            try:
                packet = self.capture(simulation_time,self.source,self.destination)
                if packet is not None:
                    stamp = packet.simulation_time
                    if (not math.isfinite(stamp) or stamp <= self.last_sensor_time or
                            not -0.02 <= simulation_time-stamp <= .5):
                        raise ValueError('stale or repeated camera packet')
                    items = list(packet.observations)+([packet.target] if packet.target else [])
                    if any(item.source != 'sensor' for item in items):
                        raise ValueError('production manipulation requires sensor provenance')
                    self.graph.publish_sensors(packet)
                    self.last_sensor_time = stamp
            except Exception as cap_exc:
                self.latest = Result('running','observe',str(cap_exc))
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
        if self.station is not None:
            try:
                self.station.stop()
            finally:
                if self.feedback is not None:
                    self.feedback.collision_time = -1.
        if self.sensing_bridge is not None and hasattr(self.sensing_bridge, 'close'):
            try:
                self.sensing_bridge.close()  # terminate the out-of-Isaac publisher subprocess
            except Exception:
                pass
        self.graph.close()  # hold; never unseal a held object on failure
