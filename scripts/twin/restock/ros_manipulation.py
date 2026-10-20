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
        # Do NOT reset _station_started: a preceding warm-up (RuntimeAdapter 'warm' phase) may have
        # already begun the station and driven collision_time fresh; resetting would start a new
        # parked epoch and throw that warm map away, re-cold-starting the ack. __init__ leaves it
        # False, so a collect with no warm-up still begins the station on the first step().
        self.last_sensor_time = -1.
        self.latest = Result('running','observe')
        if self.preparation is None:
            self.graph.start(operation,job)
            self.dispatched=True

    def _pump(self, simulation_time, parked):
        """Drive the parked-station ack + mirror collision_time + publish masked depth.

        Shared by warm() (approach/parked warm-up, before the collect job) and step() (during
        collect). Running it during the warm-up builds the host nvblox ESDF and makes
        collision_time fresh BEFORE start() dispatches the job, so the first collect capture can
        emit /sensors inside the host's 1 s sensor watchdog instead of racing it from cold.
        """
        # The port's robust net-pose-displacement parked (Packet.parked) overrides the flappy
        # instantaneous velocity for the station gate AND the capture's motion.parked().
        if parked is not None and hasattr(self.feedback, 'set_parked'):
            self.feedback.set_parked(parked)
        # begin() prepares the station on the first pump; later pumps heartbeat. Mirror its
        # collision_time onto the feedback BEFORE any capture reads it (capture gates on freshness).
        if self.station is not None and self.feedback is not None:
            if not self._station_started:
                # The parked-mapper epoch can only be OPENED with a measured parked base: StationState
                # rejects a 'prepare' whose command carries parked != True. When the mission BEGINS at a
                # manipulation stage (a start-phase shortcut, e.g. --start-at transfer) the first warm
                # ticks are not yet parked while the port's stationarity window fills, so beginning here
                # would be rejected AND still latch _station_started -> the station could never activate
                # (ds-012: station stuck 'idle', collision_time never advanced). Defer the first begin
                # until parked. Collect enters warm already parked (post-nav) so it begins immediately.
                if bool(self.feedback.parked()):
                    self.station.begin(self.station_camera_frame, simulation_time)
                    self._station_started = True
            else:
                self.station.tick(simulation_time, parked=bool(self.feedback.parked()))
            self.feedback.collision_time = self.station.collision_time
            self._sc = getattr(self, '_sc', 0) + 1
            if self._sc % 10 == 1:                       # STATION_STATS diagnostic
                snap = getattr(self.station, 'latest', None)
                sc = getattr(self.station, 'stage_clock', None)
                stage = sc() if sc is not None else None
                print(f'STATION_STATS tick={self._sc} now={simulation_time:.3f} '
                      f'stage_stamp={stage} collision_time={self.feedback.collision_time} '
                      f'ready={self.station.ready(simulation_time)} snapshot={snap}', flush=True)
        if self.sensing_bridge is not None:
            # Continuous masked depth + info + TF + (cached) target bbox for the host
            # nvblox/object_segmenter EVERY sensing tick, independent of carton verification.
            # Stamp the masked depth with the SAME host clock the /tf relay uses (world.current_time
            # via the station's stage_clock), so nvblox's TF lookup at the depth stamp resolves --
            # the dt-accumulated mission clock lagged /tf and silently dropped every lookup (no ESDF).
            sc = getattr(self.station, 'stage_clock', None) if self.station is not None else None
            self.sensing_bridge.publish(sc() if sc is not None else simulation_time)

    def pause_station(self, simulation_time):
        """fm399 (1): cleanly stop the current parked epoch at the START of the reposition drive, while
        still parked. A moving base invalidates the parked mapper, and the host faults + spams
        emergency-stop the instant a parked station stops being heartbeat-ed (run-309). Sending the
        'stop' while parked lands it as 'idle' (NOT faulted), so the host publishes no emergency stop
        during the drive. Clearing collision_time + _station_started makes the next _pump (the
        place_approach re-warm) begin() a FRESH epoch at the new pose; its ESDF re-incorporates there
        and the incorporation-ack parked_since gate only lets collision_time (hence ready) advance with
        that fresh integration."""
        if self.station is not None:
            try:
                self.station.stop()
            except Exception:
                pass            # already idle/down -> begin() still re-opens a fresh epoch cleanly
            if self.feedback is not None:
                self.feedback.collision_time = -1.
        self._station_started = False

    def warm(self, simulation_time, parked=None):
        """Approach/parked warm-up tick (no job dispatch, no capture): pump the station + masked
        depth so the ESDF builds and collision_time goes fresh before the collect job is sent.
        Returns True once the station reports a fresh collision_time (ready)."""
        try:
            self._pump(simulation_time, parked)
        except Exception as exc:
            self.latest = Result('running', 'warm', str(exc))
            return False
        return bool(self.station is not None and self.station.ready(simulation_time))

    def step(self, dt, simulation_time, parked=None):
        try:
            self._pump(simulation_time, parked)
            if not self.dispatched:
                if not self.preparation.ready(simulation_time):
                    return Result('running','prepare_station')
                self.graph.start(self.operation,self.job)
                self.dispatched=True
            # A carton-verification (capture) failure must NOT stop the bridge or the mission:
            # the bridge keeps publishing so the host ESDF builds; the perception gate is just
            # not satisfied yet, and the stage timeout still bounds a genuinely stuck capture.
            # The capture triggers a full-res render that can take > 1 s wall; bracket it with a
            # heartbeat before and after so the ack's wall heartbeat gap stays ~the render duration.
            # Only the capture() CALL is soft here. The integrity of a RETURNED packet is
            # fail-closed and lives OUTSIDE this try: a stale/repeated timestamp or a non-sensor
            # (fixture) provenance is an integrity violation that must stop the mission, never
            # degrade to 'running' -- those raises propagate to the outer handler which holds+fails.
            if self.station is not None:
                self.station.heartbeat()
            try:
                packet = self.capture(simulation_time,self.source,self.destination)
            except Exception as cap_exc:
                packet = None
                self.latest = Result('running','observe',str(cap_exc))
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
            if self.station is not None:
                self.station.heartbeat()           # close the wall gap right after the capture render
            reply = self.graph.poll(simulation_time)
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
