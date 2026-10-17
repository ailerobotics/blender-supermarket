"""Native JSON ROS transport for small simulation control-plane messages."""
import json
import time

from twin.restock_manipulation.ros_adapter import require_sim_domain

# Wall seconds between station heartbeats. Must stay well under the ack's 1 s wall heartbeat
# timeout; at the render-paced mission tick (~0.5-0.7 s wall) this sends on every tick.
_HEARTBEAT_WALL_S = 0.3
# Bounded re-prepare attempts: a reply of 'idle'/'failed' for the current epoch means the ack lost
# or faulted it; re-prepare under a fresh epoch a few times, then surface the reason.
_MAX_RESTARTS = 2
# Ack-freshness window (stage-clock seconds) for now - ack.stamp. The host relay stamps acks on
# world_current_time, which LEADS the mission stage clock by a step+, and the skew grows under heavy
# render pacing during a long contact descent (run-154/160 'stale parked-station' when the lead
# exceeded the old tight [0, 0.5]). Tolerate the lead and a wider lag; the 10 s heartbeat-unavailable
# guard still catches a genuinely dead host.
_ACK_LEAD = -1.0
_ACK_LAG = 1.0
# WALL-clock guard: fault if no station ack has arrived within this many wall seconds (genuinely dead
# host). Real-rig default 10 s. Under the sim's render-paced slow place swing a single accepted-command
# gap legitimately exceeds 10 s (run-231 tripped it mid-place), so the sim profile raises it with
# margin via StationClient(heartbeat_s=...); the 30 s /sensors wall backstop remains the one hard catch.
_HEARTBEAT_UNAVAILABLE_S = 10.0


class JsonChannel:
    def __init__(self, *, root, publish_topic, subscribe_topic, host_control=None):
        require_sim_domain()
        import omni.graph.core as og
        import omni.usd
        if omni.usd.get_context().get_stage().GetPrimAtPath(root):
            raise ValueError('refusing to overwrite native ROS channel')
        self.og,self.root=og,root
        # Both directions go through the host-DDS subprocess when host_control is set: the Isaac
        # OmniGraph ROS2 publisher is invisible to the host (run-57/59: command pub=0) AND the
        # OmniGraph subscriber delivers station/status too slowly (run-71/73: the StationClient got
        # the ack reply only after it had faulted). The subprocess publishes station/command and
        # forwards station/status back over the socket at host-side latency; read() uses it.
        self.host_control, self.publish_topic, self.subscribe_topic = host_control, publish_topic, subscribe_topic
        directions = (('send',publish_topic,'ROS2Publisher'),
                      ('receive',subscribe_topic,'ROS2Subscriber'))
        if host_control is not None:
            directions = ()  # publish + receive both out-of-Isaac via the control subprocess
        for direction,topic,kind in directions:
            key=og.Controller.Keys
            graph,_,_,_=og.Controller.edit({'graph_path':root+'/'+direction,'evaluator_name':'execution'}, {
                key.CREATE_NODES:[('tick','omni.graph.action.OnImpulseEvent'),
                    ('context','isaacsim.ros2.bridge.ROS2Context'),('wire','isaacsim.ros2.bridge.'+kind)],
                key.SET_VALUES:[('context.inputs:useDomainIDEnvVar',False),('context.inputs:domain_id',88),
                    ('wire.inputs:topicName',topic),('wire.inputs:messagePackage','std_msgs'),
                    ('wire.inputs:messageSubfolder','msg'),('wire.inputs:messageName','String')],
                key.CONNECT:[('tick.outputs:execOut','wire.inputs:execIn'),
                             ('context.outputs:context','wire.inputs:context')]})
            og.Controller.evaluate_sync(graph)

    def _pulse(self,direction):
        path=self.root+'/'+direction
        self.og.Controller.attribute(path+'/tick.state:enableImpulse').set(True)
        self.og.Controller.evaluate_sync(path)

    def publish(self,value):
        payload=json.dumps(value,allow_nan=False)
        if self.host_control is not None:
            self.host_control.publish(self.publish_topic,payload)  # host-DDS subprocess
            return
        self.og.Controller.attribute(self.root+'/send/wire.inputs:data').set(payload)
        self._pulse('send')

    def read(self):
        if self.host_control is not None:
            raw=self.host_control.read(self.subscribe_topic)   # forwarded by the control subprocess
            return json.loads(raw) if raw else None
        self._pulse('receive')
        raw=self.og.Controller.attribute(self.root+'/receive/wire.outputs:data').get()
        return json.loads(raw) if raw else None


class StationClient:
    # CONTRACT: station/command stamps (prepare, heartbeat, stop) are /clock (Isaac stage time),
    # never mission time. The host ack validates its 0.5 s freshness guard against /clock, while the
    # mission's Packet.simulation_time is zeroed at the first mission action -- two epochs. The
    # stage_clock seam reads the same stage-time source native_transport publishes on /clock, at the
    # moment of publish, so both sides share one clock domain and the ack's bound stays untouched.
    def __init__(self, channel, *, wall_clock=time.monotonic, stage_clock=None, publish_lag=0.,
                 heartbeat_s=_HEARTBEAT_UNAVAILABLE_S):
        self.channel,self.wall_clock=channel,wall_clock
        self.stage_clock=stage_clock
        # Wall heartbeat-unavailable bound (real-rig 10 s; sim profile raises it for the slow place).
        self.heartbeat_s=float(heartbeat_s)
        # Stamp commands a touch in the PAST (one /clock value the ack is sure to already hold):
        # /clock and station/command are separate topics, and the small fast command can reach the
        # ack before the matching /clock update, leaving the ack's now() ~one step BEHIND the stamp
        # -- which the ack's strict `0 <= now - stamp` lower bound rejects. publish_lag (a few clock
        # periods, << the 0.5 s window) absorbs that delivery skew while staying in the /clock domain.
        self.publish_lag=float(publish_lag)
        # Heartbeat cadence is WALL-paced, not stage-paced: the ack faults if no accepted command
        # arrives within 1 s WALL (station process/parked-heartbeat guard), but stage time advances
        # slowly under render-heavy stepping (~0.1 stage-s was ~4 s wall -> the ack timed out). Send
        # a heartbeat whenever >= HEARTBEAT_WALL_S of wall time has passed (every mission tick at the
        # render pace, comfortably under the ack's 1 s), independent of how slowly stage advances.
        self.last_wall_send=-1.
        self.epoch=0
        self.sequence=0
        # Send exactly one prepare (seq 0) per epoch, then heartbeats with a strictly increasing
        # sequence regardless of whether a reply has arrived -- the ack accepts heartbeats as soon as
        # it has accepted the prepare, and answers a DUPLICATE prepare with 'ignored' (seq<=last),
        # which previously stalled the sequence and tripped the ack's 1 s wall heartbeat guard.
        self._prepared=False
        self._restarts=0
        self.active=False
        self.latest=None
        self.last_send=-1.
        self.last_reply=None
        self.now=0.

    def begin(self,camera_frame,now):
        if self.active:
            raise RuntimeError('stop previous parked station before preparing another')
        self.epoch+=1
        self.sequence=0
        self._prepared=False
        self._restarts=0
        self.camera_frame=camera_frame
        self.active=True
        self.latest=None
        self.started=self.wall_clock()
        self.last_send=-1.
        self.last_reply=None
        self.tick(now,parked=True)
        return self

    def tick(self,now,*,parked):
        # Stamp + validate in the /clock (stage-time) domain, not the mission clock the caller
        # passes, so the host ack's /clock-validated freshness guard sees one epoch.
        if self.stage_clock is not None:
            now=self.stage_clock()
        self.now=now
        reply=self.channel.read()
        if reply and reply.get('epoch')==self.epoch and reply!=self.latest:
            # Freshness window on the ack stamp vs the mission stage clock. The original tight
            # [0, 0.5] s rejected the ack the instant the host /clock drifted AHEAD of the mission
            # stage clock (run-154/160: during the long render-paced contact descent the host led by
            # ~0.2 s, delta<0 -> 'stale'). The host relay stamps on world_current_time, which leads the
            # mission stage clock by a step+, and the skew grows under heavy render pacing. Tolerate the
            # lead and a wider lag (both << the 10 s heartbeat-unavailable guard that still catches a
            # genuinely dead host). Log the delta so a real stall is still attributable.
            delta=now-reply.get('stamp',-1)
            if not _ACK_LEAD<=delta<=_ACK_LAG:
                print(f'STATION_ACK_STALE delta={delta:.3f} now={now:.3f} '
                      f'stamp={reply.get("stamp")} window=[{_ACK_LEAD},{_ACK_LAG}]',flush=True)
                raise ValueError('stale parked-station acknowledgement')
            self.latest=reply
            self.last_reply=self.wall_clock()
        if not self.active:
            return
        if not parked:
            self.stop()
            raise ValueError('station lost measured parking')
        # A reply of 'idle'/'failed' for the current epoch means the ack lost or faulted this epoch:
        # re-prepare under a fresh epoch (bounded), rather than re-sending the same-epoch prepare
        # (which the ack ignores) or failing outright on a transient.
        if self.latest and self.latest.get('state') in ('idle','failed'):
            self._restarts+=1
            if self._restarts>_MAX_RESTARTS:
                raise RuntimeError(self.latest.get('reason') or 'host station failed')
            self.epoch+=1
            self.sequence=0
            self._prepared=False
            self.latest=None
        if self.wall_clock()-(self.last_reply or self.started)>self.heartbeat_s:
            raise RuntimeError('station supervisor heartbeat unavailable')
        wall=self.wall_clock()
        if self.last_send<0 or wall-self.last_wall_send>=_HEARTBEAT_WALL_S:
            # One prepare (seq 0) per epoch; then heartbeats with a strictly increasing sequence,
            # on the wall-paced cadence, regardless of whether a reply has arrived yet.
            if not self._prepared:
                action,seq='prepare',0
                self._prepared=True
            else:
                action,seq='heartbeat',self.sequence
            self.channel.publish(dict(action=action,epoch=self.epoch,sequence=seq,
                stamp=now-self.publish_lag,parked=True,camera_frame=self.camera_frame))
            self.sequence+=1
            self.last_send=now
            self.last_wall_send=wall

    def heartbeat(self):
        """Force one extra heartbeat NOW, bypassing the wall cadence gate, to bracket a step the
        caller knows is long (a capture render tick): the ack's wall heartbeat guard measures real
        wall time, and the StationClient cannot send mid-tick, so a heartbeat immediately before and
        after the long step keeps the accepted-command wall gap to ~the step duration. The sim does
        not advance between ticks, so the extra heartbeat is cheap and honest. No-op until the active
        epoch has been prepared."""
        if not self.active or not self._prepared:
            return
        now=self.stage_clock() if self.stage_clock is not None else self.now
        self.channel.publish(dict(action='heartbeat',epoch=self.epoch,sequence=self.sequence,
            stamp=now-self.publish_lag,parked=True,camera_frame=self.camera_frame))
        self.sequence+=1
        self.last_send=now
        self.last_wall_send=self.wall_clock()

    def ready(self,now):
        if self.stage_clock is not None:
            now=self.stage_clock()          # freshness in the /clock domain (matches the ack)
        reply=self.latest
        return bool(self.active and reply and reply.get('state')=='ready'
                    and 0<=now-reply.get('stamp',-1)<=.5
                    and 0<=now-reply.get('collision_time',-1)<=.5)

    @property
    def collision_time(self):
        return self.latest['collision_time'] if self.ready(self.now) else -1.

    def stop(self):
        if self.active:
            self.sequence+=1
            self.channel.publish(dict(action='stop',epoch=self.epoch,sequence=self.sequence,
                                      stamp=self.now-self.publish_lag,parked=True))
        self.active=False
        self.latest=None
