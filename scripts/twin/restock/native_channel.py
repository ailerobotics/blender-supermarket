"""Native JSON ROS transport for small simulation control-plane messages."""
import os
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
# With the sim-clock primary guard below wired, this wall bound is only the frozen-sim backstop.
_HEARTBEAT_UNAVAILABLE_S = max(2.0, min(120.0, float(os.environ.get('RESTOCK_HEARTBEAT_UNAVAILABLE_S', 10.0))))  # fm362
# SIM-clock (stage) supervisor-liveness bound: fault if stage time has advanced this far past the last
# accepted ack with no fresh ack. This is the PRIMARY liveness check -- it rides the render-paced /clock
# so a legitimately slow full-res render tick cannot false-trip it (ds-015 got no station reply for 10 s
# WALL during one slow warm tick and the pure-wall guard fired), while a genuinely dead/wedged station
# still trips within a few stage seconds because the mission keeps stepping /clock. The wall bound above
# stays only as a generous FROZEN-sim backstop (stage time can't advance to trip this then). Engaged only
# when a stage_clock is wired (the sim profile); real-rig (stage_clock=None) keeps the wall bound alone.
# See freshness-sim-clock-not-wall.
_STAGE_UNAVAILABLE_S = max(1.0, min(120.0, float(os.environ.get('RESTOCK_STAGE_UNAVAILABLE_S', 3.0))))  # fm362


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
                 heartbeat_s=_HEARTBEAT_UNAVAILABLE_S, stage_unavailable_s=_STAGE_UNAVAILABLE_S):
        self.channel,self.wall_clock=channel,wall_clock
        self.stage_clock=stage_clock
        # Wall heartbeat-unavailable bound (real-rig 10 s; sim profile raises it for the slow place).
        # With a stage_clock wired it is the frozen-sim backstop behind the sim-clock primary guard.
        # fm366: the sim-only env WINS over any caller-passed value (isaac_factory passes config, which
        # shadowed the env widen), unset => the passed value (byte-identical real-rig). Re-read + clamp
        # the env at ctor time (robust whether the env was set before or after import).
        self.heartbeat_s=(max(2.0,min(120.0,float(os.environ['RESTOCK_HEARTBEAT_UNAVAILABLE_S'])))
                          if 'RESTOCK_HEARTBEAT_UNAVAILABLE_S' in os.environ else float(heartbeat_s))
        # Sim-clock (stage) supervisor-liveness bound (primary when stage_clock is wired).
        self.stage_unavailable_s=(max(1.0,min(120.0,float(os.environ['RESTOCK_STAGE_UNAVAILABLE_S'])))
                                  if 'RESTOCK_STAGE_UNAVAILABLE_S' in os.environ else float(stage_unavailable_s))
        # Stage time of the last accepted fresh ack, and of the epoch's first tick; the sim-clock
        # liveness gap is measured against whichever is later. None until the stage clock is first read.
        self._last_ack_stage=None
        self._begin_stage=None
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
        # Keep sending the prepare (seq 0) every cadence tick UNTIL the host acknowledges this epoch
        # (a reply carrying epoch == self.epoch), then switch to heartbeats with a strictly increasing
        # sequence. The initial prepare can be DROPPED before pub/sub discovery completes (the station
        # subprocess's status topic had just been created); the old code sent exactly one prepare then
        # advanced to heartbeats, so a dropped prepare left the host idle at epoch 0 while the client
        # heart-beat at epoch 1 -- the host faulted 'no active matching epoch' on the first heartbeat
        # and the client IGNORED every epoch-0 reply (epoch mismatch), a permanent cross-epoch wedge
        # (hl-001: snapshot=None, collision_time=-1 the whole warm). Re-sending seq-0 prepares is safe:
        # the host accepts it while still behind (epoch 0 -> opens the epoch) and answers a duplicate at
        # the accepted epoch with 'ignored' (seq<=last), which does NOT advance its heartbeat clock --
        # so we stop re-preparing the instant a reply for our epoch lands. _prepared means ACKNOWLEDGED.
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
        self._last_ack_stage=None
        self._begin_stage=None
        self.tick(now,parked=True)
        return self

    def tick(self,now,*,parked):
        # Stamp + validate in the /clock (stage-time) domain, not the mission clock the caller
        # passes, so the host ack's /clock-validated freshness guard sees one epoch.
        if self.stage_clock is not None:
            now=self.stage_clock()
        self.now=now
        if self._begin_stage is None:
            self._begin_stage=now
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
            self._last_ack_stage=now
            # A reply carrying our epoch proves the host accepted our prepare: stop re-preparing and
            # switch to heartbeats. (A dropped initial prepare leaves the host behind at a lower epoch,
            # so this never latches until the prepare actually lands.)
            self._prepared=True
            # A fresh healthy (warming/ready) reply is progress: reset the bounded re-prepare budget
            # so transient faults recovered across a long place/verify do not accumulate toward the
            # cap and surface a stale reason on a later, unrelated transient.
            if reply.get('state') in ('warming','ready'):
                self._restarts=0
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
        # Supervisor liveness. PRIMARY is the sim/stage clock (immune to render pacing): fault if stage
        # time has advanced past the last fresh ack (or the epoch's first tick) by more than the stage
        # bound. The WALL bound stays only as a generous frozen-sim backstop. Engaged on the stage clock
        # only when it is wired (sim profile); real-rig (stage_clock=None) keeps the wall bound alone.
        if self.stage_clock is not None:
            ref_stage=self._last_ack_stage if self._last_ack_stage is not None else self._begin_stage
            if ref_stage is not None and now-ref_stage>self.stage_unavailable_s:
                raise RuntimeError(self._unavailable_reason(f'stage gap {now-ref_stage:.1f}s'))
        if self.wall_clock()-(self.last_reply or self.started)>self.heartbeat_s:
            raise RuntimeError(self._unavailable_reason(
                f'wall gap {self.wall_clock()-(self.last_reply or self.started):.1f}s'))
        wall=self.wall_clock()
        if self.last_send<0 or wall-self.last_wall_send>=_HEARTBEAT_WALL_S:
            # Re-send the prepare (seq 0) each cadence tick until the host acknowledges this epoch
            # (_prepared is set on the first reply carrying our epoch); then heartbeats with a strictly
            # increasing sequence. A dropped prepare no longer wedges the handshake -- the next tick
            # re-prepares, and a duplicate once accepted is answered 'ignored' (seq<=last, harmless).
            if not self._prepared:
                action,seq='prepare',0
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

    def _unavailable_reason(self,gap_desc):
        """Honest supervisor-unavailable reason. When the station had been HEALTHY (ready/warming)
        and then went silent, the cause is almost never the station itself -- it is the whole host
        stack being torn down because a SIBLING host node died (run-268: move_group SIGABRT'd on a
        cuMotion double-free, the launch SIGINT'd every host process, the station's acks froze, and
        this guard -- firing on the resulting staleness -- was misread as the station's fault). Name
        that so a host-stack teardown is not masked as a station lifecycle bug. Keeps the
        'station supervisor heartbeat unavailable' substring for existing log/reason matching."""
        last=(self.latest or {}).get('state')
        if last in ('ready','warming'):
            return ('station supervisor heartbeat unavailable: host stack went silent after last '
                    f'state={last} ({gap_desc}); a sibling host process likely died -- inspect '
                    'host-stack.log for an earlier process exit (e.g. move_group)')
        return f'station supervisor heartbeat unavailable ({gap_desc}; last state={last})'

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
