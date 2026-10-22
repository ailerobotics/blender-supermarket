import pytest

from twin.restock.station import StationState, _MAX_RECOVERIES
from twin.restock.native_channel import StationClient, StationAckStreamStalled


def message(action='prepare',epoch=1,sequence=0,stamp=1.,parked=True):
    return dict(action=action,epoch=epoch,sequence=sequence,stamp=stamp,parked=parked,
                camera_frame='restock_camera_source')


def test_only_current_incorporated_frame_can_make_station_ready():
    state=StationState(wall_clock=lambda:0.)
    state.accept(message(),1.)
    assert state.snapshot(1.1,endpoints_ready=True,process_alive=True)['state']=='warming'
    with pytest.raises(ValueError):
        state.acknowledge(.99,1)
    state.acknowledge(1.1,1)
    assert state.snapshot(1.2,endpoints_ready=True,process_alive=True)['state']=='ready'
    assert state.snapshot(1.2,endpoints_ready=False,process_alive=True)['state']=='warming'


def test_restart_discards_previous_map_and_reordered_ack():
    state=StationState(wall_clock=lambda:0.)
    state.accept(message(),1.)
    state.acknowledge(1.1,1)
    state.accept(message('stop',sequence=1,stamp=1.2),1.2)
    state.accept(message(epoch=2,stamp=2.),2.)
    assert state.collision_time==-1.
    with pytest.raises(ValueError):
        state.acknowledge(2.1,1)
    assert state.accept(message('heartbeat',epoch=1,sequence=9,stamp=2.),2.)=='ignored'


@pytest.mark.parametrize('field,value',[('stamp',float('nan')),('parked',False),('epoch',True)])
def test_bad_prepare_rejected(field,value):
    packet=message();packet[field]=value
    with pytest.raises(ValueError):
        StationState().accept(packet,1.)


def test_process_death_and_sim_heartbeat_fail_closed():
    # Primary liveness: a dead host process, or a stalled SIM-clock heartbeat (>0.5 s), fault closed.
    dead=StationState(wall_clock=lambda:0.)
    dead.accept(message(),1.)
    assert dead.snapshot(1.1,endpoints_ready=True,process_alive=False)['state']=='failed'
    stale=StationState(wall_clock=lambda:0.)
    stale.accept(message(),1.)
    assert stale.snapshot(1.7,endpoints_ready=True,process_alive=True)['state']=='failed'  # 0.7 s sim gap


def test_wall_backstop_tolerates_render_pacing_but_catches_frozen_sim():
    # The wall-clock check is only a loose backstop for a FROZEN sim clock; a render-paced tick
    # (~1 s wall) must NOT false-trip it (regression: ds-011/013 faulted the transfer warm here).
    wall=[0.]
    state=StationState(wall_clock=lambda:wall[0])
    state.accept(message(),1.)
    wall[0]=1.1
    assert state.snapshot(1.1,endpoints_ready=True,process_alive=True)['state']=='warming'
    # A genuinely frozen sim clock (sim time not advancing) is still caught once the wall backstop trips.
    wall[0]=20.
    assert state.snapshot(1.1,endpoints_ready=True,process_alive=True)['state']=='failed'


def test_faulted_station_recovers_on_a_fresh_parked_epoch():
    # A station fault (dead mapper child / lost heartbeat) must NOT reject the mission client's
    # re-prepare forever (ds-015). A FRESH measured parked epoch clears the fault, counted, logged.
    state=StationState(wall_clock=lambda:0.)
    state.accept(message(),1.)
    assert state.snapshot(1.1,endpoints_ready=True,process_alive=False)['state']=='failed'
    assert state.fault
    # A heartbeat/stop against the faulted epoch is still refused -- only a fresh prepare recovers.
    with pytest.raises(RuntimeError,match='explicit recovery'):
        state.accept(message('heartbeat',epoch=1,sequence=1,stamp=1.2),1.2)
    # A fresh parked epoch recovers: fault cleared, recovery counted, warming again.
    assert state.accept(message(epoch=2,stamp=2.),2.)=='prepare'
    assert state.fault is None
    assert state.recoveries==1
    assert state.recovered_from=='station process or parked heartbeat lost'
    assert state.snapshot(2.1,endpoints_ready=True,process_alive=True)['state']=='warming'
    state.acknowledge(2.1,2)
    assert state.snapshot(2.2,endpoints_ready=True,process_alive=True)['state']=='ready'


def test_recovery_is_bounded_so_a_persistently_broken_mapper_surfaces():
    state=StationState(wall_clock=lambda:0.)
    state.accept(message(epoch=1,stamp=1.),1.)          # first clean epoch
    for ep in range(2,2+_MAX_RECOVERIES):
        # kill the mapper child -> fault, then recover with a fresh epoch.
        assert state.snapshot(float(ep)-0.5,endpoints_ready=True,process_alive=False)['state']=='failed'
        assert state.accept(message(epoch=ep,stamp=float(ep)),float(ep))=='prepare'
    assert state.recoveries==_MAX_RECOVERIES
    # Budget exhausted: fault once more, the next fresh prepare is refused honestly with the reason.
    nxt=2+_MAX_RECOVERIES
    assert state.snapshot(float(nxt)-0.5,endpoints_ready=True,process_alive=False)['state']=='failed'
    with pytest.raises(RuntimeError,match='recovery budget exhausted'):
        state.accept(message(epoch=nxt,stamp=float(nxt)),float(nxt))


def test_malformed_recovery_prepare_leaves_fault_latched():
    state=StationState(wall_clock=lambda:0.)
    state.accept(message(),1.)
    assert state.snapshot(1.1,endpoints_ready=True,process_alive=False)['state']=='failed'
    bad=message(epoch=2,stamp=2.);bad['camera_frame']='wrong_frame'
    with pytest.raises(ValueError):
        state.accept(bad,2.)
    assert state.fault and state.recoveries==0   # not counted; still faulted


class Channel:
    def __init__(self): self.messages=[];self.reply=None
    def publish(self,value): self.messages.append(value)
    def read(self): return self.reply


def test_client_supervisor_heartbeat_is_sim_clocked_not_wall():
    # PRIMARY liveness is the stage (sim) clock: a dead station (no fresh ack) trips in a few stage
    # seconds regardless of how slow wall time is, and a render-paced tick (huge wall gap, fresh acks)
    # must NOT false-trip (ds-015). Wall is only a frozen-sim backstop.
    wall=[0.];stage=[1.]
    channel=Channel()
    client=StationClient(channel,wall_clock=lambda:wall[0],stage_clock=lambda:stage[0],
                         heartbeat_s=1000.,stage_unavailable_s=3.0)
    client.begin('restock_camera_source',1.)
    # Fresh acks keep arriving even though WALL jumps far past the old 10 s guard: no false trip.
    for s in (1.1,1.5,2.0,2.6):
        stage[0]=s; wall[0]+=12.
        channel.reply=dict(epoch=1,stamp=s,state='warming',collision_time=-1.)
        client.tick(0.,parked=True)   # now overridden by stage_clock
    # Station goes dead: no new ack. Stage advances (mission keeps stepping) -> sim guard fires.
    channel.reply=None
    stage[0]=2.6+2.9
    client.tick(0.,parked=True)        # within 3 s of last ack (2.6): tolerated
    stage[0]=2.6+3.1
    with pytest.raises(RuntimeError,match='heartbeat unavailable'):
        client.tick(0.,parked=True)    # > 3 s stage gap with no ack: dead station


class CoupledChannel:
    """Couple a StationClient to a REAL StationState host through one in-memory hop, dropping the
    first ``drop`` client->host messages (pub/sub not yet matched when the status topic is fresh)."""
    def __init__(self, host, host_now, *, drop=0):
        self.host, self.host_now, self.drop = host, host_now, drop
        self._status = None
    def publish(self, cmd):
        if self.drop > 0:
            self.drop -= 1
            return                          # lost before discovery
        try:
            self.host.accept(cmd, self.host_now())
        except Exception as exc:            # a rejected command faults the host, as station_host does
            self.host.fault = str(exc)
        self._status = self.host.snapshot(self.host_now(), endpoints_ready=True, process_alive=True)
    def read(self):
        return self._status


def test_dropped_initial_prepare_does_not_wedge_the_handshake():
    # hl-001: the first prepare was dropped before discovery; the old client sent exactly one prepare
    # then heart-beat, so the host stayed idle at epoch 0, faulted 'no active matching epoch' on the
    # first heartbeat, and the client ignored every epoch-0 reply -> permanent cross-epoch wedge.
    # Re-preparing until acknowledged (plus bounded host recovery) must break it.
    clock = [1.0]; wallc = [0.0]                               # render pacing: wall fast, stage slow
    host = StationState(wall_clock=lambda: wallc[0])
    channel = CoupledChannel(host, lambda: clock[0], drop=1)   # drop the first (begin's) prepare
    client = StationClient(channel, wall_clock=lambda: wallc[0], stage_clock=lambda: clock[0])
    client.begin('restock_camera_source', clock[0])            # prepare #1 -> dropped
    for _ in range(6):
        clock[0] += 0.05; wallc[0] += 0.5                      # tiny sim step, big wall step per tick
        client.tick(clock[0], parked=True)
    # The host accepted a (re-sent) prepare and is on the client's epoch, warming -- not wedged.
    assert host.active and host.epoch == client.epoch
    assert client._prepared
    assert host.snapshot(clock[0], endpoints_ready=True, process_alive=True)['state'] == 'warming'
    # Once nvblox incorporates, the client sees a fresh collision_time (handshake fully live). A
    # couple of ticks carry the host's refreshed snapshot across the one-hop read/publish lag.
    host.acknowledge(clock[0], host.epoch)
    for _ in range(3):
        clock[0] += 0.05; wallc[0] += 0.5
        host.acknowledge(clock[0], host.epoch)        # nvblox keeps incorporating each warm tick
        client.tick(clock[0], parked=True)
    assert client.ready(clock[0]) and client.collision_time > 0


def test_supervisor_fault_names_host_stack_teardown_not_the_station():
    # run-268/collect: the station was HEALTHY (ready, collision_time advancing) until move_group
    # SIGABRT'd and the launch tore down the whole host stack, freezing the acks. This guard fires on
    # the resulting staleness -- its reason must name the host-stack teardown so the real cause
    # (a sibling host process death) is not masked as a station lifecycle bug.
    wall=[0.]; stage=[1.]
    channel=Channel()
    client=StationClient(channel,wall_clock=lambda:wall[0],stage_clock=lambda:stage[0],
                         heartbeat_s=1000.,stage_unavailable_s=3.0)
    client.begin('restock_camera_source',1.)
    stage[0]=1.1; wall[0]+=0.5
    channel.reply=dict(epoch=1,stamp=1.1,state='ready',collision_time=1.05)
    client.tick(0.,parked=True)
    assert client.ready(0.)                              # healthy, collision map fresh
    # Host stack torn down: acks freeze (same stale reply), stage keeps advancing past the bound.
    stage[0]=1.1+3.1
    with pytest.raises(RuntimeError) as exc:
        client.tick(0.,parked=True)
    assert 'heartbeat unavailable' in str(exc.value)     # substring kept for reason/log matching
    assert 'host stack went silent' in str(exc.value) and 'move_group' in str(exc.value)
    assert 'state=ready' in str(exc.value)


def test_client_resets_restart_budget_on_healthy_progress():
    channel=Channel()
    client=StationClient(channel,wall_clock=lambda:0.)
    client.begin('restock_camera_source',1.)
    # Two failed epochs bump the restart counter toward the cap.
    for ep in (1,2):
        channel.reply=dict(epoch=ep,stamp=1.,state='failed',collision_time=-1.,reason='x')
        client.tick(1.1,parked=True)
    assert client._restarts==2
    # A healthy reply is progress -> budget resets, so a later transient does not surface stale.
    channel.reply=dict(epoch=client.epoch,stamp=1.1,state='warming',collision_time=-1.)
    client.tick(1.2,parked=True)
    assert client._restarts==0


def test_client_does_not_promote_heartbeat_to_mapping_readiness():
    channel=Channel()
    client=StationClient(channel,wall_clock=lambda:0.)
    client.begin('restock_camera_source',1.)
    assert channel.messages[-1]['action']=='prepare'
    channel.reply=dict(epoch=1,stamp=1.1,state='warming',collision_time=-1.)
    client.tick(1.1,parked=True)
    assert not client.ready(1.1)
    assert client.collision_time==-1.
    channel.reply=dict(epoch=1,stamp=1.2,state='ready',collision_time=1.15)
    client.tick(1.2,parked=True)
    assert client.ready(1.2)
    assert client.collision_time==1.15
    with pytest.raises(ValueError,match='parking'):
        client.tick(1.21,parked=False)
    assert not client.active
    assert channel.messages[-1]['action']=='stop'


def test_client_rejects_stale_or_different_epoch_readiness():
    channel=Channel()
    client=StationClient(channel,wall_clock=lambda:0.)
    client.begin('restock_camera_source',1.)
    channel.reply=dict(epoch=2,stamp=1.,state='ready',collision_time=1.)
    client.tick(1.1,parked=True)
    assert not client.ready(1.1)
    channel.reply=dict(epoch=1,stamp=0.,state='ready',collision_time=0.)
    with pytest.raises(ValueError,match='stale'):
        client.tick(1.1,parked=True)


# --- Frozen station-ack publisher watchdog (run-335) -------------------------------------------
# run-335 (COLD collect): the station supervisor stopped publishing at stage ~0.9 s (stamp frozen at
# 0.900000046, correlated with a sibling host IPC teardown). The host-DDS relay kept serving the last
# cached ack, so the consumer saw a FROZEN stamp. After the first collect attempt's stop() nulled
# self.latest, the frozen ack (first seen already stale) hit the generic 'stale parked-station
# acknowledgement' path, which the warm() retry swallowed every tick -- the staleness delta grew
# unbounded (delta 5.0 -> 8.2) until the opaque 30 s arm-command heartbeat backstop fired. The
# watchdog names the frozen-publisher cause and fails fast on a bounded SIM-time budget.

def test_frozen_ack_stamp_fails_fast_with_named_cause():
    # The run-335 path: a frozen ack first SEEN while already stale (self.latest reset / relay delivery
    # lag), so the stale-window check -- not the accepted-ack stage-gap guard -- is what fires. On the
    # first sighting it is only 'stale' (we cannot yet know it is frozen); a genuinely dead publisher is
    # named station_ack_stream_stalled once the SAME stamp persists past BOTH the stage budget AND the
    # wall backstop (sim-robust-watchdogs: 'stamps frozen AND process gone' -- the wall leg keeps a
    # render-lagged-but-alive publisher from false-tripping before it can recover). Here the stamp is
    # frozen FOREVER with real wall time piling up -> a true dead publisher -> fast fail.
    stage=[1.0]; wall=[0.]
    channel=Channel()
    client=StationClient(channel,wall_clock=lambda:wall[0],stage_clock=lambda:stage[0],
                         heartbeat_s=1000.,stage_unavailable_s=1000.,ack_frozen_s=2.0,ack_frozen_wall_s=10.0)
    client.begin('restock_camera_source',1.0)
    # Frozen ack delivered late: first sighting at stage 5.0 is out-of-window -> generic stale.
    stage[0]=5.0; wall[0]=1.0
    channel.reply=dict(epoch=1,stamp=0.9,state='ready',collision_time=0.85)
    with pytest.raises(ValueError,match='stale'):
        client.tick(0.,parked=True)
    # Same cached stamp, stage advanced past its budget but wall backstop NOT yet elapsed: still only
    # 'stale' -- an alive-but-render-lagged publisher is given real wall time to recover before death.
    stage[0]=5.0+2.1; wall[0]=1.0+3.0
    with pytest.raises(ValueError,match='stale'):
        client.tick(0.,parked=True)
    # Same cached stamp beyond BOTH the stage budget AND the wall backstop -> named fast fail.
    stage[0]=5.0+2.2; wall[0]=1.0+10.1
    with pytest.raises(StationAckStreamStalled) as exc:
        client.tick(0.,parked=True)
    assert 'station_ack_stream_stalled' in str(exc.value)
    assert 'publisher stopped' in str(exc.value)
    assert '0.9' in str(exc.value)


def test_frozen_ack_watchdog_tolerates_a_long_healthy_advancing_stream():
    # An ADVANCING stamp is never frozen, even across a warm far longer than the budget: no false trip.
    stage=[1.0]
    channel=Channel()
    client=StationClient(channel,wall_clock=lambda:0.,stage_clock=lambda:stage[0],
                         heartbeat_s=1000.,stage_unavailable_s=1000.,ack_frozen_s=2.0)
    client.begin('restock_camera_source',1.0)
    for s in (1.1,2.0,3.0,4.0,5.0,6.0,7.0):      # 6 s of stage time, >> the 2 s budget
        stage[0]=s
        channel.reply=dict(epoch=1,stamp=s,state='ready',collision_time=s-0.05)
        client.tick(0.,parked=True)              # fresh (delta~0), accepted, stamp advances
    assert client.ready(7.0)


def test_frozen_ack_watchdog_disengaged_on_real_rig_without_stage_clock():
    # Real-rig (stage_clock=None) keeps the generic wall path: a stale ack stays a ValueError 'stale',
    # never the sim-clock StationAckStreamStalled, so behaviour is byte-identical off the sim profile.
    channel=Channel()
    client=StationClient(channel,wall_clock=lambda:0.)      # no stage_clock
    client.begin('restock_camera_source',1.)
    channel.reply=dict(epoch=1,stamp=0.,state='ready',collision_time=0.)
    with pytest.raises(ValueError,match='stale'):
        client.tick(10.,parked=True)
    with pytest.raises(ValueError,match='stale'):           # persists -> still generic, never the stall
        client.tick(20.,parked=True)


def test_frozen_budget_env_overrides_passed_value(monkeypatch):
    # The sim-only env widen WINS over any caller-passed value (isaac_factory passes config), matching
    # the heartbeat_s / stage_unavailable_s bounds.
    monkeypatch.setenv('RESTOCK_STATION_ACK_FROZEN_S','7.5')
    client=StationClient(Channel(),wall_clock=lambda:0.,ack_frozen_s=2.0)
    assert client.ack_frozen_s==7.5


def test_accepted_then_frozen_still_uses_the_rich_host_stack_teardown_reason():
    # When the ack was ACCEPTED first and then froze (reply == self.latest), the stale-window branch is
    # skipped, so the existing stage-gap guard -- with its richer 'host stack went silent' reason --
    # still fires; the frozen-stamp watchdog does not steal that case.
    wall=[0.]; stage=[1.]
    channel=Channel()
    client=StationClient(channel,wall_clock=lambda:wall[0],stage_clock=lambda:stage[0],
                         heartbeat_s=1000.,stage_unavailable_s=3.0)   # ack_frozen_s default 2.0
    client.begin('restock_camera_source',1.)
    stage[0]=1.1; wall[0]+=0.5
    channel.reply=dict(epoch=1,stamp=1.1,state='ready',collision_time=1.05)
    client.tick(0.,parked=True)
    assert client.ready(0.)
    stage[0]=1.1+3.1
    with pytest.raises(RuntimeError) as exc:
        client.tick(0.,parked=True)
    assert 'heartbeat unavailable' in str(exc.value)
    assert not isinstance(exc.value, StationAckStreamStalled)


def test_warm_propagates_stall_but_swallows_other_transients():
    # ros_manipulation.warm() must re-raise the named stall (a dead publisher is not a recoverable warm
    # transient) so the stage fails fast, while every other warm transient stays swallowed (-> False).
    from twin.restock.ros_manipulation import RosManipulationRuntime

    class FakeFeedback:
        def __init__(self): self.collision_time=-1.; self._p=True
        def set_parked(self,v): self._p=v
        def parked(self): return self._p

    class FakeStation:
        latest=None; stage_clock=None; collision_time=-1.
        def __init__(self,exc): self.exc=exc
        def tick(self,now,*,parked): raise self.exc
        def ready(self,now): return False

    stalled=RosManipulationRuntime(None,None,prepare_station=lambda o,j:None,
        feedback=FakeFeedback(),
        station=FakeStation(StationAckStreamStalled('station_ack_stream_stalled: stamp frozen')))
    stalled._station_started=True
    with pytest.raises(StationAckStreamStalled):
        stalled.warm(1.0, parked=True)

    transient=RosManipulationRuntime(None,None,prepare_station=lambda o,j:None,
        feedback=FakeFeedback(),
        station=FakeStation(ValueError('stale parked-station acknowledgement')))
    transient._station_started=True
    assert transient.warm(1.0, parked=True) is False          # swallowed as before
    assert 'stale' in (transient.latest.reason or '')
