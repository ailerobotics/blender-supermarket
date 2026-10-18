import pytest

from twin.restock.station import StationState
from twin.restock.native_channel import StationClient


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


class Channel:
    def __init__(self): self.messages=[];self.reply=None
    def publish(self,value): self.messages.append(value)
    def read(self): return self.reply


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
