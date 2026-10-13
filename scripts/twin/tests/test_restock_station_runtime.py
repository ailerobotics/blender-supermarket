from types import SimpleNamespace

import pytest

from twin.restock.native_channel import StationClient
from twin.restock.station_runtime import StationRuntime


class Channel:
    def __init__(self): self.reply=None;self.sent=[]
    def read(self): return self.reply
    def publish(self,item): self.sent.append(item)


def runtime():
    channel=Channel();selected=[];epochs=[];captures=[]
    state=StationClient(channel,wall_clock=lambda:0.)
    feedback=SimpleNamespace(collision_time=-1.)
    result=StationRuntime(station=state,stream=SimpleNamespace(select=selected.append),
        graph=SimpleNamespace(check_health=lambda:None),hosts=SimpleNamespace(check=lambda:None),
        feedback=feedback,contact=object(),sensors=SimpleNamespace(capture=lambda *args:captures.append(args)),
        mapper_cameras={'collect':'source','transfer':'shelf'},invalidate_contexts=epochs.append)
    return result,channel,selected,epochs,captures


def test_prepare_requires_measured_parking_then_real_map_ack():
    result,channel,selected,epochs,captures=runtime()
    with pytest.raises(ValueError,match='parking'):
        result.prepare('collect',{})
    result.after_step(1.,True)
    preparation=result.prepare('collect',{})
    assert epochs==[1] and selected[-1]=='source'
    assert not preparation.ready(1.)
    assert result.capture(1.,'source','rollwagen') is None
    assert not captures
    channel.reply=dict(epoch=1,stamp=1.1,state='ready',collision_time=1.05)
    result.after_step(1.1,True)
    result.capture(1.1,'source','rollwagen')
    assert captures and result.feedback.collision_time==1.05


def test_travel_invalidates_station_but_verification_keeps_epoch():
    result,channel,selected,epochs,captures=runtime()
    result.after_step(1.,True)
    result.prepare('transfer',{})
    job=SimpleNamespace(shelf_id='shelf',sku='juice',quantity=1)
    result.begin_action('verify',job)
    assert result.station.active and result.verifier.active
    result.begin_action('return',job)
    assert not result.station.active and not result.verifier.active
    assert selected[-1] is None and result.feedback.collision_time==-1.
    assert channel.sent[-1]['action']=='stop'


def test_motion_and_global_stop_are_checked_outside_manipulation():
    result,channel,selected,epochs,captures=runtime()
    result.after_step(1.,True);result.prepare('collect',{})
    with pytest.raises(ValueError,match='parking'):
        result.after_step(1.1,False)
    def emergency(): raise RuntimeError('host emergency stop')
    result.graph.check_health=emergency
    with pytest.raises(RuntimeError,match='emergency stop'):
        result.after_step(1.2,True)


def test_station_waits_for_acquired_pair_behind_map_ack():
    result,channel,selected,epochs,captures=runtime()
    queue_epochs=[];selections=[];received=[]
    queues=SimpleNamespace(begin_epoch=queue_epochs.append,
        append=lambda *a,**k:received.append((a,k)),
        select=lambda *a,**k:selections.append(k) or False)
    result.bind_frames(queues,lambda now:[('source',{'real_fixture':True},'transform',now)])
    result.after_step(1.,True);result.prepare('collect',{})
    channel.reply=dict(epoch=1,stamp=1.1,state='ready',collision_time=1.05)
    result.after_step(1.1,True)
    assert queue_epochs==[1] and len(received)==1
    assert result.capture(1.1,'source','rollwagen') is None and not captures
    assert selections==[dict(now=1.1,collision_time=1.05)]
    queues.select=lambda *a,**k:True
    result.capture(1.1,'source','rollwagen')
    assert len(captures)==1
