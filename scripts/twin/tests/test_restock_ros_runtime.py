from pathlib import Path
import sys
from types import SimpleNamespace as NS

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.restock.ros_manipulation import RosManipulationRuntime


def make_runtime(packet):
    calls=[]
    graph=NS(start=lambda *a:calls.append('start'),
             publish_sensors=lambda s:calls.append('sensor'),
             poll=lambda *a:None,close=lambda:calls.append('hold'))
    runtime=RosManipulationRuntime(graph,lambda *a:packet,
                                  prepare_station=lambda *a:calls.append('calibrate'))
    runtime.start('collect',dict(job_id='j',source_bin='back',shelf_id='shelf'))
    return runtime,calls,graph


def test_graph_polled_without_new_camera_frame_and_does_not_replay_controls():
    runtime,calls,graph=make_runtime(None)
    graph.poll=lambda *a:dict(status='succeeded',phase='verify',controls=[{'kind':'gripper'}],
                          evidence=[{'source':'sensor'}])
    result=runtime.step(.01,1.)
    assert result.status=='succeeded' and result.controls==()
    assert calls==['calibrate','start']


def test_fixture_sensor_refused_and_seal_preserved_by_hold():
    packet=NS(simulation_time=1.,observations=[NS(source='test_fixture')],target=None)
    runtime,calls,_=make_runtime(packet)
    assert runtime.step(.01,1.).status=='failed'
    assert calls==['calibrate','start','hold']


def test_repeated_camera_timestamp_fails_closed():
    packet=NS(simulation_time=1.,observations=[NS(source='sensor')],target=None)
    runtime,calls,_=make_runtime(packet)
    assert runtime.step(.01,1.).status=='running'
    assert runtime.step(.01,1.01).status=='failed'
    assert calls.count('sensor')==1 and calls[-1]=='hold'
