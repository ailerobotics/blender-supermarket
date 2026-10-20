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


def _mission_wrapper_chain():
    # The REAL mission wrapper chain: ManipulationBridge -> HostManipulationRuntime -> RosManipulationRuntime.
    from twin.restock_manipulation.isaac_factory import HostManipulationRuntime
    from twin.restock.runtime import ManipulationBridge
    stops = []
    station = NS(stop=lambda: stops.append('stop'))
    feedback = NS(collision_time=1.0, parked=lambda: True, set_parked=lambda v: None)
    graph = NS(start=lambda *a: None, publish_sensors=lambda s: None, poll=lambda *a: None, close=lambda: None)
    inner = RosManipulationRuntime(graph, lambda *a: None, prepare_station=lambda *a: None,
                                   station=station, feedback=feedback)
    host = HostManipulationRuntime(runtime=inner, sensors=NS(), motion=NS(), profiles={})
    bridge = ManipulationBridge(host, owns_sensor_io=True, prepare_station=lambda *a: None)
    return bridge, host, inner, stops


def test_mission_wrapper_chain_resolves_every_probed_method_fm400():
    # fm400: every method the ManipulationBridge/RuntimeAdapter probes (hasattr or optional-call) MUST
    # resolve on the real mission wrapper chain, failing BY NAME if a delegation is dropped -- so the
    # run-310 class of silent no-op (pause_station missing on HostManipulationRuntime) cannot recur.
    bridge, host, inner, stops = _mission_wrapper_chain()
    PROBED = ('start', 'warm', 'step', 'stop', 'pause_station', 'set_base_repositioned')
    for name in PROBED:
        assert hasattr(host, name), f'HostManipulationRuntime missing {name!r} (silent no-op risk)'
        assert callable(getattr(host, name)), f'{name!r} not callable on HostManipulationRuntime'
    assert host.cart_location == 'rollwagen'              # attr the bridge reads at construction
    # getattr-probed ATTRS the adapter reads off the controller (reach-stall watchdog); must resolve
    # to the inner runtime, not silently default to None (the same silent-no-op family).
    assert host.last_sensor_time == inner.last_sensor_time
    # and the two reposition methods actually REACH the inner RosManipulationRuntime through the bridge:
    bridge.pause_station(1.0)
    assert inner._station_started is False and stops == ['stop']   # station cleanly stopped at drive-start
    bridge.set_base_repositioned()
    assert inner._base_repositioned is True                        # arrival latched for the next host packet


def test_set_base_repositioned_stamps_published_sensor_packet_fm400():
    # fm400 handshake, Isaac side: once the adapter signals arrival, the next published sensor packet
    # carries base_repositioned=True across the ROS boundary (the host controller reads it from Sensors).
    from twin.restock_manipulation.types import Sensors, Observation
    published = []
    def capture(simulation_time, *a):                     # fresh, advancing-stamp sensor packet
        obs = Observation('o', 'sensor', simulation_time, 'sku', 'rollwagen',
                          (0., 0., 0., 0., 0., 0., 1.), .99)
        return Sensors(simulation_time, (obs,), None, True, False, simulation_time, simulation_time,
                       (0.,) * 6, ('rollwagen',), simulation_time)
    graph = NS(start=lambda *a: None, publish_sensors=lambda s: published.append(s),
               poll=lambda *a: None, close=lambda: None)
    rt = RosManipulationRuntime(graph, capture, prepare_station=lambda *a: None)
    rt.start('collect', dict(job_id='j', source_bin='back', shelf_id='shelf'))
    rt.step(.01, 1.0)
    assert published[-1].base_repositioned is False      # before arrival: byte-identical (default False)
    rt.set_base_repositioned()
    rt.step(.01, 1.1)
    assert published[-1].base_repositioned is True        # after arrival: carried to the host on the packet
