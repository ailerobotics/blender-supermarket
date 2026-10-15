import pytest
from scripts.twin.restock_manipulation.ros_payload import relative_pose
from scripts.twin.restock_manipulation.isaac_ros_graph import IsaacRosGraph


def test_payload_pose_uses_measured_tcp_translation_and_rotation():
    assert relative_pose((2,2,3,0,0,0,1),(1,2,3,0,0,1,0)) == pytest.approx(
        (-1,0,0,0,0,-1,0))
    with pytest.raises(ValueError,match='unit quaternion'):
        relative_pose((2,2,3,0,0,0,2),(1,2,3,0,0,1,0))


def test_native_bridge_fault_latches_without_consuming_late_host_controls():
    bridge = IsaacRosGraph.__new__(IsaacRosGraph)
    holds = []
    bridge.faulted, bridge.active = True, False
    bridge.hold = lambda: holds.append(True)
    # No graph exists: a fault must not even inspect stale incoming controls.
    assert bridge.poll()['status'] == 'failed'
    assert holds == [True]
    with pytest.raises(RuntimeError,match='recovery'):
        bridge.start('transfer',{'job_id':'next'})


def _silent_bridge():
    """A bridge whose host is alive but emitting no fresh status (empty /status attr)."""
    import time
    from types import SimpleNamespace as NS
    bridge = IsaacRosGraph.__new__(IsaacRosGraph)
    bridge.og = NS(Controller=NS(attribute=lambda path: NS(get=lambda: False if '/stop/' in path else '')))
    bridge.root = '/test'
    bridge.host_control = None
    bridge.faulted = False; bridge.active = True; bridge.job_id = 'j'
    bridge.last_sequence = 0; bridge.last_status = ''
    bridge.timeout = 5.; bridge.wall_backstop = 180.
    bridge.hold = lambda: None
    return bridge, time


def test_status_watchdog_does_not_false_trip_on_render_paced_wall_gap():
    # Render-paced retry: 30 s of WALL time passed but the sim clock advanced only 0.2 s.
    # The old wall-time 5 s bound killed the mission here (run-82); the node-clock bound must not.
    bridge, time = _silent_bridge()
    bridge.last_receive = time.monotonic() - 30.
    bridge.last_receive_sim = 10.0
    assert bridge.poll(simulation_time=10.2) is None
    assert bridge.active and not bridge.faulted


def test_status_watchdog_trips_on_sim_time_silence():
    bridge, time = _silent_bridge()
    bridge.last_receive = time.monotonic()
    bridge.last_receive_sim = 10.0
    result = bridge.poll(simulation_time=16.0)   # 6 s of SIM time with no status
    assert result['status'] == 'failed' and result['reason'] == 'host process/status watchdog'
    assert bridge.faulted and not bridge.active


def test_status_watchdog_wall_backstop_trips_when_sim_clock_frozen():
    bridge, time = _silent_bridge()
    bridge.last_receive = time.monotonic() - 200.   # frozen sim: sim bound can never advance
    bridge.last_receive_sim = 10.0
    result = bridge.poll(simulation_time=10.0)
    assert result['status'] == 'failed' and result['reason'] == 'host process/status wall backstop'


def test_status_watchdog_legacy_wall_behaviour_without_sim_clock():
    bridge, time = _silent_bridge()
    bridge.last_receive = time.monotonic() - 6.
    bridge.last_receive_sim = None
    result = bridge.poll()   # no sim clock => legacy wall bound (5 s)
    assert result['status'] == 'failed' and result['reason'] == 'host process/status watchdog'


def test_native_bridge_ignores_reordered_status_before_gripper_actuation():
    import json
    import time
    from types import SimpleNamespace as NS
    bridge=IsaacRosGraph.__new__(IsaacRosGraph)
    commands=[]
    reply=dict(job_id='j',sequence=2,status='running',controls=[dict(kind='gripper',enabled=False)],
               valid_until_steady=time.monotonic()+.5)
    def attribute(path):
        return NS(get=lambda:False if '/stop/' in path else json.dumps(reply))
    bridge.og=NS(Controller=NS(attribute=attribute))
    bridge.root='/test'
    bridge.faulted=False; bridge.active=True; bridge.job_id='j'
    bridge.last_sequence=3; bridge.last_status=''
    bridge.last_receive=time.monotonic(); bridge.timeout=5.
    bridge.contact=NS(command=commands.append)
    bridge.hold=lambda:None
    assert bridge.poll() is None
    assert commands==[]
    reply['sequence']=4
    assert bridge.poll()['sequence']==4
    assert commands==[False]
    reply.update(sequence=5,valid_until_steady=time.monotonic()-.001)
    assert bridge.poll()['status']=='failed'
    assert commands==[False] and bridge.faulted and not bridge.active
