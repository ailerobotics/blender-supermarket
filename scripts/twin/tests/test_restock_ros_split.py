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


def test_native_bridge_ignores_reordered_status_before_gripper_actuation():
    import json
    import time
    from types import SimpleNamespace as NS
    bridge=IsaacRosGraph.__new__(IsaacRosGraph)
    commands=[]
    reply=dict(job_id='j',sequence=2,status='running',controls=[dict(kind='gripper',enabled=False)])
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
