from types import SimpleNamespace

import pytest

from twin.restock_manipulation.isaac_ros_graph import IsaacRosGraph


def graph(stop):
    held=[]
    instance=object.__new__(IsaacRosGraph)
    instance.faulted=False
    instance.active=False
    instance.root='/graph'
    instance.hold=lambda:held.append(True)
    instance.og=SimpleNamespace(Controller=SimpleNamespace(
        attribute=lambda path:SimpleNamespace(get=lambda:stop)))
    return instance,held


def test_global_stop_latches_outside_active_manipulation():
    instance,held=graph(True)
    with pytest.raises(RuntimeError,match='host emergency stop'):
        instance.check_health()
    assert instance.faulted and held==[True]
    assert instance.poll()['status']=='failed'
    assert held==[True,True]


def test_health_check_does_not_consume_transfer_status():
    instance,held=graph(False)
    def attr(path):
        assert path.endswith('/stop/wire.outputs:data')
        return SimpleNamespace(get=lambda:False)
    instance.og.Controller.attribute=attr
    instance.check_health()
    assert not held and not instance.faulted
