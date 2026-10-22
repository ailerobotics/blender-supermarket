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


def _closeable_graph():
    # A minimal real IsaacRosGraph with a fake shared control relay (host_control). close() must hold
    # + fault always, but only TERMINATE the relay when close_control=True.
    closed=[]
    instance=object.__new__(IsaacRosGraph)
    instance.faulted=False
    instance.active=True
    instance.hold=lambda:None
    instance.host_control=SimpleNamespace(close=lambda:closed.append(True))
    return instance,closed


def test_per_op_close_keeps_shared_control_relay_alive():
    # sim-robust-station-ack: a per-op failure closes the graph WITHOUT tearing down the shared control
    # relay the StationClient reads /restock_sim/station/status through -- closing it froze the station
    # ack -> StationAckStreamStalled (armcam/transfer2/bench1). The relay survives; graph still faults.
    instance,closed=_closeable_graph()
    instance.close(close_control=False)
    assert instance.faulted and not instance.active and closed==[]   # relay KEPT alive


def test_teardown_close_terminates_the_shared_control_relay_once():
    # The genuine mission teardown (default close_control=True) terminates the relay exactly once.
    instance,closed=_closeable_graph()
    instance.close()                     # default close_control=True
    assert instance.faulted and not instance.active and closed==[True]
