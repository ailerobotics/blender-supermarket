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


def _startable_graph(stop_value):
    """IsaacRosGraph with a dict-backed MUTABLE stop wire (get/set) + a fake host_control so start()'s
    _publish('job') never touches the OmniGraph publisher path. Returns (instance, store, sets, published)."""
    store = {'/graph/stop/wire.outputs:data': stop_value}
    sets = []
    published = []

    def attribute(path):
        return SimpleNamespace(get=lambda: store.get(path, False),
                               set=lambda v: (store.__setitem__(path, v), sets.append((path, v)))[0])
    instance = object.__new__(IsaacRosGraph)
    instance.faulted = False
    instance.active = False
    instance.job_id = None
    instance.root = '/graph'
    instance.hold = lambda: None
    instance.og = SimpleNamespace(Controller=SimpleNamespace(attribute=attribute))
    instance.host_control = SimpleNamespace(
        publish=lambda topic, data: published.append((topic, data)),
        read=lambda topic: '')
    return instance, store, sets, published


JOB = dict(job_id='restock-one', sku='juice_orange', quantity=1, source_bin='backstock_00',
           shelf_id='G3_back_s2')


def test_full_chain_transfer_start_clears_a_stale_latched_global_stop(capsys):
    # sim-robust-transfer-start-estop-v1 (the coldlivefull transfer-start e-stop): a station-liveness
    # stop published during the collect->couple->outbound->park nav gap LATCHES the bare Bool stop wire
    # on this subscriber; the station then RECOVERS at transfer's fresh prepare, but nothing un-latches
    # the wire. In the FULL CHAIN the same long-lived graph carries that stale True into the transfer op,
    # so its first check_health reported a bare 'host emergency stop'. start() must re-derive it (clear)
    # exactly like a fresh staged-start graph, so the transfer op's FIRST goal is accepted.
    graph, store, sets, published = _startable_graph(True)
    graph.start('transfer', JOB)                         # the full-chain transfer op start
    assert store['/graph/stop/wire.outputs:data'] is False   # stale latch cleared
    assert sets == [('/graph/stop/wire.outputs:data', False)]
    assert not graph.faulted and graph.active            # faulted latch re-derived; op active
    assert graph.job_id == 'restock-one'
    assert published and published[0][1]                 # the job was dispatched (first goal accepted)
    out = capsys.readouterr().out
    assert 'STALE_OP_START_STOP_CLEARED op=transfer' in out
    graph.check_health()                                 # the op's first poll gate no longer e-stops


def test_fresh_graph_op_start_is_byte_identical(capsys):
    # The staged start (start_at=transfer) and every cold collect op start begin with a fresh graph whose
    # stop wire was never True: start() must be a pure no-op for the clear (no write, no log) so those
    # paths stay byte-identical.
    graph, store, sets, published = _startable_graph(False)
    graph.start('transfer', JOB)
    assert store['/graph/stop/wire.outputs:data'] is False
    assert sets == []                                    # no clear write
    assert 'STALE_OP_START_STOP_CLEARED' not in capsys.readouterr().out
    assert graph.active and not graph.faulted


def test_a_genuine_ongoing_stop_still_fires_within_the_op():
    # Clearing a stale stop at op start must NOT mask a genuinely ongoing stop: a host that keeps
    # publishing True (the ROS2Subscriber re-takes it) re-arms the wire, so check_health still e-stops
    # within the op. Model it as the wire reading True again on the first post-start poll.
    graph, store, sets, published = _startable_graph(True)
    graph.start('transfer', JOB)                         # cleared to False
    store['/graph/stop/wire.outputs:data'] = True        # host still asserting stop -> re-taken next tick
    with pytest.raises(RuntimeError, match='host emergency stop'):
        graph.check_health()
    assert graph.faulted


def test_non_controller_stop_names_the_host_stack_trigger():
    # A station-liveness/trajectory stop sets no /manipulation/status, so the mission reason must still
    # point at the named host-stack.log trigger instead of a bare 'host emergency stop'.
    instance, _, _, _ = _startable_graph(True)
    instance.host_control = SimpleNamespace(read=lambda topic: '')   # no matching controller fault
    suffix = instance._host_fault_suffix()
    assert '_ESTOP_PUBLISH' in suffix and 'host-stack.log' in suffix


def test_controller_fault_status_still_names_the_reason():
    import json as _json
    instance, _, _, _ = _startable_graph(True)
    instance.job_id = 'restock-one'
    status = _json.dumps({'status': 'failed', 'reason': 'planning failed', 'job_id': 'restock-one'})
    instance.host_control = SimpleNamespace(read=lambda topic: status)
    assert instance._host_fault_suffix() == ' (host fault: planning failed)'


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
