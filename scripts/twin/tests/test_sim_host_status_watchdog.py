"""sim-host-status-watchdog-v1: CPU proof for the THIRD host-liveness watchdog false trip.

Recorded failure (collectretreat1-fail1-saved): a cold GROUND-TRUTH collect welded the carton cleanly
and lifted it, then the host dispatched the post-lift cuMotion plan, hit the sim planning deadline and a
``MOVE_GROUP_CRASH_RETRY`` + post-respawn carton RECONCILE. That reconcile runs ``RosPayloadSync._wait``,
a BLOCKING ack wait on the host's single-threaded main executor, so ``RosTransferBridge._sensors`` /
``_emit`` cannot publish ``/manipulation/status`` for the whole bounded wait. The host was provably alive
(nvblox served ESDF and the parked-station ack advanced -- both from INDEPENDENT nodes), but the
Isaac-side status watchdog (``IsaacRosGraph.poll``, 5 SIM-s) saw no fresh status for >5 sim s and
pre-empted the host's OWN named ack/plan/respawn timers, killing the live host
(``reason='host process/status watchdog'``).

The fix keeps the status source PUBLISHING THROUGH the blocking wait: ``_wait`` pulses
``RosTransferBridge.heartbeat`` each poll slice (a liveness-only ``/manipulation/status`` with a fresh
sequence and no controls), which (a) keeps the Isaac status watchdog's ``last_receive_sim`` fresh so it
no longer false-trips, and (b) advances the host's OWN sensor-stream ``_watchdog`` anchors
(``last_packet_sim``/``last_packet_wall``) so the sibling bound does not false-trip 'sensor stream
watchdog expired' when ``_watchdog`` resumes after the block. A genuine loss still trips by name: the
owning ``_wait``'s sim-budget TimeoutError for an instrumented wait, or the Isaac status watchdog's 5
sim-s backstop for an UNINSTRUMENTED executor hang that emits no heartbeat.

Pure CPU: no Isaac / ROS / rclpy / GPU. Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= .
"""
import json
import threading
import time
from types import SimpleNamespace as NS

import pytest

from scripts.twin.restock_manipulation.isaac_ros_graph import IsaacRosGraph, TOPICS
from scripts.twin.restock_manipulation.ros_adapter import RosTransferBridge
from scripts.twin.restock_manipulation.ros_payload import RosPayloadSync


# --------------------------------------------------------------------------------------------------
# Isaac-side status watchdog (IsaacRosGraph.poll): the external observer that tripped.
# --------------------------------------------------------------------------------------------------
def _graph_with_host_control(read):
    """A faulted-free, active graph whose host status arrives via a host_control relay stub."""
    graph = IsaacRosGraph.__new__(IsaacRosGraph)
    graph.og = NS(Controller=NS(attribute=lambda path: NS(get=lambda: False)))  # stop wire never True
    graph.root = '/test'
    graph.host_control = NS(read=read)
    graph.faulted = False
    graph.active = True
    graph.job_id = 'restock-one'
    graph.last_sequence = 0
    graph.last_status = ''
    graph.last_receive = time.monotonic()
    graph.last_receive_sim = None
    graph.timeout = 5.
    graph.wall_backstop = 180.
    graph.contact = NS(command=lambda enabled: None)
    graph.hold = lambda: None
    return graph


def _heartbeat_json(sequence):
    return json.dumps({'status': 'running', 'phase': 'lift', 'reason': 'host-alive heartbeat',
                       'controls': [], 'evidence': [], 'sequence': sequence,
                       'job_id': 'restock-one', 'heartbeat': True})


def test_recorded_silence_false_trips_without_heartbeat_and_names_the_loss(capsys):
    """BEFORE: with no host heartbeat the recorded >5 sim-s status silence trips -- now by a NAMED line."""
    graph = _graph_with_host_control(read=lambda topic: '')   # host emits nothing (blocked executor)
    # Recorded timing: last fresh status at sim ~81.8, poll at 86.85 (the collectretreat1 trip point) --
    # a 5.05 sim-s gap, just past the 5 s bound, while the host was blocked in the reconcile payload wait.
    graph.last_receive_sim = 81.80
    result = graph.poll(simulation_time=86.85)
    assert result == {'status': 'failed', 'reason': 'host process/status watchdog'}
    assert graph.faulted and not graph.active
    line = capsys.readouterr().out
    assert 'HOST_STATUS_WATCHDOG' in line and 'site=isaac_ros_graph.poll' in line
    assert 'clock=sim' in line and 'job=' in line


def test_host_heartbeat_stream_prevents_false_trip_across_the_blocking_wait():
    """AFTER: the host pulses a heartbeat through the payload wait, so poll() stays alive far past 5 sim s."""
    seq = {'n': 0}

    def read(topic):
        return _heartbeat_json(seq['n']) if topic == TOPICS['status'] else None

    graph = _graph_with_host_control(read)
    # Walk the recorded reconcile window (sim 82 -> 102, 20 sim s >> the 5 s bound); the host emits a
    # fresh heartbeat every sim second (the _wait slice is sub-second wall, so far denser in practice).
    for sim_t in [82.0 + i for i in range(21)]:
        seq['n'] += 1                                   # host advanced the status sequence (heartbeat)
        assert graph.poll(simulation_time=sim_t) is not None
        assert graph.active and not graph.faulted
    assert graph.last_receive_sim == pytest.approx(102.0)


def test_genuine_host_death_still_trips_by_name_when_no_heartbeat(capsys):
    """An UNINSTRUMENTED executor hang emits no heartbeat; the status watchdog is still the backstop."""
    graph = _graph_with_host_control(read=lambda topic: '')
    graph.last_receive_sim = 50.0
    assert graph.poll(simulation_time=54.0) is None      # within the 5 s bound: still alive
    result = graph.poll(simulation_time=55.5)             # 5.5 sim s of true silence -> trip
    assert result == {'status': 'failed', 'reason': 'host process/status watchdog'}
    assert 'HOST_STATUS_WATCHDOG' in capsys.readouterr().out


def test_frozen_sim_wall_backstop_still_named(capsys):
    graph = _graph_with_host_control(read=lambda topic: '')
    graph.last_receive = time.monotonic() - 200.         # frozen sim: the sim leg can never advance
    graph.last_receive_sim = 10.0
    result = graph.poll(simulation_time=10.0)
    assert result == {'status': 'failed', 'reason': 'host process/status wall backstop'}
    out = capsys.readouterr().out
    assert 'HOST_STATUS_WATCHDOG' in out and 'clock=wall_backstop' in out


# --------------------------------------------------------------------------------------------------
# Host-side: RosTransferBridge.heartbeat (the liveness source) + RosPayloadSync._wait (publishes through).
# --------------------------------------------------------------------------------------------------
def _bridge():
    bridge = RosTransferBridge.__new__(RosTransferBridge)
    published = []
    bridge.String = lambda data: data                    # publish(self.String(data=...)) -> the json str
    bridge.status = NS(publish=published.append)
    bridge.sequence = 7
    bridge.controller = NS(phase='grasp', job={'job_id': 'restock-one'})
    bridge.last_packet_sim = 10.0
    bridge.last_packet_wall = 0.0
    bridge.node = NS(get_clock=lambda: NS(now=lambda: NS(nanoseconds=30.0 * 1e9)))
    return bridge, published


def test_bridge_heartbeat_emits_liveness_status_and_refreshes_sensor_anchors():
    bridge, published = _bridge()
    wall0 = time.monotonic()
    bridge.heartbeat()
    assert bridge.sequence == 8                           # advanced so the Isaac poll sees a fresh status
    data = json.loads(published[-1])
    assert data['status'] == 'running' and data['heartbeat'] is True
    assert data['controls'] == [] and data['sequence'] == 8 and data['job_id'] == 'restock-one'
    # Sibling fix: the sensor-stream _watchdog anchors advanced to NOW (host-busy, not a sensor stall),
    # so the per-frame bound measures from AFTER the wait and does not false-trip on resume.
    assert bridge.last_packet_sim == pytest.approx(30.0)
    assert bridge.last_packet_wall >= wall0


def test_bridge_heartbeat_is_a_no_op_without_an_active_job():
    for phase in ('idle', 'failed', 'succeeded'):
        bridge, published = _bridge()
        bridge.controller = NS(phase=phase, job={'job_id': 'restock-one'})
        bridge.heartbeat()
        assert published == [] and bridge.sequence == 7   # nothing to keep alive -> byte-identical


def test_sensor_stream_sibling_bound_not_tripped_after_a_heartbeated_block():
    """The host sensor-stream per-frame bound (sim_now - last_packet_sim > 1 s) would false-trip after a
    multi-sim-second payload block; the heartbeat's anchor refresh keeps it below the bound."""
    watchdog_seconds = 1.0
    bridge, _ = _bridge()
    bridge.last_packet_sim = 10.0                         # last real frame before the block
    sim_now = 30.0                                        # 20 sim s of blocking reconcile elapsed
    bridge.node = NS(get_clock=lambda: NS(now=lambda: NS(nanoseconds=sim_now * 1e9)))
    # Without the refresh the bound trips (20 s > 1 s); the heartbeat refreshes the anchor to sim_now.
    assert sim_now - 10.0 > watchdog_seconds             # the sibling false trip the fix prevents
    bridge.heartbeat()
    assert sim_now - bridge.last_packet_sim <= watchdog_seconds


class _FakeFuture:
    """A ROS service/action future stub: pending until complete() fires the registered done callback."""
    def __init__(self, result='ACK'):
        self._cb = None
        self._result = result
    def add_done_callback(self, cb):
        self._cb = cb
    def complete(self):
        if self._cb is not None:
            self._cb(self)
    def result(self):
        return self._result


def _payload_for_wait(sim_now_fn, heartbeat, ack_wall_s=1.0):
    p = RosPayloadSync.__new__(RosPayloadSync)
    p.ack_wall_s = ack_wall_s                             # drives the _wait poll-slice width
    p.ack_sim_s = 15.0
    p._sim_now = sim_now_fn
    p.heartbeat = heartbeat
    return p


def test_payload_wait_pulses_heartbeat_each_slice_until_the_ack_arrives():
    beats = {'n': 0}
    # ack_wall_s=0.1 => a 0.1 s poll slice; the ack lands after ~3 slices, so heartbeats fire first.
    p = _payload_for_wait(sim_now_fn=lambda: 0.0, ack_wall_s=0.1,
                          heartbeat=lambda: beats.__setitem__('n', beats['n'] + 1))
    future = _FakeFuture('ATTACHED')
    threading.Timer(0.35, future.complete).start()
    assert p._wait(future) == 'ATTACHED'
    assert beats['n'] >= 1                                # the host kept /manipulation/status alive


def test_payload_wait_still_fails_by_name_when_its_own_sim_budget_is_exhausted():
    """A heartbeat masks the Isaac status watchdog, so liveness detection must stay with _wait's OWN
    sim-clock ack budget -- a genuinely unacknowledged attach still fails by name, not silently."""
    beats = {'n': 0}
    clock = {'t': 100.0}

    def sim_now():
        clock['t'] += 50.0                               # jump the sim clock past ack_sim_s (15) fast
        return clock['t']

    p = _payload_for_wait(sim_now_fn=sim_now, heartbeat=lambda: beats.__setitem__('n', beats['n'] + 1))
    with pytest.raises(TimeoutError, match='payload acknowledgement deadline exceeded'):
        p._wait(_FakeFuture())                           # never completes
    assert beats['n'] >= 1                                # heartbeats fired AND the named failure still won


def test_payload_wait_without_heartbeat_is_byte_identical():
    """heartbeat=None (unit tests / a real rig without the bridge) => the ack wait is unchanged."""
    p = _payload_for_wait(sim_now_fn=lambda: 0.0, heartbeat=None)
    future = _FakeFuture('OK')
    threading.Timer(0.12, future.complete).start()
    assert p._wait(future) == 'OK'
