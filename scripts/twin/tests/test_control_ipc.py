"""CPU contract tests for the host-DDS control-plane publisher IPC (fix #2).

Isaac OmniGraph ROS2 publishers do not reach the host UDPv4 DDS, so the Isaac->host control
messages (manipulation/job, sensors, station/command) are shipped over a Unix socket to an
out-of-Isaac rclpy subprocess that publishes them on the host DDS. These tests exercise the
pure framing + the routing seams (no Isaac/omni, no rclpy, no subprocess): the omni imports in
IsaacRosGraph/JsonChannel live inside __init__, so the publish routing is tested via __new__.
"""
import json
import socket
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.control_ipc import (pack_message, read_message, MessageWriter,
                                       HostControlPublisher, TOPIC_TYPES, _secs_to_time)

JOB = '/restock_sim/manipulation/job'
SENSORS = '/restock_sim/sensors'
STACMD = '/restock_sim/station/command'


def _buffer_recv(data):
    pos = {'i': 0}
    def recv(n):
        chunk = bytes(data[pos['i']:pos['i'] + n])
        pos['i'] += len(chunk)
        return chunk
    return recv


class Framing(unittest.TestCase):
    def test_round_trip_multiple_messages(self):
        buf = bytearray()
        msgs = [(JOB, json.dumps({'operation': 'collect', 'job': {'job_id': 'j1'}})),
                (SENSORS, '{}'),
                (STACMD, json.dumps({'action': 'prepare', 'epoch': 1}))]
        for t, d in msgs:
            buf += pack_message(t, d)
        recv = _buffer_recv(buf)
        got = []
        while True:
            m = read_message(recv)
            if m is None:
                break
            got.append(m)
        self.assertEqual(got, msgs)

    def test_clean_eof_returns_none(self):
        self.assertIsNone(read_message(_buffer_recv(b'')))

    def test_corrupt_magic_raises(self):
        bad = bytearray(pack_message(JOB, '{}'))
        bad[4:8] = b'XXXX'  # clobber the magic inside the body
        with self.assertRaises(ValueError):
            read_message(_buffer_recv(bad))

    def test_empty_payload_round_trips(self):
        recv = _buffer_recv(pack_message(STACMD, ''))
        self.assertEqual(read_message(recv), (STACMD, ''))


class TransportRelayFraming(unittest.TestCase):
    """The sim-time backbone (/clock, /tf, /joint_states) rides the same (topic, JSON) framing; the
    subprocess decodes by TOPIC_TYPES into the right ROS message (that decode needs ROS; the framing
    + the JSON payload shapes are what's CPU-checkable)."""

    def test_secs_to_time_split(self):
        self.assertEqual(_secs_to_time(1.25), (1, 250000000))
        self.assertEqual(_secs_to_time(0.0), (0, 0))

    def test_topic_types_cover_the_backbone_and_control_plane(self):
        self.assertEqual(TOPIC_TYPES['/clock'], 'clock')
        self.assertEqual(TOPIC_TYPES['/tf'], 'tf')
        self.assertEqual(TOPIC_TYPES['/restock_sim/joint_states'], 'joints')
        self.assertEqual(TOPIC_TYPES['/restock_sim/manipulation/job'], 'string')

    def test_typed_payloads_round_trip_through_the_framing(self):
        import json
        msgs = [('/clock', json.dumps({'t': 3.4166})),
                ('/restock_sim/joint_states', json.dumps({'t': 3.4, 'names': ['a', 'b'], 'pos': [0.1, 0.2]})),
                ('/tf', json.dumps({'t': 3.4, 'tf': [['ur_base_link', 'restock_camera_arm',
                                                      0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]]}))]
        buf = bytearray()
        for t, d in msgs:
            buf += pack_message(t, d)
        got = []
        recv = _buffer_recv(buf)
        while True:
            m = read_message(recv)
            if m is None:
                break
            got.append(m)
        self.assertEqual(got, msgs)
        self.assertEqual(json.loads(got[2][1])['tf'][0][1], 'restock_camera_arm')


class ReturnPath(unittest.TestCase):
    """Reverse framing (fix #2): the subprocess forwards station/status + manipulation/status back
    over the same socket; the Isaac-side reader keeps the latest per topic for the mission consumers."""

    def test_read_loop_populates_latest_per_topic_and_read_returns_it(self):
        import socket as _socket
        import threading
        a, b = _socket.socketpair()
        pub = HostControlPublisher.__new__(HostControlPublisher)
        pub._writer = type('W', (), {'sock': a})()
        pub._latest, pub._lock = {}, threading.Lock()
        b.sendall(pack_message('/restock_sim/station/status', '{"state":"warming"}'))
        b.sendall(pack_message('/restock_sim/manipulation/status', '{"status":"running"}'))
        b.sendall(pack_message('/restock_sim/station/status', '{"state":"ready"}'))  # newer wins
        b.close()
        pub._read_loop()                                     # drains until EOF
        self.assertEqual(pub.read('/restock_sim/station/status'), '{"state":"ready"}')
        self.assertEqual(pub.read('/restock_sim/manipulation/status'), '{"status":"running"}')
        self.assertIsNone(pub.read('/restock_sim/nope'))
        a.close()


class Writer(unittest.TestCase):
    def test_writer_sends_framed_message(self):
        a, b = socket.socketpair()
        try:
            MessageWriter('/unused', _sock=a).send(SENSORS, '{"x":1}')
            self.assertEqual(read_message(b.recv), (SENSORS, '{"x":1}'))
        finally:
            a.close(); b.close()


class HostControlRouting(unittest.TestCase):
    def test_publish_rejects_undeclared_topic_and_routes_declared(self):
        pub = HostControlPublisher.__new__(HostControlPublisher)
        pub.topics = (JOB, SENSORS, STACMD)
        sent = []
        pub._writer = type('W', (), {'send': lambda self, t, d: sent.append((t, d))})()
        pub.publish(JOB, '{"operation":"collect"}')
        self.assertEqual(sent, [(JOB, '{"operation":"collect"}')])
        with self.assertRaises(ValueError):
            pub.publish('/restock_sim/manipulation/status', '{}')  # a host->Isaac topic, not ours


class _FakeHostControl:
    def __init__(self):
        self.sent = []
    def publish(self, topic, data):
        self.sent.append((topic, data))


class GraphAndChannelRoute(unittest.TestCase):
    def test_graph_publish_routes_job_and_sensors_to_host_control(self):
        from twin.restock_manipulation.isaac_ros_graph import IsaacRosGraph, TOPICS
        g = IsaacRosGraph.__new__(IsaacRosGraph)   # bypass __init__ (imports omni)
        g.host_control = _FakeHostControl()
        g._publish('job', '{"operation":"collect"}')
        g._publish('sensors', '{"observations":[]}')
        self.assertEqual(g.host_control.sent,
                         [(TOPICS['job'], '{"operation":"collect"}'),
                          (TOPICS['sensors'], '{"observations":[]}')])

    def test_channel_publish_routes_station_command_to_host_control(self):
        from twin.restock.native_channel import JsonChannel
        c = JsonChannel.__new__(JsonChannel)
        c.host_control = _FakeHostControl()
        c.publish_topic = STACMD
        c.publish({'action': 'prepare', 'epoch': 1})
        self.assertEqual(c.host_control.sent, [(STACMD, json.dumps({'action': 'prepare', 'epoch': 1}, allow_nan=False))])


if __name__ == '__main__':
    unittest.main()
