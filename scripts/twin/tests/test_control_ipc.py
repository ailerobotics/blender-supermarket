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
                                       HostControlPublisher)

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
