"""Out-of-Isaac host-DDS publisher for the manip control-plane (job / sensors / station command).

Isaac OmniGraph ROS2 PUBLISHERS do not reach the host's UDPv4 Fast DDS (run-57/59: the
`/restock_sim/manipulation/job` and `/station/command` publishers show `Publisher count: 0`
to the host, so ros_host / the incorporation ack never receive them), while OmniGraph
SUBSCRIBERS do discover the host writers. So -- exactly as the masked-depth bridge already
does for depth -- the small std_msgs/String control messages Isaac must PUBLISH are shipped
over a length-prefixed Unix socket to THIS module running as a separate process under the
host overlay ROS env, where rclpy loads the host middleware and the publishers reach the
host DDS. The host->Isaac return topics (manipulation/status, station/status, and the
host-driven manipulation/stop + gripper/enable) stay on OmniGraph subscribers, which work.

This module is SELF-CONTAINED: framing is pure stdlib (CPU-tested); `serve` lazy-imports
rclpy so the module imports fine under the Isaac python too (for the writer + framing).

Wire format (big-endian): [u32 body_len][b'CTL1'][u32 topic_len][topic utf-8][data utf-8].
`data` is the exact std_msgs/String payload (a JSON object for job/sensors/station-command).
"""
from __future__ import annotations

import socket
import struct

MAGIC = b'CTL1'
_LEN = struct.Struct('>I')


def pack_message(topic, data):
    """Serialize one (topic, data string) to a length-prefixed message."""
    tb = topic.encode('utf-8')
    db = data.encode('utf-8')
    body = MAGIC + _LEN.pack(len(tb)) + tb + db
    return _LEN.pack(len(body)) + body


def _recvn(recv, n):
    """Read exactly n bytes via recv(k); return bytes, or None on clean EOF before any byte."""
    chunks = []
    got = 0
    while got < n:
        chunk = recv(n - got)
        if not chunk:
            return None if got == 0 else b''.join(chunks)
        chunks.append(chunk)
        got += len(chunk)
    return b''.join(chunks)


def read_message(recv):
    """Read one message via recv(k). Returns (topic, data) strings or None at EOF."""
    prefix = _recvn(recv, _LEN.size)
    if not prefix or len(prefix) < _LEN.size:
        return None
    (body_len,) = _LEN.unpack(prefix)
    body = _recvn(recv, body_len)
    if body is None or len(body) < 8 or body[:4] != MAGIC:
        raise ValueError('corrupt control message (bad magic/length)')
    (tl,) = _LEN.unpack(body[4:8])
    topic = body[8:8 + tl].decode('utf-8')
    data = body[8 + tl:].decode('utf-8')
    return topic, data


class MessageWriter:
    """Isaac-side client: connect to the subprocess socket and send control messages."""

    def __init__(self, socket_path, *, connect_timeout_s=30.0, _sock=None):
        self.socket_path = socket_path
        if _sock is not None:
            self.sock = _sock
            return
        import time
        deadline = time.monotonic() + connect_timeout_s
        last = None
        while time.monotonic() < deadline:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(socket_path)
                self.sock = s
                return
            except OSError as exc:  # subprocess not listening yet
                last = exc
                time.sleep(0.2)
        raise TimeoutError(f'control subprocess did not accept on {socket_path}: {last}')

    def send(self, topic, data):
        self.sock.sendall(pack_message(topic, data))

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class HostControlPublisher:
    """Isaac-side handle: spawn the host-DDS control publisher subprocess + relay publishes.

    Spawns `control_ipc.serve` under the host overlay ROS env (domain 88, UDPv4-only Fast DDS)
    with one std_msgs/String publisher per declared topic, and connects a Unix socket to it.
    `publish(topic, data)` ships the String payload; the subprocess publishes it on the host DDS
    so ros_host / the ack actually receive it. Mirrors `masked_depth.ArmDepthSubprocessBridge`'s
    spawn recipe (clean base env; the Isaac process env would mix two ROS middlewares).
    """

    def __init__(self, *, topics, runtime_env, python='python3', domain=88,
                 socket_path=None, idle_timeout_s=1800.0):
        self.topics = tuple(topics)
        if not self.topics:
            raise ValueError('at least one control topic is required')
        import os
        import subprocess
        self._socket_path = socket_path or f'/tmp/restock_control_{os.getpid()}.sock'
        ipc_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'control_ipc.py')
        command = (f'set +u; source {runtime_env}; '
                   f'export ROS_DOMAIN_ID={int(domain)} ROS_LOCALHOST_ONLY=1 ROS_STATIC_PEERS= '
                   f'ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp '
                   f'FASTDDS_BUILTIN_TRANSPORTS=UDPv4 SKIP_DEFAULT_XML=1; '
                   f'unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE '
                   f'ROS_DISCOVERY_SERVER CYCLONEDDS_URI RMW_FASTRTPS_USE_QOS_FROM_XML; '
                   f'exec {python} {ipc_script} serve {self._socket_path} {float(idle_timeout_s)} '
                   + ' '.join(self.topics))
        clean_env = {k: os.environ[k] for k in ('HOME', 'USER', 'LOGNAME', 'TERM', 'DISPLAY')
                     if k in os.environ}
        clean_env['PATH'] = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
        self._proc = subprocess.Popen(['bash', '-c', command], env=clean_env)
        self._writer = MessageWriter(self._socket_path)

    def publish(self, topic, data):
        if topic not in self.topics:
            raise ValueError(f'control topic not declared for the host publisher: {topic}')
        self._writer.send(topic, data)

    def close(self):
        try:
            self._writer.close()
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()


def serve(socket_path, topics, *, idle_timeout_s=1800.0):
    """Subprocess entry: bind the socket, accept the Isaac writer, publish each message.

    Runs under the host overlay ROS env (domain 88). Declares one std_msgs/String publisher per
    topic up front (so the host subscribers discover them during warm-up, before the first
    message), RELIABLE depth-10 (matching ros_host / the ack subscriptions). Exits when the
    writer closes or after `idle_timeout_s` with no message.
    """
    import os
    import rclpy
    # Guard the ROS seam: this subprocess MUST load the host rclpy, never Isaac's bundled one --
    # mixing the two leaves the publishers on a middleware the host cannot see (run-37). Fail loud.
    _rclpy_path = getattr(rclpy, '__file__', '') or ''
    if ('isaacsim' in _rclpy_path or '/env_isaaclab/' in _rclpy_path
            or not ('opt/ros' in _rclpy_path or 'runtime-closure' in _rclpy_path)):
        raise RuntimeError('control publisher loaded the wrong rclpy: ' + _rclpy_path +
                           ' (expected host /opt/ros or runtime-closure overlay, not Isaac internal)')
    from rclpy.node import Node
    from std_msgs.msg import String

    if os.path.exists(socket_path):
        os.unlink(socket_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(1)
    srv.settimeout(idle_timeout_s)

    rclpy.init(args=None)
    node = Node('restock_control_pub')
    publishers = {topic: node.create_publisher(String, topic, 10) for topic in topics}

    conn, _ = srv.accept()
    conn.settimeout(idle_timeout_s)
    published = 0
    try:
        while True:
            message = read_message(conn.recv)
            if message is None:
                break
            topic, data = message
            pub = publishers.get(topic)
            if pub is None:
                node.get_logger().warn(f'control publisher: undeclared topic {topic!r} ignored')
                continue
            pub.publish(String(data=data))
            published += 1
            node.get_logger().info(f'control publish[{published}] {topic}')
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        node.get_logger().info(f'control publisher exiting; published={published}')
        node.destroy_node()
        rclpy.shutdown()
        conn.close()
        srv.close()
        try:
            os.unlink(socket_path)
        except OSError:
            pass


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 4 and sys.argv[1] == 'serve':
        timeout = float(sys.argv[3])
        serve(sys.argv[2], sys.argv[4:], idle_timeout_s=timeout)
    else:
        print('usage: control_ipc.py serve <socket_path> <idle_timeout_s> <topic> [<topic>...]',
              file=sys.stderr)
        sys.exit(2)
