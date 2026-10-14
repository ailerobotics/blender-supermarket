"""Out-of-Isaac masked-depth ROS publisher: IPC framing + the subprocess `serve` main.

rclpy cannot run inside the Isaac SimulationApp (the ros2.bridge extension and full rclpy
abort on conflicting ROS2 middleware). So the Isaac-side bridge computes the masked float32
depth + metadata each manip sensing tick and ships each frame over a length-prefixed Unix
socket to THIS module running as a separate process under the host overlay ROS env, where
rclpy works (the one proven pattern in this repo). The subprocess publishes masked_depth
(32FC1), camera_info and target_bbox on one stamp (frame restock_camera_arm); TF stays on
the Isaac OmniGraph transform tree.

This module is SELF-CONTAINED: the framing is pure stdlib (CPU-tested); `serve` lazy-imports
rclpy so the module imports fine under the Isaac python too (for the writer + framing).

Frame wire format (big-endian): [u32 body_len][b'MDF1'][u32 header_json_len][header json]
[depth bytes = width*height*4 float32 little-endian]. The header carries stamp_sec,
stamp_nsec, frame_id, width, height, k (9 floats row-major) and bbox ([u1,v1,u2,v2] or null).
"""
from __future__ import annotations

import json
import socket
import struct

MAGIC = b'MDF1'
_LEN = struct.Struct('>I')


def pack_frame(header, depth_bytes):
    """Serialize one frame (header dict + raw depth bytes) to a length-prefixed message."""
    hj = json.dumps(header, separators=(',', ':')).encode('utf-8')
    body = MAGIC + _LEN.pack(len(hj)) + hj + bytes(depth_bytes)
    return _LEN.pack(len(body)) + body


def _recvn(recv, n):
    """Read exactly n bytes via recv(k); return bytes, or None on clean EOF before any byte."""
    chunks = []
    got = 0
    while got < n:
        chunk = recv(n - got)
        if not chunk:
            return None if got == 0 else b''.join(chunks) + b''  # partial -> truncated
        chunks.append(chunk)
        got += len(chunk)
    return b''.join(chunks)


def read_frame(recv):
    """Read one frame via recv(k). Returns (header dict, depth bytes) or None at EOF."""
    prefix = _recvn(recv, _LEN.size)
    if not prefix or len(prefix) < _LEN.size:
        return None
    (body_len,) = _LEN.unpack(prefix)
    body = _recvn(recv, body_len)
    if body is None or len(body) < 8 + _LEN.size or body[:4] != MAGIC:
        raise ValueError('corrupt masked-depth frame (bad magic/length)')
    (hl,) = _LEN.unpack(body[4:8])
    header = json.loads(body[8:8 + hl].decode('utf-8'))
    depth = body[8 + hl:]
    return header, depth


class FrameWriter:
    """Isaac-side client: connect to the subprocess socket and send masked-depth frames."""

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
        raise TimeoutError(f'masked-depth subprocess did not accept on {socket_path}: {last}')

    def send(self, header, depth_bytes):
        self.sock.sendall(pack_frame(header, depth_bytes))

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def serve(socket_path, *, idle_timeout_s=600.0):
    """Subprocess entry: bind the socket, accept the Isaac writer, publish each frame.

    Runs under the host overlay ROS env (domain 88). Publishes masked_depth (Image 32FC1),
    camera_info (CameraInfo) and target_bbox (Float32MultiArray) with SENSOR_DATA QoS -- the
    exact QoS the shelfbot object_segmenter subscribes with. Exits when the writer closes or
    after `idle_timeout_s` with no frame.
    """
    import os
    import numpy as np
    import rclpy
    # Guard the ROS seam: this subprocess MUST load the host rclpy (the /opt/ros jazzy or the
    # runtime-closure overlay), never Isaac's bundled "internal" rclpy -- mixing the two leaves
    # the publishers on an incompatible middleware the host cannot see (run-37). Fail loudly so
    # this can never silently recur.
    _rclpy_path = getattr(rclpy, '__file__', '') or ''
    if ('isaacsim' in _rclpy_path or '/env_isaaclab/' in _rclpy_path
            or not ('opt/ros' in _rclpy_path or 'runtime-closure' in _rclpy_path)):
        raise RuntimeError('masked-depth publisher loaded the wrong rclpy: ' + _rclpy_path +
                           ' (expected host /opt/ros or runtime-closure overlay, not Isaac internal)')
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo
    from std_msgs.msg import Float32MultiArray
    from builtin_interfaces.msg import Time

    if os.path.exists(socket_path):
        os.unlink(socket_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(1)
    srv.settimeout(idle_timeout_s)

    rclpy.init(args=None)
    node = Node('restock_masked_depth_pub')
    depth_pub = node.create_publisher(Image, '/restock_sim/camera/masked_depth', qos_profile_sensor_data)
    info_pub = node.create_publisher(CameraInfo, '/restock_sim/camera/masked_depth/camera_info', qos_profile_sensor_data)
    bbox_pub = node.create_publisher(Float32MultiArray, '/restock_sim/target_bbox', qos_profile_sensor_data)

    conn, _ = srv.accept()
    conn.settimeout(idle_timeout_s)
    published = 0
    try:
        while True:
            frame = read_frame(conn.recv)
            if frame is None:
                break
            header, depth = frame
            stamp = Time(sec=int(header['stamp_sec']), nanosec=int(header['stamp_nsec']))
            h, w = int(header['height']), int(header['width'])
            img = Image()
            img.header.stamp, img.header.frame_id = stamp, header['frame_id']
            img.height, img.width = h, w
            img.encoding, img.is_bigendian, img.step = '32FC1', 0, w * 4
            img.data = depth
            info = CameraInfo()
            info.header.stamp, info.header.frame_id = stamp, header['frame_id']
            info.height, info.width = h, w
            info.distortion_model = 'plumb_bob'
            info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            k = [float(v) for v in header['k']]
            info.k = k
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [k[0], 0.0, k[2], 0.0, 0.0, k[4], k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
            depth_pub.publish(img)
            info_pub.publish(info)
            if header.get('bbox') is not None:
                msg = Float32MultiArray()
                msg.data = [float(v) for v in header['bbox']]
                bbox_pub.publish(msg)
            published += 1
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        node.get_logger().info(f'masked-depth publisher exiting; published={published}')
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
    if len(sys.argv) >= 3 and sys.argv[1] == 'serve':
        timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
        serve(sys.argv[2], idle_timeout_s=timeout)
    else:
        print('usage: masked_depth_ipc.py serve <socket_path> [idle_timeout_s]', file=sys.stderr)
        sys.exit(2)
