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

# Default (eye-in-hand wrist, camera_0) ROS topics. The mast feed (camera_1) passes its own via
# the serve() kwargs / CLI, so the wrist path stays byte-identical (it passes no topic args).
WRIST_DEPTH_TOPIC = '/restock_sim/camera/masked_depth'
WRIST_INFO_TOPIC = '/restock_sim/camera/masked_depth/camera_info'
WRIST_BBOX_TOPIC = '/restock_sim/target_bbox'


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
    """Isaac-side client: connect to the subprocess socket and send masked-depth frames.

    ``send_buffer_bytes`` / ``send_timeout_s`` BOUND the client-side buffering so a stalled or
    dead publisher cannot back the Isaac physics loop up without limit: SO_SNDBUF caps the kernel
    send buffer and a send timeout turns an unbounded ``sendall`` block into a prompt OSError the
    supervisor can act on (restart the feed) instead of wedging the tick. Both default to ``None``
    (the legacy unbounded, blocking behaviour) so existing callers/tests are byte-identical; the
    bridge opts into a bound.
    """

    def __init__(self, socket_path, *, connect_timeout_s=30.0, send_buffer_bytes=None,
                 send_timeout_s=None, _sock=None):
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
                if send_buffer_bytes:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(send_buffer_bytes))
                if send_timeout_s:
                    s.settimeout(float(send_timeout_s))
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


class SubprocessSupervisor:
    """Keep a masked-depth publisher subprocess alive across an UNEXPECTED exit.

    The Isaac-side bridge ships frames to an out-of-Isaac rclpy publisher over a Unix socket
    (:class:`FrameWriter` -> :func:`serve`). A publisher that dies mid-mission -- a genuine crash
    (e.g. VRAM pressure), a broken pipe, or any non-``close()`` exit -- silently stops the host
    nvblox/incorporation-ack stream, which freezes the station ack stamp and faults the mission
    with the misleading ``StationAckStreamStalled`` (masking the real stage reason). This
    supervisor detects the death on the NEXT :meth:`send` and respawns the subprocess +
    reconnects the writer, bounded by ``max_restarts``, so the feed self-heals WITHIN the station
    ack's freshness window (stage-clock seconds; under render pacing a sub-second wall respawn is
    well inside it) instead of starving it. A death caused by our own :meth:`close` never restarts
    (``_closed`` latches).

    ``spawn()`` is the one injected seam: it (re)launches the publisher and returns
    ``(proc, writer)`` where ``proc`` exposes ``poll()``/``returncode``/``terminate()``/``kill()``/
    ``wait()`` (a ``subprocess.Popen``) and ``writer`` exposes ``send(header, depth)``/``close()``
    (a :class:`FrameWriter`). Keeping the spawn out of this class makes the whole restart policy
    CPU-testable with fakes -- no Isaac process, no real socket.
    """

    def __init__(self, spawn, *, feed='camera_0_arm', max_restarts=20, log=print):
        self._spawn = spawn
        self.feed = feed
        self.max_restarts = int(max_restarts)
        self._log = log
        self._closed = False
        self.restarts = 0
        self.proc, self.writer = spawn()

    def send(self, header, depth_bytes):
        """Ship one frame, respawning first if the publisher exited since the last send.

        Raises only when the feed is genuinely exhausted (restart budget spent) or already
        closed -- so the caller's existing failure handling still applies -- while a transient
        crash is healed silently (one frame may be lost, never the whole feed)."""
        if self._closed:
            raise RuntimeError('masked-depth supervisor is closed')
        if self.proc.poll() is not None:
            self._restart('writer_exit rc=%s' % self.proc.returncode)
        try:
            self.writer.send(header, depth_bytes)
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            # The peer died between the poll above and this send; restart once and retry so a
            # single lost frame does not become a dead feed. A second failure propagates.
            self._restart('send_error %s' % type(exc).__name__)
            self.writer.send(header, depth_bytes)

    def _restart(self, why):
        if self.restarts >= self.max_restarts:
            raise RuntimeError('masked-depth publisher exhausted restart budget (%d) after %s'
                               % (self.max_restarts, why))
        self.restarts += 1
        self._log('MASKED_DEPTH_BRIDGE_RESTART feed=%s attempt=%d/%d why=%s'
                  % (self.feed, self.restarts, self.max_restarts, why))
        self._reap()
        self.proc, self.writer = self._spawn()

    def _reap(self):
        """Close the old writer and terminate the old subprocess (best-effort, never raises)."""
        try:
            self.writer.close()
        except Exception:
            pass
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except Exception:
                    self.proc.kill()
        except Exception:
            pass

    def close(self):
        self._closed = True
        self._reap()

    def poll(self):
        return self.proc.poll()

    @property
    def returncode(self):
        return self.proc.returncode


def serve(socket_path, *, idle_timeout_s=600.0, depth_topic=WRIST_DEPTH_TOPIC,
          info_topic=WRIST_INFO_TOPIC, bbox_topic=WRIST_BBOX_TOPIC):
    """Subprocess entry: bind the socket, accept the Isaac writer, publish each frame.

    Runs under the host overlay ROS env (domain 88). Publishes masked_depth (Image 32FC1),
    camera_info (CameraInfo) and target_bbox (Float32MultiArray) with SENSOR_DATA QoS -- the
    exact QoS the shelfbot object_segmenter subscribes with. Exits when the writer closes or
    after `idle_timeout_s` with no frame.

    The topic names default to the wrist (camera_0) feed; the mast (camera_1) feed passes its own
    ``depth_topic``/``info_topic`` and ``bbox_topic=None`` (the shared ``/restock_sim/target_bbox``
    is published once by the wrist, so the mast segmenter reuses it -- no second bbox publisher).
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
    depth_pub = node.create_publisher(Image, depth_topic, qos_profile_sensor_data)
    info_pub = node.create_publisher(CameraInfo, info_topic, qos_profile_sensor_data)
    # Only the wrist publishes the shared grasp-target bbox; the mast reuses it (bbox_topic=None).
    bbox_pub = (node.create_publisher(Float32MultiArray, bbox_topic, qos_profile_sensor_data)
                if bbox_topic else None)

    conn, _ = srv.accept()
    conn.settimeout(idle_timeout_s)
    published = 0
    # fm463 lifecycle instrumentation: name WHY this publisher subprocess exits (writer EOF from a
    # bridge close, an idle timeout, or an exception such as the rclpy context going invalid after a
    # SIGTERM) and WHICH feed it is -- so a death in the run log names its own trigger even with the
    # mast off. `feed` is derived from the per-feed socket path (restock_camera_arm vs _mast).
    _feed = 'camera_1_mast' if 'mast' in str(socket_path) else 'camera_0_arm'
    _exit_reason = 'unknown'
    try:
        while True:
            try:
                frame = read_frame(conn.recv)
            except socket.timeout:
                _exit_reason = 'idle_timeout_%.0fs' % idle_timeout_s
                break
            if frame is None:
                _exit_reason = 'writer_eof'   # the Isaac-side FrameWriter closed (a bridge close())
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
            if bbox_pub is not None and header.get('bbox') is not None:
                msg = Float32MultiArray()
                msg.data = [float(v) for v in header['bbox']]
                bbox_pub.publish(msg)
            published += 1
            if published % 20 == 0:          # periodic so the count survives a SIGTERM kill
                node.get_logger().info(f'masked-depth publisher: published={published}')
            rclpy.spin_once(node, timeout_sec=0.0)
    except BaseException as _e:   # rclpy context invalid after a SIGTERM, a publish error, etc.
        _exit_reason = 'exception:%s:%s' % (type(_e).__name__, str(_e)[:160])
    finally:
        # fm463: print (not node.get_logger -- the rclpy context may already be invalid) so the reason
        # always lands even on a signal-induced context shutdown.
        print('MASKED_DEPTH_SERVE_EXIT feed=%s reason=%s published=%d' % (_feed, _exit_reason, published),
              flush=True)
        try:
            node.get_logger().info(f'masked-depth publisher exiting; published={published} '
                                   f'feed={_feed} reason={_exit_reason}')
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()                 # guard the double-shutdown RCLError when a second masked-depth
        except Exception:                    # IPC (e.g. the mast camera_1 feed) already shut the context down
            pass                             # -- the error is cosmetic at exit, but its traceback masked the
                                             # real early-exit reason in run-335's cold station-freeze log
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
        # Optional per-feed topic overrides (the mast feed): [depth_topic info_topic [bbox_topic]].
        # A bbox topic of '' or 'none' disables the bbox publisher (mast reuses the shared wrist one).
        kw = {}
        if len(sys.argv) > 5:
            kw['depth_topic'], kw['info_topic'] = sys.argv[4], sys.argv[5]
        if len(sys.argv) > 6:
            kw['bbox_topic'] = None if sys.argv[6] in ('', 'none') else sys.argv[6]
        serve(sys.argv[2], idle_timeout_s=timeout, **kw)
    else:
        print('usage: masked_depth_ipc.py serve <socket_path> [idle_timeout_s] '
              '[depth_topic info_topic [bbox_topic]]', file=sys.stderr)
        sys.exit(2)
