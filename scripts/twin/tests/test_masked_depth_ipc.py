"""CPU tests for the masked-depth IPC framing (no rclpy / no sockets)."""
import struct

import numpy as np
import pytest

from twin.restock.masked_depth_ipc import (
    pack_frame, read_frame, MAGIC, FrameWriter, SubprocessSupervisor)


def _recv_from(buf):
    """A recv(n) that serves `buf` in arbitrary small chunks, then EOF."""
    state = {'i': 0}

    def recv(n):
        i = state['i']
        chunk = buf[i:i + min(n, 3)]          # deliberately fragment to exercise _recvn
        state['i'] = i + len(chunk)
        return chunk
    return recv


def _hdr(w=4, h=2, bbox=None):
    return {'stamp_sec': 7, 'stamp_nsec': 250_000_000, 'frame_id': 'restock_camera_arm',
            'width': w, 'height': h, 'k': [500.0, 0, w / 2, 0, 500.0, h / 2, 0, 0, 1.0],
            'bbox': bbox}


def test_pack_read_roundtrip_single_frame():
    depth = np.arange(8, dtype='<f4').tobytes()
    msg = pack_frame(_hdr(), depth)
    header, got = read_frame(_recv_from(msg))
    assert header == _hdr() and got == depth


def test_pack_read_roundtrip_with_bbox():
    depth = np.ones(8, dtype='<f4').tobytes()
    header, got = read_frame(_recv_from(pack_frame(_hdr(bbox=[1.0, 2.0, 3.0, 4.0]), depth)))
    assert header['bbox'] == [1.0, 2.0, 3.0, 4.0] and got == depth


def test_two_frames_in_one_stream():
    d1, d2 = np.zeros(8, dtype='<f4').tobytes(), np.ones(8, dtype='<f4').tobytes()
    stream = pack_frame(_hdr(), d1) + pack_frame(_hdr(bbox=[0.0, 0.0, 1.0, 1.0]), d2)
    recv = _recv_from(stream)
    h1, g1 = read_frame(recv)
    h2, g2 = read_frame(recv)
    assert g1 == d1 and g2 == d2 and h2['bbox'] == [0.0, 0.0, 1.0, 1.0]
    assert read_frame(recv) is None                      # clean EOF


def test_eof_returns_none():
    assert read_frame(lambda n: b'') is None


def test_bad_magic_rejected():
    depth = b'\x00\x00\x00\x00'
    body = b'XXXX' + struct.pack('>I', 2) + b'{}' + depth
    msg = struct.pack('>I', len(body)) + body
    with pytest.raises(ValueError):
        read_frame(_recv_from(msg))


def test_frame_writer_sends_packed_bytes():
    sent = []
    writer = FrameWriter('/unused', _sock=type('S', (), {'sendall': lambda self, b: sent.append(b),
                                                         'close': lambda self: None})())
    depth = np.arange(8, dtype='<f4').tobytes()
    writer.send(_hdr(), depth)
    header, got = read_frame(_recv_from(sent[0]))
    assert header == _hdr() and got == depth


# --- SubprocessSupervisor: survive a publisher writer-EOF without starving the feed ----------

class _FakeProc:
    """A subprocess.Popen stand-in whose exit is driven by the test."""

    def __init__(self):
        self._rc = None

    def poll(self):
        return self._rc

    @property
    def returncode(self):
        return self._rc

    def die(self, rc=0):                 # the publisher exited (rc=0 == the writer_eof we observe)
        self._rc = rc

    def terminate(self):
        if self._rc is None:
            self._rc = -15

    def kill(self):
        self._rc = -9

    def wait(self, timeout=None):
        return self._rc


class _FakeWriter:
    def __init__(self):
        self.sent = []
        self.closed = False
        self.raise_once = None          # inject a one-shot send error (a broken pipe)

    def send(self, header, depth):
        if self.raise_once is not None:
            exc, self.raise_once = self.raise_once, None
            raise exc
        self.sent.append((header, depth))

    def close(self):
        self.closed = True


def _spawner():
    """A spawn() returning a fresh (FakeProc, FakeWriter); `made` records each (re)launch."""
    made = []

    def spawn():
        p, w = _FakeProc(), _FakeWriter()
        made.append((p, w))
        return p, w
    return spawn, made


def test_supervisor_restarts_on_writer_eof_before_the_feed_stalls():
    # Replay of the transfer2 / bench1 EOF: the publisher exits rc=0 (writer_eof) mid-stream; the
    # NEXT publish detects the dead proc and respawns + reconnects, so masked depth keeps flowing and
    # the host incorporation-ack never freezes into StationAckStreamStalled.
    spawn, made = _spawner()
    logs = []
    sup = SubprocessSupervisor(spawn, feed='camera_0_arm', log=logs.append)
    sup.send({'frame': 1}, b'd1')
    assert made[0][1].sent == [({'frame': 1}, b'd1')] and len(made) == 1
    made[0][0].die(0)                                   # writer_eof, rc=0 (never a signal/OOM kill)
    sup.send({'frame': 2}, b'd2')                        # restart happens HERE, before any stall
    assert len(made) == 2 and sup.restarts == 1
    assert made[1][1].sent == [({'frame': 2}, b'd2')]   # the fresh publisher carried the frame
    assert made[0][1].closed                            # the dead writer was reaped
    assert any('MASKED_DEPTH_BRIDGE_RESTART' in m and 'writer_exit rc=0' in m for m in logs)


def test_supervisor_retries_on_broken_pipe_send():
    spawn, made = _spawner()
    sup = SubprocessSupervisor(spawn, log=lambda *_: None)
    made[0][1].raise_once = BrokenPipeError()           # peer died between poll and send
    sup.send({'f': 3}, b'd3')
    assert sup.restarts == 1 and len(made) == 2
    assert made[1][1].sent == [({'f': 3}, b'd3')]       # retried on the fresh writer


def test_supervisor_does_not_restart_after_close():
    spawn, made = _spawner()
    sup = SubprocessSupervisor(spawn, log=lambda *_: None)
    sup.close()
    assert made[0][1].closed and sup.poll() == -15      # terminated, not restarted
    with pytest.raises(RuntimeError):
        sup.send({'f': 9}, b'd9')
    assert len(made) == 1                               # a deliberate close never respawns


def test_supervisor_restart_budget_is_bounded():
    spawn, made = _spawner()
    sup = SubprocessSupervisor(spawn, max_restarts=2, log=lambda *_: None)
    for _ in range(2):
        made[-1][0].die(0)
        sup.send({'f': 0}, b'x')
    assert sup.restarts == 2 and len(made) == 3
    made[-1][0].die(0)
    with pytest.raises(RuntimeError, match='restart budget'):
        sup.send({'f': 0}, b'x')                        # a hard-crashing publisher can't spin forever


def test_real_unix_socket_roundtrip(tmp_path):
    """FrameWriter over a real AF_UNIX socket -> read_frame on the server side (no rclpy)."""
    import socket, threading
    path = str(tmp_path / 'md.sock')
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    received = {}

    def server():
        conn, _ = srv.accept()
        received['frame'] = read_frame(conn.recv)
        conn.close()
        srv.close()

    t = threading.Thread(target=server)
    t.start()
    writer = FrameWriter(path, connect_timeout_s=5.0)
    depth = np.arange(8, dtype='<f4').tobytes()
    writer.send(_hdr(bbox=[1.0, 2.0, 3.0, 4.0]), depth)
    t.join(timeout=5.0)
    writer.close()
    header, got = received['frame']
    assert got == depth and header['bbox'] == [1.0, 2.0, 3.0, 4.0]
