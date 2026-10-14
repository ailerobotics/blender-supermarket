"""CPU tests for the masked-depth IPC framing (no rclpy / no sockets)."""
import struct

import numpy as np
import pytest

from twin.restock.masked_depth_ipc import pack_frame, read_frame, MAGIC, FrameWriter


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
