"""Stride-paced sensing reuse in IsaacCartonSensors.capture (CPU; no Isaac/ROS/GPU).

Integration renders the sensing camera only every Nth physics tick (IsaacPort.sensing_stride)
so a render-paced grasp fits the wall ceiling; on the intervening ticks get_current_frame()
returns the SAME frame. capture must consume that last fresh frame (reuse its decode) instead
of refusing it as a replay -- but every time-sensitive gate (frame-age/freshness, baseline
lifetime, the live collision-map clearance gate, parked, QR pixel tamper) must still re-run.
These are analytic fixtures, never a renderer/physics demonstration of decode accuracy.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.twin.restock_manipulation import isaac_sensors
from scripts.twin.restock_camera import commissioning as C
from .test_restock_commissioning import empty_frame, context
from .test_restock_manipulation import FixturePoseBridge


class Camera:
    """Return whatever frame the test installs; count renders only through frame identity."""
    def __init__(self, frame, k):
        self.frame, self.k = frame, k

    def get_current_frame(self, **kw):
        return self.frame

    def get_intrinsics_matrix(self):
        k = self.k
        return np.array([[k.fx,0,k.cx],[0,k.fy,k.cy],[0,0,1.]])

    def add_distance_to_image_plane_to_frame(self):
        pass


class Motion:
    def __init__(self, collision_time=1.0):
        self.parked_ = True
        self.collision_time = collision_time
        self.sources = []

    def parked(self):
        return self.parked_

    def joint_positions(self):
        return (0.,)*6

    def observe_source(self, observation):
        self.sources.append(observation)


def _rig(tmp_path):
    """A real fixed-source IsaacCartonSensors over a commissioned+reloaded backstock_00 baseline.

    Returns (adapter, camera, motion, contact, decode_calls, frame). ``decode_calls`` counts the
    boot context's full-tile decoder invocations, so a reused frame is provably not re-decoded.
    Callers must monkeypatch ``isaac_sensors.load_pose_bridge`` before calling.
    """
    k, camera_pose, depth, support, witness, frame = empty_frame()
    commissioned = context(k, camera_pose, location='backstock_00', camera_id='source')
    C.commission_and_persist(commissioned, frame, 1., {'backstock_00': (support, witness)},
                             dataset_dir=tmp_path)
    boot = context(k, camera_pose, location='backstock_00', camera_id='source')
    C.apply_baselines(boot, C.load_baselines(tmp_path), required_locations=['backstock_00'])

    decode_calls = []
    real_decode = boot.decoder.decode
    def counting_decode(*a, **kw):
        decode_calls.append(1)
        return real_decode(*a, **kw)
    boot.decoder.decode = counting_decode

    camera = Camera(frame, k)
    adapter = isaac_sensors.IsaacCartonSensors(camera, pose_bridge_root='fixture', profiles={},
                                               context_provider=boot, camera_id='source',
                                               steady_clock=lambda: 0.)
    motion = Motion()
    contact = NS(read=lambda: False)
    return adapter, camera, motion, contact, decode_calls, frame


def test_exact_repeat_is_consumed_without_a_second_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(isaac_sensors, 'load_pose_bridge', lambda root: FixturePoseBridge())
    adapter, camera, motion, contact, decode_calls, frame = _rig(tmp_path)

    first = adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    assert first.observations == ()                      # empty source baseline: no pick carton
    assert len(decode_calls) == 1

    # Same frame id/stamp/pixels, control tick advanced within the 0.5 s window: reused, not refused.
    reused = adapter.capture(1.05, 'backstock_00', 'backstock_00', motion, contact)
    assert reused.observations == ()
    assert reused.scan_time == frame['rendering_time']   # same frame's sim stamp
    assert len(decode_calls) == 1                         # NOT re-decoded


def test_a_genuinely_new_frame_is_still_decoded_and_advances(tmp_path, monkeypatch):
    monkeypatch.setattr(isaac_sensors, 'load_pose_bridge', lambda root: FixturePoseBridge())
    adapter, camera, motion, contact, decode_calls, frame = _rig(tmp_path)

    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    adapter.capture(1.05, 'backstock_00', 'backstock_00', motion, contact)  # reuse
    assert len(decode_calls) == 1

    motion.collision_time = 2.0                                            # mapper kept incorporating
    camera.frame = dict(frame, rendering_frame=2, rendering_time=2.)        # next render tick
    adapter.capture(2.00, 'backstock_00', 'backstock_00', motion, contact)
    assert len(decode_calls) == 2                                           # fresh frame decoded


def test_repeat_refused_once_the_frame_ages_out_of_the_freshness_window(tmp_path, monkeypatch):
    monkeypatch.setattr(isaac_sensors, 'load_pose_bridge', lambda root: FixturePoseBridge())
    adapter, camera, motion, contact, decode_calls, frame = _rig(tmp_path)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    # now - stamp = 0.6 s > the 0.5 s budget: the stride-paced frame is no longer fresh.
    with pytest.raises(ValueError, match='no new synchronized'):
        adapter.capture(1.60, 'backstock_00', 'backstock_00', motion, contact)


def test_stale_refusal_is_named_and_measured_under_the_diag_flag(tmp_path, monkeypatch, capsys):
    """RESTOCK_RENDER_SYNC_DIAG distinguishes the three 'no new synchronized' gates + prints the
    magnitude, so the flaky COLD-collect render-sync trip can be pinned to a gate from the run log.

    The stale gate is the confirmed cold-collect case: the source render did not deliver a fresh
    frame within 0.5 s of sim time (it only starts rendering at the collect boundary, renders every
    sensing_stride ticks, and RTX completes asynchronously), so now-stamp (0.6 s) exceeds the budget.
    """
    monkeypatch.setenv('RESTOCK_RENDER_SYNC_DIAG', '1')
    monkeypatch.setattr(isaac_sensors, 'load_pose_bridge', lambda root: FixturePoseBridge())
    adapter, camera, motion, contact, decode_calls, frame = _rig(tmp_path)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    with pytest.raises(ValueError, match='no new synchronized'):
        adapter.capture(1.60, 'backstock_00', 'backstock_00', motion, contact)  # now-stamp 0.6 > 0.5
    line = capsys.readouterr().out
    assert 'RENDER_SYNC_DIAG reject gate=stale' in line
    assert 'budget=0.5' in line and 'delta=0.6' in line  # the measured staleness, for the run log


def test_future_frame_refusal_is_named_under_the_diag_flag(tmp_path, monkeypatch, capsys):
    """The future/non-finite gate (render clock leads now by > _RENDER_LEAD) names itself distinctly."""
    monkeypatch.setenv('RESTOCK_RENDER_SYNC_DIAG', '1')
    monkeypatch.setattr(isaac_sensors, 'load_pose_bridge', lambda root: FixturePoseBridge())
    adapter, camera, motion, contact, decode_calls, frame = _rig(tmp_path)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    camera.frame = dict(frame, rendering_frame=2, rendering_time=5.0)   # render clock far ahead of now
    with pytest.raises(ValueError, match='no new synchronized'):
        adapter.capture(1.10, 'backstock_00', 'backstock_00', motion, contact)  # now-stamp = -3.9
    assert 'RENDER_SYNC_DIAG reject gate=future' in capsys.readouterr().out


def test_diag_is_silent_and_byte_identical_when_unset(tmp_path, monkeypatch, capsys):
    """Default (flag unset): the same ValueError raises with NO diagnostic line -- byte-identical."""
    monkeypatch.delenv('RESTOCK_RENDER_SYNC_DIAG', raising=False)
    monkeypatch.setattr(isaac_sensors, 'load_pose_bridge', lambda root: FixturePoseBridge())
    adapter, camera, motion, contact, decode_calls, frame = _rig(tmp_path)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    with pytest.raises(ValueError, match='no new synchronized'):
        adapter.capture(1.60, 'backstock_00', 'backstock_00', motion, contact)
    assert 'RENDER_SYNC_DIAG' not in capsys.readouterr().out


def test_repeat_with_tampered_pixels_falls_through_to_replay_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(isaac_sensors, 'load_pose_bridge', lambda root: FixturePoseBridge())
    adapter, camera, motion, contact, decode_calls, frame = _rig(tmp_path)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    # Same frame id/stamp but mutated pixels: the exact-frame key no longer matches, so the frame
    # is NOT reused; it falls to the context provider whose decoder refuses the replayed frame id.
    tampered = frame['rgb'].copy()
    tampered[0, 0, 0] ^= 1
    camera.frame = dict(frame, rgb=tampered)
    # now within 0.05 s of the frame stamp so the non-reuse path clears the sync gate and the
    # decoder (not the stride shortcut) is the thing that refuses the replayed frame id.
    with pytest.raises(ValueError, match='replayed'):
        adapter.capture(1.02, 'backstock_00', 'backstock_00', motion, contact)


def test_repeat_refused_when_no_longer_parked(tmp_path, monkeypatch):
    monkeypatch.setattr(isaac_sensors, 'load_pose_bridge', lambda root: FixturePoseBridge())
    adapter, camera, motion, contact, decode_calls, frame = _rig(tmp_path)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    motion.parked_ = False                                # the base moved since the last render
    with pytest.raises(ValueError, match='moving'):
        adapter.capture(1.05, 'backstock_00', 'backstock_00', motion, contact)


def test_live_clearance_gate_still_enforced_on_a_reused_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(isaac_sensors, 'load_pose_bridge', lambda root: FixturePoseBridge())
    adapter, camera, motion, contact, decode_calls, frame = _rig(tmp_path)
    adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
    # The frame is byte-identical and fresh, but the collision map has not incorporated clearance
    # recently (now - collision_time = 0.8 s > 0.5 s): the live clearance gate must still refuse.
    motion.collision_time = 0.20
    with pytest.raises(ValueError, match='collision map has not incorporated'):
        adapter.capture(1.00, 'backstock_00', 'backstock_00', motion, contact)
