"""Consumer side of the host nvblox mapper-incorporation ack (gate 7 / collision_time).

The host `incorporation_ack` speaks the parked-station protocol: it reads prepare/heartbeat on
/restock_sim/station/command and publishes a snapshot (incl. collision_time) on
/restock_sim/station/status. The mission drives that via `StationClient` and
`RosManipulationRuntime` mirrors `station.collision_time` onto `IsaacMeasuredFeedback` BEFORE each
capture. Pure CPU with a fake channel + fake graph; the OmniGraph JsonChannel is the only GPU piece.
"""
import itertools

import pytest

from scripts.twin.restock.native_channel import StationClient
from scripts.twin.restock.ros_manipulation import RosManipulationRuntime


class FakeChannel:
    def __init__(self):
        self.sent = []
        self.snapshot = None
    def publish(self, value):
        self.sent.append(value)
    def read(self):
        return self.snapshot


class FakeGraph:
    def __init__(self):
        self.started = None
    def start(self, operation, job):
        self.started = (operation, job)
    def poll(self):
        return None
    def close(self):
        pass


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _client():
    return StationClient(FakeChannel(), wall_clock=lambda c=itertools.count(100., .001): next(c))


def test_station_client_reads_collision_time_from_a_ready_snapshot():
    st = _client()
    st.begin('restock_camera_arm', now=8.0)
    assert st.channel.sent[-1]['action'] == 'prepare'          # prepared the ack station
    st.channel.snapshot = {'state': 'ready', 'stamp': 8.02, 'collision_time': 8.0, 'epoch': st.epoch}
    st.tick(8.1, parked=True)
    assert st.ready(8.1) and st.collision_time == pytest.approx(8.0)


def test_station_collision_time_is_minus_one_until_ready_or_when_stale():
    st = _client()
    st.begin('restock_camera_arm', now=8.0)
    assert st.collision_time == -1.                            # no reply yet
    st.channel.snapshot = {'state': 'ready', 'stamp': 2.0, 'collision_time': 2.0, 'epoch': st.epoch}
    with pytest.raises(ValueError):                            # a stale (now-stamp>0.5) reply is refused
        st.tick(8.5, parked=True)


def test_runtime_mirrors_station_collision_time_onto_feedback_before_capture():
    st = _client()
    feedback = NS(collision_time=-1., parked=lambda: True)
    seen = {}
    def capture(now, source, destination):
        seen['ct'] = feedback.collision_time                  # capture reads the mirrored value
        return None
    rt = RosManipulationRuntime(FakeGraph(), capture, prepare_station=lambda op, job: None,
                                feedback=feedback, station=st, station_camera_frame='restock_camera_arm')
    rt.start('collect', {'source_bin': 'backstock_00', 'job_id': 'j1'})
    rt.step(1 / 60., 8.0)                                      # first step prepares the station
    assert any(c['action'] == 'prepare' for c in st.channel.sent)
    assert feedback.collision_time == -1.                     # ack not ready yet
    st.channel.snapshot = {'state': 'ready', 'stamp': 8.5, 'collision_time': 8.48, 'epoch': st.epoch}
    rt.step(1 / 60., 8.52)                                     # heartbeat + mirror
    assert feedback.collision_time == pytest.approx(8.48)     # mirrored from the ack
    assert seen['ct'] == pytest.approx(8.48)                  # fresh for the capture this tick


def test_runtime_clears_collision_time_on_stop():
    st = _client()
    feedback = NS(collision_time=5.0, parked=lambda: True)
    rt = RosManipulationRuntime(FakeGraph(), lambda *a: None, prepare_station=lambda op, job: None,
                                feedback=feedback, station=st, station_camera_frame='restock_camera_arm')
    rt.start('collect', {'source_bin': 'backstock_00', 'job_id': 'j1'})
    rt.step(1 / 60., 8.0)
    rt.stop()
    assert feedback.collision_time == -1.                     # never leave a stale stamp after stop
