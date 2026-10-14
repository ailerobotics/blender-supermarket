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


def test_station_command_stamped_in_stage_clock_domain_not_mission_time():
    # The caller passes the mission clock (zeroed at the first action); the stamp must instead be
    # the /clock stage time the host ack validates against. stage_clock overrides the passed now.
    ch = FakeChannel()
    st = StationClient(ch, wall_clock=lambda c=itertools.count(100., .001): next(c),
                       stage_clock=lambda: 42.0)
    st.begin('restock_camera_arm', now=0.233)          # mission time 0.233 -> ignored for the stamp
    assert ch.sent[-1]['action'] == 'prepare'
    assert ch.sent[-1]['stamp'] == 42.0                # stamped from the stage clock (no publish_lag)


def test_publish_lag_past_dates_the_command_stamp_for_a_trailing_ack_clock():
    # The ack's /clock can lag the just-sampled stage time (command outruns the /clock update), so
    # publish_lag past-dates the stamp to keep the ack's now() >= stamp within its 0.5 s window.
    ch = FakeChannel()
    st = StationClient(ch, wall_clock=lambda c=itertools.count(100., .001): next(c),
                       stage_clock=lambda: 42.0, publish_lag=0.1)
    st.begin('restock_camera_arm', now=0.0)
    assert ch.sent[-1]['stamp'] == pytest.approx(41.9)  # 42.0 stage - 0.1 lag
    # An ack whose now() trails the stage clock by ~one step still clears 0 <= now - stamp <= 0.5:
    assert 0 <= (42.0 - 0.03) - ch.sent[-1]['stamp'] <= 0.5


def test_heartbeat_is_wall_paced_not_stage_paced():
    # The ack faults if no accepted command arrives within 1 s WALL, but stage time advances slowly
    # under render-heavy stepping. Heartbeats must fire on WALL advance even when stage barely moves.
    ch = FakeChannel()
    wall = {'t': 1000.0}
    stage = {'t': 5.0}
    st = StationClient(ch, wall_clock=lambda: wall['t'], stage_clock=lambda: stage['t'])
    st.begin('restock_camera_arm', now=0.0)                  # first send (prepare)
    n0 = len(ch.sent)
    stage['t'] += 0.001; wall['t'] += 0.4                    # stage ~still, wall past the 0.3 gate
    st.tick(0.0, parked=True)
    assert len(ch.sent) == n0 + 1                            # heartbeat fired on wall advance
    stage['t'] += 0.001; wall['t'] += 0.1                    # wall under the gate -> no send
    st.tick(0.0, parked=True)
    assert len(ch.sent) == n0 + 1


def test_sequence_discipline_one_prepare_then_increasing_heartbeats_regardless_of_reply():
    # The ack accepts heartbeats once it has accepted the prepare and ignores a duplicate prepare
    # (seq<=last); so send exactly ONE prepare (seq 0) then heartbeats with a strictly increasing
    # sequence on the wall cadence, WITHOUT waiting for a reply.
    ch = FakeChannel()
    wall = {'t': 1000.0}
    st = StationClient(ch, wall_clock=lambda: wall['t'])
    st.begin('restock_camera_arm', now=5.0)
    assert ch.sent[-1]['action'] == 'prepare' and ch.sent[-1]['sequence'] == 0
    sent = []
    for i in range(3):
        wall['t'] += 0.4                       # past the 0.3 s wall gate; no reply set
        st.tick(5.0 + 0.01 * i, parked=True)
        sent.append((ch.sent[-1]['action'], ch.sent[-1]['sequence']))
    assert all(a == 'heartbeat' for a, _ in sent), sent      # never a duplicate prepare
    seqs = [s for _, s in sent]
    assert seqs == sorted(seqs) and len(set(seqs)) == 3 and min(seqs) >= 1, seqs  # strictly increasing >0


def test_failed_reply_for_current_epoch_reprepares_new_epoch_then_fails_bounded():
    ch = FakeChannel()
    wall = {'t': 1000.0}
    st = StationClient(ch, wall_clock=lambda: wall['t'])
    st.begin('restock_camera_arm', now=5.0)
    start_epoch = st.epoch
    for _ in range(2):                          # _MAX_RESTARTS re-prepares under fresh epochs
        ch.snapshot = {'state': 'failed', 'stamp': 5.0, 'collision_time': -1.0,
                       'epoch': st.epoch, 'reason': 'boom'}
        wall['t'] += 0.4
        st.tick(5.0, parked=True)
    assert st.epoch > start_epoch               # epoch advanced on re-prepare
    ch.snapshot = {'state': 'failed', 'stamp': 5.0, 'collision_time': -1.0,
                   'epoch': st.epoch, 'reason': 'boom'}
    wall['t'] += 0.4
    with pytest.raises(RuntimeError):           # bounded: surfaces the reason after _MAX_RESTARTS
        st.tick(5.0, parked=True)


def test_ack_accepts_stage_stamped_prepare_within_half_second_and_refuses_beyond():
    # Firstmate contract: a prepare stamped from the stage clock clears the ack's 0.5 s freshness
    # guard at 0 and 0.4 s of skew (ack /clock now >= stamp) and is refused at 0.6 s.
    from scripts.twin.restock_host.incorporation_ack import IncorporationAckState

    def accept_at(skew):
        st = IncorporationAckState(wall_clock=lambda: 100.)
        stamp = 5.0
        cmd = dict(action='prepare', epoch=1, sequence=0, stamp=stamp, parked=True,
                   camera_frame='restock_camera_arm')
        return st.accept(cmd, now=stamp + skew)

    assert accept_at(0.0) == 'prepare'
    assert accept_at(0.4) == 'prepare'
    with pytest.raises(ValueError):
        accept_at(0.6)


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


class FlappyFeedback:
    """Instantaneous parked() flaps False (angular noise); set_parked overrides with the port's
    robust net-pose parked, which is what the station/capture must use."""
    def __init__(self):
        self.collision_time = -1.
        self._port = None
    def set_parked(self, value):
        self._port = None if value is None else bool(value)
    def parked(self):
        return False if self._port is None else self._port


def test_robust_parked_from_port_overrides_flappy_velocity_for_the_station():
    st = _client()
    fb = FlappyFeedback()
    rt = RosManipulationRuntime(FakeGraph(), lambda *a: None, prepare_station=lambda op, job: None,
                                feedback=fb, station=st, station_camera_frame='restock_camera_arm')
    rt.start('collect', {'source_bin': 'backstock_00', 'job_id': 'j1'})
    # The port reports robustly parked even though the instantaneous velocity would read moving:
    # begin() prepares the station without raising 'station lost measured parking'.
    r1 = rt.step(1 / 60., 8.0, parked=True)
    assert r1.status == 'running' and fb.parked() is True
    # A genuine not-parked from the port stops the station (fatal), as designed.
    r2 = rt.step(1 / 60., 8.1, parked=False)
    assert r2.status == 'failed'


def test_warm_pumps_station_before_dispatch_and_reports_ready_on_fresh_snapshot():
    st = _client()
    feedback = NS(collision_time=-1., parked=lambda: True, set_parked=lambda v: None)
    graph = FakeGraph()
    rt = RosManipulationRuntime(graph, lambda *a: None, prepare_station=lambda op, job: None,
                                feedback=feedback, station=st, station_camera_frame='restock_camera_arm')
    # First warm begins the station (prepare) but reports not-ready and never dispatches the job.
    assert rt.warm(8.0, parked=True) is False
    assert any(c['action'] == 'prepare' for c in st.channel.sent)
    assert feedback.collision_time == -1.
    assert graph.started is None                              # warm must NOT dispatch the grasp
    # A fresh ack snapshot -> warm reports ready and mirrors collision_time onto the feedback.
    st.channel.snapshot = {'state': 'ready', 'stamp': 8.1, 'collision_time': 8.08, 'epoch': st.epoch}
    assert rt.warm(8.12, parked=True) is True
    assert feedback.collision_time == pytest.approx(8.08)
    assert graph.started is None


def test_start_after_warm_keeps_the_warmed_station_and_dispatches():
    st = _client()
    feedback = NS(collision_time=-1., parked=lambda: True, set_parked=lambda v: None)
    graph = FakeGraph()
    rt = RosManipulationRuntime(graph, lambda *a: None, prepare_station=lambda op, job: None,
                                feedback=feedback, station=st, station_camera_frame='restock_camera_arm')
    rt.warm(8.0, parked=True)                                 # station begun during warm-up
    assert rt._station_started is True
    rt.start('collect', {'source_bin': 'backstock_00', 'job_id': 'j1'})
    assert rt._station_started is True                        # start() must not re-begin (re-cold) it
    assert graph.started == ('collect', {'source_bin': 'backstock_00', 'job_id': 'j1'})


def test_runtime_clears_collision_time_on_stop():
    st = _client()
    feedback = NS(collision_time=5.0, parked=lambda: True)
    rt = RosManipulationRuntime(FakeGraph(), lambda *a: None, prepare_station=lambda op, job: None,
                                feedback=feedback, station=st, station_camera_frame='restock_camera_arm')
    rt.start('collect', {'source_bin': 'backstock_00', 'job_id': 'j1'})
    rt.step(1 / 60., 8.0)
    rt.stop()
    assert feedback.collision_time == -1.                     # never leave a stale stamp after stop
