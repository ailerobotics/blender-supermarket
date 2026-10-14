"""Consumer side of the nvblox mapper-incorporation ack: collision_time into the sensing motion.

The host publishes a fresh `{collision_time, stamp}` ack on /restock_sim/mapper/incorporation;
`parse_incorporation` turns it into a collision_time (or -1 when absent/stale/invalid), and
`RosManipulationRuntime.step` mirrors it onto the measured feedback before each capture. Pure CPU
(fake ack string + fake graph/feedback); the OmniGraph read + the host publish are the only GPU
pieces left. The gate exists to prove the ESDF is current, so nothing is fabricated to `now`.
"""
import pytest

from scripts.twin.restock_manipulation.isaac_ros_graph import parse_incorporation
from scripts.twin.restock.ros_manipulation import RosManipulationRuntime


def test_parse_incorporation_fresh_ack_returns_collision_time():
    assert parse_incorporation('{"collision_time": 12.3, "stamp": 12.34}', now=12.5) == pytest.approx(12.3)


def test_parse_incorporation_defaults_stamp_to_collision_time():
    assert parse_incorporation('{"collision_time": 5.0}', now=5.2) == pytest.approx(5.0)


@pytest.mark.parametrize('raw,now', [
    ('', 1.0),                                      # no ack yet
    ('not json', 1.0),                              # malformed
    ('{"stamp": 1.0}', 1.0),                        # missing collision_time
    ('{"collision_time": 1.0, "stamp": 0.4}', 1.0), # stale: now-stamp = 0.6 > 0.5
    ('{"collision_time": 1.0, "stamp": 2.0}', 1.0), # future: now-stamp = -1.0
    ('{"collision_time": "x", "stamp": 1.0}', 1.0), # non-numeric
    ('[1,2,3]', 1.0),                               # not an object
])
def test_parse_incorporation_rejects_absent_stale_or_invalid(raw, now):
    assert parse_incorporation(raw, now) == -1.


class _FakeGraph:
    def __init__(self, ct):
        self._ct = ct
        self.refreshed_at = None
        self.started = None
    def refresh_incorporation(self, now):
        self.refreshed_at = now
        return self._ct
    def start(self, operation, job):
        self.started = (operation, job)
    def poll(self):
        return None
    def close(self):
        pass


class _FakeFeedback:
    collision_time = -1.


def test_step_mirrors_mapper_collision_time_onto_feedback_before_capture():
    seen = {}
    def capture(now, source, destination):
        seen['collision_time_at_capture'] = feedback.collision_time   # capture reads the mirrored value
        return None
    graph = _FakeGraph(ct=7.77)
    feedback = _FakeFeedback()
    rt = RosManipulationRuntime(graph, capture, prepare_station=lambda op, job: None, feedback=feedback)
    rt.start('collect', {'source_bin': 'backstock_00', 'job_id': 'j1'})
    result = rt.step(1/60., 8.0)
    assert graph.refreshed_at == 8.0
    assert feedback.collision_time == pytest.approx(7.77)          # mirrored from the ack
    assert seen['collision_time_at_capture'] == pytest.approx(7.77) # fresh for the capture
    assert result.status == 'running'


def test_step_without_feedback_is_a_noop_for_collision_time():
    graph = _FakeGraph(ct=1.0)
    rt = RosManipulationRuntime(graph, lambda *a: None, prepare_station=lambda op, job: None)
    rt.start('collect', {'source_bin': 'backstock_00', 'job_id': 'j1'})
    rt.step(1/60., 2.0)
    assert graph.refreshed_at is None                              # not called when no feedback
