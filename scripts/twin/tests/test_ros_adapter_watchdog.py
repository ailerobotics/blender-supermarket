"""CPU tests for RosTransferBridge's split sensor-stream bound (host-lane watchdog decision).

The tight per-frame watchdog (1 s) protects a MOVING arm and must only arm AFTER the first
/sensors frame; before any frame a looser startup grace (5 s) bounds a job that never produces
sensors, because cold first-ESDF incorporation is ~1 s and unachievable from a cold job start.
No ROS node: the bridge is built via __new__ and only its _watchdog logic is exercised with a
fake controller + a stubbed _emit (the omni/rclpy seams live in __init__).
"""
import time
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock_manipulation.ros_adapter import RosTransferBridge


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _bridge(*, phase='observe', first_frame=False, job_age=None, packet_age=None):
    b = RosTransferBridge.__new__(RosTransferBridge)
    b.watchdog_seconds = 1.0
    b.startup_grace_seconds = 5.0
    b.first_frame = first_frame
    now = time.monotonic()
    b.job_start_wall = None if job_age is None else now - job_age
    b.last_packet_wall = now if packet_age is None else now - packet_age
    emitted = []
    b._emit = lambda result: emitted.append(result)
    b.controller = NS(phase=phase, _fail=lambda reason: NS(reason=reason, controls=()))
    return b, emitted


class StartupGrace(unittest.TestCase):
    def test_no_initial_frame_within_grace_does_not_fail(self):
        b, emitted = _bridge(first_frame=False, job_age=2.0, packet_age=3.0)
        b._watchdog()
        self.assertEqual(emitted, [])            # 1 s not armed (no first frame); grace not expired

    def test_no_initial_frame_past_grace_fails_with_distinct_reason(self):
        b, emitted = _bridge(first_frame=False, job_age=6.0, packet_age=6.0)
        b._watchdog()
        self.assertEqual([e.reason for e in emitted], ['no initial sensor frame'])


class PerFrameWatchdog(unittest.TestCase):
    def test_arms_only_after_first_frame_then_trips_on_stall(self):
        # Fresh first frame, fresh stream -> no trip.
        b, emitted = _bridge(first_frame=True, packet_age=0.1)
        b._watchdog()
        self.assertEqual(emitted, [])
        # First frame seen, stream stalled > 1 s -> the tight watchdog trips.
        b, emitted = _bridge(first_frame=True, packet_age=2.0)
        b._watchdog()
        self.assertEqual([e.reason for e in emitted], ['sensor stream watchdog expired'])

    def test_terminal_phase_never_trips(self):
        for phase in ('idle', 'failed', 'succeeded'):
            b, emitted = _bridge(phase=phase, first_frame=False, job_age=99.0, packet_age=99.0)
            b._watchdog()
            self.assertEqual(emitted, [], phase)


if __name__ == '__main__':
    unittest.main()
