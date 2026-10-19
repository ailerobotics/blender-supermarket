"""CPU tests for RosTransferBridge's split sensor-stream bound (host-lane watchdog decision).

The tight per-frame watchdog (1 s) protects a MOVING arm and arms only AFTER the first
/sensors frame; before any frame a looser 5 s startup grace bounds a job that never produces
sensors. After the first frame the per-frame bound is measured in NODE-CLOCK (sim) time, so a
render-paced /sensors cadence (1.7-3.5 s apart in WALL time but a small step in SIM time) does
not false-trip, while a wall-time backstop still hard-stops a frozen clock. No ROS node: the
bridge is built via __new__ and only _watchdog is exercised with a fake controller, a fake node
clock, and a stubbed _emit (the omni/rclpy seams live in __init__).
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


class _FakeClock:
    """Minimal stand-in for node.get_clock().now().nanoseconds under use_sim_time."""
    def __init__(self, sim_seconds):
        self._ns = int(sim_seconds * 1e9)

    def get_clock(self):
        return self

    def now(self):
        return NS(nanoseconds=self._ns)


SIM_NOW = 1000.0  # fixed fake node-clock reading (seconds)


def _bridge(*, phase='observe', first_frame=False, job_age=None,
            packet_wall_age=0.1, sim_gap=0.0, wall_backstop=30.0):
    b = RosTransferBridge.__new__(RosTransferBridge)
    b.watchdog_seconds = 1.0
    b.startup_grace_seconds = 5.0
    b.wall_backstop_seconds = wall_backstop
    b.first_frame = first_frame
    now = time.monotonic()
    b.job_start_wall = None if job_age is None else now - job_age
    b.last_packet_wall = now - packet_wall_age
    # last_packet_sim exists only once a frame has armed the per-frame bound.
    b.last_packet_sim = (SIM_NOW - sim_gap) if first_frame else None
    b.node = _FakeClock(SIM_NOW)
    emitted = []
    b._emit = lambda result: emitted.append(result)
    b.controller = NS(phase=phase, _fail=lambda reason: NS(reason=reason, controls=()),
                      retry_or_fail_move=lambda reason: NS(reason=reason, controls=()))
    return b, emitted


class StartupGrace(unittest.TestCase):
    def test_no_initial_frame_within_grace_does_not_fail(self):
        b, emitted = _bridge(first_frame=False, job_age=2.0, packet_wall_age=3.0)
        b._watchdog()
        self.assertEqual(emitted, [])            # 1 s not armed (no first frame); grace not expired

    def test_no_initial_frame_past_grace_fails_with_distinct_reason(self):
        b, emitted = _bridge(first_frame=False, job_age=6.0, packet_wall_age=6.0)
        b._watchdog()
        self.assertEqual([e.reason for e in emitted], ['no initial sensor frame'])


class PerFrameWatchdog(unittest.TestCase):
    def test_fresh_frame_no_trip(self):
        b, emitted = _bridge(first_frame=True, sim_gap=0.1, packet_wall_age=0.1)
        b._watchdog()
        self.assertEqual(emitted, [])

    def test_sim_time_stall_trips(self):
        # Wall is FRESH (0.1 s) but the sim clock has advanced 2 s since the last frame: the
        # bound is measured in node-clock time, so it trips -- proving it is sim-time, not wall.
        b, emitted = _bridge(first_frame=True, sim_gap=2.0, packet_wall_age=0.1)
        b._watchdog()
        self.assertEqual([e.reason for e in emitted], ['sensor stream watchdog expired'])

    def test_render_paced_wall_gap_does_not_false_trip(self):
        # The run-74 failure: 3.5 s WALL between render-paced captures but only 0.05 s of SIM
        # time. The node-clock bound must NOT trip (and the 30 s wall backstop is nowhere near).
        b, emitted = _bridge(first_frame=True, sim_gap=0.05, packet_wall_age=3.5)
        b._watchdog()
        self.assertEqual(emitted, [])

    def test_terminal_phase_never_trips(self):
        for phase in ('idle', 'failed', 'succeeded'):
            b, emitted = _bridge(phase=phase, first_frame=True, sim_gap=99.0, packet_wall_age=99.0)
            b._watchdog()
            self.assertEqual(emitted, [], phase)


class WallBackstop(unittest.TestCase):
    def test_frozen_clock_backstop_trips(self):
        # Sim clock frozen (no sim gap) so the node-clock bound can never trip, but wall time has
        # advanced past the 30 s backstop -> a stalled simulator is still hard-stopped.
        b, emitted = _bridge(first_frame=True, sim_gap=0.0, packet_wall_age=31.0, wall_backstop=30.0)
        b._watchdog()
        self.assertEqual(len(emitted), 1)
        # reason now carries the measured gap vs backstop for diagnosis (fm362 gap logging).
        self.assertTrue(emitted[0].reason.startswith('sensor stream wall backstop expired'), emitted[0].reason)
        self.assertIn('gap=31.0s > 30.0s', emitted[0].reason)

    def test_within_backstop_does_not_trip(self):
        # Frozen clock, but within the wall backstop and within the sim bound -> no trip yet.
        b, emitted = _bridge(first_frame=True, sim_gap=0.0, packet_wall_age=10.0, wall_backstop=30.0)
        b._watchdog()
        self.assertEqual(emitted, [])


class PlanningHangSurfacesBeforeBackstop(unittest.TestCase):
    # run-220/281: a planning hang must surface as a planning failure within the total planning-wall
    # cap even when /sensors has stalled (poll() no longer driven), NOT get masked by the 30 s sensor
    # backstop. The 0.1 s watchdog checks the motion's read-only planning_wall_exceeded() flag.
    def test_planning_wall_exceeded_trips_even_with_fresh_sensors(self):
        b, emitted = _bridge(first_frame=True, sim_gap=0.0, packet_wall_age=0.1, wall_backstop=30.0)
        b.controller.motion = NS(planning_wall_exceeded=lambda: True)
        b._watchdog()
        self.assertEqual([e.reason for e in emitted], ['planning failed (exhausted within wall cap)'])

    def test_no_trip_while_planning_within_cap(self):
        b, emitted = _bridge(first_frame=True, sim_gap=0.0, packet_wall_age=0.1, wall_backstop=30.0)
        b.controller.motion = NS(planning_wall_exceeded=lambda: False)
        b._watchdog()
        self.assertEqual(emitted, [])

    def test_absent_motion_is_tolerated(self):
        b, emitted = _bridge(first_frame=True, sim_gap=0.05, packet_wall_age=0.1)   # no .motion attr
        b._watchdog()
        self.assertEqual(emitted, [])

    def test_failed_action_result_surfaces_even_when_sensors_stall(self):
        # run-224: a LIFT GOAL_MISS aborts the executor, but lift suspends sensing so /sensors stalls
        # and poll() never drains the failed result -> the watchdog must surface it (not hang to cap).
        b, emitted = _bridge(first_frame=True, sim_gap=0.0, packet_wall_age=0.1, wall_backstop=30.0)
        b.controller.motion = NS(planning_wall_exceeded=lambda: False,
                                 failed_result_pending=lambda: True)
        b._watchdog()
        self.assertEqual([e.reason for e in emitted], ['arm motion failed (goal tolerance not reached)'])

    def test_healthy_execution_not_faulted(self):
        b, emitted = _bridge(first_frame=True, sim_gap=0.05, packet_wall_age=3.5, wall_backstop=30.0)
        b.controller.motion = NS(planning_wall_exceeded=lambda: False,
                                 failed_result_pending=lambda: False)
        b._watchdog()
        self.assertEqual(emitted, [])


if __name__ == '__main__':
    unittest.main()
