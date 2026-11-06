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
from twin.restock_manipulation.isaac_sensors import sensing_render_required


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


def _bridge(*, phase='observe', first_frame=False, job_age=None, job_sim_age=None,
            packet_wall_age=0.1, sim_gap=0.0, wall_backstop=30.0,
            startup_grace_sim=5.0, startup_grace_wall_backstop=250.0,
            transit_pause_wall=1200.0, transit_start_age=None,
            suspend_place_descent_watchdog=False, held=False,
            suspend_capture_silent_watchdog=False):
    b = RosTransferBridge.__new__(RosTransferBridge)
    b.watchdog_seconds = 1.0
    b.startup_grace_seconds = 5.0
    # sim-place-descent-watchdog-suspend-v1: phase-scoped per-frame SIM-bound suspension over the loaded
    # place descent (off by default => byte-identical).
    b.suspend_place_descent_watchdog = suspend_place_descent_watchdog
    # sim-retreat-sensor-watchdog-scope-v1: broad capture-silent suspension (off by default => byte-identical).
    b.suspend_capture_silent_watchdog = suspend_capture_silent_watchdog
    b._place_watchdog_suspended = False
    b._watchdog_suspend_reason = None
    # sim-robust-host-timers-audit-fix-v1: SIM-primary startup grace + generous frozen-sim wall backstop.
    b.startup_grace_sim_seconds = startup_grace_sim
    b.startup_grace_wall_backstop_s = startup_grace_wall_backstop
    b.wall_backstop_seconds = wall_backstop
    b.transit_pause_wall_s = transit_pause_wall
    b._transit_pause_start_wall = (None if transit_start_age is None
                                   else time.monotonic() - transit_start_age)
    b.first_frame = first_frame
    now = time.monotonic()
    b.job_start_wall = None if job_age is None else now - job_age
    # The SIM job-start anchor: defaults to mirror the wall age (sim == wall) so existing tests keep their
    # semantics; a test sets job_sim_age independently to exercise render pacing.
    _sim_age = job_age if job_sim_age is None else job_sim_age
    b.job_start_sim = None if _sim_age is None else SIM_NOW - _sim_age
    b.last_packet_wall = now - packet_wall_age
    # last_packet_sim exists only once a frame has armed the per-frame bound.
    b.last_packet_sim = (SIM_NOW - sim_gap) if first_frame else None
    b.node = _FakeClock(SIM_NOW)
    emitted = []
    b._emit = lambda result: emitted.append(result)
    b.controller = NS(phase=phase, _fail=lambda reason: NS(reason=reason, controls=()),
                      retry_or_fail_move=lambda reason: NS(reason=reason, controls=()),
                      in_loaded_place_descent=lambda: phase == 'place' and bool(held),
                      # sim-retreat-sensor-watchdog-scope-v1: the authoritative capture-silent predicate,
                      # driven by the SAME sensing_render_required set the production controller delegates to.
                      in_capture_silent_window=lambda: not sensing_render_required(b.controller.phase))
    return b, emitted


class StartupGrace(unittest.TestCase):
    def test_no_initial_frame_within_grace_does_not_fail(self):
        b, emitted = _bridge(first_frame=False, job_age=2.0, packet_wall_age=3.0)
        b._watchdog()
        self.assertEqual(emitted, [])            # SIM gap 2 s < 5 s grace; not armed

    def test_no_initial_frame_past_grace_fails_with_distinct_reason(self):
        # Real rig (sim == wall): SIM gap 6 s > 5 s grace -> fire. Reason names the sim/wall gaps.
        b, emitted = _bridge(first_frame=False, job_age=6.0, job_sim_age=6.0, packet_wall_age=6.0)
        b._watchdog()
        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0].reason.startswith('no initial sensor frame'))

    def test_render_paced_startup_does_not_false_trip(self):
        # sim-robust-host-timers-audit-fix-v1: the fm465 light rig slows the first full-res frame, so the
        # WALL job age runs far ahead of the SIM age. SIM gap 2 s < 5 s grace and wall gap 200 s < 250 s
        # backstop -> must NOT fire (the old pure-wall 5 s grace false-tripped here before the env widen).
        b, emitted = _bridge(first_frame=False, job_age=200.0, job_sim_age=2.0, packet_wall_age=200.0)
        b._watchdog()
        self.assertEqual(emitted, [])

    def test_frozen_sim_startup_wall_backstop_fires(self):
        # A startup whose /clock is FROZEN (sim gap stuck at ~0) and no frame ever arrives: the SIM leg
        # can never advance, so the generous frozen-sim WALL backstop still hard-stops it.
        b, emitted = _bridge(first_frame=False, job_age=300.0, job_sim_age=0.0, packet_wall_age=300.0)
        b._watchdog()
        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0].reason.startswith('no initial sensor frame'))


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


class TransitSensingPause(unittest.TestCase):
    # fm490 (cold GT-collect): the fm390 reposition carry-drive holds the arm in TRANSIT and pauses the
    # station while the base drives to place_approach, so /sensors legitimately stops for the whole
    # render-paced drive (~31 s sim / ~740 s wall). Both the per-frame SIM bound AND the wall backstop
    # must be suspended for the pause, under a generous wall budget that still catches a never-arriving
    # drive.
    def test_sim_stall_does_not_trip_during_transit(self):
        # Exactly the cold-run trip: a huge SIM gap that WOULD fire 'sensor stream watchdog expired' in
        # any other phase is suspended while phase=='transit'.
        b, emitted = _bridge(phase='transit', first_frame=True, sim_gap=31.0, packet_wall_age=0.1)
        b._watchdog()
        self.assertEqual(emitted, [])

    def test_wall_backstop_does_not_trip_during_transit(self):
        # The render-paced drive runs ~740 s wall with no frame -- far past the 30 s wall backstop, which
        # must not fire during the intentional pause.
        b, emitted = _bridge(phase='transit', first_frame=True, sim_gap=0.0,
                             packet_wall_age=740.0, wall_backstop=30.0, transit_start_age=740.0)
        b._watchdog()
        self.assertEqual(emitted, [])

    def test_transit_resumes_cleanly_after_pause(self):
        # When transit ends, the next phase sees fresh packet timers (set each transit tick), so a tiny
        # residual gap does not spuriously trip.
        b, emitted = _bridge(phase='transit', first_frame=True, sim_gap=50.0, packet_wall_age=600.0)
        b._watchdog()
        self.assertEqual(emitted, [])
        self.assertIsNotNone(b.last_packet_sim)
        # timers refreshed to 'now' so a following non-transit tick is clean
        self.assertLessEqual(SIM_NOW - b.last_packet_sim, 1e-6)

    def test_transit_budget_exceeded_hard_stops_with_named_reason(self):
        # A reposition that never arrives still hard-stops once the generous transit budget is spent.
        b, emitted = _bridge(phase='transit', first_frame=True, sim_gap=0.0, packet_wall_age=0.1,
                             transit_pause_wall=1200.0, transit_start_age=1300.0)
        b._watchdog()
        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0].reason.startswith(
            'reposition carry-drive exceeded transit sensing-pause budget'), emitted[0].reason)

    def test_transit_start_latches_on_first_transit_tick(self):
        b, emitted = _bridge(phase='transit', first_frame=True, sim_gap=99.0, packet_wall_age=99.0)
        self.assertIsNone(b._transit_pause_start_wall)
        b._watchdog()
        self.assertIsNotNone(b._transit_pause_start_wall)   # latched, no trip this tick
        self.assertEqual(emitted, [])

    def test_leaving_transit_clears_the_pause_latch(self):
        b, emitted = _bridge(phase='preplace', first_frame=True, sim_gap=0.05, packet_wall_age=0.1,
                             transit_start_age=10.0)
        b._watchdog()
        self.assertIsNone(b._transit_pause_start_wall)      # cleared so the next transit starts fresh
        self.assertEqual(emitted, [])


class PlaceDescentSensorWatchdogSuspend(unittest.TestCase):
    # sim-place-descent-watchdog-suspend-v1 (coldwin64): during the loaded place descent (collect set-down /
    # live_pose transfer place) the /sensors stream legitimately stalls under the full-six-axis base lock +
    # deep render pacing, so the per-frame SIM bound would false-trip 'sensor stream watchdog expired'.
    # Opt-in: suspend ONLY the SIM bound over that window; keep the WALL backstop armed so a frozen sim is
    # still caught. Re-armed on release / phase exit. Key off => the legacy behaviour (byte-identical).
    def test_sim_stall_in_loaded_place_descent_does_not_trip_when_enabled(self):
        # The exact coldwin64 trip: a 30 s SIM gap in phase=place with the carton held. Suspended => no fail.
        b, emitted = _bridge(phase='place', held=True, first_frame=True, sim_gap=30.0,
                             packet_wall_age=0.1, suspend_place_descent_watchdog=True)
        b._watchdog()
        self.assertEqual(emitted, [])
        self.assertTrue(b._place_watchdog_suspended)
        # anchor refreshed to 'now' so the resume after the window is clean
        self.assertIsNotNone(b.last_packet_sim)
        self.assertLessEqual(SIM_NOW - b.last_packet_sim, 1e-6)

    def test_same_sim_stall_trips_when_key_off(self):
        # Key off (default): the IDENTICAL place-phase SIM gap still trips -> proves the known coldwin64
        # fault with the key off (byte-identical legacy behaviour).
        b, emitted = _bridge(phase='place', held=True, first_frame=True, sim_gap=30.0,
                             packet_wall_age=0.1, suspend_place_descent_watchdog=False)
        b._watchdog()
        self.assertEqual([e.reason for e in emitted], ['sensor stream watchdog expired'])

    def test_wall_backstop_still_fires_while_suspended(self):
        # A genuinely FROZEN sim inside the window (no frame ever arrives): the SIM bound is suspended but
        # the 180 s wall backstop must still hard-stop it (last_packet_wall is never touched by the suspend).
        b, emitted = _bridge(phase='place', held=True, first_frame=True, sim_gap=30.0,
                             packet_wall_age=181.0, wall_backstop=180.0,
                             suspend_place_descent_watchdog=True)
        b._watchdog()
        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0].reason.startswith('sensor stream wall backstop expired'), emitted[0].reason)
        self.assertIn('gap=181.0s > 180.0s', emitted[0].reason)

    def test_not_suspended_outside_the_window_even_when_enabled(self):
        # Enabled but NOT in the loaded descent (phase=place but carton NOT held -> in_loaded_place_descent
        # False): the SIM bound is live and still trips. Suspension never applies outside the window.
        b, emitted = _bridge(phase='place', held=False, first_frame=True, sim_gap=2.0,
                             packet_wall_age=0.1, suspend_place_descent_watchdog=True)
        b._watchdog()
        self.assertEqual([e.reason for e in emitted], ['sensor stream watchdog expired'])
        self.assertFalse(b._place_watchdog_suspended)

    def test_rearms_on_leaving_the_window(self):
        # Suspend on a held place tick, then leave the window (release): the latch clears so a following
        # non-place tick measures the SIM bound normally (and would trip a real stall there).
        b, emitted = _bridge(phase='place', held=True, first_frame=True, sim_gap=30.0,
                             packet_wall_age=0.1, suspend_place_descent_watchdog=True)
        b._watchdog()
        self.assertTrue(b._place_watchdog_suspended)
        # now in 'release' (held carton released / phase exit): rearm, and a fresh small gap does not trip
        b.controller.phase = 'release'
        b.controller.in_loaded_place_descent = lambda: False
        b.last_packet_sim = SIM_NOW - 0.05
        b._watchdog()
        self.assertEqual(emitted, [])
        self.assertFalse(b._place_watchdog_suspended)


class CaptureSilentSensorWatchdogScope(unittest.TestCase):
    # sim-retreat-sensor-watchdog-scope-v1 (coldwin76/77/78): the runs completed the loaded set-down AND the
    # contact-triggered release on the cart, then each died in the EMPTY RETREAT on a different watchdog
    # counting a legitimately quiet period. run-78's was THIS per-frame /sensors SIM freshness watchdog: the
    # phase-aware collect capture does not fire in retreat/stow by design, a 0.01 s move-deadline miss drove a
    # retry, the retreat ran past 40 sim-s, and the SIM bound expired (HOST_FAULT 'sensor stream watchdog
    # expired', phase=retreat) against RESTOCK_SENSOR_WATCHDOG_S=40. The broad opt-in key suspends the SIM
    # bound over EVERY capture-silent phase (controller.in_capture_silent_window); the WALL backstop stays
    # armed so a frozen sim is still caught. Key off => the legacy behaviour (byte-identical).

    def _capture(self, b):
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            b._watchdog()
        return buf.getvalue()

    def test_retreat_sim_stall_does_not_trip_when_enabled(self):
        # The exact run-78 condition: a 40 s SIM gap in phase=retreat against a 40 s SIM bound, with a
        # deadline-retry-driven slow retreat. Suspended (capture-silent) => no fail.
        b, emitted = _bridge(phase='retreat', first_frame=True, sim_gap=40.1,
                             packet_wall_age=0.1, suspend_capture_silent_watchdog=True)
        b.watchdog_seconds = 40.0
        out = self._capture(b)
        self.assertEqual(emitted, [])
        self.assertTrue(b._place_watchdog_suspended)
        self.assertIn('SENSOR_WATCHDOG_SUSPENDED phase=retreat reason=capture_silent', out)
        # anchor refreshed to 'now' so the resume after the window is clean
        self.assertIsNotNone(b.last_packet_sim)
        self.assertLessEqual(SIM_NOW - b.last_packet_sim, 1e-6)

    def test_same_retreat_stall_trips_when_key_off(self):
        # Key off (default): the IDENTICAL retreat SIM gap still trips -> proves the known run-78 fault with
        # the key off (byte-identical legacy behaviour).
        b, emitted = _bridge(phase='retreat', first_frame=True, sim_gap=40.1,
                             packet_wall_age=0.1, suspend_capture_silent_watchdog=False)
        b.watchdog_seconds = 40.0
        b._watchdog()
        self.assertEqual([e.reason for e in emitted], ['sensor stream watchdog expired'])
        self.assertFalse(b._place_watchdog_suspended)

    def test_stow_sim_stall_does_not_trip_when_enabled(self):
        # stow is capture-silent too (post-place arm return): a long SIM gap there must not trip when on.
        b, emitted = _bridge(phase='stow', first_frame=True, sim_gap=30.0,
                             packet_wall_age=0.1, suspend_capture_silent_watchdog=True)
        out = self._capture(b)
        self.assertEqual(emitted, [])
        self.assertTrue(b._place_watchdog_suspended)
        self.assertIn('SENSOR_WATCHDOG_SUSPENDED phase=stow reason=capture_silent', out)

    def test_wall_backstop_still_fires_in_retreat_while_suspended(self):
        # A genuinely FROZEN sim inside the retreat window: the SIM bound is suspended but the WALL backstop
        # must still hard-stop it (last_packet_wall is never touched by the suspend).
        b, emitted = _bridge(phase='retreat', first_frame=True, sim_gap=30.0,
                             packet_wall_age=601.0, wall_backstop=600.0,
                             suspend_capture_silent_watchdog=True)
        b._watchdog()
        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0].reason.startswith('sensor stream wall backstop expired'), emitted[0].reason)
        self.assertIn('gap=601.0s > 600.0s', emitted[0].reason)

    def test_not_suspended_in_a_capture_bearing_phase_even_when_enabled(self):
        # observe is a capture-BEARING phase (sensing_render_required True): the SIM bound stays live and a
        # real stall there still trips. The broad suspension never applies outside the capture-silent set.
        self.assertTrue(sensing_render_required('observe'))
        b, emitted = _bridge(phase='observe', first_frame=True, sim_gap=2.0,
                             packet_wall_age=0.1, suspend_capture_silent_watchdog=True)
        b._watchdog()
        self.assertEqual([e.reason for e in emitted], ['sensor stream watchdog expired'])
        self.assertFalse(b._place_watchdog_suspended)

    def test_rearms_on_entering_a_capture_bearing_phase(self):
        # Suspend over a retreat tick, then step into a capture-bearing phase (verify): the latch clears
        # (REARMED named capture_silent_exit) so a following stall is measured normally again.
        b, emitted = _bridge(phase='retreat', first_frame=True, sim_gap=30.0,
                             packet_wall_age=0.1, suspend_capture_silent_watchdog=True)
        self._capture(b)
        self.assertTrue(b._place_watchdog_suspended)
        b.controller.phase = 'verify'   # capture-bearing: re-validation renders a fresh frame
        b.last_packet_sim = SIM_NOW - 0.05
        out = self._capture(b)
        self.assertEqual(emitted, [])
        self.assertFalse(b._place_watchdog_suspended)
        self.assertIn('SENSOR_WATCHDOG_REARMED phase=verify reason=capture_silent_exit', out)

    def test_broad_key_off_keeps_retreat_live(self):
        # Broad key OFF (only the PR 357 place-descent key semantics apply): a retreat stall still trips --
        # the loaded-descent window never covers retreat. Proves the two keys are independent.
        b, emitted = _bridge(phase='retreat', first_frame=True, sim_gap=2.0,
                             packet_wall_age=0.1, suspend_place_descent_watchdog=True,
                             suspend_capture_silent_watchdog=False)
        b._watchdog()
        self.assertEqual([e.reason for e in emitted], ['sensor stream watchdog expired'])

    def test_loaded_descent_reason_preserved_when_both_keys_on(self):
        # Both keys on in the loaded place descent (place + held): the place-descent window is checked FIRST,
        # so the log/reason stay loaded_descent (PR 357 byte-identical) and the re-arm is place_descent_exit.
        b, emitted = _bridge(phase='place', held=True, first_frame=True, sim_gap=30.0,
                             packet_wall_age=0.1, suspend_place_descent_watchdog=True,
                             suspend_capture_silent_watchdog=True)
        out = self._capture(b)
        self.assertEqual(emitted, [])
        self.assertIn('reason=loaded_descent', out)
        self.assertEqual(b._watchdog_suspend_reason, 'loaded_descent')
        # leave the window into a capture-bearing phase: re-arm names the PR 357 exit string
        b.controller.phase = 'verify'
        b.controller.in_loaded_place_descent = lambda: False
        b.last_packet_sim = SIM_NOW - 0.05
        out2 = self._capture(b)
        self.assertIn('SENSOR_WATCHDOG_REARMED phase=verify reason=place_descent_exit', out2)


if __name__ == '__main__':
    unittest.main()
