"""CPU pacing proof for the sim-clocked manipulation-host timers (sim-robust-host-timers-audit-fix-v1).

Drives the SHIPPED sim-primary + frozen-sim-wall-backstop logic at 1x, 16x and 30x wall-to-sim pacing
(the measured live_pose arm-feed economy) through the pure harness, and binds the sweep to the real
``RosMotionPort.planning_wall_exceeded`` method. Proves every converted timer: (a) NEVER false-trips a
healthy-but-render-slow producer at 30x -- the coldlive sample-4 regression -- (b) still catches a
genuinely hung producer on the SIM leg in bounded SIM time, and (c) catches a frozen sim on the WALL
backstop. 1x stays byte-identical to the historical wall semantics. No ROS / Isaac / GPU.
"""
import unittest
from unittest import mock

from twin.restock.host_pacing_harness import (
    FakeClockPair, simulate_sim_primary_budget, simulate_station_heartbeat,
)
from twin.restock_manipulation import ros_adapter
from twin.restock_manipulation.ros_adapter import (
    RosMotionPort, PLAN_TOTAL_SIM_CAP_S, PLAN_TOTAL_WALL_CAP_S, STARTUP_GRACE_SIM_S, STARTUP_GRACE_S,
    frozen_sim_wall_backstop,
)

PACINGS = (1.0, 16.0, 30.0)

# The three converted host timers, with their shipped default budgets (sim_budget, wall_floor).
TIMERS = {
    'planning_budget': (PLAN_TOTAL_SIM_CAP_S, PLAN_TOTAL_WALL_CAP_S),
    'startup_grace': (STARTUP_GRACE_SIM_S, STARTUP_GRACE_S),
    'payload_ack': (1.0, 1.0),            # RosPayloadSync default ack_wall_s / ack_sim_s
}


class NoFalseTripAtPacing(unittest.TestCase):
    """A healthy producer that finishes WELL within its SIM budget must 'complete' at every pacing, even
    though at 30x it burns far more WALL time than the historical wall cap (the false-trip that e-stopped
    coldlive sample-4)."""

    def test_alive_producer_never_trips(self):
        for name, (sim_budget, wall_floor) in TIMERS.items():
            producer_sim = sim_budget * 0.3          # finishes at 30% of the sim budget -- clearly alive
            for pacing in PACINGS:
                r = simulate_sim_primary_budget(
                    pacing=pacing, sim_budget=sim_budget, wall_floor=wall_floor,
                    producer_sim_duration=producer_sim)
                self.assertEqual(r['outcome'], 'completed', (name, pacing, r))

    def test_coldlive_sample4_planning_case(self):
        # The exact sample-4 shape: the plan resolves after ~6 SIM seconds while ~120+ WALL seconds pass
        # (≈20x). Under the OLD pure-wall 120 s cap this e-stopped bare; the SIM-primary budget completes.
        r = simulate_sim_primary_budget(
            pacing=20.0, sim_budget=PLAN_TOTAL_SIM_CAP_S, wall_floor=PLAN_TOTAL_WALL_CAP_S,
            producer_sim_duration=6.0)
        self.assertEqual(r['outcome'], 'completed', r)
        self.assertGreater(r['wall'], 110.0)          # it really did burn >110 s WALL -- and did NOT trip


class HungProducerCaughtOnSimLeg(unittest.TestCase):
    """A producer that NEVER finishes while the sim keeps advancing is genuinely hung; the SIM leg must
    catch it in bounded SIM time (~the sim budget), at every pacing, named 'tripped_sim'."""

    def test_hang_trips_sim_leg_bounded(self):
        for name, (sim_budget, wall_floor) in TIMERS.items():
            for pacing in PACINGS:
                r = simulate_sim_primary_budget(
                    pacing=pacing, sim_budget=sim_budget, wall_floor=wall_floor,
                    producer_sim_duration=None)
                self.assertEqual(r['outcome'], 'tripped_sim', (name, pacing, r))
                # Bounded: fires within one tick of the sim budget (SIM seconds), independent of pacing.
                self.assertLessEqual(r['sim'], sim_budget + 1.0, (name, pacing, r))


class FrozenSimCaughtOnWallBackstop(unittest.TestCase):
    """A FROZEN sim (clock stuck at 0) can never advance the SIM leg, so the generous frozen-sim WALL
    backstop must catch it -- bounded in WALL time -- named 'tripped_wall'."""

    def test_frozen_sim_trips_wall_backstop(self):
        for name, (sim_budget, wall_floor) in TIMERS.items():
            r = simulate_sim_primary_budget(
                pacing=30.0, sim_budget=sim_budget, wall_floor=wall_floor,
                producer_sim_duration=None, frozen=True)
            self.assertEqual(r['outcome'], 'tripped_wall', (name, r))
            self.assertLessEqual(r['wall'], r['wall_backstop'] + 1.0, (name, r))


class RealRigByteIdentical(unittest.TestCase):
    """At 1x (sim == wall, the real rig) a hang must trip at the historical budget boundary on BOTH the
    wall and sim reading -- the no-pacing path is unchanged."""

    def test_1x_hang_trips_at_budget(self):
        for name, (sim_budget, wall_floor) in TIMERS.items():
            r = simulate_sim_primary_budget(
                pacing=1.0, sim_budget=sim_budget, wall_floor=wall_floor,
                producer_sim_duration=None, dwall=0.05)
            self.assertEqual(r['outcome'], 'tripped_sim', (name, r))
            self.assertAlmostEqual(r['sim'], r['wall'], places=6)         # sim == wall at 1x
            self.assertLessEqual(abs(r['sim'] - sim_budget), 0.1, (name, r))


class RealPlanningMethodUnderPacing(unittest.TestCase):
    """Bind the sweep to the ACTUAL RosMotionPort.planning_wall_exceeded (not just the harness predicate):
    build a port via __new__, inject a fake sim clock, and patch the adapter's time.monotonic to the fake
    wall clock, so the real method is evaluated at each pacing."""

    @staticmethod
    def _port(clock, sim_cap=PLAN_TOTAL_SIM_CAP_S, wall_floor=PLAN_TOTAL_WALL_CAP_S):
        p = RosMotionPort.__new__(RosMotionPort)
        p.state = 'plan_result'
        p.future = None                       # no delivered result -> a hang if the budget passes
        p.plan_total_sim_cap_s = sim_cap
        p.plan_wall_backstop_s = frozen_sim_wall_backstop(sim_cap, wall_floor)
        p._plan_started_wall = 0.0
        p._plan_started_sim = 0.0
        p._sim_now = lambda: clock.sim
        p._awaiting_server_respawn = False
        return p

    def test_real_method_tolerates_render_slow_plan(self):
        # 30x pacing, a plan resolving at ~6 SIM s: step to ~180 s WALL and assert the real method never
        # fires (future would be consumed before the budget at ~6 sim s; here we just prove no wall trip).
        for pacing in PACINGS:
            clock = FakeClockPair(pacing=pacing)
            p = self._port(clock)
            with mock.patch.object(ros_adapter.time, 'monotonic', lambda: clock.wall):
                tripped_before_sim_cap = False
                while clock.sim < PLAN_TOTAL_SIM_CAP_S - 1.0:
                    if p.planning_wall_exceeded():
                        tripped_before_sim_cap = True
                        break
                    clock.advance(0.5)
                self.assertFalse(tripped_before_sim_cap,
                                 'real planning_wall_exceeded false-tripped at pacing %s (wall=%.1f sim=%.1f)'
                                 % (pacing, clock.wall, clock.sim))

    def test_real_method_catches_hang_on_sim_leg(self):
        for pacing in PACINGS:
            clock = FakeClockPair(pacing=pacing)
            p = self._port(clock)
            with mock.patch.object(ros_adapter.time, 'monotonic', lambda: clock.wall):
                fired_at_sim = None
                while clock.wall <= PLAN_TOTAL_SIM_CAP_S * pacing * 3:
                    if p.planning_wall_exceeded():
                        fired_at_sim = clock.sim
                        break
                    clock.advance(0.5)
                self.assertIsNotNone(fired_at_sim, 'real method never caught the hang at pacing %s' % pacing)
                self.assertLessEqual(fired_at_sim, PLAN_TOTAL_SIM_CAP_S + 1.0, pacing)
                self.assertEqual(p.planning_budget_leg(), 'sim', pacing)

    def test_real_method_frozen_sim_wall_backstop(self):
        clock = FakeClockPair(pacing=30.0, frozen=True)
        p = self._port(clock)
        with mock.patch.object(ros_adapter.time, 'monotonic', lambda: clock.wall):
            fired = False
            while clock.wall <= p.plan_wall_backstop_s + 10.0:
                if p.planning_wall_exceeded():
                    fired = True
                    break
                clock.advance(1.0)
            self.assertTrue(fired, 'frozen sim never hit the wall backstop')
            self.assertEqual(p.planning_budget_leg(), 'wall')


class StationHeartbeatUnderPacing(unittest.TestCase):
    """sim-robust-station-heartbeat-simclock-v1: the parked-station heartbeat guard (the REAL shipped
    ``IncorporationAckState.snapshot``) must behave like the other converted timers under pacing -- no
    false stop on a healthy-but-render-slow station, a frozen sim still caught on the WALL backstop, and
    1x byte-identical."""

    # The bounded GPU proof's budgets: SIM gap 30 s (the real live_pose transfer manip gap), WALL floor at
    # the validator's 30 (left at default -- not the 120 s quick-lever widen). A 2 s SIM no-heartbeat gap
    # is the healthy manip window (coldliverecovery: ~2 SIM s = 30.75 WALL s at ~16x).
    SIM_GAP, WALL_FLOOR, HEALTHY_GAP_SIM = 30.0, 30.0, 2.0

    def test_healthy_station_never_false_stops(self):
        # The coldliverecovery regression: a 2 s SIM manip gap stretches to 30-60 s WALL at 16-30x. The
        # OLD flat wall guard (30 s) tripped; the derived frozen-sim backstop (30 x 50 = 1500 s) does not.
        for pacing in PACINGS:
            r = simulate_station_heartbeat(pacing=pacing, sim_gap_budget=self.SIM_GAP,
                                           wall_floor=self.WALL_FLOOR, no_heartbeat_sim=self.HEALTHY_GAP_SIM)
            self.assertEqual(r['outcome'], 'survived', (pacing, r))
        # And explicitly at the measured 16x it really did burn >30 s WALL (past the flat 30 s) -- no stop.
        r16 = simulate_station_heartbeat(pacing=16.0, sim_gap_budget=self.SIM_GAP,
                                         wall_floor=self.WALL_FLOOR, no_heartbeat_sim=self.HEALTHY_GAP_SIM)
        self.assertGreater(r16['wall'], 30.0, r16)

    def test_dead_station_caught_on_sim_leg_bounded(self):
        # A station that never gets heart-beat again while the sim keeps advancing (op never ends) is
        # genuinely lost; the SIM leg catches it within ~the sim budget at every pacing.
        for pacing in PACINGS:
            r = simulate_station_heartbeat(pacing=pacing, sim_gap_budget=self.SIM_GAP,
                                           wall_floor=self.WALL_FLOOR, no_heartbeat_sim=1e9)
            self.assertEqual(r['outcome'], 'tripped_sim', (pacing, r))
            self.assertLessEqual(r['sim'], self.SIM_GAP + 1.0, (pacing, r))

    def test_frozen_sim_caught_on_wall_backstop(self):
        # A FROZEN sim can never advance the SIM leg, so the derived WALL backstop (1500 s) must catch it.
        r = simulate_station_heartbeat(pacing=30.0, sim_gap_budget=self.SIM_GAP, wall_floor=self.WALL_FLOOR,
                                       no_heartbeat_sim=1e9, frozen=True)
        self.assertEqual(r['outcome'], 'tripped_wall', r)
        self.assertLessEqual(r['wall'], r['wall_backstop'] + 1.0, r)

    def test_1x_default_budgets_byte_identical(self):
        # At 1x with the shipped DEFAULT budgets (0.5 s sim gap / 3 s wall floor -> 25 s derived backstop),
        # a dead station trips on the SIM gap at 0.5 s -- identical to the historical real-rig behaviour
        # (the sim leg always fired first; the wall value is inert at sim == wall).
        r = simulate_station_heartbeat(pacing=1.0, sim_gap_budget=0.5, wall_floor=3.0, no_heartbeat_sim=1e9)
        self.assertEqual(r['outcome'], 'tripped_sim', r)
        self.assertAlmostEqual(r['sim'], r['wall'], places=6)
        self.assertLessEqual(abs(r['sim'] - 0.5), 0.1, r)


if __name__ == '__main__':
    unittest.main()
