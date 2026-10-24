"""Recorded-trace proof for sim-robust-coupled-breakaway-from-rest-v1.

Drives the REAL ArticulatedController against the REAL recorded outbound nav-traces of the clean
tow (coldwin7) and the throat-wedge stall (coldwin9), via the compact committed fixture
``fixtures/outbound_breakaway_nav.json`` (every 3rd recorded row, SI, ``world_t`` is the sim
clock). It proves the brief's acceptance criterion:

  * the coldwin9 recorded REST/STALL start trips the pinned condition (the shipped grader
    ``twin.traction.coupled_tow_margin`` reports a pin >> escape_s) and, replayed through the
    controller, ENTERS the break-away rock recovery and commands the elevated launch surge;
  * the coldwin7 recorded MOMENTUM start never pins (grader pin < escape_s) and never enters the
    recovery -- ``breakaway_recoveries == 0``;
  * rock OFF is byte-identical (same entry decision, legacy surge speed).

CPU-only; no Isaac/GPU/ROS. Run with:
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -I -m pytest \
    scripts/twin/tests/test_coupled_breakaway_rock.py -q
"""
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.mobile import MobileSpec
from twin.navigation import GridMap, Odometry, Observation
from twin.navigation.articulated import ArticulatedController, TrainState, advance
from twin.traction import coupled_tow_margin

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'outbound_breakaway_nav.json'
# The live monitor/recovery parameters the capstone241 run used (sim seconds / metres).
ESCAPE_S = 3.0


def _fixture():
    return json.loads(FIXTURE.read_text())


def _margin_rows(rows):
    return [{'x': r['x'], 'y': r['y'], 'coupled': r['coupled'], 'parked': r['parked'],
             'lin': r['lin'], 'world_t': r['world_t']} for r in rows]


class RecordedBreakawayTests(unittest.TestCase):
    def setUp(self):
        self.spec = MobileSpec()
        self.world = GridMap((-15, -15, 15, 15))
        self.fix = _fixture()
        self.goal = tuple(self.fix['goal'])

    # ---- (1) the shipped acceptance grader: coldwin9 pins, coldwin7 does not ----------------------

    def test_recorded_margin_discriminates_stall_from_clean(self):
        clean = coupled_tow_margin(_margin_rows(self.fix['coldwin7']['rows']), self.goal)
        stall = coupled_tow_margin(_margin_rows(self.fix['coldwin9']['rows']), self.goal)
        # coldwin9: pinned at the throat for far longer than the recovery's escape window -> the pinned
        # condition that FIRES the break-away recovery; coldwin7: a brief plan-wait dip only.
        self.assertFalse(stall.passed)
        self.assertFalse(stall.reached_goal)
        self.assertGreater(stall.longest_pin_s, ESCAPE_S)
        self.assertLess(stall.max_advance_m, 1.0)        # never cleared the throat (~0.54 m)
        self.assertTrue(clean.passed)
        self.assertTrue(clean.reached_goal)
        self.assertLess(clean.longest_pin_s, ESCAPE_S)   # the clean tow never pins long enough to trip

    # ---- (2) replay through the REAL controller: coldwin9 triggers the rock, coldwin7 does not ----

    def _replay(self, run, rock):
        """Reach tracking on a straight +x goal, then advance the synthetic base by the recorded
        per-tick FORWARD displacement (frozen where the real base was pinned) on the recorded SIM
        clock, feeding the recorded measured (rv, rw). Returns the controller after replay.

        The monitor/recovery key only on NET base progress over the SIM clock, so replaying the
        recorded net-progress profile reproduces the real trigger faithfully without the full
        ~800-obstacle world."""
        c = ArticulatedController(self.world, self.spec, allow_test_fixtures=True,
                                  coupled_progress_escape=True, coupled_progress_recovery=True,
                                  coupled_breakaway_rock=rock,
                                  coupled_progress_min_m=0.05, coupled_progress_escape_s=ESCAPE_S,
                                  coupled_progress_fault_s=8.0,
                                  coupled_breakaway_speed_mps=0.18, coupled_breakaway_launch_mps=0.24,
                                  coupled_breakaway_window_s=2.0, coupled_breakaway_pulse_s=0.5,
                                  coupled_breakaway_attempts=3)
        c.start((9, 0, 0), True)
        # reach tracking
        state = TrainState(0, 0, 0, 0); v = w = 0.0; now = 0.0
        for i in range(400):
            now = (i + 1) * .05
            od = Odometry(state.pose, now, v, w, 'test_fixture', .02)
            obs = Observation(now, (), True, True, state.articulation, 'test_fixture', 10)
            r = c.step(.05, od, obs, now=now)
            v, w = r.velocity
            if c.route is not None and c.route.states and r.reason == 'tracking':
                break
            state = advance(state, v * .05, w / v if v else 0, self.spec)
        self.assertTrue(c.route and c.route.states, 'controller never reached tracking')

        rows = [r for r in self.fix[run]['rows'] if r['coupled']]
        surges = []
        prev = None
        for row in rows:
            if prev is not None:
                d = math.hypot(row['x'] - prev['x'], row['y'] - prev['y'])   # recorded forward advance
                now += max(1e-3, row['world_t'] - prev['world_t'])           # recorded SIM clock
                state = advance(state, d, 0.0, self.spec)                    # straight +x goal
            prev = row
            od = Odometry(state.pose, now, row['lin'], row['rw'], 'test_fixture', .02)
            obs = Observation(now, (), True, True, state.articulation, 'test_fixture', 10)
            res = c.step(.05, od, obs, now=now)
            if c._recovery_active and res.reason == 'tracking':
                surges.append(res.velocity[0])
            if res.status != 'running':
                break
        return c, surges

    def test_recorded_stall_enters_rock_recovery_with_elevated_launch(self):
        c, surges = self._replay('coldwin9', rock=True)
        self.assertGreaterEqual(c.breakaway_recoveries, 1)        # the stall tripped the recovery
        self.assertTrue(surges)
        self.assertTrue(any(abs(s - 0.24) < 1e-6 for s in surges))   # the ELEVATED launch surge was commanded
        self.assertFalse(any(s > c.max_speed + 1e-9 for s in surges))  # bounded by max_speed

    def test_recorded_clean_tow_never_triggers_recovery(self):
        for rock in (True, False):
            c, surges = self._replay('coldwin7', rock=rock)
            self.assertEqual(c.breakaway_recoveries, 0, f'clean tow triggered recovery (rock={rock})')
            self.assertEqual(surges, [])

    def test_recorded_stall_off_uses_legacy_surge(self):
        # Rock OFF on the SAME recorded stall: the recovery still enters (byte-identical trigger) but the
        # surge is PR 236's 0.18, never the elevated launch.
        c, surges = self._replay('coldwin9', rock=False)
        self.assertGreaterEqual(c.breakaway_recoveries, 1)
        self.assertTrue(surges)
        self.assertTrue(any(abs(s - 0.18) < 1e-6 for s in surges))
        self.assertFalse(any(s > 0.18 + 1e-6 for s in surges))   # never the elevated launch when rock off


if __name__ == '__main__':
    unittest.main()
