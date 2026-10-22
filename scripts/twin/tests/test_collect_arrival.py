"""sim-robust-collect-arrival-v1 CPU tests: the tolerance-aware (strict) XY final approach for the
grasp-critical collect source park. Deterministic fixtures, never evidence of a physics run.

Context: the global XY arrival hysteresis latch (fm482, added so the couple standoff stops dithering on
the tolerance circle) latches the XY capture once inside ``arrival_xy_tol`` and only releases past
``arrival_xy_tol + hysteresis``. For the couple standoff that is correct -- the fine-dock servo closes
the residual. But for a grasp-critical leg it lets the base DECLARE arrived_stopped while the true xy sits
up to one hysteresis band OUTSIDE the tolerance (cold collect declared arrived at xy 0.169 against a 0.150
tol as the base crept out during the rotate-in-place). ``strict_xy_arrival`` makes the latch unable to, by
itself, declare arrival outside the real xy tolerance; the base keeps closing until the TRUE xy is in tol.
"""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.navigation.core import Controller, Footprint, GridMap, Odometry, Observation

STANDOFF = (-4.06, 15.25, 0.0)
DT = 1 / 60


def _grid():
    return GridMap((-8.0, 12.0, 0.0, 19.0), resolution=0.2)


def _footprint():
    return Footprint(0.65, 0.62, 1.12, margin=0.05)


class StrictXyArrivalTests(unittest.TestCase):
    def _controller(self, strict):
        c = Controller(_grid(), _footprint(), allow_test_fixtures=True, goal_timeout=1e4,
                       arrival_xy_hysteresis=0.03)
        c.start(STANDOFF, False, arrival_xy_tol=0.05, strict_xy_arrival=strict)
        return c

    def _step(self, c, dx, t, yaw=0.0, v=0.0, w=0.0):
        obs = Observation(t, (), False, True, source='test_fixture', coverage_radius=10.0)
        return c.step(DT, Odometry((STANDOFF[0] + dx, STANDOFF[1], yaw), t, v, w, 'test_fixture'),
                      obs, now=t)

    def test_loose_latch_declares_arrival_outside_the_true_tol(self):
        # Baseline (the cold-collect failure): latch the XY capture inside tol with the yaw misaligned,
        # then align the yaw while the base has crept to 0.065 m (outside the 0.05 tol but inside tol+h).
        # The loose latch declares arrived_stopped there -- the grasp then begins ~1.5 cm out of tolerance.
        c = self._controller(False)
        self._step(c, 0.04, 0.1, yaw=-0.5)                 # captured, turning to the goal yaw (not arrived)
        self.assertTrue(c._xy_latched)
        r = self._step(c, 0.065, 0.2, yaw=0.0)             # aligned + stopped, drifted outside the true tol
        self.assertEqual((r.status, r.reason), ('succeeded', 'arrived_stopped'))

    def test_strict_refuses_to_declare_arrival_outside_the_true_tol(self):
        # Same geometry, strict_xy_arrival on: the latch still holds (the drive is not re-armed by strict
        # alone) but arrival is NOT declared while the true xy is outside tol -- the base keeps closing.
        c = self._controller(True)
        self._step(c, 0.04, 0.1, yaw=-0.5)
        self.assertTrue(c._xy_latched)
        r = self._step(c, 0.065, 0.2, yaw=0.0)
        self.assertNotEqual(r.status, 'succeeded')         # does not arrive outside the true tolerance

    def test_strict_still_arrives_within_the_true_tol(self):
        # Strict only blocks a declaration OUTSIDE tol; a base genuinely within tol, aligned and stopped,
        # arrives normally (no regression on a clean park).
        c = self._controller(True)
        self._step(c, 0.04, 0.1, yaw=-0.5)                 # latch + turn
        r = self._step(c, 0.03, 0.2, yaw=0.0)              # aligned + stopped + inside the true tol
        self.assertEqual((r.status, r.reason), ('succeeded', 'arrived_stopped'))

    def test_strict_is_a_noop_without_the_hysteresis_latch(self):
        # With no hysteresis the bare xy <= tol test is already strict, so strict_xy_arrival changes
        # nothing: outside tol never arrives with OR without it => byte-identical for a no-hysteresis run.
        def run(strict):
            c = Controller(_grid(), _footprint(), allow_test_fixtures=True, goal_timeout=1e4)
            c.start(STANDOFF, False, arrival_xy_tol=0.05, strict_xy_arrival=strict)
            obs = Observation(0.1, (), False, True, source='test_fixture', coverage_radius=10.0)
            return c.step(DT, Odometry((STANDOFF[0] + 0.065, STANDOFF[1], 0.0), 0.1, 0.0, 0.0,
                                       'test_fixture'), obs, now=0.1)
        self.assertNotEqual(run(False).status, 'succeeded')
        self.assertNotEqual(run(True).status, 'succeeded')

    def test_default_start_leaves_strict_off(self):
        c = Controller(_grid(), _footprint(), allow_test_fixtures=True)
        c.start(STANDOFF, False)
        self.assertFalse(c.strict_xy_arrival)


if __name__ == '__main__':
    unittest.main()
