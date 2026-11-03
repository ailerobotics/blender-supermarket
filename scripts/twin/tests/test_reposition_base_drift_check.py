"""sim-collect-reposition-block-v1: the base-yaw-during-grasp / reposition-drift grader
(scripts/check_reposition_base_drift.py) on a committed fixture reproducing the boxc1a signature -- the
base is held during the grasp (yaw ~0) but drifts ~0.28 m during the reposition. No Isaac/ROS/GPU."""
from pathlib import Path
import sys
import unittest

_ROOT = Path(__file__).resolve().parents[2]      # scripts/
sys.path.insert(0, str(_ROOT))
import check_reposition_base_drift as grader      # scripts/check_reposition_base_drift.py

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'reposition_base_drift'


class RepositionBaseDriftCheckTests(unittest.TestCase):
    def test_grasp_held_but_reposition_drifts_fails_default_bounds(self):
        r = grader.analyze(FIXTURE, grasp_yaw_bound_deg=2.0, reposition_drift_bound_m=0.10)
        # Grasp: the base is steady (the brake held) -> yaw drift well within bound. This is the honest
        # correction of the brief's "~15 deg grasp shove" (it was in_start=15, an obstacle-point count).
        self.assertTrue(r['grasp']['within_bound'])
        self.assertLess(r['grasp']['yaw_drift_deg'], 2.0)
        # Reposition: the loaded base drifts ~0.28 m NORTH off the departure pose -> over the 0.10 m bound.
        self.assertFalse(r['reposition']['within_bound'])
        self.assertGreater(r['reposition']['base_xy_drift_m'], 0.25)
        self.assertGreater(r['reposition']['north_m'], 0.25)
        self.assertFalse(r['pass'])

    def test_passes_when_reposition_drift_is_within_the_fixed_bound(self):
        # The fix's pass criterion: with the base HELD on the departure pose the reposition drift is small.
        # A generous bound stands in for a fixed run; the grader then PASSES.
        r = grader.analyze(FIXTURE, grasp_yaw_bound_deg=2.0, reposition_drift_bound_m=0.40)
        self.assertTrue(r['reposition']['within_bound'])
        self.assertTrue(r['pass'])

    def test_windows_located_from_stage_timeline(self):
        r = grader.analyze(FIXTURE, grasp_yaw_bound_deg=2.0, reposition_drift_bound_m=0.10)
        self.assertEqual(r['windows_mission_time']['reposition'], [5.0, 10.0])
        self.assertEqual(r['reason'], 'blocked_no_route_timeout')


if __name__ == '__main__':
    unittest.main()
