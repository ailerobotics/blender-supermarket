"""CPU tests for the fm357 settle-before-execute + re-seed-from-live-state arm-execute fix.

The host plans an arm move from the begin-time measured joints, then hands the trajectory to MoveIt's
ExecuteTrajectory, which validates the FIRST waypoint against the CURRENT /joint_states. Under render
pacing the plan->validate gap is huge and a gravity-loaded joint drifts, so the start no longer matches
and the move aborts (run-280 shoulder_pan, run-282 ur_elbow). The fix: settle the arm, then re-seed the
first waypoint to the live state before dispatch. These test the pure logic; absent a live-state
provider the port keeps its original direct dispatch (byte-identical), covered below.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from twin.restock_manipulation.ros_adapter import (  # noqa: E402
    arm_settled, reseed_first_waypoint, SETTLE_VEL_DEFAULT)

JOINTS = ['ur_shoulder_pan_joint', 'ur_shoulder_lift_joint', 'ur_elbow_joint',
          'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']


class SettledTests(unittest.TestCase):
    def test_at_rest_is_settled(self):
        self.assertTrue(arm_settled([0.0, 0.01, -0.02, 0.0, 0.0, 0.0]))

    def test_moving_joint_not_settled(self):
        # run-282: the elbow was still drifting; must HOLD, not execute.
        self.assertFalse(arm_settled([0.0, 0.0, 0.2, 0.0, 0.0, 0.0]))

    def test_threshold_edge(self):
        self.assertTrue(arm_settled([SETTLE_VEL_DEFAULT] * 6))
        self.assertFalse(arm_settled([SETTLE_VEL_DEFAULT + 1e-6] * 6))

    def test_empty_is_not_settled(self):
        # Empty => unknown; the port guards this (does not treat as settled) and proceeds separately.
        self.assertFalse(arm_settled([]))

    def test_nonfinite_not_settled(self):
        self.assertFalse(arm_settled([float('nan'), 0, 0, 0, 0, 0]))


class ReseedTests(unittest.TestCase):
    def test_reseed_matches_live_so_start_gate_passes(self):
        planned = [0.10, -1.00, 1.42, -1.08, 0.56, 0.00]   # plan start (begin-time measured)
        live = {n: v for n, v in zip(JOINTS, [0.10, -1.00, 1.49, -1.08, 0.56, 0.00])}  # elbow drifted +0.07
        out = reseed_first_waypoint(JOINTS, planned, live)
        # After re-seed the first waypoint equals the live state -> allowed_start_tolerance deviation 0.
        self.assertEqual(out, [live[n] for n in JOINTS])
        self.assertEqual(max(abs(a - live[n]) for a, n in zip(out, JOINTS)), 0.0)

    def test_respects_trajectory_joint_order(self):
        names = list(reversed(JOINTS))
        planned = [0.0] * 6
        live = {n: float(i) for i, n in enumerate(JOINTS)}
        out = reseed_first_waypoint(names, planned, live)
        self.assertEqual(out, [live[n] for n in names])   # mapped by name, not by position

    def test_missing_joint_keeps_planned(self):
        planned = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        live = {JOINTS[0]: 9.9}                            # only pan known
        out = reseed_first_waypoint(JOINTS, planned, live)
        self.assertEqual(out[0], 9.9)
        self.assertEqual(out[1:], planned[1:])            # others untouched

    def test_empty_live_is_noop(self):
        # Provider-absent / empty-live path => byte-identical (no re-seed).
        planned = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.assertEqual(reseed_first_waypoint(JOINTS, planned, {}), planned)


if __name__ == '__main__':
    unittest.main()


# --- run-284 regression: the reseeded first waypoint must satisfy trajectory_host.Plan ---
import types as _types  # noqa: E402
import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
from twin.restock_manipulation.ros_adapter import JOINTS as _JOINTS, reseed_first_waypoint as _reseed  # noqa: E402
from twin.restock import trajectory_host as _th  # noqa: E402


def _traj(first_positions, first_velocities):
    def _pt(pos, vel, t):
        return _types.SimpleNamespace(positions=list(pos), velocities=list(vel), effort=[],
                                      time_from_start=_types.SimpleNamespace(sec=int(t), nanosec=int((t % 1) * 1e9)))
    start = list(first_positions)
    goal = [v + 0.001 for v in start]          # tiny, slow rest-to-rest move -> within accel limits
    return _types.SimpleNamespace(joint_names=list(_JOINTS), points=[
        _pt(first_positions, first_velocities, 0.0), _pt(goal, [0.0] * 6, 5.0)])


def test_reseeded_first_waypoint_accepted_by_trajectory_host():
    # The reseed overwrites point[0] positions to the live state and sets SIX zero velocities.
    live = {n: 0.01 * i for i, n in enumerate(_JOINTS)}
    limits = [(-6.3, 6.3)] * 6
    planned_first = [0.0] * 6
    reseeded = _reseed(list(_JOINTS), planned_first, live)
    # The fix: six explicit zeros (run-284 used [] -> rejected).
    _th.Plan(_traj(reseeded, [0.0] * 6), reseeded, limits)   # must NOT raise

    import pytest
    with pytest.raises(ValueError):                           # the run-284 bug: empty velocities
        _th.Plan(_traj(reseeded, []), reseeded, limits)
