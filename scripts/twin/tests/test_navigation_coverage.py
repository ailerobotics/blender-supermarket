"""Self-occlusion-aware coverage certification (CPU, no PhysX/GPU/ROS).

The physical restock rig's range sensor always self-occludes on its own base, mast,
stowed arm and towed cart, so a full 360 degree unoccluded scan is never produced.
Certification must therefore refuse to zero coverage merely because some beams are
self-returns (UNKNOWN): it certifies the forward braking sweep in the direction of
travel against the OBSERVED (non-self) sectors plus the robot's own footprint, keeps
the 0.05 m geometry and 0.10 m localization margins and the full cart sweep, and never
reverses. These counterfactuals pin that a self-occluded ring with a clear forward
sweep drives, while a genuinely unobserved forward sector still refuses.
"""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.navigation import (GridMap, Odometry, CoverageField, coverage_certifies,
                             sweep_extent)
from twin.navigation.core import Observation
from twin.navigation.articulated import ArticulatedController
from twin.navigation.sensors import Scan, observation

BEAMS = 360
INC = 2 * math.pi / BEAMS
RANGE_MAX = 10.0


def _bearing(i):
    return math.atan2(math.sin(-math.pi + i * INC), math.cos(-math.pi + i * INC))


def _scan(free, stamp=0.1):
    """360 beam scan; `free(bearing)` True -> observed free (range_max), else self/UNKNOWN."""
    ranges, unknown = [], []
    for i in range(BEAMS):
        clear = free(_bearing(i))
        ranges.append(RANGE_MAX if clear else math.nan)
        unknown.append(not clear)
    return Scan(stamp, tuple(ranges), -math.pi, INC, RANGE_MAX, unknown=tuple(unknown))


def _odom(stamp=0.1, linear=0.0):
    return Odometry((0., 0., 0.), stamp, linear, 0.0,
                    source='wheel_odometry_localized', position_uncertainty=0.10)


def _controller(coupled):
    c = ArticulatedController(GridMap((-10, -10, 10, 10)))
    c.start((4., 0., 0.), coupled)
    return c


def _observe(scan, coupled):
    return observation(scan, _odom(), coupled=coupled, arm_transport=True)


class SweepExtentGeometryTests(unittest.TestCase):
    def test_forward_extends_sides_and_rear_stay_at_footprint(self):
        rho, L = 0.80, 0.153
        self.assertAlmostEqual(sweep_extent(0.0, rho, L), rho + L)           # dead ahead
        self.assertEqual(sweep_extent(math.pi, rho, L), rho)                 # rear: no reverse
        self.assertEqual(sweep_extent(math.radians(100), rho, L), rho)       # past broadside
        # A near-broadside bearing the capsule barely reaches is NOT treated as swept.
        self.assertLessEqual(sweep_extent(math.radians(89), rho, L), rho)

    def test_extent_is_monotonic_into_the_heading(self):
        rho, L = 1.0, 0.2
        prev = sweep_extent(math.radians(80), rho, L)
        for deg in (60, 40, 20, 0):
            cur = sweep_extent(math.radians(deg), rho, L)
            self.assertGreaterEqual(cur + 1e-9, prev)
            prev = cur


class CoverageFieldTests(unittest.TestCase):
    def test_rear_only_self_occlusion_certifies_forward_travel(self):
        cov = _observe(_scan(lambda b: abs(b) <= math.radians(110)), False).coverage
        self.assertTrue(cov.certifies(0.0, 0.80, 0.153))

    def test_blind_forward_wedge_refuses(self):
        # Forward clear EXCEPT a wedge straight ahead that is self-occluded.
        cov = _observe(_scan(lambda b: abs(b) <= math.radians(110)
                             and not (math.radians(8) <= b <= math.radians(24))),
                       False).coverage
        self.assertFalse(cov.certifies(0.0, 0.80, 0.153))

    def test_stationary_robot_needs_no_external_observation(self):
        # Fully self-occluded, but stopped: no sector is swept -> certified.
        cov = _observe(_scan(lambda b: False), False).coverage
        self.assertTrue(cov.certifies(0.0, 0.80, 0.0))
        self.assertFalse(cov.certifies(0.0, 0.80, 0.153))  # once it tries to drive, refuse

    def test_heading_rotates_the_required_sector(self):
        # Observed only to the sides/rear (|bearing| >= 80 deg); the forward cone is blind.
        cov = _observe(_scan(lambda b: abs(b) >= math.radians(80)), False).coverage
        self.assertFalse(cov.certifies(0.0, 0.80, 0.153))      # heading east -> blind cone
        self.assertTrue(cov.certifies(math.pi, 0.80, 0.153))   # heading west -> observed

    def test_scalar_fallback_when_no_field_present(self):
        fixture = Observation(0.1, (), False, True, source='test_fixture', coverage_radius=10.)
        self.assertTrue(coverage_certifies(fixture, 0.0, 0.80, 0.153))
        blind = Observation(0.1, (), False, True, source='test_fixture', coverage_radius=0.)
        self.assertFalse(coverage_certifies(blind, 0.0, 0.80, 0.153))


class GateDriveTests(unittest.TestCase):
    """Drive the production ArticulatedController gate with self-occluded scans."""

    def _gate(self, coupled, scan, linear_cmd=0.25):
        c = _controller(coupled)
        obs = _observe(scan, coupled)
        return c.gate_velocity(0.1, _odom(linear=0.0), obs, linear_cmd, 0.0)

    def test_uncoupled_self_occluded_ring_with_clear_forward_drives(self):
        result = self._gate(False, _scan(lambda b: abs(b) <= math.radians(110)))
        self.assertEqual(result.reason, 'tracking')
        self.assertGreater(result.linear, 0.0)

    def test_coupled_full_cart_sweep_drives_with_clear_forward(self):
        result = self._gate(True, _scan(lambda b: abs(b) <= math.radians(120)))
        self.assertEqual(result.reason, 'tracking')

    def test_uncoupled_blind_forward_sector_refuses(self):
        scan = _scan(lambda b: abs(b) <= math.radians(110)
                     and not (math.radians(8) <= b <= math.radians(24)))
        self.assertEqual(self._gate(False, scan).reason, 'insufficient_sensor_coverage')

    def test_coupled_blind_forward_sector_refuses(self):
        scan = _scan(lambda b: abs(b) <= math.radians(120)
                     and not (math.radians(-12) <= b <= math.radians(12)))
        self.assertEqual(self._gate(True, scan).reason, 'insufficient_sensor_coverage')

    def test_fully_self_occluded_robot_cannot_start_driving(self):
        # The historical inside-chassis counterexample stays refused: blind ahead.
        self.assertEqual(self._gate(False, _scan(lambda b: False)).reason,
                         'insufficient_sensor_coverage')

    def test_no_reverse_command_still_refused(self):
        result = self._gate(False, _scan(lambda b: abs(b) <= math.radians(110)),
                            linear_cmd=-0.1)
        self.assertEqual(result.reason, 'reverse_or_invalid_command')

    def test_observed_forward_obstacle_is_obstacle_stop_not_coverage(self):
        # A real return ahead is observed (not blind): coverage passes, obstacle gate stops.
        ranges, unknown = [], []
        for i in range(BEAMS):
            b = _bearing(i)
            if abs(b) <= math.radians(110):
                hit = abs(b) <= math.radians(3)           # obstacle dead ahead, within sweep
                ranges.append(0.85 if hit else RANGE_MAX)
                unknown.append(False)
            else:
                ranges.append(math.nan)
                unknown.append(True)
        scan = Scan(0.1, tuple(ranges), -math.pi, INC, RANGE_MAX, unknown=tuple(unknown))
        c = _controller(False)
        obs = _observe(scan, False)
        self.assertTrue(any(obs.obstacles))                # the return is reported
        self.assertEqual(c.gate_velocity(0.1, _odom(), obs, 0.25, 0.0).reason, 'obstacle_stop')


if __name__ == '__main__':
    unittest.main()
