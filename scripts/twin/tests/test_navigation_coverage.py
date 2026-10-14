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


class CoupledCrescentTests(unittest.TestCase):
    """Coupled train: certify only the newly swept crescent, subtracting the body.

    The circular train footprint (radius >= 1.1, dominated by the towed cart behind)
    must NOT pull self-occluded near-side bearings (~+/-85-86 deg) into the required set.
    The coupled gate subtracts the actual occupied envelope and requires observation only
    for the forward crescent in the direction of travel; the lateral extent is covered by
    the footprint, but a turn's outward lateral bulge still demands observation.
    """

    FRONT = (0.52, 0.0, 0.0)  # lidar at the chassis front edge (mission mount)

    def _scan(self, free):
        r, u = [], []
        for i in range(BEAMS):
            clear = free(_bearing(i))
            r.append(RANGE_MAX if clear else math.nan)
            u.append(not clear)
        return Scan(0.1, tuple(r), -math.pi, INC, RANGE_MAX,
                    sensor_to_base=self.FRONT, unknown=tuple(u))

    def _gate(self, free, *, linear=0.25, angular=0.0):
        c = ArticulatedController(GridMap((-12, -12, 12, 12)))
        c.start((6., 0., 0.), True)
        obs = observation(self._scan(free), _odom(), coupled=True, arm_transport=True,
                          articulation=0.0)
        return c.gate_velocity(0.1, _odom(), obs, linear, angular), obs

    def test_near_side_self_occlusion_with_clear_forward_crescent_drives(self):
        # Forward cone clear; everything near-side (incl. -85.4/-86.4) and rear is UNKNOWN.
        result, obs = self._gate(lambda b: abs(b) <= math.radians(85))
        self.assertGreater(sum(1 for _, rch in obs.coverage.beams if rch == 0), 150)
        self.assertEqual(result.reason, 'tracking')
        self.assertGreater(result.linear, 0.0)

    def test_unobserved_lateral_obstacle_zone_in_crescent_refuses(self):
        # A blind wedge off to the forward-left: inside the swept crescent, not observed.
        result, _ = self._gate(lambda b: abs(b) <= math.radians(85)
                               and not (math.radians(-40) <= b <= math.radians(-10)))
        self.assertEqual(result.reason, 'insufficient_sensor_coverage')

    def test_blind_dead_ahead_refuses(self):
        result, _ = self._gate(lambda b: abs(b) <= math.radians(85)
                               and abs(b) >= math.radians(6))
        self.assertEqual(result.reason, 'insufficient_sensor_coverage')

    def test_stationary_coupled_train_certifies_despite_full_self_occlusion(self):
        c = ArticulatedController(GridMap((-12, -12, 12, 12)))
        c.start((0., 0., 0.), True)  # already at goal -> commands 0,0
        obs = observation(self._scan(lambda b: False), _odom(), coupled=True,
                          arm_transport=True, articulation=0.0)
        self.assertEqual(c.gate_velocity(0.1, _odom(), obs, 0.0, 0.0).reason, 'tracking')

    def test_certifies_sweep_exempts_lateral_face_requires_leading_face(self):
        # Direct geometry: a square body translating +x. Blind beam through the FRONT
        # face (leading) refuses; through a SIDE face (lateral) certifies.
        occupied = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]
        swept = [(-1., -1.), (1.3, -1.), (1.3, 1.), (-1., 1.)]  # translated +0.3 in x
        travel = (1.0, 0.0)
        front = CoverageField(RANGE_MAX, ((0.0, 0.0),), origin=(0.0, 0.0))   # blind ahead
        side = CoverageField(RANGE_MAX, ((math.pi / 2, 0.0),), origin=(0.0, 0.0))  # blind left
        self.assertFalse(front.certifies_sweep(occupied, swept, travel))
        self.assertTrue(side.certifies_sweep(occupied, swept, travel))


class UncoupledHeadingTests(unittest.TestCase):
    """Uncoupled collect-stage base: the same occupied-envelope subtraction applies.

    The base self-occludes its own sides and rear at short range; those beams are
    body-fixed, so the map-frame bearings that are blind rotate with the heading. The
    coverage gate must exempt them as footprint-covered at every heading (not only the
    one sampled at spawn) while still refusing a genuinely unobserved forward sector.
    Pins run-30's regression: forward-clear early ticks drove, later ticks (heading
    changed) refused under the old circular hemisphere requirement.
    """

    FRONT = (0.52, 0.0, 0.0)

    def _scan(self, body_blind):
        """Body-fixed occlusion: beam i blind iff body_blind(relative bearing) is True."""
        r, u = [], []
        for i in range(BEAMS):
            blind = body_blind(_bearing(i))
            r.append(math.nan if blind else RANGE_MAX)
            u.append(blind)
        return Scan(0.1, tuple(r), -math.pi, INC, RANGE_MAX,
                    sensor_to_base=self.FRONT, unknown=tuple(u))

    def _odom(self, yaw, lin=0.0):
        return Odometry((-5.73, 16.0, yaw), 0.1, lin, 0.0,
                        source='wheel_odometry_localized', position_uncertainty=0.05)

    def _gate(self, yaw, body_blind, *, lin=0.0, linear=0.25, angular=0.0):
        c = ArticulatedController(GridMap((-20, -20, 20, 20)))
        c.start((2., 16.0, yaw), False)  # uncoupled collect leg
        obs = observation(self._scan(body_blind), self._odom(yaw, lin),
                          coupled=False, arm_transport=True)
        return c.gate_velocity(0.1, self._odom(yaw, lin), obs, linear, angular), obs

    # The rig's real pattern: a wide forward cone is clear, sides+rear self-occlude.
    SIDE_REAR = staticmethod(lambda rel: abs(math.degrees(rel)) > 84.0)

    def test_side_rear_self_occlusion_drives_at_every_heading(self):
        for deg in (0, 37, 90, -130, 175):
            result, obs = self._gate(math.radians(deg), self.SIDE_REAR)
            self.assertGreater(sum(1 for _, rch in obs.coverage.beams if rch == 0), 150)
            self.assertEqual(result.reason, 'tracking', f'heading {deg}')

    def test_heading_change_mid_drive_stays_certified(self):
        # Simulate a turn mid-leg: the blind map-frame bearings sweep with the heading,
        # yet every heading still certifies because they stay outside the forward cone.
        reasons = [self._gate(math.radians(d), self.SIDE_REAR)[0].reason
                   for d in range(0, 360, 15)]
        self.assertTrue(all(r == 'tracking' for r in reasons), reasons)

    def test_blind_forward_sector_refuses(self):
        blind = lambda rel: self.SIDE_REAR(rel) or (math.radians(10) <= rel <= math.radians(35))
        self.assertEqual(self._gate(math.radians(90), blind)[0].reason,
                         'insufficient_sensor_coverage')

    def test_turn_in_place_low_speed_certifies(self):
        # v command 0 with tiny measured motion: no forward sweep -> covered by footprint.
        result, _ = self._gate(0.0, self.SIDE_REAR, lin=0.001, linear=0.0, angular=0.3)
        self.assertEqual(result.reason, 'tracking')

    def test_fully_self_occluded_uncoupled_cannot_start(self):
        self.assertEqual(self._gate(0.0, lambda rel: True)[0].reason,
                         'insufficient_sensor_coverage')

    def test_settle_stop_at_goal_with_residual_velocity_certifies(self):
        # run-31: the pickup goal equals the start pose, so the robot only ever issues the
        # settle STOP (v=0) with an infinitesimal residual stop_distance. The near-side
        # self-occluded beams must NOT be pulled into the crescent by that residual (the
        # old circular sweep_extent was non-monotonic there); the settle must certify.
        for residual in (0.0, 0.0005, 0.001, 0.0015, 0.002):
            c = ArticulatedController(GridMap((-20, -20, 20, 20)))
            c.start((-5.73, 16.0, 0.0), False)  # goal == current pose
            obs = observation(self._scan(self.SIDE_REAR), self._odom(0.0, residual),
                              coupled=False, arm_transport=True)
            gate = c.gate_velocity(0.1, self._odom(0.0, residual), obs, 0.0, 0.0)
            self.assertEqual(gate.reason, 'tracking', f'residual {residual}')
        # and the full settle resolves to arrived_stopped
        c = ArticulatedController(GridMap((-20, -20, 20, 20)))
        c.start((-5.73, 16.0, 0.0), False)
        obs = observation(self._scan(self.SIDE_REAR), self._odom(0.0, 0.001),
                          coupled=False, arm_transport=True)
        self.assertEqual(c.step(0.1, self._odom(0.0, 0.001), obs, now=0.2).reason,
                         'arrived_stopped')


if __name__ == '__main__':
    unittest.main()
