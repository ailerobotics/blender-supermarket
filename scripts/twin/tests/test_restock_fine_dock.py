"""CPU fixtures for the final fine-docking servo; no Isaac/ROS/GPU. Not a physics run."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.mobile import MobileSpec, Pose2, DockObservation
from twin.restock.fine_dock import FineDock, FineDockConfig, residuals, target_base, build

SPEC = MobileSpec()  # hitch_x=-0.62, cart_hitch_x=0.65


def dock(robot, cart, *, speed=0.0, yaw_rate=0.0, age=0.02):
    return DockObservation(robot=robot, cart=cart,
                           robot_hitch_z=0.28, cart_hitch_z=0.28,
                           robot_speed=speed, cart_speed=0.0,
                           robot_yaw_rate=yaw_rate, cart_yaw_rate=0.0, age=age)


def cfg(**over):
    base = dict(creep_speed=0.035, creep_yaw_rate=0.25, gain_lin=2.0, gain_yaw=1.0,
                hitch_tol_m=0.020, yaw_tol_rad=math.radians(4.0), settle_speed=0.01,
                time_cap_s=30.0)
    base.update(over)
    return FineDockConfig(**base)


class FineDockTests(unittest.TestCase):
    def test_target_base_seats_the_hitch(self):
        cart = Pose2(-5.73, 15.25, 0.0)
        tx, ty, tyaw = target_base(cart, SPEC)
        hm, yr = residuals(dock(Pose2(tx, ty, tyaw), cart), SPEC)
        self.assertAlmostEqual(hm, 0.0, places=6)
        self.assertAlmostEqual(yr, 0.0, places=6)

    def test_residuals_match_the_attach_predicate(self):
        cart = Pose2(-5.73, 15.25, 0.0)
        d = dock(Pose2(-4.53, 15.287, math.radians(-4.5)), cart)
        hm, yr = residuals(d, SPEC)
        a = d.robot.point(SPEC.hitch_x); b = d.cart.point(SPEC.cart_hitch_x)
        self.assertAlmostEqual(hm, math.dist(a, b))         # reproduces Coupler.attach's points
        self.assertGreater(hm, 0.025)                        # outside attach -> needs the servo

    def test_ready_immediately_when_seated_and_stopped(self):
        cart = Pose2(-5.73, 15.25, 0.0)
        tx, ty, tyaw = target_base(cart, SPEC)
        fd = FineDock(cfg(), SPEC); fd.start(0.0)
        r = fd.step(dock(Pose2(tx, ty, tyaw), cart), 0.1)
        self.assertEqual(r.status, 'ready')
        self.assertEqual((r.linear, r.angular), (0.0, 0.0))

    def test_seated_but_moving_brakes_and_settles(self):
        cart = Pose2(-5.73, 15.25, 0.0)
        tx, ty, tyaw = target_base(cart, SPEC)
        fd = FineDock(cfg(), SPEC); fd.start(0.0)
        r = fd.step(dock(Pose2(tx, ty, tyaw), cart, speed=0.03), 0.1)  # seated but moving
        self.assertEqual(r.status, 'creeping')
        self.assertEqual((r.linear, r.angular), (0.0, 0.0))

    def test_commands_reverse_when_the_drawbar_is_behind_the_rear_hitch(self):
        # If the rear hitch has overshot WEST past the drawbar, the base must REVERSE (v<0),
        # not rotate in place. (hitch_x<0, base heading +x: forward moves the hitch +x/east.)
        cart = Pose2(-5.73, 15.25, 0.0)
        tx, ty, tyaw = target_base(cart, SPEC)
        fd = FineDock(cfg(), SPEC); fd.start(0.0)
        # base 4 cm WEST of the seating pose, aligned: rear hitch is 4 cm west of the drawbar.
        r = fd.step(dock(Pose2(tx - 0.04, ty, 0.0), cart), 0.1)
        self.assertEqual(r.status, 'creeping')
        self.assertGreater(r.linear, 0.0)   # drawbar is EAST of the hitch -> drive forward (east)
        # base 4 cm EAST of the seating pose: rear hitch overshot east -> reverse.
        fd2 = FineDock(cfg(), SPEC); fd2.start(0.0)
        r2 = fd2.step(dock(Pose2(tx + 0.04, ty, 0.0), cart), 0.1)
        self.assertLess(r2.linear, 0.0)     # REVERSE

    def _closed_loop(self, start_pose, config, cart=Pose2(-5.73, 15.25, 0.0),
                     dt=0.05, max_t=60, pivot_off=0.13):
        """Integrate diff-drive kinematics WITH pivot drift (rotating dθ translates the base
        centre by pivot_off·dθ perpendicular) -- the real-physics effect cn-002 exposed."""
        fd = FineDock(config, SPEC); fd.start(0.0)
        x, y, yaw = start_pose
        v = w = 0.0
        t = 0.0
        last = None
        while t < max_t:
            d = dock(Pose2(x, y, yaw), cart, speed=abs(v), yaw_rate=abs(w), age=0.02)
            last = fd.step(d, t)
            if last.status in ('ready', 'failed'):
                return last, t
            v, w = last.linear, last.angular
            dth = w * dt
            x += v * math.cos(yaw) * dt - pivot_off * math.sin(yaw) * dth
            y += v * math.sin(yaw) * dt + pivot_off * math.cos(yaw) * dth
            yaw = math.atan2(math.sin(yaw + dth), math.cos(yaw + dth))
            t += dt
        return last, t

    def test_closed_loop_seats_a_cart_pinned_arrival_inside_attach(self):
        # Cart pinned at spawn; nav arrives ~65 mm short, roughly on-axis, +3 deg yaw.
        cart = Pose2(-5.73, 15.25, 0.0)
        tx, ty, _ = target_base(cart, SPEC)
        r, t = self._closed_loop((tx - 0.063, ty + 0.015, math.radians(3.0)), cfg(), cart=cart)
        self.assertEqual(r.status, 'ready')
        self.assertLessEqual(r.hitch_residual_m, 0.020)
        self.assertLessEqual(abs(r.yaw_residual_rad), math.radians(4.0))
        self.assertLess(t, 30.0)

    def test_time_cap_fails_honestly_with_residuals(self):
        cart = Pose2(-5.73, 15.25, 0.0)
        tx, ty, _ = target_base(cart, SPEC)
        r, t = self._closed_loop((tx - 0.063, ty + 0.015, math.radians(3.0)),
                                 cfg(creep_speed=0.001, creep_yaw_rate=0.001, time_cap_s=1.0),
                                 cart=cart)
        self.assertEqual(r.status, 'failed')
        self.assertIn('time cap', r.reason)
        self.assertGreater(r.hitch_residual_m, 0.020)

    def test_config_must_aim_strictly_inside_the_attach_handshake(self):
        with self.assertRaises(ValueError):
            cfg(hitch_tol_m=0.025)                      # == attach bound
        with self.assertRaises(ValueError):
            cfg(yaw_tol_rad=math.radians(5))            # == attach bound
        with self.assertRaises(ValueError):
            cfg(creep_speed=0.0)                        # nonpositive

    def test_step_before_start_refused(self):
        with self.assertRaises(ValueError):
            FineDock(cfg(), SPEC).step(dock(Pose2(0, 0, 0), Pose2(1.27, 0, 0)), 0.0)

    def test_build_absent_is_disabled_present_builds(self):
        self.assertIsNone(build({}))
        self.assertIsNone(build({'fine_dock': None}))
        fd = build({'fine_dock': dict(creep_speed=0.035, creep_yaw_rate=0.25, gain_lin=2.0,
                                      gain_yaw=1.0, hitch_tol_m=0.020, yaw_tol_rad=math.radians(4),
                                      settle_speed=0.01, time_cap_s=30.0)})
        self.assertIsInstance(fd, FineDock)


if __name__ == '__main__':
    unittest.main()
