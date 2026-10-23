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


def ct_cfg(**over):
    """The opt-in cross-track terminal-docking mode (sim-robust-couple-fine-dock-v1)."""
    ct = dict(approach_mode='crosstrack', ct_gain_approach=2.0, ct_gain_cross=8.0,
              ct_gain_yaw=1.5, ct_align_hold_m=3.0, ct_min_creep=0.015)
    ct.update(over)
    return cfg(**ct)


# --- Recorded couple standoffs (run dirs under ~/datasets/twin-robot-in-store-boot-v1/mission) ---
# The couple-nav parks the base at the couple_standoff; fine_dock then creeps onto the MEASURED cart
# drawbar. These seeds are the two recorded couple arrivals: the base ODOMETRY arrival pose from the
# last NAV_ARRIVAL_DIAG before the first FINE_DOCK_DIAG, the base yaw = measured cart yaw + the first
# FINE_DOCK_DIAG yaw residual, and the measured cart pose from the RESTOCK_TRACE_POSES nav-trace.
_CART = Pose2(-5.745, 15.237, math.radians(0.20))         # measured, world-pinned through the approach
# coldlivefinal: the real live_pose-collect run that FAILED the fine-dock (console 31552..33822,
# 'fine_dock time cap 30s: residual hitch=26.7mm yaw=4.78deg'); first FINE_DOCK yaw residual -2.93 deg.
_COLDLIVE_STANDOFF = Pose2(-4.101, 15.298, math.radians(0.20 - 2.93))
# coldhonest: the still-arm run that LATCHED the same dock (156->28 mm); first yaw residual -4.03 deg.
_COLDHONEST_STANDOFF = Pose2(-4.101, 15.299, math.radians(0.20 - 4.03))


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
                     dt=0.05, max_t=60, pivot_off=0.13, drive_deadband=(0.0, 0.0)):
        """Integrate diff-drive kinematics WITH pivot drift (rotating dθ translates the base
        centre by pivot_off·dθ perpendicular) -- the real-physics effect cn-002 exposed.

        ``drive_deadband=(v_dead, w_dead)`` models the real drive's near-zero-command motion floor:
        a velocity target too small to overcome static friction / lost in the documented ~14 mm/s
        PhysX-TGS velocity bias produces NO net motion. It is 0 by default (the pre-existing tests
        are unchanged). At the legacy law's degenerate equilibrium the commands fall BELOW this floor,
        so the base freezes there -- the coldlivefinal couple limit cycle."""
        v_dead, w_dead = drive_deadband
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
            v_eff = 0.0 if abs(v) < v_dead else v
            w_eff = 0.0 if abs(w) < w_dead else w
            dth = w_eff * dt
            x += v_eff * math.cos(yaw) * dt - pivot_off * math.sin(yaw) * dth
            y += v_eff * math.sin(yaw) * dt + pivot_off * math.cos(yaw) * dth
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

    # ---- sim-robust-couple-fine-dock-v1: the coldlivefinal couple limit cycle + the cross-track fix ----

    def _on_legacy_equilibrium(self, yaw_deg, cart=_CART):
        """A dock pose EXACTLY on the legacy law's zero-command equilibrium line: fwd=0 and
        lat = (gain_yaw*|hitch_x|/gain_lin)*yaw_err, where the legacy ω cancels to ~0."""
        c = cfg()
        ye = math.radians(yaw_deg)
        byaw = cart.yaw + ye
        lat = (c.gain_yaw * abs(SPEC.hitch_x) / c.gain_lin) * ye
        dbx, dby = cart.point(SPEC.cart_hitch_x)
        # hitch world = drawbar + lat along the base-frame normal; base centre = hitch - R(byaw)@(hitch_x,0)
        hxw = dbx - math.sin(byaw) * lat
        hyw = dby + math.cos(byaw) * lat
        return Pose2(hxw - math.cos(byaw) * SPEC.hitch_x, hyw - math.sin(byaw) * SPEC.hitch_x, byaw)

    def test_legacy_law_stalls_on_its_degenerate_equilibrium(self):
        # Reproduces the coldlivefinal failure mechanism, model-independently: at the recorded stuck
        # residual (~26 mm / ~4.78 deg) the legacy ω cancels (lateral vs yaw) and v is tiny, so BOTH
        # commands fall below the drive's near-zero-command motion floor -> the base freezes OUTSIDE
        # the latch box (hitch > 20 mm OR |yaw| > 4 deg) -> 30 s time cap.
        base = self._on_legacy_equilibrium(4.78)
        d = dock(base, _CART)
        hm, yr = residuals(d, SPEC)
        self.assertGreater(hm, 0.020)                         # out of hitch tol (matches ~26 mm)
        self.assertGreater(abs(yr), math.radians(4.0))        # out of yaw tol (matches ~4.78 deg)
        fd = FineDock(cfg(), SPEC); fd.start(0.0)
        r = fd.step(d, 0.1)
        self.assertEqual(r.status, 'creeping')
        self.assertLess(abs(r.linear), 0.005)                 # below any plausible drive floor
        self.assertLess(abs(r.angular), 0.005)                # yaw NEVER nulls: ω ~ 0 here

    def test_crosstrack_commands_real_motion_at_the_legacy_equilibrium(self):
        # The fix's guarantee: where the legacy ω cancels to ~0 (frozen), the cross-track servo still
        # commands a real steering rate above the drive floor -- so the base is never content outside
        # the latch box. (Here along≈0, so the escape is the cross-track STEER onto the axis, not v.)
        d = dock(self._on_legacy_equilibrium(4.78), _CART)
        legacy = FineDock(cfg(), SPEC); legacy.start(0.0)
        rl = legacy.step(d, 0.1)
        self.assertLess(abs(rl.angular), 0.005)               # legacy: frozen steering
        fd = FineDock(ct_cfg(), SPEC); fd.start(0.0)
        r = fd.step(d, 0.1)
        self.assertEqual(r.status, 'creeping')
        self.assertGreater(abs(r.angular), 0.02)              # cross-track: steers above the drive floor

    def test_crosstrack_seats_the_recorded_coldlivefinal_standoff(self):
        # The deliverable: from the pose the real live_pose collect leaves the robot in, the legacy law
        # limit-cycles (time cap) under the drive deadband, but the cross-track mode converges + latches.
        dead = (0.003, 0.02)
        legacy, _ = self._closed_loop(
            (_COLDLIVE_STANDOFF.x, _COLDLIVE_STANDOFF.y, _COLDLIVE_STANDOFF.yaw),
            cfg(), cart=_CART, max_t=35, drive_deadband=dead)
        self.assertEqual(legacy.status, 'failed')             # reproduces the recorded couple failure
        self.assertIn('time cap', legacy.reason)
        r, t = self._closed_loop(
            (_COLDLIVE_STANDOFF.x, _COLDLIVE_STANDOFF.y, _COLDLIVE_STANDOFF.yaw),
            ct_cfg(), cart=_CART, max_t=30, drive_deadband=dead)
        self.assertEqual(r.status, 'ready')
        self.assertLessEqual(r.hitch_residual_m, 0.020)
        self.assertLessEqual(abs(r.yaw_residual_rad), math.radians(4.0))
        self.assertLess(t, 30.0)

    def test_crosstrack_still_seats_the_coldhonest_standoff(self):
        # No regression: the cross-track mode also seats the standoff the still-arm run already latched.
        r, t = self._closed_loop(
            (_COLDHONEST_STANDOFF.x, _COLDHONEST_STANDOFF.y, _COLDHONEST_STANDOFF.yaw),
            ct_cfg(), cart=_CART, max_t=30, drive_deadband=(0.003, 0.02))
        self.assertEqual(r.status, 'ready')
        self.assertLessEqual(r.hitch_residual_m, 0.020)
        self.assertLessEqual(abs(r.yaw_residual_rad), math.radians(4.0))

    def test_crosstrack_converges_across_pivot_drift(self):
        # Robust to the base's rotation-vs-translation coupling (the cn-002 pivot effect), which the
        # real value of is unknown -- both recorded standoffs latch across a wide pivot range.
        for pivot in (0.0, 0.08, 0.13, 0.20):
            for name, s in (('coldlive', _COLDLIVE_STANDOFF), ('coldhonest', _COLDHONEST_STANDOFF)):
                r, _ = self._closed_loop((s.x, s.y, s.yaw), ct_cfg(), cart=_CART, max_t=30,
                                         pivot_off=pivot, drive_deadband=(0.003, 0.02))
                self.assertEqual(r.status, 'ready', f'{name} failed at pivot_off={pivot}')
                self.assertLessEqual(r.hitch_residual_m, 0.020)
                self.assertLessEqual(abs(r.yaw_residual_rad), math.radians(4.0))

    def test_absent_approach_mode_is_byte_identical_to_the_legacy_law(self):
        # Backward compatible: the default approach_mode reproduces the legacy creep command for every
        # pose -- a config without the new keys is byte-identical (the no-change path the spec requires).
        default = FineDockConfig(creep_speed=0.035, creep_yaw_rate=0.25, gain_lin=2.0, gain_yaw=1.0,
                                 hitch_tol_m=0.020, yaw_tol_rad=math.radians(4.0), settle_speed=0.01,
                                 time_cap_s=30.0)
        self.assertEqual(default.approach_mode, 'hitch_jacobian')
        import random
        rng = random.Random(7)
        tx, ty, tyaw = target_base(_CART, SPEC)
        for _ in range(400):
            p = Pose2(tx + rng.uniform(-0.5, 0.5), ty + rng.uniform(-0.2, 0.2),
                      tyaw + rng.uniform(-0.3, 0.3))
            d = dock(p, _CART, speed=rng.uniform(0, 0.03), yaw_rate=rng.uniform(0, 0.03))
            a = FineDock(default, SPEC); a.start(0.0)
            b = FineDock(cfg(), SPEC); b.start(0.0)     # cfg() also omits approach_mode -> default
            ra, rb = a.step(d, 0.1), b.step(d, 0.1)
            self.assertEqual((ra.status, ra.linear, ra.angular),
                             (rb.status, rb.linear, rb.angular))

    def test_crosstrack_config_validation(self):
        with self.assertRaises(ValueError):
            cfg(approach_mode='bogus')                        # unknown mode
        with self.assertRaises(ValueError):
            ct_cfg(ct_gain_cross=0.0)                         # nonpositive gain in crosstrack mode
        with self.assertRaises(ValueError):
            ct_cfg(ct_min_creep=0.035)                        # >= creep_speed
        # a nonpositive ct_* gain is ignored while the legacy mode is active (the gains are unread)
        self.assertEqual(cfg(ct_gain_cross=-1.0).approach_mode, 'hitch_jacobian')

    def test_crosstrack_builds_from_config(self):
        fd = build({'fine_dock': dict(
            creep_speed=0.035, creep_yaw_rate=0.25, gain_lin=2.0, gain_yaw=1.0,
            hitch_tol_m=0.020, yaw_tol_rad=math.radians(4), settle_speed=0.01, time_cap_s=30.0,
            approach_mode='crosstrack', ct_gain_approach=2.0, ct_gain_cross=8.0, ct_gain_yaw=1.5,
            ct_align_hold_m=3.0, ct_min_creep=0.015)})
        self.assertIsInstance(fd, FineDock)
        self.assertEqual(fd.config.approach_mode, 'crosstrack')

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
