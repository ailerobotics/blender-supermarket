"""CPU fixtures for the couple APPROACH fix (sim-robust-couple-approach-v1); no Isaac/ROS/GPU.

Root cause (reconstructed from the recorded runs' nav-trace, below): the authored ``couple_standoff``
is a FIXED map pose with yaw 0, but the pinned cart's MEASURED yaw varies run to run -- coldlivefinal
-1.58 deg, coupledock +19.17 deg -- so navigating to the fixed pose lands the base aligned to the MAP,
not the CART. The fine-dock then enters from an 8-27 deg yaw / 25-470 mm hitch residual it cannot null
(it lands on its cross-vs-yaw equilibrium) and times out at the fixed 30 s cap.

The fix: land the standoff on the MEASURED cart axis (``fine_dock.standoff_from_cart``) so the fine-dock
starts on-axis + yaw-aligned regardless of where the cart settled; gate entry on the cart-relative
cross-track + yaw (re-approach a fresh cart-measured standoff when outside, bounded); and replace the
fixed 30 s cap with a progress-based cap. These tests prove a CONSISTENT close landing from BOTH recorded
approaches and that the progress cap latches the coupledock-style monotone convergence the fixed cap cut.
"""
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.mobile import MobileSpec, Pose2, DockObservation
from twin.restock.fine_dock import (FineDock, FineDockConfig, build as build_fine_dock,
                                     standoff_from_cart, cart_relative_residuals, residuals)
from twin.restock.runtime import Packet, RuntimeAdapter
from twin.restock.inventory import Job

SPEC = MobileSpec()  # hitch_x=-0.62, cart_hitch_x=0.65

# The two recorded couple arrivals' MEASURED, world-pinned cart poses (last couple tick of each run's
# nav-trace; cart yaw = the degrees field, NOT the z height). coldlivefinal nearly latched (cart ~ the
# authored yaw 0); coupledock is the 19 deg-off cart that defeated the fixed-standoff fine-dock.
_CART_COLDLIVE = Pose2(-5.7445, 15.2366, math.radians(-1.58))
_CART_COUPLEDOCK = Pose2(-5.7423, 15.3672, math.radians(19.17))
_AUTHORED_STANDOFF = (-4.06, 15.25, 0.0)   # the fixed map pose both runs used (yaw 0)


def dock(robot, cart, *, speed=0.0, yaw_rate=0.0, age=0.02):
    return DockObservation(robot=robot, cart=cart, robot_hitch_z=0.28, cart_hitch_z=0.28,
                           robot_speed=speed, cart_speed=0.0, robot_yaw_rate=yaw_rate,
                           cart_yaw_rate=0.0, age=age)


def ct_cfg(**over):
    base = dict(creep_speed=0.035, creep_yaw_rate=0.25, gain_lin=2.0, gain_yaw=1.0,
                hitch_tol_m=0.020, yaw_tol_rad=math.radians(4.0), settle_speed=0.01, time_cap_s=30.0,
                approach_mode='crosstrack', ct_gain_approach=2.0, ct_gain_cross=8.0, ct_gain_yaw=1.5,
                ct_align_hold_m=3.0, ct_min_creep=0.015)
    base.update(over)
    return FineDockConfig(**base)


def _closed_loop(start_pose, config, cart, *, dt=0.05, max_t=140, pivot_off=0.13,
                 drive_deadband=(0.003, 0.02)):
    """Diff-drive integration WITH pivot drift + a near-zero-command motion floor (the cn-002 model the
    fine-dock tests already use), from an arbitrary start pose onto the pinned cart."""
    v_dead, w_dead = drive_deadband
    fd = FineDock(config, SPEC); fd.start(0.0)
    x, y, yaw = start_pose
    v = w = 0.0
    t = 0.0
    last = None
    while t < max_t:
        last = fd.step(dock(Pose2(x, y, yaw), cart, speed=abs(v), yaw_rate=abs(w)), t)
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


# --------------------------------------------------------------------------------------------------
# 1. Geometry: the cart-measured standoff gives a CONSISTENT on-axis, yaw-aligned entry for both carts.
# --------------------------------------------------------------------------------------------------
class CartRelativeStandoffGeometry(unittest.TestCase):
    def test_fixed_authored_standoff_misaligns_by_the_cart_yaw(self):
        # The recorded failure, reproduced as geometry: the fixed standoff yaw is 0, so the base-vs-cart
        # yaw residual IS the cart yaw, and the hitch residual grows with it (coupledock >> coldlivefinal).
        cold = residuals(dock(Pose2(*_AUTHORED_STANDOFF), _CART_COLDLIVE), SPEC)
        dockd = residuals(dock(Pose2(*_AUTHORED_STANDOFF), _CART_COUPLEDOCK), SPEC)
        # residual yaw = base yaw (0) - cart yaw, so it is the negated cart yaw.
        self.assertAlmostEqual(math.degrees(cold[1]), 1.58, places=1)
        self.assertAlmostEqual(math.degrees(dockd[1]), -19.17, places=1)
        self.assertGreater(dockd[0], 0.40)          # coupledock hitch residual ~ 0.47 m at the fixed pose
        self.assertGreater(dockd[0], cold[0])        # and far worse than coldlivefinal's

    def test_cart_measured_standoff_is_on_axis_and_aligned_for_both(self):
        for cart in (_CART_COLDLIVE, _CART_COUPLEDOCK):
            sx, sy, syaw = standoff_from_cart(cart, 0.40, SPEC)
            self.assertAlmostEqual(syaw, cart.yaw, places=9)         # base yaw == measured cart yaw
            hm, cross, yr = cart_relative_residuals(dock(Pose2(sx, sy, syaw), cart), SPEC)
            self.assertAlmostEqual(cross, 0.0, places=6)            # exactly on the cart axis
            self.assertAlmostEqual(yr, 0.0, places=9)               # exactly aligned
            self.assertAlmostEqual(hm, 0.40, places=6)              # only the deliberate along offset

    def test_worst_case_nav_arrival_stays_inside_the_capture_region(self):
        # Every LEGAL standoff arrival (base within arrival_xy_tol 0.05 m AND arrival_yaw_tol 0.08 rad,
        # ALL sign combinations) stays inside the default capture region (cross <= 0.11 m, yaw <= 0.10
        # rad) -- so a legal arrival always enters the fine-dock, never a spurious re-approach.
        for cart in (_CART_COLDLIVE, _CART_COUPLEDOCK):
            sx, sy, syaw = standoff_from_cart(cart, 0.40, SPEC)
            nx, ny = -math.sin(syaw), math.cos(syaw)
            for sxy in (+1, -1):
                for syw in (+1, -1):
                    base = Pose2(sx + sxy * 0.05 * nx, sy + sxy * 0.05 * ny, syaw + syw * 0.08)
                    _, cross, yr = cart_relative_residuals(dock(base, cart), SPEC)
                    self.assertLessEqual(abs(cross), 0.11)
                    self.assertLessEqual(abs(yr), 0.10)

    def test_standoff_offset_must_be_positive(self):
        with self.assertRaises(ValueError):
            standoff_from_cart(_CART_COLDLIVE, 0.0, SPEC)


# --------------------------------------------------------------------------------------------------
# 2. The fine-dock latches from the cart-measured standoff for BOTH recorded carts (consistent landing).
# --------------------------------------------------------------------------------------------------
class FineDockFromCartRelativeStandoff(unittest.TestCase):
    def test_both_recorded_carts_latch_from_the_cart_measured_standoff(self):
        for name, cart in (('coldlivefinal', _CART_COLDLIVE), ('coupledock', _CART_COUPLEDOCK)):
            s = standoff_from_cart(cart, 0.40, SPEC)
            r, t = _closed_loop(s, ct_cfg(progress_cap=True, progress_patience_s=12.0,
                                          progress_max_time_cap_s=120.0), cart, max_t=120)
            self.assertEqual(r.status, 'ready', f'{name} did not latch')
            self.assertLessEqual(r.hitch_residual_m, 0.020, name)
            self.assertLessEqual(abs(r.yaw_residual_rad), math.radians(4.0), name)

    def test_coupledock_cart_from_the_FIXED_standoff_fails_as_recorded(self):
        # The control: from the authored fixed standoff (yaw 0) against the 19 deg coupledock cart, the
        # fine-dock cannot null the structural yaw -> it fails (reproduces the recorded couple failure).
        r, _ = _closed_loop(_AUTHORED_STANDOFF, ct_cfg(), _CART_COUPLEDOCK, max_t=35)
        self.assertEqual(r.status, 'failed')
        self.assertGreater(abs(r.yaw_residual_rad), math.radians(4.0))

    def test_worst_case_arrival_still_latches_for_both(self):
        for name, cart in (('coldlivefinal', _CART_COLDLIVE), ('coupledock', _CART_COUPLEDOCK)):
            sx, sy, syaw = standoff_from_cart(cart, 0.40, SPEC)
            nx, ny = -math.sin(syaw), math.cos(syaw)
            start = (sx + 0.05 * nx, sy + 0.05 * ny, syaw + math.radians(4.6))
            r, _ = _closed_loop(start, ct_cfg(progress_cap=True), cart, max_t=120)
            self.assertEqual(r.status, 'ready', name)
            self.assertLessEqual(r.hitch_residual_m, 0.020, name)


# --------------------------------------------------------------------------------------------------
# 3. Progress-based cap: latches a slow monotone convergence the fixed 30 s cut; fails a limit cycle fast.
# --------------------------------------------------------------------------------------------------
class ProgressCap(unittest.TestCase):
    def _converging_dock(self, now, hitch_m):
        """A base whose rear hitch sits ``hitch_m`` straight behind the drawbar on the cart axis (yaw
        aligned, on-axis): the hitch residual == hitch_m, so a scripted schedule drives the measured
        residual however the test wants, independent of the servo command."""
        cart = Pose2(-5.73, 15.25, 0.0)
        tx, ty, tyaw = standoff_from_cart(cart, hitch_m, SPEC)
        return dock(Pose2(tx, ty, tyaw), cart, speed=0.01)

    def test_progress_cap_latches_a_slow_monotone_approach_past_the_fixed_cap(self):
        # A ~13 mm/s monotone close (coupledock's measured rate) from 0.47 m needs ~36 s -- PAST the fixed
        # 30 s cap. With progress_cap it keeps going (improving every tick) and latches; the fixed cap fails.
        cfg_fixed = ct_cfg(time_cap_s=30.0)
        cfg_prog = ct_cfg(progress_cap=True, progress_patience_s=10.0, progress_max_time_cap_s=120.0)
        fd_fixed = FineDock(cfg_fixed, SPEC); fd_fixed.start(0.0)
        fd_prog = FineDock(cfg_prog, SPEC); fd_prog.start(0.0)
        fixed_failed = prog_ready = False
        t = 0.0
        while t <= 45.0:
            hitch = max(0.0, 0.47 - 0.013 * t)      # monotone close at 13 mm/s
            d = self._converging_dock(t, hitch)
            rf = fd_fixed.step(d, t)
            rp = fd_prog.step(d, t)
            if rf.status == 'failed' and 30.0 <= t <= 31.0:
                self.assertIn('time cap', rf.reason)
                fixed_failed = True
            if rp.status == 'ready':
                prog_ready = True
                break
            t += 0.1
        self.assertTrue(fixed_failed, 'fixed cap should have failed the slow approach at 30 s')
        self.assertTrue(prog_ready, 'progress cap should latch the monotone approach past 30 s')
        self.assertGreater(t, 30.0)                 # it genuinely ran longer than the fixed cap allowed

    def test_progress_cap_fails_a_limit_cycle_at_the_patience_window(self):
        # A stuck residual (no net improvement) fails at the patience window, NOT the hard cap, with a
        # named 'no progress' reason -- fast-failing the coldlivefinal-style limit cycle honestly.
        cfg = ct_cfg(progress_cap=True, progress_patience_s=8.0, progress_max_time_cap_s=120.0)
        fd = FineDock(cfg, SPEC); fd.start(0.0)
        r, t = None, 0.0
        while t <= 30.0:
            d = self._converging_dock(t, 0.030 + 0.001 * math.sin(t * 3))   # dithers ~30 mm, never closes
            r = fd.step(d, t)
            if r.status == 'failed':
                break
            t += 0.1
        self.assertEqual(r.status, 'failed')
        self.assertIn('no progress', r.reason)
        self.assertLess(t, 20.0)                    # at ~patience, far before the 120 s hard cap

    def test_progress_cap_validation(self):
        with self.assertRaises(ValueError):
            ct_cfg(progress_cap=True, progress_max_time_cap_s=10.0)   # < time_cap_s (30)
        with self.assertRaises(ValueError):
            ct_cfg(progress_cap=True, progress_patience_s=0.0)        # nonpositive
        # off (default) => the progress knobs are unread, config is valid
        self.assertFalse(ct_cfg().progress_cap)

    def test_progress_cap_absent_is_byte_identical_to_the_fixed_cap(self):
        import random
        rng = random.Random(11)
        base = ct_cfg()
        prog_default = ct_cfg()
        cart = Pose2(-5.73, 15.25, 0.0)
        tx, ty, tyaw = standoff_from_cart(cart, 0.1, SPEC)
        for _ in range(200):
            p = Pose2(tx + rng.uniform(-0.4, 0.4), ty + rng.uniform(-0.2, 0.2),
                      tyaw + rng.uniform(-0.3, 0.3))
            d = dock(p, cart, speed=rng.uniform(0, 0.03))
            a = FineDock(base, SPEC); a.start(0.0)
            b = FineDock(prog_default, SPEC); b.start(0.0)
            for now in (0.1, 10.0, 31.0):           # 31 s straddles the fixed cap
                ra, rb = a.step(d, now), b.step(d, now)
                self.assertEqual((ra.status, ra.linear, ra.angular),
                                 (rb.status, rb.linear, rb.angular))


# --------------------------------------------------------------------------------------------------
# 4. Runtime couple stage: re-targets the standoff to the measured cart, enters/​re-approaches/​fails.
# --------------------------------------------------------------------------------------------------
class _CoupleHarness:
    """A perfect-arrival nav + port fixture: each navigate leg 'arrives' with the base placed AT the last
    requested goal (plus an optional constant arrival error), the cart pinned at ``cart``. Drives the real
    RuntimeAdapter couple path on CPU."""
    def __init__(self, cart, *, couple_cart_relative=None, fine_dock_cfg=None, arrival_err=(0.0, 0.0),
                 poses_override=None):
        self.cart = cart
        self.goals = []
        self.last_goal = (-4.46, 16.0, 0.0)
        self.arr_dxy, self.arr_dyaw = arrival_err
        self.commands = []
        self._sim = 0.0
        self._n = 0
        nav = NS(start=self._nav_start,
                 step=lambda dt, p: NS(status='succeeded', linear=0.0, angular=0.0, reason=''),
                 cancel=lambda: None)
        mobile = NS(park=lambda: None, hold=lambda: None, release=lambda: None,
                    command=lambda *a, **k: self.commands.append(a),
                    hold_cart_pin=lambda: None, attach=lambda o: None,
                    coupling_state=lambda o: {'healthy': True})
        manip = NS(start=lambda a, j: None, cart_location='rollwagen',
                   step=lambda dt, s: NS(status='running', reason='', controls=()))
        port = NS(advance=self._advance, controls=lambda c: None, stop=lambda: None)
        poses = poses_override or {'home': [0, 0, 0], 'pickup': [0, 0, 0], 'cart_dock': [-4.46, 15.25, 0],
                                   'shelf_approach': [0, 0, 0], 'couple_approach': [-4.46, 16.0, 0.0],
                                   'couple_standoff': list(_AUTHORED_STANDOFF)}
        fd = build_fine_dock({'fine_dock': fine_dock_cfg}) if fine_dock_cfg else None
        self.adapter = RuntimeAdapter(nav, mobile, manip, port, poses, fixture_mode=True,
                                      fine_dock=fd, hold_cart_through_couple=True,
                                      couple_cart_relative=couple_cart_relative)

    def _nav_start(self, goal, coupled, **kw):
        self.goals.append(tuple(round(v, 4) for v in goal))
        self.last_goal = tuple(goal)

    def _advance(self, dt):
        self._sim += dt
        self._n += 1
        gx, gy, gyaw = self.last_goal
        robot = Pose2(gx + self.arr_dxy, gy, gyaw + self.arr_dyaw)
        d = dock(robot, self.cart)
        return Packet(f'f-{self._n}', 'test_fixture', self._sim, NS(pose=(gx, gy, gyaw)),
                      None, d, None, True, False, 0.0, True)

    def run_couple(self, job, max_steps=12):
        self.adapter.start('couple', job)
        results = []
        for _ in range(max_steps):
            r = self.adapter.step(0.05)
            results.append(r)
            if self.adapter.phase == 'fine_dock' or r.status in ('failed', 'succeeded'):
                break
        return results


class CoupleStageCartRelative(unittest.TestCase):
    JOB = Job('j', 'milk', 1, 'back', 'shelf')
    FD = dict(creep_speed=0.035, creep_yaw_rate=0.25, gain_lin=2.0, gain_yaw=1.0, hitch_tol_m=0.020,
              yaw_tol_rad=math.radians(4.0), settle_speed=0.01, time_cap_s=30.0,
              approach_mode='crosstrack', ct_gain_approach=2.0, ct_gain_cross=8.0, ct_gain_yaw=1.5,
              ct_align_hold_m=3.0, ct_min_creep=0.015)

    def test_absent_cart_relative_uses_the_authored_fixed_standoff(self):
        # Backward compatible: without the config block the standoff leg navigates to the authored pose.
        h = _CoupleHarness(_CART_COUPLEDOCK, fine_dock_cfg=self.FD)
        h.run_couple(self.JOB)
        self.assertEqual(h.goals[0], (-4.46, 16.0, 0.0))                 # couple_approach
        self.assertEqual(h.goals[1], tuple(round(v, 4) for v in _AUTHORED_STANDOFF))  # authored standoff

    def test_cart_relative_retargets_the_standoff_to_the_measured_cart(self):
        # The fix: the standoff leg goes to standoff_from_cart(measured cart), NOT the fixed pose, for both.
        for cart in (_CART_COLDLIVE, _CART_COUPLEDOCK):
            h = _CoupleHarness(cart, fine_dock_cfg=self.FD, couple_cart_relative={})
            h.run_couple(self.JOB)
            expect = tuple(round(v, 4) for v in standoff_from_cart(cart, 0.40, SPEC))
            self.assertEqual(h.goals[0], (-4.46, 16.0, 0.0))             # couple_approach unchanged
            self.assertEqual(h.goals[1], expect)                         # cart-measured standoff
            self.assertEqual(h.adapter.phase, 'fine_dock')               # entered from an aligned pose

    def test_both_recorded_carts_enter_fine_dock_from_a_consistent_pose(self):
        # Consistent landing: from both recorded cart yaws (-1.6 deg and +19.2 deg) the couple stage enters
        # the fine-dock with the SAME small cart-relative residual (cross ~0, yaw ~0) -- the deliverable.
        entries = []
        for cart in (_CART_COLDLIVE, _CART_COUPLEDOCK):
            h = _CoupleHarness(cart, fine_dock_cfg=self.FD, couple_cart_relative={})
            h.run_couple(self.JOB)
            self.assertEqual(h.adapter.phase, 'fine_dock')
            sx, sy, syaw = standoff_from_cart(cart, 0.40, SPEC)
            _, cross, yr = cart_relative_residuals(dock(Pose2(sx, sy, syaw), cart), SPEC)
            entries.append((cross, yr))
        for cross, yr in entries:
            self.assertAlmostEqual(cross, 0.0, places=3)
            self.assertAlmostEqual(yr, 0.0, places=3)

    def test_reapproach_then_fail_honestly_when_arrival_stays_out_of_capture(self):
        # If nav cannot achieve the aligned standoff (here a persistent 0.3 rad yaw error every arrival),
        # the stage re-approaches the bounded number of times, then FAILS honestly with the residual --
        # never loosening the latch, never looping.
        h = _CoupleHarness(_CART_COUPLEDOCK, fine_dock_cfg=self.FD,
                           couple_cart_relative={'max_reapproach': 2}, arrival_err=(0.0, 0.3))
        results = h.run_couple(self.JOB, max_steps=20)
        self.assertEqual(results[-1].status, 'failed')
        self.assertIn('outside capture', results[-1].reason)
        self.assertIn('re-approach', results[-1].reason)
        # couple_approach + (initial standoff + 2 re-approaches) = 4 nav legs, then fail.
        self.assertEqual(len(h.goals), 4)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            _CoupleHarness(_CART_COLDLIVE, couple_cart_relative={'standoff_offset_m': 0.0})
        with self.assertRaises(ValueError):
            _CoupleHarness(_CART_COLDLIVE, couple_cart_relative={'max_reapproach': -1})
        with self.assertRaises(ValueError):
            _CoupleHarness(_CART_COLDLIVE, couple_cart_relative={'bogus': 1})


if __name__ == '__main__':
    unittest.main()
