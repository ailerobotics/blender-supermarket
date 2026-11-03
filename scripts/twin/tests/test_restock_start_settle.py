"""CPU tests for the start-phase shortcut's sim-time settle + rest/heading verify
(sim-couple-shortcut-settle-dt-v1).

No Isaac/GPU: ``start_settle`` is pure (injected step/sample), and ``apply_start_state`` runs here
against lightweight body/world/mobile fakes. Covers:

* the settle is SIM-TIME, not a tick count: a decaying disturbance converges at ~the same sim time at
  1/60 and 1/36 (different step counts), and a "paced" step changes nothing;
* the resting / heading verify (``START_STATE_UNSUPPORTED`` / ``START_STATE_SETTLE_MISMATCH``) applied to
  the RECORDED coldwin25c (box-floor mis-seat) vs towstaging3 (good seat) cart poses -- the dry-run that
  shows the verify would have caught the failure and passes the good run;
* ``apply_start_state`` for start_at=couple: holding the train at its seat through the settle makes a
  would-be-tipping cart seat correctly (the fix) while an un-held cart on the same bad floor is caught.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.start_settle import (resolve_settle_params, converge_settle, resting_check,
                                       heading_check, _DEFAULTS)
from twin.restock.shortcut import (apply_start_state, resolve_couple_dock, dock_align_scenario,
                                   couple_start_poses, verify_dock_seat, staged_scenario)


# --------------------------------------------------------------------------- #
# resolve_settle_params                                                        #
# --------------------------------------------------------------------------- #
class ResolveParamsTests(unittest.TestCase):
    def test_defaults_when_absent(self):
        self.assertEqual(resolve_settle_params({}), dict(_DEFAULTS))
        self.assertEqual(resolve_settle_params({'start_state_settle': None}), dict(_DEFAULTS))

    def test_overrides_and_degree_forms(self):
        p = resolve_settle_params({'start_state_settle': {
            'min_settle_s': 0.3, 'window_s': 0.1, 'max_settle_s': 2.0,
            'heading_tol_deg': 2.0, 'rest_tilt_tol_deg': 1.0, 'rest_z_tol_m': 0.01,
            'hold_during_settle': False}})
        self.assertEqual(p['min_settle_s'], 0.3)
        self.assertAlmostEqual(p['heading_tol_rad'], math.radians(2.0))
        self.assertAlmostEqual(p['rest_tilt_tol_rad'], math.radians(1.0))
        self.assertEqual(p['rest_z_tol_m'], 0.01)
        self.assertFalse(p['hold_during_settle'])

    def test_rejects_malformed(self):
        for bad in ({'min_settle_s': 'x'}, {'window_s': 0}, {'max_settle_s': -1},
                    {'max_settle_s': 0.1, 'min_settle_s': 0.5}, {'hold_during_settle': 'yes'}):
            with self.assertRaises(ValueError):
                resolve_settle_params({'start_state_settle': bad})


# --------------------------------------------------------------------------- #
# converge_settle -- sim-time, not tick count                                  #
# --------------------------------------------------------------------------- #
class _Decay:
    """A body whose yaw+z disturbance decays exponentially in SIM time (not tick count)."""
    def __init__(self, dt, *, tau=0.3, amp_yaw=0.15, amp_z=0.08, base_z=0.2, base_yaw=0.0):
        self.dt = dt; self.tau = tau; self.amp_yaw = amp_yaw; self.amp_z = amp_z
        self.base_z = base_z; self.base_yaw = base_yaw; self.t = 0.0; self.steps = 0

    def step(self):
        self.t += self.dt; self.steps += 1

    def sample(self):
        d = math.exp(-self.t / self.tau)
        return {'carton': (-5.73, 15.25, self.base_z + self.amp_z * d, self.base_yaw + self.amp_yaw * d)}


class ConvergeSettleTests(unittest.TestCase):
    def test_static_body_converges_after_min_settle(self):
        p = resolve_settle_params({'start_state_settle': {'min_settle_s': 0.5, 'window_s': 0.2,
                                                          'max_settle_s': 2.0}})
        dt = 1.0 / 60.0
        sample = lambda: {'carton': (-5.73, 15.25, 0.2, 0.0)}
        r = converge_settle(step=lambda: None, sample=sample, dt=dt, params=p)
        self.assertTrue(r['converged'])
        self.assertGreaterEqual(r['elapsed_s'], 0.5 - dt)        # never before the min settle floor
        self.assertLess(r['elapsed_s'], 0.5 + 0.2 + 2 * dt)      # released promptly once the window covers

    def test_sim_time_independent_of_physics_step(self):
        # Same decaying disturbance at 1/60 and 1/36: convergence happens at ~the same SIM time, with a
        # DIFFERENT number of steps -- i.e. the settle is governed by sim seconds, not a tick count.
        p = resolve_settle_params({'start_state_settle': {'min_settle_s': 0.2, 'window_s': 0.1,
                                                          'max_settle_s': 5.0, 'angular_tol': 0.02,
                                                          'linear_tol': 0.02}})
        results = {}
        for dt in (1.0 / 60.0, 1.0 / 36.0):
            m = _Decay(dt)
            r = converge_settle(step=m.step, sample=m.sample, dt=dt, params=p)
            self.assertTrue(r['converged'], f'did not converge at dt={dt}')
            results[dt] = r
        e60, e36 = results[1.0 / 60.0]['elapsed_s'], results[1.0 / 36.0]['elapsed_s']
        self.assertLess(abs(e60 - e36), 2 * (1.0 / 36.0) + 1e-6)     # same sim time within a coarse step
        self.assertNotEqual(results[1.0 / 60.0]['steps'], results[1.0 / 36.0]['steps'])  # different ticks

    def test_pacing_does_not_change_the_settle(self):
        # "max pacing" is a WALL concept; the settle advances by dt per step regardless. A step wrapper
        # that models pacing (extra wall work, no sim change) yields a byte-identical settle verdict.
        p = resolve_settle_params({'start_state_settle': {'min_settle_s': 0.2, 'window_s': 0.1,
                                                          'max_settle_s': 5.0, 'angular_tol': 0.02,
                                                          'linear_tol': 0.02}})
        dt = 1.0 / 36.0
        plain = _Decay(dt); r_plain = converge_settle(step=plain.step, sample=plain.sample, dt=dt, params=p)
        paced = _Decay(dt)
        wall = {'n': 0}

        def paced_step():
            wall['n'] += 1          # stand-in for the pacing wall-sleep; no effect on sim state
            paced.step()
        r_paced = converge_settle(step=paced_step, sample=paced.sample, dt=dt, params=p)
        self.assertEqual(r_plain['steps'], r_paced['steps'])
        self.assertEqual(r_plain['elapsed_s'], r_paced['elapsed_s'])
        self.assertEqual(r_plain['converged'], r_paced['converged'])
        self.assertEqual(wall['n'], r_paced['steps'])

    def test_fixed_tick_count_would_differ_by_step(self):
        # The regression the fix removes: 30 ticks is 0.50 s at 1/60 but 0.83 s at 1/36 -- a settle that
        # counts ticks silently changes its sim duration with the step (the coldwin25 "no coarse step"
        # workaround). The sim-time settle above does not.
        self.assertNotAlmostEqual(30 * (1.0 / 60.0), 30 * (1.0 / 36.0))


# --------------------------------------------------------------------------- #
# resting / heading verify, incl. the recorded-pose dry-run                    #
# --------------------------------------------------------------------------- #
class VerifyTests(unittest.TestCase):
    P = dict(_DEFAULTS)

    def test_resting_ok_and_unsupported(self):
        ok, reason = resting_check(dict(z=0.200, roll_rad=0.0, pitch_rad=0.0, lin_speed=0.0, ang_speed=0.0),
                                   {'z': 0.199}, self.P)
        self.assertTrue(ok); self.assertIsNone(reason)
        ok, reason = resting_check(dict(z=0.275, roll_rad=0.0, pitch_rad=0.0, lin_speed=0.0, ang_speed=0.0),
                                   {'z': 0.199}, self.P)
        self.assertFalse(ok); self.assertIn('START_STATE_UNSUPPORTED', reason)
        ok, reason = resting_check(dict(z=0.199, roll_rad=math.radians(6), pitch_rad=0.0,
                                        lin_speed=0.0, ang_speed=0.0), {'z': 0.199}, self.P)
        self.assertFalse(ok); self.assertIn('tilt', reason)
        ok, reason = resting_check(dict(z=0.199, roll_rad=0.0, pitch_rad=0.0, lin_speed=0.5, ang_speed=0.0),
                                   {'z': 0.199}, self.P)
        self.assertFalse(ok); self.assertIn('moving', reason)

    def test_heading_ok_and_mismatch(self):
        ok, res, reason = heading_check(math.radians(-90.2), math.radians(-90), self.P['heading_tol_rad'])
        self.assertTrue(ok); self.assertIsNone(reason)
        ok, res, reason = heading_check(math.radians(6.7), 0.0, self.P['heading_tol_rad'])
        self.assertFalse(ok); self.assertIn('START_STATE_SETTLE_MISMATCH', reason)

    def test_dry_run_against_recorded_run_poses(self):
        # Recorded nav-trace cart [x, y, yaw_deg, z] (the pinned post-settle pose) from the saved runs:
        #   coldwin25c (portal_floor_box ON, mis-seated) : yaw 6.71 deg, z 0.2202
        #   towstaging3 (portal_floor_box OFF, good seat): yaw -90.0 deg, z 0.1999
        # The supported-floor resting z is ~0.199 (towstaging3). The verify flags 25c and passes towstaging3.
        rest_z = 0.199
        # coldwin25c: tipped (z up 21 mm) AND mis-headed vs the -90 deg dock -> both trips fire.
        ok, reason = resting_check(dict(z=0.2202, roll_rad=0.0, pitch_rad=0.0, lin_speed=0.0, ang_speed=0.0),
                                   {'z': rest_z}, dict(_DEFAULTS))
        self.assertFalse(ok); self.assertIn('START_STATE_UNSUPPORTED', reason)
        ok, res, reason = heading_check(math.radians(6.71), math.radians(-90.0),
                                        _DEFAULTS['heading_tol_rad'])
        self.assertFalse(ok); self.assertIn('START_STATE_SETTLE_MISMATCH', reason)
        # towstaging3: rests at the dock -> passes both.
        ok, reason = resting_check(dict(z=0.1999, roll_rad=0.0, pitch_rad=0.0, lin_speed=0.0, ang_speed=0.0),
                                   {'z': rest_z}, dict(_DEFAULTS))
        self.assertTrue(ok, reason)
        ok, res, reason = heading_check(math.radians(-90.0), math.radians(-90.0),
                                        _DEFAULTS['heading_tol_rad'])
        self.assertTrue(ok, reason)


# --------------------------------------------------------------------------- #
# apply_start_state (couple) against a floor model: hold seats, un-held trips  #
# --------------------------------------------------------------------------- #
class _World:
    def __init__(self, dt=1.0 / 60.0):
        self.steps = 0; self._dt = dt
        self.scene = type('S', (), {'added': [], 'add': lambda self, p: (self.added.append(p) or p)})()

    def get_physics_dt(self):
        return self._dt

    def step(self, render=False):
        self.steps += 1


class _Mobile:
    def __init__(self):
        self.pinned = False; self.base_braked = False

    def engage_base_full_brake(self):
        self.base_braked = True

    def release_base_full_brake(self):
        self.base_braked = False

    def hold_cart_pin(self):
        self.pinned = True

    def release_cart_brake(self, *, force=False):
        self.pinned = False


class _CartBody:
    """Cart that stays put when pinned, else drifts (z rises / yaw drifts) per step -- the box-floor tip."""
    def __init__(self, world, mobile, x, y, z, yaw, *, z_drift=0.0, yaw_drift=0.0):
        self.world = world; self.mobile = mobile; self.x = x; self.y = y; self.z0 = z; self.yaw0 = yaw
        self.z_drift = z_drift; self.yaw_drift = yaw_drift

    def _eff_steps(self):
        return 0 if self.mobile.pinned else self.world.steps     # pin (at step 0) freezes at spawn

    def get_world_pose(self):
        n = self._eff_steps()
        yaw = self.yaw0 + self.yaw_drift * n
        z = self.z0 + self.z_drift * n
        return ([self.x, self.y, z], [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])

    def get_linear_velocity(self):
        return [0.0, 0.0, 0.0]

    def get_angular_velocity(self):
        return [0.0, 0.0, 0.0]


class _Carton:
    def __init__(self, prim_path, name):
        self.prim_path = prim_path; self.name = name; self._pose = ([0, 0, 0], [1, 0, 0, 0])

    def set_world_pose(self, position, orientation):
        self._pose = (list(position), list(orientation))

    def get_world_pose(self):
        return (list(self._pose[0]), list(self._pose[1]))

    def get_linear_velocity(self):
        return [0.0, 0.0, 0.0]

    def get_angular_velocity(self):
        return [0.0, 0.0, 0.0]

    def set_linear_velocity(self, v):
        pass

    def set_angular_velocity(self, w):
        pass


class _Assembly:
    def __init__(self, cart):
        self.world = _World(); self.mobile = _Mobile(); self.cart_body = cart
        self.evidence = {'composition': {'stock': '/World/Stock'}}


def _scenario(dock_yaw=0.0):
    return {
        'handling_unit': {'dimensions_m': [0.45, 0.30, 0.18]},
        'initial_staging': {
            'cart_body_pose': [-5.73, 15.25, dock_yaw],
            'cart_load_center_body_xy': [0.08, 0.0],
            'cart_support_z_world': 0.9325,
            'unit_yaws_world': {'source': math.pi, 'cart': -math.pi / 2, 'shelf': 0.0},
        },
    }


_FAST = {'min_settle_s': 0.1, 'window_s': 0.05, 'max_settle_s': 0.4}


class ApplyCoupleSettleTests(unittest.TestCase):
    def test_hold_seats_a_would_be_tipping_cart(self):
        # Same bad-floor drift that trips the un-held run below; held at the seat it cannot tip (the fix).
        cart = _CartBody(None, None, -5.73, 15.25, 0.2, 0.0, z_drift=0.002, yaw_drift=math.radians(1))
        asm = _Assembly(cart); cart.world = asm.world; cart.mobile = asm.mobile
        cfg = {'start_at': 'couple', 'start_state_settle': {**_FAST, 'hold_during_settle': True}}
        plan = apply_start_state(asm, cfg, _scenario(dock_yaw=0.0), _rigid_prim=_Carton)
        self.assertEqual(plan.start_at, 'couple')
        self.assertTrue(asm.mobile.base_braked is False)   # holds released before the mission drives
        self.assertFalse(asm.mobile.pinned)                # cart pin released (begin('couple') re-pins)

    def test_unheld_tipping_cart_is_refused_unsupported(self):
        cart = _CartBody(None, None, -5.73, 15.25, 0.2, 0.0, z_drift=0.003)
        asm = _Assembly(cart); cart.world = asm.world; cart.mobile = asm.mobile
        cfg = {'start_at': 'couple', 'start_state_settle': {**_FAST, 'hold_during_settle': False}}
        with self.assertRaises(RuntimeError) as ctx:
            apply_start_state(asm, cfg, _scenario(dock_yaw=0.0), _rigid_prim=_Carton)
        self.assertIn('START_STATE_UNSUPPORTED', str(ctx.exception))

    def test_misheaded_seat_is_refused_mismatch(self):
        # Cart static (no drift) but seated off the configured dock heading -> named heading mismatch.
        cart = _CartBody(None, None, -5.73, 15.25, 0.2, math.radians(15))
        asm = _Assembly(cart); cart.world = asm.world; cart.mobile = asm.mobile
        cfg = {'start_at': 'couple', 'start_state_settle': {**_FAST, 'hold_during_settle': False}}
        with self.assertRaises(RuntimeError) as ctx:
            apply_start_state(asm, cfg, _scenario(dock_yaw=0.0), _rigid_prim=_Carton)
        self.assertIn('START_STATE_SETTLE_MISMATCH', str(ctx.exception))

    def test_good_seat_passes(self):
        cart = _CartBody(None, None, -5.73, 15.25, 0.2, 0.0)
        asm = _Assembly(cart); cart.world = asm.world; cart.mobile = asm.mobile
        cfg = {'start_at': 'couple', 'start_state_settle': {**_FAST, 'hold_during_settle': False}}
        plan = apply_start_state(asm, cfg, _scenario(dock_yaw=0.0), _rigid_prim=_Carton)
        self.assertEqual(plan.start_at, 'couple')


# --------------------------------------------------------------------------- #
# dock-align: the assembly cart/robot spawn follows the configured couple dock  #
# (one source), for cold runs AND the start_at=couple shortcut                  #
# --------------------------------------------------------------------------- #
_SOUTH_LANE = {'name': 'outbound_dock_to_side_bay', 'start': [-3.875, 14.5, -math.pi / 2],
               'start_cart_yaw': -math.pi / 2, 'goal': [-4.3, 6.95, -math.pi / 2]}
# East lane whose derived dock equals the scenario cart spawn (-5.73,15.25,0): a legacy no-op.
_EAST_LANE = {'name': 'outbound_dock_to_bay', 'start': [-4.46, 15.25, 0.0],
              'start_cart_yaw': 0.0, 'goal': [-2.0, 15.25, 0.0]}


def _lane_cfg(lane, **extra):
    cfg = {'navigation': {'tow_lanes': [lane]}}
    cfg.update(extra)
    return cfg


class DockResolveTests(unittest.TestCase):
    def test_south_lane_derives_dock_from_start(self):
        r = resolve_couple_dock(_lane_cfg(_SOUTH_LANE))
        self.assertEqual(r['source'], 'derived_from_lane_start')
        self.assertAlmostEqual(r['cart_dock'][2], -math.pi / 2, places=5)   # SOUTH
        self.assertAlmostEqual(r['cart_dock'][0], -3.875, places=3)
        self.assertAlmostEqual(r['cart_dock'][1], 15.77, places=2)          # == towstaging3 couple_dock
        # the seat the lane start is derived from
        self.assertEqual([round(v, 4) for v in r['robot_seat']], [-3.875, 14.5, round(-math.pi / 2, 4)])

    def test_east_lane_derives_the_scenario_dock(self):
        r = resolve_couple_dock(_lane_cfg(_EAST_LANE))
        self.assertAlmostEqual(r['cart_dock'][0], -5.73, places=3)
        self.assertAlmostEqual(r['cart_dock'][2], 0.0, places=5)

    def test_explicit_couple_dock_wins(self):
        lane = {**_EAST_LANE, 'couple_dock': [-5.73, 15.25, 0.0]}
        r = resolve_couple_dock(_lane_cfg(lane))
        self.assertEqual(r['source'], 'lane.couple_dock')

    def test_no_lane_is_none(self):
        self.assertIsNone(resolve_couple_dock({}))
        self.assertIsNone(resolve_couple_dock({'navigation': {}}))


def _scenario_with_robot(dock_yaw=0.0, robot=(-5.73, 16.0, 0.0)):
    sc = _scenario(dock_yaw=dock_yaw)
    sc['initial_staging']['robot_base_pose'] = list(robot)
    return sc


class DockAlignScenarioTests(unittest.TestCase):
    def test_south_lane_reheads_the_spawn(self):
        variant, record = dock_align_scenario(_scenario(dock_yaw=0.0), _lane_cfg(_SOUTH_LANE))
        self.assertIsNotNone(record)
        self.assertAlmostEqual(variant['initial_staging']['cart_body_pose'][2], -math.pi / 2, places=5)
        self.assertEqual(record['mode'], 'couple_dock_align')
        self.assertEqual(record['dock_source'], 'derived_from_lane_start')

    def test_cold_path_moves_the_cart_only_not_the_robot(self):
        # The cold dock-align re-poses ONLY the cart to the dock heading; the robot KEEPS its collect
        # start pose (fixes PR 297, which stranded the robot at the dock seat behind the backstock Support).
        start = (-5.73, 16.0, 0.0)
        variant, record = dock_align_scenario(_scenario_with_robot(dock_yaw=0.0, robot=start),
                                              _lane_cfg(_SOUTH_LANE))
        st = variant['initial_staging']
        # robot untouched
        self.assertEqual([round(v, 4) for v in st['robot_base_pose']], [round(v, 4) for v in start])
        # cart re-headed SOUTH and moved to the dock position (NOT the robot start)
        self.assertAlmostEqual(st['cart_body_pose'][2], -math.pi / 2, places=5)
        self.assertAlmostEqual(st['cart_body_pose'][0], -3.875, places=3)
        self.assertAlmostEqual(st['cart_body_pose'][1], 15.77, places=2)
        # record names the cart-only behaviour and still carries the couple seat for provenance
        self.assertTrue(record['cart_only'])
        self.assertFalse(record['robot_repositioned'])
        self.assertEqual([round(v, 4) for v in record['robot_base_pose']], [round(v, 4) for v in start])
        self.assertEqual([round(v, 4) for v in record['couple_seat']],
                         [-3.875, 14.5, round(-math.pi / 2, 4)])

    def test_shortcut_couple_moves_both_bodies(self):
        # The start_at=couple shortcut (staged_scenario) DOES seat BOTH the robot (at the dock seat) and
        # the cart (at the dock) -- the contrast the cold cart-only path must not share.
        cfg = _lane_cfg(_SOUTH_LANE, start_at='couple')
        variant, record = staged_scenario(_scenario_with_robot(dock_yaw=0.0), cfg)
        st = variant['initial_staging']
        self.assertEqual([round(v, 4) for v in st['robot_base_pose']],
                         [-3.875, 14.5, round(-math.pi / 2, 4)])      # robot seated at the dock
        self.assertAlmostEqual(st['cart_body_pose'][2], -math.pi / 2, places=5)
        self.assertEqual(record['start_at'], 'couple')

    def test_east_derived_dock_is_no_op(self):
        # scenario cart at (-5.73,15.25,0); the east lane's derived dock equals it -> byte-identical.
        sc = _scenario(dock_yaw=0.0)
        variant, record = dock_align_scenario(sc, _lane_cfg(_EAST_LANE))
        self.assertIs(variant, sc)
        self.assertIsNone(record)

    def test_no_lane_is_no_op(self):
        sc = _scenario(dock_yaw=0.0)
        variant, record = dock_align_scenario(sc, {})
        self.assertIs(variant, sc)
        self.assertIsNone(record)


class VerifyDockSeatTests(unittest.TestCase):
    def test_good_cold_seat_passes(self):
        cart = _CartBody(None, None, -3.875, 15.77, 0.2, -math.pi / 2)
        asm = _Assembly(cart); cart.world = asm.world; cart.mobile = asm.mobile
        cfg = {'start_state_settle': {**_FAST}}
        out = verify_dock_seat(asm, cfg, _scenario(dock_yaw=-math.pi / 2))
        self.assertIsNotNone(out)
        self.assertFalse(asm.mobile.pinned)        # hold released after the verify

    def test_misheaded_cold_seat_is_refused(self):
        # cart spawned EAST while the configured dock is SOUTH -> named heading mismatch.
        cart = _CartBody(None, None, -3.875, 15.77, 0.2, 0.0)
        asm = _Assembly(cart); cart.world = asm.world; cart.mobile = asm.mobile
        cfg = {'start_state_settle': {**_FAST}}
        with self.assertRaises(RuntimeError) as ctx:
            verify_dock_seat(asm, cfg, _scenario(dock_yaw=-math.pi / 2))
        self.assertIn('START_STATE_SETTLE_MISMATCH', str(ctx.exception))

    def test_no_cart_body_is_noop(self):
        class Bare:
            cart_body = None
        self.assertIsNone(verify_dock_seat(Bare(), {}, _scenario()))


if __name__ == '__main__':
    unittest.main()
