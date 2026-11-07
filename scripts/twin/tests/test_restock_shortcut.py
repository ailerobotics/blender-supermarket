"""CPU tests for the start-phase shortcut's pure start-state planning and its apply seam.

No Isaac import: plan_start_state is pure, and apply_start_state takes a SingleRigidPrim-like
test seam so its body-placement call can be exercised without a live SimulationApp.
"""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.mission import ACTIONS, bounded_stages
from twin.restock.shortcut import (StartState, apply_start_state, isolation_matrix, plan_start_state,
                                   stage_isolation, _SUPPORTED_START_AT, _STAGE_PRECONDITIONS)
from twin.restock.native_transport import ArmCommandGate
from twin.restock_manipulation.cad_tool import ARM_JOINTS


def scenario():
    return {
        'handling_unit': {'dimensions_m': [0.6, 0.4, 0.16]},
        'initial_staging': {
            'cart_body_pose': [-5.73, 15.25, 0.0],
            'cart_load_center_body_xy': [0.08, 0.0],
            'cart_support_z_world': 0.9325,
            'unit_yaws_world': {'source': math.pi, 'cart': -math.pi / 2, 'shelf': 0.0},
        },
    }


def transfer_scenario():
    """A transfer-start variant: robot+cart spawned parked+coupled at the shelf (cart yaw pi)."""
    sc = scenario()
    sc['initial_staging']['cart_body_pose'] = [-1.78, 6.0, math.pi]
    sc['initial_staging']['unit_yaws_world']['cart'] = math.pi / 2  # carton world yaw on the pi-yaw cart
    return sc


class PlanStartStateTests(unittest.TestCase):
    def test_couple_places_carton_at_cart_load_centre(self):
        plan = plan_start_state(scenario(), 'couple')
        self.assertIsInstance(plan, StartState)
        self.assertEqual(plan.skipped, ('collect',))
        self.assertFalse(plan.couple_cart)  # the real couple stage couples the cart, not the shortcut
        # cart yaw 0 => load centre xy is the cart pose plus the body-frame offset.
        self.assertAlmostEqual(plan.carton_position[0], -5.73 + 0.08)
        self.assertAlmostEqual(plan.carton_position[1], 15.25)
        # carton centre sits half its height above the deck.
        self.assertAlmostEqual(plan.carton_position[2], 0.9325 + 0.16 / 2)
        # yaw -pi/2 -> quaternion (cos(-pi/4), 0, 0, sin(-pi/4)).
        self.assertAlmostEqual(plan.carton_orientation_wxyz[0], math.cos(-math.pi / 4))
        self.assertAlmostEqual(plan.carton_orientation_wxyz[3], math.sin(-math.pi / 4))

    def test_load_offset_rotates_into_the_cart_yaw(self):
        sc = scenario()
        sc['initial_staging']['cart_body_pose'] = [1.0, 2.0, math.pi / 2]
        sc['initial_staging']['cart_load_center_body_xy'] = [0.5, 0.0]
        plan = plan_start_state(sc, 'couple')
        # +x body offset under a +90 deg cart yaw points along world +y.
        self.assertAlmostEqual(plan.carton_position[0], 1.0, places=6)
        self.assertAlmostEqual(plan.carton_position[1], 2.5, places=6)

    def test_transfer_places_carton_on_coupled_cart_at_shelf(self):
        plan = plan_start_state(transfer_scenario(), 'transfer')
        self.assertEqual(plan.skipped, ('collect', 'couple', 'outbound', 'park'))
        self.assertTrue(plan.couple_cart)  # couple/outbound/park skipped -> shortcut authors the hitch
        self.assertEqual(plan.carton_target, 'cart')
        # cart yaw pi: +x body load offset rotates to -x world.
        self.assertAlmostEqual(plan.carton_position[0], -1.78 - 0.08, places=6)
        self.assertAlmostEqual(plan.carton_position[1], 6.0, places=6)
        self.assertAlmostEqual(plan.carton_position[2], 0.9325 + 0.16 / 2, places=6)
        self.assertAlmostEqual(plan.carton_orientation_wxyz[0], math.cos(math.pi / 4), places=6)

    def test_outbound_and_park_load_the_cart_and_couple(self):
        # outbound/park skip the collect+couple(+outbound) prefix -> the carton is loaded on the cart AND
        # the cart is coupled (authored at synthesis); the carton pose is identical to the couple case
        # (same cart load centre), only the skipped prefix and the coupled flag differ.
        for stage, skipped in (('outbound', ('collect', 'couple')),
                               ('park', ('collect', 'couple', 'outbound'))):
            plan = plan_start_state(scenario(), stage)
            self.assertEqual(plan.skipped, skipped, stage)
            self.assertTrue(plan.couple_cart, stage)
            self.assertEqual(plan.carton_target, 'cart', stage)
            self.assertAlmostEqual(plan.carton_position[0], -5.73 + 0.08, msg=stage)
            self.assertAlmostEqual(plan.carton_position[1], 15.25, msg=stage)
            self.assertAlmostEqual(plan.carton_position[2], 0.9325 + 0.16 / 2, msg=stage)

    def test_verify_places_carton_on_the_bay_top_coupled(self):
        # verify skips collect..transfer: a succeeded transfer has already PLACED the carton on the bay
        # top (centre half its height above it), level, arm empty, cart still coupled.
        sc = scenario()
        sc['transfer_bay'] = {'center_xy': [-2.0, 16.42], 'top_z': 0.73, 'dimensions_m': [0.7, 0.5, 0.73]}
        plan = plan_start_state(sc, 'verify')
        self.assertEqual(plan.skipped, ('collect', 'couple', 'outbound', 'park', 'transfer'))
        self.assertTrue(plan.couple_cart)
        self.assertEqual(plan.carton_target, 'bay')
        self.assertAlmostEqual(plan.carton_position[0], -2.0)
        self.assertAlmostEqual(plan.carton_position[1], 16.42)
        self.assertAlmostEqual(plan.carton_position[2], 0.73 + 0.16 / 2)
        # level: shelf placement yaw (0 here) -> identity-about-z quaternion.
        self.assertAlmostEqual(plan.carton_orientation_wxyz[0], 1.0)
        self.assertAlmostEqual(plan.carton_orientation_wxyz[3], 0.0)

    def test_verify_bay_target_prefers_the_config_ground_truth_target(self):
        sc = scenario()
        sc['transfer_bay'] = {'center_xy': [-2.0, 16.42], 'top_z': 0.73}
        plan = plan_start_state(sc, 'verify', config={'transfer_ground_truth': {'target': [1.0, 2.0, 0.5]}})
        self.assertAlmostEqual(plan.carton_position[0], 1.0)
        self.assertAlmostEqual(plan.carton_position[1], 2.0)
        self.assertAlmostEqual(plan.carton_position[2], 0.5 + 0.16 / 2)

    def test_verify_without_a_bay_target_refuses(self):
        with self.assertRaises(ValueError):
            plan_start_state(scenario(), 'verify')   # no transfer_bay, no config target

    def test_couple_start_is_left_uncoupled(self):
        plan = plan_start_state(scenario(), 'couple')
        self.assertFalse(plan.couple_cart)           # the real couple stage latches the hitch
        self.assertEqual(plan.carton_target, 'cart')

    def test_unsupported_start_at_refuses(self):
        # collect (stage 0, nothing to synthesize) and return (past the verify stock-commit gate) are not
        # Mission start points -- plan_start_state refuses both. return is reached via start_at=verify.
        for bad in ('collect', 'return'):
            with self.assertRaises(ValueError):
                plan_start_state(scenario(), bad)

    def test_start_at_not_a_stage_refuses(self):
        with self.assertRaises(ValueError):
            plan_start_state(scenario(), 'nonsense')

    def test_missing_staging_fields_refuse(self):
        for drop in ('cart_body_pose', 'cart_load_center_body_xy', 'cart_support_z_world'):
            sc = scenario()
            del sc['initial_staging'][drop]
            with self.assertRaises(ValueError):
                plan_start_state(sc, 'couple')
        sc = scenario()
        del sc['handling_unit']['dimensions_m']
        with self.assertRaises(ValueError):
            plan_start_state(sc, 'couple')


class FakeRigidPrim:
    """Minimal SingleRigidPrim stand-in recording the applied pose/velocity.

    Also answers the body_measurement() reads (get_world_pose / linear / angular) the sim-time settle
    samples each step, returning the last-set pose as a static, resting carton."""
    def __init__(self, prim_path, name):
        self.prim_path, self.name = prim_path, name
        self.pose = self.linear = self.angular = None

    def set_world_pose(self, position, orientation):
        self.pose = (tuple(position), tuple(orientation))

    def get_world_pose(self):
        pos, quat = self.pose
        return (list(pos), list(quat))

    def get_linear_velocity(self):
        return [0.0, 0.0, 0.0]

    def get_angular_velocity(self):
        return [0.0, 0.0, 0.0]

    def set_linear_velocity(self, v):
        self.linear = tuple(v)

    def set_angular_velocity(self, w):
        self.angular = tuple(w)


class FakeScene:
    def __init__(self):
        self.added = []

    def add(self, prim):
        self.added.append(prim)
        return prim


class FakeWorld:
    def __init__(self, dt=1.0 / 60.0):
        self.scene = FakeScene()
        self.steps = 0
        self._dt = dt

    def get_physics_dt(self):
        return self._dt

    def step(self, render=False):
        self.steps += 1


class FakeAssembly:
    def __init__(self):
        self.world = FakeWorld()
        self.evidence = {'composition': {'stock': '/World/Stock'}}


class FakeBody:
    """SingleRigidPrim-like body for body_measurement() in the shortcut's synthetic hitch."""
    def __init__(self, x=0.0, y=0.0):
        self._x, self._y = x, y

    def get_world_pose(self):
        return ([self._x, self._y, 0.2], [1.0, 0.0, 0.0, 0.0])   # Isaac wxyz, level

    def get_linear_velocity(self):
        return [0.0, 0.0, 0.0]

    def get_angular_velocity(self):
        return [0.0, 0.0, 0.0]


class FakeMobile:
    def __init__(self):
        self.spec = SimpleNamespace(manifest=lambda: {'robot_hitch_local': (0.0, 0.0, 0.3),
                                                       'cart_hitch_local': (0.0, 0.0, 0.3)})
        self.parked = 0
        self.base_brakes = 0

    def park(self):
        self.parked += 1

    def engage_base_full_brake(self):
        self.base_brakes += 1

    def coupling_state(self, dock):
        return {'joint_present': False, 'healthy': True}

    def attach(self, dock):
        return {'healthy': True, 'hitch_error_m': 0.0}


class TransferFakeAssembly:
    """A transfer-start assembly: a REAL ArmCommandGate so hold()/resume() are exercised end to end."""
    def __init__(self):
        self.world = FakeWorld()
        self.evidence = {'composition': {'stock': '/World/Stock'}}
        self.mobile = FakeMobile()
        self.robot_body = FakeBody(-1.78, 6.0)
        self.cart_body = FakeBody(-1.78 - 1.27, 6.0)
        self.applied = []
        robot = SimpleNamespace(get_dof_index=lambda name: ARM_JOINTS.index(name),
                                get_joint_positions=lambda **k: np.zeros(6))
        self.command_gate = ArmCommandGate(
            robot, [(-3, 3)] * 6,
            lambda q, v, i: self.applied.append((np.asarray(q).copy(), np.asarray(v).copy())))

    def hold(self):                     # mirrors PhysicsAssembly.hold(): park the base, LATCH the gate
        self.mobile.park()
        self.command_gate.hold()


class ApplyStartStateTests(unittest.TestCase):
    def test_apply_places_carton_zeroes_velocity_and_settles(self):
        assembly = FakeAssembly()
        # Sim-time settle (not a fixed tick count): a tiny window keeps the static-carton test fast.
        config = {'start_at': 'couple',
                  'start_state_settle': {'min_settle_s': 0.1, 'window_s': 0.05, 'max_settle_s': 0.5}}
        plan = apply_start_state(assembly, config, scenario(), _rigid_prim=FakeRigidPrim)
        carton = assembly.world.scene.added[0]
        self.assertEqual(carton.prim_path, '/World/Stock')
        self.assertAlmostEqual(carton.pose[0][0], -5.73 + 0.08)
        self.assertEqual(carton.linear, (0.0, 0.0, 0.0))
        self.assertEqual(carton.angular, (0.0, 0.0, 0.0))
        # The static carton converges in sim time (>= min_settle_s at the world's 1/60 step) rather than
        # running a hardcoded number of ticks.
        self.assertGreaterEqual(assembly.world.steps, int(0.1 / (1.0 / 60.0)))
        self.assertEqual(plan.start_at, 'couple')

    def test_transfer_shortcut_rearms_the_arm_gate_so_the_executor_can_drive(self):
        # execretest root cause: the transfer-start settle holds the arm via assembly.hold(), latching
        # command_gate.faulted; without a re-arm the mission's executor/cuMotion commands are then all
        # dropped and the arm stays frozen at the stow pose. apply_start_state must hand off a LIVE gate.
        assembly = TransferFakeAssembly()
        config = {'start_at': 'transfer',
                  'start_state_settle': {'min_settle_s': 0.1, 'window_s': 0.05, 'max_settle_s': 0.5}}
        plan = apply_start_state(assembly, config, transfer_scenario(), _rigid_prim=FakeRigidPrim)
        self.assertTrue(plan.couple_cart)
        self.assertGreater(assembly.mobile.parked, 0)           # the arm was actually held during settle
        self.assertFalse(assembly.command_gate.faulted)         # ...yet the gate is re-armed at handoff
        # The decisive proof: a streamed reference now CHANGES the applied drive target (the arm follows).
        before = len(assembly.applied)
        assembly.command_gate.command(names=list(ARM_JOINTS), positions=[0.1] * 6,
                                      velocities=[0.03] * 6, efforts=[], stamp=1.0, now=1.0)
        self.assertEqual(len(assembly.applied), before + 1)
        self.assertFalse(assembly.command_gate.faulted)
        self.assertTrue(np.allclose(assembly.applied[-1][0], 0.1))

    def test_staged_couple_engages_base_brake_so_first_stationary_stage_is_parked(self):
        # coldwin96: a staged couple_cart start begins with a STATIONARY stage (verify/transfer); the
        # attach releases the cart brake for towing, so without re-braking the still-settling train drags
        # the base past the parked window -> 'base moved during stationary operation'. apply_start_state
        # must leave the train braked; the first drive releases it.
        assembly = TransferFakeAssembly()
        config = {'start_at': 'transfer',
                  'start_state_settle': {'min_settle_s': 0.1, 'window_s': 0.05, 'max_settle_s': 0.5}}
        apply_start_state(assembly, config, transfer_scenario(), _rigid_prim=FakeRigidPrim)
        self.assertEqual(assembly.mobile.base_brakes, 1)        # braked once, after the synthetic couple

    def test_staged_couple_park_brake_opt_out(self):
        assembly = TransferFakeAssembly()
        config = {'start_at': 'transfer', 'staged_couple_park_brake': False,
                  'start_state_settle': {'min_settle_s': 0.1, 'window_s': 0.05, 'max_settle_s': 0.5}}
        apply_start_state(assembly, config, transfer_scenario(), _rigid_prim=FakeRigidPrim)
        self.assertEqual(assembly.mobile.base_brakes, 0)        # legacy: no post-couple brake


class StageIsolationTests(unittest.TestCase):
    """The pure single-stage isolation facts (docs/restock_stages.md) stay consistent with the
    mission stage machinery, so the doc/matrix cannot silently drift."""

    def test_matrix_covers_every_stage_in_order(self):
        matrix = isolation_matrix()
        self.assertEqual([r['stage'] for r in matrix], list(ACTIONS))
        self.assertEqual(set(_STAGE_PRECONDITIONS), set(ACTIONS))

    def test_stop_after_is_supported_for_every_stage(self):
        # The END-bound is universal: bounded_stages accepts every stage (no missing stop_at).
        for i, stage in enumerate(ACTIONS):
            self.assertEqual(bounded_stages(None, stage), ACTIONS[:i + 1])
            self.assertEqual(stage_isolation(stage)['stop_after'], stage)

    def test_every_stage_isolates(self):
        # The gap the lane closed: outbound/park/verify/return now each start in isolation too.
        can = {r['stage'] for r in isolation_matrix() if r['can_isolate']}
        self.assertEqual(can, set(ACTIONS))
        for r in isolation_matrix():
            self.assertTrue(r['can_isolate'], r['stage'])
            self.assertIsNone(r['blocker'], r['stage'])
            self.assertIn(f"--stop-after {r['stage']}", r['recipe'])

    def test_recipes_match_the_supported_shortcuts(self):
        self.assertEqual(stage_isolation('collect')['recipe'], '--stop-after collect')
        self.assertEqual(stage_isolation('couple')['recipe'],
                         '--start-at couple --stop-after couple')
        self.assertEqual(stage_isolation('outbound')['recipe'],
                         '--start-at outbound --start-staging dock_coupled --stop-after outbound')
        self.assertEqual(stage_isolation('park')['recipe'],
                         '--start-at park --start-staging bay_parked --stop-after park')
        self.assertEqual(stage_isolation('transfer')['recipe'],
                         '--start-at transfer --start-staging recorded_bay_park --stop-after transfer')
        self.assertEqual(stage_isolation('transfer')['start_staging'], 'recorded_bay_park')
        self.assertEqual(stage_isolation('verify')['recipe'],
                         '--start-at verify --start-staging bay_placed --stop-after verify')

    def test_return_isolates_via_verify_start_without_skipping_the_commit_gate(self):
        r = stage_isolation('return')
        self.assertTrue(r['can_isolate'])
        # start_at is verify (not return): Mission refuses a start past the stock-commit gate.
        self.assertEqual(r['start_at'], 'verify')
        self.assertEqual(r['start_staging'], 'bay_placed')
        self.assertEqual(r['stop_after'], 'return')
        self.assertEqual(r['recipe'],
                         '--start-at verify --start-staging bay_placed --stop-after return')
        self.assertIn('stock-commit gate', r['note'])

    def test_couple_needs_no_staging(self):
        self.assertIsNone(stage_isolation('couple')['start_staging'])
        self.assertIsNone(stage_isolation('collect')['start_staging'])

    def test_unknown_stage_rejected(self):
        with self.assertRaises(ValueError):
            stage_isolation('teleport')


if __name__ == '__main__':
    unittest.main()


# --------------------------------------------------------------------------- sim-verify-staging-dock-jitter-v1
from twin.restock.shortcut import _attach_with_jitter_retry, _staged_attach_retry_max


class _JitterDock:
    def __init__(self, cart_speed):
        self.cart_speed = float(cart_speed)


class _JitterCoupler:
    """Stub gate: a cart_speed above 0.02 is a pure 'moving' (jitter) reject; `extra_cats` injects a
    non-jitter reason (distant/misaligned) to prove the fast-fail path."""
    def __init__(self, extra_cats=()):
        self.extra_cats = tuple(extra_cats)

    def dock_reject_reasons(self, dock):
        out = []
        if dock.cart_speed > 0.02:
            out.append(('moving', f'cart_speed={dock.cart_speed:.4f} > 0.0200'))
        for cat in self.extra_cats:
            out.append((cat, 'injected'))
        return out


class _JitterMobile:
    def __init__(self, extra_cats=()):
        self.coupler = _JitterCoupler(extra_cats)
        self.attach_calls = 0

    def attach(self, dock):
        self.attach_calls += 1
        reasons = self.coupler.dock_reject_reasons(dock)
        if reasons:
            raise ValueError('dock rejected: stale, moving, distant, or misaligned -- '
                             + '; '.join(f'{c}: {d}' for c, d in reasons))
        return {'healthy': True, 'hitch_error_m': 0.0}


class _JitterAssembly:
    def __init__(self, extra_cats=()):
        self.mobile = _JitterMobile(extra_cats)


class StagedAttachJitterRetryTests(unittest.TestCase):
    def _run(self, speeds, *, config=None, extra_cats=()):
        asm = _JitterAssembly(extra_cats)
        seq = list(speeds)
        state = {'i': 0, 'steps': 0, 'logs': []}

        def observe():
            state['i'] += 1
            return _JitterDock(seq[min(state['i'], len(seq) - 1)])

        def held_step():
            state['steps'] += 1

        def log(*a, **k):
            state['logs'].append(a[0] if a else '')

        dock0 = _JitterDock(seq[0])
        coupling, samples = _attach_with_jitter_retry(asm, dock0, observe, held_step, config, log=log)
        return asm, coupling, samples, state

    def test_first_attempt_rest_succeeds_no_step(self):
        asm, coupling, samples, state = self._run([0.0])
        self.assertTrue(coupling['healthy'])
        self.assertEqual(samples, [0.0])
        self.assertEqual(state['steps'], 0)          # no retry needed
        self.assertEqual(asm.mobile.attach_calls, 1)

    def test_retry_catches_sub_tolerance_dip(self):
        # jitter above 0.02, dips to 0.015 on the 3rd sample -> succeed there, every sample logged
        asm, coupling, samples, state = self._run([0.0526, 0.041, 0.015, 0.0])
        self.assertTrue(coupling['healthy'])
        self.assertEqual(samples, [0.0526, 0.041, 0.015])
        self.assertEqual(state['steps'], 2)          # two sim steps before the dip
        self.assertEqual(asm.mobile.attach_calls, 3)

    def test_persistent_jitter_exhausts_and_names_samples(self):
        with self.assertRaises(ValueError) as ei:
            self._run([0.05] * 10, config={'staged_attach_retry_max': 3})
        msg = str(ei.exception)
        self.assertIn('never dipped', msg)
        self.assertIn('min cart_speed=0.0500', msg)
        self.assertIn('4 sim-step', msg)             # retry_max 3 => 4 attempts

    def test_misaligned_reject_fails_fast_without_retry(self):
        with self.assertRaises(ValueError) as ei:
            self._run([0.05], extra_cats=('distant',))
        self.assertIn('dock rejected', str(ei.exception))
        # one classify + one authentic re-raise attach, no sim steps (not a pure jitter reject)

    def test_retry_max_zero_is_single_shot(self):
        self.assertEqual(_staged_attach_retry_max({'staged_attach_retry_max': 0}), 0)
        with self.assertRaises(ValueError):
            self._run([0.05], config={'staged_attach_retry_max': 0})

    def test_default_and_bad_retry_max(self):
        self.assertEqual(_staged_attach_retry_max({}), 60)
        self.assertEqual(_staged_attach_retry_max(None), 60)
        with self.assertRaises(ValueError):
            _staged_attach_retry_max({'staged_attach_retry_max': -1})


if __name__ == '__main__':
    unittest.main()
