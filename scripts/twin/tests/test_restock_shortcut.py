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
    """Minimal SingleRigidPrim stand-in recording the applied pose/velocity."""
    def __init__(self, prim_path, name):
        self.prim_path, self.name = prim_path, name
        self.pose = self.linear = self.angular = None

    def set_world_pose(self, position, orientation):
        self.pose = (tuple(position), tuple(orientation))

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
    def __init__(self):
        self.scene = FakeScene()
        self.steps = 0

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

    def park(self):
        self.parked += 1

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
        config = {'start_at': 'couple'}
        plan = apply_start_state(assembly, config, scenario(),
                                 settle_steps=5, _rigid_prim=FakeRigidPrim)
        carton = assembly.world.scene.added[0]
        self.assertEqual(carton.prim_path, '/World/Stock')
        self.assertAlmostEqual(carton.pose[0][0], -5.73 + 0.08)
        self.assertEqual(carton.linear, (0.0, 0.0, 0.0))
        self.assertEqual(carton.angular, (0.0, 0.0, 0.0))
        self.assertEqual(assembly.world.steps, 5)
        self.assertEqual(plan.start_at, 'couple')

    def test_transfer_shortcut_rearms_the_arm_gate_so_the_executor_can_drive(self):
        # execretest root cause: the transfer-start settle holds the arm via assembly.hold(), latching
        # command_gate.faulted; without a re-arm the mission's executor/cuMotion commands are then all
        # dropped and the arm stays frozen at the stow pose. apply_start_state must hand off a LIVE gate.
        assembly = TransferFakeAssembly()
        plan = apply_start_state(assembly, {'start_at': 'transfer'}, transfer_scenario(),
                                 settle_steps=3, _rigid_prim=FakeRigidPrim)
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
