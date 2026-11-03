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
from twin.restock.shortcut import StartState, apply_start_state, plan_start_state
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
        # cart yaw pi: +x body load offset rotates to -x world.
        self.assertAlmostEqual(plan.carton_position[0], -1.78 - 0.08, places=6)
        self.assertAlmostEqual(plan.carton_position[1], 6.0, places=6)
        self.assertAlmostEqual(plan.carton_position[2], 0.9325 + 0.16 / 2, places=6)
        self.assertAlmostEqual(plan.carton_orientation_wxyz[0], math.cos(math.pi / 4), places=6)

    def test_unsupported_start_at_refuses(self):
        for bad in ('outbound', 'park', 'verify'):
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


if __name__ == '__main__':
    unittest.main()
