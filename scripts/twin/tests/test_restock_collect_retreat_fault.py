"""sim-collect-retreat-fault-v1 -- CPU proof (no Isaac/ROS/GPU) for the collect deck-place EMPTY-RETREAT jam.

Root cause (rtfcollect2-fail1, ~/datasets/twin-robot-in-store-boot-v1/mission/rtfcollect2-fail1-saved):
the collect deck place sets the carton down and the weld RELEASE re-enables the carton + the TOOL-SIDE
gripper colliders (cube/surface_plates/suction_cups -- all on the cuMotion self-collision IGNORE list). The
empty retreat right after is a collision-OFF straight-line joint smoothstep from the folded place pose to the
clear ``collect_place_preplace_shape`` standoff, so neither cuMotion nor the smoothstep model the re-enabled
gripper collider now grazing the arm/cart. The arm PINS: shoulder_lift stalls 0.438 rad short (goal -1.91,
meas -1.472), elbow 0.264 short, while wrist_1/2/3 run at ~3.13 rad/s for the whole 30 s settle window. The
collect5 wrist_guard + straight_line keys were already ON (HOST_EFFECTIVE_CONFIG) and did not prevent it.

Two parts proved here:
  1. classify_contact_pin NAMES that signature (vs the generic "pose not bounded" that masked it), and is
     distinct from classify_goal_miss's single-frozen-joint pick block and from a clean decaying settle.
  2. the cad_tool tool-side collision DEFERRAL (RESTOCK_DEFER_TOOL_COLLISION) re-enables the carton colliders
     at release but keeps the tool-side gripper colliders OFF through the retreat, restoring them at the next
     grasp close -- removing the re-enable-at-the-folded-pose trigger. Pure helpers + the deferral sequence.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock import trajectory_host as T
from twin.restock_manipulation import cad_tool as C

# rtfcollect2-fail1 TRAJ_HOST_GOAL_MISS (ARM_JOINTS order: pan, lift, elbow, wrist_1/2/3)
REC_ERR = [-0.1351, 0.4381, -0.2637, -0.1494, 0.0013, 0.0017]
REC_VEL = [-1.2047, 0.7594, -0.0986, -3.1247, -3.1367, -3.1309]
POS_TOL = 0.010


class ContactPinClassifier(unittest.TestCase):
    def test_names_the_recorded_retreat_jam(self):
        reason = T.classify_contact_pin(REC_ERR, REC_VEL, POS_TOL)
        self.assertIsNotNone(reason)
        # names the stalled proximal joint (worst) and the saturated wrists, and the real mechanism
        self.assertIn('ur_shoulder_lift_joint', reason)
        self.assertIn('ur_elbow_joint', reason)
        self.assertIn('velocity saturation', reason)
        self.assertIn('collision-OFF straight-line retreat', reason)
        self.assertIn('not a tracking-bound failure', reason)
        self.assertNotEqual(reason, T._GOAL_MISS_GENERIC)

    def test_clean_decaying_settle_is_not_a_pin(self):
        # all joints at goal, low velocity -> None (left to the settle envelope / success)
        self.assertIsNone(T.classify_contact_pin([0.0] * 6, [0.1] * 6, POS_TOL))

    def test_single_frozen_pick_block_is_not_a_pin(self):
        # one far FROZEN joint, every wrist quiet -> None, so classify_goal_miss keeps naming the pick block
        err = [0.0, 0.40, 0.0, 0.0, 0.0, 0.0]
        vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(T.classify_contact_pin(err, vel, POS_TOL))
        # and classify_goal_miss DOES name it (unchanged behaviour)
        self.assertIn('pick physically blocked', T.classify_goal_miss(err, vel, POS_TOL))

    def test_needs_two_saturated_wrists(self):
        # a proximal stall with only ONE saturated wrist is not the retreat pin signature
        err = [0.0, 0.40, 0.0, 0.0, 0.0, 0.0]
        vel = [0.0, 0.5, 0.0, 3.1, 0.1, 0.1]
        self.assertIsNone(T.classify_contact_pin(err, vel, POS_TOL))

    def test_needs_a_proximal_stall(self):
        # wrists saturating but every proximal joint at goal -> not this fault
        err = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vel = [0.0, 0.0, 0.0, 3.1, 3.1, 3.1]
        self.assertIsNone(T.classify_contact_pin(err, vel, POS_TOL))

    def test_marginal_proximal_miss_is_not_a_pin(self):
        # a proximal joint only a little past tol (< CONTACT_PIN_STALL_RAD) is settle/track noise, not a pin
        err = [0.0, 0.05, 0.0, 0.0, 0.0, 0.0]
        vel = [0.0, 0.2, 0.0, 3.1, 3.1, 3.1]
        self.assertIsNone(T.classify_contact_pin(err, vel, POS_TOL))

    def test_bad_input_returns_none(self):
        self.assertIsNone(T.classify_contact_pin([0, 1], [0, 1, 2], POS_TOL))
        self.assertIsNone(T.classify_contact_pin(None, None, POS_TOL))


class ToolSideColliderHelpers(unittest.TestCase):
    def test_collision_groups_to_toggle(self):
        self.assertEqual(C.collision_groups_to_toggle('both'), (True, True))
        self.assertEqual(C.collision_groups_to_toggle('carton'), (True, False))
        self.assertEqual(C.collision_groups_to_toggle('tool'), (False, True))
        with self.assertRaises(ValueError):
            C.collision_groups_to_toggle('nope')

    def test_is_tool_side_collider(self):
        self.assertTrue(C.is_tool_side_collider('/World/robot/cube_without_surface_plates/collisions'))
        self.assertTrue(C.is_tool_side_collider('/World/robot/surface_plate_3/coll'))
        self.assertTrue(C.is_tool_side_collider('/World/robot/suction_cups'))
        self.assertFalse(C.is_tool_side_collider('/RestockScenario/HandlingUnit/geom'))
        self.assertFalse(C.is_tool_side_collider('/World/robot/ur_forearm_link/coll'))


class _FakeStage:
    def __init__(self):
        self.removed = []

    def RemovePrim(self, p):
        self.removed.append(p)


class _FakeApi:
    def __init__(self):
        self.calls = []

    def open_gripper(self, path):
        self.calls.append(('open', path))

    def close_gripper(self, path):
        self.calls.append(('close', path))


def _gripper(defer):
    """A NativePanelContact with only the attrs the release/command collision paths touch, and
    _set_carton_collision stubbed to RECORD (body, enabled, groups) instead of touching pxr."""
    g = C.NativePanelContact.__new__(C.NativePanelContact)
    g.path = '/World/robot/gripper'
    g.enabled = False
    g.stage = _FakeStage()
    g.api = _FakeApi()
    g._diag = False
    g._sim_attach_fallback = True
    g._fallback_joint = g.path + '/sim_attach_fallback'
    g._fallback_body = '/RestockScenario/HandlingUnit'
    g._defer_tool_collision = defer
    g._tool_collision_deferred = False
    g.toggles = []
    g._set_carton_collision = lambda body, enabled, groups='both': g.toggles.append((enabled, groups))
    return g


class ToolSideDeferralSequence(unittest.TestCase):
    def test_legacy_release_re_enables_both(self):
        g = _gripper(defer=False)
        g._release_fallback()
        self.assertEqual(g.toggles, [(True, 'both')])
        self.assertFalse(g._tool_collision_deferred)
        self.assertEqual(g.stage.removed, [g.path + '/sim_attach_fallback'])

    def test_deferred_release_re_enables_carton_only_and_marks_deferred(self):
        g = _gripper(defer=True)
        g._release_fallback()
        self.assertEqual(g.toggles, [(True, 'carton')])   # carton re-enabled for the deck drop
        self.assertTrue(g._tool_collision_deferred)         # tool-side held OFF through the retreat

    def test_next_grasp_close_restores_tool_side(self):
        g = _gripper(defer=True)
        g._release_fallback()
        g.toggles.clear()
        g.command(True)                                     # next grasp close
        self.assertEqual(g.toggles, [(True, 'tool')])       # tool-side restored, gripper now clear of the cart
        self.assertFalse(g._tool_collision_deferred)
        self.assertIn(('close', g.path), g.api.calls)

    def test_close_without_deferral_does_not_touch_tool_side(self):
        g = _gripper(defer=False)
        g.command(True)
        self.assertEqual(g.toggles, [])                     # byte-identical close when nothing was deferred
        self.assertIn(('close', g.path), g.api.calls)


if __name__ == '__main__':
    unittest.main()
