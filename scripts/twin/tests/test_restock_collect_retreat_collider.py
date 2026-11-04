"""sim-collect-retreat-collider-v3 -- CPU proof (no Isaac/ROS/GPU) for the collect deck-place EMPTY-RETREAT
fault seen in collectlevel4-fail1 (HOST carton_level ON, tool-collider deferral ON).

Diagnosis (from the saved run host-stack.log):
  * RESTOCK_DEFER_TOOL_COLLISION was ACTIVE and WORKED -- at the weld release only the CARTON collider was
    re-enabled ("carry collision ENABLED on 1 collider(s) (carton)") and the tool-side gripper colliders were
    DEFERRED ("tool-side collision DEFERRED past the empty retreat"). So the tool-collider re-enable was NOT
    the pin (the brief's first two hypotheses are falsified by the log).
  * The retreat LIFT-OFF move (host plan @18894) was commanded to ur_shoulder_pan +0.187 rad (a +0.164 rad
    swing off the place pose's 0.0225) with a ~0.27 rad tool tilt -- a redundant DLS branch the CURRENT
    liftoff_goal rejects for the logged seed (ori_tol 0.02), reached because the full-DOF raise is ill-
    conditioned at the folded place pose. That base-yaw swing rakes the gripper suction-panel cluster
    ~+0.12-0.19 m sideways in base +Y into the rollwagen FRONT-LEFT cage post (arm/gripper-vs-cage is NOT
    collision-filtered -- only carton-vs-cage is), pinning the pan drive 0.204 rad short while the distal
    wrists velocity-saturate (the recorded TRAJ_HOST_FAULT contact-pin fingerprint).

Fix (controller._collect_retreat_liftoff_goal): lock ur_shoulder_pan at the seed so the straight-up lift-off
cannot swing the cluster into the cage; a straight +Z raise needs no base yaw. This file proves, on CPU:
  1. GEOMETRY -- the recorded pan-swinging retreat goal rakes the gripper into the front cage post, while the
     pan-locked lift-off (the fix) rises straight up and keeps the gripper clear of the post;
  2. CONTROLLER -- the retreat lift-off the controller issues holds ur_shoulder_pan at the seed, and a
     (hypothetical) pan-swinging IK branch is rejected rather than driven into the cage;
  3. DEFERRAL -- the tool-collider deferral is a config property (defer_tool_collision) with the
     RESTOCK_DEFER_TOOL_COLLISION env alias, DEFAULT ON; and on the release path the carton collider is
     re-enabled at release while the tool-side colliders are deferred and only restored on the next grasp.
"""
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2]
for _p in (str(SCRIPTS.parent), str(SCRIPTS), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from twin import urdf_probe as U                                                     # noqa: E402
from twin.restock_manipulation.cad_tool import (                                     # noqa: E402
    ARM_JOINTS, NativePanelContact, defer_tool_collision_enabled)
from twin.restock_manipulation import collect_retreat as crt                         # noqa: E402

FIX = Path(__file__).resolve().parent / 'fixtures'
URDF = FIX / 'collect_stow_arm.urdf'           # the faithful UR10 arm + panel gripper (ur_base_link frame)

# Recorded configs (ARM_JOINTS order: pan, lift, elbow, w1, w2, w3), from the saved run host-stack.log.
PLACE = (0.0225, -1.4689, 1.7915, -0.3163, 1.5998, 0.006)   # folded place pose (retreat start / lift-off seed)
BAD_GOAL = (0.1865, -1.5218, 1.5533, -0.084, 1.4974, 0.0061)  # the EXECUTED retreat goal that pinned (plan @18894)

# The rollwagen FRONT-LEFT cage post, xy in ur_base_link (reconstructed from the run: REPOSITION_REREGISTER
# deck=(1.12,0.101,.) at host-stack.log:9573 + the mission.usda /Rollwagen post offsets ~(-0.38,+0.32)).
# The post is a thin vertical column spanning the cart height, so the horizontal (xy) distance is the
# clearance. Assertions below use RELATIVE change (rake-in vs lift-away), robust to a few cm of post offset.
CAGE_POST_XY = np.array([0.74, 0.421])
GRIPPER_CLUSTER = ('big_cube', 'cube_without_surface_plates', 'suction_cups',
                   'gripper_pad_tcp', 'ur_wrist_3_link')


def _robot():
    return U.parse_urdf(str(URDF))


def _fk(robot, q):
    return U.link_world_transforms(robot, dict(zip(ARM_JOINTS, [float(v) for v in q])), root='ur_base_link')


def _cluster_min_xy_to_post(robot, q):
    """Min horizontal (xy) distance from any gripper-cluster link origin to the front-left cage post."""
    T = _fk(robot, q)
    best = math.inf
    for link in GRIPPER_CLUSTER:
        if link in T:
            p = T[link][:3, 3]
            best = min(best, float(math.hypot(p[0] - CAGE_POST_XY[0], p[1] - CAGE_POST_XY[1])))
    return best


def _pan_locked_liftoff(robot, seed=PLACE, dz=0.15):
    pan = seed[0]
    g = crt.liftoff_goal(robot, dict(zip(ARM_JOINTS, seed)), dz=dz, tcp_link='gripper_pad_tcp',
                         limits={'ur_shoulder_pan_joint': (pan, pan),
                                 'ur_wrist_1_joint': (-2 * math.pi, 0.65),
                                 'ur_wrist_2_joint': (0.0, 2 * math.pi)})
    return tuple(g[j] for j in ARM_JOINTS)


# ----------------------------------------------------------------- 1. geometry: rake-in vs lift-away
class RetreatCagePostGeometry(unittest.TestCase):
    def setUp(self):
        self.robot = _robot()

    def test_recorded_retreat_goal_rakes_the_gripper_into_the_cage_post(self):
        # FAILING-BEFORE: the executed pan-swinging goal drags the gripper cluster sideways (+Y) toward the
        # post, dropping the clearance from ~0.25 m to ~0.10 m -- within the ~0.13 m gripper panel width of
        # the post, i.e. the hard contact that pinned the pan drive.
        d_place = _cluster_min_xy_to_post(self.robot, PLACE)
        d_bad = _cluster_min_xy_to_post(self.robot, BAD_GOAL)
        self.assertGreater(d_place, 0.20)                    # clear at the place pose
        self.assertLess(d_bad, 0.11)                         # raked into the post at the bad goal
        self.assertLess(d_bad, d_place - 0.10)               # the swing DROVE it toward the post
        # the drive is a lateral +Y rake, not a vertical lift
        y_place = _fk(self.robot, PLACE)['gripper_pad_tcp'][1, 3]
        y_bad = _fk(self.robot, BAD_GOAL)['gripper_pad_tcp'][1, 3]
        self.assertGreater(y_bad - y_place, 0.15)

    def test_pan_locked_liftoff_lifts_away_and_clears_the_cage_post(self):
        # PASSING-AFTER: the pan-locked lift-off rises straight up -- no lateral +Y approach, so the post
        # clearance is held at its (clear) place value instead of collapsing.
        good = _pan_locked_liftoff(self.robot)
        d_place = _cluster_min_xy_to_post(self.robot, PLACE)
        d_good = _cluster_min_xy_to_post(self.robot, good)
        self.assertGreaterEqual(d_good, d_place - 1e-3)      # no lateral approach to the post
        self.assertGreater(d_good, 0.20)                     # stays well clear
        # and it is a near-vertical lift: y held, z risen
        T_place, T_good = _fk(self.robot, PLACE), _fk(self.robot, good)
        self.assertLess(abs(T_good['gripper_pad_tcp'][1, 3] - T_place['gripper_pad_tcp'][1, 3]), 0.01)
        self.assertGreater(T_good['gripper_pad_tcp'][2, 3] - T_place['gripper_pad_tcp'][2, 3], 0.12)

    def test_pan_locked_liftoff_still_reaches_the_full_rise(self):
        robot = self.robot
        for dz in (0.10, 0.15):
            good = _pan_locked_liftoff(robot, dz=dz)
            p0 = _fk(robot, PLACE)['gripper_pad_tcp'][:3, 3]
            p1 = _fk(robot, good)['gripper_pad_tcp'][:3, 3]
            self.assertAlmostEqual(p1[2] - p0[2], dz, delta=3e-3)         # rose by dz
            self.assertLess(math.hypot(p1[0] - p0[0], p1[1] - p0[1]), 5e-3)  # straight up
            self.assertAlmostEqual(good[0], PLACE[0], places=6)           # pan unmoved


# ----------------------------------------------------------------- 2. controller sequencing
class ControllerLiftoffLocksPan(unittest.TestCase):
    def _controller(self, **over):
        import test_restock_manipulation as T
        kw = dict(transport_joints=T._SL_STOW, staged_cart_swing=True, cart_pan_joint0=-0.8605,
                  cart_tuck_shape=T._SL_TUCK, reposition_before_place=True,
                  collect_place_preplace_shape=T._SL_PREPLACE_SHAPE, collect_place_wrist_guard=True,
                  collect_place_straight_line=True, collect_retreat_liftoff=True,
                  carry_wrist1_cap=0.65, carry_wrist2_min=0.0, fk_urdf=str(URDF))
        kw.update(over)
        c, m = T.make_controller('collect', **kw)
        return c

    def test_retreat_liftoff_goal_holds_shoulder_pan_at_the_seed(self):
        c = self._controller()
        self.assertTrue(c._collect_deck_place())
        goal = c._collect_retreat_liftoff_goal(PLACE)
        self.assertIsNotNone(goal)
        self.assertAlmostEqual(goal[0], PLACE[0], places=6)   # ur_shoulder_pan unmoved from the seed
        # and the lift-off still rises (a real escape, not a no-op)
        robot = _robot()
        dz = _fk(robot, goal)['gripper_pad_tcp'][2, 3] - _fk(robot, PLACE)['gripper_pad_tcp'][2, 3]
        self.assertGreater(dz, 0.12)

    def test_a_pan_swinging_branch_is_rejected_not_driven_into_the_cage(self):
        # Belt-and-suspenders guard: even if liftoff_goal were to return a pan-swinging branch (the run's
        # bad goal), the controller must REJECT it (return None -> fold), never issue it into the cage.
        c = self._controller()
        orig = crt.liftoff_goal
        try:
            crt.liftoff_goal = lambda *a, **k: dict(zip(ARM_JOINTS, BAD_GOAL))   # pan +0.164 off the seed
            self.assertIsNone(c._collect_retreat_liftoff_goal(PLACE))
        finally:
            crt.liftoff_goal = orig


# ----------------------------------------------------------------- 3. deferral config + release timeline
class DeferToolCollisionConfig(unittest.TestCase):
    def test_default_is_on(self):
        self.assertTrue(defer_tool_collision_enabled(None, {}))

    def test_config_property_wins_over_env_and_accepts_strings(self):
        self.assertFalse(defer_tool_collision_enabled({'defer_tool_collision': False},
                                                      {'RESTOCK_DEFER_TOOL_COLLISION': '1'}))
        self.assertTrue(defer_tool_collision_enabled({'defer_tool_collision': True}, {}))
        self.assertFalse(defer_tool_collision_enabled({'defer_tool_collision': 'off'}, {}))
        self.assertTrue(defer_tool_collision_enabled({'defer_tool_collision': 'on'}, {}))

    def test_env_alias(self):
        self.assertFalse(defer_tool_collision_enabled(None, {'RESTOCK_DEFER_TOOL_COLLISION': '0'}))
        self.assertTrue(defer_tool_collision_enabled(None, {'RESTOCK_DEFER_TOOL_COLLISION': 'true'}))

    def test_bad_config_value_raises(self):
        with self.assertRaises(ValueError):
            defer_tool_collision_enabled({'defer_tool_collision': 'maybe'}, {})


class _StubApi:
    def __init__(self):
        self.calls = []

    def close_gripper(self, path):
        self.calls.append(('close', path))

    def open_gripper(self, path):
        self.calls.append(('open', path))


class _StubStage:
    def __init__(self):
        self.removed = []

    def RemovePrim(self, path):           # noqa: N802 (USD API name)
        self.removed.append(path)


class ReleasePathColliderTimeline(unittest.TestCase):
    """The collider timeline on the weld-release path, without Isaac/USD: build a NativePanelContact via
    __new__ (its __init__ imports the Isaac surface-gripper plugin) and record every _set_carton_collision
    toggle. This is the SAME release path both the legacy fallback and the levelled collect place take --
    the gripper-open emitted from controller.release always routes through command(False) -> _release_fallback."""

    def _contact(self, defer):
        c = NativePanelContact.__new__(NativePanelContact)
        c.stage = _StubStage()
        c.api = _StubApi()
        c.path = '/RestockPanelGripper'
        c._defer_tool_collision = defer
        c._fallback_joint = '/RestockPanelGripper/sim_attach_fallback'
        c._fallback_body = '/RestockScenario/HandlingUnit'
        c._tool_collision_deferred = False
        c.enabled = False
        c._diag = False
        c.toggles = []
        # record (groups, enabled) instead of touching USD
        c._set_carton_collision = lambda body, enabled, groups='both': c.toggles.append((groups, bool(enabled)))
        return c

    def test_defer_on_release_reenables_carton_only_then_restores_tool_at_next_grasp(self):
        c = self._contact(defer=True)
        c.command(False)                                  # RELEASE (gripper open)
        self.assertEqual(c.toggles, [('carton', True)])   # carton re-enabled so it rests on the deck
        self.assertTrue(c._tool_collision_deferred)       # tool-side DEFERRED past the retreat
        self.assertEqual(c.stage.removed, ['/RestockPanelGripper/sim_attach_fallback'])
        c.toggles.clear()
        c.command(True)                                   # NEXT GRASP close (gripper clear of the cart)
        self.assertEqual(c.toggles, [('tool', True)])     # tool-side restored only now, after the retreat
        self.assertFalse(c._tool_collision_deferred)

    def test_defer_off_release_reenables_both_immediately(self):
        c = self._contact(defer=False)
        c.command(False)
        self.assertEqual(c.toggles, [('both', True)])     # legacy: carton + tool-side re-enabled at release
        self.assertFalse(c._tool_collision_deferred)


if __name__ == '__main__':
    unittest.main()
