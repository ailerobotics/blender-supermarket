"""sim-collect-retreat-abort-v2 -- CPU proof (no Isaac/ROS/GPU) that the collect deck-place EMPTY retreat
lifts the gripper straight UP (separating the arm from the folded-place contact zone) before folding to
the clear preplace shape, so the retreat no longer pins against the re-enabled carton / shelf / cage.

Recorded abort (physcon1-fail1 / rtfcollect2-fail1, data/sim-collect-retreat-abort-v2/report.md): after
release the single collision-OFF straight-line retreat (place config -> collect_place_preplace_shape) pins
-- shoulder_lift ~0.6 rad / elbow ~0.5 rad short for the whole settle window while the distal wrists run at
velocity saturation -- and the settle gate aborts the trajectory (moveit_err=-4). The first retreat
increment is almost pure HORIZONTAL retract, dragging the still-ENABLED wrist/forearm links along the
re-enabled carton (SIM_ATTACH re-enables the carton collider at release) at the folded place height.

This file proves, on CPU:
  1. the abort is reproduced in DETECTION -- the recorded terminal (per_joint_err, per_joint_vel) trips the
     settle gate and classify_contact_pin NAMES the proximal-pin + wrist-saturation contact (not generic);
  2. it is NOT a drive-authority failure -- the stall-pose gravity holding torques are far under the
     release-phase effort cap and the retreat is already slow, so a saturated drive implies a contact;
  3. the lift-off raises the TCP straight up (+dz, orientation preserved), stays in the clear carry wrist
     region, and raises the whole wrist/forearm cluster VERTICALLY by ~dz BEFORE any horizontal retract --
     the separation the horizontal-first direct retreat never provides;
  4. the controller issues the retreat as lift-off THEN fold, and is byte-identical when disabled / no FK.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(REPO_ROOT), str(SCRIPTS), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from twin import urdf_probe as U                                                      # noqa: E402
from twin.restock_manipulation.cad_tool import ARM_JOINTS                             # noqa: E402
from twin.restock_manipulation import collect_retreat as crt                          # noqa: E402
from twin.restock.trajectory_host import classify_contact_pin, settle_decision        # noqa: E402
from twin.restock_manipulation import (                                               # noqa: E402
    Observation, ProductProfile, Sensors, Target, TransferController)

FIX = Path(__file__).resolve().parent / 'fixtures'
URDF = FIX / 'collect_stow_arm.urdf'

# Recorded configs (ARM_JOINTS order).
PREPLACE = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)       # collect_place_preplace_shape (retreat GOAL)
STALL = (-0.0127, -1.3112, 1.5111, -0.1039, 1.6648, -0.0001)  # physcon1 GOAL_MISS meas_q (~ the release fold)
WRIST_FOREARM = ('ur_forearm_link', 'ur_wrist_1_link', 'ur_wrist_2_link', 'ur_wrist_3_link')

# physcon1-fail1 recorded terminal settle sample (host-stack.log TRAJ_HOST_SETTLE settle=29.11):
# shoulder_lift + elbow pinned ~0.5-0.6 rad short; wrist_1 and wrist_3 at velocity saturation.
REC_ERR = np.array([-0.1058, 0.6082, -0.5171, -0.0028, 0.0006, 0.0014])
REC_VEL = np.array([-0.3736, 0.5015, -1.2406, 3.15, -1.4624, -3.1408])


@pytest.fixture(scope='module')
def robot():
    return U.parse_urdf(str(URDF))


def _cfg(q):
    return dict(zip(ARM_JOINTS, [float(v) for v in q]))


def _tcp(robot, q, link='gripper_pad_tcp'):
    return U.link_world_transforms(robot, _cfg(q), root='ur_base_link')[link]


def _cluster_z(robot, q):
    """Mean z of the enabled wrist/forearm link origins -- the height of the links that drag along the
    folded-place contact zone during a horizontal retreat."""
    T = U.link_world_transforms(robot, _cfg(q), root='ur_base_link')
    return float(np.mean([T[l][2, 3] for l in WRIST_FOREARM]))


# ----------------------------------------------------------------- 1. detection reproduction
def test_recorded_terminal_trips_settle_and_names_the_contact_pin():
    # The recorded terminal is a sustained non-decaying oscillation at ~0.6 rad error, so the settle
    # envelope does NOT accept it (never 'settled').
    samples = [(float(i), float(np.linalg.norm(REC_ERR)), float(np.linalg.norm(REC_VEL)))
               for i in range(8)]
    verdict, _ = settle_decision(samples, pos_tol=0.01, vel_tol=0.05, final=True)
    assert verdict != 'settled', verdict
    # classify_contact_pin NAMES the real cause: >=2 wrists at velocity saturation AND a proximal joint
    # pinned far short -- the collect retreat contact pin (not the generic oscillation reason).
    reason = classify_contact_pin(REC_ERR, REC_VEL, 0.01)
    assert reason is not None
    assert 'collect retreat pinned by an unmodelled contact' in reason
    assert 'velocity saturation' in reason and 'ur_shoulder_lift_joint' in reason


def test_contact_pin_is_distinct_from_a_clean_settle():
    # A crisply settling move (all velocities low, errors tiny) is NOT a contact pin.
    assert classify_contact_pin(np.zeros(6), np.full(6, 0.01), 0.01) is None
    # A single far joint with the wrists quiet is a pick block, not the retreat contact pin.
    err = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5]); vel = np.zeros(6)
    assert classify_contact_pin(err, vel, 0.01) is None


# ----------------------------------------------------------------- 2. not drive authority (torque accounting)
def test_stall_gravity_torque_is_far_under_the_release_effort_cap(robot):
    # Jacobian-transpose static gravity torque at the stall config: shoulder_lift ~100 N*m, elbow ~75 N*m,
    # both far under the release-phase effort cap (400 N*m). With the retreat already slow (inertial demand
    # ~1 N*m), a drive SATURATED at 400 N*m while pinned can only be opposed by a large EXTERNAL contact --
    # not a lack of drive authority. (The heavy-inertia URDF makes the absolute masses large; the point is
    # the RATIO: gravity << cap.)
    tau = _gravity_torques(robot, STALL)
    assert abs(tau['ur_shoulder_lift_joint']) < 400.0
    assert abs(tau['ur_elbow_joint']) < 400.0
    # and well under even half the cap -- so slowing/strengthening the move cannot be the fix; a contact is.
    assert abs(tau['ur_shoulder_lift_joint']) < 200.0


# ----------------------------------------------------------------- 3. lift-off kinematics
def test_liftoff_raises_tcp_straight_up_preserving_orientation(robot):
    for dz in (0.10, 0.15, 0.20):
        goal = crt.liftoff_goal(robot, _cfg(STALL), dz=dz)
        p0, R0 = _tcp(robot, STALL)[:3, 3], _tcp(robot, STALL)[:3, :3]
        p1, R1 = _tcp(robot, [goal[j] for j in ARM_JOINTS])[:3, 3], \
            _tcp(robot, [goal[j] for j in ARM_JOINTS])[:3, :3]
        assert abs((p1[2] - p0[2]) - dz) < 3e-3, (dz, p1[2] - p0[2])   # rose by dz in Z
        assert math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 5e-3          # ~no horizontal drift
        assert np.linalg.norm(crt._rotvec(R1 @ R0.T)) < 0.03            # orientation preserved


def test_liftoff_stays_in_the_clear_carry_wrist_region(robot):
    # Bounded to wrist_1 <= 0.65, wrist_2 >= 0 so the lift-off cannot drive a wrist into the
    # ur_wrist_2-vs-forearm stop.
    goal = crt.liftoff_goal(robot, _cfg(STALL), dz=0.15,
                            limits={'ur_wrist_1_joint': (-2 * math.pi, 0.65),
                                    'ur_wrist_2_joint': (0.0, 2 * math.pi)})
    assert goal['ur_wrist_1_joint'] <= 0.65 + 1e-6
    assert goal['ur_wrist_2_joint'] >= 0.0 - 1e-6


def test_liftoff_refuses_an_unreachable_rise(robot):
    with pytest.raises(ValueError):
        crt.liftoff_goal(robot, _cfg(STALL), dz=5.0)   # way past the reach envelope


def test_liftoff_separates_the_cluster_vertically_before_the_fold(robot):
    # THE MECHANISM. The lift-off raises the wrist/forearm cluster by ~dz straight away; the direct retreat
    # (seed -> preplace in one fold) barely raises it at all (it is a near-horizontal retract), so the
    # enabled links stay at the folded-place contact height while retracting -- the drag that pins. The
    # lift-off's vertical separation up front is many times the direct retreat's TOTAL rise.
    dz = 0.15
    goal = crt.liftoff_goal(robot, _cfg(STALL), dz=dz)
    z_seed = _cluster_z(robot, STALL)
    z_liftoff = _cluster_z(robot, [goal[j] for j in ARM_JOINTS])
    z_preplace = _cluster_z(robot, PREPLACE)
    liftoff_rise = z_liftoff - z_seed
    direct_total_rise = z_preplace - z_seed                 # the direct fold's ENTIRE vertical gain
    assert liftoff_rise > 0.09, liftoff_rise                # the cluster rose ~dz
    assert liftoff_rise > 3.0 * abs(direct_total_rise), (liftoff_rise, direct_total_rise)


# ----------------------------------------------------------------- 4. controller sequencing
def _drive_capture_retreat(fk_urdf, liftoff=True):
    """Drive the full collect reposition place to verify through the REAL controller + FixtureMotion, and
    return the ordered list of begin_joint_goal goals issued DURING the retreat phase (ARM-order tuples)."""
    import test_restock_manipulation as T
    kwargs = dict(transport_joints=T._SL_STOW, staged_cart_swing=True, cart_pan_joint0=-0.8605,
                  cart_tuck_shape=T._SL_TUCK, reposition_before_place=True,
                  collect_place_preplace_shape=T._SL_PREPLACE_SHAPE, collect_place_wrist_guard=True,
                  collect_place_straight_line=True, collect_retreat_liftoff=liftoff)
    if fk_urdf is not None:
        kwargs['fk_urdf'] = str(fk_urdf)
    c, m = T.make_controller('collect', **kwargs)
    retreat_goals, k, t0 = [], 0, 1.0
    for i in range(160):
        if c.phase == 'verify':
            break
        if c.phase == 'transit':
            c.base_repositioned = True
        contact = c.held or (c.phase == 'grasp' and c.issued)
        if c.phase == 'release' and c.issued:
            contact = False
        extra = {'joint_positions': T._SL_STOW} if c.phase == 'stow' else {}
        phase_before = c.phase
        r = c.step(.1, T.packet(c, t0 + i * .1, contact=contact, **extra))
        assert r.status != 'failed', (phase_before, r.reason)
        jg = getattr(m, 'joint_goals', [])
        while k < len(jg):
            if phase_before == 'retreat':
                retreat_goals.append(tuple(round(v, 4) for v in jg[k]['goal']))
            k += 1
    assert c.phase == 'verify', c.phase
    return c, retreat_goals


def test_retreat_issues_liftoff_then_fold_with_fk():
    # With an FK URDF the retreat issues TWO collision-off straight-line joint goals: FIRST a lift-off that
    # raises the TCP (a config distinct from the preplace shape), THEN the fold to the clear preplace shape.
    import test_restock_manipulation as T
    robot = U.parse_urdf(str(URDF))
    c, goals = _drive_capture_retreat(URDF, liftoff=True)
    assert c._retreat_lifted is True
    assert len(goals) == 2, goals
    liftoff_goal_q, fold_goal = goals
    assert fold_goal == tuple(round(v, 4) for v in T._SL_PREPLACE_SHAPE)   # 2nd = the fold
    assert liftoff_goal_q != fold_goal                                    # 1st = a distinct lift-off
    # the lift-off raises the TCP above the preplace shape's own config height is not required; what matters
    # is it rises vs the controller's measured release pose (packet joints = 0): assert a net +Z rise.
    z0 = _tcp(robot, (0.,) * 6)[2, 3]
    z_liftoff = _tcp(robot, liftoff_goal_q)[2, 3]
    assert z_liftoff - z0 > 0.09, (z_liftoff - z0)


def test_retreat_without_fk_is_single_fold_byte_identical():
    # No FK model => the lift-off IK cannot run => the retreat falls through to the single direct fold to
    # the preplace shape (the prior behaviour), so a fixture controller without fk_urdf is byte-identical.
    import test_restock_manipulation as T
    c, goals = _drive_capture_retreat(None, liftoff=True)
    assert goals == [tuple(round(v, 4) for v in T._SL_PREPLACE_SHAPE)], goals


def test_retreat_liftoff_off_is_single_fold():
    # Explicitly disabling the lift-off (even with an FK URDF) restores the single direct fold.
    import test_restock_manipulation as T
    c, goals = _drive_capture_retreat(URDF, liftoff=False)
    assert goals == [tuple(round(v, 4) for v in T._SL_PREPLACE_SHAPE)], goals


# ----------------------------------------------------------------- helpers
def _gravity_torques(robot, q):
    """Static gravity holding torque (N*m) per arm joint via Jacobian transpose over the distal subtree."""
    g = np.array([0.0, 0.0, -9.81])
    cfg = _cfg(q)
    T = U.link_world_transforms(robot, cfg, root='ur_base_link')
    com, mass = {}, {}
    for name, L in robot.links.items():
        if L.inertial and L.inertial.mass and name in T:
            com[name] = T[name][:3, :3] @ np.array(L.inertial.origin_xyz, float) + T[name][:3, 3]
            mass[name] = L.inertial.mass
    children = {}
    for j in robot.joints:
        children.setdefault(j.parent, []).append(j)

    def subtree(link):
        out, stack = set(), [link]
        while stack:
            x = stack.pop(); out.add(x)
            for j in children.get(x, []):
                stack.append(j.child)
        return out
    jb = {j.name: j for j in robot.joints}
    tau = {}
    for jn in ARM_JOINTS:
        j = jb[jn]
        Tc = T.get(j.child)
        if Tc is None:
            continue
        pj = Tc[:3, 3]
        axis = Tc[:3, :3] @ np.array(j.axis, float)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        t = 0.0
        for name in subtree(j.child):
            if name in com:
                t += float(np.dot(axis, np.cross(com[name] - pj, mass[name] * g)))
        tau[jn] = t
    return tau
