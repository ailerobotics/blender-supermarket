"""sim-robust-cart-place-goal-v1 -- CPU proof that the COLLECT place-on-cart step has a collision-free,
trackable IK goal, and that the controller selects it (no Isaac/ROS/GPU).

Root cause (from the collect1 / collectsrc runs under
~/datasets/twin-robot-in-store-boot-v1/mission/, see data/sim-robust-cart-place-goal-v1/report.md):
the base drives in front of the rollwagen mouth (reposition_before_place, cart_cage_place False) and the
deck place/preplace poses are reached by the host's seeded-KDL -> cuMotion nearest-branch IK. The deck
poses have TWO collision-free IK families:

  * an UPRIGHT one (shoulder_lift ~ -1.9 rad), low gravitational droop -- trackable; and
  * a DEEP fold (shoulder_lift ~ -3.57 rad) the soft arm cannot HOLD under the 0.45 kg carton load.

Seeded from the folded carry config the host picks EITHER the deep fold -- collect1 sagged 0.39 rad short
of goal -3.596 at steady state -> PATH_TOLERANCE_VIOLATED -> 'arm command heartbeat lost' -- OR a wrist
self-colliding branch (wrist_1 > 0.65, wrist_2 < 0) -- collectsrc's repeated 'Invalid c-space position:
self collision detected', the 12-replan storm.

The fix is a GOAL fix, not a planner fix: drive the preplace to a fixed UPRIGHT clear-region joint shape
(collect_place_preplace_shape, like cage_preplace_shape) and keep the place/retreat pose-IK in the clear
wrist branch (collect_place_wrist_guard, reusing transfer_carry.in_clear_region / wrist_flip_seed). These
tests assert such a branch EXISTS for the recorded deck targets and is the one the config selects.

The arm chain below is the mission mobile-import.urdf ur_base_link -> gripper_pad_tcp chain verbatim (the
same constant as test_restock_scripted_transfer; checked against the dataset import when present).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from twin.restock import scripted_transfer as st  # noqa: E402
from twin.restock_manipulation.transfer_carry import (  # noqa: E402
    DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN, in_clear_region, wrap_angle, wrist_flip_seed)

# --- recorded place-on-cart targets in ur_base_link (collect1 REPOSITION_REREGISTER, PR 200 run) ---
# deck=(1.088, 0.122, 0.419) place=(0.889, 0.161, 0.499) preplace=(0.713, 0.196, 0.529)
# collectsrc logged (1.085, 0.115, 0.419)/(0.886, 0.154, 0.499)/(0.709, 0.189, 0.529) -- within ~8 mm, so
# a FIXED upright preplace shape (a standoff, not the sensed insert) is robust across runs.
DECK = (1.088, 0.122, 0.419)
PLACE = (0.889, 0.161, 0.499)
PREPLACE = (0.713, 0.196, 0.529)
# The carton SKU's fixed TCP orientation (scripts/restock_scenario.json handling_unit.tcp_quaternion_xyzw).
TCP_Q = (0.5, 0.5, 0.5, 0.5)                 # x, y, z, w (ros_adapter sets p.orientation.{x,y,z,w})
MAX_REACH = 1.3                              # profile max_reach gate (scenario max_reach_m)
# The folded carry configuration the place moves are seeded from (pan, lift, elbow, w1, w2, w3), from the
# mission cart_pan_joint0 / cart_tuck_shape / carry_lift_shape; the trajectory point-0 in collect1 was
# [elbow, lift, pan, ...] = [1.0, -2.27, 1.268, 0.65, 0.80, -1.2544] (host order), i.e. ARM order below.
FOLDED_CARRY = (1.268, -2.27, 1.0, 0.65, 0.80, -1.2544)

# The CPU-validated UPRIGHT clear-region preplace shape shipped in the example config fragment
# (docs/examples/cart-place-goal-mission-fragment.json) and recorded in the station contract. ARM order.
UPRIGHT_PREPLACE_SHAPE = (0.093, -1.910, 2.014, -0.103, 1.664, 0.0)

# run-356 / collect1 mobile-import.urdf, ur_base_link -> gripper_pad_tcp (name, type, parent, child, xyz, rpy)
_CHAIN = [
    ('ur_base_link-base_link_inertia', 'fixed', 'ur_base_link', 'ur_base_link_inertia', (0, 0, 0), (0, 0, math.pi), None, None),
    ('ur_shoulder_pan_joint', 'revolute', 'ur_base_link_inertia', 'ur_shoulder_link', (0, 0, 0.1273), (0, 0, 0), -2 * math.pi, 2 * math.pi),
    ('ur_shoulder_lift_joint', 'revolute', 'ur_shoulder_link', 'ur_upper_arm_link', (0, 0, 0), (1.570796327, 0, 0), -2 * math.pi, 2 * math.pi),
    ('ur_elbow_joint', 'revolute', 'ur_upper_arm_link', 'ur_forearm_link', (-0.612, 0, 0), (0, 0, 0), 0.1, math.pi),
    ('ur_wrist_1_joint', 'revolute', 'ur_forearm_link', 'ur_wrist_1_link', (-0.5723, 0, 0.163941), (0, 0, 0), -2 * math.pi, 2 * math.pi),
    ('ur_wrist_2_joint', 'revolute', 'ur_wrist_1_link', 'ur_wrist_2_link', (0, -0.1157, 0), (1.570796327, 0, 0), -2 * math.pi, 2 * math.pi),
    ('ur_wrist_3_joint', 'revolute', 'ur_wrist_2_link', 'ur_wrist_3_link', (0, 0.0922, 0), (1.570796326589793, math.pi, math.pi), -2 * math.pi, 2 * math.pi),
    ('ur_wrist_3-flange', 'fixed', 'ur_wrist_3_link', 'ur_flange', (0, 0, 0), (0, -math.pi / 2, -math.pi / 2), None, None),
    ('ur_flange-tool0', 'fixed', 'ur_flange', 'ur_tool0', (0, 0, 0), (math.pi / 2, 0, math.pi / 2), None, None),
    ('gripper_mount_joint', 'fixed', 'ur_tool0', 'big_cube', (-0.000368, -0.000549, 0.130241), (1.5707963, 0, 3.1415927), None, None),
    ('suction_cups_joint', 'fixed', 'big_cube', 'suction_cups', (0.120804, 0.117792, 0.04302), (-1.535325, 0, 0), None, None),
    ('gripper_pad_tcp_joint', 'fixed', 'suction_cups', 'gripper_pad_tcp', (-0.120313, 0.044122, 0.02245), (0.035471327, 0, 3.141592654), None, None),
]
RUN_URDF = Path('/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/collect1/mobile-import.urdf')


def _urdf_text():
    links = sorted({c[2] for c in _CHAIN} | {c[3] for c in _CHAIN})
    out = ['<robot name="cart_place_arm_chain">'] + [f'  <link name="{n}"/>' for n in links]
    for name, kind, parent, child, xyz, rpy, lo, hi in _CHAIN:
        out.append(f'  <joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
                   f'<origin xyz="{" ".join(map(repr, map(float, xyz)))}" rpy="{" ".join(map(repr, map(float, rpy)))}"/>'
                   + ('<axis xyz="0 0 1"/>' if kind == 'revolute' else '')
                   + (f'<limit lower="{lo!r}" upper="{hi!r}" effort="330" velocity="2.09"/>' if kind == 'revolute' else '')
                   + '</joint>')
    return '\n'.join(out + ['</robot>'])


@pytest.fixture(scope='module')
def ik(tmp_path_factory):
    path = tmp_path_factory.mktemp('urdf') / 'arm.urdf'
    path.write_text(_urdf_text())
    return st.ArmIK(path)


def _reach(xyz):
    return float(np.linalg.norm(xyz))


def _droop(q):
    """Gravitational-droop proxy: |shoulder_lift - (-pi/2)|. -pi/2 == upper arm horizontal/upright; the
    deep fold at -3.57 is ~2.0 rad past it (near the fully-folded gravity-max), the upright at -1.9 is ~0.34."""
    return abs(wrap_angle(q[1]) - (-math.pi / 2.0))


def _clear(q):
    return in_clear_region(q, DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN)


def _best_upright_clear(ik, xyz):
    """The lowest-droop collision-free (clear-wrist) IK solution for the TCP pose at xyz (fixed TCP_Q),
    found by seeding the upright branch and its wrist-flipped twin across shoulder_pan. Returns (q, perr)."""
    T = st.pose_T(xyz, TCP_Q)
    best = None
    seeds = [np.array([pan, -1.9, 2.0, -0.1, 1.6, 0.0]) for pan in np.linspace(-math.pi, math.pi, 9)]
    for s in seeds:
        for seed in (s, np.array(wrist_flip_seed(s))):
            q, perr, rerr = ik.solve(T, seed)
            if perr < 8e-4 and rerr < 8e-3 and _clear(q):
                key = _droop(q)
                if best is None or key < best[0]:
                    best = (key, q, perr)
    return (None, None) if best is None else (best[1], best[2])


# --------------------------------------------------------------------- the core assertions

def test_fixture_chain_matches_the_dataset_import_when_present(ik):
    if not RUN_URDF.is_file():
        pytest.skip('dataset mobile-import.urdf not present')
    real = st.ArmIK(RUN_URDF)
    q = np.array([0.1, -1.9, 2.0, -0.1, 1.6, 0.0])
    assert np.allclose(real.fk(q), ik.fk(q), atol=1e-9)


def test_deck_targets_have_a_collision_free_in_reach_ik_solution(ik):
    """The spec's core assertion: a collision-free IK solution EXISTS for the cart deck targets from the
    recorded base pose (poses in ur_base_link)."""
    for name, xyz in (('preplace', PREPLACE), ('place', PLACE)):
        q, perr = _best_upright_clear(ik, xyz)
        assert q is not None, f'{name}: no collision-free (clear-wrist) IK solution found'
        assert perr < 1e-3, f'{name}: IK position error {perr*1e3:.2f} mm'
        assert _clear(q), f'{name}: solution not in the clear wrist region'
        assert _reach(xyz) <= MAX_REACH, f'{name}: target reach {_reach(xyz):.3f} exceeds max_reach'


def test_a_self_colliding_branch_exists_so_the_guard_does_real_work(ik):
    """Seeding the transport-stow config (the host's first seed) lands the deck preplace in the
    ur_wrist_2-vs-ur_forearm self-colliding branch (wrist_1 > cap, wrist_2 < min) -- collectsrc's
    'Invalid c-space position: self collision detected'. The wrist-flip re-seed recovers a clear branch at
    the SAME TCP pose. This is exactly what collect_place_wrist_guard automates."""
    stow = np.array([0.0, -2.5, 1.7, -0.7, -1.5708, 0.0])
    T = st.pose_T(PREPLACE, TCP_Q)
    q, perr, rerr = ik.solve(T, stow)
    assert perr < 1e-3 and rerr < 1e-2          # a valid IK solution
    assert not _clear(q), 'expected the stow-seeded branch to be self-colliding (guard would be pointless)'
    qf, pf, rf = ik.solve(T, np.array(wrist_flip_seed(q)))
    assert pf < 1e-3 and rf < 1e-2
    assert _clear(qf), 'the wrist-flipped re-seed must recover a clear branch at the same TCP pose'


def test_deep_fold_clear_branch_is_untrackable_and_upright_is_the_fix(ik):
    """Both deck poses also admit a clear-wrist DEEP-fold branch (shoulder_lift ~ -3.57) -- the one the
    host's folded-carry near_start seeding picks, and the one collect1 could not HOLD. The upright clear
    branch is far lower droop. This is why the goal fix is a branch choice, not a planner tweak."""
    for xyz in (PREPLACE, PLACE):
        T = st.pose_T(xyz, TCP_Q)
        # the deep-fold clear branch (seed a deep lift; flip the wrist to reach the clear twin)
        qd = None
        for seed in (np.array([0.0, -3.5, 2.1, -1.6, 0.9, -3.14]),
                     np.array([-2.2, -3.5, 2.3, -1.9, 0.6, -3.14])):
            for s in (seed, np.array(wrist_flip_seed(seed))):
                q, p, r = ik.solve(T, s)
                if p < 1e-3 and r < 1e-2 and _clear(q) and _droop(q) > 1.5:
                    qd = q
                    break
            if qd is not None:
                break
        assert qd is not None, 'expected a clear DEEP-fold branch to exist (the one that stalled collect1)'
        # the upright clear branch
        qu, _ = _best_upright_clear(ik, xyz)
        assert qu is not None
        assert _droop(qu) < 0.6, f'upright droop {_droop(qu):.2f} unexpectedly high'
        assert _droop(qd) > 1.5, f'deep-fold droop {_droop(qd):.2f} unexpectedly low'
        assert _droop(qu) < _droop(qd) - 1.0, 'upright branch must be clearly lower droop than the deep fold'


def test_shipped_upright_preplace_shape_is_clear_in_limits_and_matches_the_standoff(ik):
    """The config value shipped in the example fragment is a valid, collision-free upright preplace: in
    joint limits, in the clear wrist region, FK TCP at the recorded preplace standoff, reach within gate."""
    q = np.array(UPRIGHT_PREPLACE_SHAPE)
    assert len(q) == 6 and np.all(np.isfinite(q))
    assert np.all(q >= ik.lower - 1e-9) and np.all(q <= ik.upper + 1e-9), 'shape outside joint limits'
    assert _clear(q), 'shipped preplace shape is not in the clear wrist region'
    assert _droop(q) < 0.6, 'shipped preplace shape is not the low-droop upright branch'
    T = ik.fk(q)
    assert _reach(T[:3, 3]) <= MAX_REACH
    # FK TCP within ~1 cm of the recorded preplace standoff (a standoff, so a few mm of run-to-run deck
    # variance is fine; the sensed place insert corrects the rest).
    assert np.linalg.norm(T[:3, 3] - np.array(PREPLACE)) < 0.01, (
        f'shape FK {np.round(T[:3,3],3).tolist()} not at the preplace standoff {PREPLACE}')


def test_place_from_the_upright_preplace_stays_upright_and_clear(ik):
    """Seeded (near_start) from the upright preplace shape, the SENSED place target solves to the upright
    clear branch with a small joint move -- the controller's place pose-IK path under
    collect_place_wrist_guard. (Confirms the preplace->place transition does not flip back to the deep fold.)"""
    pre = np.array(UPRIGHT_PREPLACE_SHAPE)
    T = st.pose_T(PLACE, TCP_Q)
    q, perr, rerr = ik.solve(T, pre)
    assert perr < 1e-3 and rerr < 1e-2
    assert _clear(q), 'place IK from the upright preplace left the clear wrist region'
    assert _droop(q) < 0.6, 'place IK from the upright preplace fell into the deep fold'
    assert float(np.max(np.abs(q - pre))) < 0.6, 'place is not a small move from the upright preplace'


def test_backward_compatible_defaults_off():
    """Controller defaults: both collect place-goal switches off => the collect place path is unchanged."""
    from twin.restock_manipulation.controller import TransferController
    from twin.restock_manipulation.types import ProductProfile

    class _Motion:                                   # the controller only needs begin/begin_joint_goal/poll/stop
        def begin(self, *a, **k): ...
        def begin_joint_goal(self, *a, **k): ...
        def poll(self): return 'running'
        def stop(self): ...

    prof = {'juice_orange': ProductProfile(sku='juice_orange', dimensions=(0.6, 0.4, 0.16),
                                           tcp_quaternion=(0.5, 0.5, 0.5, 0.5), grasp_mode='top')}
    c = TransferController(_Motion(), prof)
    assert c.collect_place_preplace_shape is None
    assert c.collect_place_wrist_guard is False
    c.operation = 'collect'
    assert c._place_wrist_guard() is None            # collect, flag off -> no guard (byte-identical begin())
    c.collect_place_wrist_guard = True
    assert c._place_wrist_guard() == (DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN)
    c.operation = 'transfer'                          # transfer path still keyed on RESTOCK_TRANSFER_CARRY only
    assert c._place_wrist_guard() is None


def test_preplace_shape_validation_rejects_bad_values():
    from twin.restock_manipulation.controller import TransferController
    from twin.restock_manipulation.types import ProductProfile

    class _Motion:
        def begin(self, *a, **k): ...
        def begin_joint_goal(self, *a, **k): ...
        def poll(self): return 'running'
        def stop(self): ...

    prof = {'juice_orange': ProductProfile(sku='juice_orange', dimensions=(0.6, 0.4, 0.16),
                                           tcp_quaternion=(0.5, 0.5, 0.5, 0.5), grasp_mode='top')}
    with pytest.raises(ValueError):
        TransferController(_Motion(), prof, collect_place_preplace_shape=(0.0, 0.0, 0.0))   # not 6 values
    with pytest.raises(ValueError):
        TransferController(_Motion(), prof, collect_place_preplace_shape=(0, 0, float('nan'), 0, 0, 0))
