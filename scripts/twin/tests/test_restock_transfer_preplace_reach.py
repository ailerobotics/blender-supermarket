"""sim-robust-transfer-preplace-reach-v1 -- CPU proof that the live_pose TRANSFER preplace has a HOLDABLE,
reachable IK goal from the pose the FULL chain parks in, and that the controller/adapter select it instead
of the deep-fold the plain wrist-guard accepted (no Isaac/ROS/GPU).

Root cause (coldwin6, head 42e154c / PR #242, run
~/datasets/twin-robot-in-store-boot-v1/mission/coldwin6; see
data/sim-robust-transfer-preplace-reach-v1/report.md):

The live_pose transfer preplace standoff above the backroom bay has (at least) two wrist-CLEAR IK branches
for the mission URDF:

  * a HOLDABLE UPRIGHT one (ur_shoulder_lift ~ -1.95 rad, fold_droop ~0.38), and
  * a DEEP fold (ur_shoulder_lift ~ -4.00 rad, fold_droop ~3.85) the arm physically cannot reach.

The staged recorded_bay_park runs (armcam4/armcameco/htsim) accept the UPRIGHT branch at the first
(measured) seed with no wrist flip. The FULL chain parks at a slightly different pose, where KDL's measured
seed lands in the deep branch (ur_shoulder_lift -3.404, wrist NOT clear); the wrist-flip re-seed that clears
the wrist then converges to the deep-fold clear twin (-4.003), which the plain wrist-guard ACCEPTED -- the
arm froze 0.35 rad short at -3.65 -> PATH_TOLERANCE_VIOLATED -> HOST_FAULT_ESTOP phase=preplace.

The fix is a GOAL/branch fix: the reach guard (transfer_preplace_reach_guard) requires a clear-wrist
solution to ALSO be a holdable fold (transfer_carry.reachable_fold) and seeds an upright config, so the
seeded-KDL IK selects the holdable branch; a clear-but-deep-fold goal is rejected by name before execution.

The arm chain below is the coldwin6 / run-356 mobile-import.urdf ur_base_link -> gripper_pad_tcp chain
verbatim (the same constant as test_restock_cart_place_goal / test_restock_scripted_transfer; checked
against the dataset import when present).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from twin.restock import scripted_transfer as st  # noqa: E402
from twin.restock_manipulation import transfer_carry as tc  # noqa: E402
from twin.restock_manipulation.transfer_carry import (  # noqa: E402
    DEFAULT_MAX_FOLD_DROOP, DEFAULT_UPRIGHT_SEED, DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN,
    fold_droop, in_clear_region, reachable_fold, resolve_reach_guard, wrap_angle, wrist_flip_seed)

# --- recorded transfer configs/poses in ur_base_link (ARM order: pan, lift, elbow, w1, w2, w3) ---
# coldwin6 host-stack.log lines 26393/26394 + TRAJ_HOST_GOAL_MISS (the FAILING full-chain preplace):
#   original KDL (rejected, wrist not clear):  ur_shoulder_lift -3.404, w1 1.02, w2 -1.128
#   wrist-flipped accepted (INFEASIBLE deep):  ur_shoulder_lift -4.003, froze at -3.65
COLDWIN6_KDL_RAW = (-1.047, -3.404, 2.383, 1.02, -1.128, 0.0)
COLDWIN6_DEEP_CLEAR = (-1.047, -4.003, 2.62, -1.759, 1.128, -3.142)
# staged armcam4/armcameco/htsim preplace, accepted at seed=0 flips=0 (the holdable UPRIGHT clear branch):
STAGED_UPRIGHT_CLEAR = (1.001, -1.735, 2.235, -0.5, 1.0, -0.0)
# pick_top (carton on the cart) in ur_base_link at the coldwin6 transfer observe (reach 0.988 < max_reach):
COLDWIN6_PICK_TOP = (-1.041, -0.007, 0.592)
# the tuck config the pick+lift chain reached cleanly before preplace (first transfer TRAJ_HOST_PLAN end):
COLDWIN6_TUCK = (2.184, -2.008, 2.100, -0.093, 0.386, 0.0)
MAX_REACH = 1.3

# coldwin6 / run-356 mobile-import.urdf, ur_base_link -> gripper_pad_tcp (name, type, parent, child, xyz, rpy, lo, hi)
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
RUN_URDF = Path('/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin6/mobile-import.urdf')


def _urdf_text():
    links = sorted({c[2] for c in _CHAIN} | {c[3] for c in _CHAIN})
    out = ['<robot name="transfer_preplace_arm_chain">'] + [f'  <link name="{n}"/>' for n in links]
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


def _clear(q):
    return in_clear_region(q, DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN)


def _solve(ik, T, seed):
    q, perr, rerr = ik.solve(T, np.asarray(seed, float))
    return q, perr, rerr


def _solved(perr, rerr):
    return perr < 8e-4 and rerr < 8e-3


# --------------------------------------------------------------------- fixture integrity

def test_fixture_chain_matches_the_dataset_import_when_present(ik):
    if not RUN_URDF.is_file():
        pytest.skip('dataset mobile-import.urdf not present')
    real = st.ArmIK(RUN_URDF)
    q = np.array([0.1, -1.9, 2.0, -0.1, 1.6, 0.0])
    assert np.allclose(real.fk(q), ik.fk(q), atol=1e-9)


def test_recorded_deep_and_raw_solutions_reach_the_same_preplace_pose(ik):
    # The original KDL branch (rejected) and the wrist-flipped deep fold (accepted) are the SAME TCP pose:
    # the deep fold is a genuine IK solution for the preplace standoff, not a different target.
    T_raw, T_deep = ik.fk(COLDWIN6_KDL_RAW), ik.fk(COLDWIN6_DEEP_CLEAR)
    assert np.linalg.norm(T_raw[:3, 3] - T_deep[:3, 3]) < 2e-3
    assert np.linalg.norm(st.rot_err(T_raw[:3, :3], T_deep[:3, :3])) < 5e-3


# --------------------------------------------------------------------- the core preplace claims

def test_coldwin6_deep_fold_is_clear_but_not_reachable(ik):
    # What the plain wrist-guard accepted: wrist-clear, but a deep (un-holdable) fold -> the fix rejects it.
    assert _clear(COLDWIN6_DEEP_CLEAR)
    assert fold_droop(COLDWIN6_DEEP_CLEAR) > 3.0
    assert not reachable_fold(COLDWIN6_DEEP_CLEAR, DEFAULT_MAX_FOLD_DROOP)


def test_coldwin6_park_pose_has_a_reachable_upright_clear_preplace_branch(ik):
    # A HOLDABLE, wrist-clear branch provably EXISTS for the SAME preplace pose at the full-chain park pose.
    T = ik.fk(COLDWIN6_DEEP_CLEAR)   # the preplace standoff TCP (deep + upright are the same pose)
    q, perr, rerr = _solve(ik, T, DEFAULT_UPRIGHT_SEED)
    assert _solved(perr, rerr), (perr, rerr)
    assert _clear(q)
    assert reachable_fold(q, DEFAULT_MAX_FOLD_DROOP)
    assert abs(wrap_angle(q[1]) - (-math.pi / 2)) < 0.6   # shoulder_lift near upright (~-1.95)


def test_staged_park_preplace_is_already_reachable_upright_and_clear(ik):
    # The staged recorded_bay_park branch the staged runs accepted at seed 0 is holdable + clear: the fix
    # accepts it unchanged (byte-identical outcome on the staged-start path).
    assert _clear(STAGED_UPRIGHT_CLEAR)
    assert reachable_fold(STAGED_UPRIGHT_CLEAR, DEFAULT_MAX_FOLD_DROOP)
    # and it is a genuine IK solution reached from the measured branch (not a different pose)
    T = ik.fk(STAGED_UPRIGHT_CLEAR)
    q, perr, rerr = _solve(ik, T, STAGED_UPRIGHT_CLEAR)
    assert _solved(perr, rerr)
    assert _clear(q) and reachable_fold(q, DEFAULT_MAX_FOLD_DROOP)


def test_coldwin6_pick_is_reachable(ik):
    # pick side: pick_top is within the arm's reach gate, and the pick+lift chain reached a clear, holdable
    # tuck config (coldwin6 reached preplace, so the pick empirically succeeded).
    assert float(np.linalg.norm(COLDWIN6_PICK_TOP)) < MAX_REACH
    assert _clear(COLDWIN6_TUCK)
    assert reachable_fold(COLDWIN6_TUCK, DEFAULT_MAX_FOLD_DROOP)


# --------------------------------------------------------------------- resolve_reach_guard normalisation

def test_resolve_reach_guard_off_and_defaults():
    assert resolve_reach_guard(None) is None
    assert resolve_reach_guard(False) is None
    assert resolve_reach_guard('off') is None
    md, seed = resolve_reach_guard(True)
    assert md == DEFAULT_MAX_FOLD_DROOP and seed == tuple(float(v) for v in DEFAULT_UPRIGHT_SEED)
    assert resolve_reach_guard({}) == (DEFAULT_MAX_FOLD_DROOP, tuple(float(v) for v in DEFAULT_UPRIGHT_SEED))
    assert resolve_reach_guard({'max_droop': 1.5})[0] == 1.5
    assert resolve_reach_guard({'upright_seed': False})[1] is None


@pytest.mark.parametrize('bad', [{'max_droop': -1}, {'max_droop': 0}, {'upright_seed': [1, 2, 3]}, 'bogus', 5])
def test_resolve_reach_guard_rejects_malformed(bad):
    with pytest.raises(ValueError):
        resolve_reach_guard(bad)


# --------------------------------------------------------------------- the adapter guard decision

from twin.restock_manipulation.ros_adapter import RosMotionPort  # noqa: E402


def _guard_double(reach_guard, n_remaining_seeds):
    """A RosMotionPort with only the attributes _guard_rejects touches (no ROS). _send_ik is a no-op so the
    real _next_seed advances purely over the seed list."""
    port = RosMotionPort.__new__(RosMotionPort)
    port._wrist_guard = (DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN)
    port._reach_guard = reach_guard
    port._seeds = [object()] * (n_remaining_seeds + 1)
    port._seed_idx = 0
    port._wrist_flips = 0
    port._send_ik = lambda seed: None   # no IK dispatch in the CPU test
    return port


def test_guard_accepts_the_upright_clear_reachable_branch():
    port = _guard_double((DEFAULT_MAX_FOLD_DROOP, DEFAULT_UPRIGHT_SEED), n_remaining_seeds=2)
    # a holdable, clear solution is accepted (not rejected)
    assert port._guard_rejects(list(STAGED_UPRIGHT_CLEAR)) is False


def test_guard_rejects_deep_fold_and_advances_to_next_seed():
    port = _guard_double((DEFAULT_MAX_FOLD_DROOP, DEFAULT_UPRIGHT_SEED), n_remaining_seeds=1)
    # clear-but-deep -> reject (True = keep seeking), advance the seed, no wrist flip consumed
    assert port._guard_rejects(list(COLDWIN6_DEEP_CLEAR)) is True
    assert port._seed_idx == 1
    assert port._wrist_flips == 0


def test_guard_raises_by_name_when_no_reachable_branch_left():
    port = _guard_double((DEFAULT_MAX_FOLD_DROOP, DEFAULT_UPRIGHT_SEED), n_remaining_seeds=0)
    with pytest.raises(RuntimeError, match=r'transfer preplace: no wrist-clear IK branch within the reachable'):
        port._guard_rejects(list(COLDWIN6_DEEP_CLEAR))


def test_without_reach_guard_the_deep_fold_is_accepted_byte_identical():
    # legacy behaviour (reach guard off): the plain wrist-guard accepts the clear deep fold -- the exact
    # coldwin6 bug, reproduced, proving the guard is the only thing that changes it.
    port = _guard_double(None, n_remaining_seeds=2)
    assert port._guard_rejects(list(COLDWIN6_DEEP_CLEAR)) is False
