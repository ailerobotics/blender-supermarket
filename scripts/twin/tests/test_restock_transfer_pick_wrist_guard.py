"""sim-robust-transfer-settle-oscillation-v1 -- CPU proof that the live_pose TRANSFER PICK (approach +
contact) is steered onto the clear wrist branch, so the coldwin12 contact limit cycle cannot recur, and
that the key OFF (and collect / non-live_pose / non-pick) is byte-identical. No Isaac/ROS/GPU.

Root cause (coldwin12, head 25511f5, run ~/datasets/twin-robot-in-store-boot-v1/mission/coldwin12; see
data/sim-robust-transfer-settle-oscillation-v1/report.md):

The carry wrist_guard (RESTOCK_TRANSFER_CARRY=wrist_guard) covers only the PLACE half
(transfer_carry.GUARDED_PHASES = preplace/place/retreat). The live_pose transfer PICK was unguarded: the
APPROACH is a cuMotion task-space pose goal and the CONTACT is a straight-line seeded-KDL IK seeded from the
approach branch. In coldwin12 the approach IK landed on the ur_wrist_2-vs-ur_forearm self-colliding family
(approach end wrist_2 -0.598; contact goal wrist_1 1.171 > the ~0.69 stop, wrist_2 -1.033 < 0) and the
straight-line contact descended into it and limit-cycled 0.169 rad / ~1.6 rad/s for the WHOLE 30 s settle
window (PR239's terminal-settle envelope correctly rejected it). coldwin11 (byte-identical config) took the
clear wrist_2-positive family (wrist_1 -0.18, wrist_2 +0.93) and settled instantly. The fix bounds the pick
to the clear region (wrist_1 <= carry_wrist1_cap, wrist_2 >= carry_wrist2_min) behind transfer_pick_wrist_guard.

The recorded configs below are ARM order (pan, lift, elbow, wrist_1, wrist_2, wrist_3).
"""
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from twin.restock_manipulation import transfer_carry as tc  # noqa: E402
from twin.restock_manipulation import transfer_mode as tm  # noqa: E402
from twin.restock_manipulation.controller import TransferController  # noqa: E402
from twin.restock_manipulation.transfer_carry import (  # noqa: E402
    DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN, in_clear_region, wrist_flip_seed)
from twin.restock_manipulation.ros_adapter import RosMotionPort, WRIST_FLIP_MAX  # noqa: E402

# --- recorded coldwin12 (FAIL) pick configs, ARM order ------------------------------------------------
# host-stack.log:25864 TRAJ_HOST_GOAL_MISS goal_q (the contact goal the straight-line IK descended to):
COLDWIN12_CONTACT_GOAL = (0.502, -3.183, 2.012, 1.171, -1.033, 0.0)   # w1 1.171 > stop, w2 -1.033 < 0
# the approach END = contact START (host-stack.log:22470 plan first point, remapped to ARM order):
COLDWIN12_APPROACH_END = (0.9371, -2.9565, 2.2084, 0.7473, -0.5983, 0.0008)   # already wrist_2-negative
# --- recorded coldwin11 (PASS) pick configs, ARM order (same move, clear family) ----------------------
# host-stack.log:22560 approach goal and :23193 contact goal (remapped to ARM order):
COLDWIN11_APPROACH_GOAL = (2.6579, -1.7401, 1.9183, -0.1782, 0.9277, 0.0)
COLDWIN11_CONTACT_GOAL = (2.6618, -1.7349, 1.9141, -0.1792, 0.9316, 0.0)


def _clear(q):
    return in_clear_region(q, DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN)


# --------------------------------------------------------------------- the recorded-trace classification

def test_coldwin12_pick_configs_are_the_self_colliding_family():
    # Both the approach the pick committed to AND the contact goal it descended to are OUTSIDE the clear
    # region -- the ur_wrist_2-vs-ur_forearm family the limit cycle jammed in.
    assert not _clear(COLDWIN12_CONTACT_GOAL)       # wrist_1 1.171 > 0.65 AND wrist_2 -1.033 < 0.0
    assert not _clear(COLDWIN12_APPROACH_END)       # wrist_2 -0.598 < 0.0 (approach already colliding)


def test_coldwin11_pick_configs_are_the_clear_family_and_stay_unchanged():
    # coldwin11 took the clear wrist_2-positive family; the guard ACCEPTS it, so that run is byte-identical.
    assert _clear(COLDWIN11_APPROACH_GOAL)          # wrist_1 -0.178 <= 0.65, wrist_2 +0.928 >= 0.0
    assert _clear(COLDWIN11_CONTACT_GOAL)


def test_coldwin12_contact_wrist_flip_lands_in_the_clear_family():
    # The wrist-flipped twin of the coldwin12 colliding contact goal is the SAME TCP in the clear branch
    # (the re-seed the straight-line guard dispatches); it matches the coldwin11 clear family's wrist signs.
    flip = wrist_flip_seed(COLDWIN12_CONTACT_GOAL)
    assert _clear(flip)
    assert flip[4] > 0.0                             # wrist_2 flipped positive, like coldwin11 (+0.93)


# --------------------------------------------------------------------- _pick_wrist_guard decision (pure)

def _controller_stub(*, pick_guard_on, mode, operation, phase, cap=None, w2min=None):
    """A TransferController carrying only the attributes _pick_wrist_guard / _live_pose_transfer read."""
    c = TransferController.__new__(TransferController)
    c.transfer_pick_wrist_guard = pick_guard_on
    c.transfer_mode = mode
    c.operation = operation
    c.phase = phase
    c.carry_wrist1_cap = cap
    c.carry_wrist2_min = w2min
    return c


def test_pick_guard_active_for_live_pose_transfer_pick_phases_only():
    for phase in ('approach', 'contact'):
        c = _controller_stub(pick_guard_on=True, mode=tm.LIVE_POSE, operation='transfer', phase=phase)
        assert c._pick_wrist_guard() == (DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN)
    # the configured carry caps override the bench defaults
    c = _controller_stub(pick_guard_on=True, mode=tm.LIVE_POSE, operation='transfer', phase='approach',
                         cap=0.5, w2min=0.1)
    assert c._pick_wrist_guard() == (0.5, 0.1)


def test_pick_guard_off_is_byte_identical():
    c = _controller_stub(pick_guard_on=False, mode=tm.LIVE_POSE, operation='transfer', phase='approach')
    assert c._pick_wrist_guard() is None


@pytest.mark.parametrize('mode,operation,phase', [
    (None, 'transfer', 'approach'),          # transfer_mode off (not live_pose)
    ('scripted', 'transfer', 'approach'),    # a different transfer mode
    (tm.LIVE_POSE, 'collect', 'approach'),   # collect is never guarded
    (tm.LIVE_POSE, 'transfer', 'lift'),      # lift has its own carry guard, not this one
    (tm.LIVE_POSE, 'transfer', 'preplace'),  # place half is the carry wrist_guard's job
    (tm.LIVE_POSE, 'transfer', 'grasp'),     # not a move-IK phase
])
def test_pick_guard_scoped_out_of_everything_else(mode, operation, phase):
    c = _controller_stub(pick_guard_on=True, mode=mode, operation=operation, phase=phase)
    assert c._pick_wrist_guard() is None


# --------------------------------------------------------------------- the straight-line contact IK guard

def _guard_double(wrist_guard, n_remaining_seeds):
    """A RosMotionPort with only the attributes _guard_rejects touches (no ROS); _send_ik is a no-op so the
    real _next_seed advances purely over the seed list. Mirrors test_restock_transfer_preplace_reach."""
    port = RosMotionPort.__new__(RosMotionPort)
    port._wrist_guard = wrist_guard
    port._reach_guard = None                      # the pick contact never carries a reach guard
    port._seeds = [object()] * (n_remaining_seeds + 1)
    port._seed_idx = 0
    port._wrist_flips = 0
    port._send_ik = lambda seed: None
    return port


def test_contact_guard_rejects_the_coldwin12_colliding_config_and_reseeds():
    port = _guard_double((DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN), n_remaining_seeds=2)
    # the colliding contact goal is rejected (True = keep seeking) and a wrist-flip re-seed is dispatched
    assert port._guard_rejects(list(COLDWIN12_CONTACT_GOAL)) is True
    assert port._wrist_flips == 1


def test_contact_guard_accepts_the_coldwin11_clear_config():
    port = _guard_double((DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN), n_remaining_seeds=2)
    assert port._guard_rejects(list(COLDWIN11_CONTACT_GOAL)) is False
    assert port._wrist_flips == 0                 # a clear solution needs no flip


def test_contact_guard_raises_by_name_when_no_clear_branch_remains():
    port = _guard_double((DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN), n_remaining_seeds=0)
    port._wrist_flips = WRIST_FLIP_MAX              # flips already spent -> no re-seed left
    with pytest.raises(RuntimeError, match=r'IK outside the carry wrist region'):
        port._guard_rejects(list(COLDWIN12_CONTACT_GOAL))


def test_contact_guard_inactive_is_byte_identical():
    # With no wrist_guard set, the straight-line contact path never consults the guard (the adapter's
    # `getattr(self, '_wrist_guard', None) is not None` gate is False) -- the legacy descent, unchanged.
    port = _guard_double(None, n_remaining_seeds=2)
    assert port._wrist_guard is None


# --------------------------------------------------------------------- end-to-end wiring through the drive
# Reuse the live_pose harness: a real controller + real near-face reach-gated live_pose transfer drive.
from scripts.twin.tests import test_restock_transfer_live_pose as lp  # noqa: E402


class _PhaseRecordingMotion:
    """A fixture motion that records, per begin(), the controller's current phase and the wrist
    constraints / straight-line flag / extra kwargs the controller passed."""
    def __init__(self, controller_ref):
        self._c = controller_ref
        self.calls = []

    def begin(self, pose, joints, straight_line=False, joint_space=False, wrist1_cap=None,
              wrist2_min=None, slow=False, goal_tol=None, near_start=False, **extra):
        self.calls.append(dict(phase=self._c[0].phase, wrist1_cap=wrist1_cap, wrist2_min=wrist2_min,
                               straight_line=bool(straight_line), extra=dict(extra)))

    def begin_joint_goal(self, goal, measured, slow=False, goal_tol=None):
        pass

    def poll(self):
        return 'succeeded'

    def last_fault_was_sim_deadline(self):
        return False

    def stop(self):
        pass

    def update_payload(self, held, profile):
        return True


def _host_controller_pickguard(policy, pick_guard_on):
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    ref = []
    m = _PhaseRecordingMotion(ref)
    c = TransferController(
        m, {'juice_orange': lp.PROFILE}, pose_policy=policy, transport_joints=stow,
        contact_straight_line=True, staged_cart_swing=True,
        cart_pan_joint0=lp.HOST['cart_pan_joint0'], cart_tuck_shape=lp.HOST['cart_tuck_shape'],
        carry_lift_shape=lp.HOST['carry_lift_shape'], carry_wrist1_cap=lp.HOST['carry_wrist1_cap'],
        carry_wrist2_min=lp.HOST['carry_wrist2_min'], grasp_track_m=lp.HOST['grasp_track_m'],
        reposition_before_place=lp.HOST['reposition_before_place'],
        transfer_ground_truth=True, transfer_pick_wrist_guard=pick_guard_on, phase_timeout=120.)
    ref.append(c)
    c.start('transfer', lp.JOB)
    return c, m, stow


def _first_call(calls, phase):
    return next(c for c in calls if c['phase'] == phase)


def test_drive_approach_and_contact_carry_the_pick_guard_when_on(monkeypatch):
    monkeypatch.setenv(tm.ENV, 'live_pose')
    monkeypatch.setenv(tc.ENV, 'wrist_guard')
    policy = lp.faithful_policy()
    c, m, stow = _host_controller_pickguard(policy, pick_guard_on=True)
    phases, r = lp.run_stage(c, lp.live_source(), stow)
    assert r.status == 'succeeded', (c.phase, r.reason)
    cap, w2min = lp.HOST['carry_wrist1_cap'], lp.HOST['carry_wrist2_min']
    # APPROACH: a cuMotion task-space pose goal carrying the clear-region joint constraints
    appr = _first_call(m.calls, 'approach')
    assert appr['straight_line'] is False
    assert appr['wrist1_cap'] == cap and appr['wrist2_min'] == w2min
    # CONTACT: a straight-line seeded-KDL IK handed the wrist_guard extra (clear-branch re-seed)
    cont = _first_call(m.calls, 'contact')
    assert cont['straight_line'] is True
    assert cont['extra'].get('wrist_guard') == (cap, w2min)


def test_drive_pick_guard_off_is_byte_identical(monkeypatch):
    monkeypatch.setenv(tm.ENV, 'live_pose')
    monkeypatch.setenv(tc.ENV, 'wrist_guard')
    policy = lp.faithful_policy()
    c, m, stow = _host_controller_pickguard(policy, pick_guard_on=False)
    phases, r = lp.run_stage(c, lp.live_source(), stow)
    assert r.status == 'succeeded', (c.phase, r.reason)
    # OFF: the pick approach carries NO wrist constraints and the contact carries NO wrist_guard extra.
    appr = _first_call(m.calls, 'approach')
    assert appr['wrist1_cap'] is None and appr['wrist2_min'] is None
    cont = _first_call(m.calls, 'contact')
    assert cont['straight_line'] is True
    assert 'wrist_guard' not in cont['extra']
