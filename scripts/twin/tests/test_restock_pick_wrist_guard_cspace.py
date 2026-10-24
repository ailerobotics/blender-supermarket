"""sim-robust-pick-wrist-guard-cspace-fix-v1 -- CPU proof that the guarded cuMotion POSE goal (the
live_pose transfer PICK approach and the fm327/fm403 task-space LIFT fallback) no longer sends a
malformed 2-coordinate c-space target, that the clear wrist family is enforced AFTER planning (re-plan
off the self-colliding family, fail by name when exhausted), that the key off / collect / non-pick stays
byte-identical, and that a shutdown-guard container death is NAMED instead of 'executor stopped'. No
Isaac/ROS/GPU except moveit_msgs (an installed message package, imported for the goal-shape assertions).

Root cause (pickguard1, head 44c8c33 = PR254; ~/datasets/twin-robot-in-store-boot-v1/mission/pickguard1;
see data/sim-robust-pick-wrist-guard-cspace-fix-v1/report.md):

PR254 passed the wrist clear-region into _pose_plan_goal, which appended a 2-entry joint_constraints list
onto the pose goal. cuMotion reads any joint_constraints as a FULL c-space target built from exactly those
entries (discarding the pose), so it called planToCSpaceTarget([[-0.675, 1]]) against a 6-DOF robot:
'Invalid c-space position [2] must equal [6]' x7 -> cumotion::FatalException CUDA OOM -> container death.
The two rejected coordinates are exactly the band midpoints the encoding produced:
  wrist_1: (-2.0 + 0.65)/2 = -0.675 ; wrist_2: (0.0 + 2.0)/2 = 1.0.

Recorded coldwin pick configs are ARM order (pan, lift, elbow, wrist_1, wrist_2, wrist_3).
"""
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from twin.restock_manipulation import ros_adapter as ra  # noqa: E402
from twin.restock_manipulation.ros_adapter import (  # noqa: E402
    JOINTS, POSE_GUARD_REPLAN_MAX, PLAN_TOTAL_ATTEMPTS_CAP, RosMotionPort,
    pose_guard_decision, pose_wrist_guard_record)
from twin.restock_manipulation.transfer_carry import (  # noqa: E402
    DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN, in_clear_region)

# coldwin12 (FAIL) colliding family and coldwin11 (PASS) clear family, ARM order.
COLDWIN12_APPROACH_END = (0.9371, -2.9565, 2.2084, 0.7473, -0.5983, 0.0008)   # wrist_2 -0.598 < 0
COLDWIN12_CONTACT_GOAL = (0.502, -3.183, 2.012, 1.171, -1.033, 0.0)           # w1 1.171, w2 -1.033
COLDWIN11_APPROACH_GOAL = (2.6579, -1.7401, 1.9183, -0.1782, 0.9277, 0.0)     # clear (w1 -0.18, w2 +0.93)
GUARD = (DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN)   # (0.65, 0.0)


# ------------------------------------------------------------------ the arithmetic that produced [-0.675, 1]
def test_pr254_midpoints_match_the_rejected_cspace_pair():
    # PR254's encoding: wrist_1 position (low+cap)/2, wrist_2 position (w2min+high)/2, low=-2, high=+2.
    w1 = (-2.0 + DEFAULT_WRIST1_CAP) / 2.0
    w2 = (DEFAULT_WRIST2_MIN + 2.0) / 2.0
    assert (round(w1, 3), w2) == (-0.675, 1.0)   # exactly the host-stack.log [[-0.675  1]] target


# ------------------------------------------------------------------ the pose goal is now CLEAN (no jc list)
def _bare_port():
    port = RosMotionPort.__new__(RosMotionPort)
    port.group, port.pipeline, port.velocity_scale, port.tcp = 'ur_manipulator', 'isaac_ros_cumotion', 1.0, 'tool0'
    return port


POSE = (0.1, -0.5, 0.6, 0.0, 0.0, 0.0, 1.0)   # x,y,z + quat
START = (0.5, -1.7, 1.9, -0.18, 0.93, 0.0)


def test_pose_plan_goal_carries_no_joint_constraints():
    # The regression was a joint_constraints list shorter than six on the pose goal. The fixed goal has
    # NONE -- a pure position+orientation Cartesian goal cuMotion plans as a pose, never a c-space target.
    pytest.importorskip('moveit_msgs')   # the installed ROS msg pkg (absent under the PYTHONPATH= test env)
    goal = _bare_port()._pose_plan_goal(POSE, START)
    c = goal.request.goal_constraints[0]
    assert list(c.joint_constraints) == []
    assert len(c.position_constraints) == 1 and len(c.orientation_constraints) == 1
    assert goal.planning_options.plan_only is True


def test_pose_plan_goal_takes_no_wrist_params_anymore():
    # The wrist cap/min are no longer goal-construction inputs (they are a post-plan guard). Passing them
    # positionally/by-keyword must be a TypeError -- a reviewer tripwire against re-introducing the encoding.
    with pytest.raises(TypeError):
        _bare_port()._pose_plan_goal(POSE, START, wrist1_cap=0.65, wrist2_min=0.0)


def test_pose_plan_goal_has_no_jointconstraint_import():
    # Belt-and-braces: the method no longer imports JointConstraint (the symbol that built the bad list).
    import inspect
    src = inspect.getsource(RosMotionPort._pose_plan_goal)
    assert 'JointConstraint' not in src


# ------------------------------------------------------------------ the recorded terminal guard (begin())
def test_pose_wrist_guard_record_on_off_and_one_sided():
    assert pose_wrist_guard_record(0.65, 0.0) == (0.65, 0.0)          # pick guard: both bounds
    assert pose_wrist_guard_record(None, None) is None               # key off / no guard => byte-identical
    cap_only = pose_wrist_guard_record(0.65, None)                   # fm327 lift fallback: wrist_1 only
    assert cap_only[0] == 0.65 and cap_only[1] == float('-inf')      # lower bound permissive
    w2_only = pose_wrist_guard_record(None, 0.0)
    assert w2_only[0] == float('inf') and w2_only[1] == 0.0          # upper bound permissive


def test_one_sided_guard_matches_a_single_cap():
    # A wrist_1-only guard must accept any wrist_2 and reject only a wrist_1 above the cap.
    g = pose_wrist_guard_record(0.65, None)
    assert in_clear_region((0, 0, 0, 0.5, -3.0, 0), *g)      # w2 very negative but w1<=cap -> accept
    assert not in_clear_region((0, 0, 0, 0.9, 5.0, 0), *g)   # w1 0.9 > 0.65 -> reject


# ------------------------------------------------------------------ the post-plan family classification
def test_coldwin11_clear_family_accepted():
    assert pose_guard_decision(list(COLDWIN11_APPROACH_GOAL), GUARD, 0, 0, False) == 'accept'
    assert in_clear_region(COLDWIN11_APPROACH_GOAL, *GUARD)


def test_coldwin12_colliding_family_replans_then_fails_by_budget():
    # The self-colliding approach/contact families are rejected; with budget left -> replan, else fail.
    for q in (COLDWIN12_APPROACH_END, COLDWIN12_CONTACT_GOAL):
        assert not in_clear_region(q, *GUARD)
        assert pose_guard_decision(list(q), GUARD, 0, 0, False) == 'replan'
        # replan budget spent -> fail by name
        assert pose_guard_decision(list(q), GUARD, POSE_GUARD_REPLAN_MAX, 0, False) == 'fail'


def test_colliding_family_fails_when_dispatch_cap_or_wall_spent():
    q = list(COLDWIN12_CONTACT_GOAL)
    assert pose_guard_decision(q, GUARD, 0, PLAN_TOTAL_ATTEMPTS_CAP, False) == 'fail'   # GPU dispatch cap
    assert pose_guard_decision(q, GUARD, 0, 0, True) == 'fail'                          # wall cap


def test_no_guard_is_always_accept_byte_identical():
    # key off / collect / non-pick: guard is None -> the terminal check never rejects (byte-identical).
    assert pose_guard_decision(list(COLDWIN12_CONTACT_GOAL), None, 0, 0, False) == 'accept'


def test_named_failure_string_mentions_the_clear_wrist_region():
    # The fail-fast reason the _poll path raises must name the clear carry wrist region (not a bare
    # 'planning failed'), so the mission surfaces the true cause.
    reason = ('pose goal could not reach the clear carry wrist region (wrist_1<=%.2f, wrist_2>=%.2f) '
              'within %d re-plans' % (GUARD[0], GUARD[1], POSE_GUARD_REPLAN_MAX))
    assert 'clear carry wrist region' in reason and 'wrist_1<=0.65' in reason


# ------------------------------------------------------------------ step 3: name the container death
from twin.restock_host import host_guard_exit as hge  # noqa: E402


def test_clean_line_strips_ansi_and_keeps_last_nonempty():
    raw = "\x1b[31mterminate called ...\x1b[0m\n\x1b[31m  what():  CUDA error: out of memory\x1b[0m\n"
    assert hge.clean_line(raw) == 'what():  CUDA error: out of memory'
    assert hge.clean_line(b'') == '' and hge.clean_line(None) == ''


def test_guard_exit_reason_names_process_code_and_last_line():
    r = hge.guard_exit_reason('host_components', -6, 'what():  CUDA error: out of memory')
    assert "host component 'host_components' exited (exit code -6)" in r
    assert 'CUDA error: out of memory' in r


def test_guard_marker_round_trips(tmp_path):
    marker = str(tmp_path / 'guard.json')
    assert hge.read_guard_exit(marker) is None            # absent -> None
    hge.write_guard_exit(marker, 'host_components', -6, 'what():  CUDA error: out of memory')
    rec = hge.read_guard_exit(marker)
    assert rec['process_name'] == 'host_components' and rec['returncode'] == -6
    assert 'CUDA error: out of memory' in rec['reason']


def test_guard_exit_fault_reason_prefers_marker_else_default():
    rec = hge.guard_exit_record('host_components', -6, 'what():  CUDA error: out of memory')
    assert hge.guard_exit_fault_reason(rec, 'executor stopped') == rec['reason']
    assert hge.guard_exit_fault_reason(None, 'executor stopped') == 'executor stopped'   # clean run
    assert hge.guard_exit_fault_reason({}, 'executor stopped') == 'executor stopped'


def test_write_guard_exit_never_raises_on_bad_path():
    hge.write_guard_exit('', 'x', -6)                     # falsy path -> no-op
    hge.write_guard_exit('/no/such/dir/guard.json', 'x', -6)   # unwritable -> swallowed
