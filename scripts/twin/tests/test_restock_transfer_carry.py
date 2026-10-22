"""CPU-only: the RESTOCK_TRANSFER_CARRY=wrist_guard transfer-carry switch (warm2 preplace wrist jam).

Fixtures are the warm2 numbers (host-stack.log TRAJ_HOST_PLAN / TRAJ_HOST_TRACK_MISS and a CPU IK of the
same preplace TCP pose on the mission URDF); planner/IK here are fakes, never live evidence.
"""
import math
from types import SimpleNamespace as NS

import pytest

from scripts.twin.restock_manipulation import transfer_carry as tc
from scripts.twin.restock_manipulation.ros_adapter import JOINTS, WRIST_FLIP_MAX
from scripts.twin.tests.test_restock_manipulation import (
    JOB, PROFILE, FixtureMotion, packet, run_to)
from scripts.twin.restock_manipulation import TransferController

# ARM order (pan, lift, elbow, wrist_1, wrist_2, wrist_3).
WARM2_PREPLACE_KDL = (1.0503, -1.3523, 2.4136, 2.0802, -1.0155, -3.1416)  # the jammed preplace goal
WARM2_JAM = (1.0341, -1.4397, 2.3087, 0.6779, 0.0872, -3.0894)           # measured at TRAJ_HOST_FAULT
WARM2_CLEAR = (1.0503, -1.7175, 2.2212, -0.5038, 1.0155, 0.0)            # same TCP pose, clear branch
CARRY_SHAPE = (1.2675, -2.27, 1.0, 0.65, 0.8, -1.2545)                   # carry_lift_shape = tuck


# ---------------------------------------------------------------- switch parsing + labels

def test_flags_off_by_default_and_parse():
    assert tc.carry_flags({}) == frozenset()
    for off in ('', '0', 'off', 'none', 'false', ' OFF '):
        assert tc.carry_flags({tc.ENV: off}) == frozenset()
    assert tc.carry_flags({tc.ENV: 'wrist_guard'}) == {tc.WRIST_GUARD}
    assert tc.carry_flags({tc.ENV: ' Wrist_Guard , '}) == {tc.WRIST_GUARD}


def test_unknown_flag_raises_loudly():
    with pytest.raises(ValueError, match='unknown transfer carry flag'):
        tc.carry_flags({tc.ENV: 'wrist_guard,tilt_bound'})


def test_stamp_summary_off_is_byte_identical():
    summary = {'note': 'n', 'walkthrough_caption': 'c'}
    assert tc.stamp_summary(dict(summary), environ={}) == summary


def test_stamp_summary_records_flag_workaround_note_and_caption_once():
    env = {tc.ENV: 'wrist_guard'}
    summary = {'note': 'base.', 'walkthrough_caption': 'BENCH RUN', 'workarounds': ['transfer_mode=live_pose']}
    tc.stamp_summary(summary, environ=env)
    tc.stamp_summary(summary, environ=env)               # idempotent (both mission.json writers call it)
    assert summary['transfer_carry'] == ['wrist_guard']
    assert summary['workarounds'] == ['transfer_mode=live_pose', 'transfer_carry=wrist_guard']
    assert summary['note'].count('TRANSFER_CARRY=wrist_guard') == 1
    assert summary['walkthrough_caption'] == 'BENCH RUN | ' + tc.CAPTIONS[tc.WRIST_GUARD]


def test_invalid_value_is_recorded_not_hidden():
    summary = tc.stamp_summary({}, environ={tc.ENV: 'bogus'})
    assert summary['transfer_carry'] == 'invalid' and 'bogus' in summary['transfer_carry_error']
    assert tc.carry_caption('cap', environ={tc.ENV: 'bogus'}) == 'cap'


def test_caption_off_unchanged_on_appended():
    assert tc.carry_caption('cap', environ={}) == 'cap'
    assert tc.carry_caption(None, environ={}) is None
    assert tc.carry_caption(None, environ={tc.ENV: 'wrist_guard'}) == tc.CAPTIONS[tc.WRIST_GUARD]


# ---------------------------------------------------------------- the clear-region geometry (pure)

def test_warm2_preplace_goal_and_jam_are_outside_the_clear_region():
    assert not tc.in_clear_region(WARM2_PREPLACE_KDL, 0.65, 0.0)   # w1 2.08 > cap and w2 -1.02 < min
    assert not tc.in_clear_region(WARM2_JAM, 0.65, 0.0)            # pinned at the stop, w1 0.678 > cap


def test_clear_solution_and_carry_shape_are_inside():
    assert tc.in_clear_region(WARM2_CLEAR, 0.65, 0.0)
    assert tc.in_clear_region(CARRY_SHAPE, 0.65, 0.0)


def test_region_uses_the_physical_wrapped_angle_not_the_branch():
    far = list(WARM2_PREPLACE_KDL)
    far[3] -= 2 * math.pi                 # -4.20: numerically <= cap, physically the SAME colliding 2.08
    far[4] += 2 * math.pi                 # +5.27: numerically >= min, physically -1.02
    assert not tc.in_clear_region(far, 0.65, 0.0)


def test_wrist_flip_seed_of_warm2_goal_lands_in_the_clear_region():
    seed = tc.wrist_flip_seed(WARM2_PREPLACE_KDL)
    assert seed[:3] == WARM2_PREPLACE_KDL[:3]                  # shoulder/elbow seed unchanged
    assert seed[3] == pytest.approx(2.0802 - math.pi, abs=1e-6)
    assert seed[4] == pytest.approx(1.0155) and abs(seed[5]) < 1e-3
    assert tc.in_clear_region(seed, 0.65, 0.0)
    # ... and the exact IK solution it converges to (CPU, mission URDF) is a few tenths away, same branch.
    assert max(abs(a - b) for a, b in zip(seed, WARM2_CLEAR)) < 0.6


# ---------------------------------------------------------------- controller wiring

class GuardMotion(FixtureMotion):
    """FixtureMotion that also accepts (and records) the wrist_guard kwarg."""
    def begin(self, pose, joints, straight_line=False, joint_space=False, wrist1_cap=None,
              wrist2_min=None, slow=False, goal_tol=None, near_start=False, **extra):
        super().begin(pose, joints, straight_line, joint_space, wrist1_cap, wrist2_min, slow, goal_tol,
                      near_start)
        self.extras = getattr(self, 'extras', [])
        self.extras.append(dict(extra))


def _controller(operation, monkeypatch, flag='wrist_guard', motion_cls=GuardMotion, **kwargs):
    if flag is None:
        monkeypatch.delenv(tc.ENV, raising=False)
    else:
        monkeypatch.setenv(tc.ENV, flag)
    motion = motion_cls()
    c = TransferController(motion, {'sku-box': PROFILE}, fixture_mode=True, **kwargs)
    c.start(operation, JOB)
    return c, motion


def _issue(c, phase):
    t = run_to(c, phase)
    c.step(.1, packet(c, t))                 # issue the phase's move
    return t


def test_transfer_place_moves_carry_the_wrist_guard(monkeypatch):
    c, m = _controller('transfer', monkeypatch, carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    _issue(c, 'preplace')
    assert m.extras[-1] == {'wrist_guard': (0.65, 0.0)}
    assert m.joint_space_moves[-1] is True
    # the pick moves before it never carry the guard
    assert all(e == {} for e in m.extras[:-1])
    labels = [e for e in c.evidence if e.get('source') == 'transfer_carry']
    assert labels and labels[0]['wrist_guard']['wrist1_cap'] == 0.65


def test_place_and_retreat_are_guarded_too(monkeypatch):
    c, m = _controller('transfer', monkeypatch, carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    _issue(c, 'retreat')                     # preplace, place and retreat moves have all been issued
    guarded = [e for e, js in zip(m.extras, m.joint_space_moves) if js]
    assert len(guarded) == 3 and all(e == {'wrist_guard': (0.65, 0.0)} for e in guarded)
    assert all(e == {} for e, js in zip(m.extras, m.joint_space_moves) if not js)


def test_guard_defaults_to_bench_region_when_caps_unset(monkeypatch):
    c, m = _controller('transfer', monkeypatch)
    _issue(c, 'preplace')
    assert m.extras[-1] == {'wrist_guard': (tc.DEFAULT_WRIST1_CAP, tc.DEFAULT_WRIST2_MIN)}


def test_flag_off_keeps_begin_byte_identical(monkeypatch):
    # The plain FixtureMotion has NO wrist_guard parameter: an off flag must never pass it.
    c, m = _controller('transfer', monkeypatch, flag=None, motion_cls=FixtureMotion,
                       carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    _issue(c, 'retreat')
    assert not [e for e in c.evidence if e.get('source') == 'transfer_carry']


def test_collect_is_never_guarded(monkeypatch):
    c, m = _controller('collect', monkeypatch, carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    _issue(c, 'preplace')
    assert all(e == {} for e in m.extras)


# ---------------------------------------------------------------- RosMotionPort IK branch selection

def _guard_port(monkeypatch, ik_solutions):
    """A RosMotionPort with fake transport. ``ik_solutions`` is a callable(seed)->q (ARM order) or None
    (no IK solution). Returns (port, ik_seeds, plan_goals, stopped)."""
    pytest.importorskip('rclpy')
    from concurrent.futures import Future
    from moveit_msgs.msg import RobotTrajectory
    from moveit_msgs.srv import GetPositionIK
    from trajectory_msgs.msg import JointTrajectoryPoint
    from scripts.twin.restock_manipulation.ros_adapter import RosMotionPort
    import rclpy.action

    monkeypatch.setenv('ROS_DOMAIN_ID', '88')
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    monkeypatch.delenv('ROS_STATIC_PEERS', raising=False)

    def ready(value):
        f = Future(); f.set_result(value); return f

    plan_goals, ik_seeds = [], []

    class Handle:
        accepted = True
        def __init__(self, result):
            self.result = result
        def get_result_async(self):
            return ready(self.result)
        def cancel_goal_async(self):
            return ready(None)

    def planned(goal):
        q = [c.position for c in goal.request.goal_constraints[0].joint_constraints]
        traj = RobotTrajectory()
        traj.joint_trajectory.joint_names = list(JOINTS)
        for i, pos in enumerate((list(goal.request.start_state.joint_state.position), q)):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in pos]
            p.time_from_start.sec = 10 * i
            traj.joint_trajectory.points.append(p)
        return NS(status=4, result=NS(error_code=NS(val=1), planned_trajectory=traj))

    class Client:
        def __init__(self, node, action_type, name):
            self.name = name
        def server_is_ready(self):
            return True
        def send_goal_async(self, goal):
            if self.name.endswith('move_action'):
                plan_goals.append(goal)
                return ready(Handle(planned(goal)))
            return ready(Handle(NS(status=4, result=NS(error_code=NS(val=1)))))

    def ik_call(req):
        seed = tuple(req.ik_request.robot_state.joint_state.position)
        ik_seeds.append(seed)
        q = ik_solutions(seed)
        resp = GetPositionIK.Response()
        if q is None:
            resp.error_code.val = -31
        else:
            resp.error_code.val = 1
            resp.solution.joint_state.name = list(JOINTS)
            resp.solution.joint_state.position = [float(v) for v in q]
        return ready(resp)

    monkeypatch.setattr(rclpy.action, 'ActionClient', Client)
    from builtin_interfaces.msg import Time
    node = NS(context=NS(get_domain_id=lambda: 88),
              get_parameter=lambda name: NS(value=True),
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=0, to_msg=lambda: Time())),
              create_client=lambda *args: NS(service_is_ready=lambda: True, call_async=ik_call),
              create_publisher=lambda *a, **k: NS(publish=lambda *a, **k: None))
    stopped = []
    port = RosMotionPort(node, joint_limits=[(-2 * math.pi, 2 * math.pi)] * 2 + [(0.1, math.pi)]
                         + [(-2 * math.pi, 2 * math.pi)] * 3,
                         payload_sync=lambda *a: True, emergency_stop=lambda: stopped.append(True))
    return port, ik_seeds, plan_goals, stopped


def _drive(port, n=40):
    status = None
    for _ in range(n):
        status = port.poll()
        if status in ('succeeded', 'failed'):
            break
    return status


def _goal_q(goal):
    return tuple(round(c.position, 4) for c in goal.request.goal_constraints[0].joint_constraints)


PREPLACE_POSE = (0.0507, 0.8028, 0.34, 0.5, 0.5, 0.5, 0.5)


def _kdl_like(seed):
    """warm2's KDL: from the measured carry seed it returns the colliding branch; from a seed already in
    the clear branch (the wrist-flipped re-seed) it converges to the clear solution."""
    return WARM2_CLEAR if tc.in_clear_region(seed, 0.65, 0.0) and seed[3] < 0 else WARM2_PREPLACE_KDL


def test_guard_rejects_colliding_kdl_branch_and_plans_the_flipped_clear_one(monkeypatch, capsys):
    port, seeds, goals, stopped = _guard_port(monkeypatch, _kdl_like)
    port.begin(PREPLACE_POSE, CARRY_SHAPE, joint_space=True, near_start=True, slow=True, goal_tol=0.12,
               wrist_guard=(0.65, 0.0))
    assert _drive(port) == 'succeeded'
    assert len(goals) == 1 and _goal_q(goals[0]) == WARM2_CLEAR      # planned the CLEAR goal, never 2.08
    assert seeds[0] == CARRY_SHAPE                                   # near_start: measured seed first
    assert seeds[1] == pytest.approx(tc.wrist_flip_seed(WARM2_PREPLACE_KDL))   # then the flipped twin
    out = capsys.readouterr().out
    assert 'WRIST_GUARD reject' in out and 'WRIST_GUARD accept' in out
    assert not stopped


def test_guard_off_plans_the_kdl_branch_unchanged(monkeypatch):
    port, seeds, goals, stopped = _guard_port(monkeypatch, _kdl_like)
    port.begin(PREPLACE_POSE, CARRY_SHAPE, joint_space=True, near_start=True, slow=True, goal_tol=0.12)
    assert _drive(port) == 'succeeded'
    assert _goal_q(goals[0]) == WARM2_PREPLACE_KDL and len(seeds) == 1   # legacy: first KDL answer wins


def test_guard_fails_honestly_when_no_seed_reaches_the_clear_region(monkeypatch, capsys):
    port, seeds, goals, stopped = _guard_port(monkeypatch, lambda seed: WARM2_PREPLACE_KDL)
    port.begin(PREPLACE_POSE, CARRY_SHAPE, joint_space=True, near_start=True, slow=True, goal_tol=0.12,
               wrist_guard=(0.65, 0.0))
    assert _drive(port) == 'failed'
    assert not goals                                  # never planned into the stop
    assert stopped                                    # the normal failed-move hold
    # bounded: the plain seeds plus at most WRIST_FLIP_MAX flipped re-seeds
    assert len(seeds) <= len(port._seed_configs(CARRY_SHAPE)) + 1 + WRIST_FLIP_MAX
    assert 'flips exhausted' in capsys.readouterr().out


def test_joint_goal_after_a_guarded_move_is_unguarded(monkeypatch):
    port, seeds, goals, stopped = _guard_port(monkeypatch, _kdl_like)
    port.begin(PREPLACE_POSE, CARRY_SHAPE, joint_space=True, near_start=True, wrist_guard=(0.65, 0.0))
    assert _drive(port) == 'succeeded'
    port.begin_joint_goal(CARRY_SHAPE, WARM2_CLEAR)
    assert port._wrist_guard is None
