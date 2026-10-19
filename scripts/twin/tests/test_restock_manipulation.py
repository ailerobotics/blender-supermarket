"""CPU-only fixtures: planner and contact here are tests, never live evidence."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.twin.restock_manipulation import (
    Observation, ProductProfile, Sensors, Target, TransferController)
from scripts.twin.restock_manipulation.perception import DepthPerception
from scripts.twin.restock_manipulation.ros_adapter import (
    JOINTS, require_sim_domain, validate_trajectory)


class FixtureMotion:
    def __init__(self):
        self.moves, self.payloads, self.stops = [], [], 0
        self.stows = []
        self.outcome, self.attach_ok = 'succeeded', True
        # fm367: when True, a 'failed' outcome is reported to the controller as a SIM-clock deadline
        # miss (retriable); default False => a collision/plan-rejection fault (fails immediately).
        self.sim_deadline_fault = False
        self.polls = 0

    def begin(self, pose, joints, straight_line=False, joint_space=False, wrist1_cap=None,
              slow=False, goal_tol=None, near_start=False):
        self.moves.append(pose)
        self.joint_space_moves = getattr(self, 'joint_space_moves', [])
        self.joint_space_moves.append(joint_space)
        self.wrist1_caps = getattr(self, 'wrist1_caps', [])
        self.wrist1_caps.append(wrist1_cap)
        # fm345: per-move PLACE slow + widened goal-gate tolerance (recorded for assertions).
        self.slows = getattr(self, 'slows', [])
        self.slows.append(slow)
        self.goal_tols = getattr(self, 'goal_tols', [])
        self.goal_tols.append(goal_tol)
        # fm347 (3): per-move IK-branch-near-the-pan flag for the PLACE moves (recorded for assertions).
        self.near_starts = getattr(self, 'near_starts', [])
        self.near_starts.append(near_start)

    def poll(self):
        self.polls += 1
        if self.outcome == 'raise':
            raise RuntimeError('planning failed (exhausted within wall cap)')
        return self.outcome

    def last_fault_was_sim_deadline(self):
        return self.sim_deadline_fault

    def begin_joint_goal(self, goal, measured, slow=False, goal_tol=None):
        self.stows.append((goal,measured,slow))

    def stop(self):
        self.stops += 1

    def update_payload(self, held, profile):
        self.payloads.append(held)
        return self.attach_ok


PROFILE = ProductProfile('sku-box', (.10, .10, .10), (1., 0., 0., 0.))
JOB = dict(job_id='job-1', sku='sku-box', quantity=1, source_bin='backroom', shelf_id='shelf-a')


def make_controller(operation='transfer', fixture_mode=True, **kwargs):
    motion = FixtureMotion()
    controller = TransferController(motion, {'sku-box': PROFILE}, fixture_mode=fixture_mode, **kwargs)
    controller.start(operation, JOB)
    return controller, motion


def packet(c, t=1., **kwargs):
    observation = Observation(f'source-{t}', 'test_fixture', t, 'sku-box', c.source,
                              (.4, 0., .2, 0., 0., 0., 1.), .99)
    target = Target(f'target-{t}', 'test_fixture', t, c.destination,
                    (.5, .2, .1, 0., 0., 0., 1.), True, (.3, .3, .4))
    import time
    values = dict(simulation_time=t, observations=(observation,), target=target,
                  parked=True, contact=c.held, feedback_time=t, collision_time=t,
                  joint_positions=(0.,)*6, scanned_locations=(c.source, c.destination), scan_time=t,
                  valid_until_steady=time.monotonic() + 100.)  # fresh acquisition deadline by default
    values.update(kwargs)
    return Sensors(**values)


def run_to(c, phase, t0=1.0):
    for i in range(80):
        if c.phase == phase:
            return t0 + i*.1
        contact = c.held or (c.phase == 'grasp' and c.issued)
        if c.phase == 'release' and c.issued:
            contact = False
        result = c.step(.1, packet(c, t0+i*.1, contact=contact))
        assert result.status != 'failed', result.reason
    pytest.fail('phase not reached')


@pytest.mark.parametrize('operation', ['collect', 'transfer'])
def test_success_needs_two_new_observations_and_source_scan(operation):
    c, m = make_controller(operation)
    t = run_to(c, 'verify')
    obs = Observation('placed-1', 'test_fixture', t, 'sku-box', c.destination,
                      (.5, .2, .2, 0., 0., 0., 1.), .99)
    assert c.step(.1, packet(c, t, observations=(obs,), scanned_locations=())).status == 'running'
    assert c.step(.1, packet(c, t+.1, observations=(replace(obs, simulation_time=t+.1),))).status == 'running'
    # A new id on the same acquisition must not double count.
    assert c.step(.1, packet(c, t+.2, observations=(replace(obs, observation_id='renamed', simulation_time=t+.1),))).status == 'running'
    result = c.step(.1, packet(c, t+.3, observations=(replace(obs, observation_id='placed-2', simulation_time=t+.3),)))
    assert result.status == 'succeeded'
    assert len(m.moves) == 6 and m.payloads == [True, False]
    assert all(e['source'] == 'test_fixture' for e in result.evidence)


def test_live_rejects_fixture_and_ground_truth():
    c, m = make_controller(fixture_mode=False, phase_timeout=.2)
    s = packet(c)
    for source in ('test_fixture', 'ground_truth'):
        r = c.step(.1, replace(s, observations=(replace(s.observations[0], source=source),)))
        assert r.status == 'running'
    assert c.step(.1, s).status == 'failed'
    assert not m.moves


def test_staged_cart_swing_routes_lift_through_tuck_pan_and_retreat_through_tuck():
    # firstmate 292: lift -> TUCK -> PAN -> preplace, and retreat -> RETREAT_TUCK -> stow, as joint-space
    # via points so the welded carton clears the robot body. TUCK/retreat_tuck fold to the STOW shape at
    # the measured shoulder-pan; PAN rotates joint0 to cart_pan_joint0.
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    tuck_shape = (-2.3, 1.0, 2.1066, 0.8, -1.2545)        # the high carry shape (joints 1-5)
    c, m = make_controller(transport_joints=stow, staged_cart_swing=True, cart_pan_joint0=-0.8605,
                           cart_tuck_shape=tuck_shape)
    t = run_to(c, 'tuck')                                 # lift transitions to tuck, not preplace
    before = len(m.stows)
    c.step(.1, packet(c, t))                              # issue the tuck joint goal
    assert m.stows[before][0] == (0.0,) + tuck_shape      # fold at measured joint0 (=0), carry shape
    assert m.stows[before][2] is True                     # tuck carries the carton -> slow fold
    c.step(.1, packet(c, t+.1))                           # poll succeeds -> pan
    assert c.phase == 'pan'
    c.step(.1, packet(c, t+.2))                           # issue the pan joint goal
    assert m.stows[-1][0] == (-0.8605,) + tuck_shape      # pan to the cart side at the carry shape
    c.step(.1, packet(c, t+.3))
    assert c.phase == 'preplace'                          # then extend to the mirrored preplace
    # Return leg: retreat -> retreat_tuck -> stow.
    t = run_to(c, 'retreat_tuck', t0=t+1.0)
    n = len(m.stows)
    c.step(.1, packet(c, t))
    assert m.stows[n][0] == (0.0,) + tuck_shape           # fold at the measured pan on the way back
    assert m.stows[n][2] is False                         # retreat_tuck is empty-handed -> full speed
    c.step(.1, packet(c, t+.1))
    assert c.phase == 'stow'


def _staged_controller():
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    tuck_shape = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
    return make_controller(transport_joints=stow, staged_cart_swing=True,
                           cart_pan_joint0=-0.8605, cart_tuck_shape=tuck_shape)


def test_staged_move_retries_then_succeeds_on_transient():
    # fm358: a staged move (run-283 pan) failed once at the plan->execute handoff after a normal plan.
    # A transient 'failed' must RETRY (fresh plan) rather than fail the stage.
    c, m = _staged_controller()
    t = run_to(c, 'pan')
    c.step(.1, packet(c, t))                 # issue the pan move
    n = len(m.stows)
    m.outcome = 'failed'
    c.step(.1, packet(c, t + .1))            # poll fails -> retry (re-issue next tick), stays in pan
    assert c.phase == 'pan'
    c.step(.1, packet(c, t + .2))            # re-plan the pan move
    assert len(m.stows) == n + 1             # a fresh begin_joint_goal was issued
    m.outcome = 'succeeded'
    c.step(.1, packet(c, t + .3))            # now it succeeds -> transition
    assert c.phase == 'preplace'


def test_staged_move_retries_on_plan_exhaust_raise():
    # run-290: a staged move's cuMotion planning can RAISE 'planning failed (exhausted within wall cap)'
    # on a stochastic bad streak; it must be caught and retried (fresh budget), not fail the mission.
    c, m = _staged_controller()
    t = run_to(c, 'pan')
    c.step(.1, packet(c, t))                 # issue the pan move
    n = len(m.stows)
    m.outcome = 'raise'                      # poll() raises the plan-exhaust RuntimeError
    c.step(.1, packet(c, t + .1))            # caught -> retry, stays in pan (not failed)
    assert c.phase == 'pan'
    c.step(.1, packet(c, t + .2))            # re-plan with a fresh budget
    assert len(m.stows) == n + 1
    m.outcome = 'succeeded'
    c.step(.1, packet(c, t + .3))
    assert c.phase == 'preplace'


def test_staged_move_retry_exhausts_then_fails():
    # The retry is BOUNDED: a persistently failing staged move still fails the stage (not an infinite loop).
    c, m = _staged_controller()
    t = run_to(c, 'pan')
    c.step(.1, packet(c, t))                 # issue
    m.outcome = 'failed'
    result = None
    for i in range(12):
        result = c.step(.1, packet(c, t + (i + 1) * .1))
        if c.phase == 'failed':
            break
    assert c.phase == 'failed'
    assert 'staged cart-swing pan move failed' in (result.reason or '')


def test_no_staged_swing_keeps_direct_lift_to_preplace():
    c, m = make_controller(transport_joints=(0.,)*6)      # staged off by default
    run_to(c, 'preplace')
    assert c.phase == 'preplace'                          # lift -> preplace directly, no tuck/pan


def test_place_moves_request_near_start_ik_branch_but_pick_moves_do_not():
    # fm347 (3): the cart PLACE pose moves (preplace/place) must seed+branch the IK to the measured PAN
    # config so shoulder_pan stays on the pan branch -- run-273 flipped j0 ~3.9 rad (pan -1.822 vs
    # preplace IK ~2.09) and the soft drive stalled. Pick moves (approach/contact/lift) must NOT.
    c, m = make_controller(transport_joints=(0.11, -2.5, 1.7, -0.7, -1.5708, 0.0),
                           staged_cart_swing=True, cart_pan_joint0=-1.822,
                           cart_tuck_shape=(-2.3, 1.0, 0.65, 0.8, -1.2545))
    t = run_to(c, 'preplace')                             # through approach/contact/grasp/lift/tuck/pan
    assert m.near_starts and not any(m.near_starts)       # every begin() so far is a pick move -> False
    c.step(.1, packet(c, t))                              # issue the preplace pose move
    assert m.near_starts[-1] is True                      # preplace stays on the pan branch
    assert m.slows[-1] is True and m.goal_tols[-1] == 0.12  # and keeps the fm345 place relaxations


def test_transport_stow_is_planned_before_fresh_inventory_verification():
    c,m=make_controller(transport_joints=(0.,)*6)
    t=run_to(c,'stow')
    assert c.step(.1,packet(c,t)).status=='running'
    assert m.stows==[((0.,)*6,(0.,)*6,False)]   # stow is empty-handed -> full speed (slow=False)
    assert c.step(.1,packet(c,t+.1)).status=='running'
    assert c.phase=='verify' and not c.verify_ids


def test_stow_executor_success_cannot_override_wrong_measured_joints():
    c,m=make_controller(transport_joints=(.2,)*6)
    t=run_to(c,'stow')
    c.step(.1,packet(c,t))
    result=c.step(.1,packet(c,t+.1))
    assert result.status=='failed'
    assert 'joint feedback' in result.reason


@pytest.mark.parametrize('change,reason', [
    ({'parked': False}, 'base moving'),
    ({'feedback_time': 0}, 'stale'),
    ({'collision_time': 0}, 'stale'),
    ({'joint_positions': (float('nan'),)*6}, 'feedback'),
])
def test_feedback_gates(change, reason):
    c, m = make_controller()
    result = c.step(.1, packet(c, **change))
    assert result.status == 'failed' and reason in result.reason and m.stops == 1


@pytest.mark.parametrize('field,value', [('clear', False), ('free_dimensions', (.05,.05,.05)),
                                        ('pose', (2.,0.,0.,0.,0.,0.,1.))])
def test_target_refusal(field, value):
    c, m = make_controller()
    s = packet(c)
    result = c.step(.1, replace(s, target=replace(s.target, **{field: value})))
    assert result.status == 'failed' and not m.moves


def test_lost_grasp_stops_without_release():
    c, m = make_controller()
    t = run_to(c, 'lift')
    r = c.step(.1, packet(c, t, contact=False))
    assert r.reason == 'lost grasp'
    assert r.controls[-1] == {'kind': 'gripper', 'enabled': True}
    with pytest.raises(ValueError, match='recovery'):
        c.start('transfer', JOB)


def _fresh():
    import time
    return time.monotonic() + 100.   # a future steady-clock acquisition deadline (require_fresh passes)


def test_contact_descent_enables_suction_then_terminates_on_latch():
    # Suction-grasp choreography + contact-terminated move (firstmate keys). In the 'contact' phase:
    # (1) issue the descent move; (2) enable suction ONCE a fresh valid_until_steady is present (the
    # grip command carries an acquisition deadline); (3) s.contact is the grip LATCH (needs suction
    # on), so once it latches, raise {'kind':'contact'} to terminate the move holding the touch pose;
    # (4) advance to grasp. Free-space moves never enable suction or signal contact.
    c, m = make_controller()
    t = run_to(c, 'contact')
    m.outcome = 'running'
    r = c.step(.1, packet(c, t, contact=False, valid_until_steady=_fresh()))        # issue the move
    assert r.controls[0]['kind'] == 'move'
    r = c.step(.1, packet(c, t+.1, contact=False, valid_until_steady=_fresh()))     # suction enabled
    assert {'kind': 'gripper', 'enabled': True} in r.controls
    r = c.step(.1, packet(c, t+.2, contact=False, valid_until_steady=_fresh()))     # running, no latch
    assert r.status == 'running' and not any(x.get('kind') == 'contact' for x in r.controls)
    r = c.step(.1, packet(c, t+.3, contact=True, valid_until_steady=_fresh()))      # latch -> terminate
    assert r.status == 'running' and c.phase == 'contact'
    assert any(x.get('kind') == 'contact' for x in r.controls)
    m.outcome = 'succeeded'                                                          # host succeeded+held
    c.step(.1, packet(c, t+.4, contact=True, valid_until_steady=_fresh()))
    assert c.phase == 'grasp'


def test_contact_suction_waits_for_a_fresh_acquisition_deadline():
    # The grip command must carry an acquisition deadline; under the sensing gate the first held ticks
    # may lack one (run-137). Do not emit the gripper until valid_until_steady is present.
    c, m = make_controller()
    t = run_to(c, 'contact')
    m.outcome = 'running'
    c.step(.1, packet(c, t, contact=False, valid_until_steady=_fresh()))            # issue the move
    r = c.step(.1, packet(c, t+.1, contact=False, valid_until_steady=None))         # no deadline yet
    assert not any(x.get('kind') == 'gripper' for x in r.controls)                  # suction withheld


def test_free_space_move_never_enables_suction_or_signals_contact():
    # The approach (free-space) move keeps its exact pose-settle: never enables suction, never signals
    # contact, even if a (spurious) contact is measured mid-flight.
    c, m = make_controller()
    run_to(c, 'approach')
    m.outcome = 'running'
    c.step(.1, packet(c, 50., contact=False, valid_until_steady=_fresh()))          # issue approach
    r = c.step(.1, packet(c, 50.1, contact=True, valid_until_steady=_fresh()))
    assert r.status == 'running' and c.phase == 'approach'
    assert not any(x.get('kind') in ('contact', 'gripper') for x in r.controls)


def test_planning_failure_and_missing_attachment():
    c, m = make_controller()
    c.step(.1, packet(c))
    m.outcome = 'failed'
    assert c.step(.1, packet(c, 1.1)).status == 'failed'
    c, m = make_controller()
    t = run_to(c, 'grasp')
    c.step(.1, packet(c, t))
    m.attach_ok = False
    assert 'attachment' in c.step(.1, packet(c, t+.1, contact=True)).reason


def test_failed_verification_never_credits_stock():
    c, m = make_controller(phase_timeout=2)
    t = run_to(c, 'verify')
    r = c.step(2.1, packet(c, t))
    assert r.status == 'failed' and 'timeout' in r.reason
    assert len(r.evidence) == 1


def test_ambiguous_duplicate_and_unreachable_source_wait_without_motion():
    for mutate in (lambda o: (replace(o, confidence=.2),), lambda o: (o,o),
                   lambda o: (replace(o, pose=(3.,0.,0.,0.,0.,0.,1.)),)):
        c, m = make_controller()
        s = packet(c)
        assert c.step(.1, replace(s, observations=mutate(s.observations[0]))).status == 'running'
        assert not m.moves


def test_target_move_tolerated_during_locked_reach_but_revalidated_at_preplace():
    # Observe-then-execute lock (key approach-sensing-occlusion): the placement target is NOT
    # re-validated from the fixed camera during the locked reach (approach/contact/lift) -- the
    # reaching arm occludes it, so a changed/absent target there is expected, not a fault. It IS
    # re-validated again at preplace, once lift has cleared the view.
    c, m = make_controller()
    c.step(.1, packet(c))
    assert c.phase == 'approach'
    moved = replace(packet(c, 1.1).target, pose=(.7, .2, .1, 0., 0., 0., 1.))
    assert c.step(.1, replace(packet(c, 1.1), target=moved)).status == 'running'  # tolerated while locked
    # Fresh controller (clean clock): the target IS re-validated again at preplace, after lift clears.
    c2, _ = make_controller()
    t = run_to(c2, 'preplace')
    s = replace(packet(c2, t), target=replace(packet(c2, t).target, pose=(.7, .2, .1, 0., 0., 0., 1.)))
    assert c2.step(.1, s).reason == 'target clearance lost'                        # re-validated after lift


def test_replayed_sensor_time_and_invalid_configuration():
    c, _ = make_controller()
    c.step(.1, packet(c, 2))
    assert c.step(.1, packet(c, 1)).status == 'failed'
    with pytest.raises(ValueError):
        c.start('transfer', dict(JOB, quantity=2))


def trajectory(points=None, names=JOINTS):
    return NS(joint_names=names, points=points or [
        NS(positions=[0.]*6, velocities=[], time_from_start=NS(sec=0,nanosec=0)),
        NS(positions=[.1]*6, velocities=[0.]*6, time_from_start=NS(sec=1,nanosec=0))])


def test_trajectory_validates_all_waypoints():
    traj = trajectory()
    validate_trajectory(traj, [0.]*6, [(-1,1)]*6)
    traj.points[-1].positions[2] = 2
    with pytest.raises(ValueError, match='limits'):
        validate_trajectory(traj, [0.]*6, [(-1,1)]*6)


@pytest.mark.parametrize('mutation', [
    lambda t: setattr(t, 'joint_names', ['wrong']*6),
    lambda t: setattr(t.points[0], 'positions', [.2]*6),
    lambda t: setattr(t.points[-1], 'velocities', [.1]*6),
    lambda t: setattr(t.points[-1], 'time_from_start', NS(sec=0,nanosec=0)),
    lambda t: setattr(t.points[-1], 'positions', [float('nan')]*6),
])
def test_bad_trajectory(mutation):
    t = trajectory()
    mutation(t)
    with pytest.raises(ValueError):
        validate_trajectory(t, [0.]*6, [(-1,1)]*6)


def test_ros_isolation(monkeypatch):
    for domain in ('0', '42', '83', ''):
        monkeypatch.setenv('ROS_DOMAIN_ID', domain)
        with pytest.raises(RuntimeError):
            require_sim_domain()
    monkeypatch.setenv('ROS_DOMAIN_ID', '88')
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    monkeypatch.delenv('ROS_STATIC_PEERS', raising=False)
    require_sim_domain()


class FixturePoseBridge:
    """Test pinhole math only; separate explicit upstream test below."""
    CameraIntrinsics = staticmethod(lambda *k: k)

    @staticmethod
    def backproject_mask(mask, depth, k):
        v, u = np.nonzero(mask)
        fx, fy, cx, cy = k
        z = depth[v,u]
        return np.column_stack(((u-cx)*z/fx, (v-cy)*z/fy, z))

    @staticmethod
    def estimate_pose(points, min_points):
        return NS(translation=points.mean(0), normal=np.array([0.,0.,1.]))


def depth_args():
    return dict(depth_m=np.full((10,10), .5), mask=np.ones((10,10), bool),
                intrinsics=(100.,100.,5.,5.), base_from_camera=np.eye(4),
                classification=([('sku-box', .8)], 'OK', .8, .1),
                observation_id='rgbd-1', simulation_time=1., location='rollwagen',
                rgb_time=1., depth_time=1., transform_time=1.)


def test_depth_pose_and_uncertainty():
    p = DepthPerception(FixturePoseBridge())
    args = depth_args()
    obs = p.observe(**args)
    assert obs.source == 'sensor' and obs.pose[2] == pytest.approx(.5)
    for change in ({'classification':([], 'UNKNOWN', .2, .01)}, {'depth_time':.8},
                   {'depth_m':np.zeros((10,10))}, {'base_from_camera':np.zeros((4,4))}):
        with pytest.raises(ValueError):
            p.observe(**dict(args, **change))


def test_target_depth_coverage_and_occupancy():
    p = DepthPerception(FixturePoseBridge())
    args = depth_args()
    for k in ('classification', 'rgb_time', 'depth_time', 'transform_time'):
        args.pop(k)
    args.update(support_z=.5, free_dimensions=(.3,.3,.4))
    assert p.target(**args).clear
    depth = args['depth_m'].copy()
    depth[0,0] = .6
    assert not p.target(**dict(args, depth_m=depth)).clear
    depth[:] = float('nan')
    with pytest.raises(ValueError):
        p.target(**dict(args, depth_m=depth))


def test_actual_ros_message_request_and_async_progression(monkeypatch):
    """Uses installed message types but fake transport: no ROS graph or GPU."""
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
        f = Future()
        f.set_result(value)
        return f

    class Handle:
        accepted = True
        cancelled = False
        def __init__(self, result):
            self.result = result
        def get_result_async(self):
            return ready(self.result)
        def cancel_goal_async(self):
            self.cancelled = True
            return ready(None)

    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = list(JOINTS)
    for index in (0, 1):
        point = JointTrajectoryPoint()
        point.positions = [float(index)*.1]*6
        point.time_from_start.sec = index
        trajectory.joint_trajectory.points.append(point)
    plan_result = NS(status=4, result=NS(error_code=NS(val=1), planned_trajectory=trajectory))
    execute_result = NS(status=4, result=NS(error_code=NS(val=1)))
    clients = []

    class Client:
        def __init__(self, node, action_type, name):
            self.name, self.goals = name, []
            clients.append(self)
        def server_is_ready(self):
            return True
        def send_goal_async(self, goal):
            self.goals.append(goal)
            return ready(Handle(plan_result if self.name.endswith('move_action') else execute_result))

    monkeypatch.setattr(rclpy.action, 'ActionClient', Client)
    response = GetPositionIK.Response()
    response.error_code.val = 1
    response.solution.joint_state.name = list(JOINTS)
    response.solution.joint_state.position = [.1]*6
    requests = []
    def request(req):
        requests.append(req)
        return ready(response)
    from builtin_interfaces.msg import Time
    node = NS(context=NS(get_domain_id=lambda:88),
              get_parameter=lambda name:NS(value=True),
              get_clock=lambda:NS(now=lambda:NS(to_msg=lambda:Time())),
              create_client=lambda *args:NS(service_is_ready=lambda:True, call_async=request))
    stopped = []
    port = RosMotionPort(node, joint_limits=[(-1,1)]*6,
                         payload_sync=lambda *args:False, emergency_stop=lambda:stopped.append(True))
    port.begin((.4,0.,.2,1.,0.,0.,0.), (0.,)*6)
    for _ in range(6):
        status = port.poll()
        if status == 'succeeded':
            break
    assert status == 'succeeded'
    assert requests[0].ik_request.avoid_collisions
    assert clients[0].goals[0].planning_options.plan_only
    assert list(clients[0].goals[0].request.start_state.joint_state.position) == [0.]*6
    assert clients[0].goals[0].request.pipeline_id == 'isaac_ros_cumotion'
    assert len(clients[1].goals) == 1
    assert not port.update_payload(True, PROFILE)
    port.stop()
    assert stopped


def test_ensemble_calls_rgb_models_and_rejects_unknown():
    from scripts.twin.restock_manipulation.perception import EnsembleObservationPipeline
    bounds = np.array([[0,0,10,10]])
    detector = NS(predict=lambda *args, **kwargs:[NS(boxes=[NS(conf=.99, xyxy=bounds)])])
    classification = lambda crop: ([('sku-box', .8)], 'OK', .8, .1)
    pipeline = EnsembleObservationPipeline(detector, classification, DepthPerception(FixturePoseBridge()),
                                          lambda rgb, depth, bounds:np.ones(depth.shape, bool),
                                          lambda bounds:'rollwagen')
    args = depth_args()
    for key in ('classification','observation_id','location','mask'):
        args.pop(key)
    observations, rejected = pipeline.observe(np.zeros((10,10,3), np.uint8), frame_id='frame-1', **args)
    assert len(observations) == 1 and not rejected
    pipeline.classify_crop = lambda crop:([], 'UNKNOWN', .2, .01)
    observations, rejected = pipeline.observe(np.zeros((10,10,3), np.uint8), frame_id='frame-2', **args)
    assert not observations and 'uncertain SKU' in rejected[0]['reason']


def test_source_pose_change_tolerated_during_locked_approach():
    # Observe-then-execute lock: once the approach is planned the reaching arm occludes/degrades the
    # fixed source camera, so a changed or absent source observation during approach/contact is
    # EXPECTED occlusion, not a fault. The grasp proceeds; a genuine miss is caught at contact and by
    # the executor's joint-tracking/heartbeat e-stops, not by per-tick fixed-camera re-validation.
    c, _ = make_controller()
    c.step(.1, packet(c))
    assert c.phase == 'approach'
    s = packet(c, 1.1)
    moved = replace(s.observations[0], pose=(.45,0.,.2,0.,0.,0.,1.))
    assert c.step(.1, replace(s, observations=(moved,))).status == 'running'


def test_ros_packet_rejects_string_booleans_and_scan_coverage():
    import json
    from dataclasses import asdict
    from scripts.twin.restock_manipulation.ros_adapter import sensors_from_json
    c, _ = make_controller()
    data = asdict(packet(c))
    parsed = sensors_from_json(json.dumps(data))
    assert parsed.observations[0].sku == 'sku-box'
    for key, value in (('parked', 'false'), ('contact', 1), ('scanned_locations', 'rollwagen')):
        with pytest.raises(ValueError):
            sensors_from_json(json.dumps(dict(data, **{key:value})))
