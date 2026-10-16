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

    def begin(self, pose, joints):
        self.moves.append(pose)

    def poll(self):
        return self.outcome

    def begin_joint_goal(self, goal, measured):
        self.stows.append((goal,measured))

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
    values = dict(simulation_time=t, observations=(observation,), target=target,
                  parked=True, contact=c.held, feedback_time=t, collision_time=t,
                  joint_positions=(0.,)*6, scanned_locations=(c.source, c.destination), scan_time=t)
    values.update(kwargs)
    return Sensors(**values)


def run_to(c, phase):
    for i in range(80):
        if c.phase == phase:
            return 1 + i*.1
        contact = c.held or (c.phase == 'grasp' and c.issued)
        if c.phase == 'release' and c.issued:
            contact = False
        result = c.step(.1, packet(c, 1+i*.1, contact=contact))
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


def test_transport_stow_is_planned_before_fresh_inventory_verification():
    c,m=make_controller(transport_joints=(0.,)*6)
    t=run_to(c,'stow')
    assert c.step(.1,packet(c,t)).status=='running'
    assert m.stows==[((0.,)*6,(0.,)*6)]
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


def test_contact_move_raises_contact_signal_only_on_measured_contact():
    # Contact-terminated grasp descent (firstmate key contact-move-termination): in the 'contact'
    # phase, while the move is RUNNING, MEASURED contact raises a {'kind':'contact'} control so
    # trajectory_host can succeed the goal holding the touch pose (the gripper is against the carton
    # and cannot reach the commanded free-space pose). Without contact it never signals, keeping the
    # normal pose-settle; free-space moves (approach/lift/...) never signal contact.
    c, m = make_controller()
    t = run_to(c, 'contact')
    m.outcome = 'running'                       # the contact descent is now executing
    r = c.step(.1, packet(c, t, contact=False))     # issues the contact move
    assert r.status == 'running'
    r = c.step(.1, packet(c, t+.1, contact=False))  # running, no contact yet -> no signal
    assert r.status == 'running'
    assert not any(ctl.get('kind') == 'contact' for ctl in r.controls)
    r = c.step(.1, packet(c, t+.2, contact=True))   # measured contact -> raise the signal
    assert r.status == 'running' and c.phase == 'contact'
    assert any(ctl.get('kind') == 'contact' for ctl in r.controls)
    m.outcome = 'succeeded'                       # trajectory_host honored the signal (succeed+hold)
    c.step(.1, packet(c, t+.3, contact=True))
    assert c.phase == 'grasp'                     # advance to grasp from the achieved contact pose


def test_free_space_move_never_raises_contact_signal():
    # The approach (free-space) move keeps its exact pose-settle: it must never emit a contact signal
    # even if a (spurious) contact were measured mid-flight -- only the 'contact' phase does.
    c, m = make_controller()
    run_to(c, 'approach')
    m.outcome = 'running'
    c.step(.1, packet(c, 50., contact=False))       # issue the approach move
    r = c.step(.1, packet(c, 50.1, contact=True))
    assert r.status == 'running' and c.phase == 'approach'
    assert not any(ctl.get('kind') == 'contact' for ctl in r.controls)


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
