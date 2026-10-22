"""Host-ROS FollowJointTrajectory client for the SIM-ONLY scripted transfer (``RESTOCK_TRANSFER_MODE=scripted``
with ``RESTOCK_SCRIPTED_TRANSFER_ARM=executor``).

Spawned per arm segment by ``scripted_transfer.ExecutorArm`` under the host overlay ROS env (domain 88, the
control_ipc spawn recipe). Sends ONE pre-timed joint trajectory -- a JSON file
``{"action", "names", "points": [{"t", "q", "v"}]}`` -- to the mission's own executor (trajectory_host) and
prints exactly one ``SCRIPTED_ARM_RESULT {"status": "succeeded"|"rejected"|"aborted", "detail": ...}`` line.
``rejected`` means no goal was accepted (nothing latched in the executor); ``aborted`` means an accepted goal
failed (the executor has latched its fault + e-stop). Self-contained: stdlib + host rclpy only.
"""
import json
import sys

SERVER_WAIT_S = 30.0
ACCEPT_WAIT_S = 60.0


def _report(status, detail=''):
    print('SCRIPTED_ARM_RESULT ' + json.dumps({'status': status, 'detail': str(detail)}), flush=True)


def build_goal(spec, goal_cls, point_cls, duration_cls):
    """FollowJointTrajectory goal from the JSON spec: immediate (zero header stamp), explicit positions and
    velocities, no effort, no tolerances -- the shape trajectory_host accepts."""
    goal = goal_cls()
    goal.trajectory.joint_names = list(spec['names'])
    for point in spec['points']:
        p = point_cls()
        p.positions = [float(x) for x in point['q']]
        p.velocities = [float(x) for x in point['v']]
        sec = int(point['t'])
        p.time_from_start = duration_cls(sec=sec, nanosec=int(round((float(point['t']) - sec) * 1e9)))
        goal.trajectory.points.append(p)
    return goal


def main(argv):
    spec = json.loads(open(argv[1]).read())
    import rclpy
    path = getattr(rclpy, '__file__', '') or ''
    if 'isaacsim' in path or '/env_isaaclab/' in path:
        _report('rejected', 'wrong rclpy (Isaac internal): ' + path)
        return 3
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    rclpy.init()
    node = Node('restock_scripted_arm_client')
    try:
        client = ActionClient(node, FollowJointTrajectory, spec['action'])
        if not client.wait_for_server(timeout_sec=SERVER_WAIT_S):
            _report('rejected', 'executor action server unavailable: ' + spec['action'])
            return 3
        sent = client.send_goal_async(build_goal(spec, FollowJointTrajectory.Goal, JointTrajectoryPoint, Duration))
        rclpy.spin_until_future_complete(node, sent, timeout_sec=ACCEPT_WAIT_S)
        handle = sent.result()
        if handle is None or not handle.accepted:
            _report('rejected', 'goal not accepted by the executor')
            return 3
        done = handle.get_result_async()
        while rclpy.ok() and not done.done():          # render-paced: no wall deadline (Isaac owns the budget)
            rclpy.spin_until_future_complete(node, done, timeout_sec=5.0)
        response = done.result()
        result = response.result if response is not None else None
        if result is not None and result.error_code == result.SUCCESSFUL:
            _report('succeeded')
            return 0
        _report('aborted', getattr(result, 'error_string', '') or 'no result')
        return 4
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
