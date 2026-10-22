#!/usr/bin/env python3
"""Reproduce integration run-81's cuMotion task-space IK failure and classify it.

Beside the launched host stack, sends the EXACT MoveGroup pose-goal (task-space) request the
mission's RosMotionPort(pose_goal=True) sends -- a PositionConstraint (4mm box) + an
OrientationConstraint on gripper_pad_tcp, pipeline isaac_ros_cumotion, which cuMotion answers
with its OWN collision-aware IK ("Using task space goal"). The empty-gripper collect approach
pose (0, 0.778, 0.181) quat (0,-0.71,-0.71,0) in ur_base_link fails trajopt
INVERSE_KINEMATICS_FAILURE. We run that case plus controls to say which it is:

  A  exact pose (expect INVERSE_KINEMATICS_FAILURE)
  B  MoveIt compute_ik (KDL, avoid_collisions=False) for the exact pose -> kinematically reachable?
  C  exact POSITION, a different (down-pointing) orientation -> orientation/tool-frame culprit?
  D  trivial reachable position (0.4,0,0.5) with the EXACT side-grip quat -> quat broadly infeasible?
  E  trivial reachable position (0.4,0,0.5) identity orientation -> cuMotion task-space IK sanity

No depth is published: the world is EMPTY, so the only collision cuMotion can flag is the robot
SELF model (the XRDF) -- isolating self-collision / orientation from any environment/ESDF cause.
Stack-probe diagnosis only; no mission run.
"""
import argparse
import json
import math
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--discovery-timeout', type=float, default=150.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(a.config.read_text())

    import numpy as np
    import rclpy
    from rclpy.parameter import Parameter
    from rclpy.action import ActionClient
    from rosgraph_msgs.msg import Clock
    from builtin_interfaces.msg import Time as RTime
    from sensor_msgs.msg import JointState, Image, CameraInfo
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import StaticTransformBroadcaster
    from nvblox_msgs.srv import EsdfAndGradients
    from moveit_msgs.srv import GetPositionIK
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (Constraints, PositionConstraint, OrientationConstraint,
                                 BoundingVolume, RobotState)
    from shape_msgs.msg import SolidPrimitive
    from geometry_msgs.msg import Pose as GPose

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from twin.restock_host.config import NS, FRAME, TCP, JOINTS

    manifest = json.loads(Path(config['mobile_manifest']).read_text())['transport_positions']
    transport = [float(manifest[j]) for j in JOINTS]
    import xml.etree.ElementTree as ET
    import os, signal as _signal, subprocess
    robot_xml = ET.parse(config['urdf']).getroot()
    movable = [j.get('name') for j in robot_xml.findall('joint') if j.get('type') != 'fixed']
    all_pos = [float(manifest.get(n, 0.0) or 0.0) for n in movable]
    rsp = subprocess.Popen(
        ['ros2', 'run', 'robot_state_publisher', 'robot_state_publisher', '--ros-args',
         '-p', 'use_sim_time:=true', '-p', 'robot_description:=' + Path(config['urdf']).read_text(),
         '-r', 'joint_states:=' + NS + '/joint_states'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)

    rclpy.init()
    node = rclpy.create_node('restock_ik_diag', namespace=NS,
                             parameter_overrides=[Parameter('use_sim_time', value=True)])
    clock_pub = node.create_publisher(Clock, '/clock', 10)
    js_pub = node.create_publisher(JointState, NS + '/joint_states', 10)
    ik_cli = node.create_client(GetPositionIK, NS + '/compute_ik')
    plan_cli = ActionClient(node, MoveGroup, NS + '/move_action')
    # cuMotion's pose-goal path does a World update (ESDF query) before IK; it aborts on an empty
    # map ("World update failed"), so we must populate the ESDF with masked depth first.
    CAM, W, H, FX, FY = 'restock_camera_shelf', 200, 200, 60.0, 60.0
    CX, CY, DEPTH_M = W / 2.0, H / 2.0, 2.9
    depth_pub = node.create_publisher(Image, NS + '/camera/masked_depth', 10)
    info_pub = node.create_publisher(CameraInfo, NS + '/camera/masked_depth/camera_info', 10)
    _depth_bytes = np.full((H, W), DEPTH_M, dtype=np.float32).tobytes()
    _cam_tf = StaticTransformBroadcaster(node)
    _tf = TransformStamped()
    _tf.header.frame_id, _tf.child_frame_id = FRAME, CAM
    _tf.transform.translation.z = 2.0
    _tf.transform.rotation.x, _tf.transform.rotation.w = 1.0, 0.0
    _cam_tf.sendTransform(_tf)
    sim_ns = [int(1_000_000_000)]
    wall = [time.monotonic()]

    def publish_depth():
        stamp = RTime(sec=sim_ns[0] // 1_000_000_000, nanosec=sim_ns[0] % 1_000_000_000)
        img = Image(); img.header.stamp, img.header.frame_id = stamp, CAM
        img.height, img.width = H, W
        img.encoding, img.is_bigendian, img.step = '32FC1', 0, W * 4
        img.data = _depth_bytes; depth_pub.publish(img)
        info = CameraInfo(); info.header.stamp, info.header.frame_id = stamp, CAM
        info.height, info.width = H, W
        info.distortion_model = 'plumb_bob'; info.d = [0.0] * 5
        info.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [FX, 0.0, CX, 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
        info_pub.publish(info)

    def tick():
        nowm = time.monotonic()
        sim_ns[0] += max(0, int((nowm - wall[0]) * 1e9)); wall[0] = nowm
        m = Clock(); m.clock = RTime(sec=sim_ns[0] // 1_000_000_000, nanosec=sim_ns[0] % 1_000_000_000)
        clock_pub.publish(m)
        js = JointState(); js.header.stamp = m.clock
        js.name = list(movable); js.position = list(all_pos); js_pub.publish(js)

    def wait(fut, timeout, depth=True):
        end = time.monotonic() + timeout
        nxt = 0.0
        while rclpy.ok() and not fut.done() and time.monotonic() < end:
            if depth and time.monotonic() >= nxt:
                publish_depth(); nxt = time.monotonic() + 0.1
            tick(); rclpy.spin_once(node, timeout_sec=0.02)
        return fut.result() if fut.done() else None

    def pose_goal(pos, quat, label):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name, req.pipeline_id = 'ur_manipulator', 'isaac_ros_cumotion'
        req.allowed_planning_time = 5.0
        req.num_planning_attempts = 1
        req.max_velocity_scaling_factor = 0.04
        req.max_acceleration_scaling_factor = 0.05
        req.start_state.joint_state.name = list(JOINTS)
        req.start_state.joint_state.position = list(transport)
        pc = PositionConstraint()
        pc.header.frame_id = FRAME
        pc.link_name = TCP
        pc.weight = 1.0
        bv = BoundingVolume()
        bv.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.004, .004, .004]))
        tp = GPose()
        tp.position.x, tp.position.y, tp.position.z = map(float, pos)
        tp.orientation.w = 1.0
        bv.primitive_poses.append(tp)
        pc.constraint_region = bv
        oc = OrientationConstraint()
        oc.header.frame_id = FRAME
        oc.link_name = TCP
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = map(float, quat)
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 0.02
        oc.weight = 1.0
        c = Constraints()
        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        req.goal_constraints = [c]
        goal.planning_options.plan_only = True
        t = time.monotonic()
        gh = wait(plan_cli.send_goal_async(goal), 20.0)
        out = {'label': label, 'pos': list(pos), 'quat': list(quat)}
        if gh is None or not gh.accepted:
            out['planned'] = False; out['reason'] = 'goal not accepted'; return out
        res = wait(gh.get_result_async(), 40.0)
        if res is None:
            out['planned'] = False; out['reason'] = 'no result'; return out
        out['error_code'] = int(res.result.error_code.val)
        out['points'] = len(res.result.planned_trajectory.joint_trajectory.points)
        out['reply_s'] = round(time.monotonic() - t, 3)
        out['planned'] = (res.result.error_code.val == 1 and out['points'] > 0)
        return out

    def kdl_ik(pos, quat, label, avoid=False):
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = 'ur_manipulator'
        r.ik_link_name = TCP
        r.avoid_collisions = avoid
        r.timeout.sec = 2
        r.robot_state.joint_state.name = list(JOINTS)
        r.robot_state.joint_state.position = list(transport)
        r.pose_stamped.header.frame_id = FRAME
        r.pose_stamped.pose.position.x, r.pose_stamped.pose.position.y, r.pose_stamped.pose.position.z = map(float, pos)
        (r.pose_stamped.pose.orientation.x, r.pose_stamped.pose.orientation.y,
         r.pose_stamped.pose.orientation.z, r.pose_stamped.pose.orientation.w) = map(float, quat)
        resp = wait(ik_cli.call_async(req), 15.0)
        out = {'label': label, 'avoid_collisions': avoid}
        if resp is None:
            out['solution'] = False; out['reason'] = 'no response'; return out
        out['error_code'] = int(resp.error_code.val)
        out['solution'] = resp.error_code.val == 1
        if out['solution']:
            js = resp.solution.joint_state
            out['q'] = [round(js.position[list(js.name).index(j)], 3) for j in JOINTS]
        return out

    def joint_plan(q, label):
        """cuMotion joint-space plan_only transport -> q (the run-82 seeded c-space path)."""
        from moveit_msgs.msg import Constraints as _C, JointConstraint as _JC
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name, r.pipeline_id = 'ur_manipulator', 'isaac_ros_cumotion'
        r.allowed_planning_time = 5.0
        r.num_planning_attempts = 1
        r.max_velocity_scaling_factor = 0.04
        r.max_acceleration_scaling_factor = 0.05
        r.start_state.joint_state.name = list(JOINTS)
        r.start_state.joint_state.position = list(transport)
        con = _C()
        for n, v in zip(JOINTS, q):
            con.joint_constraints.append(_JC(joint_name=n, position=float(v),
                tolerance_above=0.05, tolerance_below=0.05, weight=1.0))
        r.goal_constraints = [con]
        goal.planning_options.plan_only = True
        t = time.monotonic()
        gh = wait(plan_cli.send_goal_async(goal), 20.0)
        out = {'label': label, 'q': [round(v, 3) for v in q]}
        if gh is None or not gh.accepted:
            out['planned'] = False; out['reason'] = 'goal not accepted'; return out
        res = wait(gh.get_result_async(), 40.0)
        if res is None:
            out['planned'] = False; out['reason'] = 'no result'; return out
        out['error_code'] = int(res.result.error_code.val)
        out['points'] = len(res.result.planned_trajectory.joint_trajectory.points)
        out['reply_s'] = round(time.monotonic() - t, 3)
        out['planned'] = (res.result.error_code.val == 1 and out['points'] > 0)
        return out

    APPROACH = (0.0, 0.778, 0.181)
    SIDE_GRIP = (0.0, -0.71, -0.71, 0.0)
    DOWN = (1.0, 0.0, 0.0, 0.0)       # 180 deg about x: TCP z-axis points down
    IDENT = (0.0, 0.0, 0.0, 1.0)
    TRIVIAL = (0.4, 0.0, 0.5)
    rec = {'approach': list(APPROACH), 'side_grip_quat': list(SIDE_GRIP), 'cases': {}}
    try:
        ik_cli.wait_for_service(timeout_sec=a.discovery_timeout)
        plan_cli.wait_for_server(timeout_sec=a.discovery_timeout)
        esdf_cli = node.create_client(EsdfAndGradients, NS + '/nvblox_node/get_esdf_and_gradient')
        esdf_cli.wait_for_service(timeout_sec=20.0)
        # Integrate masked depth so cuMotion's pre-IK World update succeeds (empty map -> abort),
        # and settle the clock/joint_states so move_group's monitored state == transport.
        t0 = time.monotonic()
        while time.monotonic() - t0 < 4.0:
            publish_depth(); tick(); rclpy.spin_once(node, timeout_sec=0.05)
        er = EsdfAndGradients.Request()
        er.update_esdf, er.use_aabb, er.frame_id = True, True, FRAME
        er.aabb_min_m.x, er.aabb_min_m.y, er.aabb_min_m.z = -0.8, -0.8, -1.0
        er.aabb_size_m.x, er.aabb_size_m.y, er.aabb_size_m.z = 1.6, 1.6, 1.2
        eresp = wait(esdf_cli.call_async(er), 20.0)
        rec['esdf_observed'] = int(np.count_nonzero(np.isfinite(np.asarray(
            eresp.esdf_and_gradients.data, dtype=np.float32)) & (np.asarray(
            eresp.esdf_and_gradients.data, dtype=np.float32) > -999.0))) if eresp else None
        rec['cases']['A_exact_pose_cumotion'] = pose_goal(APPROACH, SIDE_GRIP, 'exact approach, side-grip quat')
        rec['cases']['B_exact_pose_kdl'] = kdl_ik(APPROACH, SIDE_GRIP, 'exact approach via KDL compute_ik', avoid=False)
        rec['cases']['B2_exact_pose_kdl_collision'] = kdl_ik(APPROACH, SIDE_GRIP, 'exact approach via KDL avoid_collisions', avoid=True)
        rec['cases']['C_exact_pos_down_quat'] = pose_goal(APPROACH, DOWN, 'exact position, down orientation')
        rec['cases']['D_trivial_pos_sidegrip_quat'] = pose_goal(TRIVIAL, SIDE_GRIP, 'trivial position, side-grip quat')
        rec['cases']['E_trivial_pos_identity'] = pose_goal(TRIVIAL, IDENT, 'trivial position, identity quat')
        # run-82 seeded c-space path: plan to the KDL seed for the approach pose (raw wrapped branch
        # and the nearest-branch-normalised version). The cuMotion "self collision detected" names
        # the arm-mount pair in the stack log; normalised succeeding => wrapped-seed, not XRDF.
        b = rec['cases'].get('B_exact_pose_kdl') or {}
        if b.get('solution') and b.get('q'):
            q_raw = [float(v) for v in b['q']]
            q_norm = [((v + math.pi) % (2 * math.pi)) - math.pi for v in q_raw]
            rec['cases']['F_cspace_plan_raw_seed'] = joint_plan(q_raw, 'c-space plan to raw KDL seed')
            rec['cases']['G_cspace_plan_norm_seed'] = joint_plan(q_norm, 'c-space plan to nearest-branch seed')
    except BaseException as exc:
        import traceback
        rec['error'] = f'{type(exc).__name__}: {exc}'; rec['traceback'] = traceback.format_exc()[-1200:]
    finally:
        try: os.killpg(os.getpgid(rsp.pid), _signal.SIGTERM)
        except Exception: pass
        try: node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass
    (a.output / 'ik_diag.json').write_text(json.dumps(rec, indent=2, default=str) + '\n')
    print(json.dumps(rec, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
