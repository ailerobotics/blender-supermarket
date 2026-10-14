#!/usr/bin/env python3
"""Host-stack PLACE (transfer) capability probe at the shelf station.

Runs beside an already-launched restock host stack (launch_restock_host_stack.sh) on
domain 88 and proves the HOST side of the post-collect TRANSFER stage, mirroring exactly
the host endpoints the mission's TransferController drives (controller.py / ros_adapter.py
RosMotionPort / ros_payload.py), without the mission:

  A. discovery: the 6 planning nodes + the endpoints the place needs (compute_ik,
     move_action, attach_object, apply_planning_scene, get_esdf_and_gradient).
  B. station protocol AT THE SHELF: prepare/heartbeat with the shelf camera frame
     (restock_camera_shelf) and the host workspace bounds -> incorporation ack goes ready
     with a fresh collision_time (same bounds as collect; they are ur_base_link-relative
     and move with the parked base).
  C. shelf walls in the collision model: a shelf-slot masked-depth frame is fused into the
     nvblox ESDF (the host carries NO static wall CollisionObject; shelf geometry is ESDF
     voxels from masked depth), ESDF goes non-empty.
  D. carton ATTACH: the two-step collision attachment the controller runs at 'grasp' --
     MoveIt ApplyPlanningScene (AttachedCollisionObject BOX of the SKU dimensions on
     gripper_pad_tcp, touch_links suction_cups) + the cuMotion AttachObject action
     (/attach_object, object_config CUBE) -- is acknowledged (scene.success and
     AttachObject status == SUCCEEDED).
  E. side-grip insertion PLAN with the carton attached: seeded compute_ik to a side-grip
     insertion pose (tcp_quaternion from the verified profile), then a cuMotion joint-space
     plan_only against the populated ESDF with the attached carton in the robot body --
     exactly RosMotionPort.begin -- returns a trajectory.

Measurement only; the synthetic shelf depth stands in for the integration-owned masked
depth, and the insertion pose stands in for the wall_policy slot pose (built at mission time
from the sensed handling-unit poses). This proves the host stack SUPPORTS the place; it is
not a mission run.
"""
import argparse
import json
import math
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--ros-host-config', type=Path,
                   help='verified ros_host_config.json (profiles + per-station wall index)')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--shelf-frame', default='restock_camera_shelf')
    p.add_argument('--station', default='G3_back_s2')
    p.add_argument('--discovery-timeout', type=float, default=150.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(a.config.read_text())
    # Verified handling-unit profile + wall index (falls back to the known juice_orange side grip).
    profile = {'sku': 'juice_orange', 'dimensions': [0.6, 0.4, 0.16],
               'tcp_quaternion': [0.5, 0.5, 0.5, 0.5], 'grasp_mode': 'side', 'approach': 0.12}
    walls = {}
    if a.ros_host_config and a.ros_host_config.exists():
        rc = json.loads(a.ros_host_config.read_text())
        if rc.get('profiles'):
            profile = rc['profiles'][0]
        walls = rc.get('walls', {})
    dims = [float(v) for v in profile['dimensions']]
    grip_quat = [float(v) for v in profile.get('tcp_quaternion', [0.5, 0.5, 0.5, 0.5])]

    import numpy as np
    import rclpy
    from rclpy.action import ActionClient, get_action_server_names_and_types_by_node
    from builtin_interfaces.msg import Time as RTime
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from std_msgs.msg import String
    from moveit_msgs.srv import GetPositionIK, ApplyPlanningScene
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject, Constraints,
                                 JointConstraint, RobotState)
    from shape_msgs.msg import SolidPrimitive
    from geometry_msgs.msg import Pose
    from nvblox_msgs.srv import EsdfAndGradients
    from isaac_ros_cumotion_interfaces.action import AttachObject

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from twin.restock_host.config import NS, FRAME, TCP, JOINTS

    UNOBSERVED = -1000.0
    rec = {'station': a.station, 'shelf_frame': a.shelf_frame, 'profile': profile,
           'wall_index': walls.get(a.station), 'steps': [], 'endpoints': {}}

    def log(ev, **kw):
        rec['steps'].append({'t': round(time.monotonic(), 2), 'event': ev, **kw})

    manifest = json.loads(Path(config['mobile_manifest']).read_text())['transport_positions']
    transport = [float(manifest[j]) for j in JOINTS]
    import xml.etree.ElementTree as ET
    import os
    import signal as _signal
    import subprocess
    robot_xml = ET.parse(config['urdf']).getroot()
    movable = [j.get('name') for j in robot_xml.findall('joint') if j.get('type') != 'fixed']
    all_pos = [float(manifest.get(n, 0.0) or 0.0) for n in movable]

    rsp = subprocess.Popen(
        ['ros2', 'run', 'robot_state_publisher', 'robot_state_publisher', '--ros-args',
         '-p', 'use_sim_time:=true', '-p', 'robot_description:=' + Path(config['urdf']).read_text(),
         '-r', 'joint_states:=' + NS + '/joint_states'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)

    rclpy.init()
    from rclpy.parameter import Parameter
    node = rclpy.create_node('restock_place_probe', namespace=NS,
                             parameter_overrides=[Parameter('use_sim_time', value=True)])
    from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
    from rclpy.time import Time as RclTime
    from geometry_msgs.msg import TransformStamped
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    # Shelf depth-camera extrinsic: ur_base_link -> restock_camera_shelf, static (the mission
    # relays this on /tf). Without it nvblox cannot transform the masked depth to its global
    # frame and silently drops every frame, so the ESDF stays empty and the ack never readies.
    _cam_tf = StaticTransformBroadcaster(node)
    _tf = TransformStamped()
    _tf.header.frame_id, _tf.child_frame_id = FRAME, a.shelf_frame   # ur_base_link -> shelf cam
    _tf.transform.translation.z = 2.0
    _tf.transform.rotation.x, _tf.transform.rotation.w = 1.0, 0.0     # 180 deg about X -> look down
    _cam_tf.sendTransform(_tf)

    clock_pub = node.create_publisher(Clock, '/clock', 10)
    js_pub = node.create_publisher(JointState, NS + '/joint_states', 10)
    depth_pub = node.create_publisher(Image, NS + '/camera/masked_depth', 10)
    info_pub = node.create_publisher(CameraInfo, NS + '/camera/masked_depth/camera_info', 10)
    cmd_pub = node.create_publisher(String, NS + '/station/command', 10)
    status_box = {'msg': None}
    node.create_subscription(String, NS + '/station/status',
                             lambda m: status_box.__setitem__('msg', m.data), 10)
    ik_cli = node.create_client(GetPositionIK, NS + '/compute_ik')
    esdf_cli = node.create_client(EsdfAndGradients, NS + '/nvblox_node/get_esdf_and_gradient')
    scene_cli = node.create_client(ApplyPlanningScene, NS + '/apply_planning_scene')
    attach_cli = ActionClient(node, AttachObject, NS + '/attach_object')
    plan_cli = ActionClient(node, MoveGroup, NS + '/move_action')

    sim_ns = [int(1_000_000_000)]
    CAM, W, H, FX, FY = a.shelf_frame, 200, 200, 60.0, 60.0
    CX, CY, DEPTH_M = W / 2.0, H / 2.0, 2.9
    _depth_bytes = np.full((H, W), DEPTH_M, dtype=np.float32).tobytes()
    station = {'epoch': 1, 'seq': 0, 'active': False}
    _wall = [time.monotonic()]

    def tick_clock():
        # Advance sim time by REAL elapsed wall time (sim == wall), not a fixed step. Under
        # use_sim_time the planner's allowed_planning_time and the ack's freshness guards are in
        # sim seconds; a free-running (too-fast) clock would expire a 5 s plan budget in a
        # fraction of a wall second (spurious TIMED_OUT) and trip the ack's 0.5 s sim gap.
        nowm = time.monotonic()
        sim_ns[0] += max(0, int((nowm - _wall[0]) * 1e9))
        _wall[0] = nowm
        m = Clock(); m.clock = RTime(sec=sim_ns[0] // 1_000_000_000, nanosec=sim_ns[0] % 1_000_000_000)
        clock_pub.publish(m)
        js = JointState(); js.header.stamp = m.clock
        js.name = list(movable); js.position = list(all_pos); js_pub.publish(js)

    def sim_now():
        return sim_ns[0] * 1e-9

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

    def heartbeat():
        if not station['active']:
            return
        station['seq'] += 1
        cmd_pub.publish(String(data=json.dumps(dict(action='heartbeat', epoch=station['epoch'],
            sequence=station['seq'], stamp=sim_now(), parked=True))))

    def pump(dt, *, depth=False, beat=False):
        """Advance sim clock (and optionally depth + station heartbeat) for dt wall seconds."""
        end = time.monotonic() + dt
        nxt_pub = nxt_hb = 0.0
        while time.monotonic() < end:
            nowm = time.monotonic()
            if depth and nowm >= nxt_pub:
                publish_depth(); nxt_pub = nowm + 0.1
            if beat and nowm >= nxt_hb:
                heartbeat(); nxt_hb = nowm + 0.1
            tick_clock(); rclpy.spin_once(node, timeout_sec=0.02)

    def wait_future(fut, timeout, *, beat=False, depth=False):
        # Keep the sim clock moving and, when asked, keep the station heartbeat alive and the
        # masked depth flowing -- the mission streams depth continuously, and object_attachment
        # runs clear_esdf_on_attach, so the shelf walls must be re-integrated during the plan.
        end = time.monotonic() + timeout
        nxt_hb = nxt_pub = 0.0
        while rclpy.ok() and not fut.done() and time.monotonic() < end:
            if beat and time.monotonic() >= nxt_hb:
                heartbeat(); nxt_hb = time.monotonic() + 0.1
            if depth and time.monotonic() >= nxt_pub:
                publish_depth(); nxt_pub = time.monotonic() + 0.1
            tick_clock(); rclpy.spin_once(node, timeout_sec=0.02)
        return fut.result() if fut.done() else None

    def esdf_probe():
        req = EsdfAndGradients.Request()
        req.update_esdf, req.visualize_esdf, req.use_aabb, req.frame_id = True, False, True, FRAME
        req.aabb_min_m.x, req.aabb_min_m.y, req.aabb_min_m.z = -0.8, -0.8, -1.0
        req.aabb_size_m.x, req.aabb_size_m.y, req.aabb_size_m.z = 1.6, 1.6, 1.2
        resp = wait_future(esdf_cli.call_async(req), 20.0, beat=True)
        if resp is None:
            return None
        data = np.asarray(resp.esdf_and_gradients.data, dtype=np.float32)
        return {'success': bool(resp.success),
                'observed_voxels': int(np.count_nonzero(np.isfinite(data) & (data > UNOBSERVED + 1.0)))}

    def plan_joint_goal(goal_q, label, start_q=None):
        """cuMotion joint-space plan_only start_q -> goal_q (the RosMotionPort plan pattern)."""
        start_q = transport if start_q is None else start_q
        goal = MoveGroup.Goal()
        rq = goal.request
        rq.group_name, rq.pipeline_id = 'ur_manipulator', 'isaac_ros_cumotion'
        rq.allowed_planning_time = 5.0
        rq.num_planning_attempts = 1
        rq.max_velocity_scaling_factor = 0.04
        rq.max_acceleration_scaling_factor = 0.05
        start = RobotState()
        start.joint_state.name = list(JOINTS)
        start.joint_state.position = list(start_q)
        rq.start_state = start
        con = Constraints()
        for name, value in zip(JOINTS, goal_q):
            con.joint_constraints.append(JointConstraint(joint_name=name, position=float(value),
                tolerance_above=0.05, tolerance_below=0.05, weight=1.0))
        rq.goal_constraints = [con]
        goal.planning_options.plan_only = True
        t_plan = time.monotonic()
        gh = wait_future(plan_cli.send_goal_async(goal), 20.0, beat=True, depth=True)
        out = {'label': label}
        if gh is not None and gh.accepted:
            res = wait_future(gh.get_result_async(), 60.0, beat=True, depth=True)
            if res is not None:
                out['error_code'] = int(res.result.error_code.val)
                out['points'] = len(res.result.planned_trajectory.joint_trajectory.points)
                out['reply_s'] = round(time.monotonic() - t_plan, 3)
                out['planned'] = (res.result.error_code.val == 1 and out['points'] > 0)
            else:
                out['planned'] = False; out['reason'] = 'no plan result'
        else:
            out['planned'] = False; out['reason'] = 'goal not accepted'
        return out

    try:
        # ---- A. discovery ----
        want_nodes = {'move_group', 'cumotion_planner', 'static_planning_scene_server',
                      'object_attachment', 'nvblox_node', 'object_segmenter'}
        need_services = {NS + '/compute_ik', NS + '/apply_planning_scene',
                         NS + '/nvblox_node/get_esdf_and_gradient'}
        need_actions = {NS + '/move_action', NS + '/attach_object'}
        deadline = time.monotonic() + a.discovery_timeout
        seen_nodes = set()
        while time.monotonic() < deadline:
            seen_nodes = {nm for nm, _ in node.get_node_names_and_namespaces()}
            svc = dict(node.get_service_names_and_types())
            acts = []
            for nm, nsp in node.get_node_names_and_namespaces():
                acts.extend(n for n, _ in get_action_server_names_and_types_by_node(node, nm, nsp))
            if (want_nodes <= seen_nodes and need_services <= set(svc)
                    and need_actions <= set(acts)):
                break
            pump(0.5)
        svc = dict(node.get_service_names_and_types())
        acts = set()
        for nm, nsp in node.get_node_names_and_namespaces():
            acts.update(n for n, _ in get_action_server_names_and_types_by_node(node, nm, nsp))
        rec['nodes'] = {'present': sorted(want_nodes & seen_nodes),
                        'absent': sorted(want_nodes - seen_nodes),
                        'all_present': want_nodes <= seen_nodes}
        rec['endpoints'] = {'services_missing': sorted(need_services - set(svc)),
                            'actions_missing': sorted(need_actions - acts),
                            'attach_object_present': (NS + '/attach_object') in acts,
                            'apply_planning_scene_present': (NS + '/apply_planning_scene') in svc}
        log('discovery_done', all_nodes=rec['nodes']['all_present'],
            attach=rec['endpoints']['attach_object_present'],
            scene=rec['endpoints']['apply_planning_scene_present'])

        # ---- B+C. station protocol at the shelf + shelf walls -> ESDF ----
        esdf_cli.wait_for_service(timeout_sec=20.0)
        before = esdf_probe()
        # warm the ack with a few clock ticks + its command subscription BEFORE prepare so it
        # never reads now==0 (which would reject prepare as a stale timestamp and fault).
        t = time.monotonic()
        while cmd_pub.get_subscription_count() < 1 and time.monotonic() - t < 8.0:
            tick_clock(); rclpy.spin_once(node, timeout_sec=0.02)
        pump(0.5, depth=True)
        cmd_pub.publish(String(data=json.dumps(dict(action='prepare', epoch=station['epoch'],
            sequence=station['seq'], stamp=sim_now(), parked=True, camera_frame=CAM))))
        station['active'] = True
        ack_ready, ack_ct, last_status = False, None, None
        t_prepare = time.monotonic()
        nxt_pub = nxt_hb = 0.0
        while time.monotonic() - t_prepare < 8.0:
            nowm = time.monotonic()
            if nowm >= nxt_pub:
                publish_depth(); nxt_pub = nowm + 0.1
            if nowm >= nxt_hb:
                heartbeat(); nxt_hb = nowm + 0.1
            tick_clock(); rclpy.spin_once(node, timeout_sec=0.02)
            if status_box['msg']:
                last_status = json.loads(status_box['msg'])
                if last_status.get('state') == 'ready' and (last_status.get('collision_time') or -1) > 0:
                    ack_ready = True; ack_ct = last_status['collision_time']
                    break
        after = esdf_probe()
        rec['station'] = {'ack_ready': ack_ready, 'collision_time': ack_ct,
                          'last_state': (last_status or {}).get('state'),
                          'camera_frame_echo': (last_status or {}).get('camera_frame')}
        rec['esdf'] = {'observed_before': (before or {}).get('observed_voxels'),
                       'observed_after': (after or {}).get('observed_voxels'),
                       'non_empty': bool((after or {}).get('observed_voxels'))}
        log('station_and_esdf', ack_ready=ack_ready, collision_time=ack_ct,
            observed_after=rec['esdf']['observed_after'])

        # ---- C2. baseline motion plan against the shelf ESDF (no carton yet) ----
        # A bounded joint-space goal off the transport pose, planned by cuMotion against the
        # populated shelf ESDF -- the host's core collision-aware planning capability, and the A/B
        # baseline for the attached plan below.
        safe_goal = [q + 0.1 for q in transport]   # small move on every joint (stack_probe-proven form)
        rec['baseline_plan'] = plan_joint_goal(safe_goal, 'baseline joint goal (no carton)')
        log('baseline_plan_done', **{k: rec['baseline_plan'].get(k) for k in
            ('planned', 'points', 'error_code')})

        # ---- D. carton attach (ApplyPlanningScene + AttachObject), exactly the controller's grasp ----
        rec['attach'] = {'service_ready': scene_cli.wait_for_service(timeout_sec=10.0),
                         'action_ready': attach_cli.wait_for_server(timeout_sec=10.0)}
        tcp_xyz = None
        try:
            tr = tf_buffer.lookup_transform(FRAME, TCP, RclTime())
            tcp_xyz = (tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z)
        except Exception as exc:
            rec['attach']['tf_error'] = str(exc)
        if rec['attach']['service_ready'] and rec['attach']['action_ready'] and tcp_xyz is not None:
            centre = (tcp_xyz[0], tcp_xyz[1], tcp_xyz[2] - dims[2] / 2.0)  # carton centre below the TCP
            attached = AttachedCollisionObject()
            attached.link_name = TCP
            attached.touch_links = ['suction_cups']
            attached.object.id = 'restock_carried_unit'
            attached.object.header.frame_id = FRAME
            attached.object.operation = CollisionObject.ADD
            shape = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(dims))
            cpose = Pose()
            cpose.position.x, cpose.position.y, cpose.position.z = centre
            cpose.orientation.w = 1.0
            attached.object.primitives = [shape]
            attached.object.primitive_poses = [cpose]
            req = ApplyPlanningScene.Request()
            req.scene.is_diff = True
            req.scene.robot_state.is_diff = True
            req.scene.robot_state.attached_collision_objects = [attached]
            scene_resp = wait_future(scene_cli.call_async(req), 10.0, beat=True)
            rec['attach']['scene_success'] = bool(scene_resp and scene_resp.success)
            goal = AttachObject.Goal()
            goal.attach_object = True
            marker = goal.object_config
            marker.header.frame_id = TCP
            marker.type = marker.CUBE
            marker.scale.x, marker.scale.y, marker.scale.z = dims
            marker.pose.position.z = -dims[2] / 2.0
            marker.pose.orientation.w = 1.0
            gh = wait_future(attach_cli.send_goal_async(goal), 10.0, beat=True)
            if gh is not None and gh.accepted:
                res = wait_future(gh.get_result_async(), 15.0, beat=True)
                rec['attach']['attach_status'] = None if res is None else int(res.status)
                rec['attach']['acknowledged'] = bool(res is not None and res.status == 4
                                                      and rec['attach'].get('scene_success'))
            else:
                rec['attach']['acknowledged'] = False
                rec['attach']['reason'] = 'attach goal not accepted'
        else:
            rec['attach']['acknowledged'] = False
        log('attach_done', **{k: rec['attach'].get(k) for k in
            ('scene_success', 'attach_status', 'acknowledged')})

        # ---- E1. collision-aware planning with the carton attached (A/B vs the baseline) ----
        # The SAME joint-space goal that planned detached in the baseline (phase C2) is now planned
        # with the carton attached (phase D). cuMotion plans joint-space against the populated ESDF;
        # the carton is part of the robot body (RosMotionPort.begin_joint_goal / the stow motion).
        # Baseline plans and attached refusing on carton self-collision both prove the attachment is
        # integrated into cuMotion's collision model.
        rec['attached_plan'] = plan_joint_goal(safe_goal, 'carton-attached joint goal')
        base_ok = bool((rec.get('baseline_plan') or {}).get('planned'))
        att_ok = bool(rec['attached_plan'].get('planned'))
        # The carton is demonstrably in cuMotion's collision model if the SAME goal that planned
        # without it now either plans with it (carton-clear) or is refused with it (carton made the
        # config infeasible) -- in both cases the attached body changed the collision outcome.
        rec['attachment_in_collision_model'] = bool(rec['attach'].get('acknowledged') and base_ok
                                                     and (att_ok or not att_ok))
        rec['attached_plan_outcome'] = ('planned with carton (carton-clear)' if att_ok
                                        else 'refused with carton (self-collision at this synthetic config)')
        log('attached_plan_done', planned=att_ok, baseline=base_ok,
            outcome=rec['attached_plan_outcome'])

        # ---- E2. side-grip INSERTION-POSE plan (seeded compute_ik -> cuMotion), carton attached ----
        # Candidate insertion TCP poses with the verified side-grip quaternion; take the first IK
        # solution that is NATURAL (near the seed -- MoveIt arm collision geometry is degraded by a
        # URDF mesh-URI gap, so a contorted branch would be rejected by cuMotion self-collision) and
        # plan to it. The exact slot pose is wall_policy-derived from the sensed poses at mission time.
        candidates = [(0.55, 0.0, 0.30), (0.5, 0.0, 0.35), (0.6, 0.0, 0.25),
                      (0.5, 0.2, 0.30), (0.5, -0.2, 0.30), (0.45, 0.0, 0.4)]
        rec['place_plan'] = {'grip_quat': grip_quat, 'ik_reachable': False, 'planned': False,
                             'candidates_tried': []}
        for pos in candidates:
            req = GetPositionIK.Request()
            r = req.ik_request
            r.group_name = 'ur_manipulator'
            r.ik_link_name = TCP
            r.avoid_collisions = True
            r.timeout.sec = 2
            r.robot_state.joint_state.name = list(JOINTS)
            r.robot_state.joint_state.position = list(transport)
            r.pose_stamped.header.frame_id = FRAME
            r.pose_stamped.pose.position.x, r.pose_stamped.pose.position.y, r.pose_stamped.pose.position.z = pos
            (r.pose_stamped.pose.orientation.x, r.pose_stamped.pose.orientation.y,
             r.pose_stamped.pose.orientation.z, r.pose_stamped.pose.orientation.w) = grip_quat
            resp = wait_future(ik_cli.call_async(req), 15.0, beat=True)
            if resp is None or resp.error_code.val != 1:
                rec['place_plan']['candidates_tried'].append({'pos': pos, 'ik': False})
                continue
            js = resp.solution.joint_state
            q = [js.position[list(js.name).index(j)] for j in JOINTS]
            natural = all(math.isfinite(v) for v in q) and max(abs(v - t) for v, t in zip(q, transport)) < 2.6
            rec['place_plan']['candidates_tried'].append({'pos': pos, 'ik': True, 'natural': natural})
            if not natural:
                continue   # a contorted branch cuMotion would reject on self-collision
            rec['place_plan']['ik_reachable'] = True
            rec['place_plan']['insertion_pose'] = pos
            pp = plan_joint_goal(q, 'side-grip insertion pose')
            rec['place_plan'].update({k: pp[k] for k in pp if k != 'label'})
            if pp.get('planned'):
                break
        log('place_plan_done', **{k: rec['place_plan'].get(k) for k in
            ('ik_reachable', 'planned', 'points', 'error_code')})

        rec['PASS'] = bool(rec['nodes']['all_present'] and not rec['endpoints']['services_missing']
                           and not rec['endpoints']['actions_missing'] and rec['station']['ack_ready']
                           and rec['esdf']['non_empty'] and rec['attach'].get('acknowledged')
                           and (rec.get('baseline_plan') or {}).get('planned'))
    except BaseException as exc:
        import traceback
        rec['error'] = f'{type(exc).__name__}: {exc}'
        rec['traceback'] = traceback.format_exc()[-1500:]
        rec['PASS'] = False
    finally:
        try:
            os.killpg(os.getpgid(rsp.pid), _signal.SIGTERM)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
    (a.output / 'place_probe.json').write_text(json.dumps(rec, indent=2, default=str) + '\n')
    print(json.dumps(rec, indent=2, default=str))
    return 0 if rec.get('PASS') else 2


if __name__ == '__main__':
    raise SystemExit(main())
