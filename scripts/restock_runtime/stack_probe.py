#!/usr/bin/env python3
"""Bounded verification probe for the restock host planning stack + mapper.

Runs beside an already-launched stack (launch_restock_host_stack.sh) on domain 88 and
measures, without commissioning anything:

  1. nodes/endpoints : the 6 planning-stack nodes are present and the host-provided
     service/action endpoints are advertised (arm_controller/follow_joint_trajectory
     is the EXTERNAL robot controller, not launched here; reported separately). The
     host no longer runs the NVIDIA RobotSegmenter self-filter: the Isaac side supplies
     robot-masked depth on robot_world_depth, which this probe stands in for.
  2. planning round-trip : a GetPositionIK (compute_ik) request against the refined
     CAD robot_description round-trips through move_group.
  3. depth integration : a synthetic registered depth frame (a flat obstacle plane) is
     published through the object_segmenter on robot_world_depth -> world_depth, with a
     matching CameraInfo and a static camera->ur_base_link TF. nvblox fuses it, and the
     get_esdf_and_gradient grid over a focused AABB goes from zero observed voxels
     (every cell == the -1000 unobserved sentinel) to many observed voxels, proving the
     mapper actually integrated depth (not just an allocated-but-empty baseline).
  4. cuMotion plan : a cuMotion MoveGroup joint-space plan (plan_only, explicit
     start_state) against the now-populated ESDF returns a trajectory.

Fixture/measurement only; a successful probe is NOT sensor/endpoint readiness. The
synthetic depth stands in for the integration-owned Isaac masked-depth publisher.
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
    p.add_argument('--discovery-timeout', type=float, default=120.0)
    p.add_argument('--plan-timeout', type=float, default=60.0)
    p.add_argument('--integrate-seconds', type=float, default=5.0)
    # Cold first-job latency measurement (vs the IsaacRosGraph 5 s process/status watchdog):
    # how long after the first masked-depth frame the ESDF first reports observed voxels, and
    # how long the first cuMotion plan reply takes. --esdf-ready-timeout caps the ESDF poll.
    p.add_argument('--esdf-ready-timeout', type=float, default=20.0)
    p.add_argument('--esdf-poll-interval', type=float, default=0.3)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(a.config.read_text())

    import numpy as np
    import rclpy
    from rclpy.action import ActionClient, get_action_server_names_and_types_by_node
    from rclpy.qos import qos_profile_sensor_data
    from builtin_interfaces.msg import Time as RTime
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import StaticTransformBroadcaster
    from moveit_msgs.srv import GetPositionIK
    from moveit_msgs.action import MoveGroup
    from nvblox_msgs.srv import EsdfAndGradients

    # nvblox fills voxels it has never observed with this sentinel (node_params.hpp
    # esdf_and_gradients_unobserved_value); any other finite value is an observed voxel.
    UNOBSERVED = -1000.0

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from twin.restock_host.config import ENDPOINTS, NS, FRAME, TCP, JOINTS
    from twin.restock_host.readiness import missing_endpoints

    rec = {'steps': [], 'endpoints': {}, 'nodes': {}, 'latencies': {}}

    def log(ev, **kw):
        rec['steps'].append({'t': round(time.monotonic(), 2), 'event': ev, **kw})

    manifest = json.loads(Path(config['mobile_manifest']).read_text())['transport_positions']
    transport = [float(manifest[j]) for j in JOINTS]

    # All movable joints (arm at transport, base/cart at their manifest value or 0) so a
    # robot_state_publisher can emit the full TF tree the planner needs (ur_base_link).
    import xml.etree.ElementTree as ET
    import os
    import signal as _signal
    import subprocess
    robot_xml = ET.parse(config['urdf']).getroot()
    movable = [(j.get('name'), j.find('child').get('link')) for j in robot_xml.findall('joint')
               if j.get('type') != 'fixed']
    all_names = [n for n, _ in movable]
    all_pos = [float(manifest.get(n, 0.0) or 0.0) for n in all_names]

    # Fixture TF source: robot_state_publisher turns joint_states into the TF tree the
    # cuMotion world query and move_group require. Real joint_states/TF are external.
    rsp = subprocess.Popen(
        ['ros2', 'run', 'robot_state_publisher', 'robot_state_publisher', '--ros-args',
         '-p', 'use_sim_time:=true', '-p', 'robot_description:=' + Path(config['urdf']).read_text(),
         '-r', 'joint_states:=' + NS + '/joint_states'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)

    rclpy.init()
    node = rclpy.create_node('restock_stack_probe', namespace=NS)

    # Keep sim time moving so use_sim_time nodes have a clock.
    clock_pub = node.create_publisher(Clock, '/clock', 10)
    js_pub = node.create_publisher(JointState, NS + '/joint_states', 10)
    sim_ns = [int(1_000_000_000)]

    def tick_clock():
        sim_ns[0] += 20_000_000
        m = Clock(); m.clock = RTime(sec=sim_ns[0] // 1_000_000_000, nanosec=sim_ns[0] % 1_000_000_000)
        clock_pub.publish(m)
        js = JointState()
        js.header.stamp = m.clock
        js.name = list(all_names)
        js.position = list(all_pos)
        js_pub.publish(js)

    def spin(dt):
        end = time.monotonic() + dt
        while time.monotonic() < end:
            tick_clock()
            rclpy.spin_once(node, timeout_sec=0.02)

    def wait_future(fut, timeout):
        end = time.monotonic() + timeout
        while rclpy.ok() and not fut.done() and time.monotonic() < end:
            tick_clock()
            rclpy.spin_once(node, timeout_sec=0.02)
        return fut.result() if fut.done() else None

    # --- synthetic masked-depth source (stands in for the integration Isaac bridge) ---
    # A wide-FOV down-looking camera 2.0 m above ur_base_link sees a flat floor at z=-0.9
    # (2.9 m away, under the 5 m max integration distance). nvblox carves the whole frustum
    # FREE from the camera down to the floor, so the robot arm's entire body envelope becomes
    # observed-free and cuMotion's start pose is not in (pessimistic-unobserved) world
    # collision, while the floor remains an observed obstacle so the ESDF is non-empty.
    # Published 32FC1 on camera/masked_depth + camera/masked_depth/camera_info with a static
    # camera->ur_base_link TF: exactly the agreed Option-B contract that the object_segmenter
    # republishes as world_depth for nvblox to fuse.
    CAM_FRAME, W, H, FX, FY = 'restock_camera_0', 200, 200, 60.0, 60.0
    CX, CY, DEPTH_M = W / 2.0, H / 2.0, 2.9
    depth_pub = node.create_publisher(Image, NS + '/camera/masked_depth', 10)
    info_pub = node.create_publisher(CameraInfo, NS + '/camera/masked_depth/camera_info', 10)
    esdf_cli = node.create_client(EsdfAndGradients, NS + '/nvblox_node/get_esdf_and_gradient')
    tf_static = StaticTransformBroadcaster(node)
    _tf = TransformStamped()
    _tf.header.frame_id, _tf.child_frame_id = FRAME, CAM_FRAME     # ur_base_link -> camera
    _tf.transform.translation.z = 2.0
    _tf.transform.rotation.x, _tf.transform.rotation.w = 1.0, 0.0  # 180 deg about X -> look down
    tf_static.sendTransform(_tf)
    _depth_bytes = np.full((H, W), DEPTH_M, dtype=np.float32).tobytes()

    def publish_depth():
        stamp = RTime(sec=sim_ns[0] // 1_000_000_000, nanosec=sim_ns[0] % 1_000_000_000)
        img = Image()
        img.header.stamp, img.header.frame_id = stamp, CAM_FRAME
        img.height, img.width = H, W
        img.encoding, img.is_bigendian, img.step = '32FC1', 0, W * 4
        img.data = _depth_bytes
        depth_pub.publish(img)
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, CAM_FRAME
        info.height, info.width = H, W
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [FX, 0.0, CX, 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
        info_pub.publish(info)

    def integrate_depth(seconds, hz=10.0):
        period, nxt, end = 1.0 / hz, 0.0, time.monotonic() + seconds
        while time.monotonic() < end:
            if time.monotonic() >= nxt:
                publish_depth(); nxt = time.monotonic() + period
            tick_clock(); rclpy.spin_once(node, timeout_sec=0.02)

    def esdf_observed():
        """Observed-voxel census over a focused AABB spanning the floor + carved free space."""
        req = EsdfAndGradients.Request()
        req.update_esdf, req.visualize_esdf, req.use_aabb, req.frame_id = True, False, True, FRAME
        req.aabb_min_m.x, req.aabb_min_m.y, req.aabb_min_m.z = -0.8, -0.8, -1.0
        req.aabb_size_m.x, req.aabb_size_m.y, req.aabb_size_m.z = 1.6, 1.6, 1.2
        resp = wait_future(esdf_cli.call_async(req), 30.0)
        if resp is None:
            return None
        data = np.asarray(resp.esdf_and_gradients.data, dtype=np.float32)
        dims = [d.size for d in resp.esdf_and_gradients.layout.dim]
        observed = int(np.count_nonzero(np.isfinite(data) & (data > UNOBSERVED + 1.0)))
        return {'success': bool(resp.success), 'grid_dims': dims, 'total_voxels': int(data.size),
                'observed_voxels': observed, 'voxel_size_m': float(resp.voxel_size_m)}

    try:
        # ---- 1. discovery: nodes + endpoints ----
        want_nodes = {'move_group', 'cumotion_planner', 'static_planning_scene_server',
                      'object_attachment', 'nvblox_node', 'object_segmenter'}
        deadline = time.monotonic() + a.discovery_timeout
        seen_nodes, missing = set(), list(ENDPOINTS)
        while time.monotonic() < deadline:
            seen_nodes = {nm for nm, _ in node.get_node_names_and_namespaces()}
            actions = []
            for nm, nsp in node.get_node_names_and_namespaces():
                actions.extend(get_action_server_names_and_types_by_node(node, nm, nsp))
            missing = missing_endpoints(node.get_service_names_and_types(), actions)
            # follow_joint_trajectory is the external controller; ignore for "servers up".
            missing_stack = [m for m in missing if 'follow_joint_trajectory' not in m]
            if want_nodes <= seen_nodes and not missing_stack:
                break
            spin(0.5)
        rec['nodes'] = {'present': sorted(want_nodes & seen_nodes),
                        'absent': sorted(want_nodes - seen_nodes),
                        'all_present': want_nodes <= seen_nodes}
        rec['endpoints'] = {'missing_all': missing,
                            'missing_excluding_external_controller':
                                [m for m in missing if 'follow_joint_trajectory' not in m],
                            'external_controller_absent':
                                any('follow_joint_trajectory' in m for m in missing)}
        log('discovery_done', present=len(want_nodes & seen_nodes),
            missing=rec['endpoints']['missing_excluding_external_controller'])

        # ---- 2a. planning round-trip: compute_ik against refined CAD kinematics ----
        ik_cli = node.create_client(GetPositionIK, NS + '/compute_ik')
        rec['compute_ik'] = {'available': ik_cli.wait_for_service(timeout_sec=10.0)}
        if rec['compute_ik']['available']:
            req = GetPositionIK.Request()
            r = req.ik_request
            r.group_name = 'ur_manipulator'
            r.robot_state.joint_state.name = list(JOINTS)
            r.robot_state.joint_state.position = list(transport)
            r.ik_link_name = TCP
            r.pose_stamped.header.frame_id = FRAME
            r.pose_stamped.pose.position.x = 0.4
            r.pose_stamped.pose.position.y = 0.0
            r.pose_stamped.pose.position.z = 0.4
            r.pose_stamped.pose.orientation.w = 1.0
            r.timeout.sec = 2
            r.avoid_collisions = False
            resp = wait_future(ik_cli.call_async(req), 30.0)
            if resp is None:
                rec['compute_ik']['round_trip'] = False
            else:
                rec['compute_ik']['round_trip'] = True
                rec['compute_ik']['error_code'] = int(resp.error_code.val)  # 1 == SUCCESS
            log('compute_ik_done', **{k: rec['compute_ik'].get(k) for k in ('round_trip', 'error_code')})

        # ---- 2b. depth integration + COLD first-ESDF-ready latency: starting from a cold
        #          mapper (every cell the -1000 unobserved sentinel), begin publishing masked
        #          depth and poll the ESDF service until it first reports observed voxels. The
        #          elapsed time from the first depth frame to that first non-empty read is the
        #          "first ESDF ready after depth starts" latency the IsaacRosGraph watchdog
        #          cares about (the ESDF service call drives update_esdf=True exactly as the
        #          cuMotion planner's world query does). ----
        rec['mapper_esdf'] = {'available': esdf_cli.wait_for_service(timeout_sec=10.0)}
        if rec['mapper_esdf']['available']:
            before = esdf_observed()  # cold baseline: should be zero observed voxels
            obs_before = (before or {}).get('observed_voxels')
            t_depth0 = time.monotonic()
            first_ready_s, after = None, None
            period, nxt_pub, nxt_poll = 0.1, 0.0, t_depth0  # 10 Hz publish; poll from t0
            deadline = t_depth0 + a.esdf_ready_timeout
            while time.monotonic() < deadline:
                now = time.monotonic()
                if now >= nxt_pub:
                    publish_depth(); nxt_pub = now + period
                if now >= nxt_poll:
                    nxt_poll = time.monotonic() + a.esdf_poll_interval
                    snap = esdf_observed()  # blocks on the service (ticks clock + spins inside)
                    if snap and snap.get('observed_voxels', 0) > (obs_before or 0) and snap['observed_voxels'] > 0:
                        first_ready_s = round(time.monotonic() - t_depth0, 3)
                        after = snap
                        break
                    nxt_poll = time.monotonic() + a.esdf_poll_interval  # re-base after blocking call
                tick_clock(); rclpy.spin_once(node, timeout_sec=0.02)
            if after is None:  # never went non-empty within the window; take a final read
                after = esdf_observed()
            obs_after = (after or {}).get('observed_voxels')
            rec['mapper_esdf'].update(
                round_trip=bool(before and after),
                success=bool(after and after['success']),
                grid_dims=(after or {}).get('grid_dims'),
                voxel_size_m=(after or {}).get('voxel_size_m'),
                total_voxels=(after or {}).get('total_voxels'),
                observed_before=obs_before, observed_after=obs_after,
                non_empty=bool(obs_after and obs_after > 0 and obs_after > (obs_before or 0)))
            rec['latencies']['first_esdf_ready_s'] = first_ready_s
            rec['latencies']['esdf_ready_timed_out'] = first_ready_s is None
        log('mapper_esdf_done', **{k: rec['mapper_esdf'].get(k)
            for k in ('round_trip', 'success', 'observed_before', 'observed_after', 'non_empty')},
            first_esdf_ready_s=rec['latencies'].get('first_esdf_ready_s'))

        # ---- 2c. cuMotion MoveGroup joint-space plan (plan_only) against the populated ESDF ----
        rec['cumotion_plan'] = {}
        mg = ActionClient(node, MoveGroup, NS + '/move_action')
        if mg.wait_for_server(timeout_sec=15.0):
            from moveit_msgs.msg import Constraints, JointConstraint, RobotState
            goal = MoveGroup.Goal()
            goal.request.group_name = 'ur_manipulator'
            goal.request.pipeline_id = 'isaac_ros_cumotion'
            goal.request.allowed_planning_time = 10.0
            goal.request.num_planning_attempts = 1
            start = RobotState()
            start.joint_state.name = list(JOINTS)
            start.joint_state.position = list(transport)
            start.is_diff = False
            goal.request.start_state = start
            con = Constraints()
            for name, value in zip(JOINTS, transport):
                jc = JointConstraint(joint_name=name, position=value + 0.1,
                                     tolerance_above=0.05, tolerance_below=0.05, weight=1.0)
                con.joint_constraints.append(jc)
            goal.request.goal_constraints = [con]
            goal.planning_options.plan_only = True
            # COLD first-cuMotion-reply latency: wall time from sending the manip (plan) job to
            # receiving the plan result. This is the reply the IsaacRosGraph process/status
            # watchdog is timing; the cuMotion node queries the (now populated) nvblox ESDF as
            # part of answering, so the reply includes its own world-sync cost.
            t_goal = time.monotonic()
            gh_fut = mg.send_goal_async(goal)
            gh = wait_future(gh_fut, 20.0)
            if gh is not None and gh.accepted:
                res = wait_future(gh.get_result_async(), a.plan_timeout)
                if res is not None:
                    code = int(res.result.error_code.val)
                    reply_s = round(time.monotonic() - t_goal, 3)
                    rec['latencies']['first_cumotion_reply_s'] = reply_s
                    rec['cumotion_plan'] = {'round_trip': True, 'error_code': code,
                                            'reply_s': reply_s,
                                            'points': len(res.result.planned_trajectory.joint_trajectory.points)}
                else:
                    rec['cumotion_plan'] = {'round_trip': False, 'reason': 'no result within timeout'}
            else:
                rec['cumotion_plan'] = {'round_trip': False, 'reason': 'goal not accepted'}
        else:
            rec['cumotion_plan'] = {'round_trip': False, 'reason': 'move_action server absent'}
        log('cumotion_plan_done', **rec['cumotion_plan'])

        # ---- cold first-job summary vs the 5 s IsaacRosGraph watchdog ----
        esdf_s = rec['latencies'].get('first_esdf_ready_s')
        reply_s = rec['latencies'].get('first_cumotion_reply_s')
        # Worst case for a cold first job: the manip request lands before any depth has been
        # integrated, so the host must build the ESDF and then plan before replying.
        rec['latencies']['cold_first_job_s'] = (
            round(esdf_s + reply_s, 3) if (esdf_s is not None and reply_s is not None) else None)
        rec['latencies']['watchdog_s'] = 5.0
        cold = rec['latencies']['cold_first_job_s']
        rec['latencies']['within_watchdog'] = (cold is not None and cold <= 5.0)
        log('latency_summary', **rec['latencies'])
    except BaseException as exc:
        rec['error'] = f'{type(exc).__name__}: {exc}'
        log('error', detail=rec['error'])
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
    (a.output / 'stack_probe.json').write_text(json.dumps(rec, indent=2, default=str) + '\n')
    print(json.dumps(rec, indent=2, default=str))
    ok = (rec.get('nodes', {}).get('all_present')
          and not rec.get('endpoints', {}).get('missing_excluding_external_controller')
          and rec.get('compute_ik', {}).get('round_trip')
          and rec.get('mapper_esdf', {}).get('non_empty')
          and (rec.get('cumotion_plan', {}).get('points') or 0) > 0)
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
