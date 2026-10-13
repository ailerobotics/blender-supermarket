#!/usr/bin/env python3
"""Bounded verification probe for the restock host planning stack + mapper.

Runs beside an already-launched stack (launch_restock_host_stack.sh) on domain 88 and
measures, without commissioning anything:

  1. nodes/endpoints : the 7 planning-stack nodes are present and the host-provided
     service/action endpoints are advertised (arm_controller/follow_joint_trajectory
     is the EXTERNAL robot controller, not launched here; reported separately).
  2. planning round-trip : a GetPositionIK (compute_ik) request against the refined
     CAD robot_description round-trips through move_group; a best-effort cuMotion
     MoveGroup joint-space plan (plan_only, explicit start_state) is also attempted.
  3. mapper slice : the nvblox get_esdf_and_gradient service returns a grid (the
     manipulator path serves the map as an ESDF/gradient grid on request, not a
     published slice topic: publish_esdf_distance_slice is false).

Fixture/measurement only; a successful probe is NOT sensor/endpoint readiness.
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
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(a.config.read_text())

    import rclpy
    from rclpy.action import ActionClient, get_action_server_names_and_types_by_node
    from builtin_interfaces.msg import Time as RTime
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState
    from moveit_msgs.srv import GetPositionIK
    from moveit_msgs.action import MoveGroup
    from nvblox_msgs.srv import EsdfAndGradients

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from twin.restock_host.config import ENDPOINTS, NS, FRAME, TCP, JOINTS
    from twin.restock_host.readiness import missing_endpoints

    rec = {'steps': [], 'endpoints': {}, 'nodes': {}}

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

    try:
        # ---- 1. discovery: nodes + endpoints ----
        want_nodes = {'move_group', 'cumotion_planner', 'static_planning_scene_server',
                      'object_attachment', 'robot_segmenter', 'nvblox_node', 'object_segmenter'}
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
                        'all_seven_present': want_nodes <= seen_nodes}
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

        # ---- 2b. best-effort cuMotion MoveGroup joint-space plan (plan_only) ----
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
            gh_fut = mg.send_goal_async(goal)
            gh = wait_future(gh_fut, 20.0)
            if gh is not None and gh.accepted:
                res = wait_future(gh.get_result_async(), a.plan_timeout)
                if res is not None:
                    code = int(res.result.error_code.val)
                    rec['cumotion_plan'] = {'round_trip': True, 'error_code': code,
                                            'points': len(res.result.planned_trajectory.joint_trajectory.points)}
                else:
                    rec['cumotion_plan'] = {'round_trip': False, 'reason': 'no result within timeout'}
            else:
                rec['cumotion_plan'] = {'round_trip': False, 'reason': 'goal not accepted'}
        else:
            rec['cumotion_plan'] = {'round_trip': False, 'reason': 'move_action server absent'}
        log('cumotion_plan_done', **rec['cumotion_plan'])

        # ---- 3. mapper slice: get_esdf_and_gradient returns a grid ----
        esdf_cli = node.create_client(EsdfAndGradients, NS + '/nvblox_node/get_esdf_and_gradient')
        rec['mapper_esdf'] = {'available': esdf_cli.wait_for_service(timeout_sec=10.0)}
        if rec['mapper_esdf']['available']:
            req = EsdfAndGradients.Request()
            req.update_esdf = True
            req.visualize_esdf = False
            req.use_aabb = True
            req.frame_id = FRAME
            lo = config['workspace_min']; hi = config['workspace_max']
            req.aabb_min_m.x, req.aabb_min_m.y, req.aabb_min_m.z = [float(v) for v in lo]
            req.aabb_size_m.x = float(hi[0] - lo[0])
            req.aabb_size_m.y = float(hi[1] - lo[1])
            req.aabb_size_m.z = float(hi[2] - lo[2])
            resp = wait_future(esdf_cli.call_async(req), 30.0)
            if resp is None:
                rec['mapper_esdf']['round_trip'] = False
            else:
                dims = [d.size for d in resp.esdf_and_gradients.layout.dim]
                rec['mapper_esdf'].update(round_trip=True, success=bool(resp.success),
                                          voxel_size_m=float(resp.voxel_size_m),
                                          grid_dims=dims, values=len(resp.esdf_and_gradients.data))
        log('mapper_esdf_done', **{k: rec['mapper_esdf'].get(k) for k in ('round_trip', 'success', 'grid_dims')})
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
    ok = (rec.get('nodes', {}).get('all_seven_present')
          and not rec.get('endpoints', {}).get('missing_excluding_external_controller')
          and rec.get('compute_ik', {}).get('round_trip')
          and rec.get('mapper_esdf', {}).get('round_trip'))
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
