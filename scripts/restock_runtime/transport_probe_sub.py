#!/usr/bin/env python3
"""Host-side masked-depth transport assertion for the focused probe.

Run UNDER THE HOST OVERLAY ROS ENV with the host DDS config (ROS_DOMAIN_ID=88,
RMW_IMPLEMENTATION=rmw_fastrtps_cpp, FASTDDS_BUILTIN_TRANSPORTS=UDPv4, SKIP_DEFAULT_XML=1,
localhost). Subscribes the live masked_depth + camera_info for a window and reports the
measured count/rate and whether depth and camera_info share one stamp. Writes a JSON verdict.

  transport_probe_sub.py <out_json> <window_s>
"""
import json
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/dev/stdout'
    window = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
    rclpy.init()
    node = Node('transport_probe_sub')
    depth, info = [], []
    node.create_subscription(Image, '/restock_sim/camera/masked_depth',
        lambda m: depth.append(((m.header.stamp.sec, m.header.stamp.nanosec), m.height, m.width,
                                m.encoding, m.header.frame_id)), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, '/restock_sim/camera/masked_depth/camera_info',
        lambda m: info.append(((m.header.stamp.sec, m.header.stamp.nanosec), m.header.frame_id)),
        qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < window:
        rclpy.spin_once(node, timeout_sec=0.2)
    elapsed = max(1e-6, time.time() - t0)
    depth_stamps = {d[0] for d in depth}
    info_stamps = {i[0] for i in info}
    stamp_match = bool(depth_stamps & info_stamps)
    verdict = {'depth_msgs': len(depth), 'info_msgs': len(info),
               'depth_rate_hz': round(len(depth) / elapsed, 2),
               'depth_info_stamp_match': stamp_match,
               'encoding': depth[0][3] if depth else None,
               'frame_id': depth[0][4] if depth else None,
               'shape': (depth[0][1], depth[0][2]) if depth else None,
               'ok': len(depth) > 0 and len(info) > 0 and stamp_match}
    with open(out, 'w') as f:
        json.dump(verdict, f)
    print('TRANSPORT_PROBE_SUB ' + json.dumps(verdict))
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0 if verdict['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
