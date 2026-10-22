#!/usr/bin/env python3
"""Host-side verdict for the focused masked-depth transport probe.

Run UNDER THE HOST OVERLAY ROS ENV with the host DDS config (ROS_DOMAIN_ID=88,
RMW_IMPLEMENTATION=rmw_fastrtps_cpp, FASTDDS_BUILTIN_TRANSPORTS=UDPv4, SKIP_DEFAULT_XML=1,
localhost). Over a window it (1) subscribes the live masked_depth + camera_info and measures
count/rate + whether depth and camera_info share one stamp, then (2) queries nvblox's
get_esdf_and_gradient service and counts OBSERVED voxels -- the definitive proof the masked
depth flowed object_segmenter -> nvblox and was integrated. Writes a JSON verdict; exit 0 iff
masked_depth arrived on one stamp AND nvblox integrated (observed voxels > 0).

  transport_probe_sub.py <out_json> <window_s>
"""
import json
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

NS = '/restock_sim'
FRAME = 'ur_base_link'
UNOBSERVED = -1000.0


def _esdf_observed(node):
    """(observed_voxels, success) from nvblox over an AABB around the robot, or (None, None)."""
    try:
        from nvblox_msgs.srv import EsdfAndGradients
    except Exception:
        return None, None
    cli = node.create_client(EsdfAndGradients, NS + '/nvblox_node/get_esdf_and_gradient')
    if not cli.wait_for_service(timeout_sec=8.0):
        return None, None
    req = EsdfAndGradients.Request()
    req.update_esdf, req.visualize_esdf, req.use_aabb, req.frame_id = True, False, True, FRAME
    req.aabb_min_m.x, req.aabb_min_m.y, req.aabb_min_m.z = -2.5, -2.5, -1.0
    req.aabb_size_m.x, req.aabb_size_m.y, req.aabb_size_m.z = 5.0, 5.0, 3.0
    fut = cli.call_async(req)
    end = time.time() + 30.0
    while time.time() < end and not fut.done():
        rclpy.spin_once(node, timeout_sec=0.1)
    resp = fut.result() if fut.done() else None
    if resp is None:
        return None, None
    data = np.asarray(resp.esdf_and_gradients.data, dtype=np.float32)
    observed = int(np.count_nonzero(np.isfinite(data) & (data > UNOBSERVED + 1.0)))
    return observed, bool(resp.success)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/dev/stdout'
    window = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    rclpy.init()
    node = Node('transport_probe_sub')
    depth, info = [], []
    node.create_subscription(Image, NS + '/camera/masked_depth',
        lambda m: depth.append(((m.header.stamp.sec, m.header.stamp.nanosec), m.height, m.width,
                                m.encoding, m.header.frame_id)), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, NS + '/camera/masked_depth/camera_info',
        lambda m: info.append(((m.header.stamp.sec, m.header.stamp.nanosec), m.header.frame_id)),
        qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < window:
        rclpy.spin_once(node, timeout_sec=0.2)
    elapsed = max(1e-6, time.time() - t0)
    observed, esdf_success = _esdf_observed(node)

    stamp_match = bool({d[0] for d in depth} & {i[0] for i in info})
    verdict = {'depth_msgs': len(depth), 'info_msgs': len(info),
               'depth_rate_hz': round(len(depth) / elapsed, 2),
               'depth_info_stamp_match': stamp_match,
               'encoding': depth[0][3] if depth else None,
               'frame_id': depth[0][4] if depth else None,
               'shape': [depth[0][1], depth[0][2]] if depth else None,
               'nvblox_observed_voxels': observed, 'nvblox_esdf_success': esdf_success,
               'ok': (len(depth) > 0 and len(info) > 0 and stamp_match
                      and observed is not None and observed > 0)}
    with open(out, 'w') as f:
        json.dump(verdict, f)
    print('TRANSPORT_PROBE_SUB ' + json.dumps(verdict))
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0 if verdict['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
