#!/usr/bin/env python3
"""No-GPU smoke for the out-of-Isaac masked-depth publisher (Firstmate option B).

Run UNDER THE HOST OVERLAY ROS ENV on domain 88 (rclpy works out-of-Isaac). It spawns the
`masked_depth_ipc.serve` subprocess, feeds it synthetic frames via FrameWriter (the exact
Isaac-side IPC), and runs an rclpy subscriber to confirm masked_depth + camera_info +
target_bbox all arrive with masked_depth and camera_info sharing one stamp. Proves the
subprocess publish path before spending a GPU shakedown. Exit 0 = pass.

  bash -c 'set +u; source <runtime-env.sh>; export ROS_DOMAIN_ID=88 \
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST; PYTHONPATH=scripts python3 \
    scripts/restock_runtime/maskdepth_pub_smoke.py'
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray

from twin.restock.masked_depth_ipc import FrameWriter

IPC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'twin', 'restock', 'masked_depth_ipc.py')


def main():
    sock = tempfile.mktemp(suffix='.sock')
    proc = subprocess.Popen([sys.executable, os.path.abspath(IPC), 'serve', sock, '40'])
    try:
        rclpy.init()
        node = Node('maskdepth_smoke_sub')
        got = {'depth': [], 'info': [], 'bbox': []}
        node.create_subscription(Image, '/restock_sim/camera/masked_depth',
            lambda m: got['depth'].append(((m.header.stamp.sec, m.header.stamp.nanosec),
                                           m.height, m.width, m.encoding, m.header.frame_id,
                                           len(m.data))), qos_profile_sensor_data)
        node.create_subscription(CameraInfo, '/restock_sim/camera/masked_depth/camera_info',
            lambda m: got['info'].append(((m.header.stamp.sec, m.header.stamp.nanosec),
                                          m.header.frame_id, tuple(m.k))), qos_profile_sensor_data)
        node.create_subscription(Float32MultiArray, '/restock_sim/target_bbox',
            lambda m: got['bbox'].append(list(m.data)), qos_profile_sensor_data)

        for _ in range(20):                      # allow pub/sub discovery
            rclpy.spin_once(node, timeout_sec=0.1)
        writer = FrameWriter(sock, connect_timeout_s=10.0)
        h, w = 4, 8
        depth = np.arange(h * w, dtype='<f4')
        for i in range(40):
            stamp = 5.0 + i * 0.1
            sec = int(stamp)
            writer.send({'stamp_sec': sec, 'stamp_nsec': int(round((stamp - sec) * 1e9)),
                         'frame_id': 'restock_camera_arm', 'width': w, 'height': h,
                         'k': [500.0, 0, w / 2, 0, 500.0, h / 2, 0, 0, 1.0],
                         'bbox': [1.0, 2.0, 3.0, 4.0]}, depth.tobytes())
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.03)
        t0 = time.time()
        while time.time() - t0 < 3.0:
            rclpy.spin_once(node, timeout_sec=0.1)
        writer.close()

        stamp_match = bool(got['depth'] and got['info'] and got['depth'][0][0] == got['info'][0][0])
        depth_ok = bool(got['depth']) and got['depth'][0][3] == '32FC1' and \
            got['depth'][0][1:3] == (h, w) and got['depth'][0][5] == h * w * 4 and \
            got['depth'][0][4] == 'restock_camera_arm'
        verdict = {'depth_msgs': len(got['depth']), 'info_msgs': len(got['info']),
                   'bbox_msgs': len(got['bbox']), 'depth_shape_encoding_ok': depth_ok,
                   'depth_info_stamp_match': stamp_match,
                   'bbox_value': got['bbox'][0] if got['bbox'] else None}
        ok = (verdict['depth_msgs'] > 0 and verdict['info_msgs'] > 0 and verdict['bbox_msgs'] > 0
              and depth_ok and stamp_match)
        verdict['ok'] = ok
        print('MASKDEPTH_PUB_SMOKE ' + json.dumps(verdict))
        return 0 if ok else 1
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


if __name__ == '__main__':
    sys.exit(main())
