#!/usr/bin/env python3
"""One-shot EmptyGeneration probe for the construction-baseline verification.

Runs beside an already-launched EvidenceProducerNode (launch_evidence_mapper.py
--execute) in DDS domain 88 and exercises exactly one behaviour per invocation:

  --mode accept  : with NOTHING published (no /clock, /tf or depth) call
                   EmptyGeneration once on the freshly-constructed map and expect
                   it ACCEPTED, recording the reported per-layer LayerCount
                   baseline. This is the case the stock producer wrongly refused
                   ("native layer not empty") because a fresh nvblox layer is
                   non-empty from construction.

  --mode depth   : publish /clock, the static map->optical_frame TF and ONE
                   synthetic depth frame (+camera_info) so the base nvblox node
                   integrates it and the layer grows beyond its construction
                   baseline, THEN call EmptyGeneration and expect it REFUSED
                   ("native layer not empty") with the grown LayerCount reported
                   and WITHOUT faulting the producer.

This is a FIXTURE sensor source and a launch/measurement probe, never a
commissioning step: it drives no admission handshake and publishes no certified
geometry. Pure policy lives in native/restock_mapper_evidence/state.hpp (CPU
tested); this adds only the ROS wiring and the native integration it cannot invent.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def _quat_from_axes(forward, up):
    """Optical-frame (x right, y down, z forward) quaternion (x,y,z,w) in map."""
    z = np.asarray(forward, float)
    z /= np.linalg.norm(z)
    down = -np.asarray(up, float)
    x = np.cross(down, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.column_stack([x, y, z])
    t = np.trace(r)
    w = np.sqrt(max(0.0, 1.0 + t)) / 2.0
    return ((r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w),
            (r[1, 0] - r[0, 1]) / (4 * w), w)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=['accept', 'depth'])
    p.add_argument('--generation', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--session', default='mission_a')
    p.add_argument('--scene', default='supermarket')
    p.add_argument('--cameras', default='front')
    p.add_argument('--localization', default='amcl')
    p.add_argument('--optical-frame', default='front_optical')
    p.add_argument('--challenge', default=None)
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--height', type=int, default=240)
    p.add_argument('--focal', type=float, default=200.0)
    p.add_argument('--depth-metres', type=float, default=2.0)
    p.add_argument('--frames', type=int, default=4)
    p.add_argument('--sim-step', type=float, default=0.1)
    p.add_argument('--integrate-wait', type=float, default=0.1)
    p.add_argument('--settle', type=float, default=4.0,
                   help='spin after the last frame so the depth queue drains and the '
                        'integration watermark (not a merely-queued frame) is what refuses')
    p.add_argument('--service-timeout', type=float, default=25.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.challenge is None:
        a.challenge = f'{a.mode}-{int(time.time())}'

    import rclpy
    from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy,
                           qos_profile_sensor_data)
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Image, CameraInfo
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import TransformStamped
    from builtin_interfaces.msg import Time
    from restock_mapper_evidence.srv import EmptyGeneration
    from restock_mapper_evidence.msg import MapIdentity as RMapIdentity

    record = {'mode': a.mode, 'generation': a.generation, 'challenge': a.challenge}
    rclpy.init(args=None)
    node = rclpy.create_node('empty_baseline_probe')
    ns = f'/restock_sim/navigation/input/{a.generation}'
    svc = '/restock_sim/navigation/nvblox_node'

    def time_msg(ns_):
        return Time(sec=int(ns_ // 1_000_000_000), nanosec=int(ns_ % 1_000_000_000))

    def spin(dt):
        end = time.monotonic() + dt
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)

    def wait_future(fut, timeout):
        end = time.monotonic() + timeout
        while rclpy.ok() and not fut.done() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)
        return fut.result() if fut.done() else None

    empty_cli = node.create_client(EmptyGeneration, f'{svc}/empty_generation')
    try:
        if not empty_cli.wait_for_service(timeout_sec=a.service_timeout):
            record['error'] = 'empty_generation service unavailable'
            raise RuntimeError(record['error'])

        if a.mode == 'depth':
            W, H = a.width, a.height
            fx = fy = a.focal
            cx, cy = W / 2.0, H / 2.0
            clock_pub = node.create_publisher(Clock, '/clock', 10)
            tf_static_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
            tf_static_pub = node.create_publisher(TFMessage, '/tf_static', tf_static_qos)
            tf_pub = node.create_publisher(TFMessage, '/tf', 10)
            depth_pub = node.create_publisher(Image, f'{ns}/camera_0/depth', qos_profile_sensor_data)
            info_pub = node.create_publisher(CameraInfo, f'{ns}/camera_0/camera_info', qos_profile_sensor_data)

            qx, qy, qz, qw = _quat_from_axes(forward=(1, 0, 0), up=(0, 0, 1))

            def tf_message(stamp_ns):
                tfm = TFMessage()
                for child, (px, py, pz, q) in {
                        a.optical_frame: (0.0, 0.0, 1.0, (qx, qy, qz, qw)),
                        'base_link': (0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 1.0))}.items():
                    t = TransformStamped()
                    t.header.stamp = time_msg(stamp_ns)
                    t.header.frame_id = 'map'
                    t.child_frame_id = child
                    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = px, py, pz
                    (t.transform.rotation.x, t.transform.rotation.y,
                     t.transform.rotation.z, t.transform.rotation.w) = q
                    tfm.transforms.append(t)
                return tfm

            info = CameraInfo()
            info.width, info.height = W, H
            info.distortion_model = 'plumb_bob'
            info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            depth_bytes = np.full((H, W), float(a.depth_metres), dtype='<f4').tobytes()

            sim_ns = int(1_000_000_000)
            for _ in range(3):
                c = Clock(); c.clock = time_msg(sim_ns)
                clock_pub.publish(c)
                tf_static_pub.publish(tf_message(sim_ns))
                tf_pub.publish(tf_message(sim_ns))
                spin(0.05)

            for _ in range(a.frames):
                sim_ns += int(a.sim_step * 1e9)
                c = Clock(); c.clock = time_msg(sim_ns)
                clock_pub.publish(c)
                tf_pub.publish(tf_message(sim_ns))
                spin(0.02)
                stamp_ns = sim_ns - 20_000_000
                img = Image()
                img.header.stamp = time_msg(stamp_ns)
                img.header.frame_id = a.optical_frame
                img.height, img.width = H, W
                img.encoding = '32FC1'
                img.is_bigendian = 0
                img.step = W * 4
                img.data = depth_bytes
                info.header = img.header
                depth_pub.publish(img)
                info_pub.publish(info)
                spin(a.integrate_wait)
            record['published_depth_frames'] = a.frames
            # Keep the sim clock ticking so the node drains and integrates its depth
            # queue; after this the refusal is the integration watermark, not a frame
            # still sitting in the queue.
            settle_end = time.monotonic() + a.settle
            while time.monotonic() < settle_end:
                sim_ns += int(a.sim_step * 1e9)
                c = Clock(); c.clock = time_msg(sim_ns)
                clock_pub.publish(c)
                tf_pub.publish(tf_message(sim_ns))
                spin(0.1)

        req = EmptyGeneration.Request()
        req.challenge = a.challenge
        req.expected_identity = RMapIdentity(
            session=a.session, scene=a.scene, cameras=a.cameras, localization=a.localization)
        resp = wait_future(empty_cli.call_async(req), a.service_timeout)
        if resp is None:
            record['error'] = 'empty_generation call timed out'
        else:
            record['success'] = bool(resp.success)
            record['failure_reason'] = resp.failure_reason
            record['layer_inventory_identity'] = resp.layer_inventory_identity
            record['layers'] = [{'name': lc.layer_name, 'type': lc.layer_type,
                                 'blocks': int(lc.allocated_blocks)} for lc in resp.layers]
            record['certificate_token'] = resp.certificate_token
    except BaseException as exc:
        record.setdefault('error', f'{type(exc).__name__}: {exc}')
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
    (a.output / f'probe_{a.mode}.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    print(json.dumps(record, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
