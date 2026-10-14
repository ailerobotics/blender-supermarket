#!/usr/bin/env python3
"""Reproduce the host-DDS-relay TF race for the nvblox depth path, per TF lag.

Beside an already-launched host stack (launch RESTOCK_NVBLOX_DIAG=1 so nvblox prints its
input-queue drops/delays), this probe publishes real-shaped masked depth (frame
restock_camera_arm) from THIS process while a SEPARATE process (tf_lag_publisher) supplies
the matching ur_base_link->restock_camera_arm transform on /tf, stamped ``lag`` behind the
sim clock. For each lag in {0, 50, 150, 500} ms it measures whether nvblox integrates the
depth (the get_esdf_and_gradient response's integration stamp advances into the trial's
published-stamp window, and/or observed voxels increase) and whether the host incorporation
ack's collision_time advances. That distinguishes "nvblox queues and retries the TF lookup"
(integrates after the lag) from "drops on first miss" (never integrates while lagged).

Measurement only; the depth stands in for the integration-owned Isaac masked-depth bridge.
"""
import argparse
import json
import os
import signal as _signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--discovery-timeout', type=float, default=150.0)
    p.add_argument('--lags-ms', type=float, nargs='+', default=[0.0, 50.0, 150.0, 500.0])
    p.add_argument('--trial-seconds', type=float, default=3.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import rclpy
    from rclpy.parameter import Parameter
    from rosgraph_msgs.msg import Clock
    from builtin_interfaces.msg import Time as RTime
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    from nvblox_msgs.srv import EsdfAndGradients

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from twin.restock_host.config import NS, FRAME

    UNOBSERVED = -1000.0
    CAM_FRAME, W, H, FX, FY = 'restock_camera_arm', 200, 200, 60.0, 60.0
    CX, CY, DEPTH_M = W / 2.0, H / 2.0, 2.9
    rec = {'lags_ms': a.lags_ms, 'trials': [], 'nvblox_params_probed': {
        'maximum_input_queue_length': 10, 'tick_period_ms': 10, 'integrate_depth_rate_hz': 40.0,
        'update_esdf_rate_hz_override': 0.0, 'use_tf_transforms': True}}

    rclpy.init()
    node = rclpy.create_node('tf_race_probe', namespace=NS,
                             parameter_overrides=[Parameter('use_sim_time', value=True)])
    clock_pub = node.create_publisher(Clock, '/clock', 10)
    depth_pub = node.create_publisher(Image, NS + '/camera/masked_depth', 10)
    info_pub = node.create_publisher(CameraInfo, NS + '/camera/masked_depth/camera_info', 10)
    cmd_pub = node.create_publisher(String, NS + '/station/command', 10)
    status_box = {'msg': None}
    node.create_subscription(String, NS + '/station/status',
                             lambda m: status_box.__setitem__('msg', m.data), 10)
    esdf_cli = node.create_client(EsdfAndGradients, NS + '/nvblox_node/get_esdf_and_gradient')
    sim_ns = [int(1_000_000_000)]
    _depth_bytes = np.full((H, W), DEPTH_M, dtype=np.float32).tobytes()

    def tick_clock():
        sim_ns[0] += 20_000_000
        m = Clock(); m.clock = RTime(sec=sim_ns[0] // 1_000_000_000, nanosec=sim_ns[0] % 1_000_000_000)
        clock_pub.publish(m)

    def sim_now():
        return sim_ns[0] * 1e-9

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

    def wait_future(fut, timeout):
        end = time.monotonic() + timeout
        while rclpy.ok() and not fut.done() and time.monotonic() < end:
            tick_clock(); rclpy.spin_once(node, timeout_sec=0.02)
        return fut.result() if fut.done() else None

    def esdf_probe():
        """Return observed-voxel count and the response's integration stamp (last fused frame)."""
        req = EsdfAndGradients.Request()
        req.update_esdf, req.visualize_esdf, req.use_aabb, req.frame_id = True, False, True, FRAME
        req.aabb_min_m.x, req.aabb_min_m.y, req.aabb_min_m.z = -0.8, -0.8, -1.0
        req.aabb_size_m.x, req.aabb_size_m.y, req.aabb_size_m.z = 1.6, 1.6, 1.2
        resp = wait_future(esdf_cli.call_async(req), 20.0)
        if resp is None:
            return None
        data = np.asarray(resp.esdf_and_gradients.data, dtype=np.float32)
        observed = int(np.count_nonzero(np.isfinite(data) & (data > UNOBSERVED + 1.0)))
        h = resp.header.stamp
        return {'success': bool(resp.success), 'observed_voxels': observed,
                'integration_stamp': round(h.sec + h.nanosec * 1e-9, 3)}

    def spawn_tf(lag_ms):
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve().parent / 'tf_lag_publisher.py'),
             '--lag-ms', str(lag_ms), '--parent-frame', FRAME, '--child-frame', CAM_FRAME],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)

    tf_child = None
    try:
        esdf_cli.wait_for_service(timeout_sec=a.discovery_timeout)
        # Bring the incorporation ack active once (prepare); heartbeat throughout so it stays
        # active (3 s wall guard) and its collision_time reflects each trial's integration.
        deadline = time.monotonic() + 8.0
        while cmd_pub.get_subscription_count() < 1 and time.monotonic() < deadline:
            tick_clock(); rclpy.spin_once(node, timeout_sec=0.02)
        epoch, seq = 1, 0
        cmd_pub.publish(String(data=json.dumps(dict(action='prepare', epoch=epoch, sequence=seq,
            stamp=sim_now(), parked=True, camera_frame=CAM_FRAME))))

        def heartbeat():
            nonlocal seq
            seq += 1
            cmd_pub.publish(String(data=json.dumps(dict(action='heartbeat', epoch=epoch,
                sequence=seq, stamp=sim_now(), parked=True))))

        for lag in a.lags_ms:
            # settle, then baseline the integration stamp / observed voxels
            t0 = time.monotonic()
            while time.monotonic() - t0 < 0.5:
                tick_clock(); heartbeat(); rclpy.spin_once(node, timeout_sec=0.02)
            before = esdf_probe()
            start_stamp = sim_now()
            tf_child = spawn_tf(lag)
            cts, states = [], []
            nxt_pub = nxt_hb = 0.0
            end = time.monotonic() + a.trial_seconds
            published = []
            while time.monotonic() < end:
                nowm = time.monotonic()
                if nowm >= nxt_pub:
                    published.append(round(sim_now(), 3)); publish_depth(); nxt_pub = nowm + 0.1
                if nowm >= nxt_hb:
                    heartbeat(); nxt_hb = nowm + 0.1
                tick_clock(); rclpy.spin_once(node, timeout_sec=0.02)
                if status_box['msg']:
                    snap = json.loads(status_box['msg'])
                    states.append(snap.get('state'))
                    ct = snap.get('collision_time')
                    if isinstance(ct, (int, float)) and ct > 0:
                        cts.append(round(float(ct), 3))
            after = esdf_probe()
            try:
                os.killpg(os.getpgid(tf_child.pid), _signal.SIGTERM)
            except Exception:
                pass
            tf_child = None
            end_stamp = sim_now()
            s_before = (before or {}).get('integration_stamp')
            s_after = (after or {}).get('integration_stamp')
            v_before = (before or {}).get('observed_voxels')
            v_after = (after or {}).get('observed_voxels')
            # Integrated iff the fused-frame stamp advanced into this trial's published window
            # (or observed voxels increased from a cold baseline).
            stamp_advanced = bool(s_after is not None and s_before is not None
                                  and s_after > s_before + 1e-3
                                  and s_after >= start_stamp - 0.3)
            voxels_grew = bool(v_after is not None and v_before is not None and v_after > v_before)
            rec['trials'].append({
                'lag_ms': lag, 'integrated': bool(stamp_advanced or voxels_grew),
                'stamp_advanced': stamp_advanced, 'voxels_grew': voxels_grew,
                'integration_stamp_before': s_before, 'integration_stamp_after': s_after,
                'observed_before': v_before, 'observed_after': v_after,
                'trial_stamp_window': [round(start_stamp, 3), round(end_stamp, 3)],
                'published_first_last': [published[0], published[-1]] if published else None,
                'ack_states': sorted(set(s for s in states if s)),
                'ack_collision_times': sorted(set(cts))[:6],
                'ack_collision_time_advanced': len(set(cts)) > 1})
    except BaseException as exc:
        rec['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if tf_child is not None:
            try:
                os.killpg(os.getpgid(tf_child.pid), _signal.SIGTERM)
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
    (a.output / 'tf_race_probe.json').write_text(json.dumps(rec, indent=2, default=str) + '\n')
    print(json.dumps(rec, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
