#!/usr/bin/env python3
"""Full native incorporation pipeline + landed-consumer acceptance probe.

Runs beside an already-launched EvidenceProducerNode (launch_evidence_mapper.py
--execute) in DDS domain 88 and drives one empty-started generation end to end,
then feeds the published EvidenceSnapshot to the LANDED navigation consumer and
records whether it ACCEPTS the slice:

  1. EmptyGeneration  -> certify the fresh map empty (baseline reported).
  2. OpenAdmission    -> open admission, fix the sensor binding and the cutoff.
  3. per cycle        -> advance /clock, RegisterAcquisition the frame about to be
                         published, publish a synthetic depth frame + camera_info +
                         the static map->optical_frame TF so nvblox integrates it and
                         raises the ESDF revision.
  4. consume          -> decode the snapshot with twin.restock_host.evidence_bridge
                         and call twin.navigation.temporal.TemporalMap.accept_envelope
                         (observing each registered acquisition first). Records the
                         adopted slice frame and whether acceptance succeeded.

This exists to verify one producer fix: processEsdf now sets the embedded slice's
header frame_id to the mapper global frame (and a ledger-consistent stamp), so the
landed consumer no longer rejects the snapshot as an "invalid native distance
slice". The synthetic depth is a FIXTURE sensor source; this is a measurement
probe, never commissioning. The sequencing/clock invariants mirror
native/restock_mapper_evidence/state.hpp.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

STEADY_CLOCK_DOMAIN = 'host_monotonic_shared'  # == mapper_launch plan evidence domain
FRESHNESS_NS = 300_000_000


def _quat_from_axes(forward, up):
    z = np.asarray(forward, float); z /= np.linalg.norm(z)
    down = -np.asarray(up, float)
    x = np.cross(down, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.column_stack([x, y, z]); t = np.trace(r)
    w = np.sqrt(max(0.0, 1.0 + t)) / 2.0
    return ((r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w),
            (r[1, 0] - r[0, 1]) / (4 * w), w)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--session', default='mission_a')
    p.add_argument('--scene', default='supermarket')
    p.add_argument('--cameras', default='front')
    p.add_argument('--localization', default='amcl')
    p.add_argument('--sensor', default='front')
    p.add_argument('--optical-frame', default='front_optical')
    p.add_argument('--challenge', default=None)
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--height', type=int, default=240)
    p.add_argument('--focal', type=float, default=200.0)
    p.add_argument('--depth-metres', type=float, default=2.0)
    p.add_argument('--sim-step', type=float, default=0.1)
    p.add_argument('--integrate-wait', type=float, default=0.08)
    p.add_argument('--service-timeout', type=float, default=25.0)
    p.add_argument('--window', type=float, default=90.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.challenge is None:
        a.challenge = f'consume-{int(time.time())}'

    import rclpy
    from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy,
                           qos_profile_sensor_data)
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Image, CameraInfo
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import TransformStamped
    from builtin_interfaces.msg import Time
    from restock_mapper_evidence.srv import EmptyGeneration, OpenAdmission, RegisterAcquisition
    from restock_mapper_evidence.msg import (EvidenceSnapshot, MapIdentity as RMapIdentity,
                                             SensorBinding as RSensorBinding, Acquisition as RAcq)
    from twin.navigation.temporal import TemporalMap, MapIdentity as CMapIdentity, ADMISSION_OPEN
    from twin.restock_host import evidence_bridge

    rec = {'generation': a.generation, 'challenge': a.challenge, 'steps': [], 'service_failures': {}}

    def log(ev, **kw):
        rec['steps'].append({'t': round(time.monotonic(), 3), 'event': ev, **kw})

    rclpy.init(args=None)
    node = rclpy.create_node('consume_snapshot_probe')
    ns = f'/restock_sim/navigation/input/{a.generation}'
    svc = '/restock_sim/navigation/nvblox_node'

    def tmsg(ns_):
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

    def rid(ident):
        return RMapIdentity(session=ident.session, scene=ident.scene,
                            cameras=ident.cameras, localization=ident.localization)

    identity = CMapIdentity(a.session, a.scene, a.cameras, a.localization)
    snapshot = [None]
    snap_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(EvidenceSnapshot, f'{svc}/evidence_snapshot',
                             lambda m: snapshot.__setitem__(0, m), snap_qos)
    clock_pub = node.create_publisher(Clock, '/clock', 10)
    tf_static_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                               durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    tf_static_pub = node.create_publisher(TFMessage, '/tf_static', tf_static_qos)
    tf_pub = node.create_publisher(TFMessage, '/tf', 10)
    depth_pub = node.create_publisher(Image, f'{ns}/camera_0/depth', qos_profile_sensor_data)
    info_pub = node.create_publisher(CameraInfo, f'{ns}/camera_0/camera_info', qos_profile_sensor_data)
    empty_cli = node.create_client(EmptyGeneration, f'{svc}/empty_generation')
    open_cli = node.create_client(OpenAdmission, f'{svc}/open_admission')
    reg_cli = node.create_client(RegisterAcquisition, f'{svc}/register_acquisition')

    W, H = a.width, a.height
    fx = fy = a.focal
    cx, cy = W / 2.0, H / 2.0
    qx, qy, qz, qw = _quat_from_axes(forward=(1, 0, 0), up=(0, 0, 1))
    sim_ns = int(1_000_000_000)

    def tf_message(stamp_ns):
        tfm = TFMessage()
        for child, (px, py, pz, q) in {
                a.optical_frame: (0.0, 0.0, 1.0, (qx, qy, qz, qw)),
                'base_link': (0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 1.0))}.items():
            t = TransformStamped()
            t.header.stamp = tmsg(stamp_ns)
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

    try:
        for cli, name in ((empty_cli, 'empty'), (open_cli, 'open'), (reg_cli, 'register')):
            if not cli.wait_for_service(timeout_sec=a.service_timeout):
                raise RuntimeError(f'{name} service unavailable')

        # 1. EmptyGeneration (map still empty: nothing published yet).
        req = EmptyGeneration.Request()
        req.challenge = a.challenge
        req.expected_identity = rid(identity)
        resp = wait_future(empty_cli.call_async(req), a.service_timeout)
        if resp is None or not resp.success:
            rec['service_failures']['empty_generation'] = getattr(resp, 'failure_reason', 'timeout') if resp else 'timeout'
            raise RuntimeError('empty generation refused')
        producer_nonce = resp.producer_nonce
        certificate_token = resp.certificate_token
        rec['empty_baseline'] = [{'name': l.layer_name, 'blocks': int(l.allocated_blocks)} for l in resp.layers]
        log('empty_certified', producer_nonce=producer_nonce)

        # 2. OpenAdmission (map still empty). Cutoff at sim 0; every acquisition after.
        cutoff_stamp, cutoff_steady = 0, time.monotonic_ns()
        req = OpenAdmission.Request()
        req.producer_nonce = producer_nonce
        req.certificate_token = certificate_token
        req.identity = rid(identity)
        req.required_sensors = [RSensorBinding(sensor=a.sensor, optical_frame=a.optical_frame)]
        req.acquisition_cutoff_stamp = tmsg(cutoff_stamp)
        req.acquisition_cutoff_steady_ns = cutoff_steady
        req.steady_clock_domain = STEADY_CLOCK_DOMAIN
        resp = wait_future(open_cli.call_async(req), a.service_timeout)
        if resp is None or not resp.success:
            rec['service_failures']['open_admission'] = getattr(resp, 'failure_reason', 'timeout') if resp else 'timeout'
            raise RuntimeError('open admission refused')
        log('admission_open')

        tmap = TemporalMap(identity, commissioned=cutoff_stamp / 1e9,
                           required_sensors=[a.sensor], generation=producer_nonce, frame='map',
                           duration=600., stale_after=.3, steady_domain=STEADY_CLOCK_DOMAIN)

        for _ in range(3):
            c = Clock(); c.clock = tmsg(sim_ns); clock_pub.publish(c)
            tf_static_pub.publish(tf_message(sim_ns)); tf_pub.publish(tf_message(sim_ns))
            spin(0.05)

        seq = 0
        registered = 0
        deadline = time.monotonic() + a.window
        while rclpy.ok() and time.monotonic() < deadline and snapshot[0] is None:
            sim_ns += int(a.sim_step * 1e9)
            c = Clock(); c.clock = tmsg(sim_ns); clock_pub.publish(c)
            tf_pub.publish(tf_message(sim_ns))
            spin(0.02)
            stamp_ns = sim_ns - 20_000_000
            steady_ns = time.monotonic_ns()
            seq += 1
            acq = RAcq()
            acq.identity = rid(identity)
            acq.sensor = a.sensor
            acq.optical_frame = a.optical_frame
            acq.sequence = seq
            acq.acquisition_stamp = tmsg(stamp_ns)
            acq.acquisition_steady_ns = steady_ns
            acq.steady_clock_domain = STEADY_CLOCK_DOMAIN
            req = RegisterAcquisition.Request()
            req.producer_nonce = producer_nonce
            req.certificate_token = certificate_token
            req.acquisition = acq
            resp = wait_future(reg_cli.call_async(req), a.service_timeout)
            if resp is None or not resp.success:
                rec['service_failures'].setdefault('register_acquisition',
                                                   getattr(resp, 'failure_reason', 'timeout') if resp else 'timeout')
                continue
            registered += 1
            # Consumer-side record from the SAME message -> float-identical to the echoed ledger.
            tmap.observe(evidence_bridge.acquisition_from_message(acq))
            img = Image()
            img.header.stamp = tmsg(stamp_ns)
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

        rec['registered_acquisitions'] = registered
        rec['snapshot_received'] = snapshot[0] is not None
        if snapshot[0] is None:
            raise RuntimeError('no EvidenceSnapshot within window')

        snap = snapshot[0]
        rec['snapshot'] = {
            'snapshot_index': int(snap.snapshot_index), 'esdf_revision': int(snap.esdf_revision),
            'admission_state': int(snap.admission_state), 'empty_state': int(snap.empty_state),
            'slice_frame_id': snap.slice.header.frame_id,
            'slice_stamp': {'sec': snap.slice.header.stamp.sec, 'nanosec': snap.slice.header.stamp.nanosec},
            'slice_wh': [int(snap.slice.width), int(snap.slice.height)],
            'slice_resolution': float(snap.slice.resolution),
            'data_len': len(snap.slice.data),
            'ledger': [{'sensor': ac.sensor, 'optical_frame': ac.optical_frame, 'sequence': int(ac.sequence)}
                       for ac in snap.incorporated],
        }
        log('snapshot_received', frame_id=snap.slice.header.frame_id,
            snapshot_index=int(snap.snapshot_index))

        # The fix under test: with the slice frame_id set, decode + consumer accept.
        envelope = evidence_bridge.envelope_from_message(snap)
        tmap.accept_envelope(envelope)
        rec['consumer_accepted'] = True
        rec['adopted'] = {
            'slice_frame': tmap.slice.frame,
            'slice_bound_to': {'sensor': tmap.slice.incorporated.sensor,
                               'sequence': tmap.slice.incorporated.sequence},
            'producer_nonce': tmap.producer_nonce,
            'certificate_token': tmap.certificate_token,
            'snapshot_index': tmap.snapshot_index,
            'esdf_revision': tmap.esdf_revision,
            'sensor_bindings': dict(tmap.sensor_bindings),
            'admission_open': envelope.admission_state == ADMISSION_OPEN,
            'full_height_certified': tmap.full_height_certified(),
        }
        log('consumer_accepted', slice_frame=tmap.slice.frame)
    except BaseException as exc:
        rec['error'] = f'{type(exc).__name__}: {exc}'
        rec.setdefault('consumer_accepted', False)
        log('error', detail=rec['error'])
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
    (a.output / 'consume_result.json').write_text(json.dumps(rec, indent=2, default=str) + '\n')
    print(json.dumps(rec, indent=2, default=str))
    return 0 if rec.get('consumer_accepted') else 2


if __name__ == '__main__':
    raise SystemExit(main())
