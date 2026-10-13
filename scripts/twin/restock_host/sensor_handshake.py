"""ROS 2 sensor + handshake driver for the restock_mapper_evidence producer.

Runs beside an already-launched ``EvidenceProducerNode`` (launch_evidence_mapper.py
--execute) in DDS domain 88 and drives one empty-started generation all the way to
a measured incorporation identity:

  1. publishes ``/clock`` (the mapper runs ``use_sim_time:=true``) and the static
     ``map -> optical_frame`` / ``map -> base_link`` transforms nvblox needs;
  2. calls EmptyGeneration (certifies the map empty) then OpenAdmission (fixes the
     per-sensor bindings and the acquisition cutoff);
  3. each cycle: advances sim time, calls RegisterAcquisition for the frame it is
     about to publish, then publishes a synthetic depth image + camera_info on the
     generation input topics so nvblox integrates it and raises the ESDF revision;
  4. consumes the resulting ``EvidenceSnapshot``, decodes it through the reference
     ``evidence_bridge`` and feeds it to ``twin.navigation.temporal.TemporalMap``
     (observing each registered acquisition first, as the consumer requires), and
     records the TemporalMap's adopted incorporation identity.

The synthetic depth is a FIXTURE sensor source, not Isaac perception: it proves the
whole empty->admit->register->integrate->snapshot->consume path and the measured
incorporation identity end to end. The same driver accepts an Isaac depth source by
replacing ``_depth_frame`` once the native camera graph is wired; nothing else about
the handshake changes. Every mapper service ``failure_reason`` and the per-step
timeline are written to result.json so a bounded GPU run is diagnostic even when a
step is refused.

Pure sequencing/clock invariants live in incorporation_driver.py (CPU tested); this
module adds only the ROS wiring and the native integration it cannot invent.
"""
import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

from twin.restock_host.incorporation_driver import (
    IncorporationSequencer,
    MapIdentity,
    SensorBinding,
    STEADY_CLOCK_DOMAIN,
)


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
    qx = (r[2, 1] - r[1, 2]) / (4 * w)
    qy = (r[0, 2] - r[2, 0]) / (4 * w)
    qz = (r[1, 0] - r[0, 1]) / (4 * w)
    return float(qx), float(qy), float(qz), float(w)


class SensorHandshakeDriver:
    def __init__(self, *, args):
        self.args = args
        self.record = {'steps': [], 'measured': {}, 'service_failures': {}}
        self._snapshot = None

    def log(self, event, **kw):
        self.record['steps'].append({'t': round(time.monotonic(), 4), 'event': event, **kw})

    # ---- ROS message construction ------------------------------------------
    def _ros(self):
        import rclpy
        from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy,
                               qos_profile_sensor_data)
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image, CameraInfo
        from tf2_msgs.msg import TFMessage
        from geometry_msgs.msg import TransformStamped
        from builtin_interfaces.msg import Time
        from restock_mapper_evidence.srv import (EmptyGeneration, OpenAdmission,
                                                 RegisterAcquisition)
        from restock_mapper_evidence.msg import (EvidenceSnapshot, MapIdentity as RMapIdentity,
                                                 SensorBinding as RSensorBinding, Acquisition as RAcq)
        return dict(rclpy=rclpy, QoSProfile=QoSProfile, QoSReliabilityPolicy=QoSReliabilityPolicy,
                    QoSDurabilityPolicy=QoSDurabilityPolicy, qos_sensor=qos_profile_sensor_data,
                    Clock=Clock, Image=Image, CameraInfo=CameraInfo, TFMessage=TFMessage,
                    TransformStamped=TransformStamped, Time=Time, EmptyGeneration=EmptyGeneration,
                    OpenAdmission=OpenAdmission, RegisterAcquisition=RegisterAcquisition,
                    EvidenceSnapshot=EvidenceSnapshot, RMapIdentity=RMapIdentity,
                    RSensorBinding=RSensorBinding, RAcq=RAcq)

    def _time_msg(self, m, ns):
        return m['Time'](sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))

    def _identity_msg(self, m, ident):
        return m['RMapIdentity'](session=ident.session, scene=ident.scene,
                                 cameras=ident.cameras, localization=ident.localization)

    def _depth_frame(self, width, height, metres):
        """Synthetic fixture: a flat plane `metres` ahead (32FC1, little-endian)."""
        arr = np.full((height, width), float(metres), dtype='<f4')
        return arr.tobytes()

    # ---- main run ----------------------------------------------------------
    def run(self):
        a = self.args
        m = self._ros()
        rclpy = m['rclpy']
        rclpy.init(args=None)
        node = rclpy.create_node('restock_sensor_handshake')
        ns = f'/restock_sim/navigation/input/{a.generation}'
        svc = '/restock_sim/navigation/nvblox_node'

        clock_pub = node.create_publisher(m['Clock'], '/clock', 10)
        tf_static_qos = m['QoSProfile'](depth=1, reliability=m['QoSReliabilityPolicy'].RELIABLE,
                                        durability=m['QoSDurabilityPolicy'].TRANSIENT_LOCAL)
        tf_static_pub = node.create_publisher(m['TFMessage'], '/tf_static', tf_static_qos)
        tf_pub = node.create_publisher(m['TFMessage'], '/tf', 10)
        depth_pub = node.create_publisher(m['Image'], f'{ns}/camera_0/depth', m['qos_sensor'])
        info_pub = node.create_publisher(m['CameraInfo'], f'{ns}/camera_0/camera_info', m['qos_sensor'])

        snap_qos = m['QoSProfile'](depth=1, reliability=m['QoSReliabilityPolicy'].RELIABLE,
                                   durability=m['QoSDurabilityPolicy'].TRANSIENT_LOCAL)

        def on_snapshot(msg):
            self._snapshot = msg
        node.create_subscription(m['EvidenceSnapshot'], f'{svc}/evidence_snapshot', on_snapshot, snap_qos)

        empty_cli = node.create_client(m['EmptyGeneration'], f'{svc}/empty_generation')
        open_cli = node.create_client(m['OpenAdmission'], f'{svc}/open_admission')
        reg_cli = node.create_client(m['RegisterAcquisition'], f'{svc}/register_acquisition')

        identity = MapIdentity(a.session, a.scene, a.cameras, a.localization)
        sensors = [SensorBinding(*s.split('=', 1)) for s in a.sensors]
        seq = IncorporationSequencer(generation=a.generation, challenge=a.challenge,
                                     identity=identity, sensors=sensors)

        # Camera intrinsics (pinhole), one synthetic front camera.
        W, H = a.width, a.height
        fx = fy = a.focal
        cx, cy = W / 2.0, H / 2.0
        optical = sensors[0].optical_frame

        def spin(dt):
            end = time.monotonic() + dt
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.01)

        def wait_future(fut, timeout):
            end = time.monotonic() + timeout
            while rclpy.ok() and not fut.done() and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.02)
            return fut.result() if fut.done() else None

        sim_ns = [int(1_000_000_000)]  # sim time starts at 1.0 s

        def publish_clock():
            msg = m['Clock']()
            msg.clock = self._time_msg(m, sim_ns[0])
            clock_pub.publish(msg)

        def tf_message(stamp_ns):
            tfm = m['TFMessage']()
            qx, qy, qz, qw = _quat_from_axes(forward=(1, 0, 0), up=(0, 0, 1))
            for child, (px, py, pz, q) in {
                    optical: (0.0, 0.0, 1.0, (qx, qy, qz, qw)),
                    'base_link': (0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 1.0))}.items():
                t = m['TransformStamped']()
                t.header.stamp = self._time_msg(m, stamp_ns)
                t.header.frame_id = 'map'
                t.child_frame_id = child
                t.transform.translation.x = px
                t.transform.translation.y = py
                t.transform.translation.z = pz
                (t.transform.rotation.x, t.transform.rotation.y,
                 t.transform.rotation.z, t.transform.rotation.w) = q
                tfm.transforms.append(t)
            return tfm

        try:
            # Certify emptiness FIRST: publish no clock/TF/depth yet, so nothing can
            # allocate a native layer block before the empty-generation + admission
            # checks (both run State::verify_empty). Only after admission is open do
            # we start the clock, TF and synthetic depth that populate the map.
            for cli, name in ((empty_cli, 'empty'), (open_cli, 'open'), (reg_cli, 'register')):
                if not cli.wait_for_service(timeout_sec=a.service_timeout):
                    self.record['measured']['service_available'] = False
                    self.log('service_unavailable', service=name)
                    raise RuntimeError(f'{name} service not available')
            self.record['measured']['service_available'] = True

            # ---- optional diagnostic: poll EmptyGeneration to time map emptiness ----
            if a.probe_empty_count > 0:
                probe = []
                for i in range(a.probe_empty_count):
                    req = m['EmptyGeneration'].Request()
                    req.challenge = f'probe-{i}'
                    req.expected_identity = self._identity_msg(m, identity)
                    r = wait_future(empty_cli.call_async(req), a.service_timeout)
                    probe.append({'i': i, 'wall': round(time.monotonic(), 3),
                                  'success': bool(r.success) if r else None,
                                  'reason': getattr(r, 'failure_reason', 'timeout') if r else 'timeout',
                                  'layers': [{'name': lc.layer_name, 'blocks': int(lc.allocated_blocks)}
                                             for lc in r.layers] if (r and r.success) else []})
                    spin(a.probe_interval)
                self.record['measured']['empty_probe'] = probe
                self.record['measured']['ever_empty'] = any(p['success'] for p in probe)
                self.log('empty_probe_done', ever_empty=self.record['measured']['ever_empty'])
                return self.record

            # ---- EmptyGeneration (map still empty: nothing published yet) ----
            req = m['EmptyGeneration'].Request()
            er = seq.empty_request()
            req.challenge = er['challenge']
            req.expected_identity = self._identity_msg(m, identity)
            resp = wait_future(empty_cli.call_async(req), a.service_timeout)
            if resp is None or not resp.success:
                self.record['service_failures']['empty_generation'] = (
                    getattr(resp, 'failure_reason', 'no response') if resp else 'timeout')
                self.log('empty_failed', reason=self.record['service_failures']['empty_generation'])
                raise RuntimeError('empty generation refused')
            seq.certified(producer_nonce=resp.producer_nonce, certificate_token=resp.certificate_token)
            self.record['measured']['producer_nonce'] = resp.producer_nonce
            self.record['measured']['certificate_token'] = resp.certificate_token
            self.record['measured']['empty_layers'] = [
                {'name': lc.layer_name, 'type': lc.layer_type, 'blocks': int(lc.allocated_blocks)}
                for lc in resp.layers]
            self.log('empty_certified', producer_nonce=resp.producer_nonce)

            # ---- OpenAdmission (map still empty) ----
            # Cutoff at sim 0: every acquisition stamp below (>= sim baseline) is
            # strictly newer. The map must still carry zero blocks here.
            cutoff_stamp = 0
            cutoff_steady = time.monotonic_ns()
            orq = seq.open_request(cutoff_stamp_ns=cutoff_stamp, cutoff_steady_ns=cutoff_steady)
            req = m['OpenAdmission'].Request()
            req.producer_nonce = orq['producer_nonce']
            req.certificate_token = orq['certificate_token']
            req.identity = self._identity_msg(m, identity)
            req.required_sensors = [m['RSensorBinding'](sensor=b['sensor'], optical_frame=b['optical_frame'])
                                    for b in orq['required_sensors']]
            req.acquisition_cutoff_stamp = self._time_msg(m, cutoff_stamp)
            req.acquisition_cutoff_steady_ns = cutoff_steady
            req.steady_clock_domain = orq['steady_clock_domain']
            resp = wait_future(open_cli.call_async(req), a.service_timeout)
            if resp is None or not resp.success:
                self.record['service_failures']['open_admission'] = (
                    getattr(resp, 'failure_reason', 'no response') if resp else 'timeout')
                self.log('open_failed', reason=self.record['service_failures']['open_admission'])
                raise RuntimeError('open admission refused')
            self.log('admission_open')

            # Admission is open: NOW start the sim clock and publish the static TF
            # nvblox needs to project depth into the map frame.
            for _ in range(3):
                publish_clock()
                tf_static_pub.publish(tf_message(sim_ns[0]))
                tf_pub.publish(tf_message(sim_ns[0]))
                spin(0.05)
            self.log('clock_tf_started', sim_ns=sim_ns[0])

            # ---- consumer TemporalMap (observes each registered acquisition) ----
            from twin.navigation.temporal import TemporalMap, MapIdentity as CMapIdentity
            from twin.restock_host import evidence_bridge
            tmap = TemporalMap(
                identity=CMapIdentity(a.session, a.scene, a.cameras, a.localization),
                commissioned=cutoff_stamp / 1e9, required_sensors=[b.sensor for b in sensors],
                generation=a.generation, frame='map', duration=600., stale_after=.3,
                steady_domain=STEADY_CLOCK_DOMAIN)

            info = m['CameraInfo']()
            info.width, info.height = W, H
            info.distortion_model = 'plumb_bob'
            info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            depth_bytes = self._depth_frame(W, H, a.depth_metres)

            registered = 0
            deadline = time.monotonic() + a.window
            while rclpy.ok() and time.monotonic() < deadline and self._snapshot is None:
                sim_ns[0] += int(a.sim_step * 1e9)
                publish_clock()
                tf_pub.publish(tf_message(sim_ns[0]))
                spin(0.02)  # let the mapper advance its sim clock past the stamp

                stamp_ns = sim_ns[0] - 20_000_000  # 20 ms before "now" for sim-clock propagation lag
                steady_ns = time.monotonic_ns()
                for b in sensors:
                    rr = seq.register_request(sensor=b.sensor, stamp_ns=stamp_ns, steady_ns=steady_ns,
                                              now_ns=sim_ns[0], steady_now_ns=time.monotonic_ns())
                    req = m['RegisterAcquisition'].Request()
                    req.producer_nonce = rr['producer_nonce']
                    req.certificate_token = rr['certificate_token']
                    acq = m['RAcq']()
                    ad = rr['acquisition']
                    acq.identity = self._identity_msg(m, identity)
                    acq.sensor = ad['sensor']
                    acq.optical_frame = ad['optical_frame']
                    acq.sequence = ad['sequence']
                    acq.acquisition_stamp = self._time_msg(m, ad['acquisition_stamp_ns'])
                    acq.acquisition_steady_ns = ad['acquisition_steady_ns']
                    acq.steady_clock_domain = ad['steady_clock_domain']
                    req.acquisition = acq
                    resp = wait_future(reg_cli.call_async(req), a.service_timeout)
                    if resp is None or not resp.success:
                        reason = getattr(resp, 'failure_reason', 'no response') if resp else 'timeout'
                        self.record['service_failures'].setdefault('register_acquisition', reason)
                        self.log('register_failed', sensor=b.sensor, reason=reason)
                        continue
                    registered += 1
                    # Consumer record from the SAME message -> float-identical to the echoed ledger.
                    tmap.observe(evidence_bridge.acquisition_from_message(acq))

                    # Publish the depth frame this acquisition declared.
                    img = m['Image']()
                    img.header.stamp = self._time_msg(m, ad['acquisition_stamp_ns'])
                    img.header.frame_id = b.optical_frame
                    img.height, img.width = H, W
                    img.encoding = '32FC1'
                    img.is_bigendian = 0
                    img.step = W * 4
                    img.data = depth_bytes
                    info.header = img.header
                    depth_pub.publish(img)
                    info_pub.publish(info)
                spin(a.integrate_wait)

            self.record['measured']['registered_acquisitions'] = registered
            self.record['measured']['snapshot_received'] = self._snapshot is not None

            if self._snapshot is not None:
                snap = self._snapshot
                self.record['measured']['snapshot'] = {
                    'producer_nonce': snap.producer_nonce,
                    'certificate_token': snap.certificate_token,
                    'snapshot_index': int(snap.snapshot_index),
                    'esdf_revision': int(snap.esdf_revision),
                    'admission_state': int(snap.admission_state),
                    'empty_state': int(snap.empty_state),
                    'full_height_observation_verified': bool(snap.full_height_observation_verified),
                    'ledger': [{'sensor': ac.sensor, 'optical_frame': ac.optical_frame,
                                'sequence': int(ac.sequence)} for ac in snap.incorporated],
                    'slice': {'width': int(snap.slice.width), 'height': int(snap.slice.height),
                              'resolution': float(snap.slice.resolution)},
                }
                envelope = evidence_bridge.envelope_from_message(snap)
                tmap.accept_envelope(envelope)
                self.record['measured']['incorporation_identity'] = {
                    'generation': tmap.generation,
                    'producer_nonce': tmap.producer_nonce,
                    'certificate_token': tmap.certificate_token,
                    'steady_clock_domain': tmap.steady_domain,
                    'snapshot_index': tmap.snapshot_index,
                    'esdf_revision': tmap.esdf_revision,
                    'sensor_bindings': dict(tmap.sensor_bindings),
                    'ledger_sensors': sorted(tmap.ledger.keys()),
                    'slice_adopted': tmap.slice is not None,
                    'full_height_certified': tmap.full_height_certified(),
                }
                self.record['measured']['CONSUMED'] = True
                self.log('incorporation_identity_measured', **self.record['measured']['incorporation_identity'])
            else:
                self.record['measured']['CONSUMED'] = False
                self.log('no_snapshot_within_window', registered=registered)
        except BaseException as exc:  # record SIGINT/teardown too, not just Exception
            self.record['error'] = f'{type(exc).__name__}: {exc}'
            self.log('error', detail=self.record['error'])
        finally:
            try:
                (a.output / 'handshake_result.json').write_text(
                    json.dumps(self.record, indent=2, default=str) + '\n')
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
        return self.record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--session', default='mission_a')
    p.add_argument('--scene', default='supermarket')
    p.add_argument('--cameras', default='front')
    p.add_argument('--localization', default='amcl')
    p.add_argument('--sensors', nargs='+', default=['front=front_optical'])
    p.add_argument('--challenge', default=None)
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--height', type=int, default=240)
    p.add_argument('--focal', type=float, default=200.0)
    p.add_argument('--depth-metres', type=float, default=2.0)
    p.add_argument('--sim-step', type=float, default=0.1)
    p.add_argument('--integrate-wait', type=float, default=0.08)
    p.add_argument('--service-timeout', type=float, default=20.0)
    p.add_argument('--window', type=float, default=90.0)
    p.add_argument('--probe-empty-count', type=int, default=0,
                   help='diagnostic: poll EmptyGeneration N times and exit (no admission)')
    p.add_argument('--probe-interval', type=float, default=1.0)
    a = p.parse_args()
    if a.challenge is None:
        a.challenge = f'challenge-{int(time.time())}'
    a.output.mkdir(parents=True, exist_ok=True)
    driver = SensorHandshakeDriver(args=a)
    record = driver.run()
    (a.output / 'handshake_result.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    print(json.dumps({k: v for k, v in record.get('measured', {}).items()}, indent=2, default=str))
    return 0 if record.get('measured', {}).get('CONSUMED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
