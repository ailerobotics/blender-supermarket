"""Isaac-side rig-masked depth bridge for the host nvblox consumer.

Option B for the manipulation collision world: instead of the host `robot_segmenter`
node masking the robot out of the depth stream (which currently fails to load on a
duplicate NITROS subscription), publish a depth image from the Isaac side with the
robot/cart (the "rig") self-pixels already invalidated, using the per-frame instance-ID
segmentation the mission already captures (`add_instance_id_segmentation_to_frame`).
nvblox then integrates only non-rig returns, so the ESDF/planning scene is built without
the robot carving itself into the obstacle set.

The masking core (`rig_instance_ids`, `mask_rig_depth`, `parse_instance_frame`,
`select_target_bbox`) is pure NumPy and CPU-tested. Publishing uses OmniGraph ROS2 nodes
(`ArmDepthOmniGraphBridge`), NOT rclpy: an rclpy Node created inside the Isaac SimulationApp
aborts (rcl_interfaces parameter_event typesupport assert) because the isaacsim.ros2.bridge
extension and full rclpy load conflicting ROS2 middleware. The OmniGraph path mirrors
`isaac_ros_graph.py` / `native_cameras.py` / `native_transport.py` (the proven in-app
pattern). The bridge is wired only when the runtime-config opts in (`masked_depth` block).

Contract with the host worker (host PR#50):
  depth topic : /restock_sim/camera/masked_depth        sensor_msgs/Image, 32FC1, metres
  info  topic : /restock_sim/camera/masked_depth/camera_info   sensor_msgs/CameraInfo
  bbox  topic : /restock_sim/target_bbox                 std_msgs/Float32MultiArray [u1,v1,u2,v2]
  frame_id    : restock_camera_arm (optical frame, added to native_transport's /tf tree)
  invalid     : 0.0 (nvblox treats 0 / non-finite depth as no-return)
  domain      : 88 (the mission's isolated ROS domain)
  stamp       : mission sim time; masked_depth + bbox published on one acquisition stamp,
                TF (restock_camera_arm -> ur_base_link) published every physics tick by
                native_transport's ROS2PublishTransformTree + /clock.
"""
from __future__ import annotations

import math

import numpy as np

INVALID_DEPTH = 0.0  # nvblox no-return sentinel

# USD camera looks down -Z with +Y up; the ROS optical frame looks +Z with +Y down.
# Converting a point from the optical frame to the USD-camera frame flips Y and Z.
_USDCAM_FROM_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])


def mat_to_quat_xyzw(rotation):
    """3x3 rotation matrix (column convention, p' = R @ p) -> (x, y, z, w) quaternion."""
    r = np.asarray(rotation, float)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s, (r[1, 0]-r[0, 1])/s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w, x, y, z = (r[2, 1]-r[1, 2])/s, 0.25 * s, (r[0, 1]+r[1, 0])/s, (r[0, 2]+r[2, 0])/s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w, x, y, z = (r[0, 2]-r[2, 0])/s, (r[0, 1]+r[1, 0])/s, 0.25 * s, (r[1, 2]+r[2, 1])/s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w, x, y, z = (r[1, 0]-r[0, 1])/s, (r[0, 2]+r[2, 0])/s, (r[1, 2]+r[2, 1])/s, 0.25 * s
    return (x, y, z, w)


def base_from_optical_transform(cam_world_col, base_world_col):
    """Pose of the camera OPTICAL frame in ur_base_link as (translation xyz, quat xyzw).

    Both inputs are 4x4 local-to-world matrices in COLUMN convention (p_world = M @ p_local).
    The USD camera's world matrix is converted to the ROS optical frame, then expressed in
    the base frame. This is the per-frame TF the host nvblox needs for the moving eye-in-hand
    camera; isolating the convention math here keeps it unit-tested off the live stage.
    """
    cam = np.asarray(cam_world_col, float)
    base = np.asarray(base_world_col, float)
    optical_world = cam @ _USDCAM_FROM_OPTICAL
    base_from_opt = np.linalg.inv(base) @ optical_world
    return tuple(base_from_opt[:3, 3]), mat_to_quat_xyzw(base_from_opt[:3, :3])


def ros_time(seconds):
    """builtin_interfaces/Time for a (sim) seconds float; lazy import for CPU tests."""
    from builtin_interfaces.msg import Time
    sec = int(seconds)
    return Time(sec=sec, nanosec=int(round((float(seconds) - sec) * 1e9)))


def parse_instance_frame(frame):
    """Extract (id_image, {instance_id: prim_path}) from an Isaac instance-ID seg frame.

    Accepts the `instance_id_segmentation` entry produced with colorize=False: a dict with
    an integer-id `data` image and an `info.idToLabels` map from stringified id to prim
    path (values may already be plain path strings or small dicts). Returns the id image as
    a 2-D int array and a normalized {int id: str path} mapping.
    """
    seg = frame.get('instance_id_segmentation') if isinstance(frame, dict) else None
    if not isinstance(seg, dict):
        raise ValueError('frame lacks instance_id_segmentation (enable it on the camera)')
    data = seg.get('data')
    ids = np.asarray(data)
    if ids.ndim != 2 or ids.size == 0:
        raise ValueError('instance_id_segmentation data must be a non-empty 2-D id image')
    info = seg.get('info') or {}
    raw = info.get('idToLabels') or info.get('idToSemantics') or {}
    if not isinstance(raw, dict):
        raise ValueError('instance_id_segmentation info.idToLabels must be a mapping')
    mapping = {}
    for key, value in raw.items():
        try:
            ik = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            # Some Isaac builds wrap the path, e.g. {'class': '/World/...'}.
            value = value.get('class') or value.get('instanceId') or next(
                (v for v in value.values() if isinstance(v, str)), '')
        mapping[ik] = str(value)
    return ids, mapping


def rig_instance_ids(id_to_path, rig_prefixes):
    """Instance IDs whose prim path lies under any rig (robot/cart) prefix.

    An empty or missing path (e.g. BACKGROUND, UNLABELLED) is never a rig id, so free space
    and the environment are preserved. Matching is exact-prefix on the USD path so a sibling
    like `/World/robot_dock` does not match the `/World/robot` rig unless a trailing `/`
    (or exact equality) confirms containment.
    """
    prefixes = tuple(p for p in rig_prefixes if p)
    if not prefixes:
        raise ValueError('at least one rig prim prefix is required')
    ids = set()
    for ident, path in id_to_path.items():
        if not path:
            continue
        for prefix in prefixes:
            if path == prefix or path.startswith(prefix.rstrip('/') + '/'):
                ids.add(int(ident))
                break
    return ids


def mask_rig_depth(depth, instance_ids, rig_ids, *, invalid=INVALID_DEPTH):
    """Return a depth copy with rig pixels (and non-finite returns) set to `invalid`.

    `depth` is HxW metres; `instance_ids` is the matching HxW id image; `rig_ids` is the set
    of instance IDs to invalidate. Non-finite or non-positive depths are also invalidated so
    the consumer never integrates a spurious return. Shapes must match exactly.
    """
    depth = np.asarray(depth, dtype=np.float32)
    ids = np.asarray(instance_ids)
    if depth.ndim != 2 or depth.shape != ids.shape:
        raise ValueError('depth and instance-id images must share a 2-D shape')
    out = depth.copy()
    bad = ~np.isfinite(out) | (out <= 0.0)
    if rig_ids:
        bad |= np.isin(ids, np.asarray(sorted(rig_ids)))
    out[bad] = float(invalid)
    return out

def bbox_payload(bbox):
    """Validate and flatten one grasp-target bbox to [u1, v1, u2, v2] floats.

    `bbox` is the target's axis-aligned image box as (u1, v1, u2, v2) pixel corners with
    u1<u2 and v1<v2 (the shelfbot object_segmenter masks this rectangle out of the depth so
    cuMotion can plan a path into the grasp target).
    """
    box = [float(v) for v in bbox]
    if len(box) != 4 or not all(np.isfinite(box)):
        raise ValueError('target bbox must be four finite pixel values [u1,v1,u2,v2]')
    u1, v1, u2, v2 = box
    if not (u2 > u1 and v2 > v1):
        raise ValueError('target bbox must have u1<u2 and v1<v2')
    return box


def select_target_bbox(sensed_bboxes, target_location, previous):
    """Pure pick of the grasp-target roi to publish, caching the last good one.

    `sensed_bboxes` are dicts with 'roi' and 'location' (from IsaacCartonSensors). Prefer the
    entry at `target_location`; fall back to the first available. Returns the roi tuple, or
    `previous` when perception produced none this frame (republished so the host's 0.5 s bbox
    freshness never lapses while the static target is momentarily unobserved).
    """
    boxes = list(sensed_bboxes or ())
    matched = [b for b in boxes if target_location is None or b.get('location') == target_location]
    pick = matched or boxes
    if pick:
        return tuple(pick[0]['roi'])
    return previous


def masked_depth_from_frame(frame, rig_prefixes, id_cache, *, invalid=INVALID_DEPTH):
    """(masked HxW float32 depth, refreshed id_cache). Pure; shared by the bridge and tests.

    `id_cache` is a (id_to_path_obj, frozenset rig_ids) tuple reused across frames (Isaac
    keeps the id->path dict object stable while the scene graph is), so rig IDs resolve once.
    """
    depth = frame.get('distance_to_image_plane')
    if depth is None:
        depth = frame.get('depth')
    if depth is None:
        raise ValueError('frame carries no depth (distance_to_image_plane)')
    ids, id_to_path = parse_instance_frame(frame)
    # Key the cache on the raw seg-info object (stable across frames while the scene graph is),
    # since parse_instance_frame rebuilds a fresh normalized mapping each call.
    raw = frame['instance_id_segmentation'].get('info')
    if raw is None or raw is not id_cache[0]:
        id_cache = (raw, frozenset(rig_instance_ids(id_to_path, rig_prefixes)))
    masked = mask_rig_depth(depth, ids, id_cache[1], invalid=invalid)
    return masked, id_cache

class ArmDepthSubprocessBridge:
    """Ship arm masked-depth frames to the out-of-Isaac rclpy publisher subprocess.

    Built only inside the guarded app (lazy pxr). On construction it (1) creates an optical
    child prim `restock_camera_arm` under the arm camera and hands it to native_transport's
    existing ROS2PublishTransformTree (so ur_base_link -> restock_camera_arm + /clock publish
    every physics tick), and (2) spawns `masked_depth_ipc.serve` under the host overlay ROS
    env (domain 88) and connects a Unix socket to it. `publish(now)` reads one arm-camera
    frame, masks the rig self-pixels, and ships (stamp, frame_id, w, h, K, bbox, depth) to the
    subprocess, which publishes masked_depth + camera_info + target_bbox on that one stamp.

    `sensors` exposes `sensed_target_bboxes`; set `target_location` per operation.
    """

    def __init__(self, *, camera, camera_path, stage, transport, sensors, rig_prefixes,
                 runtime_env, python='python3', frame_id='restock_camera_arm', domain=88,
                 invalid=INVALID_DEPTH, socket_path=None, idle_timeout_s=1800.0):
        from pxr import UsdGeom, Gf
        self.camera, self.sensors = camera, sensors
        self.rig_prefixes = tuple(rig_prefixes)
        if not self.rig_prefixes:
            raise ValueError('rig_prefixes required to identify robot/cart self-returns')
        self.frame_id, self.invalid = frame_id, float(invalid)
        self.target_location = None
        self._id_cache = (None, frozenset())
        self._bbox = None

        # Optical child prim (ROS +Z forward / +Y down) under the USD camera (-Z / +Y up),
        # handed to the existing TF tree.
        optical_path = camera_path + '/' + frame_id
        if not stage.GetPrimAtPath(optical_path):
            xf = UsdGeom.Xform.Define(stage, optical_path)
            xf.AddTransformOp().Set(Gf.Matrix4d(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1))
        transport.add_frames([optical_path])

        # Spawn the out-of-Isaac rclpy publisher under the host overlay env + domain 88.
        import os
        import subprocess
        from .masked_depth_ipc import FrameWriter
        self._socket_path = socket_path or f'/tmp/restock_maskdepth_{os.getpid()}.sock'
        ipc_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'masked_depth_ipc.py')
        # Match the host nodes' DDS config EXACTLY (twin.restock_host.config.isolated_environment):
        # rmw_fastrtps_cpp + UDPv4-only builtin transport + SKIP_DEFAULT_XML + localhost-only, and
        # strip any DDS profile/discovery-server overrides. The host forces UDPv4 (no shared memory);
        # a subprocess on the default SHM+UDP transport discovers the topics but never delivers data
        # to nvblox (run-37/38). Export AFTER sourcing the overlay so these win.
        command = (f'set +u; source {runtime_env}; '
                   f'export ROS_DOMAIN_ID={int(domain)} ROS_LOCALHOST_ONLY=1 ROS_STATIC_PEERS= '
                   f'ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp '
                   f'FASTDDS_BUILTIN_TRANSPORTS=UDPv4 SKIP_DEFAULT_XML=1; '
                   f'unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE '
                   f'ROS_DISCOVERY_SERVER CYCLONEDDS_URI RMW_FASTRTPS_USE_QOS_FROM_XML; '
                   f'exec {python} {ipc_script} serve {self._socket_path} {float(idle_timeout_s)}')
        # Spawn under a CLEAN base env, NOT the Isaac process env: Isaac loads its own bundled
        # ("internal") jazzy rclpy/rmw and sets AMENT/PYTHONPATH/LD_LIBRARY_PATH to it; inheriting
        # that and sourcing the host /opt/ros jazzy overlay on top mixes two ROS middlewares, so
        # the subprocess's DDS publishers never reach the host (the no-GPU smoke worked precisely
        # because it ran in a clean shell). Hand the overlay source a minimal env to build from.
        clean_env = {k: os.environ[k] for k in ('HOME', 'USER', 'LOGNAME', 'TERM', 'DISPLAY')
                     if k in os.environ}
        clean_env['PATH'] = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
        self._proc = subprocess.Popen(['bash', '-c', command], env=clean_env)
        self._writer = FrameWriter(self._socket_path)

    def publish(self, now):
        import numpy as np
        self._calls = getattr(self, '_calls', 0) + 1
        frame = self.camera.get_current_frame()
        # The camera's depth + instance-ID annotators are not populated on the very first
        # renders after initialize(); skip those frames rather than crash (the mission reaches
        # the arm camera many ticks in, so it never saw this, but the focused probe does).
        no_depth = not isinstance(frame, dict) or frame.get('distance_to_image_plane') is None
        no_seg = (not isinstance(frame, dict) or not isinstance(frame.get('instance_id_segmentation'), dict)
                  or frame['instance_id_segmentation'].get('data') is None)
        if self._calls % 30 == 1:                        # lightweight diagnostic (grep BRIDGE_STATS)
            rt = frame.get('rendering_time') if isinstance(frame, dict) else None
            print(f'BRIDGE_STATS calls={self._calls} no_depth={no_depth} no_seg={no_seg} '
                  f'published={getattr(self, "_published", 0)} now={now} rendering_time={rt}', flush=True)
        if no_depth or no_seg:
            return None
        # Stamp with the caller's host clock (world.current_time, the SAME source the /tf relay uses),
        # NOT the frame's rendering_time: tf2 never extrapolates forward, so the depth stamp must
        # equal the /tf stamp for that tick or nvblox silently drops the frame (no ESDF integration).
        stamp_s = float(now)
        masked, self._id_cache = masked_depth_from_frame(
            frame, self.rig_prefixes, self._id_cache, invalid=self.invalid)
        height, width = masked.shape
        k = np.asarray(self.camera.get_intrinsics_matrix(), float).reshape(3, 3).ravel().tolist()
        self._bbox = select_target_bbox(
            getattr(self.sensors, 'sensed_target_bboxes', ()), self.target_location, self._bbox)
        # ONE shared float->(sec,nanosec) helper for the depth stamp AND every relayed /tf, /clock,
        # /joint_states stamp: tf2 accepts an exact-newest match at NANOSECOND resolution, so the
        # depth and the TF for a tick must convert the identical world.current_time float identically
        # -- a truncate-vs-round or float32 difference would land the depth 1 ns after the newest TF
        # and nvblox would refuse it as future extrapolation forever (persistent non-integration).
        from .control_ipc import _secs_to_time
        sec, nsec = _secs_to_time(stamp_s)
        header = {'stamp_sec': sec, 'stamp_nsec': nsec,
                  'frame_id': self.frame_id, 'width': int(width), 'height': int(height),
                  'k': k, 'bbox': list(self._bbox) if self._bbox is not None else None}
        self._writer.send(header, np.ascontiguousarray(masked, dtype='<f4').tobytes())
        self._published = getattr(self, '_published', 0) + 1
        return stamp_s

    def close(self):
        try:
            self._writer.close()
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()
