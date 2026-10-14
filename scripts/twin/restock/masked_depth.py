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


class ArmDepthOmniGraphBridge:
    """Publish arm masked-depth (Image) + camera_info + target_bbox via OmniGraph ROS2 nodes.

    Built only inside the guarded app (lazy `omni.graph.core`). TF (restock_camera_arm ->
    ur_base_link) is NOT published here: an optical child prim `restock_camera_arm` is created
    under the arm camera and added to the existing native_transport /tf tree, which publishes
    it every physics tick alongside /clock. `publish(now)` reads one arm-camera frame, masks
    the rig self-pixels, and emits depth + camera_info + the cached target bbox on one stamp.

    - `camera` is the Isaac arm Camera (render product + get_current_frame + intrinsics);
    - `transport` is the native_transport graph exposing `add_frames([optical_path])`;
    - `sensors` exposes `sensed_target_bboxes`; set `target_location` per operation.
    """

    DEPTH_TOPIC = '/restock_sim/camera/masked_depth'
    INFO_TOPIC = '/restock_sim/camera/masked_depth/camera_info'
    BBOX_TOPIC = '/restock_sim/target_bbox'

    def __init__(self, *, camera, camera_path, stage, transport, sensors, rig_prefixes,
                 frame_id='restock_camera_arm', domain=88, invalid=INVALID_DEPTH,
                 root='/RestockMaskedDepthROS'):
        import omni.graph.core as og
        from pxr import UsdGeom, Gf, Sdf
        if stage.GetPrimAtPath(root):
            raise ValueError('refusing to overwrite the masked-depth ROS graph')
        self.og, self.root, self.frame_id = og, root, frame_id
        self.camera, self.sensors = camera, sensors
        self.rig_prefixes = tuple(rig_prefixes)
        if not self.rig_prefixes:
            raise ValueError('rig_prefixes required to identify robot/cart self-returns')
        self.invalid = float(invalid)
        self.target_location = None
        self._id_cache = (None, frozenset())
        self._bbox = None

        # Optical child prim (ROS +Z forward / +Y down) under the USD camera (-Z / +Y up).
        optical_path = camera_path + '/' + frame_id
        if not stage.GetPrimAtPath(optical_path):
            xf = UsdGeom.Xform.Define(stage, optical_path)
            xf.AddTransformOp().Set(Gf.Matrix4d(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1))
        # Hand the optical frame to the existing TF tree (parent ur_base_link via the chain).
        transport.add_frames([optical_path])

        render_product = camera.get_render_product_path()
        keys = og.Controller.Keys
        og.Controller.edit({'graph_path': root, 'evaluator_name': 'execution'}, {
            keys.CREATE_NODES: [
                ('tick', 'omni.graph.action.OnImpulseEvent'),
                ('context', 'isaacsim.ros2.bridge.ROS2Context'),
                ('info', 'isaacsim.ros2.bridge.ROS2CameraInfoHelper'),
                ('depth', 'isaacsim.ros2.bridge.ROS2PublishImage'),
                ('bbox', 'isaacsim.ros2.bridge.ROS2Publisher')],
            keys.SET_VALUES: [
                ('context.inputs:useDomainIDEnvVar', False), ('context.inputs:domain_id', domain),
                ('info.inputs:renderProductPath', render_product),
                ('info.inputs:topicName', self.INFO_TOPIC), ('info.inputs:frameId', frame_id),
                ('info.inputs:useSystemTime', False), ('info.inputs:resetSimulationTimeOnStop', False),
                ('depth.inputs:topicName', self.DEPTH_TOPIC), ('depth.inputs:frameId', frame_id),
                ('depth.inputs:encoding', '32FC1'),
                ('bbox.inputs:topicName', self.BBOX_TOPIC),
                ('bbox.inputs:messagePackage', 'std_msgs'), ('bbox.inputs:messageSubfolder', 'msg'),
                ('bbox.inputs:messageName', 'Float32MultiArray')],
            keys.CONNECT: [
                ('tick.outputs:execOut', 'info.inputs:execIn'),
                ('tick.outputs:execOut', 'depth.inputs:execIn'),
                ('tick.outputs:execOut', 'bbox.inputs:execIn'),
                ('context.outputs:context', 'info.inputs:context'),
                ('context.outputs:context', 'depth.inputs:context'),
                ('context.outputs:context', 'bbox.inputs:context')]})
        # Materialize the Float32MultiArray dynamic data attribute.
        og.Controller.evaluate_sync(root)

    def _attr(self, node, attr):
        return self.og.Controller.attribute(self.root + '/' + node + '.inputs:' + attr)

    def publish(self, now):
        frame = self.camera.get_current_frame()
        stamp = float(frame.get('rendering_time', now)) if isinstance(frame, dict) else float(now)
        masked, self._id_cache = masked_depth_from_frame(
            frame, self.rig_prefixes, self._id_cache, invalid=self.invalid)
        height, width = masked.shape
        data = np.ascontiguousarray(masked, dtype=np.float32).view(np.uint8).reshape(-1)
        self._attr('depth', 'data').set(data)
        self._attr('depth', 'width').set(int(width))
        self._attr('depth', 'height').set(int(height))
        self._attr('depth', 'timeStamp').set(float(stamp))
        self._bbox = select_target_bbox(
            getattr(self.sensors, 'sensed_target_bboxes', ()), self.target_location, self._bbox)
        if self._bbox is not None:
            self._attr('bbox', 'data').set([float(v) for v in bbox_payload(self._bbox)])
        self.og.Controller.attribute(self.root + '/tick.state:enableImpulse').set(True)
        self.og.Controller.evaluate_sync(self.root)
        return stamp
