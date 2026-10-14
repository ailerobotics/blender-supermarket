"""Isaac-side rig-masked depth bridge for the host nvblox consumer.

Option B for the manipulation collision world: instead of the host `robot_segmenter`
node masking the robot out of the depth stream (which currently fails to load on a
duplicate NITROS subscription), publish a depth image from the Isaac side with the
robot/cart (the "rig") self-pixels already invalidated, using the per-frame instance-ID
segmentation the mission already captures (`add_instance_id_segmentation_to_frame`).
nvblox then integrates only non-rig returns, so the ESDF/planning scene is built without
the robot carving itself into the obstacle set.

The masking core (`rig_instance_ids`, `mask_rig_depth`, `parse_instance_frame`) is pure
NumPy and CPU-tested; the ROS publisher (`MaskedDepthPublisher`) lazy-imports rclpy and
sensor_msgs so it is constructed only inside the guarded app. The publisher is wired only
when the runtime-config opts in (`masked_depth` block), so the legacy raw-depth OmniGraph
path is untouched until the host worker chooses option B.

Contract proposed to the host worker (confirm via status notes before enabling):
  depth topic : /restock_sim/camera/masked_depth        sensor_msgs/Image, 32FC1, metres
  info  topic : /restock_sim/camera/masked_depth/camera_info   sensor_msgs/CameraInfo
  frame_id    : restock_camera_<id> (same TF frame as the existing camera stream)
  invalid     : 0.0 (nvblox treats 0 / non-finite depth as no-return)
  domain      : 88 (the mission's isolated ROS domain)
  stamp       : mission-relative acquisition clock (sim time), matching the RGB/info stream
"""
from __future__ import annotations

import numpy as np

INVALID_DEPTH = 0.0  # nvblox no-return sentinel


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


class MaskedDepthPublisher:
    """Publish rig-masked depth + camera info on the mission's ROS domain for nvblox.

    Constructed only inside the guarded app (lazy-imports rclpy/sensor_msgs). `camera` is a
    RenderedCamera exposing `get_current_frame()` (RGB-D + instance-ID seg) and
    `get_intrinsics_matrix()`; `rig_prefixes` are the USD prim prefixes of the robot/cart.
    `publish(now)` reads one frame, masks it and publishes; callers invoke it only on
    sensing-stage ticks (when the scene is rendered).
    """

    def __init__(self, camera, rig_prefixes, *, topic='/restock_sim/camera/masked_depth',
                 info_topic=None, frame_id='restock_camera', domain=88,
                 invalid=INVALID_DEPTH, node=None, _ros=None):
        self.camera = camera
        self.rig_prefixes = tuple(rig_prefixes)
        if not self.rig_prefixes:
            raise ValueError('rig_prefixes required to identify robot/cart self-returns')
        self.topic = topic
        self.info_topic = info_topic or topic + '/camera_info'
        self.frame_id = frame_id
        self.invalid = float(invalid)
        self._id_cache = (None, frozenset())  # (id_to_path identity, resolved rig ids)
        self._ros = _ros or self._init_ros(domain, node)

    def _init_ros(self, domain, node):
        import os
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, CameraInfo
        # The isolated mission domain must already be set by the launcher; honour it.
        os.environ.setdefault('ROS_DOMAIN_ID', str(domain))
        if not rclpy.ok():
            rclpy.init(args=None)
        node = node or Node('restock_masked_depth')
        return {'node': node, 'Image': Image, 'CameraInfo': CameraInfo,
                'depth_pub': node.create_publisher(Image, self.topic, 10),
                'info_pub': node.create_publisher(CameraInfo, self.info_topic, 10)}

    def _rig_ids(self, id_to_path):
        # Resolve once per distinct id->path map (Isaac reuses the dict object across frames
        # while the scene graph is stable), then reuse for every subsequent frame.
        if id_to_path is not self._id_cache[0]:
            self._id_cache = (id_to_path, frozenset(rig_instance_ids(id_to_path, self.rig_prefixes)))
        return self._id_cache[1]

    def masked_frame(self, frame):
        """Pure step: frame -> (masked HxW float32 depth metres, KxK intrinsics)."""
        depth = frame.get('distance_to_image_plane')
        if depth is None:
            depth = frame.get('depth')
        if depth is None:
            raise ValueError('frame carries no depth (distance_to_image_plane)')
        ids, id_to_path = parse_instance_frame(frame)
        masked = mask_rig_depth(depth, ids, self._rig_ids(id_to_path), invalid=self.invalid)
        return masked, self.camera.get_intrinsics_matrix()

    def publish(self, now):
        frame = self.camera.get_current_frame()
        masked, intrinsics = self.masked_frame(frame)
        node = self._ros['node']
        stamp = self._stamp(float(now))
        height, width = masked.shape
        image = self._ros['Image']()
        image.header.stamp, image.header.frame_id = stamp, self.frame_id
        image.height, image.width = int(height), int(width)
        image.encoding, image.is_bigendian = '32FC1', 0
        image.step = int(width) * 4
        image.data = np.ascontiguousarray(masked, dtype=np.float32).tobytes()
        info = self._ros['CameraInfo']()
        info.header.stamp, info.header.frame_id = stamp, self.frame_id
        info.height, info.width = int(height), int(width)
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        k = np.asarray(intrinsics, float).reshape(3, 3)
        info.k = k.ravel().tolist()
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [k[0, 0], 0.0, k[0, 2], 0.0, 0.0, k[1, 1], k[1, 2], 0.0, 0.0, 0.0, 1.0, 0.0]
        self._ros['depth_pub'].publish(image)
        self._ros['info_pub'].publish(info)
        return image, info

    @staticmethod
    def _stamp(seconds):
        from builtin_interfaces.msg import Time
        sec = int(seconds)
        return Time(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


def bbox_payload(bbox):
    """Validate and flatten one grasp-target bbox to [u1, v1, u2, v2] floats.

    `bbox` is the target's axis-aligned image box as (u1, v1, u2, v2) pixel corners with
    u1<u2 and v1<v2 (the shelfbot object_segmenter masks this rectangle out of the depth so
    cuMotion can plan a path into the grasp target). Returns a plain float list for the
    Float32MultiArray data field.
    """
    box = [float(v) for v in bbox]
    if len(box) != 4 or not all(np.isfinite(box)):
        raise ValueError('target bbox must be four finite pixel values [u1,v1,u2,v2]')
    u1, v1, u2, v2 = box
    if not (u2 > u1 and v2 > v1):
        raise ValueError('target bbox must have u1<u2 and v1<v2')
    return box


class TargetBboxPublisher:
    """Publish the grasp-target image bbox as std_msgs/Float32MultiArray for the host.

    The shelfbot object_segmenter subscribes `/restock_sim/target_bbox` (bbox_timeout_s 0.5)
    and, for each masked-depth frame with a fresh bbox, invalidates the target rectangle in
    the depth before republishing `world_depth` to nvblox; a stale/absent bbox passes depth
    through unchanged (leaving the target an obstacle). So the bbox must be published on every
    masked-depth frame, in the SAME camera image as the depth. Constructed only inside the
    guarded app (lazy rclpy import).
    """

    def __init__(self, *, topic='/restock_sim/target_bbox', domain=88, node=None, _ros=None):
        self.topic = topic
        self._ros = _ros or self._init_ros(domain, node)

    def _init_ros(self, domain, node):
        import os
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray
        os.environ.setdefault('ROS_DOMAIN_ID', str(domain))
        if not rclpy.ok():
            rclpy.init(args=None)
        node = node or Node('restock_target_bbox')
        return {'node': node, 'Float32MultiArray': Float32MultiArray,
                'pub': node.create_publisher(Float32MultiArray, self.topic, 10)}

    def publish(self, bbox):
        msg = self._ros['Float32MultiArray']()
        msg.data = bbox_payload(bbox)
        self._ros['pub'].publish(msg)
        return msg
