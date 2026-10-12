"""Rendered RGB-D inventory for a small, physically labelled carton slice.

Uses actual camera pixels plus the existing masked depth/PCA implementation.
QR text is product identity; no USD object names, semantic IDs, poses or colors
enter perception. A registered carton has a centred top label reading its SKU.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from .perception import DepthPerception, load_pose_bridge
from .types import Sensors


@dataclass(frozen=True)
class SlotView:
    location: str
    roi: tuple[int, int, int, int]
    support_z: float      # previous empty-slot depth calibration in arm frame
    free_dimensions: tuple[float, float, float]
    yaw_rad: float = 0.0  # calibrated slot orientation in current arm frame


class IsaacCartonSensors:
    def __init__(self, camera, *, pose_bridge_root, profiles, slots, base_from_camera):
        import cv2
        self.camera, self.profiles = camera, profiles
        self.slots = tuple(slots)
        self.transform = np.asarray(base_from_camera, float)
        self.depth = DepthPerception(load_pose_bridge(pose_bridge_root))
        self.decoder = cv2.QRCodeDetector()
        self.last_frame = None
        # Camera must already be initialized by integration, and mounted rigidly
        # to the arm base (optical convention +x right, +y down, +z forward).
        camera.add_distance_to_image_plane_to_frame()

    def capture(self, now, source, destination, motion, contact):
        import cv2
        frame = self.camera.get_current_frame(clone=True)
        frame_id, stamp = frame['rendering_frame'], float(frame['rendering_time'])
        if frame_id == self.last_frame or not 0 <= now-stamp <= .5:
            raise ValueError('no new synchronized rendered camera frame')
        rgb = np.asarray(frame['rgb'])[:, :, :3]
        depth = np.asarray(frame['distance_to_image_plane'], float)
        if rgb.shape[:2] != depth.shape or depth.ndim != 2:
            raise ValueError('unregistered camera RGB/depth')
        self.last_frame = frame_id
        matrix = self.camera.get_intrinsics_matrix()
        intr = (matrix[0,0], matrix[1,1], matrix[0,2], matrix[1,2])
        ok, codes, corners, _ = self.decoder.detectAndDecodeMulti(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        observations, rejected = [], []
        mask_shape = depth.shape
        for index, (code, polygon) in enumerate(zip(codes, corners if ok else [])):
            if code not in self.profiles:
                rejected.append(code or 'undecoded label')
                continue
            centre = np.mean(polygon, axis=0)
            matches = [s for s in self.slots if s.roi[0] <= centre[0] < s.roi[2]
                       and s.roi[1] <= centre[1] < s.roi[3]]
            if len(matches) != 1:
                rejected.append('label outside unique calibrated slot')
                continue
            mask = np.zeros(mask_shape, np.uint8)
            cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
            try:
                obs = self.depth.observe(depth_m=depth, mask=mask,
                    intrinsics=intr, base_from_camera=self.transform,
                    classification=([(code,1.)], 'OK', 1., 1.),
                    observation_id=f'camera:{frame_id}:qr:{index}', simulation_time=stamp,
                    location=matches[0].location, rgb_time=stamp, depth_time=stamp, transform_time=stamp)
                angle = self._label_yaw(polygon, depth, intr)
                obs = replace(obs, pose=(*obs.pose[:3],0.,0.,math.sin(angle/2),math.cos(angle/2)))
                if abs(obs.pose[2]-matches[0].support_z-self.profiles[code].dimensions[2]) > .01:
                    raise ValueError('observed carton height differs from registered handling unit')
                # Verify the complete physical suction patch in depth, beyond the
                # small QR marker. A label floating over a gap cannot certify a seal.
                self._check_top_patch(obs, self.profiles[code], depth, intr)
                observations.append(obs)
            except ValueError as exc:
                rejected.append(str(exc))
        targets, scanned = {}, []
        for slot in self.slots:
            x1,y1,x2,y2 = slot.roi
            mask = np.zeros(mask_shape, bool)
            if not (0 <= x1 < x2 <= depth.shape[1] and 0 <= y1 < y2 <= depth.shape[0]):
                raise ValueError('slot ROI outside camera image')
            mask[y1:y2,x1:x2] = True
            target = self.depth.target(depth_m=depth, mask=mask, intrinsics=intr,
                base_from_camera=self.transform, support_z=slot.support_z,
                free_dimensions=slot.free_dimensions, observation_id=f'camera:{frame_id}:slot:{slot.location}',
                simulation_time=stamp, location=slot.location)
            target = replace(target, pose=(*target.pose[:3],0.,0.,math.sin(slot.yaw_rad/2),math.cos(slot.yaw_rad/2)))
            targets[slot.location] = target
            # Only a fully observed empty plane proves vacancy. A decoded carton
            # proves presence, not visibility of every hidden product behind it.
            if target.clear:
                scanned.append(slot.location)
            elif (slot.location == destination and not rejected and
                  sum(o.location == destination for o in observations) == 1
                  and self._known_foreground(slot, observations, depth, intr)):
                # Supported slice is a single dedicated carton slot, never a bin
                # of loose/stacked products. Target verification checks carton pose.
                scanned.append(slot.location)
        candidates = [o for o in observations if o.location == source]
        if len(candidates) == 1:
            motion.observe_source(candidates[0])
        return Sensors(now, tuple(observations), targets.get(destination),
                       motion.parked(), contact.read(), now, motion.collision_time,
                       motion.joint_positions(), tuple(scanned), stamp)

    def _known_foreground(self, slot, observations, depth, intr):
        mask = np.zeros(depth.shape, bool)
        x1,y1,x2,y2 = slot.roi
        mask[y1:y2,x1:x2] = True
        cloud = self.depth._cloud(depth, mask, intr, self.transform)
        foreground = cloud[np.abs(cloud[:,2]-slot.support_z) > .02]
        explained = np.zeros(len(foreground), bool)
        for obs in observations:
            if obs.location != slot.location:
                continue
            size = np.asarray(self.profiles[obs.sku].dimensions)
            centre = np.asarray(obs.pose[:3]) - [0,0,size[2]/2]
            angle = 2*math.atan2(obs.pose[5],obs.pose[6])
            c,s = math.cos(angle),math.sin(angle)
            local = (foreground-centre) @ np.array([[c,-s,0],[s,c,0],[0,0,1]])
            explained |= np.all(np.abs(local) <= size/2+.01, axis=1)
        return bool(explained.all())

    def _label_yaw(self, polygon, depth, intr):
        fx,fy,cx,cy = intr
        points = []
        for u,v in polygon[:2]:
            ui,vi = int(round(u)),int(round(v))
            if not (0 <= ui < depth.shape[1] and 0 <= vi < depth.shape[0]):
                raise ValueError('QR corner outside depth image')
            z = depth[vi,ui]
            if not np.isfinite(z) or z <= 0:
                raise ValueError('QR orientation lacks valid depth')
            points.append(self.transform @ np.array([(u-cx)*z/fx,(v-cy)*z/fy,z,1.]))
        axis = points[1]-points[0]
        if np.linalg.norm(axis[:2]) < .005 or abs(axis[2]) > .004:
            raise ValueError('QR top orientation is uncertain/tilted')
        return math.atan2(axis[1],axis[0])

    def _check_top_patch(self, observation, profile, depth, intr):
        # A dense interior grid requires depth agreement over the registered
        # carton footprint. It also rejects tilted/small/occluded surfaces.
        fx,fy,cx,cy = intr
        camera_from_base = np.linalg.inv(self.transform)
        x,y,z = observation.pose[:3]
        angle = 2*math.atan2(observation.pose[5], observation.pose[6])
        c,s = math.cos(angle),math.sin(angle)
        # QR centre is mechanically centred on the top; unknown mounting is unsupported.
        for dx in np.linspace(-profile.dimensions[0]/2+.01, profile.dimensions[0]/2-.01, 9):
            for dy in np.linspace(-profile.dimensions[1]/2+.01, profile.dimensions[1]/2-.01, 7):
                p = camera_from_base @ np.array([x+c*dx-s*dy,y+s*dx+c*dy,z,1.])
                if p[2] <= 0:
                    raise ValueError('carton top behind camera')
                u,v = int(round(fx*p[0]/p[2]+cx)), int(round(fy*p[1]/p[2]+cy))
                if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                    raise ValueError('carton top outside depth image')
                if not np.isfinite(depth[v,u]) or abs(depth[v,u]-p[2]) > .004:
                    raise ValueError('carton top patch occluded, tilted or smaller than catalog')


def write_carton_label(sku, output_path, *, pixels=512):
    """Generate the actual scannable top texture; caller applies it to the carton.

    The label must be centred, flat, visible and unwarped on the carton top. This
    writes a PNG only; it never injects a decoded observation or semantic label.
    """
    import cv2
    if not sku or not isinstance(sku, str) or pixels < 128:
        raise ValueError('SKU and label resolution required')
    qr = cv2.QRCodeEncoder_create().encode(sku)
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (pixels,pixels), interpolation=cv2.INTER_NEAREST)
    if not cv2.imwrite(str(output_path), qr):
        raise OSError('could not write carton label PNG')
