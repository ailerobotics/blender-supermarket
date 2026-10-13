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
from ..restock_camera.geometry import rigid
from ..restock_camera.sensed import FrameEvidence, SupportView, masked_cloud, contains


@dataclass(frozen=True)
class SlotView:
    location: str
    roi: tuple[int, int, int, int]
    support_z: float      # previous empty-slot depth calibration in arm frame
    free_dimensions: tuple[float, float, float]
    yaw_rad: float = 0.0  # calibrated slot orientation in current arm frame
    label_mask: object = None
    support_view: SupportView | None = None

    @classmethod
    def from_support(cls, location, view, label_mask):
        mask = np.asarray(label_mask)
        k = view.geometry.intrinsics
        if mask.dtype != bool or mask.shape != (k.height,k.width) or not mask.any():
            raise ValueError('separate label search mask required')
        support = view.mask(k)
        v,u = np.where(support)
        roi = (int(u.min()),int(v.min()),int(u.max()+1),int(v.max()+1))
        return cls(location,roi,view.support_z,
                   (*view.geometry.required_xy,0. if view.clearance is None else view.clearance.height_m),view.yaw_rad,
                   mask.copy(),view)


@dataclass(frozen=True)
class CameraContext:
    base_from_camera: object
    transform_stamp: float
    epoch: str
    slots: tuple
    parked: bool
    session_id: str = 'default'
    steady_domain: str = ''
    acquired_steady: float | None = None


class IsaacCartonSensors:
    def __init__(self, camera, *, pose_bridge_root, profiles, slots=(), base_from_camera=None,
                 context_provider=None, camera_id='camera'):
        import cv2
        self.camera, self.profiles = camera, profiles
        self.slots = tuple(slots)
        self.transform = None if base_from_camera is None else rigid(base_from_camera)
        self.context_provider, self.camera_id = context_provider, camera_id
        self.depth = DepthPerception(load_pose_bridge(pose_bridge_root))
        self.decoder = cv2.QRCodeDetector()
        self.last_frame = None
        self.last_stamp = None
        self.last_session = None
        self.seen_sessions = set()
        self.sensed_target_bboxes = ()
        self.evidence = ()
        # Integration initializes actual cameras; every frame uses measured TF.
        camera.add_distance_to_image_plane_to_frame()

    def capture(self, now, source, destination, motion, contact):
        import cv2
        frame = self.camera.get_current_frame(clone=True)
        frame_id, stamp = frame['rendering_frame'], float(frame['rendering_time'])
        self.sensed_target_bboxes = ()
        self.evidence = ()
        if not np.isfinite([now,stamp]).all() or not 0 <= now-stamp <= .5:
            raise ValueError('no new synchronized rendered camera frame')
        rgb = np.asarray(frame['rgb'])[:, :, :3]
        depth = np.asarray(frame['distance_to_image_plane'], float)
        if rgb.shape[:2] != depth.shape or depth.ndim != 2:
            raise ValueError('unregistered camera RGB/depth')
        matrix = self.camera.get_intrinsics_matrix()
        intr = (matrix[0,0], matrix[1,1], matrix[0,2], matrix[1,2])
        if self.context_provider is not None:
            context = self.context_provider(frame,now)
            self.transform = rigid(context.base_from_camera)
            self.slots = tuple(context.slots)
        elif self.slots and all(s.support_view is not None for s in self.slots):
            e = self.slots[0].support_view.registration.evidence
            context = CameraContext(self.transform,e.transform_stamp,e.epoch,self.slots,motion.parked(),
                                    e.session_id,e.steady_domain,e.acquired_steady)
        else:
            raise ValueError('fresh registered support views and camera context required')
        if (self.last_session == context.session_id and (frame_id <= self.last_frame or stamp <= self.last_stamp)
                or self.last_session != context.session_id and context.session_id in self.seen_sessions):
            raise ValueError('no new synchronized rendered camera frame')
        self.last_frame,self.last_stamp,self.last_session = frame_id,stamp,context.session_id
        self.seen_sessions.add(context.session_id)
        if not self.slots or len({s.location for s in self.slots}) != len(self.slots):
            raise ValueError('camera needs unique calibrated slots')
        for slot in self.slots:
            if slot.support_view is None:
                raise ValueError('fresh registered support view required')
            k = slot.support_view.geometry.intrinsics
            if (depth.shape != (k.height,k.width) or not np.allclose(intr,[k.fx,k.fy,k.cx,k.cy],atol=1e-9,rtol=0)):
                raise ValueError('camera intrinsics changed after support calibration')
            evidence = FrameEvidence(self.camera_id,int(frame_id),stamp,context.transform_stamp,
                context.epoch,slot.support_view.registration.evidence.camera_revision,
                context.session_id,context.steady_domain,context.acquired_steady)
            slot.support_view.validate(depth,k,self.transform,evidence,now,
                                       bool(context.parked and motion.parked()))
            if (not np.isfinite(motion.collision_time) or not 0 <= now-motion.collision_time <= .5
                    or (slot.support_view.clearance is not None
                        and motion.collision_time < slot.support_view.clearance.incorporated_time)):
                raise ValueError('collision map has not incorporated current clearance evidence')
            if (slot.support_z != slot.support_view.support_z
                    or slot.free_dimensions != (*slot.support_view.geometry.required_xy,
                        0. if slot.support_view.clearance is None else slot.support_view.clearance.height_m)
                    or slot.yaw_rad != slot.support_view.yaw_rad):
                raise ValueError('slot geometry differs from sensed support evidence')
            if (np.asarray(slot.label_mask).dtype != bool or np.shape(slot.label_mask) != depth.shape):
                raise ValueError('separate registered label mask required')
        marker_ids = {marker for s in self.slots for marker in s.support_view.registration.marker_ids}
        ok, codes, corners, _ = self.decoder.detectAndDecodeMulti(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        observations, rejected = [], []
        mask_shape = depth.shape
        for index, (code, polygon) in enumerate(zip(codes, corners if ok else [])):
            if code in marker_ids:
                continue  # support identity never becomes inventory/SKU
            centre = np.mean(polygon, axis=0)
            ui,vi = np.rint(centre).astype(int)
            matches = [s for s in self.slots if 0 <= ui < depth.shape[1]
                       and 0 <= vi < depth.shape[0] and s.label_mask[vi,ui]]
            if not matches:
                continue
            if code not in self.profiles:
                rejected.append(code or 'undecoded label')
                continue
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
                self._check_slot_footprint(obs,self.profiles[code],matches[0])
                observations.append(obs)
            except ValueError as exc:
                rejected.append(str(exc))
        targets, scanned = {}, []
        for slot in self.slots:
            mask = slot.support_view.mask(slot.support_view.geometry.intrinsics)
            cloud = masked_cloud(depth,mask,slot.support_view.geometry.intrinsics,self.transform)
            target = self.depth.target(depth_m=depth, mask=mask, intrinsics=intr,
                base_from_camera=self.transform, support_z=slot.support_z,
                free_dimensions=slot.free_dimensions, observation_id=f'camera:{frame_id}:slot:{slot.location}',
                simulation_time=stamp, location=slot.location)
            # The footprint centre comes from the measured quad/registration,
            # not the perspective-biased median of its image pixels.
            target = replace(target,clear=bool(slot.support_view.clearance is not None
                                               and np.all(np.abs(cloud[:,2]-slot.support_z) < .004)
                                               and not any(o.location == slot.location for o in observations)),
                pose=(*slot.support_view.center_base,0.,0.,math.sin(slot.yaw_rad/2),math.cos(slot.yaw_rad/2)))
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
        self.sensed_target_bboxes = tuple(self._object_bbox(o,self.profiles[o.sku],intr,depth.shape)
                                          for o in observations)
        self.evidence = tuple(dict(location=s.location,
            baseline_calibration_id=s.support_view.geometry.calibration_id,
            baseline_stamp=s.support_view.geometry.baseline_stamp,
            baseline_session_id=s.support_view.geometry.baseline_session_id,
            baseline_acquired_steady=s.support_view.geometry.baseline_acquired_steady,
            current_frame_id=frame_id,current_stamp=stamp,epoch=context.epoch,
            registration_uncertainty_m=s.support_view.uncertainty_m,
            clearance_observation_id=None if s.support_view.clearance is None else s.support_view.clearance.observation_id,
            provenance='dated_rigid_geometry_with_fresh_marker_witness_and_depth') for s in self.slots)
        return Sensors(now, tuple(observations), targets.get(destination),
                       motion.parked(), contact.read(), now, motion.collision_time,
                       motion.joint_positions(), tuple(scanned), stamp)

    def _known_foreground(self, slot, observations, depth, intr):
        mask = slot.support_view.mask(slot.support_view.geometry.intrinsics)
        cloud = masked_cloud(depth,mask,slot.support_view.geometry.intrinsics,self.transform)
        foreground = cloud[np.abs(cloud[:,2]-slot.support_z) >= .004]
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

    def _check_slot_footprint(self, observation, profile, slot):
        x,y,z = observation.pose[:3]
        yaw = 2*math.atan2(observation.pose[5],observation.pose[6])
        c,s = math.cos(yaw),math.sin(yaw)
        points = np.array([[x+c*dx-s*dy,y+s*dx+c*dy,z]
            for dx,dy in [(-profile.dimensions[0]/2,-profile.dimensions[1]/2),
                          (profile.dimensions[0]/2,-profile.dimensions[1]/2),
                          (profile.dimensions[0]/2,profile.dimensions[1]/2),
                          (-profile.dimensions[0]/2,profile.dimensions[1]/2)]])
        inverse = np.linalg.inv(slot.support_view.registration.base_from_support)
        local = points @ inverse[:3,:3].T+inverse[:3,3]
        if not contains(np.asarray(slot.support_view.geometry.polygon_support)[:,:2],local[:,:2],
                        slot.support_view.uncertainty_m):
            raise ValueError('sensed carton footprint outside the measured slot')

    def _object_bbox(self, observation, profile, intr, shape):
        """Project accepted sensed carton volume, never a support/label ROI.

        This is a catalog-sized box anchored to fresh QR/depth pose. Consumers
        must choose only the active target and retain mapper coverage/ack gates.
        """
        x,y,z = observation.pose[:3]
        yaw = 2*math.atan2(observation.pose[5],observation.pose[6])
        c,s = math.cos(yaw),math.sin(yaw)
        points = np.array([[x+c*dx-s*dy,y+s*dx+c*dy,z-dz]
            for dx in (-profile.dimensions[0]/2,profile.dimensions[0]/2)
            for dy in (-profile.dimensions[1]/2,profile.dimensions[1]/2)
            for dz in (0,profile.dimensions[2])])
        transform = np.linalg.inv(self.transform)
        p = points @ transform[:3,:3].T+transform[:3,3]
        if np.any(p[:,2] <= 0):
            raise ValueError('sensed object bounding box behind camera')
        uv = p[:,:2]/p[:,2,None]*intr[:2]+intr[2:]
        lo,hi = np.floor(uv.min(axis=0)).astype(int),np.ceil(uv.max(axis=0)).astype(int)+1
        if np.any(lo < 0) or np.any(hi > [shape[1],shape[0]]):
            raise ValueError('sensed object bounding box truncated')
        return dict(camera_id=self.camera_id,observation_id=observation.observation_id,
                    stamp=observation.simulation_time,location=observation.location,
                    roi=tuple(map(int,[*lo,*hi])),source='sensor_pose_registered_catalog_volume')

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
