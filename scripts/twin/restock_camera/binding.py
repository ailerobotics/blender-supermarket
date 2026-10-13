"""Executable multi-camera inventory binding. No native Isaac/ROS imports."""
from dataclasses import dataclass, replace
import numpy as np

from .sensed import (FrameEvidence, register_marker, calibrate_rigid_support,
                     RegisteredSupport)


@dataclass
class SupportSlot:
    location: str
    support_id: str
    marker_id: str
    marker_side_m: float
    revision: str
    label_mask: object
    required_xy: tuple = (.62,.42)
    auxiliary_id: str | None = None
    auxiliary_offset_xy: tuple | None = None
    tracker: RegisteredSupport | None = None


class RenderedSupportContext:
    """Decode support markers from actual RGB and register against the same depth.

    measure_context(frame, now) -> (base_from_camera, transform_stamp, epoch, parked)
    clearance_for_slot(location, support_view, frame, now) -> ClearanceEvidence | None
    None means no vacancy proof (including occupied source); recognition remains possible.
    clock_context(frame, now) -> (session_id, steady_domain, acquired_steady)
    uses one persistent acquisition clock across commissioning/mission clock resets.
    Both callbacks are integration-owned; receipt of this frame is not map incorporation.
    """
    def __init__(self, *, camera_id, intrinsics, slots, measure_context,
                 clearance_for_slot, depth_error_m, camera_revision, clock_context=None):
        import cv2
        self.camera_id,self.intrinsics = camera_id,intrinsics
        self.slots = tuple(slots)
        self.measure_context,self.clearance_for_slot = measure_context,clearance_for_slot
        self.depth_error_m,self.camera_revision = depth_error_m,camera_revision
        self.clock_context = clock_context
        self.decoder = cv2.QRCodeDetector()
        if len({slot.location for slot in slots}) != len(slots):
            raise ValueError('duplicate support location')

    def _registrations(self, frame, now):
        import cv2
        transform,tf_stamp,epoch,parked = self.measure_context(frame,now)
        clock = ('default','',None) if self.clock_context is None else self.clock_context(frame,now)
        evidence = FrameEvidence(self.camera_id,int(frame['rendering_frame']),
            float(frame['rendering_time']),tf_stamp,epoch,self.camera_revision,*clock)
        evidence.validate(now,parked)
        rgb = np.asarray(frame['rgb'])[:,:,:3]
        depth = np.asarray(frame['distance_to_image_plane'],float)
        if rgb.shape[:2] != (self.intrinsics.height,self.intrinsics.width) or depth.shape != rgb.shape[:2]:
            raise ValueError('unregistered marker RGB-D')
        ok,codes,corners,_ = self.decoder.detectAndDecodeMulti(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        decoded = {}
        for code,polygon in zip(codes,corners if ok else []):
            if code in decoded:
                raise ValueError('duplicate support/product identity in camera frame')
            decoded[code] = polygon
        registrations = []
        for slot in self.slots:
            if slot.marker_id not in decoded or slot.auxiliary_id and slot.auxiliary_id not in decoded:
                raise ValueError('fresh support marker(s) not decoded; no geometry fallback')
            extra = {} if not slot.auxiliary_id else dict(auxiliary_pixels=decoded[slot.auxiliary_id],
                auxiliary_id=slot.auxiliary_id,decoded_auxiliary_id=slot.auxiliary_id,
                auxiliary_offset_xy=slot.auxiliary_offset_xy)
            registrations.append(register_marker(depth,decoded[slot.marker_id],self.intrinsics,transform,
                marker_id=slot.marker_id,decoded_id=slot.marker_id,support_id=slot.support_id,
                revision=slot.revision,side_m=slot.marker_side_m,evidence=evidence,now=now,
                parked=parked,depth_error_m=self.depth_error_m,**extra))
        return transform,evidence,parked,depth,registrations

    def commission(self, frame, now, empty_polygons, *, lifetime_s=600.):
        """Explicitly acquire baselines from observed empty supports.

        empty_polygons[location] = (support_quad_uv, witness_quad_uv), selected
        from that measured frame. Source starts occupied, so a real prior empty
        commissioning frame is required; never manufacture it from a setup dict.
        """
        transform,e,parked,depth,registrations = self._registrations(frame,now)
        pending = []
        for slot,registration in zip(self.slots,registrations):
            support,witness = empty_polygons[slot.location]
            model = calibrate_rigid_support(depth,support,witness,self.intrinsics,transform,registration,
                required_xy=slot.required_xy,calibration_id=f'{e.session_id}:{self.camera_id}:{e.frame_id}:{slot.support_id}',
                now=now,parked=parked,lifetime_s=lifetime_s)
            pending.append(RegisteredSupport(model))
        for slot,tracker in zip(self.slots,pending):
            slot.tracker = tracker
        return tuple(t.geometry for t in pending)

    def __call__(self, frame, now):
        from ..restock_manipulation.isaac_sensors import CameraContext, SlotView
        transform,e,parked,depth,registrations = self._registrations(frame,now)
        views = []
        for slot,registration in zip(self.slots,registrations):
            if slot.tracker is None:
                raise ValueError('observed empty commissioning baseline is required')
            view = slot.tracker.prepare(depth,self.intrinsics,transform,registration,None,now=now,parked=parked)
            clearance = self.clearance_for_slot(slot.location,view,frame,now)
            if clearance is not None:
                clearance.validate(now,e,view.geometry)
                view = replace(view,clearance=clearance)
            views.append(SlotView.from_support(slot.location,view,slot.label_mask))
        return CameraContext(transform,e.transform_stamp,e.epoch,tuple(views),parked,
                             e.session_id,e.steady_domain,e.acquired_steady)

    def invalidate(self, reason):
        for slot in self.slots:
            if slot.tracker is not None:
                slot.tracker.invalidate(reason)


class StationSensors:
    """Aggregate distinct assigned cameras; no camera implicitly covers the mapper."""
    def __init__(self, *, cameras, profiles, pose_bridge_root, context_providers, location_cameras):
        from ..restock_manipulation.isaac_sensors import IsaacCartonSensors
        self.location_cameras = dict(location_cameras)
        if (not cameras or set(cameras) != set(context_providers)
                or any(camera not in cameras for camera in self.location_cameras.values())):
            raise ValueError('camera/location/context binding is incomplete')
        self.adapters = {key:IsaacCartonSensors(camera,pose_bridge_root=pose_bridge_root,
            profiles=profiles,camera_id=key,context_provider=context_providers[key])
            for key,camera in cameras.items()}
        self.sensed_target_bboxes = ()
        self.evidence = ()

    def capture(self, now, source, destination, motion, contact):
        from ..restock_manipulation.types import Sensors
        self.sensed_target_bboxes,self.evidence = (),()
        selected = sorted({self.location_cameras[source],self.location_cameras[destination]})
        packets = []
        for camera_id in selected:
            adapter = self.adapters[camera_id]
            packet = adapter.capture(now,source,destination,motion,contact)
            if any(self.location_cameras.get(slot.location) != camera_id for slot in adapter.slots):
                raise ValueError('camera observed a slot assigned to another camera')
            packets.append(packet)
        targets = [p.target for p in packets if p.target is not None]
        if len(targets) != 1:
            raise ValueError('destination requires one current calibrated camera view')
        if max(p.scan_time for p in packets)-min(p.scan_time for p in packets) > .05:
            raise ValueError('station cameras are not synchronized')
        self.sensed_target_bboxes = tuple(box for key in selected for box in self.adapters[key].sensed_target_bboxes)
        self.evidence = tuple(item for key in selected for item in self.adapters[key].evidence)
        return Sensors(now,tuple(o for p in packets for o in p.observations),targets[0],
            all(p.parked for p in packets),packets[0].contact,min(p.feedback_time for p in packets),
            min(p.collision_time for p in packets),packets[0].joint_positions,
            tuple(location for p in packets for location in p.scanned_locations),min(p.scan_time for p in packets))
