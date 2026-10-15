"""Executable multi-camera inventory binding. No native Isaac/ROS imports."""
from dataclasses import dataclass, replace
import time
import numpy as np

from .sensed import (FrameEvidence, register_marker, calibrate_rigid_support,
                     RegisteredSupport)
from .freshness import frame_deadline, require_fresh
from .qr import FixedTileQR, FrameQR, rgb_digest


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
    decode_roi: tuple | None = None   # cached pixel bbox of this slot's support (set at commission)


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
                 clearance_for_slot, depth_error_m, camera_revision, clock_context=None,
                 steady_clock=time.monotonic, occlusion_window_s=None,
                 occlusion_contradiction_m=.05, occlusion_contradiction_rad=.17, render_scale=1.0):
        self.steady_clock = steady_clock
        self.camera_id,self.intrinsics = camera_id,intrinsics
        self.slots = tuple(slots)
        self.measure_context,self.clearance_for_slot = measure_context,clearance_for_slot
        self.depth_error_m,self.camera_revision = depth_error_m,camera_revision
        self.clock_context = clock_context
        # Approach-occlusion tolerance (None = strict). When the reaching arm occludes the FIXED
        # camera's view of the support it grasps toward, the live __call__ holds the last clean
        # registration (flagged occluded) up to this window instead of refusing and starving the
        # sensor-stream watchdog. commission()/observe stay strict (full footprint required).
        if occlusion_window_s is not None and (not np.isfinite(occlusion_window_s) or occlusion_window_s < 0):
            raise ValueError('occlusion_window_s must be None or a finite non-negative duration')
        self.occlusion_window_s = occlusion_window_s
        self.occlusion_contradiction_m = occlusion_contradiction_m
        self.occlusion_contradiction_rad = occlusion_contradiction_rad
        # Scale the decoder tile grid with the sensing render resolution (a fast render preset steps
        # it down): the intrinsics are already scaled by the same factor, so the tiles follow the
        # markers and decode stays reliable at reduced resolution (see qr.tile_origins).
        self.decoder = FixedTileQR(camera_id, render_scale=render_scale)
        self.qr_observation = None
        if len({slot.location for slot in slots}) != len(slots):
            raise ValueError('duplicate support location')

    def _decode_roi(self):
        """Union pixel bbox of the slots' cached support regions (None until commissioned).

        Each slot's support and its camera are rigidly related per station, so the
        commissioned support pixel box bounds the fresh support markers and the handling
        unit on it across captures; a miss still falls back to the full certified grid.
        """
        boxes = [slot.decode_roi for slot in self.slots if slot.decode_roi is not None]
        if len(boxes) != len(self.slots) or not boxes:
            return None
        boxes = np.asarray(boxes, float)
        return (float(boxes[:,0].min()), float(boxes[:,1].min()),
                float(boxes[:,2].max()), float(boxes[:,3].max()))

    def _registrations(self, frame, now, *, allow_occlusion=False):
        """Decode support markers and register each slot against this depth.

        ``allow_occlusion`` (the live __call__ path, never commission/observe): a slot whose marker
        is not decoded, whose marker is too occluded to register, OR a frame the production decoder
        cannot read at all (unreadable/conflicting/replayed-re-presentation/over-budget as the
        reaching arm occludes the support) yields ``None`` instead of aborting — __call__ then holds
        that slot's last clean registration, and this frame carries an EMPTY held QR (no fresh
        detections). Strict (default) keeps the historical 'no geometry fallback' / decode refusal so
        commissioning/observe require full visibility."""
        self.qr_observation = None
        transform,tf_stamp,epoch,parked = self.measure_context(frame,now)
        clock = ('default','',None) if self.clock_context is None else self.clock_context(frame,now)
        evidence = FrameEvidence(self.camera_id,int(frame['rendering_frame']),
            float(frame['rendering_time']),tf_stamp,epoch,self.camera_revision,*clock)
        evidence.validate(now,parked)
        rgb = np.asarray(frame['rgb'])[:,:,:3]
        depth = np.asarray(frame['distance_to_image_plane'],float)
        if rgb.shape[:2] != (self.intrinsics.height,self.intrinsics.width) or depth.shape != rgb.shape[:2]:
            raise ValueError('unregistered marker RGB-D')
        try:
            qr = self.decoder.decode(rgb,evidence,now=now,parked=parked,roi=self._decode_roi())
        except ValueError:
            if not allow_occlusion:
                raise
            # Decode could not read a clean frame under occlusion: present an empty held QR for this
            # frame (every slot then holds its last clean registration) so /sensors keeps publishing
            # the occluded flag through the reach, instead of aborting and starving the watchdog.
            qr = FrameQR(evidence, tuple(rgb.shape), rgb_digest(rgb), (), (), 0., 0)
        decoded = {item.text:np.asarray(item.corners) for item in qr.detections}
        registrations = []
        for slot in self.slots:
            if slot.marker_id not in decoded or slot.auxiliary_id and slot.auxiliary_id not in decoded:
                if allow_occlusion:
                    registrations.append(None)   # marker fully occluded this frame; hold downstream
                    continue
                raise ValueError('fresh support marker(s) not decoded; no geometry fallback')
            extra = {} if not slot.auxiliary_id else dict(auxiliary_pixels=decoded[slot.auxiliary_id],
                auxiliary_id=slot.auxiliary_id,decoded_auxiliary_id=slot.auxiliary_id,
                auxiliary_offset_xy=slot.auxiliary_offset_xy)
            try:
                registrations.append(register_marker(depth,decoded[slot.marker_id],self.intrinsics,transform,
                    marker_id=slot.marker_id,decoded_id=slot.marker_id,support_id=slot.support_id,
                    revision=slot.revision,side_m=slot.marker_side_m,evidence=evidence,now=now,
                    parked=parked,depth_error_m=self.depth_error_m,**extra))
            except ValueError:
                if allow_occlusion:
                    registrations.append(None)   # marker partially occluded (corner depth/fit); hold
                    continue
                raise
        self.qr_observation = qr
        return transform,evidence,parked,depth,registrations

    def commission(self, frame, now, empty_polygons, *, lifetime_s=600.,
                   commissioning_freshness_s=5.):
        """Explicitly acquire baselines from observed empty supports.

        empty_polygons[location] = (support_quad_uv, witness_quad_uv), selected
        from that measured frame. Source starts occupied, so a real prior empty
        commissioning frame is required; never manufacture it from a setup dict.

        Commissioning is a static-scene step whose full-resolution decode+calibrate can
        exceed the live 0.5 s acquisition-to-use budget, so it runs under its own
        ``commissioning_freshness_s`` acquisition-to-use bound (default 5 s): a stall guard,
        not the mission freshness. The live mission capture still uses the 0.5 s gate.
        """
        if not np.isfinite(commissioning_freshness_s) or commissioning_freshness_s < .5:
            raise ValueError('commissioning_freshness_s must be finite and at least the 0.5 s live budget')
        deadline = frame_deadline(frame,now,self.steady_clock(),budget_s=commissioning_freshness_s)
        transform,e,parked,depth,registrations = self._registrations(frame,now)
        pending = []
        for slot,registration in zip(self.slots,registrations):
            support,witness = empty_polygons[slot.location]
            model = calibrate_rigid_support(depth,support,witness,self.intrinsics,transform,registration,
                required_xy=slot.required_xy,calibration_id=f'{e.session_id}:{self.camera_id}:{e.frame_id}:{slot.support_id}',
                now=now,parked=parked,lifetime_s=lifetime_s)
            pending.append(RegisteredSupport(model))
        require_fresh(deadline,self.steady_clock)
        for slot,tracker in zip(self.slots,pending):
            slot.tracker = tracker
            # Cache the support's pixel box (support + witness) padded, so the live capture can
            # ROI-bound its decode to the commissioned region and fall back to the full frame.
            support,witness = empty_polygons[slot.location]
            box = np.concatenate([np.asarray(support,float),np.asarray(witness,float)])
            lo,hi = box.min(axis=0),box.max(axis=0)
            pad = np.maximum((hi-lo)*.25,96.)
            slot.decode_roi = (float(max(0,lo[0]-pad[0])),float(max(0,lo[1]-pad[1])),
                               float(min(self.intrinsics.width,hi[0]+pad[0])),
                               float(min(self.intrinsics.height,hi[1]+pad[1])))
        return tuple(t.geometry for t in pending)

    def __call__(self, frame, now):
        from ..restock_manipulation.isaac_sensors import CameraContext, SlotView
        allow = self.occlusion_window_s is not None
        transform,e,parked,depth,registrations = self._registrations(frame,now,allow_occlusion=allow)
        views = []
        for slot,registration in zip(self.slots,registrations):
            if slot.tracker is None:
                raise ValueError('observed empty commissioning baseline is required')
            if registration is not None:
                # prepare itself holds the last clean registration if the footprint/witness is
                # occluded (occlusion_window_s set); a present registration also feeds its
                # contradiction check against the held pose.
                view = slot.tracker.prepare(depth,self.intrinsics,transform,registration,None,now=now,parked=parked,
                    occlusion_window_s=self.occlusion_window_s,contradiction_m=self.occlusion_contradiction_m,
                    contradiction_rad=self.occlusion_contradiction_rad)
            else:
                # Marker fully occluded this frame: no current registration to contradict; hold on
                # the window alone (refuses past it, or if no clean view was ever established).
                view = slot.tracker.hold(depth,self.intrinsics,transform,e,now=now,parked=parked,
                    occlusion_window_s=self.occlusion_window_s,current_registration=None,
                    contradiction_m=self.occlusion_contradiction_m,contradiction_rad=self.occlusion_contradiction_rad)
            clearance = self.clearance_for_slot(slot.location,view,frame,now)
            if clearance is not None:
                clearance.validate(now,e,view.geometry)
                view = replace(view,clearance=clearance)
            views.append(SlotView.from_support(slot.location,view,slot.label_mask))
        return CameraContext(transform,e.transform_stamp,e.epoch,tuple(views),parked,
                             e.session_id,e.steady_domain,e.acquired_steady,self.qr_observation,
                             self.occlusion_window_s)

    def invalidate(self, reason):
        for slot in self.slots:
            if slot.tracker is not None:
                slot.tracker.invalidate(reason)


class StationSensors:
    """Aggregate distinct assigned cameras; no camera implicitly covers the mapper."""
    def __init__(self, *, cameras, profiles, pose_bridge_root, context_providers, location_cameras,
                 steady_clock=time.monotonic, wall_backstop_s=60., wall_clock=time.monotonic):
        from ..restock_manipulation.isaac_sensors import IsaacCartonSensors
        self.location_cameras = dict(location_cameras)
        self.steady_clock = steady_clock
        self.wall_backstop_s, self.wall_clock = wall_backstop_s, wall_clock
        if (not cameras or set(cameras) != set(context_providers)
                or any(camera not in cameras for camera in self.location_cameras.values())):
            raise ValueError('camera/location/context binding is incomplete')
        self.adapters = {key:IsaacCartonSensors(camera,pose_bridge_root=pose_bridge_root,
            profiles=profiles,camera_id=key,context_provider=context_providers[key],
            steady_clock=steady_clock,wall_backstop_s=wall_backstop_s,wall_clock=wall_clock)
            for key,camera in cameras.items()}
        self.sensed_target_bboxes = ()
        self.evidence = ()

    def capture(self, now, source, destination, motion, contact):
        from ..restock_manipulation.types import Sensors
        self.sensed_target_bboxes,self.evidence = (),()
        wall_started = self.wall_clock()
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
        # Sequential per-camera processing must not refresh the earliest acquisition deadline: hold
        # every camera's capture to the minimum valid_until_steady, re-checked against the shared
        # steady clock now that all cameras have been processed (evidence/bboxes stay empty on expiry).
        deadlines = [p.valid_until_steady for p in packets if p.valid_until_steady is not None]
        if deadlines:
            require_fresh(min(deadlines), self.steady_clock, wall_started=wall_started,
                          wall_backstop_s=self.wall_backstop_s, wall_clock=self.wall_clock)
        self.sensed_target_bboxes = tuple(box for key in selected for box in self.adapters[key].sensed_target_bboxes)
        self.evidence = tuple(item for key in selected for item in self.adapters[key].evidence)
        return Sensors(now,tuple(o for p in packets for o in p.observations),targets[0],
            all(p.parked for p in packets),packets[0].contact,min(p.feedback_time for p in packets),
            min(p.collision_time for p in packets),packets[0].joint_positions,
            tuple(location for p in packets for location in p.scanned_locations),min(p.scan_time for p in packets),
            min(deadlines) if deadlines else None)
