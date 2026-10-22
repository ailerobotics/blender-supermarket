"""Bind parked mapper epochs, rendered sensing and post-transfer verification."""
from .verification import InventoryVerifier


class StationRuntime:
    def __init__(self, *, station, stream, graph, hosts, feedback, contact,
                 sensors, mapper_cameras, invalidate_contexts):
        self.station,self.stream,self.graph=station,stream,graph
        self.hosts,self.feedback,self.contact=hosts,feedback,contact
        self.sensors,self.mapper_cameras=sensors,dict(mapper_cameras)
        self.invalidate_contexts=invalidate_contexts
        if set(self.mapper_cameras)!={'collect','transfer'}:
            raise ValueError('explicit source and shelf mapper camera selections required')
        self.now=0.
        self.parked=False
        self.verifier=InventoryVerifier(self.capture)
        self.frames=None
        self.acquire_views=None

    def bind_frames(self, frames, acquire_views):
        """Wire native acquisition before starting a station.

        acquire_views(now) yields (camera_name, cloned_frame, measured_transform,
        transform_stamp). It must retain each frame's real acquisition clock.
        Sensors must use frames.cameras, not read newer native pixels directly.
        """
        if self.station.active or self.frames is not None:
            raise ValueError('camera queues must be bound once before station startup')
        self.frames,self.acquire_views=frames,acquire_views

    def after_step(self, now, parked):
        self.now,self.parked=now,parked
        self.hosts.check()
        self.graph.check_health()
        self.station.tick(now,parked=parked)
        self.feedback.collision_time=self.station.collision_time
        if self.frames is not None and self.station.active and parked:
            for name,frame,transform,stamp in self.acquire_views(now):
                self.frames.append(name,frame,base_from_camera=transform,
                    transform_stamp=stamp,now=now,parked=parked)

    def prepare(self, operation, job):
        if not self.parked:
            raise ValueError('measured robot and independent cart parking required')
        if operation not in self.mapper_cameras:
            raise ValueError('unsupported station operation')
        self.stop_station()
        camera=self.mapper_cameras[operation]
        self.stream.select(camera)
        try:
            result=self.station.begin('restock_camera_'+camera,self.now)
            if self.frames is not None:
                self.frames.begin_epoch(self.station.epoch)
            # Clear current observations/clearance only. Ordinary parking does
            # not invalidate the immutable empty-support commissioning geometry.
            self.invalidate_contexts(self.station.epoch)
            return result
        except Exception:
            self.stop_station()
            raise

    def capture(self, now, source, destination):
        if not self.station.ready(now):
            return None
        if self.frames is not None and not self.frames.select(source,destination,
                now=now,collision_time=self.station.collision_time):
            return None
        return self.sensors.capture(now,source,destination,self.feedback,self.contact)

    def begin_action(self, action, job):
        self.verifier.stop()
        if action in ('collect','couple','outbound','return'):
            self.stop_station()
        elif action=='verify':
            # New post-transfer frames, while retaining this parked mapper epoch.
            self.verifier.begin(job,self.now)

    def stop_station(self):
        try:
            self.station.stop()
        finally:
            self.feedback.collision_time=-1.
            self.stream.select(None)

    def close(self):
        self.verifier.stop()
        self.stop_station()
