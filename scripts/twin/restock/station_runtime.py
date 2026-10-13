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

    def after_step(self, now, parked):
        self.now,self.parked=now,parked
        self.hosts.check()
        self.graph.check_health()
        self.station.tick(now,parked=parked)
        self.feedback.collision_time=self.station.collision_time

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
            self.invalidate_contexts(self.station.epoch)
            return result
        except Exception:
            self.stop_station()
            raise

    def capture(self, now, source, destination):
        if not self.station.ready(now):
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
