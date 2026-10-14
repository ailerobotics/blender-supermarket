"""Native JSON ROS transport for small simulation control-plane messages."""
import json
import time

from twin.restock_manipulation.ros_adapter import require_sim_domain


class JsonChannel:
    def __init__(self, *, root, publish_topic, subscribe_topic, host_control=None):
        require_sim_domain()
        import omni.graph.core as og
        import omni.usd
        if omni.usd.get_context().get_stage().GetPrimAtPath(root):
            raise ValueError('refusing to overwrite native ROS channel')
        self.og,self.root=og,root
        # The Isaac->host PUBLISH (station/command) goes through the host-DDS subprocess, since the
        # OmniGraph ROS2 publisher is invisible to the host UDPv4 DDS (run-57/59: command pub=0).
        # The host->Isaac station/status SUBSCRIBER stays on OmniGraph (it discovers the ack writer).
        self.host_control, self.publish_topic = host_control, publish_topic
        directions = (('send',publish_topic,'ROS2Publisher'),
                      ('receive',subscribe_topic,'ROS2Subscriber'))
        if host_control is not None:
            directions = directions[1:]  # publish out-of-Isaac; only build the receive graph
        for direction,topic,kind in directions:
            key=og.Controller.Keys
            graph,_,_,_=og.Controller.edit({'graph_path':root+'/'+direction,'evaluator_name':'execution'}, {
                key.CREATE_NODES:[('tick','omni.graph.action.OnImpulseEvent'),
                    ('context','isaacsim.ros2.bridge.ROS2Context'),('wire','isaacsim.ros2.bridge.'+kind)],
                key.SET_VALUES:[('context.inputs:useDomainIDEnvVar',False),('context.inputs:domain_id',88),
                    ('wire.inputs:topicName',topic),('wire.inputs:messagePackage','std_msgs'),
                    ('wire.inputs:messageSubfolder','msg'),('wire.inputs:messageName','String')],
                key.CONNECT:[('tick.outputs:execOut','wire.inputs:execIn'),
                             ('context.outputs:context','wire.inputs:context')]})
            og.Controller.evaluate_sync(graph)

    def _pulse(self,direction):
        path=self.root+'/'+direction
        self.og.Controller.attribute(path+'/tick.state:enableImpulse').set(True)
        self.og.Controller.evaluate_sync(path)

    def publish(self,value):
        payload=json.dumps(value,allow_nan=False)
        if self.host_control is not None:
            self.host_control.publish(self.publish_topic,payload)  # host-DDS subprocess
            return
        self.og.Controller.attribute(self.root+'/send/wire.inputs:data').set(payload)
        self._pulse('send')

    def read(self):
        self._pulse('receive')
        raw=self.og.Controller.attribute(self.root+'/receive/wire.outputs:data').get()
        return json.loads(raw) if raw else None


class StationClient:
    def __init__(self, channel, *, wall_clock=time.monotonic):
        self.channel,self.wall_clock=channel,wall_clock
        self.epoch=0
        self.sequence=0
        self.active=False
        self.latest=None
        self.last_send=-1.
        self.last_reply=None
        self.now=0.

    def begin(self,camera_frame,now):
        if self.active:
            raise RuntimeError('stop previous parked station before preparing another')
        self.epoch+=1
        self.sequence=0
        self.camera_frame=camera_frame
        self.active=True
        self.latest=None
        self.started=self.wall_clock()
        self.last_send=-1.
        self.last_reply=None
        self.tick(now,parked=True)
        return self

    def tick(self,now,*,parked):
        self.now=now
        reply=self.channel.read()
        if reply and reply.get('epoch')==self.epoch and reply!=self.latest:
            if not 0<=now-reply.get('stamp',-1)<=.5:
                raise ValueError('stale parked-station acknowledgement')
            self.latest=reply
            self.last_reply=self.wall_clock()
        if not self.active:
            return
        if not parked:
            self.stop()
            raise ValueError('station lost measured parking')
        if self.latest and self.latest.get('state')=='failed':
            raise RuntimeError(self.latest.get('reason','host station failed'))
        if self.wall_clock()-(self.last_reply or self.started)>10.:
            raise RuntimeError('station supervisor heartbeat unavailable')
        if now-self.last_send>=.1 or self.last_send<0:
            acknowledged=self.latest is not None
            self.channel.publish(dict(action='heartbeat' if acknowledged else 'prepare',
                epoch=self.epoch,sequence=self.sequence if acknowledged else 0,
                stamp=now,parked=True,camera_frame=self.camera_frame))
            self.sequence+=1
            self.last_send=now

    def ready(self,now):
        reply=self.latest
        return bool(self.active and reply and reply.get('state')=='ready'
                    and 0<=now-reply.get('stamp',-1)<=.5
                    and 0<=now-reply.get('collision_time',-1)<=.5)

    @property
    def collision_time(self):
        return self.latest['collision_time'] if self.ready(self.now) else -1.

    def stop(self):
        if self.active:
            self.sequence+=1
            self.channel.publish(dict(action='stop',epoch=self.epoch,sequence=self.sequence,
                                      stamp=self.now,parked=True))
        self.active=False
        self.latest=None
