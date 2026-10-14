"""Measured physics-to-mission port, constructed only inside a guarded Isaac app.

Base/cart rigid-body measurements model ideal proprioception and latch sensors.
They never replace localized navigation or rendered inventory observations.
No method sets a body pose, joint position, payload parent or collision flag.
"""
import math
import time

import numpy as np

from twin.mobile import DockObservation, Pose2, wrap
from twin.navigation.core import Odometry
from twin.navigation.sensors import observation as nav_observation
from twin.urdf_probe import parse_urdf, link_world_transforms, geometry_world_points, resolve_package_uri
from .runtime import Packet


def body_measurement(body):
    position, quaternion = body.get_world_pose()  # Isaac quaternion is wxyz
    position, quaternion = np.asarray(position,float),np.asarray(quaternion,float)
    linear = np.asarray(body.get_linear_velocity(),float)
    angular = np.asarray(body.get_angular_velocity(),float)
    if (position.shape != (3,) or quaternion.shape != (4,) or
            linear.shape != (3,) or angular.shape != (3,) or
            not all(np.isfinite(v).all() for v in (position,quaternion,linear,angular)) or
            not math.isclose(float(np.linalg.norm(quaternion)),1.,abs_tol=1e-4)):
        raise ValueError('invalid physical body measurement')
    w,x,y,z = quaternion
    yaw = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return Pose2(float(position[0]),float(position[1]),yaw),float(position[2]),linear,angular


class TransportEnvelope:
    """Current encoder FK against the same collision geometry used offline."""
    def __init__(self, urdf, package_roots, radius):
        self.robot = parse_urdf(str(urdf))
        self.radius = float(radius)
        self.clouds = {}
        for name,link in self.robot.links.items():
            points=[]
            for geometry in link.collisions:
                cloud=geometry_world_points(geometry,np.eye(4),
                    lambda uri:resolve_package_uri(uri,package_roots)[0])
                if cloud is None or not np.isfinite(cloud).all():
                    raise ValueError('unresolved transport collision geometry: '+name)
                points.extend(cloud)
            if points:
                self.clouds[name]=np.asarray(points)
        self.required={joint.name for joint in self.robot.joints
                       if joint.jtype in ('revolute','continuous','prismatic')}

    def contains(self, positions):
        if not self.required <= positions.keys() or not all(
                math.isfinite(positions[name]) for name in self.required):
            return False
        transforms=link_world_transforms(self.robot,positions)
        for name,points in self.clouds.items():
            matrix=transforms[name]
            cloud=points@matrix[:3,:3].T+matrix[:3,3]
            if np.linalg.norm(cloud[:,:2],axis=1).max() > self.radius:
                return False
        return True


class MissionCamera:
    """Keep real pixels/frame IDs while expressing sensor time at mission origin."""
    def __init__(self, camera, clock_origin):
        self.camera,self.clock_origin=camera,float(clock_origin)

    def get_current_frame(self, *args, **kwargs):
        frame=dict(self.camera.get_current_frame(*args,**kwargs))
        if 'rendering_time' in frame:
            frame['rendering_time']=float(frame['rendering_time'])-self.clock_origin
        return frame

    def __getattr__(self,name):
        return getattr(self.camera,name)


class IsaacPort:
    def __init__(self, *, world, robot, robot_body, cart_body, mobile, range_sensor,
                 transport, contact, hold_arm, verification, recording_camera=None,
                 recorder=None, native_transport=None, after_step=None,
                 on_action=None, realtime=True):
        self.world,self.robot=world,robot
        self.robot_body,self.cart_body=robot_body,cart_body
        self.mobile,self.range_sensor=mobile,range_sensor
        self.transport,self.contact=transport,contact
        self.hold_arm,self.verification=hold_arm,verification
        self.recording_camera,self.recorder=recording_camera,recorder
        self.native_transport=native_transport
        self.after_step,self.on_action=after_step,on_action
        self.realtime=realtime
        self.wall_origin=time.monotonic()
        self.tick=0
        self.names=tuple(robot.dof_names)
        self.wheels=tuple(self.names.index(n) for n in ('left_wheel_joint','right_wheel_joint'))
        self.sensor_models={'docking':'ideal_simulated_rigid_body_proprioception',
                            'navigation':'wheel_encoders_and_physical_range_scan',
                            'inventory':'rendered_rgb_depth_only'}

    def advance(self, dt):
        if not math.isclose(float(self.world.get_physics_dt()),dt,abs_tol=1e-8):
            raise ValueError('mission and physics step mismatch')
        if self.realtime:
            # ROS runs on a separate host interpreter; accelerated simulation can
            # otherwise outrun fresh trajectory commands and map acknowledgements.
            remaining=self.wall_origin+(self.tick+1)*dt-time.monotonic()
            if remaining>0:
                time.sleep(remaining)
        self.world.step(render=True)
        self.tick+=1
        # Mission-relative acquisition clock on the nominal fixed timestep. Deriving it
        # from world.current_time instead drifts: Isaac's physics dt differs from the
        # mission dt by ~1e-9 per step (both pass the isclose guard above), which left
        # the packet a few nanoseconds ahead of the mission's dt-accumulated clock and
        # tripped the freshness lower bound on the very first tick.
        now=self.tick*dt
        if self.native_transport is not None:
            self.native_transport.tick(now)
        joints=np.asarray(self.robot.get_joint_positions(),float)
        if joints.shape != (len(self.names),) or not np.isfinite(joints).all():
            raise ValueError('invalid measured articulation encoders')
        positions=dict(zip(self.names,joints))
        r,rz,rv,rw=body_measurement(self.robot_body)
        c,cz,cv,cw=body_measurement(self.cart_body)
        spec=self.mobile.spec
        frames=spec.manifest()
        docking=DockObservation(r,c,rz+frames['robot_hitch_local'][2],
            cz+frames['cart_hitch_local'][2],float(np.linalg.norm(rv)),float(np.linalg.norm(cv)),
            float(rw[2]),float(cw[2]),age=0.)
        coupled=self.mobile.coupling_state(docking)['healthy']
        parked=bool(max(np.linalg.norm(v) for v in (rv,rw,cv,cw))<.005)
        if self.after_step is not None:
            self.after_step(now,parked)
        ready=self.transport.contains(positions) and not self.contact.read()
        scan=self.range_sensor.read(now)
        articulation=wrap(r.yaw-c.yaw)
        # Reference navigation consumes a localized Odometry + an obstacle Observation,
        # not raw wheel encoders and a Scan. The measured base pose is the reference
        # localization (default source 'wheel_odometry_localized'); the Scan is transformed
        # into map-frame obstacles about that pose (source 'simulated_lidar').
        odometry=Odometry((r.x,r.y,r.yaw),now,float(np.linalg.norm(rv)),float(rw[2]))
        navigation=nav_observation(scan,odometry,coupled=coupled,arm_transport=ready,
                                   articulation=articulation)
        evidence=self.verification(now)
        if self.recorder is not None and self.tick%2 == 0:
            frame=self.recording_camera.get_current_frame()
            self.recorder.capture(frame)
        return Packet(f'physics:{self.tick}','sensor',now,
            odometry,navigation,docking,None,
            ready,coupled,articulation,parked,evidence)

    def begin_action(self,action,job):
        if self.on_action is not None:
            self.on_action(action,job)

    def controls(self, controls):
        for control in controls:
            kind=control.get('kind')
            if kind=='gripper':
                self.contact.command(control['enabled'])
            elif kind=='stop':
                self.stop()
            elif kind!='move':
                raise ValueError('unknown manipulation control')
            # Planned move commands are owned by the motion server, not replayed.

    def stop(self):
        try:
            self.mobile.park()
        finally:
            self.hold_arm()
