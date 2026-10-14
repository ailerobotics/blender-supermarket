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

# Stages whose work consumes freshly rendered pixels (inventory perception, the
# manipulation camera->nvblox depth, the final inventory verification). Only these
# need world.step(render=True) every tick; the navigation-only stages (couple,
# outbound, park, return) drive the base from the PhysX lidar raycast + encoders and
# never read a rendered frame, so they render only on the recorder's cadence ticks.
SENSING_ACTIONS = frozenset({'collect', 'transfer', 'verify'})


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
                 on_action=None, realtime=True, record_stride=2, masked_depth=None):
        self.world,self.robot=world,robot
        self.robot_body,self.cart_body=robot_body,cart_body
        self.mobile,self.range_sensor=mobile,range_sensor
        self.transport,self.contact=transport,contact
        self.hold_arm,self.verification=hold_arm,verification
        self.recording_camera,self.recorder=recording_camera,recorder
        self.native_transport=native_transport
        self.after_step,self.on_action=after_step,on_action
        self.realtime=realtime
        if type(record_stride) is not int or record_stride<1:
            raise ValueError('record_stride must be a positive int (physics ticks per recorded frame)')
        self.record_stride=record_stride
        # Optional masked-depth publisher (Isaac-side rig self-pixel invalidation for the
        # host nvblox consumer). Published only on sensing-stage ticks; None keeps the
        # legacy raw-depth OmniGraph path untouched. See masked_depth.MaskedDepthPublisher.
        self.masked_depth=masked_depth
        # Current mission action (set by begin_action); gates per-tick rendering + sensing.
        self.current_action=None
        self.wall_origin=time.monotonic()
        self.tick=0
        self.names=tuple(robot.dof_names)
        self.wheels=tuple(self.names.index(n) for n in ('left_wheel_joint','right_wheel_joint'))
        self.sensor_models={'docking':'ideal_simulated_rigid_body_proprioception',
                            'navigation':'wheel_encoders_and_physical_range_scan',
                            'inventory':'rendered_rgb_depth_only'}

    def _frame_plan(self, upcoming_tick):
        """(sensing_tick, record_tick) for `upcoming_tick`: whether a sensing stage needs a
        fresh render and whether the recorder captures this tick. render = either is True."""
        sensing=self.current_action in SENSING_ACTIONS
        record=(self.recorder is not None and self.recording_camera is not None
                and upcoming_tick%self.record_stride==0)
        return sensing,record

    def advance(self, dt):
        if not math.isclose(float(self.world.get_physics_dt()),dt,abs_tol=1e-8):
            raise ValueError('mission and physics step mismatch')
        if self.realtime:
            # ROS runs on a separate host interpreter; accelerated simulation can
            # otherwise outrun fresh trajectory commands and map acknowledgements.
            remaining=self.wall_origin+(self.tick+1)*dt-time.monotonic()
            if remaining>0:
                time.sleep(remaining)
        # Render only when this tick produces a frame someone reads: a recorder-cadence
        # frame, or any sensing stage (perception / manip depth / verification). The
        # dominant per-tick cost is the whole-scene render; the navigation-only stages
        # read no pixels, so skipping their renders is the headroom under the wall cap.
        self.sensing_tick,self.record_tick=self._frame_plan(self.tick+1)
        self.world.step(render=bool(self.sensing_tick or self.record_tick))
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
        if self.recorder is not None:
            # Per-tick nav-coverage trace (JSONL): for every tick, evaluate the uncoupled
            # gate's exact coverage probe (coverage_certifies at the real gate radius
            # footprint.radius(False)+position_uncertainty) plus a bracket, and record the
            # robot pose/heading and the forward-blind bearings. The previous sparse sampler
            # (ticks 1/120/300/600) missed the failing window entirely because collect drives
            # for a while before coverage flips, so the fail tick was never captured. One line
            # per tick localizes exactly when/where insufficient_sensor_coverage first appears.
            try:
                import json as _json, math as _math
                from twin.navigation.core import coverage_certifies, sweep_extent
                heading=odometry.pose[2]
                field=navigation.coverage
                # Gate's measured stopping distance at the worst-case commanded speed and at
                # the turn-in-place command (v=0 -> stop from residual measured speed only).
                sp_drive=max(0.25,float(np.linalg.norm(rv)))
                stop_drive=sp_drive*0.3+sp_drive*sp_drive/(2*0.4)
                sp_turn=float(np.linalg.norm(rv))
                stop_turn=sp_turn*0.3+sp_turn*sp_turn/(2*0.4)
                unc=0.05  # Odometry.position_uncertainty default used in advance()
                rec={}
                for tag,R,stop in (('gate_drive',0.70,stop_drive),('gate_turn',0.70,stop_turn),
                                   ('bracket_0p60',0.60,stop_drive),('bracket_0p90',0.90,stop_drive)):
                    rad=R+unc
                    blind=[round(_math.degrees(wrap(a-heading)),1) for a,reach in field.beams
                           if sweep_extent(wrap(a-heading),rad,stop)>rad+1e-9
                           and reach<sweep_extent(wrap(a-heading),rad,stop)-1e-9]
                    rec[tag]=dict(rad=round(rad,3),stop=round(stop,3),
                                  cert=coverage_certifies(navigation,heading,rad,stop),
                                  nblind=len(blind),blind_deg=blind[:12])
                line=_json.dumps(dict(tick=self.tick,t=round(now,3),
                    wall=round(time.monotonic()-self.wall_origin,3),action=self.current_action,
                    rendered=bool(self.sensing_tick or self.record_tick),
                    x=round(r.x,3),y=round(r.y,3),yaw_deg=round(_math.degrees(heading),1),
                    lin=round(float(np.linalg.norm(rv)),4),coupled=bool(coupled),parked=bool(parked),
                    vel=dict(rv=round(float(np.linalg.norm(rv)),4),rw=round(float(np.linalg.norm(rw)),4),
                             cv=round(float(np.linalg.norm(cv)),4),cw=round(float(np.linalg.norm(cw)),4)),
                    unknown=sum(1 for u in scan.unknown if u),probe=rec))
                with (self.recorder.directory.parent/'nav-trace.jsonl').open('a') as _f:
                    _f.write(line+'\n')
            except Exception as _e:
                try:
                    if self.tick==1:
                        (self.recorder.directory.parent/'nav-trace.err').write_text(repr(_e))
                except Exception: pass
        if self.masked_depth is not None and self.sensing_tick:
            # Rig-masked depth (+ camera info) for the host nvblox consumer, published only
            # while a sensing stage needs the manipulation collision world. Off (None) by
            # default; the raw-depth OmniGraph path is unchanged until the host opts in.
            self.masked_depth.publish(now)
        evidence=self.verification(now)
        if self.record_tick:
            frame=self.recording_camera.get_current_frame()
            self.recorder.capture(frame)
        return Packet(f'physics:{self.tick}','sensor',now,
            odometry,navigation,docking,None,
            ready,coupled,articulation,parked,evidence)

    def begin_action(self,action,job):
        self.current_action=action
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
