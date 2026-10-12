"""Wheel encoders, map scan matching and Isaac PhysX range observations.

No SimulationApp is created here. The ray sensor uses simulated physical geometry
only to generate ranges. Its world pose never enters the localization output.
"""
from dataclasses import dataclass
import math
from .core import Odometry, Observation, wrap


@dataclass(frozen=True)
class Scan:
    stamp: float
    ranges: tuple
    angle_min: float
    angle_increment: float
    range_max: float
    source: str = 'simulated_lidar'

    def points(self, pose):
        x, y, yaw = pose
        return tuple((x+r*math.cos(yaw+self.angle_min+i*self.angle_increment),
                      y+r*math.sin(yaw+self.angle_min+i*self.angle_increment))
                     for i,r in enumerate(self.ranges) if 0 < r < self.range_max)


class WheelOdometry:
    """Integrate measured continuous left/right wheel joint angles, in radians."""
    def __init__(self, wheel_radius, wheel_separation, initial_pose=(0,0,0)):
        if wheel_radius <= 0 or wheel_separation <= 0:
            raise ValueError('wheel dimensions must be positive')
        self.radius, self.separation = wheel_radius, wheel_separation
        self.pose, self.previous = tuple(initial_pose), None

    def update(self, left, right, stamp):
        if not all(math.isfinite(v) for v in (left,right,stamp)):
            raise ValueError('invalid encoder sample')
        if self.previous is None:
            self.previous = left,right,stamp
            return Odometry(self.pose,stamp,source='wheel_odometry')
        pl,pr,pt = self.previous
        dt = stamp-pt
        if dt <= 0:
            raise ValueError('encoder timestamp must advance')
        dl,dr = (left-pl)*self.radius,(right-pr)*self.radius
        ds,da = (dl+dr)/2,(dr-dl)/self.separation
        x,y,a = self.pose
        scale = 1 if abs(da)<1e-9 else math.sin(da/2)/(da/2)
        self.pose = x+ds*scale*math.cos(a+da/2),y+ds*scale*math.sin(a+da/2),wrap(a+da)
        self.previous = left,right,stamp
        return Odometry(self.pose,stamp,ds/dt,da/dt,source='wheel_odometry')


class MapLocalizer:
    """Bounded scan-to-static-map matcher, initialized by a declared spawn prior.

    Not global kidnapped-robot recovery. Insufficient mapped hits or a poor fit
    raises instead of silently publishing odometry as localized. Use AMCL with
    the ROS path for production localization; this is the in-process CPU route.
    """
    def __init__(self, grid, initial_map_pose, *, max_error=0.15):
        self.grid = grid
        self.prior = tuple(initial_map_pose)
        self.last_odom = None
        self.max_error = max_error
        self.map_to_odom = (0.0,0.0,0.0)

    def _distance(self,p):
        x,y=p
        a,b,c,d=self.grid.bounds
        distances=[abs(x-a),abs(x-c),abs(y-b),abs(y-d)]
        for a,b,c,d in self.grid.obstacles:
            if a <= x <= c and b <= y <= d:
                distances.append(min(x-a,c-x,y-b,d-y))
            else:
                distances.append(math.hypot(max(a-x,0,x-c),max(b-y,0,y-d)))
        return min(distances)

    def update(self, odom, scan):
        if abs(odom.stamp-scan.stamp)>0.05:
            raise ValueError('scan/encoder time mismatch')
        if scan.source != 'simulated_lidar' or not scan.ranges or any(
                not math.isfinite(r) or r <= 0 or r > scan.range_max for r in scan.ranges):
            raise ValueError('invalid scan')
        prior=self.prior
        if self.last_odom is not None:
            dx,dy=odom.pose[0]-self.last_odom[0],odom.pose[1]-self.last_odom[1]
            a=wrap(prior[2]-self.last_odom[2])
            prior=(prior[0]+dx*math.cos(a)-dy*math.sin(a),
                   prior[1]+dx*math.sin(a)+dy*math.cos(a),
                   wrap(prior[2]+odom.pose[2]-self.last_odom[2]))
        # Downsample the scan evenly, retain enough independent mapped surfaces.
        hits=[(i,r) for i,r in enumerate(scan.ranges) if r<scan.range_max]
        hits=hits[::max(1,len(hits)//90)]
        if len(hits)<12:
            raise ValueError('insufficient mapped range returns')
        candidates=[]
        for dx in (-0.10,-0.05,0.0,0.05,0.10):
            for dy in (-0.10,-0.05,0.0,0.05,0.10):
                for da in (-0.04,-0.02,0.0,0.02,0.04):
                    p=(prior[0]+dx,prior[1]+dy,wrap(prior[2]+da))
                    distances=sorted(self._distance((p[0]+r*math.cos(p[2]+scan.angle_min+i*scan.angle_increment),
                                                    p[1]+r*math.sin(p[2]+scan.angle_min+i*scan.angle_increment)))
                                     for i,r in hits)
                    # Ignore up to 20% dynamic returns without accepting an empty fit.
                    score=sum(distances[:max(12,int(len(distances)*0.8))])/max(12,int(len(distances)*0.8))
                    candidates.append((score,dx*dx+dy*dy+da*da,p))
        score,_,pose=min(candidates)
        if score>self.max_error:
            raise ValueError('localization lost')
        self.prior,self.last_odom=pose,odom.pose
        a=wrap(pose[2]-odom.pose[2])
        self.map_to_odom=(pose[0]-odom.pose[0]*math.cos(a)+odom.pose[1]*math.sin(a),
                          pose[1]-odom.pose[0]*math.sin(a)-odom.pose[1]*math.cos(a),a)
        return Odometry(pose,odom.stamp,odom.linear,odom.angular,
                        position_uncertainty=max(0.10,2*score))


class IsaacRangeSensor:
    """Idealized multi-height 360 degree collision-ray scanner (not RTX optics).

    Construct after Isaac boot. get_sensor_pose() returns physical (x,y,yaw).
    Self hits stop rays and become range_min returns: occlusion never becomes
    invented free space. Mount outside self geometry or provide separate sensors.
    """
    def __init__(self,get_sensor_pose,*,heights=(0.15,0.5,1.0),beams=360,range_max=10.0):
        from omni.physx import get_physx_scene_query_interface
        self.query=get_physx_scene_query_interface()
        self.get_pose=get_sensor_pose
        self.heights,self.beams,self.range_max=heights,beams,range_max

    def read(self,stamp):
        x,y,yaw=self.get_pose()
        ranges=[]
        increment=2*math.pi/self.beams
        for i in range(self.beams):
            angle=yaw-math.pi+i*increment
            best=self.range_max
            for z in self.heights:
                hit=self.query.raycast_closest((x,y,z),(math.cos(angle),math.sin(angle),0.0),self.range_max)
                if hit['hit']:
                    best=min(best,max(0.001,float(hit['distance'])))
            ranges.append(best)
        return Scan(stamp,tuple(ranges),-math.pi,increment,self.range_max)


def observation(scan,localized,*,coupled,arm_transport,articulation=0.0):
    """Transform returns through estimated pose; caller supplies measured state."""
    if (not math.isfinite(scan.range_max) or scan.range_max<=0
            or not math.isfinite(scan.angle_min) or not math.isfinite(scan.angle_increment)
            or scan.angle_increment<=0 or len(scan.ranges)<3
            or any(math.isnan(r) or r<=0 or (math.isfinite(r) and r>scan.range_max)
                   for r in scan.ranges)):
        raise ValueError('invalid scan; unknown beams must not become free space')
    coverage=scan.range_max if len(scan.ranges)*scan.angle_increment>=2*math.pi-.01 else 0.0
    return Observation(scan.stamp,scan.points(localized.pose),coupled,arm_transport,
                       articulation,scan.source,coverage)
