"""Wheel encoders, map scan matching and Isaac PhysX range observations.

No SimulationApp is created here. The ray sensor uses simulated physical geometry
only to generate ranges. Its world pose never enters the localization output.
"""
from dataclasses import dataclass
import math
from .core import Odometry, Observation, CoverageField, wrap


@dataclass(frozen=True)
class Scan:
    stamp: float
    ranges: tuple
    angle_min: float
    angle_increment: float
    range_max: float
    source: str = 'simulated_lidar'
    sensor_to_base: tuple = (0.0, 0.0, 0.0)  # sensor origin x,y,yaw in base frame
    unknown: tuple = ()  # any sampled height occluded: never certifies clearing
    external_returns: tuple = ()  # (beam index, range), retained in unknown sectors

    def __post_init__(self):
        if len(self.sensor_to_base) != 3 or not all(math.isfinite(v) for v in self.sensor_to_base):
            raise ValueError('finite calibrated sensor-to-base pose required')
        if self.unknown and (len(self.unknown)!=len(self.ranges) or
                             any(type(v) is not bool for v in self.unknown)):
            raise ValueError('unknown mask must contain one bool per beam')
        for i,r in self.external_returns:
            if (type(i) is not int or not 0<=i<len(self.ranges) or
                    not math.isfinite(r) or not 0<r<self.range_max):
                raise ValueError('invalid retained external return')

    def validate(self):
        if (not math.isfinite(self.range_max) or self.range_max<=0
                or not math.isfinite(self.angle_min) or not math.isfinite(self.angle_increment)
                or self.angle_increment<=0 or len(self.ranges)<3):
            raise ValueError('invalid scan geometry')
        for i,r in enumerate(self.ranges):
            hidden=bool(self.unknown and self.unknown[i])
            # NaN survives a raw polar-scan transport as unknown, not max range.
            if (hidden and not math.isnan(r)) or (not hidden and
                    (not math.isfinite(r) or not 0<r<=self.range_max)):
                raise ValueError('unknown beams require explicit mask and NaN range')

    def hits(self):
        return tuple((i,r) for i,r in enumerate(self.ranges)
                     if math.isfinite(r) and 0<r<self.range_max)+self.external_returns

    def sensor_pose(self, base_pose):
        x,y,yaw = base_pose
        dx,dy,da = self.sensor_to_base
        return (x+dx*math.cos(yaw)-dy*math.sin(yaw),
                y+dx*math.sin(yaw)+dy*math.cos(yaw),yaw+da)

    def points(self, pose):
        x, y, yaw = self.sensor_pose(pose)
        return tuple((x+r*math.cos(yaw+self.angle_min+i*self.angle_increment),
                      y+r*math.sin(yaw+self.angle_min+i*self.angle_increment))
                     for i,r in self.hits())


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
        scan.validate()
        if scan.source != 'simulated_lidar':
            raise ValueError('invalid scan')
        prior=self.prior
        if self.last_odom is not None:
            dx,dy=odom.pose[0]-self.last_odom[0],odom.pose[1]-self.last_odom[1]
            a=wrap(prior[2]-self.last_odom[2])
            prior=(prior[0]+dx*math.cos(a)-dy*math.sin(a),
                   prior[1]+dx*math.sin(a)+dy*math.cos(a),
                   wrap(prior[2]+odom.pose[2]-self.last_odom[2]))
        # Downsample the scan evenly, retain enough independent mapped surfaces.
        hits=list(scan.hits())
        hits=hits[::max(1,len(hits)//90)]
        if len(hits)<12:
            raise ValueError('insufficient mapped range returns')
        candidates=[]
        for dx in (-0.10,-0.05,0.0,0.05,0.10):
            for dy in (-0.10,-0.05,0.0,0.05,0.10):
                for da in (-0.04,-0.02,0.0,0.02,0.04):
                    p=(prior[0]+dx,prior[1]+dy,wrap(prior[2]+da))
                    sx,sy,sa=scan.sensor_pose(p)
                    distances=sorted(self._distance((sx+r*math.cos(sa+scan.angle_min+i*scan.angle_increment),
                                                    sy+r*math.sin(sa+scan.angle_min+i*scan.angle_increment)))
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
    Self hits stop rays and become unknown. Identity is collider/rigid-body USD
    subtree membership, never inferred from point location or cart ground truth.
    Heights are WORLD Z values; they are not offsets from the lidar_frame.
    """
    def __init__(self,get_sensor_pose,*,heights=(0.15,0.5,1.0),beams=360,range_max=10.0,
                 sensor_to_base=(0.0,0.0,0.0),self_paths=(),query=None):
        if query is None:
            from omni.physx import get_physx_scene_query_interface
            query=get_physx_scene_query_interface()
        self.query=query
        self.self_paths=tuple(p.rstrip('/') for p in self_paths)
        if any(not p.startswith('/') or p=='/' or not p or '//' in p or
               any(s in ('.','..') for s in p.split('/')) for p in self.self_paths):
            raise ValueError('self paths must be explicit absolute USD subtrees')
        if (type(beams) is not int or beams<3 or not heights or
                not all(math.isfinite(z) for z in heights) or
                not math.isfinite(range_max) or range_max<=0):
            raise ValueError('invalid range sensor configuration')
        self.get_pose=get_sensor_pose
        self.heights,self.beams,self.range_max=heights,beams,range_max
        self.sensor_to_base=Scan(0,(),0,1,range_max,sensor_to_base=sensor_to_base).sensor_to_base

    def _self_hit(self,hit):
        return any(path==root or path.startswith(root+'/')
                   for path in (hit.get('collision',''),hit.get('rigidBody',''))
                   if isinstance(path,str) for root in self.self_paths)

    def read(self,stamp):
        x,y,yaw=self.get_pose()
        if not all(math.isfinite(v) for v in (x,y,yaw,stamp)):
            raise ValueError('invalid range sensor pose or timestamp')
        ranges=[];unknown=[];external=[]
        increment=2*math.pi/self.beams
        for i in range(self.beams):
            angle=yaw-math.pi+i*increment
            best=self.range_max
            hidden=False
            for z in self.heights:
                hit=self.query.raycast_closest((x,y,z),(math.cos(angle),math.sin(angle),0.0),self.range_max)
                if hit['hit']:
                    distance=float(hit['distance'])
                    if not math.isfinite(distance) or not 0<=distance<=self.range_max:
                        raise ValueError('invalid closest-hit distance')
                    if self._self_hit(hit):
                        hidden=True
                    else:
                        best=min(best,max(0.001,distance))
            unknown.append(hidden)
            ranges.append(math.nan if hidden else best)
            if hidden and best<self.range_max:
                external.append((i,best))
        return Scan(stamp,tuple(ranges),-math.pi,increment,self.range_max,
                    sensor_to_base=self.sensor_to_base,unknown=tuple(unknown),
                    external_returns=tuple(external))


def observation(scan,localized,*,coupled,arm_transport,articulation=0.0):
    """Transform returns through estimated pose; caller supplies measured state.

    Builds a self-occlusion-aware `CoverageField`: each beam is certified (not blind)
    only when it is a NON-self return (a self-occluded UNKNOWN beam certifies nothing
    beyond the rig, even if another height retained an external return). The gate then
    requires only the bearings the forward braking sweep actually enters beyond the
    footprint, so a rig that always self-occludes its own base/mast/arm/cart no longer
    zeroes the whole certification. `coverage_radius` stays the legacy full-circle
    inscribed reach for diagnostics/fixtures only; the gate consumes the field.
    """
    scan.validate()
    _,_,sa=scan.sensor_pose(localized.pose)
    base=sa+scan.angle_min
    field=CoverageField(scan.range_max,tuple(
        (base+i*scan.angle_increment,
         0.0 if (scan.unknown and scan.unknown[i]) else scan.range_max)
        for i in range(len(scan.ranges))))
    # Range is measured about the sensor, not the chassis origin. The inscribed
    # base-centred observation disk shrinks by the calibrated translation.
    coverage=max(0.0,scan.range_max-math.hypot(*scan.sensor_to_base[:2])) if not any(scan.unknown) and len(scan.ranges)*scan.angle_increment>=2*math.pi-.01 else 0.0
    return Observation(scan.stamp,scan.points(localized.pose),coupled,arm_transport,
                       articulation,scan.source,coverage,coverage=field)
