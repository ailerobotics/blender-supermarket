"""Nvblox temporal-map contract for a CONTROLLED STATIC-SCENE simulation only.

Stored observed free space survives self-occlusion within one <=600 s mission.
This is not a dynamic-intrusion, per-cell-recency or real-person safety claim.
The mapper owns fusion. This module never raytraces or invents free space.
"""
from dataclasses import dataclass,field
from collections import deque
import math
import time
import numpy as np

from twin.mobile import swept_envelope,wrap


@dataclass(frozen=True)
class MapIdentity:
    session: str
    scene: str
    cameras: str
    localization: str

    def __post_init__(self):
        if any(not isinstance(v,str) or not v for v in vars(self).values()):
            raise ValueError('explicit session and revision identity required')


@dataclass(frozen=True)
class Acquisition:
    sensor: str
    identity: MapIdentity
    sequence: int
    stamp: float
    steady_stamp: float

    def __post_init__(self):
        if (not isinstance(self.sensor,str) or not self.sensor or not isinstance(self.identity,MapIdentity)
                or type(self.sequence) is not int or self.sequence<0
                or not all(math.isfinite(v) for v in (self.stamp,self.steady_stamp))):
            raise ValueError('invalid sensor acquisition provenance')


@dataclass(frozen=True)
class DistanceSlice:
    """Decoded native DistanceMapSlice plus mapper incorporation provenance.

    Native4.4 lookup rounds (p-origin)/resolution, whereas the message documents
    a corner origin. A full-cell radius conservatively covers both conventions.
    The native publication timestamp is NOT an incorporation acknowledgement.
    """
    frame: str
    origin: tuple
    resolution: float
    width: int
    height: int
    unknown: float
    distances: tuple
    generation: str
    incorporated: Acquisition
    _array: object = field(init=False,repr=False,compare=False)

    def __post_init__(self):
        if (not isinstance(self.frame,str) or not self.frame or
                not isinstance(self.generation,str) or not self.generation or len(self.origin)!=2
                or not isinstance(self.incorporated,Acquisition)
                or type(self.width) is not int or type(self.height) is not int
                or min(self.width,self.height)<=0 or self.width*self.height>4_000_000
                or len(self.distances)!=self.width*self.height
                or not all(math.isfinite(v) for v in (*self.origin,self.resolution,self.unknown))
                or self.resolution<=0):
            raise ValueError('invalid native distance slice')
        try:
            object.__setattr__(self,'origin',tuple(float(v) for v in self.origin))
            object.__setattr__(self,'distances',tuple(float(v) for v in self.distances))
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError('nonnumeric distance slice') from exc
        array=np.asarray(self.distances,dtype=float).reshape(self.height,self.width)
        array.setflags(write=False)
        object.__setattr__(self,'_array',array)

    @classmethod
    def from_message(cls,msg,*,generation,incorporated):
        return cls(msg.header.frame_id,(msg.origin.x,msg.origin.y),msg.resolution,
                   msg.width,msg.height,msg.unknown_value,tuple(msg.data),generation,incorporated)


class TemporalMap:
    """Mission epoch + required-source watchdog + conservative train sweep.

    The host binds incorporation to the exact published map generation. A fresh
    unrelated ESDF response cannot certify an old slice. A generation identifies
    a mapper started EMPTY, not a counter incremented around a retained map.
    """
    def __init__(self,identity,*,commissioned,required_sensors,generation,frame='map',
                 duration=600.,stale_after=.3,steady_clock=time.monotonic,
                 steady_commissioned=None,measured_latency=0.):
        if (not isinstance(identity,MapIdentity) or not generation or not frame
                or not required_sensors or not all(isinstance(s,str) and s for s in required_sensors)
                or not all(math.isfinite(v) for v in (commissioned,duration,stale_after,measured_latency))
                or not 0<duration<=600 or not 0<stale_after<=.3 or not 0<=measured_latency<=.3):
            raise ValueError('invalid controlled static-scene epoch')
        self.identity=identity;self.commissioned=commissioned;self.duration=duration
        self.required=frozenset(required_sensors);self.frame=frame;self.generation=generation
        self.stale_after=stale_after;self.clock=steady_clock
        self.steady_commissioned=steady_clock() if steady_commissioned is None else steady_commissioned
        if not math.isfinite(self.steady_commissioned):raise ValueError('invalid steady epoch')
        self.measured_latency=measured_latency
        self.sources={};self.slice=None;self.invalid='';self.last_missing=None
        self.history=deque(maxlen=512)
        self.reconstruction_since=commissioned;self.retired_generations=set()
        self.clearance_changed=False
        self.last_checked_time=None;self.last_checked_steady=None

    def invalidate(self,reason):
        self.invalid=reason or 'epoch_invalidated'
        self.slice=None

    def check_identity(self,identity):
        if identity!=self.identity:
            self.invalidate('map_identity_changed')

    def invalidate_clearance(self,*,now):
        """Stock/manipulation/payload event: stop until fresh empty-map rebuild.

        Conservatively invalidates the whole local clearance map. This avoids
        asserting regional re-observation from a single global timestamp.
        The absolute mission deadline is NEVER extended by a rebuild.
        """
        if not math.isfinite(now) or now<self.reconstruction_since:
            raise ValueError('invalid clearance change time')
        if self.invalid and self.invalid!='clearance_reobservation_required':
            return  # A payload event cannot reopen an invalid mission epoch.
        self.reconstruction_since=now
        self.clearance_changed=True
        self.invalid='clearance_reobservation_required';self.slice=None

    def begin_empty_reconstruction(self,generation):
        if self.invalid!='clearance_reobservation_required':
            raise ValueError('only clearance invalidation permits same-epoch rebuild')
        if not generation or generation==self.generation or generation in self.retired_generations:
            raise ValueError('new empty mapper generation required')
        self.retired_generations.add(self.generation)
        self.generation=generation;self.sources={};self.history.clear();self.slice=None;self.invalid=''

    def observe(self,acquisition):
        if not isinstance(acquisition,Acquisition):raise ValueError('acquisition required')
        self.check_identity(acquisition.identity)
        if self.invalid:return
        if acquisition.sensor not in self.required:raise ValueError('unconfigured sensor')
        old=self.sources.get(acquisition.sensor)
        if (acquisition.stamp<self.reconstruction_since or
                (self.clearance_changed and acquisition.stamp==self.reconstruction_since) or
                (old and (acquisition.sequence<=old.sequence or acquisition.stamp<=old.stamp
                          or acquisition.steady_stamp<=old.steady_stamp))):
            raise ValueError('held, out-of-order or pre-reconstruction sensor frame')
        self.sources[acquisition.sensor]=acquisition
        self.history.append(acquisition)

    def accept(self,snapshot):
        if not isinstance(snapshot,DistanceSlice):raise ValueError('native slice required')
        self.check_identity(snapshot.incorporated.identity)
        if self.invalid:return
        if snapshot.frame!=self.frame or snapshot.generation!=self.generation:
            raise ValueError('wrong stable frame or mapper generation')
        ack=snapshot.incorporated
        if ack not in self.history:
            raise ValueError('map acknowledgement must match registered acquisition')
        if self.slice and ack.stamp<self.slice.incorporated.stamp:
            raise ValueError('out-of-order mapper incorporation')
        self.slice=snapshot

    def readiness(self,now):
        steady=self.clock()
        if not math.isfinite(now) or not math.isfinite(steady):return 'invalid_map_time'
        if (self.last_checked_time is not None and
                (now<self.last_checked_time or steady<self.last_checked_steady)):
            if not self.invalid or self.invalid=='clearance_reobservation_required':
                self.invalidate('map_clock_rollback')
        self.last_checked_time=now;self.last_checked_steady=steady
        if self.invalid:return self.invalid
        if not 0<self.duration<=600 or not 0<self.stale_after<=.3:return 'invalid_map_time_limits'
        if not math.isfinite(self.measured_latency) or not 0<=self.measured_latency<=.3:
            return 'navigation_latency_exceeds_bound'
        if (now<self.commissioned or steady<self.steady_commissioned or
                now-self.commissioned>=self.duration or
                steady-self.steady_commissioned>=self.duration):
            self.invalidate('map_epoch_expired')
            return self.invalid
        if self.required!=self.sources.keys():return 'missing_navigation_sensor'
        if self.slice is None:return 'missing_navigation_map'
        for source in (*self.sources.values(),self.slice.incorporated):
            if (not 0<=now-source.stamp<=self.stale_after or
                    not 0<=steady-source.steady_stamp<=self.stale_after):
                return 'stale_navigation_sensor_or_mapper'
        return ''

    def reaction_delay(self,now):
        """Conservative latency allowance in addition to the existing watchdog.

        Use the larger measured pipeline delay / current acquisition age. Never
        shorten the controller's existing300ms watchdog stopping allowance.
        """
        steady=self.clock()
        if self.slice is None:return self.measured_latency
        ack=self.slice.incorporated
        return max(self.measured_latency,now-ack.stamp,steady-ack.steady_stamp,0.)

    def check_train(self,states,spec,uncertainty,now):
        reason=self.readiness(now)
        if reason:return reason
        if not states:return 'missing_stopping_sweep'
        if not math.isfinite(uncertainty) or uncertainty<.1:return 'invalid_localization_bound'
        samples=[s.bodies(spec) for s in states]
        pairs=list(zip(samples,samples[1:])) or [(samples[0],samples[0])]
        for pair in pairs:
            a,b=pair[0][0],pair[1][0]
            ds=math.hypot(a.x-b.x,a.y-b.y)
            curvature=abs(wrap(a.yaw-b.yaw))/ds if ds>1e-9 else 0.
            pad=ds*ds*(curvature+1/spec.cart_hitch_x)/8+.001
            polygon=swept_envelope(pair,spec,margin=.05+uncertainty+pad)
            reason=self.check_polygon(polygon)
            if reason:return reason
        return self.readiness(now)

    def check_polygon(self,polygon):
        snapshot=self.slice
        if snapshot is None:return 'missing_navigation_map'
        res=snapshot.resolution;ox,oy=snapshot.origin
        xs,ys=zip(*polygon)
        low=(math.floor((min(xs)-ox)/res-1),math.floor((min(ys)-oy)/res-1))
        high=(math.ceil((max(xs)-ox)/res+1),math.ceil((max(ys)-oy)/res+1))
        ii,jj=np.meshgrid(np.arange(low[0],high[0]+1),np.arange(low[1],high[1]+1))
        indices=np.column_stack((ii.ravel(),jj.ravel()))
        centers=np.array((ox,oy))+indices*res
        vertices=np.asarray(polygon,float)
        edges=np.roll(vertices,-1,axis=0)-vertices
        axes=np.vstack((np.eye(2),np.column_stack((edges[:,1],-edges[:,0]))))
        points=vertices@axes.T
        projections=centers@axes.T
        half=res*np.abs(axes).sum(axis=1)
        intersects=np.all((projections+half>=points.min(axis=0)-1e-12)&
                          (projections-half<=points.max(axis=0)+1e-12),axis=1)
        indices=indices[intersects]
        outside=(indices[:,0]<0)|(indices[:,0]>=snapshot.width)|(
                 indices[:,1]<0)|(indices[:,1]>=snapshot.height)
        if outside.any():
            self.last_missing=tuple(map(int,indices[outside][0]));return 'outside_navigation_map'
        distances=snapshot._array[indices[:,1],indices[:,0]]
        unknown=~np.isfinite(distances)|(distances==snapshot.unknown)
        if unknown.any():
            self.last_missing=tuple(map(int,indices[unknown][0]));return 'unknown_navigation_cell'
        # Cover the complete map cell, not only its ESDF sample centre.
        occupied=distances<=res/math.sqrt(2)
        if occupied.any():
            self.last_missing=tuple(map(int,indices[occupied][0]));return 'occupied_navigation_cell'
        self.last_missing=None
        return ''

    def planning_grid(self,base_grid):
        """Conservative row runs of unknown/occupied cells for existing planner.

        This consumes nvblox output; it does not reconstruct a second voxel map.
        Planning is bounded to the available map; the live actuator gate rechecks.
        """
        from .core import GridMap
        if self.slice is None:raise ValueError('navigation map missing')
        m=self.slice;r=m.resolution;ox,oy=m.origin
        bounds=(max(base_grid.bounds[0],ox),max(base_grid.bounds[1],oy),
                min(base_grid.bounds[2],ox+(m.width-1)*r),
                min(base_grid.bounds[3],oy+(m.height-1)*r))
        boxes=list(base_grid.obstacles)
        blocked=~np.isfinite(m._array)|(m._array==m.unknown)|(m._array<=r/math.sqrt(2))
        for j,row in enumerate(blocked):
            changes=np.flatnonzero(np.diff(np.concatenate(([False],row,[False]))))
            for start,end in changes.reshape(-1,2):
                boxes.append((ox+(int(start)-1)*r,oy+(j-1)*r,ox+int(end)*r,oy+(j+1)*r))
        return GridMap(bounds,boxes,resolution=base_grid.resolution,corridors=base_grid.corridors)
