"""Depth-ray voxel fusion for simulated RGB-D cameras.

Resolution is a model assumption, not a guarantee for sub-voxel obstacles.
Unknown pixels do not clear cells; external occupied evidence dominates free.
No scene geometry or ground-truth visibility is consumed by this module.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class DepthMount:
    name: str
    body: str  # base, cart, or a calibrated fixed map mount
    xyz: tuple  # body-frame metres
    yaw: float = 0.
    pitch: float = -0.35
    width: int = 80
    height: int = 60
    horizontal_fov: float = math.pi/2
    near: float = .05
    far: float = 6.
    noise_bound: float = .02

    def __post_init__(self):
        if (self.body not in ('base','cart','map') or len(self.xyz)!=3 or not self.name
                or not all(math.isfinite(v) for v in (*self.xyz,self.yaw,self.pitch,
                           self.horizontal_fov,self.near,self.far,self.noise_bound))
                or self.width<2 or self.height<2 or not 0<self.horizontal_fov<math.pi
                or not 0<self.near<self.far or self.noise_bound<0):
            raise ValueError('invalid calibrated depth mount')

    @property
    def focal_pixels(self):
        return self.width/(2*math.tan(self.horizontal_fov/2))

    def world_ray(self,u,v,body_pose):
        # Camera convention: right/down/forward, depth along forward image axis.
        x,y,z,heading=body_pose
        c,s=math.cos(heading),math.sin(heading)
        origin=(x+c*self.xyz[0]-s*self.xyz[1],y+s*self.xyz[0]+c*self.xyz[1],z+self.xyz[2])
        yaw=heading+self.yaw;cp,sp=math.cos(self.pitch),math.sin(self.pitch)
        forward=(math.cos(yaw)*cp,math.sin(yaw)*cp,sp)
        right=(math.sin(yaw),-math.cos(yaw),0.)
        down=(math.cos(yaw)*sp,math.sin(yaw)*sp,-cp)
        a=(u-(self.width-1)/2)/self.focal_pixels
        b=(v-(self.height-1)/2)/self.focal_pixels
        return origin,tuple(forward[i]+a*right[i]+b*down[i] for i in range(3))

    def native_spec(self,brackets):
        """CameraRig-compatible physical housing and supplied mounting brackets.

        Brackets use parent_from_solid/size_m in body coordinates. Their physical
        fit and collision clearance must be checked with the assembled rig.
        """
        cp,sp=math.cos(self.pitch),math.sin(self.pitch)
        cy,sy=math.cos(self.yaw),math.sin(self.yaw)
        tf=np.eye(4)
        tf[:3,:3]=np.array([[sy,cy*sp,cy*cp],[-cy,sy*sp,sy*cp],[0,-cp,sp]])
        tf[:3,3]=self.xyz
        housing=tf.copy();housing[:3,3]-=.016*tf[:3,2]
        if not brackets:raise ValueError('physical bracket geometry required')
        return dict(id=self.name,parent={'base':'robot_base','cart':'cart_chassis','map':'map'}[self.body],
            parent_from_optical=tf.tolist(),intrinsics=dict(width=self.width,height=self.height,
                fx=self.focal_pixels,fy=self.focal_pixels,cx=(self.width-1)/2,
                cy=(self.height-1)/2,near=self.near,far=self.far),
            collision_solids=[dict(parent_from_solid=housing.tolist(),size_m=[.06,.025,.03]),*brackets])


@dataclass(frozen=True)
class DepthFrame:
    mount: DepthMount
    stamp: float
    pose_stamp: float
    body_pose: tuple  # measured/localized x,y,z,yaw of the specified body
    depth: tuple  # row-major distance_to_image_plane; invalid pixels stay unknown
    self_mask: tuple = ()  # known instance identity; absent identity stays external
    intrinsics: tuple = ()  # actual registered OpenCV K, flattened row-major
    map_from_optical: tuple = ()  # acquisition-time optical TF, flattened 4x4
    rendering_frame: int = -1

    def __post_init__(self):
        count=self.mount.width*self.mount.height
        if (len(self.depth)!=count or len(self.body_pose)!=4
                or not all(math.isfinite(v) for v in (*self.body_pose,self.stamp,self.pose_stamp))
                or abs(self.stamp-self.pose_stamp)>.05
                or (self.self_mask and (len(self.self_mask)!=count or
                    any(type(v) is not bool for v in self.self_mask)))):
            raise ValueError('invalid depth frame or acquisition transform')
        k=np.asarray(self.intrinsics,dtype=float)
        tf=np.asarray(self.map_from_optical,dtype=float)
        if k.size!=9 or tf.size!=16 or not np.isfinite(k).all() or not np.isfinite(tf).all():
            raise ValueError('actual camera K and optical transform required')
        k=k.reshape(3,3);tf=tf.reshape(4,4)
        if (k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],(0,0,1))
                or k[0,1]!=0 or k[1,0]!=0 or not np.allclose(tf[3],(0,0,0,1))
                or not np.allclose(tf[:3,:3].T@tf[:3,:3],np.eye(3),atol=1e-6)
                or not np.isclose(np.linalg.det(tf[:3,:3]),1)
                or type(self.rendering_frame) is not int or self.rendering_frame<0):
            raise ValueError('invalid pinhole calibration or render identity')

    def world_ray(self,u,v):
        k=np.asarray(self.intrinsics).reshape(3,3)
        tf=np.asarray(self.map_from_optical).reshape(4,4)
        direction=tf[:3,:3]@np.array(((u-k[0,2])/k[0,0],(v-k[1,2])/k[1,1],1.))
        return tuple(tf[:3,3]),tuple(direction)


class RenderedDepthReader:
    """Adapter for native_cameras.RenderedCamera; never relabel held pixels.

    The caller supplies the optical TF measured at rendering_time and the
    independently measured body pose. Camera acquisition is owned by integration.
    """
    def __init__(self,mount,*,self_paths=()):
        self.mount=mount
        self.last_identity=None
        self.self_paths=tuple(self_paths)
        if any(not isinstance(p,str) or not p.startswith('/') or p=='/' or p.endswith('/')
               or '//' in p or any(part in ('.','..') for part in p.split('/'))
               for p in self.self_paths):
            raise ValueError('exact absolute rig subtree roots required')

    def read(self,camera,*,map_from_optical,body_pose,pose_stamp):
        raw=camera.get_current_frame()
        identity=raw['rendering_frame']
        stamp=float(raw['rendering_time'])
        if (isinstance(identity,(bool,np.bool_)) or not isinstance(identity,(int,np.integer))
                or identity<0 or not math.isfinite(stamp)):
            raise ValueError('invalid native render identity/time')
        if (self.last_identity is not None and
                (identity<=self.last_identity[0] or stamp<=self.last_identity[1])):
            raise ValueError('no new rendered depth frame')
        depth=np.asarray(raw['distance_to_image_plane'])
        if depth.shape!=(self.mount.height,self.mount.width) or depth.dtype!=np.float32:
            raise ValueError('registered float32 image-plane depth required')
        mask=()
        if self.self_paths:
            # Read the segmentation carried by this exact acquisition, never a
            # separate annotator tick or a previously cached ID-to-path table.
            segmentation=raw.get('instance_id_segmentation',{})
            ids=np.asarray(segmentation.get('data'))
            labels=segmentation.get('info',{}).get('idToLabels',{})
            if ids.shape!=depth.shape or ids.dtype!=np.uint32 or not isinstance(labels,dict) or not labels:
                raise ValueError('same-acquisition rig instance identity required')
            own=[]
            for instance_id,path in labels.items():
                if not isinstance(path,str):continue
                if any(path==root or path.startswith(root+'/') for root in self.self_paths):
                    own.append(int(instance_id))
            mask=tuple(map(bool,np.isin(ids,own).ravel()))
        result=DepthFrame(self.mount,stamp,pose_stamp,tuple(body_pose),tuple(depth.ravel()),
            self_mask=mask,
            intrinsics=tuple(np.asarray(camera.get_intrinsics_matrix()).ravel()),
            map_from_optical=tuple(np.asarray(map_from_optical).ravel()),rendering_frame=int(identity))
        self.last_identity=(identity,stamp)
        return result


class ObservedSpace:
    """Short-lived 3D occupancy at declared resolution, using measured rays.

    Ray traversal clears discrete cells before the noise-eroded endpoint. It
    never clears behind either self or external surfaces, or from invalid pixels.
    Every required device must contribute a fresh frame. No persistent free map
    survives a missing/stale frame, and occupied endpoints win across sensors.
    """
    def __init__(self,frames,*,required_sensors,resolution=.10,stale_after=.3,rig=None):
        self.frames=tuple(frames)
        self.required_sensors=frozenset(required_sensors)
        self.resolution=resolution;self.stale_after=stale_after
        self.rig=rig
        if (not self.required_sensors or not math.isfinite(resolution) or resolution<=0
                or not math.isfinite(stale_after) or stale_after<=0
                or any(not isinstance(f,DepthFrame) for f in self.frames)):
            raise ValueError('invalid observed-space configuration')
        self.free=set();self.occupied=set();self.self_hits=set()
        self.last_missing=None
        for frame in self.frames:self._integrate(frame)

    def key(self,point):
        return tuple(math.floor(v/self.resolution) for v in point)

    def centre(self,key):
        return tuple((i+.5)*self.resolution for i in key)

    def _integrate(self,frame):
        mount=frame.mount
        for i,depth in enumerate(frame.depth):
            if not math.isfinite(depth) or not mount.near<=depth<=mount.far:
                continue
            origin,ray=frame.world_ray(i%mount.width,i//mount.width)
            norm=math.sqrt(sum(v*v for v in ray))
            # Conservative endpoint band in metric range, including voxel radius.
            # noise_bound is image-plane depth error, not slant range error.
            padding=self.resolution*math.sqrt(3)/2
            end=max(mount.near,depth-mount.noise_bound-padding/norm)
            n=max(1,math.ceil((end-mount.near)*norm/(self.resolution/3)))
            for j in range(n):
                distance=mount.near+(end-mount.near)*j/n
                self.free.add(self.key(tuple(origin[k]+ray[k]*distance for k in range(3))))
            endpoint=tuple(origin[k]+ray[k]*depth for k in range(3))
            target=self.self_hits if frame.self_mask and frame.self_mask[i] else self.occupied
            target.add(self.key(endpoint))
            if target is self.occupied:
                for sign in (-1,1):
                    target.add(self.key(tuple(origin[k]+ray[k]*(depth+sign*mount.noise_bound)
                                              for k in range(3))))
        self.free.difference_update(self.occupied|self.self_hits)

    def freshness(self,now):
        if not math.isfinite(now):return 'invalid_observation_time'
        names=[f.mount.name for f in self.frames]
        if len(names)!=len(set(names)):return 'duplicate_coverage_sensor'
        if not self.required_sensors<=set(names):return 'missing_coverage_sensor'
        if any(f.stamp>now+1e-6 or now-f.stamp>self.stale_after for f in self.frames):
            return 'stale_coverage_sensor'
        if self.frames and max(f.stamp for f in self.frames)-min(f.stamp for f in self.frames)>.05:
            return 'unsynchronized_coverage_sensors'
        return ''

    def check_train(self,states,spec,uncertainty,now):
        if self.rig is None:return 'missing_observed_rig_model'
        if not isinstance(self.rig,RigVoxels):return 'invalid_observed_rig_model'
        if not states:return 'missing_stopping_sweep'
        if not math.isfinite(uncertainty) or uncertainty<.10:
            return 'invalid_depth_localization_bound'
        from twin.mobile import wrap
        robot,cart=states[0].bodies(spec)
        for frame in self.frames:
            if frame.mount.body=='map':continue
            body=robot if frame.mount.body=='base' else cart
            if (math.hypot(body.x-frame.body_pose[0],body.y-frame.body_pose[1])>uncertainty+.15
                    or abs(wrap(body.yaw-frame.body_pose[3]))>.15):
                return 'coverage_body_pose_mismatch'
        own=self.rig.cells(states[:1],spec,self,0.,0.)
        # Bound displacement between samples for every point of either body.
        # Half the endpoint centre displacement plus angular arc length covers
        # the continuous interpolated sweep, not just disconnected poses.
        sweep=0.
        radii=[max(math.hypot(p[0],p[1]) for p in points)
               for points in (self.rig.base_points,self.rig.cart_points)]
        for a,b in zip(states,states[1:]):
            for first,last,radius in zip(a.bodies(spec),b.bodies(spec),radii):
                sweep=max(sweep,(math.hypot(first.x-last.x,first.y-last.y)+
                                radius*abs(wrap(first.yaw-last.yaw)))/2)
        required=self.rig.cells(states,spec,self,.05+uncertainty+sweep,.05)
        return self.check_cells(required,now,current_solid_cells=own)

    def check_cells(self,cells,now,*,current_solid_cells=()):
        reason=self.freshness(now)
        if reason:return reason
        own=set(current_solid_cells)
        self.last_missing=None
        cells=set(cells)
        occupied=cells & self.occupied
        if occupied:
            self.last_missing=min(occupied);return 'occupied_swept_cell'
        unseen=cells-self.free-own
        if unseen:
            self.last_missing=min(unseen);return 'unseen_swept_cell'
        return ''


@dataclass(frozen=True)
class RigVoxels:
    """Offline collision-solid voxel centres in base/cart body frames.

    These describe the rig, never environmental free space. Generate from the
    actual collision assets at the measured transport joint configuration.
    A radial transport envelope or whole cart cage box is NOT a solid model.
    """
    base_points: tuple
    cart_points: tuple
    resolution: float
    source_sha256: str

    def __post_init__(self):
        if (not self.base_points or not self.cart_points or not self.source_sha256
                or not math.isfinite(self.resolution) or self.resolution<=0
                or any(len(p)!=3 or not all(math.isfinite(v) for v in p)
                       for p in self.base_points+self.cart_points)):
            raise ValueError('invalid collision-solid voxel model')

    def cells(self,states,spec,space,xy_margin,z_margin):
        result=set();res=space.resolution
        # Whole local voxels are represented, not just their centre points.
        radius=self.resolution*math.sqrt(3)/2
        for state in states:
            robot,cart=state.bodies(spec)
            for body,zbase,points in ((robot,spec.base_z,self.base_points),
                                     (cart,.20,self.cart_points)):
                for x,y,z in points:
                    x,y=body.point(x,y);z+=zbase
                    if xy_margin==z_margin==0:
                        # Own-cell exemption is limited to solid model samples.
                        result.add(space.key((x,y,z)));continue
                    low=space.key((x-xy_margin-radius,y-xy_margin-radius,z-z_margin-radius))
                    high=space.key((x+xy_margin+radius,y+xy_margin+radius,z+z_margin+radius))
                    for i in range(low[0],high[0]+1):
                        for j in range(low[1],high[1]+1):
                            for k in range(max(0,low[2]),high[2]+1):
                                result.add((i,j,k))
        return result
