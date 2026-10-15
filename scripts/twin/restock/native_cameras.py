"""Isaac5.1 rendered cameras and native ROS RGB-D output; no host Python ROS.

Configuration describes physical mounts, never observed stock or support. Camera
calibration/label interpretation belongs to restock_camera and isaac_sensors.
"""
import re
import time
import uuid

import numpy as np

from twin.restock_camera.geometry import Intrinsics,rigid


def camera_spec(spec):
    name=spec.get('id','')
    if not re.fullmatch(r'[a-z][a-z0-9_]*',name):
        raise ValueError('stable lowercase camera ID required')
    if spec.get('parent') not in ('map','cart_chassis','robot_base'):
        raise ValueError('camera parent must identify its actual physical mount')
    intrinsics=Intrinsics(**spec['intrinsics'])
    transform=rigid(spec['parent_from_optical'])
    solids=spec.get('collision_solids',[])
    if not solids:
        raise ValueError('camera housing and bracket collision solids required')
    for solid in solids:
        rigid(solid['parent_from_solid'])
        size=np.asarray(solid['size_m'],float)
        if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):
            raise ValueError('positive physical camera collision box dimensions required')
    return name,intrinsics,transform,solids


class RenderedCamera:
    """Read calibrated OpenCV optics instead of Isaac's centred-aperture estimate.

    Isaac5.1 Camera.get_intrinsics_matrix derives a centred principal point from
    aperture, even when an OpenCV lens model is authored. Read the real authored
    lens properties used by rendering; never substitute observed object poses.
    """
    def __init__(self,camera):
        self.camera=camera
        self.native_denominator=None

    def get_current_frame(self,*args,**kwargs):
        frame=dict(self.camera.get_current_frame(*args,**kwargs))
        identity=frame.get('rendering_frame')
        if isinstance(identity,dict):
            numerator,denominator=identity.get('referenceTimeNumerator'),identity.get('referenceTimeDenominator')
            if type(numerator) is not int or numerator<0 or type(denominator) is not int or denominator<=0:
                raise ValueError('invalid native rendered-frame identity')
            if self.native_denominator is not None and denominator!=self.native_denominator:
                raise ValueError('native rendered-frame identity basis changed')
            self.native_denominator=denominator
            frame['native_rendering_frame']=dict(identity)
            frame['rendering_frame']=numerator
        acquisition=getattr(self.camera,'_restock_acquisition',None)
        if acquisition is not None:
            if acquisition['native_id']!=identity:
                raise ValueError('camera acquisition clock belongs to another frame')
            frame.update({k:v for k,v in acquisition.items() if k!='native_id'})
        return frame

    def get_intrinsics_matrix(self):
        cx,cy,fx,fy,distortion=self.camera.get_opencv_pinhole_properties()
        if not np.isfinite([cx,cy,fx,fy,*distortion]).all() or min(fx,fy)<=0 or any(distortion):
            raise ValueError('registered zero-distortion pinhole camera required')
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]],float)

    def __getattr__(self,name):
        return getattr(self.camera,name)


def initialize_camera(path,k,*,name,frequency=-1,session_id=None,steady_domain=None,resolution=None):
    """Initialize one authored optical view; clock metadata is captured in callback.

    The Isaac5.1 acquisition callback supplies one render-product RGB/depth frame.
    Recording the monotonic clock there avoids restamping cached frames on read.
    Callers reuse steady_domain across commissioning and mission scene resets.
    """
    from isaacsim.sensors.camera import Camera
    session_id=session_id or str(uuid.uuid4())
    steady_domain=steady_domain or str(uuid.uuid4())
    class AcquisitionCamera(Camera):
        def _data_acquisition_callback(self,event):
            began=time.monotonic()
            super()._data_acquisition_callback(event)
            native=self._current_frame.get('rendering_frame')
            previous=getattr(self,'_restock_acquisition',None)
            if isinstance(native,dict) and (previous is None or native!=previous['native_id']):
                self._restock_acquisition=dict(native_id=dict(native),acquired_steady=began,
                    session_id=session_id,steady_domain=steady_domain)
    camera=AcquisitionCamera(prim_path=path,name=name,frequency=frequency,
                             resolution=resolution or (k.width,k.height))
    camera.set_clipping_range(k.near,k.far)
    camera.set_focal_length(.02)
    camera.set_horizontal_aperture(k.width*.02/k.fx)
    camera.set_vertical_aperture(k.height*.02/k.fy)
    camera.set_opencv_pinhole_properties(cx=k.cx,cy=k.cy,fx=k.fx,fy=k.fy,pinhole=[0.]*12)
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    # Exact rendered prim IDs are used only to mark known rig pixels unknown.
    # They must never supply product identity or clear occluded free space.
    camera.add_instance_id_segmentation_to_frame(init_params={'colorize':False})
    return RenderedCamera(camera)


def collision_camera_frame(design):
    """The ROS ``frame_id`` the masked-depth bridge publishes for the collision camera.

    Resolves the single ``role == 'collision'`` camera from the design
    (``twin.restock_camera.wrist_camera.collision_camera``). For the eye-in-hand wrist camera this
    is ``restock_camera_wrist``.
    """
    from twin.restock_camera.wrist_camera import collision_camera
    return collision_camera(design)['frame']


def initialize_collision_camera(stage, design, frame_prim_path, *, session_id=None,
                                steady_domain=None, resolution=None):
    """Wrap the eye-in-hand wrist camera as a :class:`RenderedCamera` for the masked-depth bridge.

    Unlike the store/cart mapper cameras (authored by :class:`CameraRig`), the wrist collision
    camera rides the UR tool flange: its prim is authored on the robot asset
    (``twin.mobile_assets.author_wrist_camera``) under the ``restock_camera_wrist`` frame, and its
    design entry has ``role == 'collision'`` (a wide-field, non-decode-grade depth camera). This
    selects it by that role and initializes it at the authored prim with the design intrinsics, so
    the bridge can publish its masked depth on frame ``restock_camera_wrist``. Returns
    ``(RenderedCamera, frame_id)``. Refuses before any Isaac import if the prim is not authored.
    """
    from twin.restock_camera.wrist_camera import collision_camera
    spec = collision_camera(design)
    k = Intrinsics(**spec['intrinsics'])
    camera_path = frame_prim_path.rstrip('/') + '/wrist_camera'
    if not stage.GetPrimAtPath(camera_path):
        raise ValueError('wrist collision camera prim not authored on the robot: ' + camera_path)
    camera = initialize_camera(camera_path, k, name='restock_' + spec['id'], session_id=session_id,
                               steady_domain=steady_domain, resolution=resolution)
    return camera, spec['frame']


class CameraRig:
    def __init__(self, *, stage, specs, parent_paths, frequency=-1, render_scale=1.0,
                 camera_render_scales=None):
        from pxr import Gf,UsdGeom,UsdPhysics
        if not 0 < render_scale <= 1.0:
            raise ValueError('render_scale must be in (0, 1]')
        # Per-camera override of the render-product scale: a decode-grade camera (small markers,
        # e.g. the collect source) renders at 1.0 for its own full-res product while the mapper
        # and other cameras keep the reduced VRAM scale. Each camera's context/baseline intrinsics
        # must be scaled by the SAME factor (see sensing_sources / rendered_context).
        camera_render_scales = camera_render_scales or {}
        if any(not 0 < s <= 1.0 for s in camera_render_scales.values()):
            raise ValueError('every per-camera render_scale must be in (0, 1]')
        self.stage,self.cameras,self.optical_paths=stage,{},{}
        self.render_scale=render_scale
        self.camera_render_scales=dict(camera_render_scales)
        self.specs={}
        for spec in specs:
            name,k,matrix,solids=camera_spec(spec)
            if name in self.cameras:
                raise ValueError('duplicate physical camera ID')
            parent=parent_paths[spec['parent']]
            if not stage.GetPrimAtPath(parent):
                raise ValueError('camera physical parent absent: '+parent)
            root=parent+'/restock_camera_'+name
            if stage.GetPrimAtPath(root):
                raise ValueError('refusing to overwrite camera mount')
            UsdGeom.Xform.Define(stage,root).AddTransformOp().Set(Gf.Matrix4d(*matrix.T.ravel().tolist()))
            for i,solid in enumerate(solids):
                # Solids are expressed in the physical parent, independently of lens orientation.
                solid_path=parent+'/restock_mount_'+name+'_'+str(i)
                if stage.GetPrimAtPath(solid_path):
                    raise ValueError('refusing to overwrite mount collision')
                box=UsdGeom.Cube.Define(stage,solid_path)
                box.CreateSizeAttr(1.)
                m=rigid(solid['parent_from_solid'])
                m=m@np.diag([*solid['size_m'],1.])
                box.AddTransformOp().Set(Gf.Matrix4d(*m.T.ravel().tolist()))
                UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            camera_path=root+'/Camera'
            prim=UsdGeom.Camera.Define(stage,camera_path)
            # USD camera looks -Z, +Y up. ROS optical parent is +Z, +Y down.
            prim.AddTransformOp().Set(Gf.Matrix4d(1,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,1))
            # Acquire each actual render. Isaac5.1's elapsed-time >= 1/frequency
            # decimator can skip a step due to floating-point accumulation.
            # The mission recorder selects every second 60Hz physics render;
            # native ROS helpers independently use frameSkipCount=1.
            # Scale the render-product resolution (not the intrinsics) to bound VRAM when
            # the whole-scene multi-camera render would otherwise exhaust the card. Keep
            # even dimensions: the recorder refuses odd HxW. Decode-grade stages must use
            # render_scale=1.0.
            scale=self.camera_render_scales.get(name,self.render_scale)
            if scale == 1.0:
                resolution=None
            else:
                resolution=(max(2,int(k.width*scale)//2*2),
                            max(2,int(k.height*scale)//2*2))
            self.cameras[name]=initialize_camera(camera_path,k,name='restock_'+name,
                                                 frequency=frequency,resolution=resolution)
            self.optical_paths[name]=root
            self.specs[name]=spec

    def base_from_camera(self,name,base_path):
        from pxr import UsdGeom
        cache=UsdGeom.XformCache()
        camera=cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.optical_paths[name]))
        base=cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(base_path))
        return rigid(np.asarray(camera*base.GetInverse()).T)


class CameraRosStream:
    """Exactly one active registered mapper camera; switch only at a new parked epoch."""
    def __init__(self, rig, *, root='/RestockCameraROS'):
        from twin.restock_manipulation.ros_adapter import require_sim_domain
        require_sim_domain()
        import omni.graph.core as og
        if rig.stage.GetPrimAtPath(root):
            raise ValueError('refusing to overwrite camera ROS graph')
        self.og,self.root,self.rig=og,root,rig
        self.selected=None
        keys=og.Controller.Keys
        nodes=[('tick','omni.graph.action.OnImpulseEvent'),('context','isaacsim.ros2.bridge.ROS2Context')]
        values=[('context.inputs:useDomainIDEnvVar',False),('context.inputs:domain_id',88)]
        links=[]
        for name,camera in rig.cameras.items():
            for kind in ('rgb','depth','info'):
                node=name+'_'+kind
                nodes.append((node,'isaacsim.ros2.bridge.ROS2CameraInfoHelper' if kind=='info'
                              else 'isaacsim.ros2.bridge.ROS2CameraHelper'))
                values.extend([(node+'.inputs:enabled',False),
                    (node+'.inputs:renderProductPath',camera.get_render_product_path()),
                    (node+'.inputs:frameId','restock_camera_'+name),
                    (node+'.inputs:topicName','/restock_sim/camera/'+('camera_info' if kind=='info' else kind)),
                    (node+'.inputs:useSystemTime',False),
                    (node+'.inputs:resetSimulationTimeOnStop',False),
                    (node+'.inputs:frameSkipCount',1)])
                if kind!='info':
                    values.append((node+'.inputs:type',kind))
                links.extend([('tick.outputs:execOut',node+'.inputs:execIn'),
                              ('context.outputs:context',node+'.inputs:context')])
        og.Controller.edit({'graph_path':root,'evaluator_name':'execution'},
            {keys.CREATE_NODES:nodes,keys.SET_VALUES:values,keys.CONNECT:links})

    def select(self,name):
        if name is not None and name not in self.rig.cameras:
            raise ValueError('unknown calibrated mapper camera')
        self.selected=name
        for camera in self.rig.cameras:
            for kind in ('rgb','depth','info'):
                self.og.Controller.attribute(self.root+'/'+camera+'_'+kind+'.inputs:enabled').set(camera==name)
        self.og.Controller.attribute(self.root+'/tick.state:enableImpulse').set(True)
        self.og.Controller.evaluate_sync(self.root)

    def close(self):
        self.select(None)
