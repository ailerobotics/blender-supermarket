"""Isaac5.1 rendered cameras and native ROS RGB-D output; no host Python ROS.

Configuration describes physical mounts, never observed stock or support. Camera
calibration/label interpretation belongs to restock_camera and isaac_sensors.
"""
import re
import time
import uuid

import numpy as np

from twin.restock_camera.geometry import Intrinsics,rigid


class CameraRenderProductReleased(RuntimeError):
    """A camera whose render product was DESTROYED (stage-scoped teardown / per-op close) was read
    again. Raised as a NAMED mission-fail reason in place of Isaac's raw "Accessed invalid expired
    'Camera' prim" RuntimeError, so a lifecycle bug reports which camera and why, not an opaque
    prim-expired trace. The fix is the lifecycle (keep the camera alive until its last reader); this
    is the backstop so a regression fails by name instead of a bare prim-expired trace."""


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
        # Set True by destroy_camera_render_product; a read after that raises CameraRenderProductReleased
        # (a named reason) instead of Isaac's raw expired-prim RuntimeError.
        self._render_product_released=False

    def _check_live(self):
        if self._render_product_released:
            path=getattr(self.camera,'prim_path',None) or '?'
            raise CameraRenderProductReleased(
                'sensing camera render product was released before this stage read it '
                '(stage-scoped teardown / per-op close): %s' % (path,))

    def get_current_frame(self,*args,**kwargs):
        self._check_live()
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
        self._check_live()
        cx,cy,fx,fy,distortion=self.camera.get_opencv_pinhole_properties()
        if not np.isfinite([cx,cy,fx,fy,*distortion]).all() or min(fx,fy)<=0 or any(distortion):
            raise ValueError('registered zero-distortion pinhole camera required')
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]],float)

    def set_render_products_enabled(self,enabled):
        """Optional hook IsaacCartonSensors.set_render_suspended looks up to DISABLE this camera's
        decode render while a controller execution phase has sensing suspended (the fixed station
        decode cameras are RenderedCamera, so they get it; the arm masked-depth camera is a raw
        Camera without it and keeps rendering for the ESDF). Delegates to the best-effort helper on
        the wrapped Isaac camera; a pure optimisation that never raises into the mission tick."""
        return set_camera_render_products_enabled(self.camera,enabled)

    def __getattr__(self,name):
        return getattr(self.camera,name)


def set_camera_render_products_enabled(camera, enabled):
    """Best-effort enable/disable an Isaac Camera's RTX render product(s).

    ``world.step(render=True)`` renders EVERY active render product. While a controller execution
    phase has sensing suspended, the sensing camera's decode frame is unused but its whole-store
    render is the dominant wall cost -- so integration disables ONLY this product's hydra updates
    (the separate arm-depth / masked-depth / chase-recorder cameras keep their own products and
    still render, so the host nvblox ESDF and the walkthrough are unaffected; run-139 died when ALL
    rendering was skipped), re-enabling it before the next observe/verify.

    ``camera._render_product.hydra_texture.set_updates_enabled(bool)`` is the canonical Isaac pause;
    this is a pure optimisation, so any failure is LOGGED and the product left enabled (the
    pre-existing always-render behaviour) -- never raised into the mission tick. Returns True iff the
    toggle was actually applied."""
    enabled = bool(enabled)
    try:
        products = getattr(camera, '_render_product', None)
        for rp in (products if isinstance(products, (list, tuple)) else [products]):
            tex = getattr(rp, 'hydra_texture', None) if rp is not None else None
            setter = getattr(tex, 'set_updates_enabled', None) if tex is not None else None
            if setter is None:
                raise AttributeError('render product has no hydra_texture.set_updates_enabled')
            setter(enabled)
        return True
    except Exception as exc:                                      # pure optimisation; never crash a tick
        print('RENDER_PRODUCT_TOGGLE skipped enabled=%s: %r' % (enabled, exc), flush=True)
        return False


def destroy_camera_render_product(rendered_camera, stage=None):
    """fm380: best-effort DESTROY an Isaac camera's RTX render product to RETURN its VRAM -- unlike
    :func:`set_camera_render_products_enabled`, which only PAUSES hydra updates (the buffers stay
    resident). Used by stage-scoped sensing to free the collect-only station cameras at the
    collect->couple boundary, and by the release-bench ``--destroy-sensing`` mode to MEASURE how much
    VRAM Isaac actually returns (it may pool and return ~0). Layered best-effort: stop updates, call
    the render product's own ``destroy()`` (releases the GPU render target), then RemovePrim the
    render-product + camera prims so a later ``world.step`` cannot re-create them. Pure optimisation:
    logs what applied and returns that dict; NEVER raises (a failed teardown must not crash a tick)."""
    applied = {}
    camera = getattr(rendered_camera, 'camera', rendered_camera)
    try:
        set_camera_render_products_enabled(camera, False)
        applied['paused'] = True
    except Exception as exc:                                  # noqa: BLE001
        applied['paused'] = repr(exc)
    # Destroy the render product object(s) -- this is what frees the GPU render-target buffers.
    try:
        products = getattr(camera, '_render_product', None)
        rp_paths = []
        for rp in (products if isinstance(products, (list, tuple)) else [products]):
            if rp is None:
                continue
            path = getattr(rp, 'path', None) or getattr(rp, 'prim_path', None)
            if path:
                rp_paths.append(str(path))
            fn = getattr(rp, 'destroy', None)
            if callable(fn):
                fn()
                applied.setdefault('rp_destroyed', 0)
                applied['rp_destroyed'] += 1
        if not rp_paths:
            getter = getattr(camera, 'get_render_product_path', None)
            if callable(getter):
                rp_paths.append(str(getter()))
        applied['rp_paths'] = rp_paths
    except Exception as exc:                                  # noqa: BLE001
        applied['rp_destroy'] = repr(exc)
    # Remove the render-product + camera prims so nothing re-allocates them on the next step.
    if stage is not None:
        try:
            removed = []
            for p in applied.get('rp_paths', []):
                if p and stage.GetPrimAtPath(p):
                    stage.RemovePrim(p)
                    removed.append(p)
            cam_path = getattr(camera, 'prim_path', None)
            if cam_path and stage.GetPrimAtPath(cam_path):
                stage.RemovePrim(cam_path)
                removed.append(str(cam_path))
            applied['prims_removed'] = removed
        except Exception as exc:                              # noqa: BLE001
            applied['prims_removed'] = repr(exc)
    # Latch the wrapper as released so a LATER read fails by name (CameraRenderProductReleased) rather
    # than Isaac's raw expired-prim RuntimeError (a lifecycle bug then names the camera + cause).
    try:
        rendered_camera._render_product_released = True
    except Exception:                                         # noqa: BLE001 -- a non-RenderedCamera input
        pass
    print('DESTROY_RENDER_PRODUCT %r' % applied, flush=True)
    return applied


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


def initialize_mast_camera(stage, design, base_path, *, session_id=None,
                           steady_domain=None, resolution=None):
    """Author + wrap the body-mounted mast depth camera (nvblox camera_1) for its masked-depth bridge.

    Mirrors :func:`initialize_collision_camera` for the wrist, but the mast camera has no separately
    authored prim: it is authored HERE from the camera design's ``mast`` entry (``role ==
    'nvblox_depth'``, ``parent == 'robot_base'``; :func:`twin.restock_camera.mast_depth.mast_camera_entry`).
    ``base_path`` is the robot base prim (``assembly.paths['ur_base_link']``). The design
    ``parent_from_optical`` is the ROS optical frame in base; a USD ``Camera`` child is authored with
    the optical->USD flip (``+Y`` up, looks ``-Z``) exactly as :class:`CameraRig` does for the station
    cameras, then initialized with distance-to-image-plane + instance-id segmentation so the bridge can
    rig+carton-mask it. Returns ``(RenderedCamera, camera_path, frame_id)``, or ``None`` when the design
    carries no mast camera (mast depth not enabled) -- so a disabled mast authors nothing.
    """
    from twin.restock_camera.mast_depth import MAST_FRAME_ID
    spec = design.get('cameras', {}).get('mast')
    if spec is None:
        return None
    from pxr import Gf, UsdGeom
    k = Intrinsics(**spec['intrinsics'])
    matrix = rigid(spec['parent_from_optical'])
    root = base_path.rstrip('/') + '/restock_camera_mast'
    if stage.GetPrimAtPath(root):
        raise ValueError('refusing to overwrite mast camera mount: ' + root)
    UsdGeom.Xform.Define(stage, root).AddTransformOp().Set(Gf.Matrix4d(*matrix.T.ravel().tolist()))
    camera_path = root + '/Camera'
    # USD camera looks -Z, +Y up; the ROS optical parent is +Z, +Y down (CameraRig convention).
    UsdGeom.Camera.Define(stage, camera_path).AddTransformOp().Set(
        Gf.Matrix4d(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1))
    camera = initialize_camera(camera_path, k, name='restock_mast', session_id=session_id,
                               steady_domain=steady_domain, resolution=resolution)
    return camera, camera_path, MAST_FRAME_ID


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
