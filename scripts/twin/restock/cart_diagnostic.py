"""Opt-in stationary contact experiment; never used by the mission factory."""
import json
import hashlib
from pathlib import Path
import numpy as np


GROUND='/Supermarket/GRP_Structure/Ground_exterior/Ground_exterior'
FLOOR='/Supermarket/GRP_Structure/Backroom_floor/Backroom_floor'


def select_floor(stage, variant):
    from pxr import UsdPhysics
    if variant not in ('baseline','without_lower_ground'):
        raise ValueError('unknown cart diagnostic variant')
    for path in (GROUND,FLOOR):
        prim=stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('required diagnostic floor collider absent: '+path)
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            raise ValueError('diagnostic input collider already disabled: '+path)
    if variant=='without_lower_ground':
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(GROUND)).CreateCollisionEnabledAttr(False)


class CartDiagnostic:
    def __init__(self, output, variant):
        self.output=Path(output)
        self.variant=variant
        self.phase='initialization'
        self.events=[]
        self.errors=[]
        self.subscription=None

    def prepare(self,stage):
        from pxr import UsdPhysics, PhysxSchema
        from omni.physx import get_physx_simulation_interface
        select_floor(stage,self.variant)
        stage.GetRootLayer().Export(str(self.output/'pristine-authored.usda'))
        offsets=[]
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith('/Rollwagen/'):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    api=PhysxSchema.PhysxCollisionAPI(prim)
                    offsets.append(dict(path=str(prim.GetPath()),
                        contact_offset_schema=str(api.GetContactOffsetAttr().Get()),
                        rest_offset_schema=str(api.GetRestOffsetAttr().Get())))
        self.subscription=get_physx_simulation_interface().subscribe_contact_report_events(self.contacts)
        (self.output/'contact-provenance.json').write_text(json.dumps(dict(
            variant=self.variant,offsets=offsets,
            limitation='Schema offsets only; simulation-selected effective offsets unavailable through this callback. Contact callback reports actual collider paths, impulses and separations; no friction-anchor instrumentation.',
            api='Installed Isaac5.1 omni.physx ContactReportDemo callback API'),indent=2))

    def contacts(self,headers,data):
        from pxr import PhysicsSchemaTools
        try:
            for header in headers:
                paths={key:str(PhysicsSchemaTools.intToSdfPath(getattr(header,key)))
                       for key in ('actor0','actor1','collider0','collider1')}
                if not any(p.startswith('/Rollwagen/') for p in paths.values()):
                    continue
                points=[]
                for point in data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]:
                    points.append(dict(position=list(point.position),normal=list(point.normal),
                                       impulse=list(point.impulse),separation=float(point.separation)))
                self.events.append(dict(phase=self.phase,type=str(header.type),**paths,points=points))
        except Exception as exc:
            self.errors.append(str(exc))

    def initialize(self,assembly):
        from pxr import UsdPhysics,UsdUtils,PhysicsSchemaTools,Sdf
        from omni.physx import get_physx_interface,get_physx_simulation_interface
        self.stage=assembly.stage
        paths=[str(p.GetPath()) for p in self.stage.Traverse()
               if str(p.GetPath()).startswith('/Rollwagen/') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(paths)!=9:
            raise ValueError('nine native cart bodies required')
        physx=get_physx_interface()
        simulation=get_physx_simulation_interface()
        stage_id=UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()
        def external_state():
            transforms={}
            for path in paths:
                value=physx.get_rigidbody_transformation(path)
                if not value['ret_val']:
                    raise ValueError('native body pose unavailable: '+path)
                transforms[path]=dict(position=list(value['position']),rotation=list(value['rotation']),
                    sleeping=simulation.is_sleeping(stage_id,PhysicsSchemaTools.sdfPathToInt(Sdf.Path(path))))
            layers={layer.identifier:hashlib.sha256(layer.ExportToString().encode()).hexdigest()
                    for layer in self.stage.GetLayerStack()}
            return dict(layers=layers,transforms=transforms,
                simulation_time=float(assembly.world.current_time),
                chassis_velocity=assembly.cart_body.get_linear_velocity().tolist())
        before=external_state()
        # Raw backend tensor views bind handles and allocate read buffers only.
        # Never construct/initialize a late SingleRigidPrim: it authors xforms,
        # sleep thresholds and other state even when used as an observer.
        probe=assembly.world.physics_sim_view.create_rigid_body_view(paths)
        before_tensors=read_tensors(probe,properties=True)
        after_probe=external_state()
        proof=dict(before=before,after_probe=after_probe,probe_unchanged=before==after_probe)
        proof['backend']=dict(physx_gpu_override=physx.get_overwrite_gpu_setting(),
            scenes=[{str(a.GetName()):str(a.Get()) for a in prim.GetAttributes()
                     if 'enableGPUDynamics' in str(a.GetName()) or 'solverType' in str(a.GetName())
                     or 'timeStepsPerSecond' in str(a.GetName()) or 'enableStabilization' in str(a.GetName())}
                    for prim in self.stage.Traverse() if prim.IsA(UsdPhysics.Scene)],
            note='CPU PhysX and GPU RTX rendering are distinct; no backend setting changed')
        self.phase='observer-attachment-no-step'
        if proof['probe_unchanged']:
            self.view=assembly.world.physics_sim_view.create_rigid_body_view(paths)
            after_tensors=read_tensors(self.view,properties=True)
            after=external_state()
            proof.update(before_tensors=before_tensors,after_tensors=after_tensors,after=after,
                         observer_unchanged=before==after and before_tensors==after_tensors)
        (self.output/'observer-attachment.json').write_text(json.dumps(proof,indent=2,allow_nan=False))
        if not proof['probe_unchanged'] or not proof.get('observer_unchanged'):
            raise ValueError('diagnostic observer attachment changed state; abort before stepping')

    def snapshot(self):
        from pxr import Gf,UsdPhysics
        poses={}
        transforms={}
        values=read_tensors(self.view)
        for path,value in values.items():
            pose=value['get_transforms']; velocity=value['get_velocities']
            p=pose[:3];q=[pose[6],*pose[3:6]]
            poses[path]=dict(position=p,quaternion_wxyz=q,
                linear_velocity=velocity[:3],angular_velocity=velocity[3:])
            transforms[path]=(Gf.Vec3d(*map(float,p)),Gf.Rotation(Gf.Quatd(float(q[0]),Gf.Vec3d(*map(float,q[1:])))))
        residuals={}
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).startswith('/Rollwagen/') and prim.IsA(UsdPhysics.RevoluteJoint):
                joint=UsdPhysics.RevoluteJoint(prim)
                points=[]
                for rel,attr in ((joint.GetBody0Rel(),joint.GetLocalPos0Attr()),(joint.GetBody1Rel(),joint.GetLocalPos1Attr())):
                    p,r=transforms[str(rel.GetTargets()[0])]
                    points.append(p+r.TransformDir(Gf.Vec3d(attr.Get())))
                residuals[str(prim.GetPath())]=(points[0]-points[1]).GetLength()
        return dict(bodies=poses,joint_anchor_residual_m=residuals,
                    phase=self.phase,
                    sampling='raw PhysX tensor transforms and COM velocities after completed world.step; no intervening step; quaternion xyzw converted to wxyz')

    def save(self):
        (self.output/'contact-events.json').write_text(json.dumps(dict(events=self.events,errors=self.errors),indent=2))
        self.subscription=None


def read_tensors(view, *, properties=False):
    """Read and immediately copy backend buffers; do not author or flush state."""
    methods=['get_transforms','get_velocities']
    if properties:
        methods+=['get_masses','get_coms','get_inertias','get_disable_gravities',
                  'get_disable_simulations','get_material_properties','get_contact_offsets','get_rest_offsets']
    result={path:{} for path in view.prim_paths}
    for method in methods:
        value=getattr(view,method)()
        if hasattr(value,'numpy'):
            value=value.numpy()
        rows=np.array(value,copy=True)
        if len(rows)!=len(result) or not np.isfinite(rows).all():
            raise ValueError('invalid passive tensor measurement')
        for path,row in zip(result,rows):
            result[path][method]=row.tolist()
    return result
