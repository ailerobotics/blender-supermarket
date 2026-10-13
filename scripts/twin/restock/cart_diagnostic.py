"""Opt-in stationary contact experiment; never used by the mission factory."""
import json
from pathlib import Path


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
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import UsdPhysics
        self.stage=assembly.stage
        self.bodies={}
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).startswith('/Rollwagen/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body=SingleRigidPrim(prim_path=str(prim.GetPath()))
                body.initialize()
                self.bodies[str(prim.GetPath())]=body

    def snapshot(self):
        from pxr import Gf,UsdPhysics
        poses={}
        transforms={}
        for path,body in self.bodies.items():
            p,q=body.get_world_pose()
            poses[path]=dict(position=p.tolist(),quaternion_wxyz=q.tolist(),
                linear_velocity=body.get_linear_velocity().tolist(),angular_velocity=body.get_angular_velocity().tolist())
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
                    sampling='sequential native body getters after same completed world.step; no intervening step')

    def save(self):
        (self.output/'contact-events.json').write_text(json.dumps(dict(events=self.events,errors=self.errors),indent=2))
        self.subscription=None
