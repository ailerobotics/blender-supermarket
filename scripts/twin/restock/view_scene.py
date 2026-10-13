"""Static native-camera bench using the full store and actual CAD geometry.

All initial placements are explicit. Rigid dynamics are disabled for this optical
bench, never for a mission. No fixture geometry is hidden for visibility.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _apply_light_rig(stage, design, view, config):
    """Apply the camera worker's store light rig per the lighting handoff.

    The 72 Blender-exported RectLights are authored at power/pi (~14.3), ~200x too
    dim in RTX nits, which caused the dark captures. Per the handoff we author the
    designed rig (correct per-fixture nits, physical 4.44x panel/tube ratio, dome
    fill) and DISABLE the dim exported store lights to avoid doubling. For the
    recessed shelf support a task fill is co-located with the shelf camera. The
    single calibration knob is intensity_scale (config 'light_intensity_scale').
    RTX auto-exposure (histogram) is already off, so exposure is fixed; calibrate
    brightness through intensity_scale rather than editing the physical ratio.
    """
    import numpy as np
    from pxr import UsdLux
    from twin.restock_camera.lights import light_rig, author_light_rig, task_fill_fixture, LightRigTuning
    tuning=LightRigTuning(intensity_scale=float(config.get('light_intensity_scale',1.0)))
    fills=[]
    if view=='shelf':
        # Co-locate a task fill with the shelf camera (parent 'map' == /World identity)
        # so it reaches into the recessed shelf slot the ceiling panels cannot light.
        optical=np.asarray(design['cameras']['shelf']['parent_from_optical'],float)
        eye=optical[:3,3]
        target=eye+optical[:3,2]*0.35  # optical +Z looks at the support
        fills=[task_fill_fixture('fill:shelf',eye.tolist(),target.tolist())]
    # Disable the dim exported store lights before adding the rig (avoid doubling).
    disabled=0
    for prim in stage.Traverse():
        if (prim.IsA(UsdLux.RectLight) or prim.IsA(UsdLux.DomeLight)) and \
                not str(prim.GetPath()).startswith('/World/RestockLight_'):
            prim.SetActive(False)
            disabled+=1
    rig=light_rig(tuning,fills=fills)
    authored=author_light_rig(stage,rig,parent_path='/World',tuning=tuning)
    return dict(intensity_scale=tuning.intensity_scale,disabled_exported_lights=disabled,
                authored_fixtures=len(authored),dome_fill=bool(rig.get('dome',{}).get('enabled')),
                shelf_task_fill=bool(fills),tonemapping=rig['tonemapping'],
                counts=rig['counts'],
                note='RTX auto-exposure already off; calibrate via intensity_scale only')


def build(stage, config, output, *, view='source'):
    from pxr import Gf,UsdGeom,UsdPhysics
    import trimesh
    from twin.urdf_probe import parse_urdf,link_world_transforms,resolve_package_uri,rpy_to_matrix
    from twin.restock_camera.config import author_camera_design
    from twin.restock_camera.geometry import Intrinsics,pose_matrix
    from twin.restock_manipulation.isaac_sensors import write_carton_label
    from .scene import author_initial_scenario,label_stock
    from .native_cameras import initialize_camera
    output=Path(output)
    scenario=json.loads(Path(config['scenario']).read_text())
    mobile=json.loads(Path(config['mobile_manifest']).read_text())
    design=json.loads(Path(config['camera_config']).read_text())
    if view not in design['cameras']:
        raise ValueError('unknown physical candidate camera')
    stage.GetRootLayer().subLayerPaths.append(str(Path(config['store_usd']).resolve()))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.Xform.Define(stage,'/World')
    initial=author_initial_scenario(stage,scenario)
    label_stock(stage,initial['stock'],scenario['job']['sku'],output/'handling-unit-qr.png')
    cart=UsdGeom.Xform.Define(stage,'/ViewCart')
    cart.GetPrim().GetReferences().AddReference(mobile['cart_usd'])
    pose=scenario['initial_staging']['cart_body_pose']
    cart.AddTranslateOp().Set(Gf.Vec3d(pose[0],pose[1],0.))
    cart.AddRotateZOp().Set(math.degrees(pose[2]))
    robot=parse_urdf(mobile['mobile_urdf'])
    transforms=link_world_transforms(robot,mobile['transport_positions'])
    base=pose_matrix(scenario['initial_staging']['robot_base_pose'],mobile['spec']['base_z'])
    UsdGeom.Xform.Define(stage,'/StaticRobot')
    count=0
    for name,link in robot.links.items():
        for index,geometry in enumerate(link.collisions):
            path=f'/StaticRobot/{name}_{index}'
            scale=np.ones(3)
            if geometry.kind=='mesh':
                file,_=resolve_package_uri(geometry.mesh_filename,config['package_roots'])
                if not file: raise ValueError('unresolved actual CAD geometry')
                mesh=trimesh.load(file,force='mesh',process=False)
                prim=UsdGeom.Mesh.Define(stage,path)
                prim.CreatePointsAttr(mesh.vertices.tolist())
                prim.CreateFaceVertexCountsAttr([3]*len(mesh.faces))
                prim.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
                prim.CreateSubdivisionSchemeAttr('none')
                scale=geometry.mesh_scale
            elif geometry.kind=='box':
                prim=UsdGeom.Cube.Define(stage,path);prim.CreateSizeAttr(1.)
                scale=geometry.box_size
            elif geometry.kind=='cylinder':
                prim=UsdGeom.Cylinder.Define(stage,path)
                prim.CreateRadiusAttr(geometry.radius);prim.CreateHeightAttr(geometry.length)
                prim.CreateAxisAttr('Z')
            elif geometry.kind=='sphere':
                prim=UsdGeom.Sphere.Define(stage,path);prim.CreateRadiusAttr(geometry.radius)
            else:
                raise ValueError('unsupported CAD geometry')
            local=np.eye(4);local[:3,:3]=rpy_to_matrix(geometry.origin_rpy)
            local[:3,3]=geometry.origin_xyz
            matrix=base@transforms[name]@local@np.diag([*scale,1.])
            UsdGeom.Xformable(prim).AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))
            prim.CreateDisplayColorAttr([Gf.Vec3f(.16,.4,.6)])
            count+=1
    textures={}
    for support in design['supports'].values():
        for index,marker in enumerate(support['markers']):
            path=output/(support['id']+'-marker-'+str(index)+'.png')
            write_carton_label(marker['id'],path)
            textures[marker['id']]=path
    paths=author_camera_design(stage,design,parent_paths={'map':'/World','cart_chassis':'/ViewCart/chassis'},
                               marker_textures=textures)
    lighting=_apply_light_rig(stage,design,view,config)
    # This is a new static optical test scene, not a physics mission state.
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(False)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr().Set(False)
    k=Intrinsics(**design['cameras'][view]['intrinsics'])
    camera=initialize_camera(paths[view],k,name='candidate_'+view)
    evidence=dict(evidence='static optical bench, dynamics disabled at initial composition',
        runtime_mission=False,view=view,actual_robot_collision_geometries=count,
        full_store=True,cart_rails_and_camera_solids=True,lighting=lighting,
        input_sha256={name:hashlib.sha256(Path(config[name]).read_bytes()).hexdigest()
                      for name in ('scenario','mobile_manifest','camera_config','store_usd')})
    (output/'view-setup.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return camera,design['cameras'][view],{
        marker['id'] for marker in design['supports'][view]['markers']}
