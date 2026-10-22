"""CPU USD scene assembly; all placements below occur before physics starts.

An imported mobile articulation is mandatory. Semantic stock identifiers configure
actors, never perception. Input layers are read-only and output must be new.
"""
from pathlib import Path
import hashlib
import math

from twin import compose
from twin.mobile import MobileSpec
from twin.mobile_assets import author


def promote_stock(stage, path, sku, mass_kg):
    from pxr import UsdGeom, UsdPhysics, Sdf
    prim = stage.GetPrimAtPath(path)
    if (not prim or not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy() or
            prim.IsInPrototype() or prim.IsInstance()):
        raise ValueError('stock must be a non-instanced concrete mesh')
    if not math.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError('measured stock mass required')
    if not isinstance(sku, str) or not sku:
        raise ValueError('SKU required')
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set('convexHull')
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr().Set(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(mass_kg)
    # Keep the pick unit from sleeping (run-194 seal diagnostic): it settles to ~0 velocity on its
    # support during the long approach/commissioning, and Isaac's SurfaceGripper does not seal onto a
    # SLEEPING rigid body -- the seal reached the carton at a measured 0.45 mm, perfectly aligned, with
    # suction on, yet grippedObjects stayed empty. sleepThreshold 0 disables sleep so the settled unit
    # stays an active grab target. Authored as a raw PhysX attribute (no PhysxSchema import) so the CPU
    # usd-core compose path still loads; Isaac honours it at runtime.
    prim.AddAppliedSchema('PhysxRigidBodyAPI')
    prim.CreateAttribute('physxRigidBody:sleepThreshold', Sdf.ValueTypeNames.Float, False).Set(0.0)
    prim.SetCustomDataByKey('restock:sku', sku)
    prim.SetCustomDataByKey('restock:provenance', 'scenario_setup_not_sensor_evidence')
    return str(prim.GetPath())


def build(store, robot_usd, output, *, robot_pose, cart_pose, stock_path, sku,
          stock_mass_kg, spec=MobileSpec(), scenario=None, cage_collision=None):
    from pxr import Gf, UsdGeom, UsdPhysics, Usd
    from . import cage_collision as _cage
    # Validate the opt-in sim-only cage filter mode BEFORE creating the output (unknown value => refuse).
    cage_mode = _cage.mode({_cage.KEY: cage_collision})
    store, robot_usd, output = map(lambda x: Path(x).resolve(), (store, robot_usd, output))
    if output.exists() or output in (store, robot_usd):
        raise ValueError('scene output must be a new scratch file')
    for pose in (robot_pose, cart_pose):
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            raise ValueError('initial poses must be finite x/y/yaw')
    if not store.is_file() or not robot_usd.is_file():
        raise ValueError('store and imported mobile robot USD are required')
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (store,robot_usd)}
    # Fail before output creation on an incompatible robot asset.
    robot_stage = Usd.Stage.Open(str(robot_usd))
    if not robot_stage or not robot_stage.GetDefaultPrim():
        raise ValueError('imported robot must have a default prim')
    if not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in robot_stage.Traverse()):
        raise ValueError('imported robot is not an articulation')
    stage, root = compose.open_output_stage(str(output), str(store))
    compose.author_physics_scene(stage, root)
    static = compose.author_static_colliders(stage, root)
    products = compose.author_product_colliders(stage, root)
    assets = author(stage, robot_usd, spec=spec)
    # Imported base_link origin is chassis centre; cart root is at ground plane.
    for path, pose, z in ((assets['robot_root'],robot_pose,spec.base_z),
                          (assets['cart']['root'],cart_pose,0.0)):
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(pose[0],pose[1],z))
        xform.AddRotateZOp().Set(math.degrees(pose[2]))
    initial = author_initial_scenario(stage, scenario) if scenario is not None else None
    stock = initial['stock'] if initial else promote_stock(stage,stock_path,sku,stock_mass_kg)
    # Opt-in SIM-ONLY workaround (restock.cage_collision): carton-vs-cage contacts filtered; None => off.
    cage = _cage.author_filter(stage,stock,assets['cart']['body']) if cage_mode else None
    stage.GetRootLayer().Save()
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=digest for p,digest in hashes.items()):
        raise RuntimeError('input USD layer changed during composition')
    result = dict(stage=str(output),assets=assets,stock=stock,initial_scenario=initial,
                  static_colliders=static,product_colliders=products,
                  evidence='offline_usd_schema_only',physics_verified=False,
                  input_sha256={str(p):digest for p,digest in hashes.items()})
    if cage is not None:
        result['cage_collision'] = cage
    return result


def label_stock(stage, stock_path, sku, texture_path, *, side_m=.12):
    """Author a physical printed QR label for RGB-D perception, not semantic IDs.

    The label is centred on the carton top in the mesh's local coordinates.
    Texture is generated offline in caller-owned scratch storage and remains a
    normal rendered material. The detector must decode actual camera pixels.
    """
    import qrcode
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    prim=stage.GetPrimAtPath(stock_path)
    mesh=UsdGeom.Mesh(prim)
    points=mesh.GetPointsAttr().Get() if mesh else None
    if not points or not isinstance(sku,str) or not sku:
        raise ValueError('concrete carton mesh and SKU required')
    lo=[min(p[i] for p in points) for i in range(3)]
    hi=[max(p[i] for p in points) for i in range(3)]
    if not math.isfinite(side_m) or side_m<=0 or any(hi[i]-lo[i]<side_m+.02 for i in (0,1)):
        raise ValueError('QR label does not fit carton top')
    # Axis-aligned sealed carton only: sloped or arbitrary mesh tops unsupported.
    top=[p for p in points if abs(p[2]-hi[2])<1e-6]
    if len(top)<4:
        raise ValueError('carton lacks a planar rectangular top')
    texture_path=Path(texture_path).resolve()
    if texture_path.exists():
        raise ValueError('label texture output must be new')
    texture_path.parent.mkdir(parents=True,exist_ok=True)
    qrcode.make(sku).save(texture_path)
    label=UsdGeom.Mesh.Define(stage,stock_path+'/RestockLabel')
    x,y=(hi[0]+lo[0])/2,(hi[1]+lo[1])/2
    z=hi[2]+.0005;r=side_m/2
    label.CreatePointsAttr([(x-r,y-r,z),(x+r,y-r,z),(x+r,y+r,z),(x-r,y+r,z)])
    label.CreateFaceVertexCountsAttr([4]);label.CreateFaceVertexIndicesAttr([0,1,2,3])
    label.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(label).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying').Set(
        [(0,0),(1,0),(1,1),(0,1)])
    mat=UsdShade.Material.Define(stage,stock_path+'/QRMaterial')
    shader=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.7)
    tex=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Texture')
    tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(texture_path))
    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    reader=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/UV')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput('result',Sdf.ValueTypeNames.Float2))
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3))
    mat.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput('surface',Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI.Apply(label.GetPrim()).Bind(mat)
    return str(label.GetPath())


def author_initial_scenario(stage, scenario):
    """Author a depleted bay, support and one sealed unit before simulation.

    Caller must invoke only during composition, before physics starts. Authored
    transforms describe initial conditions, never motion or sensed inventory.
    Only products tagged with the exact selected slot are deactivated; fixture
    prims remain untouched. The input store layer stays read-only.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    staging, unit = scenario['initial_staging'], scenario['handling_unit']
    if not staging.get('initial_conditions_only') or not staging.get('target_initially_depleted'):
        raise ValueError('explicit initial depleted scenario required')
    if stage.GetPrimAtPath('/RestockScenario'):
        raise ValueError('scenario already authored; runtime repositioning refused')
    dimensions = unit['dimensions_m']
    if (len(dimensions) != 3 or any(not math.isfinite(v) or v <= 0 for v in dimensions)
            or not math.isfinite(unit['mass_kg']) or unit['mass_kg'] <= 0):
        raise ValueError('finite positive handling unit dimensions/mass required')
    slot = scenario['job']['shelf_id']
    matches = []
    for prim in Usd.PrimRange.Stage(stage):
        attr = prim.GetAttribute('userProperties:slot')
        if attr and attr.Get() == slot:
            if '/GRP_Products/' not in str(prim.GetPath()):
                continue  # Price tags retain the slot identity; fixtures are never depleted.
            matches.append(prim.GetPath())
    if not matches:
        raise ValueError('target product slot not found; refusing unverified depletion')
    for path in matches:
        stage.OverridePrim(path).SetActive(False)
    UsdGeom.Xform.Define(stage, '/RestockScenario')
    support = UsdGeom.Cube.Define(stage, '/RestockScenario/Support')
    support.CreateSizeAttr(1.)
    size = staging['support_dimensions_m']
    support.AddTranslateOp().Set(Gf.Vec3d(*staging['support_center_xy'],size[2]/2))
    support.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(support.GetPrim())
    support.CreateDisplayColorAttr([Gf.Vec3f(.3,.32,.34)])
    # fm459: optional backroom TRANSFER BAY -- a static support surface (collider) the carton is placed
    # onto when the mission transfers to a backroom bay instead of the store shelf (the store shelf is
    # baked south of the portal). Absent => unchanged (store-shelf transfer). Its top surface must sit at
    # the shelf camera's commissioned support z, and the shelf station camera is repositioned over it (see
    # camera-config-bay.json) so commissioning + camera-measured verify register the SAME surface. A box
    # from the floor to `top_z`, footprint `dimensions_m[:2]`.
    bay = scenario.get('transfer_bay')
    if bay:
        bx, by = bay['center_xy']
        bw, bd = bay['dimensions_m'][0], bay['dimensions_m'][1]
        top_z = float(bay['top_z'])
        if not (math.isfinite(bx) and math.isfinite(by) and bw > 0 and bd > 0 and top_z > 0):
            raise ValueError('transfer_bay needs finite center_xy, positive dimensions_m and top_z')
        bayprim = UsdGeom.Cube.Define(stage, '/RestockScenario/TransferBay')
        bayprim.CreateSizeAttr(1.)
        bayprim.AddTranslateOp().Set(Gf.Vec3d(bx, by, top_z / 2))
        bayprim.AddScaleOp().Set(Gf.Vec3f(bw, bd, top_z))
        UsdPhysics.CollisionAPI.Apply(bayprim.GetPrim())
        bayprim.CreateDisplayColorAttr([Gf.Vec3f(.28, .3, .34)])
    # Mesh is needed by the same collider and physical QR authoring used above.
    mesh = UsdGeom.Mesh.Define(stage, '/RestockScenario/HandlingUnit')
    x,y,z = [v/2 for v in dimensions]
    mesh.CreatePointsAttr([(-x,-y,-z),(x,-y,-z),(x,y,-z),(-x,y,-z),
                           (-x,-y,z),(x,-y,z),(x,y,z),(-x,y,z)])
    mesh.CreateFaceVertexCountsAttr([4]*6)
    mesh.CreateFaceVertexIndicesAttr([0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([(-x,-y,-z),(x,y,z)])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(.65,.42,.18)])
    mesh.AddTranslateOp().Set(Gf.Vec3d(*staging['support_center_xy'],staging['unit_bottom_z']+z))
    mesh.AddRotateZOp().Set(math.degrees(staging['unit_yaws_world']['source']))
    stock = promote_stock(stage,str(mesh.GetPath()),unit['sku'],unit['mass_kg'])
    return dict(stock=stock, depleted_product_prims=list(map(str,matches)),
                support=str(support.GetPath()), evidence='initial_conditions_only',
                counted_unit='sealed_handling_unit', hidden_contents_counted=False,
                transfer_bay=('/RestockScenario/TransferBay' if scenario.get('transfer_bay') else None))
