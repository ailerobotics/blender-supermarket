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
    from pxr import UsdGeom, UsdPhysics
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
    prim.SetCustomDataByKey('restock:sku', sku)
    prim.SetCustomDataByKey('restock:provenance', 'scenario_setup_not_sensor_evidence')
    return str(prim.GetPath())


def build(store, robot_usd, output, *, robot_pose, cart_pose, stock_path, sku,
          stock_mass_kg, spec=MobileSpec()):
    from pxr import Gf, UsdGeom, UsdPhysics, Usd
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
    stock = promote_stock(stage,stock_path,sku,stock_mass_kg)
    stage.GetRootLayer().Save()
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=digest for p,digest in hashes.items()):
        raise RuntimeError('input USD layer changed during composition')
    return dict(stage=str(output),assets=assets,stock=stock,
                static_colliders=static,product_colliders=products,
                evidence='offline_usd_schema_only',physics_verified=False,
                input_sha256={str(p):digest for p,digest in hashes.items()})


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
