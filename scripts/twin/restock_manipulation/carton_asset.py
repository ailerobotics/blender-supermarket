"""Author the physically labelled supported handling unit in an output stage.

Initial placement is scene construction, not a perception input. Runtime sensing
only sees its rendered QR texture/depth; no object path is passed to perception.
"""
from pathlib import Path


def author_carton(stage, *, path, profile, label_png, centre):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    if not Path(label_png).is_file():
        raise ValueError('generate the real SKU label texture before authoring')
    if stage.GetPrimAtPath(path):
        raise ValueError('carton path already exists')
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*centre))
    body = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(profile.mass_kg)
    cube = UsdGeom.Cube.Define(stage, path + '/body')
    cube.CreateSizeAttr(1.)
    cube.AddScaleOp().Set(Gf.Vec3f(*profile.dimensions))
    cube.CreateDisplayColorAttr([Gf.Vec3f(.45,.30,.15)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    material = UsdShade.Material.Define(stage, path + '/cardboard')
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(.8)
    physics.CreateDynamicFrictionAttr(.6)
    physics.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose='physics')
    # A centred 60 mm printed patch is large enough to decode at the configured
    # camera resolution while leaving the rest of the suctionable top unaltered.
    half = .03
    z = profile.dimensions[2]/2 + .00005
    label = UsdGeom.Mesh.Define(stage, path + '/printed_label')
    label.CreatePointsAttr([(-half,-half,z),(half,-half,z),(half,half,z),(-half,half,z)])
    label.CreateFaceVertexCountsAttr([4])
    label.CreateFaceVertexIndicesAttr([0,1,2,3])
    label.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(label).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                           UsdGeom.Tokens.vertex).Set([(0,0),(1,0),(1,1),(0,1)])
    mat = UsdShade.Material.Define(stage, path + '/label_material')
    reader = UsdShade.Shader.Define(stage, path + '/label_material/uv')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, path + '/label_material/texture')
    tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(label_png).resolve())))
    tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
    tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    shader = UsdShade.Shader.Define(stage, path + '/label_material/surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.7)
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(label.GetPrim()).Bind(mat)
    return path
