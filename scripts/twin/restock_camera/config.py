"""Executable physical camera design, never runtime observations.

Positions are metres; optical +X right, +Y down, +Z forward. Cart parent is
the independently moving chassis origin, 0.20 m above the wheel-ground frame.
Integration must author every solid, render the printed markers, and commission
from actual RGB-D. A config is not permission to move or a calibration result.
"""
from dataclasses import asdict
import numpy as np

from .geometry import Intrinsics, look_at, pose_matrix, usd_camera_transform
from .lights import camera_lamp


def camera_config(scenario, manifest):
    staging,unit = scenario['initial_staging'],scenario['handling_unit']
    if (unit['local_wall'] != 3 or unit['grasp_mode'] != 'side'
            or not np.allclose(unit['dimensions_m'],[.6,.4,.16])
            or not np.isclose(manifest['spec']['arm_mount_z'],.5)):
        raise ValueError('camera design requires audited mobile wall3 handling unit')
    # Per-camera pixel sampling sized so the 60 mm paired QR markers (25-module v2)
    # clear a decode-safe pixels-per-module margin (~6 px/module) while the full
    # .62x.42 support stays framed. Resolution stays 2560x1920, so the bounded QR
    # tile grid (hence decode latency) is unchanged: it is resolution-bound, not
    # marker-size-bound. Source/cart narrow the FOV to enlarge their frontal-ish
    # markers; the shelf keeps the wide FOV because its markers and support span a
    # wide, close arc inside the shelf slot. Installed Isaac
    # Camera.get_intrinsics_matrix uses width/2, height/2 for the principal point.
    def intr(f):
        return Intrinsics(2560,1920,f,f,1280.,960.,.15,3.5)
    k_source,k_cart,k_shelf = intr(2000.),intr(1500.),intr(1280.)
    source = np.array([*staging['support_center_xy'],staging['unit_bottom_z']])
    shelf = np.array(staging['target_surface_pose_world'][:3])
    deck = np.array([*staging['cart_load_center_body_xy'],staging['cart_support_z_world']-.2])
    cameras,solids,supports,lamps = {},[],{},[]

    def solid(name,parent,center,size):
        solids.append(dict(id=name,parent=parent,parent_from_solid=pose_matrix([*center[:2],0],center[2]).tolist(),
                           dimensions_m=list(size),collision=True))

    def camera(name,parent,eye,target,k):
        pose = look_at(eye,target)
        cameras[name] = dict(id=name,parent=parent,parent_from_optical=pose.tolist(),
            intrinsics=asdict(k),revision='physical-paired-marker-v3')
        housing = pose.copy()
        # Glass is the front face, lens at the face; body is behind optical origin.
        housing[:3,3] += pose[:3,:3] @ [0,0,-.014]
        solids.append(dict(id=name+'_housing',parent=parent,parent_from_solid=housing.tolist(),
                           dimensions_m=[.06,.025,.025],collision=True))

    camera('source','map',source+[0,.20,.85],source+[0,0,.08],k_source)
    camera('cart','cart_chassis',deck+[0,-.02,.655],deck+[0,0,.075],k_cart)
    # Shelf camera: same realisable aisle bracket height as before, but y-centred on
    # the marker pair (the old +.30 y offset grazed the far marker at 63.8 deg and
    # 3.65 px/module). Centred, both markers reach ~13 px/module at ~35 deg while the
    # carton-top label plane and full support stay framed. The camera must stay just
    # below the next shelf tier (~0.97 m) to see under it to the carton top; a top-down
    # view is impossible (the tier is ~0.24 m above the support).
    camera('shelf','map',shelf+[-.50,0,.23],shelf+[-.18,0,-.05],k_shelf)
    solid('source_mast','map',source+[0,.44,.15],[.03,.03,1.50])
    solid('source_boom','map',source+[0,.32,.88],[.03,.27,.025])
    solid('source_marker_tab','map',source+[0,.30,-.0015],[.72,.16,.003])
    solid('cart_bracket','cart_chassis',deck+[0,-.02,.674],[.02,.02,.017])
    solid('shelf_bracket','map',shelf+[-.50,0,.27],[.02,.55,.02])
    solid('shelf_anchor','map',shelf+[-.47,-.22,.14],[.08,.02,.26])
    # Camera-mounted ring lamp co-located with the shelf lens. The shelf support sits
    # in a recessed slot the ceiling panels cannot light (grazing_only), so the robot
    # carries its own coaxial lamp: it illuminates exactly what the camera sees. Sphere
    # (omnidirectional) avoids any aim/orientation failure; intensity shares the store
    # rig tuning so one intensity_scale drives it and the fixtures together. Its housing
    # is collision geometry and must be included in navigation/planning.
    # Mounted on the bracket just above and behind the lens so neither the lamp nor its
    # housing sits in the camera frustum (a ring light's body clears the lens); the
    # omnidirectional sphere still lights the markers ~0.35 m away.
    shelf_lamp_center = shelf+[-.53,0,.32]
    lamps.append(camera_lamp('shelf','map',shelf_lamp_center,radius=.03,power_w=6.))
    solid('shelf_lamp_housing','map',shelf_lamp_center,[.05,.05,.02])

    def support(name,parent,center,yaw,marker_centers,witness_center,witness_size):
        pose = pose_matrix([*marker_centers[0][:2],yaw],marker_centers[0][2])
        markers = []
        for index,point in enumerate(marker_centers):
            markers.append(dict(id='support:'+name+':'+str(index),side_m=.06,
                parent_from_marker=pose_matrix([*point[:2],yaw],point[2]).tolist(),
                substrate_thickness_m=.0002))
        auxiliary = np.linalg.inv(pose) @ np.r_[marker_centers[1],1]
        supports[name] = dict(id=name,parent=parent,revision='bolted-pair-v2',
            camera_id=name,required_xy_m=[.62,.42],nominal_center=list(center),yaw=yaw,
            markers=markers,auxiliary_offset_xy=auxiliary[:2].tolist(),
            witness_design=dict(center=list(witness_center),size_xy=list(witness_size),yaw=0.),
            warning='nominal geometry is design only; select support/witness quads from empty RGB-D')

    support('source','map',source,np.pi,[source+[-.30,.30,0],source+[.30,.30,0]],
            source+[0,.30,0],[.18,.06])
    support('cart','cart_chassis',deck,-np.pi/2,[deck+[-.38,-.22,0],deck+[-.38,.22,0]],
            deck+[-.38,0,0],[.06,.18])
    support('shelf','map',shelf,0.,[shelf+[-.42,-.14,0],shelf+[-.42,.14,0]],
            shelf+[-.42,0,0],[.06,.14])
    return dict(schema='restock-camera-design.v2',evidence='geometric_design_not_sensor_capture',
                runtime_ready=False,cameras=cameras,solids=solids,supports=supports,lamps=lamps,
                cart_chassis_height_m=.20,required_depth_error_bound_m=.0002,
                limitations=['Depth error bound must be established for the real renderer/sensor; not assumed.',
                             'Full-link/mount collision and visibility report must be checked before use.',
                             'No commissioning pixels or inventory are provided by this config.'])


def author_camera_design(stage, config, *, parent_paths, marker_textures):
    """Author actual cameras/collision mounts and physically printed QR planes.

    Caller supplies map/cart chassis prim paths and existing QR texture files.
    No simulation boot, fixture removal, hidden-stock toggle or sensor inference.
    Returns camera paths for integration's initialized native camera objects.
    """
    from pathlib import Path
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    def transform(prim,matrix):
        UsdGeom.Xformable(prim).AddTransformOp().Set(Gf.Matrix4d(np.asarray(matrix).T.tolist()))
    def path(parent,name):
        if parent not in parent_paths or not stage.GetPrimAtPath(parent_paths[parent]):
            raise ValueError('physical camera parent missing: '+parent)
        return parent_paths[parent]+'/RestockCamera_'+name
    for item in config['solids']:
        cube = UsdGeom.Cube.Define(stage,path(item['parent'],item['id']))
        cube.CreateSizeAttr(1.)
        matrix = np.array(item['parent_from_solid'])
        matrix[:3,:3] = matrix[:3,:3] @ np.diag(item['dimensions_m'])
        transform(cube.GetPrim(),matrix)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    paths = {}
    for name,item in config['cameras'].items():
        camera = UsdGeom.Camera.Define(stage,path(item['parent'],name))
        transform(camera.GetPrim(),usd_camera_transform(item['parent_from_optical']))
        k = Intrinsics(**item['intrinsics'])
        if (k.cx,k.cy) != (k.width/2,k.height/2):
            raise ValueError('native pinhole authoring requires centred Isaac principal point')
        # USD aperture/focal units cancel; the camera is an ideal pinhole.
        camera.CreateFocalLengthAttr(20.)
        camera.CreateHorizontalApertureAttr(20.*k.width/k.fx)
        camera.CreateVerticalApertureAttr(20.*k.height/k.fy)
        camera.CreateClippingRangeAttr(Gf.Vec2f(k.near,k.far))
        paths[name] = str(camera.GetPath())
    for name,support in config['supports'].items():
        for index,marker in enumerate(support['markers']):
            texture = Path(marker_textures[marker['id']]).resolve()
            if not texture.is_file():
                raise ValueError('printed marker texture missing')
            # Physical metric side refers to decoded QR corners, excluding the
            # quiet border. Preserve that border instead of silently scaling the
            # actual fiducial smaller than the registration model.
            import cv2
            pixels = cv2.imread(str(texture))
            code,corners,_ = cv2.QRCodeDetector().detectAndDecode(pixels)
            if code != marker['id'] or corners is None:
                raise ValueError('marker texture does not decode to its physical identity')
            corners = corners.reshape(4,2)
            edge = np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=1)
            if pixels.shape[0] != pixels.shape[1] or np.ptp(edge) > 1.5:
                raise ValueError('marker texture must be an unwarped square QR')
            base = path(support['parent'],name+'_marker_'+str(index))
            mesh = UsdGeom.Mesh.Define(stage,base)
            side = marker['side_m']*pixels.shape[0]/edge.mean()/2
            mesh.CreatePointsAttr([(-side,-side,.0002),(side,-side,.0002),
                                  (side,side,.0002),(-side,side,.0002)])
            mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
            mesh.CreateSubdivisionSchemeAttr('none')
            transform(mesh.GetPrim(),marker['parent_from_marker'])
            uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex')
            uv.Set([(0,0),(1,0),(1,1),(0,1)])
            material = UsdShade.Material.Define(stage,base+'/Material')
            shader = UsdShade.Shader.Define(stage,base+'/Material/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1.)
            reader = UsdShade.Shader.Define(stage,base+'/Material/UV')
            reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
            tex = UsdShade.Shader.Define(stage,base+'/Material/Texture')
            tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(texture))
            tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return paths
