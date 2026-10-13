"""Executable physical camera design, never runtime observations.

Positions are metres; optical +X right, +Y down, +Z forward. Cart parent is
the independently moving chassis origin, 0.20 m above the wheel-ground frame.
Integration must author every solid, render the printed markers, and commission
from actual RGB-D. A config is not permission to move or a calibration result.
"""
from dataclasses import asdict
import numpy as np

from .geometry import (Intrinsics, look_at, pose_matrix, usd_camera_transform,
                       project, transform_points, rigid)
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
    # Aimed DISK ring lamp at the slot mouth, below the ~0.97 m shelf tier, emission
    # pointed at the marker centroid. A disk is an area light sampled like the store
    # RectLights (which illuminate the native RTX render); a tiny SphereLight at the
    # same nits emitted ~nothing there (see shelf-lamp-native-diagnosis.md). The emitter
    # stays below the tier and on the clear sightline the camera uses; the body/housing
    # stays above-and-behind, out of the camera frustum and the emitter->marker rays.
    # plan_restock_lighting.py ray-tests emitter->marker against both housings, the
    # shelf tiers and products in the rendered store; keep both rays clear if retuned.
    shelf_lamp_emitter = shelf+[-.50,0,.17]
    shelf_marker_centroid = shelf+[-.42,0,0]
    shelf_lamp_body = shelf+[-.55,0,.33]
    lamps.append(camera_lamp('shelf','map',shelf_lamp_emitter,kind_type='disk',
                             aim_at=shelf_marker_centroid,radius=.06,power_w=8.))
    solid('shelf_lamp_housing','map',shelf_lamp_body,[.04,.04,.03])

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


def marker_module_grid(sku):
    """The QR module matrix for ``sku`` with a 4-module white quiet border (0 == black).

    Byte-identical content to ``write_carton_label``'s PNG (encode then a 4-module
    border), so a geometry fiducial built from it carries the same code.
    """
    import cv2
    grid = cv2.copyMakeBorder(cv2.QRCodeEncoder_create().encode(sku), 4, 4, 4, 4,
                              cv2.BORDER_CONSTANT, value=255)
    return grid


def marker_geometry_quads(grid, half):
    """White base quad and one black quad per dark module across a +/-``half`` footprint.

    The pattern is laid over the same footprint the textured marker would occupy, so the
    physical size and pose are unchanged; only the texture becomes geometry. Grid row 0 is
    the top (+y). Returns (white_quad, black_quads) as marker-local XY corner lists.
    """
    module_count = grid.shape[0]
    module = 2 * half / module_count
    white_quad = [(-half, -half), (half, -half), (half, half), (-half, half)]
    black_quads = []
    for row in range(module_count):
        for col in range(module_count):
            if grid[row, col] == 0:
                x0 = -half + col * module
                y1 = half - row * module
                black_quads.append([(x0, y1 - module), (x0 + module, y1 - module),
                                    (x0 + module, y1), (x0, y1)])
    return white_quad, black_quads


def render_marker_at_camera(grid, half, world_from_optical, parent_from_marker,
                            intrinsics, *, background=128, upscale=16):
    """Render-faithful projection of a geometry fiducial into the camera image (CPU).

    Projects the +/-``half`` white-quad world corners through the exact pinhole and warps
    the module ``grid`` into a camera-size canvas, so decoding it proves the geometric
    pattern reads at this pose and pixel size. Not a renderer: no lighting, material,
    occlusion or noise, so a successful decode is necessary, not sufficient, for the
    native render. Returns (canvas, in_fov).
    """
    import cv2
    k = intrinsics
    module_count = grid.shape[0]
    # Grid-image corners TL,TR,BR,BL <-> marker-plane corners (row 0 is +y top).
    local = np.array([[-half, half, .0002, 1], [half, half, .0002, 1],
                      [half, -half, .0002, 1], [-half, -half, .0002, 1]])
    world = (rigid(parent_from_marker) @ local.T).T[:, :3]
    uv, valid = project(transform_points(np.linalg.inv(rigid(world_from_optical)), world), k)
    canvas = np.full((k.height, k.width), background, np.uint8)
    src = cv2.resize(grid.astype(np.uint8), (module_count * upscale, module_count * upscale),
                     interpolation=cv2.INTER_NEAREST)
    h, w = src.shape
    homography = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]), uv.astype(np.float32))
    # Area-sampled warp, as a real sensor integrates over each pixel footprint, so the
    # oblique pattern is anti-aliased rather than point-sampled (point sampling aliases a
    # crisp binary grid and understates decodability).
    cv2.warpPerspective(src, homography, (k.width, k.height), dst=canvas,
                        flags=cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
    return canvas, bool(valid.all())


def render_authored_marker(stage, base_path, world_from_optical, intrinsics, *, background=128):
    """Rasterise an AUTHORED marker (white base + black Modules meshes) at the camera.

    Reads the composed world-space vertices of the authored prims and fills the projected
    polygons (white base, then each black module on top), so an authoring fault that moves
    the modules (e.g. a child mesh double-applying its parent transform) shows up as a
    blank or off-frame pattern. Returns (canvas, white_corners_uv). pxr + cv2 required.
    """
    import cv2
    from pxr import UsdGeom
    k = intrinsics
    xform = UsdGeom.XformCache()

    def world_quads(prim):
        matrix = np.array(xform.GetLocalToWorldTransform(prim)).T
        points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        world = (matrix @ np.c_[points, np.ones(len(points))].T).T[:, :3]
        return [world[i:i + 4] for i in range(0, len(world), 4)]

    def to_px(quad):
        uv, _ = project(transform_points(np.linalg.inv(rigid(world_from_optical)), quad), k)
        return uv

    canvas = np.full((k.height, k.width), background, np.uint8)
    white = world_quads(stage.GetPrimAtPath(base_path))[0]
    cv2.fillPoly(canvas, [np.round(to_px(white)).astype(np.int32)], 255)
    modules = stage.GetPrimAtPath(base_path + '/Modules')
    if modules:
        for quad in world_quads(modules):
            cv2.fillPoly(canvas, [np.round(to_px(quad)).astype(np.int32)], 0)
    return canvas, to_px(white)


def decode_authored_marker(stage, base_path, world_from_optical, intrinsics, *, upscale=16):
    """Rasterise the authored marker at the camera, rectify by its base corners, decode."""
    import cv2
    canvas, white_uv = render_authored_marker(stage, base_path, world_from_optical, intrinsics)
    side = 33 * upscale
    homography = cv2.getPerspectiveTransform(
        white_uv.astype(np.float32), np.float32([[0, side], [side, side], [side, 0], [0, 0]]))
    frontal = cv2.warpPerspective(canvas, homography, (side, side), flags=cv2.INTER_AREA)
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(cv2.cvtColor(frontal, cv2.COLOR_GRAY2BGR))
    return text


def decode_marker_render(grid, half, world_from_optical, parent_from_marker, intrinsics,
                         *, upscale=16):
    """Render the geometry marker at the camera, rectify by its projected corners and decode.

    A marker-registration pipeline detects the fiducial corners and rectifies before
    decoding; this does the same from the known projection, so a correct decode proves the
    geometric pattern carries its code at this pose and pixel size. (A bare auto-detector
    is fragile on a strongly sheared QR; rectification is the faithful test.) Returns
    (decoded_text, full_quad_in_fov).
    """
    import cv2
    canvas, in_fov = render_marker_at_camera(grid, half, world_from_optical,
                                             parent_from_marker, intrinsics, upscale=upscale)
    local = np.array([[-half, half, .0002, 1], [half, half, .0002, 1],
                      [half, -half, .0002, 1], [-half, -half, .0002, 1]])
    world = (rigid(parent_from_marker) @ local.T).T[:, :3]
    uv, _ = project(transform_points(np.linalg.inv(rigid(world_from_optical)), world), intrinsics)
    side = grid.shape[0] * upscale
    homography = cv2.getPerspectiveTransform(
        uv.astype(np.float32), np.float32([[0, 0], [side, 0], [side, side], [0, side]]))
    frontal = cv2.warpPerspective(canvas, homography, (side, side), flags=cv2.INTER_AREA)
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(cv2.cvtColor(frontal, cv2.COLOR_GRAY2BGR))
    return text, in_fov


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
    def constant_material(matpath, mesh_prim, color, emissive=None):
        material = UsdShade.Material.Define(stage,matpath)
        shader = UsdShade.Shader.Define(stage,matpath+'/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1.)
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        # Truly matte: a diffuseColor-0 UsdPreviewSurface still keeps metallic 0 + ior 1.5,
        # whose dielectric Fresnel specular rises at grazing angles, so the black modules
        # mirrored the bright white base and the lamp and floored at ~0.42 at the oblique
        # shelf view. ior 1.0 gives F0=0 -> zero specular at every angle; metallic 0 keeps
        # it dielectric. Matte the white base too so no glossy highlight blooms.
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
        shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.)
        if emissive is not None:
            shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(material)

    def quad_mesh(prim_path,quads,z,parent_from_marker):
        mesh = UsdGeom.Mesh.Define(stage,prim_path)
        points,counts,indices = [],[],[]
        for quad in quads:
            start = len(points)
            points += [(x,y,z) for (x,y) in quad]
            counts.append(4); indices += [start,start+1,start+2,start+3]
        mesh.CreatePointsAttr(points); mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices); mesh.CreateSubdivisionSchemeAttr('none')
        # Double-sided so the module/base faces render regardless of winding vs the camera.
        mesh.CreateDoubleSidedAttr(True)
        if parent_from_marker is not None:
            transform(mesh.GetPrim(),parent_from_marker)
        return mesh

    for name,support in config['supports'].items():
        for index,marker in enumerate(support['markers']):
            base = path(support['parent'],name+'_marker_'+str(index))
            if name == 'shelf':
                # Geometry fiducial (no texture): the native render never sampled the shelf
                # markers' texture, so build the pattern from a white base quad plus one
                # black quad per dark module. Same footprint/pose as the textured marker
                # (size read from the printed PNG), constant-colour with emissive white so
                # it reads in or out of the lamp; black modules sit a sliver above the base.
                import cv2
                texture = Path(marker_textures[marker['id']]).resolve()
                pixels = cv2.imread(str(texture))
                code,corners,_ = cv2.QRCodeDetector().detectAndDecode(pixels)
                if code != marker['id'] or corners is None:
                    raise ValueError('marker texture does not decode to its physical identity')
                edge = np.linalg.norm(np.roll(corners.reshape(4,2),-1,axis=0)-corners.reshape(4,2),axis=1)
                half = marker['side_m']*pixels.shape[0]/edge.mean()/2
                grid = marker_module_grid(marker['id'])
                white_quad,black_quads = marker_geometry_quads(grid,half)
                white = quad_mesh(base,[white_quad],.0002,marker['parent_from_marker'])
                # Passive diffuse white (NO emissive): the lamp lights the support now, and
                # an emissive white base glowed through/around the thin black modules
                # (0.4 mm above it) and washed the pattern to a near-uniform bright plane in
                # the native render (blacks ~122/255, no module contrast). A diffuse white
                # that the black diffuse modules occlude renders the pattern with contrast
                # regardless of lamp intensity (diffuse black reflects ~0 even over-lit).
                constant_material(base+'/Material',white.GetPrim(),(1,1,1))
                if black_quads:
                    # Child of the white base: it INHERITS base's parent_from_marker, so
                    # it must not re-apply it (that double transform put the QR modules at
                    # twice the world coords, behind the camera, leaving only a blank white
                    # plane in the native render). Author the quads in marker-local coords.
                    # Offset 2 mm in front of the base (base is at local z .0002) along the
                    # marker normal toward the camera: at .0006 (0.4 mm) the base won the depth
                    # test in the native RTX render (within the ray self-intersection epsilon)
                    # so the modules never rendered and the "black" read the white base; 2 mm
                    # clears that while the uniform in-plane shift does not corrupt the code.
                    black = quad_mesh(base+'/Modules',black_quads,.0022,None)
                    constant_material(base+'/Modules/Material',black.GetPrim(),(0,0,0))
                continue
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
            # Plain diffuse texture only. An emissive copy of the QR (tried as a render
            # hardening) reached the SOURCE markers and over-exposed them (support max
            # 158 -> 240), breaking the decode that worked before; the shelf markers are
            # geometry instead (they ignored the texture entirely). Do not add emissive or
            # a colour-space override to the source/cart texture markers.
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return paths
