"""CPU asset generation; pxr is imported only for USD authoring.

The mobile URDF reuses the input UR10/gripper links, meshes and home targets.
Import it with a FREE base and unmerged fixed joints to preserve sensor/hitch
frames. Never import the old fixed-pedestal URDF as the mobile articulation.
"""
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .mobile import MobileSpec
from .restock_camera.wrist_camera import (
    WRIST_FRAME, WRIST_PARENT_LINK, WRIST_MOUNT_XYZ, wrist_usd_camera_transform)


def _element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def _numbers(values):
    return ' '.join(str(v) for v in values)


def _joint(root, name, parent, child, xyz, kind='fixed', axis='0 0 1'):
    j = _element(root, 'joint', name=name, type=kind)
    _element(j, 'parent', link=parent)
    _element(j, 'child', link=child)
    _element(j, 'origin', xyz=_numbers(xyz), rpy='0 0 0')
    if kind != 'fixed':
        _element(j, 'axis', xyz=axis)
        _element(j, 'limit', effort=80, velocity=12)
        _element(j, 'dynamics', damping=0.05, friction=0.01)
    return j


def _link(root, name, mass, size=None, radius=None, length=None):
    l = _element(root, 'link', name=name)
    inertia = _element(l, 'inertial')
    _element(inertia, 'mass', value=mass)
    if size:
        x, y, z = size
        diagonal = (mass*(y*y+z*z)/12, mass*(x*x+z*z)/12, mass*(x*x+y*y)/12)
    else:
        # Cylinder rotated onto Y, matching wheel axis.
        side = mass*(3*radius**2+length**2)/12
        diagonal = (side, mass*radius**2/2, side)
    _element(inertia, 'inertia', ixx=diagonal[0], iyy=diagonal[1], izz=diagonal[2],
             ixy=0, ixz=0, iyz=0)
    for tag in ('visual', 'collision'):
        g = _element(l, tag)
        _element(g, 'origin', xyz='0 0 0', rpy='0 0 0' if size else f'{math.pi/2} 0 0')
        geom = _element(g, 'geometry')
        if size:
            _element(geom, 'box', size=_numbers(size))
        else:
            _element(geom, 'cylinder', radius=radius, length=length)
        if tag == 'visual':
            mat = _element(g, 'material', name=f'{name}_color')
            _element(mat, 'color', rgba='0.12 0.23 0.30 1' if size else '0.06 0.06 0.06 1')
    return l


def build_mobile_urdf(source, output, spec=MobileSpec()):
    """Replace only the known pedestal subtree, leaving arm/gripper unchanged.

    Output retains package:// meshes; caller supplies the existing ROS package
    roots at import. This function never edits or copies upstream mesh files.
    """
    root = copy.deepcopy(ET.parse(source).getroot())
    required = {'world', 'pedestal_link', 'pedestal_top', 'ur_base_link'}
    if not required <= {l.get('name') for l in root.findall('link')}:
        raise ValueError('expected shelfbot UR10 pedestal source')
    removed = {'world', 'pedestal_link', 'pedestal_top'}
    for child in list(root):
        if ((child.tag == 'link' and child.get('name') in removed)
                or (child.tag == 'joint' and child.find('child').get('link') in removed)
                or child.tag in ('gazebo', 'ros2_control', 'transmission')):
            root.remove(child)
    mount = root.find("joint[@name='ur_base_joint']")
    if mount is None or mount.find('parent').get('link') != 'pedestal_top':
        raise ValueError('unexpected arm attachment')
    mount.find('parent').set('link', 'arm_mount')
    root.set('name', 'mobile_ur10_rollwagen')
    _link(root, 'base_link', spec.mass, (spec.length, spec.width, spec.height))
    _link(root, 'arm_mount', 12, (0.24, 0.24, spec.arm_mount_z-spec.base_z-spec.height/2))
    # Mount frame is at the top; shift its geometry/inertia down half column height.
    column = root.find("link[@name='arm_mount']")
    h = spec.arm_mount_z-spec.base_z-spec.height/2
    for tag in ('visual', 'collision'):
        column.find(f'{tag}/origin').set('xyz', f'0 0 {-h/2}')
    _element(column.find('inertial'), 'origin', xyz=f'0 0 {-h/2}', rpy='0 0 0')
    _joint(root, 'arm_mount_joint', 'base_link', 'arm_mount', (0, 0, spec.arm_mount_z-spec.base_z))
    for side, sign in [('left', 1), ('right', -1)]:
        _link(root, f'{side}_wheel', 4, radius=spec.wheel_radius, length=spec.wheel_width)
        _joint(root, f'{side}_wheel_joint', 'base_link', f'{side}_wheel',
               (0, sign*spec.wheel_track/2, spec.wheel_radius-spec.base_z), 'continuous', '0 1 0')
    # Two trailing passive caster assemblies support pitch without skid-steer fiction.
    for label, x in [('front', spec.length/2-0.12), ('rear', -spec.length/2+0.12)]:
        _link(root, f'{label}_caster_fork', 0.3, (0.035, 0.07, 0.02))
        _joint(root, f'{label}_caster_swivel', 'base_link', f'{label}_caster_fork',
               (x, 0, 0.145-spec.base_z), 'continuous')
        _link(root, f'{label}_caster_wheel', 0.5, radius=0.065, length=0.035)
        _joint(root, f'{label}_caster_roll', f'{label}_caster_fork', f'{label}_caster_wheel',
               (-0.025, 0, -0.08), 'continuous', '0 1 0')
    # Drawbar: back edge -.50 to hitch -.62, below the deck.
    _link(root, 'hitch', 1, (abs(spec.hitch_x)-spec.length/2, 0.06, 0.04))
    hitch = root.find("link[@name='hitch']")
    for tag in ('visual', 'collision'):
        hitch.find(f'{tag}/origin').set('xyz', f'{(abs(spec.hitch_x)-spec.length/2)/2} 0 0')
    _element(hitch.find('inertial'), 'origin', xyz=f'{(abs(spec.hitch_x)-spec.length/2)/2} 0 0')
    _joint(root, 'hitch_joint', 'base_link', 'hitch', (spec.hitch_x, 0, spec.hitch_z-spec.base_z))
    for name, xyz in [('lidar_frame', (0.46, 0, 0.15)),
                      ('front_camera_frame', (0.46, 0, 0.35)), ('imu_frame', (0, 0, 0))]:
        _element(root, 'link', name=name)
        _joint(root, f'{name}_joint', 'base_link', name, xyz)
    # Eye-in-hand wrist RGB-D collision camera frame, rigidly on the UR tool flange so the arm's
    # side-grasp workspace is always in its depth (see twin.restock_camera.wrist_camera). The joint
    # is a pure mount offset (identity rotation); the optical rotation lives in the camera design's
    # parent_from_optical and the authored USD Camera prim.
    if root.find(f"link[@name='{WRIST_PARENT_LINK}']") is None:
        raise ValueError(f'source UR chain is missing {WRIST_PARENT_LINK!r} for the wrist camera mount')
    _element(root, 'link', name=WRIST_FRAME)
    _joint(root, f'{WRIST_FRAME}_joint', WRIST_PARENT_LINK, WRIST_FRAME, WRIST_MOUNT_XYZ)
    # Reject disconnected unexpected source structure before writing.
    links = {x.get('name') for x in root.findall('link')}
    children = [j.find('child').get('link') for j in root.findall('joint')]
    parents = {j.find('parent').get('link') for j in root.findall('joint')}
    if set(children) != links-{'base_link'} or len(children) != len(set(children)) or not parents <= links:
        raise ValueError('source contains unsupported detached links')
    ET.indent(root)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding='utf-8', xml_declaration=True)
    return transport_positions(source)


def transport_positions(source):
    root = ET.parse(source).getroot()
    positions = {}
    for joint in root.findall('.//ros2_control/joint'):
        p = joint.find("state_interface[@name='position']/param[@name='initial_value']")
        if p is not None:
            positions[joint.get('name')] = float(p.text)
    if not positions:
        raise ValueError('source has no transport joint configuration')
    # All gripper prismatic tines retract during transport.
    for joint in root.findall('joint'):
        if joint.get('type') == 'prismatic':
            positions[joint.get('name')] = float(joint.find('limit').get('lower'))
    return positions


def author_wrist_camera(stage, frame_prim_path, intrinsics=None, name='wrist_camera'):
    """Author the eye-in-hand wrist ``Camera`` prim under the ``restock_camera_wrist`` frame prim.

    ``frame_prim_path`` is the imported-URDF prim of the wrist frame (e.g.
    ``/MobileRobot/.../restock_camera_wrist``). The optical transform (USD -Z view, +Y up) comes from
    :func:`wrist_camera.wrist_usd_camera_transform`; the pinhole aperture encodes the collision
    intrinsics' field of view. Integration's ``native_cameras`` adopts this prim as the collision
    camera. CPU-safe: only ``pxr`` is imported here, nothing boots.
    """
    import math as _math
    from pxr import UsdGeom, Gf
    from .restock_camera.wrist_camera import wrist_camera_intrinsics, WRIST_HFOV_DEG
    if not stage.GetPrimAtPath(frame_prim_path):
        raise ValueError(f'wrist frame prim not found: {frame_prim_path}')
    k = intrinsics or wrist_camera_intrinsics()
    camera_path = f'{frame_prim_path}/{name}'
    if stage.GetPrimAtPath(camera_path):
        raise ValueError(f'wrist camera already authored: {camera_path}')
    cam = UsdGeom.Camera.Define(stage, camera_path)
    cam.AddTransformOp().Set(Gf.Matrix4d(*[v for row in wrist_usd_camera_transform() for v in row]))
    focal = 18.0
    horizontal_aperture = 2 * focal * _math.tan(_math.radians(WRIST_HFOV_DEG) / 2.)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(horizontal_aperture)
    cam.CreateVerticalApertureAttr(horizontal_aperture * k.height / k.width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(float(k.near), float(k.far)))
    return camera_path


def _usd_box(stage, path, size, xyz, color=(0.6, 0.63, 0.66)):
    from pxr import UsdGeom, UsdPhysics, Gf
    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1)
    box.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    box.AddScaleOp().Set(Gf.Vec3d(*size))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    return box.GetPrim()


def _rigid(stage, path, xyz, mass, diagonal):
    from pxr import UsdGeom, UsdPhysics, Gf
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    api.CreateMassAttr(mass)
    api.CreateCenterOfMassAttr(Gf.Vec3f(0))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal))
    return body.GetPrim()


def _revolute(stage, path, body0, body1, pos0, pos1=(0, 0, 0), axis='Z'):
    from pxr import UsdPhysics, Gf
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
    joint.CreateAxisAttr(axis)
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateExcludeFromArticulationAttr(True)
    return joint


def author_cart(stage, path='/Rollwagen', spec=MobileSpec()):
    """Author movable cage + 4 genuine swivel/rolling casters, initially at origin.

    Root is a placement frame. Chassis local origin is deck centre at z=.20.
    Fork and wheel bodies are siblings, not rigid-body children. Cart is a
    maximal-coordinate mechanism; there are no kinematic bodies or pose drivers.
    """
    from pxr import UsdGeom, UsdPhysics, Gf
    if stage.GetPrimAtPath(path):
        raise ValueError(f'asset already exists: {path}')
    UsdGeom.Xform.Define(stage, path)
    body = f'{path}/chassis'
    l, w, h = spec.cart_length, spec.cart_width, spec.cart_height
    # Hollow cage mass distributed mainly into deck; conservative box inertia.
    m = spec.cart_mass
    _rigid(stage, body, (0, 0, 0.20), m,
           (m*(w*w+h*h)/12, m*(l*l+h*h)/12, m*(l*l+w*w)/12))
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(body))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0.45))
    _usd_box(stage, body+'/deck', (l, w, 0.05), (0, 0, 0), (0.15, 0.25, 0.65))
    _usd_box(stage, body+'/shelf', (l-0.04, w-0.04, 0.025), (0, 0, 0.72))
    for i, (x, y) in enumerate((x, y) for x in (-l/2+0.02, l/2-0.02)
                              for y in (-w/2+0.02, w/2-0.02)):
        _usd_box(stage, body+f'/post_{i}', (0.025, 0.025, h-0.20), (x, y, (h-0.20)/2))
    # Three mesh cage walls; front (+X) open for trays and loading.
    for i, y in enumerate((-w/2+0.02, w/2-0.02)):
        for k in range(1, 10):
            _usd_box(stage, body+f'/side_{i}_horizontal_{k}', (l, 0.012, 0.012), (0, y, k*(h-0.2)/10))
        for k in range(1, 8):
            _usd_box(stage, body+f'/side_{i}_vertical_{k}', (0.01, 0.012, h-0.2),
                     (-l/2+k*l/8, y, (h-0.2)/2))
    for k in range(1, 10):
        _usd_box(stage, body+f'/back_horizontal_{k}', (0.012, w, 0.012),
                 (-l/2+0.02, 0, k*(h-0.2)/10))
    for k in range(1, 6):
        _usd_box(stage, body+f'/back_vertical_{k}', (0.012, 0.01, h-0.2),
                 (-l/2+0.02, -w/2+k*w/6, (h-0.2)/2))
    _usd_box(stage, body+'/drawbar', (spec.cart_hitch_x-l/2, 0.04, 0.04),
             ((spec.cart_hitch_x+l/2)/2, 0, spec.hitch_z-0.20))
    dock = UsdGeom.Xform.Define(stage, body+'/dock')
    dock.AddTranslateOp().Set(Gf.Vec3d(spec.cart_hitch_x, 0, spec.hitch_z-0.20))
    for i, (x, y) in enumerate((x, y) for x in (-l/2+0.10, l/2-0.10)
                              for y in (-w/2+0.08, w/2-0.08)):
        fork, wheel = f'{path}/fork_{i}', f'{path}/wheel_{i}'
        _rigid(stage, fork, (x, y, 0.135), 0.25, (0.0001, 0.0001, 0.0001))
        _usd_box(stage, fork+'/shape', (0.025, 0.07, 0.02), (0, 0, 0.035))
        _rigid(stage, wheel, (x-0.025, y, 0.075), 0.6, (0.0009, 0.00169, 0.0009))
        cyl = UsdGeom.Cylinder.Define(stage, wheel+'/shape')
        cyl.CreateAxisAttr('Y'); cyl.CreateRadiusAttr(0.075); cyl.CreateHeightAttr(0.035)
        cyl.CreateDisplayColorAttr([Gf.Vec3f(0.07)])
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
        _revolute(stage, f'{path}/swivel_{i}', body, fork, (x, y, -0.065))
        _revolute(stage, f'{path}/roll_{i}', fork, wheel, (-0.025, 0, -0.06), axis='Y')
    return {'root': path, 'body': body, 'dock': body+'/dock',
            'stock_supports': [body+'/deck', body+'/shelf']}


def author_hitch(stage, robot_body, cart_body, path='/MobileHitch', spec=MobileSpec()):
    """Create a yaw hinge ONLY from Coupler.attach's checked callback.

    Body frames must be base_link and chassis, not articulation/root Xforms.
    The 65-degree stops transmit real forces; no root transform is modified.
    """
    from pxr import UsdPhysics
    if stage.GetPrimAtPath(path):
        raise ValueError('hitch already exists')
    for body in (robot_body, cart_body):
        p = stage.GetPrimAtPath(body)
        if not p or not p.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f'hitch endpoint is not a rigid body: {body}')
    j = _revolute(stage, path, robot_body, cart_body,
                  (spec.hitch_x, 0, spec.hitch_z-spec.base_z),
                  (spec.cart_hitch_x, 0, spec.hitch_z-0.20))
    j.CreateLowerLimitAttr(-math.degrees(spec.max_articulation))
    j.CreateUpperLimitAttr(math.degrees(spec.max_articulation))
    return str(j.GetPath())


def author(stage, robot_usd, robot_path='/MobileRobot', cart_path='/Rollwagen', spec=MobileSpec()):
    """Reference the imported *mobile* URDF USD and create independent cart.

    Called in an already booted process or with CPU USD. Initial placement is the
    integration caller's responsibility BEFORE physics starts. No robot stub.
    """
    from pxr import UsdGeom
    if stage.GetPrimAtPath(robot_path):
        raise ValueError('robot already exists')
    asset = Path(robot_usd).resolve()
    if not asset.is_file():
        raise ValueError('imported mobile URDF USD required')
    robot = UsdGeom.Xform.Define(stage, robot_path)
    robot.GetPrim().GetReferences().AddReference(str(asset))
    cart = author_cart(stage, cart_path, spec)
    return {'robot_root': robot_path, 'cart': cart, 'spec': spec.manifest()}


def measure_transport(urdf, positions, package_roots, spec=MobileSpec()):
    """Resolve every collision mesh and prove transport footprint/COM offline.

    This is static FK, not a substitute for loaded moving stability or collision
    checking an extended arm. Reject missing mesh data rather than using frames.
    """
    import numpy as np
    from . import urdf_probe as probe
    from .mobile import convex_hull
    robot = probe.parse_urdf(str(urdf))
    transforms = probe.link_world_transforms(robot, positions)
    if probe.find_roots(robot) != ['base_link'] or len(transforms) != len(robot.links):
        raise ValueError('mobile URDF must be a connected free-base tree')
    points, total, weighted = [], 0.0, np.zeros(3)
    for name, link in robot.links.items():
        inertia = link.inertial
        if inertia:
            matrix = np.array([[inertia.ixx, inertia.ixy, inertia.ixz],
                               [inertia.ixy, inertia.iyy, inertia.iyz],
                               [inertia.ixz, inertia.iyz, inertia.izz]], dtype=float)
            eigen = np.linalg.eigvalsh(matrix)
            if (not np.isfinite(matrix).all() or inertia.mass is None or inertia.mass <= 0
                    or not np.isfinite(inertia.mass) or eigen[0] <= 0
                    or eigen[2] > eigen[0]+eigen[1]+1e-6):
                raise ValueError(f'invalid mass/inertia: {name}')
            total += inertia.mass
            weighted += inertia.mass*(transforms[name]@np.array([*inertia.origin_xyz, 1]))[:3]
        for geometry in link.collisions:
            vertices = probe.geometry_world_points(geometry, transforms[name],
                lambda uri: probe.resolve_package_uri(uri, package_roots)[0])
            if vertices is None or not np.isfinite(vertices).all():
                raise ValueError(f'unresolved collision geometry: {name}')
            points.extend(vertices)
    points = np.array(points)
    radius = float(np.linalg.norm(points[:, :2], axis=1).max())
    if radius > spec.transport_radius:
        raise ValueError(f'transport radius {radius:.3f} exceeds {spec.transport_radius:.3f}')
    com = weighted/total
    # Ground contacts form a diamond; caster trail can turn either direction.
    support = convex_hull([(0, spec.wheel_track/2), (0, -spec.wheel_track/2),
                           (spec.length/2-0.145, 0), (-spec.length/2+0.145, 0)])
    margin = min(((b[0]-a[0])*(com[1]-a[1])-(b[1]-a[1])*(com[0]-a[0]))/
                 math.dist(a, b) for a, b in zip(support, support[1:]+support[:1]))
    if margin < 0.08:
        raise ValueError(f'home COM insufficient static support margin: {margin:.3f}')
    return {'source': 'offline_urdf_fk_not_physics', 'mass_kg': total,
            'home_collision_radius_m': radius, 'configured_transport_radius_m': spec.transport_radius,
            'home_bounds_base_frame': [points.min(0).tolist(), points.max(0).tolist()],
            'com_base_frame_m': com.tolist(), 'static_support_margin_m': float(margin),
            'dynamic_stability': 'unverified; integration must step loaded articulated assets'}
