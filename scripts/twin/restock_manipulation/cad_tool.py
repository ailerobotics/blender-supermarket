"""Actual CAD suction-panel capability and native Isaac attachment authoring.

URDF gives geometry, not a vacuum actuator. This module adds Isaac's native
SurfaceGripper at the measured CAD seal, never a hand-authored payload weld.
It does not select a payload by path, set body1, or move any product transform.
"""
from dataclasses import dataclass
import math

import numpy as np

from ..urdf_probe import parse_urdf, link_world_transforms, geometry_world_points

ARM_JOINTS = tuple('ur_' + name for name in (
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'))
TINE_JOINTS = tuple(f'surface_plate_{i}_joint' for i in range(1, 7))


@dataclass(frozen=True)
class CadTool:
    arm_joints: tuple
    arm_limits: tuple
    panel_size: tuple
    tool0_from_pad: np.ndarray
    tool0_from_support: np.ndarray
    tine_limits: tuple
    minimum_top_gap: float = .0025
    envelope_pad_min: tuple = (-.145, -.06702, -.272034)
    envelope_pad_max: tuple = (.146, .059328, .001919)
    tcp: str = 'gripper_pad_tcp'
    base_frame: str = 'ur_base_link'

    def validate_profile(self, profile):
        if not math.isfinite(profile.mass_kg) or not 0 < profile.mass_kg <= .5:
            raise ValueError('CAD simulation slice limits carton catalog mass to 0.5 kg')
        if profile.grasp_mode == 'top':
            if (profile.dimensions[0] < self.panel_size[0] + .02
                    or profile.dimensions[1] < self.panel_size[1] + .02):
                raise ValueError('handling unit top is smaller than the CAD suction panel')
            if not np.allclose(profile.tcp_quaternion, (1, 0, 0, 0), atol=1e-5):
                raise ValueError('top slice requires CAD pad +Z vertically down')
        elif profile.grasp_mode == 'side':
            if (max(profile.dimensions[:2]) < self.panel_size[0] + .02
                    or profile.dimensions[2] < self.panel_size[1] + .02):
                raise ValueError('carton wall is smaller than the CAD suction panel')
            if profile.dimensions[2]/2 + self.envelope_pad_min[1] < .01:
                raise ValueError('retracted CAD tines would approach within 10 mm of the support')
        else:
            raise ValueError('unsupported CAD grasp mode')
        if not self.minimum_top_gap <= profile.tcp_surface_offset <= .003:
            raise ValueError('CAD panel tilt requires calibrated 2.5..3 mm approach clearance')


def inspect_tool(urdf_path):
    robot = parse_urdf(str(urdf_path))
    required = ('ur_base_link', 'ur_tool0', 'suction_cups', 'gripper_pad_tcp', 'gripper_tcp')
    if any(name not in robot.links for name in required):
        raise ValueError('URDF is not the supported CAD panel/tine tool')
    joints = robot.joints_by_name
    if any(name not in joints for name in ARM_JOINTS + TINE_JOINTS):
        raise ValueError('CAD arm/tine joint names missing')
    if any(joints[n].jtype != 'prismatic' or joints[n].axis != (0, 1, 0) for n in TINE_JOINTS):
        raise ValueError('unexpected tine kinematics')
    panels = [g for g in robot.links['suction_cups'].collisions if g.kind == 'box']
    if len(panels) != 1:
        raise ValueError('need one measured CAD suction-panel collision box')
    transforms = link_world_transforms(robot, root='ur_tool0')
    points = []
    for link, transform in transforms.items():
        for geometry in robot.links[link].collisions:
            cloud = geometry_world_points(geometry, transform, [])
            if cloud is not None:
                points.extend(cloud)
    relative = np.asarray(points) - transforms['gripper_pad_tcp'][:3,3]
    return CadTool(ARM_JOINTS, tuple((joints[n].lower, joints[n].upper) for n in ARM_JOINTS),
                   tuple(panels[0].box_size), transforms['gripper_pad_tcp'],
                   transforms['gripper_tcp'],
                   tuple((joints[n].lower, joints[n].upper) for n in TINE_JOINTS),
                   envelope_pad_min=tuple(relative.min(axis=0)),
                   envelope_pad_max=tuple(relative.max(axis=0)))


def author_native_gripper(stage, *, tool, tool0_path, body_path, gripper_path,
                           max_gap=.005, force_limit=30.):
    """Author BEFORE physics starts, on the task's composed output stage only.

    body_path must name the actual rigid body carrying the CAD panel (importers
    may merge fixed links). tool0_path supplies that link's robot-frame transform.
    Native runtime owns all body1 relationships; they start EMPTY and disabled.
    Four sample points lie inside the actual panel. Missing any seal is a miss.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    if not 0 < max_gap <= .005 or not math.isfinite(force_limit) or force_limit <= 0:
        raise ValueError('invalid seal distance/force')
    body, tool0 = stage.GetPrimAtPath(body_path), stage.GetPrimAtPath(tool0_path)
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI) or not tool0:
        raise ValueError('actual gripper rigid body and tool0 prim are required')
    if stage.GetPrimAtPath(gripper_path):
        raise ValueError('refusing to overwrite an existing gripper')
    cache = UsdGeom.XformCache()
    # USD matrices use row vectors, our URDF matrices column vectors.
    body_from_tool0 = np.asarray(cache.GetLocalToWorldTransform(tool0) *
                                cache.GetLocalToWorldTransform(body).GetInverse()).T
    body_from_pad = body_from_tool0 @ tool.tool0_from_pad
    rotation = Gf.Matrix3d(*body_from_pad[:3, :3].T.ravel().tolist()).ExtractRotation().GetQuat()
    grip = stage.DefinePrim(gripper_path, 'IsaacSurfaceGripper')
    for name, value in (('maxGripDistance', max_gap), ('coaxialForceLimit', force_limit),
                        ('shearForceLimit', force_limit), ('retryInterval', 0.)):
        grip.CreateAttribute('isaac:' + name, Sdf.ValueTypeNames.Float, False).Set(value)
    grip.CreateAttribute('isaac:status', Sdf.ValueTypeNames.Token, False).Set('Open')
    grip.CreateRelationship('isaac:grippedObjects', False)
    paths = []
    for i, (x, y) in enumerate(((-1,-1), (-1,1), (1,-1), (1,1))):
        path = f'{gripper_path}/seal_{i}'
        joint = UsdPhysics.Joint.Define(stage, path)
        prim = joint.GetPrim()
        prim.AddAppliedSchema('IsaacAttachmentPointAPI')
        prim.CreateAttribute('isaac:forwardAxis', Sdf.ValueTypeNames.Token, False).Set('Z')
        prim.CreateAttribute('isaac:clearanceOffset', Sdf.ValueTypeNames.Float, False).Set(0.)
        sample = np.array([x * (tool.panel_size[0]/2 - .015),
                           y * (tool.panel_size[1]/2 - .015), 0., 1.])
        pos = body_from_pad @ sample
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
        joint.CreateBody1Rel().SetTargets([])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pos[:3]))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(rotation.GetImaginary())))
        joint.CreateJointEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateBreakForceAttr(force_limit)
        joint.CreateBreakTorqueAttr(force_limit * .05)
        for axis in ('transX', 'transY', 'transZ', 'rotX', 'rotY', 'rotZ'):
            limit = UsdPhysics.LimitAPI.Apply(prim, axis)
            limit.CreateLowAttr(1.)
            limit.CreateHighAttr(-1.)  # lock; native gripper may tune at latch
        paths.append(Sdf.Path(path))
    grip.CreateRelationship('isaac:attachmentPoints', False).SetTargets(paths)
    return gripper_path


class NativePanelContact:
    """Native contact/latch feedback only; zero fallback welds or object poses."""
    def __init__(self, stage, gripper_path):
        import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper
        self.stage, self.path = stage, gripper_path
        self.api = surface_gripper.acquire_surface_gripper_interface()
        self.api.set_write_to_usd(True)
        self.closed_status = surface_gripper.GripperStatus.Closed
        prim = stage.GetPrimAtPath(gripper_path)
        self.points = list(prim.GetRelationship('isaac:attachmentPoints').GetTargets())
        if len(self.points) != 4:
            raise ValueError('CAD panel requires all four authored seal samples')
        self.enabled = False

    def command(self, enabled):
        self.enabled = bool(enabled)
        if self.enabled:
            self.api.close_gripper(self.path)
        else:
            self.api.open_gripper(self.path)

    def read(self):
        objects = set(map(str, self.api.get_gripped_objects(self.path) or []))
        complete = self.api.get_gripper_status(self.path) == self.closed_status and len(objects) == 1
        for path in self.points:
            joint = self.stage.GetPrimAtPath(path)
            attached = set(map(str, joint.GetRelationship('physics:body1').GetTargets()))
            complete = complete and attached == objects and bool(joint.GetAttribute('physics:jointEnabled').Get())
        # A partial native seal is explicitly released, never reported as held.
        if objects and not complete:
            self.api.open_gripper(self.path)
        return bool(complete)


def prepare_cad_gripper(stage, *, robot_root, urdf_path):
    """Resolve actual imported link paths and author only under that robot root."""
    from pxr import Usd, UsdPhysics
    root = stage.GetPrimAtPath(robot_root)
    if not root:
        raise ValueError('imported mobile robot root missing')
    named = {}
    for prim in Usd.PrimRange(root):
        if prim.GetName() in ('ur_tool0', 'suction_cups'):
            if prim.GetName() in named:
                raise ValueError('ambiguous CAD link names under robot root')
            named[prim.GetName()] = prim
    if set(named) != {'ur_tool0', 'suction_cups'}:
        raise ValueError('import did not preserve CAD tool0/panel frames')
    body = named['suction_cups']
    while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
        body = body.GetParent()
    if not body or not body.GetPath().HasPrefix(root.GetPath()):
        raise ValueError('panel is not carried by an imported robot rigid body')
    tool = inspect_tool(urdf_path)
    path = str(root.GetPath()) + '/RestockSurfaceGripper'
    author_native_gripper(stage, tool=tool, tool0_path=str(named['ur_tool0'].GetPath()),
                          body_path=str(body.GetPath()), gripper_path=path)
    return path
