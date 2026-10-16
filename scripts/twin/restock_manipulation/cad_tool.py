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


# Physical suction reach of the real CAD cup (mm-class); the real rig seals within this. UNTOUCHED.
REAL_MAX_GRIP_DISTANCE = .005
# SIM FIDELITY RELAXATION (the real value above is untouched): the simulated arm cannot settle the
# pad within 5 mm under gravity (soft drives sag ~0.5 rad; see data/surface-gripper-latch-v1), so the
# NATIVE sim suction gate is widened to this so a residual approach gap (~1-2 cm) and a few degrees of
# pad tilt still latch. Native-sim-only: the physical approach clearance (validate_profile
# tcp_surface_offset 2.5-3 mm, panel size, envelope) is unchanged. Value proven by the isolated GPU
# latch probe (largest window that still seals reliably on a flat approach + the tilt it tolerates).
SIM_MAX_GRIP_DISTANCE = .020


def author_native_gripper(stage, *, tool, tool0_path, body_path, gripper_path,
                           max_gap=SIM_MAX_GRIP_DISTANCE, force_limit=30.,
                           clearance_offset=None, retry_interval=1e6, stroke_limit=None):
    """Author BEFORE physics starts, on the task's composed output stage only.

    body_path must name the actual rigid body carrying the CAD panel (importers
    may merge fixed links). tool0_path supplies that link's robot-frame transform.
    Native runtime owns the gripped body; the seals start with an EMPTY body1 and
    the plugin records the contact on the gripper's grippedObjects relationship.
    Four sample points lie inside the actual panel. Missing any seal is a miss.

    Each seal is authored exactly the way Isaac's installed surface-gripper plugin
    (isaacsim.robot.surface_gripper) and its own authored example require, verified
    natively (data/surface-gripper-latch-v1): an ENABLED maximal-coordinate D6 joint
    (the plugin never creates or enables the joint itself -- it only fetches the
    already-instantiated runtime PhysX joint; a disabled joint is never instantiated,
    which is why the gripper logged "has no joint at attachment point" and never
    latched). body0 is the panel; body1 is empty (world) with both local poses
    authored; the shear/tilt axes are locked and the approach axis (local +Z, the
    forward/suction axis) is a 0..max_gap compliant stroke with a drive; a positive
    clearanceOffset places the suction plane off the panel face.

    max_gap is the native sim suction gate (maxGripDistance). It defaults to
    SIM_MAX_GRIP_DISTANCE -- a sim fidelity relaxation of the real mm-class cup reach so
    the simulated arm, which cannot settle the pad within 5 mm under gravity, still
    latches a residual approach gap. Crucially the gap only seals if the compliant
    suction-axis stroke AND the cup standoff (clearanceOffset) scale WITH the gate
    (stroke_limit = .8*max_gap, clearance_offset = .5*max_gap by default); widening
    maxGripDistance alone does NOT seal -- proven natively. Native-sim-only; the physical
    approach clearance (validate_profile) is unchanged. The latch distance and tilt
    tolerance at this gate are proven by the isolated GPU latch probe
    (data/surface-gripper-latch-v1).

    retryInterval is the window the plugin keeps *attempting* the seal after a single
    close command before giving up to Open. The mission enables suction ONCE at the
    start of the contact descent (controller.py `contact` phase) and expects the
    gripper to latch the instant its seals reach the carton; that descent is slow
    (time-dilation 0.05 -- run-145's prior collect move alone took ~185 s), so a short
    window (retryInterval 1 s) expires long before the seals arrive and the gripper is
    never re-commanded -> no latch (the run-145 in-mission failure). The default is
    therefore large: keep attempting across the whole descent and latch when the seals
    enter range; the mission stage watchdog bounds the overall attempt, and `release`
    (open_gripper) ends it. Verified natively (data/surface-gripper-latch-v1): with the
    panel enabled far (15 mm) and descended slowly, retryInterval 1 never latches but a
    large retryInterval latches at ~4.9 mm as the seals reach the 5 mm window.
    (retryInterval 0 abandons the close in the same step.)
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    # The sealable gap scales only if the compliant stroke AND the cup standoff scale WITH the
    # detection gate: widening maxGripDistance alone does NOT seal (proven natively,
    # data/surface-gripper-latch-v1). The ratios below are the proven window (gate 20 -> stroke 16,
    # clearance 10 seals at 19.9 mm; gate 10 -> 8/5 at 9.9 mm). Callers may override per seal.
    if stroke_limit is None:
        stroke_limit = .8 * max_gap
    if clearance_offset is None:
        clearance_offset = .5 * max_gap
    if not 0 < max_gap <= SIM_MAX_GRIP_DISTANCE or not math.isfinite(force_limit) or force_limit <= 0:
        raise ValueError('invalid seal distance/force')
    if not 0 < clearance_offset <= max_gap or not math.isfinite(retry_interval) or retry_interval <= 0:
        raise ValueError('invalid seal clearance/retry')
    if not 0 < stroke_limit <= SIM_MAX_GRIP_DISTANCE:
        raise ValueError('invalid seal stroke')
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
    # body1 is empty (world), so each seal's localPos1/localRot1 is authored in world space
    # (body0_world * seal_in_body0) so the enabled joint rests exactly satisfied at spawn.
    body_world = cache.GetLocalToWorldTransform(body)
    grip = stage.DefinePrim(gripper_path, 'IsaacSurfaceGripper')
    for name, value in (('maxGripDistance', max_gap), ('coaxialForceLimit', force_limit),
                        ('shearForceLimit', force_limit), ('retryInterval', retry_interval)):
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
        prim.CreateAttribute('isaac:clearanceOffset', Sdf.ValueTypeNames.Float, False).Set(clearance_offset)
        sample = np.array([x * (tool.panel_size[0]/2 - .015),
                           y * (tool.panel_size[1]/2 - .015), 0., 1.])
        pos = body_from_pad @ sample
        local0 = Gf.Matrix4d(1.)
        local0.SetRotateOnly(rotation)
        local0.SetTranslateOnly(Gf.Vec3d(*pos[:3]))
        world = local0 * body_world  # seal pose in world == body1 (empty) frame
        rot1 = world.ExtractRotation().GetQuat()
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
        joint.CreateBody1Rel().SetTargets([])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pos[:3]))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(rotation.GetImaginary())))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*world.ExtractTranslation()))
        joint.CreateLocalRot1Attr(Gf.Quatf(float(rot1.GetReal()), Gf.Vec3f(rot1.GetImaginary())))
        joint.CreateJointEnabledAttr(True)
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateBreakForceAttr(force_limit)
        joint.CreateBreakTorqueAttr(force_limit * .05)
        # Lock shear/tilt; the forward/suction axis (+Z) is a compliant stroke with a drive (below)
        # so the plugin can seal onto an object inside max_gap (an all-axes weld never grips). The
        # plugin seals only on a near-perpendicular approach (~1-2 deg); compliant/driven rotation
        # does NOT widen that (proven natively, data/surface-gripper-latch-v1), so rotation stays
        # locked and the arm must present the pad roughly flat.
        for axis in ('transX', 'transY', 'rotX', 'rotY', 'rotZ'):
            limit = UsdPhysics.LimitAPI.Apply(prim, axis)
            limit.CreateLowAttr(1.)
            limit.CreateHighAttr(-1.)  # low > high == locked
        stroke = UsdPhysics.LimitAPI.Apply(prim, 'transZ')
        stroke.CreateLowAttr(0.)
        stroke.CreateHighAttr(stroke_limit)
        drive = UsdPhysics.DriveAPI.Apply(prim, 'transZ')
        drive.CreateStiffnessAttr(5000.)
        drive.CreateDampingAttr(100.)
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
        # The installed surface-gripper plugin reports a latch through its OWN signals -- the
        # gripper status and the gripper's gripped-objects list -- and manages the PhysX joint
        # directly; it never writes a per-seal body1 back to USD (verified natively,
        # data/surface-gripper-latch-v1). The latch is therefore confirmed as Closed holding
        # exactly one object. A non-complete close that still reports an object (a partial grab)
        # is explicitly released, never reported as held -- no fallback weld or object pose.
        objects = set(map(str, self.api.get_gripped_objects(self.path) or []))
        complete = self.api.get_gripper_status(self.path) == self.closed_status and len(objects) == 1
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
