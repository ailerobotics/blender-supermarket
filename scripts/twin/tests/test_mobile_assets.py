import math
import xml.etree.ElementTree as ET
import pytest
from twin.mobile import MobileSpec
from twin.mobile_assets import build_mobile_urdf, author_cart, author_hitch
from twin.urdf_probe import parse_urdf, find_roots, link_world_transforms


@pytest.fixture
def source(tmp_path):
    p = tmp_path/'arm.urdf'
    p.write_text('''<robot name="test_fixture">
      <link name="world"/><link name="pedestal_link"/><link name="pedestal_top"/>
      <link name="ur_base_link"/><link name="arm"/><link name="ur_tool0"/>
      <joint name="pedestal_joint" type="fixed"><parent link="world"/><child link="pedestal_link"/></joint>
      <joint name="pedestal_top_joint" type="fixed"><parent link="pedestal_link"/><child link="pedestal_top"/></joint>
      <joint name="ur_base_joint" type="fixed"><parent link="pedestal_top"/><child link="ur_base_link"/></joint>
      <joint name="shoulder" type="revolute"><parent link="ur_base_link"/><child link="arm"/>
      <limit lower="-3" upper="3" effort="100" velocity="1"/></joint>
      <joint name="tool0_joint" type="fixed"><parent link="arm"/><child link="ur_tool0"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/></joint>
      <ros2_control name="fixture"><joint name="shoulder"><state_interface name="position">
      <param name="initial_value">-2.5</param></state_interface></joint></ros2_control>
      </robot>''')
    return p


def test_mobile_urdf_free_root_preserves_arm_and_wheel_geometry(source, tmp_path):
    out = tmp_path/'mobile.urdf'
    target = build_mobile_urdf(source, out)
    assert target == {'shoulder': -2.5}
    robot = parse_urdf(str(out))
    assert find_roots(robot) == ['base_link']
    t = link_world_transforms(robot, target)
    assert len(t) == len(robot.links)
    assert t['ur_base_link'][2, 3] == pytest.approx(.26)
    for side, sign in [('left', 1), ('right', -1)]:
        j = robot.joints_by_name[f'{side}_wheel_joint']
        assert j.axis == (0, 1, 0) and j.jtype == 'continuous'
        assert t[f'{side}_wheel'][1, 3] == pytest.approx(sign*.35)
        assert t[f'{side}_wheel'][2, 3]+.24-.16 == pytest.approx(0)
    for link in robot.links.values():
        if link.inertial:
            i = link.inertial
            assert i.mass > 0 and min(i.ixx, i.iyy, i.izz) > 0
            assert max(i.ixx, i.iyy, i.izz) <= (i.ixx+i.iyy+i.izz)/2+1e-9
    original = ET.parse(source).find("joint[@name='shoulder']")
    generated = ET.parse(out).find("joint[@name='shoulder']")
    assert generated.attrib == original.attrib
    assert generated.find('limit').attrib == original.find('limit').attrib
    assert ET.parse(out).find('ros2_control') is None  # no real-rig driver plugin
    # The eye-in-hand wrist camera frame is a fixed child of the UR tool flange (ur_tool0), not base_link.
    wrist = ET.parse(out).find("joint[@name='restock_camera_wrist_joint']")
    assert wrist is not None and wrist.find('parent').get('link') == 'ur_tool0'
    assert wrist.find('child').get('link') == 'restock_camera_wrist'
    assert 'restock_camera_wrist' in robot.links and robot.joints_by_name['restock_camera_wrist_joint']


@pytest.fixture
def stage():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    s = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(s, 'Z'); UsdGeom.SetStageMetersPerUnit(s, 1)
    return s


def test_cart_physical_bodies_and_joint_anchors(stage):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    paths = author_cart(stage)
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 9
    joints = [UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    assert len(joints) == 8
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for joint in joints:
        b0, b1 = joint.GetBody0Rel().GetTargets()[0], joint.GetBody1Rel().GetTargets()[0]
        a = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(b0)).Transform(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
        b = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(b1)).Transform(Gf.Vec3d(joint.GetLocalPos1Attr().Get()))
        assert (a-b).GetLength() < 1e-7
        assert joint.GetExcludeFromArticulationAttr().Get()
    for body in bodies:
        assert not UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get()
        assert UsdPhysics.MassAPI(body).GetMassAttr().Get() > 0
    assert stage.GetPrimAtPath(paths['dock'])
    assert all(stage.GetPrimAtPath(p).HasAPI(UsdPhysics.CollisionAPI) for p in paths['stock_supports'])


def test_hitch_and_runtime_do_not_change_cart_transforms(stage):
    from pxr import UsdGeom, UsdPhysics
    from twin.mobile import DockObservation, Pose2
    from twin.mobile_runtime import MobileController
    author_cart(stage)
    robot = UsdGeom.Xform.Define(stage, '/Robot/base_link').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(robot)
    wheel_paths = ['/Robot/left', '/Robot/right']
    for path in wheel_paths:
        UsdPhysics.RevoluteJoint.Define(stage, path)
    controller = MobileController(stage, str(robot.GetPath()), '/Rollwagen/chassis', wheel_paths)
    obs = DockObservation(Pose2(0, 0, 0), Pose2(-1.27, 0, 0), .28, .28, 0, 0, 0, 0)
    cart = stage.GetPrimAtPath('/Rollwagen/chassis')
    before = UsdGeom.Xformable(cart).GetLocalTransformation()
    assert controller.attach(obs)['healthy']
    hitch = UsdPhysics.RevoluteJoint.Get(stage, '/MobileHitch')
    assert hitch.GetAxisAttr().Get() == 'Z'
    assert hitch.GetUpperLimitAttr().Get() == pytest.approx(65)
    assert UsdGeom.Xformable(cart).GetLocalTransformation() == before
    targets = controller.command(.16, 0, obs, transport_ready=True)
    assert targets == (1, 1)
    assert controller.drives[0].GetTargetVelocityAttr().Get() == pytest.approx(180/math.pi)
    assert controller.command(.2, 0, None, transport_ready=True) == (0, 0)
    assert controller.drives[0].GetTargetVelocityAttr().Get() == 0
    assert not controller.detach(obs)['joint_present']
    assert UsdGeom.Xformable(cart).GetLocalTransformation() == before
    with pytest.raises(ValueError):
        author_hitch(stage, '/missing', '/Rollwagen/chassis')


def test_base_hold_locks_wheels_and_releases(stage):
    # The sim stationary base hold (task base-hold-under-stiff-arm): hold() locks each wheel at its
    # current angle with a stiff POSITION drive (the parking brake) so the stiff arm's reaction
    # torque cannot creep the base past the stationarity gate; it is released cleanly when driving.
    from pxr import UsdGeom, UsdPhysics
    from twin.mobile import DockObservation, Pose2
    from twin.mobile_runtime import MobileController
    author_cart(stage)
    robot = UsdGeom.Xform.Define(stage, '/Robot/base_link').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(robot)
    wheels = ['/Robot/l', '/Robot/r']
    for p in wheels:
        UsdPhysics.RevoluteJoint.Define(stage, p)
    angles = [0.5, -0.3]   # live wheel angles (radians) the reader returns
    c = MobileController(stage, str(robot.GetPath()), '/Rollwagen/chassis', wheels,
                         wheel_angle_reader=lambda: list(angles))
    d0, d1 = c.drives
    # Free driving default: a zero-stiffness velocity drive.
    assert d0.GetStiffnessAttr().Get() == pytest.approx(0.0)
    assert d0.GetDampingAttr().Get() == pytest.approx(c.DRIVE_DAMPING)
    # hold(): stiff POSITION hold locking each wheel at its current angle (degrees).
    assert c.hold() == (0., 0.) and c._held
    assert d0.GetStiffnessAttr().Get() == pytest.approx(c.HOLD_STIFFNESS) and c.HOLD_STIFFNESS > 0
    assert d0.GetMaxForceAttr().Get() == pytest.approx(c.HOLD_MAX_FORCE)
    assert d0.GetTargetPositionAttr().Get() == pytest.approx(math.degrees(0.5))
    assert d1.GetTargetPositionAttr().Get() == pytest.approx(math.degrees(-0.3))
    assert d0.GetTargetVelocityAttr().Get() == 0.
    # Idempotent: a second hold() does NOT re-capture the angle (the brake stays where it engaged).
    angles[:] = [9., 9.]
    c.hold()
    assert d0.GetTargetPositionAttr().Get() == pytest.approx(math.degrees(0.5))
    # release(): restore the free (zero-stiffness) velocity drive.
    c.release()
    assert not c._held and d0.GetStiffnessAttr().Get() == pytest.approx(0.0)
    assert d0.GetDampingAttr().Get() == pytest.approx(c.DRIVE_DAMPING)
    # command() (resuming driving) auto-releases a prior hold.
    c.hold(); assert c._held
    c.command(.16, 0, DockObservation(Pose2(0, 0, 0), Pose2(-1.27, 0, 0), .28, .28, 0, 0, 0, 0),
              transport_ready=True)
    assert not c._held and d0.GetStiffnessAttr().Get() == pytest.approx(0.0)
    # Without a reader hold() degrades to a plain zero-velocity park (not a position hold).
    c.wheel_angle_reader = None
    assert c.hold() == (0., 0.) and not c._held
    assert d0.GetStiffnessAttr().Get() == pytest.approx(0.0)


def test_sim_parking_brake_locks_chassis_and_releases(stage):
    # The sim parking brake (firstmate run-210): engage_brake welds the chassis to the world with a
    # FixedJoint at its CURRENT pose so a welded-carton reaction torque cannot pivot the yaw-saturated
    # base; released cleanly, and command() (resuming driving) auto-releases it.
    from pxr import UsdGeom, UsdPhysics
    from twin.mobile import DockObservation, Pose2
    from twin.mobile_runtime import MobileController
    author_cart(stage)
    robot = UsdGeom.Xform.Define(stage, '/Robot/base_link').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(robot)
    UsdGeom.Xformable(robot).AddTranslateOp().Set((-5.73, 16.0, 0.24))
    wheels = ['/Robot/l', '/Robot/r']
    for p in wheels:
        UsdPhysics.RevoluteJoint.Define(stage, p)
    c = MobileController(stage, str(robot.GetPath()), '/Rollwagen/chassis', wheels,
                         wheel_angle_reader=lambda: [0.0, 0.0])
    assert not c._braked and not stage.GetPrimAtPath(c.BRAKE_PATH)
    c.engage_brake()
    brake = UsdPhysics.Joint(stage.GetPrimAtPath(c.BRAKE_PATH))
    assert c._braked and brake
    # body1 is the chassis; body0 empty == world; anchored at the chassis's current pose.
    assert [str(t) for t in brake.GetBody1Rel().GetTargets()] == [str(robot.GetPath())]
    assert tuple(brake.GetLocalPos0Attr().Get()) == pytest.approx((-5.73, 16.0, 0.24))
    # ONLY yaw (rotZ) is locked (low>high); translation + roll/pitch stay free (no limit).
    z = UsdPhysics.LimitAPI.Get(brake.GetPrim(), 'rotZ')
    assert z and z.GetLowAttr().Get() > z.GetHighAttr().Get()
    assert not UsdPhysics.LimitAPI.Get(brake.GetPrim(), 'transX').GetLowAttr()
    assert not UsdPhysics.LimitAPI.Get(brake.GetPrim(), 'rotX').GetLowAttr()
    c.engage_brake()  # idempotent
    assert stage.GetPrimAtPath(c.BRAKE_PATH)
    c.release_brake()
    assert not c._braked and not stage.GetPrimAtPath(c.BRAKE_PATH)
    # command() (resuming driving) releases the brake before any wheel command.
    c.engage_brake(); assert c._braked
    c.command(.16, 0, DockObservation(Pose2(0, 0, 0), Pose2(-1.27, 0, 0), .28, .28, 0, 0, 0, 0),
              transport_ready=True)
    assert not c._braked and not stage.GetPrimAtPath(c.BRAKE_PATH)


def test_transport_measurement_and_missing_mesh_fail_closed(source, tmp_path):
    from twin.mobile_assets import measure_transport
    out = tmp_path/'mobile.urdf'
    targets = build_mobile_urdf(source, out)
    evidence = measure_transport(out, targets, [])
    assert evidence['home_collision_radius_m'] < .65
    assert evidence['static_support_margin_m'] > .08
    with pytest.raises(ValueError, match='transport radius'):
        measure_transport(out, targets, [], MobileSpec(transport_radius=.3))
    tree = ET.parse(out)
    link = tree.find("link[@name='arm']")
    geom = ET.SubElement(ET.SubElement(link, 'collision'), 'geometry')
    ET.SubElement(geom, 'mesh', filename='package://missing/part.stl')
    tree.write(out)
    with pytest.raises(ValueError, match='unresolved collision'):
        measure_transport(out, targets, [])


def test_lidar_frame_clears_forward_braking_sweep(source, tmp_path):
    """The 2D nav scanner must sit at/just beyond the chassis front edge, inside the
    transport footprint, so no authored collision geometry occludes the forward
    braking sweep at any configured scan height (navigation.core certifies the forward
    sweep strictly; a recessed scanner self-occludes the low band and nav refuses)."""
    import math
    import numpy as np
    from twin.urdf_probe import geometry_world_points, link_world_transforms
    spec = MobileSpec()
    out = tmp_path / 'mobile.urdf'
    target = build_mobile_urdf(source, out, spec)
    robot = parse_urdf(str(out))
    lx, ly, _ = robot.joints_by_child['lidar_frame'].origin_xyz
    # Mounted at/just beyond the chassis front (length/2) and inside the footprint.
    assert lx >= spec.length / 2
    assert math.hypot(lx, ly) <= spec.transport_radius
    heights = (0.15, 0.5, 1.0)              # IsaacRangeSensor configured scan heights
    tf = link_world_transforms(robot, target)
    occluders = []
    for name, link in robot.links.items():
        for g in link.collisions:
            cloud = geometry_world_points(g, tf[name], lambda uri: uri)
            if cloud is None:
                continue
            p = np.asarray(cloud, float)
            dx, dy = p[:, 0] - lx, p[:, 1] - ly
            # A forward-sweep bearing is any point ahead of the lidar in its forward
            # half-plane; it occludes height z if the box spans z.
            forward = (dx > 1e-6) & (np.abs(np.arctan2(dy, dx)) < math.pi / 2)
            if forward.any():
                zlo, zhi = float(p[:, 2].min()), float(p[:, 2].max())
                for z in heights:
                    if zlo - 1e-9 <= z <= zhi + 1e-9:
                        occluders.append((name, z))
    assert not occluders, f'forward self-occlusion from lidar_frame at: {sorted(set(occluders))}'
