"""CAD/tool and rendered-sensor boundary tests. All dynamics are fixtures here."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.twin.restock_manipulation.cad_tool import CadTool, ARM_JOINTS, TINE_JOINTS, author_native_gripper
from scripts.twin.restock_manipulation.isaac_motion import IsaacCadMotion
from scripts.twin.restock_manipulation.types import ProductProfile, Observation


def tool_fixture():
    pad = np.eye(4)
    pad[:3,3] = [-.000859,-.000827,.272034]
    support = np.eye(4)
    support[:3,3] = [-.000077,-.058637,.203907]
    return CadTool(ARM_JOINTS, ((-6.,6.),)*6, (.260424,.108204,.02558), pad, support, ((0.,.2),)*6)


def test_actual_panel_rejects_small_carton_and_wrong_orientation():
    tool = tool_fixture()
    valid = ProductProfile('carton-300x200', (.3,.2,.1), (1.,0.,0.,0.), tcp_surface_offset=.0025)
    tool.validate_profile(valid)
    for invalid in (replace(valid, dimensions=(.1,.1,.1)),
                    replace(valid, tcp_quaternion=(0.,0.,0.,1.)),
                    replace(valid, tcp_surface_offset=.05)):
        with pytest.raises(ValueError):
            tool.validate_profile(invalid)


def test_native_authoring_never_selects_or_moves_payload():
    Usd = pytest.importorskip('pxr.Usd')
    from pxr import UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, '/Robot/Wrist').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdGeom.Xform.Define(stage, '/Robot/Wrist/Tool0')
    payload = UsdGeom.Cube.Define(stage, '/Payload')
    payload.AddTranslateOp().Set((.5,0.,.3))
    before = stage.GetPrimAtPath('/Payload').GetAttribute('xformOp:translate').Get()
    path = author_native_gripper(stage, tool=tool_fixture(), tool0_path='/Robot/Wrist/Tool0',
                                 body_path='/Robot/Wrist', gripper_path='/Robot/Grip')
    grip = stage.GetPrimAtPath(path)
    points = grip.GetRelationship('isaac:attachmentPoints').GetTargets()
    assert len(points) == 4 and grip.GetTypeName() == 'IsaacSurfaceGripper'
    for p in points:
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(p))
        assert not joint.GetBody1Rel().GetTargets()
        assert joint.GetJointEnabledAttr().Get() is False
        assert joint.GetLocalPos0Attr().Get()[2] == pytest.approx(.272034)
        assert joint.GetExcludeFromArticulationAttr().Get()
    assert stage.GetPrimAtPath('/Payload').GetAttribute('xformOp:translate').Get() == before
    with pytest.raises(ValueError, match='overwrite'):
        author_native_gripper(stage, tool=tool_fixture(), tool0_path='/Robot/Wrist/Tool0',
                              body_path='/Robot/Wrist', gripper_path='/Robot/Grip')


class ArticulationFixture:
    def __init__(self):
        self.q = np.zeros(12)
        self.names = ARM_JOINTS + TINE_JOINTS
    def get_dof_index(self, name):
        return self.names.index(name)
    def get_joint_positions(self, joint_indices):
        return self.q[joint_indices]
    def get_joint_velocities(self, joint_indices):
        return np.zeros(len(joint_indices))
    def get_linear_velocity(self):
        return np.zeros(3)
    def get_angular_velocity(self):
        return np.zeros(3)


class PlannerFixture:
    is_ready = True
    joint_names = ARM_JOINTS
    def __init__(self):
        self.attached = False
        self.fail = False
    def plan_to_pose(self, joints, xyz, quat, **kwargs):
        return NS(success=not self.fail, joint_names=ARM_JOINTS,
                  positions=[joints, np.asarray(joints)+.01], reason='fixture refusal')
    def attach_object_world(self, joints, dims, centre, quat):
        self.attached = True
        self.centre = centre
    def detach_object(self):
        self.attached = False
    def update_world_from_esdf(self, grid, origin):
        self.grid = grid


def motion_fixture():
    robot, planner = ArticulationFixture(), PlannerFixture()
    contact = NS(read=lambda:True)
    motion = IsaacCadMotion(robot, planner, tool_fixture(), contact)
    commands = []
    motion._drive = lambda q:commands.append(np.array(q))
    motion.update_esdf(np.ones((4,4,4)), (0,0,0), 1.)
    return motion, robot, planner, commands


def test_arm_drive_completion_requires_measured_joints():
    motion, robot, planner, commands = motion_fixture()
    motion.begin((.4,0,.3,1,0,0,0), tuple(robot.q[:6]))
    for _ in range(30):
        motion.tick(.02)
    assert commands and motion.poll() == 'succeeded'  # goal within .03 tolerance
    # Larger target cannot succeed from time alone.
    motion.state = 'idle'
    motion.begin((.4,0,.3,1,0,0,0), tuple(robot.q[:6]))
    motion.path[-1] = .1
    for _ in range(60):
        motion.tick(.02)
    assert motion.poll() == 'running'
    robot.q[:6] = .1
    motion.tick(.02)
    assert motion.poll() == 'succeeded'


def test_tines_and_planner_fail_closed():
    motion, robot, planner, commands = motion_fixture()
    robot.q[7] = .02
    with pytest.raises(ValueError, match='retracted'):
        motion.begin((.4,0,.3,1,0,0,0), tuple(robot.q[:6]))
    robot.q[7] = 0
    planner.fail = True
    with pytest.raises(RuntimeError, match='planning failed'):
        motion.begin((.4,0,.3,1,0,0,0), tuple(robot.q[:6]))
    assert not commands


def test_payload_requires_native_contact_and_sensor_pose():
    motion, robot, planner, commands = motion_fixture()
    profile = ProductProfile('carton',(.3,.2,.1),(1,0,0,0), tcp_surface_offset=.0025)
    assert not motion.update_payload(True, profile)
    obs = Observation('image-1','sensor',1.,'carton','cart',(.4,0,.3,0,0,0,1),1.)
    motion.observe_source(obs)
    motion.contact.read = lambda:False
    assert not motion.update_payload(True, profile)
    motion.contact.read = lambda:True
    assert motion.update_payload(True, profile)
    assert planner.attached and planner.centre[2] == pytest.approx(.25)
    assert not motion.update_payload(False, profile)
    motion.contact.read = lambda:False
    assert motion.update_payload(False, profile) and not planner.attached
    with pytest.raises(ValueError, match='measured'):
        motion.observe_source(replace(obs, source='ground_truth'))


def test_unknown_voxels_are_never_free():
    motion, _, planner, _ = motion_fixture()
    motion.update_esdf(np.array([[[-1000.,np.nan,.1]]]), (0,0,0), 2.)
    assert (planner.grid[0,0,:2] < 0).all()
    assert planner.grid[0,0,2] > 0 and motion.collision_time == 2.


def test_rendered_qr_and_depth_are_actual_inputs(monkeypatch):
    cv2 = pytest.importorskip('cv2')
    from scripts.twin.restock_manipulation import isaac_sensors
    from .test_restock_manipulation import FixturePoseBridge
    monkeypatch.setattr(isaac_sensors, 'load_pose_bridge', lambda root:FixturePoseBridge())
    from .test_restock_sensed_views import commissioned, registration, pixels, quad, clearance
    from scripts.twin.restock_camera.sensed import polygon_mask
    tracker,(k,transform,empty,_,_,a,b) = commissioned()
    gray = np.full(empty.shape,255,np.uint8)
    qr = cv2.QRCodeEncoder_create().encode('carton')
    qr = cv2.resize(qr,(150,150),interpolation=cv2.INTER_NEAREST)
    gray[405:555,565:715] = qr
    rgb = np.repeat(gray[:,:,None],3,axis=2)
    depth = empty.copy()
    depth[polygon_mask(pixels(quad([0,0,.3],[.3,.2]),k,transform),k)] = .7
    frame = dict(rgb=rgb, distance_to_image_plane=depth, rendering_time=2., rendering_frame=2)
    camera = NS(get_current_frame=lambda **kw:frame,
                get_intrinsics_matrix=lambda:np.array([[k.fx,0,k.cx],[0,k.fy,k.cy],[0,0,1.]]),
                add_distance_to_image_plane_to_frame=lambda:None)
    def context(frame, now):
        depth = frame['distance_to_image_plane']
        reg = registration(k,transform,depth,a,b,stamp=now,frame_id=frame['rendering_frame'])
        proof = clearance(now) if np.all(depth == empty) else None
        view = tracker.prepare(depth,k,transform,reg,proof,now=now,parked=True)
        return isaac_sensors.CameraContext(transform,now,'park1',
            (isaac_sensors.SlotView.from_support('rollwagen',view,np.ones(depth.shape,bool)),),True)
    profile = ProductProfile('carton',(.3,.2,.1),(1,0,0,0), tcp_surface_offset=.0025)
    clock=[100.]
    sensor = isaac_sensors.IsaacCartonSensors(camera, pose_bridge_root='fixture-only',
        profiles={'carton':profile}, context_provider=context,steady_clock=lambda:clock[0])
    motion = NS(parked=lambda:True, joint_positions=lambda:(0.,)*6,
                collision_time=2., observe_source=lambda o:None)
    packet = sensor.capture(2.,'rollwagen','rollwagen',motion,NS(read=lambda:False))
    assert len(packet.observations) == 1
    assert packet.observations[0].sku == 'carton'
    assert packet.observations[0].pose[2] == pytest.approx(.3)
    assert not packet.target.clear and packet.target.free_dimensions[2] == 0
    assert sensor.sensed_target_bboxes[0]['source'] == 'sensor_pose_registered_catalog_volume'
    # Stride-paced re-presentation of this exact fresh frame (same id/stamp/pixels, still inside the
    # 0.5 s window) is CONSUMED, not refused: the decoded observation is reused and the context
    # provider / tracker is not re-run (so no monotonic-prepare refusal). A genuinely new or stale
    # frame below still takes the full path and the freshness gate still fires (see run 6).
    reused = sensor.capture(2.,'rollwagen','rollwagen',motion,NS(read=lambda:False))
    assert len(reused.observations) == 1 and reused.observations[0].sku == 'carton'
    # Remove identity pixels: no SKU appears from known scene/profile metadata.
    frame['rendering_frame'] = 3
    frame['rendering_time'] = motion.collision_time = 3.
    frame['rgb'] = np.full_like(rgb,255)
    packet = sensor.capture(3.,'rollwagen','rollwagen',motion,NS(read=lambda:False))
    assert not packet.observations and not packet.target.clear
    frame['rendering_frame'] = 4
    frame['rendering_time'] = motion.collision_time = 4.
    frame['distance_to_image_plane'] = empty
    packet = sensor.capture(4.,'rollwagen','rollwagen',motion,NS(read=lambda:False))
    assert packet.target.clear and 'rollwagen' in packet.scanned_locations
    frame['rendering_frame'] = 5
    frame['rendering_time'] = motion.collision_time = 5.
    frame['rgb'] = rgb
    sensor.profiles = {}
    with pytest.raises(ValueError,match='unknown or undecoded'):
        sensor.capture(5.,'rollwagen','rollwagen',motion,NS(read=lambda:False))
    # Native acquisition expires while registration/clearance processing runs,
    # although the caller's simulation `now` remains fixed.
    frame.update(rendering_frame=6,rendering_time=6.,rgb=np.full_like(rgb,255),
                 acquired_steady=100.,steady_domain='fixture-process')
    motion.collision_time=6.
    def delayed_context(frame,now):
        result=context(frame,now)
        clock[0]+=.501
        return result
    sensor.context_provider=delayed_context
    with pytest.raises(ValueError,match='freshness expired'):
        sensor.capture(6.,'rollwagen','rollwagen',motion,NS(read=lambda:False))
    assert sensor.evidence==() and sensor.sensed_target_bboxes==()


def test_native_partial_latch_releases_without_weld_fallback():
    from scripts.twin.restock_manipulation.cad_tool import NativePanelContact
    adapter = NativePanelContact.__new__(NativePanelContact)
    adapter.path, adapter.points, adapter.closed_status = '/Grip', ['a','b','c','d'], 'closed'
    opened = []
    adapter.api = NS(get_gripper_status=lambda p:'closed', get_gripped_objects=lambda p:['/Box'],
                     open_gripper=lambda p:opened.append(p))
    joint = lambda p:NS(GetRelationship=lambda n:NS(GetTargets=lambda:['/Box'] if p != 'd' else []),
                       GetAttribute=lambda n:NS(Get=lambda:True))
    adapter.stage = NS(GetPrimAtPath=joint)
    assert adapter.read() is False and opened == ['/Grip']


def test_new_obstacle_map_stops_rejected_active_plan():
    motion, robot, planner, commands = motion_fixture()
    motion.begin((.4,0,.3,1,0,0,0), tuple(robot.q[:6]))
    planner.fail = True
    with pytest.raises(RuntimeError, match='planning failed'):
        motion.update_esdf(np.full((4,4,4),-.1), (0,0,0), 1.1)
    assert motion.poll() == 'failed' and commands  # encoder hold, no old-path continuation


def test_labelled_carton_has_real_texture_mass_and_collision(tmp_path):
    Usd = pytest.importorskip('pxr.Usd')
    from pxr import UsdPhysics
    from scripts.twin.restock_manipulation.carton_asset import author_carton
    from scripts.twin.restock_manipulation.isaac_sensors import write_carton_label
    profile = ProductProfile('carton',(.3,.2,.1),(1,0,0,0),tcp_surface_offset=.0025)
    texture = tmp_path/'label.png'
    write_carton_label('carton',texture)
    stage = Usd.Stage.CreateInMemory()
    path = author_carton(stage,path='/Carton',profile=profile,label_png=texture,centre=(.4,0,.3))
    assert stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).GetMassAttr().Get() == .5
    assert stage.GetPrimAtPath(path+'/body').HasAPI(UsdPhysics.CollisionAPI)
    assert stage.GetPrimAtPath(path+'/label_material/texture').GetAttribute('inputs:file').Get().path == str(texture)


def test_side_carton_dimensions_and_support_clearance():
    tool = tool_fixture()
    profile = ProductProfile('case',(.6,.4,.16),(.5,.5,.5,.5),
                             tcp_surface_offset=.0025,grasp_mode='side')
    tool.validate_profile(profile)
    assert profile.dimensions[2]/2+tool.envelope_pad_min[1] > .01
    with pytest.raises(ValueError, match='tines'):
        tool.validate_profile(replace(profile,dimensions=(.6,.4,.14)))


def test_wall_policy_uses_current_station_poses_and_rejects_oversized_case():
    from scripts.twin.restock_manipulation.wall_policy import WallCartonPolicy
    from scripts.twin.restock_manipulation.types import Target
    # Explicit primitive fixture. Actual upstream functions are exercised by the
    # separately recorded source-compatibility probe, never silently substituted.
    policy = WallCartonPolicy.__new__(WallCartonPolicy)
    policy.tool,policy.walls,policy.lift = tool_fixture(),{'cart':0,'shelf':0},.03
    policy.geometry = NS(wall_grip=lambda tray,p,wall:
        (-tray.tray_size[0]/2-tray.contact_margin,0,-tray.approach_offset_x_m,0,[.5]*4))
    policy.primitives = NS(approach_contact=lambda x,y,z,rx,ry,q,pad:
        ([x+rx,y+ry,z,*q],[x,y,z,*q]))
    profile = ProductProfile('case',(.6,.4,.16),(.5,.5,.5,.5),
                             tcp_surface_offset=.0025,grasp_mode='side')
    source = Observation('s','test_fixture',1,'case','cart',(.4,0,.36,0,0,0,1),1)
    target = Target('t','test_fixture',1,'shelf',(.5,.4,.2,0,0,0,1),True,(.65,.45,.28))
    plan = policy.build(source,target,profile)
    assert plan['contact'][2] == pytest.approx(.28)
    assert plan['place'][0] == pytest.approx(.5-.3025)
    assert plan['retreat'][2] == plan['place'][2]  # horizontal shelf exit
    assert plan['preplace'][2] == pytest.approx(.31)
    with pytest.raises(ValueError,match='vertical gap'):
        policy.build(source,target,replace(profile,dimensions=(.6,.4,.32)))
    with pytest.raises(ValueError,match='three dimensions'):
        policy.build(source,replace(target,free_dimensions=(.5,.45,.28)),profile)
    assert source.unit_kind == 'handling_unit' and source.retail_quantity is None
