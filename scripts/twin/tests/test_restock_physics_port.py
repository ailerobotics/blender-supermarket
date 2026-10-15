from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.restock.isaac_port import body_measurement, MissionCamera, IsaacPort, SENSING_ACTIONS
from twin.restock.native_transport import ArmCommandGate
from twin.restock_manipulation.cad_tool import ARM_JOINTS


def test_cart_solver_tuning_preserves_physical_bodies_and_free_joints():
    Usd=pytest.importorskip('pxr.Usd')
    from pxr import UsdPhysics
    from twin.mobile_assets import author_cart
    from twin.restock.physics_setup import configure_cart_solver
    stage=Usd.Stage.CreateInMemory()
    cart=author_cart(stage)
    before={(str(p.GetPath()),a.GetName()):a.Get() for p in stage.Traverse()
            for a in p.GetAttributes() if a.HasAuthoredValueOpinion()}
    joints={str(p.GetPath()):list(UsdPhysics.Joint(p).GetBody1Rel().GetTargets())
            for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    result=configure_cart_solver(stage,cart['root'])
    assert len(result)==9
    for (path,name),value in before.items():
        assert stage.GetPrimAtPath(path).GetAttribute(name).Get()==value
    for path,targets in joints.items():
        joint=UsdPhysics.Joint(stage.GetPrimAtPath(path))
        assert list(joint.GetBody1Rel().GetTargets())==targets and joint.GetJointEnabledAttr().Get()
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get()
                   for p in stage.Traverse())


def test_cart_bearings_add_damping_without_touching_physical_model():
    Usd=pytest.importorskip('pxr.Usd')
    from twin.mobile_assets import author_cart
    from twin.restock.physics_setup import configure_cart_bearings, LINEAR_DAMPING, ANGULAR_DAMPING
    stage=Usd.Stage.CreateInMemory()
    cart=author_cart(stage)
    # Snapshot everything the fidelity correction must NOT change: mass, inertia,
    # collision geometry, joint bodies/axes and the free (undriven) revolute model.
    protected={(str(p.GetPath()),a.GetName()):a.Get() for p in stage.Traverse()
               for a in p.GetAttributes() if a.HasAuthoredValueOpinion()}
    result=configure_cart_bearings(stage,cart['root'])
    assert len(result['joints'])==8 and len(result['bodies'])==9
    for (path,name),value in protected.items():
        assert stage.GetPrimAtPath(path).GetAttribute(name).Get()==value
    # Casters stay undriven maximal joints (jointFriction is inert there, so unset).
    for path in result['joints']:
        j=stage.GetPrimAtPath(path)
        assert not j.GetAttribute('physxJoint:jointFriction').HasAuthoredValueOpinion()
        assert j.GetAttribute('physics:excludeFromArticulation').Get() is True
    # Bodies carry bearing/rolling resistance as plain damping (no drive).
    for path in result['bodies']:
        b=stage.GetPrimAtPath(path)
        assert b.GetAttribute('physxRigidBody:linearDamping').Get()==pytest.approx(LINEAR_DAMPING)
        assert b.GetAttribute('physxRigidBody:angularDamping').Get()==pytest.approx(ANGULAR_DAMPING)


def test_wheel_traction_binds_drive_wheel_colliders_despite_instance_proxy():
    """Reproduce the Isaac URDF-import layout that bound 0 colliders: the drive wheels are
    siblings of the articulation-root base_link, and each wheel's collider is referenced under
    <wheel>/collisions with instanceable=True (an instance proxy a naive PrimRange skips and
    cannot bind). The fixed lookup must bind exactly the two drive-wheel colliders."""
    Usd=pytest.importorskip('pxr.Usd')
    from pxr import UsdGeom, UsdPhysics, UsdShade
    from twin.restock.physics_setup import configure_wheel_traction
    stage=Usd.Stage.CreateInMemory()
    # Collider source geometry the importer references per link.
    src=UsdGeom.Xform.Define(stage,'/_colliders/wheel')
    cyl=UsdGeom.Cylinder.Define(stage,'/_colliders/wheel/mesh_0/cylinder')
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    robot=UsdGeom.Xform.Define(stage,'/MobileRobot').GetPrim()
    base=UsdGeom.Xform.Define(stage,'/MobileRobot/base_link').GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(base)
    # Drive wheels and a non-drive caster, all siblings of base_link under the robot root.
    for name in ('left_wheel','right_wheel','rear_caster_wheel'):
        wheel=UsdGeom.Xform.Define(stage,f'/MobileRobot/{name}').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(wheel)
        coll=UsdGeom.Xform.Define(stage,f'/MobileRobot/{name}/collisions').GetPrim()
        coll.GetReferences().AddInternalReference('/_colliders/wheel')
        coll.SetInstanceable(True)
        assert coll.IsInstanceable()
    bound=configure_wheel_traction(stage,'/MobileRobot/base_link',1.0)
    # Exactly the two drive-wheel colliders, not the caster.
    assert sorted(bound)==[
        '/MobileRobot/left_wheel/collisions/mesh_0/cylinder',
        '/MobileRobot/right_wheel/collisions/mesh_0/cylinder',
    ]
    mat=stage.GetPrimAtPath('/RestockWheelMaterial')
    api=UsdPhysics.MaterialAPI(mat)
    assert api.GetStaticFrictionAttr().Get()==pytest.approx(1.0)
    assert api.GetDynamicFrictionAttr().Get()==pytest.approx(0.9)
    for path in bound:
        rel=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).GetDirectBindingRel(materialPurpose='physics')
        assert [str(t) for t in rel.GetTargets()]==['/RestockWheelMaterial']


def test_scene_physics_enables_stabilization_only():
    Usd=pytest.importorskip('pxr.Usd')
    from pxr import UsdPhysics, Gf
    from twin.restock.physics_setup import configure_scene_physics
    stage=Usd.Stage.CreateInMemory()
    scene=UsdPhysics.Scene.Define(stage,'/World/physicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    scene.CreateGravityMagnitudeAttr(9.81)
    result=configure_scene_physics(stage,'/World/physicsScene')
    assert result['enable_stabilization'] is True
    prim=stage.GetPrimAtPath('/World/physicsScene')
    assert prim.GetAttribute('physxScene:enableStabilization').Get() is True
    # Gravity (the physical model) is untouched.
    assert list(prim.GetAttribute('physics:gravityDirection').Get())==[0,0,-1]
    assert prim.GetAttribute('physics:gravityMagnitude').Get()==pytest.approx(9.81)


def test_body_measurement_uses_isaac_wxyz_and_rejects_nonfinite_feedback():
    body=NS(get_world_pose=lambda:(np.array([2,3,.24]),np.array([2**-.5,0,0,2**-.5])),
            get_linear_velocity=lambda:np.zeros(3),get_angular_velocity=lambda:np.zeros(3))
    pose,z,_,_=body_measurement(body)
    assert pose.yaw==pytest.approx(np.pi/2) and z==.24
    body.get_linear_velocity=lambda:np.array([np.nan,0,0])
    with pytest.raises(ValueError,match='measurement'):
        body_measurement(body)


def test_mission_camera_preserves_pixels_id_and_original_frame_clock():
    frame=dict(rendering_time=8.5,rendering_frame=23,rgb=np.zeros((2,2,3)))
    camera=MissionCamera(NS(get_current_frame=lambda:frame),8.)
    measured=camera.get_current_frame()
    assert measured['rendering_time']==.5 and measured['rendering_frame']==23
    assert measured['rgb'] is frame['rgb'] and frame['rendering_time']==8.5


def _plan_port(*, action, stride=2, recorder=True, camera=True):
    port = IsaacPort.__new__(IsaacPort)   # skip Isaac wiring; only the render-gating state
    port.current_action = action
    port.record_stride = stride
    port.recorder = object() if recorder else None
    port.recording_camera = object() if camera else None
    return port


def test_navigation_stage_renders_only_on_recorder_cadence():
    port = _plan_port(action='couple', stride=3)   # couple is navigation-only
    plans = [port._frame_plan(t) for t in range(1, 7)]
    sensing = [s for s, _ in plans]
    record = [r for _, r in plans]
    assert sensing == [False] * 6                   # no sensing render in a nav stage
    assert record == [False, False, True, False, False, True]   # only every 3rd tick


def test_sensing_stage_renders_every_tick():
    for action in sorted(SENSING_ACTIONS):
        port = _plan_port(action=action, stride=100)
        assert all(port._frame_plan(t)[0] for t in range(1, 5))   # sensing True regardless of stride


def test_no_recorder_means_no_record_render():
    port = _plan_port(action='return', stride=1, recorder=False)
    assert all(not r for _, r in (port._frame_plan(t) for t in range(1, 5)))


def test_begin_action_tracks_current_action_for_gating():
    port = IsaacPort.__new__(IsaacPort)
    port.on_action = None
    port.current_action = None
    port.begin_action('collect', job={})
    assert port.current_action == 'collect'


def test_invalid_record_stride_rejected():
    import pytest as _pt
    with _pt.raises(ValueError, match='record_stride'):
        IsaacPort(world=None, robot=NS(dof_names=('left_wheel_joint', 'right_wheel_joint')),
                  robot_body=None, cart_body=None, mobile=None, range_sensor=None,
                  transport=None, contact=None, hold_arm=None, verification=None,
                  record_stride=0)


def test_port_initializes_mission_clock_origin():
    # `origin` must exist so the verify boundary (on_action -> port.mission_time) never raises
    # AttributeError -- the former `port.origin` read assumed an attribute that was never set.
    port = IsaacPort(world=None, robot=NS(dof_names=('left_wheel_joint', 'right_wheel_joint')),
                     robot_body=None, cart_body=None, mobile=None, range_sensor=None,
                     transport=None, contact=None, hold_arm=None, verification=lambda now: None)
    assert port.origin == 0.0


def test_mission_time_recovers_packet_clock_from_stage_time_origin():
    # mission_time() converts Isaac stage time back to the mission clock (tick*dt) the packets carry
    # via `origin` (stage-time value at mission t=0). It is what on_action('verify') begins the
    # InventoryVerifier on, so its scan-stamp comparisons share the packets' clock.
    port = IsaacPort.__new__(IsaacPort)
    port.world = NS(current_time=12.3456)
    port.origin = 12.3456 - 5.0
    assert port.mission_time() == pytest.approx(5.0)
    # A stage clock that cannot be read falls back to -origin (no crash at the boundary).
    port.world = NS()                     # no current_time attribute -> float(...) raises
    assert port.mission_time() == pytest.approx(-(12.3456 - 5.0))


def gate_fixture():
    applied=[]
    robot=NS(get_dof_index=lambda name:ARM_JOINTS.index(name),
             get_joint_positions=lambda **kwargs:np.zeros(6))
    gate=ArmCommandGate(robot,[(-3,3)]*6,lambda q,v,i:applied.append((q.copy(),v.copy())))
    return gate,applied


def test_arm_drive_targets_do_not_accept_old_sequence():
    gate,applied=gate_fixture()
    command=dict(names=list(ARM_JOINTS),positions=[.01]*6,velocities=[.01]*6,
                 efforts=[],stamp=1.,now=1.)
    gate.command(**command)
    gate.command(**command)
    assert len(applied)==1 and not gate.faulted


def test_arm_heartbeat_is_node_clocked_not_wall(monkeypatch):
    # The arm-command heartbeat must be measured in SIM time: a render-paced WALL gap (~3.5 s, far
    # over the old 0.5 s wall bound that emergency-stopped the executed grasp, run-91) is NOT a lost
    # heartbeat while the SIM gap is tiny; a real SIM-time gap still is.
    import twin.restock.native_transport as nt
    clock = [1000.0]
    monkeypatch.setattr(nt.time, 'monotonic', lambda: clock[0])
    gate, _ = gate_fixture()
    gate.command(names=list(ARM_JOINTS), positions=[.01]*6, velocities=[.01]*6,
                 efforts=[], stamp=5.0, now=5.0)
    clock[0] = 1003.5                       # 3.5 s WALL later, but only 0.1 s SIM later -> still fresh
    gate.watchdog(now=5.1)
    assert not gate.faulted
    with pytest.raises(RuntimeError, match='heartbeat lost'):
        gate.watchdog(now=5.7)             # 0.7 s SIM gap -> lost heartbeat


def test_arm_command_ignores_the_effort_field():
    # Isaac's ROS2SubscribeJointState fills effort to the joint count even though trajectory_host
    # publishes none (run-89: eff=6 all-zero; run-90: eff=6 non-zero measured torques). Effort is
    # never commanded (apply drives only q,v), so it must be IGNORED -- a valid position+velocity
    # command is applied regardless of the effort field (gating on it stalled the executed grasp).
    gate,applied=gate_fixture()
    gate.command(names=list(ARM_JOINTS),positions=[.01]*6,velocities=[.01]*6,
                 efforts=[0.]*6,stamp=1.,now=1.)
    gate.command(names=list(ARM_JOINTS),positions=[.02]*6,velocities=[.01]*6,
                 efforts=[1.3,-2.,0.,4.,0.,5.],stamp=2.,now=2.)
    assert len(applied)==2 and not gate.faulted


def test_arm_tracking_jump_latches_measured_hold_and_rejects_later_move():
    gate,applied=gate_fixture()
    with pytest.raises(ValueError,match='tracking'):
        gate.command(names=list(ARM_JOINTS),positions=[1.]*6,velocities=[.01]*6,
                     efforts=[],stamp=1.,now=1.)
    assert gate.faulted and np.array_equal(applied[-1][0],np.zeros(6))
    gate.command(names=list(ARM_JOINTS),positions=[.01]*6,velocities=[.01]*6,
                 efforts=[],stamp=2.,now=2.)
    assert np.array_equal(applied[-1][0],np.zeros(6))
