from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.restock.isaac_port import body_measurement, MissionCamera
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


def test_cart_bearings_add_friction_damping_without_touching_physical_model():
    Usd=pytest.importorskip('pxr.Usd')
    from pxr import UsdPhysics
    from twin.mobile_assets import author_cart
    from twin.restock.physics_setup import configure_cart_bearings
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
    # Friction authored on every caster joint; bodies carry mild damping only.
    for path in result['joints']:
        j=stage.GetPrimAtPath(path)
        assert j.GetAttribute('physxJoint:jointFriction').Get()==pytest.approx(0.01)
        assert not UsdPhysics.RevoluteJoint(j).GetBody1Rel().GetTargets()==[]  # still a real joint
    for path in result['bodies']:
        b=stage.GetPrimAtPath(path)
        assert b.GetAttribute('physxRigidBody:linearDamping').Get()==pytest.approx(0.05)
        assert b.GetAttribute('physxRigidBody:angularDamping').Get()==pytest.approx(0.05)


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


def test_arm_tracking_jump_latches_measured_hold_and_rejects_later_move():
    gate,applied=gate_fixture()
    with pytest.raises(ValueError,match='tracking'):
        gate.command(names=list(ARM_JOINTS),positions=[1.]*6,velocities=[.01]*6,
                     efforts=[],stamp=1.,now=1.)
    assert gate.faulted and np.array_equal(applied[-1][0],np.zeros(6))
    gate.command(names=list(ARM_JOINTS),positions=[.01]*6,velocities=[.01]*6,
                 efforts=[],stamp=2.,now=2.)
    assert np.array_equal(applied[-1][0],np.zeros(6))
