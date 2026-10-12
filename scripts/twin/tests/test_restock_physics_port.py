from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.restock.isaac_port import body_measurement, MissionCamera
from twin.restock.native_transport import ArmCommandGate
from twin.restock_manipulation.cad_tool import ARM_JOINTS


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
