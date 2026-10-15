import math

import pytest

from twin.restock_manipulation.ros_adapter import JOINTS,planner_velocity_scale


def write_robot(path,rate):
    path.write_text('<robot>'+''.join(
        f'<joint name="{name}"><limit velocity="{rate}"/></joint>' for name in JOINTS)+'</robot>')


def test_ur10_default_five_percent_exceeds_gate_but_derived_scale_does_not(tmp_path):
    path=tmp_path/'mobile.urdf';write_robot(path,math.pi)
    assert math.pi*.05>.15
    scale=planner_velocity_scale(path)
    assert scale*math.pi==pytest.approx(.135)


def test_sim_profile_raises_command_limit_and_scale_cap(tmp_path):
    # The SIM execution profile passes a single raised arm_max_speed as BOTH command_limit and
    # scale_cap, so the planner commands up to ~0.9*command_limit (the extra 5% scale ceiling no
    # longer binds): 0.3 rad/s -> derived scale .9*.3/pi, i.e. ~0.27 rad/s executed.
    path=tmp_path/'mobile.urdf';write_robot(path,math.pi)
    scale=planner_velocity_scale(path,command_limit=.3,scale_cap=.3)
    assert scale*math.pi==pytest.approx(.27)
    assert scale*math.pi>.135   # strictly faster than the conservative default profile


def test_scale_cap_still_binds_below_command_limit(tmp_path):
    # scale_cap remains an independent ceiling: a huge command_limit cannot exceed it.
    path=tmp_path/'mobile.urdf';write_robot(path,math.pi)
    assert planner_velocity_scale(path,command_limit=100.,scale_cap=.05)==pytest.approx(.05)


@pytest.mark.parametrize('rate',[0,-1,float('nan')])
def test_missing_or_invalid_cad_rates_refuse(tmp_path,rate):
    path=tmp_path/'mobile.urdf';write_robot(path,rate)
    with pytest.raises(ValueError,match='velocity limits'):
        planner_velocity_scale(path)


@pytest.mark.parametrize('scale_cap',[0,-1,float('nan')])
def test_invalid_scale_cap_refuses(tmp_path,scale_cap):
    path=tmp_path/'mobile.urdf';write_robot(path,math.pi)
    with pytest.raises(ValueError,match='velocity limits'):
        planner_velocity_scale(path,scale_cap=scale_cap)
