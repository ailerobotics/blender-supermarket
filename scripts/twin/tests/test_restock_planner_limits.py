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


@pytest.mark.parametrize('rate',[0,-1,float('nan')])
def test_missing_or_invalid_cad_rates_refuse(tmp_path,rate):
    path=tmp_path/'mobile.urdf';write_robot(path,rate)
    with pytest.raises(ValueError,match='velocity limits'):
        planner_velocity_scale(path)
