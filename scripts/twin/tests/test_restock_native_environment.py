from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.restock.native_environment import ros_environment


def test_native_ros_environment_separates_host_python_and_deduplicates_alias(tmp_path):
    library=tmp_path/'lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib'
    library.mkdir(parents=True)
    for name in ('librcl.so','librmw_fastrtps_cpp.so'):
        (library/name).touch()
    (tmp_path/'lib/python3.1').symlink_to(tmp_path/'lib/python3.11',target_is_directory=True)
    original=dict(LD_LIBRARY_PATH='/task/matching-gpu',PYTHONPATH='/host/cp312',AMENT_PREFIX_PATH='/host/ros',ROS_DOMAIN_ID='88')
    result=ros_environment(original,tmp_path/'bin/python')
    assert result['LD_LIBRARY_PATH']=='/task/matching-gpu:'+str(library)
    assert result['ROS_DISTRO']=='jazzy' and result['RMW_IMPLEMENTATION']=='rmw_fastrtps_cpp'
    assert 'PYTHONPATH' not in result and 'AMENT_PREFIX_PATH' not in result
    assert result['ROS_DOMAIN_ID']=='88' and original['PYTHONPATH']=='/host/cp312'
    assert ros_environment(result,tmp_path/'bin/python')['LD_LIBRARY_PATH']==result['LD_LIBRARY_PATH']


def test_missing_native_ros_library_refuses_before_boot(tmp_path):
    with pytest.raises(ValueError,match='bundled Jazzy'):
        ros_environment({},tmp_path/'bin/python')
