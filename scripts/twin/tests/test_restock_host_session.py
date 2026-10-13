import os
import sys

import pytest

from twin.restock.host_session import host_environment,validate_host_inputs
from twin.restock.station import OwnedProcess


def test_setup_paths_are_literal_and_environment_is_isolated(tmp_path):
    setup=tmp_path/'setup $(touch injected).bash'
    setup.write_text('export HOST_OVERLAY_READY=yes\nexport PYTHONPATH=/host/messages\n')
    env=host_environment([setup],{'PATH':os.defpath,'PYTHONPATH':'/isaac/python',
                                 'PYTHONHOME':'/isaac','BASH_ENV':'/unrequested/script'})
    assert env['HOST_OVERLAY_READY']=='yes'
    assert env['PYTHONPATH']=='/host/messages'
    assert env['ROS_DOMAIN_ID']=='88'
    assert env['ROS_LOCALHOST_ONLY']=='1'
    assert 'PYTHONHOME' not in env and 'BASH_ENV' not in env
    assert not (tmp_path/'injected').exists()


def test_setup_cannot_switch_to_hardware_domain(tmp_path):
    setup=tmp_path/'setup.bash'
    setup.write_text('export ROS_DOMAIN_ID=42\n')
    with pytest.raises(ValueError,match='ROS_DOMAIN_ID'):
        host_environment([setup],{'PATH':os.defpath})


def test_host_input_validation_does_not_source_scripts(tmp_path):
    setup=tmp_path/'setup.bash'
    setup.write_text('exit 99\n')
    config={'host_python':sys.executable,'host_setup_files':[str(setup)]}
    assert validate_host_inputs(config)[1]==[setup]
    with pytest.raises(ValueError,match='absolute executable'):
        validate_host_inputs(dict(config,host_python='python3'))
    with pytest.raises(ValueError,match='file list'):
        validate_host_inputs(dict(config,host_setup_files=str(setup)))


def test_owned_process_shutdown_preserves_unrelated_child(tmp_path):
    command=[sys.executable,'-c','import time; time.sleep(30)']
    first=OwnedProcess(command,tmp_path/'first.log',env=os.environ)
    other=OwnedProcess(command,tmp_path/'other.log',env=os.environ)
    try:
        first.start()
        other.start()
        assert first.alive and other.alive
        first.close()
        assert not first.alive and other.alive
        first.close()
    finally:
        first.close()
        other.close()
