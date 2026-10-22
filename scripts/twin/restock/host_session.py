"""Launch and close only this mission's host ROS interpreter processes."""
import json
import os
from pathlib import Path
import subprocess

from twin.restock_host.config import isolated_environment
from .station import OwnedProcess


def validate_host_inputs(config):
    """Validate explicit interpreter/setup paths without sourcing or starting ROS."""
    python=Path(config.get('host_python',''))
    if not python.is_absolute() or not python.is_file() or not os.access(python,os.X_OK):
        raise ValueError('absolute executable host ROS Python interpreter required')
    files=config.get('host_setup_files')
    if not isinstance(files,list) or not files or any(not isinstance(p,str) for p in files):
        raise ValueError('explicit host ROS setup file list required')
    if any(not Path(p).is_absolute() or not Path(p).is_file() for p in files):
        raise ValueError('absolute existing host ROS setup files required')
    return python.resolve(),[Path(p).resolve() for p in files]


def host_environment(setup_files, launch_environment):
    """Capture explicit setup-script environment without importing ROS into Isaac.

    Shell paths are positional arguments, not interpolated code. Output remains
    in memory; environment secrets are never printed or saved in reports.
    """
    scripts=[Path(p).resolve() for p in setup_files]
    if not scripts or any(not p.is_file() for p in scripts):
        raise ValueError('existing explicit host ROS setup files required')
    # Seed isolation before setup scripts apply ROS's default SUBNET discovery.
    # Reject an explicitly incompatible launch environment instead of hiding it.
    env=isolated_environment(launch_environment)
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','CONDA_PREFIX','CONDA_DEFAULT_ENV',
                'AMENT_PREFIX_PATH','CMAKE_PREFIX_PATH','COLCON_PREFIX_PATH','BASH_ENV','ENV'):
        env.pop(key,None)
    result=subprocess.run(['/bin/bash','--noprofile','--norc','-c',
        'for setup_file in "$@"; do source "$setup_file" >/dev/null || exit; done; exec /usr/bin/env -0',
        'restock-host-env',*(str(p) for p in scripts)],env=env,stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=10)
    captured={}
    for entry in result.stdout.split(b'\0'):
        if entry:
            key,value=entry.split(b'=',1)
            captured[os.fsdecode(key)]=os.fsdecode(value)
    return isolated_environment(captured)


class HostSession:
    def __init__(self, config, output, manipulation_config):
        self.children=[]
        python,setups=validate_host_inputs(config)
        environment=host_environment(setups,config['_launch_environment'])
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        scripts=Path(__file__).resolve().parents[2]
        environment['PYTHONPATH']=str(scripts)+os.pathsep+environment.get('PYTHONPATH','')
        environment['ROS_LOG_DIR']=str(self.output/'ros-logs')
        host_config=json.loads(Path(config['host_config']).read_text())
        commands={
            'trajectory':[str(python),'-m','twin.restock.trajectory_host','--urdf',host_config['urdf']],
            'station':[str(python),'-m','twin.restock.station_host','--config',config['host_config'],
                       '--output',str(self.output/'stations')],
            'manipulation':[str(python),'-m','twin.restock_manipulation.ros_host',
                '--config',str(manipulation_config),'--urdf',host_config['urdf'],
                '--shelfbot-root',config['shelfbot_root'],'--mobile-manifest',config['mobile_manifest']],
        }
        try:
            for name,command in commands.items():
                child=OwnedProcess(command,self.output/(name+'.log'),env=environment)
                child.start()
                self.children.append(child)
            self.check()
            (self.output/'owned-processes.json').write_text(json.dumps([
                dict(pid=child.process.pid,command=child.command,log=str(child.log))
                for child in self.children],indent=2)+'\n')
        except BaseException:
            self.close()
            raise

    def check(self):
        for child in self.children:
            if not child.alive:
                raise RuntimeError('task-owned host process exited; inspect '+str(child.log))

    def close(self):
        errors=[]
        for child in reversed(self.children):
            try:
                child.close()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('host cleanup: '+'; '.join(errors))
