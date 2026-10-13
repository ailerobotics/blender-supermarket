"""Process-local Isaac ROS libraries, separate from the host ROS interpreter."""
from pathlib import Path


def ros_environment(environment, isaac_python):
    """Resolve the bundled Jazzy runtime without importing or booting Isaac.

    This environment must be passed at process creation: changing LD_LIBRARY_PATH
    inside an already-running Python process does not update the ELF loader.
    """
    prefix=Path(isaac_python).absolute().parent.parent
    candidates=sorted({p.resolve() for p in prefix.glob(
        'lib/python*/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib')})
    if len(candidates)!=1:
        raise ValueError('one installed Isaac bundled Jazzy library directory required')
    library=candidates[0].resolve()
    if any(not (library/name).is_file() for name in ('librcl.so','librmw_fastrtps_cpp.so')):
        raise ValueError('incomplete bundled native Jazzy runtime')
    result=dict(environment)
    for key in ('PYTHONPATH','PYTHONHOME','AMENT_PREFIX_PATH','CMAKE_PREFIX_PATH',
                'COLCON_PREFIX_PATH','ROS_PACKAGE_PATH'):
        result.pop(key,None)
    result.update(ROS_DISTRO='jazzy',RMW_IMPLEMENTATION='rmw_fastrtps_cpp')
    paths=[p for p in result.get('LD_LIBRARY_PATH','').split(':') if p]
    result['LD_LIBRARY_PATH']=':'.join(dict.fromkeys([*paths,str(library)]))
    return result
