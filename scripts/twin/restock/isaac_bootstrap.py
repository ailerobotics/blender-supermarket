"""Isaac 5.1 runtime-only helpers. Call only inside the guarded SimulationApp.

API inspected against installed 5.1 and vendor importer documentation; execution
remains unverified until the GPU driver stack is coherent. No function boots Kit.
"""
from pathlib import Path
import xml.etree.ElementTree as ET


def resolve_meshes(urdf, destination, package_roots):
    """Expand every CAD mesh URI into a new task-owned import file, CPU only.

    Isaac must not silently omit unresolved package meshes. Keep upstream files
    read-only and retain the actual CAD geometry; no mesh substitution is allowed.
    """
    from twin.urdf_probe import resolve_package_uri
    urdf,destination=Path(urdf).resolve(),Path(destination).resolve()
    if destination.exists() or destination==urdf:
        raise ValueError('resolved URDF output must be new')
    tree=ET.parse(urdf)
    for mesh in tree.getroot().findall('.//mesh'):
        filename=mesh.get('filename','')
        if filename.startswith('package://'):
            resolved,_=resolve_package_uri(filename,package_roots)
            path=Path(resolved) if resolved else None
        else:
            path=Path(filename.removeprefix('file://'))
            if not path.is_absolute():
                path=urdf.parent/path
        if path is None or not path.is_file():
            raise ValueError('unresolved actual CAD mesh: '+filename)
        mesh.set('filename',str(path.resolve()))
    destination.parent.mkdir(parents=True,exist_ok=True)
    tree.write(destination,encoding='utf-8',xml_declaration=True)
    return destination


def import_mobile(urdf, destination):
    """Import the generated free-base robot, preserving actual tool/sensor frames."""
    urdf,destination=Path(urdf).resolve(),Path(destination).resolve()
    if destination.exists():
        raise ValueError('robot USD destination must be new')
    root=ET.parse(urdf).getroot()
    links={link.get('name') for link in root.findall('link')}
    if not {'base_link','left_wheel','right_wheel','gripper_pad_tcp','gripper_tcp'} <= links:
        raise ValueError('generated mobile URDF with actual gripper frames required')
    if any(root.findall(tag) for tag in ('ros2_control','gazebo','transmission')):
        raise ValueError('hardware/plugin blocks are forbidden in simulation asset')
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension('isaacsim.asset.importer.urdf')
    ok,config=omni.kit.commands.execute('URDFCreateImportConfig')
    if not ok:
        raise RuntimeError('URDF import configuration failed')
    config.set_fix_base(False)
    config.set_merge_fixed_joints(False)
    config.set_self_collision(True)
    config.set_make_default_prim(True)
    config.set_import_inertia_tensor(True)
    config.set_distance_scale(1.)
    config.set_density(0.)
    destination.parent.mkdir(parents=True,exist_ok=True)
    ok,path=omni.kit.commands.execute('URDFParseAndImportFile',urdf_path=str(urdf),
                                    import_config=config,dest_path=str(destination))
    if not ok or not destination.is_file():
        raise RuntimeError('mobile URDF import failed')
    return destination,path


def discover(stage,robot_root='/MobileRobot'):
    """Discover actual imported paths; ambiguous or missing matches fail closed."""
    from pxr import Usd,UsdPhysics
    root=stage.GetPrimAtPath(robot_root)
    if not root:
        raise ValueError('mobile root absent')
    names=('base_link','ur_base_link','ur_tool0','left_wheel_joint','right_wheel_joint','lidar_frame',
           'front_camera_frame','imu_frame','gripper_pad_tcp','gripper_tcp','suction_cups')
    found={name:[] for name in names}
    articulations=[]
    for prim in Usd.PrimRange(root):
        if prim.GetName() in found:
            found[prim.GetName()].append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulations.append(str(prim.GetPath()))
    if len(articulations)!=1 or any(len(paths)!=1 for paths in found.values()):
        raise ValueError(f'ambiguous/missing imported frames: {found}; articulations={articulations}')
    return {key:paths[0] for key,paths in found.items()} | {'articulation':articulations[0]}
