#!/usr/bin/env python3
"""List concrete runtime inputs without a GPU probe, ROS graph or simulator boot."""
import argparse
import hashlib
import json
from pathlib import Path


FILES = {
    'store_usd':'exported store with resolved assets',
    'mobile_manifest':'generated mobile URDF/cart and transport configuration',
    'scenario':'sealed handling-unit and robot/cart scenario',
    'host_config':'domain-88 MoveIt/cuMotion/nvblox launch configuration',
    'srdf':'actual CAD six-joint planning group and collision exclusions',
    'xrdf':'actual CAD collision spheres, limits and attachment configuration',
    'camera_config':'validated physical source/cart/shelf camera mounts and views',
    'ffmpeg':'H.264 encoder executable',
}
DIRECTORIES = {'shelfbot_root':'upstream wall-grasp geometry',
               'pose_bridge_root':'upstream registered-depth pose pipeline',
               'ros_overlay':'compatible host NVIDIA/Jazzy messages and libraries'}
SERVERS = {'/restock_sim/compute_ik':'moveit_msgs/srv/GetPositionIK',
           '/restock_sim/move_action':'moveit_msgs/action/MoveGroup',
           '/restock_sim/execute_trajectory':'moveit_msgs/action/ExecuteTrajectory',
           '/restock_sim/apply_planning_scene':'moveit_msgs/srv/ApplyPlanningScene',
           '/restock_sim/attach_object':'isaac_ros_cumotion_interfaces/action/AttachObject',
           '/restock_sim/nvblox_node/get_esdf_and_gradient':'nvblox_msgs/srv/EsdfAndGradients',
           '/restock_sim/arm_controller/follow_joint_trajectory':'control_msgs/action/FollowJointTrajectory'}


def check(config):
    checks=[]
    for key,description in {**FILES,**DIRECTORIES}.items():
        value=config.get(key)
        path=Path(value).expanduser() if isinstance(value,str) and value else None
        exists=bool(path and (path.is_dir() if key in DIRECTORIES else path.is_file()))
        checks.append(dict(input=key,status='present' if exists else 'missing',
                           description=description,path=str(path) if path else None))
    mobile=config.get('mobile_manifest')
    if isinstance(mobile,str) and Path(mobile).is_file():
        try:
            manifest=json.loads(Path(mobile).read_text())
            for key in ('mobile_urdf','cart_usd'):
                value=manifest.get(key)
                checks.append(dict(input='mobile_manifest.'+key,
                    status='present' if isinstance(value,str) and Path(value).is_file() else 'missing',path=value))
        except (ValueError,OSError) as exc:
            checks.append(dict(input='mobile_manifest_schema',status='invalid',reason=str(exc)))
    root=Path(__file__).resolve().parent
    for name in ('run_restock_host.py','twin/restock_camera/__init__.py',
                 'twin/restock/isaac_bundle.py','twin/restock/native_transport.py',
                 'twin/restock/trajectory_host.py','twin/restock/physics_setup.py'):
        checks.append(dict(input=name,status='present' if (root/name).is_file() else 'missing'))
    hashes={}
    for key in FILES:
        value=config.get(key)
        if isinstance(value,str) and Path(value).is_file():
            hashes[key]=hashlib.sha256(Path(value).read_bytes()).hexdigest()
    assembly=None
    if all(key in hashes for key in ('store_usd','mobile_manifest','scenario')):
        try:
            from twin.restock.physics_setup import inputs
            assembly=inputs(config,json.loads(Path(config['scenario']).read_text()))[-1]
            checks.append(dict(input='physical_assembly_inputs',status='present',
                               evidence='actual CAD limits, declared spawn state and content hashes'))
        except Exception as exc:
            checks.append(dict(input='physical_assembly_inputs',status='invalid',reason=str(exc)))
    missing=[c['input'] for c in checks if c['status']!='present']
    return dict(files_ready=not missing,unresolved_inputs=missing,checks=checks,
        input_file_sha256=hashes,physical_assembly_sha256=assembly,
        required_servers=SERVERS,runtime_verified=False,
        runtime_gates=['explicit GPU release and protected-Blender/resource guard',
                       'domain88 loopback-only service/action discovery',
                       'fresh incorporated-depth acknowledgement after station reset',
                       'measured arm trajectory feedback and collision-aware stow',
                       'depth-derived camera calibration and unobstructed rendered QR views',
                       'actual localization uncertainty and arm/cart clearance',
                       'native seal, dynamic payload, verification and return evidence'],
        evidence='CPU input inventory only; present files do not prove compatible services')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-config',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    config=json.loads(a.runtime_config.read_text()) if a.runtime_config else {}
    result=check(config)
    text=json.dumps(result,indent=2)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(text)
    print(text,end='')
    return 0 if result['files_ready'] else 2


if __name__=='__main__':
    raise SystemExit(main())
