#!/usr/bin/env python3
"""Actual-URDF CPU IK probe for representative poses; NOT a collision planner."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from twin.urdf_probe import parse_urdf, link_world_transforms


def check(urdf, mobile_manifest, scenario):
    robot=parse_urdf(str(urdf))
    config=dict(mobile_manifest['transport_positions'])
    joints=[j for j in robot.joints if j.name in config and j.jtype=='revolute' and 'ur_' in j.name]
    if len(joints)!=6:
        raise ValueError('six real UR arm joints required')
    unit,staging=scenario['handling_unit'],scenario['initial_staging']
    def local(point,pose):
        delta=np.asarray(point,float)-np.array([pose[0],pose[1],0.])
        return Rotation.from_euler('z',-pose[2]).apply(delta)
    source=local([*staging['support_center_xy'],
                  staging['unit_bottom_z']+unit['dimensions_m'][2]],staging['robot_base_pose'])
    cart_offset=Rotation.from_euler('z',staging['cart_body_pose'][2]).apply(
        [*staging['cart_load_center_body_xy'],0.])
    cart_world=np.array([*staging['cart_body_pose'][:2],
                         staging['cart_support_z_world']+unit['dimensions_m'][2]])+cart_offset
    cart=local(cart_world,staging['robot_base_pose'])
    target=local([*staging['target_surface_pose_world'][:2],
                  staging['target_surface_pose_world'][2]+unit['dimensions_m'][2]],staging['shelf_robot_pose'])
    base_z=mobile_manifest['spec']['base_z']
    home=np.array([config[j.name] for j in joints])
    bounds=(np.array([j.lower for j in joints]),np.array([j.upper for j in joints]))
    rng=np.random.default_rng(12)
    results={}
    previous=home
    def frame(q):
        values=dict(config);values.update({j.name:float(v) for j,v in zip(joints,q)})
        return link_world_transforms(robot,values)['gripper_pad_tcp']
    for label,point in [('source',source),('cart',cart),('shelf',target)]:
        if unit.get('grasp_mode', 'top') == 'side':
            wall = unit['local_wall']
            robot_pose = staging['shelf_robot_pose'] if label == 'shelf' else staging['robot_base_pose']
            angle = staging['unit_yaws_world'][label] + wall*np.pi/2 - robot_pose[2]
            rotation = (Rotation.from_euler('z', angle) *
                        Rotation.from_quat(unit['tcp_quaternion_xyzw'])).as_matrix()
            normal = np.array([-np.cos(angle), -np.sin(angle), 0.])
            half = unit['dimensions_m'][wall % 2]/2
            contact = point + normal*(half+unit['tcp_surface_offset_m'])
            contact[2] -= unit['dimensions_m'][2]/2 + base_z
            phases = [('contact', contact),
                      ('approach', contact+normal*unit['approach_m']),
                      ('lift', contact+np.array([0,0,unit['lift_m']])),
                      ('preplace', contact+normal*unit['approach_m']+np.array([0,0,unit['lift_m']]))]
        else:
            rotation = Rotation.from_quat(unit['tcp_quaternion_xyzw']).as_matrix()
            phases = [(phase, point+np.array([0,0,dz-base_z])) for phase,dz in
                      [('contact',unit['tcp_surface_offset_m']),
                       ('approach',unit['tcp_surface_offset_m']+unit['approach_m'])]]
        for phase,goal in phases:
            def residual(q):
                actual=frame(q)
                return np.r_[actual[:3,3]-goal,Rotation.from_matrix(rotation.T@actual[:3,:3]).as_rotvec()*.25]
            best=None
            for seed in [previous,home]+[rng.uniform(bounds[0],bounds[1]) for _ in range(15)]:
                candidate=least_squares(residual,seed,bounds=bounds,max_nfev=180,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                err=residual(candidate.x)
                if best is None or np.linalg.norm(err)<best[0]:
                    best=(np.linalg.norm(err),candidate.x,err)
                if np.linalg.norm(err[:3])<.002 and np.linalg.norm(err[3:])/.25<.01:
                    break
            _,q,err=best
            previous=q
            results[f'{label}_{phase}']=dict(position_error_m=float(np.linalg.norm(err[:3])),
                angular_error_rad=float(np.linalg.norm(err[3:])/.25),
                reachable=bool(np.linalg.norm(err[:3])<.002 and np.linalg.norm(err[3:])/.25<.01),
                joints={j.name:float(v) for j,v in zip(joints,q)},goal_in_base_link=goal.tolist(),
                quaternion_in_base_link=Rotation.from_matrix(rotation).as_quat().tolist())
    return dict(ok=all(v['reachable'] for v in results.values()),poses=results,
                scenario_sha256=hashlib.sha256(json.dumps(scenario,sort_keys=True).encode()).hexdigest(),
                urdf_sha256=hashlib.sha256(Path(urdf).read_bytes()).hexdigest(),
                evidence='actual_urdf_cpu_inverse_kinematics_only',
                collision_checked=False,physics_verified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mobile-manifest',type=Path,required=True)
    p.add_argument('--scenario',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=json.loads(a.mobile_manifest.read_text())
    result=check(m['mobile_urdf'],m,json.loads(a.scenario.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['ok'] else 1


if __name__=='__main__':
    raise SystemExit(main())
