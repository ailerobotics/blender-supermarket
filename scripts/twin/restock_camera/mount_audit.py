"""Conservative CPU all-link/held-payload bounds against the added camera solids.

This bounded audit complements, never replaces, integration's store/cart/self
collision checks. An AABB overlap is a rejection needing refinement, not a
claim of mesh penetration. No route or cuRobo trajectory is certified here.
"""
import hashlib
from itertools import product
import json
from pathlib import Path
import numpy as np

from ..urdf_probe import parse_urdf, link_world_transforms, geometry_world_points, resolve_package_uri
from .geometry import pose_matrix, transform_points


def corners(low,high):
    return np.array(list(product(*zip(low,high))))


def audit_mounts(config, manifest, reach, package_roots):
    if hashlib.sha256(Path(manifest['mobile_urdf']).read_bytes()).hexdigest() != reach['urdf_sha256']:
        raise ValueError('all-link IK evidence uses another URDF')
    scenario = reach['scenario']
    if hashlib.sha256(json.dumps(scenario,sort_keys=True).encode()).hexdigest() != reach['scenario_sha256']:
        raise ValueError('all-link IK scenario integrity mismatch')
    robot = parse_urdf(manifest['mobile_urdf'])
    clouds,files = {},set()
    def resolve(uri):
        path = resolve_package_uri(uri,package_roots)[0]
        if path: files.add(str(path))
        return path
    for name,link in robot.links.items():
        points = []
        for geometry in link.collisions:
            cloud = geometry_world_points(geometry,np.eye(4),resolve)
            if cloud is None: raise ValueError('missing collision geometry: '+name)
            points.extend(cloud)
        if points:
            cloud = np.asarray(points)
            clouds[name] = corners(cloud.min(axis=0),cloud.max(axis=0))
    def chain(name):
        joint = robot.joints_by_child.get(name)
        return 0. if joint is None else np.linalg.norm(joint.origin_xyz)+chain(joint.parent)
    unit,s = scenario['handling_unit'],scenario['initial_staging']
    radius = max(chain(name) for name in clouds)+max(np.linalg.norm(p,axis=1).max() for p in clouds.values())
    radius += np.linalg.norm(unit['dimensions_m'])
    payload = corners(-np.array(unit['dimensions_m'])/2,np.array(unit['dimensions_m'])/2)
    results = []
    for location in ('source','cart','cart_shelf','shelf'):
        arrival = location in ('cart_shelf','shelf')
        base = s['shelf_robot_pose'] if arrival else s['robot_base_pose']
        cart = s['shelf_cart_body_pose'] if arrival else s['cart_body_pose']
        world = pose_matrix(base,manifest['spec']['base_z'])
        obstacles = []
        for solid in config['solids']:
            parent = pose_matrix(cart,.2) if solid['parent']=='cart_chassis' else np.eye(4)
            half = np.array(solid['dimensions_m'])/2
            points = transform_points(parent @ solid['parent_from_solid'],corners(-half,half))
            obstacles.append((solid['id'],points.min(axis=0),points.max(axis=0)))
        lows,highs = np.array([o[1] for o in obstacles]),np.array([o[2] for o in obstacles])
        for start,end,held in (('approach','contact',False),('contact','lift',True),
                               ('preplace','contact',True),('contact','approach',False)):
            a,b = reach['poses'][location+'_'+start],reach['poses'][location+'_'+end]
            if not a['reachable'] or not b['reachable']: raise ValueError('unreachable sweep endpoint')
            names = list(a['joints'])
            q0,q1 = [np.array([pose['joints'][n] for n in names]) for pose in (a,b)]
            steps = max(1,int(np.ceil(radius*np.sum(np.abs(q1-q0))/.004)))
            padding = radius*np.sum(np.abs(q1-q0))/steps/2
            hits = {}
            for i in range(steps):
                values = dict(manifest['transport_positions'])
                values.update(zip(names,q0+(q1-q0)*(i+.5)/steps))
                fk = link_world_transforms(robot,values)
                moving = {name:transform_points(world @ fk[name],cloud) for name,cloud in clouds.items()}
                if held:
                    # Exact wall3 rigid pad/carton relation from this station's IK.
                    from scipy.spatial.transform import Rotation
                    angle = s['unit_yaws_world'][location]
                    half = unit['dimensions_m'][unit['local_wall']%2]/2+unit['tcp_surface_offset_m']
                    offset = pose_matrix([0,0,angle+unit['local_wall']*np.pi/2])[:3,:3] @ [-half,0,0]
                    pad_rotation = world[:3,:3] @ Rotation.from_quat(b['quaternion_in_base_link']).as_matrix()
                    relative = np.eye(4)
                    relative[:3,:3] = pad_rotation.T @ pose_matrix([0,0,angle])[:3,:3]
                    relative[:3,3] = pad_rotation.T @ (-offset)
                    moving['held_payload'] = transform_points(world @ fk['gripper_pad_tcp'] @ relative,payload)
                for name,points in moving.items():
                    low,high = points.min(axis=0)-padding,points.max(axis=0)+padding
                    overlaps = np.flatnonzero(np.all(high>lows+1e-6,axis=1)&np.all(low<highs-1e-6,axis=1))
                    for index in overlaps: hits.setdefault((name,obstacles[index][0]),i)
            results.append(dict(location=location,segment=start+'->'+end,held_payload=held,samples=steps,
                displacement_bound_m=float(padding),overlaps=[dict(link=k[0],solid=k[1],first_sample=v) for k,v in hits.items()]))
    return dict(candidate_clear=not any(r['overlaps'] for r in results),segments=results,
        checked_collision_links=sorted(clouds),self_collision_checked=False,
        existing_store_cart_collision_checked=False,transit_checked=False,
        evidence='conservative_all_link_and_held_payload_joint_linear_bounds',physics_verified=False),files
