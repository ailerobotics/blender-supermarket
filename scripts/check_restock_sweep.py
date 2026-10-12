#!/usr/bin/env python3
"""Conservative CPU rejection test for candidate joint-linear restock segments.

Uses every URDF collision link, loaded payload, cart and store mesh bounds.
Inflates segment bounds by a kinematic displacement bound between samples.
An overlap is a candidate rejection, not proof of physical penetration. This
is not a motion planner or runtime certification; self collision is not covered.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from twin.urdf_probe import parse_urdf, link_world_transforms, geometry_world_points, resolve_package_uri


def transform(pose, z=0.):
    result = np.eye(4)
    result[:3,:3] = Rotation.from_euler('z', pose[2]).as_matrix()
    result[:3,3] = [pose[0], pose[1], z]
    return result


def corners(low, high):
    return np.array(list(itertools.product(*zip(low, high))))


def apply(matrix, points):
    return points @ matrix[:3,:3].T + matrix[:3,3]


def mesh_bounds(path, matrix=None):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError('stage cannot be opened')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
    transforms = UsdGeom.XformCache()
    result = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or prim.IsInPrototype():
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        # Store frames join disconnected box solids into one mesh. Bounding the
        # whole frame fills the usable shelf opening; retain every component.
        if any(token in prim.GetName().lower() for token in ('frame', 'shelf')):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get())
            parents = list(range(len(points)))
            def root(index):
                while parents[index] != index:
                    parents[index] = parents[parents[index]]
                    index = parents[index]
                return index
            indices = list(mesh.GetFaceVertexIndicesAttr().Get())
            offset = 0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                face = indices[offset:offset+count]
                for index in face[1:]:
                    parents[root(index)] = root(face[0])
                offset += count
            world = np.asarray(transforms.GetLocalToWorldTransform(prim)).T
            points = apply(world, points)
            if matrix is not None:
                points = apply(matrix, points)
            groups = {}
            for index, point in enumerate(points):
                groups.setdefault(root(index), []).append(point)
            for index, cloud in groups.items():
                cloud = np.asarray(cloud)
                result.append((str(prim.GetPath())+'#component'+str(index),
                               cloud.min(axis=0), cloud.max(axis=0)))
            continue
        pts = corners(box.GetMin(), box.GetMax())
        if matrix is not None:
            pts = apply(matrix, pts)
        result.append((str(prim.GetPath()), pts.min(axis=0), pts.max(axis=0)))
    return result


def audit(manifest, scenario, reach, store, package_roots, max_step=.015, selected_locations=None):
    scenario_hash = hashlib.sha256(json.dumps(scenario,sort_keys=True).encode()).hexdigest()
    if reach.get('scenario_sha256') != scenario_hash:
        raise ValueError('IK evidence does not match the current scenario')
    if reach.get('urdf_sha256') != hashlib.sha256(Path(manifest['mobile_urdf']).read_bytes()).hexdigest():
        raise ValueError('IK evidence does not match the current URDF')
    robot = parse_urdf(manifest['mobile_urdf'])
    config = manifest['transport_positions']
    # Resolve once: every geometry must be present, no frame-only fallback.
    clouds = {}
    for name, link in robot.links.items():
        points = []
        for geometry in link.collisions:
            cloud = geometry_world_points(geometry, np.eye(4),
                lambda uri: resolve_package_uri(uri, package_roots)[0])
            if cloud is None:
                raise ValueError('unresolved collision geometry: '+name)
            points.extend(cloud)
        if points:
            cloud = np.asarray(points)
            clouds[name] = corners(cloud.min(axis=0), cloud.max(axis=0))
    # Any joint axis to any downstream collision point is bounded by the sum
    # of all fixed/joint translations and maximum link-local point radius.
    def chain_length(name):
        joint = robot.joints_by_child.get(name)
        return 0. if joint is None else np.linalg.norm(joint.origin_xyz)+chain_length(joint.parent)
    radius_bound = max(chain_length(name) for name in clouds)
    radius_bound += max(np.linalg.norm(p,axis=1).max() for p in clouds.values())
    unit, staging = scenario['handling_unit'], scenario['initial_staging']
    radius_bound += np.linalg.norm(unit['dimensions_m'])
    payload = corners(-np.array(unit['dimensions_m'])/2, np.array(unit['dimensions_m'])/2)
    # This actor is represented by the held-payload geometry below; treating its
    # initial pose as a second immutable obstacle would duplicate the same body.
    obstacles = [o for o in mesh_bounds(store)
                 if not o[0].startswith('/RestockScenario/HandlingUnit')]
    support_centre = np.r_[staging['support_center_xy'], staging['support_dimensions_m'][2]/2]
    support_half = np.array(staging['support_dimensions_m'])/2
    obstacles.append(('initial_staging_support',support_centre-support_half,support_centre+support_half))
    results = []
    locations = ('source','cart','cart_shelf','shelf') if 'shelf_cart_body_pose' in staging else ('source','cart','shelf')
    if selected_locations:
        if not set(selected_locations) <= set(locations):
            raise ValueError('unknown sweep station')
        locations=tuple(name for name in locations if name in selected_locations)
    for location in locations:
        pose = staging['shelf_robot_pose'] if location in ('shelf','cart_shelf') else staging['robot_base_pose']
        world = transform(pose, manifest['spec']['base_z'])
        if location in ('shelf','cart_shelf') and 'shelf_cart_body_pose' in staging:
            cart_pose = staging['shelf_cart_body_pose']
        elif location == 'shelf':
            sep = manifest['spec']['cart_hitch_x']-manifest['spec']['hitch_x']
            cart_xy = apply(transform(pose), np.array([[-sep,0,0]]))[0]
            cart_pose = [*cart_xy[:2],pose[2]]
        else:
            cart_pose = staging['cart_body_pose']
        local_obstacles = obstacles + mesh_bounds(manifest['cart_usd'], transform(cart_pose))
        # Broad phase around every possible point of this robot and payload.
        centre = world[:3,3]
        local_obstacles = [o for o in local_obstacles if
            np.all(o[2] >= centre-radius_bound) and np.all(o[1] <= centre+radius_bound)]
        lows = np.array([o[1] for o in local_obstacles]); highs = np.array([o[2] for o in local_obstacles])
        # Probe insertion/lift/retreat, with carried payload on insertion/lift.
        for start,end,held in [('approach','contact',False),('contact','lift',True),
                               ('preplace','contact',True),('contact','approach',False)]:
            a=reach['poses'][location+'_'+start]; b=reach['poses'][location+'_'+end]
            if not a['reachable'] or not b['reachable']:
                raise ValueError('cannot sweep an unreachable endpoint')
            names=list(a['joints']); q0=np.array([a['joints'][n] for n in names]); q1=np.array([b['joints'][n] for n in names])
            steps=max(1,int(np.ceil(np.max(np.abs(q1-q0))/max_step)),
                      int(np.ceil(radius_bound*np.sum(np.abs(q1-q0))/.004)))
            padding=radius_bound*np.sum(np.abs(q1-q0))/steps/2
            hits={}
            for i in range(steps):
                q=q0+(q1-q0)*((i+.5)/steps)
                values=dict(config); values.update(zip(names,q))
                fk=link_world_transforms(robot,values)
                moving={name:apply(world@fk[name],p) for name,p in clouds.items()}
                if held:
                    # Preserve the selected face's rigid pad-to-carton relation.
                    angle=staging['unit_yaws_world'][location]
                    half=unit['dimensions_m'][unit['local_wall']%2]/2+unit['tcp_surface_offset_m']
                    offset=Rotation.from_euler('z',angle+unit['local_wall']*np.pi/2).apply([-half,0,0])
                    pad_rotation=Rotation.from_quat(b['quaternion_in_base_link']).as_matrix()
                    pad_world_rotation=world[:3,:3]@pad_rotation
                    pad_from_carton=np.eye(4)
                    pad_from_carton[:3,:3]=pad_world_rotation.T@Rotation.from_euler('z',angle).as_matrix()
                    pad_from_carton[:3,3]=pad_world_rotation.T@(-offset)
                    moving['held_payload']=apply(world@fk['gripper_pad_tcp']@pad_from_carton,payload)
                for name,p in moving.items():
                    low=p.min(axis=0)-padding; high=p.max(axis=0)+padding
                    overlap=np.flatnonzero(np.all(high>lows+1e-6,axis=1)&np.all(low<highs-1e-6,axis=1))
                    for idx in overlap:
                        hits.setdefault((name,local_obstacles[idx][0]),i)
            results.append(dict(location=location,segment=start+'->'+end,held=held,samples=steps,
                displacement_bound_m=float(padding),overlaps=[dict(link=k[0],obstacle=k[1],first_sample=v) for k,v in hits.items()]))
    return dict(candidate_clear=not any(r['overlaps'] for r in results),segments=results,
        scenario_sha256=scenario_hash, store_sha256=hashlib.sha256(Path(store).read_bytes()).hexdigest(),
        evidence='conservative_all_link_joint_linear_sweep_bounds',physics_verified=False,
        self_collision_checked=False, runtime_trajectory_certified=False,
        limitations=['AABB overlap may be a false positive; rejection requires refinement.',
                     'Candidate joint-linear paths are not cuRobo trajectories.',
                     'No source-to-cart or cart-to-shelf transit path certified.',
                     'Shelf cart pose uses navigation feedback only when supplied; otherwise assumes aligned train.',
                     'Support contacts remain visible in results; no fixture exclusions.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('mobile-manifest','scenario','reach','store','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--package-root',action='append',required=True)
    p.add_argument('--navigation-evidence',type=Path)
    p.add_argument('--location',action='append',help='Bounded station audit; omitted means all stations')
    a=p.parse_args()
    from twin.usd_probe import ensure_pxr_env
    ensure_pxr_env()
    scenario=json.loads(a.scenario.read_text())
    if a.navigation_evidence:
        from check_restock_reach import with_navigation_arrival
        scenario=with_navigation_arrival(scenario,json.loads(a.navigation_evidence.read_text()))
    result=audit(json.loads(a.mobile_manifest.read_text()),scenario,
                 json.loads(a.reach.read_text()),a.store,a.package_root,selected_locations=a.location)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='segments'},indent=2))
    return 0 if result['candidate_clear'] else 1


if __name__=='__main__':
    raise SystemExit(main())
