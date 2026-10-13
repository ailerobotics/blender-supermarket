#!/usr/bin/env python3
"""CPU-only design audit of physical cameras at independently articulated poses."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from twin.restock_camera.config import camera_config
from twin.restock_camera.geometry import Intrinsics, pose_matrix, transform_points
from twin.restock_camera.visibility import (Surface, box_surface, inspect_view,
                                            marker_decodability, sample_plane, usd_surfaces)
from twin.restock_camera.mount_audit import audit_mounts


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def qr_module_count(marker_id):
    """Module side of the actual printed QR for this id (version v -> 17+4v)."""
    import cv2
    return int(cv2.QRCodeEncoder_create().encode(marker_id).shape[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scenario','navigation','reach','mobile-manifest','store','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--package-root',action='append',required=True)
    a = p.parse_args()
    scenario,nav,reach,manifest = [json.loads(path.read_text()) for path in
                                 (a.scenario,a.navigation,a.reach,a.mobile_manifest)]
    s = scenario['initial_staging']
    audited = reach['scenario']['initial_staging']
    robot,cart = [[nav['legs'][0][name][key] for key in ('x','y','yaw')]
                  for name in ('robot_body_pose','cart_body_pose')]
    if (not np.allclose(robot,audited['shelf_robot_pose'])
            or not np.allclose(cart,audited['shelf_cart_body_pose'])
            or scenario['handling_unit'] != reach['scenario']['handling_unit']
            or any(s[key] != audited[key] for key in s if key not in ('shelf_robot_pose','unit_yaws_world'))
            or any(value != audited['unit_yaws_world'][key] for key,value in s['unit_yaws_world'].items())):
        raise ValueError('current scenario/navigation and all-link IK evidence disagree')
    config = camera_config(scenario,manifest)
    roots = a.package_root
    inputs = [a.scenario,a.navigation,a.reach,a.mobile_manifest,a.store,Path(manifest['mobile_urdf']),
              Path(manifest['cart_usd']),Path(__file__),Path(__file__).parent/'twin/urdf_probe.py',
              *[Path(__file__).parent/'twin/restock_camera'/name for name in
                ('config.py','geometry.py','visibility.py','mount_audit.py')]]
    hashes = {str(path.resolve()):sha(path) for path in inputs}
    results,layers = {},set()
    dims = np.array(scenario['handling_unit']['dimensions_m'])
    for station,camera_id,base,cart_pose in (
            ('source','source',s['robot_base_pose'],s['cart_body_pose']),
            ('cart_pickup','cart',s['robot_base_pose'],s['cart_body_pose']),
            ('cart_arrival','cart',robot,cart),('shelf','shelf',robot,cart)):
        item,support = config['cameras'][camera_id],config['supports'][camera_id]
        parent = pose_matrix(cart_pose,.2) if item['parent']=='cart_chassis' else np.eye(4)
        camera = parent @ item['parent_from_optical']
        k = Intrinsics(**item['intrinsics'])
        center = transform_points(parent,[support['nominal_center']])[0]
        yaw = support['yaw']+(cart_pose[2] if item['parent']=='cart_chassis' else 0)
        bounds = (np.minimum(camera[:3,3],center)-[1,1,.4],np.maximum(camera[:3,3],center)+[1,1,.4])
        surfaces,used = usd_surfaces(a.store,bounds,exclude=('/RestockScenario/HandlingUnit',))
        layers.update(used)
        parts,used = usd_surfaces(manifest['cart_usd'],([-1,-1,.2],[1,1,1.8]))
        layers.update(used)
        ground = pose_matrix(cart_pose)
        surfaces.extend(Surface(part.name,transform_points(ground,part.triangles.reshape(-1,3)).reshape(-1,3,3))
                        for part in parts)
        # Author the current source support analytically from its immutable input;
        # this is explicitly only design geometry, never runtime observation.
        surfaces.append(box_surface('current_source_support',
            [*s['support_center_xy'],s['unit_bottom_z']-s['support_dimensions_m'][2]/2],s['support_dimensions_m']))
        for solid in config['solids']:
            parent_solid = pose_matrix(cart_pose,.2) if solid['parent']=='cart_chassis' else np.eye(4)
            surfaces.append(box_surface(solid['id'],[0,0,0],solid['dimensions_m'],
                                        parent_solid @ solid['parent_from_solid']))
        loaded = box_surface('loaded_unit',[0,0,dims[2]/2],dims,pose_matrix([*center[:2],yaw],center[2]))
        witness = support['witness_design']
        witness_points = transform_points(parent,sample_plane(np.array(witness['center'])+[0,0,.0002],
                                                              witness['size_xy'],witness['yaw'],count=5))
        marker_results = {}
        for marker in support['markers']:
            points = transform_points(parent @ marker['parent_from_marker'],
                                      sample_plane([0,0,.0002],[marker['side_m']]*2,0,count=5))
            view = inspect_view(camera,points,k,surfaces+[loaded])
            # Decodable-pixel-size proof: project the QR data-region corners and report
            # pixels-per-module at the marker's actual module count. Necessary, never
            # sufficient; says nothing about lighting or contrast.
            side = marker['side_m']
            data_corners = transform_points(parent @ marker['parent_from_marker'],
                [[-side/2,-side/2,.0002],[side/2,-side/2,.0002],
                 [side/2,side/2,.0002],[-side/2,side/2,.0002]])
            view['decodability'] = marker_decodability(camera,data_corners,k,qr_module_count(marker['id']))
            marker_results[marker['id']] = view
        results[station] = dict(camera_id=camera_id,world_from_optical=camera.tolist(),
            base_from_optical=(np.linalg.inv(pose_matrix(base,manifest['spec']['arm_mount_z'])) @ camera).tolist(),
            robot_body_pose=base,cart_body_pose=cart_pose,
            top=inspect_view(camera,sample_plane(center+[0,0,dims[2]],dims[:2]-.02,yaw),k,surfaces),
            empty_support=inspect_view(camera,sample_plane(center,[.62,.42],yaw),k,surfaces),
            loaded_support=inspect_view(camera,sample_plane(center,[.62,.42],yaw),k,surfaces+[loaded]),
            loaded_witness=inspect_view(camera,witness_points,k,surfaces+[loaded]),markers=marker_results)
    audit,meshes = audit_mounts(config,manifest,reach,roots)
    for path in [*layers,*meshes]: hashes[str(Path(path).resolve())] = sha(path)
    for path in inputs:
        if hashes[str(path.resolve())] != sha(path): raise ValueError('input changed during audit')
    report = dict(evidence='CPU_geometric_design_only_not_rendered_sensor_or_PhysX',runtime_ready=False,
        input_sha256=hashes,camera_config=config,stations=results,mount_sweep=audit,
        limitations=['Sampled ray visibility does not prove pixel quality, decoding or full-volume clearance.',
                     'All-link joint-linear bounds reject overlaps; no motion planner or transit certification.',
                     'Real commissioning must measure uncertainty margins beyond the required .62x.42 rectangle.',
                     'New static source/shelf mounts require inclusion in navigation obstacle maps.',
                     'Commissioning and loaded mission must share the same physical marker/mount revisions.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:{name:value[name]['visible_samples'] for name in ('top','empty_support','loaded_witness')}
                      for key,value in results.items()}))
    print(json.dumps({'mount_candidate_clear':audit['candidate_clear'],'segments':len(audit['segments'])}))


if __name__ == '__main__': main()
