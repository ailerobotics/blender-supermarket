#!/usr/bin/env python3
"""Optional offline CPU geometry report; never boots Isaac or certifies sensing."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from twin.restock_camera import Intrinsics, look_at, pose_matrix, relative_camera
from twin.restock_camera.geometry import transform_points, plane_roi
from twin.restock_camera.visibility import (Surface, box_surface, inspect_view,
                                            sample_plane, usd_surfaces)
from twin.restock_manipulation.cad_tool import inspect_tool


def sha(path):
    digest = hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('scenario','arrival','navigation','store','cart','urdf','mobile-manifest','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--historical-arrival',action='store_true',
                        help='Replay the embedded arrival scenario; report differences from current scenario')
    modes.add_argument('--current-source-only',action='store_true',
                       help='Inspect only the current source; requires a matching source-support USD')
    args = parser.parse_args()
    source_files = [Path(__file__),Path(inspect_tool.__code__.co_filename),
                    Path(__file__).parent/'twin/urdf_probe.py',
                    *sorted((Path(__file__).parent/'twin/restock_camera').glob('*.py'))]
    input_files = [args.scenario,args.arrival,args.navigation,args.store,args.cart,args.urdf,
                   args.mobile_manifest,*source_files]
    initial_hashes = {str(path.resolve()):sha(path) for path in input_files}
    scenario = json.loads(args.scenario.read_text())
    arrival = json.loads(args.arrival.read_text())
    nav = json.loads(args.navigation.read_text())
    mobile = json.loads(args.mobile_manifest.read_text())['spec']
    staging = scenario['initial_staging']
    final = nav['legs'][0]
    robot = [final['robot_body_pose'][key] for key in ('x','y','yaw')]
    cart = [final['cart_body_pose'][key] for key in ('x','y','yaw')]
    audited = arrival['scenario']['initial_staging']
    if not (np.allclose(robot,audited['shelf_robot_pose'])
            and np.allclose(cart,audited['shelf_cart_body_pose'])):
        raise ValueError('arrival and navigation artifacts disagree')
    comparison = dict(audited)
    comparison['unit_yaws_world'] = {key:audited['unit_yaws_world'].get(key)
                                    for key in staging['unit_yaws_world']}
    differences = {key:dict(current=value,arrival=comparison.get(key))
                   for key,value in staging.items()
                   if key != 'shelf_robot_pose' and value != comparison.get(key)}
    if scenario['handling_unit'] != arrival['scenario']['handling_unit']:
        differences['handling_unit'] = dict(current=scenario['handling_unit'],
                                            arrival=arrival['scenario']['handling_unit'])
    if differences and not (args.historical_arrival or args.current_source_only):
        raise ValueError('current scenario differs from arrival evidence; regenerate evidence or explicitly replay --historical-arrival')
    if args.historical_arrival:
        staging = audited
    from pxr import Usd, UsdGeom
    source_stage = Usd.Stage.Open(str(args.store))
    support_prim = source_stage.GetPrimAtPath('/RestockScenario/Support')
    if not support_prim:
        raise ValueError('source support geometry missing from inspected USD')
    support_range = UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(support_prim).ComputeAlignedRange()
    support_min,support_max = np.array(support_range.GetMin()),np.array(support_range.GetMax())
    if (not np.allclose((support_min+support_max)[:2]/2,staging['support_center_xy'],atol=1e-6,rtol=0)
            or not np.allclose(support_max-support_min,staging['support_dimensions_m'],atol=1e-6,rtol=0)
            or not np.isclose(support_max[2],staging['unit_bottom_z'],atol=1e-6,rtol=0)):
        raise ValueError('scenario and inspected source support geometry disagree; regenerate the source artifact')
    urdf = ET.parse(args.urdf).getroot()
    joint = urdf.find("joint[@name='ur_base_joint']")
    if (joint.find('parent').get('link') != 'arm_mount'
            or joint.find('origin').get('xyz') != '0 0 0'
            or joint.find('origin').get('rpy') != '0 0 0'):
        raise ValueError('unrecognized CAD arm mount transform')
    unit = (arrival['scenario'] if args.historical_arrival else scenario)['handling_unit']
    if unit['local_wall'] != 3 or unit['grasp_mode'] != 'side':
        raise ValueError('presets require wall3 side-contact scenario')
    dims = np.array(unit['dimensions_m'])
    tool = inspect_tool(args.urdf)
    k = Intrinsics(1280,960,640.,640.,639.5,479.5,.15,3.5)
    source = np.array([*staging['support_center_xy'],staging['unit_bottom_z']])
    shelf = np.array(staging['target_surface_pose_world'][:3])
    deck_local = np.array([*staging['cart_load_center_body_xy'],staging['cart_support_z_world']])
    cart_mount = look_at([.08,-.02,.88],[.08,0,.30])
    stations = [
        ('source',staging['robot_base_pose'],None,source,staging['unit_yaws_world']['source'],
         look_at(source+[0,.20,.85],source+[0,0,.08]),'fixed source mast with overhead boom'),
        ('cart_pickup',staging['robot_base_pose'],staging['cart_body_pose'],deck_local,
         staging['unit_yaws_world']['cart'],None,'rigid cart upper-shelf underside mount'),
        ('cart_arrival',robot,cart,deck_local,cart[2]+staging['unit_yaws_world']['cart'],
         None,'same cart mount at independently articulated arrival'),
        ('shelf',robot,None,shelf,staging['unit_yaws_world']['shelf'],
         look_at(shelf+[0,.55,.22],shelf+[0,0,.12]),'fixed shelf underside bracket extending into aisle'),
    ]
    if args.current_source_only:
        stations = stations[:1]
    results,layers = {},set()
    for name,base_pose,cart_pose,center,yaw,camera,mount in stations:
        world_base = pose_matrix(base_pose,mobile['arm_mount_z'])
        if cart_pose is not None:
            world_cart = pose_matrix(cart_pose)
            center = transform_points(world_cart,[center])[0]
            camera = world_cart @ cart_mount
        # Include entire line-of-sight volume, then ray-test actual disconnected
        # solids instead of their enclosing cage/frame AABB.
        bounds = (np.minimum(center,camera[:3,3])-[1,1,.25],
                  np.maximum(center,camera[:3,3])+[1,1,.25])
        surfaces,used = usd_surfaces(args.store,bounds,exclude=('/RestockScenario/HandlingUnit',))
        layers.update(used)
        if cart_pose is not None:
            cart_surfaces,used = usd_surfaces(args.cart,([-1,-1,.22],[1,1,1.0]))
            layers.update(used)
            surfaces.extend(Surface(s.name,transform_points(world_cart,s.triangles.reshape(-1,3)).reshape(-1,3,3))
                            for s in cart_surfaces)
        top = sample_plane(center+[0,0,dims[2]],dims[:2]-.02,yaw)
        support = sample_plane(center,dims[:2]+.02,yaw)
        roi_size = dims[:2]+.02
        if name == 'shelf':
            roi_size = np.array([dims[0]+.02,dims[1]-.02])
        corners = sample_plane(center,roi_size,yaw,count=2)[[0,2,3,1]]
        candidate_roi = plane_roi(corners,camera,k)
        # A single SlotView uses the same ROI for top-label identity and empty
        # support. Check where the top-centre ray lands when the carton is absent.
        eye = camera[:3,3]
        top_center = center+[0,0,dims[2]]
        factor = (center[2]-eye[2])/(top_center[2]-eye[2])
        empty_label_ray = eye+factor*(top_center-eye)
        local_hit = transform_points(np.linalg.inv(pose_matrix([*center[:2],yaw],center[2])),[empty_label_ray])[0]
        label_ray_on_patch = bool(factor>0 and np.all(np.abs(local_hit[:2]) <= roi_size/2))
        # Keep loaded carton as an explicit occluder when checking the support.
        unit_transform = pose_matrix([center[0],center[1],yaw],center[2])
        loaded = box_surface('loaded_handling_unit',[0,0,dims[2]/2],dims,unit_transform)
        angle = yaw+3*np.pi/2
        normal = np.array([-np.cos(angle),-np.sin(angle),0.])
        grip = center+[0,0,dims[2]/2]+normal*(dims[1]/2+unit['tcp_surface_offset_m'])
        # Conservative actual-CAD tool envelope for contact/insertion/lift only.
        # This is deliberately not an all-link arm trajectory collision verdict.
        cyclic = np.array([[0,0,1],[1,0,0],[0,1,0]])  # quaternion (.5,.5,.5,.5)
        rotation = pose_matrix([0,0,angle])[:3,:3] @ cyclic @ tool.tool0_from_pad[:3,:3].T
        envelope = np.array([[x,y,z] for x in (tool.envelope_pad_min[0],tool.envelope_pad_max[0])
            for y in (tool.envelope_pad_min[1],tool.envelope_pad_max[1])
            for z in (tool.envelope_pad_min[2],tool.envelope_pad_max[2])]) @ rotation.T
        swept = np.concatenate([envelope+grip+normal*distance+[0,0,lift]
            for distance in (0,unit['approach_m']) for lift in (0,unit['lift_m'])])
        housing = box_surface('camera_housing',[0,0,0],[.06,.025,.025],camera).triangles.reshape(-1,3)
        separation = np.maximum(0,np.maximum(housing.min(axis=0)-swept.max(axis=0),
                                             swept.min(axis=0)-housing.max(axis=0)))
        results[name] = dict(mount=mount,world_from_camera=camera.tolist(),
            base_from_camera=relative_camera(world_base,camera).tolist(),
            world_from_arm_base=world_base.tolist(),cart_pose=cart_pose,
            wall3_tcp_world=grip.tolist(),wall3_approach_world=(grip+normal*unit['approach_m']).tolist(),
            camera_to_contact_insertion_lift_tool_aabb_separation_m=float(np.linalg.norm(separation)),
            planned_empty_plane_roi=list(candidate_roi),
            roi_candidate_plane_xy_m=roi_size.tolist(),
            roi_candidate_visibility=inspect_view(camera,sample_plane(center,roi_size,yaw),k,surfaces),
            roi_candidate_fits_required_footprint=bool(np.all(roi_size>=dims[:2]+.02-1e-9)),
            top_label_center_ray_hits_empty_patch=label_ray_on_patch,
            empty_top_label_ray_world=empty_label_ray.tolist(),
            top=inspect_view(camera,top,k,surfaces),
            empty_support=inspect_view(camera,support,k,surfaces),
            loaded_support=inspect_view(camera,support,k,surfaces+[loaded]),
            occluder_solids=len(surfaces))
    inputs = {str(path.resolve()):sha(path) for path in [*input_files,*map(Path,sorted(layers))]}
    if any(inputs[path] != digest for path,digest in initial_hashes.items()):
        raise ValueError('input changed during inspection; discard prediction and rerun')
    report = dict(evidence='offline_geometric_design_prediction_not_rendered_sensor_capture',
        runtime_ready=False,intrinsics=asdict(k),input_sha256=inputs,
        historical_arrival_replay=args.historical_arrival,scenario_differences=differences,
        scope='current_source_only_no_route_assertion' if args.current_source_only else 'historical_stations' if args.historical_arrival else 'current_stations',
        actual_cad_panel_size_m=list(tool.panel_size),
        actual_cad_envelope_pad_min_m=list(tool.envelope_pad_min),
        actual_cad_envelope_pad_max_m=list(tool.envelope_pad_max),
        cart_camera_in_ground_aligned_cart_frame=cart_mount.tolist(),
        cart_camera_in_chassis_frame=(pose_matrix([0,0,0],-.20) @ cart_mount).tolist(),
        camera_housing_dimensions_m=[.06,.025,.025],stations=results,
        limitations=[
            'Sampled geometric rays do not prove QR decoding, depth quality, stereo baseline visibility or physics.',
            'Candidate housings/brackets require collision authoring and actual CAD all-link arm sweep before use.',
            'Station views assume arm withdrawn; moving/held gripper occlusion invalidates capture.',
            'USD mesh tessellation only; no opacity/texturing or lens distortion model.',
            'Inspected shelf has flat .730 m support and a raised .736 m lip; mixed support ROIs must be rejected.',
            'Known geometry is used only here; runtime support/SKU/clearance must be measured from fresh pixels.',
        ])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{key:station[key]['visible_samples'] for key in ('top','empty_support','loaded_support')}
                      for name,station in results.items()},indent=2))


if __name__ == '__main__':
    main()
