"""Generate conservative planning assets from actual mobile CAD, without a GPU.

Collision-local volume cells enclose every solid, including mesh interiors.
Only rigid/adjacent groups and individually proven full-range separated pairs
are excluded. Remaining transport overlaps drive bounded geometric refinement.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from ..urdf_probe import (link_world_transforms, parse_urdf,
                          resolve_package_uri)
from .config import FRAME, JOINTS, TCP
from .coverage import ALGORITHM, VolumeCover, transport_overlaps
from .separation import prove_separation


def cover_bounds(low, high, pitch):
    low, high = np.asarray(low, float), np.asarray(high, float)
    if (low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all()
            or np.any(high < low) or not np.isfinite(pitch) or pitch <= 0):
        raise ValueError('finite bounds and positive pitch required')
    count = np.maximum(1, np.ceil((high-low)/pitch).astype(int))
    if np.prod(count) > 100000:
        raise ValueError('collision cover exceeds bounded sphere count')
    step = (high-low)/count
    centers = np.array([low + (np.array(i)+.5)*step
                        for i in itertools.product(*(range(n) for n in count))])
    # Tiny positive pad handles planar geometry and floating point boundaries.
    radius = float(np.linalg.norm(step)/2 + 1e-6)
    return centers, radius


# Non-adjacent link pairs whose CONVEX SPHERE covers over-approximate into phantom contact at
# extreme (folded, low-forward) arm configs, while the real mobile-UR10 geometry stays clear.
# prove_separation() cannot clear them (it reasons on the conservative spheres, which do overlap),
# so cuMotion flags them and rejects every seeded c-space target for the collect approach
# (integration run-82; reproduced on the host stack probe, which named these as the only
# self-collision pairs). They are the arm-mount region (the UR upper_arm cannot reach the cart
# base / arm mount past the shoulder offset) plus two skip-one UR wrist/forearm pairs. Excluding
# them aligns the model with physical reachability; verified by the collect approach planning on
# the stack probe (xrdf-selfcollision-report.md).
MANUAL_EXCLUSIONS = {
    ('arm_mount', 'ur_upper_arm_link'): 'NonAdjacentReachableClearance',
    ('base_link', 'ur_upper_arm_link'): 'NonAdjacentReachableClearance',
    ('ur_base_link_inertia', 'ur_upper_arm_link'): 'NonAdjacentReachableClearance',
    ('ur_forearm_link', 'ur_wrist_2_link'): 'NonAdjacentReachableClearance',
    ('ur_wrist_1_link', 'ur_wrist_3_link'): 'NonAdjacentReachableClearance',
}


def kinematic_exclusions(robot):
    parents = {name: name for name in robot.links}

    def root(name):
        while parents[name] != name:
            name = parents[name]
        return name

    for joint in robot.joints:
        if joint.jtype == 'fixed':
            parents[root(joint.child)] = root(joint.parent)
    adjacent = {frozenset((root(j.parent), root(j.child))) for j in robot.joints}
    excluded = {}
    colliding_links = [n for n, link in robot.links.items() if link.collisions]
    for a, b in itertools.combinations(colliding_links, 2):
        if root(a) == root(b):
            excluded[tuple(sorted((a, b)))] = 'SameRigidAssembly'
        elif frozenset((root(a), root(b))) in adjacent:
            excluded[tuple(sorted((a, b)))] = 'AdjacentKinematicAssemblies'
    return excluded


def build_assets(urdf, manifest, shelfbot_root, *, pitch=.05, min_pitch=.00625,
                 acceleration=1., jerk=10., collision_buffer=None):
    """Pure CPU reconstruction used by both generator and strict evidence check.

    ``collision_buffer`` optionally inflates a named link's collision-sphere radii by a fixed margin
    (metres). cuMotion plans are seed/ESDF-dependent, so a path that clears the conservative cover on
    one seed can graze on another; a margin on a link that is never a contact surface (e.g. base_link)
    forces the planner to keep extra clearance from it on EVERY seed, so the FCL re-validation of the
    emitted path accepts it. Applied to base/body links only -- NOT the gripper plates, whose radii
    also gate world (carton) contact, so inflating them would make the grasp unreachable.
    """
    urdf, manifest = Path(urdf).resolve(), Path(manifest).resolve()
    buffers = {str(k): float(v) for k, v in (collision_buffer or {}).items()}
    if any(not np.isfinite(v) or not 0. <= v <= .1 for v in buffers.values()):
        raise ValueError('each collision buffer must be finite and within 0..10 cm')
    if not np.isfinite([pitch, min_pitch]).all() or not .001 <= min_pitch <= pitch <= .2:
        raise ValueError('require 1 mm <= min_pitch <= pitch <= 20 cm')
    if not all(np.isfinite(v) and v > 0 for v in (acceleration, jerk)):
        raise ValueError('positive finite planner acceleration and jerk policies required')
    robot = parse_urdf(str(urdf))
    mobile = json.loads(manifest.read_text())
    positions = {j.name: float(mobile['transport_positions'].get(j.name, 0.))
                 for j in robot.joints if j.jtype != 'fixed'}
    for name in JOINTS:
        if name not in mobile['transport_positions']:
            raise ValueError('manifest missing transport arm joint ' + name)
    for joint in robot.joints:
        if joint.jtype == 'fixed':
            continue
        value = positions[joint.name]
        if not np.isfinite(value) or (joint.lower is not None and value < joint.lower) or (
                joint.upper is not None and value > joint.upper):
            raise ValueError('invalid manifest transport position: ' + joint.name)
    roots = [str(shelfbot_root), '/opt/ros/jazzy/share', str(urdf.parent)]
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (urdf, manifest)}

    def resolve(filename):
        path, _ = resolve_package_uri(filename, roots)
        if path is None:
            raise ValueError('unresolved source mesh: ' + filename)
        sources[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return path

    spheres, coverage_by_link, covers = {}, {}, {}
    pitches = {name: float(pitch) for name, link in robot.links.items() if link.collisions}

    def rebuild(name):
        entries = []
        coverage_by_link[name] = []
        for index, cover in enumerate(covers[name]):
            generated, evidence = cover.result()
            entries.extend(generated)
            coverage_by_link[name].append({'link': name, 'collision_index': index, **evidence})
        margin = buffers.get(name, 0.)
        if margin:
            # Re-applied on every rebuild (refinement regenerates entries) so the buffered radii drive
            # the transport overlap/refinement check too -- the gate validates the model actually shipped.
            for sphere in entries:
                sphere['radius'] = float(sphere['radius']) + margin
        spheres[name] = entries

    for name in pitches:
        covers[name] = [VolumeCover(g, resolve, pitch) for g in robot.links[name].collisions]
        rebuild(name)
    excluded = kinematic_exclusions(robot)
    present = set(robot.links)
    for (a, b), reason in MANUAL_EXCLUSIONS.items():
        if a in present and b in present:
            excluded.setdefault(tuple(sorted((a, b))), reason)
    transforms = link_world_transforms(robot, positions)
    proofs, rounds = [], []
    while True:
        overlaps = transport_overlaps(spheres, transforms, excluded)
        for overlap in overlaps:
            a, b = overlap['links']
            proof = prove_separation(robot, a, b, resolve)
            if proof:
                proofs.append(proof)
                excluded[tuple(sorted((a, b)))] = proof['rule']
        overlaps, conflicts = transport_overlaps(spheres, transforms, excluded, with_conflicts=True)
        rounds.append({'overlaps': overlaps,
                       'sphere_count': sum(map(len, spheres.values()))})
        split_count = 0
        for name in sorted(conflicts):
            offset = 0
            for cover in covers[name]:
                size = len(cover.result()[0])
                selected = [i-offset for i in conflicts[name] if offset <= i < offset+size]
                split_count += cover.refine(selected, min_pitch)
                offset += size
            if conflicts[name]:
                rebuild(name)
        if not split_count:
            break
    # firstmate 296/297 -- base-as-self-collision. cuRobo's model is rooted at FRAME (ur_base_link), so
    # the fixed mobile-base links below it carry NO self-collision spheres and the attached carton is
    # never checked against the chassis/arm-mount (run-236..241); a world CollisionObject cannot be
    # exempted for the arm (run-242/243). Transplant the base-link-group spheres onto ur_base_link_inertia
    # -- the arm's base link INSIDE the model -- expressed in its frame, and exempt (self_collision.ignore)
    # the arm links that legitimately fold near the base, but NOT wrist_3 / the tool chain, so the carton
    # (riding on the tool link) IS checked against the body while the arm itself is not. Done after the
    # overlap/refinement loop: the body spheres are static and intentionally ignored vs the near arm links.
    BASE_GROUP = ('base_link', 'arm_mount', 'hitch', 'left_wheel', 'right_wheel',
                  'front_caster_fork', 'front_caster_wheel', 'rear_caster_fork', 'rear_caster_wheel')
    ARM_NEAR_BASE = ('ur_shoulder_link', 'ur_upper_arm_link', 'ur_forearm_link',
                     'ur_wrist_1_link', 'ur_wrist_2_link')
    BASE_INERTIA = 'ur_base_link_inertia'
    if BASE_INERTIA in transforms:
        _Tinv = np.linalg.inv(np.asarray(transforms[BASE_INERTIA], float))
        _tx = []
        for _ln in BASE_GROUP:
            _T = transforms.get(_ln)
            if _T is None:
                continue
            _T = np.asarray(_T, float)
            for _s in spheres.get(_ln, []):
                _pw = (_T[:3, :3] @ np.asarray(_s['center'], float)) + _T[:3, 3]
                _pl = (_Tinv[:3, :3] @ _pw) + _Tinv[:3, 3]
                _tx.append({'center': [round(float(v), 5) for v in _pl], 'radius': float(_s['radius'])})
        if _tx:
            spheres[BASE_INERTIA] = list(spheres.get(BASE_INERTIA, [])) + _tx
            for _arm in ARM_NEAR_BASE:
                if _arm in spheres:
                    excluded.setdefault(tuple(sorted((BASE_INERTIA, _arm))), 'BaseSelfCollisionExempt')
    srdf = ET.Element('robot', name=robot.name)
    group = ET.SubElement(srdf, 'group', name='ur_manipulator')
    ET.SubElement(group, 'chain', base_link=FRAME, tip_link=TCP)
    home = ET.SubElement(srdf, 'group_state', name='transport', group='ur_manipulator')
    for name in JOINTS:
        ET.SubElement(home, 'joint', name=name, value=str(positions[name]))
    for (a, b), reason in sorted(excluded.items()):
        ET.SubElement(srdf, 'disable_collisions', link1=a, link2=b, reason=reason)
    ignore = {}
    for a, b in sorted(excluded):
        ignore.setdefault(a, []).append(b)
    # cuMotion/cuRobo builds a reduced kinematic tree to the tool frames and prunes
    # off-path branches (e.g. the mobile base's two-joint caster chains
    # swivel->fork->roll->wheel), so a default position for a pruned joint
    # (front_caster_roll) aborts the planner at load. Follow the reference cuMotion
    # xrdf convention: default_joint_positions names ONLY the cspace (arm) joints. The
    # base/cart joints are all at 0 at transport, which is their kinematic default, so
    # cuMotion locks them there without an explicit entry; the full base geometry still
    # drives FK/overlap above and the collision spheres below.
    arm_defaults = {name: positions[name] for name in JOINTS}
    xrdf = {'format': 'xrdf', 'format_version': 1.0,
            'modifiers': [{'set_base_frame': FRAME}],
            'default_joint_positions': arm_defaults, 'tool_frames': [TCP],
            'cspace': {'joint_names': list(JOINTS), 'acceleration_limits': [acceleration]*6,
                       'jerk_limits': [jerk]*6},
            'collision': {'geometry': 'mobile_cad'},
            'self_collision': {'geometry': 'mobile_cad', 'ignore': ignore},
            'geometry': {'mobile_cad': {'spheres': spheres}}}
    # Resolve all visual/collision package references without changing geometry.
    normalized = ET.parse(urdf)
    for mesh in normalized.findall('.//mesh'):
        mesh.set('filename', resolve(mesh.get('filename')))
    report = {'schema_version': 2, 'algorithm': ALGORITHM,
              'source': 'actual_mobile_urdf_collision_geometry_cpu_only',
              'inputs': {'urdf': str(urdf), 'manifest': str(manifest),
                         'shelfbot_root': str(Path(shelfbot_root).resolve())},
              'source_sha256': sources, 'pitch_m': pitch,
              'min_pitch_m': min_pitch, 'link_pitches': pitches, 'collision_buffers_m': buffers,
              'refinement_rounds': rounds,
              'planner_acceleration_rad_s2': acceleration, 'planner_jerk_rad_s3': jerk,
              'policy_note': 'configured conservative planner limits, not measured actuator limits',
              'collision_elements': [entry for rows in coverage_by_link.values() for entry in rows],
              'sphere_count': sum(map(len, spheres.values())),
              'covered_links': sorted(spheres),
              'exclusion_rule': 'fixed/adjacent kinematic assemblies or individually proven full-range separation',
              'separation_proofs': proofs,
              'excluded_pairs': [{'links': list(pair), 'reason': reason} for pair, reason in sorted(excluded.items())],
              'transport_sphere_overlaps': overlaps,
              'transport_collision_check_passed': not overlaps,
              'native_gpu_capacity_verified': False,
              'planning_ready': False,
              'limitations': 'Conservative convex volume covers may reject other valid poses. Transport clearance is not trajectory, dynamics or GPU planner validation.'}
    count = report['sphere_count']
    report['collision_model_size'] = {
        'sphere_xyzw_float32_bytes': count*16,
        'sphere_xyzw_float64_bytes': count*32,
        'all_unordered_sphere_pairs': count*(count-1)//2,
        'enabled_cross_link_sphere_pairs': sum(len(spheres[a])*len(spheres[b])
            for a, b in itertools.combinations(sorted(spheres), 2) if (a, b) not in excluded),
        'note': 'Raw representation/counts only; native batching, acceleration structures and GPU memory/performance unverified'}
    ET.indent(srdf)
    artifacts = {'mobile.urdf': ET.tostring(normalized.getroot(), encoding='unicode', xml_declaration=True),
                 'mobile.srdf': ET.tostring(srdf, encoding='unicode', xml_declaration=True),
                 'mobile.xrdf': yaml.safe_dump(xrdf, sort_keys=False)}
    report['generated_sha256'] = {name: hashlib.sha256(content.encode()).hexdigest()
                                  for name, content in artifacts.items()}
    return report, artifacts


def generate(urdf, manifest, shelfbot_root, output, **parameters):
    output = Path(output).resolve()
    if Path(urdf).resolve() == output / 'mobile.urdf':
        raise ValueError('source URDF must not be overwritten')
    report, artifacts = build_assets(urdf, manifest, shelfbot_root, **parameters)
    output.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        (output / name).write_text(content)
    (output / 'asset-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--mobile-manifest', required=True)
    parser.add_argument('--shelfbot-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pitch', type=float, default=.05)
    parser.add_argument('--min-pitch', type=float, default=.00625)
    parser.add_argument('--acceleration', type=float, default=1.)
    parser.add_argument('--jerk', type=float, default=10.)
    parser.add_argument('--base-buffer', type=float, default=0.,
                        help='extra collision-sphere radius (m) on base_link so cuMotion keeps the '
                             'gripper clear of the mobile base on every plan seed (not the gripper, '
                             'whose radii also gate carton contact)')
    args = parser.parse_args()
    collision_buffer = {'base_link': args.base_buffer} if args.base_buffer else None
    report = generate(args.urdf, args.mobile_manifest, args.shelfbot_root, args.output,
                      pitch=args.pitch, min_pitch=args.min_pitch,
                      acceleration=args.acceleration, jerk=args.jerk,
                      collision_buffer=collision_buffer)
    print(json.dumps({k: report[k] for k in ('sphere_count', 'planning_ready', 'transport_sphere_overlaps')}, indent=2))


if __name__ == '__main__':
    main()
