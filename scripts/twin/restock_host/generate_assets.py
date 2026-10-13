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
                 acceleration=1., jerk=10.):
    """Pure CPU reconstruction used by both generator and strict evidence check."""
    urdf, manifest = Path(urdf).resolve(), Path(manifest).resolve()
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
        spheres[name] = entries

    for name in pitches:
        covers[name] = [VolumeCover(g, resolve, pitch) for g in robot.links[name].collisions]
        rebuild(name)
    excluded = kinematic_exclusions(robot)
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
              'min_pitch_m': min_pitch, 'link_pitches': pitches, 'refinement_rounds': rounds,
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
    args = parser.parse_args()
    report = generate(args.urdf, args.mobile_manifest, args.shelfbot_root, args.output,
                      pitch=args.pitch, min_pitch=args.min_pitch,
                      acceleration=args.acceleration, jerk=args.jerk)
    print(json.dumps({k: report[k] for k in ('sphere_count', 'planning_ready', 'transport_sphere_overlaps')}, indent=2))


if __name__ == '__main__':
    main()
