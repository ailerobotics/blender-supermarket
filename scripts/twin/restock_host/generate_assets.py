"""Generate conservative planning assets from actual mobile CAD, without a GPU.

Every source collision volume is enclosed by tiled AABB circumspheres. This is
deliberately conservative; overlap diagnostics are reported, never suppressed by
sampled or guessed collision exclusions. Only rigid/adjacent kinematic groups
are excluded from self collision.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from ..urdf_probe import (geometry_world_points, link_world_transforms, parse_urdf,
                          resolve_package_uri)
from .config import FRAME, JOINTS, TCP


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


def generate(urdf, manifest, shelfbot_root, output, *, pitch=.05, acceleration=1., jerk=10.):
    urdf, manifest, output = Path(urdf).resolve(), Path(manifest).resolve(), Path(output).resolve()
    if urdf == output / 'mobile.urdf':
        raise ValueError('source URDF must not be overwritten')
    if not all(np.isfinite(v) and v > 0 for v in (acceleration, jerk)):
        raise ValueError('positive finite planner acceleration and jerk policies required')
    robot = parse_urdf(str(urdf))
    mobile = json.loads(manifest.read_text())
    positions = {j.name: float(mobile['transport_positions'].get(j.name, 0.))
                 for j in robot.joints if j.jtype != 'fixed'}
    for name in JOINTS:
        if name not in mobile['transport_positions']:
            raise ValueError('manifest missing transport arm joint ' + name)
    roots = [str(shelfbot_root), '/opt/ros/jazzy/share', str(urdf.parent)]
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (urdf, manifest)}

    def resolve(filename):
        path, _ = resolve_package_uri(filename, roots)
        if path is None:
            raise ValueError('unresolved source mesh: ' + filename)
        sources[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return path

    spheres, coverage = {}, []
    for name, link in robot.links.items():
        entries = []
        for index, geometry in enumerate(link.collisions):
            points = geometry_world_points(geometry, np.eye(4), resolve)
            if points is None or not np.isfinite(points).all() or not len(points):
                raise ValueError(f'unreadable collision geometry {name}/{index}')
            low, high = points.min(axis=0), points.max(axis=0)
            centers, radius = cover_bounds(low, high, pitch)
            entries.extend({'center': c.tolist(), 'radius': radius} for c in centers)
            coverage.append({'link': name, 'collision_index': index, 'kind': geometry.kind,
                             'bounds_min': low.tolist(), 'bounds_max': high.tolist(),
                             'source_vertex_count': len(points), 'sphere_count': len(centers),
                             'sphere_radius': radius,
                             'proof': 'complete enclosing AABB tiled into cells; each cell enclosed by circumsphere'})
        if entries:
            spheres[name] = entries
    excluded = kinematic_exclusions(robot)
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
    xrdf = {'format': 'xrdf', 'format_version': 1.0,
            'modifiers': [{'set_base_frame': FRAME}],
            'default_joint_positions': positions, 'tool_frames': [TCP],
            'cspace': {'joint_names': list(JOINTS), 'acceleration_limits': [acceleration]*6,
                       'jerk_limits': [jerk]*6},
            'collision': {'geometry': 'mobile_cad'},
            'self_collision': {'geometry': 'mobile_cad', 'ignore': ignore},
            'geometry': {'mobile_cad': {'spheres': spheres}}}
    # Diagnose the conservative sphere model at the actual transport posture.
    transforms = link_world_transforms(robot, positions)
    world = {}
    for name, entries in spheres.items():
        local = np.array([s['center'] for s in entries])
        tf = transforms[name]
        world[name] = (local @ tf[:3, :3].T + tf[:3, 3], np.array([s['radius'] for s in entries]))
    overlaps = []
    for a, b in itertools.combinations(sorted(world), 2):
        if (a, b) in excluded:
            continue
        ac, ar = world[a]
        bc, br = world[b]
        penetration = -float('inf')
        for start in range(0, len(ac), 128):
            distance = np.linalg.norm(ac[start:start+128, None, :] - bc[None, :, :], axis=2)
            penetration = max(penetration, float(np.max(ar[start:start+128, None] + br[None, :] - distance)))
        if penetration > 0:
            overlaps.append({'links': [a, b], 'max_sphere_penetration_m': penetration})
    # Resolve all visual/collision package references without changing geometry.
    normalized = ET.parse(urdf)
    for mesh in normalized.findall('.//mesh'):
        mesh.set('filename', resolve(mesh.get('filename')))
    report = {'source': 'actual_mobile_urdf_collision_geometry_cpu_only',
              'source_sha256': sources, 'pitch_m': pitch,
              'planner_acceleration_rad_s2': acceleration, 'planner_jerk_rad_s3': jerk,
              'policy_note': 'configured conservative planner limits, not measured actuator limits',
              'collision_elements': coverage, 'sphere_count': sum(map(len, spheres.values())),
              'covered_links': sorted(spheres),
              'exclusion_rule': 'same fixed-joint component or components joined by exactly one URDF joint',
              'excluded_pairs': [{'links': list(pair), 'reason': reason} for pair, reason in sorted(excluded.items())],
              'transport_sphere_overlaps': overlaps,
              'planning_ready': False,
              'limitations': 'AABB enclosure can create false collisions; no extra exclusions added. Dynamics, parked CAD state and actual planner feasibility unverified.'}
    output.mkdir(parents=True, exist_ok=True)
    normalized.write(output / 'mobile.urdf', encoding='unicode', xml_declaration=True)
    ET.indent(srdf)
    ET.ElementTree(srdf).write(output / 'mobile.srdf', encoding='unicode', xml_declaration=True)
    (output / 'mobile.xrdf').write_text(yaml.safe_dump(xrdf, sort_keys=False))
    for name in ('mobile.urdf', 'mobile.srdf', 'mobile.xrdf'):
        report.setdefault('generated_sha256', {})[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    (output / 'asset-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--mobile-manifest', required=True)
    parser.add_argument('--shelfbot-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pitch', type=float, default=.05)
    parser.add_argument('--acceleration', type=float, default=1.)
    parser.add_argument('--jerk', type=float, default=10.)
    args = parser.parse_args()
    report = generate(args.urdf, args.mobile_manifest, args.shelfbot_root, args.output,
                      pitch=args.pitch, acceleration=args.acceleration, jerk=args.jerk)
    print(json.dumps({k: report[k] for k in ('sphere_count', 'planning_ready', 'transport_sphere_overlaps')}, indent=2))


if __name__ == '__main__':
    main()
