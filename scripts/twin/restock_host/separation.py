"""Narrow, recomputable all-configuration separation proofs, never sampled ACMs."""
import numpy as np

from ..urdf_probe import geometry_world_points, joint_transform, rpy_to_matrix

EPS = 1e-8


def _separated(a, b):
    for axis in range(3):
        for first, second, order in ((a, b, 'a_before_b'), (b, a, 'b_before_a')):
            gap = float(second[0][axis] - first[1][axis])
            if gap > EPS:
                return {'axis': axis, 'order': order, 'clearance_m': gap}
    return None


def _points(robot, name, transform, resolver):
    arrays = [geometry_world_points(g, transform, resolver) for g in robot.links[name].collisions]
    if not arrays or any(p is None or not np.isfinite(p).all() for p in arrays):
        raise ValueError('unreadable separation-proof collision geometry: ' + name)
    return np.concatenate(arrays)


def prismatic_siblings(robot, a, b, resolver):
    """Each solid sweeps linearly between finite endpoints in a shared parent.

    Disjoint bounding intervals on one parent axis prove disjointness for every
    independent setting of both joints, irrespective of ancestor arm motion.
    """
    ja, jb = robot.joints_by_child.get(a), robot.joints_by_child.get(b)
    if not ja or not jb or ja.jtype != 'prismatic' or jb.jtype != 'prismatic' or ja.parent != jb.parent:
        return None
    bounds = []
    for name, joint in ((a, ja), (b, jb)):
        if not np.isclose(np.linalg.norm(joint.axis), 1.):
            return None
        if joint.lower is None or joint.upper is None or not np.isfinite([joint.lower, joint.upper]).all() or joint.lower > joint.upper:
            return None
        points = np.concatenate([_points(robot, name, joint_transform(joint, q), resolver)
                                 for q in (joint.lower, joint.upper)])
        bounds.append([points.min(0), points.max(0)])
    separation = _separated(*bounds)
    if separation is None:
        return None
    return {'rule': 'disjoint_full_prismatic_sweeps', 'links': [a, b],
            'reference_frame': ja.parent,
            'joints': [ja.name, jb.name], 'ranges': [[ja.lower, ja.upper], [jb.lower, jb.upper]],
            'swept_bounds': [np.asarray(v).tolist() for v in bounds], **separation}


def swivel_wheel(robot, a, b, resolver):
    """A centred wheel's enclosing ball rotates below the ancestor chassis.

    Inner joint rotation preserves the ball. Outer swivel rotation preserves
    its projection along the swivel axis. A strict separating slab therefore
    holds for arbitrary, independent roll/swivel angles (not only transport).
    """
    roll = robot.joints_by_child.get(b)
    swivel = None if roll is None else robot.joints_by_child.get(roll.parent)
    if (roll is None or swivel is None or swivel.parent != a or
            roll.jtype not in ('continuous', 'revolute') or
            swivel.jtype not in ('continuous', 'revolute')):
        return None
    radii = []
    for geometry in robot.links[b].collisions:
        if geometry.kind != 'cylinder' or np.linalg.norm(geometry.origin_xyz) > EPS:
            return None
        # Exact enclosing ball for an entire solid cylinder, including end caps.
        radii.append(float(np.hypot(geometry.radius, geometry.length/2)))
    if not radii:
        return None
    radius = max(radii)
    rotation = rpy_to_matrix(swivel.origin_rpy)
    axis = rotation @ np.asarray(swivel.axis)
    if not np.isclose(np.linalg.norm(axis), 1.):
        return None
    center = np.asarray(swivel.origin_xyz) + rotation @ np.asarray(roll.origin_xyz)
    points = _points(robot, a, np.eye(4), resolver)
    projection = points @ axis
    interval = [float(center @ axis - radius), float(center @ axis + radius)]
    gap = max(float(projection.min()-interval[1]), float(interval[0]-projection.max()))
    if gap <= EPS:
        return None
    return {'rule': 'swivel_invariant_wheel_ball_separation', 'links': [a, b],
            'reference_frame': a, 'joints': [swivel.name, roll.name],
            'angle_domain': 'all independent real swivel and roll angles',
            'separating_axis': axis.tolist(), 'wheel_ball_radius': radius,
            'wheel_projection_interval': interval,
            'chassis_projection_interval': [float(projection.min()), float(projection.max())],
            'clearance_m': gap}


def prove_separation(robot, a, b, resolver):
    return (prismatic_siblings(robot, a, b, resolver) or
            swivel_wheel(robot, a, b, resolver) or swivel_wheel(robot, b, a, resolver))
