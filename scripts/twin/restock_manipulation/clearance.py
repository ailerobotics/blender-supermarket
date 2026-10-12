"""Evaluation-only conservative tool sweep for fixed-orientation linear segments.

This is not an arm trajectory validator or a replacement for cuRobo collision
checking. Bounds may come from scene fixtures here, never perception inputs.
"""
import itertools
import numpy as np


def linear_tool_sweep(tool, start, end):
    """Exact AABB union of the conservative CAD envelope over a linear TCP move.

    Reject rotation changes: checking only endpoints of a rotating box is unsafe.
    """
    a, b = np.asarray(start, float), np.asarray(end, float)
    if a.shape != (7,) or b.shape != (7,) or not np.isfinite([a, b]).all():
        raise ValueError('finite xyzw poses required')
    qa, qb = a[3:], b[3:]
    if not np.isclose(np.linalg.norm(qa), 1.) or not np.isclose(np.linalg.norm(qb), 1.):
        raise ValueError('unit quaternion required')
    if not (np.allclose(qa, qb) or np.allclose(qa, -qb)):
        raise ValueError('rotating sweep requires full trajectory validation')
    x,y,z,w = qa
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    corners = np.array(list(itertools.product(*zip(tool.envelope_pad_min,
                                                   tool.envelope_pad_max))))
    rotated = corners @ rotation.T
    return (rotated.min(axis=0) + np.minimum(a[:3], b[:3]),
            rotated.max(axis=0) + np.maximum(a[:3], b[:3]))


def overlaps(bounds, obstacle):
    low, high = map(np.asarray, bounds)
    other_low, other_high = map(np.asarray, obstacle)
    return bool(np.all(high >= other_low) and np.all(other_high >= low))
