"""CPU unit test for the fm355 gravity_ff mapping fix (map_arm_gravity).

Proves the full-DOF + view-dof-name mapping extracts the arm gravity terms correctly even when the
articulation view's DOF order is scrambled and the arm dofs sit at non-trivial indices -- and that the
OLD naive "first six" / robot-index slice would have mis-mapped (the run-279 shoulder_pan bug).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from twin.restock.physics_setup import map_arm_gravity  # noqa: E402

ARM = ['ur_shoulder_pan_joint', 'ur_shoulder_lift_joint', 'ur_elbow_joint',
       'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']

# True per-joint gravity (arm order): pan/wrists ~0, lift/elbow carry the load.
TRUE = {'ur_shoulder_pan_joint': 0.0, 'ur_shoulder_lift_joint': -140.0,
        'ur_elbow_joint': -37.0, 'ur_wrist_1_joint': -6.0,
        'ur_wrist_2_joint': 0.0, 'ur_wrist_3_joint': 0.0}

# A realistic mobile articulation view: base/wheel/caster dofs interleaved, arm dofs NOT contiguous
# and NOT in arm order (mirrors the run-153/154 "view order != robot/arm order" finding).
VIEW = ['base_x', 'base_y', 'ur_shoulder_lift_joint', 'ur_elbow_joint', 'ur_wrist_1_joint',
        'left_wheel', 'ur_shoulder_pan_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint',
        'right_wheel', 'front_caster', 'rear_caster'] + ['x%d' % i for i in range(6)]


def _full_vector():
    g = np.zeros(len(VIEW))
    for n, val in TRUE.items():
        g[VIEW.index(n)] = val
    return g


def test_mapping_extracts_arm_terms_in_arm_order():
    g = map_arm_gravity(_full_vector(), VIEW, ARM)
    assert list(np.round(g, 3)) == [TRUE[n] for n in ARM]


def test_mapping_handles_2d_single_env_shape():
    # the view returns shape (M, K); for one env reshape(-1) must still align
    g2d = _full_vector().reshape(1, -1)
    g = map_arm_gravity(g2d, VIEW, ARM)
    assert list(np.round(g, 3)) == [TRUE[n] for n in ARM]


def test_pan_term_is_zero_not_the_spurious_value():
    # The core run-279 bug: pan (vertical axis) must map to ~0, never a large lift/elbow value.
    g = map_arm_gravity(_full_vector(), VIEW, ARM)
    assert abs(g[0]) < 1e-6          # shoulder_pan
    assert abs(g[4]) < 1e-6          # wrist_2
    assert g[1] < -100 and g[2] < -10  # lift / elbow carry the real load


def test_naive_first_six_slice_would_misassign():
    # Demonstrate the bug the fix closes: a naive "first 6 dofs" slice picks base/wheel dofs, and a
    # contiguous arm slice in VIEW order is NOT arm_joints order -> shoulder_pan would get a wrong term.
    g = _full_vector()
    naive_first6 = g[:6]
    assert not np.allclose(naive_first6, [TRUE[n] for n in ARM])
    # a slice in VIEW order (lift,elbow,wrist1,...,pan,...) puts lift's -140 in the pan slot
    view_arm_order = [n for n in VIEW if n in TRUE]
    assert view_arm_order[0] != ARM[0]  # first arm dof in view is NOT shoulder_pan


def test_missing_arm_name_raises():
    import pytest
    with pytest.raises(ValueError):
        map_arm_gravity(_full_vector(), [n for n in VIEW if n != 'ur_shoulder_pan_joint'], ARM)


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
