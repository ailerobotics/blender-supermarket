"""CPU geometry proof of the rollwagen upper-shelf cage clearance (scripts/check_cage_clearance.py).

Pins the numbers behind shipping the sim-only carton-vs-cage contact filter (cage_collision=filtered) for the
first video instead of a precision path: the slot's clearances from the AUTHORED cage, and -- with the mobile
URDF present -- that the controller's existing straight_line move (one 2-point joint-space smoothstep) is
not a Cartesian line over a full-depth insert, while K Cartesian segments are.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('pxr')
_SCRIPT = Path(__file__).resolve().parents[2] / 'check_cage_clearance.py'
_spec = importlib.util.spec_from_file_location('check_cage_clearance', _SCRIPT)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


@pytest.fixture(scope='module')
def boxes():
    return cc.cage_boxes()


def test_slot_clearances_from_the_authored_cage(boxes):
    s = cc.static_clearance(boxes)
    # the 0.6 m carton side spans the cart's y; the front posts leave a 0.615 m mouth
    assert s['carton_half_extents_cart'] == pytest.approx((0.2, 0.3, 0.08))
    assert s['front_post_gap_m'] == pytest.approx(0.615)
    assert s['front_post_clearance_per_side_m'] == pytest.approx(0.0075)      # 7.5 mm
    assert s['side_bar_clearance_per_side_m'] == pytest.approx(0.014)         # 14 mm
    # the carton's mouth face is flush with the open edge, INSIDE the front posts' x range
    lo, hi = s['front_post_x_range']
    assert (lo, hi) == pytest.approx((0.3675, 0.3925)) and lo < s['carton_front_face_x'] < hi
    assert s['mouth_exit_travel_m'] == pytest.approx(0.4125)                 # out through the mouth
    assert s['cage_top_z'] == pytest.approx(1.65)
    assert s['vertical_exit_rise_m'] == pytest.approx(0.7175)                # out over the top


def test_carton_clearance_at_rest_is_the_post_gap(boxes):
    assert cc.carton_clearance(cc.carton_rest_pose(), boxes, step=0.005) == pytest.approx(0.0075, abs=5e-4)
    shifted = cc.carton_rest_pose()
    shifted[1, 3] += 0.01                                       # 10 mm sideways => into a front post
    assert cc.carton_clearance(shifted, boxes, step=0.005) < 0


@pytest.mark.parametrize('angle', [0.3, math.pi - 1e-9, math.pi])
def test_pose_error_sees_a_flipped_tool(angle):
    """The IK residual must not read a 180 deg wrist flip as an exact solution (the skew part alone does)."""
    G = np.eye(4)
    M = np.eye(4)
    M[:3, :3] = cc._rotz(angle) @ cc._quat_R(*cc.GRASP_QUAT)
    G[:3, :3] = cc._quat_R(*cc.GRASP_QUAT)
    assert np.linalg.norm(cc._pose_err(M, G)) == pytest.approx(angle, abs=1e-6)


def test_vertical_exit_is_out_of_reach_from_the_coupled_park_pose():
    """Clearing the 1.65 m cage top needs the side-grip TCP 0.7175 m higher: past the UR10's 1.3 m reach."""
    from twin.mobile import MobileSpec
    spec = MobileSpec()
    arm_z = spec.arm_mount_z                                    # ur_base_link height (base 0.24 + 0.26)
    tcp_x = (spec.cart_hitch_x - spec.hitch_x) - cc.LOAD_CENTER[0] - cc.CARTON[1] / 2 - cc.TCP_SURFACE_OFFSET
    tcp_z = cc.SUPPORT_Z + cc.CARTON[2] / 2 - arm_z
    assert math.hypot(tcp_x, tcp_z) < 1.3                       # the grasp itself is in reach (~1.03 m)
    assert math.hypot(tcp_x, tcp_z + 0.7175) > 1.3              # the vertical exit is not (~1.52 m)


@pytest.mark.skipif(not cc.DEFAULT_URDF.is_file(), reason='mobile URDF (dataset) not on this box')
def test_single_straight_line_move_bows_into_the_posts_but_segments_do_not(boxes):
    """Collect insert at place_approach, elbow-up branch: the controller's straight_line move is ONE 2-point
    joint smoothstep -- over a full-depth (0.40 m) line it bows by centimetres (~45 mm, > the 7.5 mm post
    gap) -- while the same line as 9 Cartesian segments tracks within ~1.3 mm (6.5 mm clearance)."""
    from twin.mobile import MobileSpec
    chain = cc.Chain(cc.DEFAULT_URDF)
    T_cart_arm = cc.cart_in_arm(cc.PLACE_APPROACH, cc.CART_AT_COLLECT, chain, MobileSpec().base_z)
    tcp, axis, _ = cc.mouth_face_tcp(T_cart_arm)
    q0, ok = chain.ik(tcp, (0.0, -1.2, 1.6, -0.3, 1.57, 0.0))
    assert ok
    qs, reached = cc.straight_path(chain, q0, tcp, axis, 0.40, cc.LIFT, n=20)
    assert reached == pytest.approx(0.40)
    bow1, clear1 = cc.path_metrics(chain, qs, tcp, 1, T_cart_arm, boxes)
    bow9, clear9 = cc.path_metrics(chain, qs, tcp, 9, T_cart_arm, boxes)
    assert bow1 > 0.0075 and clear1 < 0
    assert bow9 < 0.0015 and clear9 > 0.005
    assert np.allclose(axis, (-1.0, 0.0, 0.0), atol=1e-3)      # the mouth faces the arm at place_approach


@pytest.mark.skipif(not cc.DEFAULT_URDF.is_file(), reason='mobile URDF (dataset) not on this box')
def test_no_ik_branch_completes_a_straight_exit_through_the_mouth():
    """Leaving the cage takes 0.4125 m out along the slot axis; from the collect place pose (~0.41 m) and the
    coupled transfer park (~0.36 m, the wrist singularity) the fixed-orientation line stops short on EVERY
    true IK branch -- why a clear-path switch is not a cheap workaround (cage_collision=filtered is)."""
    from twin.mobile import MobileSpec
    spec = MobileSpec()
    chain = cc.Chain(cc.DEFAULT_URDF)
    exit_m = cc.static_clearance(cc.cage_boxes(spec))['mouth_exit_travel_m']
    coupled_cart = (spec.hitch_x - spec.cart_hitch_x, 0., 0.)
    for base, cart, (lo, hi) in ((cc.PLACE_APPROACH, cc.CART_AT_COLLECT, (0.39, exit_m)),
                                 ((0., 0., 0.), coupled_cart, (0.33, 0.37))):
        T_cart_arm = cc.cart_in_arm(base, cart, chain, spec.base_z)
        tcp, axis, _ = cc.mouth_face_tcp(T_cart_arm)
        found = cc.branches(chain, tcp, math.atan2(tcp[1, 3], tcp[0, 3]))
        assert found
        for q0 in found:
            assert np.linalg.norm(cc._pose_err(chain.fk(q0), tcp)) < 1e-6      # a true solution, not a flip
            _, reached = cc.straight_path(chain, q0, tcp, axis, 0.45, cc.LIFT)
            assert lo <= reached < hi
