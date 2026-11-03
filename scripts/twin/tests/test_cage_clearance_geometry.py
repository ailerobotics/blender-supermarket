"""CPU geometry proof of the rollwagen upper-shelf cage clearance (scripts/check_cage_clearance.py).

Pins the clearance numbers for the carton that the SINGLE source of truth (scripts/restock_scenario.json
``handling_unit.dimensions_m``) defines. sim-carton-size-v1 shrank the carton to 0.45 x 0.30 x 0.18 m, so the
0.45 m side now threads the 0.615 m front-post mouth with ~82.5 mm per side (was 7.5 mm for the 0.6 m carton):
the slot clearance is forgiving, and -- with the mobile URDF present -- the controller's existing single
``straight_line`` move (one 2-point joint-space smoothstep) now both COMPLETES the straight exit through the
mouth AND stays clear of the cage, where the old 0.6 m carton bowed into the posts. The sim-only carton-vs-cage
contact filter (``cage_collision=filtered``) is therefore no longer needed for straight-line clearance; it
stays available (opt-in, default off) as a belt-and-suspenders label.
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
    # the 0.45 m carton side spans the cart's y; the front posts leave a 0.615 m mouth => ~82.5 mm per side
    assert s['carton_half_extents_cart'] == pytest.approx((0.15, 0.225, 0.09))
    assert s['front_post_gap_m'] == pytest.approx(0.615)
    assert s['front_post_clearance_per_side_m'] == pytest.approx(0.0825)      # 82.5 mm (was 7.5 mm)
    assert s['side_bar_clearance_per_side_m'] == pytest.approx(0.089)         # 89 mm (was 14 mm)
    # the 0.30 m-deep carton's mouth face (x=0.33) now sits SHORT of the front posts' x range
    lo, hi = s['front_post_x_range']
    assert (lo, hi) == pytest.approx((0.3675, 0.3925)) and s['carton_front_face_x'] < lo
    assert s['mouth_exit_travel_m'] == pytest.approx(0.3625)                 # out through the mouth
    assert s['cage_top_z'] == pytest.approx(1.65)
    assert s['vertical_exit_rise_m'] == pytest.approx(0.7175)                # out over the top


def test_carton_clearance_at_rest_is_forgiving(boxes):
    # the true 3-D min distance at rest is ~89 mm (to the side bars; the short front posts sit off the
    # carton's x range) -- a forgiving slot, not the old 7.5 mm margin.
    assert cc.carton_clearance(cc.carton_rest_pose(), boxes, step=0.005) == pytest.approx(0.089, abs=5e-4)
    shifted = cc.carton_rest_pose()
    shifted[1, 3] += 0.10                                       # 100 mm sideways => into the cage bars
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
    assert math.hypot(tcp_x, tcp_z) < 1.3                       # the grasp itself is in reach (~1.07 m)
    assert math.hypot(tcp_x, tcp_z + 0.7175) > 1.3              # the vertical exit is not (~1.55 m)


@pytest.mark.skipif(not cc.DEFAULT_URDF.is_file(), reason='mobile URDF (dataset) not on this box')
def test_single_straight_line_move_now_clears_the_cage_with_the_smaller_carton(boxes):
    """Collect insert at place_approach, elbow-up branch: the controller's straight_line move is ONE 2-point
    joint smoothstep, so over a full-depth (0.40 m) line it still bows by centimetres (~39 mm) -- but with the
    0.45 m carton's ~82.5 mm per-side slack that bow no longer reaches the cage: the single move clears by
    ~75 mm (was a penetration for the 0.6 m carton). More Cartesian segments only tighten the bow further."""
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
    assert bow1 > 0.02 and clear1 > 0.03          # a centimetre-scale bow, but the carton still clears the cage
    assert bow9 < 0.0015 and clear9 > 0.08        # 9 segments: near-straight, ~82 mm clearance
    assert np.allclose(axis, (-1.0, 0.0, 0.0), atol=1e-3)      # the mouth faces the arm at place_approach


@pytest.mark.skipif(not cc.DEFAULT_URDF.is_file(), reason='mobile URDF (dataset) not on this box')
def test_every_ik_branch_completes_a_straight_exit_through_the_mouth():
    """Leaving the cage now takes only 0.3625 m out along the slot axis (the 0.30 m-deep carton); from the
    collect place pose (reaches 0.45 m) and the coupled transfer park (reaches 0.40 m) the fixed-orientation
    line clears the mouth on EVERY true IK branch -- so the smaller carton makes a clear straight-line path
    feasible, where the 0.6 m carton stopped short and needed the cage_collision=filtered workaround."""
    from twin.mobile import MobileSpec
    spec = MobileSpec()
    chain = cc.Chain(cc.DEFAULT_URDF)
    exit_m = cc.static_clearance(cc.cage_boxes(spec))['mouth_exit_travel_m']
    coupled_cart = (spec.hitch_x - spec.cart_hitch_x, 0., 0.)
    for base, cart in ((cc.PLACE_APPROACH, cc.CART_AT_COLLECT),
                       ((0., 0., 0.), coupled_cart)):
        T_cart_arm = cc.cart_in_arm(base, cart, chain, spec.base_z)
        tcp, axis, _ = cc.mouth_face_tcp(T_cart_arm)
        found = cc.branches(chain, tcp, math.atan2(tcp[1, 3], tcp[0, 3]))
        assert found
        for q0 in found:
            assert np.linalg.norm(cc._pose_err(chain.fk(q0), tcp)) < 1e-6      # a true solution, not a flip
            _, reached = cc.straight_path(chain, q0, tcp, axis, 0.45, cc.LIFT)
            assert reached >= exit_m - 1e-9                   # clears the mouth on every branch
