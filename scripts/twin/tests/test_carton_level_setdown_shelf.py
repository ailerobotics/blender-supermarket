"""CPU-only: the collect place-on-cart SET-DOWN must descend through the CLEAR opening above the
rollwagen support and land the carton FLAT, not drag its bottom-front across the shelf's +x (mouth)
edge (sim-setdown-shelf-contact-v1).

coldwin33/34/35 each tripped CARTON_TILT pitch +2.06 deg at phase=place onto the cart. The planned
descent is a SINGLE diagonal straight line from the pre-place standoff (which the fixed
collect_place_preplace_shape parks OUT IN FRONT of the cart, the held carton overhanging the open +x
mouth) down+in to the support load-centre. That diagonal drags the carton ACROSS the /Rollwagen/chassis/
shelf front-top (mouth) edge WHILE the carton bottom is already below the shelf top, so the carton's
bottom-front rides the shelf edge and pivots off level -- a CONTACT pivot, not droop (three generations
of droop compensation, PR 311/318/326, left the executed 2.06 deg unchanged, d(exec)/d(cmd) ~ 0).

The fix (carton_level.place_from_above) routes the set-down as an L-path: horizontal at the standoff
height (clear of the shelf top) to DIRECTLY ABOVE the load-centre, then VERTICAL down -- so the carton
reaches the shelf top with its whole footprint inside the shelf (lands FLAT, level) instead of over the
mouth edge. ``place_depth_bias_m`` adds extra margin by nudging the set-down deeper into the cart. The
per-waypoint reach+level+droop-cancel IK and the CARTON_TILT bound/grace are UNCHANGED; the bound is
never widened and no collider is removed.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_carton_level_setdown_shelf.py
"""
import json
import math
import os

import numpy as np
import pytest

from twin.restock_manipulation import carton_level as cl

# --------------------------------------------------------------- pure config / path (no URDF)

def test_place_from_above_default_off_and_byte_identical_path():
    lc = cl.resolve_level_config(True)
    assert lc.place_from_above is False
    assert lc.place_depth_bias_m == pytest.approx(0.0)
    assert lc.place_above_clearance_m == pytest.approx(0.08)
    # default (approach_from_above False) == the legacy straight-line TCP interpolation, exactly
    s = np.array([0.5, 0.1, 1.03]); t = np.array([0.18, 0.0, 0.90]); n = 12
    legacy = [((1.0 - i / n) * s + (i / n) * t) for i in range(1, n + 1)]
    got = cl._descent_tcp_targets(s, t, n)
    assert all(np.allclose(a, b) for a, b in zip(legacy, got))


def test_resolve_place_from_above_keys():
    lc = cl.resolve_level_config({'place_from_above': True, 'place_above_clearance_m': 0.10,
                                  'place_depth_bias_m': 0.05})
    assert lc.place_from_above is True
    assert lc.place_above_clearance_m == pytest.approx(0.10)
    assert lc.place_depth_bias_m == pytest.approx(0.05)


@pytest.mark.parametrize('bad', [{'place_above_clearance_m': -0.01}, {'place_depth_bias_m': -1},
                                 {'place_above_clearance_m': float('nan')}])
def test_resolve_rejects_bad(bad):
    with pytest.raises(ValueError):
        cl.resolve_level_config(bad)


def test_from_above_is_L_path():
    """First leg stays at/above the standoff height (clears the shelf); last point == target; the
    descent is monotone-down only after the carton is laterally over the target."""
    s = np.array([0.5, 0.1, 1.03]); t = np.array([0.18, 0.0, 0.90]); n = 12
    pts = cl._descent_tcp_targets(s, t, n, approach_from_above=True, above_clearance_m=0.08)
    assert np.allclose(pts[-1], t)                       # lands exactly on the support target
    # the horizontal leg keeps z at the via height (>= start z) while xy moves toward the target
    assert pts[0][2] >= s[2] - 1e-9
    # at least one early point is still near the standoff height while xy has advanced toward target
    assert pts[0][2] > t[2] + 0.05
    # z is non-increasing along the path (never climbs above the via then dips erratically)
    zs = [p[2] for p in pts]
    assert all(zs[i] >= zs[i + 1] - 1e-6 for i in range(len(zs) - 1))


def test_depth_bias_shifts_target_deeper_along_approach():
    s = np.array([0.5, 0.1, 1.03]); t = np.array([0.18, 0.0, 0.90])
    base = cl._descent_tcp_targets(s, t, 12, approach_from_above=True, depth_bias_m=0.0)
    deep = cl._descent_tcp_targets(s, t, 12, approach_from_above=True, depth_bias_m=0.06)
    # the biased endpoint moves further along the horizontal start->target direction (away from start)
    horiz = np.array([t[0] - s[0], t[1] - s[1], 0.0]); horiz /= np.linalg.norm(horiz)
    adv_base = np.dot(base[-1] - t, horiz)
    adv_deep = np.dot(deep[-1] - t, horiz)
    assert adv_deep > adv_base + 0.05 - 1e-6


# --------------------------------------------------------------- full reconstruction (coldwin35 URDF)

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin35/mobile-import.urdf'
PLAN = os.path.join(os.path.dirname(__file__), 'fixtures', 'setdown_shelf_descent_coldwin35.json')
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='coldwin35 mission URDF not present')

# recorded anchors (coldwin35 nav-trace.jsonl / host-stack.log, descent start t=176.4)
BASE = (-4.611, 14.672, math.radians(85.80))
CARTON0 = (-4.751, 15.489, 1.086, math.radians(175.87))
CART = (-4.5981, 15.9849, math.radians(-92.54), 0.2)   # chassis body world x,y,yaw,z
ARM_MOUNT_WORLD_Z = 0.50                                # MobileSpec.arm_mount_z
CARTON_DIMS = (0.45, 0.30, 0.18)
SHELF_HALF = ((0.80 - 0.04) / 2, (0.68 - 0.04) / 2, 0.025 / 2)
SHELF_C_LOCAL = (0.0, 0.0, 0.72)                        # chassis-frame centre
SHELF_EDGE_X = SHELF_HALF[0]                            # +0.38, the +x (mouth) edge
PLAN_NAMES = ['ur_elbow_joint', 'ur_shoulder_lift_joint', 'ur_shoulder_pan_joint',
              'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _h(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def _front_overhang_at_crossing(approach, bias):
    """Carton +x (mouth-side) edge position in the cart frame relative to the shelf mouth edge, at the
    FIRST descent point where the carton bottom reaches the shelf-top plane. >0 => the carton overhangs
    the open mouth when it lands (edge contact -> pivot); <=0 => fully over the shelf (lands flat)."""
    from twin.urdf_probe import link_world_transforms
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    robot = __import__('twin.urdf_probe', fromlist=['parse_urdf']).parse_urdf(URDF)
    plan = json.load(open(PLAN))

    def qmap(pt):
        m = dict(zip(PLAN_NAMES, pt)); return {n: float(m[n]) for n in ARM_JOINTS}

    Tbw = _h(_rz(BASE[2]), (BASE[0], BASE[1], ARM_MOUNT_WORLD_Z))

    def tool(qd):
        W = link_world_transforms(robot, qd, root='ur_base_link')
        return Tbw @ np.asarray(W['ur_tool0'], float)

    T0 = tool(qmap(plan[0][1]))
    weld = np.linalg.inv(T0) @ _h(_rz(CARTON0[3]), CARTON0[:3])
    start_tcp = T0[:3, 3].copy()
    tgt_tcp = tool(qmap(plan[-1][1]))[:3, 3].copy()
    R0 = T0[:3, :3]

    cart_T = _h(_rz(CART[2]), (CART[0], CART[1], CART[3]))
    cart_inv = np.linalg.inv(cart_T)
    shelf_top = CART[3] + SHELF_C_LOCAL[2] + SHELF_HALF[2]   # world z of the shelf top
    half = np.array(CARTON_DIMS) / 2.0
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half

    tcps = cl._descent_tcp_targets(start_tcp, tgt_tcp, 60, approach_from_above=approach,
                                   above_clearance_m=0.08, depth_bias_m=bias)
    overhang = None
    min_edge_clear = 9.9
    for tcp in tcps:
        Tc = _h(R0, tcp) @ weld
        cw = (Tc[:3, :3] @ corners.T).T + Tc[:3, 3]
        cbz = float(cw[:, 2].min())
        front_x = max(float((cart_inv @ np.array([*c, 1.0]))[0]) for c in cw)
        if cbz <= shelf_top + 1e-4:
            min_edge_clear = min(min_edge_clear, SHELF_EDGE_X - front_x)
            if overhang is None:
                overhang = front_x - SHELF_EDGE_X
    return overhang, min_edge_clear


@need_urdf
def test_old_straight_line_overhangs_shelf_edge():
    """Regression evidence: the shipped straight-line descent (from_above OFF) lands the carton with its
    front edge OVER the open mouth edge -> edge contact -> the recorded +2.06 deg pivot."""
    overhang, _ = _front_overhang_at_crossing(approach=False, bias=0.0)
    assert overhang is not None
    assert overhang > 0.02, f'expected the legacy descent to overhang the mouth edge, got {overhang:+.4f}'


@need_urdf
def test_from_above_lands_carton_flat_inside_shelf():
    """The fix: approach-from-above lands the carton with its whole footprint INSIDE the shelf edge
    (flat, level). Pins a minimum lateral clearance to the +x (mouth) edge along the whole insert."""
    overhang, min_edge_clear = _front_overhang_at_crossing(approach=True, bias=0.0)
    assert overhang is not None
    assert overhang < 0.0, f'carton must land fully inside the shelf, got overhang {overhang:+.4f}'
    # min clearance of the carton front edge to the shelf mouth edge, while the carton is at/under the
    # shelf top -- the fix keeps the whole insert clear of the mouth edge by >= 3 cm.
    assert min_edge_clear > 0.03, f'min edge clearance too small: {min_edge_clear:+.4f}'


@need_urdf
def test_depth_bias_increases_edge_clearance():
    _, mec0 = _front_overhang_at_crossing(approach=True, bias=0.0)
    _, mec5 = _front_overhang_at_crossing(approach=True, bias=0.05)
    assert mec5 > mec0 + 0.03
