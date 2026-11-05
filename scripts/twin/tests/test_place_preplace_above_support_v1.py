"""CPU tests for the support-aware collect pre-place standoff (sim-place-preplace-above-support-v1).

coldwin48 selected the Rollwagen UPPER shelf (resolved support top 0.9707 world) but the collect
pre-place standoff was still the DECK-tuned fixed ``collect_place_preplace_shape``, which parks the
carried carton BELOW the shelf top. So ``place_descent_drop_m`` read 0.0 (PLACE_CONTACT_RELEASE_ARM
DESCENT_DROP=0.0000), the contact-triggered weld release fired in mid-air and the carton fell
(verify_place: 'carton not resting', bottom_gap -0.809). With ``preplace_above_support`` on, the collect
preplace standoff is re-targeted DIRECTLY ABOVE the place load-centre so the carton bottom sits at the
resolved support top + the clearance, the from-above descent is POSITIVE and the release fires at the
true support top.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_preplace_above_support_v1.py
"""
import os

import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import cart_support
from twin.restock_manipulation import config_preflight as cp

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='dataset mobile URDF not present')
TCP = 'gripper_pad_tcp'

STOCK = ['/Rollwagen/chassis/deck', '/Rollwagen/chassis/shelf']
# coldwin48 geometry (ur_base_link): the resolved shelf place target and the fixed deck-tuned standoff.
CW48_PLACE = (0.957, 0.064, 0.538)            # REPOSITION_REREGISTER place=(0.957,0.064,0.538)
CW48_SHELF_TOP_WORLD = 0.9707                 # COLLECT_PLACE_SUPPORT_Z resolved support top
# collect4's upright clear-region fixed preplace shape (test_restock_manipulation._SL_PREPLACE_SHAPE),
# whose gripper_pad_tcp FK sits at z 0.5281 ur_base -- 34 mm below the shelf place target 0.538.
FIXED_SHAPE = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)
TOP_GRASP_WELD = (0.0,) * 6                    # the fixture's level weld up-axis


@pytest.fixture(scope='module')
def robot():
    from twin.urdf_probe import parse_urdf
    return parse_urdf(URDF)


# --------------------------------------------------------------- config resolve / validate

def test_resolve_above_support_defaults_clearance_to_place_above_clearance():
    lc = cl.resolve_level_config({'preplace_above_support': True})
    assert lc.preplace_above_support is True
    assert lc.preplace_support_clearance_m is None
    # default clearance == the from-above via clearance so the standoff and descent above-clearance match.
    assert lc.preplace_support_clearance() == pytest.approx(lc.place_above_clearance_m)
    assert lc.preplace_support_clearance() == pytest.approx(0.08)


def test_resolve_above_support_clearance_override():
    lc = cl.resolve_level_config({'preplace_above_support': True, 'preplace_support_clearance_m': 0.10})
    assert lc.preplace_support_clearance() == pytest.approx(0.10)


def test_resolve_off_is_byte_identical_default():
    assert cl.resolve_level_config(False) is None
    lc = cl.resolve_level_config(True)
    assert lc.preplace_above_support is False
    assert lc.preplace_support_clearance_m is None


def test_resolve_rejects_non_positive_clearance():
    with pytest.raises(ValueError):
        cl.resolve_level_config({'preplace_support_clearance_m': 0.0})
    with pytest.raises(ValueError):
        cl.resolve_level_config({'preplace_support_clearance_m': -0.05})


def test_preflight_resolves_above_support_keys_clean():
    host = {'carton_level': {'preplace_above_support': True, 'preplace_support_clearance_m': 0.08}}
    assert cp.check_carton_level(host) == []          # resolves clean, no refusal (distinct from above_place)


# --------------------------------------------------------------- pure descent helpers (cart_support)

def test_place_descent_from_support_is_the_clearance():
    standoff_bottom, descent = cart_support.place_descent_from_support(CW48_SHELF_TOP_WORLD, 0.08)
    assert standoff_bottom == pytest.approx(CW48_SHELF_TOP_WORLD + 0.08)
    assert descent == pytest.approx(0.08)             # the carton bottom falls exactly the clearance


def test_place_descent_problems_refuses_non_positive_descent():
    probs = cart_support.place_descent_problems(CW48_SHELF_TOP_WORLD, 0.0)
    assert probs and 'NON-POSITIVE' in probs[0]       # coldwin48's failure, named


def test_place_descent_problems_refuses_above_reach():
    probs = cart_support.place_descent_problems(CW48_SHELF_TOP_WORLD, 1.0)   # 1.97 m world standoff
    assert probs and 'ABOVE REACH' in probs[0]


def test_place_descent_problems_clean_for_the_shelf_clearance():
    assert cart_support.place_descent_problems(CW48_SHELF_TOP_WORLD, 0.08) == []


# --------------------------------------------------------------- preflight coherence (the coldwin48 config)

def _runtime(support_prim):
    cp_block = {'location': 'cart', 'load_center_body_xy': [0.135, 0.0],
                'support_z_world': 0.9712, 'free_dimensions': [1.0, 1.0, 1.0]}
    if support_prim is not None:
        cp_block['support_prim'] = support_prim
    return {'manipulation': {'cart_placement': cp_block}}


def test_preflight_refuses_shelf_selected_with_fixed_standoff():
    """The coldwin48 config: the shelf is selected but preplace_above_support is OFF -> the deck-tuned
    fixed standoff gives a non-positive descent (mid-air release). REFUSE before a GPU boot."""
    probs = cp.check_place_preplace_standoff({'carton_level': True}, _runtime('shelf'), stock_supports=STOCK)
    assert probs and 'preplace_above_support' in probs[0] and 'coldwin48' in probs[0]


def test_preflight_passes_shelf_with_above_support_on():
    host = {'carton_level': {'preplace_above_support': True}}
    assert cp.check_place_preplace_standoff(host, _runtime('shelf'),
                                            resolved_top_z=CW48_SHELF_TOP_WORLD, stock_supports=STOCK) == []


def test_preflight_shelf_on_absurd_clearance_refused_above_reach():
    host = {'carton_level': {'preplace_above_support': True, 'preplace_support_clearance_m': 1.0}}
    probs = cp.check_place_preplace_standoff(host, _runtime('shelf'),
                                             resolved_top_z=CW48_SHELF_TOP_WORLD, stock_supports=STOCK)
    assert probs and 'ABOVE REACH' in probs[0]


def test_preflight_deck_selected_is_byte_identical():
    """The DECK default surface keeps the fixed shape (historical, coldwin46): no refusal even with the
    flag off, and no requirement to turn preplace_above_support on."""
    assert cp.check_place_preplace_standoff({'carton_level': True}, _runtime('deck'), stock_supports=STOCK) == []
    assert cp.check_place_preplace_standoff({'carton_level': True}, _runtime(0), stock_supports=STOCK) == []


def test_preflight_no_support_prim_is_byte_identical():
    assert cp.check_place_preplace_standoff({'carton_level': True}, _runtime(None), stock_supports=STOCK) == []


# --------------------------------------------------------------- controller: the coldwin48 replay

def _collect_ctrl(carton_level):
    from twin.tests.test_restock_manipulation import make_controller, _SL_STOW, _SL_TUCK
    c, _m = make_controller(
        'collect', transport_joints=_SL_STOW, staged_cart_swing=True, cart_pan_joint0=-0.8605,
        cart_tuck_shape=_SL_TUCK, reposition_before_place=True,
        collect_place_preplace_shape=FIXED_SHAPE, collect_place_wrist_guard=True,
        collect_place_straight_line=True, carton_level=carton_level, fk_urdf=URDF,
        carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    c.poses = {'place': (*CW48_PLACE, 0.0, 0.0, 0.0, 1.0)}
    c._capture_level_up(TOP_GRASP_WELD)
    return c


@need_urdf
def test_coldwin48_fixed_shape_descent_is_zero(robot):
    """REPRODUCE coldwin48: with preplace_above_support OFF, the preplace standoff is the DECK-tuned fixed
    shape (gripper_pad_tcp z 0.528 ur_base), which is BELOW the shelf place target (z 0.538). The from-above
    descent drop from that standoff to the set-down at the support is 0.0 -- the mid-air release the brief
    names (PLACE_CONTACT_RELEASE_ARM DESCENT_DROP=0.0000)."""
    c = _collect_ctrl(cl.resolve_level_config({'trip': False}))
    assert c._preplace_above_place_goal(TOP_GRASP_WELD) is None          # no re-target -> the fixed shape
    fixed_tcp_z = float(cl.tool_position(robot, list(FIXED_SHAPE), tool_link=TCP)[2])
    assert fixed_tcp_z == pytest.approx(0.528, abs=0.01)                 # the deck-tuned standoff TCP z
    assert fixed_tcp_z < CW48_PLACE[2]                                   # BELOW the shelf place target 0.538
    # place_descent_drop_m is max(0, standoff_z - set_down_z): the standoff sits at/below the set-down, so
    # the descent drop is zero (a mid-air release), independent of droop noise in the set-down IK.
    drop = cl.place_descent_drop_m(robot, [list(FIXED_SHAPE), list(FIXED_SHAPE)], tool_link=TCP)
    assert drop == pytest.approx(0.0, abs=1e-9)
    assert max(0.0, fixed_tcp_z - CW48_PLACE[2]) == pytest.approx(0.0)   # straight-down drop onto the shelf


@need_urdf
def test_coldwin48_above_support_descent_is_positive_to_the_shelf(robot):
    """THE FIX: with preplace_above_support ON, the preplace standoff is re-targeted so the carton bottom
    sits at the resolved shelf top + clearance; the from-above descent onto the shelf is POSITIVE (~the
    clearance) and the contact release fires at the true support top, not in mid-air."""
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    c = _collect_ctrl(cl.resolve_level_config({'trip': False, 'preplace_above_support': True,
                                               'place_from_above': True}))
    standoff = c._preplace_above_place_goal(TOP_GRASP_WELD)
    assert standoff is not None                                          # the support-aware lane engaged
    assert c._preplace_above_mode == 'support'
    assert c._preplace_above_clearance == pytest.approx(0.08)
    # the standoff TCP is ABOVE the shelf place target (a real descent to come), not below it.
    standoff_tcp_z = float(cl.tool_position(robot, list(standoff), tool_link=TCP)[2])
    assert standoff_tcp_z > CW48_PLACE[2] + 0.05
    up = c._carton_up_in_tool
    setdown = cl.level_place_ik(robot, CW48_PLACE, up, list(standoff), tool_link=TCP,
                               limits=c._place_level_limits(), gravity_comp=c._place_gravity_compensation())
    setvec = [setdown[j] for j in ARM_JOINTS]
    drop = cl.place_descent_drop_m(robot, [list(standoff), setvec], tool_link=TCP)
    assert drop > 0.05                                                   # POSITIVE descent (was 0.0)
    # the standoff is level (carried carton parallel to the ground) and reachable on the clear branch.
    roll, pitch, _ = cl.carried_rpy(robot, list(standoff), up, TCP)
    assert abs(roll) < 2.0 and abs(pitch) < 2.0


@need_urdf
def test_deck_case_unchanged_when_no_flag(robot):
    """The deck geometry (coldwin46): with preplace_above_support OFF the standoff is the fixed shape,
    byte-identical -- _preplace_above_place_goal returns None regardless of the place pose."""
    c = _collect_ctrl(cl.resolve_level_config({'trip': False}))
    # a LOW deck place target (coldwin46 deck top ~0.2565 world): the fixed shape is far above it already.
    c.poses = {'place': (0.957, 0.064, -0.15, 0.0, 0.0, 0.0, 1.0)}
    assert c._preplace_above_place_goal(TOP_GRASP_WELD) is None
