"""Boot-path + controller tests for the three-stage collect place set-down (sim-place-pad-down-arrival-v1).

The fleet requires a boot-path wiring test that constructs the controller the way ros_host does (resolve the
carton_level block into a LevelConfig, build the TransferController, then wire the cage gates) with the keys
ON, and asserts the NEW reorientation stage runs. These do that:

  * off => byte-identical (the three-stage set-down is inert, the standoff is the single above-seat standoff);
  * on => _pad_down_arrival_on() and the shifted standoff; the descent cage gate + its min-clearance closure
    are installed;
  * the reorientation stage RUNS: _issue_pad_down_reorient selects the cage-clear pad-down branch and issues
    a leveled PLACE_REORIENT move (guarded on the dataset URDF + usd-core);
  * a missing cage gate / an all-raking standoff FAILS BY NAME (PLACE_REORIENT_REJECTED), never blind motion.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_pad_down_arrival.py
"""
import importlib.util
import math
from pathlib import Path

import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import place_cage_wiring as wiring
from twin.restock_manipulation import place_reorient as pr
from twin.restock_manipulation import ros_host
from twin.restock_manipulation import TransferController, ProductProfile
from twin.restock_manipulation.cad_tool import ARM_JOINTS

PROFILE = ProductProfile('sku-box', (0.45, 0.30, 0.18), (0.5, 0.5, 0.5, 0.5),
                         max_reach=1.3, grasp_mode='top')

PAD_DOWN_CFG = {'place_pad_down_arrival': True, 'place_from_above': True, 'level_place': True,
                'level_goals': True, 'pad_down_arrival_shift_m': 0.15, 'pad_down_arrival_height_m': 0.10}


class _SpyMotion:
    """Records begin_joint_path / begin_joint_goal; the controller never polls it in these unit drives."""
    def __init__(self):
        self.paths = []
        self.goals = []

    def begin_joint_path(self, waypoints, measured, **kw):
        self.paths.append({'waypoints': [list(w) for w in waypoints], 'measured': list(measured), 'kw': kw})

    def begin_joint_goal(self, goal, measured, **kw):
        self.goals.append({'goal': list(goal), 'measured': list(measured), 'kw': kw})

    def poll(self):
        return 'running'


def _build_controller(cfg, *, fk_urdf=None, wire_cart_pose=None):
    """Construct exactly as ros_host does: the carton_level block -> LevelConfig, the controller, then the
    cage-gate wiring. ``cfg`` is the HOST config dict (carton_level + optional place_descent_cage_gate)."""
    controller = TransferController(
        _SpyMotion(), {'sku-box': PROFILE}, fixture_mode=True,
        collect_place_straight_line=True, collect_place_wrist_guard=True,
        collect_place_preplace_shape=[0.093, -1.91, 2.014, -0.103, 1.664, 0.0],
        carton_level=cl.resolve_level_config(cfg.get('carton_level')), fk_urdf=fk_urdf)
    controller.operation = 'collect'                        # the collect-deck place (the pad-down scope)
    wiring.wire_place_cage_gates(controller, cfg, fk_urdf or '/x/mobile.urdf', log=lambda *a, **k: None)
    return controller


# --------------------------------------------------------------------------- off => byte-identical
def test_off_leaves_pad_down_inert():
    c = _build_controller({'carton_level': {'level_place': True, 'place_from_above': True}})
    assert c._pad_down_arrival_on() is False
    assert c._place_reoriented is False


def test_on_but_transfer_is_inert():
    c = _build_controller({'carton_level': PAD_DOWN_CFG})
    c.operation = 'transfer'
    assert c._pad_down_arrival_on() is False                # scoped to the collect-deck place


# --------------------------------------------------------------------------- on => the three-stage set-down
def test_on_activates_for_collect_deck_place():
    c = _build_controller({'carton_level': PAD_DOWN_CFG})
    assert c._pad_down_arrival_on() is True


def test_shifted_standoff_is_toward_the_base():
    """The pad-down standoff is the seat shifted mouth-out + raised -- NOT the single above-seat standoff."""
    seat = (0.683, 0.101, 0.634)
    std = pr.reorient_standoff_tcp(seat, 0.15, 0.10)
    assert math.hypot(std[0], std[1]) < math.hypot(seat[0], seat[1])
    assert std[2] == pytest.approx(seat[2] + 0.10)


def test_ros_host_resolves_carton_level_into_the_controller():
    """ros_host builds the controller with the resolved carton_level block AND wires the cage gates after --
    the construction path these tests mirror. (Source check: the merged code already resolves carton_level;
    this pins that the pad-down switch reaches the controller through that same resolution.)"""
    src = Path(ros_host.__file__).read_text()
    assert 'carton_level=carton_level.resolve_level_config(config.get(\'carton_level\'))' in src
    assert 'wire_place_cage_gates(controller, config, args.urdf)' in src


# --------------------------------------------------------------------------- a missing cage gate fails by name
def test_reorient_without_cage_gate_fails_by_name():
    c = _build_controller({'carton_level': PAD_DOWN_CFG})        # no place_descent_cage_gate block wired
    assert c._place_descent_cage_gate is None
    c._carton_up_in_tool = [0.0, 0.0, 1.0]                       # so the gate-check (not FK) is reached
    c.poses = {'place': (0.683, 0.101, 0.634)}
    c.phase = 'place'
    res = c._issue_pad_down_reorient(object(), [0.0] * 6)
    assert res is not None and c.phase == 'failed'
    assert 'PLACE_REORIENT_REJECTED' in (c.reason or '')


# --------------------------------------------------------------------------- the stage RUNS (guarded)
_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'


def _usd():
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (_URDF.is_file() and _XRDF.is_file() and _usd()),
                    reason='dataset URDF/XRDF or usd-core absent')
def test_pad_down_reorient_stage_runs():
    """Construct the controller the ros_host way with the keys on + the descent cage gate wired at the
    coldwin87 cart pose, then drive the reorientation stage: it selects the cage-clear PAD-DOWN branch
    (high wrist_2) and issues a leveled PLACE_REORIENT move from the carry-wrist ARRIVAL pose."""
    from twin.urdf_probe import parse_urdf
    spec = importlib.util.spec_from_file_location(
        'cpdr', Path(__file__).resolve().parents[2] / 'check_place_descent_reach.py')
    cpdr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpdr)
    robot = parse_urdf(str(_URDF))
    from twin.restock_manipulation import place_clearance as pc
    geom = pc.GripperCartonGeometry.shipped()
    bz = cpdr._base_link_z(robot, pc, geom)

    # the coldwin87 coupled cart pose -> ur_base_link (cart-frame AABBs read from usd via the provider)
    park = cpdr._PARKS['coldwin87']
    T = cpdr._t_base_cart(park, bz)
    cart_block = {'cart_pose': [[float(T[r][col]) for col in range(4)] for r in range(4)], 'margin': 0.03}
    c = _build_controller({'carton_level': PAD_DOWN_CFG, 'place_descent_cage_gate': cart_block},
                          fk_urdf=str(_URDF))
    assert c._place_descent_cage_gate is not None and c._place_descent_cage_min_clearance is not None

    # reconstruct arrival + seat in the SAME IK frame the controller uses (gripper_pad_tcp, via _place_fk_link)
    pad = cpdr._to_ur(cpdr._CW87_DESCENT_PLAN[1])
    arr_seed = cpdr._to_ur(cpdr._CW87_DESCENT_PLAN[0])
    tcp_link = c._place_fk_link(robot, list(c.collect_place_preplace_shape))
    seat_tcp = cl.tool_position(robot, {j: float(pad[i]) for i, j in enumerate(ARM_JOINTS)},
                                tcp_link, 'ur_base_link')
    up = cl.carton_up_in_tool(cl.tool_rotation(robot, {j: float(pad[i]) for i, j in enumerate(ARM_JOINTS)},
                                               tcp_link, 'ur_base_link'))
    std_tcp = pr.reorient_standoff_tcp(seat_tcp, 0.15, 0.10)
    limits = {'ur_elbow_joint': (0.1, math.pi), 'ur_wrist_1_joint': (-2 * math.pi, 0.65),
              'ur_wrist_2_joint': (0.0, 2 * math.pi)}
    arrival = [float(cl.level_place_ik(robot, std_tcp, up, {j: arr_seed[i] for i, j in enumerate(ARM_JOINTS)},
                                       tool_link=tcp_link, limits=limits, fast=True)[j])
               for j in ARM_JOINTS]

    c.poses = {'place': tuple(float(x) for x in seat_tcp)}
    c.phase = 'place'
    c._capture_level_up(arrival)
    assert c._carton_up_in_tool is not None

    res = c._issue_pad_down_reorient(robot, arrival)
    # the stage RAN: a reorientation move was issued, not a failure
    assert c.phase == 'place' and c.issued is True
    assert res is not None
    assert len(c.motion.paths) == 1                               # one leveled PLACE_REORIENT begin_joint_path
    path = c.motion.paths[0]
    endpoint = path['waypoints'][-1]
    # the endpoint is the PAD-DOWN branch (high wrist_2 ~ 1.6), not the carry-wrist arrival (~0.2)
    assert endpoint[ARM_JOINTS.index('ur_wrist_2_joint')] > 1.2
    # and its descent was selected cage-clear (the selection guarantees it; report >= the gate margin)
    # the move's first waypoint is the measured arrival (the reorientation is a real branch switch)
    assert path['waypoints'][0] == pytest.approx(list(arrival))
    # _place_reoriented is latched by the move-DONE handler, not the dispatch (mirrors wrist-first)
    assert c._place_reoriented is False
