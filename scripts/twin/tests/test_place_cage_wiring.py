"""CPU tests for the boot-layer wiring of the cart cage-clearance gates (sim-place-descent-reach-wire-v1).

PR 366 / PR 376 shipped the providers (place_clearance.preplace_cage_clearance_provider /
.place_descent_cage_clearance_provider) and the controller hooks they feed, but NOTHING ever constructed
or injected them, so both hooks stayed None and the in-op gates never ran (coldwin90 repeated run 89/90
exactly with every switch on). These tests pin the glue that fixes that:

  * the wiring installs the real provider closures on the controller hooks from the HOST blocks, and the
    installed hook ROUTES through the arm-sphere sweep (a spy proves the provider is invoked) -- so a
    configured gate can never again be left uncalled;
  * absent blocks => the hooks stay None => byte-identical;
  * malformed blocks raise loudly (bare true / missing cart_pose);
  * ros_host.main() calls the wiring right after it builds the controller (fails on the merged code, which
    never wires it);
  * a guarded cw87/cw90 replay: the WIRED descent hook REFUSES the in-cage-reorient descent and PASSES the
    pad-down remedy descent (the run-worker's exact reproduction, now through the injected gate).

No Isaac / ROS / GPU. The spy path needs no USD/FK; the replay is guarded on usd-core + the dataset URDF/XRDF.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_cage_wiring.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation import place_cage_wiring as wiring
from twin.restock_manipulation import place_clearance as pc
from twin.restock_manipulation import TransferController, ProductProfile


PROFILE = ProductProfile('sku-box', (0.45, 0.30, 0.18), (0.5, 0.5, 0.5, 0.5),
                         max_reach=1.3, grasp_mode='top')

# A cart-frame collider dict pre-baked offline (so the glue tests need no usd-core): one CAGE body
# (post_2, a corner post) the sweep keeps + one SUPPORT body (deck) it excludes.
FAKE_BOXES = {
    'post_2': [[0.30, -0.10, 0.20], [0.37, -0.03, 0.80]],
    'deck': [[-0.10, -0.40, 0.18], [0.70, 0.20, 0.25]],
}


class _NullMotion:
    """The TransferController never touches the motion port at construction; this satisfies the type."""
    def poll(self):
        return 'running'


def _controller():
    return TransferController(_NullMotion(), {'sku-box': PROFILE}, fixture_mode=True)


# --------------------------------------------------------------------------- byte-identical off
def test_absent_blocks_leave_both_hooks_none():
    c = _controller()
    assert c._preplace_cage_clearance is None and c._place_descent_cage_gate is None
    armed = wiring.wire_place_cage_gates(c, {}, '/x/mobile.urdf', log=lambda *a, **k: None)
    assert armed == []
    assert c._preplace_cage_clearance is None and c._place_descent_cage_gate is None


def test_false_blocks_are_inert():
    c = _controller()
    armed = wiring.wire_place_cage_gates(
        c, {'preplace_cage_gate': False, 'place_descent_cage_gate': None},
        '/x/mobile.urdf', log=lambda *a, **k: None)
    assert armed == []
    assert c._place_descent_cage_gate is None


# --------------------------------------------------------------------------- malformed blocks raise loudly
def test_bare_true_is_rejected():
    c = _controller()
    with pytest.raises(ValueError, match='cart_pose'):
        wiring.wire_place_cage_gates(c, {'place_descent_cage_gate': True}, '/x/mobile.urdf',
                                     log=lambda *a, **k: None)


def test_block_without_cart_pose_raises():
    c = _controller()
    with pytest.raises(ValueError, match='cart_pose'):
        wiring.wire_place_cage_gates(c, {'preplace_cage_gate': {'margin': 0.03}}, '/x/mobile.urdf',
                                     log=lambda *a, **k: None)


# --------------------------------------------------------------------------- the installed hook is INVOKED
def _spy_sweep(records):
    def _sweep(configs, urdf_path, xrdf_path, bodies, T_base_cart, *, include=None):
        records.append({'configs': [list(w) for w in configs], 'urdf': urdf_path, 'xrdf': xrdf_path,
                        'include': list(include) if include is not None else None})
        # one penetrating cage body so the sweep refuses by name
        return {name: -0.02 for name in (include or [])}
    return _sweep


def test_descent_wiring_installs_hook_that_routes_through_the_sweep(monkeypatch):
    """The wired place-descent hook is the real provider closure, and calling it (as the controller
    dispatch does) INVOKES the arm-sphere sweep and refuses a penetrating descent by name. On the merged
    code this hook is None, so it is never called -- the exact defect this lane fixes."""
    records = []
    monkeypatch.setattr(pc, 'preplace_cage_arm_clearance', _spy_sweep(records))
    c = _controller()
    block = {'cart_pose': [0.0, 0.0, 0.0, 0.0], 'collider_boxes': FAKE_BOXES,
             'xrdf': '/x/mobile.xrdf', 'margin': 0.03}
    armed = wiring.wire_place_cage_gates(
        c, {'place_descent_cage_gate': block}, '/x/mobile.urdf', log=lambda *a, **k: None)
    assert armed == ['place_descent_cage_gate']
    assert c._place_descent_cage_gate is not None
    # empty waypoints short-circuit without a sweep
    assert c._place_descent_cage_gate([]) is None
    assert records == []
    # a real descent plan invokes the sweep and is refused by name
    reject = c._place_descent_cage_gate([[0.0] * 6, [0.1] * 6])
    assert len(records) == 1
    assert records[0]['xrdf'] == '/x/mobile.xrdf'
    assert records[0]['include'] == ['post_2']          # only the cage body, not the deck support
    assert reject is not None and reject.startswith(pc.PLACE_DESCENT_CAGE_CLEARANCE)
    assert 'prim=post_2' in reject


def test_descent_wiring_installs_min_clearance_closure(monkeypatch):
    """sim-place-pad-down-arrival-v1: wiring the descent gate also installs the raw min-clearance closure
    the pad-down-branch selector ranks by (it exposes the cage margin, not a pass/fail). None off."""
    records = []
    monkeypatch.setattr(pc, 'preplace_cage_arm_clearance', _spy_sweep(records))
    c = _controller()
    assert c._place_descent_cage_min_clearance is None            # byte-identical off
    block = {'cart_pose': [0.0, 0.0, 0.0, 0.0], 'collider_boxes': FAKE_BOXES, 'margin': 0.03}
    wiring.wire_place_cage_gates(c, {'place_descent_cage_gate': block}, '/x/mobile.urdf',
                                 log=lambda *a, **k: None)
    assert c._place_descent_cage_min_clearance is not None
    assert c._place_descent_cage_min_clearance([]) == float('inf')  # no waypoints => no constraint
    # the spy returns -0.02 for post_2 (a cage body) -> the min non-support clearance is -0.02
    clr = c._place_descent_cage_min_clearance([[0.0] * 6, [0.1] * 6])
    assert clr == pytest.approx(-0.02)
    assert records and records[-1]['include'] == ['post_2']


def test_preplace_wiring_installs_hook_that_routes_through_the_sweep(monkeypatch):
    records = []
    monkeypatch.setattr(pc, 'preplace_cage_arm_clearance', _spy_sweep(records))
    c = _controller()
    block = {'cart_pose': [0.0, 0.0, 0.0, 0.0], 'collider_boxes': FAKE_BOXES, 'margin': 0.0}
    armed = wiring.wire_place_cage_gates(
        c, {'preplace_cage_gate': block}, '/x/mobile.urdf', log=lambda *a, **k: None)
    assert armed == ['preplace_cage_gate']
    assert c._preplace_cage_clearance is not None
    reject = c._preplace_cage_clearance([[0.0] * 6])
    assert len(records) == 1 and records[0]['include'] == ['post_2']
    assert reject is not None and reject.startswith(pc.PREPLACE_SWEEP_CLEARANCE)


def test_xrdf_defaults_to_urdf_sibling(monkeypatch):
    records = []
    monkeypatch.setattr(pc, 'preplace_cage_arm_clearance', _spy_sweep(records))
    c = _controller()
    block = {'cart_pose': [0.0, 0.0, 0.0, 0.0], 'collider_boxes': FAKE_BOXES}
    wiring.wire_place_cage_gates(c, {'place_descent_cage_gate': block},
                                 '/data/host-cad/mobile.urdf', log=lambda *a, **k: None)
    c._place_descent_cage_gate([[0.0] * 6])
    assert records[0]['xrdf'] == '/data/host-cad/mobile.xrdf'


def test_both_gates_wired_together(monkeypatch):
    monkeypatch.setattr(pc, 'preplace_cage_arm_clearance', _spy_sweep([]))
    c = _controller()
    block = {'cart_pose': [0.0, 0.0, 0.0, 0.0], 'collider_boxes': FAKE_BOXES}
    armed = wiring.wire_place_cage_gates(
        c, {'preplace_cage_gate': block, 'place_descent_cage_gate': dict(block, margin=0.03)},
        '/x/mobile.urdf', log=lambda *a, **k: None)
    assert set(armed) == {'preplace_cage_gate', 'place_descent_cage_gate'}
    assert c._preplace_cage_clearance is not None and c._place_descent_cage_gate is not None


# --------------------------------------------------------------------------- ros_host calls the wiring
def test_ros_host_wires_the_gates_after_controller_construction():
    """ros_host.main() must call wire_place_cage_gates right after it builds the controller. On the merged
    code there is NO such call (the providers were never constructed), so this assertion fails there."""
    from twin.restock_manipulation import ros_host
    src = Path(ros_host.__file__).read_text()
    assert 'from .place_cage_wiring import wire_place_cage_gates' in src
    assert 'wire_place_cage_gates(controller, config, args.urdf)' in src
    # the wiring is after the controller exists
    assert src.index('controller = TransferController(') < src.index('wire_place_cage_gates(controller')


# --------------------------------------------------------------------------- guarded cw87/cw90 replay
_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'


def _usd_available():
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (_URDF.is_file() and _XRDF.is_file() and _usd_available()),
                    reason='dataset URDF/XRDF or usd-core absent')
def test_cw87_replay_through_the_wired_descent_gate():
    """Wire the place-descent gate exactly as ros_host does (via wire_place_cage_gates) at the coldwin87
    coupled cart pose, then replay the run worker's two descents THROUGH the injected hook: the
    in-cage-reorient OLD descent is REFUSED by name and the pad-down REMEDY descent (to the corrected seat)
    PASSES. This is the cw87/cw90 reproduction, now proving the gate is live once wired."""
    import importlib.util
    from twin.restock_manipulation import carton_level as cl
    from twin.urdf_probe import parse_urdf

    spec = importlib.util.spec_from_file_location(
        'check_place_descent_reach',
        Path(__file__).resolve().parents[2] / 'check_place_descent_reach.py')
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)

    robot = parse_urdf(str(_URDF))
    geom = pc.GripperCartonGeometry.shipped()
    bz = chk._base_link_z(robot, pc, geom)
    old = [chk._to_ur(p) for p in chk._CW87_DESCENT_PLAN]
    standoff = chk._to_ur(chk._CW87_DESCENT_PLAN[1])
    new, new_tilt, _bottom = chk.build_clear_descent(robot, pc, cl, geom, bz, standoff_vec=standoff)
    T = chk._t_base_cart(chk._PARKS['coldwin87'], bz)   # cart-body -> ur_base_link 4x4

    # wire the gate the way the boot layer does: the 4x4 coupled cart pose carried in the HOST block
    c = _controller()
    armed = wiring.wire_place_cage_gates(
        c, {'place_descent_cage_gate': {'cart_pose': T.tolist()}},
        str(_URDF), log=lambda *a, **k: None)
    assert armed == ['place_descent_cage_gate']
    gate = c._place_descent_cage_gate

    old_reject = gate(old)
    new_reject = gate(new)
    assert old_reject is not None and old_reject.startswith(pc.PLACE_DESCENT_CAGE_CLEARANCE), \
        'the wired gate must REFUSE the cw87 in-cage-reorient descent'
    assert 'cube_without_surface_plates' in old_reject
    assert new_reject is None, 'the wired gate must PASS the pad-down remedy descent'
    assert new_tilt <= 2.0 + 1e-9                       # the untouched carton-level bound
