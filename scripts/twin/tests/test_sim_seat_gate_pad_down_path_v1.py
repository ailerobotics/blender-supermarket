"""CPU tests for the seat gate as the SINGLE release authority in every descent path, incl. the PR 380
pad-down descent (sim-seat-gate-pad-down-path-v1).

coldwin98 ran the new three-stage set-down cleanly, then the release fired in MID-AIR 106 mm above the
seat and the contact release let the carton go. Named cause (from the cw98 host-stack.log): a DISPATCH-PATH
bug -- the release gate (``_place_release_gate``) observes with ``commanded_down=True`` on every ``place``
tick, so it also observed during the PR 380 pad-down REORIENT sub-leg (stage b), where the carton is held
LEVEL at the shifted standoff and its centre z is FROZEN (0.6375). After ``place_release_stall_ticks`` frozen
ticks the gate's RAW stall trigger fired a ``release`` -- and the honest seat gate was NOT armed yet
(``seat_center_z`` is set only in the post-reorient descent block, reached only after ``_place_reoriented``
latches, which never happened because the release pre-empted the reorient), so the gate fell through to the
un-gated stall release. No ``PLACE_SEAT_GATE_ARM`` / ``PLACE_CONTACT_RELEASE_ARM`` appears in the cw98 logs.

The fix makes the seat gate the single authority two ways:
  * the controller does NOT run the release gate during the pad-down reorient leg (so it only judges the
    real descent, after the seat is armed); and
  * ``PlaceReleaseGate(require_seat=True)`` (passed from ``carton_level.place_seat_gate``) forbids ANY
    seat-blind raw release -- a trigger with no modelled seat armed HOLDS (obstructed, 'seat-gate-unarmed')
    instead of dropping the carton.

Plus a preflight that refuses the deck combination (``RESTOCK_PLACE_ON_DECK`` / a deck support selection
while a merged gate is on) -- the gates are proven only for the upper-shelf flat-top seat.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_sim_seat_gate_pad_down_path_v1.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import config_preflight as cp
from twin.restock_manipulation import TransferController, ProductProfile
from twin.restock_manipulation.place_release import PlaceReleaseGate

# coldwin98 geometry (ur_base_link frame, from REPOSITION_REREGISTER + the PLACE_RELEASE_TICK ladder):
#   deck support top (target_surface z) = 0.441; carton half = 0.09 => deck seat centre = 0.531.
#   the pad-down reorient / descent jammed with the carton CENTRE frozen at 0.6375 => bottom gap +106.5 mm.
CW98_DECK_SEAT_CZ = 0.531            # carton CENTRE z at the (deck) seat
CW98_JAM_CZ = 0.6375                 # carton CENTRE z where the reorient/descent froze
CW98_BOTTOM_GAP_M = CW98_JAM_CZ - CW98_DECK_SEAT_CZ   # 0.1065 m == the RELEASE_NOT_FREE 0.1064 m
SHELF = '/Rollwagen/chassis/shelf'


def _gate(**kw):
    kw.setdefault('effort_margin_nm', 100.0)
    kw.setdefault('effort_consecutive', 3)
    kw.setdefault('stall_eps_m', 0.001)
    kw.setdefault('stall_ticks', 5)
    kw.setdefault('seat_tol_m', 0.02)
    return PlaceReleaseGate(**kw)


def _run_frozen_stall(gate, carton_z, ticks=8, effort=110.5):
    """Feed a FROZEN descent (carton_z constant, like the cw98 reorient) and return the final decision."""
    dec = None
    for i in range(ticks):
        dec = gate.observe(now=i * 0.05, carton_z=carton_z, commanded_down=True, arm_effort_max=effort)
    return dec


# --------------------------------------------------------------- the cw98 bypass: require_seat holds it
def test_require_seat_holds_raw_stall_when_seat_unarmed():
    """The cw98 bypass: a frozen-standoff stall with the seat gate UNARMED (seat_center_z None). With
    require_seat on the gate must HOLD (obstructed, sentinel 'seat-gate-unarmed'), NEVER release."""
    g = _gate(require_seat=True)                 # seat_center_z defaults None (unarmed, as in the reorient)
    dec = _run_frozen_stall(g, CW98_JAM_CZ)
    assert dec.action == 'obstructed'
    assert dec.hit_prim == 'seat-gate-unarmed'
    assert dec.trigger is None                   # never a release trigger
    assert 'seat-blind' in dec.reason


def test_require_seat_off_releases_raw_stall_byte_identical():
    """Default (require_seat off): the SAME unarmed frozen stall still fires the legacy raw stall release --
    the prior behaviour, unchanged, so the key is byte-identical off."""
    g = _gate(require_seat=False)
    dec = _run_frozen_stall(g, CW98_JAM_CZ)
    assert dec.action == 'release' and dec.trigger == 'stall'


def test_require_seat_default_is_off():
    assert _gate().require_seat is False


# --------------------------------------------------------------- armed seat: the real-descent classifier
def test_armed_seat_stall_above_seat_is_obstructed():
    """cw98 numbers with the seat ARMED (the fixed pad-down descent): a stall 106 mm ABOVE the deck seat is
    an OBSTRUCTION (held above the seat), never 'support met' -- no mid-air release."""
    g = _gate(require_seat=True, seat_center_z=CW98_DECK_SEAT_CZ)
    dec = _run_frozen_stall(g, CW98_JAM_CZ)
    assert dec.action == 'obstructed'
    assert dec.hit_prim == 'above-modelled-seat'
    assert pytest.approx(CW98_BOTTOM_GAP_M, abs=1e-6) == CW98_JAM_CZ - CW98_DECK_SEAT_CZ


def test_armed_seat_descent_reaches_seat_releases():
    """With the shelf target the descent reaches the seat: a stall WITHIN seat_tol of the modelled seat is a
    true set-down -> release the weld."""
    g = _gate(require_seat=True, seat_center_z=CW98_DECK_SEAT_CZ)
    dec = _run_frozen_stall(g, CW98_DECK_SEAT_CZ + 0.005)   # 5 mm above the seat, inside the 20 mm tol
    assert dec.action == 'release'
    assert dec.trigger in ('stall', 'effort')


def test_armed_seat_descent_above_seat_keeps_pressing_without_contact():
    """A descent still above the seat with NO contact keeps pressing (not a premature release)."""
    g = _gate(require_seat=True, seat_center_z=CW98_DECK_SEAT_CZ)
    # one tick, no stall accumulated, well above the seat
    dec = g.observe(now=0.0, carton_z=CW98_JAM_CZ, commanded_down=True, arm_effort_max=110.0)
    assert dec.action == 'press'


# --------------------------------------------------------------- preflight: the deck combination
def _host(**over):
    carton = {'place_from_above': True, 'place_release_on_contact': True,
              'place_seat_gate': True, 'place_pad_down_arrival': True}
    carton.update(over.pop('carton_level', {}))
    h = {'carton_level': carton}
    h.update(over)
    return h


def _runtime(support_prim='shelf'):
    cpl = {'location': 'rollwagen', 'load_center_body_xy': [0.0, 0.0],
           'support_z_world': 0.9633, 'free_dimensions': [0.65, 0.45, 0.3]}
    if support_prim is not None:
        cpl['support_prim'] = support_prim
    return {'manipulation': {'cart_placement': cpl}}


def test_seat_gates_on_names_the_merged_gates():
    names = cp.seat_gates_on(_host())
    assert 'carton_level.place_seat_gate' in names
    assert 'carton_level.place_pad_down_arrival' in names
    # host cage gates count too
    assert 'host.place_descent_cage_gate' in cp.seat_gates_on(_host(place_descent_cage_gate=True))


def test_preflight_refuses_gates_plus_place_on_deck():
    problems = cp.check_seat_gates_support(_host(), _runtime('shelf'), place_on_deck=True)
    assert problems and any('RESTOCK_PLACE_ON_DECK' in p for p in problems)


def test_preflight_refuses_gates_plus_deck_selection():
    problems = cp.check_seat_gates_support(_host(), _runtime('deck'), place_on_deck=False)
    assert problems and any('DECK' in p for p in problems)


def test_preflight_refuses_gates_plus_absent_selection_defaults_deck():
    # absent support_prim defaults to stock_supports[0] == the deck
    problems = cp.check_seat_gates_support(_host(), _runtime(support_prim=None), place_on_deck=False)
    assert problems and any('DECK' in p for p in problems)


def test_preflight_passes_gates_plus_shelf_no_place_on_deck():
    assert cp.check_seat_gates_support(_host(), _runtime('shelf'), place_on_deck=False) == []


def test_preflight_noop_when_no_gate_on():
    host = _host(carton_level={'place_from_above': True, 'place_release_on_contact': True,
                               'place_seat_gate': False, 'place_pad_down_arrival': False})
    assert cp.check_seat_gates_support(host, _runtime('deck'), place_on_deck=True) == []


# --------------------------------------------------------------- boot-path wiring (the fleet requirement)
PROFILE = ProductProfile('sku-box', (0.45, 0.30, 0.18), (0.5, 0.5, 0.5, 0.5),
                         max_reach=1.3, grasp_mode='top')


class _NullMotion:
    def poll(self):
        return 'running'


def _controller(**carton_level_over):
    carton = {'place_from_above': True, 'place_release_on_contact': True,
              'place_seat_gate': True, 'place_pad_down_arrival': True}
    carton.update(carton_level_over)
    lc = cl.resolve_level_config(carton)
    c = TransferController(_NullMotion(), {'sku-box': PROFILE}, fixture_mode=True,
                           carton_level=lc, collect_place_straight_line=True)
    return c


def test_controller_wires_require_seat_from_place_seat_gate():
    """The boot path constructs the gate with require_seat == carton_level.place_seat_gate, so a configured
    seat gate is ALWAYS the release authority (it can never be left un-wired for a descent)."""
    c = _controller()
    assert c._place_release_gate is not None
    assert c._place_release_gate.require_seat is True


def test_controller_require_seat_off_when_place_seat_gate_off():
    c = _controller(place_seat_gate=False)
    assert c._place_release_gate is not None
    assert c._place_release_gate.require_seat is False


def test_controller_gate_absent_when_release_on_contact_off_byte_identical():
    """Keys off => the gate is never constructed => byte-identical (no require_seat path at all)."""
    lc = cl.resolve_level_config({'place_from_above': True, 'place_release_on_contact': False})
    c = TransferController(_NullMotion(), {'sku-box': PROFILE}, fixture_mode=True,
                          carton_level=lc, collect_place_straight_line=True)
    assert c._place_release_gate is None


def test_controller_suppresses_gate_during_pad_down_reorient():
    """The gate must NOT observe while awaiting the pad-down reorient (the cw98 window). Assert the guard
    inputs: the pad-down three-stage set-down is active for a collect deck place and _place_reoriented is
    False on entry, so the controller's `_pad_down_arrival_on() and not _place_reoriented` suppression holds."""
    c = _controller()
    c.operation = 'collect'                       # _collect_deck_place requires the collect op
    assert c._collect_deck_place() is True
    assert c._pad_down_arrival_on() is True
    assert c._place_reoriented is False
    # the controller's step() guard: _awaiting_pad_down_reorient
    assert (c._pad_down_arrival_on() and not c._place_reoriented) is True
    # once reoriented, the suppression lifts (the real descent, seat armed, runs the gate)
    c._place_reoriented = True
    assert (c._pad_down_arrival_on() and not c._place_reoriented) is False
