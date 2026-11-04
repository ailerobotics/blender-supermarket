"""CPU tests for the collect PLACE support-height single source of truth (sim-place-support-z-consistency-v1).

Root cause (CONFIRMED from coldwin43/45 logs + the mission.usda geometry): the cold ground-truth collect
aimed the place at the warm-shortcut SHELF slot (support_z_world 0.9325 world = the Rollwagen upper shelf,
measured top 0.964 world) instead of the intended DECK (/Rollwagen/chassis/deck, measured top 0.2565
world). So the planned set-down and the contact-triggered weld-release threshold sat ~0.68 m ABOVE the real
deck: the release fired in mid-air and the carton fell off the cart (verify_place: 'carton not resting on
deck', final carton z ~-0.22). Moving the place-point xy changed nothing (<2 cm) -- the HEIGHT was wrong.

Measured world heights (coldwin45 mission.usda; ur_base_link world z 0.4999):
    deck prim origin 0.2315 | deck TOP face 0.2565 | shelf centre 0.9515 | shelf TOP 0.964 | legacy 0.9325.
In ur_base_link the legacy shelf set-down is 0.9325 - 0.4999 + 0.09 = 0.5226 (= the "carton released at
~0.52" the integration worker observed), ~0.68 m above the deck. The surface the carton pressed at ~0.52
in coldwin39-42 is the upper SHELF (top 0.964 world), never the 0.2565 deck.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_support_z.py
"""
import math
import os

import pytest

from twin.restock_manipulation import cart_support
from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import config_preflight as cp
from twin.restock_manipulation import ground_truth_collect as gtc

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='dataset mobile URDF not present')
TCP = 'gripper_pad_tcp'

# --- measured truth (world z), from coldwin45 mission.usda /Rollwagen (cart at the station) ---
DECK_TOP_WORLD = 0.2565          # deck cube: chassis z 0.23148 + half-thickness 0.025
SHELF_TOP_WORLD = 0.964          # shelf cube: chassis + 0.72 + half-thickness 0.0125
LEGACY_SHELF_CONST = 0.9325      # scenario cart_support_z_world / GT place_slot / place_select default
UBL_WORLD_Z = 0.4999             # ur_base_link world z (robot root 0.24 + ur_base_link local 0.2599)
CARTON_H = 0.18                  # handling unit height (single source: handling_unit.dimensions_m)
HALF = CARTON_H / 2.0


# --------------------------------------------------------------- the truth table (documents the numbers)

def test_measured_heights_reconcile_the_023_052_09325_puzzle():
    """0.23 is the deck prim ORIGIN; the deck TOP face is 0.2565. 0.52 is the carton-centre z in
    ur_base_link where the legacy shelf set-down released it (= 0.9325 - 0.4999 + half-height), ~0.68 m
    ABOVE the deck. 0.9325 is the legacy shelf constant. They are different surfaces, not one number."""
    legacy_setdown_ubl = LEGACY_SHELF_CONST - UBL_WORLD_Z + HALF
    assert legacy_setdown_ubl == pytest.approx(0.5226, abs=1e-3)        # the "released at ~0.52" value
    # the legacy set-down is ~0.68 m above the deck top (a mid-air release over the deck)
    carton_bottom_world = legacy_setdown_ubl + UBL_WORLD_Z - HALF       # = 0.9325 (the shelf constant)
    assert carton_bottom_world == pytest.approx(LEGACY_SHELF_CONST, abs=1e-6)
    assert carton_bottom_world - DECK_TOP_WORLD == pytest.approx(0.676, abs=2e-3)
    # the surface pressed at ~0.52 in cw39-42 is the upper SHELF (top 0.964), within a carton height of it
    assert abs(carton_bottom_world - SHELF_TOP_WORLD) < CARTON_H


# --------------------------------------------------------------- cart_support pure helpers

def test_support_top_z_world_is_the_max_aabb_z():
    assert cart_support.support_top_z_world(0.2065, 0.2565) == pytest.approx(DECK_TOP_WORLD)
    assert cart_support.support_top_z_world(0.2565, 0.2065) == pytest.approx(DECK_TOP_WORLD)
    with pytest.raises(ValueError):
        cart_support.support_top_z_world(float('nan'), 0.25)


def test_resting_carton_center_z():
    assert cart_support.resting_carton_center_z(DECK_TOP_WORLD, CARTON_H) == pytest.approx(0.3465)


def test_override_slot_support_z_keeps_xy_and_yaw():
    slot = {'load_center_body_xy': (0.18, 0.0), 'support_z_world': LEGACY_SHELF_CONST,
            'yaw_cart_rel': -math.pi / 2}
    out = cart_support.override_slot_support_z(slot, DECK_TOP_WORLD)
    assert out['support_z_world'] == pytest.approx(DECK_TOP_WORLD)
    assert out['load_center_body_xy'] == (0.18, 0.0)
    assert out['yaw_cart_rel'] == pytest.approx(-math.pi / 2)
    assert slot['support_z_world'] == LEGACY_SHELF_CONST                # original untouched (copy)
    with pytest.raises(ValueError):
        cart_support.override_slot_support_z(slot, float('inf'))


def test_height_consistency_agree_and_disagree():
    # all heights at the deck top -> no problems
    assert cart_support.height_consistency_problems(
        DECK_TOP_WORLD, {'runtime': 0.2565, 'scenario': 0.2560, 'gt': 0.2570}, tol_m=0.010) == []
    # the legacy shelf constant disagrees by ~0.68 m -> refused, and the message names both values
    probs = cart_support.height_consistency_problems(
        DECK_TOP_WORLD, {'runtime cart_placement.support_z_world': LEGACY_SHELF_CONST}, tol_m=0.010)
    assert len(probs) == 1 and '0.9325' in probs[0] and '0.2565' in probs[0]
    # a None height is skipped (not configured)
    assert cart_support.height_consistency_problems(
        DECK_TOP_WORLD, {'runtime': None}, tol_m=0.010) == []


# --------------------------------------------------------------- the slot-pose retarget (the runtime fix)

def _ident_transforms():
    """A cart and a base both level at the world origin, with the base lifted to ur_base_link world z, so
    gtc.slot_pose's z maps world support_z -> ubl support z = support_z - UBL_WORLD_Z."""
    import numpy as np
    cart = np.eye(4)
    base = np.eye(4)
    base[2, 3] = UBL_WORLD_Z
    return cart, base


def test_slot_pose_z_follows_the_overridden_support_height():
    """Overriding the slot's support_z_world shifts the place target z by exactly the height delta -- so the
    deck-retargeted slot lands the place on the deck, 0.68 m below the legacy shelf slot."""
    cart, base = _ident_transforms()
    shelf_slot = {'load_center_body_xy': (0.18, 0.0), 'support_z_world': LEGACY_SHELF_CONST,
                  'yaw_cart_rel': 0.0}
    deck_slot = cart_support.override_slot_support_z(shelf_slot, DECK_TOP_WORLD)
    z_shelf = gtc.slot_pose(cart, base, shelf_slot)[2]
    z_deck = gtc.slot_pose(cart, base, deck_slot)[2]
    assert z_shelf == pytest.approx(LEGACY_SHELF_CONST - UBL_WORLD_Z, abs=1e-6)
    assert z_deck == pytest.approx(DECK_TOP_WORLD - UBL_WORLD_Z, abs=1e-6)
    assert (z_shelf - z_deck) == pytest.approx(LEGACY_SHELF_CONST - DECK_TOP_WORLD, abs=1e-6)  # ~0.676 m


# --------------------------------------------------------------- replay: cw45 false contact vs true contact

def test_coldwin45_release_threshold_shelf_is_mid_air_deck_is_true_contact():
    """Replay of the coldwin45 set-down with the MEASURED frame. The contact-release fires when the live
    carton centre falls to place_contact_release_z(standoff, drop, gap). With the OLD shelf target the drop
    is tiny (~15 mm) so it fires at ~0.52 ubl -- 0.68 m above the deck, a false contact (carton falls). With
    the deck target the drop is the real ~0.68 m and it fires at the deck top within the release gap."""
    standoff_ubl = 0.527                                              # measured preplace carton centre (cw43/45)
    gap = 0.012
    shelf_rest = LEGACY_SHELF_CONST - UBL_WORLD_Z + HALF             # carton centre resting on the shelf slot
    deck_rest = DECK_TOP_WORLD - UBL_WORLD_Z + HALF                  # carton centre resting on the deck
    drop_shelf = standoff_ubl - shelf_rest
    drop_deck = standoff_ubl - deck_rest
    assert drop_shelf < 0.05                                          # the stale "15 mm" descent
    assert drop_deck > 0.60                                           # a real full descent to the deck

    rel_shelf = cl.place_contact_release_z(standoff_ubl, drop_shelf, gap)
    rel_deck = cl.place_contact_release_z(standoff_ubl, drop_deck, gap)
    # carton BOTTOM (world) at the moment of release:
    bottom_shelf = rel_shelf + UBL_WORLD_Z - HALF
    bottom_deck = rel_deck + UBL_WORLD_Z - HALF
    # OLD: released ~0.68 m above the real deck top -> mid-air -> carton falls
    assert bottom_shelf - DECK_TOP_WORLD > 0.60
    # NEW: released within the release gap of the real deck top -> TRUE contact
    assert bottom_deck == pytest.approx(DECK_TOP_WORLD + gap, abs=2e-3)


# --------------------------------------------------------------- descent-to-contact with the real FK

@need_urdf
def test_place_descent_drop_measures_the_full_descent_not_a_fixed_15mm(robot):
    """With the deck target the planned descent is the FULL drop from the standoff to the deck, not a fixed
    ~15 mm. place_descent_drop_m reads the real TCP drop between the standoff and the set-down waypoint."""
    # a standoff config and a set-down ~0.5 m lower (shoulder_lift fold), both from the cw39 place family.
    standoff = [-0.094, -1.0, 1.2, -0.2526, 1.6056, 0.0001]
    setdown = [-0.094, -1.0 + 0.6, 1.2, -0.2526, 1.6056, 0.0001]     # lower the TCP by folding shoulder_lift
    drop = cl.place_descent_drop_m(robot, [standoff, setdown], tool_link=TCP)
    assert drop > 0.30, drop                                          # a real descent, not 15 mm
    # the release threshold then lands within `gap` of the set-down height (true contact), not at the standoff.
    start_z, gap = 0.95, 0.012
    rz = cl.place_contact_release_z(start_z, drop, gap)
    assert (start_z - drop) < rz < (start_z - drop + 2 * gap)


@pytest.fixture(scope='module')
def robot():
    from twin.urdf_probe import parse_urdf
    return parse_urdf(URDF)


# --------------------------------------------------------------- verify_place on the resolved deck height

def test_verify_place_passes_on_the_resolved_deck_height_and_fails_on_the_shelf_setdown():
    """verify_place checks the carton rests on target_surface (= the resolved place target). With the deck
    as the support, a carton resting on the deck top passes; a carton left at the legacy shelf set-down
    (what the bug released) is far above the deck -> hard-fail."""
    from twin.tests.test_restock_manipulation import make_controller, run_to, packet
    # resolved deck top in ur_base_link
    deck_ubl = DECK_TOP_WORLD - UBL_WORLD_Z
    shelf_setdown_ubl = LEGACY_SHELF_CONST - UBL_WORLD_Z + HALF       # the fallen/mid-air shelf centre

    c, _m = make_controller('collect')
    t = run_to(c, 'verify')
    c.target_surface = (0.4, 0.0, deck_ubl, 0., 0., 0., 1.)          # the deck is the support
    on_deck = (0.4, 0.0, deck_ubl + HALF, 0., 0., 0., 1.)            # carton resting on the deck
    assert c.step(.1, packet(c, t, placed_pose=on_deck)).status == 'running'
    r = c.step(.1, packet(c, t + .1, placed_pose=on_deck))
    assert r.status == 'succeeded', r.reason

    c2, _m2 = make_controller('collect')
    t2 = run_to(c2, 'verify')
    c2.target_surface = (0.4, 0.0, deck_ubl, 0., 0., 0., 1.)
    at_shelf = (0.4, 0.0, shelf_setdown_ubl, 0., 0., 0., 1.)         # carton at the shelf set-down (bug)
    r2 = None
    for i in range(4):
        r2 = c2.step(.1, packet(c2, t2 + i * .1, placed_pose=at_shelf))
    assert r2.status == 'failed' and 'place verification failed' in r2.reason


# --------------------------------------------------------------- config_preflight four-height gate

def test_check_support_height_agree_and_disagree():
    runtime_ok = {'cart_placement': {'support_z_world': 0.2565}}
    scenario_ok = {'initial_staging': {'cart_support_z_world': 0.2560}}
    assert cp.check_support_height(runtime_ok, scenario_ok, resolved_top_z=DECK_TOP_WORLD) == []
    # the legacy shelf constant in either layer is refused
    runtime_bad = {'cart_placement': {'support_z_world': LEGACY_SHELF_CONST}}
    scenario_bad = {'initial_staging': {'cart_support_z_world': LEGACY_SHELF_CONST}}
    probs = cp.check_support_height(runtime_bad, scenario_bad, resolved_top_z=DECK_TOP_WORLD)
    assert len(probs) == 2
    assert any('runtime cart_placement.support_z_world' in p for p in probs)
    assert any('place_slot' in p for p in probs)
    # neither layer carries a height -> nothing to check
    assert cp.check_support_height({}, {}, resolved_top_z=DECK_TOP_WORLD) == []


def _write(tmp_path, name, obj):
    import json
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def _baseline_host():
    """A host config carrying the always-asserted BASELINE_REQUIRED caps so main() reaches the
    support-height gate instead of failing on unrelated missing caps."""
    return {k: 0.0 for k in cp.BASELINE_REQUIRED}


def test_preflight_main_refuses_stale_shelf_height_and_passes_on_deck(tmp_path):
    host = _write(tmp_path, 'host.json', _baseline_host())
    # a deck-height config passes; the stale shelf constant is refused.
    rt_ok = _write(tmp_path, 'rt_ok.json', {'cart_placement': {'support_z_world': 0.2565}})
    sc_ok = _write(tmp_path, 'sc_ok.json', {'initial_staging': {'cart_support_z_world': 0.2560}})
    assert cp.main(['--host-config', host, '--runtime-config', rt_ok, '--scenario', sc_ok,
                    '--place-on-deck', '--support-top-z-world', str(DECK_TOP_WORLD)]) == 0

    rt_bad = _write(tmp_path, 'rt_bad.json', {'cart_placement': {'support_z_world': LEGACY_SHELF_CONST}})
    sc_bad = _write(tmp_path, 'sc_bad.json', {'initial_staging': {'cart_support_z_world': LEGACY_SHELF_CONST}})
    assert cp.main(['--host-config', host, '--runtime-config', rt_bad, '--scenario', sc_bad,
                    '--place-on-deck', '--support-top-z-world', str(DECK_TOP_WORLD)]) == 2
    # --place-on-deck without the resolved top is itself a refusal (can't verify the height).
    assert cp.main(['--host-config', host, '--runtime-config', rt_ok, '--scenario', sc_ok,
                    '--place-on-deck']) == 2


def test_preflight_byte_identical_without_place_on_deck(tmp_path):
    """Existing invocations (no --place-on-deck) never run the support-height gate, even with a stale
    shelf height in the config -- the new gate is strictly opt-in."""
    host = _write(tmp_path, 'host.json', _baseline_host())
    rt_bad = _write(tmp_path, 'rt_bad.json', {'cart_placement': {'support_z_world': LEGACY_SHELF_CONST}})
    assert cp.main(['--host-config', host, '--runtime-config', rt_bad]) == 0
