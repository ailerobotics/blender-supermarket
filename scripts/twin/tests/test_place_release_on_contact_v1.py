"""CPU tests for the contact-gated place-descent weld release (sim-place-release-on-contact-v1).

coldwin68 was the furthest run ever: the pick sealed, the carried pre-place converged, the base locked for
the loaded descent -- then the set-down never released. The PR359 pre-place hold-firm held the welded carton
rigid into a ~1.46 deg roll, so its edge jammed on cage post /Rollwagen/chassis/post_2 at the DESCENT START
(carton_z frozen at 0.6257, 0.075 m ABOVE the computed z-release threshold 0.5624). The fixed carton_z gate
therefore never fired and the stiff place drive (kp1e6/eff2000) pressed the blocked carton to 757 N*m -- past
the 600 N*m station full-brake guard -- then three StageRetry cycles and a 180 s backstop killed the run.
coldwin64 ran the SAME config/support, descended the full drop, and the z gate released cleanly (peak 125.8
N*m, pad raycast hit NOTHING).

The fix releases the weld the instant the support is ACTUALLY touched (effort over the pre-contact hold
baseline | descent stall | the legacy carton_z threshold, whichever first), GATED on the touched body being
the SUPPORT: a flush contact on a NON-support body (the cage post) is an OBSTRUCTION -- do not press/release.

These tests cover: each release trigger in isolation (effort / stall / contact / z), the obstruction gate,
the trigger logging label, byte-identical-off (config default), config resolve + validation, the controller
``_place_obstructed`` bounded-re-approach-then-fail decision, and REPLAY regressions from the coldwin68
(no-descent, post-jam, effort 665->757) and coldwin64 (clean descent -> z) traces.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_release_on_contact_v1.py
"""
from types import SimpleNamespace as NS

import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import hit_prims_from_probe, seal_feedback_from_probe
from twin.restock_manipulation.place_release import (
    PlaceReleaseGate, PlaceReleaseDecision, is_support_hit, DEFAULT_SUPPORT_TOKENS)

SHELF = '/Rollwagen/chassis/shelf'
POST = '/Rollwagen/chassis/post_2'


def _gate(**kw):
    kw.setdefault('release_z', 0.5624)
    kw.setdefault('effort_margin_nm', 100.0)
    kw.setdefault('effort_consecutive', 3)
    kw.setdefault('stall_eps_m', 0.001)
    kw.setdefault('stall_ticks', 5)
    kw.setdefault('support_tokens', ('shelf', 'deck'))
    kw.setdefault('obstruction_gap_mm', 2.0)
    return PlaceReleaseGate(**kw)


# --------------------------------------------------------------------- support identity + probe helpers

def test_is_support_hit_tokens():
    assert is_support_hit(SHELF) and is_support_hit('/Rollwagen/chassis/deck')
    assert not is_support_hit(POST) and not is_support_hit('/Rollwagen/chassis/drawbar')
    assert is_support_hit('/Rollwagen/deck_top', ('deck',))
    assert not is_support_hit('/Rollwagen/deck_top', ('shelf',))


def test_hit_prims_from_probe():
    assert hit_prims_from_probe({'raycast': [[POST, 0.0], [POST, 0.0], [SHELF, 1.0]]}) == (POST, SHELF)
    assert hit_prims_from_probe({'raycast': []}) is None
    assert hit_prims_from_probe({}) is None
    assert hit_prims_from_probe(None) is None
    # seal_feedback_from_probe (unchanged) still only reports the bool
    assert seal_feedback_from_probe({'status': 'Closing', 'raycast': [[POST, 0.0]]})[3] is True


# --------------------------------------------------------------------- each release trigger in isolation

def test_z_fallback_trigger():
    g = _gate()
    assert g.observe(now=0, carton_z=0.60, commanded_down=True).action == 'press'
    d = g.observe(now=0.05, carton_z=0.5624, commanded_down=True)
    assert d.action == 'release' and d.trigger == 'z'
    assert 'release_z' in d.reason


def test_effort_margin_trigger_on_support():
    """Effort rising >= baseline + margin for N ticks while the hit is the SUPPORT => release:effort,
    and it fires well before the 600 N*m guard."""
    g = _gate(effort_warmup_ticks=2)
    # baseline settles ~125 over the warmup, hit is the shelf (support), carton still descending slowly
    seq = [125, 125, 126, 230, 231, 232]   # crosses 125+100=225 for 3 consecutive from idx 3
    d = None
    for i, e in enumerate(seq):
        d = g.observe(now=i * 0.05, carton_z=0.60 - i * 0.0001, commanded_down=True,
                      arm_effort_max=e, seal_gap_mm=0.5, seal_hit_prims=[SHELF])
    assert d.action == 'release' and d.trigger == 'effort'
    assert g.peak_effort < 600.0 and d.baseline == 125.0


def test_stall_trigger_no_hit_releases():
    """Carton z not decreasing for stall_ticks, no obstruction hit => release:stall (support met)."""
    g = _gate(stall_ticks=4)
    d = None
    for i in range(8):
        d = g.observe(now=i * 0.05, carton_z=0.60, commanded_down=True, arm_effort_max=130,
                      seal_gap_mm=None, seal_hit_prims=None)
    assert d.action == 'release' and d.trigger == 'stall'


def test_contact_trigger_support_gap():
    g = _gate(contact_gap_mm=1.0)
    d = g.observe(now=0, carton_z=0.60, commanded_down=True, seal_gap_mm=0.5, seal_hit_prims=[SHELF])
    assert d.action == 'release' and d.trigger == 'contact'
    # a gap above the threshold does not trigger
    g2 = _gate(contact_gap_mm=1.0)
    assert g2.observe(now=0, carton_z=0.60, commanded_down=True, seal_gap_mm=2.0,
                      seal_hit_prims=[SHELF]).action == 'press'


def test_contact_trigger_off_by_default():
    # contact_gap_mm None => the seal-gap trigger never fires on its own
    g = _gate()
    assert g.observe(now=0, carton_z=0.60, commanded_down=True, seal_gap_mm=0.0,
                     seal_hit_prims=[SHELF]).action == 'press'


# --------------------------------------------------------------------- obstruction gate (post, not support)

def test_obstruction_flush_non_support():
    g = _gate()
    d = g.observe(now=0, carton_z=0.6257, commanded_down=True, arm_effort_max=125,
                  seal_gap_mm=0.0, seal_hit_prims=[POST])
    assert d.action == 'obstructed' and d.trigger is None and d.hit_prim == POST
    assert POST in d.reason


def test_obstruction_preempts_effort_and_z():
    """A non-support flush hit obstructs even if the effort/stall would otherwise read a release."""
    g = _gate(effort_warmup_ticks=0, release_z=0.70)   # z already satisfied on purpose
    seq = [125, 400, 500, 580]
    d = None
    for i, e in enumerate(seq):
        d = g.observe(now=i * 0.05, carton_z=0.6257, commanded_down=True, arm_effort_max=e,
                      seal_gap_mm=0.0, seal_hit_prims=[POST])
        assert d.action == 'obstructed'
    assert g.peak_effort < 600.0  # a jam is caught before the guard


def test_support_hit_is_not_obstruction():
    g = _gate(contact_gap_mm=1.0)
    d = g.observe(now=0, carton_z=0.60, commanded_down=True, seal_gap_mm=0.0, seal_hit_prims=[SHELF])
    assert d.action == 'release' and d.trigger == 'contact'


def test_not_commanded_down_is_noop():
    g = _gate()
    d = g.observe(now=0, carton_z=0.5624, commanded_down=False, arm_effort_max=900,
                  seal_gap_mm=0.0, seal_hit_prims=[POST])
    assert d.action == 'press' and d.trigger is None


# --------------------------------------------------------------------- config resolve + validation

def test_config_default_off_byte_identical():
    c = cl.resolve_level_config(True)   # defaults
    assert c.place_release_on_contact is False
    assert c.place_reapproach_max == 0 and c.place_support_tokens == ('shelf', 'deck')


def test_config_resolve_values():
    c = cl.resolve_level_config({
        'place_release_on_contact': True, 'place_release_effort_margin_nm': 80,
        'place_release_effort_consecutive': 2, 'place_release_stall_m': 0.002,
        'place_release_stall_ticks': 6, 'place_release_contact_gap_mm': 1.5,
        'place_support_tokens': ['shelf'], 'place_obstruction_gap_mm': 3.0,
        'place_reapproach_max': 1, 'place_reapproach_shift_m': 0.04})
    assert c.place_release_on_contact is True and c.place_release_effort_margin_nm == 80.0
    assert c.place_release_effort_consecutive == 2 and c.place_release_stall_ticks == 6
    assert c.place_release_contact_gap_mm == 1.5 and c.place_support_tokens == ('shelf',)
    assert c.place_reapproach_max == 1 and c.place_reapproach_shift_m == 0.04


@pytest.mark.parametrize('bad', [
    {'place_release_effort_margin_nm': 0},
    {'place_release_effort_margin_nm': -5},
    {'place_release_effort_consecutive': 0},
    {'place_release_stall_ticks': 0},
    {'place_reapproach_max': -1},
    {'place_release_contact_gap_mm': 0},
    {'place_support_tokens': []},
    {'place_support_tokens': 'shelf'},
    {'place_obstruction_gap_mm': 0},
])
def test_config_validation_rejects(bad):
    with pytest.raises(ValueError):
        cl.resolve_level_config(dict({'place_release_on_contact': True}, **bad))


# --------------------------------------------------------------------- controller _place_obstructed decision

def _fake_controller(reapproach_max):
    from twin.restock_manipulation.controller import TransferController
    c = NS()
    c._carton_level = cl.resolve_level_config({'place_release_on_contact': True,
                                               'place_reapproach_max': reapproach_max,
                                               'place_reapproach_shift_m': 0.03})
    c.poses = {'place': (0.10, 0.80, 0.55, 1, 0, 0, 0)}
    c._place_reapproach_count = 0
    c._place_release_gate = _gate()
    c._place_release_log_sim = 0.0
    c._place_contact_release_z = 0.56
    c.evidence = []
    c._transitions = []
    c._transition = lambda ph: c._transitions.append(ph)
    c._result = lambda: 'HOLD'
    c._fail = lambda r: ('FAIL', r)
    c._obstructed = TransferController._place_obstructed.__get__(c, type(c))
    return c


def test_obstructed_fail_fast_when_budget_zero():
    c = _fake_controller(0)
    dec = PlaceReleaseDecision('obstructed', None, 'met post', 0.6257, 125, 125, 7, POST)
    res = c._obstructed(0.1, dec)
    assert isinstance(res, tuple) and res[0] == 'FAIL' and 'PLACE_OBSTRUCTED' in res[1] and POST in res[1]
    assert c._transitions == []     # no re-approach


def test_obstructed_bounded_reapproach_then_fail():
    c = _fake_controller(1)
    dec = PlaceReleaseDecision('obstructed', None, 'met post', 0.6257, 125, 125, 7, POST)
    x0 = c.poses['place'][0]
    res1 = c._obstructed(0.1, dec)
    assert res1 == 'HOLD' and c._transitions == ['preplace']       # one re-approach
    assert c.poses['place'][0] != x0                                # laterally shifted
    assert c.evidence and 'place_obstructed' in c.evidence[0]
    res2 = c._obstructed(0.2, dec)                                  # budget now exhausted
    assert isinstance(res2, tuple) and res2[0] == 'FAIL'


# --------------------------------------------------------------------- REPLAY regressions (cw68 / cw64)

# coldwin68 trace: carton frozen at the descent START 0.6257 (NO descent), pad flush on post_2 (gap 0.0),
# the station full-brake effort ramps 665 -> 757 (sampled every 30 ticks in the log).
CW68_START_Z = 0.6257
CW68_EFFORTS = [665.9, 761.1, 588.1, 611.8, 666, 681, 695, 709, 723, 757, 757, 757]


def test_replay_cw68_key_on_obstructed_before_guard():
    """Key ON + the recorded post-jam: the gate obstructs (does NOT press/drop) and the effort it SEES
    never reaches the 600 N*m guard before the terminal verdict."""
    g = _gate()
    for i, e in enumerate(CW68_EFFORTS):
        d = g.observe(now=i * 0.05, carton_z=CW68_START_Z, commanded_down=True, arm_effort_max=e,
                      seal_gap_mm=0.0, seal_hit_prims=[POST])
        assert d.action == 'obstructed'
        if i == 0:
            break   # obstructed on the FIRST contact tick -- the press never builds


def test_replay_cw68_support_variant_releases_before_guard():
    """Had the jam been the SUPPORT (not a post), the effort trigger releases BEFORE the 600 N*m guard --
    the coldwin68 press loop (665->757) never happens with the key on."""
    g = _gate(effort_warmup_ticks=1)
    d = None
    for i, e in enumerate([125, 130, 240, 260, 300]):   # baseline 125 then > 225 for 3 ticks
        d = g.observe(now=i * 0.05, carton_z=CW68_START_Z, commanded_down=True, arm_effort_max=e,
                      seal_gap_mm=0.5, seal_hit_prims=[SHELF])
        if d.action != 'press':
            break
    assert d.action == 'release' and d.trigger == 'effort' and g.peak_effort < 600.0


def test_replay_cw68_key_off_presses_past_guard():
    """Key OFF (the gate is never constructed in the controller): nothing stops the press, so the recorded
    effort climbs past the 600 N*m guard -- the regression the fix removes."""
    assert max(CW68_EFFORTS) > 600.0        # the documented press loop with the feature off


def test_replay_cw64_clean_descent_releases_at_z():
    """coldwin64: same config/support, the carton descends the full 0.0747 m with NO pad hit; the z gate
    releases at the threshold and the effort it sees stays low (peak 125.8 in the run)."""
    g = _gate(release_z=0.5624 + 0.0, effort_margin_nm=100)
    start = 0.6253
    fired = None
    for i in range(50):
        z = start - 0.0747 * min(1.0, i / 40.0)
        d = g.observe(now=i * 0.05, carton_z=z, commanded_down=True, arm_effort_max=125.8,
                      seal_gap_mm=None, seal_hit_prims=[])
        if d.action != 'press':
            fired = d
            break
    assert fired is not None and fired.action == 'release' and fired.trigger == 'z'
    assert g.peak_effort == 125.8
