"""CPU tests for sim-place-seat-geometry-v1: the PHYSICAL cart/shelf set-down seat derivation and the
shared seat-vs-contact classification.

Two tiers:
  * PURE (no USD): the SeatModel math, slab_flat_top_z, and classify_set_down -- the shared predicate the
    controller / place_release gate use. Always run.
  * STAGE (usd-core + the recorded runs): derive_from_stage on the composed coldwin* mission.usda and the
    nav-trace carton WORLD poses -- asserts the true seat, the model error decomposition, and cw87's stall
    as an OBSTRUCTION. Skipped when pxr or the dataset is absent.

    env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-usd/bin/python -m pytest \
        scripts/twin/tests/test_place_seat_geometry.py
"""
import json
import math
import os

import pytest

from twin.restock_manipulation import place_seat_geometry as psg
from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.place_release import PlaceReleaseGate

MISSION = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission'
CARTON_H = 0.18
HALF = CARTON_H / 2.0
# The TRUE seat in WORLD frame (settled shelf flat top 0.9325 + carton half 0.09). The gate works in any
# consistent frame; the replays feed the nav-trace carton WORLD centre z and the WORLD seat centre.
TRUE_SEAT_CENTER_WORLD = 0.9325 + HALF     # 1.0225
SEAT_TOL = 0.02

try:
    import pxr  # noqa: F401
    _HAVE_PXR = True
except ImportError:
    _HAVE_PXR = False

need_stage = pytest.mark.skipif(
    not (_HAVE_PXR and os.path.isdir(MISSION)),
    reason='needs usd-core (pxr) and the twin-robot-in-store-boot-v1 mission dataset')


# ------------------------------------------------------------------- pure SeatModel + slab geometry

def test_slab_flat_top_is_center_plus_half_thickness():
    # The shelf slab centre is 0.9514 world at the composed pose; its flat top face is + the half-thickness
    # 0.0125 = 0.9639. This is the plane the carton rests on -- NOT the tilted world-AABB max-z corner.
    assert psg.slab_flat_top_z(0.9514, psg.SHELF_SLAB_HALF_THICKNESS_M) == pytest.approx(0.9639, abs=1e-4)


def test_seat_model_center_bottom_gap():
    seat = psg.seat_from_support_top(0.9325, CARTON_H)
    assert seat.carton_half_m == pytest.approx(HALF)
    assert seat.seat_center_z == pytest.approx(0.9325 + HALF)       # carton centre when resting flat
    # a carton whose centre is at the seat centre has bottom exactly on the support (gap 0)
    assert seat.bottom_gap(seat.seat_center_z) == pytest.approx(0.0, abs=1e-9)
    # 14 cm above => bottom_gap +0.14
    assert seat.bottom_gap(seat.seat_center_z + 0.14) == pytest.approx(0.14, abs=1e-9)


def test_seat_from_support_top_rejects_bad_height():
    with pytest.raises(ValueError):
        psg.seat_from_support_top(0.9325, 0.0)
    with pytest.raises(ValueError):
        psg.seat_from_support_top(float('nan'), CARTON_H)


# ------------------------------------------------------------------- classify_set_down (shared predicate)

def test_classify_seated_only_at_the_seat_with_contact():
    seat = psg.seat_from_support_top(0.9325, CARTON_H)
    at_seat = seat.seat_center_z
    # a measured contact AT the seat (within tol) -> seated (release here)
    assert psg.classify_set_down(at_seat, seat, contact=True, tol_m=0.02) == 'seated'
    assert psg.classify_set_down(at_seat + 0.015, seat, contact=True, tol_m=0.02) == 'seated'


def test_classify_contact_well_above_seat_is_obstructed():
    seat = psg.seat_from_support_top(0.9325, CARTON_H)
    # the coldwin87 stall: carton centre 14 cm above the seat, WITH a (press) contact -> OBSTRUCTED
    stall_center = seat.seat_center_z + 0.141
    assert psg.classify_set_down(stall_center, seat, contact=True, tol_m=0.02) == 'obstructed'


def test_classify_descending_without_contact_above_seat():
    seat = psg.seat_from_support_top(0.9325, CARTON_H)
    assert psg.classify_set_down(seat.seat_center_z + 0.10, seat, contact=False, tol_m=0.02) == 'descending'


def test_classify_passed_the_seat_without_contact_is_fault():
    seat = psg.seat_from_support_top(0.9325, CARTON_H)
    # descended well BELOW the seat with no contact -> the support is not where modelled (named fault)
    assert psg.classify_set_down(seat.seat_center_z - 0.05, seat, contact=False, tol_m=0.02) == 'passed'


# ------------------------------------------------------------------- stage derivation (usd-core)

@need_stage
def test_derive_from_cw87_stage_flat_top_vs_aabb_corner():
    der = psg.derive_from_stage(os.path.join(MISSION, 'coldwin87', 'mission.usda'))
    # the FLAT top face the carton rests on (build/composed pose, chassis ~0.2315)
    assert der.shelf_flat_top_z == pytest.approx(0.9639, abs=1.5e-3)
    # the world-AABB max-z CORNER (the model's over-read) is ~7 mm higher (the slab tilt)
    assert der.shelf_aabb_top_z == pytest.approx(0.9712, abs=1.5e-3)
    assert der.aabb_corner_inflation_m == pytest.approx(0.0073, abs=1.5e-3)
    assert der.aabb_corner_inflation_m > 0.004            # the AABB corner is strictly above the flat face
    # carton half-height is exactly 0.09 (extent +-0.09); the model's half-height term is RIGHT
    assert der.carton_half_m == pytest.approx(0.09, abs=1e-4)
    # arm root + chassis world z at the composed (build) pose
    assert der.ur_base_z == pytest.approx(0.4999, abs=2e-3)
    assert der.chassis_z == pytest.approx(0.2315, abs=2e-3)


@need_stage
def test_model_support_top_error_decomposes_into_aabb_corner_plus_settling():
    der = psg.derive_from_stage(os.path.join(MISSION, 'coldwin87', 'mission.usda'))
    model_top = 0.9707                  # COLLECT_PLACE_SUPPORT_Z (resolved AABB corner, un-settled chassis)
    true_top = 0.9325                   # settled shelf flat top = cw83's rested carton bottom
    total_err = model_top - true_top
    assert total_err == pytest.approx(0.0382, abs=1.5e-3)
    aabb_part = der.shelf_aabb_top_z - der.shelf_flat_top_z          # the tilt inflation (~7 mm)
    settling_part = der.shelf_flat_top_z - true_top                  # the cart settling (~31 mm)
    assert aabb_part == pytest.approx(0.0073, abs=2e-3)
    assert settling_part == pytest.approx(0.0314, abs=2e-3)
    # the two parts account for the whole model error (the resolved AABB top ~= model 0.9707)
    assert (aabb_part + settling_part) == pytest.approx(der.shelf_aabb_top_z - true_top, abs=1e-6)
    assert der.shelf_aabb_top_z == pytest.approx(model_top, abs=1.5e-3)


@need_stage
def test_cart_support_slab_flat_top_matches_the_face_not_the_aabb_corner():
    # the isaac_factory flat-face resolution (RESTOCK_PLACE_SUPPORT_FLAT_TOP) uses
    # cart_support.slab_flat_top_z_world on the shelf prim's local-to-world (the .T column-vector form).
    # Assert it recovers the FLAT TOP FACE 0.9639, strictly below the world-AABB corner 0.9712.
    import numpy as np
    from pxr import Usd, UsdGeom
    from twin.restock_manipulation import cart_support
    stage = Usd.Stage.Open(os.path.join(MISSION, 'coldwin87', 'mission.usda'))
    shelf = stage.GetPrimAtPath('/Rollwagen/chassis/shelf')
    cache = UsdGeom.XformCache()
    slab_l2w = np.asarray(cache.GetLocalToWorldTransform(shelf)).T       # isaac_factory convention
    size = shelf.GetAttribute('size').Get()
    half = 0.5 * (float(size) if size else 1.0)
    flat = cart_support.slab_flat_top_z_world(slab_l2w, half)
    assert flat == pytest.approx(0.9639, abs=1.5e-3)
    # and strictly below the AABB max-z corner the legacy resolution returns
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    aabb_top = float(bbox.ComputeWorldBound(shelf).ComputeAlignedRange().GetMax()[2])
    assert flat < aabb_top
    assert (aabb_top - flat) == pytest.approx(0.0073, abs=1.5e-3)


def _carton_final_and_place_min(run):
    """From a run's nav-trace, the carton WORLD centre z at the end and the min over the place phase."""
    nav = os.path.join(MISSION, run, 'nav-trace.jsonl')
    place_start = None
    mj = os.path.join(MISSION, run, 'mission.json')
    if os.path.exists(mj):
        data = json.load(open(mj))
        for ev in data.get('stage_timeline', []):
            d = ev.get('data', {})
            if d.get('label') == 'place' or d.get('step') == 5:
                place_start = float(d.get('sim_time', 0.0))
                break
    final = None
    place_min = math.inf
    with open(nav) as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            c = d.get('carton')
            wt = d.get('world_t')
            if not c or wt is None:
                continue
            final = float(c[2])
            if place_start is None or wt >= place_start:
                place_min = min(place_min, float(c[2]))
    return final, (None if place_min is math.inf else place_min)


@need_stage
def test_cw83_rested_carton_is_the_true_seat_within_tolerance():
    # coldwin83 released and the carton rested + verify PASSED: its final carton CENTRE world z is the true
    # seat centre. derive the carton half from the stage and assert the rested bottom == 0.9325 settled flat.
    der = psg.derive_from_stage(os.path.join(MISSION, 'coldwin83', 'mission.usda'))
    final, _ = _carton_final_and_place_min('coldwin83')
    rested_bottom = final - der.carton_half_m
    assert rested_bottom == pytest.approx(0.9325, abs=5e-3)
    # against the TRUE seat it classifies SEATED (a contact at the seat)
    seat = psg.seat_from_support_top(0.9325, CARTON_H)
    assert psg.classify_set_down(final, seat, contact=True, tol_m=0.02) == 'seated'


@need_stage
def test_cw87_stall_is_14cm_above_the_true_seat_an_obstruction():
    # the brief's central assertion: coldwin87's solid stall is ~14 cm ABOVE the corrected seat, so a
    # contact there is an OBSTRUCTION -- NOT a seat contact. (This DISPROVES PR 374's "rested on the shelf".)
    der = psg.derive_from_stage(os.path.join(MISSION, 'coldwin87', 'mission.usda'))
    final, place_min = _carton_final_and_place_min('coldwin87')
    stall_bottom = final - der.carton_half_m
    assert stall_bottom == pytest.approx(1.0734, abs=3e-3)
    seat = psg.seat_from_support_top(0.9325, CARTON_H)
    gap = seat.bottom_gap(final)
    assert gap == pytest.approx(0.141, abs=5e-3)                     # ~14 cm above the true seat
    assert gap > 0.10                                               # unambiguously not within any sane tol
    assert psg.classify_set_down(final, seat, contact=True, tol_m=0.02) == 'obstructed'
    # the carton NEVER descended near the seat during the whole place phase (the min is the stall itself)
    assert (place_min - der.carton_half_m) > 0.9325 + 0.10


@need_stage
def test_cw89_pr374_released_high_above_the_seat():
    # coldwin89: PR 374's +half threshold sat ABOVE the descent start, so it released at the standoff ~12 cm
    # above the seat (the free-check then refused). Its final carton is well above the seat (obstruction).
    der = psg.derive_from_stage(os.path.join(MISSION, 'coldwin89', 'mission.usda'))
    final, _ = _carton_final_and_place_min('coldwin89')
    seat = psg.seat_from_support_top(0.9325, CARTON_H)
    assert seat.bottom_gap(final) > 0.10
    assert psg.classify_set_down(final, seat, contact=True, tol_m=0.02) == 'obstructed'


# ------------------------------------------------------------------- per-run REPLAYS (brief part 4)
#
# Replay each recorded run's set-down against the CORRECTED seat (0.9325 world flat top). The honest
# verdict per run, stated up front (what SHOULD happen under the seat gate):
#   cw83  -> SEATED      : the carton rested AT the seat (verify passed); the corrected seat is right.
#   cw87  -> OBSTRUCTED  : the arm stalled 14 cm above the seat (cage-filtered carton on nothing) -- the
#                          gate must NOT release there (the press-to-fault PR 373 produced).
#   cw89  -> OBSTRUCTED  : PR 374 released at the standoff 12 cm high -- the gate must NOT release there.
#   cw85/86/88 -> PASSED : the carton ended BELOW the seat (fell off / onto a lower surface) -- the gate
#                          must name a fault, never read the fall as a valid set-down.
# Each run's recorded END-STATE carton centre (nav-trace) classified against the true seat with the shared
# predicate. This is the evidence that DISPROVES PR 374's "rested on the shelf at 0.6422": cw87's carton
# ended 14 cm above the seat, not on it.

_EXPECTED_ENDSTATE = {
    'coldwin83': 'seated',
    'coldwin85': 'passed',
    'coldwin86': 'passed',
    'coldwin87': 'obstructed',
    'coldwin88': 'passed',
    'coldwin89': 'obstructed',
}


@need_stage
@pytest.mark.parametrize('run,expected', sorted(_EXPECTED_ENDSTATE.items()))
def test_replay_endstate_classification_per_run(run, expected):
    final, _ = _carton_final_and_place_min(run)
    seat = psg.seat_from_support_top(0.9325, CARTON_H)
    # the carton came to rest (stuck / rested / fell) => a measured contact at the end state
    assert psg.classify_set_down(final, seat, contact=True, tol_m=SEAT_TOL) == expected


def _final_descent_series(run, dt=0.5):
    """The run's place-phase carton WORLD centre z, downsampled to ~the sensor cadence (dt sim-s, like the
    PLACE_RELEASE_TICK throttle), from the FINAL descent onward (the last time the carton is near the place
    standoff max). Mirrors what the controller's place-descent gate observes (phase=='place'): the raw
    nav-trace (physics rate) over-samples and false-stalls on render-pacing pauses; the long pre-descent
    hover at the high preplace standoff is a different phase."""
    nav = os.path.join(MISSION, run, 'nav-trace.jsonl')
    place_start = None
    mj = os.path.join(MISSION, run, 'mission.json')
    if os.path.exists(mj):
        data = json.load(open(mj))
        for ev in data.get('stage_timeline', []):
            d = ev.get('data', {})
            if d.get('label') == 'place' or d.get('step') == 5:
                place_start = float(d.get('sim_time', 0.0))
                break
    samples = []
    last_t = None
    with open(nav) as f:
        for line in f:
            try:
                o = json.loads(line)
            except ValueError:
                continue
            c = o.get('carton')
            wt = o.get('world_t')
            if not c or wt is None or (place_start is not None and wt < place_start):
                continue
            if last_t is None or wt - last_t >= dt:
                samples.append((float(wt), float(c[2])))
                last_t = wt
    if not samples:
        return []
    zmax = max(z for _, z in samples)
    last_top = max(i for i, (_, z) in enumerate(samples) if z >= zmax - 0.02)
    return samples[last_top:]


@need_stage
@pytest.mark.parametrize('run', ['coldwin87', 'coldwin89'])
def test_replay_live_gate_stuck_runs_never_release(run):
    # Feed the recorded FINAL place descent through the seat-gated gate. The stuck runs stall ~14/12 cm
    # above the seat, so the gate returns OBSTRUCTED and NEVER 'release'/'seated' (it would not drop the
    # carton from there). This is the live-gate counterpart of the end-state classification.
    series = _final_descent_series(run)
    assert series, f'no place descent samples for {run}'
    gate = PlaceReleaseGate(release_z=TRUE_SEAT_CENTER_WORLD, seat_center_z=TRUE_SEAT_CENTER_WORLD,
                            seat_tol_m=SEAT_TOL, stall_ticks=5, stall_eps_m=0.001)
    verdict = None
    for wt, z in series:
        d = gate.observe(now=wt, carton_z=z, commanded_down=True, arm_effort_max=100.0)
        if d.action in ('obstructed', 'seated', 'release', 'fault'):
            verdict = d.action
            break
    assert verdict == 'obstructed', f'{run}: expected obstructed, got {verdict}'


# ------------------------------------------------------------------- PlaceReleaseGate seat gate (pure)

def _drive(gate, zs, effort=None):
    """Feed carton-centre zs; return the first terminal verdict action (or 'press')."""
    for i, z in enumerate(zs):
        d = gate.observe(now=i * 0.5, carton_z=z, commanded_down=True,
                         arm_effort_max=(100.0 if effort is None else effort))
        if d.action in ('obstructed', 'seated', 'release', 'fault'):
            return d.action
    return 'press'


def test_seat_gate_off_is_legacy_byte_identical():
    # seat_center_z None => the seat gate never intercepts; a descent stalling above a release_z still
    # releases via the legacy stall trigger (the pre-seat-gate behaviour).
    gate = PlaceReleaseGate(release_z=0.50, stall_ticks=3, stall_eps_m=0.001)   # no seat_center_z
    assert _drive(gate, [0.70] * 6) == 'release'          # stall far above release_z => legacy stall release


def test_seat_gate_obstructed_when_stall_above_seat():
    gate = PlaceReleaseGate(release_z=0.50, seat_center_z=0.50, seat_tol_m=0.02,
                            stall_ticks=3, stall_eps_m=0.001)
    # stall 14 cm above the seat => OBSTRUCTED (do not release/drop)
    assert _drive(gate, [0.64] * 6) == 'obstructed'


def test_seat_gate_releases_at_the_seat():
    gate = PlaceReleaseGate(release_z=0.50, seat_center_z=0.50, seat_tol_m=0.02,
                            stall_ticks=3, stall_eps_m=0.001)
    # descend to the seat then stall => RELEASE
    assert _drive(gate, [0.60, 0.57, 0.54, 0.51, 0.50, 0.50, 0.50, 0.50]) == 'release'


def test_seat_gate_fault_when_descent_passes_the_seat_without_contact():
    gate = PlaceReleaseGate(release_z=None, seat_center_z=0.50, seat_tol_m=0.02,
                            stall_ticks=50, stall_eps_m=0.001)   # never stalls (keeps progressing)
    # a monotonic descent that passes BELOW seat-tol with no stall => fault (support not where modelled)
    assert _drive(gate, [0.56, 0.54, 0.52, 0.50, 0.48, 0.46]) == 'fault'


def test_seat_gate_press_while_descending_above_seat():
    gate = PlaceReleaseGate(release_z=0.50, seat_center_z=0.50, seat_tol_m=0.02,
                            stall_ticks=50, stall_eps_m=0.001)
    # still above the seat, making progress, no stall => keep pressing (no terminal verdict)
    assert _drive(gate, [0.70, 0.68, 0.66, 0.64]) == 'press'


# ------------------------------------------------------------------- carton_level config (pure)

def test_carton_level_place_seat_gate_defaults_off_byte_identical():
    c = cl.resolve_level_config({'place_release_on_contact': True})
    assert c.place_seat_gate is False
    assert c.place_seat_tol_m == pytest.approx(0.02)


def test_carton_level_place_seat_gate_resolves_and_validates():
    c = cl.resolve_level_config({'place_release_on_contact': True, 'place_seat_gate': True,
                                 'place_seat_tol_m': 0.015})
    assert c.place_seat_gate is True
    assert c.place_seat_tol_m == pytest.approx(0.015)


def test_carton_level_place_seat_gate_requires_place_release_on_contact():
    with pytest.raises(ValueError):
        cl.resolve_level_config({'place_seat_gate': True})       # without place_release_on_contact


def test_carton_level_place_seat_tol_must_be_positive():
    with pytest.raises(ValueError):
        cl.resolve_level_config({'place_release_on_contact': True, 'place_seat_gate': True,
                                 'place_seat_tol_m': 0.0})


def test_replay_synthetic_good_descent_seats():
    # A clean controlled descent that reaches the seat (then stalls on the support) RELEASES at the seat --
    # the success path the gate must still take. Seat at 1.0225 (world), descend to it and hold.
    gate = PlaceReleaseGate(release_z=TRUE_SEAT_CENTER_WORLD, seat_center_z=TRUE_SEAT_CENTER_WORLD,
                            seat_tol_m=SEAT_TOL, stall_ticks=5, stall_eps_m=0.001)
    # monotonic descent 1.20 -> 1.025 (just above the seat), then freeze AT the seat (stall => contact)
    descent = [1.20 - 0.01 * i for i in range(18)]          # 1.20 .. 1.03
    hold = [TRUE_SEAT_CENTER_WORLD] * 8                     # frozen at the seat
    verdict = None
    for i, z in enumerate(descent + hold):
        d = gate.observe(now=i * 0.5, carton_z=z, commanded_down=True, arm_effort_max=100.0)
        if d.action in ('obstructed', 'seated', 'release', 'fault'):
            verdict = d.action
            break
    assert verdict == 'release'
