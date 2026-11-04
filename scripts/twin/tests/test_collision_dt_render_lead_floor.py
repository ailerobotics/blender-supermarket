"""CPU proof for sim-collision-dt-render-lead-floor-v1.

coldwin30 (cold full cycle) tripped the collect preplace at sim 129 s with ::

    CONTROLLER_INVARIANT_TRIP phase=preplace now=129.000 collision_dt=-0.5333(band [-0.5,1.5])

The mirrored collision stamp (nvblox's ESDF integration time) LED the controller's node clock `now` by
0.533 s -- the collision world was FRESHER, not stale -- but the band FLOOR had been hand-set to -0.5
via RESTOCK_FRESH_RENDER_LEAD_S=0.5, which was 0.033 s too tight. The integration lane then bumped the
env to 0.7 by hand for coldwin31 and the trip cleared with zero over-band samples.

This lane makes that floor a DERIVED, logged default instead of a hand-set env constant:
``physics_step.collision_band_floor`` = (configured delivery render lead + the step-scaled ESDF
incorporation window), = 0.7 s at the default step. The positive (stale) CEILING and the trip name are
unchanged.

The quantified distribution these replays stand on (saved run dirs
``datasets/twin-robot-in-store-boot-v1/mission/coldwin3{0,1}-fail1-saved/``; analysis + histogram in
``firstmate/data/sim-collision-dt-render-lead-floor-v1/``): coldwin30's per-tick collision_dt sat
STEADY at ~-0.55 s (proxy ``-(world_t - t)`` over all 8219 collect ticks; min -0.5503, p50 -0.5500,
max -0.5497), so ALL ticks were below the -0.5 floor and NONE below -0.7. coldwin31 had the same
distribution and passed. The two exact host-stack.log values (boot gate -0.317, the preplace trip
-0.5333) agree with the proxy band.
"""
import contextlib
import io
import math
import os
from dataclasses import replace

import pytest

from scripts.twin.restock import physics_step as ps
from scripts.twin.restock_manipulation import ProductProfile, TransferController

DEFAULT_DT = ps.DEFAULT_DT        # 1/60
RAISED_DT = 1.0 / 36.0
RATIO = RAISED_DT / DEFAULT_DT

# coldwin30's recorded collision_dt (now - collision_time). The two EXACT values printed in
# host-stack.log, plus the steady per-tick proxy band -(world_t - t) the run held for all 8219 collect
# ticks. Every sample is a FUTURE lead (negative) -- the collision world leading `now`.
COLDWIN30_EXACT = [-0.317, -0.5333]
COLDWIN30_PROXY_BAND = [-0.5497, -0.5500, -0.5503]
COLDWIN30_TRACE = COLDWIN30_EXACT + COLDWIN30_PROXY_BAND

# The band the lane ran coldwin30 with (ceiling from RESTOCK_SENSOR_MAX_AGE_S=1.5).
AS_RUN_CEILING = 1.5
LEGACY_FLOOR = -0.5


# --------------------------------------------------------------------------- the pure derivation

def test_collision_band_floor_is_07_at_default():
    # The whole point: default step + default delivery lead -> 0.7, the value the lane found by hand.
    assert ps.collision_band_floor(0.2, DEFAULT_DT) == pytest.approx(0.7)
    # = delivery 0.2 + the ESDF incorporation window constant.
    assert ps.COLLISION_RELAY_LEAD_S == pytest.approx(0.5)


def test_collision_band_floor_scales_with_step():
    # Generalises PR 288: the whole floor is a per-frame budget, so it scales with the step.
    assert ps.collision_band_floor(0.2, RAISED_DT) == pytest.approx(0.7 * RATIO)
    # byte-identical scale: exactly the delivery+window sum at 1/60.
    assert ps.collision_band_floor(0.0, DEFAULT_DT) == pytest.approx(0.5)


def test_collision_band_floor_honours_a_configured_delivery_lead():
    # The env delivery term still raises the floor (it is simply no longer NEEDED): 0.5 -> 1.0 at 1/60.
    assert ps.collision_band_floor(0.5, DEFAULT_DT) == pytest.approx(1.0)


def test_collision_band_floor_safe_on_garbage_base():
    # A non-finite / negative base is treated as 0, never crashes a freshness read.
    assert ps.collision_band_floor(float('nan'), DEFAULT_DT) == pytest.approx(0.5)
    assert ps.collision_band_floor(-3.0, DEFAULT_DT) == pytest.approx(0.5)


# --------------------------------------------------------------------------- controller wiring

class _StubMotion:
    """TransferController.__init__ only stores the port; it never calls a motion method."""
    pass


@contextlib.contextmanager
def _physics_dt(dt):
    prev = os.environ.get(ps.PHYSICS_DT_ENV)
    if dt is None:
        os.environ.pop(ps.PHYSICS_DT_ENV, None)
    else:
        ps.export_physics_dt(dt)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(ps.PHYSICS_DT_ENV, None)
        else:
            os.environ[ps.PHYSICS_DT_ENV] = prev


def _make(**kw):
    prof = {'sku-box': ProductProfile('sku-box', (.1, .1, .1), (1., 0., 0., 0.))}
    c = TransferController(_StubMotion(), prof, fixture_mode=True, **kw)
    return c


def test_controller_default_floor_is_derived_07_without_env():
    # No RESTOCK_FRESH_RENDER_LEAD_S set: the controller derives the 0.7 floor itself.
    with _physics_dt(None):
        assert 'RESTOCK_FRESH_RENDER_LEAD_S' not in os.environ
        c = _make()
    assert c.render_lead == pytest.approx(0.7)
    assert c._render_lead_delivery == pytest.approx(0.2)


# --------------------------------------------------------------------------- the REPLAY (point 4)

def test_replay_coldwin30_trace_is_fresh_with_the_derived_floor():
    # Every recorded coldwin30 collision_dt (the exact -0.317/-0.5333 + the steady ~-0.55 proxy band) is
    # WITHIN the derived band -> _fresh True -> no trip. This is the run that failed; it now passes.
    with _physics_dt(None):
        c = _make(max_age=AS_RUN_CEILING)        # the lane's ceiling; the floor is the derived default
    assert c.render_lead == pytest.approx(0.7)
    now = 100.0
    for dt in COLDWIN30_TRACE:
        stamp = now - dt                          # collision_dt = now - stamp = dt
        assert c._fresh(stamp, now), f'collision_dt={dt} must be fresh under floor -{c.render_lead}'


def test_replay_coldwin30_trip_sample_would_trip_the_legacy_floor():
    # Guards the regression direction: the -0.5333 sample that tripped coldwin30 IS below the legacy
    # -0.5 floor (so the fix is load-bearing), but inside the derived -0.7 floor.
    with _physics_dt(None):
        c = _make(max_age=AS_RUN_CEILING)
    now = 129.0
    stamp = now - (-0.5333)                        # collision_time = now + 0.5333 (leads now)
    assert c._fresh(stamp, now) is True            # derived floor -0.7: fresh
    # legacy floor -0.5 reproduction: below the floor -> not fresh.
    legacy_fresh = (LEGACY_FLOOR <= (now - stamp) <= AS_RUN_CEILING)
    assert legacy_fresh is False


def test_replay_coldwin30_trip_sample_drives_a_running_step_not_a_trip():
    # Full-path faithfulness: feed the -0.5333 future-lead collision stamp through the real step
    # invariant (held lift). It must run, not trip and not enter COLLISION_STALE_PAUSE.
    from scripts.twin.tests.test_restock_manipulation import make_controller, packet, run_to
    with _physics_dt(None):
        c, m = make_controller('collect', max_age=AS_RUN_CEILING)
    assert c.render_lead == pytest.approx(0.7)
    t = run_to(c, 'lift'); assert c.held
    m.outcome = 'running'
    c.step(.1, packet(c, t, contact=True))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = c.step(.1, packet(c, t + .1, contact=True, collision_time=(t + .1) + 0.5333))
    assert r.status == 'running' and r.reason == '', r.reason
    assert 'CONTROLLER_INVARIANT_TRIP' not in buf.getvalue()
    assert 'COLLISION_STALE_PAUSE' not in buf.getvalue()


# --------------------------------------------------------------------------- safety: stale still trips

def test_synthetic_truly_stale_trace_still_trips():
    # The CEILING (max_age) is untouched: a genuinely STALE stamp (age beyond the ceiling, positive) is
    # still rejected at every step. Replays a synthetic stale trace at the default ceiling 0.5.
    with _physics_dt(None):
        c = _make()                                # default ceiling 0.5, derived floor 0.7
    now = 50.0
    for stale_age in (0.6, 1.0, 2.0, 5.0):         # all > max_age 0.5
        stamp = now - stale_age
        assert not c._fresh(stamp, now), f'stale age {stale_age} must trip (ceiling {c.max_age})'


def test_synthetic_stale_trace_trips_the_step_invariant_by_name():
    # End to end: a stale collision stamp pre-grasp still trips the invariant with the unchanged
    # 'collision_dt=' reason (trip name unchanged).
    from scripts.twin.tests.test_restock_manipulation import make_controller, packet
    c, _ = make_controller('collect')              # default ceiling 0.5
    r = c.step(.1, packet(c, 5.0, collision_time=4.3))   # age 0.7 > 0.5 -> stale
    assert r.status == 'failed' and 'collision_dt=' in r.reason, r.reason


def test_wildly_future_beyond_derived_floor_still_trips():
    # A future lead BEYOND the derived -0.7 floor is a real clock anomaly and still trips (never an
    # unbounded tolerance).
    with _physics_dt(None):
        c = _make(max_age=AS_RUN_CEILING)
    now = 10.0
    assert c._fresh(now - (-0.69), now) is True    # just inside -0.7
    assert c._fresh(now - (-0.71), now) is False   # just beyond -0.7 -> trips


# --------------------------------------------------------------------------- the logged band (point 3)

def test_freshness_band_summary_names_the_derived_band():
    with _physics_dt(None):
        c = _make(max_age=AS_RUN_CEILING)
    line = c.freshness_band_summary()
    assert line.startswith('COLLISION_FRESHNESS_BAND floor=-0.7000 ceiling=1.5000')
    assert 'delivery 0.2000 + esdf_window 0.5000' in line
    assert 'freshness_scale=1.000' in line
    assert 'sim-collision-dt-render-lead-floor-v1' in line


def test_freshness_band_summary_reports_the_step_scaled_band_at_raised_dt():
    with _physics_dt(RAISED_DT):
        c = _make()
    line = c.freshness_band_summary()
    # floor magnitude = 0.7 * 5/3 = 1.1667; the components are shown step-scaled and sum to it.
    assert c.render_lead == pytest.approx(0.7 * RATIO)
    assert ('floor=%.4f' % -(0.7 * RATIO)) in line
    assert 'freshness_scale=1.667' in line
