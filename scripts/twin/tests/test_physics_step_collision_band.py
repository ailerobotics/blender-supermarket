"""CPU proof for sim-physics-step-collision-band-v1.

The sim-realtime-factor-v1 physics-step lever (``physics_dt``) raised the step 1/60 -> 1/36 and the
rtfcollect GPU run FAILED collect with ``collision_dt=1.5278 (band [-0.5,1.5]) -> arm heartbeat lost``.
Root cause: the per-FRAME sensor-freshness windows (the collision/ESDF age ``max_age``, the arm-feedback
age, and the one-to-few-step ``render_lead``) are stored as FIXED sim-seconds but encode a FRAME-COUNT
budget -- nvblox integrates rendered frames and ``rendering_dt`` tracks ``physics_dt``, so one frame =
dt sim-seconds. A coarser step stretches each frame's sim-time footprint, silently SHRINKING the frame
budget and false-tripping a healthy pipeline.

These tests prove, on CPU:
  * the scale is EXACTLY 1.0 at 1/60 (byte-identical) and dt/(1/60) otherwise;
  * the controller + host + mission freshness windows rebase onto the running step;
  * a REPLAY of the exact rtfcollect trace: the legacy FIXED band trips at 1/36, the step-aware band
    does NOT (margin stated), and a genuinely-dead ESDF STILL trips at every step (safety preserved);
  * the band is a constant FRAME budget across the step (not a widening).

The dt-sensitive-threshold TABLE this delivers is in
``firstmate/data/sim-physics-step-collision-band-v1/dt_thresholds_table.md``.
"""
import os
from contextlib import contextmanager

import pytest

from scripts.twin.restock import physics_step as ps

DEFAULT_DT = ps.DEFAULT_DT           # 1/60
RAISED_DT = 1.0 / 36.0               # the rtfcollect step
RATIO = RAISED_DT / DEFAULT_DT       # 5/3


@contextmanager
def physics_dt_env(dt):
    """Set RESTOCK_PHYSICS_DT for the block (None => unset)."""
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


# --------------------------------------------------------------------------- scale helper

def test_scale_exactly_one_at_default():
    # byte-identical guarantee: the factor at 1/60 is EXACTLY 1.0 (not merely close).
    assert ps.freshness_scale(DEFAULT_DT) == 1.0
    assert ps.scale_freshness_window(1.5, DEFAULT_DT) == 1.5
    assert ps.scale_freshness_window(0.02, DEFAULT_DT) == 0.02


def test_scale_is_linear_in_dt():
    assert ps.freshness_scale(RAISED_DT) == pytest.approx(RATIO)
    # a FRAME budget: base_s/DEFAULT_DT frames, held constant across the step.
    assert ps.scale_freshness_window(1.5, RAISED_DT) == pytest.approx(2.5)
    frames_default = 1.5 / DEFAULT_DT
    frames_raised = ps.scale_freshness_window(1.5, RAISED_DT) / RAISED_DT
    assert frames_default == pytest.approx(frames_raised) == pytest.approx(90.0)


def test_scale_is_bounded_and_safe_on_garbage():
    # dt bounded [1/240, 1/15] => factor bounded [0.25, 4.0]; a bad value never crashes a freshness read.
    assert ps.freshness_scale(ps.MIN_DT) == pytest.approx(0.25)
    assert ps.freshness_scale(ps.MAX_DT) == pytest.approx(4.0)
    assert ps.freshness_scale(float('nan')) == 1.0
    assert ps.freshness_scale(0.0) == 1.0


def test_env_roundtrip_and_default():
    assert ps.dt_from_env({}) == DEFAULT_DT
    e = {}
    ps.export_physics_dt(RAISED_DT, e)
    assert ps.dt_from_env(e) == pytest.approx(RAISED_DT)
    assert ps.freshness_scale_from_env(e) == pytest.approx(RATIO)
    # a default export round-trips to EXACTLY factor 1.0 (byte-identical even when explicitly exported).
    ed = {}
    ps.export_physics_dt(DEFAULT_DT, ed)
    assert ps.freshness_scale_from_env(ed) == 1.0
    # malformed / out-of-range => default (no raise).
    assert ps.dt_from_env({ps.PHYSICS_DT_ENV: 'xyz'}) == DEFAULT_DT
    assert ps.dt_from_env({ps.PHYSICS_DT_ENV: '10'}) == DEFAULT_DT   # 10 s is past MAX_DT


# --------------------------------------------------------------------------- controller band

class _StubMotion:
    """TransferController.__init__ never calls a motion method; it only stores the port."""
    pass


def _controller_bands(physics_dt_value, env=None):
    from scripts.twin.restock_manipulation import TransferController, ProductProfile
    prof = {'sku-box': ProductProfile('sku-box', (.1, .1, .1), (1., 0., 0., 0.))}
    for k, v in (env or {}).items():
        os.environ[k] = v
    try:
        with physics_dt_env(physics_dt_value):
            c = TransferController(_StubMotion(), prof, fixture_mode=True)
            return c.max_age, c.render_lead, c.collision_stale_tol_s
    finally:
        for k in (env or {}):
            os.environ.pop(k, None)


def test_controller_band_byte_identical_at_default():
    # env unset AND env explicitly 1/60 both reproduce the legacy defaults (0.5 / 0.2 / 0.8).
    for dt in (None, DEFAULT_DT):
        max_age, lead, tol = _controller_bands(dt)
        assert max_age == pytest.approx(0.5)
        assert lead == pytest.approx(0.2)
        assert tol == pytest.approx(0.8)


def test_controller_band_scales_at_raised_step():
    max_age, lead, tol = _controller_bands(RAISED_DT)
    assert max_age == pytest.approx(0.5 * RATIO)
    assert lead == pytest.approx(0.2 * RATIO)
    assert tol == pytest.approx(0.8 * RATIO)


def test_controller_band_scales_the_integration_env_value():
    # The rtfcollect config set RESTOCK_SENSOR_MAX_AGE_S=1.5 / RESTOCK_FRESH_RENDER_LEAD_S=0.5 (the band
    # [-0.5, 1.5] in the trip). Those env values are the window AT 1/60; the step rebases them.
    env = {'RESTOCK_SENSOR_MAX_AGE_S': '1.5', 'RESTOCK_FRESH_RENDER_LEAD_S': '0.5'}
    max_age, lead, _ = _controller_bands(DEFAULT_DT, env)
    assert max_age == pytest.approx(1.5) and lead == pytest.approx(0.5)   # byte-identical at 1/60
    max_age, lead, _ = _controller_bands(RAISED_DT, env)
    assert max_age == pytest.approx(2.5)        # 1.5 * 5/3 -- the band that would have held the trip
    assert lead == pytest.approx(0.5 * RATIO)


# --------------------------------------------------------------------------- rtfcollect REPLAY

# Exact numbers from the preserved run dir
# (datasets/twin-robot-in-store-boot-v1/mission/rtfcollect-fail1-saved/):
#   last fresh host ESDF stamp before the warming gap  : collision_time = 59.1111 s (sim)
#   the mission controller held that stamp; `now` advanced; the invariant tripped at:
#   now = 60.6389 s  ->  collision_dt = 1.5278 s  >  max_age 1.5  (band [-0.5, 1.5]) at physics_dt 1/36.
TRIP_COLLISION_TIME = 59.1111
TRIP_NOW = 60.6389
TRIP_MAX_AGE_BASE = 1.5              # the window value AT 1/60 (the integration env)


def _collision_trips(now, collision_time, dt, max_age_base, render_lead_base=0.5):
    """The controller's collision-freshness predicate (controller._fresh / the collision_dt invariant),
    with the step-aware band. Returns True iff the invariant TRIPS (not fresh)."""
    max_age = ps.scale_freshness_window(max_age_base, dt)
    render_lead = ps.scale_freshness_window(render_lead_base, dt)
    age = now - collision_time
    fresh = -render_lead <= age <= max_age
    return not fresh


def _legacy_fixed_trips(now, collision_time, max_age_base, render_lead_base=0.5):
    """The PRE-FIX behaviour: the band is FIXED sim-seconds regardless of the step."""
    age = now - collision_time
    return not (-render_lead_base <= age <= max_age_base)


def test_replay_legacy_fixed_band_reproduces_the_trip_at_1_36():
    # The recorded failure: at 1/36 the FIXED 1.5 s band trips on the 1.5278 s staleness.
    assert _legacy_fixed_trips(TRIP_NOW, TRIP_COLLISION_TIME, TRIP_MAX_AGE_BASE) is True
    age = TRIP_NOW - TRIP_COLLISION_TIME
    assert age == pytest.approx(1.5278, abs=1e-3)
    assert age > TRIP_MAX_AGE_BASE                       # 1.5278 > 1.5 -> the stage-killing trip


def test_replay_step_aware_band_holds_the_trip_with_margin():
    # With the step-aware band the SAME staleness does NOT trip at 1/36.
    assert _collision_trips(TRIP_NOW, TRIP_COLLISION_TIME, RAISED_DT, TRIP_MAX_AGE_BASE) is False
    band_at_36 = ps.scale_freshness_window(TRIP_MAX_AGE_BASE, RAISED_DT)
    age = TRIP_NOW - TRIP_COLLISION_TIME
    assert band_at_36 == pytest.approx(2.5)
    margin_s = band_at_36 - age
    assert margin_s == pytest.approx(0.972, abs=2e-3)    # ~0.97 s (39 % of the band) of headroom
    # stated as a FRAME margin: the worst observed staleness was ~55 frames vs a constant 90-frame budget.
    assert age / RAISED_DT == pytest.approx(55.0, abs=0.1)
    assert band_at_36 / RAISED_DT == pytest.approx(90.0)


def test_replay_band_is_byte_identical_at_1_60():
    # At 1/60 the step-aware band equals the legacy fixed band for the identical staleness verdict.
    for extra in (-0.3, -0.05, 0.0, 0.3, 0.6):           # sweep ages on both sides of the boundary
        now = TRIP_COLLISION_TIME + TRIP_MAX_AGE_BASE + extra
        assert (_collision_trips(now, TRIP_COLLISION_TIME, DEFAULT_DT, TRIP_MAX_AGE_BASE)
                == _legacy_fixed_trips(now, TRIP_COLLISION_TIME, TRIP_MAX_AGE_BASE))


def test_replay_genuinely_dead_esdf_still_trips_at_every_step():
    # Safety preserved: a staleness BEYOND the frame budget (a genuinely frozen/dead ESDF, > 90 frames)
    # still trips at BOTH steps -- the band is rebased, never removed.
    for dt in (DEFAULT_DT, RAISED_DT):
        budget_frames = ps.scale_freshness_window(TRIP_MAX_AGE_BASE, dt) / dt   # == 90
        dead_age = (budget_frames + 5) * dt                                     # 95 frames stale
        now = TRIP_COLLISION_TIME + dead_age
        assert _collision_trips(now, TRIP_COLLISION_TIME, dt, TRIP_MAX_AGE_BASE) is True


# --------------------------------------------------------------------------- host ESDF gate

def test_incorporation_ack_esdf_window_scales():
    from scripts.twin.restock_host.incorporation_ack import IncorporationAckState
    with physics_dt_env(None):
        st = IncorporationAckState()
        assert st._esdf_fresh_window_s == pytest.approx(0.5)      # byte-identical at default
    with physics_dt_env(RAISED_DT):
        st = IncorporationAckState()
        assert st._esdf_fresh_window_s == pytest.approx(0.5 * RATIO)   # 0.833 s = 30 frames @ 1/36


def test_incorporation_ack_stays_ready_longer_at_raised_step():
    # The host 'ready' gate is what flipped to 'warming' (collision_time=-1) in rtfcollect, which aged
    # out the mission's held stamp. At 1/36 the SAME ~0.7 s ESDF cadence gap no longer drops to warming.
    from scripts.twin.restock_host.incorporation_ack import IncorporationAckState
    gap = 0.7     # sim-s between the integrated stamp and `now` (within 30 frames @ 1/36, >30 @ 1/60)
    for dt, expect_ready in ((DEFAULT_DT, False), (RAISED_DT, True)):
        with physics_dt_env(dt):
            st = IncorporationAckState()
            st.active = True
            st.parked_since = 0.0
            st.collision_time = 10.0
            st.last_heartbeat = 10.0 + gap       # fresh liveness so only the ESDF-freshness gate decides
            st.wall_heartbeat = st.wall_clock()
            snap = st.snapshot(10.0 + gap, endpoints_ready=True, process_alive=True)
        assert snap['reason'] == ''              # liveness healthy; only the ESDF-freshness gate decides
        assert (snap['state'] == 'ready') is expect_ready


# --------------------------------------------------------------------------- camera render-lead

def test_camera_render_lead_scales_and_accepts_one_frame_lead_at_1_36():
    from scripts.twin.restock_camera import freshness as fr
    with physics_dt_env(None):
        assert fr._render_lead() == pytest.approx(0.02)
    with physics_dt_env(RAISED_DT):
        lead = fr._render_lead()
        assert lead == pytest.approx(0.02 * RATIO)
        # the KEY correctness point: one physics step at 1/36 (0.0278 s) EXCEEDS the fixed 0.02 s lead,
        # so a legitimately one-frame-ahead stamp would be mis-rejected as future pre-fix; the scaled
        # lead admits it.
        one_frame = RAISED_DT
        assert one_frame > 0.02                 # would FALSE-reject under the fixed lead
        assert one_frame <= lead                # accepted under the step-aware lead


def test_camera_frame_deadline_rejects_one_frame_lead_without_scale():
    # Demonstrate the latent pre-fix defect directly through frame_deadline at 1/36 with the scale OFF.
    from scripts.twin.restock_camera import freshness as fr
    now = 5.0
    one_frame_future = now + RAISED_DT          # stamp one step ahead (render leads the sim clock)
    frame = {'rendering_time': one_frame_future}
    with physics_dt_env(None):                  # scale == 1.0 => the legacy 0.02 lead
        with pytest.raises(ValueError):
            fr.frame_deadline(frame, now, now, budget_s=0.5, use_acquired=False)
    with physics_dt_env(RAISED_DT):             # step-aware lead admits the one-frame future stamp
        fr.frame_deadline(frame, now, now, budget_s=0.5, use_acquired=False)
