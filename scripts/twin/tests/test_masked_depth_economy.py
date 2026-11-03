"""CPU tests for the opt-in arm masked-depth feed economy (sim-robust-arm-feed-economy-v1).

No Isaac/ROS/GPU: the economy policy (parse, the SIM-clock publish gate, the resolution scaling) and
its pose-error acceptance are pure. The "recorded fixtures" are synthetic depth frames of a carton
face at the RECORDED working distances (the collect grasp standoff ~1.2 m, the wrist standoff ~0.32 m);
the acceptance is the ESDF voxel half-size (the collision/grasp solver's own spatial resolution).
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import masked_depth_economy as e
from twin.restock_camera.arm_mount import collision_camera_intrinsics

# The ESDF voxel size the mission runs at (mission-runtime-*.json manipulation.voxel_size); the arm
# feed's reprojection error must stay under HALF a voxel so a coarser feed does not move an obstacle
# surface into a neighbouring voxel. This is the "existing acceptance" the economy must hold within.
VOXEL_SIZE_M = 0.02
VOXEL_HALF_M = VOXEL_SIZE_M / 2.0
# Recorded working distances (arm_mount / wrist_camera docstrings): the collect grasp standoff and the
# near eye-in-hand wrist standoff. The larger distance is the worst case for pixel-quantization error.
COLLECT_STANDOFF_M = 1.2
WRIST_STANDOFF_M = 0.32


# --- from_block: default-off, validation, fail-closed --------------------------------------------

def test_absent_or_disabled_block_is_inactive_defaults():
    for block in (None, False, {}, {'enabled': False}, {'resolution_scale': 1.0}):
        s = e.from_block(block)
        assert not s.active
        assert s.resolution_scale == 1.0
        assert s.publish_interval_s is None


def test_levers_parse():
    s = e.from_block({'resolution_scale': 0.5, 'publish_interval_s': 0.1})
    assert s.active
    assert s.resolution_scale == 0.5
    assert s.publish_interval_s == 0.1


def test_enabled_true_with_levers_is_active():
    s = e.from_block({'enabled': True, 'resolution_scale': 0.5})
    assert s.active and s.resolution_scale == 0.5


def test_unknown_key_fails_closed():
    with pytest.raises(ValueError):
        e.from_block({'resolution_scalez': 0.5})


@pytest.mark.parametrize('scale', [0.0, -0.1, 1.5, 2, True, 'half', float('nan')])
def test_bad_resolution_scale_fails_closed(scale):
    with pytest.raises(ValueError):
        e.from_block({'resolution_scale': scale})


@pytest.mark.parametrize('interval', [0.0, -0.5, True, 'fast', float('inf')])
def test_bad_publish_interval_fails_closed(interval):
    with pytest.raises(ValueError):
        e.from_block({'publish_interval_s': interval})


def test_publish_interval_must_be_below_nvblox_window():
    # At/over the 0.5 s depth window a rate-capped feed could starve the ESDF -> refuse.
    with pytest.raises(ValueError):
        e.from_block({'publish_interval_s': 0.5})
    with pytest.raises(ValueError):
        e.from_block({'publish_interval_s': 0.6})
    assert e.from_block({'publish_interval_s': 0.49}).publish_interval_s == 0.49


def test_describe_and_summary():
    s = e.from_block({'resolution_scale': 0.5, 'publish_interval_s': 0.1})
    assert 'resolution_scale=0.5' in s.describe()
    assert s.to_summary() == {'resolution_scale': 0.5, 'publish_interval_s': 0.1}


# --- scaled_resolution: even dims, identity at 1.0 -----------------------------------------------

def test_scaled_resolution_identity_at_one():
    assert e.scaled_resolution(424, 240, 1.0) == (424, 240)


def test_scaled_resolution_half_is_even():
    assert e.scaled_resolution(424, 240, 0.5) == (212, 120)


def test_scaled_resolution_floors_to_even_and_min_two():
    w, h = e.scaled_resolution(425, 241, 0.5)
    assert w % 2 == 0 and h % 2 == 0
    assert e.scaled_resolution(4, 4, 0.1) == (2, 2)   # never below 2 px


# --- due_to_publish: SIM-clock rate gate ---------------------------------------------------------

def test_no_interval_publishes_every_tick():
    assert e.due_to_publish(now=5.0, last_publish_s=4.999, interval_s=None)


def test_first_publish_always_fires():
    assert e.due_to_publish(now=0.0, last_publish_s=None, interval_s=0.1)


def test_interval_defers_then_fires():
    assert not e.due_to_publish(now=1.05, last_publish_s=1.0, interval_s=0.1)
    assert e.due_to_publish(now=1.10, last_publish_s=1.0, interval_s=0.1)
    assert e.due_to_publish(now=1.25, last_publish_s=1.0, interval_s=0.1)


def test_boundary_tolerance_does_not_defer_a_whole_tick():
    # A frame one float-epsilon short of the interval still fires (dt accumulation absorbed).
    assert e.due_to_publish(now=1.0 + 0.1 - 1e-9, last_publish_s=1.0, interval_s=0.1)


def test_backwards_clock_fires():
    # A clock reset (now < last) must not wedge the feed off forever.
    assert e.due_to_publish(now=0.1, last_publish_s=50.0, interval_s=0.1)


# --- intrinsics scaling preserves the field of view ----------------------------------------------

def test_intrinsics_scale_uniformly():
    k = collision_camera_intrinsics()
    fx, fy, cx, cy = e.intrinsics_scaled(k.fx, k.fy, k.cx, k.cy, 0.5)
    assert fx == pytest.approx(k.fx * 0.5)
    assert cx == pytest.approx(k.cx * 0.5)
    # FOV is preserved: half-angle = atan(cx/fx) is unchanged by a uniform scale.
    assert np.arctan(cx / fx) == pytest.approx(np.arctan(k.cx / k.fx))


# --- POSE-ERROR ACCEPTANCE on recorded working distances -----------------------------------------

def test_reprojection_shift_within_half_voxel_at_recommended_scale():
    # The recommended economy scale (0.5) keeps the worst-case lateral reprojection error under half a
    # voxel at BOTH recorded working distances -> the solver's collision/grasp pose is unchanged.
    k = collision_camera_intrinsics()
    for z in (COLLECT_STANDOFF_M, WRIST_STANDOFF_M):
        shift = e.reprojection_shift_m(k.fx, 0.5, z)
        assert shift < VOXEL_HALF_M, (z, shift)


def test_full_resolution_shift_is_the_native_floor():
    # scale=1.0 reproduces the native feed's own quantization floor (the baseline the economy adds to).
    k = collision_camera_intrinsics()
    native = e.reprojection_shift_m(k.fx, 1.0, COLLECT_STANDOFF_M)
    half = e.reprojection_shift_m(k.fx, 0.5, COLLECT_STANDOFF_M)
    assert half == pytest.approx(2.0 * native)       # halving resolution doubles the per-pixel error
    assert native < VOXEL_HALF_M


def test_too_aggressive_scale_breaks_acceptance():
    # A scale small enough to push the error past half a voxel at the far standoff must be visible as a
    # FAILED acceptance here (so the test is a real bound, not vacuous) -- the economy is not free.
    k = collision_camera_intrinsics()
    assert e.reprojection_shift_m(k.fx, 0.1, COLLECT_STANDOFF_M) > VOXEL_HALF_M


def _carton_face_depth(width, height, k, z, target_xyz):
    """A synthetic full-res depth frame: a fronto-parallel carton face at depth ``z`` filling the image,
    with the back-projected 3-D ``target_xyz`` marking the grasp point. Returns (depth, target pixel)."""
    depth = np.full((height, width), float(z), dtype=np.float32)
    u = target_xyz[0] / target_xyz[2] * k.fx + k.cx
    v = target_xyz[1] / target_xyz[2] * k.fy + k.cy
    return depth, (u, v)


def _backproject(u, v, z, fx, fy, cx, cy):
    return np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])


def test_recorded_fixture_pose_error_within_acceptance_at_economy_scale():
    # RECORDED-FIXTURE acceptance: an off-centre grasp target at the collect standoff, localized through
    # the FULL-res feed vs the economy (0.5) feed. The economy feed's recovered 3-D pose stays within
    # half a voxel of the true point -- i.e. nvblox builds the SAME obstacle/grasp geometry.
    k = collision_camera_intrinsics()
    scale = 0.5
    target = np.array([0.18, 0.09, COLLECT_STANDOFF_M])   # ~left/up corner of the 0.6x0.4 carton face
    full_depth, (u_f, v_f) = _carton_face_depth(k.width, k.height, k, COLLECT_STANDOFF_M, target)

    # Full-res recovery: nearest pixel centre, back-projected with the native K.
    uf, vf = round(u_f), round(v_f)
    p_full = _backproject(uf, vf, full_depth[vf, uf], k.fx, k.fy, k.cx, k.cy)

    # Economy recovery: same scene rendered at the scaled resolution + scaled K (FOV preserved).
    w, h = e.scaled_resolution(k.width, k.height, scale)
    fx, fy, cx, cy = e.intrinsics_scaled(k.fx, k.fy, k.cx, k.cy, scale)
    econ_depth = e.downscale_depth(full_depth, scale)
    assert econ_depth.shape == (h, w)
    u_e = target[0] / target[2] * fx + cx
    v_e = target[1] / target[2] * fy + cy
    ue, ve = round(u_e), round(v_e)
    p_econ = _backproject(ue, ve, econ_depth[ve, ue], fx, fy, cx, cy)

    # Both recoveries are within half a voxel of the true target; the economy adds no more than the
    # analytic half-pixel bound on top of the native floor.
    assert np.linalg.norm(p_full[:2] - target[:2]) < VOXEL_HALF_M
    assert np.linalg.norm(p_econ[:2] - target[:2]) < VOXEL_HALF_M
    assert np.linalg.norm(p_econ[:2] - target[:2]) <= e.reprojection_shift_m(k.fx, scale, COLLECT_STANDOFF_M) + 1e-9
    # Depth (range) is preserved exactly: a fronto-parallel surface area-averages to the same metres.
    assert econ_depth[ve, ue] == pytest.approx(COLLECT_STANDOFF_M)


# --- downscale_depth: preserves surface range, excludes no-returns -------------------------------

def test_downscale_preserves_constant_surface():
    d = np.full((240, 424), 1.2, dtype=np.float32)
    out = e.downscale_depth(d, 0.5)
    assert out.shape == (120, 212)
    assert np.allclose(out, 1.2)


def test_downscale_excludes_invalid_returns():
    d = np.full((4, 4), 2.0, dtype=np.float32)
    d[0, 0] = 0.0            # no-return sentinel in one corner block
    out = e.downscale_depth(d, 0.5)      # 2x2 output, each cell a 2x2 source block
    assert out.shape == (2, 2)
    # The cell over the invalid pixel averages only the 3 valid 2.0 returns -> still 2.0, not pulled to 0.
    assert np.allclose(out, 2.0)


def test_downscale_identity_at_one():
    d = np.random.default_rng(0).random((8, 8)).astype(np.float32)
    assert np.array_equal(e.downscale_depth(d, 1.0), d)
