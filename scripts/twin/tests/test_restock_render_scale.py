"""CPU tests for render_scale intrinsics scaling (the resolution/calibration fix)."""
import numpy as np
import pytest

from scripts.twin.restock_camera.geometry import (Intrinsics, scale_intrinsics, project,
                                                  backproject, transform_points)

FULL = Intrinsics(2560, 1920, 2000., 2000., 1280., 960., .15, 3.5)


def _points(n=400, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(.3, 3., n)
    x = rng.uniform(-.6, .6, n) * z
    y = rng.uniform(-.45, .45, n) * z
    return np.column_stack([x, y, z])


def test_scale_intrinsics_scales_resolution_focal_and_keeps_centered_principal():
    for scale in (.5, .375):
        k = scale_intrinsics(FULL, scale)
        assert (k.width, k.height) == (int(2560*scale)//2*2, int(1920*scale)//2*2)
        assert k.width % 2 == 0 and k.height % 2 == 0
        assert (k.cx, k.cy) == (k.width/2, k.height/2)            # stays centred
        assert np.isclose(k.fx, FULL.fx * k.width/FULL.width)
        assert np.isclose(k.fy, FULL.fy * k.height/FULL.height)
        assert (k.near, k.far) == (FULL.near, FULL.far)


@pytest.mark.parametrize('scale', [.5, .375])
def test_reprojection_error_is_bounded_under_scaling(scale):
    """A point projected with scaled intrinsics lands at scale x its full-res pixel, and a
    scaled-pixel backprojection recovers the same 3D ray — i.e. no calibration error."""
    points = _points()
    uv_full, _ = project(points, FULL)
    k = scale_intrinsics(FULL, scale)
    uv_scaled, valid = project(points, k)
    sx, sy = k.width/FULL.width, k.height/FULL.height
    # Projection is consistent with the resolution change to sub-pixel (exact for centred scaling).
    assert np.max(np.abs(uv_scaled - uv_full*[sx, sy])) < 1e-6
    # Backprojecting the scaled pixel with the scaled intrinsics recovers the 3D point.
    recovered = backproject(uv_scaled[valid], points[valid, 2], k)
    assert np.max(np.abs(recovered - points[valid])) < 1e-9


@pytest.mark.parametrize('scale', [.5, .375])
def test_unscaled_intrinsics_on_a_scaled_image_is_the_bug_scaling_fixes(scale):
    """Backprojecting a scaled-resolution pixel with the UNSCALED (full) intrinsics — the bug
    where CameraRig scaled resolution but not intrinsics — is wrong by ~1/scale; the fix removes it."""
    points = _points()
    k = scale_intrinsics(FULL, scale)
    uv_scaled, valid = project(points, k)
    z = points[valid, 2]
    wrong = backproject(uv_scaled[valid], z, FULL)      # full intrinsics on a scaled pixel grid
    right = backproject(uv_scaled[valid], z, k)         # scaled intrinsics (the fix)
    lateral_error = np.linalg.norm((wrong - right)[:, :2], axis=1)
    assert np.max(lateral_error) > .05                  # the bug mislocates points by centimetres+
    assert np.max(np.abs(right - points[valid])) < 1e-9


def test_rendered_context_uses_scaled_intrinsics():
    from scripts.twin.restock_camera.sensing import rendered_context
    camera_config = {'cameras': {'source': {'intrinsics': {
        'width': 2560, 'height': 1920, 'fx': 2000., 'fy': 2000., 'cx': 1280., 'cy': 960.,
        'near': .15, 'far': 3.5}}}, 'required_depth_error_bound_m': .0002}
    context = rendered_context(camera_config, 'source', [], measure_context=lambda f, n: None,
                               clearance_for_slot=lambda *a: None, camera_revision='lens1',
                               render_scale=.375)
    assert context.intrinsics == scale_intrinsics(FULL, .375)
    assert context.intrinsics.width == 960 and context.intrinsics.fx == 2000.*960/2560
