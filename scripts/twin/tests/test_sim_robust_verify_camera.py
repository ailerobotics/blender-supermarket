"""vr1: the post-place verify must not read a shelf camera the stage-scoped teardown destroyed.

CPU-only (no Isaac). Two layers:
  1. The PRIMARY fix -- the fm381 stage-scoped teardown KEEPS the shelf camera alive when a later
     active stage still reads it (camera-mode transfer / the verify-stage InventoryVerifier), and
     still frees it for a collect-only or fully ground-truth mission (byte-identical).
  2. The DEFENSIVE backstop -- a read of a camera whose render product was destroyed fails by name
     (CameraRenderProductReleased) instead of Isaac's raw "Accessed invalid expired 'Camera' prim".
"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.sensing_sources import (shelf_camera_needed, _run_stage_scoped_teardown,
                                           _unused_sensing_camera_ids)
from twin.restock.native_cameras import (RenderedCamera, CameraRenderProductReleased,
                                         destroy_camera_render_product)
from twin.restock.mission import bounded_stages


def _design():
    return {'supports': {'source': {'camera_id': 'source'}, 'cart': {'camera_id': 'cart'},
                         'shelf': {'camera_id': 'shelf'}}}


# ---------------------------------------------------------------- pure stage predicate

def test_shelf_needed_for_camera_transfer_and_camera_verify():
    assert shelf_camera_needed(('transfer', 'verify')) is True
    assert shelf_camera_needed(('verify',)) is True
    assert shelf_camera_needed(('transfer',)) is True


def test_shelf_not_needed_collect_only_or_all_ground_truth():
    # collect-only: no transfer/verify at all
    assert shelf_camera_needed(('collect', 'couple', 'outbound', 'park')) is False
    # transfer + verify both ground-truth -> neither reads the camera
    assert shelf_camera_needed(('transfer', 'verify'),
                               transfer_ground_truth=True, verify_ground_truth=True) is False


def test_shelf_needed_when_only_one_mode_is_ground_truth():
    # the armcam case: transfer is ground-truth, but verify runs the camera InventoryVerifier
    assert shelf_camera_needed(('transfer', 'verify'),
                               transfer_ground_truth=True, verify_ground_truth=False) is True
    # mirror: camera transfer, ground-truth verify
    assert shelf_camera_needed(('transfer', 'verify'),
                               transfer_ground_truth=False, verify_ground_truth=True) is True


def test_shelf_needed_matches_bounded_stages_for_the_armcam_config():
    # start_at=transfer, stop_after=verify, transfer ground-truth, verify = camera (no verify_ground_truth)
    stages = ('collect', 'couple', 'outbound', 'park', 'transfer', 'verify', 'return')
    active = bounded_stages(stages, 'verify', 'transfer')
    active = active[active.index('transfer'):]
    assert active == ('transfer', 'verify')
    assert shelf_camera_needed(active, transfer_ground_truth=True, verify_ground_truth=False) is True


# ---------------------------------------------------------------- teardown keep / free

def test_teardown_frees_shelf_when_not_needed():
    calls = []
    freed = _run_stage_scoped_teardown(_design(), {'shelf': object(), 'cart': object()},
                                       keep_shelf_camera=False,
                                       destroy=lambda cam, stage: calls.append(cam))
    assert freed == ['shelf'] and len(calls) == 1


def test_teardown_keeps_shelf_when_needed():
    calls = []
    freed = _run_stage_scoped_teardown(_design(), {'shelf': object(), 'cart': object()},
                                       keep_shelf_camera=True,
                                       destroy=lambda cam, stage: calls.append(cam))
    # the only teardown-eligible camera (shelf) is kept -> nothing destroyed, verify can read it
    assert freed == [] and calls == []
    # sanity: shelf really is the id the teardown would otherwise free
    assert _unused_sensing_camera_ids(_design()) == ('shelf',)


def test_teardown_absent_camera_is_skipped_not_fatal():
    # keep off, but the live rig never built the shelf camera (dropped): no destroy, no raise
    freed = _run_stage_scoped_teardown(_design(), {'cart': object()}, keep_shelf_camera=False,
                                       destroy=lambda cam, stage: pytest.fail('should not destroy'))
    assert freed == []


# ---------------------------------------------------------------- defensive named failure

class _FakeCamera:
    """Minimal stand-in for an Isaac Camera: a valid zero-distortion pinhole + a frame."""
    prim_path = '/World/restock_camera_shelf/Camera'
    _render_product = None

    def get_opencv_pinhole_properties(self):
        return (320.0, 240.0, 600.0, 600.0, [0.0] * 12)

    def get_current_frame(self, *a, **k):
        return {'rendering_frame': 7, 'rendering_time': 1.0}


def test_rendered_camera_reads_before_release():
    cam = RenderedCamera(_FakeCamera())
    k = cam.get_intrinsics_matrix()
    assert k[0][0] == 600.0 and k[1][2] == 240.0
    assert cam.get_current_frame()['rendering_frame'] == 7


def test_read_after_release_fails_by_name_not_raw_prim_expired():
    cam = RenderedCamera(_FakeCamera())
    destroy_camera_render_product(cam, stage=None)        # the stage-scoped teardown path
    assert cam._render_product_released is True
    with pytest.raises(CameraRenderProductReleased) as ei:
        cam.get_intrinsics_matrix()
    # the named reason carries the camera prim, not an opaque expired-prim trace
    assert 'restock_camera_shelf' in str(ei.value)
    with pytest.raises(CameraRenderProductReleased):
        cam.get_current_frame()


def test_destroy_tolerates_a_bare_camera_input():
    # destroy on a non-RenderedCamera (the release-bench may pass the inner camera) must not raise
    applied = destroy_camera_render_product(_FakeCamera(), stage=None)
    assert isinstance(applied, dict)
