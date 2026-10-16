"""One configured occlusion window reaches every support camera's context (CPU, no GPU).

run-145: the mission regenerates the camera design via ``config.camera_config`` in
``build_sensing_sources``; ``rendered_context`` reads ``design['occlusion_window_s']``. The design
did not emit it, so every support context fell back to the 20 s default and the held republish
lapsed mid-reach (the collect source_target wraps the SAME per-camera adapter, so it lapsed too).
These tests lock that ``camera_config`` emits the window from ONE scenario knob and that
``rendered_context`` reads it for every support camera -- so one value governs both the station
capture and the collect path.
"""
import numpy as np
import pytest

from scripts.twin.restock_camera.config import camera_config
from scripts.twin.restock_camera.sensing import rendered_context


def _scenario(**extra):
    return dict(handling_unit=dict(local_wall=3, grasp_mode='side', dimensions_m=[.6, .4, .16]),
                initial_staging=dict(support_center_xy=[-5.73, 17.1], unit_bottom_z=.6,
                    target_surface_pose_world=[-3.05, 4.826, .73, 0], cart_load_center_body_xy=[.08, 0],
                    cart_support_z_world=.225), **extra)


MANIFEST = dict(spec=dict(arm_mount_z=.5))


def test_camera_config_emits_the_occlusion_window_from_the_scenario():
    assert camera_config(_scenario(), MANIFEST)['occlusion_window_s'] == 20.0          # default
    assert camera_config(_scenario(occlusion_window_s=60), MANIFEST)['occlusion_window_s'] == 60.0
    assert camera_config(_scenario(occlusion_window_s=None), MANIFEST)['occlusion_window_s'] is None
    for bad in (-1, 901, float('nan')):
        with pytest.raises(ValueError, match='occlusion_window_s'):
            camera_config(_scenario(occlusion_window_s=bad), MANIFEST)


def _ctx(design, camera_id):
    return rendered_context(design, camera_id, [],
                            measure_context=lambda fr, now: (np.eye(4), now, 'e', True),
                            clearance_for_slot=lambda loc, v, fr, now: None, camera_revision='lens1')


@pytest.mark.parametrize('camera_id', ['source', 'shelf', 'cart'])
def test_configured_window_reaches_every_support_camera_context(camera_id):
    # One scenario value -> the single regenerated design -> every support camera's context. The
    # collect source_target wraps the same per-camera adapter, so this is the one value for both.
    design = camera_config(_scenario(occlusion_window_s=60), MANIFEST)
    assert _ctx(design, camera_id).occlusion_window_s == 60.0
    # default scenario keeps 20 s (byte-identical for runs that do not set it)
    assert _ctx(camera_config(_scenario(), MANIFEST), camera_id).occlusion_window_s == 20.0
    # None disables the tolerance (strict), still honored end to end
    assert _ctx(camera_config(_scenario(occlusion_window_s=None), MANIFEST), camera_id).occlusion_window_s is None
