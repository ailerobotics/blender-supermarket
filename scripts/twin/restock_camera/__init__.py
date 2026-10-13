"""CPU camera geometry and fail-closed depth calibration; no simulator imports."""
from .geometry import Intrinsics, backproject, project, pose_matrix, look_at, relative_camera
from .calibration import calibrate_support, SupportCalibration
from .lights import (RectFixture, LightRigTuning, store_fixtures, task_fill_fixture,
                     light_rig, tonemapping_settings, author_light_rig, camera_lamp,
                     author_camera_lamps, sphere_illuminance, rect_illuminance,
                     aimed_illuminance, lamp_illuminance, lamp_marker_occlusion)

__all__ = ['Intrinsics', 'backproject', 'project', 'pose_matrix', 'look_at',
           'relative_camera', 'calibrate_support', 'SupportCalibration',
           'RectFixture', 'LightRigTuning', 'store_fixtures', 'task_fill_fixture',
           'light_rig', 'tonemapping_settings', 'author_light_rig', 'camera_lamp',
           'author_camera_lamps', 'sphere_illuminance', 'rect_illuminance',
           'aimed_illuminance', 'lamp_illuminance', 'lamp_marker_occlusion']
