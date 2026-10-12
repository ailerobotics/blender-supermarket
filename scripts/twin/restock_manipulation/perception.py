"""RGB classification + registered depth geometry, with no USD/scene query path.

Load the existing tray_perception plane estimator read-only by explicit source
root. Feed classifier output from perception_ensemble.tray_pipeline.classify_crop.
No colors or USD names are interpreted as SKU identity.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np

from .types import Observation, Target


def load_pose_bridge(source_root):
    path = Path(source_root) / 'labeling' / 'pose_bridge.py'
    spec = importlib.util.spec_from_file_location('_restock_tray_pose_bridge', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


class DepthPerception:
    def __init__(self, pose_bridge, *, min_points=40, min_coverage=.85,
                 max_depth_spread=.02):
        self.bridge = pose_bridge
        self.min_points, self.min_coverage = min_points, min_coverage
        self.max_depth_spread = max_depth_spread

    def _cloud(self, depth_m, mask, intrinsics, base_from_camera):
        depth, mask = np.asarray(depth_m, float), np.asarray(mask, bool)
        k, transform = np.asarray(intrinsics, float), np.asarray(base_from_camera, float)
        if (depth.ndim != 2 or depth.shape != mask.shape or k.shape != (4,)
                or not np.isfinite(k).all() or np.any(k[:2] <= 0)
                or transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)):
            raise ValueError('invalid registered RGB-D calibration')
        valid = mask & np.isfinite(depth) & (depth > .05) & (depth < 4)
        if np.count_nonzero(valid) < self.min_points or valid.sum() / max(1, mask.sum()) < self.min_coverage:
            raise ValueError('insufficient depth coverage; absence is not free space')
        intr = self.bridge.CameraIntrinsics(*k)
        points = self.bridge.backproject_mask(valid, depth, intr)
        return points @ transform[:3, :3].T + transform[:3, 3]

    def observe(self, *, depth_m, mask, intrinsics, base_from_camera,
                classification, observation_id, simulation_time, location,
                rgb_time, depth_time, transform_time):
        """classification is (ranked, status, top_similarity, margin), upstream API.

        Mask must select the detected box's suctionable top face, not the entire
        tray silhouette. Require synchronized RGB/depth/TF and a horizontal plane.
        Detector/segmenter confidence is a separate upstream gate, not similarity.
        """
        stamps = [simulation_time, rgb_time, depth_time, transform_time]
        if not all(np.isfinite(stamps)) or max(stamps) - min(stamps) > .05:
            raise ValueError('RGB/depth/TF not synchronized')
        ranked, status, score, margin = classification
        if (status != 'OK' or not ranked or not np.isfinite([score, margin]).all()
                or score < .6 or margin < .05):
            raise ValueError('uncertain SKU')
        points = self._cloud(depth_m, mask, intrinsics, base_from_camera)
        pose = self.bridge.estimate_pose(points, min_points=self.min_points)
        if abs(pose.normal[2]) < .98 or np.ptp(points[:, 2]) > self.max_depth_spread:
            raise ValueError('unsupported surface: requires horizontal sealed box top')
        # A fit is not a calibrated probability. Confidence here is a gate flag;
        # raw retrieval score/margin must remain in the caller's sensor log.
        return Observation(observation_id, 'sensor', simulation_time, str(ranked[0][0]),
                           location, (*map(float, pose.translation), 0., 0., 0., 1.), 1.)

    def target(self, *, depth_m, mask, intrinsics, base_from_camera,
               support_z, free_dimensions, observation_id, simulation_time, location):
        """ROI is a calibrated shelf/cart slot; support_z comes from a prior depth fit.

        Missing/occluded pixels refuse vacancy. Any point >2 cm above the support
        marks occupied. Vertical clearance must also be measured by the upstream
        shelf scan; dimensions cannot be obtained from a single top-plane view.
        """
        points = self._cloud(depth_m, mask, intrinsics, base_from_camera)
        if not np.isfinite(support_z):
            raise ValueError('invalid support plane')
        centre = np.median(points, axis=0)
        clear = bool(np.all(np.abs(points[:, 2] - support_z) < .02))
        return Target(observation_id, 'sensor', simulation_time, location,
                      (float(centre[0]), float(centre[1]), float(support_z), 0., 0., 0., 1.),
                      clear, tuple(free_dimensions))


class EnsembleObservationPipeline:
    """Reuse loaded YOLO detector and existing classify_crop without copying models.

    Pass classify_crop bound to its loaded gallery (e.g. functools.partial).
    segment_top(rgb, depth_m, xyxy) supplies a sensor-derived top-face mask.
    This is mandatory: treating an entire tray bbox as a suction patch is unsafe.
    location_for_box(xyxy) maps calibrated image ROIs to backroom/cart/shelf IDs.
    This class does not declare a scan complete merely because detection was empty.
    """
    def __init__(self, detector, classify_crop, depth_perception, segment_top,
                 location_for_box):
        self.detector, self.classify_crop = detector, classify_crop
        self.depth_perception, self.segment_top = depth_perception, segment_top
        self.location_for_box = location_for_box

    def observe(self, rgb, *, depth_m, frame_id, **calibration):
        from PIL import Image
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != np.shape(depth_m):
            raise ValueError('RGB and registered depth must match')
        image = Image.fromarray(rgb.astype(np.uint8))
        detections = self.detector.predict(image, imgsz=960, conf=.9,
                                           save=False, verbose=False)[0]
        observations, rejected = [], []
        for index, box in enumerate(detections.boxes):
            try:
                if float(box.conf) < .9:
                    raise ValueError('uncertain detector')
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if not (0 <= x1 < x2 <= rgb.shape[1] and 0 <= y1 < y2 <= rgb.shape[0]):
                    raise ValueError('invalid or truncated detection')
                bounds = (x1, y1, x2, y2)
                location = self.location_for_box(bounds)
                if not location:
                    raise ValueError('outside calibrated inventory ROI')
                classification = self.classify_crop(image.crop(bounds))
                mask = np.asarray(self.segment_top(rgb, depth_m, bounds), bool)
                if mask.shape != rgb.shape[:2]:
                    raise ValueError('invalid top-face mask')
                outside = mask.copy()
                outside[y1:y2, x1:x2] = False
                if outside.any():
                    raise ValueError('top-face mask outside detection')
                observation = self.depth_perception.observe(
                    depth_m=depth_m, mask=mask, classification=classification,
                    observation_id=f'{frame_id}:{index}', location=location, **calibration)
                observations.append(observation)
            except (ValueError, TypeError) as exc:
                rejected.append({'detection': index, 'reason': str(exc)})
        return tuple(observations), tuple(rejected)
