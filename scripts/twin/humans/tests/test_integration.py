"""End-to-end (no simulator): people -> ground-truth detection -> reaction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans.config import PersonSpec
from twin.humans.integration import MissionSafety, build_safety
from twin.humans.paths import ScriptedPath


def test_disabled_returns_none():
    assert build_safety(None) is None
    assert build_safety({'humans': {'enabled': False}}) is None


def _one_person_config():
    return {'humans': {
        'enabled': True,
        'detection': {'source': 'ground_truth', 'cameras': ['wrist'],
                      'gt_visible_range_m': 5.0, 'gt_fov_halfangle_rad': 1.2},
        'reaction': {'slow_base_scale': 0.3, 'stop_base_scale': 0.0,
                     'resume_clear_dwell_s': 2.0},
        'people': [{'person_id': 'p1', 'role': 'shopper',
                    'waypoints': [[6.0, 0.0], [0.6, 0.0], [6.0, 0.0]], 'speed_mps': 1.0,
                    'mode': 'once'}],
    }}


def test_ground_truth_end_to_end_slow_stop_resume():
    cfg = _one_person_config()
    # Walk the configured person's own path; base at origin facing +x; camera at base.
    person = PersonSpec.from_dict(cfg['humans']['people'][0]).path()

    def people_positions(now):
        x, y, _ = person.pose_at(now)
        return [('p1', (x, y, 0.9), 0.25)]

    safety = build_safety(cfg, people_positions=people_positions,
                          base_pose=lambda: (0.0, 0.0, 0.0),
                          camera_poses=lambda: {'wrist': (0.0, 0.0, 0.0)})
    assert isinstance(safety, MissionSafety) and safety.provenance == 'ground_truth'

    zones, scales, holds = [], [], []
    for i in range(140):                       # 14 s at dt=0.1
        now = i * 0.1
        out = safety.update(now)
        zones.append(out.zone)
        scales.append(out.base_scale)
        holds.append(out.arm_hold)

    # The person starts 6 m out (beyond the 5 m view -> CLEAR), walks in to 0.6 m
    # (deep in STOP), then back out past 5 m. We must observe every regime and resume.
    assert zones[0] == 'CLEAR'                   # out of view at the start
    assert 'SLOW' in zones and 'STOP' in zones
    assert min(scales) == 0.0                   # base fully stopped at closest approach
    assert scales[0] == 1.0                      # full speed before the person is in range
    assert scales[-1] == 1.0 and not holds[-1]   # resumed after the person left + dwell
    assert any(holds) and not all(holds)        # arm paused then resumed
    # The shared obstacle contract is populated while the person is visible.
    dets = safety.detections(5.4)
    assert len(dets) == 1 and dets[0].source == 'human_ground_truth'


def test_detector_source_builds_without_pose_accessors():
    cfg = {'humans': {'enabled': True, 'detection': {'source': 'detector'},
                      'people': [{'person_id': 'p', 'role': 'staff',
                                  'waypoints': [[0, 0], [1, 0]], 'speed_mps': 1.0}]}}
    safety = build_safety(cfg)
    assert safety is not None and safety.provenance == 'detector'
    # no detector message yet -> fail-safe STOP
    assert safety.update(0.0).base_scale == 0.0 and safety.update(0.0).arm_hold
