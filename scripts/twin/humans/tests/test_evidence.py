"""The CPU evidence harness observes every regime and renders an SVG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans import evidence
from twin.humans.config import HumansConfig
from twin.humans.perception import SafetyParams

POSES = {
    'pickup': (2.0, 1.0, 0.0), 'cart_dock': (3.5, 1.0, 3.14159),
    'shelf_approach': (9.0, 5.0, 1.5708), 'home': (2.0, 5.0, 0.0),
}
HUMANS = HumansConfig.from_config({'humans': {
    'enabled': True,
    'detection': {'source': 'ground_truth', 'cameras': ['restock_camera_wrist'],
                  'gt_visible_range_m': 6.0, 'gt_fov_halfangle_rad': 1.0},
    'reaction': {'slow_base_scale': 0.35, 'stop_base_scale': 0.0,
                 'resume_clear_dwell_s': 2.0},
}})


def test_demo_observes_slow_stop_pause_resume():
    params = SafetyParams()
    rows = evidence.run(POSES, HUMANS, duration_s=22.0, dt=0.1, drive_s=8.0, params=params)
    summary = evidence.summarize(rows, params)
    assert summary['slow_seen'] and summary['stop_seen']
    assert summary['arm_paused'] and summary['arm_resumed'] and summary['base_resumed']
    assert summary['min_base_scale'] == 0.0 and summary['max_base_scale'] == 1.0
    assert summary['verdict'].startswith('OK')


def test_run_is_deterministic():
    a = evidence.run(POSES, HUMANS, duration_s=10.0, dt=0.1)
    b = evidence.run(POSES, HUMANS, duration_s=10.0, dt=0.1)
    assert a == b


def test_render_svg_is_wellformed():
    rows = evidence.run(POSES, HUMANS, duration_s=12.0, dt=0.1)
    svg = evidence.render_svg(POSES, rows, SafetyParams(), [0.0, 5.0, 10.0])
    assert svg.startswith('<svg') and svg.rstrip().endswith('</svg>')
    assert 'STOP' in svg or 'SLOW' in svg      # a state banner rendered
