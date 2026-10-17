"""Offline tests for the GPU bench's pure logic (no isaacsim/USD/GPU).

The module must import on a box without isaacsim (the Isaac imports are deferred
into ``run``), the crossing synthesis must be correct geometry, and the synthesized
crossing must actually drive the real ground-truth detection + reaction through
CLEAR -> SLOW -> STOP -> resume (the bench's pass condition).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans import gpu_bench
from twin.humans.config import DetectionConfig, HumansConfig, ReactionConfig
from twin.humans.detection import GroundTruthSafety
from twin.humans.perception import SafetyParams
from twin.humans.reaction import ReactionController


def test_module_imports_without_isaac():
    # The Isaac/USD imports are inside run(); importing the module must not need them.
    assert hasattr(gpu_bench, 'run') and hasattr(gpu_bench, 'synthesize_crossing')


def test_synthesize_crossing_geometry():
    base = (0.0, 0.0, 0.0)
    cam = (0.2, 0.0, 0.0)               # camera just ahead of base, facing +x
    spec = gpu_bench.synthesize_crossing(cam, base, start_range_m=6.5, closest_range_m=0.9)
    far, near = spec.waypoints
    # far/near are placed by distance FROM THE BASE along the camera axis (+x here)
    assert math.isclose(math.hypot(*far), 6.5, abs_tol=1e-6)
    assert math.isclose(math.hypot(*near), 0.9, abs_tol=1e-6)
    assert far[0] > near[0] > 0 and abs(far[1]) < 1e-9     # both down the +x axis


def test_crossing_drives_clear_slow_stop_resume():
    """Replays the bench's per-tick loop with a STATIC base (as the bench runs):
    the synthesized crossing must exercise every regime and resume to full speed."""
    base = (0.0, 0.0, 0.0)
    cam = (0.2, 0.0, 0.0)
    person = gpu_bench.synthesize_crossing(cam, base, speed_mps=0.9,
                                           start_range_m=6.5, closest_range_m=0.9)
    detection = DetectionConfig(source='ground_truth', cameras=('shelf',),
                                gt_visible_range_m=5.5, gt_fov_halfangle_rad=1.2)
    reaction_cfg = ReactionConfig(resume_clear_dwell_s=2.0, reach_radius_m=1.3,
                                  arm_hold_margin_m=0.3)
    humans = HumansConfig(enabled=True, people=(person,), detection=detection,
                          reaction=reaction_cfg)
    path = person.path()

    def people_positions(now):
        x, y, _ = path.pose_at(now)
        return [(person.person_id, (x, y, person.height_m * 0.5), person.radius_m)]

    params = SafetyParams()
    provider = GroundTruthSafety(config=detection, people=people_positions,
                                 base_pose=lambda: base,
                                 camera_poses=lambda: {'shelf': cam}, params=params)
    controller = ReactionController(reaction_cfg)

    zones, scales, holds = [], [], []
    dt = 0.1
    for i in range(int(round(20.0 / dt)) + 1):      # 20 s: a full ping-pong + resume dwell
        now = i * dt
        out = controller.update(provider.state(now), now)
        zones.append(out.zone); scales.append(out.base_scale); holds.append(out.arm_hold)

    assert {'CLEAR', 'SLOW', 'STOP'}.issubset(set(zones)), sorted(set(zones))
    assert any(holds), 'arm never paused'
    # resume to full base speed after being restricted (a <1 -> 1.0 transition)
    assert any(scales[i] < 1.0 and scales[i + 1] == 1.0 for i in range(len(scales) - 1))
    # arm resumes (a hold -> release transition)
    assert any(holds[i] and not holds[i + 1] for i in range(len(holds) - 1))
