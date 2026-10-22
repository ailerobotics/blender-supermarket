"""Offline tests for the opt-in mission reaction wiring (no USD/ROS/Isaac)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans import gpu_bench
from twin.humans.mission_wiring import (MissionReaction, ReversibleArmPause,
                                        build_mission_reaction)


def test_reversible_arm_pause_is_edge_triggered():
    calls = []
    pause = ReversibleArmPause(calls.append)
    pause.update(False)            # already not paused -> first call sets baseline False
    pause.update(False)            # no change -> no extra call
    pause.update(True)             # rising edge -> pause(True)
    pause.update(True)             # no change
    pause.update(False)            # falling edge -> pause(False)
    assert calls == [False, True, False]
    assert pause.paused is False


def test_build_mission_reaction_disabled_is_none():
    assert build_mission_reaction(None) is None
    assert build_mission_reaction({'humans': {'enabled': False}}) is None


def _crossing_config():
    base = (0.0, 0.0, 0.0)
    cam = (0.0, 0.0, 0.0)
    person = gpu_bench.synthesize_crossing(cam, base, speed_mps=0.9,
                                           start_range_m=6.5, closest_range_m=0.9)
    config = {'humans': {
        'enabled': True,
        'people': [person.to_dict()],
        'detection': {'source': 'ground_truth', 'cameras': ['shelf'],
                      'gt_visible_range_m': 5.5, 'gt_fov_halfangle_rad': 1.2},
        'reaction': {'slow_base_scale': 0.35, 'stop_base_scale': 0.0,
                     'reach_radius_m': 1.3, 'arm_hold_margin_m': 0.3,
                     'resume_clear_dwell_s': 2.0, 'state_timeout_s': 0.7}}}
    path = person.path()

    def people_positions(now):
        x, y, _ = path.pose_at(now)
        return [(person.person_id, (x, y, person.height_m * 0.5), person.radius_m)]

    return config, base, cam, people_positions


def test_mission_reaction_drives_base_scale_and_arm_pause():
    config, base, cam, people_positions = _crossing_config()
    pause_calls = []
    reaction = build_mission_reaction(
        config, people_positions=people_positions, base_pose=lambda: base,
        camera_poses=lambda: {'shelf': cam}, pause=pause_calls.append)
    assert isinstance(reaction, MissionReaction)

    scales, holds = [], []
    dt = 0.1
    for i in range(int(round(20.0 / dt)) + 1):
        out = reaction.step(i * dt)
        scales.append(out.base_scale)
        holds.append(out.arm_hold)

    assert min(scales) == 0.0 and max(scales) == 1.0        # STOP (0.0) and full (1.0) both seen
    assert any(holds) and not all(holds)                    # arm paused then resumed
    # the reversible pause sink saw a rising edge THEN a later falling edge (a real
    # pause->resume), proving the hold is releasable (not a permanent latch)
    assert True in pause_calls and False in pause_calls
    first_pause = pause_calls.index(True)
    assert any(v is False for v in pause_calls[first_pause + 1:]), pause_calls
    # base velocity scaling helper applies the latest scale
    assert reaction.scale_velocity(2.0, 1.0) == (2.0 * reaction.base_scale,
                                                 1.0 * reaction.base_scale)
