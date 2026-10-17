"""Offline tests for the slow/stop/pause reaction mechanism (needs numpy only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans.config import ReactionConfig
from twin.humans.perception import DistanceStatus, SafetyState, Zone
from twin.humans.reaction import ReactionController


def _state(zone, dist_base, stamp):
    return SafetyState(stamp=stamp, zone=zone, nearest_distance=max(0.0, dist_base - 1.3),
                       nearest_distance_base=dist_base,
                       distance_status=DistanceStatus.KNOWN if dist_base != float('inf')
                       else DistanceStatus.NONE,
                       nearest_track_id=1 if dist_base != float('inf') else None,
                       any_person_visible=dist_base != float('inf'))


def test_clear_start_runs_at_full_speed():
    rc = ReactionController(ReactionConfig(resume_clear_dwell_s=2.0))
    out = rc.update(_state(Zone.CLEAR, float('inf'), 0.0), now=0.0)
    assert out.base_scale == 1.0 and out.arm_hold is False and out.zone == 'CLEAR'


def test_slow_zone_slows_base_but_keeps_arm_free_when_far():
    rc = ReactionController(ReactionConfig(slow_base_scale=0.35))
    out = rc.update(_state(Zone.SLOW, 3.0, 0.0), now=0.0)          # person 3 m, outside reach
    assert out.base_scale == 0.35
    assert out.arm_hold is False                                   # not inside the arm envelope
    assert out.person_in_arm_envelope is False


def test_stop_zone_stops_base_and_holds_arm():
    rc = ReactionController(ReactionConfig(stop_base_scale=0.0))
    out = rc.update(_state(Zone.STOP, 1.4, 0.0), now=0.0)
    assert out.base_scale == 0.0 and out.arm_hold is True and out.zone == 'STOP'


def test_person_in_reach_envelope_holds_arm_even_in_slow():
    rc = ReactionController(ReactionConfig(reach_radius_m=1.3, arm_hold_margin_m=0.3))
    # 1.5 m <= 1.3 + 0.3 -> inside the arm envelope
    out = rc.update(_state(Zone.SLOW, 1.5, 0.0), now=0.0)
    assert out.person_in_arm_envelope is True and out.arm_hold is True


def test_missing_or_stale_state_fails_safe():
    rc = ReactionController(ReactionConfig(state_timeout_s=0.7))
    out = rc.update(None, now=0.0)
    assert out.base_scale == 0.0 and out.arm_hold is True and out.stale is True
    # fresh then let it go stale
    rc.update(_state(Zone.CLEAR, float('inf'), 0.0), now=0.0)
    out = rc.update(_state(Zone.CLEAR, float('inf'), 0.0), now=1.0)   # 1.0 s old > 0.7
    assert out.stale and out.base_scale == 0.0 and out.arm_hold


def test_full_cross_and_resume_lifecycle():
    rc = ReactionController(ReactionConfig(
        slow_base_scale=0.3, stop_base_scale=0.0, resume_clear_dwell_s=2.0,
        reach_radius_m=1.3, arm_hold_margin_m=0.3))
    t = 0.0
    # 1) clear
    assert rc.update(_state(Zone.CLEAR, float('inf'), t), t).base_scale == 1.0
    # 2) person approaches -> SLOW (far), base slows, arm still free
    t = 1.0
    o = rc.update(_state(Zone.SLOW, 2.5, t), t)
    assert o.base_scale == 0.3 and o.arm_hold is False
    # 3) person enters -> STOP, base stops, arm pauses
    t = 2.0
    o = rc.update(_state(Zone.STOP, 1.2, t), t)
    assert o.base_scale == 0.0 and o.arm_hold is True
    # 4) person leaves -> CLEAR, but resume is held through the dwell
    t = 3.0
    o = rc.update(_state(Zone.CLEAR, float('inf'), t), t)
    assert o.base_scale == 0.3 and o.arm_hold is True       # still dwelling
    assert 'resume dwell' in o.reason
    # 5) after the dwell elapses, base returns to full and the arm resumes
    t = 5.1
    o = rc.update(_state(Zone.CLEAR, float('inf'), t), t)
    assert o.base_scale == 1.0 and o.arm_hold is False


def test_resume_dwell_restarts_if_person_reappears():
    rc = ReactionController(ReactionConfig(resume_clear_dwell_s=2.0))
    rc.update(_state(Zone.STOP, 1.0, 0.0), 0.0)
    rc.update(_state(Zone.CLEAR, float('inf'), 1.0), 1.0)   # dwell started
    o = rc.update(_state(Zone.STOP, 1.0, 1.5), 1.5)         # person back -> reset
    assert o.base_scale == 0.0 and o.arm_hold
    o = rc.update(_state(Zone.CLEAR, float('inf'), 2.0), 2.0)
    assert o.base_scale < 1.0                                # dwell restarted, not resumed
