"""CPU tests for the per-phase arm depth-feed schedule + the phase-dependent VRAM budget."""
from twin.restock import arm_feed_schedule as sched
from twin.restock import gpu_budget


def test_wrist_live_in_approach_mast_off_when_disabled():
    a = sched.feeds_active_in_phase('approach', mast_enabled=False)
    assert a == frozenset({sched.FEED_WRIST})


def test_mast_live_during_carry_when_enabled():
    a = sched.feeds_active_in_phase('tuck', mast_enabled=True)
    assert sched.FEED_MAST in a and sched.FEED_WRIST in a


def test_mast_dropped_in_live_pose_transfer_place():
    # The mast does not fit at the live_pose transfer peak, so it is force-dropped there.
    a = sched.feeds_active_in_phase('place', mast_enabled=True, live_pose_transfer=True)
    assert sched.FEED_MAST not in a


def test_no_arm_feed_in_nav_phases():
    assert sched.feeds_active_in_phase('couple', mast_enabled=True) == frozenset()


def test_peak_over_collect_phases_is_two_feeds():
    peak = sched.peak_feeds_over_phases(['observe', 'approach', 'lift', 'tuck', 'place'],
                                        mast_enabled=True)
    assert peak == frozenset({sched.FEED_WRIST, sched.FEED_MAST})


def test_collect_budget_fits_wrist_plus_mast_transfer_does_not():
    cfg = {'masked_depth': {'enabled': True}, 'mast_depth': {'enabled': True}}
    collect = gpu_budget.check_collect_budget(cfg)
    assert collect.fits and collect.free_after_mib > 0
    # The SAME two feeds at the live_pose transfer peak do NOT fit.
    transfer = gpu_budget.check_budget([(1280, 720), (1280, 720)])
    assert not transfer.fits


def test_collect_base_is_below_transfer_base():
    # Headroom is phase-dependent: collect peaks lower than the live_pose transfer peak.
    assert gpu_budget.COLLECT_BASE_MISSION_MIB < gpu_budget.BASE_MISSION_MIB
