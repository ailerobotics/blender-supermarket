"""CPU test for the collect-gate replay harness (per-leg verdicts)."""
from twin.tools import replay_collect_gate as rcg


def test_representative_collect_is_all_clear():
    # With only the (clear-of-the-legs) representative cart boxes, every collect leg proceeds.
    source = rcg._build_source(rcg.REPRESENTATIVE_CART_BOXES, None)
    verdicts = rcg.replay_gate([tuple(l) for l in rcg.REPRESENTATIVE_COLLECT_LEGS], source)
    assert len(verdicts) == len(rcg.REPRESENTATIVE_COLLECT_LEGS)
    assert all(v.ok for v in verdicts), [(v.phase, v.action, v.min_clearance) for v in verdicts]


def test_planted_post_on_place_leg_is_refused():
    # Plant a post at the set-down column (the 'place' leg goal) -> that leg is refused by name.
    source = rcg._build_source(rcg.REPRESENTATIVE_CART_BOXES, (0.35, 0.45, 0.50))
    verdicts = rcg.replay_gate([tuple(l) for l in rcg.REPRESENTATIVE_COLLECT_LEGS], source)
    by_phase = {v.phase: v for v in verdicts}
    assert by_phase['place'].blocked
    assert by_phase['place'].nearest_body == 'planted_post'
    # The early, far-from-the-post legs (pick approach/contact) still proceed.
    assert by_phase['approach'].ok and by_phase['contact'].ok


def test_main_returns_nonzero_when_blocked():
    assert rcg.main(['--planted-post', '0.35,0.45,0.50']) == 1
    assert rcg.main([]) == 0
