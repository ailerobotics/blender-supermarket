"""fm395 (B): the SIM_ATTACH_FALLBACK weld latches deterministically -- a transient Open status flicker
does not tear down a held weld; only a sustained Open (a real suction-off) releases it. Pure/CPU (the
PhysX weld itself needs Isaac; this tests the release-decision the detect loop applies)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation.cad_tool import _release_held_on_open, SIM_ATTACH_OPEN_RELEASE_TICKS


def replay_statuses(statuses, *, release_ticks, welded_from=0):
    """Replay a plugin-status sequence through the detect loop's Open-release gate (mirrors
    sim_attach_contact_ready): True=Open, False=non-Open (Closing/Closed). The weld is considered held
    from index ``welded_from``. Returns the tick index the weld was released, or None if it held throughout."""
    open_streak = 0
    held = False
    for i, is_open in enumerate(statuses):
        if i >= welded_from:
            held = True
        if is_open:
            open_streak += 1
            if held and _release_held_on_open(True, open_streak, release_ticks):
                return i
        else:
            open_streak = 0
    return None


class WeldLatchTests(unittest.TestCase):
    def test_pure_decision(self):
        rt = SIM_ATTACH_OPEN_RELEASE_TICKS
        self.assertTrue(_release_held_on_open(False, 1, rt))      # no weld: Open is a no-op release
        self.assertFalse(_release_held_on_open(True, 1, rt))      # held + 1-tick flicker: HOLD
        self.assertFalse(_release_held_on_open(True, rt - 1, rt))  # held + below threshold: HOLD
        self.assertTrue(_release_held_on_open(True, rt, rt))      # held + sustained: release

    def test_held_weld_survives_transient_flickers(self):
        rt = SIM_ATTACH_OPEN_RELEASE_TICKS
        # A carry/grasp timeline: weld forms at tick 0, then the plugin status flickers Open for 1-2 ticks
        # amid Closing/Closed (the documented flicker + cross-instance lag). The weld must NEVER release.
        seq = [False]                                   # formed (Closing/Closed)
        for _ in range(50):
            seq += [False, False, True, False, True, True, False]   # intermittent 1-2 tick Open flickers
        self.assertIsNone(replay_statuses(seq, release_ticks=rt, welded_from=0),
                          'a transient Open flicker must not release a held weld')

    def test_sustained_open_releases(self):
        rt = SIM_ATTACH_OPEN_RELEASE_TICKS
        # A real suction-off (cross-instance): status goes and STAYS Open -> release after release_ticks.
        seq = [False] + [True] * (rt + 5)
        released = replay_statuses(seq, release_ticks=rt, welded_from=0)
        self.assertEqual(released, 1 + rt - 1, 'sustained Open must release exactly at the threshold')


if __name__ == '__main__':
    unittest.main()
