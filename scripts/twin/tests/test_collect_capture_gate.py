"""fm394: the collect phase-aware capture teardown must fire AFTER the lift, never during it."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock.collect_capture_gate import release_after_lift


class ReleaseAfterLiftTests(unittest.TestCase):
    def test_holds_through_pick_and_lift(self):
        # Pre-lift and the lift itself must NOT fire the teardown, even once the carton is welded
        # (the weld latches at grasp; the lift runs while welded). This is the fix: no mid-lift teardown.
        for phase in (None, 'observe', 'approach', 'contact'):
            self.assertFalse(release_after_lift(phase, False))
        for phase in ('grasp', 'lift'):
            self.assertFalse(release_after_lift(phase, True), f'{phase} must not fire during the lift')

    def test_fires_at_first_post_lift_phase(self):
        # Once the controller transitions PAST the lift (to the carry fold / preplace), fire.
        for phase in ('tuck', 'pan', 'transit', 'preplace', 'place', 'release',
                      'retreat', 'retreat_tuck', 'stow', 'verify'):
            self.assertTrue(release_after_lift(phase, True), f'{phase} is post-lift -> must fire')

    def test_requires_the_weld(self):
        # A post-lift phase without a confirmed weld never fires (no carton -> nothing to free against).
        self.assertFalse(release_after_lift('tuck', False))


if __name__ == '__main__':
    unittest.main()
