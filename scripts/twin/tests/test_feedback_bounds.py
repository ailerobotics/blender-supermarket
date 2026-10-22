"""fm361: the feedback-staleness watchdog bounds are SIM-ONLY env-overridable so a render-paced gap
can be masked for a proof run (sim 0.2->1.0, wall 30->90) without touching the real-rig-equivalent
defaults, while the gap logging still records the magnitude for the precise fix."""
import importlib
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _snapshot_with(env, probe=None):
    """Reload trajectory_host with env applied, capture a snapshot (bounds + an optional feedback_stale
    probe) WHILE the env is live, then restore defaults. Returns the snapshot dict (reload mutates the
    module in place, so the caller must read the snapshot, not the live module)."""
    from twin.restock import trajectory_host as th
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        importlib.reload(th)
        snap = {'sim': th.FEEDBACK_SIM_STALE_S, 'wall': th.FEEDBACK_WALL_BACKSTOP_S}
        if probe is not None:
            snap['stale'] = th.feedback_stale(*probe)
        return snap
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        importlib.reload(th)   # restore defaults for the rest of the suite


class FeedbackBoundsTests(unittest.TestCase):
    def test_default_bounds_unchanged(self):
        s = _snapshot_with({'RESTOCK_FEEDBACK_SIM_STALE_S': None, 'RESTOCK_FEEDBACK_WALL_BACKSTOP_S': None})
        self.assertEqual(s['sim'], 0.2)
        self.assertEqual(s['wall'], 30.0)

    def test_env_widens_bounds(self):
        s = _snapshot_with({'RESTOCK_FEEDBACK_SIM_STALE_S': '1.0', 'RESTOCK_FEEDBACK_WALL_BACKSTOP_S': '90'})
        self.assertEqual(s['sim'], 1.0)
        self.assertEqual(s['wall'], 90.0)

    def test_feedback_stale_respects_widened_sim_bound(self):
        # A 0.5 s SIM gap trips the default (>0.2) but is tolerated when widened to 1.0.
        probe = dict(sim_now=10.5, feedback_sim_stamp=10.0, wall_now=100.0, wall_received=100.0)
        probe = (10.5, 10.0, 100.0, 100.0)
        self.assertTrue(_snapshot_with({'RESTOCK_FEEDBACK_SIM_STALE_S': None}, probe)['stale'])
        self.assertFalse(_snapshot_with({'RESTOCK_FEEDBACK_SIM_STALE_S': '1.0'}, probe)['stale'])

    def test_bounds_clamped_sane(self):
        s = _snapshot_with({'RESTOCK_FEEDBACK_SIM_STALE_S': '999', 'RESTOCK_FEEDBACK_WALL_BACKSTOP_S': '0.1'})
        self.assertLessEqual(s['sim'], 5.0)
        self.assertGreaterEqual(s['wall'], 5.0)


if __name__ == '__main__':
    unittest.main()
