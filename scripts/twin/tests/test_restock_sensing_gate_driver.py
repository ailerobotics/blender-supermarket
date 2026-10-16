"""Mission-side driver for the opt-in sensing execution gate (PR 91 delivered the seam + replay
driver + tests; this is the IsaacPort/runtime per-tick driver that makes the gate live in the real
mission). Pure CPU: the phase->suspend mapping and the decode-grade render-skip it drives."""
import unittest

from twin.restock.isaac_port import _sensing_frame_due, SENSING_ACTIONS
from twin.restock_manipulation.isaac_sensors import sensing_render_required, _SENSING_SUSPEND_PHASES


class SensingGateDriverTests(unittest.TestCase):
    def test_phase_suspend_mapping(self):
        # Validated-reach execution phases suspend the decode render; observe and the preplace/verify
        # re-validations render; unknown/None fail safe to render-required (never wrongly suspend).
        for phase in _SENSING_SUSPEND_PHASES:
            self.assertFalse(sensing_render_required(phase), phase)
        for phase in ('observe', 'preplace', 'verify', 'idle', 'succeeded', 'failed', None, 'other'):
            self.assertTrue(sensing_render_required(phase), phase)

    def test_frame_due_honours_suspension(self):
        # On a decode-grade stage's stride tick the sensing render is due -- UNLESS suspended, then the
        # capture republishes the held registration and no render is forced.
        self.assertTrue(_sensing_frame_due('collect', 12, 6, False))
        self.assertFalse(_sensing_frame_due('collect', 12, 6, True))
        # Off-stride ticks never render; non-decode stages never force a sensing render.
        self.assertFalse(_sensing_frame_due('collect', 13, 6, False))
        self.assertFalse(_sensing_frame_due('couple', 12, 6, False))
        self.assertIn('collect', SENSING_ACTIONS)
        self.assertNotIn('couple', SENSING_ACTIONS)

    def test_driver_end_to_end_mapping(self):
        # The runtime feeds the last manipulation phase to the port each tick; the port suspends via
        # ``not sensing_render_required(phase)`` and skips the forced sensing render while suspended.
        def would_render(phase, tick, stride):
            suspended = not sensing_render_required(phase)
            return _sensing_frame_due('collect', tick, stride, suspended)
        self.assertTrue(would_render('observe', 6, 6))     # observing: render to validate registration
        self.assertFalse(would_render('approach', 6, 6))   # reaching: arm occludes -> suspend
        self.assertFalse(would_render('contact', 6, 6))
        self.assertFalse(would_render('lift', 6, 6))
        self.assertTrue(would_render('preplace', 6, 6))     # re-validation resumes the render
        self.assertTrue(would_render(None, 6, 6))           # pre-grasp: fail safe render


if __name__ == '__main__':
    unittest.main()
