"""Mission-side driver for the opt-in sensing execution gate (PR 91 delivered the seam + replay
driver + tests; this is the IsaacPort/runtime per-tick driver that makes the gate live in the real
mission). Pure CPU: the phase->suspend mapping and the decode-grade render-skip it drives."""
import unittest

from twin.restock.isaac_port import _sensing_frame_due, SENSING_ACTIONS, _render_suspended
from twin.restock_manipulation.isaac_sensors import sensing_render_required, _SENSING_SUSPEND_PHASES


class SensingGateDriverTests(unittest.TestCase):
    def test_phase_suspend_mapping(self):
        # Validated-reach execution phases (incl. the whole cart holding chain preplace/place/retreat,
        # firstmate 286) suspend the decode render; observe and the verify re-validation render;
        # unknown/None fail safe to render-required (never wrongly suspend).
        for phase in _SENSING_SUSPEND_PHASES:
            self.assertFalse(sensing_render_required(phase), phase)
        for phase in ('observe', 'verify', 'idle', 'succeeded', 'failed', None, 'other'):
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
        self.assertFalse(would_render('preplace', 6, 6))    # holding chain: arm occludes -> suspend (fm 286)
        self.assertTrue(would_render(None, 6, 6))           # pre-grasp: fail safe render

    def test_collect_keep_source_render_holds_source_through_occlusion(self):
        # fm446 (D): with collect_keep_source_render the gate does NOT suspend the decode render on the
        # collect reach phases, so the Option-B-seeded hold() keeps getting fresh source frames. Scoped
        # to the collect stage; transfer still suspends; default (off) is byte-identical to the mapping.
        for phase in ('approach', 'contact', 'grasp', 'lift'):
            # default off: suspends on the reach phase (legacy)
            self.assertTrue(_render_suspended(phase, 'collect', False), phase)
            # keep_source on + collect: NOT suspended (source stays live)
            self.assertFalse(_render_suspended(phase, 'collect', True), phase)
            # keep_source on but NOT collect (e.g. transfer): still suspends
            self.assertTrue(_render_suspended(phase, 'transfer', True), phase)
        # observe always renders regardless of the flag/stage
        for keep in (False, True):
            self.assertFalse(_render_suspended('observe', 'collect', keep))


if __name__ == '__main__':
    unittest.main()
