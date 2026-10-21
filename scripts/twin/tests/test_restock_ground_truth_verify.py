"""fm461: ground-truth transfer target + verify (demo fallback, shelf not camera-commissioned)."""
import math
import unittest

from twin.restock_manipulation.types import Target
from twin.restock_camera.shelf_target import (build_ground_truth_shelf_target_provider, ShelfSlotTarget)


class GroundTruthProvider(unittest.TestCase):
    def test_returns_bay_target_tagged_ground_truth(self):
        p = build_ground_truth_shelf_target_provider('G3_back_s2', (-2.0, 16.424, 0.73))
        t = p(12.5)
        self.assertIsInstance(t, ShelfSlotTarget)
        self.assertEqual(t.location, 'G3_back_s2')
        self.assertEqual(t.pose_station[:3], (-2.0, 16.424, 0.73))
        self.assertTrue(t.clear)
        self.assertEqual(t.provenance['source'], 'ground_truth')   # NEVER 'sensor'
        self.assertEqual(t.stamp, 12.5)


def _make_controller(transfer_ground_truth):
    # minimal controller for the _provenance gate; no motion needed for the gate itself
    from twin.restock_manipulation.controller import TransferController
    from twin.restock_manipulation.types import ProductProfile

    class _M:  # noqa
        def pose(self): return (0,)*7
    prof = ProductProfile('juice_orange', (0.2, 0.2, 0.2), (1., 0., 0., 0.))
    return TransferController(_M(), {'juice_orange': prof},
                             transfer_ground_truth=transfer_ground_truth)


class ProvenanceGate(unittest.TestCase):
    def _target(self, source):
        return Target(observation_id='shelf:1.0', source=source, simulation_time=1.0,
                      location='G3_back_s2', pose=(-2.0, 16.424, 0.73, 0, 0, 0, 1), clear=True,
                      free_dimensions=(0.6, 0.6, 0.5), frame='ur_base_link')

    def test_ground_truth_source_accepted_only_when_enabled(self):
        on = _make_controller(True)
        off = _make_controller(False)
        tgt = self._target('ground_truth')
        self.assertTrue(on._provenance(tgt, 1.2))         # gated ON -> accepted
        self.assertFalse(off._provenance(tgt, 1.2))       # default -> refused (no silent fixture trust)
        # a genuine sensor target is accepted either way (byte-identical default path)
        self.assertTrue(on._provenance(self._target('sensor'), 1.2))
        self.assertTrue(off._provenance(self._target('sensor'), 1.2))


if __name__ == '__main__':
    unittest.main()
