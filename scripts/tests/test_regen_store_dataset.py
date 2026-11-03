"""CPU-only checks for the dataset regenerator's geometry + manifest logic.

Pure logic only — no Blender, no pxr, no GPU, no running of the full pipeline.
The `*_when_present` tests additionally cross-check a real regenerated dataset
manifest if one is found on disk (otherwise skipped), so this file both guards
the pure helpers and asserts the delivered v3 geometry when it is available.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import regen_store_dataset as regen
from store_environment import PORTAL_WIDTH, PORTAL_X

# Default delivered dataset, if present on this machine.
V3 = Path("~/datasets/sim-store-realism-v3-portal26").expanduser()


def _synthetic_rows(clear_width):
    """Build portal prim X-extents the way export_usd authors them, for a width."""
    left = PORTAL_X - clear_width / 2.0
    right = PORTAL_X + clear_width / 2.0
    return {
        "Wall_back_west": [["Wall_back_west", [-13.25, left]]],
        "Wall_back_east": [["Wall_back_east", [right, 13.25]]],
        "Portal_jamb": [
            ["jw", [left - 0.07, left]],
            ["je", [right, right + 0.07]],
        ],
        "Portal_lintel": [["Portal_lintel", [left, right]]],
    }


class PortalWidthConstantTests(unittest.TestCase):
    def test_ast_reader_matches_imported_constant(self):
        self.assertEqual(regen._portal_width_constant(), PORTAL_WIDTH)

    def test_current_branch_portal_is_2p6(self):
        self.assertEqual(PORTAL_WIDTH, 2.6)


class ClearWidthComputationTests(unittest.TestCase):
    def test_computes_opening_from_rows(self):
        out = regen.clear_width_from_rows(_synthetic_rows(2.6))
        self.assertAlmostEqual(out["clear_width_m"], 2.6, places=6)
        self.assertAlmostEqual(out["west_wall_east_edge_x"], PORTAL_X - 1.3, places=6)
        self.assertAlmostEqual(out["east_wall_west_edge_x"], PORTAL_X + 1.3, places=6)

    def test_jambs_move_outward_when_widened(self):
        before = regen.clear_width_from_rows(_synthetic_rows(2.5))
        after = regen.clear_width_from_rows(_synthetic_rows(2.6))
        self.assertAlmostEqual(before["clear_width_m"], 2.5, places=6)
        self.assertAlmostEqual(after["clear_width_m"], 2.6, places=6)
        # West edge moves west (smaller x), east edge moves east (larger x), 0.05 each.
        self.assertAlmostEqual(before["west_wall_east_edge_x"] - after["west_wall_east_edge_x"], 0.05, places=6)
        self.assertAlmostEqual(after["east_wall_west_edge_x"] - before["east_wall_west_edge_x"], 0.05, places=6)

    def test_missing_walls_give_none(self):
        out = regen.clear_width_from_rows({})
        self.assertIsNone(out["clear_width_m"])


class RefusalGuardTests(unittest.TestCase):
    def test_refuses_protected_v2_dataset(self):
        with self.assertRaises(SystemExit) as ctx:
            regen.main(["--out", str(regen.PROTECTED_DATASET)])
        self.assertIn("protected", str(ctx.exception).lower())

    def test_refuses_output_inside_repo(self):
        with self.assertRaises(SystemExit) as ctx:
            regen.main(["--out", str(regen.ROOT / "export" / "scratch-dataset")])
        self.assertIn("inside the repo", str(ctx.exception).lower())


class DeliveredManifestTests(unittest.TestCase):
    def _load(self):
        mf = V3 / "manifest.json"
        if not mf.exists():
            self.skipTest(f"delivered dataset manifest not present: {mf}")
        return json.loads(mf.read_text())

    def test_manifest_portal_is_2p6_and_consistent(self):
        g = self._load()["geometry"]
        self.assertEqual(g["portal_width_source_constant_m"], 2.6)
        self.assertEqual(g["portal_clear_width_manifest_m"], 2.6)
        self.assertAlmostEqual(g["portal_clear_width_measured_usd_m"], 2.6, places=3)
        self.assertAlmostEqual(g["portal_centre_xy"][0], PORTAL_X, places=6)

    def test_manifest_jambs_moved_outward(self):
        g = self._load()["geometry"]
        # New jamb inner faces sit at PORTAL_X ∓ 1.3 (was ∓ 1.25 at 2.5 m).
        self.assertAlmostEqual(g["portal_west_wall_east_edge_x"], PORTAL_X - 1.3, places=3)
        self.assertAlmostEqual(g["portal_east_wall_west_edge_x"], PORTAL_X + 1.3, places=3)

    def test_manifest_validation_passed(self):
        v = self._load()["usd_validation"]
        self.assertIn(v["verdict"], ("PASS", "PASS_AFTER_TEXTURE_REPAIR"))


if __name__ == "__main__":
    unittest.main()
