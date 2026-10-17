"""CPU tests for the demand-driven mission job adapter (twin.restock.job_from_plan).

Validates the store-vision job-plan export -> mission Job mapping, the planogram
SKU/slot validation (and honest refusals), and that the substitution is a faithful
drop-in: the committed example maps to the scenario's hardcoded job byte-for-byte.
"""
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
from twin.restock.inventory import Job
from twin.restock.job_from_plan import (
    SCHEMA,
    PlanError,
    apply_to_scenario,
    load_plan,
    resolve_job,
    slot_candidates,
)

PLANOGRAM = SCRIPTS / "planogram.json"
SCENARIO = SCRIPTS / "restock_scenario.json"
EXAMPLE = SCRIPTS / "restock_job_plan.example.json"


def _plan(**over):
    base = {
        "schema": SCHEMA,
        "sku": "juice_orange",
        "quantity": 1,
        "source": {"location": "backstock_00"},
        "destination": {"slot_id": "G3_back_s2", "shelf_id": "G3_back"},
        "priority": 10.0,
        "provenance": {"mission_id": "restock-one"},
    }
    base.update(over)
    return base


class SlotNormalization(unittest.TestCase):
    def test_gondola_side_underscore_collapses(self):
        self.assertEqual(slot_candidates("G3_R_b0_s1"), ["G3_R_b0_s1", "G3R_b0_s1"])

    def test_endcap_and_direct_slots_pass_through(self):
        self.assertEqual(slot_candidates("G3_back_s2"), ["G3_back_s2"])


class ResolveJob(unittest.TestCase):
    def test_direct_endcap_slot_resolves(self):
        job, meta = resolve_job(_plan(), PLANOGRAM)
        self.assertEqual(job, Job("restock-one", "juice_orange", 1, "backstock_00", "G3_back_s2"))
        self.assertEqual(meta["resolved_slot_id"], "G3_back_s2")

    def test_gondola_side_slot_is_normalised_to_planogram(self):
        plan = _plan(destination={"slot_id": "G3_R_b0_s1"})
        job, meta = resolve_job(plan, PLANOGRAM)
        self.assertEqual(job.shelf_id, "G3R_b0_s1")
        self.assertEqual(meta["plan_slot_id"], "G3_R_b0_s1")

    def test_job_id_prefers_mission_then_task_then_fallback(self):
        self.assertEqual(resolve_job(_plan(provenance={"task_id": "t7"}), PLANOGRAM)[0].job_id, "t7")
        self.assertEqual(resolve_job(_plan(provenance={}), PLANOGRAM)[0].job_id,
                         "plan-juice_orange-G3_back_s2")

    def test_explicit_job_id_override(self):
        job, _ = resolve_job(_plan(), PLANOGRAM, job_id="custom-9")
        self.assertEqual(job.job_id, "custom-9")

    def test_unknown_sku_refused(self):
        with self.assertRaises(PlanError):
            resolve_job(_plan(sku="not_a_product"), PLANOGRAM)

    def test_unknown_slot_refused(self):
        with self.assertRaises(PlanError):
            resolve_job(_plan(destination={"slot_id": "ZZ_nope_s0"}), PLANOGRAM)

    def test_slot_that_does_not_stock_sku_refused(self):
        # G1L_b0_s0 is a cereal slot; refuse restocking juice there.
        with self.assertRaises(PlanError):
            resolve_job(_plan(destination={"slot_id": "G1L_b0_s0"}), PLANOGRAM)

    def test_bad_quantity_refused(self):
        with self.assertRaises(PlanError):
            resolve_job(_plan(quantity=0), PLANOGRAM)


class LoadPlan(unittest.TestCase):
    def _write(self, obj):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(obj, f)
        f.close()
        return f.name

    def test_wrong_schema_refused(self):
        with self.assertRaises(PlanError):
            load_plan(self._write(_plan(schema="other/v9")))

    def test_missing_field_refused(self):
        bad = _plan()
        del bad["source"]
        with self.assertRaises(PlanError):
            load_plan(self._write(bad))

    def test_valid_plan_loads(self):
        self.assertEqual(load_plan(self._write(_plan()))["sku"], "juice_orange")


class ApplyToScenario(unittest.TestCase):
    def test_matching_plan_reproduces_scenario_exactly(self):
        scenario = json.loads(SCENARIO.read_text())
        job, _ = resolve_job(_plan(), PLANOGRAM)
        self.assertEqual(asdict(job), scenario["job"])
        out = apply_to_scenario(scenario, job)
        self.assertEqual(out, scenario)  # faithful drop-in: no incidental drift

    def test_new_job_updates_inventory_and_handling_unit(self):
        scenario = json.loads(SCENARIO.read_text())
        job = Job("j2", "water_05l", 1, "backstock_05", "G3R_b0_s1")
        out = apply_to_scenario(scenario, job)
        self.assertEqual(out["job"], asdict(job))
        rows = [(r["location"], r["sku"], r["quantity"]) for r in out["initial_inventory"]]
        self.assertIn(("backstock_05", "water_05l", 1), rows)
        self.assertEqual(out["handling_unit"]["sku"], "water_05l")
        # unrelated fields untouched
        self.assertEqual(out["navigation_model"], scenario["navigation_model"])


class LauncherWiring(unittest.TestCase):
    """run_mobile_restock._resolve_job_override: opt-in, default-off."""

    def setUp(self):
        import run_mobile_restock  # noqa: F401 -- import must succeed on CPU
        self.r = run_mobile_restock

    def test_absent_job_from_is_a_noop(self):
        import types
        scenario = {"job": {"job_id": "x"}, "other": [1, 2]}
        out, job, meta = self.r._resolve_job_override(types.SimpleNamespace(job_from=None), scenario)
        self.assertIs(out, scenario)
        self.assertIsNone(job)
        self.assertIsNone(meta)

    def test_present_job_from_resolves_example(self):
        import types
        scenario = json.loads(SCENARIO.read_text())
        args = types.SimpleNamespace(job_from=str(EXAMPLE), job_id=None, planogram=None)
        out, job, meta = self.r._resolve_job_override(args, scenario)
        self.assertEqual(job.shelf_id, "G3_back_s2")
        self.assertEqual(out["job"], scenario["job"])


class CommittedExample(unittest.TestCase):
    def test_example_fixture_maps_to_hardcoded_job(self):
        plan = load_plan(EXAMPLE)
        job, meta = resolve_job(plan, PLANOGRAM)
        scenario = json.loads(SCENARIO.read_text())
        self.assertEqual(asdict(job), scenario["job"])
        self.assertEqual(meta["provenance"]["approval"]["state"], "approved")


if __name__ == "__main__":
    unittest.main()
