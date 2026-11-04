#!/usr/bin/env python3
"""Export one approved store-vision restock plan item as a mission job plan JSON.

Producer side of the demand-driven mission job (lane sim-smart-job-feed-v1). This
reuses store-vision's documented public restock/robot/missions models READ-ONLY to
plan a RobotMission for one (sku, slot), route it through the mission-approval gate,
and serialise the approved plan into the `storevision.restock.job_plan/v1` contract
consumed by blender-supermarket's `scripts/twin/restock/job_from_plan.py`.

It never actuates and touches no store-vision state on disk. Run it with the
store-vision virtualenv's interpreter, pointing --store-vision-src at that checkout.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "storevision.restock.job_plan/v1"

# Which store-vision planner chooses / sizes the demo job. The working default is
# ``restock`` (store-vision's restock MissionPlanner, exactly v1). The two smart
# variants consult an additional landed store-vision module READ-ONLY and record
# its distinguishing signal in provenance.planner_signal, without changing the
# v1 restock schema:
#   forecast   -- storevision.forecast gives a probabilistic time-to-empty; the
#                 task is marked STOCKOUT_FORECAST and prioritised soonest-to-empty
#                 first (ForecastAwarePolicy semantics). Quantity is unchanged.
#   freshorder -- storevision.freshorder proposes the demand-driven order quantity
#                 (service-level order-up-to with a perishability waste cap); that
#                 recommendation sets the restocked quantity.
HOUR = 3600.0


def _forecast_signal(args) -> dict:
    """Probabilistic time-to-empty for the SKU from storevision.forecast (read-only).

    Builds a short synthetic hourly demand history (demo stand-in; the mission's
    own sensing owns the real signal) and runs the landed SeasonalForecaster.
    Returns the stockout payload recorded under provenance.planner_signal.
    """
    import numpy as np

    from storevision.forecast.config import ForecastConfig
    from storevision.forecast.demand import DemandModel
    from storevision.forecast.forecaster import SeasonalForecaster
    from storevision.forecast.restock import forecast_to_stockout_payload

    cfg = ForecastConfig()
    now = float(args.now) if args.now else datetime.now(timezone.utc).timestamp()
    n = max(2, int(args.history_hours))
    times = np.array([now - (n - i) * HOUR for i in range(n)], dtype=float)
    demands = np.full(n, max(0.0, float(args.demand_per_hour)), dtype=float)
    model = DemandModel.fit(args.sku, times, demands, cfg)
    res = SeasonalForecaster(cfg).forecast(
        args.sku, current_fill=float(args.current_fill), model=model, now=now,
        capacity=1.0, facings=max(1, int(args.quantity)))
    payload = forecast_to_stockout_payload(res.to_depletion_forecast(), now)
    payload["method"] = res.method
    return payload


def _freshorder_signal(args) -> dict:
    """Demand-driven order quantity for the SKU from storevision.freshorder (read-only)."""
    from storevision.freshorder.config import FreshorderConfig
    from storevision.freshorder.engine import FreshOrderEngine
    from storevision.freshorder.types import FreshItem

    now = float(args.now) if args.now else datetime.now(timezone.utc).timestamp()
    item = FreshItem(
        sku=args.sku,
        on_hand_units=float(args.on_hand_units),
        demand_per_day=max(0.0, float(args.demand_per_hour) * 24.0),
        shelf_life_days=float(args.shelf_life_days),
        case_pack=max(1, int(args.case_pack)),
        lead_time_days=float(args.lead_time_days),
        recent_waste_rate=float(args.recent_waste_rate),
    )
    line = FreshOrderEngine(FreshorderConfig()).propose_item(item, now=now)
    if line is None:
        raise SystemExit(
            f"storevision.freshorder proposed no order for {args.sku!r} "
            "(nothing needed / protected); adjust --on-hand-units / --demand-per-hour")
    return {
        "order_units": float(line.order_units),
        "order_cases": int(line.order_cases),
        "case_pack": int(line.case_pack),
        "order_up_to_level": round(float(line.order_up_to_level), 3),
        "inventory_position": round(float(line.inventory_position), 3),
        "service_level": float(line.service_level),
        "bound_by": line.bound_by,
        "reason": line.reason,
    }


def build_plan(args) -> dict:
    sys.path.insert(0, str(Path(args.store_vision_src).resolve()))
    from storevision.core.config import load_store_config
    from storevision.missions.adapters import restock_mission_record
    from storevision.missions.gate import MissionGate
    from storevision.missions.models import MissionState
    from storevision.restock import (
        BackroomItem,
        FreeSpace,
        MissionPlanner,
        MockBackroomInventory,
        MockFreeSpaceProvider,
        MockProductCatalog,
        ProductInfo,
        RestockTask,
        RobotSpec,
        TaskKind,
    )
    from storevision.restock.mission import slot_pose_lookup_from_config

    cfg = load_store_config(args.config)
    lookup = slot_pose_lookup_from_config(cfg)
    if lookup(args.slot) is None:
        raise SystemExit(f"slot {args.slot!r} not found in store config {args.config}")

    # Confirm the slot's planogram intent matches the requested SKU (store-vision owns
    # the shelf->SKU expectation); refuse a mismatch rather than plan a wrong placement.
    expected = {s.expected_sku for sh, s in cfg.iter_slots() if s.id == args.slot}
    if expected and args.sku not in expected:
        raise SystemExit(f"slot {args.slot!r} expects {sorted(expected)}, not {args.sku!r}")

    # Physical attributes of the sealed handling unit. Defaults match the mission's
    # juice_orange unit (scripts/restock_scenario.json: 0.6x0.4x0.16 m, 0.45 kg). The
    # UR10 + surface gripper handles sub-kg sealed units, so lower the generic tray
    # robot's 1.0 kg payload floor to the unit mass. Free space is a demo stand-in
    # sized to clear the opening; the mission's own sensing owns the real evidence.
    # --- planner variant: working default restock, smart forecast / freshorder ---
    # Resolved first so the backroom source stocks the chosen (demand-driven) quantity.
    planner_name = "storevision.restock"
    planner_signal: dict | None = None
    task_kind = args.task_kind
    quantity = max(1, int(args.quantity))
    if args.planner == "forecast":
        # Forecast drives WHEN/urgency: mark the task a stockout forecast and record
        # the probabilistic time-to-empty; the restocked quantity is unchanged.
        planner_name = "storevision.forecast"
        planner_signal = _forecast_signal(args)
        task_kind = "stockout_forecast"
    elif args.planner == "freshorder":
        # Fresh-ordering drives HOW MUCH is needed (service-level order-up-to with a
        # perishability waste cap). The need is recorded in planner_signal.order_cases;
        # the mission restocks ONE sealed handling unit per trip (scenario single
        # source; validate() enforces job.quantity == 1), so the job quantity stays 1
        # and the full recommendation rides in provenance for batching / the next trip.
        planner_name = "storevision.freshorder"
        planner_signal = _freshorder_signal(args)
        planner_signal["mission_units_per_trip"] = 1
        quantity = 1

    # Physical attributes of the sealed handling unit. Defaults match the mission's
    # juice_orange unit (scripts/restock_scenario.json: 0.6x0.4x0.16 m, 0.45 kg). The
    # UR10 + surface gripper handles sub-kg sealed units, so lower the generic tray
    # robot's 1.0 kg payload floor to the unit mass. Free space is a demo stand-in
    # sized to clear the opening; the mission's own sensing owns the real evidence.
    dims_mm = tuple(float(x) for x in args.dims_mm)
    catalog = MockProductCatalog(
        {args.sku: ProductInfo(sku=args.sku, dims_mm=dims_mm, weight_kg=args.weight_kg)}
    )
    backroom = MockBackroomInventory(
        [BackroomItem(sku=args.sku, quantity=max(quantity, 1), location=args.source_bin,
                      xyz=tuple(float(x) for x in args.source_xyz))]
    )
    free = MockFreeSpaceProvider(default=FreeSpace(
        width_mm=max(dims_mm[0], dims_mm[1]) + 100.0,
        height_mm=dims_mm[2] + 100.0,
        depth_mm=max(dims_mm[0], dims_mm[1]) + 50.0,
        approach_clearance_mm=200.0, source="producer_demo_stub"))
    planner = MissionPlanner(
        RobotSpec(payload_min_kg=min(1.0, args.weight_kg)), backroom, catalog, lookup,
        free_space=free)

    task = RestockTask(
        id=args.task_id,
        kind=TaskKind(task_kind),
        sku=args.sku,
        slot_id=args.slot,
        shelf_id=args.shelf_id,
        facings_needed=quantity,
        priority=args.priority,
    )
    planner.plan(task)
    if task.mission is None:
        raise SystemExit(f"store-vision could not route a robot mission: {task.note}")

    from storevision.missions.models import ApprovalPolicyKind
    from storevision.missions.policy import MissionPolicyConfig

    record = restock_mission_record(task, catalog=catalog)
    # Firstmate default: the human-approval gate is OFF (auto_approve) and selectable.
    gate = MissionGate(MissionPolicyConfig(policy=ApprovalPolicyKind(args.approval_policy)))
    record = gate.submit(record)
    if record.state != MissionState.APPROVED:
        if args.approver:
            record = gate.approve(record.id, by=args.approver, reason="sim-smart-job-feed export")
        else:
            raise SystemExit(
                f"mission held by gate (state={record.state.value}); pass --approver to approve")

    mission = task.mission
    sp = mission.slot_pose
    provenance = {
        "planner": planner_name,
        "task_id": task.id,
        "mission_id": record.id,
        "robot_id": mission.robot_id,
        "task_kind": task.kind.value,
        "est_total_s": float(mission.est_total_s),
        "tray_mode": bool(mission.tray_mode),
        "approval": {
            "state": record.state.value,
            "required": bool(record.approval.required),
            "by": record.approval.by,
            "policy": gate.policy.policy.value,
        },
    }
    if planner_signal is not None:
        provenance["planner_signal"] = planner_signal
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "store_config": Path(args.config).stem,
        "sku": task.sku,
        "quantity": int(task.facings_needed),
        "source": {"location": mission.backroom_location, "xyz": list(mission.pick_xyz)},
        "destination": {
            "shelf_id": record.target_bay_id,
            "slot_id": task.slot_id,
            "slot_pose": {"x": sp.x, "y": sp.y, "z": sp.z, "yaw": sp.yaw},
        },
        "priority": float(task.priority),
        "provenance": provenance,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store-vision-src", required=True, help="store-vision 'src' directory")
    p.add_argument("--config", required=True, help="store config yaml (e.g. demo-freshmart.yaml)")
    p.add_argument("--sku", default="juice_orange")
    p.add_argument("--slot", default="G3_back_s2", help="destination slot id (store-vision namespace)")
    p.add_argument("--shelf-id", default="G3_back")
    p.add_argument("--source-bin", default="backstock_00", help="backroom source location id")
    p.add_argument("--source-xyz", nargs=3, type=float, default=(-5.73, 17.1, 0.6),
                   help="backroom pick pose in the store frame")
    p.add_argument("--weight-kg", type=float, default=0.45)
    p.add_argument("--dims-mm", nargs=3, type=float, default=(600.0, 400.0, 160.0),
                   help="handling unit L W H in mm")
    p.add_argument("--quantity", type=int, default=1)
    p.add_argument("--priority", type=float, default=10.0)
    p.add_argument("--task-id", default="restock-one")
    p.add_argument("--task-kind", default="low_stock")
    p.add_argument("--planner", default="restock",
                   choices=("restock", "forecast", "freshorder"),
                   help="which store-vision planner chooses/sizes the job. Working "
                        "default: restock (MissionPlanner, v1). Smart variants: forecast "
                        "(probabilistic time-to-empty => STOCKOUT_FORECAST) | freshorder "
                        "(service-level order-up-to sets the quantity).")
    # Demo inputs for the smart planners (stand-ins; the mission's own sensing owns the
    # real signal). Ignored by the default restock planner.
    p.add_argument("--now", type=float, default=0.0,
                   help="forecast/freshorder 'now' epoch seconds (0 => current UTC)")
    p.add_argument("--current-fill", type=float, default=0.25,
                   help="[forecast] current shelf fill fraction 0..1")
    p.add_argument("--demand-per-hour", type=float, default=0.08,
                   help="[forecast/freshorder] demand units/hour used to seed the signal")
    p.add_argument("--history-hours", type=int, default=48,
                   help="[forecast] synthetic demand history length (hours)")
    p.add_argument("--on-hand-units", type=float, default=2.0,
                   help="[freshorder] vision-measured on-hand units for the SKU")
    p.add_argument("--shelf-life-days", type=float, default=10.0,
                   help="[freshorder] life of freshly delivered stock (days)")
    p.add_argument("--case-pack", type=int, default=1,
                   help="[freshorder] retail units per ordered case (order granularity)")
    p.add_argument("--lead-time-days", type=float, default=1.0,
                   help="[freshorder] order->arrival lead time (days)")
    p.add_argument("--recent-waste-rate", type=float, default=0.05,
                   help="[freshorder] fraction of recent deliveries wasted (0..1)")
    p.add_argument("--approval-policy", default="auto_approve",
                   choices=("auto_approve", "approve_all", "approve_above_risk"),
                   help="mission-gate policy (firstmate default: auto_approve == gate off)")
    p.add_argument("--approver", default="",
                   help="approver identity when the gate holds the mission for a human")
    p.add_argument("--out", type=Path, help="write JSON here (default: stdout)")
    args = p.parse_args()
    plan = build_plan(args)
    text = json.dumps(plan, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
