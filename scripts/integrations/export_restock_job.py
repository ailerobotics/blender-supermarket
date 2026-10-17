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
    dims_mm = tuple(float(x) for x in args.dims_mm)
    catalog = MockProductCatalog(
        {args.sku: ProductInfo(sku=args.sku, dims_mm=dims_mm, weight_kg=args.weight_kg)}
    )
    backroom = MockBackroomInventory(
        [BackroomItem(sku=args.sku, quantity=max(args.quantity, 1), location=args.source_bin,
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
        kind=TaskKind(args.task_kind),
        sku=args.sku,
        slot_id=args.slot,
        shelf_id=args.shelf_id,
        facings_needed=args.quantity,
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
        "provenance": {
            "planner": "storevision.restock",
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
        },
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
