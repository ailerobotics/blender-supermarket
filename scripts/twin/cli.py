"""Argument parsing and dry-run rendering for compose_robot_in_store.

Pure (no pxr/isaacsim). ``build_parser`` and ``render_plan`` are unit tested.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Mapping

HOME = os.path.expanduser("~")
DEFAULT_STORE = os.path.join(HOME, "workspaces/blender-supermarket/export/supermarket.usdc")
DEFAULT_ROBOT_URDF = os.path.join(
    HOME, "workspaces/shelfbot_newgripper/description/shelfbot_gripper_isaac.urdf")
DEFAULT_SAVE = "/tmp/store_with_robot.usd"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compose_robot_in_store.py",
        description="Compose the shelfbot UR10 into the supermarket twin for Isaac Sim. "
                    "Run under env_isaaclab. Booting Isaac Sim is a GPU session — see --dry-run "
                    "and the GPU guard; do NOT run beside the captain's live Blender GUI.")
    p.add_argument("--store", default=DEFAULT_STORE, help="Store USD to reference (default: %(default)s)")
    p.add_argument("--robot-urdf", default=DEFAULT_ROBOT_URDF,
                   help="Robot URDF to import (default: %(default)s)")
    p.add_argument("--robot-usd", default=None,
                   help="Pre-converted robot USD; if given, the URDF->USD import step is skipped.")
    # Spawn
    p.add_argument("--spawn-aisle", default="Cam_Aisle",
                   help="Camera prim whose XY marks the aisle centreline for the pedestal "
                        "(default: %(default)s — the G3-G4 aisle).")
    p.add_argument("--spawn-xy", nargs=2, type=float, metavar=("X", "Y"), default=None,
                   help="Explicit pedestal XY (overrides --spawn-aisle).")
    p.add_argument("--min-clearance", type=float, default=0.6,
                   help="Required planar clearance (m) at the pedestal spot (default: %(default)s).")
    p.add_argument("--yaw-deg", type=float, default=0.0, help="Robot yaw about Z at spawn (deg).")
    # Colliders
    p.add_argument("--product-collider", choices=["convex", "none"], default="convex",
                   help="Author convex colliders on product/tag prototypes (default) or skip them "
                        "(fixtures + ground still get triangle-mesh colliders).")
    # Reactive-arm evaluation (opt-in, default OFF)
    p.add_argument("--arm-mpc", action="store_true", default=False,
                   help="EVALUATION-ONLY opt-in: flag the cuRobo MPC reactive follow-and-avoid "
                        "arm-control evaluation (lane sim-arm-mpc-avoid-v1). This does NOT change "
                        "the composed stage or the mission's cuMotion arm path -- absent it the run "
                        "is byte-identical. MPC is a SOFT collision cost, not a safety function; the "
                        "measured evidence comes from scripts/run_arm_mpc_bench.py. See "
                        "docs/arm_mpc_avoid.md.")
    # Outputs
    p.add_argument("--save", default=DEFAULT_SAVE, help="Composed stage output (default: %(default)s).")
    p.add_argument("--screenshot", default=None,
                   help="Screenshot output (default: <save> with .png).")
    p.add_argument("--json", action="store_true",
                   help="With --author-offline, also emit the composed result + measured facts as JSON.")
    p.add_argument("--steps", type=int, default=180, help="Physics steps to run (default: %(default)s "
                   "= 3 s at 60 Hz — enough for the arm to settle from its home pose).")
    p.add_argument("--sample-every", type=int, default=6,
                   help="Record the robot's world pose every N physics steps for the settle "
                        "log / base-height verdict (default: %(default)s).")
    # Modes
    p.add_argument("--dry-run", action="store_true",
                   help="Author nothing and boot nothing: probe the USD (no GPU) and print the "
                        "exact plan, spawn pose and collider counts.")
    p.add_argument("--report-usd", action="store_true",
                   help="Probe the store USD (no GPU boot) and print measured facts as JSON, then exit.")
    p.add_argument("--author-offline", action="store_true",
                   help="Actually author the composed stage with pure USD (NO GPU boot): sublayer "
                        "the store, author physics + colliders, place the robot (a faithful stub "
                        "unless --robot-usd is given), save to --save and VALIDATE it. Exit 0 = the "
                        "composition is sane, 1 = it failed a check. This is the offline proof that "
                        "everything except the simulator itself works.")
    p.add_argument("--headless", dest="headless", action="store_true", default=True,
                   help="Headless SimulationApp (default).")
    p.add_argument("--no-headless", dest="headless", action="store_false",
                   help="Windowed (needs a display; heavier — not recommended on this box).")
    # GPU guard
    p.add_argument("--vram-budget-mb", type=int, default=8000,
                   help="Refuse to boot unless at least this many MiB of VRAM are free (default: %(default)s).")
    p.add_argument("--heavy-threshold-mb", type=int, default=1500,
                   help="Treat any other GPU process using >= this many MiB as a blocker (default: %(default)s).")
    p.add_argument("--force", action="store_true",
                   help="Override the soft GPU gates (low headroom / heavy consumer). Does NOT "
                        "override the hard refusal when the protected Blender session is live.")
    return p


def resolve_screenshot(args) -> str:
    if args.screenshot:
        return args.screenshot
    base, _ = os.path.splitext(args.save)
    return base + ".png"


def _fmt_xyz(t) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in t) + ")"


def render_plan(plan: Mapping[str, Any]) -> str:
    """Render the dry-run plan (a plain dict assembled by the compose script)."""
    L = []
    L.append("=== compose_robot_in_store — DRY RUN (nothing authored, nothing booted) ===")
    L.append(f"store USD        : {plan.get('store')}")
    L.append(f"robot URDF       : {plan.get('robot_urdf')}")
    rusd = plan.get("robot_usd")
    if rusd:
        L.append(f"robot USD        : {rusd} (pre-converted — import step SKIPPED)")
    else:
        L.append("URDF -> USD      : REQUIRED as step 1 of a real run. This uses the "
                 "isaacsim.asset.importer.urdf Kit extension, which boots the RTX renderer "
                 "(a GPU session). It CANNOT run without a simulator boot, so it is deferred "
                 "behind the same GPU guard.")
        L.append(f"                   package:// deps must resolve: ur_description "
                 f"(/opt/ros/jazzy/share) and shelfbot (~/workspaces/shelfbot_newgripper — the "
                 f"repo that declares package 'shelfbot' AND holds the gripper meshes; NOT the "
                 f"older ~/workspaces/shelfbot). Run scripts/validate_robot_urdf.py to prove it.")

    facts = plan.get("usd_facts")
    if facts:
        L.append("")
        L.append("--- measured store USD facts (read now, no GPU) ---")
        L.append(f"up axis          : {facts.get('up_axis')}   metres/unit: {facts.get('meters_per_unit')}")
        L.append(f"default prim     : {facts.get('default_prim')}")
        L.append(f"floor Z          : {facts.get('floor_z')}  "
                 + ("(on ground plane)" if facts.get("floor_z") == 0.0
                    else "(NOT 0 — base Z offset applied)"))
        cp = facts.get("collider_targets", {})
        L.append(f"triangle-mesh    : {cp.get('triangle_mesh_meshes')} direct static meshes "
                 f"(structure {cp.get('structure')}, fixtures {cp.get('fixtures')}, "
                 f"signage {cp.get('signage')})")
        L.append(f"convex prototypes: {cp.get('prototypes')} instance prototypes "
                 f"-> {cp.get('instances')} instances inherit (NOT authored per-instance)")

    sp = plan.get("spawn")
    if sp:
        L.append("")
        L.append("--- spawn (pedestal ground pose, derived from geometry) ---")
        L.append(f"aisle            : {sp.get('label')}  clearance {sp.get('clearance_m')} m "
                 f">= {plan.get('min_clearance')} m required")
        L.append(f"pedestal at      : ({sp.get('x')}, {sp.get('y')}, {sp.get('z')})")
        if sp.get("root_translate"):
            L.append(f"robot root xform : {_fmt_xyz(sp['root_translate'])}  "
                     f"(compensates pedestal offset; re-measured post-import)")
        L.append(f"yaw              : {plan.get('yaw_deg')} deg")

    plan_c = plan.get("collider_plan")
    if plan_c:
        L.append("")
        L.append("--- colliders that WOULD be authored ---")
        L.append(f"triangle-mesh (static, approx=none)     : {plan_c.get('triangle_mesh')}")
        L.append(f"convex-hull  (static, approx=convexHull) : {plan_c.get('convex_hull')} "
                 f"({plan.get('product_collider')})")
        L.append(f"instances skipped (authored via proto)  : {plan_c.get('skipped_instances')}")

    gpu = plan.get("gpu")
    if gpu:
        L.append("")
        L.append("--- GPU guard (what a real run would decide right now) ---")
        L.append(gpu.get("message", ""))

    if plan.get("arm_mpc"):
        L.append("")
        L.append("--- --arm-mpc: cuRobo MPC reactive arm (EVALUATION-ONLY) ---")
        L.append("MPC follow-and-avoid is a SOFT collision cost, NOT a safety function, and is")
        L.append("NOT wired into this composition or the mission's cuMotion arm path (the stage")
        L.append("authored here is byte-identical with or without this flag). Measured evidence:")
        L.append("scripts/run_arm_mpc_bench.py (container fm-plan-mpc-curobo-v1). See")
        L.append("docs/arm_mpc_avoid.md and data/sim-arm-mpc-avoid-v1/report.md.")

    L.append("")
    L.append("--- outputs a real run would write ---")
    L.append(f"stage            : {plan.get('save')}")
    L.append(f"screenshot       : {plan.get('screenshot')}")
    L.append("=== end dry run — no GPU session was started ===")
    return "\n".join(L)
