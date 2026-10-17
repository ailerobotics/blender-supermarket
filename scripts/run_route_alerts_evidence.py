#!/usr/bin/env python3
"""Route-alerts evidence (CPU, no Blender/Isaac/GPU): the robot as a mobile sensor.

Proves the whole build spec end to end without a GPU session:

  STAGE 1 -- DETECT (needs store-vision importable). Seed a spill and an aisle
  obstruction (plus a moving shopper) into deterministic store-vision synthetic
  aisle frames, then drive the REAL ``RouteAlertsRunner`` over them. Assert:
    * the spill is detected (``spill.detected`` finding + ``robot.route_alert.*``);
    * the aisle obstruction is detected, blocks the aisle in the passability layer,
      and becomes a dynobs ``Detection``;
    * the shopper is MASKED OUT (no obstruction/finding at the person's location --
      GDPR-by-design, people-masked);
    * de-duplicated staff tasks are raised (CLEANING).

  STAGE 2 -- REACT (pure CPU, no store-vision). Take the obstruction DETECTED in
  stage 1 (same store-frame position + radius + ``source`` tag), place it on the
  route, and drive the coupled base+cart through the EXISTING navigation
  dynamic-obstacle layer (the one ``sim-restock-dynobs-v1`` owns) + articulated
  controller. Assert the robot stops for it (``stopped_for_obstacle``, clearance>0)
  and that the coupled planner re-plans a detour AROUND it. This is the (c) seam:
  the obstruction footprint feeds navigation ONLY through that costmap layer.

Self-asserting: prints PASS/FAIL per check and exits 1 on any failure. Artifacts go
OUTSIDE git (default ~/datasets/sim-route-alerts-v1). This is a kinematic +
classical-vision demonstration, not a PhysX / perception-at-GPU run.

Run stage 1 under the store-vision venv (it has store-vision + numpy/opencv/pydantic):

  PYTHONPATH=scripts \\
    /home/manishsaini/firstmate/projects/store-vision/.venv/bin/python \\
    scripts/run_route_alerts_evidence.py --out ~/datasets/sim-route-alerts-v1
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.mobile import MobileSpec  # noqa: E402
from twin.navigation.articulated import ArticulatedMap, ArticulatedPlanner, TrainState  # noqa: E402
from twin.navigation.core import GridMap  # noqa: E402
from twin.navigation.dynamic import Detection, DynamicObstacleConfig, DynamicObstacleLayer  # noqa: E402
from twin.navigation.dynamic_evidence import simulate  # noqa: E402
from twin.navigation.dynamic_scenario import DynamicScenario, Keyframe, ScriptedActor  # noqa: E402

# Faster reaction than the fixed-camera defaults so the short synthetic sequence
# confirms (production uses the detector defaults; see docs/route_alerts.md).
_TUNE = {"persist_seconds": 0.4, "warmup_frames": 2, "clear_seconds": 0.8,
         "min_object_height_m": 0.06}
_OBSTRUCTION_SOURCE = "route_obstruction"


# --------------------------------------------------------------------------- #
# STAGE 1 -- detect from pixels via store-vision
# --------------------------------------------------------------------------- #
def detect_stage(out: Path):
    """Run the real RouteAlertsRunner over seeded frames. Returns (ok, summary, detected)."""
    try:
        from storevision.obstruction.scenes import default_scenes
    except Exception as exc:  # store-vision not importable in this interpreter
        return None, {"skipped": f"store-vision unavailable: {type(exc).__name__}: {exc}"}, None

    from twin.route_alerts import RouteAlertsRunner, RouteFrame

    scene = default_scenes()[0]  # carton_blocks_aisle1: carton + moving shopper
    frames, agents_by_seq = scene.render()
    itf = scene.image_to_floor()
    aisles = {aid: dict(x0=s.x0, y0=s.y0, x1=s.x1, y1=s.y1, axis=s.axis,
                        min_width_m=s.min_width_m, zone_id=s.zone_id)
              for aid, s in scene.aisles.items()}
    cfg = {
        "spill": {"enabled": True, "gpu": False, "tuning": _TUNE},
        "obstruction": {"enabled": True, "gpu": False, "tuning": _TUNE},
        "floors": {scene.camera_id: {"image_to_floor": itf.tolist(), "zone_id": "aisle"}},
        "aisles": aisles,
        "staff_tasks": True,
        "obstruction_source": _OBSTRUCTION_SOURCE,
    }
    out.mkdir(parents=True, exist_ok=True)
    runner = RouteAlertsRunner(cfg, robot_id="shelfbot", sink_dir=str(out))

    all_events, all_robot, all_tasks = [], [], []
    last_detections, last_passability = [], {}
    # Ground-truth person track (to prove it is masked): the moving shopper(s).
    person_xy = [a for a in scene.agents]
    for i, fr in enumerate(frames):
        rf = RouteFrame(camera_id=fr.camera_id, seq=i, rgb=fr.rgb, depth=fr.depth,
                        capture_time=fr.capture_time, zone_id="aisle",
                        agents=agents_by_seq.get(i, []))
        r = runner.observe([rf], now=fr.capture_time)
        all_events += [e.type for e in r.events]
        all_robot += [e.type for e in r.robot_events]
        all_tasks += r.tasks
        if r.detections:
            last_detections = r.detections
        if r.passability:
            last_passability = r.passability
    runner.close()

    # The carton obstruction store footprints are known ground truth; a detection far
    # from any person proves people were masked (no spurious person obstruction).
    gt = {o.obj_id: o.footprint_m for o in scene.objects}
    spill_detected = any(t.startswith("spill.") for t in all_events)
    obstruction_detected = any(t.startswith("obstruction.") for t in all_events)
    blocked = last_passability.get("blocked") or []
    robot_mirrored = len(all_robot) > 0 and all(t.startswith("robot.route_alert.") for t in all_robot)
    tasks_ok = len(all_tasks) > 0
    # People masked: every emitted detection sits on a ground-truth object footprint,
    # not on a person's path. (A person detected as an obstruction would land away
    # from the static carton footprints.)
    def near_object(pos):
        for (x, y, w, h) in gt.values():
            if x - 0.6 <= pos[0] <= x + w + 0.6 and y - 0.6 <= pos[1] <= y + h + 0.6:
                return True
        return False
    people_masked = all(near_object(d.position) for d in last_detections) if last_detections else False

    summary = {
        "scene": scene.name,
        "frames": len(frames),
        "event_types": sorted(set(all_events)),
        "robot_event_types": sorted(set(all_robot)),
        "blocked_aisles": blocked,
        "detections": [{"position": [round(d.position[0], 3), round(d.position[1], 3)],
                        "radius": round(d.radius, 3), "source": d.source} for d in last_detections],
        "staff_tasks": [{"type": str(t.type), "title": t.title} for t in all_tasks],
        "disabled_reason": runner.disabled_reason,
        "checks": {
            "spill_detected": spill_detected,
            "obstruction_detected": obstruction_detected,
            "aisle_blocked": bool(blocked),
            "robot_events_mirrored": robot_mirrored,
            "staff_tasks_raised": tasks_ok,
            "people_masked": people_masked,
            "has_detection": bool(last_detections),
        },
    }
    (out / "detect.json").write_text(json.dumps(summary, indent=2) + "\n")
    ok = (runner.disabled_reason is None and spill_detected and obstruction_detected
          and bool(blocked) and robot_mirrored and tasks_ok and people_masked
          and bool(last_detections))
    # The detected obstruction to carry into stage 2 (largest-radius detection).
    detected = max(last_detections, key=lambda d: d.radius) if last_detections else None
    return ok, summary, detected


# --------------------------------------------------------------------------- #
# STAGE 2 -- react through the dynobs layer (pure CPU)
# --------------------------------------------------------------------------- #
def _aisle():
    return GridMap((-3, -2, 12, 2), ((-3, -2, 12, -1.0), (-3, 1.0, 12, 2)))


def react_stage(out: Path, detected: Detection | None):
    """Drive the coupled base toward the DETECTED obstruction; prove stop + detour."""
    out.mkdir(parents=True, exist_ok=True)
    # Place the stage-1 detection on the route at x=5 on the aisle centre line,
    # preserving its radius + source tag. Absent store-vision, use a representative
    # carton (radius 0.3) so stage 2 still runs; the source records the origin.
    radius = detected.radius if detected is not None else 0.3
    source = detected.source if detected is not None else _OBSTRUCTION_SOURCE
    on_route = Detection(position=(5.0, 0.0), radius=min(radius, 0.6), source=source)

    cfg = DynamicObstacleConfig(confirm_frames=2, moving_speed=0.08,
                                prediction_horizon=1.2, prediction_step=0.3)
    grid, goal = _aisle(), (9.0, 0.0, 0.0)
    scen = DynamicScenario(
        "route_obstruction", "obstruction detected from the robot camera, left on the route",
        (ScriptedActor("route_obstruction", on_route.radius,
                       (Keyframe(0.0, on_route.position[0], on_route.position[1]),),
                       appear=10.0, source=source),))
    run = simulate(scen, grid=grid, goal=goal, config=cfg, dt=0.1, horizon=120.0,
                   blocked_timeout=15.0)
    with (out / "react_trace.jsonl").open("w") as f:
        import dataclasses
        for k in run.ticks:
            f.write(json.dumps(dataclasses.asdict(k)) + "\n")

    # Coupled re-plan AROUND it vs the straight shot (detour proof).
    spec = MobileSpec()
    world = ArticulatedMap(GridMap((-4, -6, 14, 6)), spec)
    planner = ArticulatedPlanner(world, timeout=10)
    start, dgoal = TrainState(2.0, 0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
    straight = planner.plan(start, dgoal)
    layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=1))
    layer.update([Detection((5.0, 0.9), min(radius, 0.6), source=source)], 0.0)
    detour = planner.plan(start, dgoal, layer.emitted_points())
    straight_y = max((abs(s.y) for s in straight.states), default=None)
    detour_y = max((abs(s.y) for s in detour.states), default=None)

    summary = {
        "obstruction_on_route": {"position": list(on_route.position),
                                 "radius": round(on_route.radius, 3), "source": source},
        "closed_loop": run.summary(),
        "replan": {
            "straight_states": len(straight.states), "detour_states": len(detour.states),
            "straight_max_abs_y": round(straight_y, 3) if straight_y is not None else None,
            "detour_max_abs_y": round(detour_y, 3) if detour_y is not None else None,
        },
        "checks": {
            "stopped_for_obstacle": run.stopped_for_obstacle,
            "clearance_nonneg": run.min_clearance is None or run.min_clearance > 0
                                or run.min_clearance == math.inf,
            "detour_around": bool(straight.states and detour.states
                                  and straight_y is not None and detour_y is not None
                                  and straight_y < 0.05 and detour_y > 0.3),
        },
    }
    (out / "react.json").write_text(json.dumps(summary, indent=2) + "\n")
    ok = all(summary["checks"].values())
    return ok, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/datasets/sim-route-alerts-v1"),
                    help="artifact directory (outside git)")
    args = ap.parse_args(argv)
    base = Path(args.out)

    det_ok, det_summary, detected = detect_stage(base / "detect")
    if det_ok is None:
        print(f"[SKIP] detect: {det_summary['skipped']}")
    else:
        for name, val in det_summary["checks"].items():
            print(f"  [{'PASS' if val else 'FAIL'}] detect.{name}")
        print(f"[{'PASS' if det_ok else 'FAIL'}] detect: {json.dumps(det_summary['checks'])}")

    react_ok, react_summary = react_stage(base / "react", detected)
    for name, val in react_summary["checks"].items():
        print(f"  [{'PASS' if val else 'FAIL'}] react.{name}")
    print(f"[{'PASS' if react_ok else 'FAIL'}] react: {json.dumps(react_summary['checks'])}")

    print(f"artifacts under {base}")
    all_ok = react_ok and (det_ok or det_ok is None)
    if det_ok is None:
        print("NOTE: detect stage skipped (store-vision not importable here); "
              "run under the store-vision venv for the full chain.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
