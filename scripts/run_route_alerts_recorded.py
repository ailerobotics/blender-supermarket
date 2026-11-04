#!/usr/bin/env python3
"""Route-alerts RECORDED-FRAMES evidence (CPU, no Blender/Isaac/GPU).

Drives the identical production :class:`RouteAlertsRunner` over a SAVED mission run's
recorded camera frames (``.../mission/<run>/mission-frames/``), with a spill and an
obstruction SEEDED into those frames, and proves both delivered modes end to end:

* ``findings`` (the WORKING DEFAULT) -- the spill and obstruction are detected, mirrored
  to ``robot.route_alert.*``, and raised as staff tasks, each stamped with the run's REAL
  mission ``simulation_time``; but ``latest_detections()`` is EMPTY, so the dynobs costmap
  layer is fed nothing and navigation is byte-identical (the robot still reports, never
  swerves).
* ``act`` (the SMART variant, ``--route-alerts=act``) -- the same findings PLUS the
  confirmed obstruction footprint flows, through the EXISTING dynobs
  ``DynamicObstacleLayer`` (the layer ``sim-restock-dynobs-v1`` owns), into extra
  occupancy that raises the route cost; the coupled planner then detours around it.

Honest labelling (see docs/route_alerts.md): the RGB background and the timestamps are
the run's real recorded data; the chase camera has no depth or floor calibration, so the
seeded obstruction's depth + homography are synthetic. The detectors, the event/task
publishing, the mode gate and the dynobs layer are the identical production code.

Run under the store-vision venv (CPU):

  PYTHONPATH=scripts \\
    /home/manishsaini/firstmate/projects/store-vision/.venv/bin/python \\
    scripts/run_route_alerts_recorded.py \\
      --run /home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin33 \\
      --out /home/manishsaini/firstmate/data/sim-route-alerts-v2/recorded
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.navigation.dynamic import DynamicObstacleConfig, DynamicObstacleLayer  # noqa: E402
from twin.route_alerts import (RouteAlertsRunner, RouteFrame, iter_recorded_frames,  # noqa: E402
                               seed_obstruction, seed_spill)
from twin.route_alerts.recorded import floor_calibration  # noqa: E402

# Faster reaction than the fixed-camera defaults so the short replay confirms
# (production uses the detector defaults; see docs/route_alerts.md).
_TUNE = {"persist_seconds": 0.4, "warmup_frames": 2, "clear_seconds": 0.8,
         "min_object_height_m": 0.06}
_SOURCE = "route_obstruction"
_CAM = "route"


def _build_frames(run_dir, *, window, stride, static_bg, box_px, spill_center,
                  spill_from):
    """Load the recorded window and seed hazards. Returns (route_frames, calib)."""
    recorded = list(iter_recorded_frames(run_dir, stride=stride, limit=window))
    if not recorded:
        raise SystemExit(f"no recorded frames under {run_dir}/mission-frames")
    if static_bg:  # repeat one real dwell frame over the run's real timestamps
        bg = recorded[0][0]
        recorded = [(bg.copy(), t, seq) for (_r, t, seq) in recorded]
    shape = recorded[0][0].shape
    calib = floor_calibration(shape)
    depth = seed_obstruction(shape, box_px=box_px)
    frames = []
    for i, (rgb, t, seq) in enumerate(recorded):
        rgb_seeded = seed_spill(rgb, center_px=spill_center) if i >= spill_from else rgb
        frames.append(RouteFrame(
            camera_id=_CAM, seq=seq, rgb=rgb_seeded, depth=depth, capture_time=float(t),
            zone_id="aisle", agents=[]))  # no shoppers in the restock backroom
    return frames, calib


def _drive(frames, calib, *, mode, out):
    """Drive the runner over the seeded frames in one mode; return the summary."""
    cfg = {
        "spill": {"enabled": True, "gpu": False, "tuning": _TUNE},
        "obstruction": {"enabled": True, "gpu": False, "tuning": _TUNE},
        "floors": {_CAM: {"image_to_floor": calib.image_to_floor.tolist(),
                          "polygon": calib.floor_polygon.tolist(), "zone_id": "aisle"}},
        "aisles": {aid: dict(a) for aid, a in calib.aisle.items()},
        "staff_tasks": True,
        "obstruction_source": _SOURCE,
        "mode": mode,
    }
    sink = out / mode
    runner = RouteAlertsRunner(cfg, robot_id="shelfbot", sink_dir=str(sink))
    ev, robot_ev, tasks, dets, fed, passability = [], [], [], [], [], {}
    spill_times, obstruction_times = [], []
    for fr in frames:
        r = runner.observe([fr], now=fr.capture_time)
        if not r.ran:
            continue
        for e in r.events:
            ev.append(e.type)
            if e.type.startswith("spill."):
                spill_times.append(round(fr.capture_time, 3))
            if e.type.startswith("obstruction."):
                obstruction_times.append(round(fr.capture_time, 3))
        robot_ev += [e.type for e in r.robot_events]
        tasks += r.tasks
        if r.detections:
            dets = r.detections
        fed = runner.latest_detections()  # what nav actually receives this tick
        if r.passability:
            passability = r.passability
    runner.close()
    summary = {
        "mode": mode,
        "frames": len(frames),
        "event_types": sorted(set(ev)),
        "robot_event_types": sorted(set(robot_ev)),
        "first_spill_time_s": spill_times[0] if spill_times else None,
        "first_obstruction_time_s": obstruction_times[0] if obstruction_times else None,
        "blocked_aisles": passability.get("blocked") or [],
        "detections": [{"position": [round(d.position[0], 3), round(d.position[1], 3)],
                        "radius": round(d.radius, 3), "source": d.source} for d in dets],
        "fed_to_nav": [{"position": [round(d.position[0], 3), round(d.position[1], 3)],
                        "radius": round(d.radius, 3), "source": d.source} for d in fed],
        "staff_tasks": [{"type": str(t.type), "title": t.title} for t in tasks],
        "disabled_reason": runner.disabled_reason,
    }
    return summary, dets, fed


def _react(fed, out):
    """Feed the act-mode detections into the EXISTING dynobs layer; prove raised cost."""
    layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=1))
    layer.update(list(fed), 0.0)
    pts = layer.emitted_points()
    react = {"fed_detections": len(fed), "dynobs_emitted_points": len(pts),
             "raises_cost": bool(pts)}
    (out / "react.json").write_text(json.dumps(react, indent=2) + "\n")
    return react


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path,
                    help="a saved mission run dir (…/mission/<run>) with mission-frames/")
    ap.add_argument("--out", type=Path,
                    default=Path(os.path.expanduser(
                        "~/firstmate/data/sim-route-alerts-v2/recorded")),
                    help="artifact dir (outside git)")
    ap.add_argument("--window", type=int, default=24, help="recorded frames to replay")
    ap.add_argument("--stride", type=int, default=1, help="sub-sample recorded frames")
    ap.add_argument("--static-bg", action="store_true",
                    help="repeat one real dwell frame (a fixed viewpoint) over the run's "
                         "real timestamps; soundest for the fixed-camera spill tracker")
    ap.add_argument("--spill-from", type=int, default=4,
                    help="frame index the seeded spill appears at (after warmup)")
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    frames, calib = _build_frames(
        args.run, window=args.window, stride=args.stride, static_bg=args.static_bg,
        box_px=(410, 430, 560, 585), spill_center=(720, 640), spill_from=args.spill_from)

    findings, f_dets, f_fed = _drive(frames, calib, mode="findings", out=out)
    act, a_dets, a_fed = _drive(frames, calib, mode="act", out=out)
    react = _react(a_fed, out)

    summary = {
        "run": str(args.run), "run_id": args.run.name,
        "static_bg": args.static_bg, "window": len(frames),
        "sim_time_span_s": [round(frames[0].capture_time, 3),
                            round(frames[-1].capture_time, 3)],
        "findings": findings, "act": act, "react": react,
    }
    checks = {
        # detection on recorded frames with the run's real timestamps
        "spill_detected": findings["first_spill_time_s"] is not None,
        "obstruction_detected": findings["first_obstruction_time_s"] is not None,
        "robot_events_mirrored": bool(findings["robot_event_types"]) and all(
            t.startswith("robot.route_alert.") for t in findings["robot_event_types"]),
        "staff_tasks_raised": bool(findings["staff_tasks"]),
        "timestamps_are_run_sim_time": (
            findings["first_obstruction_time_s"] is not None
            and frames[0].capture_time <= findings["first_obstruction_time_s"]
            <= frames[-1].capture_time),
        # the mode split: findings feeds nav NOTHING; act feeds the obstruction + raises cost
        "findings_feeds_nav_nothing": findings["fed_to_nav"] == [],
        "act_feeds_obstruction": bool(act["fed_to_nav"]) and all(
            d["source"] == _SOURCE for d in act["fed_to_nav"]),
        "act_raises_dynobs_cost": react["raises_cost"],
        "disabled_quiet": findings["disabled_reason"] is None
            and act["disabled_reason"] is None,
    }
    summary["checks"] = checks
    (out / "recorded.json").write_text(json.dumps(summary, indent=2) + "\n")

    for name, val in checks.items():
        print(f"  [{'PASS' if val else 'FAIL'}] {name}")
    ok = all(checks.values())
    print(f"[{'PASS' if ok else 'FAIL'}] route-alerts recorded-frames evidence over "
          f"{len(frames)} real frames of run {args.run.name} "
          f"(sim t {summary['sim_time_span_s'][0]}..{summary['sim_time_span_s'][1]} s)")
    print(f"artifacts: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
