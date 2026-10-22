#!/usr/bin/env python3
"""Route-alerts LIVE in-Isaac smoke: detect a seeded hazard on a real rendered frame.

Two phases (so the GPU hold is minimal and the detection is the IDENTICAL production
runner):

  GPU phase (``scripts/run_restock_render_smoke.py``, run under the blocking GPU lock):
  boot Isaac and render the calibrated bench camera's view of the **static test cube on
  a floor** -- a seeded obstruction in a fixed, calibrated RGB-D camera's stationary
  view -- capturing a sequence of real RTX RGB + depth frames with the camera intrinsics
  + pose. No new Isaac code; reuses the proven render-smoke bench verbatim.

  CPU phase (this script, after the lock is released): reconstruct that real
  Isaac-rendered (rgb, depth) sequence and drive the **identical** ``RouteAlertsRunner``
  (obstruction, depth-protrusion path) over it, with the floor homography computed from
  the captured camera intrinsics + pose, so the seeded cube is confirmed as an
  ``obstruction.detected`` finding, mirrored to ``robot.route_alert.*``, raised as a
  staff task, and turned into a ``twin.navigation.dynamic.Detection`` for the dynobs
  layer -- exactly as on the route. Self-asserting (exit 1 on failure).

This proves live in-Isaac detection through the production runner. It is NOT the
"on the route while driving" run (the mission bundle has no mobile floor-facing RGB-D
camera -- see the option-A follow-up in data/sim-route-alerts-v1/report.md).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.route_alerts import RouteAlertsRunner, RouteFrame  # noqa: E402

_TUNE = {"persist_seconds": 0.4, "warmup_frames": 2, "clear_seconds": 0.8,
         "min_object_height_m": 0.06}


def image_to_floor_homography(K, parent_from_optical, z_floor):
    """3x3 image-pixel -> world-floor(x,y) homography for plane z=z_floor.

    ``K`` is the 3x3 intrinsics, ``parent_from_optical`` the 4x4 optical->world
    transform (OpenCV optical frame: +z forward, +x right, +y down). Back-projects a
    grid of pixels onto the floor plane and fits the homography.
    """
    K = np.asarray(K, float).reshape(3, 3)
    T = np.asarray(parent_from_optical, float).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    Kinv = np.linalg.inv(K)
    # a spread of image pixels (avoid the exact principal row where dir_z ~ 0)
    us = np.linspace(40, K[0, 2] * 2 - 40, 5)
    vs = np.linspace(K[1, 2] + 20, K[1, 2] * 2 - 20, 5)
    src, dst = [], []
    for u in us:
        for v in vs:
            d_opt = Kinv @ np.array([u, v, 1.0])
            d_world = R @ d_opt
            if abs(d_world[2]) < 1e-6:
                continue
            s = (z_floor - t[2]) / d_world[2]
            if s <= 0:
                continue
            p = t + s * d_world
            src.append([u, v])
            dst.append([p[0], p[1]])
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4:
        return None
    import cv2
    H, _ = cv2.findHomography(src, dst, method=0)
    return H


def _fit_floor_z(depth, K, parent_from_optical, pct=20.0):
    """Estimate the floor-plane world z as a low percentile of back-projected depth."""
    K = np.asarray(K, float).reshape(3, 3)
    T = np.asarray(parent_from_optical, float).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    Kinv = np.linalg.inv(K)
    h, w = depth.shape
    vs, us = np.mgrid[0:h:16, 0:w:16]
    us, vs = us.ravel(), vs.ravel()
    d = depth[vs, us]
    ok = np.isfinite(d) & (d > 0)
    us, vs, d = us[ok], vs[ok], d[ok]
    rays = Kinv @ np.vstack([us, vs, np.ones_like(us)])  # 3xN optical dirs (z=1)
    pts_opt = rays * d  # scale by z-depth (distance_to_image_plane)
    pts_world = (R @ pts_opt) + t[:, None]
    return float(np.percentile(pts_world[2], pct))


def load_render(render_dir: Path):
    """Reconstruct the (rgb, depth, time) sequence + camera calibration from a render."""
    cam = json.loads((render_dir / "camera.json").read_text())
    K = cam["intrinsics"]
    pfo = cam["spec"]["parent_from_optical"]
    frames_dir = render_dir / "frames"
    depth_dir = render_dir / "depth-frames"
    obs = json.loads((render_dir / "observed-frames.json").read_text())
    import cv2
    seq = []
    pngs = sorted(frames_dir.glob("*.png"))
    for i, png in enumerate(pngs):
        npz = depth_dir / f"{i:06d}.npz"
        if not npz.is_file():
            continue
        rgb = cv2.imread(str(png))  # BGR uint8 (store-vision convention)
        depth = np.asarray(np.load(npz)["depth"], dtype=np.float32)
        t = float(obs[i]["rendering_time"]) if i < len(obs) else float(i) / 60.0
        seq.append((rgb, depth, t))
    return seq, K, pfo


def run(render_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    seq, K, pfo = load_render(render_dir)
    if not seq:
        raise SystemExit("no (rgb, depth) frames in the render dir")
    z_floor = _fit_floor_z(seq[0][1], K, pfo)
    H = image_to_floor_homography(K, pfo, z_floor)
    h, w = seq[0][1].shape
    cfg = {
        "obstruction": {"enabled": True, "gpu": False, "tuning": _TUNE},
        "spill": {"enabled": True, "gpu": False, "tuning": _TUNE},
        "floors": {"bench": {"image_to_floor": H.tolist() if H is not None else None,
                             "zone_id": "bench"}},
        "staff_tasks": True,
        "obstruction_source": "route_obstruction",
    }
    runner = RouteAlertsRunner(cfg, robot_id="shelfbot", sink_dir=str(out))
    ev, robot_ev, tasks, dets = [], [], [], []
    for i, (rgb, depth, t) in enumerate(seq):
        rf = RouteFrame(camera_id="bench", seq=i, rgb=rgb, depth=depth,
                        capture_time=t, zone_id="bench", agents=[])  # no people in the bench scene
        r = runner.observe([rf], now=t)
        ev += [e.type for e in r.events]
        robot_ev += [e.type for e in r.robot_events]
        tasks += r.tasks
        if r.detections:
            dets = r.detections
    runner.close()
    summary = {
        "frames": len(seq),
        "z_floor_m": round(z_floor, 3),
        "event_types": sorted(set(ev)),
        "robot_event_types": sorted(set(robot_ev)),
        "detections": [{"position": [round(d.position[0], 3), round(d.position[1], 3)],
                        "radius": round(d.radius, 3), "source": d.source} for d in dets],
        "staff_tasks": [{"type": str(t.type), "title": t.title} for t in tasks],
        "disabled_reason": runner.disabled_reason,
        "checks": {
            "obstruction_detected": any(t.startswith("obstruction.") for t in ev),
            "robot_events_mirrored": bool(robot_ev) and all(
                t.startswith("robot.route_alert.") for t in robot_ev),
            "dynobs_detection": bool(dets),
            "staff_task_raised": bool(tasks),
        },
    }
    (out / "smoke-detect.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, val in summary["checks"].items():
        print(f"  [{'PASS' if val else 'FAIL'}] {name}")
    ok = runner.disabled_reason is None and summary["checks"]["obstruction_detected"] \
        and summary["checks"]["dynobs_detection"]
    print(f"[{'PASS' if ok else 'FAIL'}] route-alerts live smoke over {len(seq)} real Isaac frames")
    print(f"artifacts: {out}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", required=True, type=Path,
                    help="run_restock_render_smoke.py output dir (frames/, depth-frames/, camera.json)")
    ap.add_argument("--out", required=True, type=Path, help="artifact dir for the detection result")
    args = ap.parse_args(argv)
    return run(args.render, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
