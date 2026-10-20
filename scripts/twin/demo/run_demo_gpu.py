#!/usr/bin/env python3
"""Guarded GPU capture for the choreographed restock demo (RUNBOOK/demo lane).

Under the shared GPU flock (``twin.restock.guard.gpu_slot``, VRAM guard unchanged, never
``--force``), boots a lean SimulationApp, composes the mission's Isaac scene, walks the
choreography timeline capturing a scene frame per tick and writing the per-frame decision
log (values read back from the live stage). Releases the lock, then (``--encode``)
composites the panel and encodes ``walkthrough.mp4`` offline.

Example:
  python3 scripts/twin/demo/run_demo_gpu.py \
    --config /home/.../mission-runtime-cc.json --scenario /home/.../mission-scenario.json \
    --out /home/.../mission/demo-001 --fps 10 --encode
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# scripts/ on the path so `twin.*` imports resolve when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.demo import choreography as C  # noqa: E402

FFMPEG_DEFAULT = ('/home/manishsaini/miniforge3/envs/env_isaaclab/lib/python3.11/'
                  'site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')


def _app_config(width, height, low_vram=True):
    config = {'headless': True, 'width': width, 'height': height}
    settings = []
    if low_vram:
        config.update({'renderer': 'RayTracedLighting', 'anti_aliasing': 0, 'multi_gpu': False})
        settings = [('/rtx/indirectDiffuse/enabled', False),
                    ('/rtx/reflections/enabled', False),
                    ('/rtx/ambientOcclusion/enabled', False)]
    return config, settings


def _smoke_subsample(frames, n):
    """Keep ~n frames spread across the timeline (at least one per stage) for a fast
    framing/boot check."""
    if n <= 0 or n >= len(frames):
        return frames
    idx = sorted(set(round(i * (len(frames) - 1) / (n - 1)) for i in range(n)))
    # Ensure each stage is represented.
    seen = {f['stage'] for f in (frames[i] for i in idx)}
    for i, f in enumerate(frames):
        if f['stage'] not in seen:
            idx.append(i)
            seen.add(f['stage'])
    return [frames[i] for i in sorted(set(idx))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='mission runtime config JSON')
    ap.add_argument('--scenario', help='scenario JSON (default: config["scenario"])')
    ap.add_argument('--out', required=True, help='output run dir (new)')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--panel-width', type=int, default=480)
    ap.add_argument('--out-width', type=int, default=1920)
    ap.add_argument('--out-height', type=int, default=1080)
    ap.add_argument('--camera', default='wide', choices=['wide', 'follow'])
    ap.add_argument('--heavy-mb', type=int, default=2500, choices=[1500, 2500])
    ap.add_argument('--ffmpeg', default=FFMPEG_DEFAULT)
    ap.add_argument('--smoke', type=int, default=0, help='capture ~N frames only (framing check)')
    ap.add_argument('--frame-start', type=int, default=0, help='first timeline frame to capture')
    ap.add_argument('--frame-end', type=int, default=0, help='last timeline frame (exclusive; 0=all)')
    ap.add_argument('--cam-height', type=float, default=2.1,
                    help='chase-camera height above base_link (m); keep below the store ceiling')
    ap.add_argument('--encode', action='store_true', help='composite + encode after capture')
    ap.add_argument('--heavy-vram', action='store_true',
                    help='accept the whispr heavy consumer (sets heavy-mb 2500)')
    args = ap.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    scenario_path = args.scenario or config.get('scenario')
    scenario = json.loads(Path(scenario_path).read_text())
    out = Path(args.out)
    if out.exists():
        print(f'refusing to reuse existing out dir {out}', file=sys.stderr)
        return 2

    frames, meta = C.build_timeline(fps=args.fps)
    if args.frame_end:
        frames = frames[args.frame_start:args.frame_end]
    elif args.frame_start:
        frames = frames[args.frame_start:]
    if args.smoke:
        frames = _smoke_subsample(frames, args.smoke)
    print(f'DEMO timeline frames={len(frames)} duration~{meta["duration_s"]}s fps={args.fps}', flush=True)

    from twin.restock.guard import gpu_slot
    app = None
    try:
        with gpu_slot(heavy_threshold_mb=args.heavy_mb):
            from isaacsim import SimulationApp
            cfg, rtx = _app_config(args.width, args.height, low_vram=True)
            app = SimulationApp(cfg)
            if rtx:
                try:
                    import carb
                    s = carb.settings.get_settings()
                    for k, v in rtx:
                        s.set(k, v)
                except Exception as exc:
                    print(f'DEMO rtx carb settings skipped: {exc!r}', flush=True)
            from twin.demo.scene_reader import DemoScene
            from twin.demo import driver
            scene = DemoScene(app=app, config=config, scenario=scenario, out_dir=out,
                              resolution=(args.width, args.height), camera=(args.camera,),
                              cam_height=args.cam_height)
            driver.run(frames, out_dir=out, reads_fn=scene.reads, render_fn=scene.capture_to_file,
                       pre_fn=scene.apply_and_step)
            print('DEMO capture complete', flush=True)
        # Lock released here. Encode BEFORE app.close() -- Isaac's close() can hard-exit the
        # process and skip anything after it. The encode is CPU-only and holds no GPU lock.
        if args.encode:
            from twin.demo import compositor
            res = compositor.build_video(out / 'scene-frames', out / 'decision_log.json', out,
                                         ffmpeg=args.ffmpeg, out_size=(args.out_width, args.out_height),
                                         panel_w=args.panel_width, fps=args.fps)
            print(f'DEMO encoded -> {res["file"]} ({res["frames"]} frames)', flush=True)
    finally:
        if app is not None:
            try:
                app.close()
            except Exception:
                pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
