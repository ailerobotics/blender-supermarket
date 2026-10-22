"""Orchestrate the demo: walk the choreography timeline, gather per-frame scene READS,
accumulate the measured decision reasons, write the per-frame decision log JSON and the
scene frames. The read SOURCE is pluggable:

* ``synthetic_reads`` -- derive the reads from the scripted poses (CPU preview/fallback,
  no Isaac); the schematic renderer draws the scene.
* the Isaac path (:mod:`twin.demo.scene_reader`) -- read the live USD prim world poses,
  so every panel value is measured from the simulated scene.

Both feed the identical decision-log + panel + composite pipeline, so the offline half is
validated on CPU before any GPU time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import choreography as C
from . import decision_log as DL

# Static obstacle points (x, y, radius) used for the forward-clearance read in the
# synthetic path -- the backstock support and the shelf endcap. The Isaac path reads the
# equivalent prim AABB centres from the live stage instead.
SYNTHETIC_OBSTACLES = [(-5.73, 17.1, 0.45), (-3.05, 5.0, 0.6), (-7.0, 15.0, 0.3)]


def synthetic_reads(frame, prev=None):
    """Reads derived from the scripted poses (preview/fallback; NOT the deliverable)."""
    bx, by, byaw = frame['base']
    cxx, cyy, cyaw = frame['cart']
    kx, ky, kz, kyaw = frame['carton']
    hitch = (bx - C.HITCH_BEHIND * math.cos(byaw), by - C.HITCH_BEHIND * math.sin(byaw))
    dock = (cxx + C.DOCK_AHEAD * math.cos(cyaw), cyy + C.DOCK_AHEAD * math.sin(cyaw))
    # Commanded base speed -> braking lookahead (same shape as the mission gate).
    brake = 0.3
    if prev is not None:
        dist = math.hypot(bx - prev['base'][0], by - prev['base'][1])
        dt = max(1e-3, frame['t'] - prev['t'])
        speed = dist / dt
        brake = max(0.05, min(0.6, speed * 0.3 + speed * speed / (2 * 0.4)))
    return {
        'base_xy': (bx, by), 'base_yaw': byaw, 'cart_yaw': cyaw,
        'hitch_xy': hitch, 'dock_xy': dock,
        'carton_xyz': (kx, ky, kz), 'carton_face_xyz': (C.P_SUPPORT[0], C.P_SUPPORT[1] - 0.2, C.P_SUPPORT[2]),
        # With IK the gripper tracks the carton, so the pad sits ~on it (few-mm contact gap).
        'pad_xyz': (kx + 0.003, ky, kz), 'carton_top_z': kz + C.CARTON_HALF_H,
        'cage_top_z': 1.65,
        'cage_target_xyz': (cxx + math.cos(cyaw) * C.CAGE_LOAD_BODY_XY[0],
                            cyy + math.sin(cyaw) * C.CAGE_LOAD_BODY_XY[0], C.CAGE_SHELF_Z + C.CARTON_HALF_H),
        'shelf_target_xyz': (C.P_SHELF[0], C.P_SHELF[1], C.P_SHELF[2]),
        'obstacles': SYNTHETIC_OBSTACLES, 'brake_m': brake,
        'articulation_rad': (cyaw - byaw + math.pi) % (2 * math.pi) - math.pi,
    }


def run(frames, *, out_dir, reads_fn, render_fn, pre_fn=None, scene_size=(960, 720),
        save_scene=True):
    """Walk the timeline; write scene frames + ``decision_log.json``. Returns the log dict.

    ``pre_fn(frame)`` (optional) runs first each frame -- the Isaac path uses it to apply
    the scripted poses, step+render and capture the viewport. ``reads_fn(frame, prev) ->
    reads`` then reads the resulting scene; ``render_fn(frame, scene_path) -> None`` writes
    the scene PNG (saving the just-captured frame, or drawing the synthetic schematic).
    """
    out_dir = Path(out_dir)
    scene_dir = out_dir / 'scene-frames'
    scene_dir.mkdir(parents=True, exist_ok=True)
    step_reasons = {}
    prev = None
    entries = []
    for frame in frames:
        if pre_fn is not None:
            pre_fn(frame)
        reads = reads_fn(frame, prev)
        key = f"{frame['stage']}:{frame['step']}"
        step_reasons[key] = DL.compute_reason(frame['stage'], frame['step'], reads)
        chips = DL.compute_chips(reads, frame)
        state = 'complete' if frame is frames[-1] else 'running'
        entry = DL.build_entry(frame, step_reasons, chips, mission_state=state)
        scene_file = f"{frame['frame']:06d}.png"
        if save_scene and render_fn is not None:
            render_fn(frame, scene_dir / scene_file)
        entries.append({'scene_file': scene_file, 'panel': entry, 'reads': dict(reads)})
        prev = frame
    log = {'meta': {'frames': len(frames)}, 'frames': entries}
    (out_dir / 'decision_log.json').write_text(json.dumps(log) + '\n')
    return log


def run_synthetic(out_dir, *, fps=10):
    """CPU preview: schematic scene + full decision log + composited mp4."""
    from . import schematic
    frames, meta = C.build_timeline(fps=fps)
    run(frames, out_dir=out_dir, reads_fn=synthetic_reads,
        render_fn=lambda f, p: schematic.render_scene(f).save(p))
    return meta


def recompute(src_decision_log, out_dir, *, fps=15):
    """Rebuild the per-frame decision log from an EXISTING capture's logged prim reads with
    the current measure logic (no GPU, no re-capture). The scene frames are reused in place.
    Returns the new log dict; the caller composites against the source capture's scene-frames.

    Two reason refinements live here that need a whole-segment view (not per-frame):
    the grasp contact gap is reported at the ATTACH frame (the minimum carton-surface gap
    over the grasp step), frozen for the step.
    """
    from . import measures as M
    src = json.loads(Path(src_decision_log).read_text())['frames']
    frames, _ = C.build_timeline(fps=fps)
    if len(frames) != len(src):
        raise ValueError(f'timeline/log length mismatch {len(frames)} vs {len(src)}')
    reads = [e['reads'] for e in src]
    # Re-inject the static forward-clearance obstacles (the capture logged reads without the
    # obstacle list): the backstock support + the shelf endcap, the same set scene_reader used.
    obstacles = [(-5.73, 17.1, 0.34), (C.P_SHELF[0], C.P_SHELF[1] - 0.2, 0.6)]
    for r in reads:
        r.setdefault('obstacles', obstacles)
    # Pre-pass: minimum pad->carton-surface gap over the grasp step (collect:2) == the attach gap.
    grasp_gaps = []
    for fr, r in zip(frames, reads):
        if fr['stage'] == 'collect' and fr['step'] == 2:
            kx, ky, kz, kyaw = fr['carton']
            grasp_gaps.append(M.box_surface_gap(r['pad_xyz'], (kx, ky, kz), kyaw,
                                                (0.6, 0.4, 0.16)))
    grasp_mm = min(grasp_gaps) * 1000.0 if grasp_gaps else 0.0
    grasp_reason = ('contact gap <1 mm at grasp · sealed' if grasp_mm < 1.0
                    else f'contact gap {grasp_mm:.0f} mm at grasp')
    # Pre-pass: the cage placement (collect:5) reports where the carton SETTLES in the cage
    # once the gripper RELEASES it (the collect:6 verify-placed frames), not the transient
    # offset while the kinematic IK pad is still pressed against the free carton body (which
    # physics shoves aside during the hold). This is the honest placed-position read.
    place_reason = None
    for fr, r in zip(frames, reads):
        if fr['stage'] == 'collect' and fr['step'] == 6:
            err = M.placement_error(r['carton_xyz'], r['cage_target_xyz'])['error_mm']
            place_reason = f'placed Δ {err:.0f} mm from target'
            break
    # Same for the shelf place (transfer:1): report the carton's SETTLED placement once the
    # gripper releases it (the verify stage reads it stationary on the shelf), not the
    # gripper-hold transient while the IK pad is still pressed against the free carton body.
    shelf_place_reason = None
    for fr, r in zip(frames, reads):
        if fr['stage'] == 'verify':
            err = M.placement_error(r['carton_xyz'], r['shelf_target_xyz'])['error_mm']
            shelf_place_reason = f'placed Δ {err:.0f} mm from target'
            break
    step_reasons = {}
    entries = []
    for i, (fr, r) in enumerate(zip(frames, reads)):
        key = f"{fr['stage']}:{fr['step']}"
        if key == 'collect:2':
            step_reasons[key] = grasp_reason
        elif key == 'collect:5' and place_reason is not None:
            step_reasons[key] = place_reason
        elif key == 'transfer:1' and shelf_place_reason is not None:
            step_reasons[key] = shelf_place_reason
        else:
            step_reasons[key] = DL.compute_reason(fr['stage'], fr['step'], r)
        chips = DL.compute_chips(r, fr)
        state = 'complete' if i == len(frames) - 1 else 'running'
        entry = DL.build_entry(fr, step_reasons, chips, mission_state=state)
        entries.append({'scene_file': src[i]['scene_file'], 'panel': entry, 'reads': r})
    log = {'meta': {'frames': len(frames), 'recomputed_from': str(src_decision_log)}, 'frames': entries}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'decision_log.json').write_text(json.dumps(log) + '\n')
    return log
