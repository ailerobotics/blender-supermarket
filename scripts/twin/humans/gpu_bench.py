"""Bounded native RTX bench: scripted people crossing the robot, with the
ground-truth human-safety detection + reaction driven on the real rendered frames.

This is the standalone guarded-GPU evidence for the opt-in humans layer, dispatched
by ``scripts/run_restock_render_smoke.py --humans-only`` (same lock/VRAM guard and
task-local NVIDIA runtime as the other bounded diagnostics). It is NOT a mission and
runs no planner/navigation/arm: it reuses the static store optical bench
(``twin.restock.view_scene.build``) for the full store + a real candidate camera,
authors the scripted shoppers/staff (``twin.humans.isaac_people``) on deterministic
paths that cross the robot and enter the arm envelope, and every frame teleports the
people, renders the camera, and runs the EXPLICIT ground-truth detection provider +
the real reaction mechanism (``twin.humans.detection`` / ``reaction``). The output is
labelled GROUND-TRUTH evidence (known poses projected into the real camera), not a
neural-detector measurement -- the detector path needs the fleet package + model on
the same single GPU and is left to the mission-wiring follow-up.

The pure crossing synthesis (:func:`synthesize_crossing`) and the trace/summary are
CPU-tested in ``tests/test_gpu_bench.py``; only the USD/Isaac authoring + render loop
here require a GPU and run inside the worker.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Sequence, Tuple

from .config import HumansConfig, PersonSpec
from .detection import GroundTruthSafety
from .paths import PINGPONG
from .perception import SafetyParams
from .reaction import ReactionController

Pose2D = Tuple[float, float, float]


def synthesize_crossing(camera_xyyaw: Pose2D, base_xyyaw: Pose2D, *,
                        role: str = 'shopper', speed_mps: float = 0.9,
                        start_range_m: float = 6.5, closest_range_m: float = 0.9,
                        radius_m: float = 0.3) -> PersonSpec:
    """A deterministic person who walks from off-camera (beyond the detection
    range) in toward the robot (inside the stop ring) and back, down the camera axis.

    ``camera_xyyaw`` is the candidate camera's world pose (so the person is framed);
    ``base_xyyaw`` the robot base (so approaching the camera also approaches the
    base). The path is a ping-pong along the camera's forward axis: far point at
    ``start_range_m`` from the base -- set BEYOND ``detection.gt_visible_range_m`` so
    the person is unseen there (CLEAR, the base at full speed) -- and near point at
    ``closest_range_m`` from the base (inside reach+stop => STOP). A single run then
    shows CLEAR -> SLOW -> STOP on the approach and STOP -> SLOW -> CLEAR (resume) on
    the return. (With the fleet ``slow_on_any_person`` floor, any visible person
    forces >= SLOW, so resume to full speed requires the person to leave view -- the
    far point does exactly that.)
    """
    cx, cy, cyaw = camera_xyyaw
    bx, by, _ = base_xyyaw
    fx, fy = math.cos(cyaw), math.sin(cyaw)        # camera forward (its viewing axis)
    # Walk along the camera axis in front of the camera. Place the near/far points
    # by distance FROM THE BASE along that axis, measured from the base outwards.
    # Direction from base toward the camera-forward region: use the camera forward.
    far = (bx + fx * start_range_m, by + fy * start_range_m)
    near = (bx + fx * closest_range_m, by + fy * closest_range_m)
    return PersonSpec(person_id='bench_shopper', role=role,
                      waypoints=(far, near), speed_mps=speed_mps, mode=PINGPONG,
                      dwell_s=1.0, start_offset_s=0.0, radius_m=radius_m)


def _camera_world_xyyaw(stage, prim_path: str) -> Pose2D:
    """World (x, y, yaw) of a USD camera prim; yaw is its viewing axis (-Z) in XY."""
    from pxr import Usd, UsdGeom, Gf
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f'camera prim not found at {prim_path}')
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    # A USD camera looks down its local -Z; the forward world vector is -Z column.
    fwd = Gf.Vec3d(-m[2][0], -m[2][1], -m[2][2])
    yaw = math.atan2(fwd[1], fwd[0])
    return (float(t[0]), float(t[1]), float(yaw))


def _people_from_config(humans: HumansConfig, camera_xyyaw: Pose2D,
                        base_xyyaw: Pose2D) -> Tuple[PersonSpec, ...]:
    if humans.people:
        return humans.people
    return (synthesize_crossing(camera_xyyaw, base_xyyaw),)


def summarize(rows, params: SafetyParams) -> dict:
    """Headline verdict for the bench: were SLOW, STOP, arm pause and resume seen."""
    from .evidence import summarize as _summarize
    return _summarize(rows, params)


def run(app, config, output, *, ffmpeg, sensing_stride: int = 8,
        record_stride: int = 20, max_steps: int = 900, target_frames: int = 16,
        dt: float = 1 / 60):  # pragma: no cover - requires GPU/Isaac/USD
    """Author the store + crossing people, render, and drive GT detection+reaction.

    Mirrors the validated capture/identity/depth checks of the render smoke, but
    teleports the scripted people each step and evaluates the ground-truth safety
    machine + reaction on the rendered timeline. Returns a result dict; the parent
    (`run_restock_render_smoke.py`) gates `status` and the slow/stop/resume verdict.
    """
    import numpy as np
    import omni.usd
    from pxr import UsdGeom, UsdLux
    from isaacsim.core.api import World

    from twin.restock.view_scene import build as build_view
    from twin.restock.recording import Recorder
    from .isaac_people import PeopleStage

    output = Path(output)
    humans_block = config.get('humans') or {}
    humans = HumansConfig.from_config({'humans': humans_block})
    if not humans.enabled:
        raise ValueError('humans-only bench requires humans.enabled in the runtime config')
    if humans.detection.source != 'ground_truth':
        raise ValueError('the GPU bench runs the EXPLICIT ground_truth provider; '
                         'the neural detector path is the mission-wiring follow-up')
    view = config.get('humans_bench_view', 'shelf')

    world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.)
    stage = omni.usd.get_context().get_stage()
    camera, cam_design, _codes = build_view(stage, config, output, view=view)

    scenario = json.loads(Path(config['scenario']).read_text())
    staging_base_pose = tuple(float(v) for v in scenario['initial_staging']['robot_base_pose'])
    # The candidate camera prim lives under the map/cart; read its world pose so the
    # people are framed by the real lens and the GT cone matches the render.
    cam_prim_path = getattr(camera, 'prim_path', None)
    try:
        cam_pose = _camera_world_xyyaw(stage, cam_prim_path) if cam_prim_path else staging_base_pose
    except Exception:
        cam_pose = staging_base_pose                 # fall back to the staging base pose/yaw

    # The candidate camera IS the robot's eye here (eye-in-hand / on-robot), so the
    # safety distance is person-to-CAMERA, not person-to-staging-base: the fixed
    # store candidate camera sits metres from where the mobile base happens to park,
    # and a person approaching the camera is a person approaching the robot. Using the
    # staging base pose instead leaves the person forever outside the camera's range
    # (always CLEAR). The staging base pose is kept only for the evidence record.
    base_pose = cam_pose

    people = _people_from_config(humans, cam_pose, base_pose)
    humans = humans.with_people(people)
    stage_people = PeopleStage(humans)
    stage_people.author(stage, physical=False)

    radii = {p.person_id: p.radius_m for p in people}
    heights = {p.person_id: p.height_m for p in people}

    def people_positions(now: float):
        out = []
        for p in people:
            x, y, _yaw = stage_people.world_pose(p.person_id, now)
            out.append((p.person_id, (x, y, heights[p.person_id] * 0.5), radii[p.person_id]))
        return out

    base_ref = {'pose': base_pose}
    params = SafetyParams()
    provider = GroundTruthSafety(
        config=humans.detection, people=people_positions,
        base_pose=lambda: base_ref['pose'],
        camera_poses=lambda: {f: cam_pose for f in humans.detection.cameras},
        params=params)
    reaction = ReactionController(humans.reaction)

    world.reset()
    for _ in range(60):
        world.step(render=True)

    recorder = Recorder(output / 'frames', fps=30)
    (output / 'depth-frames').mkdir()
    rows = []
    frame_records = []
    for i in range(max_steps):
        now = i * dt
        stage_people.update(now)
        # Render economy: detection is pose-based (ground truth, no pixels), so only
        # the frame-capture steps pay for an RTX render; the rest advance cheaply.
        do_capture = (i % record_stride == 0)
        world.step(render=do_capture)
        if i % sensing_stride == 0:
            state = provider.state(now)
            out = reaction.update(state, now)
            rows.append({
                't': round(now, 4),
                'zone': out.zone, 'base_scale': out.base_scale, 'arm_hold': out.arm_hold,
                'person_in_arm_envelope': out.person_in_arm_envelope,
                'nearest_distance_base_m': (None if not math.isfinite(state.nearest_distance_base)
                                            else round(state.nearest_distance_base, 3)),
                'people': {pid: [round(xyz[0], 3), round(xyz[1], 3)]
                           for pid, xyz, _r in people_positions(now)},
                'reason': out.reason})
            (output / 'reaction-trace.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in rows) + '\n')
        if do_capture:
            frame = camera.get_current_frame(clone=True)
            if recorder.capture(frame):
                depth = np.asarray(frame.get('distance_to_image_plane'), float)
                np.savez_compressed(
                    output / 'depth-frames' / f'{len(recorder.frames) - 1:06d}.npz',
                    depth=depth)
                # Annotate the frame with the most recent reaction state (the trace
                # above owns the authoritative tick; this is a read-only label).
                last = rows[-1] if rows else {}
                frame_records.append(dict(index=len(recorder.frames) - 1,
                                          simulation_time=float(frame['rendering_time']),
                                          zone=last.get('zone'),
                                          base_scale=last.get('base_scale'),
                                          arm_hold=last.get('arm_hold')))
        if len(recorder.frames) >= target_frames and rows and \
                {'SLOW', 'STOP'}.issubset({r['zone'] for r in rows}):
            # Enough frames AND both regimes already seen: keep going a little to
            # capture the return-to-CLEAR (resume), then stop.
            if any(r['zone'] == 'CLEAR' for r in rows[len(rows) // 2:]):
                break

    summary = summarize(rows, params)
    video = recorder.encode(output / 'humans-bench.mp4', ffmpeg=ffmpeg) \
        if len(recorder.frames) >= 2 else None
    (output / 'reaction-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'frame-reactions.json').write_text(json.dumps(frame_records, indent=2) + '\n')
    try:
        stage.GetRootLayer().Export(str(output / 'humans-bench.usda'))
    except Exception:
        pass

    complete = (summary['slow_seen'] and summary['stop_seen']
                and summary['arm_paused'] and summary['base_resumed']
                and len(recorder.frames) >= 2)
    return dict(status='passed' if complete else 'failed',
                evidence='native RTX ground-truth human-safety bench (NOT a neural '
                         'detector measurement, NOT a mission); scripted people, '
                         'ground-truth projection into the real candidate camera',
                view=view, provenance='ground_truth', frames=len(recorder.frames),
                video=video, trace_rows=len(rows), summary=summary,
                base_pose=list(base_pose), camera_pose=list(cam_pose),
                staging_base_pose=list(staging_base_pose),
                base_model='camera-as-robot-eye (safety distance is person-to-camera)',
                robot_motion=False, autonomous_mission=False, neural_detector=False)
