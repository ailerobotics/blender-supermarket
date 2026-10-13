"""Bounded native straight-towing validation; records VFR H.264, never a mission.

Couples the robot+cart at a portal-aligned dock, then commands a short forward crawl
through the sales/backroom portal. It proves forward towing (the train advances and
the cart follows through the physical hitch), bounded articulation (< 46 deg) and no
cart wheel lift, and records the motion as variable-frame-rate H.264 with the same
provenance as the native render smoke. No navigation planner is used: a single
constant forward velocity is commanded straight down the portal lane; steering,
sensing and manipulation are not exercised.
"""
import json
from pathlib import Path

import numpy as np

from twin.restock_camera.geometry import look_at
from .physics_setup import assemble
from .settle_metrics import settle_report
from .coupling_check import dock_observation, _row
from .recording import Recorder
from .native_cameras import CameraRig

WHEELS = [f'/Rollwagen/wheel_{i}' for i in range(4)]
WHEEL_REST_Z = 0.075            # wheel centre height when resting on z=0 floor
WHEEL_LIFT_TOL = 0.010          # 10 mm: above this a wheel has left the floor
MAX_ARTICULATION_RAD = 0.80     # ~46 deg coupled articulation limit
MIN_FORWARD_M = 0.30            # minimum demonstrated forward travel


def _wheel_heights(physx, paths):
    heights = {}
    for path in paths:
        value = physx.get_rigidbody_transformation(path)
        if not value['ret_val']:
            raise ValueError('cart wheel pose unavailable: ' + path)
        heights[path] = float(value['position'][2])
    return heights


def run(app, config, output, *, ffmpeg, settle_steps=150, tow_steps=540,
        crawl_speed=0.35, capture_every=2):
    output = Path(output)
    scenario = json.loads(Path(config['scenario']).read_text())
    assembly = assemble(app=app, config=config, scenario=scenario, output=output / 'assembly')
    from omni.physx import get_physx_interface
    from pxr import UsdGeom
    physx = get_physx_interface()

    # Fixed map camera overlooking the portal lane; housing solid is up and aside.
    UsdGeom.Xform.Define(assembly.stage, '/TowingCamera')
    eye, target = [0.6, 11.0, 2.6], [-3.875, 11.6, 0.5]
    mount = np.eye(4); mount[:3, 3] = eye
    intr = dict(width=640, height=360, fx=500., fy=500., cx=320., cy=180., near=.1, far=40.)
    spec = dict(id='tow', parent='map', parent_from_optical=look_at(eye, target).tolist(),
                intrinsics=intr,
                collision_solids=[dict(parent_from_solid=mount.tolist(), size_m=[.06, .06, .04])])
    rig = CameraRig(stage=assembly.stage, specs=[spec], parent_paths={'map': '/TowingCamera'})
    camera = rig.cameras['tow']
    recorder = Recorder(output / 'frames', fps=30)
    history = []
    try:
        # Warm up the render product before recording.
        for _ in range(30):
            assembly.hold(); assembly.world.step(render=True)

        # 1) Settle the uncoupled train at the portal-aligned dock.
        for i in range(settle_steps):
            assembly.hold(); assembly.world.step(render=True)
            history.append(_row(assembly, i + 1, 'pre_couple'))
        pre_settled, pre_settle = settle_report(history[-31:])
        if not pre_settled:
            raise RuntimeError('train did not settle before docking')

        # 2) Create the physical hitch from the fresh stopped aligned observation.
        dock = dock_observation(assembly)
        attach_state = assembly.mobile.attach(dock)
        if not attach_state['healthy']:
            raise RuntimeError('coupling not healthy immediately after attach')
        start_xy = np.array(history[-1]['robot']['pose_xyz_yaw'][:2])

        # 3) Tow straight forward at crawl speed down the portal lane.
        max_wheel_z, max_artic, captured = WHEEL_REST_Z, 0.0, 0
        for i in range(tow_steps):
            obs = dock_observation(assembly)
            assembly.command_gate.hold()
            targets = assembly.mobile.command(crawl_speed, 0.0, obs, transport_ready=True)
            assembly.world.step(render=True)
            post = dock_observation(assembly)
            cstate = assembly.mobile.coupling_state(post)
            artic = abs(post.robot.yaw - post.cart.yaw)
            heights = _wheel_heights(physx, WHEELS)
            peak_wheel = max(heights.values())
            max_wheel_z, max_artic = max(max_wheel_z, peak_wheel), max(max_artic, artic)
            history.append(_row(assembly, settle_steps + i + 1, 'towing',
                coupling=dict(healthy=cstate['healthy'], hitch_error_m=cstate['hitch_error_m'],
                    articulation_rad=float(post.robot.yaw - post.cart.yaw),
                    wheel_z=heights, commanded_wheel_rad_s=list(targets))))
            if not cstate['healthy']:
                raise RuntimeError(f'coupling lost during tow at step {i + 1}: '
                                   f"residual {cstate['hitch_error_m']:.4f} m")
            if artic >= MAX_ARTICULATION_RAD:
                raise RuntimeError(f'articulation {artic:.3f} rad exceeded 46 deg at step {i + 1}')
            if peak_wheel > WHEEL_REST_Z + WHEEL_LIFT_TOL:
                raise RuntimeError(f'cart wheel lift to {peak_wheel:.4f} m at step {i + 1}')
            if i % capture_every == 0:
                frame = camera.get_current_frame(clone=True)
                rgb = np.asarray(frame.get('rgb'))
                if rgb.ndim == 3 and rgb.shape[0] == intr['height'] and rgb.shape[1] == intr['width']:
                    if recorder.capture(frame):
                        captured += 1

        end_xy = np.array(history[-1]['robot']['pose_xyz_yaw'][:2])
        forward = float(np.linalg.norm(end_xy - start_xy))
        if forward < MIN_FORWARD_M:
            raise RuntimeError(f'insufficient forward towing: {forward:.3f} m < {MIN_FORWARD_M}')
        if captured < 10:
            raise RuntimeError(f'insufficient recorded tow frames: {captured}')
        video = recorder.encode(output / 'native-towing.mp4', ffmpeg=ffmpeg)
        return dict(status='passed', towed=True,
            forward_distance_m=forward, crawl_speed_m_s=crawl_speed,
            max_articulation_rad=float(max_artic), max_articulation_deg=float(np.degrees(max_artic)),
            max_cart_wheel_z_m=float(max_wheel_z), wheel_rest_z_m=WHEEL_REST_Z, no_wheel_lift=True,
            start_robot_xy=start_xy.tolist(), end_robot_xy=end_xy.tolist(),
            steps=len(history), captured_frames=captured, video=video,
            evidence='native coupled forward towing at constant crawl velocity down the portal lane; '
                     'cart followed through the physical hitch with bounded articulation and no wheel '
                     'lift; VFR H.264 recorded from a fixed map camera. No planner, steering or sensing.',
            autonomous_mission=False, planner_started=False, navigation_used=False)
    finally:
        assembly.hold()
        (output / 'towing-feedback.json').write_text(json.dumps(history, indent=2) + '\n')
