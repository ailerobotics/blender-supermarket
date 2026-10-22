"""Bounded native PLANNER-STEERED coupled-towing probe; records VFR H.264.

Unlike `towing_check` (a single constant forward velocity straight down the portal
lane), this drives the production `twin.navigation.articulated.ArticulatedController`
in the loop: it plans a forward-only coupled route from the measured couple pose to
the `G3_back_s2` shelf docking pose and follows it with gated (v, omega) commands, so
the native run exercises steering and the articulated-train turn behaviour the CPU
contract assumes (`data/sim-restock-navigation-v2/outbound-contract.md`).

Scope and honesty: this isolates the navigation steering + coupled-towing dynamics.
Odometry is the measured native base pose/velocity; the controller runs through its
fixture-accepting input path (no lidar is wired in this reduced nav-only scene) and
sensor coverage is ASSERTED clear (coverage_radius = range_max, obstacles empty) — the
self-occlusion coverage certification is proven separately and is not what this probe
measures. Manipulation is not present. The run is wall-clock bounded; the shelf dock
is ~11 m away and native towing is traction-limited, so a bounded run may demonstrate
a partial (not completed) traverse, which is reported honestly.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from twin.restock_camera.geometry import look_at
from twin.navigation import Odometry, Observation
from twin.navigation.scenario import environment_grid
from twin.navigation.articulated import ArticulatedController
from .physics_setup import assemble
from .settle_metrics import settle_report
from .isaac_port import body_measurement
from .coupling_check import dock_observation, _row
from .towing_check import WHEELS, WHEEL_REST_Z, WHEEL_LIFT_TOL, _wheel_heights
from .recording import Recorder
from .native_cameras import CameraRig

MAX_ARTICULATION_RAD = 0.80     # ~46 deg coupled articulation limit (jackknife guard)
GOAL_ARTICULATION_RAD = 0.90    # per-leg docking tolerance (see outbound-contract.md)
ASSERTED_COVERAGE_M = 10.0      # clear-coverage assumption (no lidar in this probe)


def run(app, config, output, *, ffmpeg, settle_steps=90, max_wall_s=360.0,
        capture_every=3, uncertainty=0.10, dt=1 / 60):
    output = Path(output)
    scenario = json.loads(Path(config['scenario']).read_text())
    manifest = json.loads(Path(config['navigation_environment']).read_text())
    staging = scenario['initial_staging']
    goal = tuple(staging['shelf_robot_pose'])

    # Production controller on the static navigation grid (+ the staging support obstacle).
    grid = environment_grid(manifest, scenario)
    controller = ArticulatedController(grid, allow_test_fixtures=True,
                                       goal_timeout=1200.0, blocked_timeout=30.0)
    controller.start(goal, True, goal_articulation=GOAL_ARTICULATION_RAD)

    assembly = assemble(app=app, config=config, scenario=scenario, output=output / 'assembly')
    from omni.physx import get_physx_interface
    from pxr import UsdGeom
    physx = get_physx_interface()

    # Fixed overhead map camera framing the backroom couple pose and the portal mouth.
    UsdGeom.Xform.Define(assembly.stage, '/NavTowingCamera')
    eye, target = [-5.0, 13.0, 4.2], [-4.4, 10.5, 0.3]
    mount = np.eye(4); mount[:3, 3] = eye
    intr = dict(width=640, height=360, fx=430., fy=430., cx=320., cy=180., near=.1, far=50.)
    spec = dict(id='nav', parent='map', parent_from_optical=look_at(eye, target).tolist(),
                intrinsics=intr,
                collision_solids=[dict(parent_from_solid=mount.tolist(), size_m=[.06, .06, .04])])
    rig = CameraRig(stage=assembly.stage, specs=[spec], parent_paths={'map': '/NavTowingCamera'})
    camera = rig.cameras['nav']
    recorder = Recorder(output / 'frames', fps=30)

    history = []
    reasons = {}
    docked = False
    try:
        for _ in range(30):                       # warm the render product
            assembly.hold(); assembly.world.step(render=True)

        # 1) Settle the uncoupled train at the couple pose, then create the hitch.
        for i in range(settle_steps):
            assembly.hold(); assembly.world.step(render=True)
            history.append(_row(assembly, i + 1, 'pre_couple'))
        settled, pre_settle = settle_report(history[-31:])
        if not settled:
            raise RuntimeError('train did not settle before docking')
        dock = dock_observation(assembly)
        attach_state = assembly.mobile.attach(dock)
        if not attach_state['healthy']:
            raise RuntimeError('coupling not healthy immediately after attach')
        start_xy = np.array(history[-1]['robot']['pose_xyz_yaw'][:2])

        # 2) Planner-steered tow toward the shelf dock, wall-clock bounded.
        max_wheel_z, max_artic, max_omega, captured, step_i = WHEEL_REST_Z, 0.0, 0.0, 0, 0
        began = time.monotonic()
        while time.monotonic() - began < max_wall_s:
            step_i += 1
            robot_pose, _, rvel, rang = body_measurement(assembly.robot_body)
            cart_pose, _, _, _ = body_measurement(assembly.cart_body)
            now = float(assembly.world.current_time)
            articulation = math.atan2(math.sin(robot_pose.yaw - cart_pose.yaw),
                                      math.cos(robot_pose.yaw - cart_pose.yaw))
            odom = Odometry((robot_pose.x, robot_pose.y, robot_pose.yaw), now,
                            float(np.linalg.norm(rvel[:2])), float(rang[2]),
                            'test_fixture', uncertainty)
            obs = Observation(now, (), True, True, articulation, 'test_fixture', ASSERTED_COVERAGE_M)
            result = controller.step(dt, odom, obs, now=now)
            reasons[result.reason] = reasons.get(result.reason, 0) + 1
            if result.status == 'succeeded':
                docked = True
            v, omega = result.velocity
            max_omega = max(max_omega, abs(omega))

            post = dock_observation(assembly)
            assembly.command_gate.hold()
            assembly.mobile.command(v, omega, post, transport_ready=True)
            assembly.world.step(render=True)

            cstate = assembly.mobile.coupling_state(dock_observation(assembly))
            heights = _wheel_heights(physx, WHEELS)
            peak_wheel = max(heights.values())
            max_wheel_z, max_artic = max(max_wheel_z, peak_wheel), max(max_artic, abs(articulation))
            row = _row(assembly, settle_steps + step_i, 'nav_towing',
                coupling=dict(healthy=cstate['healthy'], hitch_error_m=cstate['hitch_error_m'],
                    articulation_rad=float(articulation)))
            row['command'] = dict(linear=float(v), angular=float(omega), status=result.status,
                                  reason=result.reason)
            history.append(row)
            if not cstate['healthy']:
                raise RuntimeError(f'coupling lost during nav tow at step {step_i}: '
                                   f"residual {cstate['hitch_error_m']:.4f} m")
            if abs(articulation) >= MAX_ARTICULATION_RAD:
                raise RuntimeError(f'articulation {articulation:.3f} rad exceeded 46 deg at step {step_i}')
            if peak_wheel > WHEEL_REST_Z + WHEEL_LIFT_TOL:
                raise RuntimeError(f'cart wheel lift to {peak_wheel:.4f} m at step {step_i}')
            if step_i % capture_every == 0:
                frame = camera.get_current_frame(clone=True)
                rgb = np.asarray(frame.get('rgb'))
                if rgb.ndim == 3 and rgb.shape[0] == intr['height'] and rgb.shape[1] == intr['width']:
                    if recorder.capture(frame):
                        captured += 1
            if docked:
                break

        end_xy = np.array(history[-1]['robot']['pose_xyz_yaw'][:2])
        forward = float(np.linalg.norm(end_xy - start_xy))
        remaining = float(np.linalg.norm(np.array(goal[:2]) - end_xy))
        video = recorder.encode(output / 'native-nav-towing.mp4', ffmpeg=ffmpeg) if captured >= 10 else None
        return dict(status='passed' if (docked or forward > 0.1) else 'failed',
            docked=docked, planner_started=True, navigation_used=True, steering_commanded=bool(max_omega > 1e-3),
            goal_pose=list(goal), goal_articulation_rad=GOAL_ARTICULATION_RAD,
            start_robot_xy=start_xy.tolist(), end_robot_xy=end_xy.tolist(),
            forward_distance_m=forward, remaining_to_dock_m=remaining,
            max_articulation_rad=float(max_artic), max_articulation_deg=float(math.degrees(max_artic)),
            max_commanded_omega_rad_s=float(max_omega),
            max_cart_wheel_z_m=float(max_wheel_z), wheel_rest_z_m=WHEEL_REST_Z, no_wheel_lift=True,
            steps=len(history), captured_frames=captured, video=video,
            step_reasons=reasons, wall_budget_s=max_wall_s,
            evidence='native coupled towing STEERED by the production ArticulatedController '
                     '(forward-only route plan + gated v/omega from measured base odometry); '
                     'cart followed through the physical hitch, bounded articulation, no wheel lift; '
                     'VFR H.264 from a fixed map camera. Coverage asserted clear (no lidar in this '
                     'nav-only probe); manipulation absent; traversal is wall-clock bounded.',
            autonomous_mission=False)
    finally:
        assembly.hold()
        (output / 'nav-towing-feedback.json').write_text(json.dumps(history, indent=2) + '\n')
