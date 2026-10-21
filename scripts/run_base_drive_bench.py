#!/usr/bin/env python3
"""Guarded UNCOUPLED base-drive-fidelity bench (fm434).

Boots Isaac ONCE under the canonical GPU lock in the task-local NVIDIA runtime, assembles the mission's
mobile base (NO cart coupling), commands an OPEN-LOOP step profile (straight 0.25 m/s, then an arc at the
1.5 m min radius) directly via the production wheel-target conversion, and records commanded-vs-achieved
(v, omega) per tick. This isolates the DRIVE's ability to achieve the command from the follower/gate --
run-325/326 showed the base under-tracks even uncoupled (cart drag refuted). One physics lever per run
(fm434 order: wheel traction, then solver iterations, then physics_dt; the wheel-radius/diff-drive units
error was ruled out statically). Short: <= 15 min wall under /tmp/aile-gpu.lock, --heavy-threshold-mb 2500
for whispr, never --force. Writes base-drive-bench.json with per-phase tracking ratios + within_tol.

Parent/worker like run_nav_towing_probe: the parent takes the GPU slot + builds the NVIDIA-runtime env and
spawns the ISAAC-python worker; the worker boots SimulationApp and runs the profile.
"""
import argparse
from dataclasses import asdict
import json
import os
import time
import traceback
from pathlib import Path

from twin.restock.guard import gpu_slot
from twin.restock.station import OwnedProcess
from twin.restock.native_environment import ros_environment
from run_restock_render_smoke import write, runtime_environment, services, sample
from twin.tools.base_drive_bench import step_profile, tracking_report


def worker(config_path, output, *, wheel_friction, solver_iters, physics_dt,
           straight_s=4.0, arc_s=6.0, v=0.25, min_radius=1.5, max_wall_s=540.0):
    import numpy as np
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 320, 'height': 240})
    try:
        from twin.restock.physics_setup import assemble, body_measurement
        from twin.mobile import DockObservation, Pose2
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        config = json.loads(Path(config_path).read_text())
        scenario = json.loads(Path(config['scenario']).read_text())
        cfg = dict(config)
        if wheel_friction is not None:
            cfg['wheel_friction'] = float(wheel_friction)          # lever 1: traction
        dt = float(physics_dt) if physics_dt else 1.0 / 60.0       # lever 4
        assembly = assemble(app=app, config=cfg, scenario=scenario, output=output / 'assembly')
        if solver_iters is not None:                               # lever 3: articulation position iters
            from pxr import PhysxSchema
            root = assembly.stage.GetPrimAtPath(assembly.paths['articulation'])
            api = PhysxSchema.PhysxArticulationAPI.Apply(root)
            api.CreateSolverPositionIterationCountAttr().Set(int(solver_iters))
        assembly.world.reset()
        mobile, rb = assembly.mobile, assembly.robot_body
        for _ in range(60):
            assembly.world.step(render=False)
        cmds = step_profile(dt=dt, straight_s=straight_s, arc_s=arc_s, v=v, min_radius=min_radius)
        achieved = []
        t0 = time.monotonic()
        for c in cmds:
            r, _rz, _rv, _rw = body_measurement(rb)
            obs = DockObservation(Pose2(float(r[0]), float(r[1]), 0.0), Pose2(-1.27, 0.0, 0.0),
                                  0.28, 0.28, 0.0, 0.0, 0.0, 0.0)
            mobile.command(c.v, c.omega, obs, transport_ready=True)
            assembly.world.step(render=False)
            _r2, _z2, rv2, rw2 = body_measurement(rb)
            achieved.append((float(np.linalg.norm(rv2)), float(rw2[2])))
            if time.monotonic() - t0 > max_wall_s:
                break
        report = tracking_report(cmds[:len(achieved)], achieved)
        report.update(wheel_friction=cfg.get('wheel_friction'), solver_iters=solver_iters,
                      physics_dt=dt, ticks=len(achieved), v=v, min_radius=min_radius, status='ok')
        (output / 'base-drive-bench.json').write_text(json.dumps(report, indent=2))
        print('BASE_DRIVE_BENCH ' + json.dumps(report), flush=True)
    except Exception:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / 'base-drive-bench.err').write_text(traceback.format_exc())
        raise
    finally:
        app.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--isaac-python', type=Path)
    p.add_argument('--nvidia-runtime', type=Path)
    p.add_argument('--wheel-friction', type=float)
    p.add_argument('--solver-iters', type=int)
    p.add_argument('--physics-dt', type=float)
    p.add_argument('--heavy-threshold-mb', type=int, choices=(1500, 2500), default=1500)
    p.add_argument('--monitor-cap-s', type=float, default=500.0)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args.config, args.output, wheel_friction=args.wheel_friction,
               solver_iters=args.solver_iters, physics_dt=args.physics_dt); return
    if not args.isaac_python or not args.nvidia_runtime:
        p.error('--isaac-python and --nvidia-runtime required')
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    environment = ros_environment(runtime_environment(args.nvidia_runtime, args.output), args.isaac_python)
    rec = dict(status='failed', runtime_root=str(args.nvidia_runtime), telemetry=[])
    with gpu_slot(heavy_threshold_mb=args.heavy_threshold_mb) as decision:
        if decision.free_mb < 8192:
            raise RuntimeError('need at least 8 GiB free VRAM')
        rec['guard'] = asdict(decision)
        cmd = [str(args.isaac_python), str(Path(__file__).resolve()), '--worker',
               '--config', str(args.config), '--output', str(args.output)]
        for flag, val in (('--wheel-friction', args.wheel_friction), ('--solver-iters', args.solver_iters),
                          ('--physics-dt', args.physics_dt)):
            if val is not None:
                cmd += [flag, str(val)]
        child = OwnedProcess(cmd, args.output / 'isaac.log', env=environment)
        child.start(); rec['pid'] = child.process.pid
        started = time.monotonic()
        try:
            while child.alive:
                rec['telemetry'].append(sample()); write(args.output / 'run.json', rec)
                if time.monotonic() - started > args.monitor_cap_s:
                    rec['status'] = 'timeout'; break
                time.sleep(5)
            else:
                rec['status'] = 'done' if child.process.returncode == 0 else 'failed'
        finally:
            child.close()
        rec['returncode'] = child.process.returncode
        write(args.output / 'run.json', rec)
    print('BASE_DRIVE_BENCH_PARENT ' + json.dumps({k: rec[k] for k in ('status', 'returncode') if k in rec}), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
