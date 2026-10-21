#!/usr/bin/env python3
"""Guarded UNCOUPLED base-drive-fidelity bench (fm434).

Boots Isaac ONCE under the canonical GPU lock, assembles the mission's mobile base (NO cart coupling),
commands an OPEN-LOOP step profile (straight 0.25 m/s, then an arc at the 1.5 m min radius) directly via
the production wheel-target conversion, and records commanded-vs-achieved (v, omega) per tick. This
isolates the DRIVE's ability to achieve the command from the follower/gate -- run-325/326 showed the base
under-tracks even uncoupled (cart drag refuted). One physics lever per run (fm434 order: wheel traction,
then solver iterations, then physics_dt; the wheel-radius/diff-drive units error was ruled out
statically). Short: <= 15 min wall under /tmp/aile-gpu.lock, --heavy-threshold-mb 2500 for whispr, never
--force. Writes base-drive-bench.json with the per-phase tracking ratios + within_tol verdict.

Levers (CLI, one per run): --wheel-friction (traction material), --solver-iters (articulation position
iterations), --physics-dt. Omitted => the assembled defaults (baseline run first).
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from twin.restock.guard import gpu_slot
from twin.tools.base_drive_bench import step_profile, tracking_report


def _boot():
    from isaacsim import SimulationApp
    return SimulationApp({'headless': True, 'width': 320, 'height': 240})


def run(app, config, output, *, wheel_friction=None, solver_iters=None, physics_dt=None,
        straight_s=4.0, arc_s=6.0, v=0.25, min_radius=1.5, max_wall_s=600.0):
    from twin.restock.physics_setup import assemble, body_measurement
    from twin.mobile import DockObservation, Pose2
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
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
    # settle the free base
    for _ in range(60):
        assembly.world.step(render=False)
    cmds = step_profile(dt=dt, straight_s=straight_s, arc_s=arc_s, v=v, min_radius=min_radius)
    achieved = []
    t0 = time.monotonic()
    for c in cmds:
        r, _rz, rv, rw = body_measurement(rb)
        # command the base open-loop (uncoupled); obs carries the measured base pose only
        obs = DockObservation(Pose2(float(r[0]), float(r[1]), 0.0), Pose2(-1.27, 0.0, 0.0),
                              0.28, 0.28, 0.0, 0.0, 0.0, 0.0)
        mobile.command(c.v, c.omega, obs, transport_ready=True)
        assembly.world.step(render=False)
        r2, _z2, rv2, rw2 = body_measurement(rb)
        achieved.append((float(np.linalg.norm(rv2)), float(rw2[2])))
        if time.monotonic() - t0 > max_wall_s:
            break
    report = tracking_report(cmds[:len(achieved)], achieved)
    report.update(wheel_friction=cfg.get('wheel_friction'), solver_iters=solver_iters,
                  physics_dt=dt, ticks=len(achieved), v=v, min_radius=min_radius)
    (output / 'base-drive-bench.json').write_text(json.dumps(report, indent=2))
    print('BASE_DRIVE_BENCH ' + json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wheel-friction', type=float)
    p.add_argument('--solver-iters', type=int)
    p.add_argument('--physics-dt', type=float)
    p.add_argument('--heavy-threshold-mb', type=int, default=1500)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    with gpu_slot(heavy_threshold_mb=args.heavy_threshold_mb):
        app = _boot()
        try:
            run(app, config, args.output, wheel_friction=args.wheel_friction,
                solver_iters=args.solver_iters, physics_dt=args.physics_dt)
        except Exception:
            (args.output).mkdir(parents=True, exist_ok=True)
            (args.output / 'base-drive-bench.err').write_text(traceback.format_exc())
            raise
        finally:
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
