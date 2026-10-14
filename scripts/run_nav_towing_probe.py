#!/usr/bin/env python3
"""Bounded guarded native PLANNER-STEERED coupled-towing probe.

Boots Isaac in the task-local NVIDIA runtime under the GPU lock, assembles the
mobile base + rollwagen at the couple pose, latches the hitch, then drives the
production navigation controller (`twin.restock.nav_towing_check`) to steer the
coupled train toward the G3_back_s2 shelf dock and records VFR H.264. This is the
nav side of the outbound stage, no manipulation. It is wall-clock bounded; native
towing is traction-limited so the traverse is partial and reported honestly.

Same guard discipline as run_restock_render_smoke.py: the parent holds the GPU slot
(`/tmp/aile-gpu.lock`) while monitoring one task-owned child, enforcing RAM/VRAM
floors and a hard wall cap.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from twin.restock.guard import gpu_slot
from twin.restock.station import OwnedProcess
from twin.restock.native_environment import ros_environment
from run_restock_render_smoke import write, renderer_metadata, runtime_environment, services, sample


def worker(output, ffmpeg, runtime_config, max_wall_s):
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 640, 'height': 360,
        'renderer': 'RayTracedLighting', 'anti_aliasing': 0, 'multi_gpu': False})
    try:
        write(output / 'renderer-start.json', renderer_metadata())
        from twin.restock.nav_towing_check import run
        config = json.loads(Path(runtime_config).read_text())
        result = run(app, config, output, ffmpeg=ffmpeg, max_wall_s=max_wall_s)
        write(output / 'native-result.json', dict(status=result['status'], nav_towing=result,
            evidence='native planner-steered coupled-towing probe with VFR recording, not an autonomous mission'))
    except BaseException:
        import traceback
        # Isaac's SimulationApp.close() can hard-exit the process, so persist the traceback
        # BEFORE the finally runs or it is lost (child exits 0 with no result).
        (output / 'worker-error.txt').write_text(traceback.format_exc())
        raise
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ffmpeg', required=True, type=Path)
    parser.add_argument('--isaac-python', type=Path)
    parser.add_argument('--nvidia-runtime', type=Path)
    parser.add_argument('--runtime-config', required=True, type=Path)
    parser.add_argument('--heavy-threshold-mb', type=int, choices=(1500, 2500), default=1500)
    parser.add_argument('--max-wall-s', type=float, default=240.0,
                        help='nav-towing stepping budget inside the worker (boot/settle/encode are extra)')
    parser.add_argument('--monitor-cap-s', type=float, default=460.0,
                        help='hard parent wall cap for the whole child (boot + probe + encode)')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.output, args.ffmpeg, args.runtime_config, args.max_wall_s); return
    if not args.isaac_python or not args.nvidia_runtime:
        parser.error('--isaac-python and --nvidia-runtime required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = runtime_environment(args.nvidia_runtime, args.output)
    environment = ros_environment(environment, args.isaac_python)
    os.environ.clear(); os.environ.update(environment)
    record = dict(status='failed', code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  telemetry=[], runtime_root=str(args.nvidia_runtime), monitor_cap_s=args.monitor_cap_s)
    child = None
    try:
        with gpu_slot(heavy_threshold_mb=args.heavy_threshold_mb) as decision:
            if decision.free_mb < 8192:
                raise RuntimeError('need at least 8 GiB free VRAM')
            record['guard'] = asdict(decision)
            record['services_before'] = services()
            command = [str(args.isaac_python), str(Path(__file__).resolve()), '--worker',
                       '--output', str(args.output), '--ffmpeg', str(args.ffmpeg),
                       '--runtime-config', str(args.runtime_config),
                       '--max-wall-s', str(args.max_wall_s)]
            child = OwnedProcess(command, args.output / 'isaac.log', env=environment)
            child.start()
            record['pid'] = child.process.pid
            started = time.monotonic()
            while child.alive:
                value = sample(); record['telemetry'].append(value)
                write(args.output / 'run.json', record)
                if value['ram_available_mib'] < 4096 or value['gpu_free_mib'] < 1024:
                    raise RuntimeError('task resource floor reached')
                if time.monotonic() - started > args.monitor_cap_s:
                    raise TimeoutError('native nav-towing probe exceeded monitor cap')
                time.sleep(1.)
            record['exit_code'] = child.process.returncode
            record['services_after'] = services()
            if child.process.returncode:
                raise RuntimeError('native child failed; inspect isaac.log')
            native = json.loads((args.output / 'native-result.json').read_text())
            record['native'] = native
            record['status'] = native.get('status', 'failed')
    except BaseException as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if child is not None:
            child.close()
        write(args.output / 'run.json', record)
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
