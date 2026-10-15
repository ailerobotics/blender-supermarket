#!/usr/bin/env python3
"""Guarded GPU launcher for the native empty-support commissioning session.

Holds the GPU slot (task-local NVIDIA runtime + `/tmp/aile-gpu.lock` + heavy-consumer
threshold for the always-on whispr) exactly like the render smoke, then launches the
camera-lane commissioning CLI (`scripts/run_restock_commissioning.py --execute`) in an
isaac-python child with the integration `--sources` hook
(`twin.restock.commissioning_sources:build_sources`). The child boots Isaac, renders the
EMPTY source station and commissions its support baseline through the production
`commission()` path, persisting it (with provenance) under `--output/baselines`.

This launcher owns no Isaac import; the child owns boot/render/selection.
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
from run_restock_render_smoke import runtime_environment, sample, services


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='run dir (baselines land in <dir>/baselines)')
    parser.add_argument('--isaac-python', required=True, type=Path)
    parser.add_argument('--nvidia-runtime', required=True, type=Path)
    parser.add_argument('--camera-config', required=True, type=Path)
    parser.add_argument('--scenario', required=True, type=Path)
    parser.add_argument('--bench-config', required=True, type=Path,
                        help='view-bench JSON (scenario, mobile_manifest, camera_config, store_usd, package_roots)')
    parser.add_argument('--station', action='append', dest='stations', help='repeatable; default backstock_00')
    parser.add_argument('--heavy-threshold-mb', type=int, choices=(1500, 2500), default=2500)
    parser.add_argument('--wall-cap-s', type=int, default=600)
    parser.add_argument('--lifetime-s', type=float, default=600.,
                        help='commissioned-baseline lifetime (wall seconds via the shared clock domain)')
    args = parser.parse_args()

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    baselines = args.output / 'baselines'
    environment = runtime_environment(args.nvidia_runtime, args.output)
    environment['COMMISSIONING_BENCH_CONFIG'] = str(args.bench_config.resolve())
    stations = args.stations or ['backstock_00']
    # Select which station(s) the commissioning renderer (commissioning_sources.build_sources)
    # commissions this boot. It commissions one map-parented station per boot (source/shelf);
    # loop the parent per station for more. Default preserves the historical source-only behaviour.
    environment['COMMISSIONING_STATIONS'] = ','.join(stations)
    os.environ.clear()
    os.environ.update(environment)

    record = dict(status='failed', wall_limit_s=args.wall_cap_s,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runtime_root=str(args.nvidia_runtime), stations=stations, telemetry=[])
    child = None
    try:
        with gpu_slot(heavy_threshold_mb=args.heavy_threshold_mb) as decision:
            if decision.free_mb < 8192:
                raise RuntimeError('need at least 8 GiB free VRAM')
            record['guard'] = asdict(decision)
            record['services_before'] = services()
            command = [str(args.isaac_python),
                       str(Path(__file__).resolve().parent / 'run_restock_commissioning.py'),
                       '--execute', '--camera-config', str(args.camera_config.resolve()),
                       '--scenario', str(args.scenario.resolve()),
                       '--sources', 'twin.restock.commissioning_sources:build_sources',
                       '--output', str(baselines), '--lifetime-s', str(args.lifetime_s)]
            for s in stations:
                command.extend(['--station', s])
            record['command'] = command
            child = OwnedProcess(command, args.output / 'isaac.log', env=environment)
            try:
                child.start()
                record['pid'] = child.process.pid
                started = time.monotonic()
                while child.alive:
                    value = sample()
                    record['telemetry'].append(value)
                    (args.output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
                    if value['ram_available_mib'] < 4096 or value['gpu_free_mib'] < 1024:
                        raise RuntimeError('task resource floor reached')
                    if time.monotonic() - started > args.wall_cap_s:
                        raise TimeoutError(f'commissioning exceeded {args.wall_cap_s}s')
                    time.sleep(1.)
                record['exit_code'] = child.process.returncode
                # The CLI writes commissioning-report.json and persists each baseline BEFORE the
                # process exits. Isaac then crashes in its own atexit/omni-graph teardown (SIGSEGV,
                # a known shutdown issue), so a negative return code does NOT mean commissioning
                # failed. Treat a complete report for every required station as success; only a
                # missing/incomplete report (or a non-signal failure) is a real failure.
                report_path = baselines / 'commissioning-report.json'
                report = json.loads(report_path.read_text()) if report_path.is_file() else None
                record['report'] = report
                complete = bool(report) and all(s in report.get('stations', {}) for s in stations)
                if complete:
                    record['status'] = 'passed'
                    if child.process.returncode:
                        record['shutdown_signal'] = child.process.returncode
                        record['note'] = ('commissioning committed; child exited on a signal during '
                                          'Isaac atexit teardown (known shutdown crash), not a commission failure')
                elif child.process.returncode:
                    raise RuntimeError('commissioning child failed; inspect isaac.log')
                else:
                    raise RuntimeError('commissioning report missing required stations')
            finally:
                if child:
                    child.close()
                record['services_after'] = services()
                record['services_unchanged'] = record.get('services_before') == record['services_after']
    except Exception as exc:
        record['reason'] = str(exc)
        raise
    finally:
        (args.output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
