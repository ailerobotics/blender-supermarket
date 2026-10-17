#!/usr/bin/env python3
"""Guarded GPU launcher for the native empty-support commissioning session.

Holds the GPU slot (task-local NVIDIA runtime + `/tmp/aile-gpu.lock` + heavy-consumer
threshold for the always-on whispr) exactly like the render smoke, then commissions one
or more station baselines inside that single guarded session and assembles them into one
dataset dir the following mission boot loads.

The commissioning renderer (`twin.restock.commissioning_sources:build_sources`) commissions
exactly ONE station per Isaac boot (its empty-support optical bench renders one station
camera's view). To produce every baseline the full mission needs — the collect source
(`backstock_00`), the cart (`rollwagen`) and the transfer shelf (`G3_back_s2`) — in ONE
guarded session, this launcher loops the camera-lane commissioning CLI
(`scripts/run_restock_commissioning.py --execute`) once per requested `--station`, each child
in an isaac-python boot that writes into its own per-station dir, all under a single
`/tmp/aile-gpu.lock` hold. It then copies each station's persisted baseline into the shared
`<output>/baselines` dir and rebuilds the union manifest, so one `commissioning_dir` carries
every station — produced in one guarded boot window, session-bound and within the lifetime,
consumable by the mission that follows. A station whose baseline the renderer does not produce
is refused, never fabricated.

This launcher owns no Isaac import; each child owns boot/render/selection, and the final
manifest assembly is pure CPU (`twin.restock_camera.commissioning`).
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from twin.restock.guard import gpu_slot
from twin.restock.station import OwnedProcess
from run_restock_render_smoke import runtime_environment, sample, services


def assemble_baselines(per_station_dirs, final_dir, stations):
    """Copy each station's persisted baseline into ``final_dir`` and rebuild the union manifest.

    ``per_station_dirs[station]`` is the child's ``--output`` baselines dir (holding
    ``<station>.json`` and the optional ``<station>.npz`` audit sidecar). Each requested
    station's baseline is copied into ``final_dir``; ``rebuild_manifest`` then unions every
    ``<station>.json`` present into one manifest. A requested station whose baseline the
    renderer never produced is refused (never fabricated). Pure CPU — no Isaac.

    Returns the loaded union-manifest dict.
    """
    from twin.restock_camera.commissioning import rebuild_manifest, MANIFEST
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    for station in stations:
        src = Path(per_station_dirs[station])
        baseline = src / f'{station}.json'
        if not baseline.is_file():
            raise RuntimeError(f'commissioning produced no baseline for station {station!r} '
                               f'(expected {baseline}); inspect isaac-{station}.log')
        shutil.copy2(baseline, final_dir / baseline.name)
        npz = src / f'{station}.npz'                    # raw-frame audit sidecar (optional)
        if npz.is_file():
            shutil.copy2(npz, final_dir / npz.name)
    # Rebuild BEFORE the combined report lands in final_dir: rebuild_manifest globs '*.json'
    # and only skips the manifest itself, so the per-station baselines must be the only json
    # files present when it runs (a report file would fail to parse as a baseline).
    rebuild_manifest(final_dir)
    manifest = json.loads((final_dir / MANIFEST).read_text())
    missing = [s for s in stations if s not in manifest.get('stations', {})]
    if missing:
        raise RuntimeError(f'assembled commissioning manifest missing required station(s): {missing}')
    return manifest


def _await_child(child, record, output, wall_cap_s):
    """Run the telemetry + resource-floor + wall-cap watch loop until ``child`` exits.

    Appends telemetry to ``record`` and flushes ``run.json`` each second. Returns the child's
    exit code. Raises on the resource floor or the per-child wall cap.
    """
    started = time.monotonic()
    while child.alive:
        value = sample()
        record['telemetry'].append(value)
        (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
        if value['ram_available_mib'] < 4096 or value['gpu_free_mib'] < 1024:
            raise RuntimeError('task resource floor reached')
        if time.monotonic() - started > wall_cap_s:
            raise TimeoutError(f'commissioning exceeded {wall_cap_s}s')
        time.sleep(1.)
    return child.process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='run dir (baselines land in <dir>/baselines)')
    parser.add_argument('--isaac-python', required=True, type=Path)
    parser.add_argument('--nvidia-runtime', required=True, type=Path)
    parser.add_argument('--camera-config', required=True, type=Path)
    parser.add_argument('--scenario', required=True, type=Path)
    parser.add_argument('--bench-config', required=True, type=Path,
                        help='view-bench JSON (scenario, mobile_manifest, camera_config, store_usd, package_roots)')
    parser.add_argument('--station', action='append', dest='stations',
                        help='repeatable mission location to commission this guarded session; '
                             'default backstock_00. Pass all three (backstock_00 rollwagen G3_back_s2) '
                             'to produce every mission baseline in one boot window')
    parser.add_argument('--heavy-threshold-mb', type=int, choices=(1500, 2500), default=2500)
    parser.add_argument('--wall-cap-s', type=int, default=600,
                        help='wall-clock cap for EACH station commissioning child (per boot)')
    parser.add_argument('--lifetime-s', type=float, default=600.,
                        help='commissioned-baseline lifetime (wall seconds via the shared clock domain)')
    parser.add_argument('--render-preset', choices=('full', 'fast'),
                        help='opt into a sensing render preset (full|fast; fast=0.6 scale); '
                             'forwarded to the commissioning CLI. Use the SAME preset for the mission.')
    args = parser.parse_args()

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    baselines = args.output / 'baselines'
    stations = args.stations or ['backstock_00']
    environment = runtime_environment(args.nvidia_runtime, args.output)
    environment['COMMISSIONING_BENCH_CONFIG'] = str(args.bench_config.resolve())

    record = dict(status='failed', wall_limit_s=args.wall_cap_s,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runtime_root=str(args.nvidia_runtime), stations=stations, telemetry=[],
                  per_station={})
    cli = Path(__file__).resolve().parent / 'run_restock_commissioning.py'
    per_station_dirs = {}
    try:
        with gpu_slot(heavy_threshold_mb=args.heavy_threshold_mb) as decision:
            if decision.free_mb < 8192:
                raise RuntimeError('need at least 8 GiB free VRAM')
            record['guard'] = asdict(decision)
            record['services_before'] = services()
            # Loop the renderer once per station inside this single guarded session (the renderer
            # commissions one station per boot). Each child writes into its own per-station dir;
            # the baselines are assembled into the shared dir afterwards.
            for station in stations:
                station_out = args.output / 'stations' / station
                station_baselines = station_out / 'baselines'
                station_out.mkdir(parents=True, exist_ok=True)
                per_station_dirs[station] = station_baselines
                child_env = dict(environment)
                # The renderer commissions exactly the station named here (one per boot).
                child_env['COMMISSIONING_STATIONS'] = station
                command = [str(args.isaac_python), str(cli),
                           '--execute', '--camera-config', str(args.camera_config.resolve()),
                           '--scenario', str(args.scenario.resolve()),
                           '--sources', 'twin.restock.commissioning_sources:build_sources',
                           '--output', str(station_baselines), '--lifetime-s', str(args.lifetime_s),
                           '--station', station]
                if args.render_preset:
                    command.extend(['--render-preset', args.render_preset])
                record['per_station'][station] = dict(command=command, status='running')
                (args.output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')

                child = OwnedProcess(command, args.output / f'isaac-{station}.log', env=child_env)
                try:
                    child.start()
                    record['per_station'][station]['pid'] = child.process.pid
                    returncode = _await_child(child, record, args.output, args.wall_cap_s)
                finally:
                    child.close()
                record['per_station'][station]['exit_code'] = returncode

                # The CLI writes <station>/baselines/commissioning-report.json and persists the
                # baseline BEFORE the process exits. Isaac then usually crashes in its own
                # atexit/omni-graph teardown (SIGSEGV, a known shutdown issue), so a negative
                # return code does NOT mean commissioning failed. Treat a complete per-station
                # report as success; only a missing report (or a non-signal failure) is real.
                report_path = station_baselines / 'commissioning-report.json'
                report = json.loads(report_path.read_text()) if report_path.is_file() else None
                record['per_station'][station]['report'] = report
                committed = bool(report) and station in report.get('stations', {})
                if committed:
                    record['per_station'][station]['status'] = 'passed'
                    if returncode:
                        record['per_station'][station]['shutdown_signal'] = returncode
                        record['per_station'][station]['note'] = (
                            'commissioning committed; child exited on a signal during Isaac atexit '
                            'teardown (known shutdown crash), not a commission failure')
                elif returncode:
                    raise RuntimeError(f'commissioning child for {station!r} failed; inspect isaac-{station}.log')
                else:
                    raise RuntimeError(f'commissioning report for {station!r} missing the station')

            # Assemble every per-station baseline into the shared dir + union manifest the
            # mission's commissioning_dir points at, and write the combined report.
            manifest = assemble_baselines(per_station_dirs, baselines, stations)
            report = dict(schema='restock-commissioning-report.v1', dataset_dir=str(baselines),
                          stations={s: manifest['provenance'][s] for s in stations})
            (baselines / 'commissioning-report.json').write_text(json.dumps(report, indent=1, sort_keys=True))
            record['report'] = report
            record['manifest_stations'] = sorted(manifest.get('stations', {}))
            record['status'] = 'passed'
            record['services_after'] = services()
            record['services_unchanged'] = record.get('services_before') == record['services_after']
    except Exception as exc:
        record['reason'] = str(exc)
        raise
    finally:
        (args.output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
