#!/usr/bin/env python3
"""Bounded, guarded GPU run: measure a native incorporation identity end to end.

Holds the exclusive GPU guard (flock /tmp/aile-gpu.lock + protected-Blender refusal
+ >=8 GiB free VRAM + >=12 GiB RAM + domain-88 isolation), launches the navigation
nvblox mapper as restock_mapper_evidence::EvidenceProducerNode (launch_evidence_mapper.py
--execute), lets it construct, then runs the sensor+handshake driver
(twin.restock_host.sensor_handshake) in the SAME domain. The driver certifies the
empty map, opens admission, registers + integrates synthetic depth frames, consumes
the resulting EvidenceSnapshot through evidence_bridge and TemporalMap, and records
the MEASURED incorporation identity. Both children are torn down by exact PID group.

This proves the whole empty->admit->register->integrate->snapshot->consume path with
a FIXTURE depth source (not Isaac perception); it is honest about that in the report.
Nothing here is commissioning: the full mission still gates on real sensing, nav and
collision preflight. Absolute wall ceiling and the per-phase caps keep it well inside
the 20-minute GPU budget.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def _available_ram_mb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    raise RuntimeError('cannot determine available RAM')


def _protected_blender():
    for comm in Path('/proc').glob('[0-9]*/comm'):
        try:
            if 'blender' in comm.read_text().lower():
                return True
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            raise RuntimeError('cannot inspect protected process state')
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(('127.0.0.1', 9876)) == 0


def _gpu_free_mib(nvsmi):
    out = subprocess.run([nvsmi, '--query-gpu=memory.free,memory.used,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15).stdout.strip()
    free, used, total = [int(v) for v in out.splitlines()[0].split(',')]
    return free, used, total


@contextmanager
def _inline_guard(nvsmi, heavy_threshold_mb):
    import fcntl
    os.environ.update(ROS_DOMAIN_ID='88', ROS_LOCALHOST_ONLY='1', ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')
    with open('/tmp/aile-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if _protected_blender():
            raise RuntimeError('protected Blender process or MCP socket present')
        if _available_ram_mb() < 12 * 1024:
            raise RuntimeError('need >=12 GiB available RAM')
        free, used, total = _gpu_free_mib(nvsmi)
        if free < 8000:
            raise RuntimeError(f'need >=8000 MiB free VRAM; have {free}')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1', 7400 + 250 * 88))
        yield type('D', (), {'ok': True, 'free_mb': free, 'message': f'inline guard: {free} MiB free of {total}'})


@contextmanager
def guard(nvsmi, heavy_threshold_mb):
    try:
        from twin.restock.guard import gpu_slot
    except Exception:
        with _inline_guard(nvsmi, heavy_threshold_mb) as d:
            yield d
        return
    with gpu_slot(heavy_threshold_mb=heavy_threshold_mb) as d:
        yield d


def _spawn(cmd_sh, logpath):
    log = open(logpath, 'wb')
    child = subprocess.Popen(['bash', '-lc', cmd_sh], stdout=log, stderr=subprocess.STDOUT,
                             preexec_fn=os.setsid)
    return child, log


def _teardown(child):
    try:
        os.killpg(os.getpgid(child.pid), signal.SIGINT)
        time.sleep(4)
        if child.poll() is None:
            os.killpg(os.getpgid(child.pid), signal.SIGTERM)
            time.sleep(3)
        if child.poll() is None:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--plugin-env', required=True)
    p.add_argument('--nvidia-runtime', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--generation', default='mission_incorp_a')
    p.add_argument('--camera-count', type=int, default=1)
    p.add_argument('--transport-height', type=float, default=1.9)
    p.add_argument('--sensors', nargs='+', default=['front=front_optical'])
    p.add_argument('--session', default='mission_a')
    p.add_argument('--scene', default='supermarket')
    p.add_argument('--cameras', default='front')
    p.add_argument('--localization', default='amcl')
    p.add_argument('--warmup', type=float, default=40.0, help='mapper construction grace (s)')
    p.add_argument('--driver-window', type=float, default=120.0, help='driver incorporation window (s)')
    p.add_argument('--probe-empty-count', type=int, default=0,
                   help='diagnostic: driver only polls EmptyGeneration N times and exits')
    p.add_argument('--probe-interval', type=float, default=1.0)
    p.add_argument('--mapper-param', action='append', default=[],
                   help='override a native mapper parameter, KEY=VALUE (repeatable)')
    p.add_argument('--heavy-threshold-mb', type=int, default=2500)
    a = p.parse_args()
    ceiling = a.warmup + a.driver_window + 60
    if ceiling > 1200:
        raise SystemExit('absolute wall ceiling exceeds the 20-minute GPU budget')
    a.output.mkdir(parents=True, exist_ok=True)
    runtime = Path(a.nvidia_runtime)
    lib = runtime / 'usr/lib/x86_64-linux-gnu'
    nvsmi = str(runtime / 'usr/bin/nvidia-smi')

    os.environ['PATH'] = str(runtime / 'usr/bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = str(lib) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ.update(ROS_DOMAIN_ID='88', ROS_LOCALHOST_ONLY='1', ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')

    env_prefix = (f'set -e; export RESTOCK_WITH_GPU_USERSPACE=1; source "{a.plugin_env}"; '
                  f'export LD_LIBRARY_PATH="{lib}:$LD_LIBRARY_PATH"; export PATH="{runtime}/usr/bin:$PATH"; '
                  f'export PYTHONPATH="{a.repo}/scripts:$PYTHONPATH"; '
                  f'export ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST; ')

    sensors = ' '.join(a.sensors)
    mapper_params = ' '.join(f'--mapper-param {mp}' for mp in a.mapper_param)
    mapper_sh = (env_prefix +
                 f'exec python3 "{a.repo}/scripts/launch_evidence_mapper.py" --execute '
                 f'--generation {a.generation} --camera-count {a.camera_count} '
                 f'--transport-height {a.transport_height} --session {a.session} --scene {a.scene} '
                 f'--cameras {a.cameras} --localization {a.localization} --sensors {sensors} {mapper_params}')
    driver_sh = (env_prefix +
                 f'exec python3 "{a.repo}/scripts/twin/restock_host/sensor_handshake.py" '
                 f'--generation {a.generation} --output "{a.output}" --session {a.session} '
                 f'--scene {a.scene} --cameras {a.cameras} --localization {a.localization} '
                 f'--sensors {sensors} --window {a.driver_window} '
                 f'--probe-empty-count {a.probe_empty_count} --probe-interval {a.probe_interval}')

    record = {'generation': a.generation, 'warmup': a.warmup, 'driver_window': a.driver_window}
    mapper = driver = None
    mlog = dlog = None
    try:
        with guard(nvsmi, a.heavy_threshold_mb) as decision:
            record['guard'] = {'ok': getattr(decision, 'ok', True),
                               'free_mb': getattr(decision, 'free_mb', None),
                               'message': getattr(decision, 'message', '')}
            record['gpu_free_before'] = _gpu_free_mib(nvsmi)[0]
            mapper, mlog = _spawn(mapper_sh, a.output / 'mapper.log')
            record['mapper_pid'] = mapper.pid
            # Warm up: fail fast if the mapper exits during construction.
            t0 = time.monotonic()
            while time.monotonic() - t0 < a.warmup:
                if mapper.poll() is not None:
                    record['mapper_exited_during_warmup'] = mapper.returncode
                    break
                time.sleep(2)
            record['gpu_free_after_warmup'] = _gpu_free_mib(nvsmi)[0]
            if mapper.poll() is None:
                driver, dlog = _spawn(driver_sh, a.output / 'driver.log')
                record['driver_pid'] = driver.pid
                t1 = time.monotonic()
                while time.monotonic() - t1 < a.driver_window + 20:
                    if driver.poll() is not None:
                        record['driver_exit'] = driver.returncode
                        break
                    time.sleep(3)
                else:
                    record['driver_timed_out'] = True
            record['gpu_free_after_run'] = _gpu_free_mib(nvsmi)[0]
            if driver is not None:
                _teardown(driver)
            if mapper is not None:
                _teardown(mapper)
            for lg in (mlog, dlog):
                if lg:
                    lg.close()
            record['gpu_free_after_teardown'] = _gpu_free_mib(nvsmi)[0]
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        for ch in (driver, mapper):
            if ch is not None:
                _teardown(ch)
        for lg in (mlog, dlog):
            if lg:
                try:
                    lg.close()
                except Exception:
                    pass

    hs = a.output / 'handshake_result.json'
    if hs.exists():
        try:
            record['handshake'] = json.loads(hs.read_text()).get('measured', {})
        except json.JSONDecodeError:
            record['handshake'] = {'parse_error': True}
    mapper_log = (a.output / 'mapper.log')
    if mapper_log.exists():
        record['mapper_log_tail'] = mapper_log.read_text(errors='replace')[-2500:]
    (a.output / 'result.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'mapper_log_tail'}, indent=2, default=str))
    return 0 if record.get('handshake', {}).get('CONSUMED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
