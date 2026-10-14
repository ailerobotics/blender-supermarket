#!/usr/bin/env python3
"""Guarded GPU run of the host-stack PLACE capability probe (place_probe.py).

Honours the shared-GPU discipline: the integration mission has priority, so instead of
failing fast on a held /tmp/aile-gpu.lock (or on VRAM the mission is using) this WAITS and
retries the guard until the GPU frees or a bounded lock-wait expires -- the flock -w
equivalent. The probe subprocess is capped under 8 minutes.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_evidence_mapper_gpu as reg  # noqa: E402


def _retryable(exc):
    # Back off and retry while the mission holds the lock or the VRAM/RAM; never retry a live
    # protected-Blender refusal (that is a hard stop, not a transient busy GPU).
    m = str(exc).lower()
    if 'blender' in m:
        return False
    return isinstance(exc, BlockingIOError) or any(t in m for t in
        ('resource temporarily unavailable', 'free vram', 'available ram', 'lock'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--plugin-env', required=True)
    p.add_argument('--nvidia-runtime', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--ros-host-config', default='')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cap-seconds', type=float, default=460.0)      # < 8 min per probe
    p.add_argument('--lock-wait-seconds', type=float, default=1500.0)  # wait out a 25-min attempt
    p.add_argument('--retry-interval', type=float, default=20.0)
    p.add_argument('--discovery-timeout', type=float, default=150.0)
    p.add_argument('--heavy-threshold-mb', type=int, default=2500)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    runtime = Path(a.nvidia_runtime)
    lib = runtime / 'usr/lib/x86_64-linux-gnu'
    nvsmi = str(runtime / 'usr/bin/nvidia-smi')

    env_prefix = (f'set -e; export RESTOCK_WITH_GPU_USERSPACE=1; source "{a.plugin_env}"; '
                  f'export LD_LIBRARY_PATH="{lib}:$LD_LIBRARY_PATH"; export PATH="{runtime}/usr/bin:$PATH"; '
                  f'export PYTHONPATH="{a.repo}/scripts:$PYTHONPATH"; '
                  f'export ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST '
                  f'ROS_STATIC_PEERS="" RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTDDS_BUILTIN_TRANSPORTS=UDPv4 '
                  f'SKIP_DEFAULT_XML=1; ')

    os.environ['PATH'] = str(runtime / 'usr/bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = str(lib) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ.update(ROS_DOMAIN_ID='88', ROS_LOCALHOST_ONLY='1', ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')

    rhc = f'--ros-host-config "{a.ros_host_config}" ' if a.ros_host_config else ''
    record = {'note': 'host-stack PLACE capability probe at the shelf station', 'config': a.config}
    started = time.monotonic()

    def run_once():
        with reg.guard(nvsmi, a.heavy_threshold_mb) as decision:
            record['entered_guard'] = True
            record['guard'] = {'free_mb': getattr(decision, 'free_mb', None)}
            chk = subprocess.run(['bash', '-lc', env_prefix +
                                  f'exec python3 "{a.repo}/scripts/run_restock_host.py" --config "{a.config}" --check'],
                                 capture_output=True, text=True, timeout=180)
            record['preflight_rc'] = chk.returncode
            if chk.returncode != 0:
                record['preflight_tail'] = chk.stdout[-1200:]
                raise SystemExit('CPU preflight failed; not launching')
            log = open(a.output / 'stack.log', 'wb')
            launch = (f'export RESTOCK_REPO="{a.repo}" RESTOCK_PLUGIN_ENV="{a.plugin_env}" '
                      f'RESTOCK_HOST_CONFIG="{a.config}" RESTOCK_NVIDIA_RUNTIME="{runtime}"; '
                      f'exec bash "{a.repo}/scripts/restock_runtime/launch_restock_host_stack.sh"')
            child = subprocess.Popen(['bash', '-lc', launch], stdout=log, stderr=subprocess.STDOUT,
                                     preexec_fn=os.setsid)
            record['stack_pid'] = child.pid
            try:
                probe = subprocess.run(['bash', '-lc', env_prefix +
                                        f'exec python3 "{a.repo}/scripts/restock_runtime/place_probe.py" '
                                        f'--config "{a.config}" {rhc}--output "{a.output}" '
                                        f'--discovery-timeout {a.discovery_timeout}'],
                                       capture_output=True, text=True, timeout=a.cap_seconds)
                record['probe_rc'] = probe.returncode
                record['probe_stderr_tail'] = probe.stderr[-1200:]
                pj = a.output / 'place_probe.json'
                record['probe'] = json.loads(pj.read_text()) if pj.exists() else None
            finally:
                try:
                    os.killpg(os.getpgid(child.pid), signal.SIGINT); time.sleep(5)
                    if child.poll() is None:
                        os.killpg(os.getpgid(child.pid), signal.SIGTERM); time.sleep(4)
                    if child.poll() is None:
                        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                log.close()

    deadline = time.monotonic() + a.lock_wait_seconds
    attempts = 0
    while True:
        attempts += 1
        record['entered_guard'] = False
        try:
            run_once()
            break
        except Exception as exc:
            if not record.get('entered_guard') and _retryable(exc) and time.monotonic() < deadline:
                record['waits'] = record.get('waits', 0) + 1
                time.sleep(a.retry_interval)
                continue
            record['error'] = f'{type(exc).__name__}: {exc}'
            break
    record['attempts'] = attempts
    record['elapsed_s'] = round(time.monotonic() - started, 1)
    pr = record.get('probe') or {}
    record['summary'] = {'PASS': pr.get('PASS'), 'nodes': (pr.get('nodes') or {}).get('all_present'),
                         'station': pr.get('station'), 'esdf': pr.get('esdf'),
                         'attach': pr.get('attach'), 'baseline_plan': pr.get('baseline_plan'), 'attached_plan': pr.get('attached_plan'), 'attached_plan_outcome': pr.get('attached_plan_outcome'),
                         'place_plan': pr.get('place_plan'),
                         'probe_error': pr.get('error'), 'waits': record.get('waits', 0),
                         'elapsed_s': record['elapsed_s'], 'runner_error': record.get('error')}
    (a.output / 'result.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    print(json.dumps(record['summary'], indent=2, default=str))
    return 0 if pr.get('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
