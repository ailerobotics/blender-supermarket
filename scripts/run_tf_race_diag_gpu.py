#!/usr/bin/env python3
"""Guarded GPU diagnosis of the nvblox depth-before-/tf relay race.

ONE guarded job (same guard as verify_restock_host_stack_gpu): launch the task-isolated
host stack with RESTOCK_NVBLOX_DIAG=1 (nvblox prints every input-queue drop and per-message
delay to the stack log), run tf_race_probe.py (real-shaped masked depth from this process +
the matching ur_base_link->restock_camera_arm /tf from a separate lagging process, per TF
lag), tear down by PID group, and summarise per-lag integration plus the nvblox drop/delay
lines so we can see whether nvblox queues+retries the TF lookup or drops on first miss.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_evidence_mapper_gpu as reg  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--plugin-env', required=True)
    p.add_argument('--nvidia-runtime', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cap-seconds', type=float, default=400.0)
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

    record = {'note': 'nvblox depth-before-tf relay race diagnosis', 'config': a.config}
    started = time.monotonic()
    try:
        with reg.guard(nvsmi, a.heavy_threshold_mb) as decision:
            record['guard'] = {'free_mb': getattr(decision, 'free_mb', None)}
            chk = subprocess.run(['bash', '-lc', env_prefix +
                                  f'exec python3 "{a.repo}/scripts/run_restock_host.py" --config "{a.config}" --check'],
                                 capture_output=True, text=True, timeout=180)
            record['preflight_rc'] = chk.returncode
            if chk.returncode != 0:
                record['preflight_tail'] = chk.stdout[-1500:]
                raise SystemExit('CPU preflight failed; not launching')

            log = open(a.output / 'stack.log', 'wb')
            launch = (f'export RESTOCK_REPO="{a.repo}" RESTOCK_PLUGIN_ENV="{a.plugin_env}" '
                      f'RESTOCK_HOST_CONFIG="{a.config}" RESTOCK_NVIDIA_RUNTIME="{runtime}" '
                      f'RESTOCK_NVBLOX_DIAG=1; '
                      f'exec bash "{a.repo}/scripts/restock_runtime/launch_restock_host_stack.sh"')
            child = subprocess.Popen(['bash', '-lc', launch], stdout=log, stderr=subprocess.STDOUT,
                                     preexec_fn=os.setsid)
            record['stack_pid'] = child.pid
            try:
                probe = subprocess.run(['bash', '-lc', env_prefix +
                                        f'exec python3 "{a.repo}/scripts/restock_runtime/tf_race_probe.py" '
                                        f'--config "{a.config}" --output "{a.output}" '
                                        f'--discovery-timeout {a.discovery_timeout}'],
                                       capture_output=True, text=True, timeout=a.cap_seconds)
                record['probe_rc'] = probe.returncode
                record['probe_stderr_tail'] = probe.stderr[-1500:]
                pj = a.output / 'tf_race_probe.json'
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
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    record['elapsed_s'] = round(time.monotonic() - started, 1)

    # Harvest nvblox's own drop/delay/TF lines from the stack log.
    try:
        text = (a.output / 'stack.log').read_text(errors='replace')
        pat = re.compile(r'(drop|delay|could not|lookup|transform|extrapolat|queue|tf)', re.I)
        hits = [ln for ln in text.splitlines() if 'nvblox' in ln.lower() and pat.search(ln)]
        record['nvblox_diag_lines'] = hits[-60:]
        record['nvblox_diag_line_count'] = len(hits)
    except Exception as exc:
        record['nvblox_diag_error'] = str(exc)

    pr = record.get('probe') or {}
    record['summary'] = {'trials': [{k: t.get(k) for k in ('lag_ms', 'integrated', 'stamp_advanced',
                                     'voxels_grew', 'ack_collision_time_advanced')}
                                    for t in (pr.get('trials') or [])],
                         'elapsed_s': record['elapsed_s'], 'error': record.get('error')}
    (a.output / 'result.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    print(json.dumps(record['summary'], indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
