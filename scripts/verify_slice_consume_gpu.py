#!/usr/bin/env python3
"""Bounded, guarded GPU verification that the landed consumer ACCEPTS the snapshot.

ONE guarded job (flock /tmp/aile-gpu.lock + protected-Blender refusal + >=8 GiB free
VRAM + >=12 GiB RAM + domain-88 isolation, <=5 min, exact-PID teardown): launch the
EvidenceProducerNode, drive the full native incorporation pipeline with a fixture
sensor source (EmptyGeneration -> OpenAdmission -> RegisterAcquisition -> synthetic
depth integration -> EvidenceSnapshot), then decode the snapshot with the reference
bridge and feed it to the LANDED navigation consumer
(twin.navigation.temporal.TemporalMap.accept_envelope), recording that it ACCEPTS the
slice (previously refused "invalid native distance slice" because processEsdf left the
slice header frame_id empty).

Fixture sensor source; a launch/measurement verification, never commissioning.
Guard via the sibling run_evidence_mapper_gpu module (canonical guard if importable,
else its inline equivalent with the same refusals).
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--plugin-env', required=True)
    p.add_argument('--nvidia-runtime', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--generation', default='slice_consume_verify')
    p.add_argument('--window', type=float, default=90.0)
    p.add_argument('--service-timeout', type=float, default=30.0)
    p.add_argument('--heavy-threshold-mb', type=int, default=2500)
    p.add_argument('--cap-seconds', type=float, default=290.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    runtime = Path(a.nvidia_runtime)
    lib = runtime / 'usr/lib/x86_64-linux-gnu'
    nvsmi = str(runtime / 'usr/bin/nvidia-smi')
    prefix = (f'set -e; export RESTOCK_WITH_GPU_USERSPACE=1; source "{a.plugin_env}"; '
              f'export LD_LIBRARY_PATH="{lib}:$LD_LIBRARY_PATH"; export PATH="{runtime}/usr/bin:$PATH"; '
              f'export PYTHONPATH="{a.repo}/scripts:$PYTHONPATH"; '
              f'export ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST; ')

    os.environ['PATH'] = str(runtime / 'usr/bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = str(lib) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ.update(ROS_DOMAIN_ID='88', ROS_LOCALHOST_ONLY='1', ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')

    record = {'note': 'full pipeline + landed-consumer acceptance; fixture sensor; not commissioning',
              'commissioned': False}
    started = time.monotonic()
    try:
        with reg.guard(nvsmi, a.heavy_threshold_mb) as decision:
            record['guard'] = {'free_mb': getattr(decision, 'free_mb', None),
                               'message': getattr(decision, 'message', '')}
            record['gpu_free_before'] = reg._gpu_free_mib(nvsmi)[0]
            mlog = open(a.output / 'mapper.log', 'wb')
            launch = (prefix + f'exec python3 "{a.repo}/scripts/launch_evidence_mapper.py" --execute '
                      f'--generation {a.generation} --camera-count 1 --transport-height 1.9 '
                      f'--sensors front=front_optical')
            child = subprocess.Popen(['bash', '-lc', launch], stdout=mlog, stderr=subprocess.STDOUT,
                                     preexec_fn=os.setsid)
            record['mapper_pid'] = child.pid
            try:
                probe_sh = (prefix + f'exec python3 "{a.repo}/scripts/restock_runtime/consume_snapshot_probe.py" '
                            f'--generation {a.generation} --output "{a.output}" '
                            f'--window {a.window} --service-timeout {a.service_timeout}')
                r = subprocess.run(['bash', '-lc', probe_sh], capture_output=True, text=True,
                                   timeout=a.window + 90)
                record['probe_rc'] = r.returncode
                record['probe_stderr_tail'] = r.stderr[-1500:]
                cj = a.output / 'consume_result.json'
                record['consume'] = json.loads(cj.read_text()) if cj.exists() else None
                record['gpu_free_during'] = reg._gpu_free_mib(nvsmi)[0]
            finally:
                try:
                    os.killpg(os.getpgid(child.pid), signal.SIGINT); time.sleep(4)
                    if child.poll() is None:
                        os.killpg(os.getpgid(child.pid), signal.SIGTERM); time.sleep(3)
                    if child.poll() is None:
                        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                mlog.close()
            time.sleep(2)
            record['gpu_free_after_teardown'] = reg._gpu_free_mib(nvsmi)[0]
            record['child_returncode'] = child.poll()
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    record['elapsed_s'] = round(time.monotonic() - started, 1)

    c = record.get('consume') or {}
    snap = c.get('snapshot') or {}
    record['verdict'] = {
        'snapshot_received': c.get('snapshot_received') is True,
        'slice_frame_id': snap.get('slice_frame_id'),
        'consumer_accepted': c.get('consumer_accepted') is True,
        'adopted_slice_frame': (c.get('adopted') or {}).get('slice_frame'),
        'esdf_revision': snap.get('esdf_revision'),
        'snapshot_index': snap.get('snapshot_index'),
    }
    v = record['verdict']
    record['PASS'] = bool(v['consumer_accepted'] and v['slice_frame_id'] == 'map'
                          and v['adopted_slice_frame'] == 'map' and v['snapshot_received'])
    (a.output / 'result.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    print(json.dumps({'PASS': record['PASS'], 'verdict': v, 'elapsed_s': record['elapsed_s'],
                      'error': record.get('error'),
                      'service_failures': (c.get('service_failures') or {})}, indent=2, default=str))
    return 0 if record['PASS'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
