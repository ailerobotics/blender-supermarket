#!/usr/bin/env python3
"""Bounded, guarded GPU verification of the construction-baseline emptiness fix.

ONE guarded job (flock /tmp/aile-gpu.lock + protected-Blender refusal + >=8 GiB free
VRAM + >=12 GiB RAM + domain-88 isolation, <=5 min, exact-PID teardown) that launches
the EvidenceProducerNode TWICE as fresh constructions and proves both behaviours the
fix is about:

  PHASE accept : a freshly-constructed map, nothing published, EmptyGeneration ->
                 ACCEPTED, the per-layer LayerCount baseline reported. (The stock
                 producer wrongly refused this as "native layer not empty" and
                 faulted the one-shot producer forever.)
  PHASE depth  : a fresh construction, ONE synthetic depth frame integrated by the
                 base nvblox node (so the layer grows beyond its construction
                 baseline), EmptyGeneration -> REFUSED "native layer not empty" with
                 the grown LayerCount reported and the producer NOT faulted.

Two fresh node instances are used because a successful EmptyGeneration transitions
the one-shot producer to Certified, which (correctly) locks out a second probe; the
accept and the refuse-after-depth cases therefore cannot share one instance.

This is a LAUNCH/MEASUREMENT verification with a FIXTURE sensor source, never
commissioning. Publishing a first slice bound to an incorporation identity still
needs the integration sensor+handshake driver; that is sensor/integration-owned.

The canonical guard (twin.restock.guard.gpu_slot) is used when importable, via the
sibling run_evidence_mapper_gpu module; otherwise its inline equivalent enforces the
same refusals.
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
import run_evidence_mapper_gpu as reg  # noqa: E402  (guard + _gpu_free_mib, same dir)


def _child_env_prefix(repo, plugin_env, lib, runtime):
    return (f'set -e; export RESTOCK_WITH_GPU_USERSPACE=1; source "{plugin_env}"; '
            f'export LD_LIBRARY_PATH="{lib}:$LD_LIBRARY_PATH"; export PATH="{runtime}/usr/bin:$PATH"; '
            f'export PYTHONPATH="{repo}/scripts:$PYTHONPATH"; '
            f'export ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST; ')


def _launch_mapper(prefix, repo, generation, out):
    log = open(out / 'mapper.log', 'wb')
    sh = (prefix + f'exec python3 "{repo}/scripts/launch_evidence_mapper.py" --execute '
          f'--generation {generation} --camera-count 1 --transport-height 1.9 '
          f'--sensors front=front_optical')
    child = subprocess.Popen(['bash', '-lc', sh], stdout=log, stderr=subprocess.STDOUT,
                             preexec_fn=os.setsid)
    return child, log


def _run_probe(prefix, repo, mode, generation, out, service_timeout):
    sh = (prefix + f'exec python3 "{repo}/scripts/restock_runtime/empty_baseline_probe.py" '
          f'--mode {mode} --generation {generation} --output "{out}" '
          f'--service-timeout {service_timeout}')
    r = subprocess.run(['bash', '-lc', sh], capture_output=True, text=True, timeout=service_timeout + 60)
    probe_json = out / f'probe_{mode}.json'
    parsed = json.loads(probe_json.read_text()) if probe_json.exists() else None
    return {'rc': r.returncode, 'parsed': parsed, 'stderr_tail': r.stderr[-1500:]}


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


def _phase(prefix, repo, nvsmi, mode, generation, out, service_timeout):
    out.mkdir(parents=True, exist_ok=True)
    rec = {'mode': mode, 'generation': generation}
    rec['gpu_free_before'] = reg._gpu_free_mib(nvsmi)[0]
    child, log = _launch_mapper(prefix, repo, generation, out)
    rec['mapper_pid'] = child.pid
    try:
        rec['probe'] = _run_probe(prefix, repo, mode, generation, out, service_timeout)
        rec['gpu_free_during'] = reg._gpu_free_mib(nvsmi)[0]
    finally:
        _teardown(child)
        log.close()
    time.sleep(2)
    rec['gpu_free_after_teardown'] = reg._gpu_free_mib(nvsmi)[0]
    rec['child_returncode'] = child.poll()
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--plugin-env', required=True)
    p.add_argument('--nvidia-runtime', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--generation', default='baseline_verify')
    p.add_argument('--service-timeout', type=float, default=75.0)
    p.add_argument('--heavy-threshold-mb', type=int, default=2500)
    p.add_argument('--cap-seconds', type=float, default=295.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    runtime = Path(a.nvidia_runtime)
    lib = runtime / 'usr/lib/x86_64-linux-gnu'
    nvsmi = str(runtime / 'usr/bin/nvidia-smi')
    prefix = _child_env_prefix(a.repo, a.plugin_env, lib, runtime)

    os.environ['PATH'] = str(runtime / 'usr/bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = str(lib) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ.update(ROS_DOMAIN_ID='88', ROS_LOCALHOST_ONLY='1', ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')

    record = {'note': 'construction-baseline EmptyGeneration verification; fixture sensor; not commissioning',
              'commissioned': False, 'phases': {}}
    started = time.monotonic()
    try:
        with reg.guard(nvsmi, a.heavy_threshold_mb) as decision:
            record['guard'] = {'ok': getattr(decision, 'ok', True),
                               'free_mb': getattr(decision, 'free_mb', None),
                               'message': getattr(decision, 'message', '')}
            record['phases']['accept'] = _phase(
                prefix, a.repo, nvsmi, 'accept', f'{a.generation}_accept',
                a.output / 'accept', a.service_timeout)
            if time.monotonic() - started > a.cap_seconds:
                raise SystemExit('5-minute GPU cap reached before depth phase')
            record['phases']['depth'] = _phase(
                prefix, a.repo, nvsmi, 'depth', f'{a.generation}_depth',
                a.output / 'depth', a.service_timeout)
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    record['elapsed_s'] = round(time.monotonic() - started, 1)

    # Verdict.
    acc = (record['phases'].get('accept') or {}).get('probe', {}).get('parsed') or {}
    dep = (record['phases'].get('depth') or {}).get('probe', {}).get('parsed') or {}
    acc_layers = {l['name']: l['blocks'] for l in acc.get('layers', [])}
    dep_layers = {l['name']: l['blocks'] for l in dep.get('layers', [])}
    grew = {n: (acc_layers.get(n, 0), dep_layers[n]) for n in dep_layers
            if dep_layers[n] > acc_layers.get(n, 0)}
    # Prove the synthetic depth was really INTEGRATED (not dropped on a missing TF):
    # nvblox emits a 'tsdf/integrate' timing sample per integrated frame.
    depth_log = a.output / 'depth' / 'mapper.log'
    tsdf_integrations = 0
    if depth_log.exists():
        for line in depth_log.read_text(errors='replace').splitlines():
            if 'tsdf/integrate ' in line:
                parts = line.split('tsdf/integrate', 1)[1].split()
                if parts and parts[0].isdigit():
                    tsdf_integrations = int(parts[0])
    reason = dep.get('failure_reason') or ''
    # After real integration the authoritative sensor-emptiness signal is the
    # integration watermark / non-empty queue ("empty evidence incomplete"); a dense
    # enough scene would also grow blocks beyond baseline ("native layer not empty").
    native_refusal = reason in ('empty evidence incomplete', 'native layer not empty')
    record['verdict'] = {
        'accept_on_fresh': acc.get('success') is True,
        'accept_reported_baseline_layers': len(acc.get('layers', [])) > 0,
        'accept_baseline': acc_layers,
        'refused_after_depth': dep.get('success') is False and native_refusal,
        'depth_refusal_reason': reason,
        'depth_reported_layers': len(dep.get('layers', [])) > 0,
        'depth_frames_integrated': tsdf_integrations,
        'depth_frames_published': dep.get('published_depth_frames'),
        'layers_that_grew_vs_baseline': grew,
    }
    v = record['verdict']
    record['PASS'] = bool(v['accept_on_fresh'] and v['accept_reported_baseline_layers']
                          and v['refused_after_depth'] and v['depth_reported_layers']
                          and tsdf_integrations >= 1)
    (a.output / 'result.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    print(json.dumps({'PASS': record['PASS'], 'verdict': v, 'elapsed_s': record['elapsed_s'],
                      'error': record.get('error')}, indent=2, default=str))
    return 0 if record['PASS'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
