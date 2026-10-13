#!/usr/bin/env python3
"""Bounded, guarded GPU verification of the restock_mapper_evidence producer.

Holds the exclusive GPU guard (flock /tmp/aile-gpu.lock + protected-Blender refusal
+ >=8 GiB free VRAM + >=12 GiB RAM + domain-88 isolation), launches the navigation
nvblox mapper as restock_mapper_evidence::EvidenceProducerNode via
launch_evidence_mapper.py --execute, and MEASURES construction: the node appears,
the three incorporation services and the evidence_snapshot topic are advertised,
and no GXF-closure/CUDA/constructor error occurs. Teardown is by exact PID group.

This is a LAUNCH/CONSTRUCTION check, never commissioning. Publishing a first
slice bound to an incorporation identity additionally requires the integration
sensor+handshake driver (a NitrosImage depth publisher + camera_info + camera TF
into map + /clock, plus EmptyGeneration -> OpenAdmission -> RegisterAcquisition);
that is sensor/integration-owned and NOT driven here.

The canonical guard (twin.restock.guard.gpu_slot) is used when importable;
otherwise an inline equivalent enforces the same refusals.
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
        from twin.restock.guard import gpu_slot  # canonical guard, if on PYTHONPATH
    except Exception:
        with _inline_guard(nvsmi, heavy_threshold_mb) as d:
            yield d
        return
    with gpu_slot(heavy_threshold_mb=heavy_threshold_mb) as d:
        yield d


PROBE = r'''
import json, sys
import rclpy
from rclpy.node import Node
rclpy.init()
n = Node("evidence_probe")
import time; t=time.time()
nodes, services, topics = set(), {}, {}
while time.time()-t < 12:
    for nm, ns in n.get_node_names_and_namespaces():
        nodes.add(ns.rstrip("/")+"/"+nm)
    services.update(dict(n.get_service_names_and_types()))
    topics.update(dict(n.get_topic_names_and_types()))
    time.sleep(1)
n.destroy_node(); rclpy.shutdown()
print(json.dumps({"nodes":sorted(nodes),"services":sorted(services),"topics":sorted(topics)}))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--plugin-env', required=True)
    p.add_argument('--nvidia-runtime', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--generation', default='mission_a')
    p.add_argument('--camera-count', type=int, default=1)
    p.add_argument('--transport-height', type=float, default=1.9)
    p.add_argument('--sensors', nargs='+', default=['front=front_optical'])
    p.add_argument('--window', type=float, default=70.0)
    p.add_argument('--heavy-threshold-mb', type=int, default=2500)
    a = p.parse_args()
    if a.window > 300:
        raise SystemExit('window exceeds the 5-minute GPU cap')
    a.output.mkdir(parents=True, exist_ok=True)
    runtime = Path(a.nvidia_runtime)
    lib = runtime / 'usr/lib/x86_64-linux-gnu'
    nvsmi = str(runtime / 'usr/bin/nvidia-smi')
    ns = '/restock_sim/navigation/nvblox_node'
    want_services = [ns + s for s in ('/empty_generation', '/open_admission', '/register_acquisition')]
    want_topic = ns + '/evidence_snapshot'

    os.environ['PATH'] = str(runtime / 'usr/bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = str(lib) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ.update(ROS_DOMAIN_ID='88', ROS_LOCALHOST_ONLY='1', ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')

    sensors = ' '.join(a.sensors)
    child_sh = (f'set -e; export RESTOCK_WITH_GPU_USERSPACE=1; source "{a.plugin_env}"; '
                f'export LD_LIBRARY_PATH="{lib}:$LD_LIBRARY_PATH"; export PATH="{runtime}/usr/bin:$PATH"; '
                f'export PYTHONPATH="{a.repo}/scripts:$PYTHONPATH"; '
                f'export ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST; '
                f'exec python3 "{a.repo}/scripts/launch_evidence_mapper.py" --execute '
                f'--generation {a.generation} --camera-count {a.camera_count} '
                f'--transport-height {a.transport_height} --sensors {sensors}')

    record = {'want_services': want_services, 'want_topic': want_topic, 'commissioned': False,
              'note': 'EvidenceProducerNode construction/advertisement check; no sensor/handshake driver'}
    try:
        with guard(nvsmi, a.heavy_threshold_mb) as decision:
            record['guard'] = {'ok': getattr(decision, 'ok', True),
                               'free_mb': getattr(decision, 'free_mb', None),
                               'message': getattr(decision, 'message', '')}
            record['gpu_free_before'] = _gpu_free_mib(nvsmi)[0]
            log = open(a.output / 'mapper.log', 'wb')
            child = subprocess.Popen(['bash', '-lc', child_sh], stdout=log, stderr=subprocess.STDOUT,
                                     preexec_fn=os.setsid)
            record['pid'] = child.pid
            started = time.monotonic()
            probe = None
            while time.monotonic() - started < a.window:
                if child.poll() is not None:
                    record['child_exited_early'] = child.returncode
                    break
                time.sleep(5)
                if time.monotonic() - started > 25 and probe is None:
                    r = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True, timeout=40)
                    probe = r.stdout.strip()
                    record['probe_raw'] = probe[-4000:]
                    record['probe_err'] = r.stderr[-1500:]
            record['gpu_free_after_run'] = _gpu_free_mib(nvsmi)[0]
            try:
                os.killpg(os.getpgid(child.pid), signal.SIGINT); time.sleep(4)
                if child.poll() is None:
                    os.killpg(os.getpgid(child.pid), signal.SIGTERM); time.sleep(3)
                if child.poll() is None:
                    os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            log.close()
            record['gpu_free_after_teardown'] = _gpu_free_mib(nvsmi)[0]
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'

    mlog = (a.output / 'mapper.log').read_text(errors='replace') if (a.output / 'mapper.log').exists() else ''
    discovered = {}
    if record.get('probe_raw'):
        try:
            discovered = json.loads(record['probe_raw'])
        except json.JSONDecodeError:
            pass
    nodes = discovered.get('nodes', [])
    services = discovered.get('services', [])
    topics = discovered.get('topics', [])
    record['measured'] = {
        'node_present': any(ns.split('/')[-1] in nd for nd in nodes),
        'services_advertised': {s: (s in services) for s in want_services},
        'all_evidence_services_advertised': all(s in services for s in want_services),
        'evidence_topic_advertised': want_topic in topics,
        'gxf_helpers_loaded': 'libgxf_isaac_gxf_helpers.so' in mlog,
        'gxf_closure_error': ("gxf_isaac_gxf_helpers' not found" in mlog) or ('Component constructor threw' in mlog),
        'real_cuda_error': any(s in mlog for s in ('cudaError', 'no CUDA-capable', 'Driver/library version', 'CUDA error')),
        'reached_tick_loop': 'ros/tick' in mlog or 'NvbloxNode::tick' in mlog or 'subscribeToTopics' in mlog,
        'log_tail': mlog[-3500:],
    }
    (a.output / 'result.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    print(json.dumps({k: v for k, v in record['measured'].items() if k != 'log_tail'}, indent=2))


if __name__ == '__main__':
    main()
