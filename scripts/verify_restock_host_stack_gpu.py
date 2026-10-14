#!/usr/bin/env python3
"""Bounded, guarded GPU verification of the restock host planning stack + mapper.

ONE guarded job (flock /tmp/aile-gpu.lock + protected-Blender refusal + >=8 GiB free
VRAM + >=12 GiB RAM + domain-88 isolation, CPU preflight, 10-minute cap, exact-PID
teardown): launch the task-isolated stack (launch_restock_host_stack.sh -> MoveIt
move_group + the Isaac cuMotion servers + the shelfbot object_segmenter + the co-launched
nvblox mapper, all under the task-local runtime closure on domain 88, never the real
rig containers; the NVIDIA RobotSegmenter self-filter is delegated to the Isaac side), sample
GPU memory to its peak, run stack_probe.py (6 nodes present with no load failure, endpoints
advertised, a compute_ik planning round-trip against the refined CAD config, a synthetic
masked-depth frame fused into a NON-EMPTY nvblox ESDF, and a cuMotion plan that returns a
trajectory against it), then
tear down by exact PID group and report the measured VRAM so firstmate can judge
whether the stack fits beside the Isaac renderer (~6.6 GB) and the protected whispr
(~2.2 GB) on the 12 GB card.

Guard via the sibling run_evidence_mapper_gpu module (canonical guard if importable).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_evidence_mapper_gpu as reg  # noqa: E402

RENDERER_MB = 6600  # Isaac renderer footprint firstmate asked us to judge fit against


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--plugin-env', required=True)
    p.add_argument('--nvidia-runtime', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cap-seconds', type=float, default=580.0)  # under the 10-minute cap
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

    record = {'note': 'task-isolated planning stack + mapper; VRAM measurement; not commissioning',
              'commissioned': False, 'config': a.config}
    started = time.monotonic()
    try:
        with reg.guard(nvsmi, a.heavy_threshold_mb) as decision:
            record['guard'] = {'free_mb': getattr(decision, 'free_mb', None),
                               'message': getattr(decision, 'message', '')}
            free0, used0, total = reg._gpu_free_mib(nvsmi)
            record['gpu'] = {'total_mb': total, 'used_before_mb': used0, 'free_before_mb': free0}

            # ---- CPU preflight (dependency + static coverage check) ----
            chk = subprocess.run(['bash', '-lc', env_prefix +
                                  f'exec python3 "{a.repo}/scripts/run_restock_host.py" --config "{a.config}" --check'],
                                 capture_output=True, text=True, timeout=180)
            record['preflight'] = {'rc': chk.returncode, 'tail': chk.stdout[-1500:]}
            if chk.returncode != 0:
                raise SystemExit('CPU preflight failed; not launching the stack')

            # ---- launch the stack ----
            log = open(a.output / 'stack.log', 'wb')
            launch = (f'export RESTOCK_REPO="{a.repo}" RESTOCK_PLUGIN_ENV="{a.plugin_env}" '
                      f'RESTOCK_HOST_CONFIG="{a.config}" RESTOCK_NVIDIA_RUNTIME="{runtime}"; '
                      f'exec bash "{a.repo}/scripts/restock_runtime/launch_restock_host_stack.sh"')
            child = subprocess.Popen(['bash', '-lc', launch], stdout=log, stderr=subprocess.STDOUT,
                                     preexec_fn=os.setsid)
            record['stack_pid'] = child.pid

            # ---- sample VRAM to peak in the background ----
            samples, stop = [], threading.Event()

            def sampler():
                while not stop.is_set():
                    try:
                        free, used, _ = reg._gpu_free_mib(nvsmi)
                        samples.append((time.monotonic() - started, free, used))
                    except Exception:
                        pass
                    stop.wait(2.0)
            th = threading.Thread(target=sampler, daemon=True)
            th.start()

            try:
                probe = subprocess.run(['bash', '-lc', env_prefix +
                                        f'exec python3 "{a.repo}/scripts/restock_runtime/stack_probe.py" '
                                        f'--config "{a.config}" --output "{a.output}" '
                                        f'--discovery-timeout {a.discovery_timeout}'],
                                       capture_output=True, text=True, timeout=a.cap_seconds)
                record['probe_rc'] = probe.returncode
                record['probe_stderr_tail'] = probe.stderr[-1500:]
                pj = a.output / 'stack_probe.json'
                record['probe'] = json.loads(pj.read_text()) if pj.exists() else None
            finally:
                stop.set(); th.join(timeout=5)
                try:
                    os.killpg(os.getpgid(child.pid), signal.SIGINT); time.sleep(5)
                    if child.poll() is None:
                        os.killpg(os.getpgid(child.pid), signal.SIGTERM); time.sleep(4)
                    if child.poll() is None:
                        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                log.close()
            time.sleep(3)
            free_after = reg._gpu_free_mib(nvsmi)[0]
            record['gpu']['free_after_teardown_mb'] = free_after
            record['child_returncode'] = child.poll()

            if samples:
                min_free = min(s[1] for s in samples)
                max_used = max(s[2] for s in samples)
                record['gpu']['min_free_during_mb'] = min_free
                record['gpu']['max_used_during_mb'] = max_used
                record['gpu']['stack_vram_mb'] = free0 - min_free  # stack footprint (renderer not running)
                record['gpu']['samples'] = samples
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    record['elapsed_s'] = round(time.monotonic() - started, 1)

    # ---- verdict + fit analysis ----
    pr = record.get('probe') or {}
    g = record.get('gpu', {})
    stack_vram = g.get('stack_vram_mb')
    total = g.get('total_mb', 12288)
    whispr = g.get('used_before_mb', 0)
    free_before = g.get('free_before_mb', 0)
    free_after = g.get('free_after_teardown_mb', 0)
    fit = None
    if stack_vram is not None:
        headroom = total - whispr - RENDERER_MB - stack_vram
        fit = {'stack_vram_mb': stack_vram, 'whispr_mb': whispr, 'renderer_mb': RENDERER_MB,
               'total_mb': total, 'headroom_mb': headroom, 'fits_beside_renderer': headroom >= 0}
    present = set((pr.get('nodes') or {}).get('present') or [])
    # The host no longer runs the NVIDIA RobotSegmenter self-filter: its incompatible reload
    # subscription aborted the whole container, and the self mask is now delegated to the
    # Isaac side (robot-masked depth on camera/masked_depth). The six remaining nodes are the
    # full required server set, so all of them must be present with no load failure.
    core = {'move_group', 'cumotion_planner', 'static_planning_scene_server',
            'object_attachment', 'nvblox_node', 'object_segmenter'}
    me = pr.get('mapper_esdf') or {}
    lat = pr.get('latencies') or {}
    ack = pr.get('incorporation_ack') or {}
    record['verdict'] = {
        'incorporation_ack_became_ready': ack.get('became_ready'),
        'incorporation_ack_collision_time': ack.get('collision_time'),
        'incorporation_ack_fresh': ack.get('collision_time_fresh'),
        'incorporation_ack_matches_published_stamp': ack.get('collision_time_matches_published_stamp'),
        'incorporation_ack_advanced': ack.get('collision_time_advanced'),
        'incorporation_ack_fires_with_right_stamp': ack.get('fires_after_integration_with_right_stamp'),
        'first_esdf_ready_s': lat.get('first_esdf_ready_s'),
        'first_cumotion_reply_s': lat.get('first_cumotion_reply_s'),
        'cold_first_job_s': lat.get('cold_first_job_s'),
        'watchdog_s': lat.get('watchdog_s'),
        'within_watchdog': lat.get('within_watchdog'),
        'core_planner_nodes_present': core <= present,
        'all_nodes_present': (pr.get('nodes') or {}).get('all_present'),
        'nodes_present': sorted(present),
        'nodes_absent': (pr.get('nodes') or {}).get('absent'),
        'stack_endpoints_advertised': not (pr.get('endpoints') or {}).get('missing_excluding_external_controller'),
        'external_controller_absent_expected': (pr.get('endpoints') or {}).get('external_controller_absent'),
        'planning_round_trip_compute_ik': (pr.get('compute_ik') or {}).get('round_trip'),
        'compute_ik_error_code': (pr.get('compute_ik') or {}).get('error_code'),
        'cumotion_plan': pr.get('cumotion_plan'),
        'cumotion_trajectory_points': (pr.get('cumotion_plan') or {}).get('points'),
        'mapper_esdf_round_trip': me.get('round_trip'),
        'mapper_esdf_success': me.get('success'),
        'mapper_esdf_grid_dims': me.get('grid_dims'),
        'mapper_esdf_observed_before': me.get('observed_before'),
        'mapper_esdf_observed_after': me.get('observed_after'),
        'mapper_esdf_non_empty': me.get('non_empty'),
        'vram_fit': fit,
        'gpu_reclaimed': (free_after >= free_before - 200) if free_before else None,
    }
    v = record['verdict']
    record['PASS'] = bool(v['core_planner_nodes_present'] and v['all_nodes_present']
                          and v['stack_endpoints_advertised']
                          and v['planning_round_trip_compute_ik']
                          and v['mapper_esdf_non_empty']
                          and (v['cumotion_trajectory_points'] or 0) > 0
                          and v['incorporation_ack_fires_with_right_stamp']
                          and stack_vram is not None)
    (a.output / 'result.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    print(json.dumps({'PASS': record['PASS'], 'verdict': {k: v[k] for k in v if k != 'cumotion_plan'},
                      'cumotion_plan': v['cumotion_plan'], 'elapsed_s': record['elapsed_s'],
                      'error': record.get('error')}, indent=2, default=str))
    return 0 if record['PASS'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
