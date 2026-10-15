#!/usr/bin/env python3
"""Offline contract validation or explicit guarded Isaac runtime (RUNBOOK §11)."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback

from twin.restock.guard import available_ram_mb, gpu_slot
from twin.restock.inventory import Inventory, Job
from twin.restock.mission import Mission
from twin.restock.scenario import validate


def _launch_host_stack(*, repo, plugin_env, nvidia_runtime, host_config, logpath):
    """Co-launch the task-isolated MoveIt/cuMotion planner stack + nvblox mapper on
    domain 88, inside the caller's single GPU-lock hold (the launch recipe does NOT
    re-flock). A sanitized env keeps the Isaac interpreter's PYTHONHOME/PYTHONPATH out
    of the host ROS python; the recipe sources its own runtime closure. Returns the
    child process group leader + the open log handle."""
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP')
           and not k.startswith('CONDA')}
    env.update(RESTOCK_REPO=str(repo), RESTOCK_PLUGIN_ENV=str(plugin_env),
               RESTOCK_HOST_CONFIG=str(host_config), RESTOCK_NVIDIA_RUNTIME=str(nvidia_runtime),
               RESTOCK_WITH_GPU_USERSPACE='1',
               ROS_DOMAIN_ID='88', ROS_LOCALHOST_ONLY='1', ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')
    script = str(Path(repo) / 'scripts/restock_runtime/launch_restock_host_stack.sh')
    log = open(logpath, 'wb')
    child = subprocess.Popen(['bash', '-lc', f'exec bash "{script}"'],
                             stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env)
    return child, log


def _teardown_pg(child):
    if child is None:
        return
    try:
        # Graceful: SIGTERM the launcher so the ROS launch shuts its nodes down and they FLUSH
        # host-stack.log (a hard SIGKILL truncated run-73's nvblox evidence mid-buffer). Poll up to
        # 15 s for the whole planner/mapper stack to exit, then hard-kill only if it hangs.
        os.killpg(os.getpgid(child.pid), signal.SIGTERM)
        deadline = time.monotonic() + 15.
        while time.monotonic() < deadline and child.poll() is None:
            time.sleep(0.5)
        if child.poll() is None:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def check(args):
    manifest = json.loads(args.manifest.read_text())
    scenario = json.loads(args.scenario.read_text())
    report = validate(manifest, scenario)
    report['input_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for name, path in (('manifest',args.manifest),('scenario',args.scenario))}
    root = Path(__file__).resolve().parent
    required = ['twin/navigation/core.py', 'twin/navigation/nav2.py',
                'twin/mobile_runtime.py', 'twin/restock_manipulation/controller.py']
    report['missing_modules'] = [p for p in required if not (root/p).is_file()]
    report['ok'] = report['ok'] and not report['missing_modules']
    return report, manifest, scenario


def _run_transport_probe(bundle, scenario, args, app):
    """Drive the arm masked-depth bridge for N frames at the collect staging pose (no mission).

    The robot is already composed at the staging pose; this configures the source station and
    ticks the port (rendering the arm camera) + the bridge, so the host object_segmenter/nvblox
    can prove the masked-depth transport end to end. Pass/fail is asserted host-side.
    """
    from dataclasses import asdict
    adapter = bundle.adapter
    port = adapter.port
    manip = adapter.manip
    runtime = getattr(getattr(manip, 'controller', None), '_runtime', None)
    bridge = getattr(runtime, 'sensing_bridge', None)
    if bridge is None:
        raise RuntimeError('transport probe requires the arm masked_depth bridge '
                           '(enable masked_depth in the runtime-config)')
    manip.start('collect', asdict(Job(**scenario['job'])))  # source station + bridge.target_location
    port.current_action = 'collect'                         # force the sensing render each tick
    frames = int(args.transport_probe)
    # Warm up the camera render + annotators (depth/instance-ID populate a few frames in).
    for _ in range(20):
        if not app.is_running():
            break
        port.advance(1 / 60)
    # Publish continuously for a HOLD window so the host stays alive and keeps integrating while
    # the host-side verdict (masked_depth rate + the nvblox ESDF query) runs against it; stop
    # once at least `frames` were published AND the hold elapsed.
    hold_until = time.monotonic() + 70.0
    published = 0
    while app.is_running():
        packet = port.advance(1 / 60)
        if bridge.publish(packet.simulation_time) is not None:
            published += 1
        if published >= frames and time.monotonic() >= hold_until:
            break
        if time.monotonic() >= hold_until + 20.0:           # safety cap
            break
    return {'state': 'complete' if published >= frames else 'failed',
            'reason': f'transport probe published {published}/{frames} masked-depth frames (held for host verdict)',
            'physics_verified': False, 'frames': published}


def run(args, manifest, scenario):
    if not args.factory or ':' not in args.factory:
        raise ValueError('runtime requires --factory module:function; no mock fallback')
    if not args.output:
        raise ValueError('runtime requires an explicit scratch --output directory')
    if not args.runtime_config:
        raise ValueError('runtime requires --runtime-config with exact sensor/host/CAD inputs')
    from preflight_mobile_restock import check as check_runtime_inputs
    config=json.loads(args.runtime_config.read_text())
    readiness=check_runtime_inputs(config)
    if not readiness['files_ready']:
        raise ValueError('runtime inputs unresolved: '+', '.join(readiness['unresolved_inputs']))
    configured_scenario=json.loads(Path(config['scenario']).read_text())
    if configured_scenario!=scenario:
        raise ValueError('runtime configuration and mission scenario differ')
    # Preserve the launch shell before SimulationApp mutates Python/native-library
    # paths; the host ROS interpreter receives its own explicitly sourced overlay.
    config['_launch_environment']=dict(os.environ)
    # New output directory only: a restart must reconcile existing physical/job state.
    # Reject any existing path before loading the simulator.
    args.output.mkdir(parents=True, exist_ok=False)
    db = Inventory(args.output/'inventory.sqlite')
    for row in scenario['initial_inventory']:
        db.seed(row['location'], row['sku'], row['quantity'])
    mission = app = bundle = None
    finished = threading.Event()
    def interrupted(signum, frame):
        raise RuntimeError(f'runtime interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def watch_ram():
        while not finished.wait(1):
            try:
                safe = available_ram_mb() >= 4096
            except Exception:
                safe = False
            if not safe:
                os.kill(os.getpid(), signal.SIGTERM)
                return
    summary = {'state': 'failed', 'reason': 'runtime did not start', 'physics_verified': False}
    stack = stack_log = None
    try:
        with gpu_slot(heavy_threshold_mb=args.heavy_threshold_mb):
            watcher = threading.Thread(target=watch_ram, daemon=True)
            watcher.start()
            try:
                # Co-launch the host planner stack + mapper (domain 88) inside this single
                # GPU-lock hold, before Isaac, so the manip can connect and the mapper can
                # build the ESDF world from the mission's live depth. The recipe does not
                # re-flock; teardown is by exact PID group. The 4 GiB RAM watchdog + the
                # wall ceiling still bound the whole run.
                if args.host_stack:
                    if not args.plugin_env or not args.nvidia_runtime:
                        raise ValueError('--host-stack requires --plugin-env and --nvidia-runtime')
                    repo = Path(__file__).resolve().parent.parent
                    stack, stack_log = _launch_host_stack(
                        repo=repo, plugin_env=args.plugin_env, nvidia_runtime=args.nvidia_runtime,
                        host_config=config['host_config'], logpath=args.output / 'host-stack.log')
                    summary['host_stack_pid'] = stack.pid
                    warm = time.monotonic()
                    while time.monotonic() - warm < args.host_stack_warmup:
                        if stack.poll() is not None:
                            raise RuntimeError(f'host stack exited during warmup rc={stack.returncode}')
                        time.sleep(3)
                    summary['host_stack_warmup_s'] = round(time.monotonic() - warm, 1)
                # NVIDIA requires runtime/plugin imports after SimulationApp startup.
                from isaacsim import SimulationApp
                app = SimulationApp({'headless': True, 'width': 960, 'height': 540})
                module, name = args.factory.split(':', 1)
                factory = getattr(importlib.import_module(module), name)
                # Factory authors stage, initializes sensors and returns an object with
                # adapter (RuntimeAdapter), close(), and recording metadata. It must not
                # create another app or issue commands outside the simulation domain.
                bundle = factory(app=app, manifest=manifest, scenario=scenario,
                                 config=config, output=args.output)
                if getattr(bundle.adapter, 'fixture_mode', True):
                    raise RuntimeError('production runtime refuses fixture adapters')
                if args.transport_probe:
                    # Focused native masked-depth transport probe (no mission/navigation): the
                    # robot is already at the collect staging pose; drive the arm masked-depth
                    # bridge for N frames so the host (object_segmenter -> nvblox) can prove the
                    # transport. Assertions are host-side (see run_transport_probe.sh).
                    summary = _run_transport_probe(bundle, scenario, args, app)
                    (args.output / 'mission.json').write_text(
                        json.dumps(dict(summary, events=db.events(scenario['job']['job_id'])),
                                   indent=2, allow_nan=False) + '\n')
                    return 0 if summary['state'] == 'complete' else 1
                mission = Mission(db, Job(**scenario['job']), bundle.adapter,
                                  stages=config.get('stages'))
                mission.start()
                for _ in range(60*20*60):  # bounded simulation time; shell also bounds wall time
                    if mission.terminal:
                        break
                    if not app.is_running():
                        raise RuntimeError('simulation app stopped')
                    mission.tick(1/60)
                if not mission.terminal:
                    mission.fail('simulation step budget exceeded')
                summary = {'state': mission.state, 'reason': mission.reason,
                           'physics_verified': False,
                           'note': 'Mission success alone does not verify video, contacts or autonomy.'}
                # Persist the per-stage summary now, before the finally runs app.close():
                # Isaac's SimulationApp.close() can terminate the process hard and skip the
                # outer finally (which re-writes mission.json with events as a fallback).
                try:
                    early = dict(summary, events=db.events(scenario['job']['job_id']))
                    (args.output/'mission.json').write_text(json.dumps(early, indent=2, allow_nan=False)+'\n')
                except Exception:
                    pass
            except Exception as exc:
                if mission is not None and not mission.terminal:
                    mission.fail(str(exc))
                # Persist the failure before app.close(): Isaac's SimulationApp.close()
                # can terminate the process hard, so the outer finally that writes
                # mission.json may never run and the traceback would otherwise be lost.
                try:
                    (args.output/'mission-error.txt').write_text(
                        ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                except Exception:
                    pass
                raise
            finally:
                # Cleanup stays inside GPU lock, including partial boot failures.
                try:
                    if bundle is not None:
                        try:
                            bundle.adapter.stop()
                        finally:
                            bundle.close()
                finally:
                    # Encode a persistent VFR H.264 walkthrough BEFORE app.close(): Isaac's
                    # SimulationApp.close() can hard-exit and skip the outer finally, so encoding there
                    # (or leaving it to an on-demand click) means a run's clip may not exist. frames.json
                    # is written incrementally so it is complete here. Every run (success OR failure)
                    # thus leaves <run>/walkthrough.mp4 for the watch page. Never fatal.
                    try:
                        frames_manifest = args.output / 'mission-frames' / 'frames.json'
                        mp4 = args.output / 'walkthrough.mp4'
                        ffmpeg = config.get('ffmpeg') if isinstance(config, dict) else None
                        if frames_manifest.is_file() and ffmpeg and not mp4.exists():
                            from twin.restock.recording import Recorder
                            recorded = Recorder.from_manifest(frames_manifest)
                            if recorded.frames:
                                info = recorded.encode(mp4, ffmpeg=ffmpeg)
                                summary['walkthrough_mp4'] = str(mp4)
                                summary['walkthrough_frames'] = info['frames']
                    except Exception as exc:
                        summary['walkthrough_error'] = str(exc)
                    if app is not None:
                        app.close()
    except Exception as exc:
        summary['state'] = 'failed'
        summary['reason'] = str(exc)
        raise
    finally:
        finished.set()
        _teardown_pg(stack)
        if stack_log is not None:
            try:
                stack_log.close()
            except Exception:
                pass
        summary['events'] = db.events(scenario['job']['job_id'])
        (args.output/'mission.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        db.close()
    return 0 if summary['state'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--scenario', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--factory', help='installed, reviewed simulator bundle module:function')
    parser.add_argument('--runtime-config',type=Path,help='exact host, CAD and camera runtime inputs')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--heavy-threshold-mb', type=int, choices=(1500,2500), default=1500)
    parser.add_argument('--host-stack', action='store_true',
                        help='co-launch the MoveIt/cuMotion planner stack + nvblox mapper (domain 88) '
                             'inside the GPU-lock hold before Isaac')
    parser.add_argument('--plugin-env', type=Path, help='host runtime closure env.sh for --host-stack')
    parser.add_argument('--nvidia-runtime', type=Path, help='task-local NVIDIA runtime root for --host-stack')
    parser.add_argument('--host-stack-warmup', type=float, default=180.0,
                        help='seconds to let the planner stack + mapper advertise before booting Isaac')
    parser.add_argument('--transport-probe', type=int, default=0,
                        help='focused masked-depth transport probe: drive the arm bridge for N '
                             'frames at the staging pose instead of running the mission')
    args = parser.parse_args()
    report, manifest, scenario = check(args)
    print(json.dumps(report, indent=2))
    if not report['ok']:
        return 2
    if args.execute:
        return run(args, manifest, scenario)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
