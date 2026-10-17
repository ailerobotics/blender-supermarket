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


def run_warm_session(run_once, *, warm_attempts, warm_reset, write_attempts=None):
    """Run ``run_once`` up to ``warm_attempts`` times in one live session; return (mission, summary).

    Pure control flow (no Isaac): ``run_once()`` returns ``(mission, summary)`` for one attempt and
    is expected to have persisted that attempt's mission.json already. After a FAILED attempt with
    budget remaining, the scene is re-seated for the next attempt via ``warm_reset(attempt, now)``
    (a callable or None) which MUST re-validate the commissioned baseline (lifetime + session
    binding) and raise to refuse rather than bypass it. ``warm_attempts <= 1`` is exactly one call
    to ``run_once`` (byte-identical to a single run). On a stop, the reason is recorded on the final
    summary (``warm_note`` when no provider is wired, ``warm_stop`` when the provider refused).

    ``write_attempts`` (optional) is called with the per-attempt record list when more than one
    attempt was configured, so the launcher can persist warm-attempts.json.
    """
    warm_attempts = max(1, int(warm_attempts or 1))
    attempts = []
    mission = summary = None
    for attempt in range(1, warm_attempts + 1):
        mission, summary = run_once()
        attempts.append({'attempt': attempt, 'state': summary['state'],
                         'reason': summary.get('reason')})
        if summary['state'] == 'complete' or attempt >= warm_attempts:
            break
        if warm_reset is None:
            summary['warm_note'] = ('--warm-attempts > 1 but no config.warm_reset provider; '
                                    'ran a single attempt (no in-session re-seat available)')
            break
        try:
            # Re-seat to the commissioned initial state + re-validate the baseline; the provider
            # raises (guard refused / baseline expired / session unbound) to stop -- never bypassed.
            warm_reset(attempt=attempt, now=attempt)
        except Exception as exc:
            summary['warm_stop'] = f'warm re-run refused (reset/guard): {exc}'
            break
    if warm_attempts > 1:
        summary['warm_attempts'] = attempts
        if write_attempts is not None:
            write_attempts(attempts)
    return mission, summary


def _apply_render_preset(config, name):
    """Opt into a sensing render preset for this run -- the SAME ``--render-preset`` that
    commissioning takes. Sets the NON-decode render-product scale (mapper/nav/arm). The QR-decode
    support cameras (source, shelf, cart) are always rendered full-res by the mission (isaac_bundle
    pins every decode support location): a native sub-1.0 decode render does not decode (run-72;
    the native px/module floor in config.render_presets), so ``fast`` never demotes a decode camera.
    Decode baselines are commissioned at 1.0 too, so ``commissioning._check_match`` stays satisfied
    at 1.0 for those cameras."""
    from twin.restock_camera.config import preset_render_scale
    config['render_scale'] = preset_render_scale(name)
    return config


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


def _maybe_smart_verify(args, scenario, summary):
    """Opt-in (``--smart-verify``) ADDITIONAL planogram/realogram compliance report.

    Pure CPU post-processing that runs AFTER the mission and its mission.json are
    written. It consumes the just-finished run's artifacts (mission.json + the
    committed inventory DB + the planogram) and writes ``smart-verify.json`` (full
    verdict) and ``reset-jobs.json`` (residual jobs in the sim-smart-job-feed-v1
    ``--job-from`` shape) into the run output. It NEVER changes the mission's
    stock-commit gate or ``summary``; wrapped so it can never fail the run. Absent
    ``--smart-verify`` this function is not called, so a mission run is byte-identical.

    ``summary`` already carries ``events`` (set just before mission.json is written),
    so the verify evidence is read from it directly rather than re-reading the file.
    """
    from twin.restock import smart_verify as sv
    try:
        evidence = sv.mission_verify_evidence(summary)
        if evidence is None:
            summary['smart_verify'] = {'ran': False,
                                       'reason': 'mission verify did not succeed; nothing to check'}
            return
        slot_id = evidence.get('location') or scenario['job']['shelf_id']
        plan_path = Path(__file__).resolve().parent / 'planogram.json'
        plan = sv.load_planogram(plan_path)
        # Bookkeeping basis: the committed DB counts at the slot after the verified
        # transfer -- the fullest cheap realogram, labelled honestly as a belief.
        stock = sv.load_stock(args.output / 'inventory.sqlite')
        report = sv.report_from_mission(
            mission=summary, plan=plan, slot_id=slot_id, stock=stock,
            source_bin=scenario['job'].get('source_bin'),
            quantity_policy=getattr(args, 'smart_verify_policy', 'at_least'),
            mission_json_path=str(args.output / 'mission.json'))
        out = report.to_dict()
        if getattr(args, 'smart_verify_storevision', False):
            from twin.restock import smart_verify_storevision as svx
            out['storevision_crosscheck'] = svx.cross_check(report)
        (args.output / 'smart-verify.json').write_text(
            json.dumps(out, indent=2, allow_nan=False) + '\n')
        (args.output / 'reset-jobs.json').write_text(
            json.dumps([j.to_job_feed() for j in report.residual_jobs], indent=2,
                       allow_nan=False) + '\n')
        summary['smart_verify'] = {'ran': True, 'compliant': out['compliant'],
                                   'evidence_basis': out['evidence_basis'],
                                   'residual_jobs': len(out['residual_jobs'])}
    except Exception as exc:
        summary['smart_verify'] = {'ran': False, 'error': str(exc)}


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
    if getattr(args, 'render_preset', None):
        _apply_render_preset(config, args.render_preset)
    readiness=check_runtime_inputs(config)
    if not readiness['files_ready']:
        raise ValueError('runtime inputs unresolved: '+', '.join(readiness['unresolved_inputs']))
    configured_scenario=json.loads(Path(config['scenario']).read_text())
    if configured_scenario!=scenario:
        raise ValueError('runtime configuration and mission scenario differ')
    # Opt-in tray/slot fit preflight (default off => fit_verdict stays None, byte-identical).
    # Computed once here on CPU, off the simulation hot path; a 'wont_fit' refuses the mission
    # in Mission.start() before any reservation/boot. 'ok'/'unknown' are advisory (logged).
    fit_verdict=None
    if getattr(args, 'fit_preflight', False):
        from twin.restock.fit_preflight import preflight_from_config
        fit_verdict=preflight_from_config(scenario, config)
        print('FIT_PREFLIGHT '+json.dumps(fit_verdict.event() if fit_verdict else
                                          {'fit': 'disabled', 'reason': 'no fit_preflight config'}),
              flush=True)
    # Preserve the launch shell before SimulationApp mutates Python/native-library
    # paths; the host ROS interpreter receives its own explicitly sourced overlay.
    config['_launch_environment']=dict(os.environ)
    # New output directory only: a restart must reconcile existing physical/job state.
    # Reject any existing path before loading the simulator.
    args.output.mkdir(parents=True, exist_ok=False)
    db = Inventory(args.output/'inventory.sqlite')
    for row in scenario['initial_inventory']:
        db.seed(row['location'], row['sku'], row['quantity'])
    # A proven won't-fit refuses the mission here, BEFORE paying the Isaac boot / GPU slot --
    # the job is never reserved and no robot is sent. 'ok'/'unknown' fall through to the run.
    if fit_verdict is not None and fit_verdict.status == 'wont_fit':
        db.close()
        refused = {'state': 'failed', 'reason': 'unsupported: ' + fit_verdict.message,
                   'physics_verified': False, 'fit_preflight': fit_verdict.event(),
                   'note': 'Refused by --fit-preflight before boot; robot was not sent.'}
        (args.output / 'mission.json').write_text(json.dumps(refused, indent=2, allow_nan=False) + '\n')
        print('FIT_PREFLIGHT_REFUSED ' + refused['reason'], flush=True)
        return 1
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
                def _make_mission():
                    return Mission(db, Job(**scenario['job']), bundle.adapter,
                                   stages=config.get('stages'), start_at=config.get('start_at'),
                                   fit_preflight=fit_verdict)
                def run_once():
                    """One full mission attempt against the live bundle; returns (mission, summary)
                    and persists mission.json. Unchanged from the single-run behaviour -- the warm
                    loop below runs it once by default (--warm-attempts 1), composing cleanly with
                    the downstream --start-at shortcut (which selects WHERE a run begins; warm
                    selects HOW MANY runs per boot).

                    Opt-in: runtime-config ``"bt": true`` drives the SAME Mission through the
                    behaviour-tree orchestrator (reactive event handling + live view in the run
                    dir); default off keeps the production state machine below byte-identical."""
                    if config.get('bt'):
                        from twin.restock.bt.integration import run_once_bt
                        return run_once_bt(db, scenario, bundle, config, args.output,
                                           mission_factory=_make_mission,
                                           is_running=app.is_running)
                    m = _make_mission()
                    m.start()
                    for _ in range(60*20*60):  # bounded simulation time; shell also bounds wall time
                        if m.terminal:
                            break
                        if not app.is_running():
                            raise RuntimeError('simulation app stopped')
                        m.tick(1/60)
                    if not m.terminal:
                        m.fail('simulation step budget exceeded')
                    s = {'state': m.state, 'reason': m.reason, 'physics_verified': False,
                         'note': 'Mission success alone does not verify video, contacts or autonomy.'}
                    if fit_verdict is not None:
                        s['fit_preflight'] = fit_verdict.event()
                    # Flag a start-phase shortcut run so mission.json cannot be read as a full,
                    # grasp-performed cycle: the skipped prefix was synthesized, not actuated.
                    if m.shortcut is not None:
                        s['shortcut'] = m.shortcut
                        s['note'] += (' START-PHASE SHORTCUT: stages '
                                      + ', '.join(m.shortcut['skipped'])
                                      + ' were synthesized (robot pose, braked base, carton '
                                        'attach), not performed; see events *_shortcut.')
                    # Persist the per-stage summary now, before the finally runs app.close():
                    # Isaac's SimulationApp.close() can terminate the process hard and skip the
                    # outer finally (which re-writes mission.json with events as a fallback).
                    try:
                        early = dict(s, events=db.events(scenario['job']['job_id']))
                        (args.output/'mission.json').write_text(json.dumps(early, indent=2, allow_nan=False)+'\n')
                    except Exception:
                        pass
                    return m, s

                # Warm-session retry: re-run the mission inside the SAME live Isaac session instead
                # of re-paying the ~30 s boot + separate ~43 s commissioning per attempt. Default
                # --warm-attempts 1 == today's single run (byte-identical). The in-session re-seat +
                # baseline re-validation is delegated to config['warm_reset'] (bundle.warm_reset), the
                # same module:function contract as the lane factories; it raises to refuse, never
                # bypassing the lifetime/session guard. Absent provider => a single attempt.
                def _write_attempts(records):
                    try:
                        (args.output/'warm-attempts.json').write_text(
                            json.dumps(records, indent=2, allow_nan=False) + '\n')
                    except Exception:
                        pass
                mission, summary = run_warm_session(
                    run_once, warm_attempts=getattr(args, 'warm_attempts', 1),
                    warm_reset=getattr(bundle, 'warm_reset', None), write_attempts=_write_attempts)
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
        # Opt-in (--smart-verify) ADDITIONAL planogram/realogram compliance report.
        # CPU-only, never changes the stock-commit gate, never raises. Absent the
        # flag this is skipped and mission.json is byte-identical.
        if getattr(args, 'smart_verify', False):
            _maybe_smart_verify(args, scenario, summary)
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
    parser.add_argument('--render-preset', choices=('full','fast'),
                        help='opt into a sensing render preset (full|fast). fast renders the '
                             'collect source + sensing at 0.6 scale (~2.8x fewer pixels) to speed '
                             'iteration; use the SAME preset for commissioning so the baseline matches.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--warm-attempts', type=int, default=1,
                        help='dev iteration: re-run the mission up to N times inside the SAME live '
                             'Isaac session (no re-boot / re-commission per attempt), re-seating the '
                             'scene between failed attempts via config.warm_reset and re-validating the '
                             'commissioned baseline. 1 (default) = a single run, byte-identical.')
    parser.add_argument('--heavy-threshold-mb', type=int, choices=(1500,2500), default=1500)
    parser.add_argument('--host-stack', action='store_true',
                        help='co-launch the MoveIt/cuMotion planner stack + nvblox mapper (domain 88) '
                             'inside the GPU-lock hold before Isaac')
    parser.add_argument('--plugin-env', type=Path, help='host runtime closure env.sh for --host-stack')
    parser.add_argument('--nvidia-runtime', type=Path, help='task-local NVIDIA runtime root for --host-stack')
    parser.add_argument('--host-stack-warmup', type=float, default=180.0,
                        help='seconds to let the planner stack + mapper advertise before booting Isaac')
    parser.add_argument('--smart-verify', action='store_true',
                        help='after a mission, also write an ADDITIONAL planogram/realogram '
                             'compliance report (smart-verify.json) + residual reset jobs '
                             '(reset-jobs.json). CPU-only, never gates the mission; absent this '
                             'flag the run is byte-identical.')
    parser.add_argument('--smart-verify-policy', choices=('at_least', 'exact', 'identity'),
                        default='at_least',
                        help='how --smart-verify judges slot quantity vs the planogram (default: at_least)')
    parser.add_argument('--smart-verify-storevision', action='store_true',
                        help='have --smart-verify also cross-check through store-vision '
                             'planoreset.verify (advisory; degrades if store-vision is absent)')
    parser.add_argument('--transport-probe', type=int, default=0,
                        help='focused masked-depth transport probe: drive the arm bridge for N '
                             'frames at the staging pose instead of running the mission')
    parser.add_argument('--fit-preflight', action='store_true',
                        help='opt-in: refuse a job whose handling unit will not fit the destination '
                             'slot (measured free space in runtime-config["fit_preflight"]) BEFORE '
                             'sending the robot, instead of failing opaquely at place. Default off; '
                             'absent => byte-identical run. See twin.restock.fit_preflight.')
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
