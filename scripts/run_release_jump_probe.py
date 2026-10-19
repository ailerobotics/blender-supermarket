#!/usr/bin/env python3
"""Guarded release-jump VRAM probe (fm378).

Boots a SimulationApp ONCE under the GPU lock, assembles the mission's physics scene and WELDS the
carton at the tool via the mission's own SIM_ATTACH_FALLBACK (reusing the arm_carry_bench boot+weld
glue verbatim), then drives the exact scene-graph change the mission's RELEASE phase makes:
``contact._release_fallback()`` removes the carton FixedJoint and re-enables the carton + tool-side
colliders, so the carton becomes a FREE DYNAMIC body. VRAM is sampled at high cadence across pre-weld,
welded-baseline, the detach tick and the post-detach settle, and the verdict reports whether the
hypothesised ~+2.68 GB jump reproduces AT the detach.

render is ON (RTX active) -- the full mission's jump is an RTX render-pass reallocation
('MemoryManager-allocate' / 'Indirect diffuse texture' / 'DLSS Output'), so the acceleration structure
must exist for the detach to rebuild it. (arm_carry_bench runs render OFF; this probe does not.)

GPU etiquette mirrors run_arm_carry_bench: NON-BLOCKING lock via guard.gpu_slot (which honours an
outer flock through AILE_GPU_LOCK_INHERITED) or --no-gpu-lock -> gpu_guards_no_flock; VRAM guard never
bypassed; --heavy-threshold-mb 2500 accepts the always-on whispr (~2.2 GB); <=15 min; hard-refuse
beside the live Blender session; NO walkthrough recording. Never --force.
"""
import argparse
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

# Import the sibling arm_carry_bench runner (same scripts/ dir, not a package) to REUSE its proven
# IsaacCarryStage boot+weld glue, its no-flock guard and the mission-representative poses/gains.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_arm_carry_bench as carry  # noqa: E402
from twin.restock.guard import available_ram_mb, gpu_slot  # noqa: E402
from twin.tools import release_jump_probe as probe  # noqa: E402


def _persist(path, summary):
    """Write the summary json and fsync, so it survives Isaac's fastShutdown hard-exit."""
    import os as _os
    data = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    with open(path, 'w') as f:
        f.write(data)
        f.flush()
        _os.fsync(f.fileno())


def _hold_and_sample(stage, trace, phase, reset_pose, *, steps, sample_every, t0):
    """Step physics with RTX render ON, holding the arm at ``reset_pose`` via the real ArmCommandGate,
    sampling VRAM every ``sample_every`` steps (and always on the last step)."""
    zeros = np.zeros(6)
    for i in range(steps):
        stage.gate.apply(reset_pose, zeros, stage.gate.indices)
        stage.world.step(render=True)
        if i % sample_every == 0 or i == steps - 1:
            trace.add(phase, time.monotonic() - t0, probe.vram_used_mib())


def run(args):
    config = json.loads(args.runtime_config.read_text())
    scenario = json.loads(Path(config['scenario']).read_text())
    # No out-of-Isaac transport relay needed (we never drive the host trajectory_host here).
    md = dict(config.get('masked_depth') or {})
    md.pop('runtime_env', None)
    config['masked_depth'] = md
    arm_cfg = config.get('armbench', {})
    lift_joints = args.lift_joints if args.lift_joints is not None else arm_cfg.get('lift_joints', carry.DEFAULT_LIFT)
    grasp_joints = args.grasp_joints if args.grasp_joints is not None else arm_cfg.get('grasp_joints', carry.DEFAULT_GRASP)
    if len(lift_joints) != 6 or len(grasp_joints) != 6:
        raise ValueError('lift_joints and grasp_joints must each be six arm joints (ARM_JOINTS order)')
    carton_dims = tuple(scenario.get('handling_unit', {}).get('dimensions', (0.6, 0.4, 0.16)))

    args.output.mkdir(parents=True, exist_ok=True)
    finished = threading.Event()

    def interrupted(signum, frame):
        raise RuntimeError(f'release probe interrupted by signal {signum}')
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

    trace = probe.ReleaseTrace()
    summary = {'state': 'failed', 'reason': 'probe did not start',
               'grasp_joints': list(grasp_joints), 'lift_joints': list(lift_joints),
               'carton_dims': list(carton_dims),
               'note': ('fm378 release-jump probe: boots the mission physics scene, welds the carton '
                        'via SIM_ATTACH_FALLBACK, then calls the REAL contact._release_fallback() '
                        '(FixedJoint removed + carton/tool colliders re-enabled -> free dynamic body) '
                        'with RTX render ON, watching VRAM for the mission\'s post-release jump. '
                        'CAVEAT: the bench welds/releases at the GRASP pose (backstock), not over the '
                        'cart, and the physics scene has the main RTX viewport but not the mission\'s '
                        'walkthrough+sensing cameras, so the ABSOLUTE jump may be smaller than the '
                        'mission\'s; the scene-graph transition (attached->free dynamic) is identical.')}
    app = None
    slot = (carry.gpu_guards_no_flock(heavy_threshold_mb=args.heavy_threshold_mb) if args.no_gpu_lock
            else gpu_slot(heavy_threshold_mb=args.heavy_threshold_mb))
    try:
        with slot:
            threading.Thread(target=watch_ram, daemon=True).start()
            from isaacsim import SimulationApp
            app = SimulationApp({'headless': True, 'width': args.width, 'height': args.height})
            from twin.restock.physics_setup import assemble
            assembly = assemble(app=app, config=config, scenario=scenario, output=args.output / 'assembly')
            # fm379 fuller bench: author the mission's FULL camera set (4 station sensing cams + the
            # walkthrough_chase) so the render-product count/sizes match the mission -- the decisive
            # scales-with-render-products test. Reuses _real_rig verbatim, so it ALSO honours the
            # RESTOCK_DROP_UNUSED_SENSING / RESTOCK_NO_WALKTHROUGH_RENDER / RESTOCK_WALKTHROUGH_* knobs.
            cams_note = 'viewport-only'
            if args.full_cameras:
                from twin.restock.sensing_sources import _real_rig
                from twin.restock_camera.config import camera_config as _camera_config
                if not args.manifest:
                    raise ValueError('--full-cameras requires --manifest (mission environment.json)')
                _design = _camera_config(scenario, json.loads(args.manifest.read_text()))
                _rig = _real_rig(assembly=assembly, design=_design, render_scale=1.0,
                                 camera_render_scales=None)
                cams_note = 'full mission set (%d station cams%s)' % (
                    len(_rig['cameras']), '' if _rig['recording_camera'] is None else ' + walkthrough_chase')
            summary['cameras'] = cams_note
            stage = carry.IsaacCarryStage(assembly, lift_joints, grasp_joints, render=True,
                                          carton_dims=carton_dims)
            reset_pose = stage.lift_joints if stage.start_at_lift else stage.grasp_joints
            t0 = time.monotonic()
            # fm379 mitigation 1 (prewarm): BEFORE the measured sequence, run one weld+release cycle so
            # the RTX acceleration-structure pool is grown while VRAM has headroom; if the real release
            # then shows no jump, pre-warm is the fix. Gated on RESTOCK_PREWARM_ATTACH or --prewarm.
            if args.prewarm or os.environ.get('RESTOCK_PREWARM_ATTACH'):
                print('RELEASE_PROBE prewarm: weld+release cycle to pre-size the pool', flush=True)
                stage.reset_to_grasp(carry.REF_SOFT)
                for _ in range(30):
                    stage.gate.apply(reset_pose, np.zeros(6), stage.gate.indices)
                    stage.world.step(render=True)
                stage.contact._release_fallback()
                for _ in range(30):
                    stage.gate.apply(reset_pose, np.zeros(6), stage.gate.indices)
                    stage.world.step(render=True)
                summary['prewarmed'] = True

            # 1) pre-weld: RTX renders the scene (no carton weld yet)
            _hold_and_sample(stage, trace, 'pre_weld', reset_pose,
                             steps=args.pre_steps, sample_every=args.sample_every, t0=t0)
            # 2) ATTACH: the mission's own weld (via the proven reset_to_grasp)
            stage.reset_to_grasp(carry.REF_SOFT)
            print('RELEASE_PROBE welded; holding for baseline', flush=True)
            # 3) welded baseline
            _hold_and_sample(stage, trace, 'welded', reset_pose,
                             steps=args.welded_steps, sample_every=args.sample_every, t0=t0)
            # 4) DETACH: the exact mission release scene-graph change
            print('RELEASE_PROBE detaching (contact._release_fallback) now', flush=True)
            stage.contact._release_fallback()
            trace.add('detach', time.monotonic() - t0, probe.vram_used_mib())
            # 5) post-detach settle, high-cadence VRAM sampling -> watch for the jump
            _hold_and_sample(stage, trace, 'post_detach', reset_pose,
                             steps=args.post_steps, sample_every=args.post_sample_every, t0=t0)

            v = probe.verdict(trace, threshold_mib=args.threshold_mib)
            print(probe.format_verdict(v), flush=True)
            summary.update(state='complete', reason='', verdict=v, samples=trace.samples)
            # Persist BEFORE app.close(): --app/fastShutdown hard-exits on close and skipped this
            # write in the first probe (lost the json+samples). Write+fsync inside the slot.
            _persist(args.output / 'release-jump.json', summary)
    except Exception as exc:
        summary['reason'] = str(exc)
        summary['samples'] = trace.samples
        (args.output / 'probe-error.txt').write_text(
            ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        _persist(args.output / 'release-jump.json', summary)
    finally:
        finished.set()
        if app is not None:
            try:
                app.close()
            except Exception:
                pass
    return 0 if summary['state'] == 'complete' else 1


def main():
    p = argparse.ArgumentParser(description='fm378 release-jump VRAM probe (guarded GPU)')
    p.add_argument('--runtime-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=None,
                   help='mission environment.json (required with --full-cameras)')
    p.add_argument('--full-cameras', action='store_true',
                   help='author the mission full camera set (4 station cams + walkthrough_chase) via _real_rig')
    p.add_argument('--prewarm', action='store_true',
                   help='run one weld+release cycle before the measured sequence (pre-size the RTX pool)')
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--height', type=int, default=270)
    p.add_argument('--pre-steps', type=int, default=120, help='render steps before the weld')
    p.add_argument('--welded-steps', type=int, default=180, help='render steps holding the welded carton')
    p.add_argument('--post-steps', type=int, default=420, help='render steps after the detach')
    p.add_argument('--sample-every', type=int, default=20, help='VRAM sample cadence (steps) pre-weld/welded')
    p.add_argument('--post-sample-every', type=int, default=3, help='VRAM sample cadence (steps) post-detach (high)')
    p.add_argument('--threshold-mib', type=int, default=1000, help='delta >= this at post-detach => jump detected')
    p.add_argument('--lift-joints', type=float, nargs=6, default=None)
    p.add_argument('--grasp-joints', type=float, nargs=6, default=None)
    p.add_argument('--heavy-threshold-mb', type=int, choices=(1500, 2500), default=1500)
    p.add_argument('--no-gpu-lock', action='store_true',
                   help='skip taking the flock (an outer flock -w /tmp/aile-gpu.lock already holds it)')
    args = p.parse_args()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
