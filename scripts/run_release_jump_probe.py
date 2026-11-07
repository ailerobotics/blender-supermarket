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


def _read_bind(stage, tick, phase, t0):
    """One residual-bind sample dict (see release_jump_probe.classify_bind for the schema). Reads the
    carton world pose+velocity, the gripper pad (TCP) world position, and the carton's live USD scene-
    graph state after release: RigidBody kinematicEnabled, any carton-subtree CollisionAPI enabled, and
    every stage joint whose body0/body1 targets the carton (+ its jointEnabled). Contacts come from the
    bench's own PhysX contact reporting (probe_contacts), which is armed on carton+arm+chassis."""
    from pxr import Usd, UsdPhysics
    pos, _orient = stage.carton.get_world_pose()
    try:
        vel = stage.carton.get_linear_velocity()
    except Exception:
        vel = (0.0, 0.0, 0.0)
    pad_t = stage._pad_world().ExtractTranslation()
    usd = stage.stage
    carton_prim = usd.GetPrimAtPath(stage.carton_path)
    # kinematicEnabled
    kin = None
    try:
        a = carton_prim.GetAttribute('physics:kinematicEnabled')
        kin = bool(a.Get()) if (a and a.IsValid() and a.HasAuthoredValue()) else (
            bool(UsdPhysics.RigidBodyAPI(carton_prim).GetKinematicEnabledAttr().Get())
            if carton_prim.HasAPI(UsdPhysics.RigidBodyAPI) else None)
    except Exception:
        kin = None
    # any carton-subtree collider enabled
    coll = None
    try:
        states = []
        for p in Usd.PrimRange(carton_prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                v = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
                states.append(bool(v) if v is not None else True)
        coll = (any(states) if states else None)
    except Exception:
        coll = None
    # every stage joint still targeting the carton
    joint_refs = []
    try:
        cp = stage.carton_path
        for p in usd.Traverse():
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            tgts = []
            for rel in (j.GetBody0Rel(), j.GetBody1Rel()):
                if rel:
                    tgts += [str(t) for t in rel.GetTargets()]
            if any(cp in t for t in tgts):
                en = p.GetAttribute('physics:jointEnabled')
                enabled = bool(en.Get()) if (en and en.IsValid() and en.Get() is not None) else True
                joint_refs.append({'path': str(p.GetPath()), 'enabled': enabled})
    except Exception:
        pass
    contacts = []
    try:
        contacts = sorted({h.get('prim') for h in stage.probe_contacts() if h.get('prim')})
    except Exception:
        pass
    return {'phase': phase, 'tick': int(tick), 't_s': round(time.monotonic() - t0, 3),
            'carton_pos': [float(pos[0]), float(pos[1]), float(pos[2])],
            'carton_vel': [float(vel[0]), float(vel[1]), float(vel[2])],
            'pad_pos': [float(pad_t[0]), float(pad_t[1]), float(pad_t[2])],
            'kinematic_enabled': kin, 'collision_enabled': coll,
            'joint_refs': joint_refs, 'contacts': contacts}


def _drive_to(stage, start, goal, *, steps, hold=False):
    """Drive the arm from ``start`` toward ``goal`` over ``steps`` ticks via the real ArmCommandGate
    (a slow position-tracked move when hold=False; a still hold at ``goal`` when hold=True), yielding
    (tick, target) each tick AFTER stepping physics so the caller can sample post-step state."""
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    for i in range(steps):
        frac = 1.0 if hold else (i + 1) / steps
        target = start + frac * (goal - start)
        stage.gate.apply(target, np.zeros(6), stage.gate.indices)
        stage.world.step(render=True)
        yield i, target


def run_residual_bind(stage, args, summary):
    """sim-place-release-residual-bind-v1 driver: raise the WELDED carton clear of the source, then
    sample the bind EVERY tick across pre_release (still) -> real _release_fallback() -> post_release
    (still, does it fall or hover?) -> a short further LIFT (does it track the arm?)."""
    from twin.tools import release_jump_probe as _probe
    grasp = np.asarray(stage.grasp_joints, float)
    lift = np.asarray(stage.lift_joints, float)
    # A) weld at grasp, then raise the welded carton to the lift pose so it is AIRBORNE (clear of the
    #    source) before the release -- so a freed carton visibly falls and the fall/hover test is clean.
    stage.reset_to_grasp(carry.REF_SOFT)
    for _ in _drive_to(stage, grasp, lift, steps=args.raise_steps):
        pass
    for _ in _drive_to(stage, lift, lift, steps=args.settle_steps, hold=True):
        pass
    samples = []
    t0 = time.monotonic()
    tick = 0
    # B) pre_release: arm still at the lift pose, carton still welded
    for _, _tgt in _drive_to(stage, lift, lift, steps=args.pre_release_steps, hold=True):
        samples.append(_read_bind(stage, tick, 'pre_release', t0)); tick += 1
    # C) the REAL mission release
    print('RESIDUAL_BIND detaching (contact._release_fallback) now', flush=True)
    stage.contact._release_fallback()
    # D) post_release_still: arm HELD at the lift pose -- a free carton falls, a bound one hovers
    for _, _tgt in _drive_to(stage, lift, lift, steps=args.post_still_steps, hold=True):
        samples.append(_read_bind(stage, tick, 'post_release_still', t0)); tick += 1
    # E) a short further LIFT (continue the grasp->lift direction) -- does the carton track the arm?
    lift_goal = lift + float(args.lift_frac) * (lift - grasp)
    for _, _tgt in _drive_to(stage, lift, lift_goal, steps=args.lift_steps):
        samples.append(_read_bind(stage, tick, 'lift', t0)); tick += 1
    ev = _probe.classify_bind(samples)
    print(_probe.format_bind(ev), flush=True)
    summary.update(state='complete', reason='', residual_bind=ev, bind_samples=samples)
    _persist(args.output / 'residual-bind.json', summary)
    return 0


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
            _rig = _design = None
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
            if args.residual_bind:
                return run_residual_bind(stage, args, summary)
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
            # fm380 STAGE-SCOPED SENSING measurement: after the welded (full-camera) baseline, DESTROY
            # the collect-only station cameras' render products the way the collect->couple transition
            # would, step render a few seconds, and measure how much VRAM Isaac actually RETURNS (it may
            # pool and return ~0). Needs --full-cameras (so there are station render products to free).
            if args.destroy_sensing:
                if _rig is None:
                    raise ValueError('--destroy-sensing requires --full-cameras (needs station cameras)')
                from twin.restock.native_cameras import destroy_camera_render_product
                from twin.restock.sensing_sources import _stage_scoped_teardown_camera_ids
                base_mib = probe.vram_used_mib()
                ids = _stage_scoped_teardown_camera_ids(_design)
                for cid in ids:
                    cam = _rig['cameras'].get(cid)
                    if cam is not None:
                        destroy_camera_render_product(cam, assembly.stage)
                for i in range(args.post_steps):
                    stage.gate.apply(reset_pose, np.zeros(6), stage.gate.indices)
                    stage.world.step(render=True)
                    if i % args.post_sample_every == 0 or i == args.post_steps - 1:
                        trace.add('post_destroy', time.monotonic() - t0, probe.vram_used_mib())
                after_mib = probe.vram_used_mib()
                returned = base_mib - after_mib
                print('STAGE_SCOPED_FREED baseline=%d after_destroy=%d returned=%d ids=%s'
                      % (base_mib, after_mib, returned, ','.join(ids) or 'none'), flush=True)
                summary['stage_scoped'] = dict(baseline_mib=base_mib, after_destroy_mib=after_mib,
                                               returned_mib=returned, torn_down=list(ids))
                summary.update(state='complete', reason='', samples=trace.samples)
                _persist(args.output / 'release-jump.json', summary)
                return 0
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
    p.add_argument('--residual-bind', action='store_true',
                   help='sim-place-release-residual-bind-v1: raise the welded carton clear of the source, '
                        'call the REAL contact._release_fallback(), and sample every tick across '
                        'pre_release/post_release_still/lift to read which release bind survives (carton '
                        'pose+velocity, kinematicEnabled, carton-targeting joints, collision-enabled, '
                        'contacts). Writes residual-bind.json. Pass a runtime-config with release_detach '
                        'on (or env RESTOCK_RELEASE_DETACH_ROBUST=1) to test the robust ordered detach.')
    p.add_argument('--raise-steps', type=int, default=90, help='steps to raise the welded carton grasp->lift')
    p.add_argument('--settle-steps', type=int, default=30, help='steps to settle at the lift pose before sampling')
    p.add_argument('--pre-release-steps', type=int, default=10, help='still ticks sampled BEFORE the release')
    p.add_argument('--post-still-steps', type=int, default=30, help='still ticks sampled AFTER the release (fall vs hover)')
    p.add_argument('--lift-steps', type=int, default=60, help='ticks of the post-release further lift (slow)')
    p.add_argument('--lift-frac', type=float, default=0.3,
                   help='further-lift goal = lift + lift_frac*(lift-grasp); ~0.3 ~= a 60 mm pad rise')
    p.add_argument('--destroy-sensing', action='store_true',
                   help='fm380: after the welded baseline, DESTROY the collect-only station cameras '
                        '(source+shelf; keeps cart) and measure VRAM returned (needs --full-cameras)')
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
