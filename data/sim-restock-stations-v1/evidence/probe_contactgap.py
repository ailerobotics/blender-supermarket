#!/usr/bin/env python3
"""Why the 20 mm surface gripper does not seal at the MISSION contact pose.

PR100's latch window (20 mm) + ~1 deg tilt were measured on a top approach that stops the
descent the instant it latches (~20 mm). The MISSION does a SIDE grasp pressing the pad to the
real tcp_surface_offset = 2.75 mm -- far CLOSER than the authored clearanceOffset (0.5*max_gap =
10 mm), which places the seal plane 10 mm in FRONT of the panel. This probe reproduces the mission
choreography against the REAL author_native_gripper + NativePanelContact (+ the new probe()
diagnostic) and answers, with numbers, whether the gripper seals when the object sits at the tight
2.75 mm contact gap, and whether the enable-timing (suction on DURING the descent vs only AFTER the
arm has reached contact) matters:

  mission_slow   : max_gap .020 clear .010 stroke .016, suction ON during descent, stop at 2.75 mm
  mission_direct : same authoring, snap the carton to the 2.75 mm gap FIRST, THEN enable suction
  tight_slow     : max_gap .006 clear .003 stroke .005 (seal plane matched to the tight contact),
                   suction ON during descent, stop at 2.75 mm      <- candidate fix
  tight_direct   : same tight authoring, snap to 2.75 mm first, then enable

Each recipe records the per-step probe() (gripper status, gripped-object count, measured raycast
gap, pad tilt) across the descent + hold, the gap at which the latch first forms (if any), and a
lift check. Same mission GPU guards. Carton top = world z 0.30. Diagnostic only; no mission, no video.
"""
import argparse, json, os, sys, time, traceback, math
from pathlib import Path

REPO = Path("/home/manishsaini/.treehouse/blender-supermarket-dc8579/1/blender-supermarket")
SCRIPTS = REPO / "scripts"
# recipe -> (max_gap_m, stroke_m, clearance_m, enable_during_descent)
RECIPES = {
    "mission_slow":   (.020, .016, .010, True),
    "mission_direct": (.020, .016, .010, False),
    "tight_slow":     (.006, .005, .003, True),
    "tight_direct":   (.006, .005, .003, False),
}
CARTON_H = 0.16              # the mission handling-unit thickness (grasped face)
CARTON_TOP = CARTON_H        # carton rests on the ground plane, top face at z = CARTON_H
PANEL = (0.26, 0.108)         # the real CAD suction-panel footprint
START_GAP = 0.030
CONTACT_GAP = 0.00275         # the mission tcp_surface_offset
STEP = 0.00005


def worker(recipe, output):
    output = Path(output)
    max_gap, stroke_limit, clearance, enable_during = RECIPES[recipe]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 320, 'height': 240,
                         'renderer': 'RayTracedLighting', 'anti_aliasing': 0, 'multi_gpu': False})
    result = {'recipe': recipe, 'status': 'failed', 'max_gap_mm': max_gap * 1000,
              'stroke_mm': stroke_limit * 1000, 'clearance_mm': clearance * 1000,
              'enable_during_descent': enable_during, 'contact_gap_mm': CONTACT_GAP * 1000, 'trace': []}
    try:
        import numpy as np, omni.usd, omni.timeline
        from pxr import UsdGeom, UsdLux, UsdPhysics, Gf, Sdf
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('isaacsim.robot.surface_gripper')
        from twin.restock_manipulation.cad_tool import CadTool, ARM_JOINTS, author_native_gripper, NativePanelContact
        pad_z = CARTON_TOP + START_GAP
        down = Gf.Quatf(0, 1, 0, 0)   # tool +Z points straight down onto the carton top

        world = World(physics_dt=1/60, rendering_dt=1/60, stage_units_in_meters=1.)
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, '/World')
        UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1000.)
        world.scene.add_default_ground_plane()
        # Carton = the mission handling unit (0.6 x 0.4 x 0.16), a DYNAMIC body resting on the
        # ground (top face at z = CARTON_H). We control the gap by moving the PAD, not the carton,
        # so no mid-sim kinematic toggle is needed.
        carton = UsdGeom.Cube.Define(stage, '/World/Carton'); carton.CreateSizeAttr(1.0)
        carton.AddTranslateOp().Set(Gf.Vec3d(0, 0, CARTON_H / 2))
        carton.AddScaleOp().Set(Gf.Vec3d(0.6, 0.4, CARTON_H))
        cp = carton.GetPrim(); UsdPhysics.CollisionAPI.Apply(cp); UsdPhysics.RigidBodyAPI.Apply(cp)
        UsdPhysics.MassAPI.Apply(cp).CreateMassAttr(0.45)
        UsdGeom.Xform.Define(stage, '/World/Tool')
        def frame(path):
            x = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, path).GetPrim())
            x.AddTranslateOp().Set(Gf.Vec3d(0, 0, pad_z)); x.AddOrientOp().Set(down); return x.GetPrim()
        def rigid(path, kinematic, collider=False):
            prim = frame(path); UsdPhysics.RigidBodyAPI.Apply(prim)
            prim.CreateAttribute('physics:kinematicEnabled', Sdf.ValueTypeNames.Bool, False).Set(kinematic)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)
            if collider:
                box = UsdGeom.Cube.Define(stage, path + '/panel'); box.CreateSizeAttr(1.0)
                xb = UsdGeom.Xformable(box.GetPrim())
                xb.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.01)); xb.AddScaleOp().Set(Gf.Vec3d(PANEL[0], PANEL[1], 0.02))
                UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            return prim
        carrier = '/World/Tool/carrier'; rigid(carrier, True)
        body_path = '/World/Tool/suction_cups'; rigid(body_path, False, collider=True)
        tool0_path = '/World/Tool/ur_tool0'; frame(tool0_path)
        f = UsdPhysics.FixedJoint.Define(stage, '/World/Tool/mount')
        f.CreateBody0Rel().SetTargets([Sdf.Path(carrier)]); f.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
        tool = CadTool(arm_joints=ARM_JOINTS, arm_limits=((-6., 6.),) * 6, panel_size=(PANEL[0], PANEL[1], .02558),
                       tool0_from_pad=np.eye(4), tool0_from_support=np.eye(4), tine_limits=((0., 0.),) * 6)
        gpath = '/World/Tool/RestockPanelGripper'
        author_native_gripper(stage, tool=tool, tool0_path=tool0_path, body_path=body_path,
                              gripper_path=gpath, max_gap=max_gap, clearance_offset=clearance, stroke_limit=stroke_limit)

        world.reset()
        carrier_prim = SingleRigidPrim(prim_path=carrier, name='carrier')
        panel = SingleRigidPrim(prim_path=body_path, name='panel')
        carton_prim = SingleRigidPrim(prim_path='/World/Carton', name='carton')
        contact = NativePanelContact(stage, gpath, diagnostic=False)   # we call probe() explicitly each step
        tl = omni.timeline.get_timeline_interface(); tl.play()
        for _ in range(30):
            world.step(render=False)

        def panel_gap_mm():
            return round((float(panel.get_world_pose()[0][2]) - CARTON_TOP) * 1000, 3)

        def record(step_i):
            info = contact.probe()
            info['step'] = step_i
            info['panel_gap_mm'] = panel_gap_mm()
            result['trace'].append(info)
            return info

        cpos, cor = carrier_prim.get_world_pose(); cpos = np.asarray(cpos, float)
        if not enable_during:
            # Snap straight to the mission contact gap (arm already at the contact pose), settle,
            # THEN enable suction -- the object sits at 2.75 mm before the gripper is ever commanded.
            cpos[2] = CARTON_TOP + CONTACT_GAP
            carrier_prim.set_world_pose(position=cpos.copy(), orientation=cor)
            for _ in range(30):
                world.step(render=False)
            contact.command(True)
        else:
            contact.command(True)
        # Descend with suction on (enable_during) until the seal LATCHES, then FREEZE the pad --
        # the mission's contact move is contact-terminated (the controller emits 'contact' and the
        # trajectory host holds the touch pose the instant read() reports the latch), so the arm
        # does not keep pressing past the latch. Confirm the latch PERSISTS for 30 steps (a stable
        # seal, not a one-step transient) before the lift.
        latched = False
        latch_hold = 0
        for i in range(1000):
            moving = enable_during and not latched and cpos[2] > CARTON_TOP + CONTACT_GAP
            if moving:
                cpos[2] = max(CARTON_TOP + CONTACT_GAP, cpos[2] - STEP)
                carrier_prim.set_world_pose(position=cpos.copy(), orientation=cor)
            world.step(render=False)
            held = contact.read()
            if i % 10 == 0 or i < 5 or (held and not latched):
                record(i)
            if held and not latched:
                latched = True
                result['latch_step'] = i
                result['gap_at_latch_mm'] = panel_gap_mm()
            if latched:
                latch_hold = latch_hold + 1 if held else 0
                if latch_hold >= 30:
                    break
            elif enable_during and cpos[2] <= CARTON_TOP + CONTACT_GAP + 1e-9 and i > 400:
                break
            elif not enable_during and i > 150:
                break
        result['latched'] = latched
        result['latch_persisted'] = latch_hold >= 30
        result['final_gap_mm'] = panel_gap_mm()
        result['final_status'] = str(contact.api.get_gripper_status(gpath))
        result['final_gripped'] = [str(x) for x in (contact.api.get_gripped_objects(gpath) or [])]
        # Lift check: raise the pad and see whether the carton (dynamic, resting on the ground)
        # follows -- the latch physically holds it against gravity.
        if latched:
            z0 = float(carton_prim.get_world_pose()[0][2])
            cpos, cor = carrier_prim.get_world_pose(); cpos = np.asarray(cpos, float)
            for _ in range(60):
                cpos[2] += 0.001; carrier_prim.set_world_pose(position=cpos.copy(), orientation=cor); world.step(render=False)
            result['lift'] = {'carton_dz_mm': round((float(carton_prim.get_world_pose()[0][2]) - z0) * 1000, 2),
                              'still_held': bool(contact.read())}
        result['status'] = 'passed'
        (output / f'{recipe}.json').write_text(json.dumps(result, indent=2) + '\n')
    except BaseException as exc:
        result['reason'] = str(exc); result['traceback'] = traceback.format_exc()
        (output / f'{recipe}.json').write_text(json.dumps(result, indent=2) + '\n'); raise
    finally:
        app.close()


def parent(args):
    sys.path.insert(0, str(SCRIPTS))
    from twin.restock.guard import gpu_slot
    from twin.restock.station import OwnedProcess
    from twin.restock.native_environment import ros_environment
    import run_restock_render_smoke as smoke
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    env = smoke.runtime_environment(args.nvidia_runtime, out)
    env = ros_environment(env, str(args.isaac_python)); env['PYTHONPATH'] = str(SCRIPTS)
    summary = {'recipes': {}}
    with gpu_slot(heavy_threshold_mb=2500) as decision:
        if decision.free_mb < 8192:
            raise RuntimeError('need >= 8 GiB free VRAM')
        for recipe in (list(RECIPES) if args.recipe == 'all' else [args.recipe]):
            log = out / f'{recipe}.log'
            cmd = [str(args.isaac_python), str(Path(__file__).resolve()), '--worker', '--recipe', recipe, '--output', str(out)]
            child = OwnedProcess(cmd, log, env=env); child.start(); started = time.monotonic()
            while child.alive:
                if time.monotonic() - started > 600:
                    child.close(); raise TimeoutError(f'{recipe} exceeded 600s')
                time.sleep(1.)
            child.close()
            rp = out / f'{recipe}.json'; res = json.loads(rp.read_text()) if rp.exists() else {'status': 'no-result'}
            summary['recipes'][recipe] = {k: res.get(k) for k in ('status', 'latched', 'latch_persisted',
                'gap_at_latch_mm', 'final_gap_mm', 'max_gap_mm', 'stroke_mm', 'clearance_mm', 'enable_during_descent',
                'final_status', 'final_gripped', 'lift', 'reason')} | {'exit': child.process.returncode}
            (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(f'[{recipe}] exit={child.process.returncode} latched={res.get("latched")} '
                  f'persisted={res.get("latch_persisted")} gap_latch={res.get("gap_at_latch_mm")} '
                  f'max_gap={res.get("max_gap_mm")} clear={res.get("clearance_mm")} '
                  f'enable_during={res.get("enable_during_descent")} final_status={res.get("final_status")} '
                  f'gripped={res.get("final_gripped")} lift={res.get("lift")}', flush=True)
    print('SWEEP DONE', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', action='store_true'); ap.add_argument('--recipe', default='all')
    ap.add_argument('--output', required=True)
    ap.add_argument('--isaac-python', type=Path); ap.add_argument('--nvidia-runtime', type=Path)
    a = ap.parse_args()
    worker(a.recipe, a.output) if a.worker else parent(a)


if __name__ == '__main__':
    main()
