#!/usr/bin/env python3
"""Locate the native surface-gripper seal gate: world-down? world-axis? or surface-relative?

Mission run-177 (a SIDE grasp) never seals. Earlier proven seals (contactgap / probe_tilt) were
ALL top-down. This drives the REAL author_native_gripper + NativePanelContact statically (pad held
GAP from the carton face, suction enabled) for:

  top_flat  : TOP grasp  (forward world -Z), carton flat      -> SANITY: must latch (replicates contactgap)
  side_flat : SIDE grasp (forward world -Y), carton flat      -> does a flush SIDE grasp seal at all?
  side_pad4 : SIDE, pad yaw 4, carton 0 (pad 4deg off face)   -> reproduces the mission misalignment
  side_both4: SIDE, pad yaw 4, carton yaw 4 (pad FLUSH)        -> surface-relative test at 4 deg
  side_both8: SIDE, pad yaw 8, carton yaw 8 (pad FLUSH)        -> surface-relative test at 8 deg

Verdict: top_flat latch + side_flat FAIL  => gate keys to world-down (side grasps never seal natively).
         side_flat latch + side_both* FAIL => gate world-keyed to a fixed surface axis.
         side_both* latch                  => gate is surface-relative (the 4.2deg is the observation).
Carton rests on the ground (no anchor); both carton and panel body are dynamic (the plugin only grips
dynamic actors). Same mission GPU guards.
"""
import argparse, json, sys, time, traceback, math
from pathlib import Path

REPO = Path("/home/manishsaini/.treehouse/blender-supermarket-dc8579/1/blender-supermarket")
SCRIPTS = REPO / "scripts"
# recipe -> (mode, pad_yaw_deg, carton_yaw_deg)
RECIPES = {
    "top_flat":  ("top", 0.0, 0.0),
    "side_flat": ("side", 0.0, 0.0),
    "side_pad4": ("side", 4.0, 0.0),
    "side_both4": ("side", 4.0, 4.0),
    "side_both8": ("side", 8.0, 8.0),
}
CARTON = 0.30            # cube edge (any face fits the 0.20 panel)
PANEL = (0.20, 0.20)
GAP = 0.003              # pad face 3 mm from the carton face (mission contact clearance)
MAXGAP = 0.020           # SIM_MAX_GRIP_DISTANCE


def worker(recipe, output):
    output = Path(output)
    mode, pad_deg, cart_deg = RECIPES[recipe]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 320, 'height': 240,
                         'renderer': 'RayTracedLighting', 'anti_aliasing': 0, 'multi_gpu': False})
    result = {'recipe': recipe, 'status': 'failed', 'mode': mode, 'pad_yaw_deg': pad_deg, 'carton_yaw_deg': cart_deg}
    try:
        import numpy as np, omni.usd, omni.timeline
        from pxr import UsdGeom, UsdLux, UsdPhysics, Gf, Sdf
        from isaacsim.core.api import World
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('isaacsim.robot.surface_gripper')
        from twin.restock_manipulation.cad_tool import CadTool, ARM_JOINTS, author_native_gripper, NativePanelContact

        def qz(deg):
            a = math.radians(deg)
            return Gf.Quatf(math.cos(a / 2), 0, 0, math.sin(a / 2))

        def rot(q, v):
            r = q.GetReal(); im = q.GetImaginary()
            qv = np.array([im[0], im[1], im[2]], float)
            t = 2 * np.cross(qv, v)
            return v + r * t + np.cross(qv, t)

        # pad local +Z = suction/forward axis. top: -> world -Z (down); side: -> world -Y (into +Y face).
        if mode == "top":
            q_fwd = Gf.Quatf(0, 1, 0, 0)                               # 180 deg about X: +Z -> -Z
            face_n0 = np.array([0., 0., 1.])                           # top face outward normal (flat)
        else:
            a = math.radians(90)
            q_fwd = Gf.Quatf(math.cos(a / 2), math.sin(a / 2), 0, 0)   # +90 deg about X: +Z -> -Y
            face_n0 = np.array([0., 1., 0.])                           # +Y face outward normal (flat)
        cart_q = qz(cart_deg)
        pad_q = qz(pad_deg) * q_fwd
        outward = rot(cart_q, face_n0)                                 # yawed face normal, world
        carton_cz = CARTON / 2                                        # rest on ground
        face_centre = np.array([0., 0., carton_cz]) + outward * (CARTON / 2)
        pad_pos = face_centre + outward * GAP

        world = World(physics_dt=1 / 60, rendering_dt=1 / 60, stage_units_in_meters=1.)
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, '/World')
        UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1000.)
        world.scene.add_default_ground_plane()

        carton = UsdGeom.Cube.Define(stage, '/World/Carton'); carton.CreateSizeAttr(1.0)
        cx = UsdGeom.Xformable(carton.GetPrim())
        cx.AddTranslateOp().Set(Gf.Vec3d(0, 0, carton_cz)); cx.AddOrientOp().Set(cart_q)
        cx.AddScaleOp().Set(Gf.Vec3d(CARTON, CARTON, CARTON))
        cp = carton.GetPrim(); UsdPhysics.CollisionAPI.Apply(cp); UsdPhysics.RigidBodyAPI.Apply(cp)
        cp.CreateAttribute('physics:kinematicEnabled', Sdf.ValueTypeNames.Bool, False).Set(False)
        UsdPhysics.MassAPI.Apply(cp).CreateMassAttr(50.0)   # heavy so it does not shift before the grip

        UsdGeom.Xform.Define(stage, '/World/Tool')

        def frame(path, pos, q):
            x = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, path).GetPrim())
            x.AddTranslateOp().Set(Gf.Vec3d(*[float(c) for c in pos])); x.AddOrientOp().Set(q)
            return x.GetPrim()

        def rigid(path, pos, q, kinematic, collider=False):
            prim = frame(path, pos, q); UsdPhysics.RigidBodyAPI.Apply(prim)
            prim.CreateAttribute('physics:kinematicEnabled', Sdf.ValueTypeNames.Bool, False).Set(kinematic)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)
            if collider:
                box = UsdGeom.Cube.Define(stage, path + '/panel'); box.CreateSizeAttr(1.0)
                xb = UsdGeom.Xformable(box.GetPrim())
                xb.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.01)); xb.AddScaleOp().Set(Gf.Vec3d(PANEL[0], PANEL[1], 0.02))
                UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            return prim

        # carrier kinematic (immovable mount); panel body dynamic, pinned by a fixed joint (probe_tilt pattern).
        carrier = '/World/Tool/carrier'; rigid(carrier, pad_pos, pad_q, True)
        body_path = '/World/Tool/suction_cups'; rigid(body_path, pad_pos, pad_q, False, collider=True)
        tool0_path = '/World/Tool/ur_tool0'; frame(tool0_path, pad_pos, pad_q)
        f = UsdPhysics.FixedJoint.Define(stage, '/World/Tool/mount')
        f.CreateBody0Rel().SetTargets([Sdf.Path(carrier)]); f.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])

        tool = CadTool(arm_joints=ARM_JOINTS, arm_limits=((-6., 6.),) * 6, panel_size=PANEL,
                       tool0_from_pad=np.eye(4), tool0_from_support=np.eye(4), tine_limits=((0., 0.),) * 6)
        gpath = '/World/Tool/RestockPanelGripper'
        author_native_gripper(stage, tool=tool, tool0_path=tool0_path, body_path=body_path,
                              gripper_path=gpath, max_gap=MAXGAP,
                              clearance_offset=.5 * MAXGAP, stroke_limit=.8 * MAXGAP)

        world.reset()
        contact = NativePanelContact(stage, gpath, diagnostic=False)
        tl = omni.timeline.get_timeline_interface(); tl.play()
        for _ in range(30):
            world.step(render=False)
        contact.command(True)
        latched = False
        trace = []
        for i in range(300):
            world.step(render=False)
            held = contact.read()
            if i % 25 == 0 or held:
                pr = contact.probe()
                trace.append({'step': i, 'held': bool(held), 'status': pr.get('status'),
                              'gripped_n': pr.get('gripped_n'), 'min_gap_mm': pr.get('min_gap_mm')})
            if held:
                latched = True; result['latch_step'] = i; break
        pr = contact.probe()
        result.update({'latched': latched, 'final_status': pr.get('status'), 'gripped': pr.get('gripped'),
                       'gripped_n': pr.get('gripped_n'), 'forward_world': pr.get('forward_world'),
                       'min_gap_mm': pr.get('min_gap_mm'),
                       'face_normal_world': [round(float(c), 4) for c in outward]})
        fwd = np.array(pr.get('forward_world') or [0, 0, 0], float)
        inward = -outward
        if np.linalg.norm(fwd):
            ang = math.degrees(math.acos(max(-1., min(1., float(np.dot(fwd / np.linalg.norm(fwd), inward))))))
            result['pad_vs_surface_deg'] = round(ang, 3)
        result['trace'] = trace
        result['status'] = 'passed' if latched else 'failed'
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
                if time.monotonic() - started > 480:
                    child.close(); raise TimeoutError(f'{recipe} exceeded 480s')
                time.sleep(1.)
            child.close()
            rp = out / f'{recipe}.json'; res = json.loads(rp.read_text()) if rp.exists() else {'status': 'no-result'}
            summary['recipes'][recipe] = {k: res.get(k) for k in ('status', 'latched', 'latch_step', 'mode',
                'pad_yaw_deg', 'carton_yaw_deg', 'pad_vs_surface_deg', 'final_status', 'gripped', 'gripped_n',
                'min_gap_mm', 'forward_world', 'face_normal_world', 'reason')} | {'exit': child.process.returncode}
            (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(f'[{recipe}] exit={child.process.returncode} latched={res.get("latched")} mode={res.get("mode")} '
                  f'pad_vs_surface_deg={res.get("pad_vs_surface_deg")} gripped={res.get("gripped")} '
                  f'min_gap_mm={res.get("min_gap_mm")}', flush=True)
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
