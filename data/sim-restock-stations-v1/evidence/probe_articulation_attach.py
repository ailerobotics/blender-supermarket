#!/usr/bin/env python3
"""Isolate why the native SurfaceGripper never attaches the HandlingUnit in-mission.

The mission (run-191..202) reaches CONFIRMED 0 mm raycast contact, pad flush
(pad_vs_surface 0.08 deg), suction on, force_limit 200, pad stationary -- and the
native Isaac SurfaceGripper STILL never attaches (gripped_n=0). The isolated stations
harness (probe_seal_frame side_flat) attaches /World/Carton at 0 mm with the SAME
author_native_gripper + NativePanelContact code. Two structural differences between the
mission panel and that harness panel remained untested in isolation:

  (1) ARTICULATION: the mission gripper body0 = /MobileRobot/suction_cups is a TERMINAL
      ARTICULATION LINK of the UR10 (reduced-coordinate). The harness panel was a MAXIMAL
      rigid body pinned by a fixed joint to a kinematic carrier. The plugin may not weld
      an articulation link (body0) to a free rigid body (the carton).
  (2) COLLIDER: the mission panel collider is DISABLED (physics:collisionEnabled=0, the
      self-hit exclusion), so there is no physical gripper-carton collision, only a
      raycast hit. The harness panel had a collider. The plugin attach may require real
      contact.

This drives the REAL author_native_gripper + NativePanelContact on a FLUSH side grasp
(carton flat, pad forward world -Y, gap 0 mm -- the mission contact config) in a clean
2x2 so each variable moves alone from the proven-latching baseline:

  max_col   : MAXIMAL panel on a kinematic carrier, collider ON  -> POSITIVE CONTROL (= side_flat, must LATCH)
  max_nocol : MAXIMAL panel on a kinematic carrier, collider OFF -> does disabling the collider alone break it?
  art_col   : ARTICULATION-LINK panel, collider ON               -> does the articulation alone break it?
  art_nocol : ARTICULATION-LINK panel, collider OFF              -> the MISSION structure (both differences)

Verdict logic (carton/geometry/author code are identical across all four):
  max_col LATCH + max_nocol LATCH + art_* FAIL   => ARTICULATION LINK blocks the attach.
  max_col LATCH + max_nocol FAIL                  => COLLIDER (disabled) blocks the attach.
  max_col LATCH + art_col LATCH + art_nocol FAIL  => COLLIDER blocks, only on the articulation.
  art_nocol LATCH                                 => neither isolates it (the real UR10 differs further).

The articulation is a faithful MINIMAL reproduction: a fixed-base 2-link PhysX articulation
(base root + a terminal `suction_cups` link via a fixed articulation joint), so `suction_cups`
is a reduced-coordinate link exactly like the mission, with everything else (carton, pad pose,
gap, gripper authoring, force limit) held equal to the baseline. Same mission GPU guards.
"""
import argparse, json, sys, time, traceback, math
from pathlib import Path

REPO = Path("/home/manishsaini/.treehouse/blender-supermarket-dc8579/1/blender-supermarket")
SCRIPTS = REPO / "scripts"
# recipe -> (mount, collider, carton)  carton: 'cube' (50 kg analytic cube) | 'realunit' (faithful HandlingUnit mesh)
RECIPES = {
    "max_col":           ("maximal", True, "cube"),
    "max_nocol":         ("maximal", False, "cube"),
    "art_col":           ("articulation", True, "cube"),
    "art_nocol":         ("articulation", False, "cube"),
    # the FULLY faithful mission config: articulation link + collider off + the real convexHull-mesh
    # 0.45 kg HandlingUnit + the real ~0.26x0.108 CAD panel footprint. Isolates whether the real
    # carton BODY (vs the active 6-DOF arm drive, only reproducible in a full mission) blocks the attach.
    "art_nocol_realunit": ("articulation", False, "realunit"),
}
CARTON = 0.30            # default cube edge (any face fits the 0.20 panel)
PANEL = (0.20, 0.20)     # default harness panel
GAP = 0.000              # pad face AT the carton face -- the mission's confirmed 0 mm contact
MAXGAP = 0.020           # SIM_MAX_GRIP_DISTANCE
FORCE_LIMIT = 200.0      # the mission's gripper_force_limit (run-201/202)
# Faithful mission HandlingUnit (run-202 mission.usda /RestockScenario/HandlingUnit): a convexHull
# Mesh box 0.6 x 0.4 x 0.16 m, mass 0.45 kg, physxRigidBody:sleepThreshold=0; the grasped Y-face is
# 0.6 wide x 0.16 tall. The real CAD suction panel footprint (from the mission seal localPos0 spread)
# is ~0.26 x 0.108 m, which fits that face (long axis along the 0.6 m width).
REAL_UNIT = (0.6, 0.4, 0.16)
REAL_UNIT_MASS = 0.45
REAL_PANEL = (0.26, 0.108)


def worker(recipe, output):
    output = Path(output)
    mount, collider, carton_kind = RECIPES[recipe]
    panel_dims = REAL_PANEL if carton_kind == 'realunit' else PANEL
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 320, 'height': 240,
                         'renderer': 'RayTracedLighting', 'anti_aliasing': 0, 'multi_gpu': False})
    result = {'recipe': recipe, 'status': 'failed', 'mount': mount, 'collider': collider,
              'carton': carton_kind, 'panel': list(panel_dims)}
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

        # FLUSH side grasp: pad local +Z (suction/forward) -> world -Y (into the +Y carton face).
        a = math.radians(90)
        q_fwd = Gf.Quatf(math.cos(a / 2), math.sin(a / 2), 0, 0)   # +90 deg about X: +Z -> -Y
        face_n0 = np.array([0., 1., 0.])                           # +Y face outward normal (flat)
        pad_q = q_fwd
        outward = face_n0
        if carton_kind == 'realunit':
            hx, hy, hz = REAL_UNIT[0] / 2, REAL_UNIT[1] / 2, REAL_UNIT[2] / 2
        else:
            hx = hy = hz = CARTON / 2
        carton_cz = hz                                            # rest on ground
        face_centre = np.array([0., 0., carton_cz]) + outward * hy
        pad_pos = face_centre + outward * GAP

        world = World(physics_dt=1 / 60, rendering_dt=1 / 60, stage_units_in_meters=1.)
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, '/World')
        UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1000.)
        world.scene.add_default_ground_plane()

        if carton_kind == 'realunit':
            # Faithful /RestockScenario/HandlingUnit: a convexHull Mesh box (0.6 x 0.4 x 0.16), 0.45 kg,
            # keep-awake. Verbatim physics APIs/attrs the mission authors.
            pts = [(-hx,-hy,-hz),(hx,-hy,-hz),(hx,hy,-hz),(-hx,hy,-hz),
                   (-hx,-hy,hz),(hx,-hy,hz),(hx,hy,hz),(-hx,hy,hz)]
            mesh = UsdGeom.Mesh.Define(stage, '/World/Carton')
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
            mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
            mesh.CreateFaceVertexIndicesAttr([0,3,2,1, 4,5,6,7, 0,1,5,4, 1,2,6,5, 2,3,7,6, 3,0,4,7])
            mesh.CreateExtentAttr([Gf.Vec3f(-hx,-hy,-hz), Gf.Vec3f(hx,hy,hz)])
            cx = UsdGeom.Xformable(mesh.GetPrim())
            cx.AddTranslateOp().Set(Gf.Vec3d(0, 0, carton_cz)); cx.AddOrientOp().Set(qz(0.0))
            cp = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(cp); UsdPhysics.MeshCollisionAPI.Apply(cp)
            cp.CreateAttribute('physics:approximation', Sdf.ValueTypeNames.Token, False).Set('convexHull')
            UsdPhysics.RigidBodyAPI.Apply(cp)
            cp.CreateAttribute('physics:kinematicEnabled', Sdf.ValueTypeNames.Bool, False).Set(False)
            cp.CreateAttribute('physxRigidBody:sleepThreshold', Sdf.ValueTypeNames.Float, False).Set(0.0)
            UsdPhysics.MassAPI.Apply(cp).CreateMassAttr(REAL_UNIT_MASS)
        else:
            carton = UsdGeom.Cube.Define(stage, '/World/Carton'); carton.CreateSizeAttr(1.0)
            cx = UsdGeom.Xformable(carton.GetPrim())
            cx.AddTranslateOp().Set(Gf.Vec3d(0, 0, carton_cz)); cx.AddOrientOp().Set(qz(0.0))
            cx.AddScaleOp().Set(Gf.Vec3d(CARTON, CARTON, CARTON))
            cp = carton.GetPrim(); UsdPhysics.CollisionAPI.Apply(cp); UsdPhysics.RigidBodyAPI.Apply(cp)
            cp.CreateAttribute('physics:kinematicEnabled', Sdf.ValueTypeNames.Bool, False).Set(False)
            UsdPhysics.MassAPI.Apply(cp).CreateMassAttr(50.0)   # heavy so it does not shift before the grip

        UsdGeom.Xform.Define(stage, '/World/Tool')

        def frame(path, pos, q):
            x = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, path).GetPrim())
            x.AddTranslateOp().Set(Gf.Vec3d(*[float(c) for c in pos])); x.AddOrientOp().Set(q)
            return x.GetPrim()

        def add_collider(path):
            box = UsdGeom.Cube.Define(stage, path + '/panel'); box.CreateSizeAttr(1.0)
            xb = UsdGeom.Xformable(box.GetPrim())
            xb.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.01)); xb.AddScaleOp().Set(Gf.Vec3d(panel_dims[0], panel_dims[1], 0.02))
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())

        body_path = '/World/Tool/suction_cups'
        tool0_path = '/World/Tool/ur_tool0'
        if mount == 'maximal':
            # BASELINE: panel is a maximal dynamic body pinned by a fixed joint to a kinematic carrier.
            carrier = '/World/Tool/carrier'
            cpr = frame(carrier, pad_pos, pad_q); UsdPhysics.RigidBodyAPI.Apply(cpr)
            cpr.CreateAttribute('physics:kinematicEnabled', Sdf.ValueTypeNames.Bool, False).Set(True)
            UsdPhysics.MassAPI.Apply(cpr).CreateMassAttr(1.0)
            panel = frame(body_path, pad_pos, pad_q); UsdPhysics.RigidBodyAPI.Apply(panel)
            panel.CreateAttribute('physics:kinematicEnabled', Sdf.ValueTypeNames.Bool, False).Set(False)
            UsdPhysics.MassAPI.Apply(panel).CreateMassAttr(1.0)
            if collider:
                add_collider(body_path)
            f = UsdPhysics.FixedJoint.Define(stage, '/World/Tool/mount')
            f.CreateBody0Rel().SetTargets([Sdf.Path(carrier)]); f.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
            frame(tool0_path, pad_pos, pad_q)
        else:
            # ARTICULATION: fixed-base 2-link articulation; `suction_cups` is a terminal reduced-coord link.
            root = '/World/Tool/base'
            bpr = frame(root, pad_pos, pad_q); UsdPhysics.RigidBodyAPI.Apply(bpr)
            UsdPhysics.MassAPI.Apply(bpr).CreateMassAttr(10.0)
            UsdPhysics.ArticulationRootAPI.Apply(bpr)
            # fixed base: pin the root link to the world at the authored pose (body0 empty == world)
            fb = UsdPhysics.FixedJoint.Define(stage, '/World/Tool/base_fix')
            fb.CreateBody0Rel().SetTargets([]); fb.CreateBody1Rel().SetTargets([Sdf.Path(root)])
            fb.CreateLocalPos0Attr(Gf.Vec3f(*[float(c) for c in pad_pos]))
            fb.CreateLocalRot0Attr(pad_q)
            fb.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            panel = frame(body_path, pad_pos, pad_q); UsdPhysics.RigidBodyAPI.Apply(panel)
            UsdPhysics.MassAPI.Apply(panel).CreateMassAttr(1.0)
            if collider:
                add_collider(body_path)
            # articulation joint base -> panel (fixed, PART of the articulation: no excludeFromArticulation),
            # so `suction_cups` is a distinct reduced-coordinate articulation link.
            aj = UsdPhysics.FixedJoint.Define(stage, '/World/Tool/arm_joint')
            aj.CreateBody0Rel().SetTargets([Sdf.Path(root)]); aj.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
            aj.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0)); aj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            frame(tool0_path, pad_pos, pad_q)

        tool = CadTool(arm_joints=ARM_JOINTS, arm_limits=((-6., 6.),) * 6, panel_size=panel_dims,
                       tool0_from_pad=np.eye(4), tool0_from_support=np.eye(4), tine_limits=((0., 0.),) * 6)
        gpath = '/World/Tool/RestockPanelGripper'
        author_native_gripper(stage, tool=tool, tool0_path=tool0_path, body_path=body_path,
                              gripper_path=gpath, max_gap=MAXGAP, force_limit=FORCE_LIMIT,
                              clearance_offset=.5 * MAXGAP, stroke_limit=.8 * MAXGAP)

        world.reset()
        # Confirm the panel actually became an articulation link when requested (else the test is void).
        art_info = {}
        try:
            from isaacsim.core.prims import SingleArticulation
            if mount == 'articulation':
                art = SingleArticulation(prim_path='/World/Tool/base')
                art.initialize()
                art_info = {'articulation_dof': int(art.num_dof),
                            'articulation_bodies': int(getattr(art, 'num_bodies', -1))}
        except Exception as exc:
            art_info = {'articulation_probe_err': '%s: %s' % (type(exc).__name__, exc)}
        result['articulation'] = art_info

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
            summary['recipes'][recipe] = {k: res.get(k) for k in ('status', 'latched', 'latch_step', 'mount',
                'collider', 'carton', 'panel', 'final_status', 'gripped', 'gripped_n', 'min_gap_mm',
                'forward_world', 'face_normal_world', 'articulation', 'reason')} | {'exit': child.process.returncode}
            (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(f'[{recipe}] exit={child.process.returncode} latched={res.get("latched")} mount={res.get("mount")} '
                  f'collider={res.get("collider")} gripped={res.get("gripped")} gripped_n={res.get("gripped_n")} '
                  f'min_gap_mm={res.get("min_gap_mm")} art={res.get("articulation")}', flush=True)
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
