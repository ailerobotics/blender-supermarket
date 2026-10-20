"""Isaac-side scene for the demo: a LEAN composer + kinematic poser + capture + reads.

Reuses the mission's own scene composition (``twin.restock.scene.build`` -- the same store
USD, mobile robot, rollwagen and carton as the autonomous mission) but deliberately skips
the autonomy stack ``physics_setup.assemble`` wires on top (gripper/contact/ArmCommandGate,
the ROS2 bridge and the host-DDS NativeTransport subprocess) so the demo stays light on
VRAM and spawns no host services.

Choreography is applied kinematically (gravity zeroed; base pose + arm joints + cart +
carton set each frame). The panel's values are then READ BACK from the live USD prim
world transforms by :meth:`DemoScene.reads` -- genuine scene measurements, not the scripted
targets. Isaac-only; every heavy import is lazy so the module imports on CPU.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import choreography as C


def _yaw_to_wxyz(yaw):
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _wxyz_to_yaw(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    """4x4 world transform for a USD camera at ``eye`` looking at ``target`` (camera looks
    down its local -Z, +Y up)."""
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd = fwd / (np.linalg.norm(fwd) or 1.0)
    zaxis = -fwd
    xaxis = np.cross(np.asarray(up, float), zaxis)
    xaxis = xaxis / (np.linalg.norm(xaxis) or 1.0)
    yaxis = np.cross(zaxis, xaxis)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = xaxis, yaxis, zaxis, eye
    return M


class DemoScene:
    def __init__(self, *, app, config, scenario, out_dir, resolution=(1280, 960),
                 camera=('wide',), cam_height=2.1):
        self.app = app
        self.out_dir = Path(out_dir)
        self.resolution = resolution
        self._cam_height = float(cam_height)  # chase-camera height above base_link (m)
        self._last_rgb = None
        self._build(config, scenario)
        self._hide_overhead_from_render()
        self._author_camera(camera)

    # ---- lean composition (mirrors the head of physics_setup.assemble) ----
    def _build(self, config, scenario):
        from twin.restock.physics_setup import inputs
        from twin.restock.isaac_bootstrap import import_mobile, discover, resolve_meshes
        from twin.restock.scene import build, label_stock
        from isaacsim.core.utils.stage import open_stage, is_stage_loading
        mobile, urdf, store, spec, tool, _ = inputs(config, scenario)
        out = self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        resolved = resolve_meshes(urdf, out / 'mobile-import.urdf', config['package_roots'])
        robot_usd, _ = import_mobile(resolved, out / 'mobile-import.usd')
        staging = scenario['initial_staging']
        comp = build(store, robot_usd, out / 'mission.usda',
                     robot_pose=staging['robot_base_pose'], cart_pose=staging['cart_body_pose'],
                     stock_path=None, sku=scenario['job']['sku'],
                     stock_mass_kg=scenario['handling_unit']['mass_kg'], spec=spec, scenario=scenario)
        if not open_stage(comp['stage']):
            raise RuntimeError('could not open composed demo stage')
        for _ in range(300):
            self.app.update()
            if not is_stage_loading():
                break
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        stage = omni.usd.get_context().get_stage()
        self.stage = stage
        self.paths = discover(stage)
        self.spec = spec
        self.tool = tool
        self.composition = comp
        try:
            label_stock(stage, comp['stock'], scenario['job']['sku'], out / 'handling-unit-qr.png')
        except Exception as exc:  # cosmetic QR; never fail the demo
            print(f'DEMO label_stock skipped: {exc!r}', flush=True)
        from pxr import UsdPhysics
        scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scenes) != 1:
            raise ValueError('one composed physics scene required')
        self.world = World(physics_dt=1 / 60, rendering_dt=1 / 60, stage_units_in_meters=1.,
                           physics_prim_path=scenes[0])
        self.robot = self.world.scene.add(SingleArticulation(prim_path=self.paths['articulation'], name='demo_robot'))
        cart_body = comp['assets']['cart']['body']
        self.cart_root = comp['assets']['cart']['root']
        self.cart = self.world.scene.add(SingleRigidPrim(prim_path=cart_body, name='demo_cart'))
        self.carton = self.world.scene.add(SingleRigidPrim(prim_path=comp['stock'], name='demo_carton'))
        self.world.reset()
        # Gravity OFF: this is choreography; the scripted poses must hold for the shot.
        try:
            self.world.get_physics_context().set_gravity(0.0)
        except Exception as exc:
            print(f'DEMO gravity-off skipped: {exc!r}', flush=True)
        # Arm joint indices + a home/stow preset from the mobile manifest transport positions.
        self._arm_names = list(self.tool.arm_joints)
        self._arm_idx = np.array([self.robot.get_dof_index(n) for n in self._arm_names])
        import json as _json
        man = _json.loads(Path(config['mobile_manifest']).read_text())
        self._stow_q = np.array([float(man['transport_positions'][n]) for n in self._arm_names])
        self._q_full0 = np.asarray(self.robot.get_joint_positions(), float).copy()
        # IK so the gripper tracks the carton on the manip/carry phases (arm visibly grasps it).
        from .arm_ik import ArmIK
        self._ik = ArmIK(resolved, self._arm_names, base_link='base_link',
                         tip_link='gripper_pad_tcp', limits=list(self.tool.arm_limits))
        self._ik_q = self._stow_q.copy()  # warm-start across frames
        # Obstacle reads: the backstock support prim centre + the shelf slot target.
        self._support_path = '/RestockScenario/Support'
        self._base_z = float(self.spec.base_z)
        self._cage_top_z = self._prim_max_z(self.cart_root)  # rollwagen cage top rail (cart z==0)

    def _hide_overhead_from_render(self, z_floor=2.5):
        """Make purely-overhead occluders (ceiling/roof/high signage) invisible so the
        top-down follow camera never looks down THROUGH them (the black-frame cause). Only
        meshes whose whole AABB sits above ``z_floor`` are hidden; the ceiling LIGHTS
        (GRP_Lighting, non-geometry emitters) and everything at human height are kept, so the
        store stays lit and in-context. Best-effort; never raises into the run."""
        try:
            from pxr import Usd, UsdGeom
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            hidden = 0
            for root in ('/Supermarket/GRP_Structure', '/Supermarket/GRP_Fixtures'):
                base = self.stage.GetPrimAtPath(root)
                if not base or not base.IsValid():
                    continue
                for prim in Usd.PrimRange(base):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    if rng.IsEmpty():
                        continue
                    if float(rng.GetMin()[2]) > z_floor:   # whole mesh is overhead
                        UsdGeom.Imageable(prim).MakeInvisible()
                        hidden += 1
            print(f'DEMO hid {hidden} overhead occluder meshes above z={z_floor}', flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f'DEMO overhead-hide skipped: {exc!r}', flush=True)

    def _author_camera(self, camera):
        """Replicate the mission's PROVEN overhead walkthrough capture: a top-down chase
        camera parented to the base (follows the robot) via the mission's own
        initialize_camera (its annotators force the render product to render), PLUS a fill
        light riding above the base -- the store ceiling rig alone leaves the top-down view
        black (run-110). Exact recipe from restock.sensing_sources."""
        from twin.restock.native_cameras import initialize_camera
        from twin.restock_camera.geometry import Intrinsics
        from pxr import UsdLux, UsdGeom, Gf
        import numpy as _np
        self._cam_mode = camera[0] if isinstance(camera, (tuple, list)) else camera
        base_root = self.paths['base_link']
        w, h = self.resolution
        sx, sy = w / 960.0, h / 720.0
        # Keep the ground footprint ~constant as the height drops: fx scales with camera
        # height above the floor (430 was tuned at ~3.54 m total).
        fcore = 430.0 * (self._cam_height + self._base_z) / 3.54
        chase_k = Intrinsics(width=w, height=h, fx=fcore * sx, fy=fcore * sy,
                             cx=w / 2.0, cy=h / 2.0, near=.1, far=18.)
        chase_path = base_root.rstrip('/') + '/demo_chase'
        self.camera = initialize_camera(chase_path, chase_k, name='demo_chase')
        # Looking straight down (USD camera -Z; identity child points down). Height is kept
        # BELOW the store ceiling/overhead fixtures (~3.5 m) so the camera never looks down
        # through them -- the black-aisle-frame cause. A wider FOV keeps the footprint.
        cz = self._cam_height
        self.camera.set_local_pose(translation=_np.array([0.0, 0.0, cz]),
                                   orientation=_np.array([1., 0., 0., 0.]), camera_axes='usd')
        fill_path = base_root.rstrip('/') + '/demo_fill'
        fill = UsdLux.DiskLight.Define(self.stage, fill_path)
        fill.CreateIntensityAttr(26000.)
        fill.CreateRadiusAttr(1.8)
        fill.CreateColorAttr(Gf.Vec3f(1., .97, .92))
        UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0., 0., min(cz + 0.2, 2.6)))
        for _ in range(20):
            self.world.step(render=True)

    def _grab_rgb(self):
        frame = self.camera.get_current_frame()
        data = frame.get('rgba', frame.get('rgb')) if isinstance(frame, dict) else None
        if data is None:
            try:
                data = self.camera.get_rgba()
            except Exception:
                data = None
        arr = np.asarray(data)
        if arr.ndim == 3 and arr.shape[2] >= 3 and arr.size > 0:
            return np.ascontiguousarray(arr[:, :, :3].astype(np.uint8))
        return None

    def _base_matrix(self, bx, by, byaw):
        c, s = math.cos(byaw), math.sin(byaw)
        return np.array([[c, -s, 0, bx], [s, c, 0, by], [0, 0, 1, self._base_z], [0, 0, 0, 1.]])

    def _arm_config(self, frame):
        """Arm joint 6-vec: IK the gripper onto the scripted carton on TRACK frames, else stow."""
        from . import choreography as C
        if frame['arm'] != C.TRACK:
            self._ik_q = self._stow_q.copy()
            return self._stow_q
        kx, ky, kz, _ = frame['carton']
        bx, by, byaw = frame['base']
        target_base = np.linalg.inv(self._base_matrix(bx, by, byaw)) @ np.array([kx, ky, kz, 1.0])
        self._ik_q = self._ik.solve(target_base[:3], self._ik_q)
        return self._ik_q

    # ---- per-frame kinematic apply ----
    def apply(self, frame):
        bx, by, byaw = frame['base']
        self.robot.set_world_pose(position=np.array([bx, by, self._base_z]),
                                  orientation=np.array(_yaw_to_wxyz(byaw)))
        q = self._q_full0.copy()
        arm_q = self._arm_config(frame)
        for j, idx in enumerate(self._arm_idx):
            q[int(idx)] = arm_q[j]
        self.robot.set_joint_positions(q)
        try:
            self.robot.set_joint_velocities(np.zeros_like(q))
        except Exception:
            pass
        cxx, cyy, cyaw = frame['cart']
        self.cart.set_world_pose(position=np.array([cxx, cyy, 0.0]),
                                 orientation=np.array(_yaw_to_wxyz(cyaw)))
        kx, ky, kz, kyaw = frame['carton']
        self.carton.set_world_pose(position=np.array([kx, ky, kz]),
                                   orientation=np.array(_yaw_to_wxyz(kyaw)))
        for body in (self.cart, self.carton):
            try:
                body.set_linear_velocity(np.zeros(3))
                body.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        # The chase camera + fill light are parented to base_link, so they follow the base
        # automatically; no per-frame camera move needed.

    def apply_and_step(self, frame):
        self.apply(frame)
        self.world.step(render=True)

    def capture_to_file(self, frame, path):
        from PIL import Image
        rgb = self._grab_rgb()
        if rgb is None:
            for _ in range(3):
                self.world.step(render=True)
            rgb = self._grab_rgb()
        if rgb is None:
            rgb = np.zeros((self.resolution[1], self.resolution[0], 3), np.uint8)
        Image.fromarray(rgb).save(str(path))

    # ---- read the live stage for the panel values ----
    def _world_xy_yaw(self, prim_handle):
        pos, quat = prim_handle.get_world_pose()
        return (float(pos[0]), float(pos[1])), _wxyz_to_yaw(quat), float(pos[2])

    def _prim_center(self, path):
        from pxr import Usd, UsdGeom
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        cx, cy = (mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0
        r = max(mx[0] - mn[0], mx[1] - mn[1]) / 2.0
        return (cx, cy, max(0.15, r))

    def _prim_max_z(self, path):
        from pxr import Usd, UsdGeom
        prim = self.stage.GetPrimAtPath(path)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        return float(cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMax()[2])

    def reads(self, frame, prev):
        base_xy, base_yaw, _ = self._world_xy_yaw(self.robot)
        cart_xy, cart_yaw, _ = self._world_xy_yaw(self.cart)
        carton_pos, _, carton_z = self._world_xy_yaw(self.carton)
        pad = self._prim_world(self.paths['gripper_pad_tcp'])
        hitch = (base_xy[0] - C.HITCH_BEHIND * math.cos(base_yaw),
                 base_xy[1] - C.HITCH_BEHIND * math.sin(base_yaw))
        dock = (cart_xy[0] + C.DOCK_AHEAD * math.cos(cart_yaw),
                cart_xy[1] + C.DOCK_AHEAD * math.sin(cart_yaw))
        obstacles = []
        for p in (self._support_path,):
            c = self._prim_center(p)
            if c is not None:
                obstacles.append(c)
        # shelf slot target as an obstacle point (read from the composed scenario target).
        obstacles.append((C.P_SHELF[0], C.P_SHELF[1] - 0.2, 0.6))
        brake = 0.3
        if prev is not None:
            d = math.hypot(base_xy[0] - prev['base'][0], base_xy[1] - prev['base'][1])
            dt = max(1e-3, frame['t'] - prev['t'])
            speed = d / dt
            brake = max(0.05, min(0.6, speed * 0.3 + speed * speed / (2 * 0.4)))
        art = (cart_yaw - base_yaw + math.pi) % (2 * math.pi) - math.pi
        return {
            'base_xy': base_xy, 'base_yaw': base_yaw, 'cart_yaw': cart_yaw,
            'hitch_xy': hitch, 'dock_xy': dock,
            'carton_xyz': (carton_pos[0], carton_pos[1], carton_z),
            'carton_face_xyz': (carton_pos[0], carton_pos[1] - 0.2, carton_z),
            'pad_xyz': pad, 'carton_top_z': carton_z + C.CARTON_HALF_H,
            'cage_top_z': self._cage_top_z,
            'cage_target_xyz': (cart_xy[0] + math.cos(cart_yaw) * 0.08,
                                cart_xy[1] + math.sin(cart_yaw) * 0.08, C.CAGE_SHELF_Z + C.CARTON_HALF_H),
            'shelf_target_xyz': (C.P_SHELF[0], C.P_SHELF[1], C.P_SHELF[2]),
            'obstacles': obstacles, 'brake_m': brake, 'articulation_rad': art,
        }

    def _prim_world(self, path):
        from pxr import Usd, UsdGeom
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        m = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))
        t = np.asarray(m).T[:3, 3]
        return (float(t[0]), float(t[1]), float(t[2]))
