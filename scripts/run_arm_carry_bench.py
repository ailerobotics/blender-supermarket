#!/usr/bin/env python3
"""Boot-once arm-carry bench runner (guarded GPU; RUNBOOK §16).

Boots a SimulationApp ONCE under the canonical GPU lock, assembles the mission's physics
scene, welds the carton at the tool through the mission's own SIM_ATTACH_FALLBACK fixed
joint, brakes the base in the airborne-carry state, then runs MANY trials over a gain /
velocity-accel / tolerance grid -- each trial folds the carton-carrying arm through the
mission's joint-space carry sequence (tuck -> pan -> ...) and emits per-trial metrics.

It drives the REAL ArmCommandGate and the REAL trajectory_host Plan + tracking/settle gates
(imported from those modules), so the drive gains, the carry velocity/acceleration scaling
and the sim-only tracking/settle tolerances are exercised identically to the live mission.
The ONLY honest differences from the live mission are the planning path (carry trajectories
are generated analytically here, not by MoveIt/cuMotion) and the in-process executor loop
(no ROS relay round-trip); both are documented in twin/tools/arm_carry_bench.py.

Runs headless with physics-only stepping (render off) so a full tuck+pan trial is seconds of
wall time -- many trials per 15-minute GPU session. No walkthrough recording.

Modes (--expect):
  report      (default) emit metrics, always exit 0 (exploratory sweep)
  all-settle  exit 0 iff EVERY trial settled EVERY move (a candidate-set gate)
  reproduce   exit 0 iff the stiff reference trial limit-cycles AND the soft one under-tracks
              (the run-248 / run-249 signatures) -- the verification gate before any claim

GPU etiquette: the integration lane holds priority on /tmp/aile-gpu.lock; keep sessions short
(<15 min). --heavy-threshold-mb 2500 accepts the always-on whispr (~2.2 GB). Never --force.
"""
import argparse
import json
import os
import signal
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from contextlib import contextmanager

from twin.restock.guard import available_ram_mb, gpu_slot
from twin.tools import arm_carry_bench as bench


@contextmanager
def gpu_guards_no_flock(*, heavy_threshold_mb=1500):
    """The same protected-Blender / RAM / VRAM / domain-88-port guards as guard.gpu_slot, but WITHOUT
    taking the flock -- for when an OUTER `flock -w /tmp/aile-gpu.lock` already holds the lock (the
    fleet blocking-wait discipline, firstmate inbox 008). Never bypasses the VRAM guard."""
    import socket
    from twin.restock.guard import isolation, protected_blender
    from twin.gpu_guard import query_and_assess
    if heavy_threshold_mb not in (1500, 2500):
        raise ValueError('heavy threshold must use default 1500 or explicit opt-in 2500')
    isolation()
    if protected_blender():
        raise RuntimeError('protected Blender process or MCP socket is present')
    if available_ram_mb() < 12 * 1024:
        raise RuntimeError('need at least 12 GiB available RAM')
    decision = query_and_assess(required_free_mb=8000, heavy_threshold_mb=heavy_threshold_mb)
    if not decision.ok:
        raise RuntimeError(decision.message)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(('127.0.0.1', 7400 + 250 * 88))
        yield decision

# A plausible ESTIMATE of the mission lift pose (arm joints, ARM_JOINTS order) at the start of
# the carry fold -- carton grasped at backstock_00 and lifted, before the tuck. Override with
# --lift-joints or runtime-config['armbench']['lift_joints'] with the mission's MEASURED lift
# pose for exact fidelity; the bench records whatever start pose it actually used.
# Measured mission poses (IK-solved, integration lane reach-check.json, ARM_JOINTS order).
# GRASP = source_contact (pad at the carton surface); LIFT = source_approach (grasp raised by the
# approach offset). The bench welds the carton at GRASP in place, then drives grasp->lift->tuck->pan.
DEFAULT_GRASP = [1.4051, -1.0610, 1.1213, -1.6311, -1.5708, -0.1657]
DEFAULT_LIFT = [1.4051, -1.0499, 0.8940, -1.4149, -1.5708, -0.1657]
# The carry fold's body-clear shape + cart pan (ros_host_config.json; handoff (b)).
CART_TUCK_SHAPE = [-2.3, 1.0, 2.1066, 0.8, -1.2545]
CART_PAN_JOINT0 = -0.8605
# Measured side-grip offset of the carton CENTRE along tool-z from gripper_pad_tcp (handoff (a);
# xrdf_selfcollision_check.py). 0.6 m carton, side grip ~= 0.20 m.
CARTON_TOOL_Z_OFFSET = 0.20
# Carton centre offset in the gripper_pad_tcp frame (the model world_obstacles.carton_spheres_world
# uses: 0.20 m along tool-z for the 0.6 m side-grip). Confirmed against the integration lane's CPU
# sphere model.
CARTON_OFFSET_TCP = (0.0, 0.0, 0.20)
# The run-248 / run-249 reference gains (the two failure signatures to reproduce).
REF_STIFF = bench.DriveGains(1.0e6, 1.0e4, 4000.0)
REF_SOFT = bench.DriveGains(1.0e5, 2.0e3, 400.0)


class IsaacCarryStage(bench.ArmStage):
    """Isaac glue: welds the carton, brakes the base, and implements the ArmStage the executor
    drives against the live articulation. Isaac-only; never imported by the CPU tests."""

    def __init__(self, assembly, lift_joints, grasp_joints, *, carton_path='/RestockScenario/HandlingUnit',
                 pad_path_key='gripper_pad_tcp', render=False, carton_dims=(0.6, 0.4, 0.16),
                 obstacle_prims=None, no_carton=False, arm_contacts=False, start_at_lift=False):
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import UsdGeom
        self._UsdGeom = UsdGeom
        self.assembly = assembly
        self.robot = assembly.robot
        self.gate = assembly.command_gate
        self.world = assembly.world
        self.stage = assembly.stage
        self.contact = assembly.contact
        self.mobile = assembly.mobile
        self.robot_body = assembly.robot_body
        self.render = render
        self.lift_joints = np.asarray(lift_joints, float)
        self.grasp_joints = np.asarray(grasp_joints, float)
        self.arm_idx = np.asarray(self.gate.indices)
        self.pad_path = assembly.paths[pad_path_key]
        self.carton_path = carton_path
        self.carton_half = np.asarray(carton_dims, float) / 2.0
        if not self.stage.GetPrimAtPath(carton_path):
            raise RuntimeError(f'carton prim not in assembly: {carton_path}')
        self.carton = self.world.scene.add(SingleRigidPrim(prim_path=carton_path, name='armbench_carton'))
        self.arm_ctrl = self.robot.get_articulation_controller()
        self.no_carton = bool(no_carton)
        self._arm_contacts = bool(arm_contacts)
        self.start_at_lift = bool(start_at_lift)
        self._weld_rel0 = None
        # physics dt for converting accumulated contact impulse -> force (inbox 013)
        try:
            self._phys_dt = float(self.world.get_physics_dt())
        except Exception:
            self._phys_dt = 1.0 / 60.0
        try:
            self._carton_mass = float(np.asarray(self.carton.get_masses()).reshape(-1)[0])
        except Exception:
            try:
                self._carton_mass = float(self.carton.get_mass())
            except Exception:
                self._carton_mass = 0.0
        print('ARMBENCH_CARTON_MASS %.4f kg (weight %.2f N)' % (self._carton_mass, self._carton_mass * 9.81), flush=True)
        self._current_drive_type = 'acceleration'   # physics_setup leaves the arm drives acceleration
        self.arm_joint_prims = self._discover_arm_joints()
        self.arm_links = self._discover_arm_links()
        self.obstacles = self._discover_obstacles(obstacle_prims)
        self._contact_buf = []
        self._contact_sub = None
        self._enable_contact_reporting()

    def _discover_arm_joints(self):
        """Prim paths of the 6 arm revolute joints (for USD DriveAPI drive-type/gain control)."""
        from pxr import Usd, UsdPhysics
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        root = self.stage.GetPrimAtPath(self.assembly.paths['articulation'].rsplit('/', 1)[0])
        byname = {}
        if root:
            for prim in Usd.PrimRange(root):
                if prim.GetName() in ARM_JOINTS and (prim.IsA(UsdPhysics.RevoluteJoint)
                                                     or prim.HasAPI(UsdPhysics.DriveAPI)
                                                     or prim.GetTypeName() == 'PhysicsRevoluteJoint'):
                    byname[prim.GetName()] = str(prim.GetPath())
        ordered = [byname.get(n) for n in ARM_JOINTS]
        print('ARMBENCH_ARM_JOINTS %s' % ordered, flush=True)
        return ordered

    def _discover_arm_links(self):
        """Prim paths of the robot's own arm links + gripper, so carton-vs-arm self-collision
        (the run-250 upper-arm hypothesis) is measured, not just carton-vs-static-scene."""
        from pxr import Usd
        root = self.assembly.paths['articulation'].rsplit('/', 1)[0]
        root = self.stage.GetPrimAtPath(root)
        links = {}
        if root:
            for prim in Usd.PrimRange(root):
                name = prim.GetName()
                if (name.startswith('ur_') and name.endswith('_link')) or \
                        name in ('gripper_pad_tcp', 'gripper_tcp', 'suction_cups', 'ur_tool0',
                                 'RestockPanelGripper'):
                    links[name] = str(prim.GetPath())
        print('ARMBENCH_ARM_LINKS %s' % sorted(links), flush=True)
        return links

    # ---- obstacle discovery + clearance/contact probing (the redirect's diagnosis) ----
    def _discover_obstacles(self, override):
        obstacles = {}
        try:
            cart_root = self.assembly.evidence['composition']['assets']['cart']['root']
            if self.stage.GetPrimAtPath(cart_root):
                obstacles['cart'] = cart_root
        except Exception:
            pass
        if self.stage.GetPrimAtPath('/RestockScenario/Support'):
            obstacles['support'] = '/RestockScenario/Support'
        # NB: the whole store root (/Supermarket) is NOT added as a clearance obstacle -- its AABB
        # engulfs the robot and every fixture, so AABB clearance to it is meaningless. The rack/shelf
        # the carton actually hits is named authoritatively by PhysX contact reporting instead; add a
        # SPECIFIC shelf prim via config['armbench']['obstacle_prims'] for a coarse AABB rack metric.
        if override:
            obstacles.update(override)
        print('ARMBENCH_OBSTACLES %s' % obstacles, flush=True)
        return obstacles

    def _enable_contact_reporting(self):
        """Best-effort PhysX contact reporting. DEFAULT: the carton only (a separate rigid prim --
        safe). Applying ContactReportAPI to the ARM links / chassis (part of the articulation) after
        init invalidates the physics TENSOR VIEW, breaking get_gains() -- so arm/body self-collision
        reporting is OPT-IN (``armbench_arm_contacts``) and must not be combined with a gain sweep;
        the carton-vs-arm self-collision is instead measured offline by
        twin.tools.carton_arm_fold_clearance (CPU sphere model). Each reported contact records both
        actors. Degrades to geometric clearance if unavailable."""
        try:
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysxSchema, PhysicsSchemaTools
            self._PhysicsSchemaTools = PhysicsSchemaTools
            tracked = [self.carton_path]
            if self._arm_contacts:   # opt-in: breaks the articulation tensor view / get_gains
                tracked += list(self.arm_links.values())
                base = self.assembly.paths.get('base_link')
                if base:
                    tracked.append(base)
            self._tracked_contacts = set(tracked)
            for path in tracked:
                prim = self.stage.GetPrimAtPath(path)
                if not prim:
                    continue
                api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                try:
                    api.CreateThresholdAttr().Set(0.0)
                except Exception:
                    pass

            def _on_contact(contact_headers, contact_data):
                for ch in contact_headers:
                    try:
                        a0 = str(PhysicsSchemaTools.intToSdfPath(ch.actor0))
                        a1 = str(PhysicsSchemaTools.intToSdfPath(ch.actor1))
                    except Exception:
                        continue
                    hit0 = any(t in a0 for t in self._tracked_contacts)
                    hit1 = any(t in a1 for t in self._tracked_contacts)
                    if not (hit0 or hit1):
                        continue
                    pos = None
                    impulse = np.zeros(3)
                    try:
                        n = int(ch.num_contact_data)
                        off = int(ch.contact_data_offset)
                        if n > 0:
                            cp = contact_data[off].position
                            pos = [float(cp[0]), float(cp[1]), float(cp[2])]
                            # sum the per-point contact impulses (inbox 013: "with what force")
                            for k in range(off, off + n):
                                try:
                                    imp = contact_data[k].impulse
                                    impulse += np.array([float(imp[0]), float(imp[1]), float(imp[2])], float)
                                except Exception:
                                    pass
                    except Exception:
                        pos = None
                    # force = |impulse| / physics dt (the contact report accumulates one physics step)
                    force_N = float(np.linalg.norm(impulse)) / max(self._phys_dt, 1e-6)
                    # record the PAIR so an arm-vs-body / arm-vs-arm self-collision is visible
                    self._contact_buf.append({'prim': '%s<->%s' % (a0.rsplit('/', 1)[-1], a1.rsplit('/', 1)[-1]),
                                              'position': pos, 'a0': a0, 'a1': a1, 'force_N': force_N})
            self._contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(_on_contact)
            print('ARMBENCH_CONTACT_REPORTING enabled on %d bodies (carton+arm+chassis)' % len(tracked), flush=True)
        except Exception as exc:
            print('ARMBENCH_CONTACT_REPORTING unavailable (%s); geometric clearance only' % exc, flush=True)

    def _carton_world_aabb(self):
        pos, orient = self.carton.get_world_pose()          # orient is wxyz
        pos = np.asarray(pos, float)
        w, x, y, z = [float(v) for v in orient]
        # wxyz quaternion -> rotation matrix
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)
        corners = []
        hx, hy, hz = self.carton_half
        for sx in (-hx, hx):
            for sy in (-hy, hy):
                for sz in (-hz, hz):
                    corners.append(pos + R @ np.array([sx, sy, sz], float))
        corners = np.asarray(corners, float)
        return corners.min(axis=0), corners.max(axis=0)

    def _obstacle_aabb(self, prim_path):
        UsdGeom = self._UsdGeom
        from pxr import Usd
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                  [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = cache.ComputeWorldBound(self.stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
        lo, hi = rng.GetMin(), rng.GetMax()
        return (np.array([lo[0], lo[1], lo[2]], float), np.array([hi[0], hi[1], hi[2]], float))

    @staticmethod
    def _aabb_signed_gap(a, b):
        """Signed clearance between two AABBs (a=(lo,hi), b=(lo,hi)). Positive = disjoint distance;
        negative = penetration depth on the tightest axis (overlap)."""
        alo, ahi = a
        blo, bhi = b
        sep = np.maximum(blo - ahi, alo - bhi)   # per-axis separation (negative where overlapping)
        if np.any(sep > 0):
            return float(np.linalg.norm(np.maximum(sep, 0.0)))
        return float(np.max(sep))                # all axes overlap -> least-negative = tightest penetration

    def clearance_snapshot(self):
        try:
            carton = self._carton_world_aabb()
        except Exception:
            return {}
        out = {}
        for name, path in self.obstacles.items():
            try:
                out[name] = self._aabb_signed_gap(carton, self._obstacle_aabb(path))
            except Exception:
                continue
        return out

    def probe_contacts(self):
        hits = self._contact_buf
        self._contact_buf = []
        # de-dupe within a tick by prim, keeping the LARGEST contact force seen for that pair
        best = {}
        for h in hits:
            prev = best.get(h['prim'])
            if prev is None or h.get('force_N', 0.0) > prev.get('force_N', 0.0):
                best[h['prim']] = h
        # cache the net contact magnitude on the carton this tick for the weld-reaction estimate
        self._last_net_contact = sum(h.get('force_N', 0.0) for h in best.values())
        return list(best.values())

    def weld_reaction_force(self):
        """Best-effort magnitude (N) of the SIM_ATTACH_FALLBACK fixed-joint reaction this tick.

        The welded carton is a FREE rigid body held only by the fixed joint, so Newton on the carton
        gives F_weld = m*a - m*g - F_contact. Quasi-statically (a~0) the weld carries the carton weight
        PLUS whatever a link-contact fight pushes back, so |F_weld| ~ m*g + |contact force on the
        carton|. ``get_net_contact_forces`` is unavailable on SingleRigidPrim in this build, so the
        contact magnitude is taken from the SAME carton PhysX contact reports the per-link breakdown
        uses (a magnitude upper bound, labelled as such in the report)."""
        return self._carton_mass * 9.81 + float(getattr(self, '_last_net_contact', 0.0))

    def applied_efforts(self):
        """(efforts, caps) 6-vectors in ARM_JOINTS order: per-joint APPLIED/measured drive effort (Nm)
        this tick + the current max-effort cap. The measured joint effort is the torque PhysX applied
        through the position drive (clamped to the cap); a joint pinned at its cap while short of goal
        is drive-saturated (inbox 015). () if the build cannot report it."""
        try:
            e = self.robot.get_measured_joint_efforts()
        except Exception:
            try:
                e = self.robot.get_applied_joint_efforts()
            except Exception:
                return ()
        try:
            efforts = np.asarray(e, float).reshape(-1)[self.arm_idx]
            caps = np.asarray(self.arm_ctrl.get_max_efforts(), float).reshape(-1)[self.arm_idx]
            return (efforts.tolist(), caps.tolist())
        except Exception:
            return ()

    def arm_link_clearances(self):
        """Signed carton-AABB gap (m) to each of the robot's OWN arm links -- the run-250 self-
        collision check (carton hanging under the tool vs the upper arm / forearm)."""
        try:
            carton = self._carton_world_aabb()
        except Exception:
            return {}
        out = {}
        for name, path in self.arm_links.items():
            try:
                out[name] = self._aabb_signed_gap(carton, self._obstacle_aabb(path))
            except Exception:
                continue
        return out

    def probe_pinned_config(self, config_joints, carton_offset, *, hold_steps=40):
        """Pinned-config contact dump (inbox 004/005): weld the carton under the tool at
        ``carton_offset`` m along tool-z, hold the arm STATICALLY at ``config_joints`` (the measured
        run-250 jam config), step, and report every PhysX prim the carton contacts + the per-arm-link
        and per-obstacle AABB clearance + the carton world centre. Names the pinning collider."""
        import numpy as _np
        self.contact._release_fallback()
        self.mobile.release_brake(); self.mobile.release()
        full = _np.asarray(self.robot.get_joint_positions(), float).copy()
        full[self.arm_idx] = _np.asarray(config_joints, float)
        self.robot.set_joint_positions(full)
        self.robot.set_joint_velocities(_np.zeros(len(full)))
        g = self.gate
        g.faulted = False; g.hold_target = None; g.last_stamp = -1.
        g._host_now = float('-inf'); g.started = False
        g.apply(_np.asarray(config_joints, float), _np.zeros(6), g.indices)
        self.world.step(render=self.render)   # settle the arm at the config before placing the carton
        carton_world = None
        if not self.no_carton:
            pad = self._pad_world()
            rot = pad.ExtractRotationMatrix()
            origin = pad.ExtractTranslation()
            zaxis = _np.array([rot[0][2], rot[1][2], rot[2][2]], float)
            centre = _np.array([origin[0], origin[1], origin[2]], float) + float(carton_offset) * zaxis
            q = pad.ExtractRotationQuat()
            self.carton.set_world_pose(position=centre,
                                       orientation=_np.array([q.GetReal(), *[float(v) for v in q.GetImaginary()]], float))
            self.carton.set_linear_velocity(_np.zeros(3)); self.carton.set_angular_velocity(_np.zeros(3))
            self.contact._fallback_weld(self.carton_path)
            carton_world = centre.tolist()
        self._contact_buf = []
        contacts = {}
        for _ in range(hold_steps):
            g.apply(_np.asarray(config_joints, float), _np.zeros(6), g.indices)
            self.world.step(render=self.render)
            for h in self.probe_contacts():
                contacts.setdefault(h['prim'], h.get('position'))
        try:
            cp, _ = self.carton.get_world_pose()
            carton_world = [float(v) for v in cp]
        except Exception:
            pass
        return dict(config=list(config_joints), carton_offset=float(carton_offset),
                    carton_world_centre=carton_world,
                    contacts=[{'prim': p, 'position': pos} for p, pos in contacts.items()],
                    arm_link_clearance_mm={k: round(v * 1000, 1) for k, v in self.arm_link_clearances().items()},
                    obstacle_clearance_mm={k: round(v * 1000, 1) for k, v in self.clearance_snapshot().items()})

    # ---- pose helpers
    def _pad_world(self):
        m = self._UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.pad_path))
        return m

    def _carton_in_pad(self):
        UsdGeom = self._UsdGeom
        cache = UsdGeom.XformCache()
        pad = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.pad_path))
        car = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.carton_path))
        rel = car * pad.GetInverse()
        t = rel.ExtractTranslation()
        return np.array([t[0], t[1], t[2]], float)

    # ---- ArmStage interface
    def measured(self):
        return self.gate.measured()

    def measured_velocity(self):
        v = np.asarray(self.robot.get_joint_velocities(joint_indices=self.arm_idx), float)
        return v

    def command(self, q, v, stamp):
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        self.gate.command(names=list(ARM_JOINTS), positions=list(q), velocities=list(v),
                          efforts=[0.0] * 6, stamp=float(stamp), now=float(stamp))

    def step(self, dt):
        self.world.step(render=self.render)

    def sim_time(self):
        return float(self.world.current_time)

    def carton_in_tool(self):
        return self._carton_in_pad()

    def base_xy(self):
        from twin.restock.isaac_port import body_measurement
        pose, _z, _v, _w = body_measurement(self.robot_body)
        return np.array([pose.x, pose.y], float)

    # ---- bench setup / per-trial reset
    def _set_drive_type_usd(self, dtype, per_joint):
        """Author the arm joints' angular DriveAPI TYPE + per-joint stiffness/damping/maxforce on USD,
        then world.reset() so PhysX re-parses the drives (drive TYPE is a parse-time property). Only
        called when the drive type actually changes (expensive). Returns True if a reset happened."""
        from pxr import UsdPhysics
        kps, kds, effs = per_joint
        for path, kp, kd, eff in zip(self.arm_joint_prims, kps, kds, effs):
            if not path:
                continue
            prim = self.stage.GetPrimAtPath(path)
            drive = UsdPhysics.DriveAPI.Get(prim, 'angular')
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, 'angular')
            drive.GetTypeAttr().Set(dtype) if drive.GetTypeAttr() else drive.CreateTypeAttr(dtype)
            (drive.GetStiffnessAttr() or drive.CreateStiffnessAttr()).Set(float(kp))
            (drive.GetDampingAttr() or drive.CreateDampingAttr()).Set(float(kd))
            (drive.GetMaxForceAttr() or drive.CreateMaxForceAttr()).Set(float(eff))
        self.world.reset()   # re-parse the authored drives (type change needs a re-init)
        self._current_drive_type = dtype
        return True

    def apply_gains(self, gains: bench.DriveGains):
        kps, kds, effs = gains.per_joint()
        if gains.drive_type != self._current_drive_type:
            # drive TYPE change: author on USD + reset (parse-time), then the runtime set_gains below
            # refines the values on the freshly-parsed drives
            self._set_drive_type_usd(gains.drive_type, (kps, kds, effs))
        rkp, rkd = self.arm_ctrl.get_gains()
        rkp = np.asarray(rkp, float).copy(); rkd = np.asarray(rkd, float).copy()
        for n, i in enumerate(self.arm_idx):
            rkp[int(i)] = kps[n]; rkd[int(i)] = kds[n]
        self.arm_ctrl.set_gains(rkp, rkd)
        self.arm_ctrl.set_max_efforts(np.asarray(effs, float), joint_indices=self.arm_idx)
        gkp, gkd = self.arm_ctrl.get_gains()
        gme = self.arm_ctrl.get_max_efforts()
        gkp = np.asarray(gkp, float).reshape(-1); gkd = np.asarray(gkd, float).reshape(-1)
        gme = np.asarray(gme, float).reshape(-1)
        print('ARMBENCH_GAINS type=%s kp=%s kd=%s eff=%s' % (
            self._current_drive_type,
            [round(float(gkp[int(i)]), 1) for i in self.arm_idx],
            [round(float(gkd[int(i)]), 1) for i in self.arm_idx],
            [round(float(gme[int(i)]), 1) for i in self.arm_idx]), flush=True)

    def _place_carton_at_offset(self):
        """Teleport the carton RigidBody so its origin sits at pad + R_tcp @ CARTON_OFFSET_TCP with the
        pad's orientation -- the faithful attached-carton pose (world_obstacles.carton_spheres_world)."""
        pad = self._pad_world()
        rot = pad.ExtractRotationMatrix()
        origin = pad.ExtractTranslation()
        off = np.asarray(CARTON_OFFSET_TCP, float)
        R = np.array([[rot[0][0], rot[0][1], rot[0][2]],
                      [rot[1][0], rot[1][1], rot[1][2]],
                      [rot[2][0], rot[2][1], rot[2][2]]], float)
        centre = np.array([origin[0], origin[1], origin[2]], float) + R @ off
        q = pad.ExtractRotationQuat()
        self.carton.set_world_pose(position=centre,
                                   orientation=np.array([q.GetReal(), *[float(v) for v in q.GetImaginary()]], float))
        self.carton.set_linear_velocity(np.zeros(3))
        self.carton.set_angular_velocity(np.zeros(3))

    def _pad_carton_gap_mm(self):
        pad = self._pad_world().ExtractTranslation()
        cp, _ = self.carton.get_world_pose()
        cp = np.asarray(cp, float)
        return float(np.linalg.norm(np.array([pad[0], pad[1], pad[2]], float) - cp) * 1000)

    def reset_to_grasp(self, gains: bench.DriveGains, settle_steps=30):
        """Faithful per-trial reset: set the arm to the measured GRASP pose (source_contact), where
        the gripper pad is already at the authored carton by IK, then weld the carton to the gripper
        IN PLACE (no manual carton teleport -- the weld captures the real grasp relationship, exactly
        as the mission's SIM_ATTACH does). The subsequent 'lift' move drives grasp->approach so the
        carton is carried by the weld. Re-applies the trial gains and the airborne-carry brake, then
        settles. A bench teleport of the ARM only; never a mission action."""
        # 1. release any prior weld + brake so the re-weld/brake re-anchor cleanly at the grasp pose
        self.contact._release_fallback()
        self.mobile.release_brake()
        self.mobile.release()
        # 2. apply the trial gains FIRST -- a drive-TYPE change does a world.reset() (re-homing the
        #    arm), so it must precede the grasp teleport below
        self.apply_gains(gains)
        # 3. teleport the arm to the GRASP pose at rest (the carton stays at its authored pose;
        #    the pad lands on it by construction of the source_contact IK)
        reset_pose = self.lift_joints if self.start_at_lift else self.grasp_joints
        full = np.asarray(self.robot.get_joint_positions(), float).copy()
        full[self.arm_idx] = reset_pose
        self.robot.set_joint_positions(full)
        self.robot.set_joint_velocities(np.zeros(len(full)))
        self.carton.set_linear_velocity(np.zeros(3))
        self.carton.set_angular_velocity(np.zeros(3))
        # 4. reset the gate's transient fault/freshness state (config untouched) and hold grasp
        g = self.gate
        g.faulted = False; g.hold_target = None; g.last_stamp = -1.
        g._host_now = float('-inf'); g.started = False
        g._contact_hold = False; g._contact_hold_target = None
        g.apply(reset_pose, np.zeros(6), g.indices)
        # 5. place the carton at the measured side-grip offset in the gripper_pad_tcp frame
        #    (centre = pad + R_tcp @ (0,0,0.20), the model world_obstacles.carton_spheres_world uses),
        #    then weld it there -- robust to a slightly-off grasp pose (skipped in --no-carton control).
        if not self.no_carton:
            self._place_carton_at_offset()
            self.contact._fallback_weld(self.carton_path)
        self._weld_rel0 = self._carton_in_pad()
        print('ARMBENCH_WELD carton_in_pad=%s (target offset_tcp=%s)'
              % (self._weld_rel0.tolist(), list(CARTON_OFFSET_TCP)), flush=True)
        # 5. brake the base in the airborne-carry state (yaw-lock D6 over the wheel hold)
        self.mobile.hold()
        self.mobile.engage_brake()
        # 6. settle: hold grasp for a few ticks so the weld + brake quiesce
        for _ in range(settle_steps):
            g.apply(reset_pose, np.zeros(6), g.indices)
            self.world.step(render=self.render)


def _run_trial(stage: IsaacCarryStage, spec: bench.TrialSpec, goals, dt, max_sim_s):
    stage.reset_to_grasp(spec.gains)
    moves = []
    start = np.asarray(stage.measured(), float)
    for phase, goal in goals:
        m = bench.run_move(stage, start, goal, spec.carry, spec.tolerances,
                           label=phase, dt=dt, max_sim_s=max_sim_s)
        moves.append(m)
        j, err = m.worst_tracking()
        _names = ['pan', 'lift', 'elbow', 'wrist1', 'wrist2', 'wrist3']
        _sat = [_names[k] for k in range(6) if m.effort_saturated[k]]
        print('ARMBENCH_MOVE trial=%s phase=%s outcome=%s worst_j=%d worst_track=%.4f '
              'wrist1_track=%.4f term_vel_max=%.4f settle=%s hit=%s reason=%s' % (
                  spec.label, phase, m.outcome, j, err, m.peak_tracking_error[bench.WRIST_1],
                  max(abs(x) for x in m.terminal_velocity), m.settle_time_s,
                  [p.rsplit('/', 1)[-1] for p in m.contact_prims], m.reason), flush=True)
        print('ARMBENCH_EFFORT trial=%s phase=%s applied=%s cap=%s saturated=%s term_poserr=%s' % (
            spec.label, phase, [round(x, 1) for x in m.peak_applied_effort],
            [round(x, 1) for x in m.effort_cap], (_sat or 'none'),
            [round(x, 3) for x in m.terminal_pos_error]), flush=True)
        if m.first_contact:
            fc = m.first_contact
            print('ARMBENCH_FIRST_CONTACT trial=%s phase=%s prim=%s fold_fraction=%.2f wrist1=%.3f pos=%s'
                  % (spec.label, phase, fc['prim'], fc['fold_fraction'], fc['wrist1_rad'], fc['position']),
                  flush=True)
        for name, gap in sorted(m.min_clearance_m.items()):
            print('ARMBENCH_CLEARANCE trial=%s phase=%s obstacle=%s min_gap_mm=%.1f at_wrist1=%.3f'
                  % (spec.label, phase, name, gap * 1000, m.clearance_at_wrist1.get(name, float('nan'))),
                  flush=True)
        if not m.ok:
            break                                   # a failing move aborts the trial's sequence
        start = np.asarray(stage.measured(), float)  # next move starts from the measured end
    return moves


def _write_partial(output, summary):
    """Persist the running summary after each trial so a wall-cap kill keeps the results so far."""
    try:
        (output / 'bench.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    except Exception:
        pass


def _move_dict(m: bench.MoveMetrics):
    return dict(label=m.label, outcome=m.outcome, reason=m.reason, duration_s=m.duration_s,
                settle_time_s=m.settle_time_s, peak_tracking_error=m.peak_tracking_error,
                peak_joint_velocity=m.peak_joint_velocity, terminal_pos_error=m.terminal_pos_error,
                terminal_velocity=m.terminal_velocity, peak_carton_drift_m=m.peak_carton_drift_m,
                peak_base_displacement_m=m.peak_base_displacement_m, sim_elapsed_s=m.sim_elapsed_s,
                min_clearance_m=m.min_clearance_m, clearance_at_wrist1=m.clearance_at_wrist1,
                contact_prims=m.contact_prims, first_contact=m.first_contact,
                peak_contact_force_N=m.peak_contact_force_N, peak_weld_reaction_N=m.peak_weld_reaction_N,
                peak_applied_effort=m.peak_applied_effort, effort_cap=m.effort_cap,
                effort_saturated=m.effort_saturated, terminal_applied_effort=m.terminal_applied_effort)


# Candidate carry-shape reshapings (deltas on cart_tuck_shape joints 1-5: shoulder_lift, elbow,
# wrist_1, wrist_2, wrist_3) that raise the carton and/or fold wrist_1 less so it clears the cage.
POSE_DELTAS = [
    [0.0, 0.0, 0.0, 0.0, 0.0],      # baseline carry shape
    [0.0, 0.0, -0.5, 0.0, 0.0],     # fold wrist_1 less
    [0.0, 0.0, -1.0, 0.0, 0.0],
    [0.3, 0.0, 0.0, 0.0, 0.0],      # raise shoulder_lift (carton higher)
    [0.3, 0.0, -0.5, 0.0, 0.0],
    [0.5, -0.3, -0.5, 0.0, 0.0],    # raise + open elbow + fold less
    [0.3, -0.3, -1.0, 0.0, 0.0],
]


def build_jobs(args, lift_joints, tuck_shape, pan_joint0, extra):
    """Return the ordered list of (label, goals, spec) the bench runs. The diagnosis trials come
    first (the redirect), then the carry-pose sweep (primary), then an optional gain sweep."""
    profile = 'trapezoid' if args.replay else 'cubic'
    carry = bench.CarryProfile(peak_speed=args.carry_speed, peak_accel=args.carry_accel,
                               profile=profile)
    # The weld forms at GRASP; the first move drives grasp->lift (carton carried by the weld), then
    # the tuck/pan fold follows. Prepend 'lift' so every trial exercises the full carried sequence.
    lift_move = ('lift', [float(v) for v in lift_joints])
    base_goals = ([] if args.start_at_lift else [lift_move]) + bench.carry_goals(lift_joints, tuck_shape, pan_joint0, extra_goals=extra)
    jobs = []
    if args.diagnose or args.expect == 'reproduce' or args.reproduce:
        # reproduce the mission fold in the full scene with contact reporting, at BOTH gain
        # extremes -- proves the stall is gain-independent (rigid contact) and names the prim
        jobs.append(('ref_soft_run249', base_goals, bench.TrialSpec('ref_soft_run249', REF_SOFT, carry)))
        jobs.append(('ref_stiff_run248', base_goals, bench.TrialSpec('ref_stiff_run248', REF_STIFF, carry)))
    if args.pose_sweep:
        safe = bench.DriveGains(args.pose_kp, args.pose_kd, args.pose_eff)
        for dv in POSE_DELTAS:
            shape = bench.tuck_shape_variant(tuck_shape, dv)
            goals = ([] if args.start_at_lift else [lift_move]) + bench.carry_goals(lift_joints, shape, pan_joint0, extra_goals=extra)
            label = 'pose_' + '_'.join(f'{x:+g}' for x in dv)
            jobs.append((label, goals, bench.TrialSpec(label, safe, carry)))
    # Explicit gain SETS: each --gain-set is exactly ONE trial, so a stiffness can be PAIRED with its
    # own (near-critical) damping instead of the cartesian product --sweep produces. Format
    # "kp,kd,eff[,drive_type[,pj]]" (drive_type 'acceleration'|'force'; pj=1 -> per-joint UR10 efforts).
    for gs in (args.gain_set or []):
        parts = [p.strip() for p in gs.split(',')]
        if len(parts) < 3:
            raise ValueError(f'--gain-set needs at least kp,kd,eff: {gs!r}')
        kp, kd, eff = float(parts[0]), float(parts[1]), float(parts[2])
        dtype = parts[3] if len(parts) > 3 and parts[3] else 'acceleration'
        pj = len(parts) > 4 and parts[4] in ('1', 'true', 'pj', 'yes')
        label = f'set_{dtype[:3]}_kp{kp:g}_kd{kd:g}_eff{eff:g}' + ('_pj' if pj else '')
        jobs.append((label, base_goals, bench.TrialSpec(
            label, bench.DriveGains(kp, kd, eff, drive_type=dtype, per_joint_effort=pj), carry)))
    if args.sweep:
        carries = [carry]
        if args.carry_speeds:
            carries = [bench.CarryProfile(peak_speed=s, peak_accel=args.carry_accel,
                                          profile=carry.profile)
                       for s in args.carry_speeds]
        for spec in bench.sweep_grid([float(x) for x in args.stiffnesses],
                                     [float(x) for x in args.dampings],
                                     [float(x) for x in args.max_efforts], carries,
                                     drive_types=tuple(args.drive_types),
                                     per_joint_effort=args.per_joint_effort):
            jobs.append((spec.label, base_goals, spec))
    if not jobs:
        jobs.append(('baseline', base_goals, bench.TrialSpec('baseline', REF_SOFT, carry)))
    for _, _, s in jobs:
        s.validate()
    return jobs


def run(args):
    config = json.loads(args.runtime_config.read_text())
    scenario = json.loads(Path(config['scenario']).read_text())
    # The bench drives the ArmCommandGate IN-PROCESS (no host trajectory_host), so it does not
    # need the out-of-Isaac /clock+/tf+/joint_states relay NativeTransport would otherwise spawn.
    # Drop runtime_env so assemble() builds the transport graph without launching that subprocess.
    md = dict(config.get('masked_depth') or {})
    md.pop('runtime_env', None)
    config['masked_depth'] = md
    arm_cfg = config.get('armbench', {})
    lift_joints = (args.lift_joints if args.lift_joints is not None
                   else arm_cfg.get('lift_joints', DEFAULT_LIFT))
    grasp_joints = (args.grasp_joints if args.grasp_joints is not None
                    else arm_cfg.get('grasp_joints', DEFAULT_GRASP))
    if len(lift_joints) != 6 or len(grasp_joints) != 6:
        raise ValueError('lift_joints and grasp_joints must each be six arm joints (ARM_JOINTS order)')
    tuck_shape = (args.cart_tuck_shape if args.cart_tuck_shape is not None
                  else arm_cfg.get('cart_tuck_shape', CART_TUCK_SHAPE))
    pan_joint0 = arm_cfg.get('cart_pan_joint0', CART_PAN_JOINT0)
    extra = arm_cfg.get('extra_goals')   # {phase: 6-vector} for downstream place moves (optional)
    carton_dims = tuple(scenario.get('handling_unit', {}).get('dimensions', (0.6, 0.4, 0.16)))
    obstacle_override = arm_cfg.get('obstacle_prims')
    jobs = build_jobs(args, lift_joints, tuck_shape, pan_joint0, extra)
    specs = [s for _, _, s in jobs]

    args.output.mkdir(parents=True, exist_ok=True)
    finished = threading.Event()

    def interrupted(signum, frame):
        raise RuntimeError(f'bench interrupted by signal {signum}')
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

    summary = {'state': 'failed', 'reason': 'bench did not start',
               'grasp_joints': list(grasp_joints), 'lift_joints': list(lift_joints),
               'carton_dims': list(carton_dims),
               'carry': {'peak_speed': args.carry_speed, 'peak_accel': args.carry_accel},
               'note': ('Planning path differs from the live mission (carry trajectories generated '
                        'analytically, not by MoveIt/cuMotion); the drive, gate, carton weld and '
                        'tolerances are the mission\'s own. Clearance is carton-AABB vs obstacle-AABB; '
                        'contact is PhysX (hollow-geometry-aware) when available.'),
               'trials': []}
    app = None
    slot = (gpu_guards_no_flock(heavy_threshold_mb=args.heavy_threshold_mb) if args.no_gpu_lock
            else gpu_slot(heavy_threshold_mb=args.heavy_threshold_mb))
    try:
        with slot:
            watcher = threading.Thread(target=watch_ram, daemon=True)
            watcher.start()
            from isaacsim import SimulationApp
            app = SimulationApp({'headless': True, 'width': 640, 'height': 360})
            from twin.restock.physics_setup import assemble
            assembly = assemble(app=app, config=config, scenario=scenario, output=args.output / 'assembly')
            stage = IsaacCarryStage(assembly, lift_joints, grasp_joints, render=args.render,
                                    carton_dims=carton_dims, obstacle_prims=obstacle_override,
                                    no_carton=args.no_carton, arm_contacts=args.arm_contacts, start_at_lift=args.start_at_lift)
            summary['obstacles'] = stage.obstacles
            summary['arm_links'] = sorted(stage.arm_links)
            if args.pinned_config is not None:
                # inbox 004/005: dump carton contacts + per-arm-link clearance at the measured jam config
                probes = []
                for off in args.carton_offsets:
                    p = stage.probe_pinned_config(args.pinned_config, off)
                    probes.append(p)
                    print('ARMBENCH_PINNED offset=%.3f carton_world=%s contacts=%s arm_link_clear_mm=%s obstacle_clear_mm=%s'
                          % (off, p['carton_world_centre'], [c['prim'] for c in p['contacts']],
                             p['arm_link_clearance_mm'], p['obstacle_clearance_mm']), flush=True)
                summary['pinned_config_probes'] = probes
                _write_partial(args.output, summary)
            for label, goals, spec in (jobs if args.pinned_config is None else []):
                if not app.is_running():
                    raise RuntimeError('simulation app stopped')
                moves = _run_trial(stage, spec, goals, args.dt, args.max_sim_s)
                print('ARMBENCH_DIAGNOSIS trial=%s\n%s' % (label, bench.contact_summary(moves)), flush=True)
                summary['trials'].append(dict(
                    label=label,
                    goals=[p for p, _ in goals],
                    gains=dict(stiffness=spec.gains.stiffness, damping=spec.gains.damping,
                               max_effort=spec.gains.max_effort, drive_type=spec.gains.drive_type,
                               per_joint_effort=spec.gains.per_joint_effort),
                    carry=dict(peak_speed=spec.carry.peak_speed, peak_accel=spec.carry.peak_accel),
                    passed=bool(moves and all(m.ok for m in moves)),
                    moves=[_move_dict(m) for m in moves]))
                # persist after EVERY trial so a wall-cap kill still leaves the results so far
                summary['state'] = 'partial'
                _write_partial(args.output, summary)
            summary['state'] = 'complete'
    except Exception as exc:
        summary['reason'] = str(exc)
        (args.output / 'bench-error.txt').write_text(
            ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    finally:
        finished.set()
        if app is not None:
            try:
                app.close()
            except Exception:
                pass
    # results + ranking + table
    results = [dict(label=t['label'], spec=next(s for s in specs if s.label == t['label']),
                    moves=_rehydrate(t['moves'])) for t in summary['trials']]
    ranked = bench.summarize(results)
    summary['ranking'] = [dict(label=r['label'], robust=r['robust'],
                               score=(None if r['score'] == float('-inf') else r['score']))
                          for r in ranked]
    table = bench.metrics_table(results)
    summary['metrics_table'] = table
    (args.output / 'bench.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print('\n' + table + '\n', flush=True)
    robust = next((r for r in ranked if r['robust']), None)
    if robust is not None:
        s = robust['spec']
        mc = robust.get('min_clearance_m')
        print('ARMBENCH_RECOMMEND label=%s kp=%g kd=%g eff=%g peak_speed=%g peak_accel=%g '
              'min_clearance_mm=%s score=%.4f' % (
                  s.label, s.gains.stiffness, s.gains.damping, s.gains.max_effort,
                  s.carry.peak_speed, s.carry.peak_accel,
                  ('%.1f' % (mc * 1000)) if mc is not None else 'n/a', robust['score']), flush=True)
    else:
        print('ARMBENCH_RECOMMEND none robust (every trial failed to settle or hit an obstacle)', flush=True)
    return _exit_code(args, summary, results)


def _rehydrate(move_dicts):
    out = []
    for d in move_dicts:
        m = bench.MoveMetrics(label=d['label'], outcome=d['outcome'], reason=d['reason'],
                              duration_s=d['duration_s'], settle_time_s=d['settle_time_s'],
                              peak_tracking_error=d['peak_tracking_error'],
                              peak_joint_velocity=d['peak_joint_velocity'],
                              terminal_pos_error=d['terminal_pos_error'],
                              terminal_velocity=d['terminal_velocity'],
                              peak_carton_drift_m=d['peak_carton_drift_m'],
                              peak_base_displacement_m=d['peak_base_displacement_m'],
                              sim_elapsed_s=d['sim_elapsed_s'],
                              min_clearance_m=d.get('min_clearance_m', {}),
                              clearance_at_wrist1=d.get('clearance_at_wrist1', {}),
                              contact_prims=d.get('contact_prims', []),
                              first_contact=d.get('first_contact'),
                              peak_contact_force_N=d.get('peak_contact_force_N', {}),
                              peak_weld_reaction_N=d.get('peak_weld_reaction_N', 0.0),
                              peak_applied_effort=d.get('peak_applied_effort', [0.0] * 6),
                              effort_cap=d.get('effort_cap', [0.0] * 6),
                              effort_saturated=d.get('effort_saturated', [False] * 6),
                              terminal_applied_effort=d.get('terminal_applied_effort', [0.0] * 6))
        out.append(m)
    return out


def _trial_moves(results, label):
    return next((r['moves'] for r in results if r['label'] == label), [])


def _exit_code(args, summary, results):
    if summary['state'] != 'complete':
        return 2
    if args.expect == 'report':
        return 0
    if args.expect == 'all-settle':
        ok = bool(results) and all(m.ok for r in results for m in r['moves']) \
            and all(r['moves'] for r in results)
        return 0 if ok else 1
    if args.expect == 'reproduce':
        # The redirect's hypothesis: the fold stall is GAIN-INDEPENDENT (a rigid contact), so the
        # carry fold must FAIL at BOTH gain extremes, with wrist_1 pinned off its goal. Success =
        # the gain-independent stall reproduced. The diagnosis (which prim / clearance) is in the
        # summary; print whether a physical contact was named.
        stiff = _trial_moves(results, 'ref_stiff_run248')
        soft = _trial_moves(results, 'ref_soft_run249')

        def stalled(moves):
            return any(m.outcome in ('tracking_fault', 'goal_miss')
                       and abs(m.terminal_pos_error[bench.WRIST_1]) > 0.5 for m in moves)
        gain_independent = bool(stiff) and bool(soft) and stalled(stiff) and stalled(soft)
        contacts = sorted({p for moves in (stiff, soft) for m in moves for p in m.contact_prims})
        clearances = {name: gap for moves in (stiff, soft) for m in moves
                      for name, gap in m.min_clearance_m.items()}
        negative = sorted(n for n, g in clearances.items() if g < 0)
        print('ARMBENCH_REPRODUCE gain_independent_stall=%s contacted=%s negative_clearance=%s'
              % (gain_independent, contacts, negative), flush=True)
        return 0 if gain_independent else 1
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-config', type=Path, required=True,
                   help='the mission runtime config (same one run_mobile_restock --execute consumes)')
    p.add_argument('--output', type=Path, required=True, help='scratch output dir (must not pre-exist content)')
    p.add_argument('--lift-joints', type=float, nargs=6, default=None,
                   help='arm lift pose source_approach (ARM_JOINTS order); default measured value')
    p.add_argument('--grasp-joints', type=float, nargs=6, default=None,
                   help='arm grasp pose source_contact where the carton welds (ARM_JOINTS order)')
    p.add_argument('--expect', choices=('report', 'all-settle', 'reproduce'), default='report')
    p.add_argument('--diagnose', action='store_true',
                   help='trial-1: fold the carton in the full scene at BOTH ref gains with contact '
                        'reporting (reproduce the mission stall + name the obstacle)')
    p.add_argument('--reproduce', action='store_true', help='alias: prepend the run-248/run-249 reference trials')
    p.add_argument('--pose-sweep', action='store_true',
                   help='sweep carry-shape (tuck) variants ranked by cart/rack clearance (primary axis)')
    p.add_argument('--pose-kp', type=float, default=3.0e5, help='fixed drive stiffness for the pose sweep')
    p.add_argument('--pose-kd', type=float, default=5.0e3, help='fixed drive damping for the pose sweep')
    p.add_argument('--pose-eff', type=float, default=1000.0, help='fixed drive max effort for the pose sweep')
    p.add_argument('--gain-set', action='append', default=None, metavar='kp,kd,eff[,type[,pj]]',
                   help='one trial at an EXPLICIT gain set (repeatable); pairs a stiffness with its own '
                        'damping (e.g. near-critical) instead of the --sweep cartesian product. '
                        'type=acceleration|force, pj=1 for UR10 per-joint efforts (wrists ~56 Nm).')
    p.add_argument('--sweep', action='store_true', help='run the gain/carry grid below (secondary axis)')
    p.add_argument('--drive-types', nargs='+', choices=('acceleration', 'force'),
                   default=['acceleration'], help='drive-type axis (integration msg 006)')
    p.add_argument('--per-joint-effort', action='store_true',
                   help='cap each joint at its UR10 effort (wrists ~56 Nm) and scale stiffness/damping '
                        'per joint, instead of a uniform ceiling')
    p.add_argument('--stiffnesses', type=float, nargs='+', default=[1.0e6, 5.0e5, 3.0e5, 1.0e5])
    p.add_argument('--dampings', type=float, nargs='+', default=[2.0e3, 5.0e3, 1.0e4])
    p.add_argument('--max-efforts', type=float, nargs='+', default=[400.0, 1000.0, 2000.0])
    p.add_argument('--cart-tuck-shape', type=float, nargs=5, default=None,
                   help='override the tuck shape joints 1-5 (e.g. the reshaped -2.3 1.0 0.9 0.8 -1.2545)')
    p.add_argument('--replay', action='store_true',
                   help='reconstruct the fold as a synchronized TRAPEZOIDAL (MoveIt-TOTG-shaped) arrival '
                        '-- ramp to cruise, flat cruise, hard terminal decel -- so the arm ARRIVES '
                        'OVERSHOOTING as the live mission fold does, reproducing the terminal-hold '
                        'effort-saturation limit cycle a gentle rest-to-rest cubic misses. Implies the '
                        'mission arm envelope (0.3 rad/s / 1.0 rad/s^2) unless --carry-speed/--carry-accel '
                        'are given.')
    p.add_argument('--carry-speed', type=float, default=None, help='carry move peak joint speed (rad/s); '
                   'default 0.05 (cubic) or 0.3 (--replay, the mission trajectory_host max_speed)')
    p.add_argument('--carry-accel', type=float, default=None, help='carry move peak joint accel (rad/s^2); '
                   'default 0.03 (cubic) or 1.0 (--replay, the mission max_accel = max_speed*0.5/0.15)')
    p.add_argument('--carry-speeds', type=float, nargs='+', default=None,
                   help='sweep these carry peak speeds instead of the single --carry-speed')
    p.add_argument('--pinned-config', type=float, nargs=6, default=None,
                   help='dump carton PhysX contacts + per-arm-link clearance at this static arm config '
                        '(the measured run-250 jam config); names the pinning collider (inbox 004/005)')
    p.add_argument('--carton-offsets', type=float, nargs='+', default=[0.20],
                   help='carton-centre offsets (m) along tool-z to bracket for the pinned-config probe')
    p.add_argument('--start-at-lift', action='store_true',
                   help='start the arm directly at --lift-joints and weld the carton there (skip the '
                        'grasp->lift move), for a clean short terminal settle at a collision-free pose')
    p.add_argument('--arm-contacts', action='store_true',
                   help='OPT-IN: also PhysX-report arm/chassis self-collisions (names arm-vs-body '
                        'contacts) -- but this invalidates the articulation tensor view, so it CANNOT '
                        'be combined with a gain sweep (get_gains fails); use carton_arm_fold_clearance '
                        'for carton-vs-arm instead')
    p.add_argument('--no-carton', action='store_true',
                   help='control: run the fold / pinned config with NO carton welded (isolates the '
                        'carried payload as the cause of the stall)')
    p.add_argument('--dt', type=float, default=1.0 / 60.0)
    p.add_argument('--max-sim-s', type=float, default=90.0, help='per-move sim-time budget')
    p.add_argument('--heavy-threshold-mb', type=int, choices=(1500, 2500), default=1500)
    p.add_argument('--no-gpu-lock', action='store_true',
                   help='skip the internal flock because an OUTER `flock -w /tmp/aile-gpu.lock` already '
                        'holds it (fleet blocking-wait discipline); the VRAM/RAM/Blender guards still run')
    p.add_argument('--render', action='store_true', help='render each physics step (slower; only for evidence)')
    args = p.parse_args()
    # Resolve the carry envelope: --replay defaults to the live mission arm envelope (trajectory_host
    # max_speed 0.3 / max_accel = 0.3*0.5/0.15 = 1.0) so the reconstructed arrival matches the fold;
    # the gentle cubic keeps its original slow defaults. An explicit flag always wins.
    if args.carry_speed is None:
        args.carry_speed = 0.3 if args.replay else 0.05
    if args.carry_accel is None:
        args.carry_accel = 1.0 if args.replay else 0.03
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
