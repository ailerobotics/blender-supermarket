"""GPU-side of the arm collision bench: author the scenario obstacles into the Isaac stage, and map
a bounded run's measured metrics back into the SAME per-scenario schema the CPU mock-ESDF mode emits
(sim-arm-collision-bench-v1).

Design (per the scout steer): do NOT write a second Isaac-boot harness.  ``arm_carry_bench`` already
boots Isaac ONCE and runs seconds-long arm-leg trials against discovered obstacle prims, probing
per-trial AABB clearance + PhysX contact.  This module only adds the planted obstacles:

  * :func:`author_obstacles` authors each scenario obstacle as a USD Cube/Sphere with a CollisionAPI
    (a STATIC one as a collider; a MOVING one as a kinematic rigid body the runner teleports along the
    scripted path) at the obstacle's ``ur_base_link``-frame pose, as children of a bench root carrying
    the arm base world transform.  It returns ``{name: prim_path}`` -- exactly the ``obstacle_prims``
    override ``IsaacCarryStage`` consumes, so the existing clearance/contact probe measures them.
    Pure ``pxr`` (usd-core) -- CPU-testable without Isaac/GPU.

  * :func:`ingest_trial` maps the measured per-leg trial metrics (min clearance, contact prim/force,
    re-plan count, cycle-time, carton tilt) into an :class:`arm_collision_bench.ScenarioResult`, so the
    GPU numbers drop into the same report/CSV as the CPU numbers (the BEFORE/AFTER comparison is
    apples-to-apples).

The live boot is the bounded GPU proof (``run_arm_collision_bench_gpu.py``); it is gated by the
canonical GPU lock + ``gpu_budget`` + the Blender hard-refuse and runs <= 20 min, never ``--force``.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from . import arm_collision_bench as bench


def find_base_prim(stage, name='ur_base_link'):
    """Path of the arm base-link prim (the ``ur_base_link`` the curobo spheres are FK'd from) in a
    booted assembly stage, so obstacles authored in base-frame coords land correctly in the world."""
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def _base_world_transform(stage, base_prim_path: str):
    """4x4 world transform of the arm base prim (so obstacles authored in ur_base_link coords land in
    the right world place).  Identity if the prim is missing (a bare test stage)."""
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(base_prim_path)
    if not prim or not prim.IsValid():
        return np.eye(4)
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array([[m[r][c] for c in range(4)] for r in range(4)], float).T


def author_obstacles(stage, scenario: bench.Scenario, *, base_prim_path='/World/ur_base_link',
                     root='/ArmCollisionBench', t=0.0) -> Dict[str, str]:
    """Author every scenario obstacle into ``stage`` at time ``t`` (obstacle poses are ur_base_link
    frame).  Static obstacles get a CollisionAPI; moving ones additionally get a kinematic RigidBodyAPI
    the runner advances along ``Motion.position_at``.  Returns ``{obstacle_name: prim_path}``.

    An ``observed=False`` obstacle is still authored (it physically exists and the contact probe must
    see a real strike) -- the ESDF blindness is a SENSING property, modelled on the perception side by
    masking it out of the depth feed, NOT by removing the body."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    Twb = _base_world_transform(stage, base_prim_path)
    UsdGeom.Xform.Define(stage, root)
    out = {}
    for ob in scenario.obstacles:
        path = '%s/%s' % (root, _safe(ob.name))
        c_base = np.asarray(ob.center_at(t), float)
        c_world = (Twb[:3, :3] @ c_base) + Twb[:3, 3]
        if ob.kind == 'sphere':
            geom = UsdGeom.Sphere.Define(stage, path)
            geom.CreateRadiusAttr(float(ob.radius))
            geom.CreateExtentAttr([(-ob.radius,) * 3, (ob.radius,) * 3])
        else:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
            half = [s / 2.0 for s in ob.size]
            geom.CreateExtentAttr([(-h for h in half), tuple(half)])
            UsdGeom.Xformable(geom).AddScaleOp().Set(Gf.Vec3f(*[float(s) for s in ob.size]))
        UsdGeom.Xformable(geom).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in c_world]))
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        if ob.motion.kind != 'static':
            rb = UsdPhysics.RigidBodyAPI.Apply(geom.GetPrim())
            rb.CreateKinematicEnabledAttr(True)  # teleported along the scripted path, not dynamics-driven
        geom.GetPrim().CreateAttribute('arm_bench:observed', Sdf.ValueTypeNames.Bool).Set(bool(ob.observed))
        geom.GetPrim().CreateAttribute('arm_bench:class', Sdf.ValueTypeNames.String).Set(
            ob.motion.obstacle_class)
        out[ob.name] = path
    return out


def moving_obstacle_world(scenario: bench.Scenario, stage, t: float, *, base_prim_path='/World/ur_base_link'):
    """``{prim_path_leaf -> world (x,y,z)}`` of each MOVING obstacle at sim time ``t`` -- the runner
    teleports the kinematic prims here each step."""
    Twb = _base_world_transform(stage, base_prim_path)
    out = {}
    for ob in scenario.obstacles:
        if ob.motion.kind == 'static':
            continue
        c = (Twb[:3, :3] @ np.asarray(ob.center_at(t), float)) + Twb[:3, 3]
        out[_safe(ob.name)] = tuple(float(v) for v in c)
    return out


def _safe(name: str) -> str:
    return ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in str(name)) or 'obstacle'


def ingest_trial(scenario: bench.Scenario, measured: dict,
                 thresholds: bench.Thresholds = bench.Thresholds()) -> bench.ScenarioResult:
    """Map a bounded run's MEASURED metrics for one scenario leg into a :class:`ScenarioResult`, so the
    GPU numbers share the CPU report/CSV schema.

    ``measured`` keys (all from the arm_carry_bench probe / mission logs):
      ``min_true_clearance_m`` (AABB), ``min_perceived_clearance_m`` (from the served ESDF, if read),
      ``contact`` (bool, from PhysX contact reports), ``gate_behavior`` (the gate's verdict this run),
      ``n_replans``, ``extra_cycle_time_s``, ``tilt_peak_deg`` (CARTON_TILT), ``worst_obstacle``.
    """
    true_c = float(measured.get('min_true_clearance_m', float('inf')))
    perc_c = float(measured.get('min_perceived_clearance_m', true_c))
    contact = bool(measured.get('contact', true_c < 0.0))
    behavior = str(measured.get('gate_behavior', 'none'))
    n_replans = int(measured.get('n_replans', 0))
    extra = float(measured.get('extra_cycle_time_s', 0.0))
    tilt = float(measured.get('tilt_peak_deg', 0.0))
    overhead = extra / scenario.nominal_cycle_s()
    executed_free = not contact
    executed_min = true_c if executed_free else min(true_c, 0.0)
    verdict_match = behavior == scenario.expected_behavior
    min_req = (thresholds.min_true_clearance_m if thresholds.min_true_clearance_m is not None
               else scenario.margin_m)
    fails = []
    if thresholds.no_contact and contact:
        fails.append('contact: PhysX contact reported (min true %.1f mm)' % (true_c * 1e3))
    if executed_free and executed_min < min_req:
        fails.append('clearance: executed min true %.1f mm < required %.1f mm'
                     % (executed_min * 1e3, min_req * 1e3))
    if overhead > thresholds.cycle_time_overhead_frac:
        fails.append('cycle-time: overhead %.0f%% > %.0f%%' % (overhead * 100,
                     thresholds.cycle_time_overhead_frac * 100))
    if thresholds.require_verdict_match and not verdict_match:
        fails.append('verdict: gate %r != expected %r' % (behavior, scenario.expected_behavior))
    return bench.ScenarioResult(
        scenario_id=scenario.id, arm_leg=scenario.arm_leg, expected_behavior=scenario.expected_behavior,
        gate_behavior=behavior, verdict_match=verdict_match, collision_free_nominal=not contact,
        min_true_clearance_m=round(true_c, 5), min_perceived_clearance_m=round(perc_c, 5),
        esdf_error_m=round(perc_c - true_c, 5), worst_obstacle=measured.get('worst_obstacle'),
        n_replans=n_replans, extra_cycle_time_s=round(extra, 3),
        cycle_time_overhead_frac=round(overhead, 4), tilt_peak_deg=round(tilt, 4),
        executed_collision_free=executed_free, executed_min_true_clearance_m=round(executed_min, 5),
        passed=not fails, fail_reasons=tuple(fails), reason='gpu run (arm_carry_bench stage)')
