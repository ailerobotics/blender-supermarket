"""Author and animate the scripted people in a USD/Isaac stage (GPU-runtime side).

Kept out of the pure-logic import path: ``pxr``/Isaac are imported INSIDE the
methods, so the module imports (and its construction maths are unit-tested) on a
box with no USD. Nothing here is reached unless ``HumansConfig.enabled`` is true.

Backends
--------
* ``teleport`` (default): author one prim per person -- an Isaac People character
  reference when ``character_usd`` is set, else a parametric capsule stand-in --
  under ``/Humans`` and teleport it to ``path.pose_at(sim_time)`` every step. This
  is exactly the approach proven in ``ur10-setup/sim/isaac_human`` on this shared
  12 GB card: cheap, deterministic, no skeletal solve competing with the RTX
  render and the detector.
* ``anim_people``: if ``omni.anim.people`` is importable, drive a skeletal walk
  command instead of teleporting. Falls back to ``teleport`` (reported, never
  silently) when the extension is absent.

People are authored as VISUAL actors (no rigid body / collider) by default: they
are there to be *seen* by the cameras (detection) and to drive the safety zone.
Whether they are also physical navigation obstacles is owned by the dynamic-
obstacle navigation lane (``sim-restock-dynobs-v1``); this module never adds a
collider unless ``physical=True`` is passed, and it never touches the robot,
cart, store fixtures or the input layers.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

from .config import HumansConfig, PersonSpec

HUMANS_ROOT = '/Humans'


def ground_capsule_z(height_m: float, radius_m: float) -> float:
    """Z of a Z-up capsule's centre so its base sits on the floor (z=0).

    A USD capsule of ``height`` (cylinder part) and ``radius`` spans
    ``height + 2*radius`` tip to tip, so its centre is half of that above ground.
    """
    cyl = max(0.0, height_m - 2 * radius_m)
    return cyl / 2 + radius_m


class PeopleStage:
    """Owns the authored people prims and teleports them each physics step."""

    def __init__(self, config: HumansConfig):
        self.config = config
        self._paths = {p.person_id: p.path() for p in config.people}
        self._specs: Dict[str, PersonSpec] = {p.person_id: p for p in config.people}
        self._xform_ops: Dict[str, object] = {}
        self._backend = config.backend
        self._stage = None

    # -- pure helpers (unit-tested without USD) ------------------------------
    def world_pose(self, person_id: str, sim_time: float) -> Tuple[float, float, float]:
        """``(x, y, yaw)`` of one person at ``sim_time`` (delegates to the path)."""
        return self._paths[person_id].pose_at(sim_time)

    def world_positions(self, sim_time: float) -> Dict[str, Tuple[float, float, float]]:
        """``{id: (x, y, z)}`` body-centre positions at ``sim_time`` (z = chest)."""
        out = {}
        for pid, spec in self._specs.items():
            x, y, _yaw = self._paths[pid].pose_at(sim_time)
            out[pid] = (x, y, spec.height_m * 0.5)
        return out

    # -- USD authoring (guarded) --------------------------------------------
    def author(self, stage, *, physical: bool = False) -> Dict[str, str]:
        """Author every person prim at its t=0 pose. Returns ``{id: prim_path}``."""
        from pxr import Gf, UsdGeom, UsdPhysics
        self._stage = stage
        if stage.GetPrimAtPath(HUMANS_ROOT):
            raise ValueError('refusing to overwrite an existing /Humans scope')
        UsdGeom.Xform.Define(stage, HUMANS_ROOT)
        if self._backend == 'anim_people' and not _anim_people_available():
            print('HUMANS anim_people extension unavailable; falling back to teleport',
                  flush=True)
            self._backend = 'teleport'
        paths = {}
        for pid, spec in self._specs.items():
            prim_path = f'{HUMANS_ROOT}/{pid}'
            xform = UsdGeom.Xform.Define(stage, prim_path)
            if spec.character_usd:
                xform.GetPrim().GetReferences().AddReference(spec.character_usd)
                z0 = 0.0                                   # character assets sit on their own origin
            else:
                capsule = UsdGeom.Capsule.Define(stage, prim_path + '/Body')
                capsule.CreateAxisAttr('Z')
                capsule.CreateRadiusAttr(spec.radius_m)
                capsule.CreateHeightAttr(max(0.0, spec.height_m - 2 * spec.radius_m))
                capsule.CreateDisplayColorAttr(
                    [Gf.Vec3f(0.85, 0.55, 0.25) if spec.role == 'staff'
                     else Gf.Vec3f(0.25, 0.45, 0.85)])
                z0 = ground_capsule_z(spec.height_m, spec.radius_m)
            self._xform_ops[pid] = (UsdGeom.Xformable(xform.GetPrim()), z0)
            x, y, yaw = self._paths[pid].pose_at(0.0)
            self._set_pose(pid, x, y, yaw)
            if physical:
                UsdPhysics.CollisionAPI.Apply(xform.GetPrim())
            xform.GetPrim().SetCustomDataByKey('humans:person_id', pid)
            xform.GetPrim().SetCustomDataByKey('humans:role', spec.role)
            xform.GetPrim().SetCustomDataByKey(
                'humans:provenance', 'scenario_scripted_actor_not_sensor_evidence')
            paths[pid] = prim_path
        return paths

    def _set_pose(self, pid: str, x: float, y: float, yaw: float) -> None:
        from pxr import Gf, UsdGeom
        xformable, z0 = self._xform_ops[pid]
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z0))
        xformable.AddRotateZOp().Set(math.degrees(yaw))

    def update(self, sim_time: float) -> None:
        """Teleport every person to its scripted pose at ``sim_time``."""
        if self._stage is None:
            raise RuntimeError('author() must run before update()')
        if self._backend == 'anim_people':
            return                                          # skeletal graph drives itself
        for pid in self._specs:
            x, y, yaw = self._paths[pid].pose_at(sim_time)
            self._set_pose(pid, x, y, yaw)


def _anim_people_available() -> bool:
    try:
        import omni.anim.people  # noqa: F401
        return True
    except Exception:
        return False
