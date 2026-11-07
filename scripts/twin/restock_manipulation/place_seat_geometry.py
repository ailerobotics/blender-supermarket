"""sim-place-seat-geometry-v1: establish the PHYSICAL cart/shelf set-down seat from the simulation
geometry itself, and classify a measured set-down against it.

Pure, CPU-testable decision logic (``math`` only; the optional :func:`derive_from_stage` lazily imports
``pxr`` so the module imports with no USD). No Isaac / ROS / numpy.

WHY (the brief: six runs reached the cart set-down and the release failed three ways; the model of where
the carton seats and the physics disagreed by ~9 cm and nobody had established what the carton rested on):

  Ground truth, from the composed ``mission.usda`` (usd-core) + the ``nav-trace.jsonl`` carton WORLD poses:

  * The handling unit collider is the box ``extent`` ``(-0.225,-0.15,-0.09)..(0.225,0.15,0.09)`` -- a clean
    0.45 x 0.30 x 0.18 m box, convexHull, carton BOTTOM = carton CENTRE z - 0.09. Half-height 0.09 is RIGHT.
  * The carton is filtered (``PhysicsFilteredPairsAPI``) against the ENTIRE rollwagen cage (posts, side/back
    rails); its only collision surfaces are the DECK (top ~0.26 world) and the upper SHELF. So the carton
    can only rest on the deck or the shelf -- never a cage bar.
  * The upper shelf is a thin slab (``scale (0.76,0.64,0.025)``, local translate z 0.72 on the chassis). Its
    FLAT TOP FACE world z = chassis_z + 0.72 + 0.0125. At the composed/un-settled chassis (z 0.23148) that is
    0.9639 world; the world-aligned AABB MAX-Z corner of the slightly-tilted slab is 0.9712 (+7.3 mm: the
    AABB corner, not the face the carton sits on). The DYNAMIC cart SETTLES ~3.1 cm over the run (chassis
    0.2315 -> ~0.20), so the shelf FLAT TOP at the SETTLED set-down pose is ~0.9325 world.
  * PROOF of the true seat: coldwin83 released the weld and the carton RESTED and VERIFIED -- its final
    carton CENTRE world z (nav-trace) is 1.0225, i.e. carton BOTTOM 0.9325 = the settled shelf flat top.

  So the MODEL support top used for the descent/release/verify, ``0.9707`` world, is +3.8 cm too high: it is
  the tilted slab's world-AABB CORNER (+0.7 cm vs the flat face) resolved at the cart's UN-SETTLED build
  chassis (+3.1 cm vs the settled set-down chassis). The half-height (0.09) and the carton CENTRE frame are
  CORRECT.

  The coldwin87 "solid stall" at carton CENTRE 1.1634 world (carton BOTTOM 1.0734) is 14 cm ABOVE the true
  seat (0.9325) and 10 cm above even the model support top (0.9707). Because the carton collides only with
  the deck/shelf (both far below), it is NOT resting on the cart at all -- it is held by the arm (effort
  peaked 1013 N*m) with a side-grip robot link jammed on the cage (robot links are NOT in the carton's cage
  filter). The horizontal side-grip seal raycast sees nothing. So the stall is an OBSTRUCTION ~14 cm above
  the seat, NOT a shelf contact: releasing there drops the carton 14 cm (coldwin85/86 fell off during the
  retreat; coldwin83 happened to rest). PR 374 calibrated the release threshold TO that 1.073/0.6422 stall
  (it added a half-height to fire at the measured rest) -- i.e. it trusted the obstruction as the seat.

This module provides the test-backed derivation (:class:`SeatModel`, :func:`seat_from_support_top`,
:func:`classify_set_down`) and a usd-core reader (:func:`derive_from_stage`) so the geometry is reproducible
and a replay can DISPROVE the diagnosis. The controller / place_release gate use the SAME pure predicate
(:func:`classify_set_down`) so a measured set-down is released only AT the modelled seat and anything well
above it is named an obstruction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Default contact-vs-seat tolerance (m): a measured contact (stall/effort) counts as a true set-down only
# when the carton BOTTOM is within this of the modelled seat (support top). 0.02 m is the brief's value --
# comfortably inside the verify z-tol (0.12) and the free-check rest gap (0.03), tight enough that the
# coldwin87 14 cm stall is unambiguously an OBSTRUCTION, not a seat contact.
DEFAULT_SEAT_TOL_M = 0.02

# The upper-shelf slab half-thickness (m): USD Cube size 1 x scale-z 0.025 => +-0.0125 about the slab
# centre. The FLAT TOP FACE is the slab centre z + this; the world-AABB MAX-Z over-reads it by the slab's
# tilt. Single source: twin.mobile_assets.author_cart authors scale (0.76,0.64,0.025).
SHELF_SLAB_HALF_THICKNESS_M = 0.0125
# The deck slab half-thickness (m): scale-z 0.05 => +-0.025.
DECK_SLAB_HALF_THICKNESS_M = 0.025


@dataclass(frozen=True)
class SeatModel:
    """The modelled set-down seat for one place, derived from geometry.

    ``support_top_z`` is the support surface TOP FACE (the plane the carton bottom rests on), in whatever
    frame the caller works in (world for the geometry tool; ur_base_link for the live release gate -- the
    same frame as ``sensors.placed_pose``). ``carton_half_m`` is the carton half-height. ``seat_center_z``
    is the carton CENTRE z when the carton bottom rests flat on the support (support_top + half). All pure.
    """
    support_top_z: float
    carton_half_m: float

    @property
    def seat_center_z(self) -> float:
        """The carton CENTRE z at the true seat (carton bottom on the support top)."""
        return float(self.support_top_z) + float(self.carton_half_m)

    def carton_bottom_z(self, carton_center_z: float) -> float:
        """The carton BOTTOM z for a given carton CENTRE z (centre - half)."""
        return float(carton_center_z) - float(self.carton_half_m)

    def bottom_gap(self, carton_center_z: float) -> float:
        """Carton BOTTOM minus support TOP for a given carton CENTRE z. 0 == resting flat; >0 == above the
        support; <0 == below/penetrating."""
        return self.carton_bottom_z(carton_center_z) - float(self.support_top_z)


def seat_from_support_top(support_top_z: float, carton_height_m: float) -> SeatModel:
    """A :class:`SeatModel` from a support TOP-face z and the carton HEIGHT (not half). Validates finite,
    positive carton height. Pure."""
    h = float(carton_height_m)
    if not (math.isfinite(float(support_top_z)) and math.isfinite(h) and h > 0):
        raise ValueError('seat_from_support_top needs a finite support_top_z and a positive carton_height_m')
    return SeatModel(support_top_z=float(support_top_z), carton_half_m=h / 2.0)


def slab_flat_top_z(slab_center_z: float, half_thickness_m: float) -> float:
    """The slab FLAT TOP FACE z = slab centre z + half-thickness. This is the plane the carton rests on --
    the quantity the model must use, NOT the world-aligned AABB max-z corner (which over-reads a tilted
    slab by its tilt). Pure."""
    if not (math.isfinite(float(slab_center_z)) and math.isfinite(float(half_thickness_m))):
        raise ValueError('slab_flat_top_z needs finite inputs')
    return float(slab_center_z) + float(half_thickness_m)


def classify_by_gap(bottom_gap: float, *, contact: bool, tol_m: float = DEFAULT_SEAT_TOL_M) -> str:
    """The core seat classification on the carton-BOTTOM-minus-support-TOP gap (equivalently the carton
    CENTRE minus the seat CENTRE -- the half-height cancels). Pure. See :func:`classify_set_down` for the
    verdict meanings. This is the shared primitive the live release gate calls with ``carton_center_z -
    seat_center_z`` so it never has to reconstruct the support top / half-height split."""
    if not (math.isfinite(float(tol_m)) and float(tol_m) >= 0):
        raise ValueError('classify_by_gap needs a finite non-negative tol_m')
    if not math.isfinite(float(bottom_gap)):
        raise ValueError('classify_by_gap needs a finite bottom_gap')
    gap = float(bottom_gap)
    tol = float(tol_m)
    if gap > tol:                         # the carton bottom is ABOVE the seat
        return 'obstructed' if contact else 'descending'
    if gap < -tol:                        # BELOW the modelled seat: fell through / landed on a LOWER
        return 'passed'                   # surface than modelled -> a fault, never a valid set-down
    return 'seated' if contact else 'descending'   # within tol of the seat: release only on a contact


def classify_set_down(carton_center_z: float, seat: SeatModel, *, contact: bool,
                      tol_m: float = DEFAULT_SEAT_TOL_M) -> str:
    """Classify a place-descent tick against the modelled seat. Pure; the single shared predicate the
    geometry tool AND the live release gate use so a verdict matches.

    Returns one of:
      * ``'seated'``   -- the carton is AT the seat (``|bottom_gap| <= tol_m``) WITH a measured contact:
        release the weld here (true set-down).
      * ``'obstructed'`` -- a measured CONTACT occurred while the carton bottom is still ABOVE the seat by
        more than ``tol_m`` (blocked well above the seat): do NOT release/press here; re-approach or fail by
        name. This is the coldwin87 14 cm stall.
      * ``'passed'``   -- the carton bottom descended BELOW the seat by more than ``tol_m`` (with OR without
        a contact): the support is not where modelled -- the carton fell through, or already landed on a
        LOWER surface than the shelf (a named fault; a controlled descent should contact AT the seat).
      * ``'descending'`` -- none of the above yet: above the seat with no contact, keep descending.

    ``contact`` is the caller's measured support-touched signal (stall and/or effort over baseline).
    """
    return classify_by_gap(seat.bottom_gap(carton_center_z), contact=contact, tol_m=tol_m)


# --------------------------------------------------------------------- usd-core derivation (optional)

@dataclass(frozen=True)
class StageSeatDerivation:
    """Geometry derived from a composed stage (usd-core). All WORLD frame. ``shelf_flat_top_z`` is the
    slab flat top face (the correct support top); ``shelf_aabb_top_z`` is the world-AABB max-z corner (the
    model's over-read); their difference is the tilt inflation. ``carton_half_m`` is from the handling-unit
    extent; ``ur_base_z`` is the arm-root world z at the composed (build) pose."""
    shelf_flat_top_z: float
    shelf_aabb_top_z: float
    shelf_center_z: float
    carton_half_m: float
    ur_base_z: float
    chassis_z: float
    aabb_corner_inflation_m: float


def derive_from_stage(usd_path: str, *,
                      support_path: str = '/Rollwagen/chassis/shelf',
                      chassis_path: str = '/Rollwagen/chassis',
                      carton_path: str = '/RestockScenario/HandlingUnit',
                      base_path: str = '/MobileRobot/ur_base_link',
                      slab_half_thickness_m: float = SHELF_SLAB_HALF_THICKNESS_M) -> StageSeatDerivation:
    """Open a composed ``mission.usda`` with usd-core and derive the set-down geometry (WORLD frame). Lazy
    ``pxr`` import so the module imports with no USD. Pure read (never mutates the stage)."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415 -- lazy so the pure core imports without USD

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError(f'could not open USD stage {usd_path!r}')
    cache = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def center_z(path: str) -> float:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f'prim {path!r} not in stage {usd_path!r}')
        return float(cache.GetLocalToWorldTransform(prim).ExtractTranslation()[2])

    def aabb_top_z(path: str) -> float:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f'prim {path!r} not in stage {usd_path!r}')
        rng = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        return float(rng.GetMax()[2])

    shelf_center = center_z(support_path)
    shelf_flat_top = slab_flat_top_z(shelf_center, slab_half_thickness_m)
    shelf_aabb_top = aabb_top_z(support_path)
    chassis_z = center_z(chassis_path)
    ur_base_z = center_z(base_path)

    carton = stage.GetPrimAtPath(carton_path)
    if not carton or not carton.IsValid():
        raise ValueError(f'carton prim {carton_path!r} not in stage {usd_path!r}')
    ext = carton.GetAttribute('extent').Get()
    if not ext:
        raise ValueError(f'carton prim {carton_path!r} has no extent attribute')
    carton_half = (float(ext[1][2]) - float(ext[0][2])) / 2.0

    return StageSeatDerivation(
        shelf_flat_top_z=shelf_flat_top,
        shelf_aabb_top_z=shelf_aabb_top,
        shelf_center_z=shelf_center,
        carton_half_m=carton_half,
        ur_base_z=ur_base_z,
        chassis_z=chassis_z,
        aabb_corner_inflation_m=shelf_aabb_top - shelf_flat_top,
    )
