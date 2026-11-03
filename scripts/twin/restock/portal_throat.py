"""CPU geometry proof that the ``sales_backroom_portal`` throat is collider-clear.

This is the authoritative disproof of the harvest item-5 / repeatability-ledger msg-628
hypothesis: that the coupled tow's fixed from-rest pin at **x = -3.935 m (to the millimetre,
seven-run cluster)** is a latent VERTICAL collider at that x against the base body, the cart
body, or the cart's casters. It is not. Enumerated on CPU against the store's own static
collision geometry (``scripts/twin/colliders`` authors a collider on exactly these meshes; the
navigation manifest's 797 world AABBs are the same set):

  1. **The throat corridor is clear.** The coupled train's swept envelope through the portal
     from the recorded from-rest pin pose (the eastward backroom leg handing off into the
     portal-axis turn) intersects NO static collision prim in the drive-height band on any
     segment. The nearest static obstacle to the pin footprint is ~2.66 m away (the Rack_3
     beams at y~=17.9), and the nearest collider the base faces into the portal is the east
     jamb/wall, well beyond the opening -- not at x=-3.935.

  2. **The load path is off every floor edge.** The trace shows the two DRIVE WHEELS bear the
     load and are the arrested contacts (casters airborne). Both drive wheels sit >= 0.47 m
     from EITHER ``Backroom_floor`` cooked triangle diagonal (:mod:`portal_floor`), so the
     floor's internal ("ghost") edge is not under the load path -- the floor-diagonal pin
     mechanism cannot arrest the drive wheels.

  3. **The one caster over a diagonal is airborne.** The cart front-left caster does overlie a
     cooked diagonal (perp ~0.01 m -- the original portal_floor front-caster observation), but
     ``caster_lift_m`` holds its wheel ~0.03 m above the floor, so it bears no load and cannot be
     arrested by the floor edge either.

Conclusion (named per the captain's rule): the fixed x is NOT geometry. It is the deterministic
lane hand-off pose plus hitch kinematics -- the base comes to rest at
``cart_x + hitch_offset`` = -3.935 where the follower seats the cart at its turn-entry pose --
and from that rest the from-rest PhysX drive authority is insufficient to re-break-away. That is
the ``sim-robust-coupled-traction`` / ``coupled-breakaway-from-rest`` / ``throat-momentum-launch``
domain (caster lift, wheel drive force, momentum launch, break-away rock, pre-throat heading
hold), already mitigated by the 2.6 m portal + aligned dock (coldwin22 crossed clean). There is
no store/rig/cart collider to remove.

Pure geometry; imports no pxr. Store obstacle AABBs are supplied by the caller (the committed
throat fixture, or a live ``usd-core`` scan via ``scripts/check_portal_throat_clear.py``). This
module is the regression guard: if a future store edit narrows the portal or drops a jamb footing
/ duplicated wall segment into the corridor, :func:`assess` reports it on CPU before a GPU run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..mobile import MobileSpec, Pose2, occupied_envelope, swept_envelope
from . import portal_floor as pf

# The seven-run from-rest pin cluster (base_link), x fixed to the millimetre. y is mid-corner on
# the eastward backroom leg as the follower hands off into the portal-axis turn (launch1b
# first_stop [-3.935, 15.208]; check_tow_traction).
PIN_BASE_XY: Tuple[float, float] = (-3.935, 15.21)
PORTAL_X: float = -3.875

# The drive-height band the mobile base/cart/casters occupy -- the store manifest's obstacle band
# (store_environment.write_manifest: a mesh is a nav obstacle iff it spans [0.02, 2.6] m).
DRIVE_BAND_Z: Tuple[float, float] = (0.02, 2.6)

# A load-bearing drive wheel must stay at least this far from a cooked floor diagonal for the
# floor-edge ghost-contact mechanism to be ruled out as the arrest. The measured minimum is
# ~0.47 m, far above this bound.
DRIVE_DIAGONAL_MIN_M: float = 0.10


def hitch_offset(spec: MobileSpec = MobileSpec()) -> float:
    """Coupled ``base_x - cart_x`` for the aligned train (``cart_hitch_x - hitch_x`` = 1.27 m)."""
    return pf.hitch_offset(spec)


def cart_pose_trailing(base: Pose2, spec: MobileSpec = MobileSpec()) -> Pose2:
    """The aligned cart pose trailing ``base`` by the hitch offset along the base heading."""
    ho = hitch_offset(spec)
    return Pose2(base.x - ho * math.cos(base.yaw), base.y - ho * math.sin(base.yaw), base.yaw)


def drive_wheel_world(base: Pose2, spec: MobileSpec = MobileSpec()) -> List[Tuple[float, float]]:
    """World XY of the two DRIVE-wheel ground contacts (jointed at base-local ``(0, +-track/2)``)
    -- the load-bearing contacts the trace shows are arrested at the pin."""
    return [base.point(0.0, +spec.wheel_track / 2.0), base.point(0.0, -spec.wheel_track / 2.0)]


def caster_wheel_bottom_z(spec: MobileSpec = MobileSpec()) -> float:
    """World z of a caster wheel's lowest point. The URDF roll joint recesses the wheel
    ``-0.08 + caster_lift_m`` below the fork (fork world z 0.145), wheel radius 0.065. With
    ``caster_lift_m=0.03`` the bottom sits at z=+0.030 -- airborne over the z=0 floor top."""
    fork_world_z = 0.145
    wheel_centre_z = fork_world_z + (-0.08 + spec.caster_lift_m)
    return wheel_centre_z - 0.065


def cooked_diagonals(top_quad=pf.BACKROOM_FLOOR_TOP_QUAD) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Both candidate cooked internal diagonals of the ``Backroom_floor`` top quad. PhysX cooks a
    quad into two triangles sharing ONE of these; which one depends on the exported vertex order,
    so the proof must clear BOTH."""
    return [(top_quad[0], top_quad[2]), (top_quad[1], top_quad[3])]


# --------------------------------------------------------------------------- #
# Swept envelope through the portal from the from-rest pin, and clearance.     #
# --------------------------------------------------------------------------- #
def corner_samples(spec: MobileSpec = MobileSpec(), pin: Tuple[float, float] = PIN_BASE_XY,
                   portal_x: float = PORTAL_X) -> List[Tuple[Pose2, Pose2]]:
    """Coupled (robot, cart) pose samples for the from-rest throat crossing: the eastward
    backroom leg into the portal-axis turn and out the south side of the portal. This is the
    bounded ``start_at=couple ... stop_after=outbound`` throat recipe's path."""
    xpin = pin[0]
    base = [
        Pose2(xpin - 0.4, pin[1], 0.0),          # rolling up to the pin on the eastward leg
        Pose2(xpin, pin[1], 0.0),                # the from-rest pin pose (aligned, facing +x)
        Pose2(portal_x, pin[1] - 0.2, -math.pi / 8),   # entering the turn
        Pose2(portal_x, 14.0, -math.pi / 2),           # aligned south on the portal axis
        Pose2(portal_x, 11.125, -math.pi / 2),         # in the portal opening
        Pose2(portal_x, 9.0, -math.pi / 2),            # out the sales side
    ]
    return [(b, cart_pose_trailing(b, spec)) for b in base]


def polygon_aabb_overlap(poly: Sequence[Tuple[float, float]], mn, mx) -> bool:
    """SAT overlap test between a convex polygon and an axis-aligned XY rectangle [mn, mx]."""
    axes: List[Tuple[float, float]] = [(1.0, 0.0), (0.0, 1.0)]
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        axes.append((-(y2 - y1), (x2 - x1)))
    rect = [(mn[0], mn[1]), (mx[0], mn[1]), (mx[0], mx[1]), (mn[0], mx[1])]
    for ax, ay in axes:
        d = math.hypot(ax, ay)
        if d == 0:
            continue
        ax, ay = ax / d, ay / d
        pp = [p[0] * ax + p[1] * ay for p in poly]
        rr = [p[0] * ax + p[1] * ay for p in rect]
        if max(pp) < min(rr) or max(rr) < min(pp):
            return False   # a separating axis exists -> no overlap
    return True


def _in_drive_band(mn, mx, z_band) -> bool:
    return mx[2] >= z_band[0] and mn[2] <= z_band[1]


def segment_intrusions(obstacles: Sequence[dict], spec: MobileSpec = MobileSpec(),
                       samples=None, z_band=DRIVE_BAND_Z, margin: float = 0.05
                       ) -> List[Tuple[int, List[str]]]:
    """Per-segment swept-envelope intrusions (how the nav collision-checks the lane, NOT a single
    convex union over the whole 90 deg turn -- which would over-approximate and graze the portal
    wall). Returns ``[(segment_index, [intruding obstacle ids]), ...]`` for the segments that DO
    intrude; an empty list means the whole corridor is clear in the drive-height band.

    ``obstacles``: dicts with ``id``, ``min`` [x,y,z], ``max`` [x,y,z] (world AABBs, metres)."""
    samples = list(samples if samples is not None else corner_samples(spec))
    band_obs = [o for o in obstacles if _in_drive_band(o['min'], o['max'], z_band)]
    out: List[Tuple[int, List[str]]] = []
    for i, (a, b) in enumerate(zip(samples, samples[1:])):
        poly = swept_envelope([a, b], spec, margin=margin)
        hit = [o['id'] for o in band_obs if polygon_aabb_overlap(poly, o['min'], o['max'])]
        if hit:
            out.append((i, hit))
    return out


def _aabb_point_distance(mn, mx, px, py) -> float:
    cx = max(mn[0], min(px, mx[0]))
    cy = max(mn[1], min(py, mx[1]))
    return math.hypot(px - cx, py - cy)


def nearest_obstacle(obstacles: Sequence[dict], pin: Tuple[float, float] = PIN_BASE_XY,
                     z_band=DRIVE_BAND_Z) -> Optional[Tuple[str, float]]:
    """The nearest drive-height static obstacle to the pin base centre (id, distance m)."""
    band_obs = [o for o in obstacles if _in_drive_band(o['min'], o['max'], z_band)]
    if not band_obs:
        return None
    best = min(band_obs, key=lambda o: _aabb_point_distance(o['min'], o['max'], pin[0], pin[1]))
    return best['id'], round(_aabb_point_distance(best['min'], best['max'], pin[0], pin[1]), 3)


@dataclass(frozen=True)
class ThroatClearance:
    corridor_clear: bool                 # no static collider intrudes any swept segment
    intrusions: Tuple[Tuple[int, Tuple[str, ...]], ...]
    nearest_obstacle_id: Optional[str]
    nearest_obstacle_m: Optional[float]
    drive_wheel_diagonal_min_m: float    # min perp of either drive wheel to either cooked diagonal
    load_path_off_diagonal: bool         # drive_wheel_diagonal_min_m >= DRIVE_DIAGONAL_MIN_M
    caster_on_diagonal_label: str        # the cart caster nearest a cooked diagonal
    caster_on_diagonal_perp_m: float
    caster_airborne_m: float             # caster wheel bottom clearance above the floor top (z=0)
    verdict: str

    @property
    def collider_free(self) -> bool:
        """The full disproof: corridor clear AND the load path is off every floor edge AND the one
        caster over a diagonal is airborne."""
        return (self.corridor_clear and self.load_path_off_diagonal
                and self.caster_airborne_m > 0.0)


def assess(obstacles: Sequence[dict], spec: MobileSpec = MobileSpec(),
           pin: Tuple[float, float] = PIN_BASE_XY, z_band=DRIVE_BAND_Z,
           margin: float = 0.05) -> ThroatClearance:
    """Full CPU assessment of the throat for the coupled from-rest crossing."""
    samples = corner_samples(spec, pin)
    intr = segment_intrusions(obstacles, spec, samples=samples, z_band=z_band, margin=margin)
    near = nearest_obstacle(obstacles, pin, z_band)

    # load path: both drive wheels vs both cooked diagonals at the aligned pin pose
    base = Pose2(pin[0], pin[1], 0.0)
    diags = cooked_diagonals()
    dmin = min(abs(pf.perp_distance(w, d[0], d[1]))
               for w in drive_wheel_world(base, spec) for d in diags)

    # the cart caster nearest a cooked diagonal (the original front-caster observation)
    cart = cart_pose_trailing(base, spec)
    cart_xy_yaw = (cart.x, cart.y, cart.yaw)
    labels = ['rear_right', 'rear_left', 'front_right', 'front_left']
    best_label, best_perp = '', math.inf
    for lab, w in zip(labels, pf.cart_caster_world(cart_xy_yaw, spec)):
        perp = min(abs(pf.perp_distance(w, d[0], d[1])) for d in diags)
        if perp < best_perp:
            best_label, best_perp = lab, perp

    airborne = caster_wheel_bottom_z(spec)
    corridor_clear = not intr
    load_off = dmin >= DRIVE_DIAGONAL_MIN_M

    if corridor_clear and load_off and airborne > 0.0:
        verdict = ('throat collider-clear: no static collider in the swept corridor; load-bearing '
                   'drive wheels >= %.2f m off every floor diagonal; caster over a diagonal is '
                   '%.3f m airborne -> the fixed-x pin is NOT geometry (hitch kinematics + '
                   'from-rest drive authority)' % (dmin, airborne))
    elif not corridor_clear:
        verdict = 'THROAT BLOCKED: static collider(s) intrude the swept corridor: %r' % (intr,)
    elif not load_off:
        verdict = ('a drive wheel is only %.3f m from a cooked floor diagonal (< %.2f) -- the '
                   'floor-edge mechanism is NOT ruled out' % (dmin, DRIVE_DIAGONAL_MIN_M))
    else:
        verdict = 'caster over a diagonal is not airborne (%.3f m) -- floor-edge via caster possible' % airborne

    return ThroatClearance(
        corridor_clear=corridor_clear,
        intrusions=tuple((i, tuple(ids)) for i, ids in intr),
        nearest_obstacle_id=(near[0] if near else None),
        nearest_obstacle_m=(near[1] if near else None),
        drive_wheel_diagonal_min_m=round(dmin, 3),
        load_path_off_diagonal=load_off,
        caster_on_diagonal_label=best_label,
        caster_on_diagonal_perp_m=round(best_perp, 3),
        caster_airborne_m=round(airborne, 3),
        verdict=verdict,
    )
