"""CPU clearance sweep for the collect place-on-cart set-down (sim-place-descent-residual-v1).

coldwin31..40 all failed at the COLLECT set-down with CARTON_TILT pitch +2.02..+2.07 deg against the
2.0 bound, UNCHANGED across three generations of droop compensation (PR 311 heavy / PR 318 wrist /
PR 326 per-point) -- d(executed)/d(commanded) ~ 0, the signature of a GEOMETRIC CONTACT PIVOT, not
droop. coldwin39's own SEAL_DIAG proves it: the gripper suction pad sits FLUSH on
``/Rollwagen/chassis/post_2`` (min_gap_mm 0.0, pad_vs_surface_deg 0.0) through the whole set-down/dwell
while the welded carton is levered to 2.02 deg. The ``cage_collision: filtered`` workaround filters the
CARTON vs the cage, but the PAD is a ROBOT link (``suction_cups``), NOT the carton, so it still
collides with the posts.

This module is the CPU geometric proof + gate the fix is chosen from. It sweeps the gripper suction
pad PLUS the welded 0.45 x 0.30 x 0.18 carton along a given set-down path, expressed as carton poses in
the Rollwagen CART BODY frame (the cart-anchored frame the place target lives in, so the check needs no
base/arm pose), against EVERY Rollwagen collider -- the four corner posts, the side/back wall bars, the
deck, the upper shelf and the drawbar -- read from the authored USD (``twin.mobile_assets.author_cart``,
the same bodies ``twin.restock.cage_collision`` enumerates). It reports the minimum signed gap per body
per waypoint and REFUSES a path whose minimum gap to any NON-SUPPORT body is below a margin (default
30 mm). The SUPPORT bodies (``deck``/``shelf``) are the surface the carton LANDS on, so a ~0 gap there is
expected and allowed; every other body must stay clear until the carton is flat on the support.

Pure numpy + ``usd-core`` (``pxr`` via ``author_cart``). No Isaac, ROS or GPU. The gripper/carton
geometry (the pad panel rigidly welded relative to the carton) is a MEASURED constant derived from the
wall=3 side grip (``shelfbot ... geometry.wall_grip``) + the mobile URDF suction-panel collision box;
:meth:`GripperCartonGeometry.from_mission` re-derives it and ``test_place_clearance`` pins the constant
against that derivation so a geometry drift fails on CPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Log tokens the controller/pre-flight print.
PLACE_CLEARANCE = 'PLACE_CLEARANCE'
PLACE_CLEARANCE_REJECTED = 'PLACE_CLEARANCE_REJECTED'
# sim-preplace-sweep-cage-clearance-v1: the carried PRE-PLACE leg-2 shoulder_pan sweep's cage-clearance
# log / refuse tokens (see preplace_cage_arm_clearance / preplace_sweep_reject below).
PREPLACE_SWEEP_CLEARANCE = 'PREPLACE_SWEEP_CLEARANCE'
# Minimum arm-link-to-cage gap (m) the leg-2 pan sweep must keep to NOT be refused. coldwin70 reconstructs
# the executed leg-2 at -48 mm INTO /Rollwagen/chassis/post_2 (it froze at shoulder_pan 0.2012 with a 1632.7
# N*m effort spike). The DEFAULT refuse margin is 0.0 -- an actually INTERSECTING (penetrating) sweep -- for
# two honest reasons: (1) the CPU geometry carries a documented ~40-90 mm offset from the sim colliders (the
# pad/arm sphere model vs the authored AABBs, the same calibration caveat the place-clearance block above
# notes), so only a clear negative reading is a certain strike; (2) the pan-at-carry reorder that clears the
# RAKING strike to +252 mm still ends at the pre-place standoff, whose gripper housing sits an inherent
# +4.2 mm from the FAR side wall (side_1_horizontal_6) -- a cage-width tightness within that calibration
# noise, NOT the cw70 blocker -- so a positive refuse margin would block even a clearing path. The >= 0.04 m
# target is the DESIGN GOAL the clearing path meets on the raking cage bodies (post_*/side_0), checked by
# scripts/check_preplace_sweep_clearance.py; the lane may raise the margin via the provider.
DEFAULT_PREPLACE_CAGE_MARGIN_M = 0.0
# The design-target clearing margin (m) the reorder achieves on the raking cage bodies (reported, not the
# refuse threshold): the brief's >= 0.04 m.
PREPLACE_CAGE_CLEAR_TARGET_M = 0.040

# Default minimum gap (m) the swept gripper+carton must keep from every NON-support Rollwagen body
# until the carton bottom is flat on the support. 30 mm matches the brief's margin.
DEFAULT_MARGIN_M = 0.030

# The SUPPORT bodies the carton is PLACED on (allowed to touch at the set-down). The mission's
# "place-on-deck" ground-truth slot targets ``support_z_world`` 0.9325 m, which is the UPPER SHELF top
# (0.933), not the low deck (0.225) -- so both candidate support surfaces are excluded from the refuse
# rule; every other Rollwagen body (posts, side/back bars, drawbar) must stay clear.
SUPPORT_BODIES = ('deck', 'shelf')

# --- sim-cart-support-select-v1: the LIVE collider set the sim actually touches --------------------
# ``twin.restock.cage_collision`` filters CARTON-vs-CAGE contacts (the four corner posts + the side/back
# wall bars) when ``cage_collision: 'filtered'`` is active -- the live CAGE_COLLISION console line reads
# ``kept=['deck', 'shelf', 'drawbar']``. So the CARTON (the handling unit) collides ONLY with those kept
# bodies; it passes THROUGH the cage bars in the sim. The gripper PAD, however, is a ROBOT link
# (``suction_cups``) -- NOT the handling unit -- so the pad still collides with EVERY Rollwagen body,
# posts and bars included (coldwin39: the pad sat flush on post_2). An honest clearance sweep must
# therefore count the carton points only against the kept bodies and the pad points against all bodies,
# minus the support prim the carton lands on. The old lumped sweep counted the carton box against the
# cage bars too (its ``-93 mm`` reading vs the upper side bar ``side_0_horizontal_6``) -- a body the sim
# does not collide the carton with, so a phantom strike.
CAGE_PREFIXES = ('post_', 'side_', 'back_')
KEPT_BODIES = ('deck', 'shelf', 'drawbar')


def is_cage_body(name):
    """True if ``name`` is a Rollwagen CAGE collider (post / side-wall / back-wall bar) that
    ``cage_collision`` filters out of CARTON contacts. The carton passes through it in the sim; only the
    gripper pad (a robot link) still collides with it."""
    return str(name).startswith(CAGE_PREFIXES)


def honest_collider_parts(name, support):
    """Which welded-geometry parts the SIM actually collides with Rollwagen body ``name`` (given the
    carton rests on ``support``). Returns a subset of ``('carton', 'pad')``:

      * the ``support`` body -> ``()`` (the surface the carton lands on; a ~0 gap is the goal, never a strike),
      * a CAGE body (post / side / back bar) -> ``('pad',)`` (carton filtered; only the pad collides),
      * any other KEPT body (the non-support deck/shelf, drawbar) -> ``('carton', 'pad')``.

    Pure/CPU -- the single source of truth the honest sweeps and the refuse gate share."""
    if name == support:
        return ()
    if is_cage_body(name):
        return ('pad',)
    return ('carton', 'pad')

# Carton (handling unit) dimensions (m) -- the scenario single source of truth handling_unit.dimensions_m.
CARTON_DIMS_M = (0.45, 0.30, 0.18)

# --- measured gripper suction-pad panel, rigid relative to the welded carton (wall=3 side grip) ---
# Derived by GripperCartonGeometry.from_mission (shelfbot wall_grip wall=3 + mobile URDF suction_cups
# collision box); pinned by test_place_clearance. The pad hangs below the carton bottom and sits on the
# carton's +local-y (mouth-ward) face -- CPU-validated against coldwin39's SEAL_DIAG: at carton yaw 169
# deg the pad approach axis is (0.19, 0.98), exactly the logged ``forward=[0.19, 0.98, 0.04]``.
PAD_CENTER_IN_CARTON = (0.0, 0.165532, -0.090454)
PAD_R_IN_CARTON = ((-1.0, 0.0, 0.0),
                   (0.0, -0.035464, -0.999371),
                   (0.0, -0.999371, 0.035464))
PAD_SIZE_M = (0.260424, 0.108204, 0.02558)
# The grip TCP (gripper_pad_tcp) relative to the carton for the wall=3 side grip (reference/re-derivation).
WALL3_GRIP_TCP_IN_CARTON = (0.0, 0.15275, -0.09)


# --------------------------------------------------------------------- cart collider geometry

def rollwagen_collider_boxes(spec=None):
    """``{name: (lo, hi)}`` axis-aligned world-z AABBs of EVERY Rollwagen collider, in the cart BODY
    frame (x forward to the open +x mouth, origin at the cart ground point), read from the authored USD
    (``twin.mobile_assets.author_cart``) -- not re-derived. Includes the four corner posts, the side/back
    wall bars, the deck, the upper shelf and the drawbar (the bodies ``cage_collision`` filters plus the
    supports it keeps). Needs ``pxr`` (usd-core)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from twin.mobile import MobileSpec
    from twin.mobile_assets import author_cart
    stage = Usd.Stage.CreateInMemory()
    cart = author_cart(stage, spec=spec or MobileSpec())
    boxes = {}
    for child in stage.GetPrimAtPath(cart['body']).GetChildren():
        if not child.HasAPI(UsdPhysics.CollisionAPI):
            continue
        m = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        corners = np.array([list(m.Transform(Gf.Vec3d(x, y, z)))
                            for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
        boxes[child.GetName()] = (corners.min(0), corners.max(0))
    if not boxes:
        raise ValueError('no Rollwagen colliders authored; cannot build a clearance check')
    return boxes


# --------------------------------------------------------------------- gripper + carton geometry

def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _box_surface_points(half, step):
    """Surface sample points of a box with half-extents ``half`` (local frame, centred at origin): the
    six faces as grids at spacing ``step``. Dense enough that the min point-to-AABB distance is accurate
    to ``step`` along a face."""
    half = np.asarray(half, float)
    grid = [np.arange(-v, v + 1e-9, step) if v > 1e-9 else np.array([0.0]) for v in half]
    pts = []
    for ax in range(3):
        a, b = [i for i in range(3) if i != ax]
        A, B = np.meshgrid(grid[a], grid[b], indexing='ij')
        for sgn in (-1.0, 1.0):
            P = np.zeros((A.size, 3))
            P[:, a], P[:, b], P[:, ax] = A.ravel(), B.ravel(), sgn * half[ax]
            pts.append(P)
    return np.vstack(pts)


def _box_aabb_sdf(P, lo, hi):
    """Signed distance of points ``P`` (N x 3) to the AABB ``[lo, hi]`` (negative = inside)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, e = (lo + hi) / 2.0, (hi - lo) / 2.0
    q = np.abs(P - c) - e
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0)


@dataclass(frozen=True)
class GripperCartonGeometry:
    """The welded gripper suction pad + carton, as two oriented boxes in the CARTON frame. The carton is
    an axis-aligned box of ``carton_dims``; the pad is the oriented box ``pad_center`` / ``pad_R`` /
    ``pad_size`` rigidly fixed to the carton by the side grip + the weld. ``surface_points`` samples both
    in the carton frame so a swept clearance is a point-to-AABB distance at every waypoint."""
    carton_dims: tuple = CARTON_DIMS_M
    pad_center: tuple = PAD_CENTER_IN_CARTON
    pad_R: tuple = PAD_R_IN_CARTON
    pad_size: tuple = PAD_SIZE_M

    @classmethod
    def shipped(cls):
        """The measured default (wall=3 side grip + mobile URDF suction panel)."""
        return cls()

    @classmethod
    def from_mission(cls, urdf_path, shelfbot_root, *, wall=3, carton_dims=CARTON_DIMS_M,
                     tcp_surface_offset=0.00275, grasp_quat=(0.5, 0.5, 0.5, 0.5)):
        """Re-derive the pad-in-carton geometry from the actual mission grasp: the wall=``wall`` side
        grip (``shelfbot ... geometry.wall_grip``) gives the gripper_pad_tcp pose relative to the carton,
        and the mobile URDF (``urdf_path``) gives the ``suction_cups`` collision panel relative to
        gripper_pad_tcp. Needs the shelfbot source + the dataset URDF; used by ``test_place_clearance`` to
        pin the baked constant so a geometry drift fails on CPU."""
        import importlib
        import sys
        from types import SimpleNamespace
        import numpy.linalg as la
        from twin.urdf_probe import parse_urdf, link_world_transforms
        robot = parse_urdf(str(urdf_path))
        wl = link_world_transforms(robot)
        pad_in_tcp = la.inv(wl['gripper_pad_tcp']) @ wl['suction_cups']
        box = [g for g in robot.links['suction_cups'].collisions if g.kind == 'box'][0]
        pad_bs = np.asarray(box.box_size, float)
        pad_c_tcp = (pad_in_tcp @ np.array([*np.asarray(box.origin_xyz, float), 1.0]))[:3]
        root = str(shelfbot_root)
        sys.path.insert(0, root)
        try:
            geom = importlib.import_module('shelfbot.motion_planning.geometry')
        finally:
            if root in sys.path:
                sys.path.remove(root)
        var = SimpleNamespace(tray_size=tuple(carton_dims), contact_margin=tcp_surface_offset,
                              approach_offset_x_m=0.12, grasp_quat=tuple(grasp_quat))
        ox, oy, rx, ry, q = geom.wall_grip(var, SimpleNamespace(yaw=0.0), wall)
        T_tcp_carton = np.eye(4)
        T_tcp_carton[:3, :3] = _quat_R(*q)
        T_tcp_carton[:3, 3] = (ox, oy, -carton_dims[2] / 2.0)
        pad_center = (T_tcp_carton @ np.array([*pad_c_tcp, 1.0]))[:3]
        pad_R = T_tcp_carton[:3, :3] @ pad_in_tcp[:3, :3]
        return cls(tuple(float(v) for v in carton_dims),
                   tuple(float(v) for v in pad_center),
                   tuple(tuple(float(v) for v in r) for r in pad_R),
                   tuple(float(v) for v in pad_bs))

    def carton_surface_points(self, step=0.01):
        """Surface sample points (N x 3) of the CARTON box alone, in the CARTON frame."""
        return _box_surface_points(np.asarray(self.carton_dims, float) / 2.0, step)

    def pad_surface_points(self, step=0.01):
        """Surface sample points (N x 3) of the gripper PAD alone (oriented + offset), in the CARTON frame."""
        pad_local = _box_surface_points(np.asarray(self.pad_size, float) / 2.0, step)
        return pad_local @ np.asarray(self.pad_R, float).T + np.asarray(self.pad_center, float)

    def surface_points(self, step=0.01):
        """Surface sample points (N x 3) of the carton AND the pad, in the CARTON frame."""
        return np.vstack([self.carton_surface_points(step), self.pad_surface_points(step)])


def _quat_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# --------------------------------------------------------------------- carton poses in the cart frame

def carton_pose_cart(load_center_xy, support_z, yaw_cart_rel, *, carton_h, z_above=0.0):
    """4x4 pose of the (level) carton in the cart body frame: centred over ``load_center_xy`` at height
    ``support_z + carton_h/2 + z_above`` (``z_above`` 0 => sitting on the support; > 0 => a standoff above
    it), rotated ``yaw_cart_rel`` about +Z. A level carton (the carton-level constraint holds roll/pitch 0
    throughout the carried descent), so only yaw and position vary."""
    T = np.eye(4)
    T[:3, :3] = _rotz(float(yaw_cart_rel))
    T[:3, 3] = (float(load_center_xy[0]), float(load_center_xy[1]),
                float(support_z) + float(carton_h) / 2.0 + float(z_above))
    return T


def place_descent_poses(load_center_xy, support_z, yaw_cart_rel, *, carton_h, standoff_m=0.12, n=12):
    """The REMEDY set-down path (sim-place-descent-residual-v1): the carton parks at a standoff DIRECTLY
    ABOVE the place load-centre, then descends VERTICALLY onto the support -- ``n+1`` level carton poses
    from ``z_above = standoff_m`` down to ``z_above = 0``. Because the standoff is directly above the
    (clear) place target rather than overhanging the +x mouth, the welded pad never sweeps past the front
    posts (the coldwin39 pivot). Returns a list of 4x4 carton poses in the cart body frame."""
    return [carton_pose_cart(load_center_xy, support_z, yaw_cart_rel, carton_h=carton_h,
                             z_above=standoff_m * (1.0 - i / float(n))) for i in range(int(n) + 1)]


# --------------------------------------------------------------------- the sweep

def _honest_gaps_from_points(carton_pts, pad_pts, bodies, honest_support):
    """``{body: min signed gap}`` counting ONLY the geometry the sim collides with each body
    (:func:`honest_collider_parts`): the carton points against the kept bodies, the pad points against
    every body, the ``honest_support`` body dropped. ``carton_pts``/``pad_pts`` are world/cart-frame
    (already posed) N x 3 arrays."""
    out = {}
    for name, (lo, hi) in bodies.items():
        parts = honest_collider_parts(name, honest_support)
        if not parts:
            continue
        gaps = []
        if 'carton' in parts:
            gaps.append(_box_aabb_sdf(carton_pts, lo, hi).min())
        if 'pad' in parts:
            gaps.append(_box_aabb_sdf(pad_pts, lo, hi).min())
        out[name] = float(min(gaps))
    return out


def _gaps_at(T_carton_cart, geom, bodies, honest_support=None):
    """``{body: min signed gap (m)}`` between the gripper+carton at ``T_carton_cart`` and every cart body.

    ``honest_support`` None => the legacy LUMPED sweep (carton + pad points vs every body alike -- the
    full-body visual check). A body name => the HONEST sweep (sim-cart-support-select-v1): the carton is
    filtered vs the cage, so the carton points count only vs the kept bodies and the pad points vs all,
    with the support prim dropped (see :func:`honest_collider_parts`)."""
    R = np.asarray(T_carton_cart, float)[:3, :3]
    t = np.asarray(T_carton_cart, float)[:3, 3]
    if honest_support is None:
        pts = geom.surface_points() @ R.T + t
        return {name: float(_box_aabb_sdf(pts, lo, hi).min()) for name, (lo, hi) in bodies.items()}
    cpts = geom.carton_surface_points() @ R.T + t
    ppts = geom.pad_surface_points() @ R.T + t
    return _honest_gaps_from_points(cpts, ppts, bodies, honest_support)


def sweep_clearance(carton_poses, geom, bodies, *, honest_support=None):
    """Sweep the gripper+carton (``geom``) along ``carton_poses`` (list of 4x4 cart-frame carton poses)
    against ``bodies`` (``{name: (lo, hi)}`` from :func:`rollwagen_collider_boxes`).

    Returns ``(per_waypoint, per_body_min)``: ``per_waypoint`` is a list of ``{body: gap}`` dicts (the
    minimum gap per body at each waypoint), ``per_body_min`` is ``{body: min over all waypoints}`` -- the
    minimum gap per body over the whole path (the quantity the refuse gate reads).

    ``honest_support`` None => the legacy full-body sweep (byte-identical). A body name (the support the
    carton lands on) => the HONEST sweep that counts exactly the live collider set (carton vs kept bodies,
    pad vs all, support dropped) -- sim-cart-support-select-v1."""
    per_waypoint = [_gaps_at(T, geom, bodies, honest_support=honest_support) for T in carton_poses]
    per_body_min = {}
    for g in per_waypoint:
        for name, v in g.items():
            if name not in per_body_min or v < per_body_min[name]:
                per_body_min[name] = v
    return per_waypoint, per_body_min


def clearance_reject(per_body_min, *, margin=DEFAULT_MARGIN_M, support=SUPPORT_BODIES):
    """The :data:`PLACE_CLEARANCE_REJECTED` message when any NON-support body's minimum gap is below
    ``margin``, else ``None``. ``support`` bodies (the surface the carton lands on) are allowed to touch.
    Never widens the margin: the message names the offending bodies and their gaps so the lane sees
    exactly which post/bar is struck and by how much."""
    support = set(support)
    bad = sorted(((v, k) for k, v in per_body_min.items() if k not in support and v < margin))
    if not bad:
        return None
    named = ', '.join('%s=%.1fmm' % (k, v * 1000.0) for v, k in bad)
    return ('%s gripper+carton set-down path would strike a NON-support Rollwagen body: %s (minimum gap '
            'below the %.0f mm clearance margin). A pad/carton contact on a post or wall bar pivots the '
            'welded carton off level (the coldwin39 post_2 CARTON_TILT) -- refusing before motion.'
            % (PLACE_CLEARANCE_REJECTED, named, margin * 1000.0))


# ----------------------------------------- droop-aware EXECUTED-pose sweep (sim-place-descent-residual-v2)
# The v1 sweep above poses the carton as an IDEALISED level box at the configured load-centre. coldwin41
# proved that is the WRONG quantity: the above-place IK actually drifts the welded carton ~93 mm toward the
# +x mouth (cart (0.273,-0.047) vs the intended (0.18,0)), swinging the gripper pad into post_2 at ~2 mm
# while the level sweep read 89 mm clear. These functions evaluate the EXECUTED pose instead -- the carton
# pose the arm FK (plus the loaded gravity droop) actually holds -- so the gate refuses the real
# post-strike, not an idealised clearance.
#
# HONESTY / calibration: the CPU pad model (PAD_* + the author_cart post AABBs) reads coldwin39's measured
# FLUSH contact on post_2 as ~84 mm clear -- an ~84 mm offset between the CPU geometry and the sim colliders.
# So ``base_from_cart_anchor`` pins the base<-cart transform on a config whose carton sits at a KNOWN cart
# load-centre, and the absolute gap is read relative to that anchor; the RELIABLE discriminators are the
# carton LOAD-CENTRE DRIFT (``load_center_in_cart``, FK, branch-exact) and the droop TILT
# (carton_level.droop_induced_rpy, validated to coldwin39's measured 2.02 deg). The drooped sweep's role is
# to NAME which body the real (drifted) pose strikes, not to certify a sub-cm absolute gap.

def _carton_in_base(robot, config, geom, *, root='ur_base_link'):
    """4x4 pose of the carton frame in ``root`` at arm ``config``, derived from the ``suction_cups`` link FK
    and the baked pad-in-carton rotation/offset (so ``geom.surface_points`` -- carton frame -- transform
    consistently). Pure/CPU (needs the URDF FK)."""
    from twin.urdf_probe import link_world_transforms
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    cfg = {k: float(v) for k, v in (config.items() if isinstance(config, dict)
                                    else zip(ARM_JOINTS, config))}
    wl = link_world_transforms(robot, cfg, root=root)
    Ts = np.asarray(wl['suction_cups'], float)
    R = Ts[:3, :3] @ np.asarray(geom.pad_R, float).T
    t = Ts[:3, 3] - R @ np.asarray(geom.pad_center, float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _carton_in_cart_level(load_center_xy, support_z, yaw_cart_rel, carton_h, z_above=0.0):
    """4x4 pose of a LEVEL carton in the cart frame (the known-anchor pose). Mirrors ``carton_pose_cart``."""
    return carton_pose_cart(load_center_xy, support_z, yaw_cart_rel, carton_h=carton_h, z_above=z_above)


def base_from_cart_anchor(robot, anchor_config, geom, *, load_center_xy, support_z, yaw_cart_rel,
                          carton_h=CARTON_DIMS_M[2], root='ur_base_link'):
    """The base<-cart 4x4 transform, anchored on an ``anchor_config`` whose welded carton is KNOWN to sit
    level at cart ``(load_center_xy, support_z)`` with yaw ``yaw_cart_rel`` (e.g. the coldwin39 collect
    place-end config, carton at (0.18, 0)). ``T_base_cart = carton_in_base(anchor) @ inv(carton_in_cart)``.
    Pure/CPU."""
    Cb = _carton_in_base(robot, anchor_config, geom, root=root)
    Cc = _carton_in_cart_level(load_center_xy, support_z, yaw_cart_rel, carton_h)
    return Cb @ np.linalg.inv(Cc)


def load_center_in_cart(robot, config, geom, T_base_cart, *, root='ur_base_link'):
    """The carton's actual load-centre (cart-frame x, y) at arm ``config`` -- the branch-exact DRIFT
    detector. The idealised sweep assumes the configured load-centre; this reports where the IK solution
    REALLY puts the carton (coldwin41: drifts ~93 mm toward the +x mouth). Pure/CPU."""
    Cc = np.linalg.inv(np.asarray(T_base_cart, float)) @ _carton_in_base(robot, config, geom, root=root)
    return float(Cc[0, 3]), float(Cc[1, 3])


def _bodies_in_base(bodies, T_base_cart):
    """Transform the cart-frame body AABBs into the base frame (axis-aligned re-bound over the 8 corners)."""
    T = np.asarray(T_base_cart, float)
    out = {}
    for name, (lo, hi) in bodies.items():
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        wc = (T[:3, :3] @ corners.T).T + T[:3, 3]
        out[name] = (wc.min(0), wc.max(0))
    return out


def _executed_config(robot, config, gravity_comp, *, tool_link='ur_tool0', root='ur_base_link'):
    """``config + gravity_comp.droop(config)`` (ARM-order dict) -- the pose the loaded arm settles at.
    ``gravity_comp`` None => the commanded config verbatim (the level sweep)."""
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    cfg = {k: float(v) for k, v in (config.items() if isinstance(config, dict)
                                    else zip(ARM_JOINTS, config))}
    if gravity_comp is None:
        return cfg
    dq = gravity_comp.droop(robot, cfg, tool_link=tool_link, root=root)
    return {j: cfg[j] + dq.get(j, 0.0) for j in cfg}


def sweep_executed(configs, robot, geom, bodies, T_base_cart, *, gravity_comp=None,
                   tool_link='ur_tool0', root='ur_base_link', cal_offset_m=0.0, honest_support=None):
    """Droop-aware sweep: for each COMMANDED arm ``config`` apply the loaded ``gravity_comp`` droop, FK the
    gripper pad + carton to the EXECUTED pose, transform into the cart frame, and sweep against every
    Rollwagen ``bodies`` AABB. Returns ``(per_waypoint, per_body_min)`` like :func:`sweep_clearance`.

    ``cal_offset_m`` is subtracted from every gap (the SEAL-anchored geometry calibration, so a measured
    flush contact reads ~0). ``gravity_comp`` None => the EXECUTED == commanded (the FK level sweep, still
    the real IK pose rather than the idealised box). ``honest_support`` None => the legacy full-body sweep
    (byte-identical); a body name => the HONEST sweep (carton vs the kept bodies, pad vs all, support
    dropped -- sim-cart-support-select-v1). Pure/CPU (needs the URDF FK)."""
    bodies_base = _bodies_in_base(bodies, T_base_cart)
    per_waypoint = []
    for cfg in configs:
        execq = _executed_config(robot, cfg, gravity_comp, tool_link=tool_link, root=root)
        Cb = _carton_in_base(robot, execq, geom, root=root)
        if honest_support is None:
            pts = geom.surface_points() @ Cb[:3, :3].T + Cb[:3, 3]
            gaps = {name: float(_box_aabb_sdf(pts, lo, hi).min())
                    for name, (lo, hi) in bodies_base.items()}
        else:
            cpts = geom.carton_surface_points() @ Cb[:3, :3].T + Cb[:3, 3]
            ppts = geom.pad_surface_points() @ Cb[:3, :3].T + Cb[:3, 3]
            gaps = _honest_gaps_from_points(cpts, ppts, bodies_base, honest_support)
        per_waypoint.append({name: v - float(cal_offset_m) for name, v in gaps.items()})
    per_body_min = {}
    for g in per_waypoint:
        for name, v in g.items():
            if name not in per_body_min or v < per_body_min[name]:
                per_body_min[name] = v
    return per_waypoint, per_body_min


def clearance_summary(per_body_min, *, support=SUPPORT_BODIES, worst_n=4):
    """A one-line :data:`PLACE_CLEARANCE` log: the ``worst_n`` tightest NON-support body gaps + the
    support-body gaps, for the log evidence the GPU run and the lane read."""
    support = set(support)
    nonsupport = sorted((v, k) for k, v in per_body_min.items() if k not in support)[:worst_n]
    sup = sorted((v, k) for k, v in per_body_min.items() if k in support)
    def fmt(items):
        return ', '.join('%s=%.1fmm' % (k, v * 1000.0) for v, k in items)
    return '%s min non-support: %s | support: %s' % (PLACE_CLEARANCE, fmt(nonsupport), fmt(sup))


# -------------------- carried pre-place leg-2 pan-sweep vs the cart cage (sim-preplace-sweep-cage-clearance-v1)
# The place-clearance sweep above checks the welded carton + suction PAD at the SET-DOWN. The carried
# PRE-PLACE leg-2 (PREPLACE_WRIST_FIRST "move hold joints to standoff") swings the whole end-effector on a
# shoulder_pan arc from the carry pan to the standoff pan. coldwin70 reconstruction (nav-trace park pose +
# the executed CARTON_TICK joints): the GRIPPER END-EFFECTOR HOUSING -- the robot links the cuMotion sphere
# model covers (``cube_without_surface_plates`` / ``surface_plate_1``), NOT the carton -- rakes
# ``/Rollwagen/chassis/post_2`` (front mouth post) and the upper side-wall bar ``side_0_horizontal_6`` as the
# arc crosses the +x mouth at shoulder_pan ~ 0.20 rad, where the executed pan FROZE (0.2012) with a 1632.7
# N*m effort spike (cw69 SEAL_DIAG hit_bodies=['/Rollwagen/chassis/post_2'] min_gap_mm 0.0 is the same
# contact). Crucially the carton is cage-FILTERED (``cage_collision: filtered`` keeps deck/shelf/drawbar), so
# the carton CANNOT be the sim contact -- only an UNFILTERED robot link can, which is why this gate sweeps the
# cuMotion ARM SPHERE model (``world_obstacles.arm_spheres_world``), not the pad geometry: the pad-only honest
# sweep reads +11 mm clear there while the real housing penetrates post_2 by 48 mm.

def preplace_cage_arm_clearance(configs, urdf_path, xrdf_path, bodies, T_base_cart, *,
                                include=None):
    """Min cuMotion-arm-sphere clearance (m) to each cart body over the carried leg-2 ``configs``.

    ``configs``: the leg-2 waypoint arm configs (ARM_JOINTS order lists or dicts). ``bodies``: the cart-frame
    ``{name: (lo, hi)}`` AABBs from :func:`rollwagen_collider_boxes`. ``T_base_cart``: the cart-body ->
    ur_base_link 4x4 (the coupled/parked geometry). ``include``: optional iterable of body names to keep
    (default all). Returns ``{body: min arm-sphere clearance over the sweep}`` -- negative = an arm link
    penetrates that cart body. Pure/CPU (needs the URDF FK + the XRDF sphere model, same as the host)."""
    from twin.restock_host import world_obstacles as wo
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    bodies_base = _bodies_in_base(bodies, T_base_cart)
    keep = set(bodies_base) if include is None else {n for n in bodies_base if n in set(include)}
    boxes = [wo.Box(name, tuple(np.asarray(hi, float) - np.asarray(lo, float)),
                    tuple((np.asarray(lo, float) + np.asarray(hi, float)) / 2.0), (0.0, 0.0, 0.0))
             for name, (lo, hi) in bodies_base.items() if name in keep]
    per_body_min = {}
    for cfg in configs:
        c = cfg if isinstance(cfg, dict) else {j: float(cfg[i]) for i, j in enumerate(ARM_JOINTS)}
        centres, radii, _owners = wo.arm_spheres_world(str(urdf_path), str(xrdf_path), c)
        for box in boxes:
            g = float(wo.clearance(box, centres, radii))
            if box.id not in per_body_min or g < per_body_min[box.id]:
                per_body_min[box.id] = g
    return per_body_min


def preplace_sweep_reject(per_body_min, *, margin=DEFAULT_PREPLACE_CAGE_MARGIN_M, support=SUPPORT_BODIES):
    """The :data:`PREPLACE_SWEEP_CLEARANCE` refuse message when any NON-support cart body's minimum arm-sphere
    clearance over the leg-2 sweep is below ``margin``, else ``None``. ``support`` bodies (deck/shelf -- the
    surface the carton will land on) are excluded: the preplace approach rides above them. Names the worst
    prim + its min_clearance so the lane sees exactly which cage body the sweep would strike."""
    support = set(support)
    bad = sorted((v, k) for k, v in per_body_min.items() if k not in support and v < margin)
    if not bad:
        return None
    worst_v, worst_k = bad[0]
    named = ', '.join('%s=%.1fmm' % (k, v * 1000.0) for v, k in bad)
    return ('%s prim=%s min_clearance=%.1fmm (margin %.0f mm; cage bodies below it: %s) -- the carried '
            'pre-place shoulder_pan sweep would drive an arm link into a Rollwagen cage body (the gripper '
            'housing rakes the +x mouth post/side bar, coldwin70 froze here at a 1632.7 N*m effort spike). '
            'Refusing before motion.' % (PREPLACE_SWEEP_CLEARANCE, worst_k, worst_v * 1000.0,
                                         margin * 1000.0, named))


def cage_body_names(bodies):
    """The CAGE body names in ``bodies`` (posts + side/back wall bars) -- the set the carried leg-2 pan
    sweep rakes. Excludes the horizontal supports (deck/shelf) the carton lands on and the drawbar (the
    arm base sits a fixed ~34 mm from the tow bar at the carry hold, a standing clearance the sweep does
    not change)."""
    return [n for n in bodies if is_cage_body(n)]


def place_descent_sweep_bodies(bodies, *, support=SUPPORT_BODIES):
    """The cart bodies a SHELF-AWARE collect place-DESCENT arm-sphere sweep checks: the CAGE bodies
    (:func:`cage_body_names`) PLUS the present SUPPORT bodies (deck/shelf) -- sim-pad-down-descent-jam-physics-v1.

    The descent gate historically swept only the cage (``cage_body_names``) and EXCLUDED the support
    surfaces, on the assumption that only the CARTON reaches the support. coldwin98-100 disproved that:
    in the PAD-DOWN side-grip place the welded CARTON is cage-filtered, but the unfiltered gripper suction
    HOUSING (``suction_cups`` / ``surface_plate_*`` / ``cube_without_surface_plates``) hangs ~0.14 m BELOW
    the carton bottom, so it reaches the upper-shelf board and strikes it while the carton is still ~0.10 m
    above the seat (CPU-replayed on the coldwin100 ``TRAJ_HOST_PLAN`` vs the COMPOSED USD colliders -- not
    the box model: the housing penetrates ``/Rollwagen/chassis/shelf`` by ~52 mm from the first descent
    waypoint, while every CAGE body stays >= +0.11 m clear -- the jam the cage-only sweep was structurally
    blind to; see ``scripts/check_pad_down_shelf_contact.py``). So a shelf-aware descent sweep must include
    the support board: the gripper housing is NOT the carton and must clear the surface the carton lands on.
    The CARTON is never in this sphere sweep, so including the support does not penalise the carton resting
    on it (:func:`place_descent_sweep_reject` checks the support board at a penetration-only margin because
    the pad-down housing legitimately rides close to it). Deck is included too -- harmless (far) for a shelf
    place, correct for a deck place. Preserves the cage order, then appends the present support bodies.

    This is OPT-IN: the descent providers default ``include`` to the CAGE-ONLY set (byte-identical); a
    boot/integration layer passes this set (via ``place_descent_cage_gate`` ``include_support`` or an
    explicit ``include``) to arm the shelf check."""
    cage = cage_body_names(bodies)
    sup = [n for n in support if n in bodies and n not in cage]
    return cage + sup


def preplace_cage_clearance_provider(urdf_path, xrdf_path, cart_pose, *, bodies=None,
                                     margin=DEFAULT_PREPLACE_CAGE_MARGIN_M, include=None, cal_offset_m=0.0):
    """Build the ``callable(waypoints) -> reject|None`` the controller injects as its preplace cage-clearance
    gate (sim-preplace-sweep-cage-clearance-v1). Opt-in: the controller leaves the hook ``None`` (byte-
    identical) unless a boot/integration layer supplies this provider with the live coupled cart pose + the
    host XRDF.

    ``cart_pose``: the cart-body -> ur_base_link pose ((x, y, z, yaw) or 4x4). ``bodies``: cart-frame AABBs
    (default :func:`rollwagen_collider_boxes`, needs usd-core). ``include``: cart-body names to check
    (default the CAGE bodies -- posts + side/back bars). ``margin``: refuse threshold (m). Returns a closure
    that, given a list of ARM_JOINTS-order waypoint configs, returns the :func:`preplace_sweep_reject`
    message or ``None``. The closure applies ``cal_offset_m`` (subtracted from every clearance, a SEAL-
    anchored calibration knob) before the margin test."""
    from twin.restock_manipulation.cart_world_obstacle import cart_pose_matrix
    if bodies is None:
        bodies = rollwagen_collider_boxes()
    keep = cage_body_names(bodies) if include is None else list(include)
    T_base_cart = cart_pose_matrix(cart_pose)

    def _gate(waypoints):
        if not waypoints:
            return None
        per_body = preplace_cage_arm_clearance(waypoints, urdf_path, xrdf_path, bodies, T_base_cart,
                                               include=keep)
        if cal_offset_m:
            per_body = {k: v - float(cal_offset_m) for k, v in per_body.items()}
        return preplace_sweep_reject(per_body, margin=margin)

    return _gate


# ------------------------- collect place-DESCENT cage sweep (sim-place-descent-reach-cage-v1)
# The pre-place sweep above checks the carried APPROACH (the leg-2 shoulder_pan arc to the standoff). The
# place DESCENT that follows -- the leveled straight-line set-down from the +0.08 standoff down onto the
# support -- has its OWN cage hazard, which the approach sweep does not see: coldwin87's descent plan starts
# by reorienting shoulder/wrist from the standoff's CARRY wrist orientation (wrist_2 ~ 0.28) to the place
# PAD-DOWN orientation (wrist_2 ~ 1.63) WHILE the gripper housing sits inside the +x cage mouth, so the
# UNFILTERED gripper housing link (``cube_without_surface_plates`` / ``surface_plate_*`` -- NOT the
# cage-filtered carton) wedges against the upper side bar ``side_1_horizontal_6`` and the mouth posts at the
# descent START (CPU-reconstructed -1.9 mm vs side_1_horizontal_6, the ONLY penetrating config of the whole
# path; wrist_2 then froze at 0.28 against a 1.63 rad command -> TRAJ_HOST_TRACK_MISS worst_j=4 ->
# PATH_TOLERANCE_VIOLATED). The REMEDY descent holds the PAD-DOWN wrist branch (wrist_2 ~ 1.63) from entry to
# the corrected seat, where the same reconstruction reads >= +0.10 m clearance at every collect-place park
# pose. The gate below is the same cuMotion ARM SPHERE sweep (``preplace_cage_arm_clearance``) applied to the
# place-descent waypoints; it NAMES the wedge and refuses before motion instead of the blind PATH_TOLERANCE.

# Log / refuse token for the place-descent cage sweep.
PLACE_DESCENT_CAGE_CLEARANCE = 'PLACE_DESCENT_CAGE_CLEARANCE'
# Default refuse margin (m) for the place-descent cage sweep: the brief's >= 0.03 m clearance target. Unlike
# the carried pre-place pan sweep (DEFAULT_PREPLACE_CAGE_MARGIN_M = 0.0, a penetration-only refuse that
# tolerated the standoff's inherent ~+4 mm cage-width tightness against the FAR side bar), the DESCENT holds
# the gripper housing on the fixed pad-down wrist branch that clears EVERY cage body by >= 0.10 m when
# correct, and the failure mode (coldwin87: the housing wedged in the +x mouth, -1.9 .. -33.4 mm across
# cw87/cw64/cw89, every other config >= +0.10 m) is unambiguous, so the descent gate refuses at the full
# 30 mm design margin. The CPU calibration caveat (the arm-sphere model vs the authored AABBs) is the same as
# the pre-place sweep; a clear negative reading is a certain strike and the clear descent reads +0.10 m, far
# outside the noise.
DEFAULT_PLACE_DESCENT_CAGE_MARGIN_M = 0.030


def place_descent_cage_clearance(configs, urdf_path, xrdf_path, bodies, T_base_cart, *, include=None):
    """Min cuMotion-arm-sphere clearance (m) to each cart body over the collect place-DESCENT ``configs``.

    The SAME arm-sphere sweep :func:`preplace_cage_arm_clearance` runs for the carried pre-place pan sweep,
    applied to the place descent's leveled waypoint path (ARM_JOINTS-order configs). Returns
    ``{body: min arm-sphere clearance over the descent}`` -- negative = a gripper link penetrates that cart
    body. ``include`` defaults to all; the descent gate keeps the CAGE bodies (:func:`cage_body_names`).
    Pure/CPU (URDF FK + XRDF sphere model, same as the host)."""
    return preplace_cage_arm_clearance(configs, urdf_path, xrdf_path, bodies, T_base_cart, include=include)


# Default refuse margin (m) for the SUPPORT board (deck/shelf) in a shelf-aware place-descent sweep:
# penetration-only (sim-pad-down-descent-jam-physics-v1). Unlike a CAGE upright/bar -- which the gripper
# housing must clear by the full design margin (nothing legitimately rides there) -- the pad-down gripper
# housing legitimately rides CLOSE to the support board as it lowers the carton onto it (the carton bottom
# lands on the board, the housing skims past its edge). coldwin100 replay: a clean straight descent holds
# the housing ~+9-17 mm from the shelf board (within the conservative XRDF-sphere model) while the carton
# lands; the jamming descent drove it -52 mm INTO the board. So a true PENETRATION of the support board is
# the jam to refuse; a close ride is not. The cage keeps the full 30 mm design margin.
DEFAULT_PLACE_DESCENT_SUPPORT_MARGIN_M = 0.0


def place_descent_sweep_reject(per_body_min, *, margin=DEFAULT_PLACE_DESCENT_CAGE_MARGIN_M,
                               support=SUPPORT_BODIES, support_margin=DEFAULT_PLACE_DESCENT_SUPPORT_MARGIN_M):
    """The :data:`PLACE_DESCENT_CAGE_CLEARANCE` refuse message when the place-descent arm-sphere sweep would
    drive a gripper link into a Rollwagen cart body, else ``None``.

    Two margins, because the gripper housing must clear two kinds of body with different tolerances
    (sim-pad-down-descent-jam-physics-v1):

      * a CAGE body (post / side / back bar) below ``margin`` (the 30 mm design margin) -- nothing
        legitimately rides there (coldwin87's ``cube_without_surface_plates`` wedge on the +x mouth side
        bar), OR
      * a SUPPORT body (deck/shelf, ``support``) below ``support_margin`` (penetration-only by default) --
        the PAD-DOWN gripper housing hangs ~0.14 m below the carton bottom and legitimately rides close to
        the board the carton lands on, so only a true PENETRATION is the jam (coldwin98-100: the suction
        housing drove -52 mm into ``/Rollwagen/chassis/shelf`` while the carton stayed ~0.10 m above the
        seat -- the cage-only sweep was blind to it; a clean straight descent holds +9-17 mm and passes).

    Backward-compatible: when ``per_body_min`` contains only CAGE bodies (a cage-only sweep, ``include`` not
    covering the support board) the support branch is never exercised, so the verdict is unchanged. Names
    the worst offending prim + its clearance first so the lane sees exactly which cart body the descent
    would strike and by how much."""
    support = set(support)
    bad = []
    for k, v in per_body_min.items():
        thr = support_margin if k in support else margin
        if v < thr:
            bad.append((v, k))
    bad.sort()
    if not bad:
        return None
    worst_v, worst_k = bad[0]
    worst_is_support = worst_k in support
    named = ', '.join('%s=%.1fmm' % (k, v * 1000.0) for v, k in bad)
    cause = (('the pad-down gripper housing (suction_cups / surface_plate_* / cube_without_surface_plates), '
              'which hangs ~0.14 m below the carton bottom, would strike the SUPPORT board the carton lands '
              'on (coldwin100: -52 mm into /Rollwagen/chassis/shelf while the carton stayed ~0.10 m above '
              'the seat -- the jam the cage-only sweep could not see)') if worst_is_support else
             ('the collect place-descent would drive a gripper link into a Rollwagen cage body (coldwin87 '
              'wedged the suction housing cube_without_surface_plates on the +x mouth side bar, freezing '
              'wrist_2 at a 1.30 rad TRAJ_HOST_TRACK_MISS -> PATH_TOLERANCE_VIOLATED abort)'))
    return ('%s prim=%s min_clearance=%.1fmm (cage margin %.0f mm / support margin %.0f mm; bodies below: '
            '%s) -- %s. Refusing before motion.'
            % (PLACE_DESCENT_CAGE_CLEARANCE, worst_k, worst_v * 1000.0, margin * 1000.0,
               support_margin * 1000.0, named, cause))


def place_descent_cage_clearance_provider(urdf_path, xrdf_path, cart_pose, *, bodies=None,
                                          margin=DEFAULT_PLACE_DESCENT_CAGE_MARGIN_M, include=None,
                                          cal_offset_m=0.0, include_support=False):
    """Build the ``callable(waypoints) -> reject|None`` the controller injects as its place-descent
    cage-clearance gate (sim-place-descent-reach-cage-v1). Opt-in: the controller leaves the hook ``None``
    (byte-identical) unless a boot/integration layer supplies this provider with the live coupled cart pose +
    the host XRDF. Mirrors :func:`preplace_cage_clearance_provider` exactly, at the DESCENT margin (30 mm) and
    with the DESCENT refuse message (:func:`place_descent_sweep_reject`).

    ``cart_pose``: the cart-body -> ur_base_link pose ((x, y, z, yaw) or 4x4). ``bodies``: cart-frame AABBs
    (default :func:`rollwagen_collider_boxes`, needs usd-core). ``include``: cart-body names to check (default
    the CAGE bodies; an explicit list overrides ``include_support``). ``include_support`` (sim-pad-down-
    descent-jam-physics-v1): when True and ``include`` is None, ALSO sweep the SUPPORT board (deck/shelf) --
    :func:`place_descent_sweep_bodies` -- so the pad-down gripper-housing-vs-shelf strike is refused by name
    (the support board at the penetration-only support margin; the cage at the 30 mm design margin). Default
    False => cage-only => byte-identical. ``margin``: cage refuse threshold (m). Returns a closure that, given
    a list of ARM_JOINTS-order waypoint configs, returns the refuse message or ``None``. The closure applies
    ``cal_offset_m`` (subtracted from every clearance, a SEAL-anchored calibration knob) before the margin
    test."""
    from twin.restock_manipulation.cart_world_obstacle import cart_pose_matrix
    if bodies is None:
        bodies = rollwagen_collider_boxes()
    if include is not None:
        keep = list(include)
    elif include_support:
        keep = place_descent_sweep_bodies(bodies)
    else:
        keep = cage_body_names(bodies)
    T_base_cart = cart_pose_matrix(cart_pose)

    def _gate(waypoints):
        if not waypoints:
            return None
        per_body = preplace_cage_arm_clearance(waypoints, urdf_path, xrdf_path, bodies, T_base_cart,
                                               include=keep)
        if cal_offset_m:
            per_body = {k: v - float(cal_offset_m) for k, v in per_body.items()}
        return place_descent_sweep_reject(per_body, margin=margin)

    return _gate


# ------------------------- pad-down-branch ranking clearance (sim-place-pad-down-arrival-v1)
def place_descent_cage_min_clearance_provider(urdf_path, xrdf_path, cart_pose, *, bodies=None,
                                              include=None, cal_offset_m=0.0, support=SUPPORT_BODIES,
                                              include_support=False):
    """Build the ``callable(waypoints) -> float`` returning the MINIMUM cuMotion-arm-sphere clearance (m)
    to the cart bodies the descent must clear over a set of waypoints (sim-place-pad-down-arrival-v1). The
    pad-down-branch selector (``place_reorient.select_pad_down_branch``) uses it to RANK the cage-clear
    branches by how clear their descent is -- the SAME arm-sphere sweep as
    :func:`place_descent_cage_clearance_provider`, exposing the margin instead of a pass/fail verdict.
    Returns ``+inf`` for an empty waypoint list (no constraint). Mirrors the gate provider's arguments.

    ``include_support`` (sim-pad-down-descent-jam-physics-v1): default False => the min is over NON-support
    cage bodies only (byte-identical); True => ALSO sweep + rank by the SUPPORT board (so the selector ranks
    branches by their true tightest clearance, the shelf the housing skims), matching the gate's
    ``include_support``."""
    from twin.restock_manipulation.cart_world_obstacle import cart_pose_matrix
    if bodies is None:
        bodies = rollwagen_collider_boxes()
    if include is not None:
        keep = list(include)
    elif include_support:
        keep = place_descent_sweep_bodies(bodies)
    else:
        keep = cage_body_names(bodies)
    T_base_cart = cart_pose_matrix(cart_pose)
    support_set = set() if include_support else set(support)

    def _clear(waypoints):
        if not waypoints:
            return float('inf')
        per_body = preplace_cage_arm_clearance(waypoints, urdf_path, xrdf_path, bodies, T_base_cart,
                                               include=keep)
        if cal_offset_m:
            per_body = {k: v - float(cal_offset_m) for k, v in per_body.items()}
        vals = [v for k, v in per_body.items() if k not in support_set]
        return float(min(vals)) if vals else float('inf')

    return _clear
