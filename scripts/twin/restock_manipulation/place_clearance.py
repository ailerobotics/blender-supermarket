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

# Default minimum gap (m) the swept gripper+carton must keep from every NON-support Rollwagen body
# until the carton bottom is flat on the support. 30 mm matches the brief's margin.
DEFAULT_MARGIN_M = 0.030

# The SUPPORT bodies the carton is PLACED on (allowed to touch at the set-down). The mission's
# "place-on-deck" ground-truth slot targets ``support_z_world`` 0.9325 m, which is the UPPER SHELF top
# (0.933), not the low deck (0.225) -- so both candidate support surfaces are excluded from the refuse
# rule; every other Rollwagen body (posts, side/back bars, drawbar) must stay clear.
SUPPORT_BODIES = ('deck', 'shelf')

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

    def surface_points(self, step=0.01):
        """Surface sample points (N x 3) of the carton AND the pad, in the CARTON frame."""
        half = np.asarray(self.carton_dims, float) / 2.0
        carton = _box_surface_points(half, step)
        pad_local = _box_surface_points(np.asarray(self.pad_size, float) / 2.0, step)
        pad = pad_local @ np.asarray(self.pad_R, float).T + np.asarray(self.pad_center, float)
        return np.vstack([carton, pad])


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

def _gaps_at(T_carton_cart, geom, bodies):
    """``{body: min signed gap (m)}`` between the gripper+carton at ``T_carton_cart`` and every cart body."""
    pts = geom.surface_points() @ np.asarray(T_carton_cart, float)[:3, :3].T + np.asarray(T_carton_cart, float)[:3, 3]
    return {name: float(_box_aabb_sdf(pts, lo, hi).min()) for name, (lo, hi) in bodies.items()}


def sweep_clearance(carton_poses, geom, bodies):
    """Sweep the gripper+carton (``geom``) along ``carton_poses`` (list of 4x4 cart-frame carton poses)
    against ``bodies`` (``{name: (lo, hi)}`` from :func:`rollwagen_collider_boxes`).

    Returns ``(per_waypoint, per_body_min)``: ``per_waypoint`` is a list of ``{body: gap}`` dicts (the
    minimum gap per body at each waypoint), ``per_body_min`` is ``{body: min over all waypoints}`` -- the
    minimum gap per body over the whole path (the quantity the refuse gate reads)."""
    per_waypoint = [_gaps_at(T, geom, bodies) for T in carton_poses]
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


def clearance_summary(per_body_min, *, support=SUPPORT_BODIES, worst_n=4):
    """A one-line :data:`PLACE_CLEARANCE` log: the ``worst_n`` tightest NON-support body gaps + the
    support-body gaps, for the log evidence the GPU run and the lane read."""
    support = set(support)
    nonsupport = sorted((v, k) for k, v in per_body_min.items() if k not in support)[:worst_n]
    sup = sorted((v, k) for k, v in per_body_min.items() if k in support)
    def fmt(items):
        return ', '.join('%s=%.1fmm' % (k, v * 1000.0) for v, k in items)
    return '%s min non-support: %s | support: %s' % (PLACE_CLEARANCE, fmt(nonsupport), fmt(sup))
