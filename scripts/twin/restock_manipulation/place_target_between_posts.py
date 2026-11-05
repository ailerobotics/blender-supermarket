"""CPU geometry + reach check for the collect place-on-cart target (sim-place-target-between-posts-v1).

WHAT THIS PROVES (read the evidence before trusting a number):

The brief asked to pick a place load-centre on the rollwagen UPPER shelf whose carton footprint +
descent clears every cage post with margin from the real park pose, because run coldwin69 stopped at
``PLACE_OBSTRUCTED prim=/Rollwagen/chassis/post_2 gap=0.00`` while coldwin64 placed cleanly at the
IDENTICAL target ``load_center_body_xy=(0.135, 0.0)``.

The measured runs (``data/sim-place-target-between-posts-v1/``) show the current spot is NOT
geometrically marginal at its nominal: the honest carton+pad sweep clears post_2 by ~185 mm there
(:func:`idealized_clearance`). The runs diverged for a different reason:

  * All three runs PARKED the base at the SAME world pose ``(-4.61, 14.77)`` (nav-trace, t<=195).
  * In coldwin64 the base STAYED there through the place and the carton landed at the cart-frame
    nominal ``(0.131, 0.015)`` -- cleared every post.
  * In coldwin68 / coldwin69 the base PHYSICALLY SHOVED ~0.33 m west DURING the loaded place phase
    (t~191 -> 227, still flagged ``parked=True``), so the arm -- executing in ``ur_base_link`` --
    set the welded carton down ~0.5 m off-nominal in cart-y ``(0.33, -0.50)`` / ``(0.33, -0.52)``,
    straight onto post_2.

So the executed place is effectively BASE-relative EVEN THOUGH the place TARGET is already resolved
cart-relative from the live cart prim (``isaac_factory._gt_place_target`` ->
``ground_truth_collect.slot_pose``): the base moving between target-resolve and set-down is what
breaks the cart-relative landing. A ~0.5 m cart-y excursion cannot be absorbed by ANY spot on the
0.64 m-wide shelf with a 0.45 m carton -- :func:`evaluate` names this by the numbers.

This module therefore delivers two honest things, not one false "fix":

  1. :func:`idealized_clearance` + :func:`choose_spot` -- the between-posts spot that MAXIMISES the
     static post clearance (the symmetric centre ``(0.0, 0.0)``: post_2/post_3 ~258 mm vs ~185 mm at
     the old ``(0.135, 0.0)``), reach-proven from the real park pose. This is the nominal to adopt
     ONCE the base is held; it buys margin against small residual drift, it does NOT absorb the shove.
  2. :func:`evaluate` -- the verdict that under the MEASURED execution drift (:func:`measured_drift`)
     NO shelf spot clears post_2, naming the loaded-place base shove as the blocking cause. The real
     fix (hold the base through the loaded set-down / re-resolve the place target at descent time) is
     the collect place-descent control, out of this lane's scope.

Pure numpy + ``usd-core`` (cart colliders via :mod:`place_clearance`); the reach check also needs the
dataset mobile URDF (``level_place_ik`` FK). CPU only -- no Isaac, ROS or GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import place_clearance as pc

# Log token a caller/preflight can print.
PLACE_TARGET = 'PLACE_TARGET_BETWEEN_POSTS'

# --- cart geometry constants (cart BODY frame; match place_clearance.rollwagen_collider_boxes) ------
# Upper shelf TOP in the cart body frame (author_cart: shelf z in [0.907, 0.933]); the place support.
SHELF_TOP_CART_Z = 0.933
# The shelf TOP in world at the staged cart height (scenario initial_staging.cart_support_z_world /
# the runtime manipulation.cart_placement.support_z_world the runs used).
SUPPORT_Z_WORLD = 0.9707
# => the cart body z-origin sits at this world height (SUPPORT_Z_WORLD - SHELF_TOP_CART_Z).
CART_Z_ORIGIN_WORLD = SUPPORT_Z_WORLD - SHELF_TOP_CART_Z
# The carton's cart-relative yaw for the ONE rollwagen slot (ground_truth_collect: -90 deg, gripped
# face toward the hitch); the footprint is 0.30 m (cart x) x 0.45 m (cart y).
YAW_CART_REL = -math.pi / 2.0

# The fixed clear-region preplace shape the host seeds the standoff IK from (ros_host_config.json
# collect_place_preplace_shape); the reach check seeds level_place_ik from it, like the controller.
DEFAULT_PREPLACE_SHAPE = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)
# The standoff raise above the support the controller uses (place_above_clearance_m default 0.08 m).
DEFAULT_STANDOFF_M = 0.08
# The FK tool link the place reach targets (controller._place_fk_link prefers this).
PLACE_TCP_LINK = 'gripper_pad_tcp'
# ur_base_link mount height on the mobile base (mobile URDF ur_base_link z); used to build the
# world<-base transform from a 2-D (x, y, yaw) park pose.
UR_BASE_MOUNT_Z = 0.26

# Minimum static clearance margin to any NON-support cage body (the brief's 0.04 m floor).
DEFAULT_MARGIN_M = 0.04

# The posts the drift drives the carton into.
POSTS = ('post_0', 'post_1', 'post_2', 'post_3')
# The shelf half-width to the inner face of post_2/post_3 in cart-y (|post inner y| = 0.307).
SHELF_POST_INNER_Y = 0.307


# --------------------------------------------------- measured run evidence (do NOT silently edit) ----
# From data/sim-place-target-between-posts-v1/ (nav-trace + host-stack of the three runs). Each entry:
# the base chassis (x, y, yaw_rad) and cart body (x, y, yaw_rad) world poses at the PLACE tick, the
# measured carton-centre position in the CART frame at that tick, and whether the carton cleared.
# The cart-frame carton centre is the quantity that decides a post strike; the TARGET was (0.135, 0.0)
# in every run, so the spread of ``carton_in_cart`` is the EXECUTION drift, not a target difference.
@dataclass(frozen=True)
class RunEvidence:
    name: str
    base_world: tuple      # (x, y, yaw_rad) chassis at the place tick
    cart_world: tuple      # (x, y, yaw_rad) cart body at the place tick
    park_base_world: tuple  # (x, y, yaw_rad) chassis where ALL runs parked (t<=195)
    carton_in_cart: tuple  # (x, y) measured carton centre in the cart frame at the place tick
    cleared: bool


RUN_EVIDENCE = (
    RunEvidence('coldwin64', (-4.620, 14.743, math.radians(85.30)),
                (-4.5977, 15.9859, math.radians(-92.48)), (-4.610, 14.767, math.radians(85.2)),
                (0.131, 0.015), True),
    RunEvidence('coldwin68', (-4.930, 14.610, math.radians(85.30)),
                (-4.6056, 15.9728, math.radians(-93.20)), (-4.610, 14.768, math.radians(85.3)),
                (0.332, -0.500), False),
    RunEvidence('coldwin69', (-4.963, 14.615, math.radians(84.90)),
                (-4.6041, 15.9745, math.radians(-93.40)), (-4.610, 14.770, math.radians(84.9)),
                (0.330, -0.516), False),
)

# The TARGET load-centre every run commanded (runtime manipulation.cart_placement.load_center_body_xy).
OLD_TARGET_XY = (0.135, 0.0)
# The shipped between-posts nominal: the symmetric shelf centre (max min-post clearance; see choose_spot).
SHIPPED_TARGET_XY = (0.0, 0.0)


# ------------------------------------------------------------------------- measured execution drift

def measured_drift(target_xy=OLD_TARGET_XY, evidence=RUN_EVIDENCE):
    """The measured carton execution drift (cart frame) vs ``target_xy`` across the runs.

    Returns a dict:
      * ``per_run``   -- ``{name: (dx, dy)}`` carton_in_cart minus the target,
      * ``max_abs``   -- ``(max|dx|, max|dy|)`` over all runs (the worst-case envelope),
      * ``base_shove_m`` -- the largest base displacement from the common park pose at the place tick
                           (world distance) -- the mechanism behind the drift.

    The target landed near-nominal in the cleared run and ~0.5 m off in cart-y in the struck runs; the
    cart-y component is what drives the carton onto post_2."""
    per_run, dxs, dys, shoves = {}, [], [], []
    for e in evidence:
        dx = e.carton_in_cart[0] - float(target_xy[0])
        dy = e.carton_in_cart[1] - float(target_xy[1])
        per_run[e.name] = (dx, dy)
        dxs.append(abs(dx))
        dys.append(abs(dy))
        shoves.append(math.hypot(e.base_world[0] - e.park_base_world[0],
                                 e.base_world[1] - e.park_base_world[1]))
    return {'per_run': per_run, 'max_abs': (max(dxs), max(dys)), 'base_shove_m': max(shoves)}


def park_variance_m(evidence=RUN_EVIDENCE):
    """The lateral (cart-y) execution spread the new spot would have to absorb to never strike a post
    -- the ``max|dy|`` of :func:`measured_drift` from the OLD target. This is the measured quantity the
    brief's '0.04 m + park variance' margin adds; it is ~0.5 m, far larger than any shelf spot's slack."""
    return measured_drift(OLD_TARGET_XY, evidence)['max_abs'][1]


# ------------------------------------------------------------------------- idealised static clearance

def _geom():
    return pc.GripperCartonGeometry.shipped()


def idealized_clearance(load_center_xy, *, bodies=None, geom=None, yaw_cart_rel=YAW_CART_REL,
                        support='shelf', standoff_m=DEFAULT_STANDOFF_M, n=12):
    """Honest carton+pad sweep clearance for a LEVEL carton descending onto ``support`` at
    ``load_center_xy`` (cart frame), against every rollwagen body. Reuses the PR 336 honest sweep
    (carton vs the kept bodies, gripper pad vs all, support dropped). Returns a dict:
      * ``per_body_min`` -- ``{body: min signed gap m}`` over the descent,
      * ``min_nonsupport`` -- the tightest non-support gap (and its body),
      * ``posts`` -- ``{post: gap}`` for the four corner posts,
      * ``min_post`` -- the tightest of the four posts.
    This is the STATIC clearance at the nominal; it does NOT include the measured execution drift."""
    bodies = bodies if bodies is not None else pc.rollwagen_collider_boxes()
    geom = geom if geom is not None else _geom()
    poses = pc.place_descent_poses(load_center_xy, SHELF_TOP_CART_Z, yaw_cart_rel,
                                   carton_h=pc.CARTON_DIMS_M[2], standoff_m=standoff_m, n=n)
    _, per_body_min = pc.sweep_clearance(poses, geom, bodies, honest_support=support)
    ns = sorted((v, k) for k, v in per_body_min.items() if k not in pc.SUPPORT_BODIES)
    posts = {p: per_body_min[p] for p in POSTS if p in per_body_min}
    min_post = min(posts.items(), key=lambda kv: kv[1]) if posts else (None, float('inf'))
    return {'per_body_min': per_body_min,
            'min_nonsupport': (ns[0][1], ns[0][0]) if ns else (None, float('inf')),
            'posts': posts, 'min_post': (min_post[0], min_post[1])}


def executed_clearance(load_center_xy, drift_xy, *, bodies=None, geom=None, **kw):
    """Static clearance with the load-centre shifted by a measured execution ``drift_xy`` (cart frame)
    -- i.e. where the carton would ACTUALLY descend given that drift. Negative = a strike. Used to show
    the measured ~0.5 m cart-y drift strikes post_2 for every shelf spot."""
    shifted = (float(load_center_xy[0]) + float(drift_xy[0]),
               float(load_center_xy[1]) + float(drift_xy[1]))
    return idealized_clearance(shifted, bodies=bodies, geom=geom, **kw)


def worst_executed_clearance(load_center_xy, *, bodies=None, geom=None, evidence=RUN_EVIDENCE,
                             target_xy=OLD_TARGET_XY, **kw):
    """The tightest executed clearance when each run's MEASURED drift vector (carton_in_cart minus the
    commanded target, cart frame -- mouth-ward +x and post_2-ward -y) is applied to ``load_center_xy``.
    Returns ``(worst_min_nonsupport, worst_min_post, per_run)``; a near-zero/negative min post means the
    carton would descend onto a post under that shove. Reproduces the real landing: cw68's drift applied
    to the old target (0.135, 0) lands at ~(0.33, -0.52), onto post_2."""
    drift = measured_drift(target_xy, evidence)
    per_run = {}
    worst_ns = worst_post = float('inf')
    for name, dxy in drift['per_run'].items():
        ic = executed_clearance(load_center_xy, dxy, bodies=bodies, geom=geom, **kw)
        per_run[name] = {'min_nonsupport': ic['min_nonsupport'][1], 'min_post': ic['min_post'][1],
                         'min_post_body': ic['min_post'][0]}
        worst_ns = min(worst_ns, ic['min_nonsupport'][1])
        worst_post = min(worst_post, ic['min_post'][1])
    return worst_ns, worst_post, per_run


# ------------------------------------------------------------------------------------- reach check

def world_from_pose(x, y, z, yaw):
    """4x4 world transform from a planar ``(x, y, z, yaw)`` pose (yaw about +Z). Pure."""
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = [x, y, z]
    return T


def _preshape_up_in_tool(robot, preplace_shape, tool_link=PLACE_TCP_LINK):
    """The tool-frame direction that points world-up at the (level) preplace shape, so ``level_place_ik``
    levels the welded carton the same way the controller does (arm +Z = world up at the clear preplace)."""
    from twin.urdf_probe import link_world_transforms
    from .cad_tool import ARM_JOINTS
    wl = link_world_transforms(robot, dict(zip(ARM_JOINTS, [float(v) for v in preplace_shape])),
                               root='ur_base_link')
    Rtool = np.asarray(wl[tool_link], float)[:3, :3]
    return Rtool.T @ np.array([0.0, 0.0, 1.0])


def standoff_target_base(load_center_xy, base_world, cart_world, *, standoff_m=DEFAULT_STANDOFF_M,
                         support_z_world=SUPPORT_Z_WORLD):
    """The place STANDOFF target (tool origin) in ``ur_base_link`` for ``load_center_xy`` on the cart at
    the given world poses -- the cart-relative support point expressed in the base frame, raised by
    ``standoff_m`` (the controller's ``standoff = (place_x, place_y, place_z + clearance)``). Pure."""
    Tbase = world_from_pose(base_world[0], base_world[1], UR_BASE_MOUNT_Z, base_world[2])
    Tcart = world_from_pose(cart_world[0], cart_world[1], CART_Z_ORIGIN_WORLD, cart_world[2])
    w = Tcart @ np.array([float(load_center_xy[0]), float(load_center_xy[1]), 0.0, 1.0])
    p = np.linalg.inv(Tbase) @ np.array([w[0], w[1], float(support_z_world), 1.0])
    return np.array([p[0], p[1], p[2] + float(standoff_m)])


@dataclass(frozen=True)
class ReachResult:
    ok: bool
    pos_err_m: float
    tilt_deg: float
    config: tuple
    reason: str = ''


def reach_check(load_center_xy, base_world, cart_world, robot, *,
                preplace_shape=DEFAULT_PREPLACE_SHAPE, tool_link=PLACE_TCP_LINK,
                standoff_m=DEFAULT_STANDOFF_M, limits=None, tilt_bound_deg=2.0):
    """Prove the place STANDOFF above ``load_center_xy`` is reachable AND level from the given park pose,
    via the existing ``carton_level.level_place_ik`` seeded from the fixed preplace shape -- the same
    reach+level path the controller runs. Returns a :class:`ReachResult` (ok, position error, predicted
    tilt, solved config). ``ok`` requires pos_err <= 1 mm and tilt <= ``tilt_bound_deg`` (the never-
    loosened 2.0 deg bound)."""
    from twin.urdf_probe import link_world_transforms
    from . import carton_level as cl
    from .cad_tool import ARM_JOINTS
    up_in_tool = _preshape_up_in_tool(robot, preplace_shape, tool_link=tool_link)
    tgt = standoff_target_base(load_center_xy, base_world, cart_world, standoff_m=standoff_m)
    try:
        sol = cl.level_place_ik(robot, tgt, up_in_tool, list(preplace_shape), tool_link=tool_link,
                                limits=limits, gravity_comp=None)
    except ValueError as exc:
        return ReachResult(False, float('inf'), float('inf'), (), f'unreachable: {exc}')
    q = [float(sol[j]) for j in ARM_JOINTS]
    wl = link_world_transforms(robot, dict(zip(ARM_JOINTS, q)), root='ur_base_link')
    pos = np.asarray(wl[tool_link], float)[:3, 3]
    pos_err = float(np.linalg.norm(pos - tgt))
    uw = np.asarray(wl[tool_link], float)[:3, :3] @ up_in_tool
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(uw[2])))))
    ok = pos_err <= 1e-3 and tilt <= float(tilt_bound_deg)
    return ReachResult(ok, pos_err, tilt, tuple(round(v, 4) for v in q),
                       '' if ok else f'pos_err={pos_err*1000:.1f}mm tilt={tilt:.2f}deg')


# --------------------------------------------------------------------------------------- choose spot

# Cage interior planes in cart-x the centred carton footprint should sit between (post_2/post_3 inner
# face at +0.367; back wall outer bars near -0.374). Centring the footprint between them keeps the
# carton visually inside the cage (the captain's "nothing visually wrong" rule) with symmetric margin,
# and the honest sweep's min NON-support clearance is side-bar-limited (flat ~184 mm) across the range,
# so centring -- not pushing to a grid edge -- is the right tie-break.
CAGE_FRONT_X = 0.367    # post_2/post_3 inner -x face (cart frame)
CAGE_BACK_X = -0.374    # back wall inner +x face (cart frame)


def _centering_margin(lx, carton_x_half):
    """min(front, back) cart-x margin from the carton footprint edges to the cage walls at load-centre
    ``lx`` -- maximised (=> carton centred in the cage) at ``(CAGE_FRONT_X + CAGE_BACK_X)/2 ~ 0``."""
    front = CAGE_FRONT_X - (lx + carton_x_half)
    back = (lx - carton_x_half) - CAGE_BACK_X
    return min(front, back)


def choose_spot(*, bodies=None, geom=None, x_grid=None, robot=None, standoff_m=DEFAULT_STANDOFF_M):
    """Scan candidate load-centres (ly fixed 0 -- symmetric between post_2/post_3 and the side bars) and
    return the best. Objective = MAXIMISE the brief's clearance (the honest min NON-support gap, rounded
    to mm so the flat side-bar-limited plateau ties), tie-broken by CENTRING the carton footprint in the
    cage (:func:`_centering_margin`) so it is visually inside the cage with symmetric wall margin. When
    ``robot`` is given, every candidate is reach-checked from the common park pose and unreachable ones
    are dropped. Returns ``(best_xy, table)`` where ``table`` is a list of ``{xy, min_post,
    min_nonsupport, reach_ok, centering}`` dicts."""
    bodies = bodies if bodies is not None else pc.rollwagen_collider_boxes()
    geom = geom if geom is not None else _geom()
    if x_grid is None:
        x_grid = [round(-0.15 + 0.025 * i, 3) for i in range(13)]  # -0.15 .. 0.15
    # carton x half-extent at YAW_CART_REL (-90 deg => cart-x extent = 0.30 m).
    cxh = pc.CARTON_DIMS_M[1] / 2.0 if abs(abs(YAW_CART_REL) - math.pi / 2) < 1e-6 else pc.CARTON_DIMS_M[0] / 2.0
    park = RUN_EVIDENCE[0]
    table, best, best_key = [], None, None
    for lx in x_grid:
        xy = (lx, 0.0)
        ic = idealized_clearance(xy, bodies=bodies, geom=geom, standoff_m=standoff_m)
        reach_ok = True
        if robot is not None:
            reach_ok = reach_check(xy, park.park_base_world, park.cart_world, robot,
                                   standoff_m=standoff_m).ok
        centering = _centering_margin(lx, cxh)
        row = {'xy': xy, 'min_post': ic['min_post'][1], 'min_post_body': ic['min_post'][0],
               'min_nonsupport': ic['min_nonsupport'][1], 'reach_ok': reach_ok, 'centering': centering}
        table.append(row)
        key = (round(ic['min_nonsupport'][1], 3), round(centering, 4))
        if reach_ok and (best_key is None or key > best_key):
            best, best_key = row, key
    return (best['xy'] if best else None), table


# -------------------------------------------------------------------------------------- the verdict

def evaluate(load_center_xy, *, bodies=None, geom=None, robot=None, margin_m=DEFAULT_MARGIN_M,
             evidence=RUN_EVIDENCE, standoff_m=DEFAULT_STANDOFF_M):
    """Full CPU verdict for a candidate place load-centre. Returns a dict with:

      * ``idealized``  -- static clearance at the nominal (:func:`idealized_clearance`),
      * ``required_m`` -- ``margin_m + park_variance`` (the brief's 0.04 m + measured spread),
      * ``park_variance_m`` -- the measured cart-y execution spread (~0.5 m),
      * ``executed``   -- the clearance under the WORST measured drift (strikes a post => negative),
      * ``reach``      -- :class:`ReachResult` from the common park pose (when ``robot`` given),
      * ``static_ok``  -- idealised min non-support >= margin_m,
      * ``absorbs_drift`` -- idealised min post >= required_m (can the spot swallow the measured spread),
      * ``verdict``    -- 'PASS' only when static_ok AND absorbs_drift AND reach ok; else 'FAIL',
      * ``reason``     -- a named explanation (the base shove, with numbers, when it fails).
    """
    bodies = bodies if bodies is not None else pc.rollwagen_collider_boxes()
    geom = geom if geom is not None else _geom()
    ic = idealized_clearance(load_center_xy, bodies=bodies, geom=geom, standoff_m=standoff_m)
    drift = measured_drift(OLD_TARGET_XY, evidence)
    pv = drift['max_abs'][1]
    required = float(margin_m) + pv
    # The tightest clearance when each run's MEASURED drift vector is applied to THIS candidate.
    exc_ns, exc_post, exc_per_run = worst_executed_clearance(
        load_center_xy, bodies=bodies, geom=geom, evidence=evidence, standoff_m=standoff_m)
    exc = {'worst_min_nonsupport': exc_ns, 'worst_min_post': exc_post, 'per_run': exc_per_run}
    reach = None
    if robot is not None:
        park = evidence[0]
        reach = reach_check(load_center_xy, park.park_base_world, park.cart_world, robot,
                            standoff_m=standoff_m)
    static_ok = ic['min_nonsupport'][1] >= float(margin_m)
    absorbs = ic['min_post'][1] >= required
    reach_ok = reach is None or reach.ok
    verdict = 'PASS' if (static_ok and absorbs and reach_ok) else 'FAIL'
    if not static_ok:
        reason = ('static clearance below the %.0f mm floor: %s=%.1f mm'
                  % (margin_m * 1000.0, ic['min_nonsupport'][0], ic['min_nonsupport'][1] * 1000.0))
    elif not absorbs:
        reason = ('idealised min post clearance %.1f mm (%s) < required %.0f mm (= %.0f mm floor + '
                  '%.0f mm MEASURED cart-y execution spread). No shelf spot absorbs it: the base SHOVES '
                  '%.2f m west during the loaded place (coldwin68/69, parked=True) so the carton lands '
                  'base-relative ~%.2f m off in cart-y onto post_2. Blocking fix = hold the base through '
                  'the loaded set-down / re-resolve the place target at descent time (collect place-'
                  'descent control, out of this lane). The nominal is necessary, not sufficient.'
                  % (ic['min_post'][1] * 1000.0, ic['min_post'][0], required * 1000.0,
                     margin_m * 1000.0, pv * 1000.0, drift['base_shove_m'], pv))
    elif not reach_ok:
        reason = 'unreachable/level-fail from the park pose: %s' % (reach.reason if reach else '')
    else:
        reason = ('static %s=%.1f mm, min post %.1f mm >= required %.0f mm, reach ok'
                  % (ic['min_nonsupport'][0], ic['min_nonsupport'][1] * 1000.0,
                     ic['min_post'][1] * 1000.0, required * 1000.0))
    return {'load_center_xy': tuple(float(v) for v in load_center_xy),
            'idealized': ic, 'required_m': required, 'park_variance_m': pv,
            'executed': exc, 'reach': reach, 'static_ok': static_ok, 'absorbs_drift': absorbs,
            'reach_ok': reach_ok, 'verdict': verdict, 'reason': reason, 'base_shove_m': drift['base_shove_m']}


# ----------------------------------------------------------------------------------------------- CLI

def _load_robot(urdf_path):
    from twin.urdf_probe import parse_urdf
    return parse_urdf(str(urdf_path))


def main(argv=None):
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description='CPU geometry + reach check for the collect place-on-cart '
                                             'between-posts target (sim-place-target-between-posts-v1).')
    ap.add_argument('--urdf', default=str(Path.home() /
                    'datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'),
                    help='mobile URDF for the reach check (omit/--no-reach to skip)')
    ap.add_argument('--no-reach', action='store_true', help='skip the reach check (clearance only)')
    ap.add_argument('--load-center', nargs=2, type=float, metavar=('X', 'Y'),
                    help='evaluate this load-centre instead of the shipped + old pair')
    args = ap.parse_args(argv)

    bodies = pc.rollwagen_collider_boxes()
    geom = _geom()
    robot = None
    if not args.no_reach:
        try:
            robot = _load_robot(args.urdf)
        except Exception as exc:  # noqa: BLE001
            print(f'{PLACE_TARGET}: reach check skipped (URDF load failed: {exc})')

    drift = measured_drift(OLD_TARGET_XY)
    print(f'{PLACE_TARGET} measured execution drift vs old target {OLD_TARGET_XY}:')
    for name, (dx, dy) in drift['per_run'].items():
        ev = next(e for e in RUN_EVIDENCE if e.name == name)
        print('  %-10s carton_in_cart=%s drift=(%.3f, %.3f) cleared=%s'
              % (name, ev.carton_in_cart, dx, dy, ev.cleared))
    print('  base shove (max from common park pose) = %.3f m; cart-y spread to absorb = %.3f m'
          % (drift['base_shove_m'], drift['max_abs'][1]))

    best, table = choose_spot(bodies=bodies, geom=geom, robot=robot)
    print(f'\n{PLACE_TARGET} candidate scan (ly=0), min corner-post clearance:')
    for row in table:
        mark = ' <= best' if row['xy'] == best else ''
        print('  lx=%+.3f  min_post=%6.1f mm (%s)  min_nonsupport=%6.1f mm  reach_ok=%s%s'
              % (row['xy'][0], row['min_post'] * 1000.0, row['min_post_body'],
                 row['min_nonsupport'] * 1000.0, row['reach_ok'], mark))
    print(f'  => max-clearance between-posts spot: {best}')

    targets = ([tuple(args.load_center)] if args.load_center else [OLD_TARGET_XY, SHIPPED_TARGET_XY])
    rc = 0
    for xy in targets:
        v = evaluate(xy, bodies=bodies, geom=geom, robot=robot)
        ic = v['idealized']
        print(f'\n{PLACE_TARGET} evaluate load_center={xy}:')
        print('  idealised: min_nonsupport=%.1f mm (%s), posts=%s'
              % (ic['min_nonsupport'][1] * 1000.0, ic['min_nonsupport'][0],
                 {k: round(x * 1000.0, 1) for k, x in ic['posts'].items()}))
        print('  required = %.0f mm (0.04 + park variance %.0f mm); absorbs_drift=%s; static_ok=%s'
              % (v['required_m'] * 1000.0, v['park_variance_m'] * 1000.0, v['absorbs_drift'], v['static_ok']))
        if v['reach'] is not None:
            print('  reach: ok=%s pos_err=%.1f mm tilt=%.3f deg'
                  % (v['reach'].ok, v['reach'].pos_err_m * 1000.0, v['reach'].tilt_deg))
        print('  VERDICT: %s -- %s' % (v['verdict'], v['reason']))
        if v['verdict'] != 'PASS':
            rc = 1
    return rc


if __name__ == '__main__':
    import sys
    sys.exit(main())
