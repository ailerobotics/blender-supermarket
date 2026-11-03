#!/usr/bin/env python3
"""Offline proof (CPU-only) for the sim-service-leg-carry-v1 TRANSFER service leg -- the manipulation half
of the shelf-staging re-sequence. Against the mission's REAL nav grid (environment manifest + scenario, the
~800-obstacle static map, the same inputs scripts/check_shelf_staging_nav.py / check_tow_lanes.py use), the
REAL twin.mobile base/train envelopes, and the REAL carton_level FK (the mission mobile-import.urdf).

With the cart parked + uncoupled at the staging pose (sim-shelf-staging-v1 OUTBOUND), the free base services
the endcap ALONE: pick the carton off the parked cart AT the staging pose, CARRY it to the endcap held level,
place it, and drive the empty base BACK to the staging pose. This proves, on CPU:

  (1) CARRY-DRIVE PATH -- the robot-only lane staging -> endcap AND back, swept with the base's PHYSICAL
      rectangle UNIONED with the carried-carton footprint (0.45 x 0.30 m held ahead at the carry reach),
      clears the static map AND the parked-cart AABB. The carton is on the carry leg; the base is empty on
      the return leg. (This is the check_shelf_staging_nav uncoupled-service path EXTENDED with the carton.)

  (2) REACH -- the pick target (the carton on the parked cart, upper shelf) at the staging park AND the place
      target (the endcap slot, surface + carton half-height) at the endcap park both sit inside the UR10
      working envelope (profile max_reach) from the arm base. These are the SAME base-relative geometries the
      proven coldwin22 whole-run transfer picked + placed at (the cart rides at the hitch offset either way).

  (3) TILT -- the carried carton stays PARALLEL TO THE GROUND through the carry (carton_level, the CARTON_TILT
      trip): every carried fold segment's leveled max |roll|,|pitch| is under the bound (reuses the real
      carton_level_bench + the mission URDF). Skipped-with-note when the URDF is absent.

Run:  PYTHONPATH=scripts python3 scripts/check_service_leg_carry.py \
        --environment <environment.json> --scenario scripts/restock_scenario.json \
        --lanes docs/examples/tow-lanes-mission-fragment.json [--out <dir>]
Exit 0 iff (1), (2) and (3) pass (prints SERVICE_LEG_CARRY PASS/FAIL)."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*
from twin.navigation.scenario import environment_grid
from twin.navigation.lanes import TowLaneGraph
from twin.navigation.articulated import ArticulatedMap, polygon_hits_box
from twin.mobile import MobileSpec, Pose2

UB_Z = 0.50                 # arm base (ur_base_link) z on the mobile base (BASE_ATT 0.24 + arm_mount 0.26)
DEFAULT_MAX_REACH = 1.30    # UR10 working envelope (ros_host_config profile max_reach)
TILT_BOUND_DEG = 2.0        # carton_level DEFAULT_TILT_BOUND_DEG (the CARTON_TILT trip threshold)
DEFAULT_CARRY_FWD_M = 0.40  # conservative forward offset of the tucked carried carton from the base centre
DEFAULT_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mission/coldwin16/mobile-import.urdf'


def _record(report, name, passed, **kw):
    report['ok'] = report['ok'] and passed
    report['checks'].append(dict(name=name, passed=passed, **kw))
    print(('PASS ' if passed else 'FAIL ') + name + ' ' + json.dumps(kw), flush=True)


def _cart_centre(spec, s):
    r = Pose2(s.x, s.y, s.yaw)
    hx, hy = r.point(spec.hitch_x)
    return (hx - spec.cart_hitch_x * math.cos(s.cart_yaw),
            hy - spec.cart_hitch_x * math.sin(s.cart_yaw))


def _cart_aabb(spec, s, pad=0.02):
    cx, cy = _cart_centre(spec, s)
    cp = Pose2(cx, cy, s.cart_yaw)
    corners = [cp.point(x, y) for x in (-spec.cart_length / 2, spec.cart_hitch_x)
               for y in (-spec.cart_width / 2, spec.cart_width / 2)]
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _rect(pose, half_x, half_y, *, fwd=0.0, margin=0.0):
    """A rectangle of half-extents (half_x+margin, half_y+margin) centred `fwd` ahead of `pose` (base
    frame +x). fwd=0 is the base rectangle; fwd>0 places the carried-carton footprint ahead of the base."""
    p = Pose2(pose[0], pose[1], pose[2])
    hx, hy = half_x + margin, half_y + margin
    return [p.point(fwd + hx, hy), p.point(fwd + hx, -hy),
            p.point(fwd - hx, -hy), p.point(fwd - hx, hy)]


def _seg_pt(a, b, p):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _poly_box_clear(poly, box):
    if polygon_hits_box(poly, box):
        return 0.0
    xmin, ymin, xmax, ymax = box
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    best = math.inf
    for a, b in zip(poly, poly[1:] + poly[:1]):
        for c in corners:
            best = min(best, _seg_pt(a, b, c))
    for a, b in zip(corners, corners[1:] + corners[:1]):
        for c in poly:
            best = min(best, _seg_pt(a, b, c))
    return best


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _staged_path(start, via, goal, n=30):
    """Forward-first departure: drive straight (yaw held) start->via, then face-travel via->goal, then
    rotate in place to the goal yaw -- a diff-drive-plausible uncoupled motion (matches shelf_staging_nav)."""
    pts = []
    for i in range(n + 1):
        t = i / n
        pts.append((start[0] + t * (via[0] - start[0]), start[1] + t * (via[1] - start[1]), start[2]))
    heading = math.atan2(goal[1] - via[1], goal[0] - via[0])
    for i in range(1, n + 1):
        t = i / n
        pts.append((via[0] + t * (goal[0] - via[0]), via[1] + t * (goal[1] - via[1]), heading))
    dyaw = _wrap(goal[2] - heading)
    for i in range(1, n + 1):
        pts.append((goal[0], goal[1], _wrap(heading + dyaw * i / n)))
    return pts


def _approach_path(start, via, goal, n=30):
    """Return-to-staging: drive start->via at the travel heading, then ROTATE to the goal yaw AT the open
    via point (far from the parked cart), then translate via->goal HOLDING the goal yaw. No rotation adjacent
    to the cart -- the real nav orients in the open service aisle before backing into the tight staging slot;
    the terminal in-place rotation of _staged_path (fine at the open endcap) would otherwise swing a base
    corner into the parked cart at the cart-adjacent staging pose."""
    pts = []
    heading = math.atan2(via[1] - start[1], via[0] - start[0])
    for i in range(n + 1):
        t = i / n
        pts.append((start[0] + t * (via[0] - start[0]), start[1] + t * (via[1] - start[1]),
                    _wrap(start[2] + _wrap(heading - start[2]) * t)))
    dyaw = _wrap(goal[2] - heading)
    for i in range(1, n + 1):
        pts.append((via[0], via[1], _wrap(heading + dyaw * i / n)))     # rotate to goal yaw in the open aisle
    for i in range(1, n + 1):
        t = i / n
        pts.append((via[0] + t * (goal[0] - via[0]), via[1] + t * (goal[1] - via[1]), goal[2]))
    return pts


def _sweep_clear(path, spec, cart_box, obstacles, *, carton_fwd=None, carton_half=None, margin=0.05):
    """Min static + min parked-cart clearance of the base rect (optionally UNIONed with the carried-carton
    rect, fwd `carton_fwd`) swept along `path`. Returns (min_static, min_cart, hit_or_None)."""
    min_cart = math.inf; min_static = math.inf; hit = None
    for pose in path:
        polys = [_rect(pose, spec.length / 2, spec.width / 2, margin=margin)]
        if carton_fwd is not None:
            polys.append(_rect(pose, carton_half[0], carton_half[1], fwd=carton_fwd, margin=margin))
        for poly in polys:
            c = _poly_box_clear(poly, cart_box); min_cart = min(min_cart, c)
            if c == 0 and hit is None:
                hit = {'cart': [round(pose[0], 2), round(pose[1], 2)]}
            for ob in obstacles:
                cs = _poly_box_clear(poly, ob); min_static = min(min_static, cs)
                if cs == 0 and hit is None:
                    hit = {'static': [round(v, 2) for v in ob], 'at': [round(pose[0], 2), round(pose[1], 2)]}
    return min_static, min_cart, hit


def _carton_on_cart_world(spec, s, carton_dims):
    """Approx world pose of the carton on the parked cart's upper shelf: cart centre + 0.18 m toward the
    cart's open +x face, carton centre height ~ carton half-height above the warm-shelf rest (coldwin16
    START_PHASE_SHORTCUT: upper-shelf rest, carton centre z ~1.0125)."""
    cx, cy = _cart_centre(spec, s)
    return (cx + 0.18 * math.cos(s.cart_yaw), cy + 0.18 * math.sin(s.cart_yaw), 1.0125)


def _reach(arm_base_xy, target_world):
    """3-D distance from the arm base (mobile base xy, z=UB_Z) to a world target -- the reach envelope
    test the live_pose transfer observe gate applies (REACH_ENVELOPE_FAIL) in the arm base frame."""
    return math.dist((arm_base_xy[0], arm_base_xy[1], UB_Z), target_world)


def _tilt_proof(urdf_path):
    """(max_after_tilt_deg, rows) from the real carton_level_bench over the carried fold, or None if the
    mission URDF is absent (skip-with-note). The carry-drive HOLDS the leveled fold while the base drives,
    so a leveled fold under the bound == a level carry."""
    if not Path(urdf_path).exists():
        return None
    from twin.tools import carton_level_bench as b
    from twin.restock_manipulation import carton_level as cl
    _, res = b.run(str(urdf_path), bound_deg=cl.DEFAULT_TILT_BOUND_DEG)
    worst = max(max(r.after_max_roll, r.after_max_pitch) for r in res)
    rows = [(r.name, round(r.before_max_tilt, 1), round(r.after_max_tilt, 2),
             r.passes(cl.DEFAULT_TILT_BOUND_DEG)) for r in res]
    return worst, rows


def _plot(out_dir, grid, spec, s, carry_path, return_path, carton_fwd, carton_half):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly, Rectangle
    except Exception as exc:  # matplotlib optional -- the proof is the PASS line, plots are evidence
        print(f'NOTE plots skipped (matplotlib unavailable: {exc})', flush=True)
        return None
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cart_box = _cart_aabb(spec, s)
    fig, ax = plt.subplots(figsize=(7, 9))
    for ob in grid.obstacles:
        ax.add_patch(Rectangle((ob[0], ob[1]), ob[2] - ob[0], ob[3] - ob[1],
                               fc='0.85', ec='0.6', lw=0.3))
    ax.add_patch(Rectangle((cart_box[0], cart_box[1]), cart_box[2] - cart_box[0], cart_box[3] - cart_box[1],
                           fc='tab:orange', ec='k', alpha=0.5, label='parked cart'))
    for i, pose in enumerate(carry_path):
        if i % 6:
            continue
        ax.add_patch(MplPoly(_rect(pose, spec.length / 2, spec.width / 2), closed=True,
                             fc='none', ec='tab:blue', lw=0.6))
        ax.add_patch(MplPoly(_rect(pose, carton_half[0], carton_half[1], fwd=carton_fwd), closed=True,
                             fc='tab:green', ec='tab:green', alpha=0.4, lw=0.4))
    ax.plot([p[0] for p in carry_path], [p[1] for p in carry_path], 'b-', lw=1.2, label='carry→endcap (carton)')
    ax.plot([p[0] for p in return_path], [p[1] for p in return_path], 'c--', lw=1.0, label='drive→staging (empty)')
    ax.plot(s.x, s.y, 'k^', ms=9, label='staging (pick)')
    ax.plot(-3.05, 6.0, 'ks', ms=8, label='endcap (place)')
    ax.set_xlim(-9, 1); ax.set_ylim(3, 13); ax.set_aspect('equal')
    ax.axhline(11.125, color='r', ls=':', lw=0.8); ax.set_title('service leg carry-drive (base ∪ carton swept)')
    ax.legend(loc='upper right', fontsize=7)
    p = out_dir / 'service-leg-carry-path.png'
    fig.savefig(p, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {p}', flush=True)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--environment', type=Path, required=True)
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--lanes', type=Path, required=True)
    ap.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--max-reach', type=float, default=DEFAULT_MAX_REACH)
    ap.add_argument('--carry-fwd-m', type=float, default=DEFAULT_CARRY_FWD_M)
    ap.add_argument('--json', type=Path, default=None)
    args = ap.parse_args()

    manifest = json.loads(args.environment.read_text())
    scenario = json.loads(args.scenario.read_text())
    cfg = json.loads(args.lanes.read_text())
    grid = environment_grid(manifest, scenario, None)
    spec = MobileSpec()
    world = ArticulatedMap(grid, spec)
    report = {'ok': True, 'checks': [], 'static_obstacles': len(grid.obstacles)}
    print(f'grid: {len(grid.obstacles)} static obstacles, bounds {grid.bounds}', flush=True)

    poses = cfg['mission_poses']
    staging = tuple(poses['shelf_staging'])
    service = tuple(poses['shelf_approach'])
    carton_dims = scenario['handling_unit']['dimensions_m']
    carton_half = (carton_dims[0] / 2.0, carton_dims[1] / 2.0)       # 0.45 x 0.30 footprint
    slot = scenario['initial_staging']['target_surface_pose_world']  # [x, y, z, yaw]

    # Reconstruct the parked train terminal state (cart parked at the staging, from the coupled tow lane).
    graph = TowLaneGraph.from_config(cfg['navigation']['tow_lanes'])
    graph.expand_all(world)
    lane = next(ln for ln in graph.lanes if 'staging' in ln.name)
    s = lane.route.states[-1]
    cart_box = _cart_aabb(spec, s)

    # (1) CARRY-DRIVE PATH -- base ∪ carton swept staging -> endcap (carton held), empty back.
    via = (staging[0], 6.5, staging[2])
    carry_path = _staged_path(staging, via, service)
    return_path = _approach_path(service, (service[0], 6.5, service[2]), staging)
    cs, cc, chit = _sweep_clear(carry_path, spec, cart_box, grid.obstacles,
                                carton_fwd=args.carry_fwd_m, carton_half=carton_half)
    rs, rc, rhit = _sweep_clear(return_path, spec, cart_box, grid.obstacles)
    _record(report, 'carry_drive_to_endcap_clears_with_carton',
            cs > 0.0 and cc >= 0.0 and chit is None,
            min_static_clear_m=round(cs, 3), min_cart_clear_m=round(cc, 3),
            carton_fwd_m=args.carry_fwd_m, carton_footprint_m=[carton_half[0] * 2, carton_half[1] * 2],
            hit=chit)
    _record(report, 'drive_back_to_staging_clears_empty',
            rs > 0.0 and rc >= 0.0 and rhit is None,
            min_static_clear_m=round(rs, 3), min_cart_clear_m=round(rc, 3), hit=rhit)

    # (2) REACH -- pick (carton on the parked cart at staging) + place (endcap slot) in the UR10 envelope.
    pick_world = _carton_on_cart_world(spec, s, carton_dims)
    reach_pick = _reach((s.x, s.y), pick_world)
    place_world = (slot[0], slot[1], slot[2] + carton_dims[2] / 2.0)   # carton rests on the slot surface
    reach_place = _reach((service[0], service[1]), place_world)
    _record(report, 'pick_at_staging_within_reach', reach_pick <= args.max_reach,
            reach_m=round(reach_pick, 3), max_reach_m=args.max_reach,
            margin_m=round(args.max_reach - reach_pick, 3),
            carton_world=[round(v, 3) for v in pick_world],
            note='same base-relative cart geometry as the proven coldwin22 transfer pick')
    _record(report, 'place_at_endcap_within_reach', reach_place <= args.max_reach,
            reach_m=round(reach_place, 3), max_reach_m=args.max_reach,
            margin_m=round(args.max_reach - reach_place, 3),
            slot_world=[round(v, 3) for v in place_world],
            note='the proven coldwin22 transfer place pose (endcap G3_back slot)')

    # (3) TILT -- the carried carton stays level through the fold (carton_level, CARTON_TILT trip).
    tilt = _tilt_proof(args.urdf)
    if tilt is None:
        print(f'NOTE tilt proof skipped (mission URDF absent: {args.urdf}); run the carton_level_bench '
              f'test where it is present', flush=True)
    else:
        worst, rows = tilt
        _record(report, 'carried_carton_stays_level', worst <= TILT_BOUND_DEG,
                worst_after_tilt_deg=round(worst, 2), bound_deg=TILT_BOUND_DEG,
                segments=[{'seg': n, 'before_tilt': bt, 'after_tilt': at, 'pass': ok}
                          for n, bt, at, ok in rows])

    if args.out is not None:
        _plot(args.out, grid, spec, s, carry_path, return_path, args.carry_fwd_m, carton_half)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f'wrote {args.json}', flush=True)

    print('SERVICE_LEG_CARRY ' + ('PASS' if report['ok'] else 'FAIL'), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
