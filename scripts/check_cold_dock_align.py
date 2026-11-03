#!/usr/bin/env python3
"""Offline proof (CPU-only) for sim-cold-dock-align-cart-only-v1: with the cold-path
``couple_dock_align: true`` re-posing ONLY the cart (robot kept at its collect start pose), the cold
collect drive routes on the REAL mission nav grid with the south-facing cart in place, and the PR 293
dock-frame couple still reaches the cart.

Background (PR 297 regression, collectlevel2): the cold ``couple_dock_align`` moved BOTH the robot and
the cart to the configured couple dock (south, side-bay layout). That stranded the robot at the dock
SEAT, so its collect drive had to go BACK to the collect station through the backstock Support corridor
and was blocked ~1 m short -- no pick happened. The fix (:func:`twin.restock.shortcut.dock_align_scenario`)
re-poses only the cart; the robot keeps its collect start pose and runs the normal collect drive, then
drives to the dock to couple. The start_at=couple shortcut still seats both bodies (unchanged).

This drives the REAL inputs the mission uses -- the environment manifest + mission scenario (the full
~800-obstacle static map, the same grid scripts/check_tow_lanes.py / check_cart_side_bay.py use), the
REAL twin.mobile envelopes, the REAL ``dock_align_scenario`` / ``resolve_couple_dock`` and the REAL
``fine_dock.standoff_from_cart`` / ``target_base`` geometry -- it is not a fixture of the decision.

It proves, on CPU, with the cart dock-aligned SOUTH and present as an obstacle box:
  (1) CART-ONLY: dock_align_scenario leaves the robot at its scenario collect start pose and moves only
      the cart (+ its carton) to the dock heading.
  (2) COLLECT DRIVE ROUTES: start -> collect station -> the dock-frame couple standoff all plan on the
      grid (free-base transport radius) with the south cart in place.
  (3) CART DOES NOT BLOCK the Support corridor: the free base still routes past the cart to home / the
      shelf-staging goal, and the collect station start cell is free.
  (4) PR 293 DOCK-FRAME COUPLE STILL FINDS THE CART: the fine-dock standoff (standoff_from_cart on the
      pinned cart) is reachable by the free base, and the straight-reverse insert from that standoff onto
      the cart drawbar clears the static map.
  (5) REGRESSION CONTRAST: the PR 297 both-bodies spawn seats the robot at the dock seat, which lies
      INSIDE the south cart's transport inflation -- a degenerate free-base start (its collect drive back
      to the station cannot plan cleanly), reproducing why collectlevel2 failed.

Writes a before/after top-down plot to --out-dir. Exit 0 iff all claims pass (prints COLD_DOCK_ALIGN
PASS/FAIL)."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*
from twin.navigation.scenario import environment_grid
from twin.navigation.core import GridMap
from twin.navigation.articulated import polygon_hits_box
from twin.mobile import MobileSpec, Pose2
from twin.restock.shortcut import dock_align_scenario, resolve_couple_dock
from twin.restock.fine_dock import standoff_from_cart, target_base


def _cart_box_at(spec, centre, yaw, pad=0.02):
    """Cart AABB (world) at a cart body pose (origin + heading). Matches check_cart_side_bay."""
    cp = Pose2(centre[0], centre[1], yaw)
    corners = [cp.point(x, y) for x in (-spec.cart_length / 2, spec.cart_hitch_x)
               for y in (-spec.cart_width / 2, spec.cart_width / 2)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _base_rect(spec, pose, margin=0.0):
    p = Pose2(pose[0], pose[1], pose[2])
    hx, hy = spec.length / 2 + margin, spec.width / 2 + margin
    return [p.point(hx, hy), p.point(hx, -hy), p.point(-hx, -hy), p.point(-hx, hy)]


def _record(report, name, passed, **kw):
    report['ok'] = report['ok'] and passed
    report['checks'].append(dict(name=name, passed=passed, **kw))
    print(('PASS ' if passed else 'FAIL ') + name + ' ' + json.dumps(kw), flush=True)


def _plot(path, spec, scenario_cart, dock_cart, robot_start, dock_seat, standoff, grid, cart_box):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def cart_patch(ax, centre, yaw, color, label):
        b = _cart_box_at(spec, centre, yaw)
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                               edgecolor=color, lw=1.8, label=label))

    def obstacles(ax):
        for o in grid.obstacles:
            ax.add_patch(Rectangle((o[0], o[1]), o[2] - o[0], o[3] - o[1],
                                   facecolor='0.82', edgecolor='none'))

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, title in ((a0, 'BEFORE (PR 297): robot + cart both at dock'),
                      (a1, 'AFTER (cart-only): robot at collect start')):
        obstacles(ax)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-7.0, -2.5)
        ax.set_ylim(13.0, 18.0)
        ax.set_xlabel('x (m)')
    a0.set_ylabel('y (m)')

    # BEFORE: robot AND cart at dock (the regression)
    cart_patch(a0, dock_cart[:2], dock_cart[2], 'tab:red', 'cart @ dock (south)')
    a0.plot(*dock_seat[:2], 'o', color='tab:blue', ms=9, label='robot @ dock seat')
    a0.plot(*robot_start[:2], 'x', color='0.5', ms=9, label='collect start (unused)')
    a0.annotate('stranded: must drive\nback to station through\nthe Support corridor',
                xy=dock_seat[:2], xytext=(-6.6, 13.5), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='tab:red'))

    # AFTER: cart at dock, robot at collect start
    cart_patch(a1, dock_cart[:2], dock_cart[2], 'tab:red', 'cart @ dock (south)')
    a1.plot(*robot_start[:2], 'o', color='tab:blue', ms=9, label='robot @ collect start')
    a1.plot(*standoff[:2], 's', color='tab:green', ms=8, label='dock-frame standoff')
    a1.plot(*dock_seat[:2], '+', color='tab:green', ms=10, label='couple seat (at couple time)')
    a1.annotate('collect drive: start -> station -> standoff -> couple',
                xy=standoff[:2], xytext=(-6.9, 17.4), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='tab:green'))

    for ax in (a0, a1):
        ax.legend(loc='lower right', fontsize=7)
    fig.suptitle('cold couple_dock_align: cart-only re-pose (sim-cold-dock-align-cart-only-v1)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--environment', type=Path, required=True)
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True,
                    help='mission runtime config with couple_dock_align + the outbound lane')
    ap.add_argument('--out-dir', type=Path,
                    default=Path('/home/manishsaini/firstmate/data/sim-cold-dock-align-cart-only-v1'))
    args = ap.parse_args()

    manifest = json.loads(args.environment.read_text())
    scenario = json.loads(args.scenario.read_text())
    cfg = json.loads(args.config.read_text())
    cfg = dict(cfg)
    cfg['couple_dock_align'] = True
    cfg.pop('start_at', None)   # cold path only

    spec = MobileSpec()
    grid = environment_grid(manifest, scenario, None)
    R = spec.transport_radius
    report = {'ok': True, 'checks': [], 'static_obstacles': len(grid.obstacles)}
    print(f'grid: {len(grid.obstacles)} static obstacles, bounds {grid.bounds}, transport_radius {R}',
          flush=True)

    scenario_cart = tuple(float(v) for v in scenario['initial_staging']['cart_body_pose'])
    robot_start = tuple(float(v) for v in scenario['initial_staging']['robot_base_pose'])

    # (1) CART-ONLY
    variant, record = dock_align_scenario(scenario, cfg, hitch_x=spec.hitch_x, cart_hitch_x=spec.cart_hitch_x)
    if record is None:
        _record(report, 'dock_align_active', False, note='no dock resolved / no-op; config lacks the south lane')
        print('COLD_DOCK_ALIGN FAIL', flush=True)
        sys.exit(1)
    v_robot = tuple(float(v) for v in variant['initial_staging']['robot_base_pose'])
    v_cart = tuple(float(v) for v in variant['initial_staging']['cart_body_pose'])
    robot_unchanged = (math.dist(v_robot[:2], robot_start[:2]) <= 1e-6
                       and abs(v_robot[2] - robot_start[2]) <= 1e-6)
    cart_moved = math.dist(v_cart[:2], scenario_cart[:2]) > 1e-3 or abs(v_cart[2] - scenario_cart[2]) > 1e-3
    _record(report, 'cart_only_repose', robot_unchanged and cart_moved and record.get('cart_only') is True
            and record.get('robot_repositioned') is False,
            robot_start=[round(v, 3) for v in robot_start], robot_after=[round(v, 3) for v in v_robot],
            cart_after=[round(v, 3) for v in v_cart])

    res = resolve_couple_dock(cfg, hitch_x=spec.hitch_x, cart_hitch_x=spec.cart_hitch_x)
    dock_cart = res['cart_dock']
    dock_seat = res['robot_seat']
    cart = Pose2(dock_cart[0], dock_cart[1], dock_cart[2])
    cart_box = _cart_box_at(spec, dock_cart[:2], dock_cart[2])
    g_cart = GridMap(grid.bounds, tuple(list(grid.obstacles) + [tuple(cart_box)]), grid.resolution)

    cr = cfg.get('couple_cart_relative') or {}
    standoff_off = float(cr.get('standoff_offset_m', 0.40))
    seat = target_base(cart, spec)
    standoff = standoff_from_cart(cart, standoff_off, spec)

    pickup = tuple(cfg['mission_poses']['pickup'][:2])

    def routes(g, a, b):
        return bool(g.plan(tuple(a[:2]), tuple(b[:2]), R))

    # (2) COLLECT DRIVE ROUTES (with the south cart present)
    legA = routes(g_cart, robot_start, pickup)
    legB = routes(g_cart, pickup, standoff)
    _record(report, 'collect_drive_start_to_station', legA,
            start=[round(v, 3) for v in robot_start], station=list(pickup))
    _record(report, 'collect_drive_station_to_couple_standoff', legB,
            station=list(pickup), standoff=[round(v, 3) for v in standoff])

    # (3) CART DOES NOT BLOCK the Support corridor / collect start cell
    home = cfg['mission_poses'].get('home', [-3.9, 14.0, math.pi])
    shelf = cfg['mission_poses'].get('shelf_staging', [-3.875, 8.0, -math.pi / 2])
    start_cell_free = bool(g_cart.plan(tuple(robot_start[:2]), tuple(robot_start[:2]), R)) or routes(g_cart, robot_start, pickup)
    corridor_home = routes(g_cart, pickup, home)
    corridor_shelf = routes(g_cart, pickup, shelf)
    _record(report, 'cart_clear_of_support_corridor', start_cell_free and corridor_home and corridor_shelf,
            start_cell_free=start_cell_free, pickup_to_home=corridor_home, pickup_to_shelf=corridor_shelf)

    # (4) PR 293 DOCK-FRAME COUPLE STILL FINDS THE CART
    caw = dock_cart[2]
    reverse_clear = True
    hit = None
    for i in range(13):
        d = standoff_off * (1 - i / 12.0)      # from the standoff (behind) up to the seat
        px = seat[0] + d * math.cos(caw)
        py = seat[1] + d * math.sin(caw)
        rect = _base_rect(spec, (px, py, caw), margin=0.05)
        if any(polygon_hits_box(rect, ob) for ob in grid.obstacles):
            reverse_clear = False
            hit = [round(px, 2), round(py, 2)]
            break
    standoff_reachable = routes(g_cart, pickup, standoff)
    _record(report, 'dock_frame_couple_finds_cart', standoff_reachable and reverse_clear,
            standoff=[round(v, 3) for v in standoff], seat=[round(v, 3) for v in seat],
            straight_reverse_clears=reverse_clear, reverse_hit=hit, standoff_offset_m=standoff_off)

    # (5) REGRESSION CONTRAST: the PR 297 robot-at-dock-seat start is degenerate (inside the cart inflation)
    both_start_degenerate = not routes(g_cart, dock_seat, pickup)
    gap_seat_to_cart = cart_box[1] - dock_seat[1]    # cart south face minus seat y (both on the dock axis)
    _record(report, 'pr297_both_bodies_strands_the_robot', both_start_degenerate,
            dock_seat=[round(v, 3) for v in dock_seat],
            seat_inside_cart_inflation=bool(gap_seat_to_cart < R),
            gap_seat_to_cart_m=round(gap_seat_to_cart, 3), transport_radius_m=round(R, 3),
            note='robot spawned here (PR 297) cannot plan a clean collect drive back to the station')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.out_dir / 'cold-dock-align-before-after.png'
    try:
        _plot(plot_path, spec, scenario_cart, dock_cart, robot_start, dock_seat, standoff, grid, cart_box)
        report['plot'] = str(plot_path)
        print(f'wrote {plot_path}', flush=True)
    except Exception as exc:   # plotting is evidence, not a gate
        print(f'(plot skipped: {exc})', flush=True)

    (args.out_dir / 'cold-dock-align-report.json').write_text(json.dumps(report, indent=2))
    print('COLD_DOCK_ALIGN ' + ('PASS' if report['ok'] else 'FAIL'), flush=True)
    sys.exit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
