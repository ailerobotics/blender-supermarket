"""Top-down tow-path plot for a coupled mobile-restock run (fm456).

Two modes:

* **Run mode** (``--run-dir`` given): overlays, in the store map frame, the static obstacle
  map, the AUTHORED tow lane centreline, the ACTUAL driven base path from the run's
  ``nav-trace.jsonl`` (the chosen action, cart pose reconstructed from the measured cart
  yaw-rate), and the STALL point.
* **Authored mode** (``--run-dir`` omitted; sim-portal-widen-v1): no run needed -- plots the
  static map, the AUTHORED base + towed-cart centrelines, the sales/backroom portal opening,
  and the train footprint (base disc + towed-cart rectangle) sampled through the portal, so the
  straight (or not) portal crossing is visible offline. Prints the crossing clearance.

CPU/offline, no GPU/Isaac. Reads only the given artifacts. Saves a PNG.

Run (authored, before/after of the portal crossing):
  python3 scripts/twin/tools/plot_tow_path.py --lanes <fragment.json> --lane-name outbound \\
    --environment <environment.json> --output <out>/outbound-after.png

Run (driven trace):
  python3 scripts/twin/tools/plot_tow_path.py --run-dir <mission>/run-336b --lanes <fragment> \\
    --environment <env.json> --scenario <scen.json> --camera-config <cam.json> \\
    --lane-name outbound --output <run-dir>/tow-path.png
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # scripts/ -> twin.*
from twin.navigation.scenario import environment_grid
from twin.navigation.lanes import TowLaneGraph
from twin.navigation.articulated import ArticulatedMap, TrainState
from twin.mobile import MobileSpec, Pose2, occupied_envelope, wrap

WALL_Y = 11.125


def _base_radius(spec):
    """The circumscribed transport disc twin.mobile.occupied_envelope builds for the robot."""
    return (max(spec.transport_radius,
                math.hypot(spec.length / 2, (spec.wheel_track + spec.wheel_width) / 2))
            + 0.05) / math.cos(math.pi / 16)


def _cart_pose(x, y, yaw, cart_yaw, spec):
    robot = Pose2(x, y, yaw)
    hx, hy = robot.point(spec.hitch_x)
    return Pose2(hx - spec.cart_hitch_x * math.cos(cart_yaw),
                 hy - spec.cart_hitch_x * math.sin(cart_yaw), cart_yaw)


def _rect(pose, length, width):
    c = [(length / 2, width / 2), (length / 2, -width / 2),
         (-length / 2, -width / 2), (-length / 2, width / 2)]
    return [pose.point(px, py) for px, py in c]


def _crossing_clearance(route, spec, left, right):
    """Min clearance of the (base+cart) occupied envelope to the opening edges AT the wall line."""
    for i in range(len(route.states) - 1):
        a, b = route.states[i], route.states[i + 1]
        if (a.y - WALL_Y) * (b.y - WALL_Y) <= 0 and a.y != b.y:
            env = occupied_envelope(*a.bodies(spec), spec, margin=0.05)
            xs = [p[0] for p in env]
            return dict(x=a.x, art_deg=math.degrees(abs(a.articulation)),
                        clear=min(min(xs) - left, right - max(xs)))
    return None


def _draw_obstacles(ax, grid):
    from matplotlib.patches import Polygon
    for (x0, y0, x1, y1) in grid.obstacles:
        ax.add_patch(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                             closed=True, facecolor='0.85', edgecolor='0.7', lw=0.3, zorder=0))


def _plot_authored(ax, lane, spec, left, right):
    from matplotlib.patches import Polygon, Circle
    br = _base_radius(spec)
    st = lane.route.states
    # base + cart centrelines
    ax.plot([s.x for s in st], [s.y for s in st], '-', color='tab:green', lw=1.6,
            label='authored base centreline', zorder=2)
    carts = [_cart_pose(s.x, s.y, s.yaw, s.cart_yaw, spec) for s in st]
    ax.plot([c.x for c in carts], [c.y for c in carts], '--', color='tab:orange', lw=1.1,
            label='towed-cart centreline', zorder=2)
    # the portal opening (gap in the wall at y=WALL_Y)
    ax.plot([left, left], [WALL_Y - 0.6, WALL_Y + 0.6], '-', color='tab:red', lw=2.5, zorder=5)
    ax.plot([right, right], [WALL_Y - 0.6, WALL_Y + 0.6], '-', color='tab:red', lw=2.5, zorder=5,
            label=f'portal opening ({right - left:.2f} m)')
    # footprints: states nearest to evenly spaced y through the portal band
    band = [s for s in st if WALL_Y - 2.2 <= s.y <= WALL_Y + 2.2]
    picks = []
    if band:
        ys = [s.y for s in band]
        for yt in [max(ys) - f * (max(ys) - min(ys)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]:
            picks.append(min(band, key=lambda s: abs(s.y - yt)))
    for s in picks:
        ax.add_patch(Circle((s.x, s.y), spec.length / 2, fill=False, edgecolor='tab:blue', lw=1.0, zorder=4))
        ax.add_patch(Circle((s.x, s.y), br, fill=False, edgecolor='tab:blue', lw=0.6, ls=':', zorder=4))
        cart = _cart_pose(s.x, s.y, s.yaw, s.cart_yaw, spec)
        ax.add_patch(Polygon(_rect(cart, spec.cart_length, spec.cart_width), closed=True,
                             fill=False, edgecolor='tab:orange', lw=1.0, zorder=4))
        ax.plot([s.x, cart.x], [s.y, cart.y], '-', color='0.4', lw=0.8, zorder=4)


def _plot_driven(ax, a, spec):
    from matplotlib.patches import Polygon, Circle
    rows = [json.loads(l) for l in (a.run_dir / 'nav-trace.jsonl').read_text().splitlines() if l.strip()]
    drive = [r for r in rows if r.get('action') == a.action]
    if not drive:
        print(f'no nav-trace rows for action={a.action!r}', flush=True)
        return None
    cart_yaw = drive[0]['yaw_deg'] * math.pi / 180.0
    path = []
    prev_t = drive[0]['t']
    for r in drive:
        dt = max(0.0, r['t'] - prev_t); prev_t = r['t']
        cart_yaw = wrap(cart_yaw + r.get('vel', {}).get('cw', 0.0) * dt)
        path.append((r['x'], r['y'], r['yaw_deg'] * math.pi / 180.0, cart_yaw, r['tick']))
    stall_i = len(path) - 1
    for i in range(len(path) - 1):
        if all(math.hypot(path[j][0] - path[i][0], path[j][1] - path[i][1]) < 0.02
               for j in range(i, len(path))):
            stall_i = i; break
    ax.plot([p[0] for p in path], [p[1] for p in path], '-', color='tab:blue', lw=1.2,
            label=f'driven base path ({a.action})', zorder=3)
    keys = sorted(set([0, stall_i] + [round(stall_i * f) for f in (0.25, 0.5, 0.75)]))
    for ki in keys:
        x, y, yaw, cy, tick = path[ki]
        ax.add_patch(Circle((x, y), spec.length / 2, fill=False, edgecolor='tab:blue', lw=1.0, zorder=4))
        cart = _cart_pose(x, y, yaw, cy, spec)
        ax.add_patch(Polygon(_rect(cart, spec.cart_length, spec.cart_width), closed=True,
                             fill=False, edgecolor='tab:orange', lw=1.0, zorder=4))
        ax.plot([x, cart.x], [y, cart.y], '-', color='0.4', lw=0.8, zorder=4)
    sx, sy = path[stall_i][0], path[stall_i][1]
    ax.plot([sx], [sy], 'rx', ms=14, mew=3,
            label=f'stall @({sx:.2f},{sy:.2f}) tick {path[stall_i][4]}', zorder=5)
    return path[stall_i]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', type=Path, default=None,
                    help='mission run dir with nav-trace.jsonl; omit for authored-only mode (no GPU run)')
    ap.add_argument('--lanes', required=True, type=Path)
    ap.add_argument('--environment', required=True, type=Path)
    ap.add_argument('--scenario', type=Path, default=None)
    ap.add_argument('--camera-config', type=Path, default=None)
    ap.add_argument('--lane-name', default='outbound')
    ap.add_argument('--action', default='outbound')
    ap.add_argument('--portal-width', type=float, default=None,
                    help='override the opening width drawn (default: environment manifest clear_width)')
    ap.add_argument('--title', default=None)
    ap.add_argument('--output', type=Path, default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    spec = MobileSpec()
    manifest = json.loads(a.environment.read_text())
    grid = environment_grid(manifest,
                            json.loads(a.scenario.read_text()) if a.scenario else None,
                            json.loads(a.camera_config.read_text()) if a.camera_config else None)
    graph = TowLaneGraph.from_config(json.loads(a.lanes.read_text())['navigation']['tow_lanes'])
    graph.expand_all(ArticulatedMap(grid, spec))
    lane = next((ln for ln in graph.lanes if a.lane_name in ln.name), graph.lanes[0])

    portal = manifest.get('portal', {})
    px = portal.get('centre', [-3.875])[0]
    width = a.portal_width if a.portal_width is not None else portal.get('clear_width', 2.6)
    left, right = px - width / 2, px + width / 2

    fig, ax = plt.subplots(figsize=(9, 10))
    _draw_obstacles(ax, grid)
    ax.plot([s.x for s in lane.route.states], [s.y for s in lane.route.states],
            '-', color='tab:green', lw=1.5, zorder=2) if a.run_dir else None

    crossing = _crossing_clearance(lane.route, spec, left, right)
    if a.run_dir:
        ax.plot([left, left], [WALL_Y - 0.6, WALL_Y + 0.6], '-', color='tab:red', lw=2.0, zorder=5)
        ax.plot([right, right], [WALL_Y - 0.6, WALL_Y + 0.6], '-', color='tab:red', lw=2.0, zorder=5,
                label=f'portal opening ({width:.2f} m)')
        _plot_driven(ax, a, spec)
    else:
        _plot_authored(ax, lane, spec, left, right)

    ax.set_aspect('equal'); ax.set_xlim(-8, 0); ax.set_ylim(5, 17.5)
    note = ''
    if crossing:
        note = (f'\ncrossing: x={crossing["x"]:.3f} (portal centre {px:.3f}), '
                f'base/cart articulation {crossing["art_deg"]:.1f} deg, '
                f'min opening clearance {crossing["clear"]:.3f} m')
    ax.set_title((a.title or f'{lane.name}: coupled tow at the portal') + note, fontsize=10)
    ax.legend(loc='upper left', fontsize=8); ax.grid(True, ls=':', alpha=0.4)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    out = a.output or ((a.run_dir / 'tow-path.png') if a.run_dir else Path('tow-path.png'))
    fig.savefig(out, dpi=130, bbox_inches='tight')
    if crossing:
        print(f'{lane.name}: crossing x={crossing["x"]:.3f} art={crossing["art_deg"]:.1f}deg '
              f'clear={crossing["clear"]:.3f}m -> {out}', flush=True)
    else:
        print(f'{lane.name}: no portal crossing found -> {out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
