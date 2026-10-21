"""Top-down tow-path plot for a coupled mobile-restock run (fm456).

Overlays, in the store map frame:
  * the static obstacle map (light),
  * the AUTHORED tow lane centreline (base) for the chosen lane,
  * the ACTUAL driven base path from the run's ``nav-trace.jsonl`` (the chosen action),
  * the TRAIN footprint (base disc + towed-cart rectangle) at key ticks, the cart pose
    reconstructed by integrating the measured cart yaw-rate (``vel.cw``) from the aligned
    entry, and
  * the STALL point (where the base stops making net progress).

CPU/offline, no GPU/Isaac. Reads only the run's own artifacts. Saves a PNG beside them.

Run:  python3 scripts/twin/tools/plot_tow_path.py \\
        --run-dir <mission>/run-336b --lanes <mission-runtime-*.json> \\
        --environment <environment.json> --scenario <mission-scenario.json> \\
        --camera-config <camera-config.json> --lane-name outbound --output <run-dir>/tow-path.png
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
from twin.mobile import MobileSpec, Pose2, wrap


def _cart_pose(x, y, yaw, cart_yaw, spec):
    robot = Pose2(x, y, yaw)
    hx, hy = robot.point(spec.hitch_x)
    return Pose2(hx - spec.cart_hitch_x * math.cos(cart_yaw),
                 hy - spec.cart_hitch_x * math.sin(cart_yaw), cart_yaw)


def _rect(pose, length, width):
    c = [(length / 2, width / 2), (length / 2, -width / 2),
         (-length / 2, -width / 2), (-length / 2, width / 2)]
    return [pose.point(px, py) for px, py in c]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--lanes', required=True, type=Path)
    ap.add_argument('--environment', required=True, type=Path)
    ap.add_argument('--scenario', required=True, type=Path)
    ap.add_argument('--camera-config', required=True, type=Path)
    ap.add_argument('--lane-name', default='outbound')
    ap.add_argument('--action', default='outbound')
    ap.add_argument('--output', type=Path, default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle

    spec = MobileSpec()
    grid = environment_grid(json.loads(a.environment.read_text()),
                            json.loads(a.scenario.read_text()),
                            json.loads(a.camera_config.read_text()))
    lanes_cfg = json.loads(a.lanes.read_text())
    graph = TowLaneGraph.from_config(lanes_cfg['navigation']['tow_lanes'])
    graph.expand_all(ArticulatedMap(grid, spec))
    lane = next((ln for ln in graph.lanes if a.lane_name in ln.name), graph.lanes[0])

    rows = [json.loads(l) for l in (a.run_dir / 'nav-trace.jsonl').read_text().splitlines() if l.strip()]
    drive = [r for r in rows if r.get('action') == a.action]
    if not drive:
        print(f'no nav-trace rows for action={a.action!r}', flush=True)
        return 1

    # reconstruct cart_yaw by integrating the measured cart yaw-rate from the aligned entry
    cart_yaw = drive[0]['yaw_deg'] * math.pi / 180.0
    path = []
    prev_t = drive[0]['t']
    for r in drive:
        dt = max(0.0, r['t'] - prev_t); prev_t = r['t']
        cart_yaw = wrap(cart_yaw + r.get('vel', {}).get('cw', 0.0) * dt)
        path.append((r['x'], r['y'], r['yaw_deg'] * math.pi / 180.0, cart_yaw, r['tick']))

    # stall = last point beyond which the base makes < 2 cm net progress for the rest of the drive
    stall_i = len(path) - 1
    for i in range(len(path) - 1):
        if all(math.hypot(path[j][0] - path[i][0], path[j][1] - path[i][1]) < 0.02
               for j in range(i, len(path))):
            stall_i = i
            break

    fig, ax = plt.subplots(figsize=(9, 10))
    for (x0, y0, x1, y1) in grid.obstacles:
        ax.add_patch(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                             closed=True, facecolor='0.85', edgecolor='0.7', lw=0.3, zorder=0))
    ax.plot([s.x for s in lane.route.states], [s.y for s in lane.route.states],
            '-', color='tab:green', lw=1.5, label='authored lane (base centreline)', zorder=2)
    ax.plot([p[0] for p in path], [p[1] for p in path], '-', color='tab:blue', lw=1.2,
            label=f'driven base path ({a.action})', zorder=3)

    # key ticks: start, evenly spaced to the stall, and the stall itself
    keys = sorted(set([0, stall_i] + [round(stall_i * f) for f in (0.25, 0.5, 0.75)]))
    for ki in keys:
        x, y, yaw, cy, tick = path[ki]
        ax.add_patch(Circle((x, y), spec.length / 2, fill=False, edgecolor='tab:blue', lw=1.0, zorder=4))
        cart = _cart_pose(x, y, yaw, cy, spec)
        ax.add_patch(Polygon(_rect(cart, spec.cart_length, spec.cart_width), closed=True,
                             fill=False, edgecolor='tab:orange', lw=1.0, zorder=4))
        ax.plot([x, cart.x], [y, cart.y], '-', color='0.4', lw=0.8, zorder=4)   # drawbar
        ax.annotate(f't{tick}', (x, y), fontsize=7, color='tab:blue')
    sx, sy = path[stall_i][0], path[stall_i][1]
    ax.plot([sx], [sy], 'rx', ms=14, mew=3, label=f'stall @({sx:.2f},{sy:.2f}) tick {path[stall_i][4]}', zorder=5)

    ax.set_aspect('equal'); ax.set_xlim(-8, 0); ax.set_ylim(5, 17.5)
    ax.set_title(f'{a.run_dir.name}: coupled tow at the portal — {lane.name}\n'
                 f'base disc + towed-cart rect at key ticks; stall marked')
    ax.legend(loc='upper left', fontsize=8); ax.grid(True, ls=':', alpha=0.4)
    out = a.output or (a.run_dir / 'tow-path.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'wrote {out} | driven {len(path)} ticks, stall at tick {path[stall_i][4]} '
          f'({sx:.2f},{sy:.2f})', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
