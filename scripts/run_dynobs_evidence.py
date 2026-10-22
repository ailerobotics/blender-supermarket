#!/usr/bin/env python3
"""Dynamic-obstacle detection + avoidance evidence (CPU, no Blender/Isaac/GPU).

Runs deterministic closed-loop demonstrations of the mobile base, while towing the
cart, detecting a scripted moving obstacle through the navigation dynamic-obstacle
layer and reacting:

  * person  -- a shopper crosses the aisle: detect -> stop and wait -> resume ->
               arrive.
  * box     -- a box is dropped on the route and left: detect -> stop -> honest
               give-up once the avoidance patience (blocked_timeout) expires. A
               permanently blocked aisle never reports arrival.
  * cart    -- a cart rolled in off-centre: the coupled planner re-plans a route
               AROUND it (vs the straight shot with no obstacle).

Writes a per-tick trace, a run.json summary and top-down PNG frames per scenario,
and prints a self-asserting verdict (exit 1 on any failure). Artifacts go OUTSIDE
git (default ~/datasets/sim-restock-dynobs-v1); nothing here touches the live
Blender session or the GPU.

This is a quasi-static kinematic demonstration, not a PhysX, sensing or perception
run: detections are scenario ground truth and the robot integrates the gated
command through the trailer model the planner checks.
"""
import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin.mobile import MobileSpec, Pose2, occupied_envelope
from twin.navigation.core import GridMap
from twin.navigation.articulated import ArticulatedMap, ArticulatedPlanner, TrainState
from twin.navigation.dynamic import DynamicObstacleConfig, DynamicObstacleLayer, Detection, predicted_points
from twin.navigation.dynamic_scenario import DynamicScenario, crossing_person, dropped_box, rolled_cart
from twin.navigation.dynamic_evidence import simulate

CFG = DynamicObstacleConfig(confirm_frames=2, moving_speed=0.08,
                            prediction_horizon=1.2, prediction_step=0.3)


def _aisle():
    return GridMap((-3, -2, 12, 2), ((-3, -2, 12, -1.0), (-3, 1.0, 12, 2)))


def _render(path, grid, spec, state, detections, goal, config, title):
    """Top-down PNG: walls, aisle, train footprint, detections + predicted cells, goal."""
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = grid.bounds
    scale = 60.0  # px per metre
    pad = 10
    W = int((x1 - x0) * scale) + 2 * pad
    H = int((y1 - y0) * scale) + 2 * pad
    img = Image.new('RGB', (W, H), (245, 245, 245))
    d = ImageDraw.Draw(img)

    def px(x, y):  # world -> image (flip y)
        return (pad + (x - x0) * scale, pad + (y1 - y) * scale)

    for a, b, c, e in grid.obstacles:  # walls
        d.rectangle([px(a, e), px(c, b)], fill=(170, 170, 175))
    # predicted occupancy cells (light red) then raw detection disks (red outline).
    # A single-frame render confirms tracks immediately so the costmap cells show.
    render_cfg = dataclasses.replace(config, confirm_frames=1)
    layer = DynamicObstacleLayer(render_cfg)
    layer.update(detections, 0.0)
    for t in layer.tracks:
        for cx, cy in predicted_points(t, render_cfg):
            d.ellipse([px(cx - 0.03, cy + 0.03), px(cx + 0.03, cy - 0.03)], fill=(255, 190, 190))
    for det in detections:
        cx, cy = det.position
        d.ellipse([px(cx - det.radius, cy + det.radius), px(cx + det.radius, cy - det.radius)],
                  outline=(210, 30, 30), width=2)
    # train footprint
    robot = Pose2(state.x, state.y, state.yaw)
    hx, hy = robot.point(spec.hitch_x)
    import math
    cart = Pose2(hx - spec.cart_hitch_x * math.cos(state.cart_yaw),
                 hy - spec.cart_hitch_x * math.sin(state.cart_yaw), state.cart_yaw)
    hull = occupied_envelope(robot, cart, spec, margin=0.0)
    d.polygon([px(x, y) for x, y in hull], outline=(30, 90, 210), fill=(200, 215, 245))
    d.ellipse([px(state.x - 0.08, state.y + 0.08), px(state.x + 0.08, state.y - 0.08)], fill=(30, 90, 210))
    d.ellipse([px(goal[0] - 0.12, goal[1] + 0.12), px(goal[0] + 0.12, goal[1] - 0.12)],
              outline=(20, 150, 20), width=3)
    d.text((pad, pad), title, fill=(20, 20, 20))
    img.save(path)


def _write_trace(out, run):
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'trace.jsonl').open('w') as f:
        for k in run.ticks:
            f.write(json.dumps(dataclasses.asdict(k)) + '\n')
    (out / 'run.json').write_text(json.dumps(run.summary(), indent=2))


def _frames_for(out, grid, spec, scen, run, goal, n=10):
    frames = out / 'frames'
    frames.mkdir(parents=True, exist_ok=True)
    idx = [round(i * (len(run.ticks) - 1) / (n - 1)) for i in range(n)] if len(run.ticks) > 1 else [0]
    for fi, i in enumerate(dict.fromkeys(idx)):
        k = run.ticks[i]
        st = TrainState(k.x, k.y, k.yaw, k.cart_yaw)
        title = f't={k.t:.1f}s  {k.reason}  v={k.linear:.2f}  det={k.detections} trk={k.tracks}'
        _render(frames / f'frame_{fi:02d}.png', grid, spec, st, scen.detections_at(k.t), goal, CFG, title)


def run_person(out):
    grid, goal = _aisle(), (9.0, 0.0, 0.0)
    scen = DynamicScenario('person', 'shopper crosses the aisle',
                           (crossing_person(lane_x=5.0, enter=16.0, cross_duration=6.0, dwell=6.0),))
    run = simulate(scen, grid=grid, goal=goal, config=CFG, dt=0.1, horizon=160.0, blocked_timeout=50.0)
    _write_trace(out, run)
    _frames_for(out, grid, MobileSpec(), scen, run, goal, n=10)
    ok = run.stopped_for_obstacle and run.resumed_after_clear and run.arrived and run.min_clearance > 0
    return ok, run.summary()


def run_box(out):
    grid, goal = _aisle(), (9.0, 0.0, 0.0)
    scen = DynamicScenario('box', 'box dropped on the route and left',
                           (dropped_box(at_x=5.0, at_y=0.0, drop=12.0),))
    run = simulate(scen, grid=grid, goal=goal, config=CFG, dt=0.1, horizon=120.0, blocked_timeout=15.0)
    _write_trace(out, run)
    _frames_for(out, grid, MobileSpec(), scen, run, goal, n=8)
    ok = run.stopped_for_obstacle and not run.arrived and run.ticks[-1].status == 'failed'
    return ok, run.summary()


def run_cart(out):
    spec = MobileSpec()
    grid = GridMap((-4, -6, 14, 6))
    world = ArticulatedMap(grid, spec)
    planner = ArticulatedPlanner(world, timeout=10)
    start, goal = TrainState(2.0, 0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
    straight = planner.plan(start, goal)
    layer = DynamicObstacleLayer(DynamicObstacleConfig(confirm_frames=1))
    layer.update([Detection((5.0, 0.9), 0.5)], 0.0)
    detour = planner.plan(start, goal, layer.emitted_points())
    out.mkdir(parents=True, exist_ok=True)
    straight_y = max(abs(s.y) for s in straight.states) if straight.states else None
    detour_y = max(abs(s.y) for s in detour.states) if detour.states else None
    summary = dict(scenario='cart', straight_reason=straight.reason, detour_reason=detour.reason,
                   straight_states=len(straight.states), detour_states=len(detour.states),
                   straight_max_abs_y=round(straight_y, 3) if straight_y is not None else None,
                   detour_max_abs_y=round(detour_y, 3) if detour_y is not None else None,
                   note='coupled planner re-plans AROUND the cart; the closed-loop follower '
                        'reacts locally (stop/wait) to a cart met head-on')
    (out / 'run.json').write_text(json.dumps(summary, indent=2))
    if detour.states:
        mid = detour.states[len(detour.states) // 2]
        _render(out / 'detour.png', grid, spec, mid,
                [Detection((5.0, 0.9), 0.5)], goal, DynamicObstacleConfig(confirm_frames=1),
                'cart detour (coupled re-plan around)')
    ok = bool(straight.states and detour.states and straight_y is not None
              and detour_y is not None and straight_y < 0.05 and detour_y > 0.3)
    return ok, summary


SCENARIOS = {'person': run_person, 'box': run_box, 'cart': run_cart}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=os.path.expanduser('~/datasets/sim-restock-dynobs-v1'),
                    help='artifact directory (outside git)')
    ap.add_argument('--scenario', choices=(*SCENARIOS, 'all'), default='all')
    args = ap.parse_args(argv)
    base = Path(args.out)
    chosen = SCENARIOS if args.scenario == 'all' else {args.scenario: SCENARIOS[args.scenario]}
    all_ok = True
    for name, fn in chosen.items():
        ok, summary = fn(base / name)
        all_ok = all_ok and ok
        print(f'[{"PASS" if ok else "FAIL"}] {name}: {json.dumps(summary)}')
    print(f'artifacts under {base}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
