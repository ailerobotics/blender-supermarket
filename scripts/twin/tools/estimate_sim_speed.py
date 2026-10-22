#!/usr/bin/env python3
"""Replay a recorded mission run's per-tick wall cost and estimate its sim:wall under ``sim_speed``.

CPU only, read-only over the run directory (sim-video-speedup-v1). Every tick of a run is one 1/60 s
physics step and ``nav-trace.jsonl`` records, per tick, the wall clock and whether that tick rendered.
The tool groups the wall deltas per stage into PLAIN ticks (physics + sensors + mission logic) and
RENDERED ticks (+ a whole-scene RTX render), then re-prices them under the speed profile's levers:

* ``pause_idle_cameras`` -- a rendered tick in couple/outbound/park/return no longer renders the
  2560x1920 station decode cameras: its cost drops to the render FLOOR (the same tick with the decode
  products off). Evidence for the floor: warm2's transfer rendered ticks with the decode cameras
  suspended by the sensing gate (median 0.288 s, mean 0.364 s, vs 0.79 / 0.70 s on), historical
  collect rendered ticks with the decode render suspended (runs 240-320, median 0.26-0.27 s), and the
  chase-camera-only demo render (demo-004: 2040 frames in 612 s, 0.30 s/frame).
* ``pause_unread_decode`` -- the same for transfer / verify rendered ticks when the run's config makes
  those stages ground truth (``transfer_ground_truth`` / ``verify_ground_truth``).
* ``walkthrough_hz`` -- navigation-stage rendered ticks rescale to the profile's record stride.
* ``trace_probe`` off + ``bt_snapshot_hz`` -- the per-tick Python they remove, measured offline on this
  box (coverage probe 0.83 ms, BT snapshot serialise + atomic write ~0.3 ms per tick).

Nothing else is assumed faster: physics, lidar, host bridge and mission logic keep their recorded cost,
so the estimate is what the profile buys WITHOUT touching physics or any controller. The pacer's
real-time cap in host-coupled stages only binds above 1x, so it is ignored when the estimate is below.

    python3 scripts/twin/tools/estimate_sim_speed.py --run ~/datasets/.../mission/run-356 \\
        [--runtime-config ~/datasets/.../mission/mission-runtime-couple.json] [--floor 0.30] [--json out.json]

``--console <run>.console.log`` instead summarises the MEASURED ``TICK_PROFILE`` lines of a run made
with the profile on (per-action ratio and ms/tick sections).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.sim_speed import HOST_COUPLED_ACTIONS, from_config, unread_station_cameras  # noqa: E402

DT = 1 / 60
NAV_ACTIONS = ('couple', 'outbound', 'park', 'return')
STAGES = ('collect', 'couple', 'outbound', 'park', 'transfer', 'verify', 'return')
# Per-tick Python the profile removes (measured offline on the mission box, .scratch/time_tick_parts.py
# in the PR): the nav-trace coverage probe and the every-tick BT snapshot rewrite.
PROBE_MS = 0.83
BT_SNAPSHOT_MS = 0.30
# Rendered-tick cost with the station decode cameras off (see the module docstring for the evidence).
DEFAULT_FLOOR_S = 0.30


def tick_costs(rows):
    """{action: {'plain': [wall s], 'rendered': [wall s]}} from consecutive nav-trace rows (pure)."""
    out = {}
    prev = None
    for row in rows:
        if prev is not None and row.get('tick') == prev.get('tick', 0) + 1 and 'wall' in row and 'wall' in prev:
            bucket = out.setdefault(row.get('action'), {'plain': [], 'rendered': []})
            bucket['rendered' if row.get('rendered') else 'plain'].append(float(row['wall']) - float(prev['wall']))
        prev = row
    return out


def _stride_of(costs):
    """Observed ticks per rendered tick in a navigation stage (the run's walkthrough stride)."""
    plain, rendered = len(costs['plain']), len(costs['rendered'])
    return (plain + rendered) / rendered if rendered else None


def estimate(rows, config, *, floor_s=DEFAULT_FLOOR_S, station_ids=('source', 'cart', 'shelf'),
             location_cameras=None):
    """Per-stage {sim_s, wall_s, ratio, est_wall_s, est_ratio, ...} for the run rows under ``config``'s
    sim_speed profile (``{'sim_speed': {'profile': 'max'}}`` merged in when the config has none). Pure."""
    cfg = dict(config or {})
    cfg.setdefault('sim_speed', {'profile': 'max'})
    speed = from_config(cfg)
    costs = tick_costs(rows)
    plain_saving = ((PROBE_MS if not speed.trace_probe else 0.0)
                    + (BT_SNAPSHOT_MS if speed.bt_snapshot_hz else 0.0)) / 1e3
    stages = {}
    for action in [a for a in STAGES if a in costs] + sorted(set(costs) - set(STAGES), key=str):
        c = costs[action]
        n_plain, n_render = len(c['plain']), len(c['rendered'])
        ticks = n_plain + n_render
        wall = sum(c['plain']) + sum(c['rendered'])
        render_mean = statistics.mean(c['rendered']) if n_render else 0.0
        plain_mean = statistics.mean(c['plain']) if n_plain else 0.0
        paused = False
        if action in NAV_ACTIONS and speed.pause_idle_cameras:
            paused = True
        elif action in HOST_COUPLED_ACTIONS and speed.pause_unread_decode:
            paused = bool(unread_station_cameras(cfg, action, station_ids, location_cameras))
        est_render = min(render_mean, floor_s) if paused else render_mean
        est_n_render = n_render
        if action in NAV_ACTIONS and speed.walkthrough_hz and n_render:
            stride = max(_stride_of(c) or 1.0, round(1.0 / (speed.walkthrough_hz * DT)))
            est_n_render = round(ticks / stride)
        est_n_plain = ticks - est_n_render
        est_wall = est_n_plain * max(0.0, plain_mean - plain_saving) + est_n_render * est_render
        sim = ticks * DT
        stages[action] = dict(
            ticks=ticks, sim_s=round(sim, 1), wall_s=round(wall, 1), ratio=round(sim / wall, 3) if wall else None,
            plain_ms=round(1e3 * plain_mean, 1), rendered=n_render, rendered_ms=round(1e3 * render_mean, 1),
            decode_paused=paused, est_rendered=est_n_render, est_rendered_ms=round(1e3 * est_render, 1),
            est_wall_s=round(est_wall, 1), est_ratio=round(sim / est_wall, 3) if est_wall else None)
    sim = sum(s['ticks'] for s in stages.values()) * DT
    wall = sum(s['wall_s'] for s in stages.values())
    est = sum(s['est_wall_s'] for s in stages.values())
    total = dict(sim_s=round(sim, 1), wall_s=round(wall, 1), ratio=round(sim / wall, 3) if wall else None,
                 est_wall_s=round(est, 1), est_ratio=round(sim / est, 3) if est else None,
                 speedup=round(wall / est, 2) if est else None)
    return {'profile': speed.profile, 'floor_s': floor_s, 'stages': stages, 'total': total}


_PROFILE = re.compile(r'TICK_PROFILE action=(\S+) sim=([\d.]+) sim_s=([\d.]+) wall_s=([\d.]+) ratio=([\d.]+) '
                      r'ticks=(\d+) rendered=(\d+) ms_per_tick: (.*)$')


def measured(lines):
    """Per-action summary of the MEASURED TICK_PROFILE lines (pure): sim/wall totals and the
    tick-weighted mean ms/tick per section."""
    out = {}
    for line in lines:
        m = _PROFILE.search(line)
        if not m:
            continue
        action, _, sim_s, wall_s, _, ticks, rendered, parts = m.groups()
        acc = out.setdefault(action, {'sim_s': 0.0, 'wall_s': 0.0, 'ticks': 0, 'rendered': 0, 'ms': {}})
        acc['sim_s'] += float(sim_s)
        acc['wall_s'] += float(wall_s)
        acc['ticks'] += int(ticks)
        acc['rendered'] += int(rendered)
        for kv in parts.split():
            key, _, value = kv.partition('=')
            acc['ms'][key] = acc['ms'].get(key, 0.0) + float(value) * int(ticks)
    for acc in out.values():
        acc['ratio'] = round(acc['sim_s'] / acc['wall_s'], 3) if acc['wall_s'] else None
        acc['ms'] = {k: round(v / max(1, acc['ticks']), 2)
                     for k, v in sorted(acc['ms'].items(), key=lambda kv: -kv[1])}
        acc['sim_s'], acc['wall_s'] = round(acc['sim_s'], 1), round(acc['wall_s'], 1)
    return out


def _print_estimate(name, result):
    print(f'== {name}  profile={result["profile"]}  render floor={result["floor_s"]:.2f}s')
    print(f'{"stage":10s} {"sim_s":>7s} {"wall_s":>8s} {"ratio":>6s} {"plain_ms":>8s} {"rend":>5s} '
          f'{"rend_ms":>7s} {"est_rend":>8s} {"est_ms":>6s} {"est_wall":>8s} {"est_ratio":>9s}')
    for action, s in result['stages'].items():
        print(f'{str(action):10s} {s["sim_s"]:7.1f} {s["wall_s"]:8.1f} {s["ratio"] or 0:6.3f} {s["plain_ms"]:8.1f} '
              f'{s["rendered"]:5d} {s["rendered_ms"]:7.1f} {s["est_rendered"]:8d} {s["est_rendered_ms"]:6.1f} '
              f'{s["est_wall_s"]:8.1f} {s["est_ratio"] or 0:9.3f}')
    t = result['total']
    print(f'{"TOTAL":10s} {t["sim_s"]:7.1f} {t["wall_s"]:8.1f} {t["ratio"] or 0:6.3f} {"":>8s} {"":>5s} {"":>7s} '
          f'{"":>8s} {"":>6s} {t["est_wall_s"]:8.1f} {t["est_ratio"] or 0:9.3f}   speedup x{t["speedup"]}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, action='append', default=[], help='run directory with nav-trace.jsonl')
    ap.add_argument('--runtime-config', type=Path, default=None,
                    help='mission runtime config (its sim_speed block, or profile max when absent)')
    ap.add_argument('--floor', type=float, default=DEFAULT_FLOOR_S,
                    help='rendered-tick wall with the decode cameras off (s)')
    ap.add_argument('--console', type=Path, action='append', default=[],
                    help='summarise the measured TICK_PROFILE lines of a run console log instead')
    ap.add_argument('--json', type=Path, default=None)
    args = ap.parse_args(argv)
    report = {}
    config = json.loads(args.runtime_config.read_text()) if args.runtime_config else {}
    for run in args.run:
        rows = [json.loads(line) for line in (run / 'nav-trace.jsonl').read_text().splitlines() if line.strip()]
        result = estimate(rows, config, floor_s=args.floor)
        report[run.name] = result
        _print_estimate(run.name, result)
    for console in args.console:
        result = measured(console.read_text(errors='replace').splitlines())
        report[console.name] = result
        print(f'== {console.name} measured TICK_PROFILE')
        for action, acc in result.items():
            print(f'  {action:10s} sim {acc["sim_s"]:7.1f}s wall {acc["wall_s"]:8.1f}s ratio {acc["ratio"]} '
                  f'ticks {acc["ticks"]} rendered {acc["rendered"]} ms/tick {acc["ms"]}')
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report else 2


if __name__ == '__main__':
    sys.exit(main())
