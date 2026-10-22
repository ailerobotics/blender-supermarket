"""CPU evidence harness: drive the whole people -> detection -> reaction pipeline
on a virtual clock and emit a trace + a top-down SVG, with NO simulator or GPU.

This is the CPU-first proof the mechanism behaves: the scripted crowd crosses a
robot that drives the outbound route and then transfers at the shelf; the
ground-truth detection provider feeds the vendored safety machine; the reaction
mechanism produces the base speed / arm hold each tick. The output is LABELLED
ground-truth evidence (known poses projected into the cameras), NOT sensor pixels
-- the GPU run that renders the cameras and runs the real detector is a separate,
lowest-priority step (see the report's PENDING section).

``run`` is pure (returns rows); the CLI (:mod:`scripts.run_humans_safety_demo`)
owns file IO.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .config import HumansConfig
from .detection import GroundTruthSafety
from .integration import resolve_people
from .perception import SafetyParams
from .reaction import ReactionController

Pose2D = Tuple[float, float, float]


def _xy(pose: Sequence[float]) -> Tuple[float, float]:
    return (float(pose[0]), float(pose[1]))


def base_pose_at(poses: Dict[str, Sequence[float]], t: float, drive_s: float) -> Pose2D:
    """Robot base pose: drive home -> shelf over ``drive_s``, then hold (transfer)."""
    home, shelf = _xy(poses['home']), _xy(poses['shelf_approach'])
    f = 1.0 if drive_s <= 0 else max(0.0, min(1.0, t / drive_s))
    x = home[0] + (shelf[0] - home[0]) * f
    y = home[1] + (shelf[1] - home[1]) * f
    yaw = math.atan2(shelf[1] - home[1], shelf[0] - home[0])
    return (x, y, yaw)


def run(poses: Dict[str, Sequence[float]], humans: HumansConfig, *,
        duration_s: float = 20.0, dt: float = 0.1, drive_s: float = 8.0,
        params: Optional[SafetyParams] = None) -> List[dict]:
    """Step the pipeline over ``[0, duration_s]``; return one trace row per tick."""
    people = resolve_people(humans, poses)
    paths = {p.person_id: p.path() for p in people}
    radii = {p.person_id: p.radius_m for p in people}
    heights = {p.person_id: p.height_m for p in people}

    def people_positions(now: float):
        out = []
        for pid, path in paths.items():
            x, y, _ = path.pose_at(now)
            out.append((pid, (x, y, heights[pid] * 0.5), radii[pid]))
        return out

    base_ref = {'pose': base_pose_at(poses, 0.0, drive_s)}

    provider = GroundTruthSafety(
        config=humans.detection, people=people_positions,
        base_pose=lambda: base_ref['pose'],
        camera_poses=lambda: {f: base_ref['pose'] for f in humans.detection.cameras},
        params=params)
    reaction = ReactionController(humans.reaction)

    rows = []
    n = int(round(duration_s / dt))
    for i in range(n + 1):
        now = i * dt
        base_ref['pose'] = base_pose_at(poses, now, drive_s)
        state = provider.state(now)
        out = reaction.update(state, now)
        rows.append({
            't': round(now, 3),
            'base_xyyaw': [round(v, 4) for v in base_ref['pose']],
            'people': {pid: [round(xyz[0], 3), round(xyz[1], 3)]
                       for pid, xyz, _r in people_positions(now)},
            'zone': out.zone,
            'base_scale': out.base_scale,
            'arm_hold': out.arm_hold,
            'person_in_arm_envelope': out.person_in_arm_envelope,
            'nearest_distance_base_m': (None if not math.isfinite(state.nearest_distance_base)
                                        else round(state.nearest_distance_base, 3)),
            'reason': out.reason,
        })
    return rows


def summarize(rows: List[dict], params: Optional[SafetyParams] = None) -> dict:
    """Headline facts a reviewer checks: every regime seen, and resume happened."""
    zones = [r['zone'] for r in rows]
    scales = [r['base_scale'] for r in rows]
    holds = [r['arm_hold'] for r in rows]
    stop_ticks = [r for r in rows if r['zone'] == 'STOP']
    closest = min((r['nearest_distance_base_m'] for r in rows
                   if r['nearest_distance_base_m'] is not None), default=None)
    # Did a hold->release (arm resume) and a slow/stop->full (base resume) occur?
    arm_resumed = any(holds[i] and not holds[i + 1] for i in range(len(holds) - 1))
    base_resumed = any(scales[i] < 1.0 and scales[i + 1] == 1.0 for i in range(len(scales) - 1))
    return {
        'ticks': len(rows),
        'zones_observed': sorted(set(zones)),
        'slow_seen': 'SLOW' in zones,
        'stop_seen': 'STOP' in zones,
        'min_base_scale': min(scales) if scales else None,
        'max_base_scale': max(scales) if scales else None,
        'arm_paused': any(holds),
        'arm_resumed': arm_resumed,
        'base_resumed': base_resumed,
        'stop_ticks': len(stop_ticks),
        'closest_person_m': closest,
        'verdict': ('OK: slow, stop, arm-pause and resume all observed'
                    if ('SLOW' in zones and 'STOP' in zones and arm_resumed and base_resumed)
                    else 'INCOMPLETE: not every regime observed in this window'),
    }


# ---- top-down SVG (pure string building; no plotting dependency) -----------
_ZONE_FILL = {'CLEAR': '#2e7d32', 'SLOW': '#f9a825', 'STOP': '#c62828'}


def render_svg(poses: Dict[str, Sequence[float]], rows: List[dict], params: SafetyParams,
               snapshot_times: Sequence[float], *, panel_px: int = 240) -> str:
    """A row of top-down panels at the given times: rings, people, robot, state."""
    stop_ring = params.reach_radius + params.stop_radius      # protective ring (2.0 m)
    slow_ring = params.reach_radius + params.slow_radius      # outer ring
    # world extent to frame (robot path + people), with a margin
    xs = [poses['home'][0], poses['shelf_approach'][0]]
    ys = [poses['home'][1], poses['shelf_approach'][1]]
    for r in rows:
        for (x, y) in r['people'].values():
            xs.append(x); ys.append(y)
    minx, maxx, miny, maxy = min(xs) - slow_ring, max(xs) + slow_ring, \
        min(ys) - slow_ring, max(ys) + slow_ring
    span = max(maxx - minx, maxy - miny, 1e-3)
    scale = (panel_px - 20) / span

    def sx(x): return 10 + (x - minx) * scale
    def sy(y): return panel_px - (10 + (y - miny) * scale)      # flip y for screen

    def nearest_row(t):
        return min(rows, key=lambda r: abs(r['t'] - t))

    panels = []
    for k, t in enumerate(snapshot_times):
        r = nearest_row(t)
        bx, by, _ = r['base_xyyaw']
        fill = _ZONE_FILL.get(r['zone'], '#555')
        ox = k * (panel_px + 8)
        parts = [f'<g transform="translate({ox},0)">',
                 f'<rect width="{panel_px}" height="{panel_px}" fill="#111" stroke="#333"/>',
                 f'<circle cx="{sx(bx):.1f}" cy="{sy(by):.1f}" r="{slow_ring*scale:.1f}" '
                 f'fill="none" stroke="#f9a825" stroke-dasharray="4 3" opacity="0.6"/>',
                 f'<circle cx="{sx(bx):.1f}" cy="{sy(by):.1f}" r="{stop_ring*scale:.1f}" '
                 f'fill="none" stroke="#c62828" stroke-dasharray="4 3" opacity="0.8"/>',
                 f'<rect x="{sx(bx)-5:.1f}" y="{sy(by)-5:.1f}" width="10" height="10" '
                 f'fill="#42a5f5"/>']
        for (x, y) in r['people'].values():
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="#eee"/>')
        label = (f't={r["t"]:.1f}s  {r["zone"]}  base x{r["base_scale"]:.2f}  '
                 f'arm {"HOLD" if r["arm_hold"] else "run"}')
        parts.append(f'<rect x="0" y="{panel_px-20}" width="{panel_px}" height="20" fill="{fill}"/>')
        parts.append(f'<text x="6" y="{panel_px-6}" font-family="monospace" font-size="10" '
                     f'fill="#fff">{label}</text>')
        parts.append('</g>')
        panels.append(''.join(parts))
    width = len(snapshot_times) * (panel_px + 8)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{panel_px}" '
            f'viewBox="0 0 {width} {panel_px}">{"".join(panels)}</svg>')
