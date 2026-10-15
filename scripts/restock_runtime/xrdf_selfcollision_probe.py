#!/usr/bin/env python3
"""CPU: at the run-82 rejected c-space config, find which XRDF sphere pair self-collides.

cuMotion logs only "Invalid c-space position: self collision detected" without the pair. This
places the XRDF (host-cad/mobile.xrdf) collision spheres via pure-numpy URDF FK at the rejected
approach config and reports the non-ignored frame pairs whose spheres overlap (signed gap < 0),
so we can name the pair + penetration and decide the smallest fix (ignore vs sphere vs base frame).
No GPU, no ROS.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twin.urdf_probe import parse_urdf, link_world_transforms
from twin.restock_host.config import JOINTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf', required=True)
    p.add_argument('--xrdf', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--config', nargs=6, type=float,
                   default=[1.165, -0.988, 2.484, -4.633, -1.165, 3.14], help='6 arm joint values')
    a = p.parse_args()

    robot = parse_urdf(a.urdf)
    xrdf = yaml.safe_load(open(a.xrdf))
    spheres = xrdf['geometry'][xrdf['self_collision']['geometry']]['spheres']
    ignore = xrdf['self_collision'].get('ignore', {}) or {}
    # symmetric ignore set
    ign = set()
    for k, vs in ignore.items():
        for v in vs:
            ign.add(frozenset((k, v)))

    manifest = json.loads(Path(a.manifest).read_text())['transport_positions']
    cfg = {n: float(manifest.get(n, 0.0) or 0.0) for j in robot.joints for n in [j.name]}
    for j, v in zip(JOINTS, a.config):
        cfg[j] = float(v)
    world = link_world_transforms(robot, cfg)

    # world sphere centres per frame
    frames = {}
    for frame, slist in spheres.items():
        T = world.get(frame)
        if T is None or not isinstance(slist, list):
            continue
        pts = []
        for s in slist:
            c = np.array(s['center'] + [1.0])
            pts.append((T @ c)[:3])
            s_r = s['radius']
        frames[frame] = [(np.array(s['center']), s['radius'], (T @ np.array(s['center'] + [1.0]))[:3])
                         for s in slist]

    names = sorted(frames)
    overlaps = []
    pair_min = {}
    for i in range(len(names)):
        for k in range(i + 1, len(names)):
            fa, fb = names[i], names[k]
            if frozenset((fa, fb)) in ign:
                continue
            best = None
            for _, ra, pa in frames[fa]:
                for _, rb, pb in frames[fb]:
                    gap = float(np.linalg.norm(pa - pb) - (ra + rb))
                    if best is None or gap < best:
                        best = gap
            pair_min[(fa, fb)] = round(best, 4)
            if best < 0:
                overlaps.append((fa, fb, round(best, 4)))
    overlaps.sort(key=lambda x: x[2])
    out = {'config': a.config, 'overlapping_non_ignored_pairs': overlaps,
           'closest_10_pairs': sorted(pair_min.items(), key=lambda kv: kv[1])[:10]}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
