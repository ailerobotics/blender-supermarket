#!/usr/bin/env python3
"""Verdict for the wrist eye-in-hand camera depth proof (camera lane; CPU).

Division of labour: integration renders the wrist camera depth natively, under the GPU lock
(e.g. by extending ``run_restock_render_smoke.py``: boot the guarded SimulationApp, place the robot
at the collect staging with the arm at the collect joint configuration, author + select the wrist
camera via ``native_cameras.initialize_collision_camera`` / ``mobile_assets.author_wrist_camera``,
render, and save the wrist ``distance_to_image_plane`` to an ``.npz``/``.npy``). This script is the
CPU verdict on that render: it projects the grasp-region points (in ``ur_tool0``) into the wrist
image and measures depth coverage with :func:`twin.restock_camera.wrist_camera.grasp_region_depth_coverage`.
High coverage at the grasp and pre-grasp points proves the side-grasp region is actually in the wrist
depth at the collect pose — what the fixed forward body camera could never show.

It does not boot Isaac and never fabricates a depth frame: a missing or empty render fails the proof.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from twin.restock_camera.wrist_camera import (
    GRASP_POINT_TOOL0, grasp_region_depth_coverage, wrist_camera_intrinsics)


def _load_depth(path):
    path = Path(path)
    if path.suffix == '.npz':
        data = np.load(path)
        key = 'depth' if 'depth' in data else data.files[0]
        return np.asarray(data[key], float)
    return np.asarray(np.load(path), float)


def proof(depth, *, window_px=16, min_coverage=0.6):
    """Proof record over the grasp region: coverage per point and an overall pass/fail."""
    tcp = np.asarray(GRASP_POINT_TOOL0, float)
    points = {'grasp': tcp, 'approach': tcp + [0, 0, 0.12], 'lift': tcp + [0, 0, 0.03]}
    coverage = grasp_region_depth_coverage(depth, list(points.values()),
                                           intrinsics=wrist_camera_intrinsics(), window_px=window_px)
    per_point = {name: coverage[i] for i, name in enumerate(points)}
    # The proof requires the grasp point AND the pre-grasp approach to be in the depth.
    passed = per_point['grasp'] >= min_coverage and per_point['approach'] >= min_coverage
    return dict(status='passed' if passed else 'failed', min_coverage=min_coverage, window_px=window_px,
                coverage=per_point, depth_shape=list(np.asarray(depth).shape),
                finite_depth_pixels=int(np.count_nonzero(np.isfinite(depth) & (depth > 0))),
                evidence='grasp-region depth coverage of a NATIVE wrist render; camera-lane verdict only')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--depth', required=True, type=Path, help='native wrist depth .npz (key "depth") or .npy')
    parser.add_argument('--output', type=Path, help='proof JSON path (default: stdout)')
    parser.add_argument('--window-px', type=int, default=16)
    parser.add_argument('--min-coverage', type=float, default=0.6)
    args = parser.parse_args()
    record = proof(_load_depth(args.depth), window_px=args.window_px, min_coverage=args.min_coverage)
    text = json.dumps(record, indent=2)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
