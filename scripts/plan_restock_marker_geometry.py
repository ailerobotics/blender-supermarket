#!/usr/bin/env python3
"""CPU proof that the geometry shelf fiducials decode at the shelf camera; never a render.

The native render never sampled the shelf markers' texture, so they are authored as
geometry (a white base quad plus one black quad per dark module). This renders each
geometry fiducial through the exact pinhole at the shelf camera pose and pixel size,
rectifies it by its projected corners (as a registration pipeline does) and decodes it.
A decode proves the geometric pattern carries its code at this pose/size; it is not a
renderer (no lighting, material, occlusion or noise), so it is necessary, not sufficient.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from twin.restock_camera.config import (camera_config, marker_module_grid,
                                        decode_marker_render)
from twin.restock_camera.geometry import Intrinsics, project, transform_points
from twin.restock_manipulation.isaac_sensors import write_carton_label


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scenario', 'mobile-manifest', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    scenario = json.loads(a.scenario.read_text())
    manifest = json.loads(a.mobile_manifest.read_text())
    config = camera_config(scenario, manifest)
    sources = [a.scenario, a.mobile_manifest, Path(__file__),
               Path(__file__).parent / 'twin/restock_camera/config.py',
               Path(__file__).parent / 'twin/restock_camera/geometry.py']
    hashes = {str(path.resolve()): sha(path) for path in sources}
    camera = config['cameras']['shelf']
    world = np.array(camera['parent_from_optical'])          # shelf camera parent is 'map'
    k = Intrinsics(**camera['intrinsics'])
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        detector = cv2.QRCodeDetector()
        for marker in config['supports']['shelf']['markers']:
            png = Path(tmp) / (marker['id'].replace(':', '_') + '.png')
            write_carton_label(marker['id'], png)
            pixels = cv2.imread(str(png))
            code, corners, _ = detector.detectAndDecode(pixels)
            if code != marker['id']:
                raise ValueError('printed marker does not decode to its identity')
            edge = np.linalg.norm(np.roll(corners.reshape(4, 2), -1, 0)
                                  - corners.reshape(4, 2), axis=1)
            half = marker['side_m'] * pixels.shape[0] / edge.mean() / 2
            grid = marker_module_grid(marker['id'])
            parent_from_marker = np.array(marker['parent_from_marker'])
            decoded, in_fov = decode_marker_render(grid, half, world, parent_from_marker, k)
            # data-region projected edge (pixels) for the pixel-size record
            data_half = half * (grid.shape[0] - 8) / grid.shape[0]
            quad = (parent_from_marker @ np.array(
                [[-data_half, -data_half, .0006, 1], [data_half, -data_half, .0006, 1],
                 [data_half, data_half, .0006, 1], [-data_half, data_half, .0006, 1]]).T).T[:, :3]
            uv, _ = project(transform_points(np.linalg.inv(world), quad), k)
            edges_px = np.linalg.norm(np.roll(uv, -1, 0) - uv, axis=1)
            results[marker['id']] = dict(
                decoded=decoded, decodes_to_identity=bool(decoded == marker['id']),
                full_quad_in_fov=bool(in_fov), module_count=int(grid.shape[0]),
                black_modules=int((grid == 0).sum()), marker_full_side_m=float(2 * half),
                data_side_m=float(marker['side_m']),
                data_edge_px_min=float(edges_px.min()),
                px_per_module=float(edges_px.min() / (grid.shape[0] - 8)))
    for path in sources:
        if hashes[str(path.resolve())] != sha(path):
            raise ValueError('input changed during proof')
    report = dict(evidence='CPU_geometry_fiducial_projection_decode_not_rendered_capture',
                  runtime_ready=False, input_sha256=hashes,
                  shelf_camera_intrinsics=dict(fx=k.fx, fy=k.fy, width=k.width, height=k.height),
                  markers=results,
                  all_decode=all(v['decodes_to_identity'] for v in results.values()),
                  limitations=[
                      'Rectified-projection decode proves the geometry pattern/size/pose, not '
                      'lighting, material, occlusion, noise or the native decoder path.',
                      'The data region is framed; the quiet-border corners clip the frame edges '
                      'at this oblique view, so integration should confirm on a real render.',
                      'Only the shelf support markers are geometry; source and cart stay textured.'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({mid: dict(decoded=v['decoded'], in_fov=v['full_quad_in_fov'],
                                px_per_module=round(v['px_per_module'], 1))
                      for mid, v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
