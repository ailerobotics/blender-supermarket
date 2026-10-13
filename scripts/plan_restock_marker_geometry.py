#!/usr/bin/env python3
"""CPU proof that the geometry shelf fiducials decode at the shelf camera; never a render.

The native render never sampled the shelf markers' texture, so they are authored as
geometry (a white base quad plus one black quad per dark module). This AUTHORS the markers
into a USD stage and decodes them from the composed world-space mesh vertices projected
through the exact shelf camera — reading the authored geometry, not the design model, so an
authoring fault (e.g. the black-module child mesh double-applying its parent transform,
which put the pattern behind the camera and rendered a blank white plane) is caught. It is
not a renderer (no lighting, material, occlusion or noise), so a decode is necessary, not
sufficient, for the native render.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from twin.restock_camera.config import (camera_config, author_camera_design,
                                        decode_authored_marker)
from twin.restock_camera.geometry import Intrinsics
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
    from pxr import Usd, UsdGeom
    camera = config['cameras']['shelf']
    world = np.array(camera['parent_from_optical'])          # shelf camera parent is 'map'
    k = Intrinsics(**camera['intrinsics'])
    results = {}

    def world_centroid(stage, path):
        prim = stage.GetPrimAtPath(path)
        matrix = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        return ((matrix @ np.c_[points, np.ones(len(points))].T).T[:, :3]).mean(axis=0)

    with tempfile.TemporaryDirectory() as tmp:
        textures = {}
        for support in config['supports'].values():
            for marker in support['markers']:
                png = Path(tmp) / (marker['id'].replace(':', '_') + '.png')
                write_carton_label(marker['id'], png)
                textures[marker['id']] = png
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, '/World'); UsdGeom.Xform.Define(stage, '/Cart')
        author_camera_design(stage, config, parent_paths={'map': '/World', 'cart_chassis': '/Cart'},
                             marker_textures=textures)
        for index, marker in enumerate(config['supports']['shelf']['markers']):
            base = f'/World/RestockCamera_shelf_marker_{index}'
            decoded = decode_authored_marker(stage, base, world, k)
            base_c = world_centroid(stage, base)
            modules_c = world_centroid(stage, base + '/Modules')
            # The black module mesh must share the white base's plane; a double transform
            # would move it far away (the native double-transform put it behind the camera).
            aligned = bool(np.linalg.norm(modules_c[:2] - base_c[:2]) < .02
                           and abs(modules_c[2] - base_c[2]) < .002)
            results[marker['id']] = dict(
                decoded=decoded, decodes_to_identity=bool(decoded == marker['id']),
                base_world_centroid=base_c.round(4).tolist(),
                modules_world_centroid=modules_c.round(4).tolist(),
                modules_share_base_plane=aligned)
    for path in sources:
        if hashes[str(path.resolve())] != sha(path):
            raise ValueError('input changed during proof')
    report = dict(evidence='CPU_authored_geometry_fiducial_decode_not_rendered_capture',
                  runtime_ready=False, input_sha256=hashes,
                  shelf_camera_intrinsics=dict(fx=k.fx, fy=k.fy, width=k.width, height=k.height),
                  markers=results,
                  all_decode=all(v['decodes_to_identity'] for v in results.values()),
                  all_modules_aligned=all(v['modules_share_base_plane'] for v in results.values()),
                  limitations=[
                      'Decode of the authored world geometry proves pattern/size/pose and that '
                      'the black modules sit on the base plane; it is not lighting, material, '
                      'occlusion, noise or the native decoder path.',
                      'The data region is framed; the quiet-border corners clip the frame edges '
                      'at this oblique view, so integration should confirm on a real render.',
                      'Only the shelf support markers are geometry; source and cart stay textured.'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({mid: dict(decoded=v['decoded'], modules_aligned=v['modules_share_base_plane'])
                      for mid, v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
