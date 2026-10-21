#!/usr/bin/env python3
"""Write the restock ``camera_config.json`` from the live design (``restock_camera.config``).

The station view design — camera placement, the paired QR marker layout and the support/witness
geometry for the collect source, the cart and the transfer shelf — is a GENERATED artifact of
``restock_camera.config.camera_config(scenario, manifest)``, not a hand-authored file. ONE
``camera_config.json`` is consumed by BOTH the commissioning bench (``twin.restock.view_scene``
authors the markers from it) AND the mission scene/sensing, so the two must share the same file:
a baseline commissioned against one design only re-registers at mission boot against the identical
design.

Regenerate it from the current design whenever ``config.py`` changes. A stale committed-design
artifact silently feeds the OLD geometry into commissioning — e.g. the pre-fix cart support
rectangle (0.62x0.42) overshoots the cart rails, so the sampled empty-deck patch catches the
raised rails and the production ``_flat_patch`` gate refuses it as "tilted, occupied or occluded"
(the current design insets it to 0.54x0.34, which stays on the flat deck). CPU-only; no Isaac/USD.
"""
import argparse
import json
from pathlib import Path

from twin.restock_camera.config import camera_config


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', required=True, type=Path, help='mission scenario JSON')
    p.add_argument('--manifest', required=True, type=Path, help='mobile manifest JSON')
    p.add_argument('--output', required=True, type=Path, help='camera_config.json path to write')
    p.add_argument('--mast-depth', action='store_true',
                   help='add the body-mounted mast depth camera (nvblox camera_1) that the carried '
                        'carton cannot block -- fixes the place re-park ESDF freeze (B\'). Requires '
                        'host.json mast_depth.enabled=true to be consumed.')
    a = p.parse_args()
    cfg = camera_config(json.loads(a.scenario.read_text()), json.loads(a.manifest.read_text()),
                        mast_depth=a.mast_depth)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(cfg, indent=1) + '\n')
    summary = {'wrote': str(a.output),
               'cart_required_xy_m': cfg['supports']['cart']['required_xy_m'],
               'shelf_required_xy_m': cfg['supports']['shelf']['required_xy_m'],
               'source_required_xy_m': cfg['supports']['source']['required_xy_m']}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
