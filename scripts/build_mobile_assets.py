#!/usr/bin/env python3
"""Build mobile UR10 URDF + passive rollwagen USD offline, with no simulator boot."""
import argparse
import json
from pathlib import Path
from twin.mobile import MobileSpec
from twin.mobile_assets import build_mobile_urdf, author_cart, measure_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-urdf', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--package-root', type=Path, action='append', required=True,
                        help='repeat for ROS share directory and shelfbot package directory')
    parser.add_argument('--config', type=Path, help='JSON MobileSpec keyword overrides')
    args = parser.parse_args()
    spec = MobileSpec(**(json.loads(args.config.read_text()) if args.config else {}))
    out = args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    if any(out == repo/d or repo/d in out.parents for d in ('scenes', 'renders', 'export')):
        parser.error('choose a task scratch output directory')
    out.mkdir(parents=True, exist_ok=True)
    target = build_mobile_urdf(args.source_urdf, out/'mobile_ur10.urdf', spec)
    validation = measure_transport(out/'mobile_ur10.urdf', target,
                                   [str(p) for p in args.package_root], spec)
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateNew(str(out/'rollwagen.usda'))
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    paths = author_cart(stage, spec=spec)
    stage.SetDefaultPrim(stage.GetPrimAtPath(paths['root']))
    stage.GetRootLayer().Save()
    manifest = {'spec': spec.manifest(), 'transport_positions': target,
                'transport_validation': validation, 'cart': paths, 'mobile_urdf': str(out/'mobile_ur10.urdf'),
                'cart_usd': str(out/'rollwagen.usda'),
                'physics_validation': 'pending integration Isaac execution',
                'import': {'fix_base': False, 'merge_fixed_joints': False}}
    (out/'mobile_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
