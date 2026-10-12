#!/usr/bin/env python3
"""Generate scene-specific Nav2 assets and optionally launch isolated ROS nodes."""
import argparse
import json
import subprocess
from pathlib import Path
from twin.navigation import Footprint,GridMap
from twin.navigation.nav2 import require_isolation,write_assets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',type=Path,required=True,
                   help='JSON with bounds, obstacles, resolution and footprint fields')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--uncoupled',action='store_true')
    p.add_argument('--launch',action='store_true')
    args=p.parse_args()
    spec=json.loads(args.scenario.read_text())
    grid=GridMap(spec['bounds'],spec['obstacles'],spec.get('resolution',0.2))
    footprint=Footprint(**spec['footprint'])
    # This does not install missing packages, change shared environments or boot Isaac.
    mapfile,params=write_assets(args.output,grid,footprint,coupled=not args.uncoupled)
    print(json.dumps({'map':str(mapfile),'params':str(params),'namespace':'restock'}))
    if args.launch:
        require_isolation()
        return subprocess.call(['ros2','launch','nav2_bringup','bringup_launch.py',
            'namespace:=restock','use_namespace:=True','use_sim_time:=True',
            'autostart:=True','use_composition:=False',
            f'map:={mapfile.resolve()}',f'params_file:={params.resolve()}'])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
