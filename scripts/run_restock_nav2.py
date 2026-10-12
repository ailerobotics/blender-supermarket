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
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--scenario',type=Path,
                   help='JSON with bounds, obstacles, resolution and footprint fields')
    inputs.add_argument('--environment',type=Path,help='Exported schema-2 environment manifest')
    p.add_argument('--integration-scenario',type=Path,help='Include staging support with --environment')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--uncoupled',action='store_true')
    p.add_argument('--launch',action='store_true')
    p.add_argument('--articulated',action='store_true',help='Use with ArticulatedNav2Adapter FollowPath')
    args=p.parse_args()
    if args.integration_scenario and not args.environment:
        p.error('--integration-scenario requires --environment')
    from twin.mobile import MobileSpec
    import math
    if args.environment:
        from twin.navigation.scenario import environment_grid
        spec=json.loads(args.environment.read_text())
        scenario=json.loads(args.integration_scenario.read_text()) if args.integration_scenario else None
        grid=environment_grid(spec,scenario)
        rig=MobileSpec()
        footprint=Footprint(rig.transport_radius,abs(rig.hitch_x),
            math.hypot(rig.cart_hitch_x+rig.cart_length/2,rig.cart_width/2),
            margin=.05,min_turn_radius=1.5,max_articulation=rig.max_articulation)
    else:
        spec=json.loads(args.scenario.read_text())
        grid=GridMap(spec['bounds'],spec['obstacles'],spec.get('resolution',0.2))
        footprint=Footprint(**spec['footprint'])
    # This does not install missing packages, change shared environments or boot Isaac.
    mobile_spec=MobileSpec(**spec.get('mobile_spec',{})) if args.articulated else None
    mapfile,params=write_assets(args.output,grid,footprint,coupled=not args.uncoupled,
                               articulated_spec=mobile_spec)
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
