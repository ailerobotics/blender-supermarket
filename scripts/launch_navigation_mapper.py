#!/usr/bin/env python3
"""Print a navigation mapper plan; --execute is integration-owned GPU execution."""
import argparse
import json
from twin.navigation.mapper_launch import mapper_plan,description


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation',required=True)
    parser.add_argument('--camera-count',required=True,type=int)
    parser.add_argument('--transport-height',required=True,type=float)
    parser.add_argument('--execute',action='store_true',help='start mapper; integration GPU guards required')
    args=parser.parse_args()
    config=dict(generation=args.generation,camera_count=args.camera_count,transport_height=args.transport_height)
    plan=mapper_plan(**config)
    print(json.dumps(plan,indent=2),flush=True)
    if args.execute:
        from launch import LaunchService
        service=LaunchService()
        service.include_launch_description(description(**config))
        return service.run()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
