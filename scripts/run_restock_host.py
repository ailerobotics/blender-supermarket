#!/usr/bin/env python3
"""CPU preflight/launch plan by default; execution requires explicit --launch."""
import argparse
import json
import os
from pathlib import Path

from twin.restock_host.config import isolated_environment, launch_plan
from twin.restock_host.preflight import dependencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--probe', action='store_true', help='bounded graph discovery only; no service calls')
    args = parser.parse_args()
    try:
        env = isolated_environment(os.environ)
        config = json.loads(args.config.read_text())
        plan = launch_plan(config)
        if args.plan:
            print(json.dumps(plan, indent=2))
            return 0
        if args.probe:
            os.environ.update(env)
            from twin.restock_host.readiness import probe
            result = probe()
            print(json.dumps(result, indent=2))
            return 2 if result['missing_endpoints'] else 0
        result = dependencies()
        result['errors'].extend(plan['static_errors'])
        print(json.dumps({**result, 'runtime_verified': False,
                          'required_external': plan['required_external']}, indent=2), flush=True)
        if result['errors']:
            return 2
        if args.check:
            return 0
        # Explicitly choosing --launch is not evidence of sensor/endpoint readiness.
        # Owner must authorize compute; integration gates jobs on live evidence.
        os.environ.update(env)
        from launch import LaunchService
        from twin.restock_host.launch import description
        service = LaunchService()
        service.include_launch_description(description(config))
        return service.run()
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print(json.dumps({'errors': [str(exc)], 'runtime_verified': False}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
