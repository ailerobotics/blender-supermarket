#!/usr/bin/env python3
"""Offline contract validation or explicit guarded Isaac runtime (RUNBOOK §9)."""
import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import threading

from twin.restock.guard import available_ram_mb, gpu_slot
from twin.restock.inventory import Inventory, Job
from twin.restock.mission import Mission
from twin.restock.scenario import validate


def check(args):
    manifest = json.loads(args.manifest.read_text())
    scenario = json.loads(args.scenario.read_text())
    report = validate(manifest, scenario)
    root = Path(__file__).resolve().parent
    required = ['twin/navigation/core.py', 'twin/navigation/nav2.py',
                'twin/mobile_runtime.py', 'twin/restock_manipulation/controller.py']
    report['missing_modules'] = [p for p in required if not (root/p).is_file()]
    report['ok'] = report['ok'] and not report['missing_modules']
    return report, manifest, scenario


def run(args, manifest, scenario):
    if not args.factory or ':' not in args.factory:
        raise ValueError('runtime requires --factory module:function; no mock fallback')
    if not args.output:
        raise ValueError('runtime requires an explicit scratch --output directory')
    # New output directory only: a restart must reconcile existing physical/job state.
    # Reject any existing path before loading the simulator.
    args.output.mkdir(parents=True, exist_ok=False)
    db = Inventory(args.output/'inventory.sqlite')
    for row in scenario['initial_inventory']:
        db.seed(row['location'], row['sku'], row['quantity'])
    mission = app = bundle = None
    finished = threading.Event()
    def interrupted(signum, frame):
        raise RuntimeError(f'runtime interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def watch_ram():
        while not finished.wait(1):
            try:
                safe = available_ram_mb() >= 4096
            except Exception:
                safe = False
            if not safe:
                os.kill(os.getpid(), signal.SIGTERM)
                return
    summary = {'state': 'failed', 'reason': 'runtime did not start', 'physics_verified': False}
    try:
        with gpu_slot(heavy_threshold_mb=args.heavy_threshold_mb):
            watcher = threading.Thread(target=watch_ram, daemon=True)
            watcher.start()
            try:
                # NVIDIA requires runtime/plugin imports after SimulationApp startup.
                from isaacsim import SimulationApp
                app = SimulationApp({'headless': True, 'width': 960, 'height': 540})
                module, name = args.factory.split(':', 1)
                factory = getattr(importlib.import_module(module), name)
                # Factory authors stage, initializes sensors and returns an object with
                # adapter (RuntimeAdapter), close(), and recording metadata. It must not
                # create another app or issue commands outside the simulation domain.
                bundle = factory(app=app, manifest=manifest, scenario=scenario, output=args.output)
                if getattr(bundle.adapter, 'fixture_mode', True):
                    raise RuntimeError('production runtime refuses fixture adapters')
                mission = Mission(db, Job(**scenario['job']), bundle.adapter)
                mission.start()
                for _ in range(60*20*60):  # bounded simulation time; shell also bounds wall time
                    if mission.terminal:
                        break
                    if not app.is_running():
                        raise RuntimeError('simulation app stopped')
                    mission.tick(1/60)
                if not mission.terminal:
                    mission.fail('simulation step budget exceeded')
                summary = {'state': mission.state, 'reason': mission.reason,
                           'physics_verified': False,
                           'note': 'Mission success alone does not verify video, contacts or autonomy.'}
            except Exception as exc:
                if mission is not None and not mission.terminal:
                    mission.fail(str(exc))
                raise
            finally:
                # Cleanup stays inside GPU lock, including partial boot failures.
                try:
                    if bundle is not None:
                        try:
                            bundle.adapter.stop()
                        finally:
                            bundle.close()
                finally:
                    if app is not None:
                        app.close()
    except Exception as exc:
        summary['reason'] = str(exc)
        raise
    finally:
        finished.set()
        summary['events'] = db.events(scenario['job']['job_id'])
        (args.output/'mission.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        db.close()
    return 0 if summary['state'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--scenario', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--factory', help='installed, reviewed simulator bundle module:function')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--heavy-threshold-mb', type=int, choices=(1500,2500), default=1500)
    args = parser.parse_args()
    report, manifest, scenario = check(args)
    print(json.dumps(report, indent=2))
    if not report['ok']:
        return 2
    if args.execute:
        return run(args, manifest, scenario)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
