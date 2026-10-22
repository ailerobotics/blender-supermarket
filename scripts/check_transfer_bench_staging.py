#!/usr/bin/env python3
"""CPU proof for the warm start_at=transfer bench: staged poses == recorded end-of-park poses.

Reads the raw evidence of real coupled tows (each run's nav-trace.jsonl + console log), takes the
park->transfer handover (first nav-trace row whose action is 'transfer', right after park_succeeded),
and checks the start staging ``shortcut.staged_scenario`` builds for ``start_at=transfer`` +
``start_staging=recorded_bay_park`` against it:

* robot base pose vs every run's handover pose (xy / yaw tolerance), coupled + parked there;
* rollwagen: deck centre in ur_base_link vs the run's own TRANSFER_CAPTURE ``cart=`` measurement, the
  hitch residual (must be ~0) and the cart yaw vs the run's last tow-lane ``cart_err``;
* carton centre at the decided spot: the scenario's upper shelf (cart_support_z_world) at its
  cart_load_center_body_xy -- the same spot the warm start_at=couple shortcut uses;
* the mission (state machine, and the behaviour tree when py_trees imports) begins at transfer with
  collect/couple/outbound/park recorded as ``*_shortcut``.

No Isaac, no GPU. Exit 0 = PASS. Example:

  python3 scripts/check_transfer_bench_staging.py \\
      --mission-dir ~/datasets/twin-robot-in-store-boot-v1/mission --runs 351 353 354 356
"""
import argparse
import json
import math
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin.mobile import Pose2                                        # noqa: E402
from twin.restock.inventory import Inventory, Job                    # noqa: E402
from twin.restock.mission import Mission, Result                     # noqa: E402
from twin.restock.shortcut import plan_start_state, staged_scenario  # noqa: E402

DEFAULT_MISSION_DIR = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mission'
CAPTURE = re.compile(r'^TRANSFER_CAPTURE calls=\d+ .*?cart=\(([-\d.]+),([-\d.]+),([-\d.]+)\)')
TRACK = re.compile(r'^TRACK_DIAG .*?cart_err=([-\d.]+)')


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def recorded_handover(mission_dir, run):
    """The recorded park->transfer handover of one run (pure file reads)."""
    rows = [json.loads(line) for line in (mission_dir / run / 'nav-trace.jsonl').read_text().splitlines()
            if line.strip()]
    i = next(k for k, r in enumerate(rows) if r['action'] == 'transfer')
    row, park = rows[i], rows[i - 1]
    console = (mission_dir / f'{run}.console.log').read_text(errors='replace').splitlines()
    capture = next((tuple(float(v) for v in m.groups()) for m in map(CAPTURE.match, console) if m), None)
    cart_errs = [float(m.group(1)) for m in map(TRACK.match, console) if m]
    events = json.loads((mission_dir / run / 'mission.json').read_text()).get('events', [])
    return {'run': run, 'tick': row['tick'], 't': row['t'], 'prev_action': park['action'],
            'robot': (row['x'], row['y'], math.radians(row['yaw_deg'])),
            'coupled': row['coupled'], 'parked': row['parked'],
            'park_succeeded': any(e.get('state') == 'park_succeeded' for e in events),
            'cart_in_base': capture, 'last_cart_err': cart_errs[-1] if cart_errs else None}


class _Fixture:
    def __init__(self):
        self.begun, self.time = [], 0.

    def start(self, action, job):
        self.begun.append(action)
        self.action, self.job = action, job

    def step(self, dt):
        self.time += dt
        return Result('succeeded', dict(
            source='test_fixture', observation_id=f'{self.action}-{self.time}', simulation_time=self.time,
            loaded=True, coupled=True, arrived=True, stationary=True, released=True, home=True,
            sku=self.job.sku, quantity=self.job.quantity, location=self.job.shelf_id,
            transfer_confirmed=True), '')

    def stop(self):
        pass


def entry_check(job, tree):
    """Run a fixture mission start_at=transfer; return (first stage begun, *_shortcut events)."""
    with tempfile.TemporaryDirectory() as tmp:
        inv = Inventory(str(Path(tmp) / 'stock.sqlite'))
        try:
            inv.seed(job['source_bin'], job['sku'], 1)
            adapter = _Fixture()
            m = Mission(inv, Job(**{k: job[k] for k in ('job_id', 'sku', 'quantity', 'source_bin', 'shelf_id')}),
                        adapter, fixture_mode=True, start_at='transfer')
            if tree:
                from twin.restock.bt.driver import build_bus, run_mission_tree
                run_mission_tree(m, build_bus(None)[0], max_ticks=200)
            else:
                m.start()
                for _ in range(100):
                    if m.terminal:
                        break
                    m.tick(1 / 60)
            events = [e['state'] for e in inv.events(job['job_id'])]
            return {'orchestrator': 'behaviour_tree' if tree else 'state_machine', 'state': m.state,
                    'begun': adapter.begun, 'shortcut_events': [e for e in events if e.endswith('_shortcut')]}
        finally:
            inv.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mission-dir', type=Path, default=DEFAULT_MISSION_DIR)
    p.add_argument('--runs', nargs='+', default=['351', '353', '354', '356'])
    p.add_argument('--scenario', type=Path, help='default: <mission-dir>/mission-scenario.json')
    p.add_argument('--runtime-config', type=Path,
                   help='bay runtime config (hitch geometry via its mobile_manifest); '
                        'default: <mission-dir>/mission-runtime-couple.json')
    p.add_argument('--xy-tol', type=float, default=0.01)
    p.add_argument('--yaw-tol', type=float, default=0.01)
    p.add_argument('--json', type=Path, help='write the proof record here')
    a = p.parse_args()
    scenario = json.loads((a.scenario or a.mission_dir / 'mission-scenario.json').read_text())
    runtime = json.loads((a.runtime_config or a.mission_dir / 'mission-runtime-couple.json').read_text())
    spec = json.loads(Path(runtime['mobile_manifest']).read_text())['spec']
    config = {'start_at': 'transfer', 'start_staging': 'recorded_bay_park', 'stages': runtime.get('stages')}
    variant, record = staged_scenario(scenario, config, hitch_x=spec['hitch_x'],
                                      cart_hitch_x=spec['cart_hitch_x'])
    st = variant['initial_staging']
    robot, cart = Pose2(*st['robot_base_pose']), Pose2(*st['cart_body_pose'])
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    dx, dy = cart.x - robot.x, cart.y - robot.y
    cart_in_base = (c * dx + s * dy, -s * dx + c * dy)
    hitch_residual = math.dist(robot.point(spec['hitch_x']), cart.point(spec['cart_hitch_x']))
    plan = plan_start_state(variant, 'transfer', stages=runtime.get('stages'))

    checks, rows = [], []
    def check(name, ok, detail):
        checks.append({'check': name, 'pass': bool(ok), 'detail': detail})

    for run in (f'run-{r}' if not str(r).startswith('run-') else r for r in a.runs):
        h = recorded_handover(a.mission_dir, run)
        rx, ry, ryaw = h['robot']
        dxy, dyaw = math.hypot(robot.x - rx, robot.y - ry), abs(_wrap(robot.yaw - ryaw))
        rows.append(dict(h, dxy=round(dxy, 4), dyaw_deg=round(math.degrees(dyaw), 2)))
        check(f'{run} robot pose', dxy <= a.xy_tol and dyaw <= a.yaw_tol,
              f'staged ({robot.x:.3f},{robot.y:.3f},{math.degrees(robot.yaw):.2f}deg) vs recorded '
              f'({rx:.3f},{ry:.3f},{math.degrees(ryaw):.2f}deg): dxy={dxy*1e3:.1f}mm dyaw={math.degrees(dyaw):.2f}deg')
        check(f'{run} coupled+parked at handover', h['coupled'] and h['parked'] and h['park_succeeded']
              and h['prev_action'] == 'park', f"coupled={h['coupled']} parked={h['parked']} "
              f"park_succeeded={h['park_succeeded']} prev={h['prev_action']}")
        if h['cart_in_base'] is not None:
            d = math.dist(cart_in_base, h['cart_in_base'][:2])
            check(f'{run} cart (deck centre in ur_base_link)', d <= 0.006,
                  f'staged ({cart_in_base[0]:.3f},{cart_in_base[1]:.3f}) vs TRANSFER_CAPTURE '
                  f"({h['cart_in_base'][0]:.2f},{h['cart_in_base'][1]:.2f}): {d*1e3:.1f}mm (print res 5mm)")
        if h['last_cart_err'] is not None:
            check(f'{run} cart yaw vs tow lane', abs(cart.yaw) <= h['last_cart_err'] + a.yaw_tol,
                  f"staged cart yaw {cart.yaw:+.4f} rad vs recorded |cart_err| {h['last_cart_err']:.3f} rad")
    check('hitch residual', hitch_residual <= 1e-6, f'{hitch_residual:.2e} m (Coupler gate 0.025 m)')
    shelf_z = scenario['initial_staging']['cart_support_z_world']
    load = scenario['initial_staging']['cart_load_center_body_xy']
    want = (cart.x + load[0] * math.cos(cart.yaw) - load[1] * math.sin(cart.yaw),
            cart.y + load[0] * math.sin(cart.yaw) + load[1] * math.cos(cart.yaw),
            shelf_z + scenario['handling_unit']['dimensions_m'][2] / 2)
    check('carton on upper shelf at the decided spot', math.dist(plan.carton_position, want) <= 1e-9
          and plan.couple_cart, f'carton centre {tuple(round(v, 4) for v in plan.carton_position)} = shelf '
          f'top {shelf_z} + half height, body {load} (warm couple shortcut spot); couple_cart={plan.couple_cart}')
    entries = [entry_check(scenario['job'], tree=False)]
    try:
        import py_trees  # noqa: F401
        entries.append(entry_check(scenario['job'], tree=True))
    except ImportError:
        print('NOTE behaviour-tree entry check skipped: py_trees not importable '
              '(set PYTHONPATH to the RESTOCK_BT_PYTHONPATH dir)')
    for e in entries:
        check(f"{e['orchestrator']} enters transfer", e['begun'][:1] == ['transfer'] and e['shortcut_events']
              == ['collect_shortcut', 'couple_shortcut', 'outbound_shortcut', 'park_shortcut']
              and e['state'] == 'complete', f"begun={e['begun']} shortcut_events={e['shortcut_events']}")

    for c_ in checks:
        print(('PASS ' if c_['pass'] else 'FAIL ') + c_['check'] + ': ' + c_['detail'])
    ok = all(c_['pass'] for c_ in checks)
    print('TRANSFER_BENCH_STAGING_PROOF', 'PASS' if ok else 'FAIL',
          f"robot={st['robot_base_pose']} cart={[round(v, 4) for v in st['cart_body_pose']]}")
    if a.json:
        a.json.write_text(json.dumps({'pass': ok, 'staging': record, 'recorded': rows, 'checks': checks,
                                      'entry': entries}, indent=2) + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
