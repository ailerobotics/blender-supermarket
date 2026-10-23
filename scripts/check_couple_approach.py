#!/usr/bin/env python3
"""Offline proof (CPU-only) for sim-robust-couple-approach-v1: the couple APPROACH lands the base
consistently close to the hitch alignment so the fine-dock starts inside its capture box, regardless of
where the pinned cart settled.

Root cause (reconstructed from the recorded runs' nav-trace.jsonl): the authored ``couple_standoff`` is a
FIXED map pose with yaw 0, but the pinned cart's MEASURED yaw varies run to run (coldlivefinal -1.58 deg,
coupledock +19.17 deg). Navigating to the fixed pose lands the base aligned to the MAP, not the CART, so
the fine-dock enters from an 8-27 deg yaw / 25-470 mm hitch residual it cannot null (it lands on the
``ct_gain_cross*cross == ct_gain_yaw*yaw_err`` equilibrium) and times out at the fixed 30 s cap.

This proves, with the REAL ``fine_dock`` servo + the REAL cart-relative standoff geometry + the RECORDED
measured cart poses:
  (A) FIXED standoff: the fine-dock FAILS to seat the 19 deg-off coupledock cart (reproduces the failure);
  (B) CART-MEASURED standoff: the fine-dock seats BOTH recorded carts inside the attach handshake, from a
      consistent on-axis / aligned entry (the deliverable);
  (C) PROGRESS CAP: a slow monotone close (coupledock's ~13 mm/s) that the fixed 30 s cap cut off now
      latches, while a limit cycle still fails FAST at the patience window.

Run:  python3 scripts/check_couple_approach.py --runs-dir <.../mission> --runs coldlivefinal coupledock
      (a run name resolves to <runs-dir>/<name>/nav-trace.jsonl; omit --runs-dir to use the two baked
       recorded cart poses). Exit 0 iff every check passes (prints COUPLE_APPROACH_PROOF PASS/FAIL)."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin.mobile import MobileSpec, Pose2, DockObservation
from twin.restock.fine_dock import (FineDock, FineDockConfig, standoff_from_cart,
                                     cart_relative_residuals, residuals)

SPEC = MobileSpec()
AUTHORED_STANDOFF = (-4.06, 15.25, 0.0)      # the fixed map pose both recorded runs used (yaw 0)
# Baked measured cart poses (last couple tick of each run's nav-trace; cart yaw is the degrees field).
BAKED = {'coldlivefinal': Pose2(-5.7445, 15.2366, math.radians(-1.58)),
         'coupledock': Pose2(-5.7423, 15.3672, math.radians(19.17))}

FD = dict(creep_speed=0.035, creep_yaw_rate=0.25, gain_lin=2.0, gain_yaw=1.0, hitch_tol_m=0.020,
          yaw_tol_rad=math.radians(4.0), settle_speed=0.01, time_cap_s=30.0, approach_mode='crosstrack',
          ct_gain_approach=2.0, ct_gain_cross=8.0, ct_gain_yaw=1.5, ct_align_hold_m=3.0, ct_min_creep=0.015)
PROG = dict(FD, progress_cap=True, progress_patience_s=12.0, progress_max_time_cap_s=120.0)


def _dock(robot, cart, speed=0.0, yaw_rate=0.0):
    return DockObservation(robot=robot, cart=cart, robot_hitch_z=0.28, cart_hitch_z=0.28,
                           robot_speed=speed, cart_speed=0.0, robot_yaw_rate=yaw_rate,
                           cart_yaw_rate=0.0, age=0.02)


def _closed_loop(start, cart, cfg, *, pivot=0.13, dead=(0.003, 0.02), dt=0.05, max_t=140):
    """REAL FineDock servo against the cn-002 pivot-drift + motion-floor plant (same model the
    fine-dock unit tests use), from ``start`` onto the pinned ``cart``."""
    fd = FineDock(FineDockConfig(**cfg), SPEC); fd.start(0.0)
    x, y, yaw = start
    v = w = 0.0
    t = 0.0
    last = None
    vd, wd = dead
    while t < max_t:
        last = fd.step(_dock(Pose2(x, y, yaw), cart, abs(v), abs(w)), t)
        if last.status in ('ready', 'failed'):
            return last, t
        v, w = last.linear, last.angular
        ve = 0.0 if abs(v) < vd else v
        we = 0.0 if abs(w) < wd else w
        dth = we * dt
        x += ve * math.cos(yaw) * dt - pivot * math.sin(yaw) * dth
        y += ve * math.sin(yaw) * dt + pivot * math.cos(yaw) * dth
        yaw = math.atan2(math.sin(yaw + dth), math.cos(yaw + dth))
        t += dt
    return last, t


def cart_from_trace(path):
    """Measured, world-pinned cart pose from the last couple tick of a run's nav-trace."""
    last = None
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get('action') == 'couple':
                last = d
    if last is None:
        raise SystemExit(f'no couple ticks in {path}')
    c = last['cart']           # [x, y, yaw_deg, z]
    return Pose2(c[0], c[1], math.radians(c[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', type=Path, help='mission runs dir (<runs-dir>/<run>/nav-trace.jsonl)')
    ap.add_argument('--runs', nargs='*', default=['coldlivefinal', 'coupledock'])
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()

    carts = {}
    for name in args.runs:
        if args.runs_dir is not None and (args.runs_dir / name / 'nav-trace.jsonl').exists():
            carts[name] = cart_from_trace(args.runs_dir / name / 'nav-trace.jsonl')
        elif name in BAKED:
            carts[name] = BAKED[name]
        else:
            raise SystemExit(f'no trace for {name!r} and no baked pose')

    out = {'checks': [], 'carts': {}}
    ok = True

    def check(name, passed, detail):
        nonlocal ok
        ok = ok and passed
        out['checks'].append({'name': name, 'pass': bool(passed), 'detail': detail})
        print(f'  [{"PASS" if passed else "FAIL"}] {name}: {detail}')

    # (A) + (B): per recorded cart, fixed standoff vs cart-measured standoff through the REAL fine-dock.
    for name, cart in carts.items():
        out['carts'][name] = {'cart_yaw_deg': round(math.degrees(cart.yaw), 2)}
        print(f'\n{name}: measured cart yaw {math.degrees(cart.yaw):+.2f} deg')
        # (A) fixed authored standoff
        fixed_res = residuals(_dock(Pose2(*AUTHORED_STANDOFF), cart), SPEC)
        rf, tf = _closed_loop(AUTHORED_STANDOFF, cart, FD, max_t=35)
        # (B) cart-measured standoff
        s = standoff_from_cart(cart, 0.40, SPEC)
        _, cr0, yr0 = cart_relative_residuals(_dock(Pose2(*s), cart), SPEC)
        rc, tc = _closed_loop(s, cart, PROG, max_t=120)
        out['carts'][name].update(
            fixed_entry_mm=round(fixed_res[0] * 1000, 1), fixed_entry_yaw_deg=round(math.degrees(fixed_res[1]), 2),
            fixed_result=rf.status, cart_rel_entry_cross_mm=round(cr0 * 1000, 2),
            cart_rel_entry_yaw_deg=round(math.degrees(yr0), 3), cart_rel_result=rc.status,
            cart_rel_seat_mm=round(rc.hitch_residual_m * 1000, 1), cart_rel_seat_s=round(tc, 1))
        # cart-measured entry must be on-axis + aligned, and the fine-dock must seat it.
        check(f'{name}: cart-measured standoff entry aligned',
              abs(cr0) < 1e-3 and abs(yr0) < 1e-3, f'cross={cr0*1e3:.2f}mm yaw={math.degrees(yr0):.3f}deg')
        check(f'{name}: fine-dock seats from cart-measured standoff',
              rc.status == 'ready' and rc.hitch_residual_m <= 0.020 and abs(rc.yaw_residual_rad) <= math.radians(4),
              f'{rc.status} seat={rc.hitch_residual_m*1e3:.1f}mm/{math.degrees(rc.yaw_residual_rad):.2f}deg in {tc:.1f}s')

    # (A) control: the 19 deg-off cart must FAIL from the fixed standoff (reproduce the recorded failure).
    worst = max(carts.values(), key=lambda c: abs(c.yaw))
    rwf, _ = _closed_loop(AUTHORED_STANDOFF, worst, FD, max_t=35)
    check('fixed standoff FAILS the most-off cart (recorded failure reproduced)',
          rwf.status == 'failed' and abs(rwf.yaw_residual_rad) > math.radians(4),
          f'cart yaw {math.degrees(worst.yaw):+.1f}deg -> {rwf.status} '
          f'yaw={math.degrees(rwf.yaw_residual_rad):.2f}deg')

    # (C) progress cap: a slow monotone close past the fixed 30 s cap latches; a limit cycle fails fast.
    def scripted(cfg, schedule, max_t=60):
        cart = Pose2(-5.73, 15.25, 0.0)
        fd = FineDock(FineDockConfig(**cfg), SPEC); fd.start(0.0)
        t = 0.0
        r = None
        while t <= max_t:
            hitch = schedule(t)
            sx, sy, syaw = standoff_from_cart(cart, max(hitch, 1e-4), SPEC)
            r = fd.step(_dock(Pose2(sx, sy, syaw), cart, speed=0.01), t)
            if r.status in ('ready', 'failed'):
                return r, t
            t += 0.1
        return r, t
    rprog, tprog = scripted(PROG, lambda t: max(0.0, 0.47 - 0.013 * t), max_t=60)
    rfixed, _ = scripted(FD, lambda t: max(0.0, 0.47 - 0.013 * t), max_t=60)
    check('progress cap latches the slow monotone close the fixed cap cut',
          rprog.status == 'ready' and tprog > 30.0 and rfixed.status == 'failed',
          f'progress={rprog.status}@{tprog:.1f}s fixed={rfixed.status}')
    rcyc, tcyc = scripted(dict(PROG, progress_patience_s=8.0),
                          lambda t: 0.030 + 0.001 * math.sin(t * 3), max_t=60)
    check('progress cap fails a limit cycle fast (no-progress, before the hard cap)',
          rcyc.status == 'failed' and 'no progress' in rcyc.reason and tcyc < 20.0,
          f'{rcyc.status}@{tcyc:.1f}s: {rcyc.reason}')

    out['pass'] = ok
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
    print(f'\nCOUPLE_APPROACH_PROOF {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
