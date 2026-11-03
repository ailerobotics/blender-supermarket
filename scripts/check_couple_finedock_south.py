#!/usr/bin/env python3
"""Offline proof (CPU-only) for sim-couple-finedock-south-v1: the couple docks a SOUTH-headed cart by a
straight reverse with no rotation, once the approach is computed in the DOCK frame instead of routed
through the stale authored (east-world) couple_approach pose.

Root cause (reconstructed from the saved runs towstaging1/towstaging2 under
~/datasets/twin-robot-in-store-boot-v1/mission): PR 271 re-headed the couple dock SOUTH (cart pinned at
~(-3.875, 15.80, -90 deg)) and PR 284 spawns the start_at=couple seat on-axis behind it
(~(-3.875, 14.5, -90 deg)) -- a perfect straight-reverse start. But the couple nav's FIRST leg still drove
the authored, world-frame couple_approach = (-4.46, 16.0, 0 deg): an EAST-dock pose. From the south seat
that pose is 0.585 m off the cart's hitch axis and 90 deg off the dock heading, so the base rotated ~180 deg
off-axis and stalled (recorded final base pose (-4.325, 14.88, +91 deg); nav 'tracking' forever, parked,
120 s no-base-progress). The fine-dock was NEVER reached.

The fix (opt-in couple_cart_relative.dock_frame_approach): drop the authored approach and drive a SINGLE
cart-relative standoff leg (fine_dock.standoff_from_cart on the MEASURED cart), then the fine-dock
straight-reverse insert -- collinear on the cart hitch axis for ANY dock yaw.

This proves, with the REAL fine_dock geometry + servo and the RECORDED south poses:
  (A) BEFORE: the authored east couple_approach is far OFF the south cart's hitch axis (large cross-track
      + a 90 deg heading mismatch) -- it demands a rotation, reproducing the off-axis stall.
  (B) AFTER: the dock-frame standoff is ON the measured cart axis (cross ~0, base yaw == dock yaw), and the
      fine-dock servo seats the drawbar INSIDE the attach handshake from there -- a straight reverse.
  (C) The SAME servo seats an EAST dock unchanged (yaw-general; no per-heading tuning).

Writes before/after error data + a plot to --out-dir (default the task data dir). Exit 0 iff BEFORE is
off-axis (fault reproduced) AND AFTER + EAST both seat inside attach. Prints COUPLE_FINEDOCK_SOUTH PASS/FAIL.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin.mobile import MobileSpec, Pose2, DockObservation
from twin.restock.fine_dock import (FineDock, FineDockConfig, standoff_from_cart,
                                     cart_relative_residuals, residuals, target_base, wrap)

SPEC = MobileSpec()  # hitch_x=-0.62, cart_hitch_x=0.65

# Recorded south couple (towstaging2 nav-trace: last couple tick). cart = [x, y, yaw_deg, art].
BAKED_SOUTH_CART = Pose2(-3.875, 15.7956, math.radians(-90.0))
BAKED_SOUTH_SEAT = Pose2(-3.875, 14.516, math.radians(-90.0))   # start_at=couple spawn (on-axis)
BAKED_SOUTH_STALL = Pose2(-4.325, 14.883, math.radians(91.2))   # where the legacy east approach stalled
AUTHORED_EAST_APPROACH = Pose2(-4.46, 16.0, 0.0)                # the stale world-frame couple_approach
EAST_CART = Pose2(-4.46, 15.25, 0.0)                            # coldwin22-style east dock

FD = dict(creep_speed=0.035, creep_yaw_rate=0.25, gain_lin=2.0, gain_yaw=1.0, hitch_tol_m=0.020,
          yaw_tol_rad=math.radians(4.0), settle_speed=0.01, time_cap_s=30.0, approach_mode='crosstrack',
          ct_gain_approach=2.0, ct_gain_cross=8.0, ct_gain_yaw=1.5, ct_align_hold_m=3.0, ct_min_creep=0.015,
          progress_cap=True, progress_patience_s=12.0, progress_min_improve_m=0.002,
          progress_max_time_cap_s=120.0)
STANDOFF_OFFSET = 0.40


def _dock(robot, cart, *, speed=0.0, yaw_rate=0.0, age=0.02):
    return DockObservation(robot=robot, cart=cart, robot_hitch_z=0.28, cart_hitch_z=0.28,
                           robot_speed=speed, cart_speed=0.0, robot_yaw_rate=yaw_rate,
                           cart_yaw_rate=0.0, age=age)


def axis_errors(base, cart):
    """Cross-track of the rear hitch off the cart axis (m), and base-vs-dock heading error (rad)."""
    _, cross, yr = cart_relative_residuals(_dock(base, cart), SPEC)
    return cross, yr


def closed_loop(start_pose, cart, *, dt=0.05, max_t=30.0, pivot_off=0.13, deadband=(0.003, 0.02)):
    """Integrate diff-drive kinematics (with the cn-002 pivot drift + a near-zero-command motion floor)
    under the REAL fine_dock servo. Returns (result, t, trace) where trace is per-tick axis errors."""
    v_dead, w_dead = deadband
    fd = FineDock(FineDockConfig(**FD), SPEC)
    fd.start(0.0)
    x, y, yaw = start_pose
    v = w = 0.0
    t = 0.0
    trace = []
    last = None
    while t < max_t:
        d = _dock(Pose2(x, y, yaw), cart, speed=abs(v), yaw_rate=abs(w))
        cross, yr = axis_errors(Pose2(x, y, yaw), cart)
        hm, _ = residuals(d, SPEC)
        trace.append({'t': round(t, 3), 'hitch_mm': hm * 1000.0,
                      'cross_mm': cross * 1000.0, 'yaw_deg': math.degrees(yr)})
        last = fd.step(d, t)
        if last.status in ('ready', 'failed'):
            break
        v, w = last.linear, last.angular
        v_eff = 0.0 if abs(v) < v_dead else v
        w_eff = 0.0 if abs(w) < w_dead else w
        dth = w_eff * dt
        x += v_eff * math.cos(yaw) * dt - pivot_off * math.sin(yaw) * dth
        y += v_eff * math.sin(yaw) * dt + pivot_off * math.cos(yaw) * dth
        yaw = math.atan2(math.sin(yaw + dth), math.cos(yaw + dth))
        t += dt
    return last, t, trace


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/home/manishsaini/firstmate/data/sim-couple-finedock-south-v1')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args(argv)

    cart = BAKED_SOUTH_CART
    seat = BAKED_SOUTH_SEAT

    # (A) BEFORE: the authored EAST approach, measured against the SOUTH cart axis.
    before_cross, before_yaw = axis_errors(AUTHORED_EAST_APPROACH, cart)
    #     The seat->approach leg's required heading turn (base at dock yaw must reach the east pose yaw 0).
    seat_to_approach_turn = abs(math.degrees(wrap(AUTHORED_EAST_APPROACH.yaw - seat.yaw)))
    stall_turn = abs(math.degrees(wrap(BAKED_SOUTH_STALL.yaw - seat.yaw)))  # observed rotation in the run
    before_off_axis = abs(before_cross) > 0.11 or abs(before_yaw) > math.radians(10.0)

    # (B) AFTER: dock-frame standoff on the measured cart axis + fine-dock seats it.
    sx, sy, syaw = standoff_from_cart(cart, STANDOFF_OFFSET, SPEC)
    standoff = Pose2(sx, sy, syaw)
    after_cross, after_yaw = axis_errors(standoff, cart)
    # A realistic arrival residual (within the capture band) so the servo is actually exercised.
    arrival = Pose2(sx + 0.05 * (-math.sin(cart.yaw)), sy + 0.05 * math.cos(cart.yaw),
                    syaw + math.radians(3.0))
    south_res, south_t, south_trace = closed_loop((arrival.x, arrival.y, arrival.yaw), cart)

    # (C) EAST dock unchanged (same servo, yaw 0).
    ex, ey, eyaw = standoff_from_cart(EAST_CART, STANDOFF_OFFSET, SPEC)
    east_arrival = Pose2(ex + 0.05 * (-math.sin(EAST_CART.yaw)), ey + 0.05 * math.cos(EAST_CART.yaw),
                         eyaw + math.radians(3.0))
    east_res, east_t, east_trace = closed_loop((east_arrival.x, east_arrival.y, east_arrival.yaw), EAST_CART)

    south_ok = (south_res.status == 'ready' and south_res.hitch_residual_m <= 0.020
                and abs(south_res.yaw_residual_rad) <= math.radians(4.0))
    east_ok = (east_res.status == 'ready' and east_res.hitch_residual_m <= 0.020
               and abs(east_res.yaw_residual_rad) <= math.radians(4.0))
    standoff_on_axis = abs(after_cross) < 1e-6 and abs(after_yaw) < 1e-6
    ok = before_off_axis and south_ok and east_ok and standoff_on_axis

    report = {
        'pass': bool(ok),
        'before_authored_east_approach': {
            'pose': [AUTHORED_EAST_APPROACH.x, AUTHORED_EAST_APPROACH.y,
                     math.degrees(AUTHORED_EAST_APPROACH.yaw)],
            'cross_track_m': round(before_cross, 4),
            'heading_error_deg': round(math.degrees(before_yaw), 2),
            'seat_to_approach_required_turn_deg': round(seat_to_approach_turn, 1),
            'observed_stall_rotation_deg': round(stall_turn, 1),
            'off_axis': bool(before_off_axis),
        },
        'after_dock_frame_standoff': {
            'pose': [round(sx, 4), round(sy, 4), round(math.degrees(syaw), 2)],
            'cross_track_m': round(after_cross, 6),
            'heading_error_deg': round(math.degrees(after_yaw), 6),
            'on_cart_axis': bool(standoff_on_axis),
            'fine_dock_status': south_res.status,
            'fine_dock_hitch_mm': round(south_res.hitch_residual_m * 1000.0, 2),
            'fine_dock_yaw_deg': round(math.degrees(south_res.yaw_residual_rad), 2),
            'fine_dock_seat_s': round(south_t, 2),
        },
        'east_dock_unchanged': {
            'fine_dock_status': east_res.status,
            'fine_dock_hitch_mm': round(east_res.hitch_residual_m * 1000.0, 2),
            'fine_dock_yaw_deg': round(math.degrees(east_res.yaw_residual_rad), 2),
            'fine_dock_seat_s': round(east_t, 2),
        },
        'traces': {'south': south_trace, 'east': east_trace},
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'finedock_south_errors.json').write_text(json.dumps(report, indent=2))

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            # Left: the BEFORE vs AFTER entry error (bar), showing the off-axis east approach vs on-axis.
            labels = ['cross-track (mm)', 'heading error (deg)']
            before_vals = [before_cross * 1000.0, math.degrees(before_yaw)]
            after_vals = [after_cross * 1000.0, math.degrees(after_yaw)]
            xpos = range(len(labels))
            axes[0].bar([p - 0.18 for p in xpos], before_vals, width=0.36,
                        label='BEFORE: authored east approach', color='#c0392b')
            axes[0].bar([p + 0.18 for p in xpos], after_vals, width=0.36,
                        label='AFTER: dock-frame standoff', color='#27ae60')
            axes[0].set_xticks(list(xpos)); axes[0].set_xticklabels(labels)
            axes[0].axhline(0, color='k', lw=0.6)
            axes[0].set_title('Couple entry alignment to the SOUTH cart axis')
            axes[0].legend(fontsize=8)
            # Right: the AFTER fine-dock closing the residual to the latch (south + east).
            for name, tr, col in (('south (-90 deg)', south_trace, '#2980b9'),
                                   ('east (0 deg)', east_trace, '#8e44ad')):
                axes[1].plot([r['t'] for r in tr], [r['hitch_mm'] for r in tr], label=name, color=col)
            axes[1].axhline(20.0, color='#e67e22', ls='--', lw=0.9, label='attach hitch tol (20 mm)')
            axes[1].set_xlabel('fine-dock time (s)'); axes[1].set_ylabel('hitch residual (mm)')
            axes[1].set_title('Fine-dock seats the drawbar (straight reverse)')
            axes[1].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / 'finedock_south_before_after.png', dpi=120)
            plt.close(fig)
            report['plot'] = str(out / 'finedock_south_before_after.png')
        except Exception as exc:   # plotting is best-effort; the JSON is the proof of record
            print(f'(plot skipped: {exc})', flush=True)

    print(f'COUPLE_FINEDOCK_SOUTH {"PASS" if ok else "FAIL"} '
          f'before[cross={before_cross*1000:.0f}mm yaw={math.degrees(before_yaw):.0f}deg '
          f'turn={seat_to_approach_turn:.0f}deg stall_rot={stall_turn:.0f}deg] '
          f'after[cross={after_cross*1000:.1f}mm seat={south_res.status} '
          f'hitch={south_res.hitch_residual_m*1000:.1f}mm in {south_t:.1f}s] '
          f'east[seat={east_res.status} hitch={east_res.hitch_residual_m*1000:.1f}mm]', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
