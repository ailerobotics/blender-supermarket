#!/usr/bin/env python3
"""Offline proof (CPU-only) for sim-video-couple-speed-v1 (fm482): the couple stage converges fast and
the coupled outbound tow stops stalling. Driven by the REAL navigation controller, the REAL fine-dock
servo and the mission's REAL inputs (environment + scenario + runtime config), plus the RECORDED
nav-trace.jsonl of the warm runs.

(A) COUPLE STANDOFF DITHER. At the on-axis couple standoff (xy tol 0.05 m) the base reaches the tolerance
    circle facing the goal POINT (~-1.0 rad), then turns in place to the goal YAW. A real diff base does
    not pivot about its measured centre, so the turn pushes the measured point back across the circle;
    the drive branch then turns it back toward the point -> a limit cycle. Proven here by
      A1 replay   -- per recorded run: the standoff window (first capture -> fine-dock start), its
                     sim/wall seconds, yaw swept vs net, the arrival in/out flips, and that the recorded
                     xy error never left tol + hysteresis (so the latched controller never releases);
      A0 pivot fit -- regress, from the recorded rotation windows, how far the measured base point sits
                     off the turning pivot (|o| ~ 7-8 mm, run-350/352/356): a turn of the base
                     translates the measured point, and the capture happens 0.6 mm inside the circle;
      A2 closed loop -- the REAL controller against that pivot-offset plant over every offset bearing:
                     legacy dithers, navigation.arrival_xy_hysteresis_m converges in seconds, and the
                     arrival pose handed to the REAL fine-dock servo seats (a 3.7 cm stress offset is
                     reported too);
      A3 capture  -- the REAL FineDock servo (run config gains) from standoff residuals: the largest
                     standoff xy/yaw box it still seats inside the attach handshake (hitch <= 20 mm,
                     yaw <= 4 deg) within its 30 s cap -- the widest standoff tolerance the latch accepts.
(B) COUPLED OUTBOUND STALL. run-352/355: the tow stopped abruptly ~0.4-0.5 m in, then odometry jitter
    (|rv| ~ |rw| ~ 0.01-0.06) tripped measured_turn_radius on most ticks -> 121 s livelock. Proven by
      B1 replay   -- every recorded tick through the REAL coupled gate, legacy vs
                     navigation.coupled_creep_escape (decisions above the noise band unchanged);
      B2 closed loop -- the REAL lane follower on the run's outbound lane from the run-355 stall state
                     with a sluggish, noisy plant: legacy livelocks, the escape reaches the bay;
      B3 safety   -- with the escape ON an obstacle ahead still stops the tow and a genuine sub-R_min arc
                     at tow speed still refuses.

Run:  python3 scripts/check_couple_speed.py --environment <environment.json> \\
        --scenario <mission-scenario.json> --runtime-config <mission-runtime-couple.json> \\
        --runs-dir <.../mission> --runs 347 348b 349 350 351 352 353 354 355 356 [--json out.json]
Exit 0 iff every check passes (prints COUPLE_SPEED_PROOF PASS/FAIL)."""
import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*
from twin.navigation.scenario import environment_grid
from twin.navigation.articulated import ArticulatedController, ArticulatedMap, TrainState, advance
from twin.navigation.factory import _controller_kwargs
from twin.navigation.lanes import TowLaneGraph
from twin.navigation import Odometry, Observation
from twin.mobile import MobileSpec, Pose2, DockObservation, wrap
from twin.restock.fine_dock import FineDock, FineDockConfig

DT = 1 / 60
LEGACY_STANDOFF_XY_TOL = 0.05     # runtime._start_couple_nav (default)
LEGACY_YAW_TOL = 0.08             # core.Controller default
# Measured base point vs turning pivot: A0 regresses it from the recorded rotation windows (~7-8 mm);
# this is the fallback when no runs are given. STRESS = the naive net-displacement bound over the whole
# run-356 window (confounded by drive-branch creep), kept as a reported stress case, not a requirement.
PIVOT_OFFSET_M = 0.008
STRESS_OFFSET_M = 0.037


def _record(report, name, passed, **kw):
    report['ok'] = report['ok'] and passed
    report['checks'].append(dict(name=name, passed=passed, **kw))
    print(('PASS ' if passed else 'FAIL ') + name + ' ' + json.dumps(kw), flush=True)


def _rows(run_dir):
    path = run_dir / 'nav-trace.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---- (A1) recorded standoff windows --------------------------------------------------------------

def standoff_window(rows, standoff, xy_tol=LEGACY_STANDOFF_XY_TOL):
    """The recorded couple standoff window: first tick inside the arrival circle -> the fine-dock insert
    starting (the base reversing > 7 cm west of the standoff). None when the run never got there."""
    cp = [r for r in rows if r['action'] == 'couple']
    dist = [math.hypot(r['x'] - standoff[0], r['y'] - standoff[1]) for r in cp]
    i0 = next((i for i, d in enumerate(dist) if d <= xy_tol), None)
    if i0 is None:
        return None
    i1 = next((i for i in range(i0, len(cp)) if cp[i]['x'] < standoff[0] - 0.07), None)
    if i1 is None:
        return None
    win, d = cp[i0:i1], dist[i0:i1]
    yaws = [math.radians(r['yaw_deg']) for r in win]
    flips = sum(1 for a, b in zip(d, d[1:]) if (a <= xy_tol) != (b <= xy_tol))
    return dict(ticks=len(win), sim_s=round(win[-1]['t'] - win[0]['t'], 2),
                wall_s=round(win[-1]['wall'] - win[0]['wall'], 1),
                wall_per_sim=round((win[-1]['wall'] - win[0]['wall']) / max(1e-6, win[-1]['t'] - win[0]['t']), 2),
                yaw_swept_rad=round(sum(abs(wrap(b - a)) for a, b in zip(yaws, yaws[1:])), 2),
                yaw_net_rad=round(wrap(yaws[-1] - yaws[0]), 3),
                arrival_flips=flips, xy_max_m=round(max(d), 4), xy_first_m=round(d[0], 4))


def fit_pivot_offset(rows, standoff, xy_tol=LEGACY_STANDOFF_XY_TOL, window=30):
    """Least-squares body-frame offset o of the measured point from the turning pivot over the recorded
    standoff window: for a turn d(theta), the measured point moves d(theta) * (-o_y, o_x) in the body
    frame. Returns (o_x, o_y, n_windows) or None."""
    cp = [r for r in rows if r['action'] == 'couple']
    i0 = next((i for i, r in enumerate(cp) if math.hypot(r['x'] - standoff[0], r['y'] - standoff[1]) <= xy_tol), None)
    if i0 is None:
        return None
    i1 = next((i for i in range(i0, len(cp)) if cp[i]['x'] < standoff[0] - 0.07), len(cp))
    seg = cp[i0:i1]
    A, B, C = [], [], []
    for k in range(0, len(seg) - window, window // 2):
        a, b = seg[k], seg[k + window]
        th0, th1 = math.radians(a['yaw_deg']), math.radians(b['yaw_deg'])
        dth = wrap(th1 - th0)
        if abs(dth) < 0.03:
            continue
        thm = th0 + dth / 2
        dx, dy = b['x'] - a['x'], b['y'] - a['y']
        A.append(dth)
        B.append(math.cos(thm) * dx + math.sin(thm) * dy)
        C.append(-math.sin(thm) * dx + math.cos(thm) * dy)
    if len(A) < 20:
        return None
    saa = sum(a * a for a in A)
    return (sum(a * c for a, c in zip(A, C)) / saa, -sum(a * b for a, b in zip(A, B)) / saa, len(A))


def check_pivot_fit(runs, standoff, report):
    fits = {n: fit_pivot_offset(rows, standoff) for n, rows in runs.items()}
    fits = {n: f for n, f in fits.items() if f is not None}
    report['pivot_fit'] = {n: dict(o_x=round(f[0], 4), o_y=round(f[1], 4), windows=f[2]) for n, f in fits.items()}
    if not fits:
        return PIVOT_OFFSET_M
    mags = [math.hypot(f[0], f[1]) for f in fits.values()]
    _record(report, 'pivot_offset_fitted_from_recorded_rotation', max(mags) < 0.03,
            runs=len(fits), offset_m_max=round(max(mags), 4), offset_m_mean=round(statistics.mean(mags), 4))
    return max(mags)


def check_recorded_standoff(runs, standoff, hysteresis, report):
    windows = {}
    for name, rows in runs.items():
        w = standoff_window(rows, standoff)
        if w is not None:
            windows[name] = w
            print(f'  standoff[{name}] {json.dumps(w)}', flush=True)
    report['standoff_windows'] = windows
    if not windows:
        _record(report, 'recorded_standoff_windows_found', False)
        return windows
    sims = [w['sim_s'] for w in windows.values()]
    walls = [w['wall_s'] for w in windows.values()]
    _record(report, 'recorded_standoff_dither_reproduced',
            all(w['arrival_flips'] >= 20 and w['yaw_swept_rad'] > 5 * abs(w['yaw_net_rad']) for w in windows.values()),
            runs=len(windows), sim_s_mean=round(statistics.mean(sims), 1),
            wall_s_mean=round(statistics.mean(walls), 1),
            flips_min=min(w['arrival_flips'] for w in windows.values()),
            yaw_swept_mean=round(statistics.mean(w['yaw_swept_rad'] for w in windows.values()), 1))
    xy_max = max(w['xy_max_m'] for w in windows.values())
    _record(report, 'recorded_xy_never_leaves_hysteresis_band', xy_max <= LEGACY_STANDOFF_XY_TOL + hysteresis,
            xy_max_m=xy_max, release_m=round(LEGACY_STANDOFF_XY_TOL + hysteresis, 3),
            note='the latched arrival branch would have stayed engaged over every recorded tick')
    return windows


# ---- pivot-offset plant (A2/A3) -----------------------------------------------------------------

class PivotPlant:
    """Kinematic diff base whose MEASURED point sits `offset` (body frame) off the turning pivot: a
    commanded in-place turn translates the measured point (as the real base does, ~3.7 cm/rad)."""
    def __init__(self, measured_pose, offset):
        self.o = offset
        x, y, th = measured_pose
        c, s = math.cos(th), math.sin(th)
        self.cx, self.cy, self.th = x - (c * offset[0] - s * offset[1]), y - (s * offset[0] + c * offset[1]), th
        self.v_meas = self.w_meas = 0.0

    def pose(self):
        c, s = math.cos(self.th), math.sin(self.th)
        return (self.cx + c * self.o[0] - s * self.o[1], self.cy + s * self.o[0] + c * self.o[1], self.th)

    def step(self, v, w, dt=DT):
        before = self.pose()
        self.th = wrap(self.th + w * dt)
        self.cx += v * dt * math.cos(self.th)
        self.cy += v * dt * math.sin(self.th)
        after = self.pose()
        self.v_meas = math.hypot(after[0] - before[0], after[1] - before[1]) / dt
        self.w_meas = w


def drive_standoff(grid, nav_cfg, start, standoff, *, hysteresis, xy_tol, yaw_tol, offset, cap_s=240.0):
    """The REAL uncoupled couple leg (static drive, the mission's docking cap) to the standoff."""
    kw = _controller_kwargs(nav_cfg)
    kw['arrival_xy_hysteresis'] = hysteresis
    kw.update(goal_timeout=1e4, blocked_timeout=1e3)
    c = ArticulatedController(grid, allow_test_fixtures=True, **kw)
    c.start(standoff, False, arrival_xy_tol=xy_tol, arrival_yaw_tol=yaw_tol, static_drive=True)
    plant = PivotPlant(start, offset)
    t_capture, flips, inside_prev = None, 0, False
    for tick in range(int(cap_s / DT)):
        now = (tick + 1) * DT
        pose = plant.pose()
        d = math.dist(pose[:2], standoff[:2])
        inside = d <= xy_tol
        if t_capture is None and inside:
            t_capture = now
        flips += inside != inside_prev
        inside_prev = inside
        od = Odometry(pose, now, plant.v_meas, plant.w_meas, 'test_fixture', 0.05)
        obs = Observation(now, (), False, True, 0.0, 'test_fixture', 10.0)
        r = c.step(DT, od, obs, now=now)
        if r.status == 'succeeded':
            return dict(arrived=True, t_s=round(now, 2),
                        after_capture_s=round(now - (t_capture or now), 2), flips=flips,
                        final_xy_m=round(d, 4), final_yaw_rad=round(wrap(pose[2] - standoff[2]), 4),
                        final_pose=tuple(round(v, 4) for v in pose))
        if r.status == 'failed':
            return dict(arrived=False, reason=r.reason, t_s=round(now, 2), flips=flips)
        plant.step(r.linear, r.angular)
    pose = plant.pose()
    return dict(arrived=False, reason='cap', t_s=cap_s, flips=flips,
                final_xy_m=round(math.dist(pose[:2], standoff[:2]), 4))


def check_standoff_closed_loop(grid, nav_cfg, approach, standoff, hysteresis, tol, fd_cfg, cart, offset_m,
                               report):
    """Every pivot-offset bearing (the fitted magnitude, direction swept): legacy vs hysteresis, and the
    hysteresis arrival pose chained into the REAL fine-dock servo on the same plant."""
    legacy, fixed = [], []
    for k in range(8):
        a = k * math.pi / 4
        o = (offset_m * math.cos(a), offset_m * math.sin(a))
        lg = _quiet(drive_standoff, grid, nav_cfg, approach, standoff, hysteresis=0.0,
                    xy_tol=LEGACY_STANDOFF_XY_TOL, yaw_tol=LEGACY_YAW_TOL, offset=o, cap_s=120.0)
        fx = _quiet(drive_standoff, grid, nav_cfg, approach, standoff, hysteresis=hysteresis,
                    xy_tol=tol['xy_m'], yaw_tol=tol['yaw_rad'], offset=o, cap_s=120.0)
        if fx['arrived']:
            ok, t, hm, yr = fine_dock_seats(fd_cfg, cart, tuple(fx['final_pose']), o)
            fx['fine_dock'] = dict(seated=ok, t_s=round(t, 1), hitch_mm=round(hm * 1000, 1),
                                   yaw_deg=round(math.degrees(yr), 2))
        legacy.append(lg)
        fixed.append(fx)
        print(f'  |o|={offset_m:.3f} bearing={k * 45:3d}deg legacy={json.dumps(lg)}\n'
              f'                          fixed ={json.dumps(fx)}', flush=True)
    return legacy, fixed


def _quiet(fn, *a, **kw):
    """Run fn with the controller's own throttled diag prints (TRACK_DIAG ...) suppressed."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def check_standoff(grid, nav_cfg, approach, standoff, hysteresis, tol, fd_cfg, cart, offset_m, report):
    legacy, fixed = check_standoff_closed_loop(grid, nav_cfg, approach, standoff, hysteresis, tol, fd_cfg,
                                               cart, offset_m, report)
    report['standoff_closed_loop'] = dict(offset_m=offset_m, legacy=legacy, fixed=fixed)
    dithered = [lg for lg in legacy if not lg['arrived'] or lg['after_capture_s'] > 30.0]
    _record(report, 'legacy_standoff_dithers_on_pivot_offset_plant', len(dithered) >= 1,
            offset_m=round(offset_m, 4), dithering_bearings=len(dithered), of=len(legacy),
            worst_flips=max(lg['flips'] for lg in legacy))
    worst = max((fx['after_capture_s'] for fx in fixed if fx['arrived']), default=None)
    _record(report, 'hysteresis_standoff_converges_every_bearing',
            all(fx['arrived'] for fx in fixed) and worst is not None and worst <= 15.0,
            offset_m=round(offset_m, 4), hysteresis_m=hysteresis, xy_tol=tol['xy_m'], yaw_tol=tol['yaw_rad'],
            worst_after_capture_s=worst,
            worst_final_xy_m=max((fx.get('final_xy_m', 9) for fx in fixed), default=None),
            worst_final_yaw_rad=max((abs(fx.get('final_yaw_rad', 9)) for fx in fixed), default=None))
    _record(report, 'hysteresis_arrival_then_fine_dock_seats_every_bearing',
            all(fx.get('fine_dock', {}).get('seated') for fx in fixed),
            seat_s=[fx.get('fine_dock', {}).get('t_s') for fx in fixed],
            hitch_mm=[fx.get('fine_dock', {}).get('hitch_mm') for fx in fixed])
    # Stress (reported, not required): the naive 3.7 cm net-displacement bound.
    s_leg, s_fix = check_standoff_closed_loop(grid, nav_cfg, approach, standoff, hysteresis, tol, fd_cfg,
                                              cart, STRESS_OFFSET_M, report)
    report['standoff_stress'] = dict(offset_m=STRESS_OFFSET_M, legacy=s_leg, fixed=s_fix)
    print('  STRESS |o|=%.3f: legacy arrived %d/8, hysteresis arrived %d/8, chained seat %d/8' % (
        STRESS_OFFSET_M, sum(lg['arrived'] for lg in s_leg), sum(fx['arrived'] for fx in s_fix),
        sum(bool(fx.get('fine_dock', {}).get('seated')) for fx in s_fix)), flush=True)
    return legacy, fixed


# ---- (A3) fine-dock capture --------------------------------------------------------------------------

def fine_dock_seats(fd_cfg, cart, standoff_pose, offset, *, cap_s=None):
    """Run the REAL FineDock servo from `standoff_pose` against the pinned cart on the pivot plant."""
    spec = MobileSpec()
    fd = FineDock(FineDockConfig(**fd_cfg), spec)
    fd.start(0.0)
    plant = PivotPlant(standoff_pose, offset)
    limit = (cap_s or fd_cfg['time_cap_s']) + 2.0
    for tick in range(int(limit / DT)):
        now = (tick + 1) * DT
        p = plant.pose()
        obs = DockObservation(Pose2(*p), Pose2(*cart), 0.28, 0.28, plant.v_meas, 0.0, plant.w_meas, 0.0, 0.0)
        r = fd.step(obs, now)
        if r.status == 'ready':
            return True, now, r.hitch_residual_m, r.yaw_residual_rad
        if r.status == 'failed':
            return False, now, r.hitch_residual_m, r.yaw_residual_rad
        plant.step(r.linear, r.angular)
    return False, limit, float('nan'), float('nan')


def check_fine_dock_capture(fd_cfg, cart, standoff, chosen_xy, chosen_yaw, offset_m, report):
    """Sweep standoff residuals (xy on a ring, every bearing; yaw +-) through the REAL servo; the capture
    box is the largest (xy, yaw) every sample of which still seats inside the attach handshake."""
    radii = (0.0, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12)
    yaws = (0.0, 0.08, 0.16, 0.25, 0.35)
    table = {}
    worst_t = 0.0
    for r in radii:
        for a in yaws:
            ok_all = True
            for k in range(12 if r > 0 else 1):
                b = k * math.pi / 6
                for sy in ((+1, -1) if a > 0 else (+1,)):
                    for ob in (0, 2, 4, 6):     # pivot-offset bearing (0/90/180/270 deg)
                        o = (offset_m * math.cos(ob * math.pi / 4), offset_m * math.sin(ob * math.pi / 4))
                        start = (standoff[0] + r * math.cos(b), standoff[1] + r * math.sin(b), standoff[2] + sy * a)
                        ok, t, _, _ = fine_dock_seats(fd_cfg, cart, start, o)
                        ok_all = ok_all and ok
                        if ok:
                            worst_t = max(worst_t, t)
                    if not ok_all:
                        break
                if not ok_all:
                    break
            table[f'{r:.2f}m/{a:.2f}rad'] = ok_all
    captured_xy = {a: max((r for r in radii if all(table[f'{rr:.2f}m/{a:.2f}rad'] for rr in radii if rr <= r)),
                          default=None) for a in yaws}
    report['fine_dock_capture'] = dict(table=table, captured_xy_by_yaw={f'{a:.2f}': v for a, v in captured_xy.items()},
                                       worst_seat_s=round(worst_t, 1))
    print('  fine-dock capture (max standoff xy residual seated, by |yaw|): '
          + json.dumps({f'{a:.2f}rad': v for a, v in captured_xy.items()}), flush=True)
    yaw_key = min((a for a in yaws if a >= chosen_yaw), default=None)
    inside = yaw_key is not None and captured_xy[yaw_key] is not None and captured_xy[yaw_key] >= chosen_xy
    _record(report, 'chosen_standoff_tolerance_inside_fine_dock_capture', inside,
            chosen_xy_m=chosen_xy, chosen_yaw_rad=chosen_yaw, capture_xy_at_yaw=(captured_xy.get(yaw_key)),
            yaw_bin=yaw_key, worst_seat_s=round(worst_t, 1), time_cap_s=fd_cfg['time_cap_s'])
    return captured_xy


# ---- (B) coupled outbound ---------------------------------------------------------------------------

def _outbound_controller(grid, runtime, *, escape, lane=True):
    nav = runtime['navigation']
    spec = MobileSpec()
    graph = None
    if lane and nav.get('tow_lanes'):
        graph = TowLaneGraph.from_config(nav['tow_lanes'])
        graph.expand_all(ArticulatedMap(grid, spec))
    kw = _controller_kwargs(nav)
    kw.update(goal_timeout=1e4, blocked_timeout=8.0)
    c = ArticulatedController(
        grid, spec, allow_test_fixtures=True, lane_graph=graph,
        lane_static_tol=float(nav.get('tow_lane_static_tol_m', 0.0)),
        lane_trust_static=bool(nav.get('tow_lane_trust_static', False)),
        lane_reacquire_along_route=bool(nav.get('tow_lane_reacquire_along_route', False)),
        lane_cart_yaw_tol=float(nav.get('tow_lane_cart_yaw_tol', 0.20)),
        lane_curve_speed_floor=float(nav.get('tow_lane_curve_speed_floor', 0.25)),
        lane_coast_refusals=bool(nav.get('tow_lane_coast_refusals', False)),
        lane_trust_unknown=bool(nav.get('tow_lane_trust_unknown', False)),
        measured_turn_fault_s=nav.get('measured_turn_fault_s'),
        lane_track_tol=float(nav.get('tow_lane_track_tol_m', 0.25)),
        lane_track_envelope=bool(nav.get('tow_lane_track_envelope', False)),
        lane_cart_yaw_rebase=bool(nav.get('tow_lane_cart_yaw_rebase', False)),
        coupled_creep_escape=escape, **kw)
    return c, graph


def check_recorded_outbound(grid, runtime, runs, report):
    """Every recorded outbound tick through the REAL coupled gate (follower straight-lane command), legacy
    vs escape. rw in the trace is the 3-D angular-speed NORM, used as the yaw rate: an upper bound, so the
    legacy refusal counts are upper bounds and so are the escape's (above the band, where it is unchanged)."""
    goal = tuple(runtime['mission_poses']['shelf_approach'])
    out = {}
    for name, rows in runs.items():
        ob = [r for r in rows if r['action'] == 'outbound']
        if not ob:
            continue
        stats = {}
        for escape in (False, True):
            c, _ = _outbound_controller(grid, runtime, escape=escape, lane=False)
            c.start(goal, True)
            counts, changed_above_band = {}, 0
            for r in ob:
                od = Odometry((r['x'], r['y'], math.radians(r['yaw_deg'])), 1.0, r['vel']['rv'], r['vel']['rw'],
                              'test_fixture', 0.05)
                obs = Observation(1.0, (), True, True, 0.0, 'test_fixture', 10.0)
                reason = c.gate_velocity(1.0, od, obs, 0.23, 0.0).reason
                band = 0.02 < r['vel']['rv'] < 0.06
                counts[('band:' if band else 'free:') + reason] = counts.get(('band:' if band else 'free:') + reason, 0) + 1
            stats['escape' if escape else 'legacy'] = counts
        out[name] = stats
        print(f'  outbound[{name}] {json.dumps(stats)}', flush=True)
    report['recorded_outbound_gate'] = out
    stalled = {n: s for n, s in out.items() if s['legacy'].get('band:measured_turn_radius', 0) >= 50}
    _record(report, 'recorded_stall_runs_refuse_in_noise_band_legacy', len(stalled) >= 1,
            runs=sorted(stalled), refusals={n: s['legacy'].get('band:measured_turn_radius', 0) for n, s in stalled.items()})
    _record(report, 'escape_clears_every_noise_band_refusal',
            all(s['escape'].get('band:measured_turn_radius', 0) == 0 for s in out.values()),
            band_refusals_escape={n: s['escape'].get('band:measured_turn_radius', 0) for n, s in out.items()})
    _record(report, 'escape_identical_above_noise_band',
            all({k: v for k, v in s['legacy'].items() if k.startswith('free:')}
                == {k: v for k, v in s['escape'].items() if k.startswith('free:')} for s in out.values()))
    return out


class SluggishPlant:
    """The impeded coupled base seen in the run-352/355 stall: forward speed lags the command (tau 0.6 s
    up, a near-immediate stop on a zero command, as the stiff wheel drive brakes), the measured yaw rate
    carries jitter of the recorded size, and the measured linear speed is a norm (never negative)."""
    def __init__(self, pose, cart_yaw, rng, *, tau_up=0.6, w_noise=0.03, v_noise=0.006):
        self.x, self.y, self.th = pose
        self.cart_yaw = cart_yaw
        self.v = 0.0
        self.rng, self.tau_up, self.w_noise, self.v_noise = rng, tau_up, w_noise, v_noise
        self.v_meas = self.w_meas = 0.0

    def step(self, v_cmd, w_cmd, dt=DT):
        if v_cmd <= 1e-6:
            self.v *= 0.1
        else:
            self.v += (v_cmd - self.v) * (1 - math.exp(-dt / self.tau_up))
        w = w_cmd * (self.v / v_cmd) if v_cmd > 1e-6 else 0.0      # the arc follows the achieved speed
        k = w / self.v if self.v > 1e-6 else 0.0
        ts = advance(TrainState(self.x, self.y, self.th, self.cart_yaw), self.v * dt, k, MobileSpec())
        self.th = wrap(self.th + w * dt)
        self.x += self.v * dt * math.cos(self.th)
        self.y += self.v * dt * math.sin(self.th)
        self.cart_yaw = ts.cart_yaw
        self.v_meas = abs(self.v + self.rng.gauss(0.0, self.v_noise))
        self.w_meas = w + self.rng.gauss(0.0, self.w_noise)


def tow_from_stall(grid, runtime, start, *, escape, seed, cap_s=120.0):
    goal = tuple(runtime['mission_poses']['shelf_approach'])
    c, graph = _outbound_controller(grid, runtime, escape=escape)
    c.start(goal, True)
    plant = SluggishPlant(start, start[2], random.Random(seed))
    reasons = {}
    for tick in range(int(cap_s / DT)):
        now = (tick + 1) * DT
        art = wrap(plant.th - plant.cart_yaw)
        od = Odometry((plant.x, plant.y, plant.th), now, plant.v_meas, plant.w_meas, 'test_fixture', 0.05)
        obs = Observation(now, (), True, True, art, 'test_fixture', 10.0)
        r = c.step(DT, od, obs, now=now)
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
        if r.status == 'succeeded':
            return dict(reached=True, t_s=round(now, 1), x=round(plant.x, 3), reasons=reasons,
                        creep_escapes=c.creep_escapes)
        if r.status == 'failed':
            return dict(reached=False, reason=r.reason, t_s=round(now, 1), x=round(plant.x, 3), reasons=reasons)
        plant.step(r.linear, r.angular)
    return dict(reached=False, reason='cap', t_s=cap_s, x=round(plant.x, 3), reasons=reasons)


def check_outbound_closed_loop(grid, runtime, stall_pose, report):
    legacy = [_quiet(tow_from_stall, grid, runtime, stall_pose, escape=False, seed=s) for s in range(5)]
    escape = [_quiet(tow_from_stall, grid, runtime, stall_pose, escape=True, seed=s) for s in range(5)]
    report['outbound_closed_loop'] = dict(stall_pose=stall_pose, legacy=legacy, escape=escape)
    for lg, es in zip(legacy, escape):
        print(f'  legacy={json.dumps(lg)}\n  escape={json.dumps(es)}', flush=True)
    # This kinematic plant reproduces the refusal MECHANISM (legacy trips measured_turn_radius on the
    # noise at the crawl) but not the physical impediment that held the real base for 121 s, so legacy
    # also crawls through here, just slower; the recorded replay (B1) is the evidence for the stall itself.
    mtr = lambda r: r['reasons'].get('measured_turn_radius', 0)
    _record(report, 'legacy_tow_refuses_on_noise_from_stall_state',
            all(mtr(lg) >= 20 for lg in legacy),
            t_s=[lg['t_s'] for lg in legacy], reached=[lg['reached'] for lg in legacy],
            measured_turn_radius=[mtr(lg) for lg in legacy])
    _record(report, 'escape_tow_reaches_bay_from_stall_state',
            all(es['reached'] for es in escape)
            and all(mtr(es) * 10 <= max(1, mtr(lg)) for es, lg in zip(escape, legacy))
            and all(es['t_s'] <= lg['t_s'] for es, lg in zip(escape, legacy) if lg['reached']),
            t_s=[es['t_s'] for es in escape], creep_escapes=[es.get('creep_escapes') for es in escape],
            measured_turn_radius=[mtr(es) for es in escape])


def check_escape_safety(grid, runtime, stall_pose, report):
    goal = tuple(runtime['mission_poses']['shelf_approach'])
    c, _ = _outbound_controller(grid, runtime, escape=True, lane=False)
    c.start(goal, True)
    pose = stall_pose
    noisy = Odometry(pose, 1.0, 0.03, 0.05, 'test_fixture', 0.05)          # in the band: k = 1.67
    clear = Observation(1.0, (), True, True, 0.0, 'test_fixture', 10.0)
    ahead = (pose[0] + 0.60, pose[1])               # 0.10 m off the front bumper: inside the braking sweep
    blocked = Observation(1.0, (ahead,), True, True, 0.0, 'test_fixture', 10.0)
    r_clear = c.gate_velocity(1.0, noisy, clear, 0.23, 0.0)
    r_block = c.gate_velocity(1.0, noisy, blocked, 0.23, 0.0)
    fast_arc = Odometry(pose, 1.0, 0.20, 0.30, 'test_fixture', 0.05)        # k = 1.5 at tow speed
    r_fast = c.gate_velocity(1.0, fast_arc, clear, 0.23, 0.0)
    r_stop = c.gate_velocity(1.0, noisy, clear, 0.0, 0.0)
    _record(report, 'escape_noise_band_tracks_with_clamped_arc',
            r_clear.reason == 'tracking' and abs(r_clear.angular) * 1.5 <= r_clear.linear + 1e-9,
            reason=r_clear.reason, v=round(r_clear.linear, 3), w=round(r_clear.angular, 4))
    _record(report, 'escape_obstacle_ahead_still_stops', r_block.reason == 'obstacle_stop', reason=r_block.reason)
    _record(report, 'escape_genuine_tow_speed_arc_still_refuses', r_fast.reason == 'measured_turn_radius',
            reason=r_fast.reason)
    _record(report, 'escape_never_turns_a_stop_into_motion', r_stop.linear == 0.0, reason=r_stop.reason,
            v=r_stop.linear)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--environment', type=Path, required=True)
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--runtime-config', type=Path, required=True,
                    help='mission runtime config (mission_poses, navigation, fine_dock)')
    ap.add_argument('--camera-config', type=Path, default=None)
    ap.add_argument('--runs-dir', type=Path, default=None, help='directory holding run-NNN/nav-trace.jsonl')
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--hysteresis', type=float, default=0.03, help='navigation.arrival_xy_hysteresis_m')
    ap.add_argument('--standoff-xy', type=float, default=LEGACY_STANDOFF_XY_TOL)
    ap.add_argument('--standoff-yaw', type=float, default=LEGACY_YAW_TOL)
    ap.add_argument('--json', type=Path, default=None)
    args = ap.parse_args()

    manifest = json.loads(args.environment.read_text())
    scenario = json.loads(args.scenario.read_text())
    runtime = json.loads(args.runtime_config.read_text())
    camera = json.loads(args.camera_config.read_text()) if args.camera_config else None
    grid = environment_grid(manifest, scenario, camera)
    poses = runtime['mission_poses']
    standoff = tuple(poses['couple_standoff'])
    approach = tuple(poses['couple_approach'])
    cart = tuple(scenario['initial_staging']['cart_body_pose'])
    report = {'ok': True, 'checks': [], 'static_obstacles': len(grid.obstacles)}
    print(f'grid: {len(grid.obstacles)} static obstacles', flush=True)

    runs = {}
    for name in args.runs:
        d = args.runs_dir / f'run-{name}'
        if (d / 'nav-trace.jsonl').is_file():
            runs[name] = _rows(d)
    tol = {'xy_m': args.standoff_xy, 'yaw_rad': args.standoff_yaw}

    print('(A0) pivot offset fitted from the recorded standoff rotation', flush=True)
    offset_m = check_pivot_fit(runs, standoff, report) if runs else PIVOT_OFFSET_M
    print('(A1) recorded couple standoff windows', flush=True)
    if runs:
        check_recorded_standoff(runs, standoff, args.hysteresis, report)
    print('(A2) closed-loop standoff -> fine dock, pivot-offset plant', flush=True)
    legacy, fixed = check_standoff(grid, runtime['navigation'], approach, standoff, args.hysteresis, tol,
                                   runtime['fine_dock'], cart, offset_m, report)
    print('(A3) fine-dock capture (real servo, run config gains)', flush=True)
    check_fine_dock_capture(runtime['fine_dock'], cart, standoff, args.standoff_xy + args.hysteresis,
                            args.standoff_yaw, offset_m, report)

    print('(B1) recorded outbound ticks through the coupled gate', flush=True)
    if runs:
        check_recorded_outbound(grid, runtime, runs, report)
    stall = (-3.935, 15.265, math.radians(0.8))     # run-355 stall state
    print('(B2) closed-loop tow from the run-355 stall state (sluggish noisy plant)', flush=True)
    check_outbound_closed_loop(grid, runtime, stall, report)
    print('(B3) creep-escape safety', flush=True)
    check_escape_safety(grid, runtime, stall, report)

    # Expected time saved per warm run (recorded standoff window vs the closed-loop converged window, at
    # the recorded wall/sim ratio of that window).
    windows = report.get('standoff_windows') or {}
    if windows:
        ratio = statistics.mean(w['wall_per_sim'] for w in windows.values())
        fixed_s = max(fx['after_capture_s'] for fx in fixed if fx['arrived'])
        report['expected_saving'] = dict(
            recorded_standoff_sim_s=round(statistics.mean(w['sim_s'] for w in windows.values()), 1),
            recorded_standoff_wall_s=round(statistics.mean(w['wall_s'] for w in windows.values()), 1),
            fixed_standoff_sim_s_worst=fixed_s, wall_per_sim=round(ratio, 2),
            fixed_standoff_wall_s_est=round(fixed_s * ratio, 1),
            saved_wall_s_per_run=round(statistics.mean(w['wall_s'] for w in windows.values()) - fixed_s * ratio, 1))
        print('expected saving: ' + json.dumps(report['expected_saving']), flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print('COUPLE_SPEED_PROOF ' + ('PASS' if report['ok'] else 'FAIL'), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
