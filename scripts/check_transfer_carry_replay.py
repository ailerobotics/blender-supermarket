#!/usr/bin/env python3
"""CPU replay of a recorded transfer-carry failure against RESTOCK_TRANSFER_CARRY=wrist_guard (no GPU, no ROS).

Reads ONLY a finished run's artifacts -- ``<run>/host-stack.log`` (TRAJ_HOST_PLAN / TRAJ_HOST_TRACK_MISS /
TRAJ_HOST_FAULT / HOST_EFFECTIVE_CONFIG), ``<run>.console.log`` (SIM_ATTACH_FALLBACK attach/release),
``<run>/nav-trace.jsonl`` (carton world pose, RESTOCK_TRACE_POSES=1) and the run's own imported URDF -- and
answers, for the move that tripped the trajectory host:

1. was the carton held (weld) or lost before the fault?
2. which joints pinned, and where did the planned reference leave the clear carry wrist region?
3. would the wrist guard have rejected that IK goal BEFORE any motion (the guard's own predicate)?
4. does the SAME tool pose solve in the clear branch from the wrist-flipped re-seed (the guard's re-seed;
   6-DoF DLS IK on the mission URDF stands in for MoveIt KDL), and does the joint path to it stay clear?

    python3 scripts/check_transfer_carry_replay.py ~/datasets/twin-robot-in-store-boot-v1/mission/warm2

Exit 0 when the replay shows the guard prevents the recorded jam, 1 otherwise. ``--json`` prints the
verdict dict. Evidence class: CPU replay of recorded telemetry + CPU IK -- not a physics demonstration.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.restock_manipulation.transfer_carry import (  # noqa: E402
    DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN, in_clear_region, wrap_angle, wrist_flip_seed)

ARM = ('ur_shoulder_pan_joint', 'ur_shoulder_lift_joint', 'ur_elbow_joint',
       'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint')
PLAN = re.compile(r'TRAJ_HOST_PLAN names=(\[.*?\]) npoints=\d+ goal_pos_tol=[\d.]+ points=(.*)$')
MISS = re.compile(r'TRAJ_HOST_TRACK_MISS err=([\d.]+) .*?worst_j=(\d) .*?meas_q=\[([^\]]*)\] ref_q=\[([^\]]*)\]')
FAULT = re.compile(r'TRAJ_HOST_FAULT (.*)$')
CLOCK = re.compile(r'CLOCK_DIAG .*?stage_time=([\d.]+)')


def _floats(text):
    return [float(v) for v in re.split(r'[,\s]+', text.strip()) if v]


def parse_host(lines):
    """(plans, misses, fault_line_index, effective_config) from host-stack.log lines. Each plan is
    dict(line, names, points=[(t, q_arm)]) with q reordered to ARM order."""
    plans, misses, fault_at, cfg = [], [], None, {}
    for i, line in enumerate(lines):
        m = PLAN.search(line)
        if m:
            names = ast.literal_eval(m.group(1))
            order = [names.index(j) for j in ARM]
            pts = [(float(t), [float(q[k]) for k in order]) for t, q in ast.literal_eval(m.group(2))]
            plans.append(dict(line=i, names=names, points=pts))
            continue
        m = MISS.search(line)
        if m:
            misses.append(dict(line=i, err=float(m.group(1)), worst=int(m.group(2)),
                               meas=_floats(m.group(3)), ref=_floats(m.group(4))))
            continue
        if fault_at is None and FAULT.search(line):
            fault_at = i
        if 'HOST_EFFECTIVE_CONFIG ' in line and not cfg:
            try:
                cfg = json.loads(line.split('HOST_EFFECTIVE_CONFIG ', 1)[1])
            except ValueError:
                cfg = {}
    return plans, misses, fault_at, cfg


def parse_console(lines):
    """[(stage_time, 'attach'|'release', line)] SIM_ATTACH_FALLBACK weld events, timed by CLOCK_DIAG."""
    now, events = None, []
    for line in lines:
        m = CLOCK.search(line)
        if m:
            now = float(m.group(1))
        elif 'SIM_ATTACH_FALLBACK attach ' in line:
            events.append((now, 'attach', line.strip()))
        elif 'SIM_ATTACH_FALLBACK release ' in line:
            events.append((now, 'release', line.strip()))
    return events


def carton_track(trace_lines, t_from, t_to):
    """Carton world-pose samples [(t, x, y, z)] in [t_from, t_to] from nav-trace (RESTOCK_TRACE_POSES=1)."""
    out = []
    for line in trace_lines:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        c, t = d.get('carton'), d.get('t')
        if c and t is not None and t_from <= t <= t_to:
            out.append((float(t), float(c[0]), float(c[1]), float(c[2])))
    return out


def first_exit(points, cap, w2min):
    """(t, q) of the first planned reference point outside the clear region, else None."""
    for t, q in points:
        if not in_clear_region(q, cap, w2min):
            return t, q
    return None


def first_crossing(points, joint, value):
    """t at which the planned reference of ``joint`` first passes ``value`` (linear interp), else None."""
    for (t0, q0), (t1, q1) in zip(points, points[1:]):
        a, b = q0[joint] - value, q1[joint] - value
        if a == 0:
            return t0
        if a * b < 0:
            return t0 + (t1 - t0) * a / (a - b)
    return None


def replay(run_dir, console=None, urdf=None):
    run_dir = Path(run_dir)
    console = Path(console) if console else run_dir.parent / f'{run_dir.name}.console.log'
    urdf = Path(urdf) if urdf else run_dir / 'mobile-import.urdf'
    host = (run_dir / 'host-stack.log').read_text(errors='replace').splitlines()
    plans, misses, fault_at, cfg = parse_host(host)
    if fault_at is None:
        return dict(verdict='no TRAJ_HOST_FAULT in host-stack.log -- nothing to replay', prevented=False)
    faulting = [p for p in plans if p['line'] < fault_at][-1]
    pts = faulting['points']
    start, goal = pts[0][1], pts[-1][1]
    cap = cfg.get('carry_wrist1_cap')
    cap = DEFAULT_WRIST1_CAP if cap is None else float(cap)
    w2min = cfg.get('carry_wrist2_min')
    w2min = DEFAULT_WRIST2_MIN if w2min is None else float(w2min)
    last_miss = [m for m in misses if m['line'] < fault_at][-1] if misses else None
    out = dict(run=str(run_dir), guard=dict(wrist1_cap=cap, wrist2_min=w2min),
               fault=FAULT.search(host[fault_at]).group(1),
               faulting_plan=dict(duration_s=pts[-1][0], npoints=len(pts),
                                  start=[round(v, 4) for v in start], goal=[round(v, 4) for v in goal]))
    if last_miss:
        pinned = [ARM[j] for j in range(6) if abs(last_miss['ref'][j] - last_miss['meas'][j]) > 0.2]
        out['pinned'] = dict(joints=pinned, measured=[round(v, 4) for v in last_miss['meas']],
                             reference=[round(v, 4) for v in last_miss['ref']], err=last_miss['err'])
        stall = last_miss['meas'][3]
        t_cross = first_crossing(pts, 3, stall)
        out['pinned']['wrist_1_stall'] = round(stall, 4)
        out['pinned']['reference_reached_stall_at_s'] = None if t_cross is None else round(t_cross, 2)
    ex = first_exit(pts, cap, w2min)
    out['reference_left_clear_region_at_s'] = None if ex is None else round(ex[0], 2)
    out['guard_on_recorded_goal'] = ('REJECT (re-seed)' if not in_clear_region(goal, cap, w2min)
                                     else 'accept')
    out['start_in_clear_region'] = in_clear_region(start, cap, w2min)
    # 4. same tool pose, clear branch, from the guard's own wrist-flipped re-seed.
    from twin.restock.scripted_transfer import ArmIK
    ik = ArmIK(str(urdf))
    T = ik.fk(goal)
    seed = wrist_flip_seed(goal)
    q, pe, re_ = ik.solve(T, seed, iters=800)
    q = [float(v) for v in q]
    clear_path = all(in_clear_region([a + (b - a) * k / 50. for a, b in zip(start, q)], cap, w2min)
                     for k in range(51))
    out['flip_reseed'] = dict(seed=[round(v, 4) for v in seed], solution=[round(v, 4) for v in q],
                              tcp_position_err_m=round(pe, 6), tcp_rotation_err_rad=round(re_, 6),
                              tcp=[round(v, 4) for v in T[:3, 3]],
                              in_clear_region=in_clear_region(q, cap, w2min),
                              joint_linear_path_clear=clear_path,
                              wrist=[round(wrap_angle(q[3]), 4), round(wrap_angle(q[4]), 4)])
    # 1. the carton: weld attach/release + world-pose continuity up to the fault.
    events = parse_console(console.read_text(errors='replace').splitlines()) if console.is_file() else []
    attaches = [e for e in events if e[1] == 'attach']
    releases = [e for e in events if e[1] == 'release']
    carton = {}
    if attaches:
        t_attach = attaches[-1][0] or 0.0
        trace = run_dir / 'nav-trace.jsonl'
        track = carton_track(trace.read_text().splitlines(), t_attach, 1e9) if trace.is_file() else []
        rel_after = [e for e in releases if (e[0] or 0) >= t_attach]
        carton = dict(weld_attach_t=t_attach, weld_released=bool(rel_after),
                      weld_release_t=rel_after[0][0] if rel_after else None)
        if len(track) > 2:
            rates = [abs(b[3] - a[3]) / max(b[0] - a[0], 1e-6) for a, b in zip(track, track[1:])]
            carton.update(samples=len(track), z_start=round(track[0][3], 3), z_end=round(track[-1][3], 3),
                          z_max=round(max(s[3] for s in track), 3),
                          max_abs_dz_dt_m_s=round(max(rates), 3), end_xy=[round(track[-1][1], 3),
                                                                       round(track[-1][2], 3)])
            # a drop from the tool is free fall: >=1 m/s within ~0.15 s; carried motion stays << 1 m/s.
            carton['held_continuously'] = (not rel_after) and max(rates) < 1.0
    out['carton'] = carton
    prevented = (out['guard_on_recorded_goal'].startswith('REJECT') and out['start_in_clear_region']
                 and out['flip_reseed']['in_clear_region'] and out['flip_reseed']['joint_linear_path_clear']
                 and pe < 1e-3 and re_ < 1e-2)
    out['prevented'] = bool(prevented)
    out['verdict'] = (
        'wrist_guard PREVENTS the recorded jam: the faulting goal is outside the clear wrist region (rejected '
        'at IK, before motion) and the same tool pose solves in the clear branch from the wrist-flipped '
        're-seed with a clear joint path' if prevented else
        'wrist_guard does NOT explain/prevent this fault -- inspect the fields')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--console', help='default: <run_dir>.console.log beside the run dir')
    ap.add_argument('--urdf', help='default: <run_dir>/mobile-import.urdf (the run\'s own imported URDF)')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)
    out = replay(args.run_dir, args.console, args.urdf)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for key, value in out.items():
            print(f'{key}: {value}')
    return 0 if out.get('prevented') else 1


if __name__ == '__main__':
    sys.exit(main())
