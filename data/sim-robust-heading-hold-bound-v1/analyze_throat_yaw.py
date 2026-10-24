#!/usr/bin/env python3
"""CPU reconstruction of the coldwin17 coupled-outbound throat-yaw growth.

Rebuilds the REAL authored outbound lane (outbound_dock_to_bay) and replays the recorded
coldwin17 / coldwin16 / coldwin11 approach creeps through the REAL ArticulatedController
coupled follower + PR 253 heading hold (as coldwin17 ran: hold ON). For each follow tick it
reports the quantity the hold actually bounds (base_yaw - goal_heading), the local authored
SEGMENT axis (ref_yaw) the follower steers to, the signed lateral offset, the follower's two
curvature terms, whether the hold fired, and the commanded (v, w) vs the recorded cw.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -I \
    data/sim-robust-heading-hold-bound-v1/analyze_throat_yaw.py
"""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from twin.mobile import MobileSpec
from twin.navigation import GridMap, Odometry, Observation
from twin.navigation.articulated import ArticulatedController, ArticulatedMap, TrainState, advance
from twin.navigation.lanes import TowLaneGraph
from twin.navigation.core import wrap

MISSION = Path('/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission')
THROAT_X = -3.875
# The real authored outbound lane (PR 253 report §1; from mission-runtime-capstone241.json).
LANE_CFG = [{'name': 'outbound_dock_to_bay', 'start': [-4.46, 15.5, 0.0], 'goal': [-2.0, 15.25, 0.0],
             'start_cart_yaw': 0.0, 'goal_articulation': 0.3, 'start_tol_cart_yaw': 0.35,
             'lead_in_max_m': 3.0}]


def load_creep(run):
    rows = []
    p = MISSION / run / 'nav-trace.jsonl'
    for ln in p.open():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        rows.append(r)
    cr = [r for r in rows if r.get('coupled') and -4.75 <= r['x'] <= -3.70 and r['y'] > 14.5]
    # Keep only the ADVANCING approach: from the first creep row until x first stalls at its
    # max (the pinned tail that follows adds thousands of identical rows). Cut at the first row
    # whose x is within 2 mm of the running max AND the next 20 rows never advance beyond it.
    if not cr:
        return cr
    out = []
    xmax = -1e9
    stall = 0
    for i, r in enumerate(cr):
        out.append(r)
        if r['x'] > xmax - 1e-3:
            xmax = max(xmax, r['x'])
            stall = 0
        else:
            stall += 1
        if stall >= 40:            # 40 consecutive non-advancing rows => entered the pin
            break
    return out


def build_controller(**kw):
    spec = MobileSpec()
    world = GridMap((-20, 5, 10, 25))
    graph = TowLaneGraph.from_config(LANE_CFG)
    graph.expand_all(ArticulatedMap(world, spec))
    params = dict(allow_test_fixtures=True, lane_graph=graph, tow_lane_goal_stop=True,
                  lane_trust_unknown=True, goal_timeout=900.0, blocked_timeout=120.0)
    params.update(kw)
    c = ArticulatedController(world, spec, **params)
    c.start((-2.0, 15.25, 0.0), True)
    return c, spec


def reach_tracking(c, spec):
    state = TrainState(-4.46, 15.5, 0.0, 0.0)
    v = w = 0.0
    now = 0.0
    for i in range(600):
        now = (i + 1) * .05
        od = Odometry(state.pose, now, v, w, 'test_fixture', .02)
        obs = Observation(now, (), True, True, 0.0, 'test_fixture', 10)
        r = c.step(.05, od, obs, now=now)
        v, w = r.velocity
        if c.route is not None and c.route.states and r.reason == 'tracking':
            return now
        state = advance(state, v * .05, w / v if v else 0.0, spec)
    raise AssertionError('never reached tracking')


def dump_lane_axis(c):
    """Print the authored route ref_yaw + curvature through the throat band."""
    st = c.route.states
    cv = c.route.curvatures
    print('  authored route states near throat (x, y, ref_yaw_deg, curvature):')
    for i in range(len(st) - 1):
        if -4.6 <= st[i].x <= -3.7:
            print('    x=%.3f y=%.3f ref_yaw=%+6.2f  curv=%+.4f' %
                  (st[i].x, st[i].y, math.degrees(st[i].yaw), cv[i]))


def nearest_segment(route, state):
    best = None
    for j in range(len(route.states) - 1):
        a, b = route.states[j:j + 2]
        dx, dy = b.x - a.x, b.y - a.y
        t = max(0.0, min(1.0, ((state[0] - a.x) * dx + (state[1] - a.y) * dy) / max(1e-9, dx * dx + dy * dy)))
        x, y = a.x + t * dx, a.y + t * dy
        d = math.hypot(state[0] - x, state[1] - y)
        if best is None or d < best[0]:
            best = (d, j, t, x, y)
    _, j, t, x, y = best
    a, b = route.states[j:j + 2]
    ref_yaw = wrap(a.yaw + t * wrap(b.yaw - a.yaw))
    return j, ref_yaw, x, y, route.curvatures[j]


def replay(run, every=2):
    rows = load_creep(run)
    c, spec = build_controller(coupled_heading_hold=True, heading_hold_bound_deg=1.0,
                               heading_hold_creep_mps=0.14, coupled_progress_escape=True,
                               coupled_progress_recovery=True, coupled_breakaway_rock=True,
                               coupled_breakaway_launch_mps=0.30)
    now = reach_tracking(c, spec)
    goal_yaw = c.goal[2]
    print('\n==== %s ==== goal_heading=%.2f deg, %d creep rows' % (run, math.degrees(goal_yaw), len(rows)))
    dump_lane_axis(c)
    print('  tick replay (every %d rows): x  base_yaw  ae(vs_goal)  ref_yaw(seg)  '
          'yaw_vs_seg  lateral  head_term  lat_term  curv  HOLD  cmd_w  rec_cw' % every)
    prev = None
    holds0 = 0
    n = 0
    first_hold_x = None
    cw_pos = cw_neg = 0
    for row in rows:
        if prev is not None:
            now += max(1e-3, row['world_t'] - prev['world_t'])
        prev = row
        yaw = math.radians(row['yaw_deg'])
        rec_cw = row.get('vel', {}).get('cw', float('nan'))
        od = Odometry((row['x'], row['y'], yaw), now, row['lin'], row.get('vel', {}).get('rw', 0.0),
                      'test_fixture', .02)
        obs = Observation(now, (), True, True, 0.0, 'test_fixture', 10)
        # compute follower terms against the controller's own route BEFORE stepping
        terms = None
        if c.route is not None and c.route.states:
            j, ref_yaw, px, py, curv_j = nearest_segment(c.route, (row['x'], row['y']))
            lateral = -math.sin(ref_yaw) * (px - row['x']) + math.cos(ref_yaw) * (py - row['y'])
            head_term = 1.5 * wrap(ref_yaw - yaw)
            lat_term = 2 * lateral
            follower_curv = curv_j + head_term + lat_term
            terms = (ref_yaw, lateral, head_term, lat_term, follower_curv, curv_j)
        res = c.step(.05, od, obs, now=now)
        fired = c.heading_holds > holds0
        holds0 = c.heading_holds
        if fired and first_hold_x is None:
            first_hold_x = row['x']
        cmd_w = res.velocity[1]
        if res.reason == 'tracking':
            if cmd_w > 1e-9:
                cw_pos += 1
            elif cmd_w < -1e-9:
                cw_neg += 1
        if n % every == 0 and terms is not None:
            ref_yaw, lateral, head_term, lat_term, follower_curv, curv_j = terms
            ae = math.degrees(wrap(yaw - goal_yaw))
            print('    %.3f  %+5.1f  %+6.2f  %+6.2f  %+6.2f  %+.3f  %+.3f  %+.3f  %+.3f  %s  %+.4f  %+.4f' %
                  (row['x'], row['yaw_deg'], ae, math.degrees(ref_yaw),
                   math.degrees(wrap(yaw - ref_yaw)), lateral, head_term, lat_term,
                   follower_curv, 'YES' if fired else ' . ', cmd_w, rec_cw))
        n += 1
    print('  SUMMARY %s: heading_holds=%d breakaway_recoveries=%d successes=%d first_hold_x=%s'
          % (run, c.heading_holds, c.breakaway_recoveries, c.breakaway_successes, first_hold_x))
    print('  tracking ticks commanding CCW(+w)=%d  CW(-w)=%d' % (cw_pos, cw_neg))


def raw_creep(run, lo=-4.35, hi=-3.95):
    rows = []
    for ln in (MISSION / run / 'nav-trace.jsonl').open():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        rows.append(r)
    # the FIRST contiguous advancing approach in the band (drop the pinned tail / post-throat)
    band = [r for r in rows if r.get('coupled') and lo <= r['x'] <= hi and r['y'] > 14.5]
    out = []
    for r in band:
        out.append(r)
        if len(out) > 5 and r['x'] <= out[-2]['x'] - 0.0 and abs(r['x'] - out[-2]['x']) < 1e-4:
            pass
    return band


def approach_only(run):
    """Approach-window reproduction (x in [-4.35,-3.95], before any pin): apply the REAL follower
    curvature LAW (articulated.py: curv = authored + 1.5*(ref_yaw-yaw) + 2*lateral) and the PR 253
    hold CONDITION to the recorded poses, and compare the follower's commanded turn SIGN to the
    recorded cw. This isolates the drive direction from the measured-curvature gate's coast."""
    rows = raw_creep(run)
    c, spec = build_controller(coupled_heading_hold=True, heading_hold_bound_deg=1.0,
                               heading_hold_creep_mps=0.14)
    reach_tracking(c, spec)
    route = c.route
    goal_yaw = c.goal[2]
    follower_ccw = follower_cw = 0
    hold_would_fire = 0
    recorded_ccw = 0
    yaw0 = yawN = None
    lat0 = latN = None
    for row in rows:
        if yaw0 is None:
            yaw0 = row['yaw_deg']
        yawN = row['yaw_deg']
        if row.get('vel', {}).get('cw', 0) > 1e-4:
            recorded_ccw += 1
        yaw = math.radians(row['yaw_deg'])
        j, ref_yaw, px, py, curv_j = nearest_segment(route, (row['x'], row['y']))
        lateral = -math.sin(ref_yaw) * (px - row['x']) + math.cos(ref_yaw) * (py - row['y'])
        if lat0 is None:
            lat0 = lateral
        latN = lateral
        follower_curv = curv_j + 1.5 * wrap(ref_yaw - yaw) + 2 * lateral
        if follower_curv > 1e-6:
            follower_ccw += 1
        elif follower_curv < -1e-6:
            follower_cw += 1
        # PR 253 hold condition: straight segment AND yaw CCW of goal beyond bound
        if abs(curv_j) <= c.heading_hold_straight_kappa and wrap(yaw - goal_yaw) > c.heading_hold_bound:
            hold_would_fire += 1
    print('  APPROACH[-4.35..-3.95] %s: rows=%d  yaw %.1f->%.1f deg  lateral %+.3f->%+.3f m\n'
          '      recorded cw>0 (CCW) ticks=%d/%d  | follower-law turn: CCW=%d CW=%d'
          '  | PR253 hold-condition would fire=%d'
          % (run, len(rows), yaw0 or 0, yawN or 0, lat0 or 0, latN or 0,
             recorded_ccw, len(rows), follower_ccw, follower_cw, hold_would_fire))


if __name__ == '__main__':
    print('==== APPROACH-WINDOW SUMMARY (PR 253 hold ON, real descending lane) ====')
    for run in ('coldwin17', 'coldwin16', 'coldwin11'):
        try:
            approach_only(run)
        except FileNotFoundError:
            print(run, 'NO TRACE')
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        for run in ('coldwin17', 'coldwin16', 'coldwin11'):
            try:
                replay(run)
            except FileNotFoundError:
                print(run, 'NO TRACE')
