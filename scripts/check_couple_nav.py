#!/usr/bin/env python3
"""Offline proof: the COUPLE nav legs route to the on-axis standoff from BOTH possible start poses
(the shortcut start at the pickup/home pose, and place_approach where a real collect leaves the base),
once the couple legs are driven as TRUSTED-STATIC drives (RuntimeAdapter couple_static_drive=True).

Why this must be proven offline (run-319b, start_at=couple):
  The couple approach waypoint sits at y=16.0; the backstock staging support /RestockScenario/Support
  (scenario staging, south face y=16.86) is the only obstacle beside that corridor. Its TRUE AABB
  clears the 0.75 m planner footprint (+0.05 m segment padding) by 0.11 m -- the route IS geometrically
  free against the static map. But the LIVE lidar returns of that known-static support sit a few cm
  proud of the true surface (run-319b NAV_DIAG near5=0.782 m vs the 0.86 m true face; RANGE_DIAG names
  every near return /RestockScenario/Support), which erases the thin margin and makes the reference
  planner refuse the straight leg from the shortcut start pose (directly SOUTH of the support):
      NAV_DIAG blocked_no_route pose=(-5.74,16.0) goal=(-4.46,16.0) r=0.750 seg_free=False near5=0.782
  A real collect leaves the base at place_approach (EAST of the support), from which the leg never grazes
  the support and routes either way. The honest fix for the shortcut start is the same TRUSTED-STATIC
  drive the mid-collect reposition legs already use (fm396): route the couple legs against the static map
  (true support AABB) instead of the few-cm-proud live cloud. The static map still carries the support,
  so a genuine overlap is still refused -- this does not mask the obstacle.

This proof drives the REAL reference controller (twin.navigation.core.Controller) and the REAL planner
(GridMap.plan / segment_free) -- it is not a fixture of the decision. The live support cloud is modelled
at the run-319b-MEASURED distance (0.782 m), labelled, not invented.

Run:  python3 scripts/check_couple_nav.py \
         --environment <environment.json> --scenario <mission-scenario.json>
Exit 0 iff: (A) the live cloud blocks the shortcut-start leg and clears it under static drive, AND
            (B) the full couple sequence (approach->standoff) reaches the standoff from BOTH start poses.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*
from twin.navigation.scenario import environment_grid
from twin.navigation.core import Controller
from twin.navigation.sensors import Scan, observation
from twin.navigation import Odometry
from twin.mobile import MobileSpec
from twin.navigation.articulated import Footprint


# run-319b MEASURED sensed-vs-true gap of the static support returns (NAV_DIAG near5 0.782 m vs the
# 0.86 m true south face from the shortcut-start pose). The live cloud is the known-static support
# surface, displaced toward the sensor by this bias.
SUPPORT_PROUD_M = 0.078


def support_cloud(scenario, base_xy, *, proud=SUPPORT_PROUD_M, step=0.04):
    """The sensed /RestockScenario/Support returns: the AABB faces visible from `base_xy`, each sample
    pulled `proud` metres toward the base to model the measured few-cm sensing bias."""
    st = scenario['initial_staging']
    cx, cy = st['support_center_xy']
    dx, dy, _ = st['support_dimensions_m']
    x0, x1, y0, y1 = cx - dx / 2, cx + dx / 2, cy - dy / 2, cy + dy / 2
    pts = []
    nx = max(1, int(dx / step))
    ny = max(1, int(dy / step))
    # south + north faces
    for i in range(nx + 1):
        x = x0 + (x1 - x0) * i / nx
        for y in (y0, y1):
            pts.append((x, y))
    # east + west faces
    for j in range(ny + 1):
        y = y0 + (y1 - y0) * j / ny
        for x in (x0, x1):
            pts.append((x, y))
    bx, by = base_xy
    out = []
    for px, py in pts:
        d = math.hypot(px - bx, py - by)
        if d < 1e-6:
            continue
        out.append((px - (px - bx) / d * proud, py - (py - by) / d * proud))
    return out


def make_controller(grid):
    spec = MobileSpec()
    fp = Footprint(spec.transport_radius, abs(spec.hitch_x),
                   math.hypot(spec.cart_hitch_x + spec.cart_length / 2, spec.cart_width / 2),
                   margin=0.05, min_turn_radius=0.0, max_articulation=spec.max_articulation)
    return Controller(grid, fp, allow_test_fixtures=True, goal_timeout=120.0)


RADIUS = 0.70 + 0.05   # footprint.radius(False) + position_uncertainty (== run-319b NAV_DIAG r=0.750)


def straight_leg_free(grid, start, goal, cloud, *, static_drive):
    """The EXACT NAV_DIAG seg_free signal (core.Controller.advance L516): is the designed straight
    couple leg clear at the footprint? Live obstacles when not static, static map only when static."""
    dyn = () if static_drive else tuple(cloud)
    return grid.segment_free(start, goal, RADIUS, dyn)


def plan_leg(grid, start, goal, cloud, *, static_drive):
    """The full planner decision (GridMap.plan): straight if free, else A* detour."""
    dyn = () if static_drive else tuple(cloud)
    return grid.plan(start, goal, RADIUS, dyn)


def drive_to(controller, grid, start, goal, cloud, *, static_drive, arrival_xy_tol=None):
    """Quasi-statically advance the REAL controller to a goal (test-fixture provenance). Returns
    (reached, ticks, final_pose). Uses the real gate/planner; static drives carry no live obstacles."""
    controller.start(goal, False, static_drive=static_drive,
                     **({'arrival_xy_tol': arrival_xy_tol} if arrival_xy_tol else {}))
    pose = tuple(start)
    v = w = 0.0
    for tick in range(4000):
        now = (tick + 1) * 0.05
        odom = Odometry(pose, now, abs(v), w, 'test_fixture', 0.05)
        # Build the observation with the support cloud as map obstacles. A static drive ignores them
        # (empty live set); a live drive sees them. Coverage is full (the open backroom is observed).
        obs = _observation(now, pose, cloud, static_drive)
        r = controller.step(0.05, odom, obs, now=now)
        v, w = r.linear, r.angular
        if r.status == 'succeeded':
            return True, tick, pose
        if r.status == 'failed':
            return False, tick, pose, r.reason
        # integrate the differential command
        th = pose[2] + w * 0.05
        pose = (pose[0] + v * 0.05 * math.cos(th), pose[1] + v * 0.05 * math.sin(th), th)
    return False, 4000, pose, 'timeout'


def _observation(now, pose, cloud, static_drive):
    """A fixture Observation carrying the support cloud and a FULL coverage field (open backroom)."""
    from twin.navigation import Observation
    # Live drives see the cloud; static drives are given empty live obstacles (the controller also
    # empties them internally, this keeps the fixture honest about what the gate would see).
    obstacles = () if static_drive else tuple(cloud)
    return Observation(now, obstacles, False, True, 0.0, 'test_fixture', 10.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--environment', type=Path, required=True)
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--couple-approach', type=float, nargs=3, default=(-4.46, 16.0, 0.0))
    ap.add_argument('--couple-standoff', type=float, nargs=3, default=(-4.06, 15.25, 0.0))
    ap.add_argument('--shortcut-start', type=float, nargs=3, default=(-5.74, 16.0, 0.0),
                    help='run-319b shortcut start (pickup/home pose, directly south of the support)')
    ap.add_argument('--place-approach', type=float, nargs=3, default=(-4.5, 15.4, math.pi),
                    help='where a real position-first collect leaves the base (east of the support)')
    args = ap.parse_args()

    manifest = json.loads(args.environment.read_text())
    scenario = json.loads(args.scenario.read_text())
    grid = environment_grid(manifest, scenario)       # NO camera mounts: the mission nav grid
    approach = tuple(args.couple_approach)
    standoff = tuple(args.couple_standoff)

    ok = True
    report = {'support_proud_m': SUPPORT_PROUD_M, 'checks': []}

    def record(name, passed, **kw):
        nonlocal ok
        ok = ok and passed
        report['checks'].append(dict(name=name, passed=passed, **kw))
        print(('PASS ' if passed else 'FAIL ') + name + ' ' + json.dumps(kw), flush=True)

    # (A) The NAV_DIAG seg_free decision at the shortcut start: the live cloud blocks the designed
    # straight couple leg (seg_free=False, as run-319b), the static drive clears it (seg_free=True).
    cloud = support_cloud(scenario, args.shortcut_start[:2])
    near = min((math.hypot(args.shortcut_start[0] - q[0], args.shortcut_start[1] - q[1]) for q in cloud),
               default=math.inf)
    live_free = straight_leg_free(grid, args.shortcut_start, approach, cloud, static_drive=False)
    stat_free = straight_leg_free(grid, args.shortcut_start, approach, cloud, static_drive=True)
    record('shortcut_live_cloud_blocks_straight_leg', not live_free, nearest_sensed_m=round(near, 3))
    record('shortcut_static_drive_clears_straight_leg', stat_free)

    # The leg from place_approach is clear EITHER way (east of the support): seg_free with the live cloud.
    cloud_pa = support_cloud(scenario, args.place_approach[:2])
    pa_free = straight_leg_free(grid, args.place_approach, approach, cloud_pa, static_drive=False)
    record('place_approach_live_cloud_clears_straight_leg', pa_free)

    # (B) Full couple sequence (approach -> standoff) reaches the standoff from BOTH start poses,
    # driven as trusted-static (the couple_static_drive path).
    for label, start in (('shortcut_start', args.shortcut_start),
                         ('place_approach', args.place_approach)):
        c = make_controller(grid)
        r1 = drive_to(c, grid, start, approach, support_cloud(scenario, start[:2]),
                      static_drive=True)
        reached1 = r1[0]
        c2 = make_controller(grid)
        r2 = drive_to(c2, grid, approach, standoff, support_cloud(scenario, approach[:2]),
                      static_drive=True, arrival_xy_tol=0.05)
        reached2 = r2[0]
        record(f'couple_sequence_reaches_standoff[{label}]', reached1 and reached2,
               approach_ticks=r1[1], standoff_ticks=r2[1],
               approach_detail=(r1[3] if len(r1) > 3 else 'ok'),
               standoff_detail=(r2[3] if len(r2) > 3 else 'ok'))

    # (C) Blocker 2: the COUPLED train routes dock -> shelf_approach via the CHEAP corridor route (the
    # sales_backroom_portal hints), NOT the expensive full A* search that timed out under load in the
    # fullscope nav05 proof. Prove at both the floor (0.05) and conservative (0.10) localisation
    # allowances that the plan resolves fast via `validated_corridor`.
    _blocker2_coupled_outbound(grid, scenario, manifest, record)

    print('COUPLE_NAV_PROOF ' + ('PASS' if ok else 'FAIL'), flush=True)
    return 0 if ok else 1


def _blocker2_coupled_outbound(grid, scenario, manifest, record):
    """Coupled dock->shelf_approach must resolve via the cheap corridor route within a tight budget."""
    import time as _time
    from twin.navigation.articulated import ArticulatedController, TrainState
    st = scenario['initial_staging']
    dock = st.get('robot_base_pose', manifest['poses'].get('robot_dock'))
    shelf = st.get('shelf_robot_pose', manifest['poses'].get('shelf_approach'))
    cart_yaw = st.get('cart_body_pose', [0, 0, 0])[2]
    c = ArticulatedController(grid, allow_test_fixtures=True)
    spec = c.spec
    start = TrainState(dock[0], dock[1], dock[2], cart_yaw)
    for unc in (0.05, 0.10):
        c.planner.seeds = []                      # cold cache: worst case, like the first plan under load
        t0 = _time.monotonic()
        route = c.planner.plan(start, shelf, (), unc, spec.max_articulation)
        dt = _time.monotonic() - t0
        # Cheap path = a route found without touching the expensive full search (expanded==0 for the
        # corridor route). A plan that resolves in well under a second cannot be starved by host load.
        cheap = bool(route.states) and getattr(route, 'expanded', 0) == 0 and dt < 2.0
        record(f'coupled_outbound_routes_cheaply[unc={unc}]', cheap,
               reason=route.reason, states=len(route.states),
               expanded=getattr(route, 'expanded', None), cpu_s=round(dt, 3))


if __name__ == '__main__':
    raise SystemExit(main())
