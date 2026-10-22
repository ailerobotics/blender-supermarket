#!/usr/bin/env python3
"""Offline (CPU-only) decision evidence for ``navigation.lidar_sensor_yaw_fix`` (vr1), grounded in a
recorded COMPLETE run's drive poses (default: the twin-robot-in-store-boot-v1 warm8 baseline).

THE BUG. ``twin.navigation.sensors.Scan.sensor_to_base`` is ``(x, y, YAW)`` -- the sensor origin x/y in
base_link plus the sensor's yaw offset about base_link. ``isaac_bundle`` (legacy wiring) passes the lidar
URDF ``origin_xyz`` ``(0.52, 0.0, 0.15)`` there, so the sensor's Z (0.15 m height) is read as a 0.15 rad
(8.59 deg) YAW. The rays are CAST at the true base yaw (``read()``), but ``observation()`` reconstructs
every return through ``Scan.points`` -> ``sensor_pose`` = ``yaw + da``, i.e. rotated 8.59 deg off its cast
bearing. The lidar_frame is a fixed rpy-0 child of base_link, so the correct yaw offset is EXACTLY 0;
``lidar_sensor_yaw_fix: true`` passes ``(x, y, 0.0)``.

WHY IT MATTERS. A flat surface in front of the robot reconstructs as a surface rotated 8.59 deg about the
sensor: its near end swings TOWARD the robot by r*sin(8.59deg) ~ 0.15*r. A wall 3 m ahead reads as points
spread over x 2.25..3.70 -- a phantom return 0.75 m PROUD of the true surface, straight into the forward
braking capsule and the tow-lane swept envelope. This is the documented "lidar-proud phantom" the tow
lanes paper over with ``tow_lane_static_tol_m`` (proud tolerance), ``tow_lane_trust_static`` and
``tow_lane_trust_unknown``.

WHAT THIS PROVES (all on CPU, no GPU, no Isaac/USD):
  (1) the rotation + its proud depth at several ranges (synthetic wall, the vr1 wiring verbatim);
  (2) replays the recorded run's COUPLED-tow drive poses (couple/outbound/return) through the SAME
      production ``IsaacRangeSensor`` + ``observation`` wiring over a box ray-caster of the mission's own
      static map, legacy vs fixed, and quantifies: how many returns land > the proud tolerance from any
      mapped AABB (the "kept" phantom class that forces trust_static), the min clearance each wiring
      leaves along the two authored tow lanes, and how many poses the fixed vs legacy cloud trips the
      forward braking capsule (``obstacle_stop`` -- the capsule-vs-planner deadlock cases).

The box map is the SAME AABB set the live nav grid gates against, so under the correct (fixed) wiring every
return lands on its source AABB (near_static, discounted even at tol 0); the legacy rotation is what pushes
them off into open space. Exit 0 iff the fix strictly reduces every phantom metric. Run:

  python3 scripts/check_lidar_yaw_fix.py \
    --environment <environment.json> --scenario <mission-scenario.json> \
    --camera-config <camera-config.json> --lanes <mission-runtime-couple.json> \
    --nav-trace <warm8/nav-trace.jsonl>
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*

from twin.navigation.scenario import environment_grid
from twin.navigation.core import Controller, Footprint, GridMap
from twin.navigation.articulated import ArticulatedMap, ArticulatedController
from twin.navigation.lanes import TowLaneGraph
from twin.navigation.sensors import IsaacRangeSensor, observation as nav_observation
from twin.navigation import Odometry, Observation
from twin.mobile import MobileSpec
from twin.restock.tail_nav import BoxRaycaster, LIDAR_XYZ, lidar_wall_probe

# The orchestrator's sim-clock nav bound used live (run-356): a 3.0 s stale window makes the forward
# braking capsule ~ speed*3.0 + speed^2/(2*0.4) long -- ~0.83 m at the 0.25 m/s cruise. The capsule-
# vs-planner deadlock the waivers work around lives at this length, so probe the gate with it.
STALE_AFTER_S = 3.0
PROBE_SPEED = 0.25
COUPLED_ACTIONS = ('couple', 'outbound', 'return')
PROUD_TOL_M = 0.12          # navigation.tow_lane_static_tol_m in warm8's config -- the proud tolerance
LIDAR_YAW_RAD = LIDAR_XYZ[2]


def _record(report, name, passed, **kw):
    report['ok'] = report['ok'] and passed
    report['checks'].append(dict(name=name, passed=passed, **kw))
    print(('PASS ' if passed else 'FAIL ') + name + ' ' + json.dumps(kw), flush=True)


def wall_smear_proof(report):
    """(1) The rotation + proud depth, verbatim vr1 wiring, on a synthetic wall at several ranges."""
    print('--- (1) rotation / proud-depth proof (synthetic wall) ---', flush=True)
    _record(report, 'sensor_yaw_offset_is_the_urdf_z',
            abs(LIDAR_YAW_RAD - 0.15) < 1e-9,
            urdf_z_m=LIDAR_XYZ[2], read_as_yaw_rad=round(LIDAR_YAW_RAD, 4),
            read_as_yaw_deg=round(math.degrees(LIDAR_YAW_RAD), 2))
    worst = 0.0
    for wx in (1.0, 2.0, 3.0, 5.0):
        legacy = lidar_wall_probe(LIDAR_XYZ, wall_x=wx)
        fixed = lidar_wall_probe((LIDAR_XYZ[0], LIDAR_XYZ[1], 0.0), wall_x=wx)
        proud = round(wx - legacy[0], 3)              # how far the near edge swings toward the robot
        worst = max(worst, proud)
        _record(report, f'wall_{wx:g}m_fixed_reproduces_exactly', fixed == (wx, wx), fixed_span=fixed)
        _record(report, f'wall_{wx:g}m_legacy_reads_proud', proud > 0.3,
                legacy_span=legacy, proud_depth_m=proud)
    report['worst_proud_depth_m'] = round(worst, 3)
    print(f'worst legacy proud depth across probed walls: {worst:.3f} m', flush=True)


def _box_map_from_grid(grid, z=(0.0, 2.0)):
    """Lift the nav grid's 2-D AABBs to 3-D boxes the ray-caster can hit, so a correctly-wired return lands
    on the SAME AABB ``near_static`` measures against (fixed wiring => distance ~0, discounted at any tol)."""
    return [(xmin, ymin, z[0], xmax, ymax, z[1], f'/static/{i}')
            for i, (xmin, ymin, xmax, ymax) in enumerate(grid.obstacles)]


def _sample_poses(nav_trace, actions, stride):
    poses = []
    with open(nav_trace) as f:
        for i, line in enumerate(f):
            if i % stride:
                continue
            d = json.loads(line)
            if d.get('action') in actions:
                poses.append((d['x'], d['y'], math.radians(d.get('yaw_deg', 0.0)), d.get('action')))
    return poses


def _reconstruct(sensor, pose, da):
    """Read the scan at `pose` (rays cast at the true base yaw) and reconstruct its world obstacle points
    through sensor_to_base yaw `da` -- the ONLY thing the fix changes. Returns the production Observation
    (its .obstacles cloud + its coverage field, both consumed by the braking-capsule gate)."""
    sensor.sensor_to_base = (sensor.sensor_to_base[0], sensor.sensor_to_base[1], da)
    scan = sensor.read(1.0)
    od = Odometry(pose[:3], 1.0, 0.0, 0.0, 'wheel_odometry_localized', 0.05)
    return nav_observation(scan, od, coupled=False, arm_transport=True)


def replay_run(grid, lanes_cfg, nav_trace, report, *, stride, beams):
    """(2) Replay the recorded coupled-tow poses, legacy vs fixed, over the mission's own static map."""
    print(f'--- (2) recorded-run replay: {Path(nav_trace).parent.name} coupled tow, stride {stride} ---',
          flush=True)
    spec = MobileSpec()
    amap = ArticulatedMap(grid, spec)
    boxes = _box_map_from_grid(grid)
    poses = _sample_poses(nav_trace, COUPLED_ACTIONS, stride)
    report['replay'] = {'poses': len(poses), 'beams': beams, 'stale_after_s': STALE_AFTER_S}
    if not poses:
        _record(report, 'replay_has_poses', False, note='no coupled-tow poses sampled')
        return

    # Lane centrelines (expanded against the static map), for the clearance metric.
    graph = TowLaneGraph.from_config(lanes_cfg['navigation']['tow_lanes'])
    graph.expand_all(amap)
    lane_pts = {ln.name: [(s.x, s.y) for s in ln.route.states] for ln in graph.lanes}

    # An EMPTY grid isolates the LIVE reconstructed cloud: the capsule gate then trips only on the
    # reconstructed returns (not the static map itself), so legacy-vs-fixed counts the PHANTOM intrusion.
    empty = GridMap((-20, -20, 20, 20), (), 0.2)
    fp = Footprint(spec.transport_radius, abs(spec.hitch_x),
                   math.hypot(spec.cart_hitch_x + spec.cart_length / 2, spec.cart_width / 2),
                   margin=0.05, min_turn_radius=0.0, max_articulation=spec.max_articulation)

    state = {'pose': None}
    sensor = IsaacRangeSensor(lambda: (state['pose'][0] + LIDAR_XYZ[0] * math.cos(state['pose'][2]),
                                       state['pose'][1] + LIDAR_XYZ[0] * math.sin(state['pose'][2]),
                                       state['pose'][2]),
                              sensor_to_base=LIDAR_XYZ, self_paths=('/World/Robot',),
                              query=BoxRaycaster(boxes), beams=beams)

    agg = {}
    for tag, da in (('legacy', LIDAR_YAW_RAD), ('fixed', 0.0)):
        kept = 0            # returns > PROUD_TOL_M from every mapped AABB (the trust_static-forcing class)
        total = 0
        stops = 0           # poses whose reconstructed cloud trips the forward braking capsule
        lane_clear = {name: float('inf') for name in lane_pts}
        for x, y, yaw, _action in poses:
            state['pose'] = (x, y, yaw)
            ob = _reconstruct(sensor, (x, y, yaw), da)
            cloud = ob.obstacles
            total += len(cloud)
            for px, py in cloud:
                if not amap.near_static(px, py, PROUD_TOL_M):
                    kept += 1
                for name, pts in lane_pts.items():
                    for lx, ly in pts:
                        d = math.hypot(px - lx, py - ly)
                        if d < lane_clear[name]:
                            lane_clear[name] = d
            # forward braking capsule probe at the cruise speed, reconstructed returns as the live cloud
            c = Controller(empty, fp, allow_test_fixtures=True, directional_zone=True,
                           stale_after=STALE_AFTER_S)
            c.start((x + math.cos(yaw), y + math.sin(yaw), yaw), False)
            od = Odometry((x, y, yaw), 1.0, PROBE_SPEED, 0.0, 'wheel_odometry_localized', 0.05)
            if c.gate_velocity(1.0, od, ob, PROBE_SPEED, 0.0).reason == 'obstacle_stop':
                stops += 1
        agg[tag] = dict(total_returns=total, kept_beyond_tol=kept,
                        capsule_stops=stops,
                        lane_min_clearance_m={k: round(v, 3) for k, v in lane_clear.items()})
        print(f'  {tag}: returns={total} kept>{PROUD_TOL_M}m={kept} capsule_stops={stops}/{len(poses)} '
              f'lane_clear={agg[tag]["lane_min_clearance_m"]}', flush=True)

    report['replay'].update(agg)
    L, F = agg['legacy'], agg['fixed']
    _record(report, 'fix_reduces_kept_phantom_returns', F['kept_beyond_tol'] < L['kept_beyond_tol'],
            legacy_kept=L['kept_beyond_tol'], fixed_kept=F['kept_beyond_tol'],
            reduction_pct=round(100 * (1 - F['kept_beyond_tol'] / max(1, L['kept_beyond_tol'])), 1))
    _record(report, 'fix_reduces_capsule_deadlock_cases', F['capsule_stops'] < L['capsule_stops'],
            legacy_stops=L['capsule_stops'], fixed_stops=F['capsule_stops'], poses=len(poses))
    for name in lane_pts:
        _record(report, f'fix_opens_lane_clearance[{name}]',
                F['lane_min_clearance_m'][name] > L['lane_min_clearance_m'][name],
                legacy_min_clearance_m=L['lane_min_clearance_m'][name],
                fixed_min_clearance_m=F['lane_min_clearance_m'][name])


def check_waivers(grid, lanes_cfg, nav_trace, report, *, stride):
    """(3) WAIVER DECISION, through the REAL ArticulatedController on warm8's own outbound lane: replay the
    recorded poses' real reconstructed clouds and show that ``tow_lane_trust_static`` was compensating for
    the yaw rotation -- the LEGACY cloud blocks the lane plan (``lane_blocked``) and needs trust_static to
    reopen it, while the FIXED cloud VALIDATES the lane with trust_static OFF (and the modest proud tol),
    yet a genuinely NEW obstacle on the lane still blocks it (the gate is not merely dropped)."""
    print('--- (3) waiver decision: real controller, legacy needs trust_static, fixed does not ---',
          flush=True)
    spec = MobileSpec()
    amap = ArticulatedMap(grid, spec)
    boxes = _box_map_from_grid(grid)
    graph = TowLaneGraph.from_config(lanes_cfg['navigation']['tow_lanes'])
    graph.expand_all(amap)
    lane = next((ln for ln in graph.lanes if 'outbound' in ln.name), graph.lanes[0])
    st = lane.start_state()
    tol = float(lanes_cfg['navigation'].get('tow_lane_static_tol_m', 0.12))

    state = {'p': None}
    sensor = IsaacRangeSensor(lambda: (state['p'][0] + LIDAR_XYZ[0] * math.cos(state['p'][2]),
                                       state['p'][1] + LIDAR_XYZ[0] * math.sin(state['p'][2]),
                                       state['p'][2]),
                              sensor_to_base=LIDAR_XYZ, self_paths=('/World/Robot',),
                              query=BoxRaycaster(boxes), beams=180)

    def cloud(pose, da):
        state['p'] = pose
        sensor.sensor_to_base = (LIDAR_XYZ[0], LIDAR_XYZ[1], da)
        od = Odometry(pose, 1.0, 0.0, 0.0, 'wheel_odometry_localized', 0.05)
        return nav_observation(sensor.read(1.0), od, coupled=False, arm_transport=True).obstacles

    def plan(pts, **kw):
        c = ArticulatedController(grid, spec, allow_test_fixtures=True, lane_graph=graph, **kw)
        c.start(lane.goal, True)
        obs = Observation(0.1, tuple(pts), True, True, st.articulation, 'test_fixture', 10.0)
        return c.plan_observed_route(st, obs, 0.05).reason

    poses = [(x, y, yaw) for (x, y, yaw, _a) in _sample_poses(nav_trace, ('outbound', 'return'), stride)]
    legacy_block = [p for p in poses if plan(cloud(p, LIDAR_YAW_RAD), lane_static_tol=tol) != 'validated_lane']
    report['waivers'] = {'poses': len(poses), 'legacy_blocked_poses': len(legacy_block),
                         'proud_tol_m': tol}
    print(f'  outbound/return poses: {len(poses)}; LEGACY cloud blocks the lane plan '
          f'(trust_static OFF) at {len(legacy_block)}', flush=True)
    if not legacy_block:
        _record(report, 'legacy_reproduces_lane_blocked', False,
                note='no legacy pose reproduced lane_blocked at this stride -- increase --stride density')
        return
    worst = legacy_block[0]
    _record(report, 'legacy_cloud_blocks_lane_without_trust_static',
            plan(cloud(worst, LIDAR_YAW_RAD), lane_static_tol=tol) == 'lane_blocked',
            pose=[round(c, 2) for c in worst])
    _record(report, 'legacy_cloud_needs_trust_static_to_reopen',
            plan(cloud(worst, LIDAR_YAW_RAD), lane_static_tol=tol, lane_trust_static=True) == 'validated_lane',
            pose=[round(c, 2) for c in worst])
    fixed_still_block = [p for p in legacy_block
                         if plan(cloud(p, 0.0), lane_static_tol=tol) != 'validated_lane']
    _record(report, 'fix_validates_all_blocked_lanes_without_trust_static',
            not fixed_still_block, legacy_blocked=len(legacy_block),
            fixed_still_blocked=len(fixed_still_block))
    # SAFETY: with the fix and trust_static OFF, a genuinely new obstacle on the lane centre still blocks.
    mid = lane.route.states[len(lane.route.states) // 2]
    _record(report, 'fix_new_obstacle_on_lane_still_blocks',
            plan(tuple(cloud(worst, 0.0)) + ((mid.x, mid.y),), lane_static_tol=tol) == 'lane_blocked',
            new_obstacle=[round(mid.x, 2), round(mid.y, 2)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--environment', type=Path, required=True)
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--camera-config', type=Path, required=True)
    ap.add_argument('--lanes', type=Path, required=True,
                    help='mission config/fragment carrying navigation.tow_lanes')
    ap.add_argument('--nav-trace', type=Path, required=True,
                    help="a complete run's nav-trace.jsonl (e.g. warm8/nav-trace.jsonl)")
    ap.add_argument('--stride', type=int, default=40, help='sample every Nth trace tick (default 40)')
    ap.add_argument('--beams', type=int, default=180)
    ap.add_argument('--json', type=Path, default=None, help='write the full report here')
    args = ap.parse_args()

    manifest = json.loads(args.environment.read_text())
    scenario = json.loads(args.scenario.read_text())
    camera = json.loads(args.camera_config.read_text())
    lanes_cfg = json.loads(args.lanes.read_text())
    grid = environment_grid(manifest, scenario, camera)

    report = {'ok': True, 'checks': [], 'static_obstacles': len(grid.obstacles),
              'nav_trace': str(args.nav_trace)}
    print(f'grid: {len(grid.obstacles)} static obstacles, bounds {grid.bounds}', flush=True)
    wall_smear_proof(report)
    replay_run(grid, lanes_cfg, args.nav_trace, report, stride=args.stride, beams=args.beams)
    check_waivers(grid, lanes_cfg, args.nav_trace, report, stride=max(4, args.stride // 2))
    print('LIDAR_YAW_FIX_PROOF ' + ('PASS' if report['ok'] else 'FAIL'), flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
