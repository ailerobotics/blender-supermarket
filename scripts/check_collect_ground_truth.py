#!/usr/bin/env python3
"""Replay the GROUND-TRUTH COLLECT capture (RESTOCK_COLLECT_GROUND_TRUTH) against RECORDED mission runs.

CPU only -- no Isaac, no ROS, no GPU. For each run directory it opens the run's OWN recorded stage
(``run-*/mission.usda``, usd-core, read-only) with its ``assembly.json`` and the ``--scenario`` /
``--runtime-config`` its console log was launched with, then drives the production
``isaac_factory.CollectTargetBridge`` ground-truth capture on it:

* the run's recorded camera-collect outcome (COLLECT_CAPTURE target poses vs provider errors), the
  host ``no initial sensor frame`` failure and the mission.json reason -- what the switch works around;
* the pick Observation the bridge serves parked at pickup (``source='ground_truth'``) and, where the
  run's camera DID localise the carton (e.g. run-326), the camera-vs-ground-truth pose delta;
* the warm-slot place target re-registered with the base at ``mission_poses.place_approach`` (robot
  root moved in the in-memory SESSION layer; nothing is written anywhere);
* with the host's side-grip ``WallCartonPolicy`` (repo ros_host_config.json, shelfbot root from the
  runtime config): every pick/place station pose against the controller's ``max_reach`` gate, i.e.
  the observe gate at pickup and fm390's ``_reregister_place`` gate at place_approach.

    PYTHONPATH=scripts .venv-usd/bin/python scripts/check_collect_ground_truth.py RUN_DIR [RUN_DIR ...]

Exit 0 when every run yields a ground-truth pick and slot target whose station poses pass the reach
gate. A kinematic + reach replay, not a physics or collision-planning proof.
"""
import argparse
import ast
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CAPTURE_POSE = re.compile(r'COLLECT_CAPTURE calls=\d+ sku=\S+ .*?pose=\(([^)]*)\)')
CAPTURE_ERROR = re.compile(r'COLLECT_CAPTURE calls=\d+ provider_error=([^\n]*)')


def launch_args(console):
    """--scenario / --runtime-config of the recorded launch (the kit 'Passing the following args' line)."""
    for line in console.splitlines():
        if 'Passing the following args to the base kit application:' in line:
            argv = ast.literal_eval(line.split(':', 1)[1].strip())
            return {k: argv[argv.index(k) + 1] for k in ('--scenario', '--runtime-config') if k in argv}
    raise ValueError('console log has no kit launch-args line')


def signature(run, console):
    """What the recorded run did at collect (camera path) and how it ended."""
    poses = [tuple(float(v) for v in m.group(1).split(',')) for m in CAPTURE_POSE.finditer(console)]
    errors = {}
    for m in CAPTURE_ERROR.finditer(console):
        errors[m.group(1).strip()] = errors.get(m.group(1).strip(), 0) + 1
    manip = run.parent / (run.name + '-manip.txt')
    host = manip.read_text(errors='replace') if manip.exists() else ''
    mission = json.loads((run / 'mission.json').read_text()) if (run / 'mission.json').exists() else {}
    # COLLECT_CAPTURE error/None lines are rate-limited (every 10th call): counts are samples, not totals.
    return {'camera_target_frames': len(poses), 'first_camera_pose': poses[0] if poses else None,
            'camera_provider_errors': errors,
            'camera_target_none_logged': len(re.findall(r'COLLECT_CAPTURE calls=\d+ target=None', console)),
            'host_no_initial_sensor_frame': 'no initial sensor frame' in host,
            'mission': '%s: %s' % (mission.get('state'), str(mission.get('reason'))[:110])}


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw(q):
    return 2.0 * math.atan2(q[5], q[6])


def _bridge(stage, assembly, params, slot, *, held):
    from twin.restock_manipulation.isaac_factory import CollectTargetBridge
    cart = assembly['composition']['assets']['cart']
    paths = assembly['paths']
    b = CollectTargetBridge(
        stage=stage, arm_camera=None, arm_camera_path=None, base_path=paths['ur_base_link'],
        profiles=params.profiles, mobile_base_path=paths.get('base_link', paths['ur_base_link']),
        motion=NS(parked=lambda: True, collision_time=1.0),
        contact=NS(read=lambda: held, path='/RestockPanelGripper'), joints=lambda: (0.,) * 6,
        source_location=params.collect_source_location, cart_placement=params.cart_placement,
        cart_root=cart['root'], verify_place=True, gt_place_slot=slot)
    b._reproject = lambda target: None      # the image ROI needs the live arm camera; not replayed
    return b


def _move_robot(stage, root, pose):
    """Put the robot root at map pose (x, y, yaw) in the SESSION layer (in memory, never saved)."""
    from pxr import Gf, Usd, UsdGeom
    prim = stage.GetPrimAtPath(root)
    ops = {o.GetOpName(): o for o in UsdGeom.Xformable(prim).GetOrderedXformOps()}
    if set(ops) != {'xformOp:translate', 'xformOp:rotateZ'}:
        raise ValueError('robot root %s xformOps %s: expected translate + rotateZ' % (root, sorted(ops)))
    z = ops['xformOp:translate'].Get()[2]
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        ops['xformOp:translate'].Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(z)))
        ops['xformOp:rotateZ'].Set(math.degrees(float(pose[2])))


def replay(run, host_config=None):
    from pxr import Usd
    from twin.restock_manipulation import ground_truth_collect as gtc
    from twin.restock_manipulation.isaac_factory import parse_manipulation_config
    os.environ[gtc.ENV] = '1'               # this process only: the bridge reads the switch at init
    run = Path(run)
    console = (run.parent / (run.name + '.console.log')).read_text(errors='replace')
    args = launch_args(console)
    config = json.loads(Path(args['--runtime-config']).read_text())
    scenario = json.loads(Path(args['--scenario']).read_text())
    assembly = json.loads((run / 'assembly.json').read_text())
    stage = Usd.Stage.Open(str(run / 'mission.usda'))
    sku = stage.GetPrimAtPath('/RestockScenario/HandlingUnit').GetCustomDataByKey('restock:sku')
    params = parse_manipulation_config(config, sku)
    slot = gtc.warm_slot(scenario)          # what isaac_factory.create() loads when the switch is on
    out = {'run': run.name, 'recorded': signature(run, console), 'sku': sku, 'slot': slot}

    pick = _bridge(stage, assembly, params, slot, held=False).capture(
        1.0, params.collect_source_location, params.cart_placement['location'])
    if pick is None or len(pick.observations) != 1 or pick.target is None:
        out['verdict'] = 'FAIL: no ground-truth pick packet at pickup'
        return out
    obs = pick.observations[0]
    gt_station = _bridge(stage, assembly, params, slot, held=False)._ground_truth_target(1.0).pose_station
    out['pick'] = {'source': obs.source, 'pose_base': tuple(round(v, 4) for v in obs.pose),
                   'pose_station': tuple(round(v, 4) for v in gt_station)}
    cam = out['recorded']['first_camera_pose']
    if cam:
        out['camera_vs_ground_truth'] = {
            'dxyz_m': tuple(round(a - b, 4) for a, b in zip(gt_station[:3], cam[:3])),
            'dist_m': round(math.dist(gt_station[:3], cam[:3]), 4),
            'dyaw_deg': round(math.degrees(_wrap(_yaw(gt_station) - _yaw(cam))), 2)}

    _move_robot(stage, assembly['composition']['assets']['robot_root'], config['mission_poses']['place_approach'])
    held = _bridge(stage, assembly, params, slot, held=True).capture(
        9.0, params.collect_source_location, params.cart_placement['location'])
    if held is None or held.observations or held.target is None:
        out['verdict'] = 'FAIL: no slot place target at place_approach'
        return out
    target = held.target
    out['place_target'] = {'source': target.source, 'location': target.location,
                           'pose_base': tuple(round(v, 4) for v in target.pose),
                           'yaw_base_deg': round(math.degrees(_wrap(_yaw(target.pose))), 1)}

    shelfbot = Path(config.get('shelfbot_root') or config['manipulation'].get('shelfbot_root', ''))
    if not shelfbot.is_dir():
        out['verdict'] = 'PASS (targets only; shelfbot root %s absent, policy reach not replayed)' % shelfbot
        return out
    from twin.restock_manipulation.cad_tool import inspect_tool
    from twin.restock_manipulation.types import ProductProfile
    from twin.restock_manipulation.wall_policy import WallCartonPolicy
    hcfg = json.loads(Path(host_config or ROOT / 'twin/restock_manipulation/ros_host_config.json').read_text())
    profile = {p['sku']: ProductProfile(**p) for p in hcfg['profiles']}[sku]
    mobile = json.loads(Path(config['mobile_manifest']).read_text())
    policy = WallCartonPolicy(str(shelfbot), inspect_tool(mobile['mobile_urdf']), walls=hcfg['walls'])
    poses = policy.build(obs, target, profile)   # pick side == observe build; place side == _reregister_place
    reach = {k: round(math.dist(v[:3], (0, 0, 0)), 3) for k, v in poses.items()}
    out['policy'] = {'max_reach': profile.max_reach, 'reach_m': reach,
                     'contact': tuple(round(v, 4) for v in poses['contact'][:3]),
                     'place': tuple(round(v, 4) for v in poses['place'][:3])}
    bad = sorted(k for k, r in reach.items() if r > profile.max_reach)
    out['verdict'] = 'FAIL: outside max_reach %s' % bad if bad else 'PASS'
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('runs', nargs='+', help='recorded mission run directories (run-*/ with mission.usda)')
    ap.add_argument('--host-config', help='host manipulation config (default: the repo ros_host_config.json)')
    a = ap.parse_args(argv)
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):   # the bridge's COLLECT_GT log lines; stdout = JSON only
        results = [replay(r, a.host_config) for r in a.runs]
    print(json.dumps(results, indent=1))
    return 0 if all(r['verdict'].startswith('PASS') for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
