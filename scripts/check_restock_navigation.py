#!/usr/bin/env python3
"""CPU-only actual-map planning/control evidence. All feedback is TEST_FIXTURE.

This is not a simulator demonstration, localization test, or dynamics test.
It proves geometry and controller feasibility at explicit full margins against
the supplied environment's conservative obstacle manifest.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from twin.navigation import GridMap,Odometry,Observation
from twin.navigation.articulated import ArticulatedController,TrainState,advance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenario',type=Path,help='Integration scenario: staging support and actual chassis goals')
    args=parser.parse_args()
    manifest=json.loads(args.environment.read_text())
    scenario=json.loads(args.scenario.read_text()) if args.scenario else None
    from twin.navigation.scenario import environment_grid
    grid=environment_grid(manifest,scenario)
    poses=dict(manifest['poses'])
    if scenario:
        staging=scenario['initial_staging']
        poses['robot_dock']=staging['robot_base_pose']
        poses['cart_dock']=staging['cart_body_pose']
        poses['shelf_approach']=staging['shelf_robot_pose']
    controller=ArticulatedController(grid,allow_test_fixtures=True,goal_timeout=250)
    state=TrainState(*poses['robot_dock'],poses['cart_dock'][2])
    report={'source':'test_fixture_quasistatic_not_physx',
        'environment_sha256':hashlib.sha256(args.environment.read_bytes()).hexdigest(),
        'obstacles':len(grid.obstacles),'geometry_margin':.05,'localization_allowance':.02,
        'minimum_turn_radius':1.5,'max_articulation_degrees':65,'legs':[]}
    report['scenario_sha256']=hashlib.sha256(args.scenario.read_bytes()).hexdigest() if args.scenario else None
    report['goals']=poses
    args.output.mkdir(parents=True,exist_ok=True)
    v=w=0.0
    for name,goal in [('outbound',poses['shelf_approach']),
                      ('return',poses['robot_dock'])]:
        articulation_limit = (scenario or {}).get('navigation_goal_articulation_rad', {}).get(name)
        controller.start(goal,True,goal_articulation=articulation_limit)
        trace=[];began=time.monotonic();peak_angle=0.0
        for tick in range(5200):
            now=(tick+1)*.05
            odom=Odometry(state.pose,now,v,w,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            result=controller.step(.05,odom,obs,now=now)
            v,w=result.velocity
            if v<0 or abs(w)*1.5>v+1e-9:
                raise AssertionError('illegal towing command')
            next_state=advance(state,v*.05,w/v if v else 0,controller.spec)
            if not controller.world.valid(next_state,extra_margin=.02):
                raise AssertionError('feedback trajectory collision')
            state=next_state
            peak_angle=max(peak_angle,abs(state.articulation))
            trace.append(dict(time=now,state=vars(state),linear=v,angular=w,
                              status=result.status,reason=result.reason))
            if result.status!='running':
                break
        robot,cart=state.bodies(controller.spec)
        leg=dict(cart_body_pose=vars(cart),robot_body_pose=vars(robot),name=name,status=result.status,reason=result.reason,
                 goal_articulation_limit_rad=articulation_limit,
                 goal=goal,final=vars(state),simulation_seconds=now,
                 cpu_seconds=time.monotonic()-began,peak_articulation_degrees=math.degrees(peak_angle),
                 final_articulation_degrees=math.degrees(state.articulation))
        report['legs'].append(leg)
        (args.output/(name+'-feedback.json')).write_text(json.dumps(trace))
        if controller.route and controller.route.states:
            (args.output/(name+'-route.json')).write_text(json.dumps(dict(
                source='validated_planning_hint_not_feedback',
                states=[vars(s) for s in controller.route.states],
                curvatures=controller.route.curvatures,goal=goal)))
        print(json.dumps(leg),flush=True)
        if result.status!='succeeded':
            break
    report['passed']=len(report['legs'])==2 and all(l['status']=='succeeded' for l in report['legs'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
