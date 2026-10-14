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
from twin.mobile import Pose2,occupied_envelope,wrap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenario',type=Path,help='Integration scenario: staging support and actual chassis goals')
    parser.add_argument('--camera-config',type=Path,help='Physical camera config: include fixed mount colliders')
    parser.add_argument('--position-uncertainty','--localization-allowance',type=float,default=.10,
                        help='Fixture bound from localization evidence (default: MapLocalizer floor)')
    parser.add_argument('--goal-articulation',type=float,nargs=2,metavar=('OUTBOUND','RETURN'),
                        help='Per-leg cart articulation tolerance in radians (mission runtime enforces '
                             'the same limit at the parked destination). Omitted: the MobileSpec max (65 deg).')
    args=parser.parse_args()
    if not math.isfinite(args.position_uncertainty) or args.position_uncertainty<.02:
        parser.error('position uncertainty must be finite and at least 0.02 m')
    if args.goal_articulation is not None and not all(
            math.isfinite(a) and 0<a<=math.radians(65) for a in args.goal_articulation):
        parser.error('goal articulation tolerances must be finite and in (0, 65 deg]')
    manifest=json.loads(args.environment.read_text())
    scenario=json.loads(args.scenario.read_text()) if args.scenario else None
    from twin.navigation.scenario import environment_grid
    cameras=json.loads(args.camera_config.read_text()) if args.camera_config else None
    grid=environment_grid(manifest,scenario,cameras)
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
        'obstacles':len(grid.obstacles),'geometry_margin':.05,'localization_allowance':args.position_uncertainty,
        'minimum_turn_radius':1.5,'max_articulation_degrees':65,'legs':[]}
    report['scenario_sha256']=hashlib.sha256(args.scenario.read_bytes()).hexdigest() if args.scenario else None
    report['camera_config_sha256']=hashlib.sha256(args.camera_config.read_bytes()).hexdigest() if args.camera_config else None
    report['goals']=poses
    # Towing-geometry record so the integration isaac_bundle defaults trace to one source:
    # the uncoupled transport envelope must enclose the 0.621 m hitch extent (default 0.55 is
    # too small) and a towed cart must be masked as self-geometry, not read as a rear obstacle.
    spec=controller.spec
    aligned=occupied_envelope(Pose2(0,0,0),Pose2(spec.hitch_x-spec.cart_hitch_x,0,0),spec=spec,margin=0.0)
    xs=[p[0] for p in aligned];ys=[p[1] for p in aligned]
    report['towing_geometry']=dict(
        transport_radius_m=spec.transport_radius,
        base_rear_corner_radius_m=math.hypot(spec.length/2,spec.width/2),
        hitch_offset_m=abs(spec.hitch_x),
        cart_self_geometry=dict(dimensions_m=[spec.cart_length,spec.cart_width,spec.cart_height],
            body_center_behind_base_m=abs(spec.hitch_x-spec.cart_hitch_x),
            rearmost_behind_base_m=abs(spec.hitch_x-spec.cart_hitch_x)+spec.cart_length/2,
            far_corner_radius_m=math.hypot(abs(spec.hitch_x-spec.cart_hitch_x)+spec.cart_length/2,spec.cart_width/2),
            swings_about_hitch_deg=math.degrees(spec.max_articulation)),
        coupled_occupied_envelope_aligned=dict(x_range_m=[min(xs),max(xs)],y_range_m=[min(ys),max(ys)]),
        docking_tolerance=dict(position_m=0.08,heading_rad=0.08,
            base_settle_linear_m_s=0.02,base_settle_angular_rad_s=0.03))
    report['goal_articulation_rad']=dict(outbound=args.goal_articulation[0],
        return_=args.goal_articulation[1]) if args.goal_articulation else None
    args.output.mkdir(parents=True,exist_ok=True)
    v=w=0.0
    legs=[('outbound',poses['shelf_approach'],args.goal_articulation[0] if args.goal_articulation else None),
          ('return',poses['robot_dock'],args.goal_articulation[1] if args.goal_articulation else None)]
    for name,goal,goal_art in legs:
        controller.start(goal,True,goal_articulation=goal_art)
        trace=[];began=time.monotonic();peak_angle=0.0;cart_yaws=[]
        for tick in range(5200):
            now=(tick+1)*.05
            odom=Odometry(state.pose,now,v,w,'test_fixture',args.position_uncertainty)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            result=controller.step(.05,odom,obs,now=now)
            v,w=result.velocity
            if v<0 or abs(w)*1.5>v+1e-9:
                raise AssertionError('illegal towing command')
            next_state=advance(state,v*.05,w/v if v else 0,controller.spec)
            if not controller.world.valid(next_state,extra_margin=args.position_uncertainty):
                raise AssertionError('feedback trajectory collision')
            state=next_state
            peak_angle=max(peak_angle,abs(state.articulation))
            cart_yaws.append((now,state.cart_yaw))
            trace.append(dict(time=now,state=vars(state),linear=v,angular=w,
                              status=result.status,reason=result.reason))
            if result.status!='running':
                break
        robot,cart=state.bodies(controller.spec)
        # Cart settle: the largest cart-yaw rate over the final 0.5 s. In the no-slip tow model
        # the cart rate is bounded by the base settle, so this certifies the quasi-static arrival;
        # PhysX pendulum settle of the trailer stays an integration-verified parked-check concern.
        tail=[cy for cy in cart_yaws if cy[0]>=now-0.5]
        cart_rate=max((abs(wrap(tail[i+1][1]-tail[i][1]))/max(1e-9,tail[i+1][0]-tail[i][0])
                       for i in range(len(tail)-1)),default=0.0)
        leg=dict(cart_body_pose=vars(cart),robot_body_pose=vars(robot),name=name,status=result.status,reason=result.reason,
                 goal=goal,final=vars(state),simulation_seconds=now,
                 cpu_seconds=time.monotonic()-began,peak_articulation_degrees=math.degrees(peak_angle),
                 final_articulation_degrees=math.degrees(state.articulation),
                 goal_articulation_rad=goal_art,cart_settle_rate_rad_s=cart_rate,
                 within_goal_articulation=(goal_art is None or abs(state.articulation)<goal_art))
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
    report['passed']=(len(report['legs'])==2
                      and all(l['status']=='succeeded' and l['within_goal_articulation']
                              for l in report['legs']))
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
