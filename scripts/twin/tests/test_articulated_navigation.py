"""Quasi-static CPU fixtures: these do not certify PhysX dynamics or sensing."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.mobile import MobileSpec, wrap
from twin.navigation import GridMap, Odometry, Observation
from twin.navigation.articulated import (ArticulatedMap,ArticulatedPlanner,
    ArticulatedController,TrainState,advance,dubins_connections)


class ArticulatedTests(unittest.TestCase):
    def setUp(self):
        self.spec=MobileSpec()
        self.world=ArticulatedMap(GridMap((-15,-15,15,15)))

    def test_actual_train_straight_through_2_5m_portal(self):
        world=ArticulatedMap(GridMap((-8,-8,8,8),((-8,-.15,-1.25,.15),(1.25,-.15,8,.15))))
        state=TrainState(0,4,-math.pi/2,-math.pi/2)
        samples=world.rollout(state,8,0,extra_margin=.02)
        self.assertTrue(samples)
        self.assertTrue(all(world.valid(s,extra_margin=.02) for s in samples))

    def test_narrow_aisle_straight_passes_but_tow_turn_fails(self):
        world=ArticulatedMap(GridMap((-8,-8,8,8),((-8,-8,-.75,8),(.75,-8,8,8))))
        state=TrainState(0,4,-math.pi/2,-math.pi/2)
        self.assertTrue(world.rollout(state,4,0))
        self.assertFalse(world.rollout(state,3,1/1.5))
        self.assertFalse(world.rollout(state,4,0,extra_margin=.05))

    def test_cart_is_independent_and_hitches_remain_coincident(self):
        state=TrainState(0,0,0,0)
        for _ in range(150):
            state=advance(state,.02,1/1.5,self.spec)
            base,cart=state.bodies(self.spec)
            self.assertLess(math.dist(base.point(self.spec.hitch_x),cart.point(self.spec.cart_hitch_x)),1e-10)
        self.assertGreater(abs(state.articulation),.5)
        self.assertLess(abs(state.articulation),self.spec.max_articulation)

    def test_dubins_shots_reach_pose_with_legal_curvature(self):
        start=TrainState(0,0,.3,.3)
        goal=(3,-2,-1.2)
        for shot in dubins_connections(start,goal):
            s=start
            for length,k in shot:
                self.assertLessEqual(abs(k),1/1.5)
                for _ in range(max(1,math.ceil(length/.02))):
                    s=advance(s,length/max(1,math.ceil(length/.02)),k,self.spec)
            self.assertLess(math.dist(s.pose[:2],goal[:2]),.001)
            self.assertLess(abs(wrap(s.yaw-goal[2])),1e-8)

    def test_planner_aligns_cart_and_returns(self):
        planner=ArticulatedPlanner(self.world,timeout=5)
        route=planner.plan(TrainState(0,0,0,0),(5,3,0),goal_articulation=.08)
        self.assertTrue(route.states,route.reason)
        self.assertLess(abs(route.states[-1].articulation),.08)
        back=planner.plan(route.states[-1],(0,0,0),goal_articulation=.08)
        self.assertTrue(back.states,back.reason)
        self.assertLess(abs(back.states[-1].articulation),.08)

    def test_articulated_braking_and_joint_limit_stop(self):
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,0,0),True)
        od=Odometry((0,0,0),.1,.2,source='test_fixture',position_uncertainty=.02)
        for points,angle in [(((-1.5,0),),0),((),1.2)]:
            obs=Observation(.1,points,True,True,angle,'test_fixture',10)
            self.assertEqual(c.gate_velocity(.1,od,obs,.2,0).velocity,(0,0))

    def test_no_pivot_reverse_or_too_tight_radius(self):
        self.assertFalse(self.world.rollout(TrainState(0,0,0,0),-1,0))
        self.assertFalse(self.world.rollout(TrainState(0,0,0,0),1,1))
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,0,0),True)
        od=Odometry((0,0,0),.1,source='test_fixture')
        obs=Observation(.1,(),True,True,0,'test_fixture',10)
        self.assertEqual(c.gate_velocity(.1,od,obs,0,.1).reason,'turn_radius')

    def test_feedback_controller_curved_goal(self):
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,3,0),True)
        state=TrainState(0,0,0,0); v=w=0
        for i in range(2000):
            now=(i+1)*.05
            od=Odometry(state.pose,now,v,w,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            result=c.step(.05,od,obs,now=now)
            v,w=result.velocity
            self.assertGreaterEqual(v,0)
            self.assertLessEqual(abs(w)*1.5,v+1e-9)
            state=advance(state,v*.05,w/v if v else 0,self.spec)
            if result.status!='running': break
        self.assertEqual(result.status,'succeeded',result.reason)

    def test_transient_measured_turn_radius_keeps_route(self):
        # A transient measured-curvature trip (base momentarily under commanded speed on a
        # tight turn) must keep the planned route and coast, not discard it + full stop/replan.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,3,0),True)
        state=TrainState(0,0,0,0); v=w=0.0; now=0.0
        for i in range(400):
            now=(i+1)*.05
            od=Odometry(state.pose,now,v,w,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,od,obs,now=now)
            v,w=r.velocity
            if c.route is not None and c.route.states and r.reason=='tracking':
                break
            state=advance(state,v*.05,w/v if v else 0,self.spec)
        self.assertTrue(c.route and c.route.states,'controller never reached tracking')
        route=c.route
        # Inject odometry: base just above the 0.02 gate but its MEASURED arc is well past the
        # 1/1.5 min-radius cap -> the braking-curvature gate trips measured_turn_radius.
        now+=.05
        tight=Odometry(state.pose,now,0.05,0.05*(1/1.5+0.3),'test_fixture',.02)
        obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
        r=c.step(.05,tight,obs,now=now)
        self.assertEqual(r.reason,'measured_turn_radius')
        self.assertIs(c.route,route)          # route PRESERVED (no discard, no replan)
        self.assertNotEqual(r.status,'failed')  # non-terminal wait, resumes the route on recovery
        # Contrast: a genuine obstacle refusal still drops the route for replanning.
        now+=.05
        ring=tuple((state.x+.1*math.cos(a),state.y+.1*math.sin(a))
                   for a in [k*math.pi/8 for k in range(16)])
        od=Odometry(state.pose,now,v,w,'test_fixture',.02)
        obs=Observation(now,ring,True,True,state.articulation,'test_fixture',10)
        c.step(.05,od,obs,now=now)
        self.assertIsNone(c.route)            # dropped -> will replan

    def _reach_tracking(self, c):
        state=TrainState(0,0,0,0); v=w=0.0; now=0.0
        for i in range(400):
            now=(i+1)*.05
            od=Odometry(state.pose,now,v,w,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,od,obs,now=now)
            v,w=r.velocity
            if c.route is not None and c.route.states and r.reason=='tracking':
                break
            state=advance(state,v*.05,w/v if v else 0,self.spec)
        self.assertTrue(c.route and c.route.states,'controller never reached tracking')
        return state, now

    def test_persistent_measured_turn_radius_faults_fast(self):
        # fm446 (C): a measured_turn_radius that trips EVERY tick (a sub-min-radius turn the coupled base
        # cannot make) must FAIL FAST with a named fault, not livelock. measured_turn_fault_s=1.0 => after
        # ~1 s (20 ticks at dt .05) of net over-rotation coast it terminates 'measured_turn_radius_persistent'.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True,measured_turn_fault_s=1.0)
        c.start((5,3,0),True)
        state,now=self._reach_tracking(c)
        reasons=[]
        for _ in range(40):
            now+=.05
            tight=Odometry(state.pose,now,0.05,0.05*(1/1.5+0.3),'test_fixture',.02)   # over-rotating arc
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,tight,obs,now=now)
            reasons.append((r.status,r.reason))
            if r.status=='failed':
                break
        self.assertEqual(reasons[-1],('failed','measured_turn_radius_persistent'))
        self.assertLessEqual(len(reasons),25)                 # ~20 ticks (1.0 s / 0.05), bounded fast

    def test_measured_turn_fault_off_by_default_coasts(self):
        # Default (no measured_turn_fault_s) is byte-identical: persistent measured_turn_radius keeps
        # coasting (never the new fault) -- the legacy behaviour.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,3,0),True)
        state,now=self._reach_tracking(c)
        for _ in range(40):
            now+=.05
            tight=Odometry(state.pose,now,0.05,0.05*(1/1.5+0.3),'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,tight,obs,now=now)
            self.assertNotEqual(r.reason,'measured_turn_radius_persistent')

    def test_tight_turn_commands_reduced_speed(self):
        # The tracking command slows for tight turns so the coupled base can keep up: the
        # feedback curved-goal run must never command max_speed while turning hard.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,3,0),True)
        state=TrainState(0,0,0,0); v=w=0.0
        turning_speeds=[]
        for i in range(2000):
            now=(i+1)*.05
            od=Odometry(state.pose,now,v,w,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,od,obs,now=now)
            v,w=r.velocity
            if v>1e-6 and abs(w)/v>0.5*(1/1.5):      # a meaningfully curved command
                turning_speeds.append(v)
            state=advance(state,v*.05,w/v if v else 0,self.spec)
            if r.status!='running': break
        self.assertTrue(turning_speeds,'no curved commands observed')
        self.assertLess(max(turning_speeds),c.max_speed-1e-9)

    def test_seed_replay_checks_full_margin_and_obstacles(self):
        from twin.navigation.articulated import Route
        state=TrainState(0,0,0,0)
        states=[state]
        for _ in range(50): states.append(advance(states[-1],.08,0,self.spec))
        seed=Route(states,[0.0]*50,'fixture')
        planner=ArticulatedPlanner(self.world)
        checked=planner.validate_seed(seed,state,(4,0,0),extra_margin=.02)
        self.assertTrue(checked.states)
        rejected=planner.validate_seed(seed,state,(4,0,0),dynamic=((2,0),),extra_margin=.02)
        self.assertEqual(rejected.reason,'seed_collision')
        # A path valid without localization allowance is NOT silently accepted.
        wall=ArticulatedMap(GridMap((-10,-10,10,10),((-10,.72,10,1.0),)))
        self.assertTrue(ArticulatedPlanner(wall).validate_seed(seed,state,(4,0,0),extra_margin=0).states)
        self.assertFalse(ArticulatedPlanner(wall).validate_seed(seed,state,(4,0,0),extra_margin=.02).states)

    def test_coupled_gate_is_direction_aware_rear_clears_ahead_stops(self):
        # The coupled gate already rolls the braking sweep FORWARD along the commanded curvature
        # (it never bounds the arc with an omnidirectional disk): a static return in the train's
        # rear hemisphere does not trip, while the same return AHEAD, inside the forward braking
        # rollout, does. This locks the coupled equivalent of the direction-aware base zone.
        spec = MobileSpec()
        world = ArticulatedMap(GridMap((-20, -20, 20, 20)))
        c = ArticulatedController(GridMap((-20, -20, 20, 20)), spec, allow_test_fixtures=True)
        c.start((5, 0, 0), True)
        od = Odometry((0, 0, 0), 0.1, 0.0, 0.0, 'test_fixture', 0.05)
        # The train spans robot nose (+0.5) to cart rear (~-1.77 with margins). An obstacle BEYOND
        # the cart rear (-2.0) is behind the whole no-reverse train -> no trip; one just beyond the
        # nose, inside the forward braking rollout (~0.7 m), stops.
        rear = Observation(0.1, ((-2.0, 0.0),), True, True, 0.0, 'test_fixture', 10.0)
        ahead = Observation(0.1, ((0.7, 0.0),), True, True, 0.0, 'test_fixture', 10.0)
        self.assertEqual(c.gate_velocity(0.1, od, rear, 0.25, 0.0).reason, 'tracking')
        self.assertEqual(c.gate_velocity(0.1, od, ahead, 0.25, 0.0).reason, 'obstacle_stop')

    def test_seed_rejects_teleport(self):
        from twin.navigation.articulated import Route
        a=TrainState(0,0,0,0)
        b=TrainState(0,.08,0,0)
        result=ArticulatedPlanner(self.world).validate_seed(Route([a,b],[0],'fixture'),a,b.pose)
        self.assertEqual(result.reason,'seed_kinematics_mismatch')


if __name__=='__main__': unittest.main()
