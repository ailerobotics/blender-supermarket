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


if __name__=='__main__': unittest.main()
