"""Portal guidance proposes trajectories; full train clearance decides acceptance."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.navigation import GridMap
from twin.navigation.articulated import ArticulatedMap,ArticulatedPlanner,TrainState
from twin.navigation.scenario import environment_grid


class CorridorTests(unittest.TestCase):
    def setUp(self):
        # Critical AABBs from the portal/cabinet/endcap arrangement. Decorative
        # stock is unnecessary for this regression; the full manifest is tested
        # separately by the scenario feedback checker.
        boxes=[(-13.25,11,-5.125,11.25),(-2.625,11,13.25,11.25),
               (-2.5,9.95,-1.25,10.85),(-11.525,10.254,-5.225,10.89),
               (-3.575,4.51,-2.275,5.146),(-6.625,4.51,-5.325,5.146),
               (-6.07,16.86,-5.39,17.34)]
        self.world=ArticulatedMap(GridMap((-13.25,-11.25,13.25,19.5),boxes,
                              corridors=[(-3.875,11.125,-math.pi/2)]))
        self.start=TrainState(-5.73,16,0,0)
        self.goal=(-3.05,6,math.pi)

    def test_alignment_retains_full_uncertainty_and_cart_sweeps(self):
        planner=ArticulatedPlanner(self.world)
        result=planner.plan(self.start,self.goal,extra_margin=.10)
        self.assertEqual(result.reason,'validated_corridor')
        self.assertTrue(result.states)
        for a,b,k in zip(result.states,result.states[1:],result.curvatures):
            self.assertTrue(self.world.rollout(a,math.dist(a.pose[:2],b.pose[:2]),k,
                                               extra_margin=.10))
        # A stale corridor hint must not ignore a newly blocked door.
        rejected=planner.validate_seed(result,self.start,self.goal,
                                      dynamic=[(-3.875,11.125)],extra_margin=.10)
        self.assertFalse(rejected.states)
        self.assertEqual(rejected.reason,'seed_collision')

    def test_corridor_heading_comes_from_map_route_not_world_axis(self):
        manifest={'map_bounds':[-5,5,-5,5],'static_obstacles':[],
                  'portal':{'centre':[0,0,0]},'route_hints':[[-2,0],[0,0],[2,0]]}
        self.assertEqual(environment_grid(manifest).corridors,((0,0,0.),))
        manifest['route_hints']=[[0,2],[0,0],[0,-2]]
        self.assertAlmostEqual(environment_grid(manifest).corridors[0][2],-math.pi/2)
        manifest.pop('route_hints')
        self.assertEqual(environment_grid(manifest).corridors,())

    def test_bad_corridor_metadata_rejected(self):
        with self.assertRaises(ValueError):
            GridMap((-5,-5,5,5),corridors=[(0,0,float('nan'))])


if __name__=='__main__':unittest.main()
