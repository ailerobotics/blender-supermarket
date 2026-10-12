"""Sensor extrinsics must not be mistaken for chassis displacement."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.navigation import GridMap, Odometry
from twin.navigation.sensors import Scan, MapLocalizer, observation
from twin.navigation.articulated import ArticulatedController, TrainState


def room_scan(extrinsic):
    x,y,yaw=extrinsic
    ranges=[]
    for i in range(360):
        a=yaw-math.pi+i*math.pi/180
        dx,dy=math.cos(a),math.sin(a)
        hits=[]
        if abs(dx)>1e-8: hits.append(((5 if dx>0 else -4)-x)/dx)
        if abs(dy)>1e-8: hits.append(((6 if dy>0 else -3)-y)/dy)
        ranges.append(min(hits))
    return Scan(.1,tuple(ranges),-math.pi,math.pi/180,10,sensor_to_base=extrinsic)


class CalibrationTests(unittest.TestCase):
    def test_translation_and_rotation_compose_before_world_pose(self):
        scan=Scan(0,(2,),0,1,10,sensor_to_base=(.46,.1,math.pi/2))
        p=scan.points((1,2,math.pi/2))[0]
        self.assertAlmostEqual(p[0],-1.1)
        self.assertAlmostEqual(p[1],2.46)

    def test_offset_correction_does_not_shrink_uncertainty(self):
        grid=GridMap((-4,-3,5,6))
        scan=room_scan((.46,.1,.02))
        calibrated=MapLocalizer(grid,(0,0,0)).update(Odometry((0,0,0),.1),scan)
        self.assertLess(math.dist(calibrated.pose[:2],(0,0)),1e-8)
        self.assertGreaterEqual(calibrated.position_uncertainty,.10)
        wrong=Scan(scan.stamp,scan.ranges,scan.angle_min,scan.angle_increment,scan.range_max)
        try:
            biased=MapLocalizer(grid,(0,0,0)).update(Odometry((0,0,0),.1),wrong)
        except ValueError:
            pass  # offset can exceed the fit threshold instead of biasing pose
        else:
            self.assertGreater(math.dist(biased.pose[:2],(0,0)),.04)

    def test_offset_reduces_base_centred_coverage(self):
        scan=room_scan((.46,0,0))
        obs=observation(scan,Odometry((0,0,0),.1),coupled=True,arm_transport=True)
        self.assertAlmostEqual(obs.coverage_radius,9.54)
        with self.assertRaises(ValueError):
            Scan(0,(1,),0,1,10,sensor_to_base=(float('nan'),0,0))

    def test_pipeline_uncertainty_rejects_too_close_staging_support(self):
        scan=room_scan((.46,0,0))
        localized=MapLocalizer(GridMap((-4,-3,5,6)),(0,0,0)).update(
            Odometry((0,0,0),.1),scan)
        controller=ArticulatedController(GridMap((-10,-10,10,10),[(-.34,.76,.34,1.24)]))
        state=TrainState(0,0,0,0)
        self.assertTrue(controller.world.valid(state,extra_margin=.02))
        self.assertFalse(controller.world.valid(state,extra_margin=localized.position_uncertainty))
        controller.start((3,0,0),True)
        sensed=observation(scan,localized,coupled=True,arm_transport=True)
        result=controller.gate_velocity(.1,localized,sensed,.1,0)
        self.assertEqual(result.reason,'obstacle_stop')
        self.assertEqual(result.velocity,(0,0))


if __name__=='__main__':
    unittest.main()
