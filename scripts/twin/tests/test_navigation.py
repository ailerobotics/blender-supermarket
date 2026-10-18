"""Deterministic CPU fixtures, never evidence of a physics run."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.navigation import Controller,Footprint,GridMap,Odometry,Observation
from twin.navigation.sensors import WheelOdometry,MapLocalizer,Scan
from twin.navigation.nav2 import require_isolation
from unittest.mock import patch
import tempfile


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.grid=GridMap((-6,-6,6,6),resolution=0.25)
        self.fp=Footprint(0.4,0.5,1.0)

    def inputs(self,pose=(0,0,0),stamp=0.1,coupled=False,obstacles=()):
        return (Odometry(pose,stamp,source='test_fixture'),
                Observation(stamp,obstacles,coupled,True,source='test_fixture',coverage_radius=10))

    def controller(self,grid=None,fp=None,**kwargs):
        return Controller(grid or self.grid,fp or self.fp,allow_test_fixtures=True,**kwargs)

    def test_narrow_door_base_fits_cart_rejected(self):
        grid=GridMap((-6,-6,6,6),((-0.2,-6,0.2,-0.8),(-0.2,0.8,0.2,6)),0.25)
        self.assertTrue(grid.plan((-3,0),(3,0),self.fp.radius(False)))
        self.assertFalse(grid.plan((-3,0),(3,0),self.fp.radius(True)))

    def test_cart_rear_and_swing_obstacle(self):
        c=self.controller(); c.start((3,0,0),True)
        od,obs=self.inputs(coupled=True,obstacles=((-1.4,0),))
        self.assertEqual(c.step(0.1,od,obs).velocity,(0,0))
        self.assertFalse(self.grid.free((0,0),self.fp.radius(True),((0,1.4),)))

    def test_full_circle_contains_articulated_corners(self):
        for a in range(0,360,5):
            x=-self.fp.hitch_offset+math.cos(math.radians(a))*self.fp.cart_hitch_to_farthest_corner
            y=math.sin(math.radians(a))*self.fp.cart_hitch_to_farthest_corner
            self.assertLess(math.hypot(x,y),self.fp.radius(True))

    def test_route_detours_obstacle_no_corner_cutting(self):
        grid=GridMap((-6,-6,6,6),((-0.5,-2,0.5,2),),0.25)
        route=grid.plan((-4,0),(4,0),0.6)
        self.assertTrue(route)
        for a,b in zip([(-4,0)]+route,route):
            self.assertTrue(grid.segment_free(a,b,0.6))
        self.assertTrue(any(abs(p[1])>2.6 for p in route))

    def test_dynamic_replan(self):
        c=self.controller(); c.start((4,0,0),False)
        c.step(0.1,*self.inputs())
        old=list(c.path)
        c.step(0.1,*self.inputs(stamp=0.2,obstacles=((2,0),)))
        self.assertNotEqual(old,c.path)
        for a,b in zip([(0,0)]+c.path,c.path):
            self.assertTrue(self.grid.segment_free(a,b,0.55,((2,0),)))

    def test_blocked_timeout_latches(self):
        c=self.controller(blocked_timeout=0.2); c.start((3,0,0),False)
        a=c.step(0.1,*self.inputs(obstacles=((0.6,0),)))
        b=c.step(0.1,*self.inputs(stamp=0.2,obstacles=((0.6,0),)))
        self.assertEqual(a.velocity,(0,0)); self.assertEqual(b.status,'failed')
        self.assertEqual(c.step(0.1,*self.inputs(stamp=0.3)).status,'failed')

    def test_stale_coupling_arm_and_coverage_fail_closed(self):
        for obs in (Observation(0.1,(),True,True,coverage_radius=10),
                    Observation(0.1,(),False,False,coverage_radius=10),
                    Observation(0.1,(),False,True,coverage_radius=0)):
            c=Controller(self.grid,self.fp); c.start((3,0,0),False)
            self.assertEqual(c.step(0.1,Odometry((0,0,0),0.1),obs).velocity,(0,0))
        c=self.controller(); c.start((3,0,0),False)
        self.assertEqual(c.step(0.1,*self.inputs(stamp=-1)).reason,'stale_sensing')

    def test_ground_truth_rejected(self):
        c=Controller(self.grid,self.fp); c.start((3,0,0),False)
        result=c.step(0.1,Odometry((0,0,0),0.1,source='ground_truth'),
                      Observation(0.1,(),False,True,coverage_radius=10))
        self.assertEqual(result.reason,'unsupported_observation_source')

    def test_goal_requires_orientation_and_stopped(self):
        c=self.controller(); c.start((0,0,math.pi/2),False)
        self.assertNotEqual(c.step(0.1,*self.inputs()).status,'succeeded')
        od,ob=self.inputs((0,0,math.pi/2),0.2)
        moving=Odometry(od.pose,od.stamp,0.1,source='test_fixture')
        self.assertEqual(c.step(0.1,moving,ob).reason,'settling_at_goal')
        self.assertEqual(c.step(0.1,*self.inputs(od.pose,0.3)).status,'succeeded')

    def test_tight_arrival_tolerance_delays_goal_success(self):
        # A base 0.06 m from the goal (aligned, stopped) arrives with the default 0.08 m tol, but
        # NOT with a tight 0.03 m tol (the on-axis couple standoff) -- it keeps closing instead.
        grid=GridMap((-6,-6,6,6),resolution=0.25)
        od,obs=self.inputs(pose=(0.06,0,0))  # 0.06 m short of goal (0,0,0), aligned, stopped
        loose=self.controller(grid=grid); loose.start((0,0,0),False)
        self.assertEqual(loose.step(0.1,od,obs,now=0.1).status,'succeeded')
        tight=self.controller(grid=grid); tight.start((0,0,0),False,arrival_xy_tol=0.03)
        self.assertEqual(tight.step(0.1,od,obs,now=0.1).status,'running')  # still closing
        with self.assertRaises(ValueError):
            self.controller(grid=grid).start((0,0,0),False,arrival_xy_tol=0.0)

    def test_reference_closed_loop_arrival(self):
        c=self.controller(); c.start((2,1,0.4),False)
        pose=(0.,0.,0.); v=w=0.
        for tick in range(2000):
            now=(tick+1)*0.05
            od=Odometry(pose,now,v,w,source='test_fixture')
            obs=Observation(now,(),False,True,source='test_fixture',coverage_radius=10)
            result=c.step(0.05,od,obs,now=now)
            v,w=result.velocity
            pose=(pose[0]+v*math.cos(pose[2])*0.05,pose[1]+v*math.sin(pose[2])*0.05,pose[2]+w*0.05)
            if result.status!='running':
                break
        self.assertEqual(result.status,'succeeded')
        self.assertLess(math.dist(pose[:2],(2,1)),0.08)

    def test_turn_radius_and_reverse_gate(self):
        c=self.controller(fp=Footprint(0.4,0.5,1.0,min_turn_radius=1.0))
        c.start((3,0,0),True); od,obs=self.inputs(coupled=True)
        self.assertEqual(c.gate_velocity(0.1,od,obs,0.1,0.3).reason,'turn_radius')
        self.assertEqual(c.gate_velocity(0.1,od,obs,-0.1,0).reason,'reverse_or_invalid_command')
        self.assertEqual(c.step(0.1,od,obs).reason,'requires_nav2_curvature_planner')

    def test_measured_braking_speed_inflates_obstacle_stop(self):
        c=self.controller(); c.start((3,0,0),False)
        od=Odometry((0,0,0),0.1,linear=0.7,source='test_fixture')
        _,obs=self.inputs(obstacles=((1.2,0),))
        self.assertEqual(c.gate_velocity(0.1,od,obs,0,0).reason,'obstacle_stop')

    def test_docking_cap_shrinks_braking_disk_past_rear_obstacle(self):
        # ds-007 shape: a dock goal (3,0,0) with the robot 1.0 m short of it (2,0,0) and
        # the backstock staging support sensed 0.60 m directly BEHIND it (rear hemisphere
        # of a no-reverse base). radius(False)=0.50 + default 0.05 position uncertainty
        # = 0.55; at full speed stop=0.153 so the omnidirectional disk 0.703 engulfs the
        # rear return -> obstacle_stop. The dock cap (0.05 m/s -> stop 0.018 -> disk 0.568)
        # clears it; the footprint radius itself is UNCHANGED.
        from twin.navigation import DockingApproach
        fp=Footprint(0.45,0.5,1.0,margin=0.05)  # radius(False)=0.50
        dock=DockingApproach(goal=(3,0,0),speed=0.05,approach_distance=1.5)
        _,obs=self.inputs(obstacles=((1.40,0.0),))  # 0.60 m behind the robot at (2,0)
        od=Odometry((2,0,0),0.1,source='test_fixture')
        base=self.controller(fp=fp); base.start((3,0,0),False)
        self.assertEqual(base.gate_velocity(0.1,od,obs,0.25,0.0).reason,'obstacle_stop')
        capped=self.controller(fp=fp,docking=dock); capped.start((3,0,0),False)
        self.assertEqual(capped.gate_velocity(0.1,od,obs,0.25,0.0).reason,'tracking')
        self.assertTrue(capped._docking_active)
        # The cap binds only to the configured dock goal: a different goal never engages it.
        other=self.controller(fp=fp,docking=dock); other.start((2,3,0),False)
        self.assertFalse(other._docking_active)
        self.assertEqual(other.gate_velocity(0.1,od,obs,0.25,0.0).reason,'obstacle_stop')

    def test_docking_cap_does_not_shrink_the_body_footprint(self):
        # An obstacle inside the bare footprint still stops, even under the dock cap:
        # the cap never masks geometry, it only shortens the forward braking reach.
        from twin.navigation import DockingApproach
        fp=Footprint(0.45,0.5,1.0,margin=0.05)  # radius(False)=0.50, +0.05 unc = 0.55
        dock=DockingApproach(goal=(3,0,0),speed=0.01,approach_distance=1.5)
        _,obs=self.inputs(obstacles=((2.40,0.0),))  # 0.40 m ahead < 0.55 footprint radius
        od=Odometry((2,0,0),0.1,source='test_fixture')
        c=self.controller(fp=fp,docking=dock); c.start((3,0,0),False)
        self.assertTrue(c._docking_active)
        self.assertEqual(c.gate_velocity(0.1,od,obs,0.25,0.0).reason,'obstacle_stop')

    def test_docking_cap_caps_the_commanded_linear_speed(self):
        from twin.navigation import DockingApproach
        dock=DockingApproach(goal=(3,0,0),speed=0.06,approach_distance=1.5)
        _,obs=self.inputs()
        c=self.controller(docking=dock); c.start((3,0,0),False)
        near=Odometry((2,0,0),0.1,source='test_fixture')  # 1.0 m from the dock goal
        self.assertAlmostEqual(c.gate_velocity(0.1,near,obs,0.25,0.0).linear,0.06)
        # Outside the approach zone the cap is inert (full commanded speed passes).
        far=Odometry((0,0,0),0.1,source='test_fixture')  # 3.0 m from the dock goal
        self.assertAlmostEqual(c.gate_velocity(0.1,far,obs,0.25,0.0).linear,c.max_speed)

    def test_wheel_odometry_straight_arc_and_time(self):
        wheel=WheelOdometry(0.1,0.5)
        wheel.update(0,0,0)
        self.assertAlmostEqual(wheel.update(10,10,1).pose[0],1)
        result=wheel.update(10,15,2)
        self.assertAlmostEqual(result.pose[2],1)
        self.assertAlmostEqual(result.pose[0],1+0.25*math.sin(1))
        with self.assertRaises(ValueError): wheel.update(10,15,2)

    def test_localization_uses_ranges_not_world_pose(self):
        grid=GridMap((-4,-3,5,6))
        ranges=[]
        for i in range(360):
            a=-math.pi+i*math.pi/180
            dx,dy=math.cos(a),math.sin(a)
            distances=[]
            if abs(dx)>1e-8: distances.append((5 if dx>0 else -4)/dx)
            if abs(dy)>1e-8: distances.append((6 if dy>0 else -3)/dy)
            ranges.append(min(distances))
        scan=Scan(0.1,tuple(ranges),-math.pi,math.pi/180,10)
        localizer=MapLocalizer(grid,(0.1,-0.05,0.02))
        result=localizer.update(Odometry((0,0,0),0.1),scan)
        self.assertLess(math.dist(result.pose[:2],(0,0)),0.01)
        self.assertAlmostEqual(result.pose[2],0)
        with self.assertRaises(ValueError):
            localizer.update(Odometry((0,0,0),0.2),Scan(0.2,(10,)*360,-math.pi,math.pi/180,10))

    def test_nav2_assets_preserve_cart_radius_and_thin_wall(self):
        import yaml
        from twin.navigation.nav2 import write_assets
        template={'amcl':{'ros__parameters':{}},
                  'global_costmap':{'global_costmap':{'ros__parameters':{
                      'plugins':['static_layer','inflation_layer'],'inflation_layer':{}}}},
                  'local_costmap':{'local_costmap':{'ros__parameters':{
                      'plugins':['voxel_layer','inflation_layer'],'inflation_layer':{}}}}}
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'template.yaml'; source.write_text(yaml.safe_dump(template))
            grid=GridMap((-2,-2,2,2),((0.01,-2,0.02,2),),0.2)
            mapfile,params=write_assets(d,grid,self.fp,template=source)
            cfg=yaml.safe_load(params.read_text())
            for name in ('global_costmap','local_costmap'):
                self.assertGreaterEqual(cfg[name][name]['ros__parameters']['robot_radius'],
                                        self.fp.radius(True))
            self.assertEqual(cfg['planner_server']['ros__parameters']['GridBased']['motion_model_for_search'],'DUBIN')
            self.assertFalse(cfg['controller_server']['ros__parameters']['FollowPath']['allow_reversing'])
            pixels=(Path(d)/'map.pgm').read_bytes().split(b'\n',3)[3]
            self.assertEqual(pixels[10*20+10],0)  # thin wall cannot disappear between cell centres
            self.assertTrue(mapfile.exists())

    def test_runtime_missing_scan_stops_and_times_out(self):
        from twin.navigation.runtime import SensorNavigation
        c=self.controller(blocked_timeout=0.2)
        runtime=SensorNavigation(c,WheelOdometry(0.1,0.5),MapLocalizer(self.grid,(0,0,0)))
        runtime.start((2,0,0),False)
        for now in (0.1,0.2):
            result=runtime.step(0.1,(0,0),None,now=now,coupled=False,arm_transport=True)
            self.assertEqual(result.velocity,(0,0))
        self.assertEqual(result.status,'failed')

    def test_ros_domain_guard(self):
        with patch.dict('os.environ',{'ROS_DOMAIN_ID':'42','ROS_LOCALHOST_ONLY':'1'}):
            with self.assertRaises(RuntimeError): require_isolation()

    def test_invalid_and_partial_scans_do_not_certify_full_coverage(self):
        from twin.navigation.sensors import observation
        odom=Odometry((0,0,0),.1)
        with self.assertRaises(ValueError):
            observation(Scan(.1,(float('nan'),)*360,-math.pi,math.pi/180,10),
                        odom,coupled=True,arm_transport=True)
        partial=observation(Scan(.1,(10,)*180,-math.pi/2,math.pi/180,10),
                            odom,coupled=True,arm_transport=True)
        self.assertEqual(partial.coverage_radius,0)


if __name__=='__main__': unittest.main()
