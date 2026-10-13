"""Controlled static-scene contract tests. Native mapping is not mocked as proof."""
from dataclasses import replace
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.navigation.temporal import MapIdentity,Acquisition,DistanceSlice,TemporalMap
from twin.navigation.mapper_config import navigation_mapper_parameters
from twin.navigation import GridMap,Observation,Odometry
from twin.navigation.articulated import ArticulatedController,TrainState
from twin.mobile import MobileSpec


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.clock=[1.]
        self.identity=MapIdentity('simulation','layout-a','calibration-a','localization-a')
        self.gate=TemporalMap(self.identity,commissioned=0.,required_sensors=['front','cart'],
            generation='empty-1',steady_clock=lambda:self.clock[0],steady_commissioned=0.)
        self.snapshot=None
        self.refresh(1.)

    def refresh(self,now,distances=None):
        self.clock[0]=now
        for name in ('front','cart'):
            acquisition=Acquisition(name,self.identity,int(now*1000),now,now)
            self.gate.observe(acquisition)
        values=(2.,)*(240*160) if distances is None else tuple(distances)
        self.snapshot=DistanceSlice('map',(-6.,-4.),.05,240,160,-1000.,values,
                                    self.gate.generation,acquisition)
        self.gate.accept(self.snapshot)

    def check(self,state=None):
        state=state or TrainState(0,0,0,0)
        return self.gate.check_train([state],MobileSpec(),.10,self.clock[0])

    def test_historical_static_free_survives_new_acquisitions(self):
        self.assertEqual(self.check(),'')
        old_values=self.snapshot.distances
        self.refresh(50.,old_values)
        self.assertEqual(self.check(),'')
        self.assertEqual(self.gate.commissioned,0.)
        self.assertEqual(self.snapshot.distances,old_values)

    def test_absolute_expiry_cannot_be_extended_by_payload_rebuild(self):
        self.gate.invalidate_clearance(now=590.)
        self.assertEqual(self.check(),'clearance_reobservation_required')
        self.gate.begin_empty_reconstruction('empty-2')
        self.refresh(599.)
        self.assertEqual(self.check(),'')
        self.refresh(600.)
        self.assertEqual(self.check(),'map_epoch_expired')

    def test_scene_camera_localization_session_changes_latch_stop(self):
        for field in ('session','scene','cameras','localization'):
            with self.subTest(field=field):
                gate=TemporalMap(self.identity,commissioned=0,required_sensors=['front'],
                    generation='one',steady_clock=lambda:1.,steady_commissioned=0)
                gate.check_identity(replace(self.identity,**{field:'changed'}))
                gate.check_identity(self.identity)
                self.assertEqual(gate.readiness(1),'map_identity_changed')
                gate.invalidate_clearance(now=1)
                with self.assertRaises(ValueError):gate.begin_empty_reconstruction('replacement')
                self.assertEqual(gate.readiness(1),'map_identity_changed')
                with self.assertRaises(ValueError):gate.begin_empty_reconstruction('two')

    def test_stale_sensor_mapper_and_frozen_sim_clock(self):
        self.clock[0]=1.301
        self.assertEqual(self.gate.readiness(1.),'stale_navigation_sensor_or_mapper')
        self.clock[0]=1.1
        self.assertEqual(self.gate.readiness(.9),'stale_navigation_sensor_or_mapper')
        self.clock[0]=2.
        for name in ('front','cart'):
            self.gate.observe(Acquisition(name,self.identity,2000,2.,2.))
        self.assertEqual(self.gate.readiness(2.),'stale_navigation_sensor_or_mapper')

    def test_held_frame_cannot_refresh_liveness(self):
        frame=self.gate.sources['front']
        with self.assertRaises(ValueError):self.gate.observe(replace(frame,stamp=1.1,steady_stamp=1.1))
        with self.assertRaises(ValueError):self.gate.observe(replace(frame,sequence=2000))
        self.assertEqual(self.gate.sources['front'],frame)

    def test_mapper_ack_must_match_acquisition_and_generation(self):
        for snapshot in (replace(self.snapshot,generation='other'),
                         replace(self.snapshot,frame='ur_base_link'),
                         replace(self.snapshot,incorporated=replace(self.snapshot.incorporated,stamp=1.01))):
            with self.assertRaises(ValueError):self.gate.accept(snapshot)

    def test_incorporated_frame_can_lag_latest_live_camera(self):
        self.gate.observe(Acquisition('cart',self.identity,1100,1.1,1.1))
        self.gate.accept(self.snapshot)
        self.clock[0]=1.1
        self.assertEqual(self.gate.readiness(1.1),'')

    def test_required_map_cannot_fall_back_to_clear_scan(self):
        controller=ArticulatedController(GridMap((-10,-10,10,10),[]),require_temporal_map=True,
                                         allow_test_fixtures=True)
        controller.start((2,0,0),True)
        obs=Observation(1.,(),True,True,0.,'test_fixture',100.)
        odom=Odometry((0,0,0),1.,source='test_fixture')
        self.assertEqual(controller.gate_velocity(1.,odom,obs,.2,0.).reason,'missing_navigation_map')

    def test_planner_receives_unknown_and_blocked_aisle(self):
        values=list(self.snapshot.distances)
        # A complete cross-aisle barrier, represented by native distance samples.
        for row in range(160):values[row*240+144]=0.
        self.gate.slice=replace(self.snapshot,distances=tuple(values))
        grid=self.gate.planning_grid(GridMap((-10,-10,10,10),[]))
        self.assertFalse(grid.free((1.2,0),0))
        self.assertTrue(grid.free((0,0),0))
        values[80*240+120]=-1000.
        self.gate.slice=replace(self.snapshot,distances=tuple(values))
        self.assertFalse(self.gate.planning_grid(GridMap((-10,-10,10,10),[])).free((0,0),0))

    def test_blocked_motion_stops_and_replans_after_new_clear_map(self):
        controller=ArticulatedController(GridMap((-5,-3,5,3),[]),require_temporal_map=True,
                                         allow_test_fixtures=True)
        controller.start((2,0,0),True)
        odom=Odometry((0,0,0),1.,source='test_fixture',position_uncertainty=.10)
        obs=Observation(1.,(),True,True,0.,'test_fixture',0.,temporal_map=self.gate)
        self.assertEqual(controller.step(.05,odom,obs,now=1.).reason,'planning_stop')
        self.assertEqual(controller.step(.05,odom,obs,now=1.).reason,'route_ready')
        self.assertGreater(controller.step(.05,odom,obs,now=1.).linear,0.)
        blocked=list(self.snapshot.distances);blocked[80*240+138]=0. # x=.9 ahead of base
        self.gate.slice=replace(self.snapshot,distances=tuple(blocked))
        result=controller.step(.05,odom,obs,now=1.)
        self.assertEqual(result.linear,0.);self.assertEqual(result.reason,'occupied_navigation_cell')
        self.assertIsNone(controller.route)
        self.refresh(1.1)
        odom=replace(odom,stamp=1.1);obs=replace(obs,stamp=1.1)
        self.assertEqual(controller.step(.05,odom,obs,now=1.1).reason,'route_ready')
        self.assertGreater(controller.step(.05,odom,obs,now=1.1).linear,0.)

    def test_clearance_change_requires_new_empty_generation_and_observations(self):
        old=self.snapshot
        self.gate.invalidate_clearance(now=1.1)
        self.gate.accept(old)
        self.assertEqual(self.check(),'clearance_reobservation_required')
        with self.assertRaises(ValueError):self.gate.begin_empty_reconstruction('empty-1')
        self.gate.begin_empty_reconstruction('empty-2')
        with self.assertRaises(ValueError):self.gate.observe(old.incorporated)
        self.assertEqual(self.gate.readiness(1.1),'missing_navigation_sensor')
        self.refresh(1.2,[-1000.]*(240*160))
        self.assertEqual(self.check(),'unknown_navigation_cell')

    def test_unknown_occupied_nonfinite_and_outside_stop(self):
        for value,reason in [(-1000.,'unknown_navigation_cell'),(math.nan,'unknown_navigation_cell'),
                             (0.,'occupied_navigation_cell'),(-.1,'occupied_navigation_cell')]:
            self.gate.slice=replace(self.snapshot,distances=(value,)*(240*160))
            self.assertEqual(self.check(),reason)
        self.gate.slice=self.snapshot
        self.assertEqual(self.check(TrainState(7,0,0,0)),'outside_navigation_map')

    def test_independent_cart_sweep_and_interior_unknown(self):
        aligned=TrainState(0,0,0,0);turned=TrainState(0,0,0,1.)
        # Search a few cart-side cells to obtain a geometric counterexample,
        # not a point-robot-only acceptance assertion.
        found=False
        for x,y in [(-.8,-.9),(-1.,-.85),(-1.2,-.8),(-1.3,-.75)]:
            values=list(self.snapshot.distances)
            i=round((x+6)/.05);j=round((y+4)/.05);values[j*240+i]=0.
            self.gate.slice=replace(self.snapshot,distances=tuple(values))
            if self.check(aligned)=='' and self.check(turned)=='occupied_navigation_cell':
                found=True;break
        self.assertTrue(found,'cart heading must change checked swept clearance')

    def test_command_moves_only_with_valid_temporal_map(self):
        controller=ArticulatedController(GridMap((-10,-10,10,10),[]),allow_test_fixtures=True)
        controller.start((2,0,0),True)
        odom=Odometry((0,0,0),1.,source='test_fixture',position_uncertainty=.10)
        obs=Observation(1.,(),True,True,0.,'test_fixture',0.,temporal_map=self.gate)
        result=controller.gate_velocity(1.,odom,obs,.25,0.)
        self.assertEqual(result.reason,'tracking');self.assertGreater(result.linear,0)
        self.gate.invalidate_clearance(now=1.)
        self.assertEqual(controller.gate_velocity(1.,odom,obs,.25,0.).linear,0.)
        self.assertEqual(controller.gate_velocity(1.,odom,replace(obs,observed_space=object()),.25,0.).reason,
                         'instantaneous_coverage_retired')

    def test_measured_latency_extends_actual_stopping_sweep(self):
        values=list(self.snapshot.distances);values[80*240+141]=0. # (1.05, 0.0)
        self.gate.slice=replace(self.snapshot,distances=tuple(values))
        controller=ArticulatedController(GridMap((-10,-10,10,10),[]),allow_test_fixtures=True)
        controller.start((2,0,0),True)
        odom=Odometry((0,0,0),1.,source='test_fixture',position_uncertainty=.10)
        obs=Observation(1.,(),True,True,0.,'test_fixture',0.,temporal_map=self.gate)
        self.assertEqual(controller.gate_velocity(1.,odom,obs,.25,0.).reason,'tracking')
        self.gate.measured_latency=.3
        self.assertEqual(controller.gate_velocity(1.,odom,obs,.25,0.).reason,'occupied_navigation_cell')

    def test_native_slice_layout_and_mapper_configuration(self):
        msg=SimpleNamespace(header=SimpleNamespace(frame_id='map'),origin=SimpleNamespace(x=-6.,y=-4.),
            resolution=.05,width=240,height=160,unknown_value=-1000.,data=self.snapshot.distances)
        decoded=DistanceSlice.from_message(msg,generation='empty-1',incorporated=self.snapshot.incorporated)
        self.assertEqual(decoded,self.snapshot)
        params=navigation_mapper_parameters(camera_count=2,transport_height=1.9)
        self.assertEqual(params['global_frame'],'map')
        self.assertGreater(params['static_mapper.esdf_slice_max_height'],1.9)
        self.assertFalse(params['static_mapper.tsdf_set_free_distance_on_decayed'])
        with self.assertRaises(ValueError):navigation_mapper_parameters(camera_count=24,transport_height=1.9)

    def test_vectorized_cell_contacts_match_established_polygon_checker(self):
        from twin.navigation.articulated import polygon_hits_box
        from twin.mobile import occupied_envelope
        for angle in (0.,.4,1.):
            state=TrainState(0,0,.2,angle)
            polygon=occupied_envelope(*state.bodies(MobileSpec()),MobileSpec(),margin=.15)
            for x in (-1.9,-1.2,0.,.8):
                for y in (-.6,0.,.6):
                    i=round((x+6)/.05);j=round((y+4)/.05)
                    values=list(self.snapshot.distances);values[j*240+i]=0.
                    self.gate.slice=replace(self.snapshot,distances=tuple(values))
                    expected=polygon_hits_box(polygon,(x-.05,y-.05,x+.05,y+.05))
                    self.assertEqual(self.gate.check_polygon(polygon)=='occupied_navigation_cell',expected)


if __name__=='__main__':unittest.main()
