"""Contract tests with explicit fixtures; no Isaac/ROS/GPU imports."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.runtime import Packet, RuntimeAdapter, NavigationBridge
from twin.restock.inventory import Job
from twin.restock.scenario import validate
from twin.restock import guard


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.stops = []
        self.commands = []
        self.p = Packet('frame-1', 'test_fixture', .1, None, None, None, None,
                        True, False, 0., True)
        self.port = NS(advance=lambda dt: self.p, controls=lambda c:None,
                       stop=lambda:self.stops.append('port'))
        self.nav = NS(start=lambda goal,coupled:None,
                      step=lambda dt,p:NS(status='succeeded',linear=0.,angular=0.,reason=''),
                      cancel=lambda:self.stops.append('nav'))
        self.mobile = NS(park=lambda:self.stops.append('base'),
                         command=lambda *a,**k:self.commands.append(a),
                         attach=lambda obs:None, coupling_state=lambda o:{'healthy':True})
        self.manip = NS(start=lambda a,j:None,cart_location='rollwagen',
                        step=lambda dt,s:NS(status='running',reason='',controls=()))
        poses = {k:[0,0,0] for k in ('home','pickup','cart_dock','shelf_approach')}
        self.adapter = RuntimeAdapter(self.nav,self.mobile,self.manip,self.port,poses,fixture_mode=True)
        self.job = Job('j','milk',1,'back','shelf')

    def test_coupling_needs_later_observation(self):
        self.adapter.start('couple',self.job)
        self.assertEqual(self.adapter.step(.1).status,'running')
        self.p = replace(self.p,observation_id='frame-2',simulation_time=.2,coupled=True)
        result = self.adapter.step(.1)
        self.assertEqual(result.status,'succeeded')
        self.assertTrue(result.evidence['coupled'])

    def test_stale_packet_stops_every_adapter(self):
        self.adapter.start('outbound',self.job)
        self.p = replace(self.p,simulation_time=-2.)
        self.assertEqual(self.adapter.step(.1).status,'failed')
        self.assertIn('nav',self.stops)
        self.assertIn('port',self.stops)
        self.assertEqual(self.commands,[])

    def test_fixture_rejected_in_production(self):
        self.adapter.fixture_mode = False
        self.adapter.start('collect',self.job)
        self.assertEqual(self.adapter.step(.1).status,'failed')

    def test_stop_attempts_all_ports_on_exception(self):
        def broken():
            raise RuntimeError('base stop failure')
        self.mobile.park = broken
        with self.assertRaisesRegex(RuntimeError,'base stop failure'):
            self.adapter.stop()
        self.assertEqual(self.stops,['nav','port'])

    def test_return_requires_measured_cart(self):
        self.adapter.start('return',self.job)
        self.assertEqual(self.adapter.step(.1).status,'failed')

    def test_navigation_bridge_passes_absolute_mission_clock(self):
        calls=[]
        bridge=NavigationBridge(NS(step=lambda *a,**kw:calls.append(kw)))
        bridge.step(.1,self.p)
        self.assertEqual(calls,[{'now':.1}])

    def test_verify_does_not_invent_observation(self):
        self.adapter.start('verify',self.job)
        self.assertEqual(self.adapter.step(.1).status,'running')


class ScenarioTests(unittest.TestCase):
    def test_environment_order_is_converted_and_unsafe_portal_rejected(self):
        manifest=dict(frame='map',units='metres',up_axis='Z',ground_truth_only=True,
            backroom_stocks=[dict(id='back',sku='milk')],
            stocking_stations=[dict(slots=[dict(id='shelf',sku_ids=['milk'])])],
            static_obstacles=[{'id':'wall'}],map_bounds=[-13,13,-11,19],
            portal={'clear_width':2.5})
        scenario=dict(job=dict(job_id='j',sku='milk',quantity=1,source_bin='back',shelf_id='shelf'),
            inventory_unit='sealed_handling_unit',dynamic_stock_ready=True,
            initial_inventory=[dict(location='back',sku='milk',quantity=1)],
            coupled_navigation_radius_m=1.92)
        result=validate(manifest,scenario)
        self.assertEqual(result['navigation_bounds'],[-13,-11,13,19])
        self.assertFalse(result['ok'])
        self.assertIn('coupled navigation envelope cannot fit backroom portal',result['errors'])
        manifest['portal']['clear_width']=4.5
        self.assertTrue(validate(manifest,scenario)['ok'])


class GuardTests(unittest.TestCase):
    def test_wrong_ros_domain_refused_without_probe(self):
        with patch.dict('os.environ',{'ROS_DOMAIN_ID':'42'}):
            with self.assertRaisesRegex(RuntimeError,'ROS_DOMAIN_ID'):
                guard.isolation()

    def test_protected_blender_prevents_gpu_query(self):
        with patch.object(guard,'isolation'), patch('builtins.open',unittest.mock.mock_open()), \
             patch.object(guard.fcntl,'flock'), patch.object(guard,'protected_blender',return_value=True), \
             patch.object(guard,'query_and_assess') as gpu:
            with self.assertRaisesRegex(RuntimeError,'Blender'):
                with guard.gpu_slot():
                    self.fail('must not yield')
            gpu.assert_not_called()

    def test_low_ram_prevents_gpu_query(self):
        with patch.object(guard,'isolation'), patch('builtins.open',unittest.mock.mock_open()), \
             patch.object(guard.fcntl,'flock'), patch.object(guard,'protected_blender',return_value=False), \
             patch.object(guard,'available_ram_mb',return_value=100), \
             patch.object(guard,'query_and_assess') as gpu:
            with self.assertRaisesRegex(RuntimeError,'12 GiB'):
                with guard.gpu_slot():
                    self.fail('must not yield')
            gpu.assert_not_called()


if __name__ == '__main__':
    unittest.main()


class LaunchTests(unittest.TestCase):
    def test_guard_refusal_records_failure_without_importing_factory(self):
        import tempfile
        from run_mobile_restock import run
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'new-run'
            args=NS(factory='must_not_import:factory',output=output,heavy_threshold_mb=1500)
            scenario={'job':dict(job_id='j',sku='milk',quantity=1,source_bin='back',shelf_id='shelf'),
                      'initial_inventory':[dict(location='back',sku='milk',quantity=1)]}
            with patch('run_mobile_restock.gpu_slot',side_effect=RuntimeError('test guard refusal')):
                with self.assertRaisesRegex(RuntimeError,'test guard refusal'):
                    run(args,{},scenario)
            import json
            record=json.loads((output/'mission.json').read_text())
            self.assertEqual(record['state'],'failed')
            self.assertEqual(record['events'],[])
            self.assertNotIn('must_not_import',sys.modules)
            with self.assertRaises(FileExistsError):
                run(args,{},scenario)

    def test_schema2_uses_robot_dock_not_cart_pose(self):
        from twin.restock.scenario import poses_for_job
        manifest=dict(poses=dict(pickup=[1,2,0],robot_dock=[1,2,0],cart_dock=[-.27,2,0]),
                      pose_bodies=dict(pickup='robot_base',robot_dock='robot_base',cart_dock='cart_body'),
                      stocking_stations=[dict(pose=[3,4,0],pose_body='robot_base',slots=[dict(id='shelf')])])
        poses=poses_for_job(manifest,Job('j','milk',1,'back','shelf'))
        self.assertEqual(poses['cart_dock'],[1,2,0])
        self.assertNotEqual(poses['cart_dock'],manifest['poses']['cart_dock'])
