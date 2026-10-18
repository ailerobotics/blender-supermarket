"""Contract tests with explicit fixtures; no Isaac/ROS/GPU imports."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.runtime import Packet, RuntimeAdapter, NavigationBridge, ManipulationBridge
from twin.restock.inventory import Job
from twin.restock.scenario import validate
from twin.restock import guard


class RuntimeTests(unittest.TestCase):
    def test_owned_camera_runtime_receives_clock_and_does_not_repeat_gripper_command(self):
        from dataclasses import dataclass
        @dataclass(frozen=True)
        class Reply:
            controls: tuple = ({'kind':'gripper','enabled':True},)
        calls=[]
        runtime=NS(controller=NS(cart_location='rollwagen'),
                   step=lambda dt,stamp,parked=None:(calls.append((dt,stamp,parked)) or Reply()),
                   start=lambda op,job:None,
                   motion=NS(stop=lambda:calls.append('stop')))
        bridge=ManipulationBridge(runtime,owns_sensor_io=True)
        result=bridge.step(.1,NS(simulation_time=4.2,parked=True))
        self.assertEqual(calls,[(.1,4.2,True)])
        self.assertEqual(result.controls,())
        bridge.stop()
        self.assertEqual(calls[-1],'stop')

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
                         hold=lambda:self.stops.append('hold'),   # stationary base parking brake
                         release=lambda:None,
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

    def test_couple_runs_fine_dock_servo_before_attach(self):
        # With a fine_dock servo attached, couple must creep (commanding the base) until the
        # servo reports 'ready', THEN attach -- not attach straight off the nav arrival.
        from twin.restock.fine_dock import FineDockResult
        seq = ['creeping', 'creeping', 'ready']
        fd_calls = []
        fine = NS(start=lambda now: fd_calls.append('start'),
                  step=lambda dock, now: (fd_calls.append('step') or
                      FineDockResult(seq.pop(0), 0.02, 0.01, 0.01, 0.0, 'x')))
        attached = []
        events = []
        mobile = NS(park=lambda: None, hold=lambda: None, release=lambda: None,
                    command=lambda *a, **k: self.commands.append(a),
                    hold_cart_pin=lambda: events.append('pin'),
                    attach=lambda obs: (attached.append(obs), events.append('attach')),
                    coupling_state=lambda o: {'healthy': True})
        poses = {k: [0, 0, 0] for k in ('home', 'pickup', 'cart_dock', 'shelf_approach')}
        adapter = RuntimeAdapter(self.nav, mobile, self.manip, self.port, poses,
                                 fixture_mode=True, fine_dock=fine,
                                 hold_cart_through_couple=True)
        adapter.start('couple', self.job)
        self.assertEqual(events, ['pin'])  # cart pinned at couple start, before any base drive
        # Nav succeeds this tick -> servo starts (no attach yet).
        r = adapter.step(.1)
        self.assertEqual(r.status, 'running'); self.assertIn('fine-dock', r.reason)
        self.assertEqual(fd_calls, ['start']); self.assertEqual(attached, [])
        # Two creeping ticks command the base and still do not attach.
        for i in range(2):
            self.p = replace(self.p, observation_id=f'f{i}', simulation_time=.2 + i * .1)
            r = adapter.step(.1)
            self.assertEqual(r.status, 'running'); self.assertEqual(attached, [])
        self.assertTrue(self.commands)  # the servo drove the base
        # 'ready' -> hand over to attach this same tick.
        self.p = replace(self.p, observation_id='fr', simulation_time=.4)
        r = adapter.step(.1)
        self.assertEqual(r.status, 'running'); self.assertEqual(len(attached), 1)
        # Post-attach observation shows coupled+healthy -> succeeded.
        self.p = replace(self.p, observation_id='fc', simulation_time=.5, coupled=True)
        self.assertEqual(adapter.step(.1).status, 'succeeded')

    def _couple_seq_harness(self, poses_extra):
        from twin.restock.fine_dock import FineDockResult
        starts = []
        nav = NS(start=lambda goal, coupled, **kw: starts.append((tuple(goal), kw)),
                 step=lambda dt, p: NS(status='succeeded', linear=0., angular=0., reason=''),
                 cancel=lambda: None)
        fine = NS(start=lambda now: None,
                  step=lambda dock, now: FineDockResult('ready', 0., 0., 0.01, 0.0, 'ok'))
        attached = []
        mobile = NS(park=lambda: None, hold=lambda: None, release=lambda: None,
                    command=lambda *a, **k: None, hold_cart_pin=lambda: None,
                    attach=lambda obs: attached.append(obs), coupling_state=lambda o: {'healthy': True})
        poses = {k: [0, 0, 0] for k in ('home', 'pickup', 'shelf_approach')}
        poses['cart_dock'] = [-4.46, 15.25, 0.0]
        poses.update(poses_extra)
        adapter = RuntimeAdapter(nav, mobile, self.manip, self.port, poses,
                                 fixture_mode=True, fine_dock=fine, hold_cart_through_couple=True)
        return adapter, starts, attached

    def test_couple_navigates_approach_then_cart_dock_without_standoff(self):
        # couple_approach but no standoff: approach -> cart_dock -> fine-dock (back-compat).
        adapter, starts, attached = self._couple_seq_harness({'couple_approach': [-4.46, 16.2, 0.0]})
        adapter.start('couple', self.job)
        self.assertEqual(starts[-1][0], (-4.46, 16.2, 0.0))        # approach first
        r = adapter.step(.1)
        self.assertEqual(starts[-1][0], (-4.46, 15.25, 0.0))       # then cart_dock
        self.assertIn('advancing', r.reason); self.assertEqual(attached, [])
        self.p = replace(self.p, observation_id='f2', simulation_time=.2)
        r = adapter.step(.1); self.assertIn('fine-dock', r.reason)  # dock reached -> fine-dock
        self.p = replace(self.p, observation_id='f3', simulation_time=.3)
        self.assertEqual(adapter.step(.1).status, 'running'); self.assertEqual(len(attached), 1)

    def test_couple_navigates_approach_then_standoff_with_tight_arrival(self):
        # couple_approach + couple_standoff: approach -> standoff (TIGHT arrival) -> fine-dock insert.
        adapter, starts, attached = self._couple_seq_harness(
            {'couple_approach': [-4.46, 16.0, 0.0], 'couple_standoff': [-4.06, 15.25, 0.0]})
        adapter.start('couple', self.job)
        self.assertEqual(starts[-1][0], (-4.46, 16.0, 0.0)); self.assertEqual(starts[-1][1], {})
        r = adapter.step(.1)
        self.assertEqual(starts[-1][0], (-4.06, 15.25, 0.0))       # standoff next
        self.assertEqual(starts[-1][1], {'arrival_xy_tol': 0.05})  # with a TIGHT arrival
        self.assertIn('advancing', r.reason); self.assertEqual(attached, [])
        self.p = replace(self.p, observation_id='f2', simulation_time=.2)
        r = adapter.step(.1); self.assertIn('insert', r.reason)    # standoff reached -> fine-dock insert
        self.p = replace(self.p, observation_id='f3', simulation_time=.3)
        self.assertEqual(adapter.step(.1).status, 'running'); self.assertEqual(len(attached), 1)

    def test_couple_without_fine_dock_attaches_directly(self):
        # Absent servo: unchanged legacy path (attach straight off the parked nav arrival).
        self.adapter.start('couple', self.job)
        self.assertEqual(self.adapter.step(.1).status, 'running')  # attach happened this tick
        self.p = replace(self.p, observation_id='frame-2', simulation_time=.2, coupled=True)
        self.assertEqual(self.adapter.step(.1).status, 'succeeded')

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

    def test_arrival_rejects_wrong_cart_alignment_even_if_navigation_says_success(self):
        self.adapter.goal_articulation={'outbound':.1}
        self.nav.start=lambda *a,**kw:None
        self.p=replace(self.p,coupled=True,articulation=.8)
        self.adapter.start('outbound',self.job)
        result=self.adapter.step(.1)
        self.assertEqual(result.status,'failed')
        self.assertIn('articulation',result.reason)

    def test_sensor_navigation_gets_encoder_scan_clock_and_latch_measurements(self):
        calls=[]
        controller=NS(step=lambda *a,**kw:calls.append((a,kw)))
        packet=replace(self.p,odometry=(.1,.2),navigation='scan',coupled=True)
        NavigationBridge(controller,sensor_navigation=True).step(.1,packet)
        self.assertEqual(calls[0][0],(.1,(.1,.2),'scan'))
        self.assertEqual(calls[0][1]['articulation'],0.)
        self.assertTrue(calls[0][1]['coupled'])

    def test_navigation_bridge_passes_absolute_mission_clock(self):
        calls=[]
        bridge=NavigationBridge(NS(step=lambda *a,**kw:calls.append(kw)))
        bridge.step(.1,self.p)
        self.assertEqual(calls,[{'now':.1}])

    def test_verify_does_not_invent_observation(self):
        self.adapter.start('verify',self.job)
        self.assertEqual(self.adapter.step(.1).status,'running')

    def test_transfer_warms_station_before_dispatching_the_place_grasp(self):
        # Mirror the collect warm-up: transfer must not dispatch the place grasp until the shelf
        # station's collision map is warm (fresh collision_time), so the first observe does not
        # race a cold nvblox incorporation against the phase timeout.
        from dataclasses import dataclass
        @dataclass(frozen=True)
        class Reply:
            status: str = 'running'
            reason: str = ''
            controls: tuple = ()
            evidence: tuple = ()
        events, ready = [], {'v': False}
        controller = NS(cart_location='rollwagen',
                        warm=lambda sim, parked=None: (events.append(('warm', parked)) or ready['v']),
                        start=lambda op, job: events.append(('start', op)),
                        step=lambda dt, sim, parked=None: (events.append(('step',)) or Reply()),
                        stop=lambda: None)
        manip = ManipulationBridge(controller, owns_sensor_io=True)
        poses = {k:[0,0,0] for k in ('home','pickup','cart_dock','shelf_approach')}
        adapter = RuntimeAdapter(self.nav, self.mobile, manip, self.port, poses, fixture_mode=True)
        adapter.start('transfer', self.job)
        self.assertEqual(adapter.phase, 'warm')                 # warm up, do not execute yet
        self.assertNotIn(('start', 'transfer'), events)         # grasp NOT dispatched on start
        r = adapter.step(.1)                                    # station not ready -> keep warming
        self.assertEqual(r.status, 'running')
        self.assertTrue(any(e[0] == 'warm' for e in events))
        self.assertNotIn(('start', 'transfer'), events)
        ready['v'] = True
        self.p = replace(self.p, observation_id='frame-2', simulation_time=.2)
        adapter.step(.1)                                        # warm -> dispatch the place grasp
        self.assertIn(('start', 'transfer'), events)
        self.assertEqual(adapter.phase, 'execute')


class ScenarioTests(unittest.TestCase):
    def test_environment_order_is_converted_and_unsafe_portal_rejected(self):
        manifest=dict(frame='map',units='metres',up_axis='Z',ground_truth_only=True,
            backroom_stocks=[dict(id='back',sku='milk')],
            stocking_stations=[dict(slots=[dict(id='shelf',sku_ids=['milk'])])],
            static_obstacles=[{'id':'wall'}],map_bounds=[-13,13,-11,19],
            portal={'clear_width':1.3})
        scenario=dict(job=dict(job_id='j',sku='milk',quantity=1,source_bin='back',shelf_id='shelf'),
            inventory_unit='sealed_handling_unit',dynamic_stock_ready=True,
            initial_inventory=[dict(location='back',sku='milk',quantity=1)],
            navigation_model='articulated')
        result=validate(manifest,scenario)
        self.assertEqual(result['navigation_bounds'],[-13,-11,13,19])
        self.assertFalse(result['ok'])
        self.assertIn('aligned articulated envelope cannot fit backroom portal',result['errors'])
        manifest['portal']['clear_width']=2.5
        self.assertTrue(validate(manifest,scenario)['ok'])
        self.assertFalse(validate(manifest,scenario)['route_verified'])


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
        import json
        from run_mobile_restock import run
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'new-run'
            args=NS(factory='must_not_import:factory',output=output,heavy_threshold_mb=1500,
                    runtime_config=Path(tmp)/'config.json')
            scenario={'job':dict(job_id='j',sku='milk',quantity=1,source_bin='back',shelf_id='shelf'),
                      'initial_inventory':[dict(location='back',sku='milk',quantity=1)]}
            scenario_path=Path(tmp)/'scenario.json'
            scenario_path.write_text(json.dumps(scenario))
            args.runtime_config.write_text(json.dumps({'scenario':str(scenario_path)}))
            ready=patch('preflight_mobile_restock.check',return_value={'files_ready':True})
            with ready, patch('run_mobile_restock.gpu_slot',side_effect=RuntimeError('test guard refusal')):
                with self.assertRaisesRegex(RuntimeError,'test guard refusal'):
                    run(args,{},scenario)
            import json
            record=json.loads((output/'mission.json').read_text())
            self.assertEqual(record['state'],'failed')
            self.assertEqual(record['events'],[])
            self.assertNotIn('must_not_import',sys.modules)
            with patch('preflight_mobile_restock.check',return_value={'files_ready':True}):
                with self.assertRaises(FileExistsError):
                    run(args,{},scenario)

    def test_missing_runtime_inputs_prevent_run_state_and_gpu_probe(self):
        import tempfile
        from run_mobile_restock import run
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json'
            config.write_text('{}')
            args=NS(factory='must_not_import:factory',output=Path(tmp)/'run',runtime_config=config)
            with patch('run_mobile_restock.gpu_slot') as gpu:
                with self.assertRaisesRegex(ValueError,'runtime inputs unresolved'):
                    run(args,{}, {})
                gpu.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_schema2_uses_robot_dock_not_cart_pose(self):
        from twin.restock.scenario import poses_for_job
        manifest=dict(poses=dict(pickup=[1,2,0],robot_dock=[1,2,0],cart_dock=[-.27,2,0]),
                      pose_bodies=dict(pickup='robot_base',robot_dock='robot_base',cart_dock='cart_body'),
                      stocking_stations=[dict(pose=[3,4,0],pose_body='robot_base',slots=[dict(id='shelf')])])
        poses=poses_for_job(manifest,Job('j','milk',1,'back','shelf'))
        self.assertEqual(poses['cart_dock'],[1,2,0])
        self.assertNotEqual(poses['cart_dock'],manifest['poses']['cart_dock'])



class StallGuardTests(unittest.TestCase):
    """Fail-fast: a stalled collect/transfer reach reaches terminal in seconds (sim-clock
    window) instead of spinning 'running' until the slower host sensor-stream watchdog."""

    def _packet(self, obs, t):
        return Packet(obs, 'test_fixture', t, None, None, None, None, True, False, 0., True)

    def _sensing_controller(self, *, phase='observe', advance_sensor):
        from twin.restock_manipulation.types import Result as MResult
        controller = NS(cart_location='rollwagen', last_sensor_time=-1.,
                        start=lambda a, j: None, warm=lambda sim, parked=None: True,
                        stop=lambda: None)

        def step(dt, sim, parked=None):
            if advance_sensor:
                controller.last_sensor_time = sim   # a fresh /sensors frame accepted this tick
            return MResult('running', phase, '', (), ())
        controller.step = step
        return controller

    def _collect_adapter(self, controller, *, stall_window_s=20.):
        manip = ManipulationBridge(controller, owns_sensor_io=True)
        nav = NS(start=lambda *a, **k: None, cancel=lambda: None)
        mobile = NS(park=lambda: None, hold=lambda: None, release=lambda: None)
        self._pkt = [self._packet('reach-0', 0.)]
        port = NS(advance=lambda dt: self._pkt[0], controls=lambda c: None, stop=lambda: None)
        poses = {k: [0, 0, 0] for k in ('home', 'pickup', 'cart_dock', 'shelf_approach')}
        adapter = RuntimeAdapter(nav, mobile, manip, port, poses,
                                 fixture_mode=True, stall_window_s=stall_window_s)
        adapter.start('collect', Job('j', 'milk', 1, 'back', 'shelf'))
        adapter.phase, adapter.action = 'execute', 'collect'   # drive the reach branch directly
        adapter._reach_start = None
        return adapter, port

    def _drive(self, adapter, steps, dt=.1):
        t = adapter.time
        for i in range(steps):
            t = round(t + dt, 6)
            self._pkt[0] = self._packet(f'reach-{i}', t)
            result = adapter.step(dt)
            if result.status != 'running':
                return result, i
        return result, steps

    def test_fails_fast_when_sensors_stall_in_observe(self):
        adapter, _ = self._collect_adapter(self._sensing_controller(advance_sensor=False))
        result, i = self._drive(adapter, steps=400)             # 20 s window / 0.1 ~ step 200
        self.assertEqual(result.status, 'failed')
        self.assertIn('stall', result.reason)
        self.assertTrue(190 < i < 230, i)                       # fails just past the 20 s window

    def test_keeps_running_while_sensors_flow(self):
        adapter, _ = self._collect_adapter(self._sensing_controller(advance_sensor=True))
        result, i = self._drive(adapter, steps=400)
        self.assertEqual(result.status, 'running')
        self.assertEqual(i, 400)

    def test_collect_freshness_checked_on_sim_clock_not_wall(self):
        # fm346 / freshness-sim-clock-not-wall (run-272 regression guard): the collect/transfer reach
        # re-checks p.manipulation.valid_until_steady -- a SIM-clock deadline -- and MUST use the sim
        # clock (packet.simulation_time), never wall time.monotonic. Before the fix the adapter passed
        # time.monotonic (a different epoch, seconds-since-boot), so the check tripped the instant the
        # place first set valid_until_steady with a passing status. Case A proves a sim-FRESH deadline
        # passes even though wall is far ahead (a revert to the wall clock would fail this); Case B
        # proves a sim-STALE deadline still fails fast with the freshness reason.
        adapter, _ = self._collect_adapter(self._sensing_controller(advance_sensor=True))
        t = round(adapter.time + .1, 6)
        self._pkt[0] = Packet('fresh-a', 'test_fixture', t, None, None, None,
                              NS(valid_until_steady=t + 1.0), True, False, 0., True)
        self.assertEqual(adapter.step(.1).status, 'running')   # sim-fresh -> passes (wall is huge)
        t = round(adapter.time + .1, 6)
        self._pkt[0] = Packet('stale-b', 'test_fixture', t, None, None, None,
                              NS(valid_until_steady=t - 1.0), True, False, 0., True)
        r = adapter.step(.1)                                   # sim-stale -> fails fast
        self.assertEqual(r.status, 'failed')
        self.assertIn('freshness', r.reason)

    def test_scoped_to_observe_and_approach(self):
        # contact/lift (arm grasping) is out of scope; host watchdog backstops it.
        adapter, _ = self._collect_adapter(self._sensing_controller(phase='contact', advance_sensor=False))
        result, i = self._drive(adapter, steps=400)
        self.assertEqual(result.status, 'running')
        self.assertEqual(i, 400)

    def test_inactive_without_a_sensing_controller(self):
        controller = NS(cart_location='rollwagen', start=lambda a, j: None,
                        warm=lambda sim, parked=None: True, stop=lambda: None,
                        step=lambda dt, sim, parked=None: __import__(
                            'twin.restock_manipulation.types', fromlist=['Result']
                        ).Result('running', 'observe', '', (), ()))
        adapter, _ = self._collect_adapter(controller)
        result, i = self._drive(adapter, steps=300)
        self.assertEqual(result.status, 'running')
        self.assertEqual(i, 300)

    def test_window_none_disables_the_guard(self):
        adapter, _ = self._collect_adapter(self._sensing_controller(advance_sensor=False),
                                           stall_window_s=None)
        result, i = self._drive(adapter, steps=300)
        self.assertEqual(result.status, 'running')
        self.assertEqual(i, 300)
