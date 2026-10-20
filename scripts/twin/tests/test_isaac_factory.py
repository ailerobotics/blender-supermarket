"""CPU contract tests for the manipulation bundle factory (manip_factory).

No Isaac/ROS: exercises parse_manipulation_config and the create() wiring via the
`_builder` seam. The injected runtime is a fixture; it never stands in for a physical
manipulation run, and the real CadRestockRuntime construction is not exercised here.
"""
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock_manipulation import isaac_factory
from twin.restock_manipulation.isaac_factory import parse_manipulation_config, create

EYE4 = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
JOB_SKU = 'juice_orange'


def manip_cfg(**over):
    m = {
        'profiles': [{'sku': JOB_SKU, 'dimensions': [.6, .4, .16],
                      'tcp_quaternion': [.5, .5, .5, .5], 'tcp_surface_offset': .0025,
                      'grasp_mode': 'side'}],
        'stations': {'source': {'slots': [{'id': 's0'}], 'base_from_camera': EYE4},
                     'shelf': {'slots': [{'id': 'd0'}], 'base_from_camera': EYE4}},
        'shelfbot_root': '/home/x/shelfbot_newgripper',
        'walls': {}, 'voxel_dims': [2.4, 2.4, 2.4], 'voxel_size': .02,
    }
    m.update(over)
    return {'manipulation': m}


class ParseTests(unittest.TestCase):
    def test_valid(self):
        c = parse_manipulation_config(manip_cfg(), JOB_SKU)
        self.assertIn(JOB_SKU, c.profiles)
        self.assertEqual(c.grasp_mode, 'side')
        self.assertEqual(c.profiles[JOB_SKU].dimensions, (.6, .4, .16))
        self.assertEqual(set(c.stations), {'source', 'shelf'})
        self.assertEqual(c.shelfbot_root, '/home/x/shelfbot_newgripper')

    def test_missing_manipulation(self):
        with self.assertRaises(ValueError):
            parse_manipulation_config({}, JOB_SKU)

    def test_cart_placement_absent_defaults_empty(self):
        c = parse_manipulation_config(manip_cfg(), JOB_SKU)
        self.assertEqual(c.cart_placement, {})

    def test_cart_placement_parsed(self):
        cfg = manip_cfg(cart_placement={'location': 'rollwagen', 'load_center_body_xy': [0.08, 0],
                                        'support_z_world': 0.225, 'free_dimensions': [0.65, 0.45, 0.30]})
        c = parse_manipulation_config(cfg, JOB_SKU)
        self.assertEqual(c.cart_placement['location'], 'rollwagen')
        self.assertEqual(c.cart_placement['load_center_body_xy'], (0.08, 0.0))
        self.assertEqual(c.cart_placement['support_z_world'], 0.225)
        self.assertEqual(c.cart_placement['free_dimensions'], (0.65, 0.45, 0.30))

    def test_station_heartbeat_absent_defaults_real_rig(self):
        # Real-rig default (10 s) when the sim profile does not relax it (firstmate 289 sweep).
        self.assertEqual(parse_manipulation_config(manip_cfg(), JOB_SKU).station_heartbeat_s, 10.0)

    def test_station_heartbeat_sim_profile_parsed(self):
        cfg = manip_cfg()
        cfg['station_heartbeat_s'] = 30.0
        self.assertEqual(parse_manipulation_config(cfg, JOB_SKU).station_heartbeat_s, 30.0)

    def test_arm_camera_reaim_absent_defaults_empty(self):
        self.assertEqual(parse_manipulation_config(manip_cfg(), JOB_SKU).arm_camera_reaim, {})

    def test_arm_camera_reaim_parsed(self):
        cfg = manip_cfg(arm_camera_reaim={'origin': [0.46, 0, 0.09], 'aim_targets': [[0.0, 1.101, 0.261]]})
        c = parse_manipulation_config(cfg, JOB_SKU)
        self.assertEqual(c.arm_camera_reaim['origin'], (0.46, 0.0, 0.09))
        self.assertEqual(c.arm_camera_reaim['aim_targets'], [(0.0, 1.101, 0.261)])

    def test_arm_camera_reaim_malformed_rejected(self):
        for bad in ({'origin': [0, 0]}, {'origin': [0, 0, 0], 'aim_targets': []},
                    {'origin': [0, 0, 0], 'aim_targets': [[0, 0]]}):
            with self.assertRaises(ValueError):
                parse_manipulation_config(manip_cfg(arm_camera_reaim=bad), JOB_SKU)

    def test_cart_placement_malformed_rejected(self):
        for bad in ({'location': 'rollwagen'},                       # missing fields
                    {'location': '', 'load_center_body_xy': [0, 0],  # empty location
                     'support_z_world': 0.2, 'free_dimensions': [1, 1, 1]},
                    {'location': 'c', 'load_center_body_xy': [0],     # wrong-length xy
                     'support_z_world': 0.2, 'free_dimensions': [1, 1, 1]},
                    {'location': 'c', 'load_center_body_xy': [0, 0],  # wrong-length dims
                     'support_z_world': 0.2, 'free_dimensions': [1, 1]}):
            with self.assertRaises(ValueError):
                parse_manipulation_config(manip_cfg(cart_placement=bad), JOB_SKU)

    def test_sku_without_profile(self):
        with self.assertRaises(ValueError):
            parse_manipulation_config(manip_cfg(), 'no_such_sku')

    def test_mixed_grasp_modes(self):
        cfg = manip_cfg()
        cfg['manipulation']['profiles'].append(
            {'sku': 'other', 'dimensions': [.1, .1, .1], 'tcp_quaternion': [1, 0, 0, 0], 'grasp_mode': 'top'})
        with self.assertRaises(ValueError):
            parse_manipulation_config(cfg, JOB_SKU)

    def test_missing_station(self):
        cfg = manip_cfg()
        del cfg['manipulation']['stations']['shelf']
        with self.assertRaises(ValueError):
            parse_manipulation_config(cfg, JOB_SKU)

    def test_bad_base_from_camera(self):
        cfg = manip_cfg()
        cfg['manipulation']['stations']['source']['base_from_camera'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        with self.assertRaises(ValueError):
            parse_manipulation_config(cfg, JOB_SKU)

    def test_empty_slots(self):
        cfg = manip_cfg()
        cfg['manipulation']['stations']['shelf']['slots'] = []
        with self.assertRaises(ValueError):
            parse_manipulation_config(cfg, JOB_SKU)

    def test_bad_voxel(self):
        with self.assertRaises(ValueError):
            parse_manipulation_config(manip_cfg(voxel_size=0.2), JOB_SKU)


class _StubRuntime:
    def __init__(self):
        self.controller = SimpleNamespace(cart_location='rollwagen')
        self.configured = []

    def configure_station(self, *, slots, base_from_camera):
        self.configured.append((slots, base_from_camera))


class CreateTests(unittest.TestCase):
    def _create(self, runtime):
        plan = SimpleNamespace(job_sku=JOB_SKU)
        built = {}

        def builder(*, app, assembly, params, output):
            built['params'] = params
            return runtime
        bridge = create(app=object(), assembly=object(), plan=plan, config=manip_cfg(),
                        output=Path('/tmp/x'), _builder=builder)
        return bridge, built

    def test_create_returns_sensor_owning_bridge(self):
        rt = _StubRuntime()
        bridge, built = self._create(rt)
        from twin.restock.runtime import ManipulationBridge
        self.assertIsInstance(bridge, ManipulationBridge)
        self.assertTrue(bridge.owns_sensor_io)
        self.assertEqual(bridge.cart_location, 'rollwagen')  # from controller.cart_location
        self.assertEqual(built['params'].grasp_mode, 'side')

    def test_prepare_station_selects_per_action(self):
        rt = _StubRuntime()
        bridge, _ = self._create(rt)
        bridge.prepare_station('collect', {'sku': JOB_SKU})
        bridge.prepare_station('transfer', {'sku': JOB_SKU})
        self.assertEqual([s for s, _ in rt.configured], [({'id': 's0'},), ({'id': 'd0'},)])
        with self.assertRaises(ValueError):
            bridge.prepare_station('park', {})


class HostRuntimeTests(unittest.TestCase):
    """The host cuMotion runtime: delegates to RosManipulationRuntime and exposes the
    base-parked / profile views the sensing lane reads. No in-process cuRobo."""

    def _host(self, art=None):
        from twin.restock_manipulation.isaac_factory import HostManipulationRuntime, _ArticulationParked
        calls = []
        inner = SimpleNamespace(
            start=lambda op, job: calls.append(('start', op)),
            step=lambda dt, t, parked=None: calls.append(('step', t)) or 'result',
            stop=lambda: calls.append(('stop',)))
        sensors = SimpleNamespace(slots=None, transform=None, last_frame='old')
        # fm411: parked() reads get_world_pose() displacement (wxyz quat), not instantaneous velocity.
        if art is None:
            art = SimpleNamespace(get_world_pose=lambda: ((0., 0., 0.), (1., 0., 0., 0.)))  # frozen -> parked
        rt = HostManipulationRuntime(runtime=inner, sensors=sensors,
                                     motion=_ArticulationParked(art), profiles={JOB_SKU: 'p'})
        return rt, calls, sensors

    def test_no_curobo_or_in_process_planner_imported(self):
        src = (Path(__file__).resolve().parents[2] /
               'twin/restock_manipulation/isaac_factory.py').read_text()
        self.assertNotIn('curobo', src)
        self.assertNotIn('CadRestockRuntime', src)
        self.assertNotIn('build_cad_planner', src)

    def test_exposes_parked_and_profiles_for_sensing_lane(self):
        rt, _, _ = self._host()                      # frozen pose -> parked
        self.assertEqual(rt.cart_location, 'rollwagen')
        self.assertTrue(rt.motion.parked())
        self.assertEqual(rt.controller.profiles, {JOB_SKU: 'p'})
        # fm411: a DRIVING base (pose advances each read) reads not-parked once the window shows net
        # displacement past tol -- the displacement gate, not the (bias-prone) instantaneous velocity.
        x = [0.0]
        def driving():
            x[0] += 0.05
            return ((x[0], 0., 0.), (1., 0., 0., 0.))
        moving, _, _ = self._host(art=SimpleNamespace(get_world_pose=driving))
        results = [moving.motion.parked() for _ in range(4)]
        self.assertFalse(all(results))      # a driving base is detected as NOT parked within a couple ticks
        self.assertFalse(results[-1])       # and stays not-parked while it keeps moving

    def test_delegates_and_reconfigures_station(self):
        rt, calls, sensors = self._host()
        rt.start('transfer', {'sku': JOB_SKU})
        self.assertEqual(rt.step(0.1, 2.0), 'result')
        rt.configure_station(slots=({'id': 'd0'},), base_from_camera=EYE4)
        rt.stop()
        self.assertEqual(calls, [('start', 'transfer'), ('step', 2.0), ('stop',)])
        self.assertEqual(sensors.slots, ({'id': 'd0'},))
        self.assertIsNone(sensors.last_frame)  # station change invalidates the stale frame

    def test_create_wraps_host_runtime_and_prepares_station(self):
        rt, _, sensors = self._host()
        plan = SimpleNamespace(job_sku=JOB_SKU)
        bridge = create(app=object(), assembly=object(), plan=plan, config=manip_cfg(),
                        output=Path('/tmp/x'), _builder=lambda **kw: rt)
        from twin.restock.runtime import ManipulationBridge
        self.assertIsInstance(bridge, ManipulationBridge)
        self.assertEqual(bridge.cart_location, 'rollwagen')
        bridge.prepare_station('transfer', {'sku': JOB_SKU})
        self.assertEqual(sensors.slots, ({'id': 'd0'},))


if __name__ == '__main__':
    unittest.main()
