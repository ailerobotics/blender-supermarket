"""CPU contract tests. The generated robot here is explicitly a test fixture."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as S
import tempfile
import unittest
from unittest.mock import patch

import yaml

from twin.restock_host.config import (ENDPOINTS, ENV, FRAME, JOINTS, NS,
                                      isolated_environment, launch_plan)
from twin.restock_host.preflight import dependencies, incorporated_time, link_errors
from twin.restock_host.readiness import missing_endpoints
from twin.restock_host.evidence import COVERAGE_ABS_TOL, _assert_equivalent


def fixture(directory):
    directory = Path(directory)
    (directory / 'shelfbot/perception').mkdir(parents=True)
    for name in ('object_segmenter', 'depth_mask'):
        (directory / f'shelfbot/perception/{name}.py').write_text('# test fixture, never executed\n')
    links = ['base_link', 'arm_mount', FRAME, 'suction_cups', 'gripper_pad_tcp']
    links += ['link' + str(i) for i in range(6)]
    joints = []
    previous = FRAME
    for i, name in enumerate(JOINTS):
        child = 'link' + str(i)
        joints.append(f'<joint name="{name}" type="revolute"><parent link="{previous}"/>'
                      f'<child link="{child}"/><origin xyz="0.1 0 0"/><axis xyz="0 0 1"/>'
                      '<limit lower="-3" upper="3" velocity="1" effort="1"/></joint>')
        previous = child
    for name, parent, child in [('mount', 'base_link', 'arm_mount'),
                                ('arm', 'arm_mount', FRAME),
                                ('tool', previous, 'suction_cups'),
                                ('tcp', 'suction_cups', 'gripper_pad_tcp')]:
        joints.append(f'<joint name="{name}" type="fixed"><parent link="{parent}"/><child link="{child}"/></joint>')
    (directory / 'robot.urdf').write_text('<robot name="test_fixture">' +
        ''.join(f'<link name="{n}"><collision><geometry><sphere radius="0.005"/></geometry></collision></link>'
                for n in links) + ''.join(joints) + '</robot>')
    (directory / 'mobile.json').write_text(json.dumps({'transport_positions': dict.fromkeys(JOINTS, 0.)}))
    from twin.restock_host.generate_assets import generate
    generate(directory / 'robot.urdf', directory / 'mobile.json', directory, directory / 'generated')
    return {name: str(directory / 'generated' / ('mobile.' + name)) for name in ('urdf', 'srdf', 'xrdf')} | {
        'workspace_min': [-2., -2., -1.], 'workspace_max': [2., 2., 2.],
        'shelfbot_root': str(directory), 'mobile_manifest': str(directory / 'mobile.json'),
        'asset_report': str(directory / 'generated/asset-report.json')}


class HostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = fixture(self.temp.name)

    def test_domain_and_loopback(self):
        self.assertEqual(isolated_environment({}), ENV)
        for key, value in [('ROS_DOMAIN_ID', '0'), ('ROS_DOMAIN_ID', '42'),
                           ('ROS_STATIC_PEERS', 'robot'), ('ROS_LOCALHOST_ONLY', '0'),
                           ('ROS_DISCOVERY_SERVER', 'server'), ('CYCLONEDDS_URI', 'file'),
                           ('FASTDDS_DEFAULT_PROFILES_FILE', 'file')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                isolated_environment({key: value})

    def test_names_frames_and_no_hardware(self):
        plan = launch_plan(self.config)
        self.assertEqual(plan['frame'], FRAME)
        self.assertEqual(set(plan['endpoints']), {NS + '/' + n for n in ENDPOINTS})
        self.assertIn('moveit_ros_move_group/move_group', plan['nodes'])
        self.assertFalse(any('driver' in n or 'fake' in n for n in plan['nodes']))
        self.assertIn('sensor-derived', ' '.join(plan['required_external']))
        self.assertIn('--mobile-manifest', plan['integration_host_arguments'])

    def test_manifest_stow_missing_or_out_of_bounds(self):
        path = Path(self.config['mobile_manifest'])
        for positions in ({}, dict.fromkeys(JOINTS, 100.)):
            path.write_text(json.dumps({'transport_positions': positions}))
            with self.assertRaisesRegex(ValueError, 'transport_positions'):
                launch_plan(self.config)

    def test_reject_unsafe_config(self):
        for key, value in [('domain', 42), ('namespace', '/'), ('tcp', 'tool0'),
                           ('frame', 'world'), ('workspace_max', [float('nan')] * 3),
                           ('urdf', '/nonexistent/mobile.urdf'), ('unknown', 1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch_plan(self.config | {key: value})

    def test_reject_placeholder_and_missing_self_collision(self):
        path = Path(self.config['xrdf'])
        original = yaml.safe_load(path.read_text())
        for section in ('collision', 'self_collision'):
            invalid = copy.deepcopy(original)
            invalid[section] = {'geometry': []}
            path.write_text(yaml.safe_dump(invalid))
            with self.assertRaisesRegex(ValueError, section):
                launch_plan(self.config)
        invalid = copy.deepcopy(original)
        del invalid['geometry']['mobile_cad']['spheres']['suction_cups']
        path.write_text(yaml.safe_dump(invalid))
        with self.assertRaisesRegex(ValueError, 'suction_cups'):
            launch_plan(self.config)

    def test_reject_bad_srdf_and_chain(self):
        path = Path(self.config['srdf'])
        path.write_text(path.read_text().replace('ur_manipulator', 'manipulator'))
        with self.assertRaisesRegex(ValueError, 'ur_manipulator'):
            launch_plan(self.config)

    def test_exact_dependency_failures(self):
        def unavailable(package):
            raise LookupError(package)
        result = dependencies(unavailable)
        self.assertTrue(any('isaac_ros_cumotion:' in e and 'task-local overlay' in e for e in result['errors']))
        self.assertTrue(any('nvblox_ros:' in e for e in result['errors']))

    def test_missing_elf_dependency_fails_preflight(self):
        with patch('twin.restock_host.preflight.subprocess.run',
                   return_value=S(stdout='librequired.so => not found\n', stderr='', returncode=0)):
            self.assertIn('librequired.so', link_errors('/fixture/component.so')[0])

    def test_asset_diagnostics_are_hash_bound_and_block_known_overlaps(self):
        path = Path(self.config['asset_report'])
        report = json.loads(path.read_text())
        original = path.read_text()
        report['transport_sphere_overlaps'] = [{'links': ['a', 'b']}]
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'recomputed'):
            launch_plan(self.config)
        path.write_text(original)
        Path(self.config['srdf']).write_text(Path(self.config['srdf']).read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            launch_plan(self.config)

    def test_xrdf_default_joint_positions_are_cspace_only(self):
        # cuMotion/cuRobo prunes off-arm-path branches (e.g. the mobile base's two-joint
        # caster chains swivel->fork->roll->wheel); a default position for a pruned joint
        # aborts the planner at load. The generated xrdf must name ONLY the six cspace
        # arm joints in default_joint_positions, while still covering the base link with
        # collision spheres. Build a robot with an extra continuous base chain and check.
        import yaml
        from twin.restock_host.generate_assets import generate
        d = Path(self.temp.name) / 'wheeled'
        d.mkdir()
        links = ['base_link', 'arm_mount', FRAME, 'suction_cups', 'gripper_pad_tcp',
                 'caster_fork', 'caster_wheel'] + ['link' + str(i) for i in range(6)]
        joints, previous = [], FRAME
        for i, name in enumerate(JOINTS):
            joints.append(f'<joint name="{name}" type="revolute"><parent link="{previous}"/>'
                          f'<child link="link{i}"/><origin xyz="0.1 0 0"/><axis xyz="0 0 1"/>'
                          '<limit lower="-3" upper="3" velocity="1" effort="1"/></joint>')
            previous = 'link' + str(i)
        for name, parent, child in [('mount', 'base_link', 'arm_mount'), ('arm', 'arm_mount', FRAME),
                                    ('tool', previous, 'suction_cups'), ('tcp', 'suction_cups', 'gripper_pad_tcp')]:
            joints.append(f'<joint name="{name}" type="fixed"><parent link="{parent}"/><child link="{child}"/></joint>')
        # Two-joint continuous caster chain off the base, like the real mobile CAD.
        joints.append('<joint name="caster_swivel" type="continuous"><parent link="base_link"/>'
                      '<child link="caster_fork"/><axis xyz="0 0 1"/></joint>')
        joints.append('<joint name="caster_roll" type="continuous"><parent link="caster_fork"/>'
                      '<child link="caster_wheel"/><axis xyz="0 1 0"/></joint>')
        (d / 'robot.urdf').write_text('<robot name="wheeled">' + ''.join(
            f'<link name="{n}"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>'
            for n in links) + ''.join(joints) + '</robot>')
        (d / 'mobile.json').write_text(json.dumps({'transport_positions': dict.fromkeys(JOINTS, 0.)}))
        generate(d / 'robot.urdf', d / 'mobile.json', d, d / 'out')
        xrdf = yaml.safe_load((d / 'out' / 'mobile.xrdf').read_text())
        self.assertEqual(list(xrdf['default_joint_positions']), list(JOINTS))
        self.assertNotIn('caster_roll', xrdf['default_joint_positions'])
        # The pruned base links are still covered by collision spheres.
        self.assertTrue(xrdf['geometry']['mobile_cad']['spheres'].get('caster_wheel'))

    def test_asset_report_tolerates_env_float_noise_but_not_real_change(self):
        # The recompute is numerically env-dependent (scipy qhull/KDTree): a float
        # leaf may drift at ~1e-17. A drift within tolerance must still validate; a
        # physically meaningful change to the same float must still be rejected.
        path = Path(self.config['asset_report'])
        report = json.loads(path.read_text())
        element = report['collision_elements'][0]
        original_value = element['local_min'][0]
        element['local_min'][0] = original_value + COVERAGE_ABS_TOL / 100.0
        path.write_text(json.dumps(report))
        launch_plan(self.config)  # within tolerance -> still validates under launch
        element['local_min'][0] = original_value + 1e-3  # 1 mm: a real geometric change
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            launch_plan(self.config)

    def test_equivalent_tolerates_floats_but_rejects_discrete_forgery(self):
        base = {'count': 3, 'passed': True, 'reason': 'AdjacentKinematicAssemblies',
                'pen': [0.015083112454027665], 'bounds': {'lo': -0.4, 'hi': 0.4}}
        _assert_equivalent(dict(base), dict(base))  # identical
        noisy = json.loads(json.dumps(base)); noisy['pen'][0] += 2.8e-17
        _assert_equivalent(noisy, base)  # machine-epsilon float drift tolerated
        for forged, pattern in (
                ({**base, 'pen': [0.015083112454027665 + 1e-3]}, 'tolerance'),  # real float change
                ({**base, 'count': 4}, 'differs'),                              # count forged
                ({**base, 'passed': False}, 'verdict'),                         # verdict flipped
                ({**base, 'reason': 'SameRigidAssembly'}, 'differs'),           # exclusion forged
                ({**base, 'extra': 1}, 'structure'),                           # key added
                ({**base, 'pen': [0.01, 0.02]}, 'structure')):                  # list length
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                _assert_equivalent(forged, base)

    @unittest.skipUnless(os.environ.get('RESTOCK_HOST_VALIDATE_INSTALLED') == '1',
                         'explicit CPU-only installed-image validation')
    def test_installed_launch_description_without_starting(self):
        from twin.restock_host.launch import description
        result = dependencies()
        self.assertEqual(result['errors'], [])
        launch = description(self.config)
        self.assertEqual(len(launch.entities), 6)

    def test_graph_requires_exact_types_and_all_servers(self):
        services = [(NS + '/' + k, [v]) for k, v in ENDPOINTS.items() if '/srv/' in v]
        actions = [(NS + '/' + k, [v]) for k, v in ENDPOINTS.items() if '/action/' in v]
        self.assertEqual(missing_endpoints(services, actions), [])
        self.assertTrue(missing_endpoints(actions, services))
        self.assertTrue(any('attach_object' in e for e in missing_endpoints(services, [])))

    def test_sensor_incorporation_not_receive_time(self):
        response = S(success=True, header=S(frame_id=FRAME, stamp=S(sec=10, nanosec=100000000)),
                     esdf_and_gradients=S(data=[1., 0., 0., 0.]), voxel_size_m=.02)
        self.assertAlmostEqual(incorporated_time(response, now=10.2, parked_since=10.), 10.1)
        for changes in ({'success': False}, {'voxel_size_m': 0.},
                        {'esdf_and_gradients': S(data=[])},
                        {'header': S(frame_id='map', stamp=response.header.stamp)}):
            with self.assertRaises(ValueError):
                incorporated_time(S(**(vars(response) | changes)), now=10.2, parked_since=10.)
        for now, parked in [(11., 10.), (10., 9.), (10.2, 10.15)]:
            with self.assertRaises(ValueError):
                incorporated_time(response, now=now, parked_since=parked)


if __name__ == '__main__':
    unittest.main()
