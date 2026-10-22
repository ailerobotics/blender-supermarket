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

    def test_self_filter_delegated_to_isaac_masked_depth(self):
        # The NVIDIA RobotSegmenter is removed from the host graph (its incompatible reload
        # subscription aborted the container); the rig self mask is delegated to the Isaac
        # side, which publishes robot-masked depth on camera/masked_depth. The host keeps the
        # grasp-target object_segmenter and the nvblox mapper.
        plan = launch_plan(self.config)
        self.assertFalse(any('RobotSegmenter' in n or 'robot_segmenter' in n for n in plan['nodes']))
        self.assertIn('shelfbot.perception.object_segmenter', plan['nodes'])
        self.assertIn('nvblox_ros/NvbloxNode', plan['nodes'])
        external = ' '.join(plan['required_external'])
        self.assertIn('camera/masked_depth', external)
        self.assertIn('32FC1', external)

    def test_host_publishes_incorporation_ack_node(self):
        # The host now publishes the nvblox mapper-incorporation ack (collision_time) itself,
        # as an allowlisted co-launched node; the required-external note reflects that the
        # host publishes it (it is no longer a purely external responsibility).
        plan = launch_plan(self.config)
        self.assertIn('twin.restock_host.incorporation_ack', plan['nodes'])
        external = ' '.join(plan['required_external'])
        self.assertIn('collision_time', external)
        self.assertIn('station/status', external)

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
        from twin.urdf_probe import parse_urdf
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
        # The attached carton (added at runtime as link 'attached_object') must NOT be registered as a
        # cuMotion self-collision frame (firstmate sim-restock-planner-obstacles-v1, decision A1): no
        # add_frame modifier, no collision.buffer_distance slot, no self_collision.ignore row. Making
        # it a planner collision body (PR #139's add_frame) made the carry UNPLANNABLE (run-261: a
        # just-picked carton is born in collision with the source-shelf ESDF); without the frame its
        # runtime spheres stay orphaned and dropped, exactly as in the plannable run-260. The arm-vs-arm
        # carry stop is handled at the IK layer by the integration lane's caps, not the sphere model.
        from twin.restock_host.world_obstacles import (resolve_self_collision_masks,
                                                        simulate_object_attachment_reload)
        add_frames = {m['add_frame']['frame_name']: m['add_frame']
                      for m in xrdf['modifiers'] if isinstance(m, dict) and 'add_frame' in m}
        self.assertNotIn('attached_object', add_frames)
        self.assertNotIn('attached_object', xrdf['collision'].get('buffer_distance', {}))
        self.assertNotIn('attached_object', xrdf['self_collision']['ignore'])
        self.assertNotIn('attached_object', xrdf['geometry']['mobile_cad']['spheres'])
        # Exercise the SAME reload code path the node runs (UpdateRobotDescriptionWithSpheres: write
        # the carton spheres under geometry.spheres.attached_object, reload) and model cuMotion's mask
        # resolution. Because attached_object is NOT an add_frame kinematic frame, the reloaded carton
        # spheres are orphaned and every attached_object mask is SKIPPED -- the invisible-carton state
        # that let run-260 plan. (This is the deliberate A1 behaviour, not the run-256/259 bug.)
        urdf_links = set(parse_urdf(str(d / 'robot.urdf')).links)
        reloaded = simulate_object_attachment_reload(xrdf)
        masks = resolve_self_collision_masks(reloaded, urdf_links)
        self.assertEqual([p for p in masks['resolved'] if 'attached_object' in p], [],
                         'attached_object must not enter the self-collision set (A1: carton invisible)')

    def test_attached_object_excluded_from_static_model_and_overlap_gate(self):
        # The carried carton (attached_object) is dynamic (added at runtime) and, per decision A1, is
        # NOT a cuMotion collision frame -- so at generation it must NOT appear in the geometry, the
        # arm/body sphere_count, the covered-links model, or the static transport self-overlap gate.
        import yaml
        from twin.restock_host.generate_assets import build_assets
        source_urdf = Path(self.config['mobile_manifest']).parent / 'robot.urdf'
        report, artifacts = build_assets(source_urdf, self.config['mobile_manifest'],
                                         self.config['shelfbot_root'])
        xrdf = yaml.safe_load(artifacts['mobile.xrdf'])
        geom = xrdf['geometry']['mobile_cad']['spheres']
        self.assertNotIn('attached_object', geom)
        self.assertNotIn('attached_object', report['covered_links'])
        self.assertEqual(report['attached_object_registration'], 'none')
        # Not a planner collision frame: no add_frame modifier, no self-collision / buffer slot.
        self.assertNotIn('attached_object', [m.get('add_frame', {}).get('frame_name')
                                             for m in xrdf['modifiers'] if isinstance(m, dict)])
        self.assertNotIn('attached_object', xrdf['self_collision']['ignore'])
        self.assertNotIn('buffer_distance', xrdf['collision'])
        # sphere_count is exactly the emitted (arm/body) geometry: no attached_object contribution.
        self.assertEqual(report['sphere_count'], sum(len(v) for v in geom.values()))
        self.assertTrue(report['transport_collision_check_passed'])
        self.assertEqual(report['transport_sphere_overlaps'], [])

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
        # guards (one OnProcessExit shutdown guard per process) + processes + ack:
        # processes = [moveit, container, masker, manipulation, trajectory] = 5.
        self.assertEqual(len(launch.entities), 2 * 5 + 1)

    def test_mast_depth_off_by_default_single_feed(self):
        # Absent or disabled mast_depth: the plan reports a single nvblox feed and carries no
        # mast contract -- the single-feed default is unchanged.
        for cfg in (self.config, self.config | {'mast_depth': {'enabled': False}}):
            plan = launch_plan(cfg)
            self.assertFalse(plan['mast_depth']['enabled'])
            self.assertFalse(any('mast' in req for req in plan['required_external']))

    def test_mast_depth_switch_adds_second_feed_contract(self):
        from twin.restock_camera.mast_depth import nvblox_feed
        feed = nvblox_feed()
        plan = launch_plan(self.config | {'mast_depth': {'enabled': True}})
        self.assertTrue(plan['mast_depth']['enabled'])
        mast_reqs = [req for req in plan['required_external'] if 'mast_depth on' in req]
        self.assertEqual(len(mast_reqs), 1)
        # The Isaac-side contract names the mast masked-depth topic and frame the host wires
        # to nvblox camera_1, and states the carton is masked as a rig-instance subtree.
        self.assertIn(feed['masked_depth'], mast_reqs[0])
        self.assertIn(feed['frame_id'], mast_reqs[0])
        self.assertIn('camera_1', mast_reqs[0])
        self.assertIn('carton', mast_reqs[0])

    def test_reject_malformed_mast_depth(self):
        for bad in ({'enabled': 'yes'}, {'on': True}, [], {'enabled': True, 'extra': 1}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'mast_depth'):
                launch_plan(self.config | {'mast_depth': bad})

    @unittest.skipUnless(os.environ.get('RESTOCK_HOST_VALIDATE_INSTALLED') == '1',
                         'explicit CPU-only installed-image validation')
    def test_installed_mast_depth_adds_second_segmenter(self):
        # With the switch on, the launch gains exactly one extra process (the mast segmenter)
        # and its shutdown guard; off, it is the single-feed default (same baseline count).
        from twin.restock_host.launch import description
        self.assertEqual(dependencies()['errors'], [])
        baseline = description(self.config)
        enabled = description(self.config | {'mast_depth': {'enabled': True}})
        self.assertEqual(len(enabled.entities), len(baseline.entities) + 2)

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


class IncorporationAckTest(unittest.TestCase):
    """The host-published mapper-incorporation ack wire contract (collision_time)."""

    def state(self):
        from twin.restock_host.incorporation_ack import IncorporationAckState
        clock = [100.0]
        return IncorporationAckState(wall_clock=lambda: clock[0]), clock

    def prepared(self):
        st, clock = self.state()
        self.assertEqual(st.accept(dict(action='prepare', epoch=1, sequence=0, stamp=10.0,
                                        parked=True, camera_frame='restock_camera_collect'), 10.0), 'prepare')
        return st, clock

    def test_prepare_then_fresh_integration_becomes_ready(self):
        st, clock = self.prepared()
        # Before any acknowledgement: active but warming, collision_time withheld (-1).
        snap = st.snapshot(10.1, endpoints_ready=True, process_alive=True)
        self.assertEqual(snap['state'], 'warming')
        self.assertEqual(snap['collision_time'], -1.)
        self.assertEqual(snap['epoch'], 1)
        # A real NVIDIA integration stamp strictly after the parked epoch advances it.
        st.acknowledge(10.2, 1)
        snap = st.snapshot(10.3, endpoints_ready=True, process_alive=True)
        self.assertEqual(snap['state'], 'ready')
        self.assertAlmostEqual(snap['collision_time'], 10.2)
        self.assertAlmostEqual(snap['stamp'], 10.3)

    def test_collision_time_withheld_until_endpoints_and_freshness(self):
        st, _ = self.prepared()
        st.acknowledge(10.2, 1)
        # endpoints not ready -> not ready, collision_time withheld.
        self.assertEqual(st.snapshot(10.3, endpoints_ready=False, process_alive=True)['collision_time'], -1.)
        # stale incorporation (now - collision_time > 0.5) -> withheld, matching the
        # mission's 0 <= now - collision_time <= 0.5 freshness gate.
        self.assertEqual(st.snapshot(10.9, endpoints_ready=True, process_alive=True)['collision_time'], -1.)

    def test_postattach_esdf_kick_arms_on_phase_into_lift_fm430(self):
        # fm430 (2a): the post-attach ESDF re-integration kick arms once on the mission phase rising INTO
        # 'lift' (the move right after the grasp->weld), so the tick drops any in-flight pre-attach request
        # and integrates the post-attach ESDF a cycle sooner. Pure sequencing (ROS glue is separate).
        st, _ = self.prepared()
        st.kick_enabled = True
        self.assertFalse(st.note_manip_phase('grasp'))        # not lift -> no arm
        self.assertFalse(st.consume_kick())
        self.assertTrue(st.note_manip_phase('lift'))          # rising into lift -> arm
        self.assertTrue(st.consume_kick())                    # consumed once
        self.assertFalse(st.consume_kick())                   # one-shot
        self.assertFalse(st.note_manip_phase('lift'))         # steady lift -> no re-arm
        self.assertFalse(st.note_manip_phase('preplace'))     # off lift -> no arm
        self.assertTrue(st.note_manip_phase('lift'))          # rising into lift again -> re-arms

    def test_postattach_esdf_kick_default_off_never_arms_fm430(self):
        st, _ = self.prepared()
        self.assertFalse(st.kick_enabled)                     # default off (env unset)
        self.assertFalse(st.note_manip_phase('lift'))
        self.assertFalse(st.consume_kick())

    def test_postattach_esdf_kick_inactive_epoch_never_arms_fm430(self):
        st, _ = self.state()                                  # NOT prepared -> inactive
        st.kick_enabled = True
        self.assertFalse(st.note_manip_phase('lift'))         # no active parked epoch -> no arm
        self.assertFalse(st.consume_kick())

    def test_acknowledgement_monotonic_and_epoch_scoped(self):
        st, _ = self.prepared()
        st.acknowledge(10.2, 1)
        for bad in [(10.1, 1),   # goes backwards
                    (10.3, 2),   # wrong epoch
                    (9.5, 1)]:   # predates parked_since (10.0)
            with self.assertRaises(ValueError):
                st.acknowledge(*bad)
        st.acknowledge(10.4, 1)  # advancing is accepted
        self.assertAlmostEqual(st.collision_time, 10.4)

    def test_stop_and_new_epoch_reset_collision_time(self):
        st, _ = self.prepared()
        st.acknowledge(10.2, 1)
        st.accept(dict(action='stop', epoch=1, sequence=1, stamp=10.3, parked=True), 10.3)
        self.assertFalse(st.active)
        self.assertEqual(st.collision_time, -1.)
        self.assertEqual(st.snapshot(10.3, endpoints_ready=True, process_alive=True)['state'], 'idle')
        # A new parked epoch must re-earn its acknowledgement from zero.
        st.accept(dict(action='prepare', epoch=2, sequence=0, stamp=11.0, parked=True,
                       camera_frame='restock_camera_transfer'), 11.0)
        self.assertEqual(st.collision_time, -1.)

    def test_fault_names_process_lost(self):
        st, _ = self.prepared()
        st.acknowledge(10.2, 1)
        # process_alive False (the stack died) -> failed with the named condition, stop
        # asserted by the node. Thresholds unchanged; only the reason text is specific.
        snap = st.snapshot(10.3, endpoints_ready=True, process_alive=False)
        self.assertEqual(snap['state'], 'failed')
        self.assertEqual(snap['reason'], 'station process lost')

    def test_fault_names_sim_heartbeat_gap_with_value(self):
        st, _ = self.prepared()  # parked/last_heartbeat at sim 10.0, wall at 100.0
        st.acknowledge(10.2, 1)
        # sim gap now(10.73) - last_heartbeat(10.0) = 0.73 s > 0.5 s; wall gap still 0.
        snap = st.snapshot(10.73, endpoints_ready=True, process_alive=True)
        self.assertEqual(snap['state'], 'failed')
        self.assertEqual(snap['reason'], 'parked heartbeat lost: sim gap 0.73 s')

    def test_wall_guard_tolerates_render_paced_capture_tick(self):
        st, clock = self.prepared()
        st.acknowledge(10.2, 1)
        # A render-paced client blocks its heartbeat for a ~2 s capture tick; the 3 s wall
        # guard must NOT fault it while the sim gap stays within 0.5 s (render-paced /clock).
        clock[0] = 102.0
        snap = st.snapshot(10.2, endpoints_ready=True, process_alive=True)
        self.assertNotEqual(snap['state'], 'failed')
        self.assertEqual(snap['reason'], '')

    def test_fault_names_wall_heartbeat_gap_with_value(self):
        st, clock = self.prepared()
        st.acknowledge(10.2, 1)
        # A genuinely hung client (no tick completes) exceeds the 3 s wall guard while the
        # sim gap stays within 0.5 s, so the wall-heartbeat condition trips, named with value.
        clock[0] = 103.4
        snap = st.snapshot(10.2, endpoints_ready=True, process_alive=True)
        self.assertEqual(snap['state'], 'failed')
        self.assertEqual(snap['reason'], 'parked heartbeat lost: wall gap 3.40 s')

    def test_stopped_station_does_not_fault_across_a_long_gap_run309_replay(self):
        # fm399 (1) run-309 replay at the host-ack level: the host tick() publishes emergency stop on
        # every tick that state.fault is set, and an ACTIVE parked epoch faults on the 0.5 s sim gap
        # the instant it stops being heartbeat-ed -- which is what spammed /manipulation/stop through
        # the ~190 s-sim / ~1400 s-wall reposition drive. The fix CLEANLY STOPS the station before the
        # drive: a stopped (idle) epoch must NOT fault across that same long gap, so NO emergency stop.
        st, clock = self.prepared()
        st.acknowledge(10.2, 1)
        st.accept(dict(action='stop', epoch=1, sequence=1, stamp=10.3, parked=True), 10.3)
        clock[0] = 100.0 + 1400.0   # ~1400 s WALL later
        snap = st.snapshot(10.3 + 190.0, endpoints_ready=True, process_alive=True)  # ~190 s SIM later
        self.assertEqual(snap['state'], 'idle')          # idle, NOT failed -> tick() publishes no stop
        self.assertEqual(snap['reason'], '')
        # Contrast: an ACTIVE un-heartbeat-ed epoch DOES fault across the same gap (the run-309 trigger).
        st2, _ = self.prepared()
        st2.acknowledge(10.2, 1)
        self.assertEqual(st2.snapshot(10.3 + 190.0, endpoints_ready=True,
                                      process_alive=True)['state'], 'failed')

    def test_default_gap_budgets_are_byte_identical(self):
        # fm399 (2): unset env => the documented real-rig contract (0.5 s sim / 3.0 s wall).
        st, _ = self.state()
        self.assertEqual(st.sim_gap_s, .5)
        self.assertEqual(st.wall_gap_s, 3.)

    def test_sim_only_env_widens_gap_budgets_to_survive_render_paced_transit(self):
        # fm399 (2): a widened sim-gap budget lets a long render-paced reposition/re-incorporation
        # window pass without a spurious re-fault, while a dead host (process_alive False) is still
        # caught immediately. Clamped to the floor so it can never go BELOW the real-rig contract.
        from twin.restock_host.incorporation_ack import IncorporationAckState
        with patch.dict(os.environ, {'RESTOCK_STATION_SIM_GAP_S': '30',
                                           'RESTOCK_STATION_WALL_GAP_S': '45'}):
            st = IncorporationAckState(wall_clock=lambda: 100.0)
        self.assertEqual(st.sim_gap_s, 30.)
        self.assertEqual(st.wall_gap_s, 45.)
        st.accept(dict(action='prepare', epoch=1, sequence=0, stamp=10.0, parked=True,
                       camera_frame='restock_camera_collect'), 10.0)
        st.acknowledge(10.2, 1)
        # A 5 s sim gap that WOULD fault under the 0.5 s default is tolerated here.
        snap = st.snapshot(15.2, endpoints_ready=True, process_alive=True)
        self.assertNotEqual(snap['state'], 'failed')
        # A dead host is still caught regardless of the widened gap budgets.
        self.assertEqual(st.snapshot(15.3, endpoints_ready=True, process_alive=False)['state'], 'failed')

    def test_gap_budgets_clamped_to_real_rig_floor(self):
        # fm399 (2): a value below the real-rig contract is clamped up, never weakening the guard.
        from twin.restock_host.incorporation_ack import IncorporationAckState
        with patch.dict(os.environ, {'RESTOCK_STATION_SIM_GAP_S': '0.01',
                                           'RESTOCK_STATION_WALL_GAP_S': '0.1'}):
            st = IncorporationAckState(wall_clock=lambda: 100.0)
        self.assertEqual(st.sim_gap_s, .5)
        self.assertEqual(st.wall_gap_s, 3.)

    def test_fault_recovers_on_fresh_parked_epoch_bounded(self):
        from twin.restock_host.incorporation_ack import _MAX_RECOVERIES
        st, _ = self.prepared()  # epoch 1 at sim 10.0
        st.acknowledge(10.2, 1)
        # Lost process -> fault; a heartbeat against the dead epoch is still refused.
        self.assertEqual(st.snapshot(10.3, endpoints_ready=True, process_alive=False)['state'], 'failed')
        with self.assertRaisesRegex(RuntimeError, 'explicit recovery'):
            st.accept(dict(action='heartbeat', epoch=1, sequence=2, stamp=10.4, parked=True), 10.4)
        # A fresh measured parked epoch recovers the ack instead of rejecting it forever (ds-015).
        self.assertEqual(st.accept(dict(action='prepare', epoch=2, sequence=0, stamp=11.0,
                                        parked=True, camera_frame='restock_camera_transfer'), 11.0), 'prepare')
        self.assertIsNone(st.fault)
        self.assertEqual(st.recoveries, 1)
        self.assertEqual(st.snapshot(11.1, endpoints_ready=True, process_alive=True)['state'], 'warming')
        # Recovery is bounded: once the budget is spent a persistently broken mapper surfaces honestly.
        ep = 2
        while st.recoveries < _MAX_RECOVERIES:
            ep += 1
            self.assertEqual(st.snapshot(float(ep) - 0.5 + 10, endpoints_ready=True,
                                         process_alive=False)['state'], 'failed')
            st.accept(dict(action='prepare', epoch=ep, sequence=0, stamp=float(ep) + 10,
                           parked=True, camera_frame='restock_camera_transfer'), float(ep) + 10)
        self.assertEqual(st.snapshot(float(ep) + 9.6, endpoints_ready=True, process_alive=False)['state'], 'failed')
        with self.assertRaisesRegex(RuntimeError, 'recovery budget exhausted'):
            st.accept(dict(action='prepare', epoch=ep + 1, sequence=0, stamp=float(ep) + 11,
                           parked=True, camera_frame='restock_camera_transfer'), float(ep) + 11)

    def test_invalid_commands_rejected(self):
        st, _ = self.state()
        for bad in [dict(action='prepare', epoch=1, sequence=0, stamp=10.0, parked=True,
                         camera_frame='wrong_frame'),          # uncalibrated camera frame
                    dict(action='heartbeat', epoch=1, sequence=0, stamp=10.0, parked=True),  # no active epoch
                    dict(action='prepare', epoch=1, sequence=0, stamp=9.0, parked=True,
                         camera_frame='restock_camera_collect')]:  # stale stamp vs now
            with self.assertRaises((ValueError, RuntimeError)):
                st.accept(bad, 10.0)


class RosHostLaunchWiring(unittest.TestCase):
    """The mission host stack must launch ros_host (the manipulation bridge) so
    /restock_sim/manipulation/status is published; without it every manip job dies on the
    Isaac 5 s process/status watchdog. Verifies the launch wires ros_host with the verified
    {profiles, walls} fixture + the plan's integration_host_arguments, and the fixture is usable."""

    def test_launch_wires_ros_host_with_fixture_and_plan_args(self):
        src = (Path(__file__).resolve().parents[1]
               / 'restock_host' / 'launch.py').read_text()
        self.assertIn('twin.restock_manipulation.ros_host', src)
        self.assertIn('ros_host_config.json', src)
        self.assertIn("*plan['integration_host_arguments']", src)
        # It must be a shutdown-guarded process (in `processes`), not an auxiliary like the ack.
        self.assertIn('masker, manipulation, trajectory]', src)

    def test_launch_wires_follow_joint_trajectory_executor(self):
        # MoveIt is configured with a MoveItSimpleControllerManager pointing at
        # arm_controller/follow_joint_trajectory; the host must also LAUNCH that action
        # server, or ExecuteTrajectory aborts "action server not connected" after a
        # successful grasp plan and the host emergency-stops (run-85). The executor is the
        # host-owned twin.restock.trajectory_host, launched with the plan's generated urdf.
        src = (Path(__file__).resolve().parents[1]
               / 'restock_host' / 'launch.py').read_text()
        self.assertIn('twin.restock.trajectory_host', src)
        self.assertIn("'--urdf', robot['urdf']", src)
        # The plan lists it as a host-launched node, and no longer as a purely external one.
        with tempfile.TemporaryDirectory() as tmp:
            plan = launch_plan(fixture(tmp))
        self.assertIn('twin.restock.trajectory_host', plan['nodes'])

    def test_move_group_respawns_and_is_excluded_from_teardown_guard_fm404(self):
        # fm404 B-lite: a move_group SIGABRT (run-266/268/312) must RESPAWN, not tear the host stack
        # down. The move_group Node declares respawn=True + a respawn delay, and move_group is NOT in the
        # OnProcessExit shutdown-guard set (whose exits EmitEvent Shutdown), so its recoverable crash does
        # not SIGINT the rest of the stack; the other components stay genuinely fatal.
        src = (Path(__file__).resolve().parents[1] / 'restock_host' / 'launch.py').read_text()
        self.assertRegex(src, r"name='move_group'[^\n]*respawn=True")      # move_group respawns
        self.assertRegex(src, r"respawn_delay=[0-9.]+")                    # with a delay
        guarded_line = next(l for l in src.splitlines() if l.strip().startswith('guarded ='))
        self.assertNotIn('moveit', guarded_line)                          # move_group excluded from teardown
        for comp in ('container', 'masker', 'manipulation', 'trajectory'):
            self.assertIn(comp, guarded_line)                             # the genuinely-fatal set stays
        # The OnProcessExit guards are built over `guarded` (not all processes), so move_group's exit
        # does NOT EmitEvent Shutdown -- it respawns instead.
        self.assertRegex(src, r"for process in guarded")

    def test_ros_host_config_fixture_is_usable(self):
        from twin.restock_manipulation.types import ProductProfile
        cfg = json.loads((Path(__file__).resolve().parents[1]
                          / 'restock_manipulation' / 'ros_host_config.json').read_text())
        # pose_goal selects cuMotion task-space (pose) planning over joint-space; it is an
        # optional bool the host reads via config.get('pose_goal', False). arm_max_speed is the
        # SIM execution-profile joint-speed gate (rad/s) the host reads via config.get, raised above
        # the conservative real-rig default so the render-paced grasp completes in the wall budget.
        # The base keys plus the SIM render-pacing / cart-place profile knobs added across firstmate
        # 282..296 (all optional, host reads via config.get; real-rig defaults in code): the cart-place
        # policy + staged body-clear swing + pan/tuck shapes, the wall-window relaxations
        # (payload ack, plan cap) and the base-as-self-collision gate.
        self.assertEqual(set(cfg), {'profiles', 'walls', 'pose_goal', 'contact_straight_line',
                                    'contact_overpress_step_m', 'contact_overpress_cap_m',
                                    'arm_max_speed', 'startup_grace_seconds',
                                    'sim_cart_place_policy', 'payload_ack_wall_s', 'staged_cart_swing',
                                    'cart_pan_joint0', 'cart_tuck_shape', 'plan_wall_cap_s',
                                    'base_collision_object', 'reposition_before_place',
                                    # Cart-cage open-face place (host-config keys read by ros_host). run-279
                                    # dropped world_obstacles (the real beside-cart interleaves the arm, so a
                                    # solid obstacle breaks planning) and added the wrist_1-safe fixed
                                    # cage_preplace_shape (begin_joint_goal preplace below the self-collision stop).
                                    'cart_cage_place', 'cage_axis', 'cage_mouth_standoff',
                                    'cage_insert_gap', 'cage_preplace_shape',
                                    # fm402/fm403: the loaded-lift self-collision guards -- MUST be present
                                    # (the preflight baseline enforces them). carry_wrist1_cap + the
                                    # clear-region joint goal carry_lift_shape + the wrist_2 lower bound
                                    # keep the lift off the wrist_2-vs-forearm stop (runs 300d/305/311).
                                    'carry_wrist1_cap', 'carry_lift_shape', 'carry_wrist2_min',
                                    # fm410: physics grasp-confirm (lift-probe carton-tracking) threshold --
                                    # also preflight-baseline-enforced so it can never silently de-activate
                                    # back to the camera-dependent confirm that stalled run-316.
                                    'grasp_track_m',
                                    # fm449: gate the collect observe->reach on a clean source frame + decoded
                                    # label at park (Sensors.collect_source_clean, set Isaac-side). Optional,
                                    # host reads via config.get; default False => byte-identical.
                                    'require_collect_source_evidence',
                                    # fm461 (demo fallback): the controller accepts a GROUND-TRUTH bay transfer
                                    # target + ground-truth verify when the shelf is not camera-commissioned.
                                    # Inert without a ground-truth provider (runtime sensing); default path unchanged.
                                    'transfer_ground_truth'})
        self.assertIsInstance(cfg['carry_wrist1_cap'], (int, float))
        self.assertLessEqual(cfg['carry_wrist1_cap'], 0.65)   # at/below the wrist_2-vs-forearm stop
        self.assertEqual(len(cfg['carry_lift_shape']), 6)     # full 6-joint clear-region lift goal
        self.assertLessEqual(cfg['carry_lift_shape'][3], 0.65)  # wrist_1 at/below the stop
        self.assertGreaterEqual(cfg['carry_lift_shape'][4], 0.0)  # wrist_2 on the clear (positive) side
        self.assertIsInstance(cfg['carry_wrist2_min'], (int, float))
        self.assertIsInstance(cfg['grasp_track_m'], (int, float))
        self.assertGreater(cfg['grasp_track_m'], 0.0)         # a positive rise threshold (metres)
        self.assertGreaterEqual(cfg['carry_wrist2_min'], 0.0)  # wrist_2 lower bound on the clear side
        for k in ('sim_cart_place_policy', 'staged_cart_swing', 'base_collision_object', 'cart_cage_place',
                  'reposition_before_place'):
            self.assertIsInstance(cfg[k], bool)
        for k in ('payload_ack_wall_s', 'cart_pan_joint0', 'plan_wall_cap_s'):
            self.assertIsInstance(cfg[k], (int, float))
        self.assertEqual(len(cfg['cart_tuck_shape']), 5)   # arm joints 1-5 of the high carry pose
        self.assertEqual(len(cfg['cage_preplace_shape']), 6)  # full 6-joint begin_joint_goal preplace target
        self.assertLessEqual(cfg['cage_preplace_shape'][3], 0.65)  # wrist_1 at/below the self-collision stop
        self.assertIsInstance(cfg['pose_goal'], bool)
        # contact_straight_line opts the contact descent into the host-executor straight-line path
        # (bypassing the collision planner for that move only); optional bool read via config.get.
        self.assertIsInstance(cfg['contact_straight_line'], bool)
        # The sim contact-feedback descent steps the pad toward the carton by step_m (bounded by cap_m)
        # until the seal latches, for ~0 mm contact (the SurfaceGripper seals only on contact); the real
        # rig leaves both 0 and uses a single collision-planned move. Non-negative floats, step <= cap.
        for k in ('contact_overpress_step_m', 'contact_overpress_cap_m'):
            self.assertIsInstance(cfg[k], (int, float))
            self.assertGreaterEqual(cfg[k], 0)
        self.assertLessEqual(cfg['contact_overpress_step_m'], cfg['contact_overpress_cap_m'])
        self.assertIsInstance(cfg['arm_max_speed'], (int, float))
        self.assertGreater(cfg['arm_max_speed'], 0)
        # Startup grace (first-frame wall window) is widened for render pacing; a positive float.
        self.assertIsInstance(cfg['startup_grace_seconds'], (int, float))
        self.assertGreater(cfg['startup_grace_seconds'], 0)
        # Every profile constructs; walls cover source (backstock_00), cart (rollwagen) and
        # target (G3_back_s2) so WallCartonPolicy.build never raises on a missing location.
        self.assertTrue({p['sku']: ProductProfile(**p) for p in cfg['profiles']})
        for location in ('backstock_00', 'rollwagen', 'G3_back_s2'):
            self.assertIn(location, cfg['walls'])


class IncorporationAckLivenessOrderingTest(unittest.TestCase):
    """sim-robust-collect-place-v1: the host tick() now publishes the station liveness ack BEFORE its
    ESDF-service maintenance, so a cuMotion replan storm saturating the shared get_esdf_and_gradient
    during a long place press cannot delay the heartbeat (collectsrc 30 s station-heartbeat gap). The
    reorder is correct iff these pure-state invariants hold: the liveness snapshot is computable WITHOUT
    a fresh integration, a stale/contended ESDF leaves a LIVE (non-failed) heartbeat, and readiness
    still advances ONLY through a real integration -- so a genuinely dead host is never masked."""

    def prepared(self):
        from twin.restock_host.incorporation_ack import IncorporationAckState
        clock = [100.0]
        st = IncorporationAckState(wall_clock=lambda: clock[0])
        self.assertEqual(st.accept(dict(action='prepare', epoch=1, sequence=0, stamp=10.0,
                                        parked=True, camera_frame='restock_camera_collect'), 10.0), 'prepare')
        return st, clock

    def test_liveness_heartbeat_without_any_integration(self):
        # No ESDF result consumed yet (the maintenance runs AFTER publish): the snapshot is still a live,
        # fresh-stamped heartbeat -- active, warming, not failed -- so the mission keeps seeing the host.
        st, _ = self.prepared()
        snap = st.snapshot(10.1, endpoints_ready=True, process_alive=True)
        self.assertEqual(snap['state'], 'warming')
        self.assertEqual(snap['reason'], '')
        self.assertAlmostEqual(snap['stamp'], 10.1)      # advancing stamp == a live heartbeat
        self.assertEqual(snap['collision_time'], -1.)    # but readiness withheld (no real integration)

    def test_stale_integration_keeps_heartbeat_but_not_ready(self):
        # A planner-contended ESDF means the last integration is now stale (> 0.5 s old). Readiness drops
        # (collision_time withheld) but the station STILL heartbeats (not failed): the press does not
        # starve liveness. Heartbeat gaps (sim/wall) stay within budget here, so no fault.
        st, _ = self.prepared()
        st.acknowledge(10.2, 1)
        st.accept(dict(action='heartbeat', epoch=1, sequence=1, stamp=10.9, parked=True), 10.9)
        snap = st.snapshot(11.0, endpoints_ready=True, process_alive=True)   # now - collision_time = 0.8 > 0.5
        self.assertNotEqual(snap['state'], 'failed')     # heartbeat alive through a stale/contended ESDF
        self.assertEqual(snap['state'], 'warming')
        self.assertEqual(snap['collision_time'], -1.)    # not ready: the stale integration is withheld

    def test_readiness_only_from_a_real_fresh_integration(self):
        # No faking: ready requires a real integration stamp strictly after parked_since AND fresh.
        st, _ = self.prepared()
        snap = st.snapshot(10.1, endpoints_ready=True, process_alive=True)
        self.assertNotEqual(snap['state'], 'ready')
        st.acknowledge(10.2, 1)
        snap = st.snapshot(10.3, endpoints_ready=True, process_alive=True)
        self.assertEqual(snap['state'], 'ready')
        self.assertAlmostEqual(snap['collision_time'], 10.2)

    def test_dead_process_still_goes_silent_not_masked(self):
        # The reorder must not mask a real death: process_alive False still faults 'station process lost'.
        st, _ = self.prepared()
        st.acknowledge(10.2, 1)
        snap = st.snapshot(10.3, endpoints_ready=True, process_alive=False)
        self.assertEqual(snap['state'], 'failed')
        self.assertEqual(snap['reason'], 'station process lost')


if __name__ == '__main__':
    unittest.main()
