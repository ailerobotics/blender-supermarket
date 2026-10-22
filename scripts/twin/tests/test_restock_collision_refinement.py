"""Conservative coverage, all-motion proofs, and evidence-tampering regressions."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from twin.urdf_probe import Geometry, Joint, Link, Robot, rpy_to_matrix
from twin.restock_host.coverage import VolumeCover, transport_overlaps
from twin.restock_host.separation import prove_separation
from twin.restock_host.config import launch_plan
from twin.restock_host.generate_assets import build_assets, generate
from scripts.twin.tests.test_restock_host import fixture


def assert_covered(test, points, spheres):
    centers = np.array([s['center'] for s in spheres])
    radii = np.array([s['radius'] for s in spheres])
    for start in range(0, len(points), 128):
        residual = np.linalg.norm(points[start:start+128, None] - centers[None, :], axis=2) - radii
        test.assertLessEqual(float(residual.min(axis=1).max()), 1e-8)


class CoverageTests(unittest.TestCase):
    def test_rotated_box_partition_and_interior_survive_local_refinement(self):
        g = Geometry('box', box_size=(.12, .04, .008), origin_xyz=(.2, -.1, .3), origin_rpy=(.2, .4, .8))
        cover = VolumeCover(g, None, .04)
        before = sum(np.prod(hi-lo) for lo, hi in cover.leaves.values())
        cover.refine([0], .005)
        after = sum(np.prod(hi-lo) for lo, hi in cover.leaves.values())
        self.assertAlmostEqual(before, np.prod(g.box_size))
        self.assertAlmostEqual(after, before)
        rng = np.random.default_rng(2)
        local = (rng.random((1000, 3))-.5)*g.box_size
        points = local @ rpy_to_matrix(g.origin_rpy).T + g.origin_xyz
        assert_covered(self, points, cover.result()[0])

    def test_cylinder_solid_including_caps_is_covered(self):
        g = Geometry('cylinder', radius=.065, length=.035, origin_rpy=(np.pi/2, 0., 0.))
        cover = VolumeCover(g, None, .015)
        radius, angle, z = np.meshgrid(np.linspace(0, g.radius, 10),
                                      np.linspace(0, 2*np.pi, 30),
                                      np.linspace(-g.length/2, g.length/2, 9), indexing='ij')
        points = np.stack((radius*np.cos(angle), radius*np.sin(angle), z), -1).reshape(-1, 3)
        assert_covered(self, points @ rpy_to_matrix(g.origin_rpy).T, cover.result()[0])
        self.assertGreater(cover.discarded, 0)

    def test_mesh_hull_covers_interior_not_only_surface_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            mesh = Path(temp) / 'tetra.obj'
            mesh.write_text('v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')
            g = Geometry('mesh', mesh_filename=str(mesh), mesh_scale=(1., 2., .5), origin_xyz=(.1, 0., .2))
            cover = VolumeCover(g, lambda _: str(mesh), .025)
            cover.refine([0], .005)
            rng = np.random.default_rng(7)
            weights = rng.dirichlet(np.ones(4), size=1000)
            vertices = np.array([[0, 0, 0], [.1, 0, 0], [0, .2, 0], [0, 0, .05]])
            assert_covered(self, np.vstack((weights @ vertices, vertices)) + g.origin_xyz, cover.result()[0])
            self.assertGreater(cover.discarded, 0)

    def _refine_pair(self, offset):
        covers = {'a': VolumeCover(Geometry('box', box_size=(.04, .04, .02)), None, .02),
                  'b': VolumeCover(Geometry('box', box_size=(.04, .04, .02), origin_xyz=(0., 0., offset)), None, .02)}
        initial = None
        for _ in range(6):
            spheres = {name: cover.result()[0] for name, cover in covers.items()}
            overlaps, indices = transport_overlaps(spheres, {name: np.eye(4) for name in covers}, {}, with_conflicts=True)
            if initial is None:
                initial = overlaps
            count = sum(covers[name].refine(indices[name], .003) for name in covers)
            if not count:
                return initial, overlaps
        self.fail('bounded refinement did not terminate')

    def test_false_positive_refines_away_but_true_collision_remains(self):
        initial, final = self._refine_pair(.03)
        self.assertTrue(initial)
        self.assertFalse(final)
        _, collided = self._refine_pair(.015)
        self.assertTrue(collided)


class SeparationTests(unittest.TestCase):
    def test_sibling_prismatic_proof_covers_independent_full_strokes(self):
        robot = Robot('plates', {
            'root': Link('root'),
            'a': Link('a', collisions=[Geometry('box', box_size=(.04781, .098, .003))]),
            'b': Link('b', collisions=[Geometry('box', box_size=(.04781, .098, .003), origin_xyz=(-.048438, 0, 0))])},
            [Joint(name, 'prismatic', 'root', name, axis=(0., 1., 0.), lower=0., upper=.2) for name in ('a', 'b')])
        proof = prove_separation(robot, 'a', 'b', None)
        self.assertAlmostEqual(proof['clearance_m'], .000628)
        self.assertEqual(proof['ranges'], [[0., .2], [0., .2]])
        # Counterfactual: independent travel can now close the separating gap.
        robot.joints[1].axis = (1., 0., 0.)
        self.assertIsNone(prove_separation(robot, 'a', 'b', None))

    def test_caster_all_angle_proof_rejects_larger_intersecting_wheel(self):
        robot = Robot('caster', {'base': Link('base', collisions=[Geometry('box', box_size=(1., .6, .2))]),
                                'fork': Link('fork'),
                                'wheel': Link('wheel', collisions=[Geometry('cylinder', radius=.065, length=.035, origin_rpy=(np.pi/2, 0, 0))])},
            [Joint('swivel', 'continuous', 'base', 'fork', origin_xyz=(.38, 0., -.095), axis=(0., 0., 1.)),
             Joint('roll', 'continuous', 'fork', 'wheel', origin_xyz=(-.025, 0., -.08), axis=(0., 1., 0.))])
        proof = prove_separation(robot, 'base', 'wheel', None)
        self.assertAlmostEqual(proof['clearance_m'], .007685439910818687)
        larger = deepcopy(robot)
        larger.links['wheel'].collisions[0].radius = .08
        self.assertIsNone(prove_separation(larger, 'base', 'wheel', None))
        # The actual larger cylinder's top enters the chassis by 5 mm at zero angle.
        self.assertGreater(-.175 + .08, -.1)
        tilted = deepcopy(robot)
        tilted.joints[0].axis = (1., 0., 0.)
        self.assertIsNone(prove_separation(tilted, 'base', 'wheel', None))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = fixture(self.temp.name)

    def test_missing_and_empty_evidence_do_not_pass(self):
        config = dict(self.config)
        del config['asset_report']
        self.assertTrue(launch_plan(config)['static_errors'])
        Path(self.config['asset_report']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'unsupported/stale'):
            launch_plan(self.config)

    def test_stale_source_or_manifest_rejected_even_with_unchanged_outputs(self):
        for name in ('robot.urdf', 'mobile.json'):
            path = Path(self.temp.name) / name
            original = path.read_text()
            path.write_text(original + '\n')
            with self.assertRaisesRegex(ValueError, 'stale or missing source'):
                launch_plan(self.config)
            path.write_text(original)

    def test_rehashed_undercoverage_or_missing_proof_rejected(self):
        report_path = Path(self.config['asset_report'])
        original = report_path.read_text()
        report = json.loads(original)
        report['collision_elements'] = []
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'recomputed'):
            launch_plan(self.config)

        report = json.loads(original)
        xrdf_path = Path(self.config['xrdf'])
        xrdf = yaml.safe_load(xrdf_path.read_text())
        xrdf['geometry']['mobile_cad']['spheres']['suction_cups'][0]['radius'] *= .5
        xrdf_path.write_text(yaml.safe_dump(xrdf))
        report['generated_sha256']['mobile.xrdf'] = hashlib.sha256(xrdf_path.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'recomputed'):
            launch_plan(self.config)

    def test_changed_mesh_and_forged_exclusion_rejected(self):
        source = Path(self.temp.name) / 'robot.urdf'
        mesh = Path(self.temp.name) / 'collision.obj'
        mesh.write_text('v 0 0 0\nv .005 0 0\nv 0 .005 0\nv 0 0 .005\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')
        import xml.etree.ElementTree as ET
        tree = ET.parse(source)
        geometry = tree.find('.//collision/geometry')
        geometry.clear()
        ET.SubElement(geometry, 'mesh', filename=str(mesh))
        tree.write(source)
        generate(source, self.config['mobile_manifest'], self.temp.name, Path(self.temp.name) / 'generated')
        original = mesh.read_text()
        mesh.write_text(original + '\n')
        with self.assertRaisesRegex(ValueError, 'stale or missing source'):
            launch_plan(self.config)
        mesh.write_text(original)
        report_path = Path(self.config['asset_report'])
        report = json.loads(report_path.read_text())
        report['separation_proofs'].append({'rule': 'invented_never_collide'})
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'recomputed'):
            launch_plan(self.config)

    def test_true_source_collision_blocks_even_with_authentic_evidence(self):
        source = Path(self.temp.name) / 'robot.urdf'
        # Retract the second link to the root; it is not adjacent to the base.
        original = source.read_text()
        first = original.index('xyz="0.1 0 0"')
        second = original.index('xyz="0.1 0 0"', first+1)
        source.write_text(original[:second] + original[second:].replace('xyz="0.1 0 0"', 'xyz="-0.1 0 0"', 1))
        generate(source, self.config['mobile_manifest'], self.temp.name, Path(self.temp.name) / 'generated')
        self.assertTrue(launch_plan(self.config)['static_errors'])


class CollisionBufferTests(unittest.TestCase):
    def _inputs(self, directory):
        fixture(directory)  # writes robot.urdf + mobile.json (base_link carries a 0.005 m sphere)
        return Path(directory) / 'robot.urdf', Path(directory) / 'mobile.json', Path(directory)

    def test_base_buffer_inflates_only_named_link_and_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf, manifest, root = self._inputs(directory)
            buffed, art = build_assets(urdf, manifest, root, collision_buffer={'base_link': 0.02})
            spheres0 = yaml.safe_load(art['mobile.xrdf'])['geometry']['mobile_cad']['spheres']
            baseline0 = yaml.safe_load(build_assets(urdf, manifest, root)[1]['mobile.xrdf']
                                       )['geometry']['mobile_cad']['spheres']
            # base_link radii rise by exactly the buffer; a non-buffered link is untouched
            self.assertTrue(np.allclose([s['radius'] for s in spheres0['base_link']],
                                        np.array([s['radius'] for s in baseline0['base_link']]) + 0.02))
            self.assertTrue(np.allclose([s['radius'] for s in spheres0['link0']],
                                        [s['radius'] for s in baseline0['link0']]))
            self.assertEqual(buffed['collision_buffers_m'], {'base_link': 0.02})
            # deterministic: the same buffer reproduces byte-identical artifacts (evidence replay)
            self.assertEqual(hashlib.sha256(art['mobile.xrdf'].encode()).hexdigest(),
                             hashlib.sha256(build_assets(urdf, manifest, root,
                                 collision_buffer={'base_link': 0.02})[1]['mobile.xrdf'].encode()).hexdigest())

    def test_buffer_out_of_range_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf, manifest, root = self._inputs(directory)
            with self.assertRaisesRegex(ValueError, 'collision buffer'):
                build_assets(urdf, manifest, root, collision_buffer={'base_link': 0.5})


@unittest.skipUnless(os.environ.get('RESTOCK_HOST_ACTUAL_CAD') == '1', 'explicit actual local CAD regression')
class ActualCadTests(unittest.TestCase):
    def test_recorded_mobile_cad_transport(self):
        root = Path('/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile')
        report, _ = build_assets(root / 'mobile_ur10.urdf', root / 'mobile_manifest.json',
                                '/home/manishsaini/workspaces/shelfbot_newgripper')
        self.assertTrue(report['transport_collision_check_passed'])
        self.assertFalse(report['planning_ready'])
        self.assertEqual(len(report['separation_proofs']), 7)
        self.assertLessEqual(report['sphere_count'], 2500)  # task regression budget, not a vendor runtime limit
        excluded = {tuple(item['links']) for item in report['excluded_pairs']}
        for pair in (('arm_mount', 'ur_upper_arm_link'), ('ur_base_link_inertia', 'ur_upper_arm_link'),
                     ('ur_forearm_link', 'ur_wrist_2_link')):
            self.assertNotIn(pair, excluded)


if __name__ == '__main__':
    unittest.main()
