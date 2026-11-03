"""CPU proof for sim-robust-movegroup-collision-meshes-v1 (no ROS, no GPU).

Proves, the way MoveIt's ``resource_retriever`` classifies a ``<mesh filename>``, that the
resolved URDF the host ships today (bare ABSOLUTE mesh paths) loads NO link geometry, and that
rewriting each bare path to a ``file://`` URI makes the geometry loadable -- without touching
any mesh file or any geometry. The retriever model here is verified against the real
``resource_retriever`` (``file://`` + ``package://`` load the ur_description meshes; a bare path
raises; move_group's C++/libcurl retriever logs "URL rejected: No host part in the URL").
"""
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from twin.restock_host.mesh_uri import (  # noqa: E402
    loadable_meshes, normalize_mesh_uris, retriever_loadable)


def _urdf(mesh_filenames, role='collision'):
    """A minimal two-link URDF carrying the given mesh filenames in the given role."""
    links = ''.join(
        f'<link name="l{i}"><{role}><geometry>'
        f'<mesh filename="{name}"/></geometry></{role}></link>'
        for i, name in enumerate(mesh_filenames))
    return f"<?xml version='1.0'?><robot name='r'>{links}</robot>"


class RetrieverModelTests(unittest.TestCase):
    def test_bare_absolute_path_is_rejected_no_host_part(self):
        ok, reason = retriever_loadable('/opt/ros/jazzy/share/ur_description/collision/base.stl',
                                        exists=lambda p: True)
        self.assertFalse(ok)
        self.assertIn('No host part in the URL', reason)   # the move_group log line this fix targets

    def test_bare_relative_path_is_rejected(self):
        ok, reason = retriever_loadable('meshes/base.stl', exists=lambda p: True)
        self.assertFalse(ok)
        self.assertIn('No host part', reason)

    def test_file_uri_loads_when_the_file_exists(self):
        ok, reason = retriever_loadable('file:///real/base.stl', exists=lambda p: p == '/real/base.stl')
        self.assertTrue(ok)
        self.assertEqual(reason, '')

    def test_file_uri_missing_file_is_a_distinct_failure_not_the_no_scheme_one(self):
        ok, reason = retriever_loadable('file:///gone/base.stl', exists=lambda p: False)
        self.assertFalse(ok)
        self.assertIn('missing file', reason)
        self.assertNotIn('No host part', reason)           # a different failure mode than the defect

    def test_package_uri_is_loadable(self):
        ok, reason = retriever_loadable('package://ur_description/meshes/base.stl', exists=lambda p: False)
        self.assertTrue(ok)


class NormalizeTests(unittest.TestCase):
    def test_bare_absolute_mesh_becomes_a_file_uri(self):
        out = normalize_mesh_uris(_urdf(['/opt/ros/jazzy/share/ur/collision/base.stl']))
        names = [m.get('filename') for m in ET.fromstring(out).iter('mesh')]
        self.assertEqual(names, ['file:///opt/ros/jazzy/share/ur/collision/base.stl'])

    def test_package_and_file_uris_are_left_untouched(self):
        srcs = ['package://ur/meshes/base.stl', 'file:///abs/shoulder.stl']
        out = normalize_mesh_uris(_urdf(srcs))
        names = [m.get('filename') for m in ET.fromstring(out).iter('mesh')]
        self.assertEqual(names, srcs)

    def test_relative_bare_path_is_left_as_is(self):
        # A relative path has no absolute root to turn into a file:// URI without inventing a cwd;
        # the host never ships one, so leave it (so a genuinely unresolved mesh still surfaces).
        out = normalize_mesh_uris(_urdf(['meshes/base.stl']))
        names = [m.get('filename') for m in ET.fromstring(out).iter('mesh')]
        self.assertEqual(names, ['meshes/base.stl'])

    def test_idempotent(self):
        once = normalize_mesh_uris(_urdf(['/abs/base.stl']))
        twice = normalize_mesh_uris(once)
        self.assertEqual(once, twice)

    def test_rewrite_changes_only_mesh_filenames_not_geometry(self):
        src = _urdf(['/abs/base.stl'])
        before, after = ET.fromstring(src), ET.fromstring(normalize_mesh_uris(src))
        # same link/geometry structure, only the one filename attribute differs
        self.assertEqual([l.tag for l in before], [l.tag for l in after])
        self.assertEqual(len(list(before.iter('geometry'))), len(list(after.iter('geometry'))))


class LoaderLevelEvaluationTests(unittest.TestCase):
    """Point 2: the collision geometry move_group loads is EMPTY with the key off (bare paths)
    and NON-EMPTY with it on (file:// URIs), proved on real mesh files on disk."""

    def test_empty_off_nonempty_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name in ('base.stl', 'shoulder.stl', 'forearm.stl'):
                p = os.path.join(tmp, name)
                with open(p, 'wb') as fh:
                    fh.write(b'solid x\nendsolid x\n')
                paths.append(p)
            shipped = _urdf(paths)                         # bare absolute paths, as generate_assets writes
            self.assertEqual(loadable_meshes(shipped, role='collision'), [],
                             'the shipped URDF must load NO collision geometry (the defect)')
            fixed = normalize_mesh_uris(shipped)
            loaded = loadable_meshes(fixed, role='collision')
            self.assertEqual(len(loaded), len(paths))      # every link mesh now loads
            self.assertTrue(all(f.startswith('file://') for f in loaded))

    def test_evaluation_covers_both_visual_and_collision(self):
        # move_group logs a rejection for BOTH roles; after the fix neither role rejects.
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'm.stl')
            with open(p, 'wb') as fh:
                fh.write(b'x')
            for role in ('visual', 'collision'):
                shipped = _urdf([p], role=role)
                self.assertEqual(loadable_meshes(shipped, role=role), [])
                self.assertEqual(len(loadable_meshes(normalize_mesh_uris(shipped), role=role)), 1)


if __name__ == '__main__':
    unittest.main()
