"""CPU tests for the Isaac-side MAST depth publisher wiring (nvblox camera_1).

No Isaac/GPU: the mast camera authoring and the out-of-Isaac rclpy publisher subprocess are
injected behind seams (``_initialize``/``_bridge_cls`` on ``build_mast_depth_bridge``;
``_rig_builder``/``_mast_bridge_builder`` on ``build_sensing_sources``), and the rig+carton
self-mask is exercised through the pure ``masked_depth_from_frame`` core -- exactly as the wrist
(camera_0) masked-depth tests do. Asserts: (a) absent flag => no mast wiring (byte-identical), (b)
enabled => the mast feed uses the MAST_* topics/frame and the rig+carton self-mask zeroes rig/carton
pixels, (c) the design gains the ``mast`` camera only when enabled.
"""
from types import SimpleNamespace as NS
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.masked_depth import (
    build_mast_depth_bridge, mast_sim_topics, _serve_topic_args, masked_depth_from_frame,
    _default_socket_path)
from twin.restock.sensing_sources import build_sensing_sources
from twin.restock_camera import mast_depth as M
import twin.restock_camera.config as cc

SCENARIO = {
    'job': {'sku': 'juice_orange', 'quantity': 1, 'shelf_id': 'G3_back_s2'},
    'handling_unit': {'local_wall': 3, 'grasp_mode': 'side', 'dimensions_m': [.6, .4, .16]},
    'initial_staging': {'support_center_xy': [-5.73, 17.1], 'unit_bottom_z': .6,
                        'target_surface_pose_world': [-3.05, 4.826, .73, 0],
                        'cart_load_center_body_xy': [.08, 0], 'cart_support_z_world': .225},
}
MANIFEST = {'spec': {'arm_mount_z': .5}}
ROBOT_ROOT = '/mobile_ur10_rollwagen'
CART_ROOT = '/Rollwagen'           # the mission cart is a top-level prim (BESIDE the robot)
GRIPPER = '/RestockPanelGripper'


def assembly():
    return NS(stage=object(), transport=object(), contact=NS(read=lambda: False),
              world=NS(current_time=7.5),
              paths={'ur_base_link': ROBOT_ROOT + '/ur_base_link',
                     'articulation': ROBOT_ROOT + '/ur_base_link'},
              evidence={'composition': {'assets': {'cart': {'root': CART_ROOT}}}})


def manip_runtime():
    return NS(motion=NS(parked=lambda: True, collision_time=2., joint_positions=lambda: (0.,) * 6),
              controller=NS(profiles={'juice_orange': object()}))


class FakeRig:
    """Records the design it is handed (so we can assert the mast camera presence per flag)."""
    def __init__(self):
        self.design = None

    def builder(self, *, assembly, design, render_scale=1.0, camera_render_scales=None, light_rig_scale=None):
        self.design = design
        return {'cameras': {cid: NS(id=cid) for cid in design['cameras']},
                'base_from_camera': lambda cid: np.eye(4).tolist(),
                'recording_camera': NS(id='chase')}


class RecordingBridgeBuilder:
    """Stands in for build_mast_depth_bridge; records the kwargs build_sensing_sources passes."""
    def __init__(self, result='MAST_BRIDGE'):
        self.calls = []
        self.result = result

    def __call__(self, *, stage, assembly, design, settings):
        self.calls.append(dict(stage=stage, assembly=assembly, design=design, settings=settings))
        return self.result


def build(*, mast_depth=False, mast_settings=None, mast_builder=None):
    rig = FakeRig()
    src = build_sensing_sources(
        assembly=assembly(), scenario=SCENARIO, manifest=MANIFEST, manip_runtime=manip_runtime(),
        pose_bridge_root='/pkg/tray_perception', mast_depth=mast_depth,
        mast_depth_settings=mast_settings, _rig_builder=rig.builder,
        _mast_bridge_builder=mast_builder)
    return src, rig


class MastAbsentIsByteIdentical(unittest.TestCase):
    def test_default_adds_no_mast_camera_and_no_bridge(self):
        builder = RecordingBridgeBuilder()
        src, rig = build(mast_builder=builder)
        self.assertIsNone(src['mast_depth'])                 # no publisher wired
        self.assertNotIn('mast', rig.design['cameras'])      # design unchanged
        self.assertEqual(builder.calls, [])                  # bridge builder never invoked

    def test_design_handed_to_rig_matches_camera_config_off(self):
        import json
        _, rig = build(mast_depth=False)
        self.assertEqual(json.dumps(rig.design, sort_keys=True),
                         json.dumps(cc.camera_config(SCENARIO, MANIFEST), sort_keys=True))


class MastEnabledWiring(unittest.TestCase):
    def test_design_gains_only_the_mast_camera_when_enabled(self):
        _, off = build(mast_depth=False)
        _, on = build(mast_depth=True, mast_builder=RecordingBridgeBuilder())
        self.assertEqual(set(on.design['cameras']) - set(off.design['cameras']), {'mast'})
        self.assertEqual(on.design['cameras']['mast']['parent'], 'robot_base')
        self.assertEqual(set(on.design['supports']), set(off.design['supports']))

    def test_enabled_builds_and_exposes_the_bridge_with_the_design_and_settings(self):
        builder = RecordingBridgeBuilder(result='THE_BRIDGE')
        src, rig = build(mast_depth=True, mast_settings={'runtime_env': '/env', 'domain': 88},
                         mast_builder=builder)
        self.assertEqual(src['mast_depth'], 'THE_BRIDGE')
        self.assertEqual(len(builder.calls), 1)
        call = builder.calls[0]
        self.assertIn('mast', call['design']['cameras'])
        self.assertEqual(call['settings'], {'runtime_env': '/env', 'domain': 88})
        self.assertIs(call['assembly'].transport, call['assembly'].transport)

    def test_none_settings_becomes_empty_dict_for_the_builder(self):
        builder = RecordingBridgeBuilder()
        build(mast_depth=True, mast_settings=None, mast_builder=builder)
        self.assertEqual(builder.calls[0]['settings'], {})


class MastTopicsAndFrame(unittest.TestCase):
    def test_sim_topics_are_the_mast_namespaced_feed_distinct_from_the_wrist(self):
        topics = mast_sim_topics()
        self.assertEqual(topics['depth'], '/restock_sim/' + M.MAST_MASKED_DEPTH)
        self.assertEqual(topics['info'], '/restock_sim/' + M.MAST_MASKED_DEPTH_INFO)
        self.assertEqual(topics['depth'], '/restock_sim/camera_mast/masked_depth')
        self.assertIsNone(topics['bbox'])                     # mast reuses the shared wrist bbox
        # Distinct from the wrist (camera_0) feed.
        from twin.restock.masked_depth_ipc import WRIST_DEPTH_TOPIC
        self.assertNotEqual(topics['depth'], WRIST_DEPTH_TOPIC)

    def test_serve_topic_args_wrist_default_is_byte_identical_mast_appends(self):
        self.assertEqual(_serve_topic_args(None), '')         # wrist command unchanged
        self.assertEqual(_serve_topic_args({}), '')
        self.assertEqual(
            _serve_topic_args(mast_sim_topics()),
            ' /restock_sim/camera_mast/masked_depth'
            ' /restock_sim/camera_mast/masked_depth/camera_info none')


class MastBridgeConstruction(unittest.TestCase):
    """build_mast_depth_bridge wiring, injecting the authoring + publisher seams (no GPU)."""

    def _fake_initialize(self, stage, design, base_path):
        self.init_args = (stage, design, base_path)
        return NS(id='mast_cam'), base_path + '/restock_camera_mast/Camera', M.MAST_FRAME_ID

    def test_uses_mast_frame_topics_no_bbox_and_rig_plus_carton_prefixes(self):
        captured = {}

        def fake_bridge(**kw):
            captured.update(kw)
            return 'BRIDGE'

        out = build_mast_depth_bridge(
            stage='STAGE', assembly=assembly(), design=cc.camera_config(SCENARIO, MANIFEST, mast_depth=True),
            settings={'runtime_env': '/env'}, _initialize=self._fake_initialize, _bridge_cls=fake_bridge)
        self.assertEqual(out, 'BRIDGE')
        self.assertEqual(captured['frame_id'], M.MAST_FRAME_ID)
        self.assertEqual(captured['topics'], mast_sim_topics())
        self.assertIsNone(captured['sensors'])                # no bbox publisher (shared wrist one)
        self.assertEqual(captured['camera'].id, 'mast_cam')
        self.assertTrue(captured['camera_path'].endswith('/restock_camera_mast/Camera'))
        # rig+carton self-mask prefixes: robot + cart + gripper subtrees.
        self.assertEqual(set(captured['rig_prefixes']), {ROBOT_ROOT, CART_ROOT, GRIPPER})
        # authored under robot_base (ur_base_link).
        self.assertEqual(self.init_args[2], ROBOT_ROOT + '/ur_base_link')

    def test_returns_none_when_design_has_no_mast_camera(self):
        out = build_mast_depth_bridge(
            stage='STAGE', assembly=assembly(), design=cc.camera_config(SCENARIO, MANIFEST),
            settings={'runtime_env': '/env'},
            _initialize=lambda stage, design, base: None, _bridge_cls=lambda **kw: 'X')
        self.assertIsNone(out)

    def test_explicit_rig_prefixes_override(self):
        captured = {}
        build_mast_depth_bridge(
            stage='STAGE', assembly=assembly(), design=cc.camera_config(SCENARIO, MANIFEST, mast_depth=True),
            settings={'runtime_env': '/env', 'rig_prefixes': ['/A', '/B']},
            _initialize=self._fake_initialize,
            _bridge_cls=lambda **kw: captured.update(kw))
        self.assertEqual(list(captured['rig_prefixes']), ['/A', '/B'])


class SocketPathPerFeedUniqueness(unittest.TestCase):
    """The wrist (camera_0) and mast (camera_1) publisher subprocesses run in ONE Isaac process,
    so their default Unix-socket paths MUST differ (serve() unlinks its path on bind + exit, so a
    shared name lets one feed clobber the sibling). Regression for the fm458 collision."""

    def test_wrist_and_mast_default_socket_paths_differ_in_one_process(self):
        pid = 4242
        wrist = _default_socket_path('restock_camera_arm', pid=pid)
        mast = _default_socket_path(M.MAST_FRAME_ID, pid=pid)        # same process (same pid)
        self.assertNotEqual(wrist, mast)
        self.assertIn(str(pid), wrist)
        self.assertIn('restock_camera_arm', wrist)
        self.assertIn('restock_camera_mast', mast)
        # No path separators leak into the socket filename.
        self.assertNotIn('/', wrist.rsplit('/', 1)[1])
        self.assertNotIn('/', mast.rsplit('/', 1)[1])


class MastRigCartonSelfMask(unittest.TestCase):
    """The mast feed removes rig AND the carried carton (a rig-instance subtree) pixels."""

    def _frame(self, depth, ids, id_to_path):
        return {'distance_to_image_plane': np.asarray(depth, np.float32),
                'instance_id_segmentation': {'data': np.asarray(ids),
                                             'info': {'idToLabels': id_to_path}}}

    def test_rig_and_carton_pixels_zeroed_free_space_kept(self):
        # Pixel ids: 0 free scene, 1 robot, 2 cart, 3 the carried carton welded under the gripper
        # subtree (a rig-instance subtree -> masked like any rig pixel).
        id_to_path = {0: '/World/Store/floor', 1: ROBOT_ROOT + '/ur_base_link',
                      2: CART_ROOT + '/chassis', 3: GRIPPER + '/attached_carton/box'}
        frame = self._frame([[2.0, 2.0], [2.0, 2.0]], [[0, 1], [2, 3]], id_to_path)
        prefixes = (ROBOT_ROOT, CART_ROOT, GRIPPER)
        masked, cache = masked_depth_from_frame(frame, prefixes, (None, frozenset()))
        self.assertEqual(masked.tolist(), [[2.0, 0.0], [0.0, 0.0]])   # robot/cart/carton -> invalid
        self.assertEqual(cache[1], {1, 2, 3})                         # carton id included


if __name__ == '__main__':
    unittest.main()
