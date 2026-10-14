"""CPU contract tests for the integration-owned assembly.sensing_sources builder.

No Isaac/GPU: the camera rig is injected via `_rig_builder`; the camera design comes from
the real camera_config so the slot/context wiring is exercised. Asserts the bindings the
camera-lane sensing_factory requires, and that clearance is conservatively absent (sensing
fails closed, never fabricated) when no live mapper is supplied.
"""
from types import SimpleNamespace as NS
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.sensing_sources import build_sensing_sources

SCENARIO = {
    'job': {'sku': 'juice_orange', 'quantity': 1, 'shelf_id': 'G3_back_s2'},
    'handling_unit': {'local_wall': 3, 'grasp_mode': 'side', 'dimensions_m': [.6, .4, .16]},
    'initial_staging': {'support_center_xy': [-5.73, 17.1], 'unit_bottom_z': .6,
                        'target_surface_pose_world': [-3.05, 4.826, .73, 0],
                        'cart_load_center_body_xy': [.08, 0], 'cart_support_z_world': .225},
}
MANIFEST = {'spec': {'arm_mount_z': .5}}


class FakeRig:
    def __init__(self, design):
        # one RenderedCamera stand-in per design camera id
        self.cameras = {cid: NS(id=cid) for cid in design['cameras']}

    def builder(self, *, assembly, design, render_scale=1.0):
        return {'cameras': self.cameras,
                'base_from_camera': lambda cid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                'recording_camera': self.cameras.get('shelf') or next(iter(self.cameras.values()))}


def manip_runtime():
    return NS(motion=NS(parked=lambda: True, collision_time=2., joint_positions=lambda: (0.,) * 6),
              controller=NS(profiles={'juice_orange': object()}))


def assembly():
    return NS(paths={'ur_base_link': '/MobileRobot/ur_base_link'}, contact=NS(read=lambda: False),
              evidence={'composition': {'assets': {'cart': {'root': '/World/Cart'}}}})


class BuildSensingSourcesTests(unittest.TestCase):
    def _build(self, clearance=None):
        import twin.restock_camera.config as cc
        rig = FakeRig(cc.camera_config(SCENARIO, MANIFEST))
        return build_sensing_sources(assembly=assembly(), scenario=SCENARIO, manifest=MANIFEST,
                                     manip_runtime=manip_runtime(), clearance_for_slot=clearance,
                                     pose_bridge_root='/pkg/tray_perception',
                                     _rig_builder=rig.builder), rig

    def test_bindings_present(self):
        src, rig = self._build()
        for key in ('cameras', 'context_providers', 'location_cameras', 'profiles',
                    'pose_bridge_root', 'motion', 'contact', 'recording_camera'):
            self.assertIn(key, src)
        self.assertEqual(src['pose_bridge_root'], '/pkg/tray_perception')
        self.assertIn('juice_orange', src['profiles'])  # profiles passed through from manip runtime

    def test_location_cameras_map_mission_locations(self):
        src, _ = self._build()
        # design supports source/cart/shelf -> mission locations source/rollwagen/<shelf_id>
        self.assertEqual(src['location_cameras'].get('rollwagen'), 'cart')
        self.assertEqual(src['location_cameras'].get('G3_back_s2'), 'shelf')
        self.assertEqual(src['location_cameras'].get('source'), 'source')

    def test_context_providers_per_camera(self):
        src, _ = self._build()
        # every mapped camera that has a support slot gets a context provider
        self.assertTrue(set(src['context_providers']) <= set(src['cameras']))
        self.assertIn('shelf', src['context_providers'])

    def test_clearance_conservative_none_without_mapper(self):
        src, _ = self._build()
        ctx = next(iter(src['context_providers'].values()))
        # the default clearance callback yields None (no live mapper -> sensing fails closed)
        self.assertIsNone(ctx.clearance_for_slot('shelf', object(), object(), 1.0))

    def test_injected_clearance_is_used(self):
        sentinel = object()
        src, _ = self._build(clearance=lambda *a: sentinel)
        ctx = next(iter(src['context_providers'].values()))
        self.assertIs(ctx.clearance_for_slot('shelf', object(), object(), 1.0), sentinel)


if __name__ == '__main__':
    unittest.main()
