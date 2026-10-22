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
from twin.restock.sensing_sources import build_sensing_sources, author_mission_markers
import twin.restock_camera.config as cc

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

    def builder(self, *, assembly, design, render_scale=1.0, camera_render_scales=None, light_rig_scale=None):
        self.camera_render_scales = camera_render_scales
        return {'cameras': self.cameras,
                'base_from_camera': lambda cid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                'recording_camera': self.cameras.get('shelf') or next(iter(self.cameras.values()))}


def manip_runtime():
    return NS(motion=NS(parked=lambda: True, collision_time=2., joint_positions=lambda: (0.,) * 6),
              controller=NS(profiles={'juice_orange': object()}))


def assembly():
    return NS(paths={'ur_base_link': '/MobileRobot/ur_base_link'}, contact=NS(read=lambda: False),
              world=NS(current_time=7.5),
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

    def test_steady_clock_reads_the_sim_world_clock(self):
        # build_sensing passes sources['steady_clock'] to StationSensors so the acquisition-to-use
        # freshness gate measures the 0.5 s budget on the SIM clock (not wall) under render pacing.
        src, _ = self._build()
        self.assertIn('steady_clock', src)
        self.assertEqual(src['steady_clock'](), 7.5)

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

    def test_full_res_location_renders_its_camera_at_1_0(self):
        import twin.restock_camera.config as cc
        rig = FakeRig(cc.camera_config(SCENARIO, MANIFEST))
        build_sensing_sources(assembly=assembly(), scenario=SCENARIO, manifest=MANIFEST,
                              manip_runtime=manip_runtime(), pose_bridge_root='/pkg/tray_perception',
                              render_scale=0.375, full_res_locations=('source',), _rig_builder=rig.builder)
        # the source camera (serving the full-res collect location) renders at 1.0; others at 0.375
        self.assertEqual(rig.camera_render_scales['source'], 1.0)
        self.assertEqual(rig.camera_render_scales['shelf'], 0.375)
        self.assertEqual(rig.camera_render_scales['cart'], 0.375)

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


class FakeStage:
    """Records authored marker prim paths; GetPrimAtPath is truthy only for already-seen paths."""
    def __init__(self, existing=()):
        self.prims = set(existing)
    def GetPrimAtPath(self, path):
        return path in self.prims


class AuthorMissionMarkersTests(unittest.TestCase):
    PARENTS = {'map': '/World', 'cart_chassis': '/World/Cart'}

    def _run(self, stage, **kw):
        design = cc.camera_config(SCENARIO, MANIFEST)
        calls = []
        labels = []
        author_mission_markers(
            stage, design, self.PARENTS,
            _author=lambda s, cfg, *, parent_paths, marker_textures: calls.append(
                (cfg, marker_textures)),
            _write_label=lambda marker_id, png: labels.append((marker_id, str(png))),
            _mkdtemp=lambda: Path('/tmp/markers'), **kw)
        return design, calls, labels

    def test_authors_source_cart_and_shelf_markers_only(self):
        design, calls, labels = self._run(FakeStage())
        authored = {next(iter(cfg['supports'])) for cfg, _ in calls}
        self.assertEqual(authored, {'source', 'cart', 'shelf'})
        for cfg, textures in calls:
            name = next(iter(cfg['supports']))
            # markers-only: cameras dropped, only this support, textures for its markers
            self.assertEqual(cfg['cameras'], {})
            self.assertEqual(set(cfg['supports']), {name})
            self.assertEqual(set(textures), {m['id'] for m in design['supports'][name]['markers']})
            # the SOURCE keeps its witness-patch tab substrate; cart/shelf add no tab solid
            solid_ids = {s['id'] for s in cfg['solids']}
            self.assertEqual(solid_ids, {'source_marker_tab'} if name == 'source' else set())

    def test_idempotent_skips_already_authored_support(self):
        # source already present (its first marker prim exists) -> only cart + shelf authored
        stage = FakeStage(existing={'/World/RestockCamera_source_marker_0'})
        _, calls, _ = self._run(stage)
        authored = {next(iter(cfg['supports'])) for cfg, _ in calls}
        self.assertEqual(authored, {'cart', 'shelf'})

    def test_missing_parent_path_refused(self):
        # the cart support is parented to 'cart_chassis'; absent from parent_paths -> fail closed
        with self.assertRaisesRegex(ValueError, 'parent'):
            author_mission_markers(
                FakeStage(), cc.camera_config(SCENARIO, MANIFEST), {'map': '/World'},
                support_names=('cart',), _author=lambda *a, **k: None,
                _write_label=lambda *a, **k: None, _mkdtemp=lambda: Path('/tmp/m'))


class StationCameraSpecsTests(unittest.TestCase):
    PARENTS = {'map': '/World', 'cart_chassis': '/World/Cart'}

    def test_eye_in_hand_wrist_camera_is_skipped_station_cameras_kept(self):
        from twin.restock.sensing_sources import station_camera_specs
        design = cc.camera_config(SCENARIO, MANIFEST)
        # The design carries the eye-in-hand 'wrist' camera (parent restock_camera_wrist, no
        # housing solid); it must be skipped, not raise, and the fixed stations must remain.
        self.assertIn('wrist', design['cameras'])
        specs = station_camera_specs(design, self.PARENTS)
        self.assertEqual({s['id'] for s in specs}, {'source', 'cart', 'shelf'})
        for s in specs:
            self.assertTrue(s['collision_solids'])
            self.assertIn(s['parent'], self.PARENTS)

    def test_station_camera_without_solid_still_fails_closed(self):
        from twin.restock.sensing_sources import station_camera_specs
        design = cc.camera_config(SCENARIO, MANIFEST)
        design['solids'] = [s for s in design['solids'] if not s['id'].startswith('source')]
        with self.assertRaisesRegex(ValueError, "camera 'source' has no housing/bracket"):
            station_camera_specs(design, self.PARENTS)


if __name__ == '__main__':
    unittest.main()
