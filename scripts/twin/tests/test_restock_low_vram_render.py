"""fm373: the mission SimulationApp low-VRAM render config is SIM-ONLY opt-in and byte-identical
when unset. run-297/299 OOM'd the 12 GB card on long runs; the OOM'd resources were RTX render-pass
buffers ('Indirect diffuse texture', 'DLSS Output') and the peak was resolution-INDEPENDENT (480x270
~= 640x360). Isaac's SimulationApp defaults to renderer=RaytracedLighting + anti_aliasing=3 (DLSS),
so the heavy GI+DLSS pipeline runs unless we opt out. These tests pin the pure config builder:
unset => exactly today's dict + no carb settings; set => aa=0 (no DLSS) + explicit RayTracedLighting
+ the indirect-diffuse/reflections/AO carb disables (QR sensing stays full-res, unaffected)."""
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
SCRIPTS = os.path.abspath(os.path.join(HERE, '..', '..'))   # .../scripts
sys.path.insert(0, SCRIPTS)

import run_mobile_restock as rmr  # noqa: E402


class LowVramRenderConfigTests(unittest.TestCase):
    def test_default_is_byte_identical(self):
        # Env unset / low_vram=False => exactly the pre-fm373 config dict and NO carb settings.
        config, settings = rmr._render_app_config(960, 540, low_vram=False)
        self.assertEqual(config, {'headless': True, 'width': 960, 'height': 540})
        self.assertEqual(settings, [])
        # No renderer/anti_aliasing/multi_gpu keys are introduced (defaults stay Isaac's).
        self.assertNotIn('renderer', config)
        self.assertNotIn('anti_aliasing', config)

    def test_default_honours_resolution(self):
        config, settings = rmr._render_app_config(480, 270, low_vram=False)
        self.assertEqual(config, {'headless': True, 'width': 480, 'height': 270})
        self.assertEqual(settings, [])

    def test_low_vram_disables_dlss_and_pins_raytraced(self):
        config, settings = rmr._render_app_config(480, 270, low_vram=True)
        # Width/height preserved; DLSS off (anti_aliasing 0); explicit RayTracedLighting; single GPU.
        self.assertEqual(config['width'], 480)
        self.assertEqual(config['height'], 270)
        self.assertEqual(config['headless'], True)
        self.assertEqual(config['renderer'], 'RayTracedLighting')
        self.assertEqual(config['anti_aliasing'], 0)
        self.assertEqual(config['multi_gpu'], False)

    def test_low_vram_disables_gi_reflections_ao_carb_passes(self):
        _config, settings = rmr._render_app_config(480, 270, low_vram=True)
        # The exact carb keys that free the OOM'd GI/reflection/AO buffers, all set False.
        self.assertEqual(settings, [
            ('/rtx/indirectDiffuse/enabled', False),
            ('/rtx/reflections/enabled', False),
            ('/rtx/ambientOcclusion/enabled', False)])
        for _k, v in settings:
            self.assertIs(v, False)

    def test_sensing_resolution_knobs_are_not_touched(self):
        # The builder only shapes the APP/walkthrough render; it never carries sensing-camera
        # resolution (those are full-res in isaac_bundle). Nothing render-scale-ish leaks in.
        config, _settings = rmr._render_app_config(640, 360, low_vram=True)
        self.assertNotIn('render_scale', config)
        self.assertNotIn('sensing', config)


if __name__ == '__main__':
    unittest.main()
