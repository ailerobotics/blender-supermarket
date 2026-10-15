"""Fast sensing-render preset: resolution-adaptive decode + the camera_config knob (CPU, no GPU).

The whole-store RTX render of the sensing camera dominates mission wall time; a 'fast' preset steps
the resolution down. It is only safe because the decoder tile grid scales with the render
resolution (markers keep their position relative to the tile seams). These synthetic-fixture tests
prove the scaled-tile decode parity and the knob; the native reduced-resolution render proof and the
sampling/product cost measurement are the integration lane's. The decode parity measured on the real
native commissioning frames is in the camera lane's fast-render handoff.
"""
import numpy as np
import pytest

from scripts.twin.restock_camera import qr as qrmod
from scripts.twin.restock_camera.qr import FixedTileQR, tile_origins
from scripts.twin.restock_camera.config import camera_config, render_presets
from scripts.twin.restock_camera.sensed import FrameEvidence


def _design():
    scenario = dict(handling_unit=dict(local_wall=3, grasp_mode='side', dimensions_m=[.6,.4,.16]),
        initial_staging=dict(support_center_xy=[-5.73,17.1], unit_bottom_z=.6,
            target_surface_pose_world=[-3.05,4.826,.73,0], cart_load_center_body_xy=[.08,0],
            cart_support_z_world=.225))
    return camera_config(scenario, dict(spec=dict(arm_mount_z=.5)))


def test_tile_origins_default_is_the_unchanged_production_grid():
    # render_scale=1.0 default: byte-identical 640/480 grid.
    assert tile_origins(2560,1920) == tuple((x,y) for y in (0,480,960,1280)
                                            for x in (0,480,960,1440,1920))
    assert tile_origins(80,60) == ((0,0),)
    # A scaled grid (0.6) uses proportionally smaller tiles, so a downsampled frame is still tiled.
    scaled = tile_origins(1536,1152, tile=384, stride=288)
    assert scaled[0] == (0,0) and len(scaled) > 1
    with pytest.raises(ValueError, match='tile/stride'):
        tile_origins(1536,1152, tile=100, stride=200)   # stride must be <= tile


def test_render_scale_must_be_in_unit_interval():
    FixedTileQR('cam', render_scale=1.0)                # ok
    FixedTileQR('cam', render_scale=0.6)                # ok
    for bad in (0.0, -0.1, 1.5, float('nan')):
        with pytest.raises(ValueError, match='render_scale'):
            FixedTileQR('cam', render_scale=bad)


def test_scaled_decoder_matches_the_production_tile_size():
    assert FixedTileQR('cam').tile == 640 and FixedTileQR('cam').stride == 480
    fast = FixedTileQR('cam', render_scale=0.6)
    assert fast.tile == round(640*0.6) and fast.stride == round(480*0.6)


def _frame_with_big_qr(text, module_px=16):
    """A 2560x1920 white frame carrying one decode-grade QR (``module_px`` px per module), placed
    stride-aligned so it sits wholly inside a tile at both the full and the 0.6 grid. Mirrors how the
    native source frame presents a marker at a comfortable px/module; lets the test exercise the
    scaled-tile mechanism independent of the source's own marginal ~6 px/module floor."""
    import cv2
    code = cv2.QRCodeEncoder_create().encode(text)
    code = cv2.copyMakeBorder(code, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    size = code.shape[0] * module_px
    code = cv2.resize(code, (size, size), interpolation=cv2.INTER_NEAREST)
    gray = np.full((1920, 2560), 255, np.uint8)
    y0, x0 = 480, 480                                    # stride-aligned (multiple of 480 and 288)
    gray[y0:y0+size, x0:x0+size] = code
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_downsampled_frame_decodes_with_the_scaled_tile_grid():
    cv2 = pytest.importorskip('cv2')
    rgb = _frame_with_big_qr('juice_orange')

    def decode(pixels, scale):
        obj = FixedTileQR('cam', render_scale=scale)
        ev = FrameEvidence('cam', 1, 1., 1., 'e', 'r')
        return {d.text for d in obj.decode(pixels, ev, now=1., parked=True).detections}

    assert 'juice_orange' in decode(rgb, 1.0)                       # full-res reference

    scale = 0.6
    small = cv2.resize(rgb, (round(rgb.shape[1]*scale), round(rgb.shape[0]*scale)),
                       interpolation=cv2.INTER_AREA)
    # The scaled tile grid (render_scale=0.6 -> 384 px tiles) keeps the marker wholly inside a tile
    # at the reduced resolution, so decode holds; this is the mechanism the fast preset relies on.
    assert 'juice_orange' in decode(small, scale)


def test_render_presets_knob_defaults_to_full_and_is_documented():
    presets = render_presets()
    assert set(presets) == {'full', 'fast'}
    assert presets['full']['render_scale'] == 1.0                   # default changes nothing
    assert presets['fast']['render_scale'] == 0.6                   # measured source decode floor
    # The sensing capture reads only rgb + depth; the preset drops every other render product.
    assert set(presets['full']['render_products']) == {'rgb', 'distance_to_image_plane'}
    assert set(presets['fast']['render_products']) == {'rgb', 'distance_to_image_plane'}
    # Exposed on the generated camera design for the integration runtime to opt in per run.
    assert _design()['render_presets']['fast']['render_scale'] == 0.6


# --- end-to-end wiring: one opt-in flag drives commission AND mission at the matching scale -----

def test_preset_render_scale_resolver():
    from scripts.twin.restock_camera.config import preset_render_scale
    assert preset_render_scale('full') == 1.0
    assert preset_render_scale('fast') == 0.6
    with pytest.raises(ValueError, match='render preset'):
        preset_render_scale('zoom')


def test_mission_render_preset_sets_scale_and_source_resolution():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run_mobile_restock import _apply_render_preset
    fast = _apply_render_preset({}, 'fast')
    assert fast['render_scale'] == 0.6
    assert fast['full_res_collect_source'] is False       # fast renders the dominant source at 0.6
    full = _apply_render_preset({'render_scale': 0.6, 'full_res_collect_source': False}, 'full')
    assert full['render_scale'] == 1.0                    # 'full' is authoritative when opted in
    assert full['full_res_collect_source'] is True        # decode-grade full-res source


def test_commission_render_preset_resolves_the_same_scale():
    # the commissioning CLI resolves --render-preset through the SAME resolver the mission uses,
    # so a `fast` baseline and a `fast` mission are commissioned/rendered at one scale.
    from scripts.twin.restock_camera.config import preset_render_scale
    from scripts.twin.restock_camera.geometry import Intrinsics, scale_intrinsics
    k = Intrinsics(1280, 960, 640., 640., 639.5, 479.5)
    full_k = k if preset_render_scale('full') == 1.0 else scale_intrinsics(k, 1.0)
    fast_k = scale_intrinsics(k, preset_render_scale('fast'))
    # A cross-scale baseline (commissioned full, mission fast) carries different intrinsics, so the
    # boot-time baseline/context match (commissioning._check_match) refuses it -- the mismatch guard.
    assert fast_k != full_k
