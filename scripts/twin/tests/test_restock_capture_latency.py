"""CPU tests for the per-capture latency cuts: ROI-bounded decode + cached decode ROI."""
import numpy as np
import pytest

from scripts.twin.restock_camera.sensed import FrameEvidence
from scripts.twin.restock_camera.qr import FixedTileQR, tile_origins, _tile_overlaps


def _frame_with_qr(width=1280, height=960, corner=(60, 60)):
    cv2 = pytest.importorskip('cv2')
    qr = cv2.copyMakeBorder(cv2.QRCodeEncoder_create().encode('support:a'), 4, 4, 4, 4,
                            cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (240, 240), interpolation=cv2.INTER_NEAREST)
    gray = np.full((height, width), 255, np.uint8)
    x, y = corner
    gray[y:y+240, x:x+240] = qr
    return np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)


def _evidence(frame_id=1, stamp=1.):
    return FrameEvidence('source', frame_id, stamp, stamp, 'park1', 'lens1')


def test_roi_decode_restricts_tiles_and_matches_full_frame():
    rgb = _frame_with_qr()
    roi = (0., 0., 320., 320.)                       # top-left region containing the code
    overlapping = sum(_tile_overlaps(o, roi, 1280, 960) for o in tile_origins(1280, 960))
    assert overlapping < len(tile_origins(1280, 960))  # a genuine subset of the certified grid
    full = FixedTileQR('source').decode(rgb, _evidence(), now=1., parked=True)
    roied = FixedTileQR('source').decode(rgb, _evidence(), now=1., parked=True, roi=roi)
    assert {d.text for d in roied.detections} == {d.text for d in full.detections} == {'support:a'}
    assert roied.tile_count == overlapping and roied.tile_count < full.tile_count   # fewer tiles scanned


def test_roi_decode_falls_back_to_full_frame_on_a_miss():
    rgb = _frame_with_qr(corner=(60, 60))
    roi = (900., 600., 1280., 960.)                  # bottom-right: excludes the code
    result = FixedTileQR('source').decode(rgb, _evidence(), now=1., parked=True, roi=roi)
    # Total miss in the ROI -> full-frame fallback still finds the code (correctness preserved).
    assert {d.text for d in result.detections} == {'support:a'}
    assert result.tile_count == len(tile_origins(1280, 960))


def test_commission_caches_decode_roi_and_reload_restores_it(tmp_path):
    from scripts.twin.restock_camera import commissioning as C
    from .test_restock_commissioning import empty_frame, context
    k, camera, depth, support, witness, frame = empty_frame()
    ctx = context(k, camera, location='backstock_00', camera_id='source')
    C.commission_and_persist(ctx, frame, 1., {'backstock_00': (support, witness)}, dataset_dir=tmp_path)
    roi = ctx.slots[0].decode_roi
    assert roi is not None and roi[0] < roi[2] and roi[1] < roi[3]      # a real pixel box
    # The cached ROI bounds the commissioned support (support + witness) pixels.
    box = np.concatenate([support, witness])
    assert roi[0] <= box[:, 0].min() and roi[2] >= box[:, 0].max()
    # Persisted and restored onto a fresh context at reload.
    boot = context(k, camera, location='backstock_00', camera_id='source')
    assert boot.slots[0].decode_roi is None
    C.apply_baselines(boot, C.load_baselines(tmp_path), required_locations=['backstock_00'])
    assert boot.slots[0].decode_roi == roi
