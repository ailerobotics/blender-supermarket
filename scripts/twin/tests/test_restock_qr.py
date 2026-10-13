"""Bounded decoder behavior; synthetic tests are not held-out render evidence."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.twin.restock_camera import qr
from scripts.twin.restock_camera.sensed import FrameEvidence


def evidence(frame=1,stamp=1.,session='mission'):
    return FrameEvidence('camera',frame,stamp,stamp,'park','lens',session)


def box(x=20,y=20):
    return np.array([[[x,y],[x+40,y],[x+40,y+40],[x,y+40]]],np.float32)


def decoder(responses):
    obj=qr.FixedTileQR('camera')
    calls=[]
    values=iter(responses)
    def detect(image):
        calls.append(image.shape)
        text,points=next(values)
        return text,points,None
    obj.detector=SimpleNamespace(detectAndDecode=detect)
    return obj,calls


def test_grid_is_fixed_bounded_and_not_marker_specific():
    assert qr.tile_origins(2560,1920)==tuple((x,y) for y in (0,480,960,1280)
                                          for x in (0,480,960,1440,1920))
    assert qr.tile_origins(80,60)==((0,0),)
    with pytest.raises(ValueError,match='bounded'):
        qr.tile_origins(2561,1920)


def test_translated_duplicates_merge_without_synthesizing_corners():
    obj,calls=decoder([('actual',box(500)),('actual',box(20))])
    rgb=np.zeros((640,1120,3),np.uint8)
    result=obj.decode(rgb,evidence(),now=1.,parked=True)
    assert len(calls)==2 and len(result.detections)==1
    assert result.detections[0].text=='actual'
    assert result.detections[0].sightings==2
    np.testing.assert_array_equal(result.detections[0].corners,box(500)[0])
    result.validate(rgb,evidence(),now=1.1,parked=True)
    for changed in (replace(evidence(),epoch='other'),replace(evidence(),session_id='other'),
                    replace(evidence(),frame_id=2)):
        with pytest.raises(ValueError,match='another frame'):
            result.validate(rgb,changed,now=1.1,parked=True)
    changed=rgb.copy();changed[0,0,0]=1
    with pytest.raises(ValueError,match='another frame'):
        result.validate(changed,evidence(),now=1.1,parked=True)
    with pytest.raises(ValueError,match='stale'):
        result.validate(rgb,evidence(),now=1.6,parked=True)
    with pytest.raises(ValueError,match='replayed'):
        obj.decode(rgb,evidence(),now=1.1,parked=True)
    assert len(calls)==2


@pytest.mark.parametrize('responses',[
    [('first',box(500)),('conflict',box(20))],
    [('same',box(20)),('same',box(20))],
    [('same',box(500)),('same',box(23,23))],
    [('first',box(500)),('overlapping',box(30))]])
def test_conflicting_or_nonspatial_duplicate_identity_refuses(responses):
    obj,_=decoder(responses)
    with pytest.raises(ValueError,match='conflict|duplicate'):
        obj.decode(np.zeros((640,1120,3),np.uint8),evidence(),now=1.,parked=True)


def test_negative_unreadable_and_unknown_text_never_filled_from_configuration():
    obj,_=decoder([('',None)])
    blank=obj.decode(np.zeros((100,100,3),np.uint8),evidence(),now=1.,parked=True)
    assert not blank.detections and not blank.unreadable_corners
    obj,_=decoder([('',box())])
    unknown=obj.decode(np.zeros((100,100,3),np.uint8),evidence(),now=1.,parked=True)
    assert not unknown.detections and len(unknown.unreadable_corners)==1
    obj,_=decoder([('unexpected-pixel-text',box())])
    observed=obj.decode(np.zeros((100,100,3),np.uint8),evidence(),now=1.,parked=True)
    assert observed.detections[0].text=='unexpected-pixel-text'


def test_budget_failure_discards_partial_result_and_cannot_retry_frame(monkeypatch):
    clock=[0.]
    monkeypatch.setattr(qr.time,'perf_counter',lambda:clock[0])
    obj=qr.FixedTileQR('camera')
    def slow(image):
        clock[0]=.501
        return 'actual',box(),None
    obj.detector=SimpleNamespace(detectAndDecode=slow)
    with pytest.raises(ValueError,match='500 ms'):
        obj.decode(np.zeros((100,100,3),np.uint8),evidence(),now=1.,parked=True)
    with pytest.raises(ValueError,match='replayed'):
        obj.decode(np.zeros((100,100,3),np.uint8),evidence(),now=1.,parked=True)


def test_replay_after_session_rollback_and_invalid_pixels_refuse():
    obj,_=decoder([('',None),('',None)])
    rgb=np.zeros((100,100,3),np.uint8)
    obj.decode(rgb,evidence(),now=1.,parked=True)
    obj.decode(rgb,evidence(session='new'),now=1.,parked=True)
    with pytest.raises(ValueError,match='replayed'):
        obj.decode(rgb,evidence(2,2.),now=2.,parked=True)
    for pixels in (rgb.astype(float),rgb[:,:,0]):
        with pytest.raises(ValueError,match='uint8 RGB'):
            qr.FixedTileQR('camera').decode(pixels,evidence(),now=1.,parked=True)


def test_real_blank_pixels_do_not_create_inventory():
    result=qr.FixedTileQR('camera').decode(np.full((320,320,3),255,np.uint8),
                                         evidence(),now=1.,parked=True)
    assert result.detections==() and result.unreadable_corners==()


@pytest.mark.parametrize('layout', ['contiguous','rgba_view','reversed','transposed','readonly'])
def test_rgb_digest_preserves_original_c_order_bytes_for_all_layouts(layout):
    original = np.arange(13*17*4,dtype=np.uint8).reshape(13,17,4)
    pixels = original[:,:,:3]
    if layout == 'contiguous':
        pixels = pixels.copy()
    elif layout == 'reversed':
        pixels = pixels[::-1,::-1,::-1]
    elif layout == 'transposed':
        pixels = pixels.transpose(1,0,2)
    elif layout == 'readonly':
        pixels.setflags(write=False)
    before = original.copy()
    assert qr.rgb_digest(pixels) == hashlib.sha256(pixels.tobytes(order='C')).hexdigest()
    np.testing.assert_array_equal(original,before)


@pytest.mark.parametrize('strided', [False,True])
def test_hash_uses_independent_snapshot_even_if_source_changes_during_hash(monkeypatch,strided):
    backing = np.zeros((20,30,4 if strided else 3),np.uint8)
    pixels = backing[:,:,:3]
    expected = hashlib.sha256(pixels.tobytes()).hexdigest()
    actual_sha256 = hashlib.sha256
    def mutate_source(buffer):
        assert buffer.flags.c_contiguous
        assert not np.shares_memory(buffer,pixels)
        backing[0,0,0] = 1
        return actual_sha256(buffer)
    monkeypatch.setattr(qr.hashlib,'sha256',mutate_source)
    assert qr.rgb_digest(pixels) == expected
    assert qr.rgb_digest(pixels) != expected  # every call resnapshots current pixels


def test_shared_validation_rehashes_mutated_strided_pixels_every_time():
    backing = np.zeros((100,100,4),np.uint8)
    rgb = backing[:,:,:3]
    obj,_ = decoder([('actual',box())])
    result = obj.decode(rgb,evidence(),now=1.,parked=True)
    result.validate(rgb,evidence(),now=1.,parked=True)
    backing[99,99,2] = 1  # outside the detected QR; still part of exact frame
    with pytest.raises(ValueError,match='another frame'):
        result.validate(rgb,evidence(),now=1.,parked=True)
    backing[99,99,2] = 0
    result.validate(rgb,evidence(),now=1.,parked=True)
    with pytest.raises(ValueError,match='another frame'):
        replace(result,processing_seconds=.501).validate(rgb,evidence(),now=1.,parked=True)
