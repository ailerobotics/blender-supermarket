"""Synthetic clock/pixel fixtures test buffering, never native sensing quality."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.restock.camera_frames import CameraFrames


def frame(identity,stamp):
    return dict(session_id='fixture',steady_domain='fixture-clock',acquired_steady=100+stamp,
        rendering_frame=identity,rendering_time=stamp,rgb=np.zeros((4,4,4),np.uint8),
        distance_to_image_plane=np.ones((4,4),np.float32))


def queues(**kwargs):
    result=CameraFrames({'source':SimpleNamespace(),'cart':SimpleNamespace()},
                        {'stock':'source','rollwagen':'cart'},**kwargs)
    result.begin_epoch(1)
    return result


def append(q,name,i,t):
    return q.append(name,frame(i,t),base_from_camera=np.eye(4),transform_stamp=t,now=t,parked=True)


def test_mapper_lag_selects_original_pair_not_latest_pixels():
    q=queues()
    for i,t in enumerate((1.,1.1,1.2)):
        append(q,'source',i,t);append(q,'cart',i,t+.01)
    assert not q.select('stock','rollwagen',now=1.22,collision_time=.99)
    assert q.select('stock','rollwagen',now=1.22,collision_time=1.12)
    f=q.cameras['source'].get_current_frame()
    assert f['rendering_frame']==1 and f['rendering_time']==1.1
    assert q.context('source',f,1.22)[1:]==(1.1,'1',True)
    assert not f['rgb'].flags.writeable
    assert not q.select('stock','rollwagen',now=1.22,collision_time=1.12)
    assert q.select('stock','rollwagen',now=1.22,collision_time=1.22)


def test_epoch_discards_old_pixels_without_accepting_replayed_ids():
    q=queues();append(q,'source',1,1.);append(q,'cart',1,1.)
    assert q.select('stock','rollwagen',now=1.,collision_time=1.)
    q.begin_epoch(2)
    assert not append(q,'source',1,1.)
    with pytest.raises(ValueError,match='backwards'):
        append(q,'source',0,.9)
    with pytest.raises(ValueError,match='no mapper-incorporated'):
        q.cameras['source'].get_current_frame()
    assert not q.select('stock','rollwagen',now=1.,collision_time=1.)


def test_skew_stale_map_and_memory_eviction_never_create_coverage():
    q=queues(max_bytes=256)
    append(q,'source',1,1.);append(q,'cart',1,1.1)
    assert not q.select('stock','rollwagen',now=1.1,collision_time=1.1)
    append(q,'source',2,1.11)
    assert q._bytes()<=256
    assert not q.select('stock','rollwagen',now=2.,collision_time=1.11)
    assert q.select('stock','rollwagen',now=1.12,collision_time=1.12)
    with pytest.raises(ValueError,match='context'):
        q.context('source',frame(9,1.11),1.12)


def test_cached_frame_cannot_receive_new_clock_or_unsynchronized_transform():
    q=queues();append(q,'source',1,1.)
    with pytest.raises(ValueError,match='restamped'):
        append(q,'source',1,1.01)
    with pytest.raises(ValueError,match='synchronized'):
        q.append('cart',frame(2,1.1),base_from_camera=np.eye(4),transform_stamp=1.2,now=1.2,parked=True)
