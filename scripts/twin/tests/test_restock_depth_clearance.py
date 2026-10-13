from dataclasses import replace
import numpy as np
import pytest

from twin.restock.depth_clearance import DepthView,measure_clearance
from twin.restock_camera.geometry import Intrinsics


def view():
    return DepthView('test_camera','fixture_frame',1.,1.,1,np.ones((120,160)),
        Intrinsics(160,120,100.,100.,80.,60.,.1,2.),
        np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,1],[0,0,0,1.]]))


def check(v,**extra):
    return measure_clearance([v],base_from_support=np.eye(4),xy_bounds=(-.1,-.1,.1,.1),
        required_height=.2,location='fixture_only',epoch=1,now=1.1,collision_time=1.,
        parked=True,support_plane_stamp=1.,**extra)


def test_depth_plane_certifies_only_observed_requested_volume():
    result=check(view())
    assert result['free_dimensions']==(.2,.2,.2)
    assert result['cells']==1000
    assert result['verified_above_support_m']==.005


def test_foreground_and_missing_depth_refuse_volume():
    for value in (.85,0.,float('nan')):
        v=view();v.depth[55:65,75:85]=value
        with pytest.raises(ValueError,match='clearance cells'):
            check(v)


def test_epoch_and_transform_staleness_refused():
    for v in (replace(view(),epoch=2),replace(view(),transform_stamp=.5)):
        with pytest.raises(ValueError,match='no fresh'):
            check(v)


def test_cell_budget_is_bounded():
    with pytest.raises(ValueError,match='cell budget'):
        check(view(),max_cells=10)
