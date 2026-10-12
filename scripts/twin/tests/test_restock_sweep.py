"""Offline sweep regression checks; these do not certify robot motion."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
pytest.importorskip('pxr')
from pxr import Usd, UsdGeom
from check_restock_sweep import audit, corners, mesh_bounds


def test_disconnected_frame_preserves_opening_without_omitting_solid(tmp_path):
    path = tmp_path/'frame.usda'
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, '/shelf_frame')
    left = corners((-1.,-.1,0.),(-.9,.1,2.))
    right = corners((.9,-.1,0.),(1.,.1,2.))
    mesh.CreatePointsAttr(np.r_[left,right].tolist())
    faces = [0,1,3,2,4,6,7,5,0,4,5,1,2,3,7,6,0,2,6,4,1,5,7,3]
    mesh.CreateFaceVertexCountsAttr([4]*12)
    mesh.CreateFaceVertexIndicesAttr(faces+[v+8 for v in faces])
    stage.GetRootLayer().Save()
    boxes = mesh_bounds(path)
    assert len(boxes) == 2
    assert all(not (low[0] < 0 < high[0]) for _,low,high in boxes)
    assert np.allclose(np.min([b[1] for b in boxes],axis=0),(-1,-.1,0))
    assert np.allclose(np.max([b[2] for b in boxes],axis=0),(1,.1,2))


def test_stale_pose_evidence_fails_before_geometry_loading():
    with pytest.raises(ValueError,match='current scenario'):
        audit({}, {'changed': True}, {'scenario_sha256': 'old'}, None, [])
