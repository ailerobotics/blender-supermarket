"""Recorder refusal tests use pixel fixtures, not actual simulator evidence."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.restock.recording import Recorder
from twin.restock.isaac_bootstrap import import_mobile


def test_missing_tool_frames_fail_before_isaac_import(tmp_path):
    source=tmp_path/'fixture.urdf'
    source.write_text('<robot name="fixture"><link name="base_link"/></robot>')
    with pytest.raises(ValueError,match='actual gripper frames'):
        import_mobile(source,tmp_path/'robot.usda')
    assert not (tmp_path/'robot.usda').exists()


def test_recording_rejects_repeated_or_missing_frames(tmp_path):
    np=pytest.importorskip('numpy')
    pytest.importorskip('PIL')
    r=Recorder(tmp_path/'frames')
    rgb=np.zeros((4,4,3),dtype=np.uint8)
    r.add(rgb,render_id='fixture1',simulation_time=0.)
    with pytest.raises(ValueError,match='render ID'):
        r.add(rgb,render_id='fixture1',simulation_time=1/30)
    with pytest.raises(ValueError,match='cadence'):
        r.add(rgb,render_id='fixture2',simulation_time=1.)
    r.add(rgb,render_id='fixture2',simulation_time=1/30)
    assert len(r.frames)==2
    assert (tmp_path/'frames/frames.json').is_file()


def test_recording_refuses_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        Recorder(tmp_path)
