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


def test_recording_preserves_gaps_but_rejects_repeated_ids_or_clock_rollback(tmp_path):
    np=pytest.importorskip('numpy')
    pytest.importorskip('PIL')
    r=Recorder(tmp_path/'frames')
    rgb=np.zeros((4,4,3),dtype=np.uint8)
    r.add(rgb,render_id='fixture1',simulation_time=0.)
    with pytest.raises(ValueError,match='render ID'):
        r.add(rgb,render_id='fixture1',simulation_time=1/30)
    r.add(rgb,render_id='fixture2',simulation_time=1.)
    with pytest.raises(ValueError,match='strictly increase'):
        r.add(rgb,render_id='fixture3',simulation_time=1.)
    with pytest.raises(ValueError,match='strictly increase'):
        r.add(rgb,render_id='fixture3',simulation_time=.9)
    assert len(r.frames)==2
    assert (tmp_path/'frames/frames.json').is_file()


def test_vfr_encode_decodes_exact_count_and_irregular_timestamps(tmp_path):
    np=pytest.importorskip('numpy')
    encoder=pytest.importorskip('imageio_ffmpeg').get_ffmpeg_exe()
    r=Recorder(tmp_path/'frames',fps=60)
    stamps=[1.4833334107,1.5500000808,1.5666667475,1.7166667]
    for i,t in enumerate(stamps):
        r.add(np.full((16,16,3),i*60,dtype=np.uint8),render_id=f'fixture:{i}',simulation_time=t)
    result=r.encode(tmp_path/'fixture.mp4',ffmpeg=encoder)
    assert result['frames']==4 and result['timing']=='VFR'
    assert result['decoded_presentation_times']==pytest.approx([t-stamps[0] for t in stamps],abs=2e-6)
    restored=Recorder.from_manifest(r.directory/'frames.json')
    (r.directory/'000000.png').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='hash mismatch'):
        restored.encode(tmp_path/'corrupted.mp4',ffmpeg=encoder)


def test_recording_refuses_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        Recorder(tmp_path)
