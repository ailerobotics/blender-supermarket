"""CPU tests for twin.demo.stage_video (the shared stage + sim-time overlay / MP4 writer).

Uses the small committed stage_audit fixtures (real mission.json + mission-progress.jsonl + a
downsampled nav-trace + a handful of synthetic frames) plus synthetic item lists. cv2 required.
"""
import sys
from pathlib import Path

import cv2
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.demo import stage_video as V

FIX = Path(__file__).resolve().parent / "fixtures" / "stage_audit"


def test_load_stage_ladder_from_mission_progress():
    ladder = V.load_stage_ladder(FIX / "coldwin16")
    stages = [s for _, s in ladder]
    # mission-progress.jsonl records one begin per stage, in order.
    assert stages == ["collect", "couple", "outbound", "park", "transfer", "verify", "return"]
    times = [t for t, _ in ladder]
    assert times == sorted(times)
    assert times[0] == 0.0


def test_stage_at_picks_the_last_entered_stage():
    ladder = V.load_stage_ladder(FIX / "coldwin16")
    assert V.stage_at(ladder, 0.0) == "collect"
    assert V.stage_at(ladder, 200.0) == "couple"      # between couple(164.8) and outbound(210.9)
    assert V.stage_at(ladder, 10_000.0) == "return"


def test_ladder_falls_back_to_nav_trace_actions(tmp_path):
    run = tmp_path / "run"
    (run).mkdir()
    (run / "nav-trace.jsonl").write_text(
        '{"t": 0.0, "action": "collect", "x": 0, "y": 0}\n'
        '{"t": 1.0, "action": "collect", "x": 0, "y": 0}\n'
        '{"t": 2.0, "action": "couple", "x": 0, "y": 0}\n')
    ladder = V.load_stage_ladder(run)
    assert ladder == [(0.0, "collect"), (2.0, "couple")]


def test_write_overlay_video_counts_and_opens(tmp_path):
    # 5 synthetic frames across two stages.
    items = []
    for i in range(5):
        p = tmp_path / f"f{i}.png"
        Image.new("RGB", (64, 48), (30, 60, 90)).save(p)
        items.append((p, float(i)))
    ladder = [(0.0, "collect"), (3.0, "couple")]
    out = tmp_path / "clip.mp4"
    summary = V.write_overlay_video(items, out, run_name="r", state="COMPLETE", ladder=ladder, fps=10)
    assert summary["frames"] == 5
    assert summary["stage_order"] == ["collect", "couple"]
    assert out.is_file() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    cap.release()


def test_write_overlay_video_rejects_empty(tmp_path):
    with pytest.raises(RuntimeError):
        V.write_overlay_video([], tmp_path / "x.mp4", run_name="r", state="S", ladder=[], fps=10)


def test_whole_run_video_over_fixture(tmp_path):
    out = tmp_path / "whole.mp4"
    summary = V.whole_run_video(FIX / "boxc1a", out, fps=10)
    # boxc1a fixture has 3 synthetic frames, all in collect.
    assert summary["frames"] == 3
    assert summary["state"] == "FAILED"
    assert summary["stage_order"] == ["collect"]
    assert out.is_file() and out.stat().st_size > 0
