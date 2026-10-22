"""CPU tests for twin.demo.extract_stage_segment.

Synthetic run dirs only: a mission.json event log with a performed (or failed) collect stage,
a nav-trace whose ``action`` is 'collect' over a window, and overhead frames whose
simulation_time maps (via the clock offset) into that window.
"""
import json

import pytest
from PIL import Image

from twin.demo import extract_stage_segment as X
from twin.demo import run_evidence as E


def _make_collect_run(tmp_path, *, succeeded=True, n_collect=6, n_couple=3, offset=0.55):
    run_dir = tmp_path / "run-col"
    run_dir.mkdir()

    events = [{"sequence": 1, "state": "reserved", "evidence": {}},
              {"sequence": 2, "state": "collect", "evidence": {}}]
    if succeeded:
        events.append({"sequence": 3, "state": "collect_succeeded",
                       "evidence": {"source": "sensor", "simulation_time": 1.0}})
        events.append({"sequence": 4, "state": "couple", "evidence": {}})
        events.append({"sequence": 5, "state": "couple_succeeded",
                       "evidence": {"source": "sensor", "simulation_time": 2.0}})
    else:
        events.append({"sequence": 3, "state": "failed",
                       "evidence": {"reason": "arm command heartbeat lost [x.py:1 f]"}})
    mission = {"state": "complete" if succeeded else "failed",
               "reason": "" if succeeded else "boom", "events": events}
    (run_dir / "mission.json").write_text(json.dumps(mission))

    # nav-trace: collect window t=0.0..(n_collect-1)*0.1, then (if succeeded) couple.
    rows = []
    t = 0.0
    for i in range(n_collect):
        rows.append({"tick": i + 1, "t": round(t, 3), "action": "collect",
                     "x": -5.0 + 0.1 * i, "y": 16.0, "yaw_deg": 0.0, "lin": 0.2,
                     "coupled": False, "parked": False, "world_t": round(t + offset, 3)})
        t += 0.1
    if succeeded:
        t += 0.5   # gap between the collect and couple windows so the boundary frame is unambiguous
        for i in range(n_couple):
            rows.append({"tick": n_collect + i + 1, "t": round(t, 3), "action": "couple",
                         "x": -4.5, "y": 15.25, "yaw_deg": 0.0, "lin": 0.1,
                         "coupled": i > 0, "parked": False, "world_t": round(t + offset, 3)})
            t += 0.1
    with (run_dir / "nav-trace.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # bt_events: a collect stage_begin/step + (if succeeded) a couple event.
    bt = [{"t": 0.0, "type": "stage_begin", "data": {"stage": "collect", "steps": ["grasp"]},
           "action": "stage"},
          {"t": 0.2, "type": "stage_step", "data": {"stage": "collect", "label": "grasp"},
           "action": "stage"}]
    if succeeded:
        bt.append({"t": 1.0, "type": "stage_begin", "data": {"stage": "couple"}, "action": "stage"})
    with (run_dir / "bt_events.jsonl").open("w") as f:
        for e in bt:
            f.write(json.dumps(e) + "\n")

    # frames: one per trace row, simulation_time = world_t (so offset lands them in-window).
    fdir = run_dir / "mission-frames"
    fdir.mkdir()
    frames = []
    for i, r in enumerate(rows):
        name = f"{i:06d}.png"
        Image.new("RGB", (960, 720), (200, 200, 200)).save(fdir / name)
        # +1 ms past the row's world_t so the offset-corrected frame time lands cleanly INSIDE
        # the row's action interval (not exactly on the boundary, where float error could spill it
        # into the adjacent stage).
        frames.append({"render_id": f"id{i}", "simulation_time": r["world_t"] + 0.001,
                       "shape": [720, 960, 3], "file": name, "sha256": f"sha{i}"})
    (fdir / "frames.json").write_text(json.dumps({"nominal_fps": 30, "frames": frames}))
    return run_dir


def test_extracts_succeeded_collect_segment(tmp_path):
    run_dir = _make_collect_run(tmp_path, succeeded=True, n_collect=6, n_couple=3)
    out = tmp_path / "collect-src-001"
    summary = X.extract(run_dir, "collect", out)

    assert summary["status"] == "succeeded"
    assert summary["n_frames"] == 6                  # only the collect-window frames
    assert summary["frame_range"] == [0, 5]

    manifest = json.loads((out / "mission-frames" / "frames.json").read_text())
    assert manifest["source_run"] == "run-col"
    assert manifest["source_stage"] == "collect"
    assert len(manifest["frames"]) == 6
    # renumbered from 0, original metadata preserved
    assert manifest["frames"][0]["file"] == "000000.png"
    assert manifest["frames"][0]["render_id"] == "id0"
    assert manifest["frames"][0]["sha256"] == "sha0"
    assert (out / "mission-frames" / "000005.png").is_file()

    bt = [json.loads(l) for l in (out / "bt_events.jsonl").read_text().splitlines()]
    assert bt and all(e["data"]["stage"] == "collect" for e in bt)

    nav = [json.loads(l) for l in (out / "nav-trace.jsonl").read_text().splitlines()]
    assert len(nav) == 6 and all(r["action"] == "collect" for r in nav)

    readme = (out / "README.md").read_text()
    assert "run-col" in readme and "collect" in readme and "0..5" in readme


def test_refuses_incomplete_collect_without_flag(tmp_path):
    run_dir = _make_collect_run(tmp_path, succeeded=False)
    out = tmp_path / "seg"
    with pytest.raises(SystemExit):
        X.extract(run_dir, "collect", out)


def test_allows_incomplete_collect_with_flag(tmp_path):
    run_dir = _make_collect_run(tmp_path, succeeded=False, n_collect=4, n_couple=0)
    out = tmp_path / "seg"
    summary = X.extract(run_dir, "collect", out, require_succeeded=False)
    assert summary["n_frames"] == 4
    assert summary["status"] in ("started", "failed")


def test_overwrites_existing_out_dir(tmp_path):
    run_dir = _make_collect_run(tmp_path, succeeded=True)
    out = tmp_path / "seg"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    X.extract(run_dir, "collect", out)
    assert not (out / "stale.txt").exists()
