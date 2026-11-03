"""CPU tests for the per-stage mission audit (twin.demo.stage_audit_core + scripts/stage_audit.py).

Runs the audit on the small committed coldwin16 (complete cold cycle) and boxc1a (collect-only,
failed) fixtures and asserts the table (results, named reasons, sim spans, attempts) and the clip
list. Fixtures carry real mission.json + mission-progress.jsonl + a downsampled nav-trace + a few
synthetic frames (NOT the real PNGs). cv2 required for the clip assertions.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # scripts/ for stage_audit.py
from twin.demo import stage_audit_core as A

FIX = Path(__file__).resolve().parent / "fixtures" / "stage_audit"
STAGES = ("collect", "couple", "outbound", "park", "transfer", "verify", "return")


def _rows(audit):
    return {r["stage"]: r for r in audit["rows"]}


def test_coldwin16_table_all_pass():
    audit = A.audit_run(FIX / "coldwin16")
    assert audit["state"] == "complete"
    assert audit["result_line"] == "MISSION COMPLETE"
    assert audit["stop_after"] == "return"
    rows = _rows(audit)
    assert [rows[s]["result"] for s in STAGES] == ["PASS"] * 7
    assert all(rows[s]["reason"] == "" for s in STAGES)
    assert all(rows[s]["attempts"] == 1 for s in STAGES)
    # Sim-time spans come from mission.json stage_timeline (exact, trim-independent).
    assert rows["collect"]["sim_start_s"] == 0.0
    assert abs(rows["couple"]["sim_start_s"] - 164.8) < 0.1
    assert abs(rows["return"]["sim_end_s"] - 400.93) < 0.1
    # Wall spans are present (interpolated from the trace) and monotonic across the run.
    assert rows["collect"]["wall_start_s"] is not None
    assert rows["return"]["wall_end_s"] > rows["collect"]["wall_start_s"]
    # Key metrics the stages log.
    assert rows["couple"]["metrics"].get("coupled") is True
    assert rows["outbound"]["metrics"].get("arrived") is True
    assert rows["return"]["metrics"].get("home") is True


def test_boxc1a_collect_failed_named_reason():
    audit = A.audit_run(FIX / "boxc1a")
    assert audit["state"] == "failed"
    assert "COLLECT" in audit["result_line"]
    rows = _rows(audit)
    assert rows["collect"]["result"] == "FAIL"
    assert rows["collect"]["reason"] == "blocked_no_route_timeout"
    assert rows["collect"]["attempts"] == 3          # three collect attempts in the timeline
    # Everything after the stop point is SKIPPED, with the stop_after note.
    for s in ("couple", "outbound", "park", "transfer", "verify", "return"):
        assert rows[s]["result"] == "SKIPPED"
        assert "stop_after=collect" in rows[s]["note"]


def test_markdown_table_renders():
    audit = A.audit_run(FIX / "boxc1a")
    md = A.markdown_table(audit)
    assert "boxc1a — MISSION FAILED at COLLECT" in md
    assert "| collect | FAIL | blocked_no_route_timeout |" in md


def test_cli_writes_md_json_html_and_clips(tmp_path):
    import stage_audit as CLI
    out = tmp_path / "audit"
    CLI.main([str(FIX / "coldwin16"), str(FIX / "boxc1a"), "--out", str(out), "--fps", "10"])
    assert (out / "stage_audit.md").is_file()
    assert (out / "index.html").is_file()
    data = json.loads((out / "stage_audit.json").read_text())
    assert [r["run"] for r in data["runs"]] == ["coldwin16", "boxc1a"]

    clips = {r["run"]: set(r["clips"]) for r in data["runs"]}
    # The fixtures seed frames only for the stages that had recorded frames in the real runs.
    assert clips["coldwin16"] == {"collect", "couple", "outbound", "transfer", "return"}
    assert clips["boxc1a"] == {"collect"}
    for run in ("coldwin16", "boxc1a"):
        for stage in clips[run]:
            clip = out / "clips" / f"{run}__{stage}.mp4"
            assert clip.is_file() and clip.stat().st_size > 0


def test_cli_no_clips_table_only(tmp_path):
    import stage_audit as CLI
    out = tmp_path / "audit_nc"
    CLI.main([str(FIX / "coldwin16"), "--out", str(out), "--no-clips"])
    assert (out / "stage_audit.md").is_file()
    assert not (out / "clips").exists()
    data = json.loads((out / "stage_audit.json").read_text())
    assert data["runs"][0]["clips"] == {}
