"""CPU tests for the read-only fleet dashboard aggregation, store overlay, trigger
queue and the HTTP server's degrade path. No network, no GPU, no simulator: the live
``status.json`` fetch is stubbed and every file read goes through ``tmp_path``.

Run (dodging the box's ROS pytest plugins)::

    PYTHONPATH=scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
      python3 -m pytest scripts/twin/tests/test_fleet_dashboard.py -q
"""
import json

import pytest

from twin.dashboard import aggregate, server, store_state, triggers


# ---------------------------------------------------------------- stage timeline

def _mission(events, state=None, reason=None, physics_verified=None):
    m = {"events": events}
    if state is not None:
        m["state"] = state
    if reason is not None:
        m["reason"] = reason
    if physics_verified is not None:
        m["physics_verified"] = physics_verified
    return m


def test_timeline_running_marks_active_stage():
    tl = aggregate.stage_timeline(
        _mission([{"state": "reserved"}, {"state": "collect"}]),
        current_action="couple",
    )
    assert tl["verdict"] == "running"
    assert tl["active"] == "couple"
    by = {s["stage"]: s["status"] for s in tl["steps"]}
    assert by["reserved"] == "done"
    assert by["collect"] == "done"
    assert by["couple"] == "active"
    assert by["transfer"] == "pending"


def test_timeline_failed_stage_renders_failed_not_done():
    # The run recorded a `collect` entry event then a `failed` event: collect is where
    # it died, so it must read 'failed', not 'done'.
    tl = aggregate.stage_timeline(
        _mission(
            [{"state": "reserved"}, {"state": "collect"}, {"state": "failed"}],
            state="failed", reason="tuck move failed",
        ),
        current_action="collect",
    )
    assert tl["verdict"] == "failed"
    assert tl["reason"] == "tuck move failed"
    assert tl["failed_stage"] == "collect"
    by = {s["stage"]: s["status"] for s in tl["steps"]}
    assert by["reserved"] == "done"
    assert by["collect"] == "failed"
    assert by["couple"] == "pending"


def test_timeline_complete_marks_all_done():
    tl = aggregate.stage_timeline(
        _mission(
            [{"state": s} for s in aggregate.PIPELINE],
            state="complete", physics_verified=True,
        ),
    )
    assert tl["verdict"] == "complete"
    assert all(s["status"] == "done" for s in tl["steps"])
    assert tl["physics_verified"] is True


def test_timeline_tolerates_missing_mission():
    tl = aggregate.stage_timeline(None, None)
    assert tl["verdict"] == "running"
    assert tl["active"] is None
    assert all(s["status"] == "pending" for s in tl["steps"])
    assert tl["event_count"] == 0


# ---------------------------------------------------------------- run resolution

def _write_run(root, name, mission):
    d = root / name
    d.mkdir(parents=True)
    (d / "mission.json").write_text(json.dumps(mission))
    return d


def test_read_run_mission_by_name_and_fallback(tmp_path):
    _write_run(tmp_path, "run-1", _mission([{"state": "collect"}]))
    r2 = _write_run(tmp_path, "run-2", _mission([{"state": "verify"}]))
    # make run-2 the newest
    import os
    os.utime(r2 / "mission.json", (10 ** 9, 10 ** 9))

    name, mission = aggregate.read_run_mission(str(tmp_path), "run-1")
    assert name == "run-1"
    assert mission["events"][0]["state"] == "collect"

    # unknown run name -> newest run
    name, mission = aggregate.read_run_mission(str(tmp_path), "nope")
    assert name == "run-2"

    # no name -> newest
    name, _ = aggregate.read_run_mission(str(tmp_path), None)
    assert name == "run-2"


def test_read_run_mission_empty_root(tmp_path):
    assert aggregate.read_run_mission(str(tmp_path), None) == (None, None)
    assert aggregate.read_run_mission(str(tmp_path / "missing"), "x") == (None, None)


# ---------------------------------------------------------------- upstream merge

def test_normalise_upstream_tolerates_garbage():
    up = aggregate.normalise_upstream(None)
    assert up["alive"] is False
    assert up["frames"] == 0
    assert up["trace"]["action"] is None
    assert up["videos"] == []


def test_build_state_shape():
    upstream = {
        "run": "run-250", "alive": True, "frames": 215, "run_age_s": 100,
        "trace": {"tick": 5, "t": 1.0, "action": "collect", "coupled": False},
        "mission": {"state": "failed", "reason": "x"},
        "videos": ["mission/run-250/walkthrough.mp4"],
        "runs": [{"name": "run-250"}],
    }
    state = aggregate.build_state(
        upstream=upstream, upstream_ok=True, run_name="run-250",
        mission=_mission([{"state": "collect"}], state="failed", reason="x"),
        store=None, behaviour_tree=None, triggers=[],
        upstream_base="http://host:8787",
    )
    assert state["schema"] == "fm-fleet-dashboard.v1"
    assert state["upstream"]["ok"] is True
    assert state["mission"]["run"] == "run-250"
    assert state["mission"]["timeline"]["verdict"] == "failed"
    assert state["media"]["latest_frame"] == "http://host:8787/latest.jpg"
    assert state["mission"]["videos"] == ["mission/run-250/walkthrough.mp4"]


def test_build_state_result_falls_back_to_run_mission():
    # upstream down (no mission block): the result comes from the run's mission.json.
    state = aggregate.build_state(
        upstream=None, upstream_ok=False, run_name="run-9",
        mission=_mission([{"state": "verify"}], state="complete",
                         physics_verified=True),
        store=None, behaviour_tree=None, triggers=[],
        upstream_base="http://host:8787",
    )
    assert state["upstream"]["ok"] is False
    assert state["mission"]["result"]["state"] == "complete"
    assert state["mission"]["result"]["physics_verified"] is True


# ---------------------------------------------------------------- store overlay

def test_store_state_summary(tmp_path):
    export = {
        "generated_at": "2026-10-17T20:40:00Z",
        "source": "storevision.shelf",
        "on_shelf_availability_pct": 91.0,
        "slots": [
            {"slot": "A1", "sku": "S1", "fill_pct": 0.0, "state": "oos"},
            {"slot": "A2", "sku": "S2", "fill_pct": 60.0, "state": "low"},
            {"slot": "A3", "sku": "S3", "fill_pct": 100.0, "state": "ok"},
            # store-vision's native "limited" counts as needing restock, not oos.
            {"slot": "A4", "sku": "S4", "fill_pct": 30.0, "state": "limited"},
        ],
        "alerts": [
            {"type": "spill", "severity": "warn", "count": 1, "location": "aisle-3"},
            {"type": "obstruction", "severity": "critical", "count": 2},
            {"type": "misplace", "severity": "info", "count": 0},
        ],
    }
    p = tmp_path / "store.json"
    p.write_text(json.dumps(export))
    out = store_state.load_store_state(str(p))
    s = out["summary"]
    assert s["slot_count"] == 4
    assert s["needs_restock"] == 3       # oos + low + limited
    assert s["out_of_stock"] == 1        # oos only
    assert s["active_alerts"] == 2       # count>0
    assert s["top_severity"] == "critical"


def test_store_state_absent_or_bad(tmp_path):
    assert store_state.load_store_state(None) is None
    assert store_state.load_store_state(str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json {")
    assert store_state.load_store_state(str(bad)) is None


# ---------------------------------------------------------------- trigger queue

def test_trigger_validate_rejects_bad_input():
    with pytest.raises(triggers.TriggerError):
        triggers.validate_request("nope")
    with pytest.raises(triggers.TriggerError):
        triggers.validate_request({"quantity": 1})          # no sku
    with pytest.raises(triggers.TriggerError):
        triggers.validate_request({"sku": "S", "quantity": 0})
    with pytest.raises(triggers.TriggerError):
        triggers.validate_request({"sku": "S", "quantity": True})
    with pytest.raises(triggers.TriggerError):
        triggers.validate_request({"sku": "S", "quantity": 1, "priority": "high"})


def test_trigger_enqueue_and_list(tmp_path):
    d = str(tmp_path / "reqs")
    rec = triggers.enqueue_trigger(d, {"sku": " SKU-1 ", "quantity": 3, "shelf": "G3"})
    assert rec["status"] == "queued"
    assert rec["request"]["sku"] == "SKU-1"       # trimmed
    assert rec["request"]["quantity"] == 3
    listed = triggers.list_triggers(d)
    assert len(listed) == 1
    assert listed[0]["id"] == rec["id"]
    # a second, newer request lists first
    rec2 = triggers.enqueue_trigger(d, {"sku": "SKU-2", "quantity": 1})
    listed = triggers.list_triggers(d)
    assert [r["id"] for r in listed][0] in (rec["id"], rec2["id"])
    assert len(listed) == 2


def test_list_triggers_missing_dir():
    assert triggers.list_triggers("/nonexistent/path/xyz") == []


# ---------------------------------------------------------------- server degrade

def test_fetch_upstream_never_raises():
    # unreachable port -> (None, False), no exception
    data, ok = server.fetch_upstream("http://127.0.0.1:1", timeout=0.1)
    assert data is None and ok is False


def test_compute_status_with_upstream_down(tmp_path, monkeypatch):
    _write_run(tmp_path, "run-7", _mission([{"state": "collect"}], state="failed",
                                            reason="boom"))
    monkeypatch.setattr(server, "fetch_upstream", lambda base, timeout: (None, False))
    cfg = server.Config(run_root=str(tmp_path), upstream_base="http://x:8787")
    state = server.compute_status(cfg)
    assert state["upstream"]["ok"] is False
    # fell back to the newest run directory
    assert state["mission"]["run"] == "run-7"
    assert state["mission"]["timeline"]["verdict"] == "failed"


def test_compute_status_merges_upstream_run(tmp_path, monkeypatch):
    _write_run(tmp_path, "run-250", _mission(
        [{"state": "reserved"}, {"state": "collect"}], state="failed", reason="x"))
    fake = {
        "run": "run-250", "alive": True, "frames": 10,
        "trace": {"action": "collect"}, "mission": {"state": "failed", "reason": "x"},
        "videos": [], "runs": [],
    }
    monkeypatch.setattr(server, "fetch_upstream", lambda base, timeout: (fake, True))
    cfg = server.Config(run_root=str(tmp_path), upstream_base="http://x:8787")
    state = server.compute_status(cfg)
    assert state["upstream"]["ok"] is True
    assert state["mission"]["alive"] is True
    assert state["mission"]["run"] == "run-250"
    assert state["mission"]["timeline"]["active"] == "collect"


# ---------------------------------------------------------------- behaviour tree

def _bt_snapshot():
    # Mirrors data/sim-restock-bt-v1/view-demo/bt_snapshot.json (report §4).
    return {
        "tick": 49, "clock": 0.833,
        "mission": {"state": "complete", "reason": "", "degraded": []},
        "control": {"pause": False, "slow": True, "replan": False, "abort": None},
        "active_events": {"human_detected": {"zone": "slow"}},
        "last_event": {"t": 0.6, "type": "obstacle_cleared", "data": {}},
        "log": [{"t": i / 10, "type": "human_detected", "data": {}, "action": "slow"}
                for i in range(20)],
        "tree": {
            "name": "restock_root", "status": "success", "kind": "Sequence", "feedback": "",
            "children": [
                {"name": "event_pump", "status": "success", "kind": "Behaviour"},
                {"name": "reactive", "status": "success", "kind": "Sequence", "children": [
                    {"name": "human_safety", "status": "running", "kind": "Behaviour",
                     "feedback": "slow"},
                ]},
            ],
        },
    }


def test_flatten_bt_tree_depths():
    nodes = aggregate.flatten_bt_tree(_bt_snapshot()["tree"])
    names = [(n["name"], n["depth"]) for n in nodes]
    assert names == [
        ("restock_root", 0), ("event_pump", 1), ("reactive", 1), ("human_safety", 2),
    ]
    assert nodes[3]["status"] == "running"
    assert nodes[3]["feedback"] == "slow"


def test_flatten_bt_tree_tolerates_garbage():
    assert aggregate.flatten_bt_tree(None) == []
    assert aggregate.flatten_bt_tree({"name": "x", "status": "ok"}) == [
        {"name": "x", "status": "ok", "depth": 0, "feedback": None, "kind": None}]


def test_summarise_bt():
    out = aggregate.summarise_bt(_bt_snapshot(), view_url="http://host/mission/run-9/bt_view.html",
                                 log_tail=5)
    assert out["tick"] == 49
    assert out["control"]["slow"] is True
    assert out["last_event"]["type"] == "obstacle_cleared"
    assert len(out["nodes"]) == 4
    assert len(out["log"]) == 5           # tail
    assert out["view_url"].endswith("/bt_view.html")


def test_summarise_bt_none():
    assert aggregate.summarise_bt(None) is None
    assert aggregate.summarise_bt("nope") is None


def test_resolve_bt_autodiscovers_run_snapshot(tmp_path, monkeypatch):
    run = _write_run(tmp_path, "run-9", _mission([{"state": "collect"}]))
    (run / "bt_snapshot.json").write_text(json.dumps(_bt_snapshot()))
    fake = {"run": "run-9", "alive": True, "trace": {"action": "collect"},
            "mission": {}, "videos": [], "runs": []}
    monkeypatch.setattr(server, "fetch_upstream", lambda base, timeout: (fake, True))
    cfg = server.Config(run_root=str(tmp_path), upstream_base="http://host:8787")
    state = server.compute_status(cfg)
    bt = state["behaviour_tree"]
    assert bt is not None
    assert bt["tick"] == 49
    # view_url points at the watch page's run tree (<run_root basename>/<run>/bt_view.html)
    assert bt["view_url"] == f"http://host:8787/{tmp_path.name}/run-9/bt_view.html"


def test_resolve_bt_absent_is_none(tmp_path, monkeypatch):
    _write_run(tmp_path, "run-9", _mission([{"state": "collect"}]))  # no bt_snapshot.json
    monkeypatch.setattr(server, "fetch_upstream", lambda base, timeout: (None, False))
    cfg = server.Config(run_root=str(tmp_path), upstream_base="http://host:8787")
    assert server.compute_status(cfg)["behaviour_tree"] is None


def test_resolve_bt_explicit_path_wins(tmp_path):
    p = tmp_path / "snap.json"
    p.write_text(json.dumps(_bt_snapshot()))
    cfg = server.Config(run_root=str(tmp_path), bt_path=str(p), upstream_base="http://host:8787")
    bt = server.resolve_bt(cfg, None, None, "http://host:8787")
    assert bt["tick"] == 49
    assert bt["view_url"] is None          # explicit file -> no run-dir view link


def test_resolve_run_dir(tmp_path):
    _write_run(tmp_path, "run-a", _mission([]))
    assert aggregate.resolve_run_dir(str(tmp_path), "run-a").endswith("run-a")
    assert aggregate.resolve_run_dir(str(tmp_path), "ghost").endswith("run-a")  # fallback
    assert aggregate.resolve_run_dir(str(tmp_path / "missing"), None) is None


def test_default_bind_is_loopback_and_free_port():
    # Public-safe default: bind loopback on a free port (>=8794). The deploy host is
    # supplied via RESTOCK_DASHBOARD_HOST (e.g. a tailnet address), never baked in.
    assert server.DEFAULT_HOST == "127.0.0.1"
    assert server.DEFAULT_PORT >= 8794
    assert server.DEFAULT_PORT not in (8787, 8790, 8791, 8793)
