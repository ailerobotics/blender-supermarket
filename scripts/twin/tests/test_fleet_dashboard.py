"""CPU tests for the read-only fleet dashboard aggregation, store overlay, trigger
queue and the HTTP server's degrade path. No network, no GPU, no simulator: the live
``status.json`` fetch is stubbed and every file read goes through ``tmp_path``.

Run (dodging the box's ROS pytest plugins)::

    PYTHONPATH=scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
      python3 -m pytest scripts/twin/tests/test_fleet_dashboard.py -q
"""
import json
import os

import pytest

from twin.dashboard import aggregate, server, snapshot, store_state, triggers


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


# ---------------------------------------------------------------- static run-dir panel

def _replay_run(root, name="run-5"):
    """A saved run directory: mission.json + nav-trace tail + bt snapshot + a frame."""
    d = root / name
    (d / "mission-frames").mkdir(parents=True)
    (d / "mission.json").write_text(json.dumps(_mission(
        [{"state": "reserved"}, {"state": "collect"}, {"state": "couple"}],
        state="failed", reason="transfer jam", physics_verified=False)))
    # two trace lines; the panel reads the last well-formed one
    (d / "nav-trace.jsonl").write_text(
        '{"tick": 1, "t": 0.1, "action": "collect", "coupled": false, "parked": true}\n'
        '{"tick": 42, "t": 9.9, "action": "transfer", "coupled": true, "parked": true,'
        ' "x": -2.0, "y": 15.2, "lin": 0.0}\n')
    (d / "bt_snapshot.json").write_text(json.dumps(_bt_snapshot()))
    (d / "mission-frames" / "000042.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (d / "walkthrough.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return d


def test_pulse_from_run_dir_reads_tail_and_counts(tmp_path):
    run = _replay_run(tmp_path)
    pulse = snapshot.pulse_from_run_dir(str(run))
    assert pulse["run"] == "run-5"
    assert pulse["frames"] == 1
    assert pulse["alive"] is False                      # honest: unknown from files
    assert pulse["trace"]["action"] == "transfer"       # last well-formed line
    assert pulse["trace"]["tick"] == 42
    assert pulse["mission"]["state"] == "failed"


def test_pulse_tolerates_empty_or_missing(tmp_path):
    (tmp_path / "empty").mkdir()
    pulse = snapshot.pulse_from_run_dir(str(tmp_path / "empty"))
    assert pulse["frames"] == 0
    assert pulse["trace"]["action"] is None
    assert pulse["mission"]["state"] is None


def test_build_run_snapshot_relative_media_and_timeline(tmp_path):
    _replay_run(tmp_path)
    snap = snapshot.build_run_snapshot(str(tmp_path))     # newest run, no media base
    assert snap["schema"] == "fm-fleet-dashboard.v1"
    assert snap["mission"]["run"] == "run-5"
    assert snap["mission"]["timeline"]["verdict"] == "failed"
    # the live-trace action overrides the inferred stage: transfer is active→failed
    by = {s["stage"]: s["status"] for s in snap["mission"]["timeline"]["steps"]}
    assert by["transfer"] == "failed"
    # media is run-relative (served over the static mount), not an upstream URL
    assert snap["media"]["latest_frame"] == os.path.join("mission-frames", "000042.png")
    assert snap["media"]["walkthrough"] == "walkthrough.mp4"
    # bt auto-discovered; its full-view link is relative to the run dir
    assert snap["behaviour_tree"]["tick"] == 49
    assert snap["behaviour_tree"]["view_url"] == "bt_view.html"


def test_build_run_snapshot_media_base_links_absolute(tmp_path):
    _replay_run(tmp_path)
    snap = snapshot.build_run_snapshot(str(tmp_path), media_base="http://host:8787")
    assert snap["media"]["latest_frame"] == "http://host:8787/latest.jpg"
    assert snap["behaviour_tree"]["view_url"] == "http://host:8787/bt_view.html"


def test_build_run_snapshot_none_when_empty(tmp_path):
    assert snapshot.build_run_snapshot(str(tmp_path / "missing")) is None


def test_render_static_site_writes_pair(tmp_path):
    _replay_run(tmp_path)
    out = tmp_path / "run-5"
    snap = snapshot.generate(str(tmp_path), str(out), "run-5")
    assert snap is not None
    assert (out / "fleet.json").exists()
    assert (out / "fleet.html").exists()
    # the static page polls fleet.json relative to itself (no live endpoint)
    html = (out / "fleet.html").read_text()
    assert "fetch('fleet.json" in html
    assert "/status.json" not in html
    reloaded = json.loads((out / "fleet.json").read_text())
    assert reloaded["mission"]["run"] == "run-5"


def test_generate_none_run_writes_nothing(tmp_path):
    out = tmp_path / "out"
    assert snapshot.generate(str(tmp_path / "empty"), str(out), None) is None
    assert not out.exists()


# ---------------------------------------------------------------- approval variant

def test_enqueue_approval_lands_pending(tmp_path):
    d = str(tmp_path / "reqs")
    rec = triggers.enqueue_trigger(d, {"sku": "S1", "quantity": 2, "shelf": "G3_back_s2",
                                       "source": "backstock_00"}, approval_required=True)
    assert rec["status"] == "pending_approval"
    assert rec["approval"]["state"] == "pending"
    # plain queue path is unchanged (default off)
    plain = triggers.enqueue_trigger(d, {"sku": "S2", "quantity": 1})
    assert plain["status"] == "queued"
    assert "approval" not in plain


def test_decide_approve_emits_job_plan(tmp_path):
    d = str(tmp_path / "reqs")
    rec = triggers.enqueue_trigger(d, {"sku": "juice", "quantity": 1,
                                       "shelf": "G3_back_s2", "source": "backstock_00"},
                                   approval_required=True)
    decided = triggers.decide_trigger(d, rec["id"], approve=True, by="phone")
    assert decided["status"] == "approved"
    assert decided["approval"]["state"] == "approved"
    assert decided["approval"]["by"] == "phone"
    # an approved request is written as a job-plan the lane-1 --job-from feed honours
    plan_path = os.path.join(d, "approved", rec["id"] + ".json")
    plan = json.loads(open(plan_path).read())
    assert plan["schema"] == triggers.JOB_PLAN_SCHEMA
    assert plan["sku"] == "juice"
    assert plan["source"]["location"] == "backstock_00"
    assert plan["destination"]["slot_id"] == "G3_back_s2"
    assert plan["provenance"]["approval"]["state"] == "approved"


def test_decide_reject_no_plan(tmp_path):
    d = str(tmp_path / "reqs")
    rec = triggers.enqueue_trigger(d, {"sku": "S", "quantity": 1}, approval_required=True)
    decided = triggers.decide_trigger(d, rec["id"], approve=False, by="phone")
    assert decided["status"] == "rejected"
    assert not os.path.exists(os.path.join(d, "approved"))


def test_decide_rejects_unknown_or_wrong_state(tmp_path):
    d = str(tmp_path / "reqs")
    with pytest.raises(triggers.TriggerError):
        triggers.decide_trigger(d, "nope", approve=True)
    plain = triggers.enqueue_trigger(d, {"sku": "S", "quantity": 1})  # queued, not pending
    with pytest.raises(triggers.TriggerError):
        triggers.decide_trigger(d, plain["id"], approve=True)


def test_build_state_approval_flag_omitted_by_default():
    base = dict(upstream=None, upstream_ok=False, run_name="r", mission=None,
                store=None, behaviour_tree=None, triggers=[], upstream_base="http://h:8787")
    assert "approval_required" not in aggregate.build_state(**base)           # v1 byte-identical
    assert aggregate.build_state(**base, approval_required=True)["approval_required"] is True


def test_compute_status_surfaces_approval(tmp_path, monkeypatch):
    _write_run(tmp_path, "run-1", _mission([{"state": "collect"}]))
    monkeypatch.setattr(server, "fetch_upstream", lambda base, timeout: (None, False))
    cfg = server.Config(run_root=str(tmp_path), upstream_base="http://x:8787",
                        trigger_enabled=True, approval_required=True,
                        requests_dir=str(tmp_path / "reqs"))
    assert server.compute_status(cfg)["approval_required"] is True


def test_server_trigger_approve_roundtrip(tmp_path, monkeypatch):
    """End-to-end over the real handler: POST /trigger → /approve, and read back."""
    import threading
    import urllib.request

    _write_run(tmp_path, "run-1", _mission([{"state": "collect"}]))
    monkeypatch.setattr(server, "fetch_upstream", lambda base, timeout: (None, False))
    cfg = server.Config(host="127.0.0.1", port=0, run_root=str(tmp_path),
                        upstream_base="http://x:8787", trigger_enabled=True,
                        approval_required=True, requests_dir=str(tmp_path / "reqs"))
    httpd = server.make_server(cfg)
    threading.Thread(target=httpd.handle_request, daemon=True).start()  # /trigger
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(route, body):
        req = urllib.request.Request(base + route, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())

    code, rec = post("/trigger", {"sku": "juice", "quantity": 1, "shelf": "G3_back_s2"})
    assert code == 201 and rec["status"] == "pending_approval"

    threading.Thread(target=httpd.handle_request, daemon=True).start()  # /approve
    code, decided = post("/approve", {"id": rec["id"], "by": "tester"})
    assert code == 200 and decided["status"] == "approved"
    httpd.server_close()
    assert os.path.exists(os.path.join(str(tmp_path / "reqs"), "approved", rec["id"] + ".json"))


def test_server_rejects_approve_without_approval_mode(tmp_path, monkeypatch):
    import threading
    import urllib.error
    import urllib.request

    monkeypatch.setattr(server, "fetch_upstream", lambda base, timeout: (None, False))
    cfg = server.Config(host="127.0.0.1", port=0, run_root=str(tmp_path),
                        trigger_enabled=True, approval_required=False,
                        requests_dir=str(tmp_path / "reqs"))
    httpd = server.make_server(cfg)
    threading.Thread(target=httpd.handle_request, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    req = urllib.request.Request(base + "/approve", data=b"{}",
                                 headers={"content-type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=5)
    assert ei.value.code == 403
    httpd.server_close()
