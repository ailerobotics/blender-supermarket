"""Offline unit tests for the mission-run wall profiler (no GPU, no datasets tree).

Synthetic console + nav-trace fixtures exercise the boot/shutdown anchoring, the
phase grouping and the render-vs-floor attribution; a round-trip over a tmp run
dir checks the top-level assembly and that the phase slices sum to the loop wall.
"""
import json

from scripts.twin.tools.profile_mission_run import (
    parse_console,
    parse_nav_trace,
    commissioning_wall_s,
    profile_run,
)

CONSOLE = """\
[0.001s] boot begin
[6.900s] app ready
2026-10-15T14:31:56Z [10,116ms] [Warning] [omni.usd] Startup Complete
[10.116s] Simulation App Startup Complete
2026-10-15T14:32:11Z [25,100ms] [Warning] something late in boot
CLOCK_DIAG tick=1 stage_time=0.0167 world_current_time=0.0667 wall=0.07
MANIP_START operation=collect
COLLECT_CAPTURE calls=1 sku=juice_orange roi=(1,2,3,4) pose=(0,0,0)
COLLECT_CAPTURE calls=2 sku=juice_orange roi=(1,2,3,4) pose=(0,0,0)
COLLECT_CAPTURE calls=3 provider_error=ValueError:no new synchronized rendered camera frame
COLLECT_CAPTURE calls=4 provider_error=ValueError:QR decoder refuses replayed camera frame/session
CLOCK_DIAG tick=21 stage_time=0.35 world_current_time=0.4 wall=7.0
[60.000s] Simulation App Shutting Down
"""


def test_parse_console_anchors_and_outcomes():
    cp = parse_console(CONSOLE)
    assert cp.app_ready_s == 6.9
    assert cp.startup_complete_s == 10.116
    # boot = largest process stamp before the loop's first tick (the 25.1s ms stamp)
    assert cp.loop_start_s == 25.1
    assert cp.shutdown_s == 60.0
    assert cp.collect_capture_ok == 2
    assert cp.provider_errors == {
        "no new synchronized rendered camera frame": 1,
        "QR decoder refuses replayed camera frame/session": 1,
    }


def _trace(deltas):
    """Build nav-trace rows from (wall_delta, action, rendered) tuples."""
    rows = [{"wall": 0.0, "t": 0.0, "action": deltas[0][1], "rendered": False}]
    wall = 0.0
    t = 0.0
    for dw, action, rendered in deltas:
        wall += dw
        t += 0.0167
        rows.append({"wall": round(wall, 4), "t": round(t, 4),
                     "action": action, "rendered": rendered})
    return rows


def test_parse_nav_trace_phase_grouping_and_sum():
    rows = _trace([
        (0.1, "collect", False),
        (0.1, "collect", False),
        (2.0, "collect", True),      # walkthrough render landed
        (0.1, "collect", False),
        (2.5, "collect", True),
        (0.5, "approach", False),
    ])
    lp = parse_nav_trace(rows)
    assert lp.ticks == len(rows)
    # phase wall slices partition the whole loop wall
    total_phase = sum(lp.phase_wall_s.values())
    assert abs(total_phase - lp.loop_wall_s) < 1e-6
    assert lp.phase_wall_s["collect_observe"] > lp.phase_wall_s["approach_reach"]
    # render + floor-bound time partition the loop exactly
    assert abs((lp.render_s + lp.step_ros_idle_s) - lp.loop_wall_s) < 1e-6
    # the two 2.0s+ rendered ticks dominate render
    assert lp.render_s > lp.step_ros_idle_s
    assert lp.walkthrough_render_s > 0
    assert lp.sensing_render_s >= 0


def test_empty_nav_trace_is_safe():
    lp = parse_nav_trace([])
    assert lp.ticks == 0 and lp.loop_wall_s == 0.0


def test_commissioning_wall_from_telemetry(tmp_path):
    rj = tmp_path / "run.json"
    rj.write_text(json.dumps({"telemetry": [
        {"wall_time": 1000.0}, {"wall_time": 1001.0}, {"wall_time": 1042.5},
    ]}))
    assert commissioning_wall_s(rj) == 42.5
    assert commissioning_wall_s(tmp_path / "missing.json") is None


def test_profile_run_assembles_from_dir(tmp_path):
    run = tmp_path / "run-x"
    run.mkdir()
    (run / "mission.json").write_text(json.dumps(
        {"state": "failed", "reason": "sensor stream watchdog expired"}))
    (run.parent / "run-x.console.log").write_text(CONSOLE)
    rows = _trace([(0.1, "collect", False), (2.0, "collect", True),
                   (0.1, "collect", False)])
    (run / "nav-trace.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    frames = run / "mission-frames"
    frames.mkdir()
    (frames / "frames.json").write_text(json.dumps({"frames": [{}, {}]}))

    comm = tmp_path / "comm"
    comm.mkdir()
    (comm / "run.json").write_text(json.dumps(
        {"telemetry": [{"wall_time": 0.0}, {"wall_time": 40.0}]}))

    rp = profile_run(run, comm)
    assert rp.state == "failed"
    assert rp.boot_s == 25.1
    assert rp.commission_s == 40.0
    assert rp.frames_saved == 2
    assert rp.collect_capture_ok == 2
    # collect is the only controller phase present
    assert rp.collect_observe_s > 0 and rp.approach_reach_s == 0.0
    # shutdown tail = 60 - 25.1 - loop_wall, clamped at >= 0
    assert rp.shutdown_s is not None and rp.shutdown_s >= 0.0
    # mission process time is the shutdown stamp
    assert rp.mission_process_s == 60.0
    assert rp.attempt_total_s == 100.0
