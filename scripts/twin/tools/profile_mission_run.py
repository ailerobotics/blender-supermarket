#!/usr/bin/env python3
"""Per-phase wall-time profile of a recorded Isaac mobile-restock mission run.

Read-only. Breaks a run's wall clock into the phases the integration lane cares
about -- boot, commissioning, collect-observe, approach/reach, per-frame render,
and ROS/host waits + idle -- from the artifacts a run already writes, with no GPU
and no re-run. The point is to see where a 5-20 minute iteration actually goes so
the top levers are chosen from numbers, not guesses.

Artifacts consumed (all under a run directory, e.g.
``.../mission/run-101`` plus the sibling ``run-101.console.log``):

* ``<run>.console.log`` -- the Isaac process stdout. Timestamped either
  ``[7.152s]`` (seconds since process start) or ``2026-..Z [25,122ms]``
  (ms since start). Boot = process start -> the first ``CLOCK_DIAG tick=1``
  line (the mission control loop's first tick). Shutdown = last loop activity ->
  ``Simulation App Shutting Down``. The orchestrator's own markers
  (``CLOCK_DIAG``, ``COLLECT_CAPTURE``, ``NAV_STEP``) live here too.
* ``nav-trace.jsonl`` -- one JSON object per control tick with ``wall`` (seconds
  since loop start), ``action`` (the mission phase: collect/approach/contact/
  lift/couple/tow/place/park) and ``rendered`` (whether the walkthrough camera
  rendered that tick). This is the primary source for the in-loop breakdown.
* ``mission.json`` -- terminal state + reason.
* ``mission-frames/frames.json`` -- saved capture count (optional).

Commissioning is a *separate* GPU run in the current dev loop (the mission run
loads a pre-commissioned, 900 s-lifetime baseline at boot; it does not commission
in-process). Point ``--commissioning`` at a commissioning run dir to fold its
GPU-hold wall (from ``run.json`` 1 Hz telemetry) into the per-attempt total.

Render attribution within the loop: the control loop steps physics at a fixed
rate, so each tick's irreducible cost is one physics step; everything above the
fastest-observed tick (the ``step_floor``, 5th percentile of tick wall-deltas) is
GPU render + the ROS/host round-trip the capture waits on. We split that overhead
into (a) the walkthrough-camera render -- the marginal extra on ticks flagged
``rendered`` over the median plain tick -- and (b) the remainder, dominated by the
per-tick sensing-camera capture the collect loop renders every tick. The three
sum exactly to the measured loop wall; see ``LoopProfile``.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# --- console timestamp extraction -------------------------------------------

# "[7.152s]" seconds-since-start, or "2026-..Z [25,122ms]" ms-since-start.
_SEC = re.compile(r"\[(\d+\.\d+)s\]")
_MS = re.compile(r"\[(\d+(?:,\d+)?)ms\]")


def _secs_before(text: str) -> list[float]:
    """All process-relative timestamps (in seconds) appearing in ``text``."""
    out = [float(x) for x in _SEC.findall(text)]
    out += [int(x.replace(",", "")) / 1000.0 for x in _MS.findall(text)]
    return out


@dataclass
class ConsoleProfile:
    app_ready_s: Optional[float] = None
    startup_complete_s: Optional[float] = None
    loop_start_s: Optional[float] = None       # process time at CLOCK_DIAG tick=1
    shutdown_s: Optional[float] = None          # process time at "Shutting Down"
    collect_capture_ok: int = 0
    provider_errors: dict = field(default_factory=dict)

    @property
    def boot_s(self) -> Optional[float]:
        return self.loop_start_s


def parse_console(text: str) -> ConsoleProfile:
    """Pull boot/shutdown anchors and capture outcomes from a console log."""
    p = ConsoleProfile()
    m = re.search(r"\[(\d+\.\d+)s\] app ready", text)
    if m:
        p.app_ready_s = float(m.group(1))
    m = re.search(r"\[(\d+\.\d+)s\] Simulation App Startup Complete", text)
    if m:
        p.startup_complete_s = float(m.group(1))

    idx = text.find("CLOCK_DIAG tick=1 ")
    if idx != -1:
        before = _secs_before(text[:idx])
        if before:
            # Boot is the largest process-relative stamp seen before the loop's
            # first tick (stamps are monotonic but interleaved across formats).
            p.loop_start_s = max(before)

    m = re.search(r"\[(\d+\.\d+)s\] Simulation App Shutting Down", text)
    if m:
        p.shutdown_s = float(m.group(1))

    p.collect_capture_ok = len(re.findall(r"COLLECT_CAPTURE calls=\d+ sku=", text))
    for mm in re.finditer(r"provider_error=ValueError:([^\n]+)", text):
        k = mm.group(1).strip()
        p.provider_errors[k] = p.provider_errors.get(k, 0) + 1
    return p


# --- nav-trace in-loop breakdown --------------------------------------------

# Controller phases grouped for the report.
_OBSERVE = {"collect"}
_APPROACH = {"approach", "reach", "contact", "lift", "grasp"}
# everything else (couple/tow/place/park/...) rolls into "other".


@dataclass
class LoopProfile:
    ticks: int = 0
    loop_wall_s: float = 0.0
    sim_s: float = 0.0
    phase_wall_s: dict = field(default_factory=dict)      # action-group -> wall
    rendered_ticks: int = 0
    step_floor_s: float = 0.0
    step_ros_idle_s: float = 0.0        # irreducible step + ROS/host waits + idle
    render_s: float = 0.0               # all GPU render overhead above the floor
    walkthrough_render_s: float = 0.0   # marginal nav-camera render
    sensing_render_s: float = 0.0       # per-tick sensing capture (render_s - walkthrough)

    @property
    def sim_to_wall(self) -> float:
        return self.loop_wall_s / self.sim_s if self.sim_s else float("nan")


def parse_nav_trace(rows: list[dict]) -> LoopProfile:
    """Decompose the control loop from nav-trace tick rows."""
    lp = LoopProfile()
    if not rows:
        return lp
    lp.ticks = len(rows)
    lp.loop_wall_s = rows[-1]["wall"] - rows[0]["wall"]
    lp.sim_s = rows[-1].get("t", 0.0) - rows[0].get("t", 0.0)

    deltas = []           # (wall_delta, action, rendered) per inter-tick gap
    for prev, cur in zip(rows, rows[1:]):
        dw = cur["wall"] - prev["wall"]
        if dw < 0:
            dw = 0.0
        deltas.append((dw, cur.get("action", "?"), bool(cur.get("rendered"))))

    # phase wall by action group
    groups = {"collect_observe": 0.0, "approach_reach": 0.0, "other": 0.0}
    for dw, action, _ in deltas:
        if action in _OBSERVE:
            groups["collect_observe"] += dw
        elif action in _APPROACH:
            groups["approach_reach"] += dw
        else:
            groups["other"] += dw
    lp.phase_wall_s = groups

    lp.rendered_ticks = sum(1 for _, _, r in deltas if r)
    all_d = [d for d, _, _ in deltas]
    if not all_d:
        return lp
    # step_floor: the fastest ticks are a near-pure physics step (no heavy render
    # landed that tick). Use the 5th percentile as the irreducible floor.
    lp.step_floor_s = _percentile(all_d, 5)
    plain_d = [d for d, _, r in deltas if not r]
    plain_median = statistics.median(plain_d) if plain_d else lp.step_floor_s

    render_total = 0.0
    walkthrough = 0.0
    for d, _, rendered in deltas:
        over = max(d - lp.step_floor_s, 0.0)
        render_total += over
        if rendered:
            walkthrough += max(d - plain_median, 0.0)
    lp.render_s = render_total
    lp.walkthrough_render_s = min(walkthrough, render_total)
    lp.sensing_render_s = lp.render_s - lp.walkthrough_render_s
    lp.step_ros_idle_s = lp.loop_wall_s - lp.render_s
    return lp


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# --- commissioning (separate run) -------------------------------------------

def commissioning_wall_s(run_json_path: Path) -> Optional[float]:
    """GPU-hold wall of a commissioning run from its 1 Hz telemetry samples."""
    try:
        d = json.loads(run_json_path.read_text())
    except (OSError, ValueError):
        return None
    tel = d.get("telemetry") or []
    stamps = [s.get("wall_time") for s in tel if isinstance(s, dict) and "wall_time" in s]
    if len(stamps) >= 2:
        return max(stamps) - min(stamps)
    return None


# --- top-level run profile ---------------------------------------------------

@dataclass
class RunProfile:
    run: str
    state: Optional[str] = None
    reason: Optional[str] = None
    boot_s: Optional[float] = None
    commission_s: Optional[float] = None
    collect_observe_s: float = 0.0
    approach_reach_s: float = 0.0
    other_phase_s: float = 0.0
    render_s: float = 0.0
    sensing_render_s: float = 0.0
    walkthrough_render_s: float = 0.0
    ros_host_idle_s: float = 0.0
    shutdown_s: Optional[float] = None
    loop_wall_s: float = 0.0
    sim_s: float = 0.0
    sim_to_wall: float = float("nan")
    rendered_ticks: int = 0
    ticks: int = 0
    collect_capture_ok: int = 0
    provider_errors: dict = field(default_factory=dict)
    frames_saved: Optional[int] = None

    @property
    def mission_process_s(self) -> Optional[float]:
        if self.boot_s is None:
            return None
        return self.boot_s + self.loop_wall_s + (self.shutdown_s or 0.0)

    @property
    def attempt_total_s(self) -> Optional[float]:
        mp = self.mission_process_s
        if mp is None:
            return None
        return mp + (self.commission_s or 0.0)


def _find_console(run_dir: Path) -> Optional[Path]:
    """Locate the console log for a mission run dir.

    Mission logs sit beside the run dir as ``run-NN.console.log``; some runs keep
    an ``isaac.log`` inside the dir instead.
    """
    sibling = run_dir.parent / f"{run_dir.name}.console.log"
    if sibling.exists() and sibling.stat().st_size > 0:
        return sibling
    inner = run_dir / "isaac.log"
    if inner.exists() and inner.stat().st_size > 0:
        return inner
    return None


def profile_run(run_dir: Path, commissioning_dir: Optional[Path] = None) -> RunProfile:
    run_dir = Path(run_dir)
    rp = RunProfile(run=run_dir.name)

    mj = run_dir / "mission.json"
    if mj.exists():
        try:
            d = json.loads(mj.read_text())
            rp.state = d.get("state")
            rp.reason = d.get("reason")
        except ValueError:
            pass

    console = _find_console(run_dir)
    if console is not None:
        cp = parse_console(console.read_text(errors="replace"))
        rp.boot_s = cp.boot_s
        rp.collect_capture_ok = cp.collect_capture_ok
        rp.provider_errors = cp.provider_errors
        if cp.shutdown_s is not None and cp.loop_start_s is not None:
            # shutdown phase = tail after the loop; filled below once loop known
            rp._shutdown_abs = cp.shutdown_s  # type: ignore[attr-defined]
        rp._loop_start_abs = cp.loop_start_s  # type: ignore[attr-defined]

    nt_path = run_dir / "nav-trace.jsonl"
    if nt_path.exists():
        rows = [json.loads(l) for l in nt_path.read_text().splitlines() if l.strip()]
        lp = parse_nav_trace(rows)
        rp.ticks = lp.ticks
        rp.loop_wall_s = lp.loop_wall_s
        rp.sim_s = lp.sim_s
        rp.sim_to_wall = lp.sim_to_wall
        rp.rendered_ticks = lp.rendered_ticks
        rp.collect_observe_s = lp.phase_wall_s.get("collect_observe", 0.0)
        rp.approach_reach_s = lp.phase_wall_s.get("approach_reach", 0.0)
        rp.other_phase_s = lp.phase_wall_s.get("other", 0.0)
        rp.render_s = lp.render_s
        rp.sensing_render_s = lp.sensing_render_s
        rp.walkthrough_render_s = lp.walkthrough_render_s
        rp.ros_host_idle_s = lp.step_ros_idle_s

    # shutdown = process time at "Shutting Down" minus (boot + loop wall)
    shutdown_abs = getattr(rp, "_shutdown_abs", None)
    loop_start_abs = getattr(rp, "_loop_start_abs", None)
    if shutdown_abs is not None and loop_start_abs is not None:
        rp.shutdown_s = max(shutdown_abs - loop_start_abs - rp.loop_wall_s, 0.0)

    frames = run_dir / "mission-frames" / "frames.json"
    if frames.exists():
        try:
            rp.frames_saved = len(json.loads(frames.read_text()).get("frames", []))
        except ValueError:
            pass

    if commissioning_dir is not None:
        rj = Path(commissioning_dir) / "run.json"
        rp.commission_s = commissioning_wall_s(rj)

    return rp


# --- rendering (CLI) ---------------------------------------------------------

def _fmt(v, nd=1):
    if v is None:
        return "   -"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def format_table(profiles: list[RunProfile]) -> str:
    cols = [
        ("run", "run", 8),
        ("boot", "boot_s", 6),
        ("comm", "commission_s", 6),
        ("collect", "collect_observe_s", 8),
        ("appr/reach", "approach_reach_s", 10),
        ("render", "render_s", 7),
        (" sensing", "sensing_render_s", 8),
        (" walkthr", "walkthrough_render_s", 8),
        ("ros/idle", "ros_host_idle_s", 8),
        ("shutdwn", "shutdown_s", 7),
        ("loop", "loop_wall_s", 7),
        ("sim", "sim_s", 5),
        ("s:w", "sim_to_wall", 6),
        ("state", "state", 8),
    ]
    head = "  ".join(f"{h:>{w}}" for h, _, w in cols)
    lines = [head, "-" * len(head)]
    for rp in profiles:
        cells = []
        for _, attr, w in cols:
            v = getattr(rp, attr)
            if attr == "sim_to_wall" and isinstance(v, float):
                cells.append(f"{v:>{w}.0f}" if v == v else f"{'nan':>{w}}")
            else:
                cells.append(f"{_fmt(v):>{w}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def _resolve_runs(args) -> list[tuple[Path, Optional[Path]]]:
    out = []
    comm = Path(args.commissioning) if args.commissioning else None
    if args.runs_root:
        root = Path(args.runs_root)
        names = args.runs.split(",") if args.runs else []
        for n in names:
            out.append((root / n, comm))
    for r in args.run or []:
        out.append((Path(r), comm))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="*", help="mission run directory/directories")
    ap.add_argument("--runs-root", help="parent dir holding run-NN directories")
    ap.add_argument("--runs", help="comma list of run names under --runs-root")
    ap.add_argument("--commissioning", help="a commissioning run dir to fold in")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    targets = _resolve_runs(args)
    if not targets:
        ap.error("give one or more run dirs, or --runs-root with --runs")

    profiles = [profile_run(rd, cd) for rd, cd in targets]

    if args.json:
        payload = []
        for rp in profiles:
            d = asdict(rp)
            d["mission_process_s"] = rp.mission_process_s
            d["attempt_total_s"] = rp.attempt_total_s
            payload.append(d)
        print(json.dumps(payload, indent=2))
    else:
        print(format_table(profiles))
        print()
        for rp in profiles:
            if rp.provider_errors:
                errs = ", ".join(f"{k}={v}" for k, v in sorted(rp.provider_errors.items()))
                print(f"{rp.run}: capture_ok={rp.collect_capture_ok} provider_errors: {errs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
