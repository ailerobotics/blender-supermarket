"""Offline (CPU, no GPU) tests for the reach-replay harness.

Three layers:
* pure tape/schedule parsing against a committed ``frames.json`` fixture (fast, no cv2);
* a dataset-free end-to-end replay over the real capture+hold pipeline on an analytic
  bench, proving the occluded-reach starvation (strict) vs the PR #80 sustained hold;
* a gated replay of the recorded run-101 / run-102 tapes when the dataset is present,
  reproducing their failure mode and showing PR #80 behaviour on the same frames.

The bench RGB-D is a committed analytic fixture, never sensor/physics evidence; the
recorded frame *cadence and identity* (with input hashes) come from the run.
"""
from pathlib import Path

import pytest

from scripts.twin.tools.replay_reach import (
    FrameRecord, PhasePlan, COMMISSION_STAMP,
    load_frame_tape, tape_digest, build_schedule,
)

FIXTURE = Path(__file__).parent / "data" / "replay_reach_fixture"
DATASET = Path.home() / "datasets" / "twin-robot-in-store-boot-v1" / "mission"


# --- pure tape + schedule (no cv2, fast) ------------------------------------

def test_load_tape_records_identity_timing_and_input_hashes():
    tape = load_frame_tape(FIXTURE)
    assert len(tape) == 6
    assert [r.frame_id for r in tape] == [28, 30, 32, 34, 36, 38]   # render numerators
    assert all(len(r.rgb_sha256) == 64 for r in tape)               # recorded pixel hashes
    assert tape[0].stamp < tape[-1].stamp                           # monotonic acquisition
    # digest is a stable fingerprint of the ordered recorded-frame set
    assert tape_digest(tape) == tape_digest(load_frame_tape(FIXTURE))
    assert len(tape_digest(tape)) == 64


def test_schedule_rebases_clock_and_covers_observe_then_reach():
    tape = load_frame_tape(FIXTURE)
    steps = build_schedule(tape, plan=PhasePlan(observe=0.34, approach=0.33,
                                                contact=0.16, lift=0.17))
    assert len(steps) == len(tape)
    # the replayed clock is rebased so the first frame lands on the commission stamp,
    # while frame-to-frame deltas are preserved
    assert steps[0].now == pytest.approx(COMMISSION_STAMP)
    deltas_in = [b.stamp - a.stamp for a, b in zip(tape, tape[1:])]
    deltas_out = [b.now - a.now for a, b in zip(steps, steps[1:])]
    assert deltas_out == pytest.approx(deltas_in)
    phases = [s.phase for s in steps]
    assert phases[0] == "observe" and phases[-1] in ("approach", "contact", "lift")
    # the reach phases are flagged occluded; observe is clean
    assert not steps[0].occluded
    assert any(s.occluded for s in steps)


def test_repeats_re_present_each_frame():
    tape = load_frame_tape(FIXTURE)
    steps = build_schedule(tape, repeats=3)
    assert len(steps) == 3 * len(tape)


def test_max_frames_bounds_the_replay():
    tape = load_frame_tape(FIXTURE)
    assert len(build_schedule(tape, max_frames=3)) == 3


# --- end-to-end over the real capture+hold pipeline (cv2; analytic bench) ----

cv2 = pytest.importorskip("cv2")

from scripts.twin.tools.replay_reach import (build_source_bench, replay, replay_source_target,  # noqa: E402
                                             ReplayStep)


def _synthetic_tape(n=7):
    """A clean observe prefix then an (occluded) reach, independent of any dataset."""
    return [FrameRecord(i, 10 + i, 1.0 + 0.05 * i, f"hash{i}") for i in range(n)]


def test_strict_mode_starves_on_the_occluded_reach():
    # strict (observe/commissioning default): the fresh full-footprint view refuses once the
    # arm occludes the support -- the live run-99..102 starvation that expires the watchdog.
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=None)
    steps = build_schedule(_synthetic_tape(), plan=PhasePlan())
    rep = replay(adapter, cam, motion, contact, steps, source=loc, destination=loc)
    assert rep.ok >= 1                       # the clean observe frames pass
    assert rep.ok < rep.steps                # the occluded reach frames refuse
    assert sum(rep.errors.values()) >= 1
    assert rep.occluded_published == 0       # strict never publishes a held view


def test_pr80_hold_sustains_sensors_through_the_reach():
    # the merged PR #80 behaviour on the same reach: the hold re-expresses the last clean
    # registration from live TF and keeps /sensors publishing (flagged occluded).
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0)
    steps = build_schedule(_synthetic_tape(), plan=PhasePlan())
    rep = replay(adapter, cam, motion, contact, steps, source=loc, destination=loc)
    assert rep.ok == rep.steps               # every frame publishes
    assert rep.occluded_published >= 1       # the reach frames are held, not refused


def test_sensing_gate_holds_the_reach_without_rendering():
    # Opt-in sensing gate: observe renders and validates the registration; the occluded reach steps
    # republish the HELD registration WITHOUT a new render (set_render_suspended), so /sensors keeps
    # publishing (flagged occluded) while the decode-camera render -- the dominant wall cost -- is
    # suspended. This is the run-126 wall lever.
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=True)
    steps = build_schedule(_synthetic_tape(), plan=PhasePlan())
    rep = replay(adapter, cam, motion, contact, steps, source=loc, destination=loc, gate=True)
    assert rep.ok == rep.steps                                   # every tick still publishes
    assert rep.occluded_published >= 1                           # the reach is held, not rendered
    assert rep.rendered == sum(1 for s in steps if not s.occluded)   # only observe actually rendered
    assert rep.rendered < rep.steps                              # the reach frames were not rendered
    # the decode render PRODUCT is disabled on every held step (so world.step(render=True) would
    # skip the expensive decode render), and it is re-enabled by the end (resumed on the last... )
    assert rep.product_off_during_hold == sum(1 for s in steps if s.occluded)


def test_sensing_gate_off_renders_every_frame_like_pr80():
    # Default (gate off): the harness renders every frame -- the PR #80 behaviour (hold still fetches
    # and decodes a frame each tick). The gate is purely additive; off changes nothing.
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=False)
    steps = build_schedule(_synthetic_tape(), plan=PhasePlan())
    rep = replay(adapter, cam, motion, contact, steps, source=loc, destination=loc, gate=False)
    assert rep.rendered == rep.steps                             # every frame rendered (the wall cost)
    assert rep.ok == rep.steps


def test_collect_source_target_holds_fresh_through_the_gated_reach():
    # The run-131 coverage gap: under the sensing gate the decode frame freezes, so the collect
    # source_target_provider's pick observation stamp goes stale -> the consumer's 0.5 s freshness
    # gate rejects every tick -> /sensors starves. With the provider honoring suspension, the held
    # pick is republished with a FRESH stamp, so the collect path stays fresh through the reach while
    # the render -- the wall cost -- is suspended.
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=True, with_pick=True)
    steps = build_schedule(_synthetic_tape(), plan=PhasePlan())
    rep = replay_source_target(adapter, cam, motion, contact, steps, location=loc, gate=True)
    assert rep.fresh == rep.steps                        # every tick delivers a fresh collect target
    assert rep.stale == 0                                # never the stale-stamp starvation
    assert rep.held >= 1                                 # the reach targets are held (occluded)
    assert rep.rendered == sum(1 for s in steps if not s.occluded)   # only observe rendered


def test_collect_source_target_without_gate_renders_every_frame():
    # Gate off: the provider path is unchanged -- every frame renders and the collect target carries
    # the measuring-frame stamp (fresh because the frame is re-rendered each tick).
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=False, with_pick=True)
    steps = build_schedule(_synthetic_tape(), plan=PhasePlan())
    rep = replay_source_target(adapter, cam, motion, contact, steps, location=loc, gate=False)
    assert rep.rendered == rep.steps                     # every frame rendered (the wall cost)
    assert rep.held == 0                                 # nothing is gate-held when off


def test_exact_repeat_is_reused_but_a_stale_older_frame_is_refused():
    # Two production behaviours on re-presented frames:
    #  * an EXACT repeat (same id+stamp+pixels) is the legitimate stride re-presentation and
    #    is reused -> success (not a replay);
    #  * an OLDER / non-monotonic frame is the replay-refusal family ("no new synchronized
    #    rendered camera frame" / "needs a new monotonic frame") -> refused by the real gate.
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0)
    clean = [ReplayStep(0, 1.00, FrameRecord(0, 5, 1.00, "h"), "observe", False),
             ReplayStep(1, 1.05, FrameRecord(0, 6, 1.05, "h"), "observe", False)]
    assert replay(adapter, cam, motion, contact, clean, source=loc, destination=loc).ok == 2
    # exact stride re-presentation of the last frame -> reused (accepted)
    reuse = [ReplayStep(2, 1.10, FrameRecord(0, 6, 1.05, "h"), "observe", False)]
    assert replay(adapter, cam, motion, contact, reuse, source=loc, destination=loc).ok == 1
    # an older frame id (a genuinely replayed/stale frame) -> refused
    older = [ReplayStep(3, 1.12, FrameRecord(0, 3, 1.08, "h"), "observe", False)]
    rep3 = replay(adapter, cam, motion, contact, older, source=loc, destination=loc)
    assert rep3.ok == 0 and sum(rep3.errors.values()) == 1


# --- gated replay of the recorded run-101 / run-102 tapes --------------------

@pytest.mark.skipif(not (DATASET / "run-101" / "mission-frames" / "frames.json").exists(),
                    reason="recorded mission dataset not present")
@pytest.mark.parametrize("run", ["run-101", "run-102"])
def test_recorded_run_reproduces_failure_and_pr80_fix(run):
    from scripts.twin.tools.replay_reach import run_tape
    strict = run_tape(DATASET / run, occlusion_window_s=None, max_frames=6)
    held = run_tape(DATASET / run, occlusion_window_s=20.0, max_frames=6)
    assert strict.tape_digest == held.tape_digest and len(strict.tape_digest) == 64
    # recorded failure: strict refuses the occluded reach frames (what expired the watchdog)
    assert strict.ok >= 1 and strict.ok < strict.steps
    assert sum(strict.errors.values()) >= 1
    # PR #80 on the same recorded frames: the reach is held, /sensors keeps publishing
    assert held.ok == held.steps and held.occluded_published >= 1
