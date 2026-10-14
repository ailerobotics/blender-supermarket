"""People trajectories: session-scoped tokens, monotonic frames, no biometrics."""
import people as people_mod
import scenario_defs as sd


def test_sessions_unique_and_tokenised(plan):
    ppl = people_mod.generate_people(plan, n_people=6, n_frames=30, fps=6, seed=3)
    tokens = [p.meta["session_id"] for p in ppl]
    assert len(set(tokens)) == len(tokens)
    for t in tokens:
        assert t.startswith("s_") and len(t) == 12   # random session token, not a name


def test_trajectory_frames_monotonic(plan):
    ppl = people_mod.generate_people(plan, n_people=4, n_frames=30, fps=6, seed=3)
    gt = people_mod.trajectories_gt(ppl, 30)
    for sess in gt.values():
        frames = [pt["frame"] for pt in sess["track"]]
        assert frames == sorted(frames)
        assert all(0 <= f < 30 for f in frames)


def test_people_scenario_has_trajectories(plan):
    scn = sd.build_people(plan, n_people=5)
    assert len(scn.ground_truth["sessions"]) == 5
    assert set(scn.ground_truth["trajectories"]) == set(scn.ground_truth["sessions"])


def test_deterministic(plan):
    a = people_mod.generate_people(plan, 5, 30, 6, seed=7)
    b = people_mod.generate_people(plan, 5, 30, 6, seed=7)
    assert [p.meta["session_id"] for p in a] == [p.meta["session_id"] for p in b]
